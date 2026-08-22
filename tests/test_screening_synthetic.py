"""Required test 3: planted signal recovery and the pure noise FDR bound.

This file implements required test 3 of the project: "Screening, planted signal and
pure noise. On synthetic data with one planted association among noise columns,
screening recovers it. On pure noise, the FDR survivor count stays within its
expected bound across seeds."

Every test here is built entirely from synthetic data generated with
`numpy.random.default_rng(seed)` and a fixed seed or fixed seed list. Nothing in
this file reads the real dataset, and nothing here depends on wall clock time or
an unseeded random source, so every test is exactly reproducible on every run.

Four groups of tests live here:

1. Planted signal recovery: one genuinely associated column, buried among pure
   noise columns, is recovered by the screen and ranks first by effect size.
2. Pure noise behaviour: under the complete null, the FDR survivor count stays
   within the bound Benjamini-Hochberg promises, across a fixed set of 100 seeds.
3. Permutation null sanity: the permutation machinery agrees that pure noise looks
   like noise, and that a family of unambiguous true signals does not.
4. Fallback labelling (DECISIONS.md D7): when BH clears fewer than
   FALLBACK_TRIGGER_BELOW features, the top FALLBACK_TOP_K by effect size are
   carried forward and labelled as such, and BH survivors are labelled as such too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_screening.config import (
    FALLBACK_LABEL,
    FALLBACK_TOP_K,
    FALLBACK_TRIGGER_BELOW,
    FDR_Q,
    SURVIVOR_LABEL,
)
from yield_screening.screening import mannwhitney_screen, permutation_null


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def make_planted(
    seed: int,
    n: int = 400,
    n_noise: int = 120,
    shift: float = 1.0,
    fail_rate: float = 0.15,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build `n_noise` standard normal noise columns plus one shifted `planted` column.

    `y` is a binary vector at roughly `fail_rate`. The planted column is standard
    normal for pass rows and standard normal shifted by `shift` for fail rows, so it
    carries a genuine, known association with `y` and every other column carries
    none. At least 20 fails are guaranteed regardless of seed: for the defaults used
    throughout this file (n=400, fail_rate=0.15, expected fails around 60) this
    safety net is never actually exercised, it only guards the unlikely case of an
    unfavourable draw.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fail_rate).astype(int)
    if y.sum() < 20:
        deficit = 20 - int(y.sum())
        pass_rows = np.flatnonzero(y == 0)
        flipped = rng.choice(pass_rows, size=deficit, replace=False)
        y[flipped] = 1

    noise = rng.normal(size=(n, n_noise))
    columns = [f"noise_{j:03d}" for j in range(n_noise)]
    X = pd.DataFrame(noise, columns=columns)

    planted = rng.normal(size=n)
    planted[y == 1] += shift
    X["planted"] = planted

    return X, y


def make_pure_noise(
    seed: int,
    n: int = 300,
    n_noise: int = 100,
    fail_rate: float = 0.15,
) -> tuple[pd.DataFrame, np.ndarray]:
    """`n_noise` standard normal columns with no planted association: the complete null.

    Every column, including the choice of which rows fail, is independent noise, so
    any FDR survivor produced from this data is by construction a false discovery.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fail_rate).astype(int)
    noise = rng.normal(size=(n, n_noise))
    columns = [f"noise_{j:03d}" for j in range(n_noise)]
    X = pd.DataFrame(noise, columns=columns)
    return X, y


def make_multi_planted(
    seed: int,
    n: int = 400,
    n_noise: int = 100,
    n_planted: int = 5,
    shift: float = 1.5,
    fail_rate: float = 0.3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """`n_noise` noise columns plus `n_planted` independently shifted associated columns.

    Used only by the permutation null sanity check in group 3, to demonstrate that
    an unambiguous family of true signals is well separated from the permutation
    null. A single planted column is the wrong tool for that specific job: under the
    permutation null, the family-wise chance of at least one spurious BH survivor is
    itself on the order of q, regardless of family size, which is the same fact
    `test_pure_noise_rarely_produces_any_survivor` measures below. That means an
    observed count of exactly one true feature sits inside the ordinary range of the
    null distribution rather than clearly outside it, so a single planted column
    cannot reliably demonstrate a small permutation p-value. Five independently
    planted columns push the observed survivor count comfortably clear of the null
    distribution instead, which is what this test needs to show.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fail_rate).astype(int)
    noise = rng.normal(size=(n, n_noise))
    columns = [f"noise_{j:03d}" for j in range(n_noise)]
    X = pd.DataFrame(noise, columns=columns)
    for k in range(n_planted):
        column = rng.normal(size=n)
        column[y == 1] += shift
        X[f"planted_{k:02d}"] = column
    return X, y


# Fixed seed lists. Every seed below is an explicit literal, never a random draw.
PLANTED_SEEDS: list[int] = [101, 202, 303, 404, 505]
PURE_NOISE_SEEDS: list[int] = list(range(1000, 1100))  # exactly 100 fixed seeds
PERMUTATION_NULL_NOISE_SEED = 2000
PERMUTATION_NULL_SIGNAL_SEED = 7000


# ---------------------------------------------------------------------------
# Group 1: planted signal recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", PLANTED_SEEDS)
def test_planted_signal_is_recovered(seed: int) -> None:
    """The one planted association clears BH and ranks first by |effect size|.

    Protects the core promise of the screen: a real association buried among many
    pure noise columns is not just detected, it stands out as the strongest effect
    in the whole family, checked across 5 independent seeds.
    """
    X, y = make_planted(seed)
    result = mannwhitney_screen(X, y)
    table = result.table.set_index("feature")

    assert bool(table.loc["planted", "survives_fdr"])

    top_feature = result.table.loc[result.table["abs_effect_size"].idxmax(), "feature"]
    assert top_feature == "planted"


def test_planted_effect_size_has_the_expected_sign() -> None:
    """The rank-biserial effect size for the planted column is positive and large.

    The planted column is shifted up for fails, so a fail-versus-pass comparison
    must report a positive effect, and one clearly distinguishable from the small
    positive or negative noise seen on columns with no real association.
    """
    X, y = make_planted(seed=PLANTED_SEEDS[0])
    result = mannwhitney_screen(X, y)
    table = result.table.set_index("feature")

    assert table.loc["planted", "effect_size_rank_biserial"] > 0.3


def test_weaker_planted_signal_is_still_found_but_with_a_smaller_effect() -> None:
    """Effect size responds monotonically to the true shift, not just a p-value.

    Runs the same seed at shift=1.0 and shift=0.4, holding every random draw
    identical except for the size of the planted shift itself, and checks that the
    reported effect size is strictly larger for the larger shift. This is a
    stronger property than "both were found": a screen could clear FDR at both
    shifts while reporting an effect size that does not actually track the true
    effect, and that would not be caught by a p-value threshold alone.
    """
    seed = PLANTED_SEEDS[0]
    X_strong, y_strong = make_planted(seed, shift=1.0)
    X_weak, y_weak = make_planted(seed, shift=0.4)

    strong = mannwhitney_screen(X_strong, y_strong).table.set_index("feature")
    weak = mannwhitney_screen(X_weak, y_weak).table.set_index("feature")

    assert weak.loc["planted", "effect_size_rank_biserial"] > 0
    assert strong.loc["planted", "effect_size_rank_biserial"] > 0
    assert weak.loc["planted", "abs_effect_size"] < strong.loc["planted", "abs_effect_size"]


# ---------------------------------------------------------------------------
# Group 2: pure noise behaviour, the important one
# ---------------------------------------------------------------------------


def test_pure_noise_rarely_produces_any_survivor() -> None:
    """Bound P(at least one FDR survivor) under the complete null, across 100 seeds.

    Under the complete null (no column has any real association), every BH
    rejection is a false rejection, so the false discovery rate collapses to
    exactly P(at least one survivor). Benjamini-Hochberg's guarantee is
    FDR <= q, so under the complete null this becomes
    P(at least one survivor) <= q, with q = FDR_Q = 0.10 here.

    The assertion below allows a small margin (0.05) above that theoretical bound,
    to absorb finite sample slack from using exactly 100 seeds rather than
    infinitely many, and from the normal approximation the screen uses for the
    Mann-Whitney statistic. The test is deterministic because PURE_NOISE_SEEDS is a
    fixed list of literal seeds, not a sample size chosen or drawn at run time, so
    the measured fraction below is exactly reproducible on every run.

    n=300 and n_noise=100 (smaller than group 1's n=400 and n_noise=120) are used so
    that all 100 seeds run in well under a second, comfortably inside this file's
    runtime budget, while remaining large enough for the normal approximation behind
    the reported p-values to stay accurate. If a future change to this file needs
    more seeds and 100 becomes too slow, shrink n and n_noise further rather than
    the seed count, since the seed count is what makes the bound meaningful.
    """
    seeds_with_a_survivor = 0
    for seed in PURE_NOISE_SEEDS:
        X, y = make_pure_noise(seed)
        result = mannwhitney_screen(X, y)
        if result.n_survivors > 0:
            seeds_with_a_survivor += 1

    fraction = seeds_with_a_survivor / len(PURE_NOISE_SEEDS)
    assert fraction <= FDR_Q + 0.05


def test_pure_noise_mean_survivor_count_is_near_zero() -> None:
    """Over the same 100 seeds, the mean survivor count must stay far below one.

    A complementary check to the fraction bound above: not only must a survivor be
    rare, the typical number of survivors on the seeds where one does appear must
    not be large either, otherwise a low "any survivor" fraction could still hide
    an occasional badly miscalibrated run.
    """
    counts = [mannwhitney_screen(*make_pure_noise(seed)).n_survivors for seed in PURE_NOISE_SEEDS]
    assert (sum(counts) / len(counts)) < 1.0


# ---------------------------------------------------------------------------
# Group 3: permutation null sanity
# ---------------------------------------------------------------------------


def test_permutation_null_on_pure_noise_gives_a_large_p_value() -> None:
    """The permutation null must not manufacture significance out of pure noise.

    On one fixed seed of pure noise, the observed screen and its permutation null
    should agree that nothing is there: a large p-value, and the full requested
    count of null draws actually produced.
    """
    X, y = make_pure_noise(seed=PERMUTATION_NULL_NOISE_SEED)
    result = mannwhitney_screen(X, y)
    null = permutation_null(X, y, observed_count=result.n_survivors, n_permutations=300, seed=4242)

    assert null.p_value > 0.10
    assert len(null.null_counts) == 300


def test_permutation_null_on_planted_signal_gives_a_small_p_value() -> None:
    """A family of unambiguous true signals is clearly separated from the null.

    Uses shift=1.5 and fail_rate=0.3 (a larger fail count than group 1's default)
    across five independently planted columns, per `make_multi_planted`'s docstring,
    so that the observed survivor count is unambiguously outside the range the
    permutation null produces on its own.
    """
    X, y = make_multi_planted(seed=PERMUTATION_NULL_SIGNAL_SEED)
    result = mannwhitney_screen(X, y)
    null = permutation_null(X, y, observed_count=result.n_survivors, n_permutations=300, seed=4242)

    assert null.p_value <= 0.01
    assert result.n_survivors > null.summary()["null_p95"]


# ---------------------------------------------------------------------------
# Group 4: fallback labelling (DECISIONS.md D7)
# ---------------------------------------------------------------------------


def test_fallback_labels_carried_features_when_few_survive() -> None:
    """When BH clears fewer than FALLBACK_TRIGGER_BELOW features, D7's fallback
    carries forward exactly FALLBACK_TOP_K features by effect size, and every
    carried feature that did not itself clear FDR is labelled FALLBACK_LABEL.

    Seed PURE_NOISE_SEEDS[0] (seed 1000) is pre-verified to produce zero FDR
    survivors on this pure noise construction, which is comfortably below
    FALLBACK_TRIGGER_BELOW and triggers the fallback.
    """
    X, y = make_pure_noise(seed=PURE_NOISE_SEEDS[0])
    result = mannwhitney_screen(X, y)

    assert result.n_survivors < FALLBACK_TRIGGER_BELOW
    assert result.fallback_used

    shortlisted = result.table[result.table["shortlisted"]]
    assert len(shortlisted) == FALLBACK_TOP_K

    carried = shortlisted[~shortlisted["survives_fdr"]]
    assert (carried["status"] == FALLBACK_LABEL).all()


def test_survivors_are_labelled_as_cleared_fdr() -> None:
    """A feature that clears BH must carry SURVIVOR_LABEL, not a generic status.

    On planted data, the planted column survives FDR outright (group 1 above), so
    its status column must read exactly SURVIVOR_LABEL.
    """
    X, y = make_planted(seed=PLANTED_SEEDS[0])
    result = mannwhitney_screen(X, y)
    table = result.table.set_index("feature")

    assert table.loc["planted", "status"] == SURVIVOR_LABEL
