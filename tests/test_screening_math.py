"""Pin the screening statistic against independent reference implementations.

The permutation null is only meaningful if the fast matrix path computes exactly the
statistic the observed screen reports. These tests are the reason that claim can be
made. They also pin both against scipy and statsmodels, so a future refactor cannot
quietly change what is being tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from yield_screening.screening import (
    benjamini_hochberg,
    bh_survivor_counts,
    circular_shift_null,
    column_midranks,
    mannwhitney_screen,
    p_from_u,
    rank_biserial,
    tie_terms,
    u_from_rank_sums,
    u_sigma,
)


def _frame(seed: int, n: int = 200, p: int = 12, tie_blocks: bool = True):
    """A frame with deliberate tie structure, which is the hard case."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    if tie_blocks:
        # Emulate median imputation: force a chunk of each column to one value.
        for j in range(p):
            idx = rng.choice(n, size=n // 5, replace=False)
            X[idx, j] = np.median(X[:, j])
        X[:, 0] = np.round(X[:, 0])  # heavy ties
    y = (rng.random(n) < 0.15).astype(int)
    if y.sum() < 5:
        y[:5] = 1
    return pd.DataFrame(X, columns=[f"f{j:02d}" for j in range(p)]), y


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_screen_p_values_match_scipy(seed: int) -> None:
    """Every reported p-value and effect size matches a per column scipy call."""
    X, y = _frame(seed)
    result = mannwhitney_screen(X, y)
    row_by_feature = result.table.set_index("feature")

    for col in X.columns:
        fail = X.loc[y == 1, col].to_numpy()
        ok = X.loc[y == 0, col].to_numpy()
        reference = stats.mannwhitneyu(
            fail, ok, alternative="two-sided", method="asymptotic", use_continuity=False
        )
        assert row_by_feature.loc[col, "p_raw"] == pytest.approx(
            reference.pvalue, rel=1e-10, abs=1e-12
        )
        expected_r = 2.0 * (reference.statistic / (y.sum() * (len(y) - y.sum()))) - 1.0
        assert row_by_feature.loc[col, "effect_size_rank_biserial"] == pytest.approx(
            expected_r, rel=1e-10, abs=1e-12
        )


@pytest.mark.parametrize("seed", [4, 5])
def test_fast_matrix_path_reproduces_the_observed_statistic(seed: int) -> None:
    """The permutation path, fed the unpermuted labels, reproduces the screen exactly.

    This is the test that makes the permutation null trustworthy. It exercises the
    matrix product form used inside `permutation_null` rather than a copy of it.
    """
    X, y = _frame(seed)
    observed = mannwhitney_screen(X, y).table.set_index("feature")

    values = X.to_numpy(dtype=np.float64)
    n1 = int(y.sum())
    n2 = int(len(y) - n1)
    ranks = column_midranks(values)
    sigma = u_sigma(n1, n2, tie_terms(values))

    labels = y.astype(np.float64)[:, None]
    u1 = u_from_rank_sums(ranks.T @ labels, n1)
    p_fast = p_from_u(u1, n1, n2, sigma[:, None])[:, 0]
    r_fast = rank_biserial(u1, n1, n2)[:, 0]

    np.testing.assert_allclose(
        p_fast, observed.loc[X.columns, "p_raw"].to_numpy(), rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        r_fast,
        observed.loc[X.columns, "effect_size_rank_biserial"].to_numpy(),
        rtol=1e-12,
        atol=1e-15,
    )


def test_benjamini_hochberg_matches_statsmodels() -> None:
    """Survivor set and adjusted p-values match the statsmodels reference."""
    multipletests = pytest.importorskip("statsmodels.stats.multitest").multipletests
    rng = np.random.default_rng(11)
    for q in (0.05, 0.10, 0.20):
        for _ in range(20):
            p = np.concatenate(
                [rng.uniform(0, 1, 90), rng.uniform(0, 0.01, 10)]
            )
            rng.shuffle(p)
            survives, adjusted = benjamini_hochberg(p, q)
            ref_reject, ref_adjusted, _, _ = multipletests(p, alpha=q, method="fdr_bh")
            np.testing.assert_array_equal(survives, ref_reject)
            np.testing.assert_allclose(adjusted, ref_adjusted, rtol=1e-12, atol=1e-15)


def test_bh_survivor_counts_matches_per_column_bh() -> None:
    """The vectorised counter agrees with looping the scalar BH implementation."""
    rng = np.random.default_rng(12)
    p_matrix = np.concatenate(
        [rng.uniform(0, 1, (80, 40)), rng.uniform(0, 0.005, (20, 40))]
    )
    counts = bh_survivor_counts(p_matrix, q=0.10)
    expected = [
        int(benjamini_hochberg(p_matrix[:, b], 0.10)[0].sum())
        for b in range(p_matrix.shape[1])
    ]
    np.testing.assert_array_equal(counts, expected)


def test_bh_survivor_counts_handles_no_survivors() -> None:
    """A family where nothing clears must return zero, not the family size."""
    p_matrix = np.full((50, 5), 0.9)
    np.testing.assert_array_equal(bh_survivor_counts(p_matrix, q=0.10), np.zeros(5, int))


def test_screen_rejects_missing_values() -> None:
    """Screening imputed data is a declared decision, so unimputed input must fail."""
    X, y = _frame(7)
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        mannwhitney_screen(X, y)


def test_circular_shift_null_matches_the_reference_screen_shift_by_shift() -> None:
    """Each null count is the survivor count of the readable screen on np.roll(y, k).

    Uses labels and features that both trend, so the counts vary across shifts and the
    comparison is not a string of zeros.
    """
    rng = np.random.default_rng(21)
    n = 240
    y = (rng.random(n) < np.where(np.arange(n) < n // 4, 0.3, 0.05)).astype(int)
    X = pd.DataFrame(
        np.cumsum(rng.normal(size=(n, 30)), axis=0), columns=[f"w{j:02d}" for j in range(30)]
    )
    null = circular_shift_null(X, y, observed_count=mannwhitney_screen(X, y).n_survivors)

    assert null.n_permutations == n - 1 and len(null.null_counts) == n - 1
    assert null.seed is None
    assert len(set(null.null_counts.tolist())) > 3
    for k in (1, 17, 60, 120, 181, n - 1):
        assert null.null_counts[k - 1] == mannwhitney_screen(X, np.roll(y, k)).n_survivors
