"""Univariate screen, BH-FDR, and the permutation null, per DECISIONS.md D4 to D7.

The screen is a two sided Mann-Whitney U test per feature, reported with a
rank-biserial effect size, corrected by Benjamini-Hochberg across the whole family.

Two implementations of the same statistic live here on purpose:

`mannwhitney_screen` computes the observed result one column at a time and is the
readable reference. `permutation_null` computes the identical statistic as a matrix
product so that 2000 permutations are affordable, exploiting the fact that ranks do
not change when labels are permuted. A permutation null computed from a subtly
different statistic than the observed value looks rigorous and is worthless, so
`tests/test_screening_math.py` pins the two together and both against scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    BOOTSTRAP_INTERVAL,
    FALLBACK_LABEL,
    FALLBACK_TOP_K,
    FALLBACK_TRIGGER_BELOW,
    FDR_Q,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    RANDOM_SEED,
    SURVIVOR_LABEL,
)


# ---------------------------------------------------------------------------
# Rank machinery, shared by the observed screen and the permutation null
# ---------------------------------------------------------------------------


def column_midranks(X: np.ndarray) -> np.ndarray:
    """Average ranks within each column, ties resolved by midrank."""
    return stats.rankdata(X, method="average", axis=0)


def tie_terms(X: np.ndarray) -> np.ndarray:
    """Per column tie term T = sum over tied groups of (t**3 - t).

    Median imputation creates large tie blocks. Ignoring this term would understate
    the variance of U and inflate significance, so it is never optional here.
    """
    n_cols = X.shape[1]
    out = np.zeros(n_cols, dtype=np.float64)
    for j in range(n_cols):
        counts = np.unique(X[:, j], return_counts=True)[1].astype(np.float64)
        out[j] = float(np.sum(counts**3 - counts))
    return out


def u_sigma(n1: int, n2: int, tie_term: np.ndarray) -> np.ndarray:
    """Tie corrected standard deviation of U under the null."""
    n = n1 + n2
    variance = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1.0)))
    variance = np.clip(variance, 0.0, None)
    return np.sqrt(variance)


def u_from_rank_sums(rank_sum_group1: np.ndarray, n1: int) -> np.ndarray:
    """U for group 1, from its summed midranks."""
    return rank_sum_group1 - n1 * (n1 + 1.0) / 2.0


def p_from_u(u1: np.ndarray, n1: int, n2: int, sigma: np.ndarray) -> np.ndarray:
    """Two sided p-value from the tie corrected normal approximation.

    No continuity correction, matching
    `scipy.stats.mannwhitneyu(..., method="asymptotic", use_continuity=False)`.
    Degenerate columns, where sigma is zero because every value is tied, get p = 1.
    """
    mu = n1 * n2 / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, (u1 - mu) / np.where(sigma > 0, sigma, 1.0), 0.0)
    return 2.0 * stats.norm.sf(np.abs(z))


def bootstrap_rank_biserial(
    values: np.ndarray,
    y: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    interval: tuple[float, float] = BOOTSTRAP_INTERVAL,
) -> tuple[float, float]:
    """Percentile interval for one feature's rank-biserial effect size.

    Stratified so the class counts are preserved in every resample. At 26 holdout
    failures an unstratified resample would sometimes contain very few, and the
    resulting spread would describe the resampling scheme as much as the data.

    A point estimate of an effect size computed from 26 positives is not a result on
    its own, which is why this exists: the same standard already applied to PR-AUC in
    DECISIONS.md D10 applies to effect sizes too.
    """
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y).astype(int)
    positive_idx = np.flatnonzero(y == 1)
    negative_idx = np.flatnonzero(y == 0)
    if positive_idx.size < 2 or negative_idx.size < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n1, n2 = positive_idx.size, negative_idx.size
    scores = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        fail = values[rng.choice(positive_idx, size=n1, replace=True)]
        ok = values[rng.choice(negative_idx, size=n2, replace=True)]
        combined = np.concatenate([fail, ok])
        ranks = stats.rankdata(combined, method="average")
        u1 = ranks[:n1].sum() - n1 * (n1 + 1.0) / 2.0
        scores[b] = 2.0 * (u1 / (n1 * n2)) - 1.0
    low, high = np.percentile(scores, interval)
    return float(low), float(high)


def rank_biserial(u1: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Rank-biserial correlation, r = 2 * AUC - 1, in [-1, 1].

    Positive r means the fail group tends to sit higher on that sensor.
    """
    auc = u1 / (n1 * n2)
    return 2.0 * auc - 1.0


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


def benjamini_hochberg(p_values: np.ndarray, q: float = FDR_Q) -> tuple[np.ndarray, np.ndarray]:
    """Return (survives, adjusted_p) for a single family of p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)

    adjusted_sorted = np.minimum.accumulate((m / ranks * p_sorted)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    below = p_sorted <= q * ranks / m
    n_survivors = int(m - np.argmax(below[::-1])) if below.any() else 0
    survives_sorted = np.zeros(m, dtype=bool)
    survives_sorted[:n_survivors] = True

    survives = np.empty(m, dtype=bool)
    adjusted = np.empty(m, dtype=np.float64)
    survives[order] = survives_sorted
    adjusted[order] = adjusted_sorted
    return survives, adjusted


def bh_survivor_counts(p_matrix: np.ndarray, q: float = FDR_Q) -> np.ndarray:
    """Survivor count per column of a (features, replicates) p-value matrix."""
    m = p_matrix.shape[0]
    p_sorted = np.sort(p_matrix, axis=0)
    thresholds = (q * np.arange(1, m + 1, dtype=np.float64) / m)[:, None]
    below = p_sorted <= thresholds
    counts = m - np.argmax(below[::-1, :], axis=0)
    return np.where(below.any(axis=0), counts, 0).astype(int)


# ---------------------------------------------------------------------------
# The observed screen
# ---------------------------------------------------------------------------


@dataclass
class ScreenResult:
    """Per feature screen output plus the family level facts."""

    table: pd.DataFrame
    n_tested: int
    n_survivors: int
    q: float
    expected_false_discoveries: float
    smallest_failing_p: float
    fallback_used: bool

    @property
    def shortlist(self) -> pd.DataFrame:
        """Survivors, or the effect-size ranked fallback, in report order."""
        return self.table[self.table["shortlisted"]].copy()


def mannwhitney_screen(
    X: pd.DataFrame,
    y: np.ndarray,
    q: float = FDR_Q,
    fallback_trigger_below: int = FALLBACK_TRIGGER_BELOW,
    fallback_top_k: int = FALLBACK_TOP_K,
) -> ScreenResult:
    """Screen every column of `X` against binary `y`.

    `X` must already be imputed, since the rank machinery has no missing value
    handling by design: deciding what a missing value ranks as is a modelling choice
    and it was already made in DECISIONS.md D3.
    """
    values = X.to_numpy(dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("screen received missing values, impute before screening")

    y = np.asarray(y).astype(int)
    n1 = int(y.sum())
    n2 = int(len(y) - n1)
    if n1 == 0 or n2 == 0:
        raise ValueError("screen needs both classes present")

    ranks = column_midranks(values)
    sigma = u_sigma(n1, n2, tie_terms(values))
    rank_sum_fail = ranks[y == 1].sum(axis=0)
    u1 = u_from_rank_sums(rank_sum_fail, n1)

    p_raw = p_from_u(u1, n1, n2, sigma)
    effect = rank_biserial(u1, n1, n2)
    survives, p_adj = benjamini_hochberg(p_raw, q)

    table = pd.DataFrame(
        {
            "feature": list(X.columns),
            "effect_size_rank_biserial": effect,
            "abs_effect_size": np.abs(effect),
            "auc_fail_over_pass": u1 / (n1 * n2),
            "p_raw": p_raw,
            "p_bh_adjusted": p_adj,
            "survives_fdr": survives,
        }
    )

    n_survivors = int(survives.sum())
    fallback_used = n_survivors < fallback_trigger_below

    table["shortlisted"] = table["survives_fdr"]
    table["status"] = np.where(table["survives_fdr"], SURVIVOR_LABEL, "not shortlisted")

    if fallback_used:
        ranked = table.sort_values(
            ["abs_effect_size", "p_raw"], ascending=[False, True], kind="stable"
        )
        carried = ranked.index[:fallback_top_k]
        newly = [i for i in carried if not table.loc[i, "survives_fdr"]]
        table.loc[newly, "status"] = FALLBACK_LABEL
        table.loc[carried, "shortlisted"] = True

    failing = table.loc[~table["survives_fdr"], "p_raw"]
    smallest_failing_p = float(failing.min()) if len(failing) else float("nan")

    table = table.sort_values(
        ["survives_fdr", "abs_effect_size"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)

    return ScreenResult(
        table=table,
        n_tested=int(len(table)),
        n_survivors=n_survivors,
        q=q,
        expected_false_discoveries=float(q * n_survivors),
        smallest_failing_p=smallest_failing_p,
        fallback_used=fallback_used,
    )


# ---------------------------------------------------------------------------
# Permutation null
# ---------------------------------------------------------------------------


@dataclass
class PermutationNullResult:
    """Null distribution of the survivor count under permuted labels."""

    null_counts: np.ndarray
    observed_count: int
    n_permutations: int
    q: float
    seed: int

    @property
    def p_value(self) -> float:
        """Permutation p-value, with the standard plus one correction."""
        at_least = int(np.sum(self.null_counts >= self.observed_count))
        return (1.0 + at_least) / (1.0 + self.n_permutations)

    def summary(self) -> dict[str, float]:
        counts = self.null_counts
        return {
            "null_mean": float(counts.mean()),
            "null_median": float(np.median(counts)),
            "null_p95": float(np.percentile(counts, 95)),
            "null_max": float(counts.max()),
            "null_zero_share": float((counts == 0).mean()),
            "observed": float(self.observed_count),
            "permutation_p_value": self.p_value,
        }


def permutation_null(
    X: pd.DataFrame,
    y: np.ndarray,
    observed_count: int,
    n_permutations: int = N_PERMUTATIONS,
    q: float = FDR_Q,
    seed: int = RANDOM_SEED,
) -> PermutationNullResult:
    """Rerun the whole screen, BH correction included, on permuted labels.

    Only the labels move, so the correlation structure among features is preserved
    exactly rather than assumed. This is what makes the result independent of the
    dependence assumptions BH itself relies on.

    Ranks and the tie corrected sigma depend on the feature values alone, so both are
    computed once and every permutation reduces to a single matrix product.
    """
    values = X.to_numpy(dtype=np.float64)
    y = np.asarray(y).astype(int)
    n = len(y)
    n1 = int(y.sum())
    n2 = n - n1

    ranks = column_midranks(values)
    sigma = u_sigma(n1, n2, tie_terms(values))

    rng = np.random.default_rng(seed)
    # Each column of `labels` is one permutation of y. argsort of uniform noise gives
    # an exact permutation per replicate without a Python loop.
    order = np.argsort(rng.random((n_permutations, n)), axis=1)
    labels = np.take(y, order).T.astype(np.float64)

    rank_sums = ranks.T @ labels
    u1 = u_from_rank_sums(rank_sums, n1)
    p_matrix = p_from_u(u1, n1, n2, sigma[:, None])
    counts = bh_survivor_counts(p_matrix, q)

    return PermutationNullResult(
        null_counts=counts,
        observed_count=int(observed_count),
        n_permutations=int(n_permutations),
        q=q,
        seed=seed,
    )
