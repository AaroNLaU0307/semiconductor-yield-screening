"""I-MR control charts, Nelson rules, and the OOC against failure contingency test.

Per DECISIONS.md D8. Control limits are computed on train and applied unchanged to
holdout.

Two conventions were not fixed at Gate 1 because they are implementation detail rather
than methodology. Both are recorded here and in `reports/spc.md`, and both were chosen
before the holdout contingency table was computed:

1. A run based rule (2, 3, 5, 6) flags the point at which the pattern **completes**,
   not every point in the window. Flagging the whole window would multiply the out of
   control count by the window length for a single signal.
2. Run rules are evaluated **within a partition**, so a run is never assembled from a
   mixture of train and holdout records. A run spanning the boundary would let train
   records create holdout flags.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    IMR_INDIVIDUALS_CONSTANT,
    IMR_MOVING_RANGE_CONSTANT,
    NELSON_RULES_APPLIED,
)

#: d2 for a moving range of length 2. sigma is estimated as MR-bar / D2_CONSTANT.
D2_CONSTANT = 1.128

RULE_DESCRIPTIONS = {
    1: "one point beyond 3 sigma",
    2: "nine consecutive points on the same side of centre",
    3: "six consecutive points steadily increasing or decreasing",
    5: "two of three consecutive points beyond 2 sigma, same side",
    6: "four of five consecutive points beyond 1 sigma, same side",
}


@dataclass(frozen=True)
class ControlLimits:
    """I-MR limits estimated from train individuals."""

    centre: float
    sigma: float
    ucl: float
    lcl: float
    mr_bar: float
    mr_ucl: float

    def zone(self, values: np.ndarray) -> np.ndarray:
        """Signed distance from centre in sigma units."""
        if self.sigma <= 0:
            return np.zeros_like(values, dtype=float)
        return (values - self.centre) / self.sigma


def fit_control_limits(train_values: np.ndarray) -> ControlLimits:
    """Estimate I-MR limits from the train series.

    The declared individuals constant 2.66 is 3 / 1.128 rounded, so the individuals
    limits and the 3 sigma zone agree to within rounding. Zone rules use sigma
    directly; rule 1 uses the declared limits, which is what an engineer would read
    off the chart.
    """
    values = np.asarray(train_values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("need at least two observations to estimate a moving range")

    centre = float(np.mean(values))
    moving_range = np.abs(np.diff(values))
    mr_bar = float(np.mean(moving_range))
    sigma = mr_bar / D2_CONSTANT

    return ControlLimits(
        centre=centre,
        sigma=sigma,
        ucl=centre + IMR_INDIVIDUALS_CONSTANT * mr_bar,
        lcl=centre - IMR_INDIVIDUALS_CONSTANT * mr_bar,
        mr_bar=mr_bar,
        mr_ucl=IMR_MOVING_RANGE_CONSTANT * mr_bar,
    )


# ---------------------------------------------------------------------------
# Nelson rules
# ---------------------------------------------------------------------------


def _rule_1(values: np.ndarray, limits: ControlLimits) -> np.ndarray:
    return (values > limits.ucl) | (values < limits.lcl)


def _rule_2(z: np.ndarray, run: int = 9) -> np.ndarray:
    flags = np.zeros(z.size, dtype=bool)
    above = z > 0
    below = z < 0
    for i in range(run - 1, z.size):
        window = slice(i - run + 1, i + 1)
        if above[window].all() or below[window].all():
            flags[i] = True
    return flags


def _rule_3(values: np.ndarray, run: int = 6) -> np.ndarray:
    flags = np.zeros(values.size, dtype=bool)
    if values.size < run:
        return flags
    diffs = np.diff(values)
    for i in range(run - 1, values.size):
        window = diffs[i - run + 1 : i]
        if window.size and ((window > 0).all() or (window < 0).all()):
            flags[i] = True
    return flags


def _k_of_n_beyond(z: np.ndarray, k: int, n: int, threshold: float) -> np.ndarray:
    flags = np.zeros(z.size, dtype=bool)
    for i in range(n - 1, z.size):
        window = z[i - n + 1 : i + 1]
        if (window > threshold).sum() >= k or (window < -threshold).sum() >= k:
            flags[i] = True
    return flags


def apply_nelson_rules(
    values: np.ndarray,
    limits: ControlLimits,
    rules: tuple[int, ...] = NELSON_RULES_APPLIED,
) -> pd.DataFrame:
    """Return a boolean frame, one column per applied rule, plus `any_rule`.

    A run based rule flags the point at which the pattern completes.
    """
    values = np.asarray(values, dtype=np.float64)
    z = limits.zone(values)

    computed = {
        1: lambda: _rule_1(values, limits),
        2: lambda: _rule_2(z),
        3: lambda: _rule_3(values),
        5: lambda: _k_of_n_beyond(z, k=2, n=3, threshold=2.0),
        6: lambda: _k_of_n_beyond(z, k=4, n=5, threshold=1.0),
    }
    unknown = set(rules) - set(computed)
    if unknown:
        raise ValueError(f"no implementation for Nelson rules {sorted(unknown)}")

    frame = pd.DataFrame({f"rule_{r}": computed[r]() for r in rules})
    frame["any_rule"] = frame.any(axis=1)
    return frame


# ---------------------------------------------------------------------------
# Contingency test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContingencyResult:
    """2x2 table of out of control flag against actual failure, with a test."""

    table: np.ndarray  # [[ooc_fail, ooc_pass], [clean_fail, clean_pass]]
    odds_ratio: float
    p_value: float
    risk_ooc: float
    risk_clean: float
    risk_ratio: float
    risk_ratio_ci: tuple[float, float]
    n_ooc: int
    n_clean: int

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.table,
            index=["out of control", "in control"],
            columns=["failed", "passed"],
        )


def ooc_failure_contingency(
    ooc_flag: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> ContingencyResult:
    """Do out of control records fail more often than in control records?

    Fisher's exact test rather than chi-square, because with 26 holdout failures the
    expected cell counts are small and the chi-square approximation is not trustworthy
    there. The risk ratio is reported alongside, because an odds ratio on a rare
    outcome is easy to misread as a risk ratio.
    """
    ooc_flag = np.asarray(ooc_flag).astype(bool)
    y = np.asarray(y).astype(int)

    ooc_fail = int(np.sum(ooc_flag & (y == 1)))
    ooc_pass = int(np.sum(ooc_flag & (y == 0)))
    clean_fail = int(np.sum(~ooc_flag & (y == 1)))
    clean_pass = int(np.sum(~ooc_flag & (y == 0)))
    table = np.array([[ooc_fail, ooc_pass], [clean_fail, clean_pass]])

    odds_ratio, p_value = stats.fisher_exact(table, alternative="two-sided")

    n_ooc = ooc_fail + ooc_pass
    n_clean = clean_fail + clean_pass
    risk_ooc = ooc_fail / n_ooc if n_ooc else float("nan")
    risk_clean = clean_fail / n_clean if n_clean else float("nan")

    if ooc_fail > 0 and clean_fail > 0 and n_ooc > 0 and n_clean > 0:
        risk_ratio = risk_ooc / risk_clean
        # Katz log method for the interval on a risk ratio.
        se = np.sqrt(1 / ooc_fail - 1 / n_ooc + 1 / clean_fail - 1 / n_clean)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci = (
            float(np.exp(np.log(risk_ratio) - z_crit * se)),
            float(np.exp(np.log(risk_ratio) + z_crit * se)),
        )
    else:
        risk_ratio = float("nan")
        ci = (float("nan"), float("nan"))

    return ContingencyResult(
        table=table,
        odds_ratio=float(odds_ratio),
        p_value=float(p_value),
        risk_ooc=float(risk_ooc),
        risk_clean=float(risk_clean),
        risk_ratio=float(risk_ratio),
        risk_ratio_ci=ci,
        n_ooc=int(n_ooc),
        n_clean=int(n_clean),
    )


def chart_series(
    train_values: np.ndarray,
    holdout_values: np.ndarray,
    limits: ControlLimits,
    rules: tuple[int, ...] = NELSON_RULES_APPLIED,
) -> dict[str, object]:
    """Assemble one sensor's chart payload, flags evaluated within each partition."""
    train_flags = apply_nelson_rules(train_values, limits, rules)
    holdout_flags = apply_nelson_rules(holdout_values, limits, rules)
    return {
        "limits": {
            "centre": limits.centre,
            "ucl": limits.ucl,
            "lcl": limits.lcl,
            "sigma": limits.sigma,
            "mr_bar": limits.mr_bar,
            "mr_ucl": limits.mr_ucl,
        },
        "train_values": np.asarray(train_values, dtype=float).tolist(),
        "holdout_values": np.asarray(holdout_values, dtype=float).tolist(),
        "train_ooc": train_flags["any_rule"].to_numpy().tolist(),
        "holdout_ooc": holdout_flags["any_rule"].to_numpy().tolist(),
        "train_flags": {c: train_flags[c].to_numpy().tolist() for c in train_flags},
        "holdout_flags": {
            c: holdout_flags[c].to_numpy().tolist() for c in holdout_flags
        },
    }
