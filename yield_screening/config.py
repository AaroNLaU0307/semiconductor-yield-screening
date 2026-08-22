"""Every frozen constant from DECISIONS.md, in one place.

DECISIONS.md is the authority. This module is its executable form. A number that
appears in pipeline code but not here, or here but not there, is a defect.

Frozen at Gate 1 and approved by the owner. Per DECISIONS.md D12, none of these
values may be changed on the basis of a result.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# D13. Determinism
# ---------------------------------------------------------------------------

#: Global seed. Passed explicitly at every call site, never relied on globally.
RANDOM_SEED: Final[int] = 20080719

# ---------------------------------------------------------------------------
# D1. Chronological split
# ---------------------------------------------------------------------------

#: Fraction of records, ordered by timestamp, assigned to train.
TRAIN_FRACTION: Final[float] = 0.70

# ---------------------------------------------------------------------------
# D2. Column drop rules, fitted on train only
# ---------------------------------------------------------------------------

#: Drop a sensor if its most common non-null train value covers at least this share.
MODAL_SHARE_CEILING: Final[float] = 0.99

#: Drop a sensor outright if its train missing rate exceeds this.
MISSING_RATE_HARD_CEILING: Final[float] = 0.95

# ---------------------------------------------------------------------------
# D3. Missing data strategy, two bands
# ---------------------------------------------------------------------------

#: Sensors at or below this train missing rate keep their values and are median imputed.
#: Sensors above it, and at or below MISSING_RATE_HARD_CEILING, contribute only a
#: missingness indicator. Measured: no sensor has a train missing rate in [0.02, 0.20),
#: so any ceiling in that interval yields an identical partition.
DENSE_BAND_CEILING: Final[float] = 0.20

#: Suffix for the emitted binary missingness indicator columns.
INDICATOR_SUFFIX: Final[str] = "__notmeasured"

# ---------------------------------------------------------------------------
# D4, D5. Screening and false discovery control
# ---------------------------------------------------------------------------

#: Benjamini-Hochberg level, applied across the whole screen family.
FDR_Q: Final[float] = 0.10

# ---------------------------------------------------------------------------
# D6. Permutation null
# ---------------------------------------------------------------------------

#: Number of label permutations used to build the null distribution of survivor counts.
N_PERMUTATIONS: Final[int] = 2000

# ---------------------------------------------------------------------------
# D7. Fallback when the screen returns little or nothing
# ---------------------------------------------------------------------------

#: If fewer than this many features clear BH, carry forward the top k by effect size.
FALLBACK_TRIGGER_BELOW: Final[int] = 3

#: Number of effect-size ranked features carried forward by the fallback.
FALLBACK_TOP_K: Final[int] = 20

#: Label attached to any feature carried by the fallback. Emitted as data, not prose.
FALLBACK_LABEL: Final[str] = "effect-size ranked, did not clear FDR"

#: Label attached to any feature that cleared BH.
SURVIVOR_LABEL: Final[str] = "cleared FDR"

# ---------------------------------------------------------------------------
# D8. SPC
# ---------------------------------------------------------------------------

#: Maximum number of sensors charted, ranked by absolute effect size.
SPC_MAX_SENSORS: Final[int] = 10

#: I-MR constants for a moving range of length 2.
IMR_INDIVIDUALS_CONSTANT: Final[float] = 2.66
IMR_MOVING_RANGE_CONSTANT: Final[float] = 3.267

#: The Nelson rules applied. Rules 4, 7 and 8 are excluded, see DECISIONS.md D8.
NELSON_RULES_APPLIED: Final[tuple[int, ...]] = (1, 2, 3, 5, 6)

# ---------------------------------------------------------------------------
# D9. Model
# ---------------------------------------------------------------------------

#: Number of forward-chaining folds used on train for selection. The holdout takes
#: no part in selection.
N_CV_FOLDS: Final[int] = 5

#: The complete declared grid. No search beyond it.
HGB_GRID: Final[tuple[dict[str, object], ...]] = (
    {"learning_rate": 0.05, "max_leaf_nodes": 15},
    {"learning_rate": 0.05, "max_leaf_nodes": 31},
    {"learning_rate": 0.10, "max_leaf_nodes": 15},
    {"learning_rate": 0.10, "max_leaf_nodes": 31},
)

#: Fixed HistGradientBoostingClassifier settings, applied to every grid point.
HGB_FIXED: Final[dict[str, object]] = {
    "min_samples_leaf": 20,
    "max_iter": 200,
    "early_stopping": True,
    "class_weight": "balanced",
}

#: L2 logistic regression, the optional second model.
LOGIT_C: Final[float] = 1.0

# ---------------------------------------------------------------------------
# D10. Metrics
# ---------------------------------------------------------------------------

#: Operational false alarm budget: at most this share of passing lots may be flagged.
FALSE_ALARM_BUDGET_FPR: Final[float] = 0.10

#: Bootstrap resamples used for the PR-AUC interval on holdout.
N_BOOTSTRAP: Final[int] = 2000

#: Percentile bounds of the reported interval.
BOOTSTRAP_INTERVAL: Final[tuple[float, float]] = (2.5, 97.5)

# ---------------------------------------------------------------------------
# D11. Cost aware operating point
# ---------------------------------------------------------------------------

#: Base case cost of a missed failure relative to a false alarm.
COST_RATIO_BASE: Final[float] = 10.0

#: Sweep range and resolution, log spaced.
COST_RATIO_MIN: Final[float] = 1.0
COST_RATIO_MAX: Final[float] = 100.0
COST_RATIO_POINTS: Final[int] = 25
