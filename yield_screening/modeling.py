"""Predictive baseline, metrics and the cost aware operating point, per D9 to D11.

Two properties of this module carry the honesty of the reported numbers:

1. **Selection never touches the holdout.** The grid point and the decision threshold
   are both chosen from forward chaining out of fold predictions on train. The holdout
   is scored once, with the already chosen configuration.
2. **The out of fold estimate is not reported as performance.** The feature screen was
   run on all of train rather than nested inside each fold, so the out of fold PR-AUC
   is optimistic. It is used for selection only. The holdout number is the one that
   pays for the screen in full.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    BOOTSTRAP_INTERVAL,
    COST_RATIO_BASE,
    COST_RATIO_MAX,
    COST_RATIO_MIN,
    COST_RATIO_POINTS,
    FALSE_ALARM_BUDGET_FPR,
    HGB_FIXED,
    HGB_GRID,
    LOGIT_C,
    N_BOOTSTRAP,
    N_CV_FOLDS,
    RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def make_hgb(params: dict[str, object] | None = None):
    """Class weighted histogram gradient boosting, the declared primary model."""
    settings = dict(HGB_FIXED)
    settings.update(params or {})
    return HistGradientBoostingClassifier(random_state=RANDOM_SEED, **settings)


def make_logit():
    """L2 penalised logistic regression on standardised features, the second model.

    The scaler lives inside the pipeline so that it is refitted within each training
    fold and never sees the data it is scoring.
    """
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                # L2 is the default penalty. Passing it explicitly is deprecated from
                # scikit-learn 1.8, so the default is relied on and named here instead.
                LogisticRegression(
                    C=LOGIT_C,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Forward chaining out of fold predictions on train
# ---------------------------------------------------------------------------


def forward_chaining_oof(
    factory, X: pd.DataFrame, y: np.ndarray, n_splits: int = N_CV_FOLDS
) -> tuple[np.ndarray, np.ndarray]:
    """Out of fold probabilities from a forward chaining split.

    Returns (probabilities, mask) where `mask` marks the rows that were ever in a
    validation fold. TimeSeriesSplit never validates on the first block, so those
    rows have no out of fold prediction and must be excluded rather than filled.
    """
    probabilities = np.full(len(y), np.nan, dtype=float)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, validate_idx in splitter.split(X):
        model = factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        probabilities[validate_idx] = model.predict_proba(X.iloc[validate_idx])[:, 1]
    mask = ~np.isnan(probabilities)
    return probabilities, mask


def select_grid_point(X: pd.DataFrame, y: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Pick the declared grid point with the best out of fold PR-AUC on train."""
    rows = []
    for params in HGB_GRID:
        probabilities, mask = forward_chaining_oof(lambda p=params: make_hgb(p), X, y)
        score = average_precision_score(y[mask], probabilities[mask])
        rows.append({**params, "oof_pr_auc": float(score)})
    table = pd.DataFrame(rows).sort_values("oof_pr_auc", ascending=False, kind="stable")
    best = {k: v for k, v in table.iloc[0].items() if k != "oof_pr_auc"}
    # Grid values are declared as floats and ints; restore int typing after the frame.
    if "max_leaf_nodes" in best:
        best["max_leaf_nodes"] = int(best["max_leaf_nodes"])
    return best, table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recall_at_fpr_budget(
    y: np.ndarray, proba: np.ndarray, budget: float = FALSE_ALARM_BUDGET_FPR
) -> tuple[float, float]:
    """Best recall achievable without exceeding the false alarm budget.

    Returns (recall, threshold). The budget is a false positive rate, that is, the
    share of passing lots flagged.
    """
    fpr, tpr, thresholds = roc_curve(y, proba)
    affordable = fpr <= budget
    if not affordable.any():
        return 0.0, float("inf")
    best = int(np.argmax(np.where(affordable, tpr, -1.0)))
    return float(tpr[best]), float(thresholds[best])


def bootstrap_pr_auc(
    y: np.ndarray,
    proba: np.ndarray,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    interval: tuple[float, float] = BOOTSTRAP_INTERVAL,
) -> tuple[float, float, np.ndarray]:
    """Percentile interval for PR-AUC, stratified so the class counts are preserved.

    Stratifying matters at this positive count: an unstratified resample would
    sometimes contain very few failures, and the resulting spread would describe the
    resampling scheme as much as the model.
    """
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    positive_idx = np.flatnonzero(y == 1)
    negative_idx = np.flatnonzero(y == 0)

    scores = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        take = np.concatenate(
            [
                rng.choice(positive_idx, size=positive_idx.size, replace=True),
                rng.choice(negative_idx, size=negative_idx.size, replace=True),
            ]
        )
        scores[b] = average_precision_score(y[take], proba[take])
    low, high = np.percentile(scores, interval)
    return float(low), float(high), scores


@dataclass
class HoldoutMetrics:
    """Everything reported for one model on the holdout."""

    name: str
    pr_auc: float
    pr_auc_ci: tuple[float, float]
    roc_auc: float
    no_skill: float
    recall_at_budget: float
    threshold_at_budget: float
    accuracy_floor: float


def evaluate(name: str, y: np.ndarray, proba: np.ndarray) -> HoldoutMetrics:
    """Compute the declared metric set. PR-AUC leads, ROC-AUC is reported not headlined."""
    y = np.asarray(y).astype(int)
    low, high, _ = bootstrap_pr_auc(y, proba)
    recall, threshold = recall_at_fpr_budget(y, proba)
    return HoldoutMetrics(
        name=name,
        pr_auc=float(average_precision_score(y, proba)),
        pr_auc_ci=(low, high),
        roc_auc=float(roc_auc_score(y, proba)),
        no_skill=float(y.mean()),
        recall_at_budget=recall,
        threshold_at_budget=threshold,
        accuracy_floor=float(1.0 - y.mean()),
    )


# ---------------------------------------------------------------------------
# Cost aware operating point
# ---------------------------------------------------------------------------


def expected_cost(
    y: np.ndarray, proba: np.ndarray, threshold: float, ratio: float
) -> float:
    """Cost per record, with a false alarm costing 1 and a missed failure `ratio`."""
    predicted = proba >= threshold
    false_negative = int(np.sum((~predicted) & (y == 1)))
    false_positive = int(np.sum(predicted & (y == 0)))
    return float((ratio * false_negative + false_positive) / len(y))


def select_threshold(
    y: np.ndarray, proba: np.ndarray, ratio: float = COST_RATIO_BASE
) -> float:
    """Cost minimising threshold, chosen from the supplied (train) predictions.

    Candidates are the distinct predicted probabilities, so the search covers every
    achievable confusion matrix and no grid resolution is imposed.
    """
    candidates = np.unique(np.concatenate([[0.0], proba, [1.0 + 1e-9]]))
    costs = np.array([expected_cost(y, proba, t, ratio) for t in candidates])
    return float(candidates[int(np.argmin(costs))])


def cost_sweep(
    y_train: np.ndarray,
    proba_train: np.ndarray,
    y_holdout: np.ndarray,
    proba_holdout: np.ndarray,
) -> pd.DataFrame:
    """Selected threshold against cost ratio, with the holdout consequence of each.

    The threshold is always selected on train and then applied to holdout, at every
    ratio. The sweep is the deliverable: a reader whose real ratio is 40 to 1 must be
    able to find their own operating point rather than inherit the base case.
    """
    ratios = np.geomspace(COST_RATIO_MIN, COST_RATIO_MAX, COST_RATIO_POINTS)
    rows = []
    for ratio in ratios:
        threshold = select_threshold(y_train, proba_train, ratio)
        predicted = proba_holdout >= threshold
        true_positive = int(np.sum(predicted & (y_holdout == 1)))
        false_positive = int(np.sum(predicted & (y_holdout == 0)))
        false_negative = int(np.sum((~predicted) & (y_holdout == 1)))
        n_positive = int(np.sum(y_holdout == 1))
        n_negative = int(np.sum(y_holdout == 0))
        rows.append(
            {
                "cost ratio": round(float(ratio), 2),
                "threshold": round(threshold, 4),
                "lots flagged": int(predicted.sum()),
                "failures caught": true_positive,
                "failures missed": false_negative,
                "false alarms": false_positive,
                "recall": round(true_positive / n_positive, 4) if n_positive else float("nan"),
                "precision": round(true_positive / predicted.sum(), 4)
                if predicted.sum()
                else float("nan"),
                "false alarm rate": round(false_positive / n_negative, 4)
                if n_negative
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------


def associational_importance(
    model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 20
) -> pd.DataFrame:
    """Permutation importance, by average precision.

    **This ranks association, not causation.** Permuting a column measures how much
    the fitted model leans on it, which reflects both the process and every
    correlation among the sensors. A sensor can rank high because it proxies another,
    and a genuinely important sensor can rank low because a correlated twin covers
    for it.
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
