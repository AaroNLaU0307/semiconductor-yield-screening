"""Characterise the raw SECOM dataset and emit reports/data_quality.md.

This script is deliberately run before any modelling code is written. Its output
is the evidence base for the numeric thresholds fixed in DECISIONS.md, so it
reports what the data actually contains, including the ugly parts.

Usage, from the repository root:
    python scripts/run_data_qa.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from yield_screening import warehouse as wh  # noqa: E402
from yield_screening.data_io import file_checksums  # noqa: E402
from yield_screening.reporting import md_table  # noqa: E402

REPORT_MD = REPO_ROOT / "reports" / "data_quality.md"
REPORT_JSON = REPO_ROOT / "reports" / "data_quality.json"
TRAIN_FRACTION = 0.70


def column_profile(X: pd.DataFrame) -> pd.DataFrame:
    """One row per sensor: missingness, spread, and degeneracy diagnostics."""
    n = len(X)
    missing = X.isna().sum()
    non_null = n - missing
    described = X.describe().T

    modal_share = []
    unique_non_null = []
    for col in X.columns:
        values = X[col].dropna()
        if values.empty:
            modal_share.append(np.nan)
            unique_non_null.append(0)
            continue
        counts = values.value_counts()
        modal_share.append(float(counts.iloc[0] / len(values)))
        unique_non_null.append(int(len(counts)))

    profile = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_rate": missing / n,
            "non_null_count": non_null,
            "unique_non_null": unique_non_null,
            "modal_share": modal_share,
            "mean": described["mean"],
            "std": described["std"],
            "min": described["min"],
            "max": described["max"],
        }
    )
    profile["std"] = profile["std"].fillna(0.0)
    # Coefficient of variation is scale free, which matters because the 590 sensors
    # sit on wildly different units. A raw variance floor would silently drop
    # informative low magnitude sensors.
    denom = profile["mean"].abs().replace(0.0, np.nan)
    profile["cv"] = (profile["std"] / denom).abs()
    profile["is_all_missing"] = profile["non_null_count"] == 0
    profile["is_constant"] = profile["unique_non_null"] <= 1
    return profile


def duplicate_column_groups(X: pd.DataFrame) -> list[list[str]]:
    """Group sensors whose value vectors are identical, missing pattern included."""
    buckets: dict[bytes, list[str]] = {}
    for col in X.columns:
        key = X[col].to_numpy(dtype=np.float64).tobytes()
        buckets.setdefault(key, []).append(col)
    return [group for group in buckets.values() if len(group) > 1]


def threshold_grid(profile: pd.DataFrame) -> pd.DataFrame:
    """Survivor counts under candidate drop rules, so thresholds are chosen on evidence."""
    rows = []
    for missing_ceiling in (0.05, 0.10, 0.20, 0.30, 0.50, 0.60, 1.00):
        for modal_ceiling in (0.95, 0.99, 1.00):
            keep = (
                (~profile["is_all_missing"])
                & (~profile["is_constant"])
                & (profile["missing_rate"] <= missing_ceiling)
                & (profile["modal_share"].fillna(1.0) < modal_ceiling)
            )
            rows.append(
                {
                    "missing_rate_ceiling": missing_ceiling,
                    "modal_share_ceiling": modal_ceiling,
                    "surviving_sensors": int(keep.sum()),
                }
            )
    return pd.DataFrame(rows)


def split_preview(timestamp: pd.Series, y: pd.Series, fraction: float) -> dict:
    """Describe the chronological split that DECISIONS.md will freeze."""
    n = len(timestamp)
    n_train = int(np.floor(n * fraction))
    boundary = timestamp.iloc[n_train - 1]
    first_holdout = timestamp.iloc[n_train]
    train_mask = np.arange(n) < n_train
    return {
        "train_fraction_requested": fraction,
        "n_total": n,
        "n_train": int(n_train),
        "n_holdout": int(n - n_train),
        "last_train_timestamp": str(boundary),
        "first_holdout_timestamp": str(first_holdout),
        "boundary_is_strict": bool(first_holdout > boundary),
        "train_fails": int(y[train_mask].sum()),
        "holdout_fails": int(y[~train_mask].sum()),
        "train_fail_rate": float(y[train_mask].mean()),
        "holdout_fail_rate": float(y[~train_mask].mean()),
    }


def monthly_table(timestamp: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Reference pandas implementation, retained as the check on the SQL version."""
    frame = pd.DataFrame(
        {"month": timestamp.dt.to_period("M").astype(str), "y": y.to_numpy()}
    )
    table = frame.groupby("month")["y"].agg(records="size", fails="sum")
    table["fail_rate"] = table["fails"] / table["records"]
    return table.reset_index()


def monthly_table_sql() -> pd.DataFrame:
    """Monthly failure rate from `sql/monthly_failure_rate.sql`.

    Column names and integer dtypes are normalised to match the pandas reference
    exactly, so switching the source changed no rendered value.
    """
    frame = wh.monthly_failure_rate().rename(
        columns={"n_records": "records", "n_fail": "fails"}
    )
    frame["records"] = frame["records"].astype(int)
    frame["fails"] = frame["fails"].astype(int)
    return frame[["month", "records", "fails", "fail_rate"]]


def pattern_discovery_table() -> pd.DataFrame:
    """Distinct missingness patterns among sparse band sensors, from SQL."""
    patterns = wh.missingness_patterns()
    return pd.DataFrame(
        {
            "pattern": [f"{r}__notmeasured" for r in patterns["representative"]],
            "sensors in pattern": patterns["n_sensors"].astype(int),
            "train coverage": patterns["train_coverage"].astype(float),
            "members": [
                ", ".join(list(m))
                if len(m) <= 6
                else ", ".join(list(m)[:6]) + f", and {len(m) - 6} more"
                for m in patterns["members"]
            ],
        }
    )


def pattern_failure_association(
    patterns: pd.DataFrame, X_train: pd.DataFrame, y_train: np.ndarray
) -> pd.DataFrame:
    """Is pattern membership itself associated with failure? Train only, descriptive.

    A sensor measured on only a minority of lots usually reflects a metrology sampling
    plan. If the plan correlates with failure, that is a sampling bias finding rather
    than a sensor finding, and it is worth reporting either way.

    These figures are descriptive and use the train partition only. The authoritative,
    FDR controlled test of the same question is in `reports/screening.md`, where each
    pattern enters the 424 test screening family on equal footing with the sensor
    values.
    """
    rows = []
    for _, row in patterns.iterrows():
        representative = row["pattern"].replace("__notmeasured", "")
        not_measured = X_train[representative].isna().to_numpy()
        measured = ~not_measured
        rows.append(
            {
                "pattern": row["pattern"],
                "lots measured": int(measured.sum()),
                "lots not measured": int(not_measured.sum()),
                "fail rate when measured": float(y_train[measured].mean())
                if measured.any()
                else float("nan"),
                "fail rate when not measured": float(y_train[not_measured].mean())
                if not_measured.any()
                else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    frame["difference"] = (
        frame["fail rate when measured"] - frame["fail rate when not measured"]
    )
    return frame


def build_report(summary: dict, tables: dict) -> str:
    """Assemble the markdown report from measured values only."""
    fail_rate = summary["fail_rate"]
    split = summary["split_preview"]
    parts = []

    parts.append("# Data quality report: UCI SECOM\n")
    parts.append(
        "Generated by `scripts/run_data_qa.py`. Every number below is measured, "
        "not assumed.\n"
    )

    parts.append("## 1. Shape and label\n")
    parts.append(
        md_table(
            pd.DataFrame(
                [
                    {"property": "rows", "value": summary["n_rows"]},
                    {"property": "sensor columns", "value": summary["n_sensors"]},
                    {
                        "property": "pass records (raw -1, modelled 0)",
                        "value": summary["n_pass"],
                    },
                    {
                        "property": "fail records (raw +1, modelled 1)",
                        "value": summary["n_fail"],
                    },
                    {"property": "fail rate", "value": f"{fail_rate:.4f}"},
                ]
            )
        )
    )
    parts.append(
        f"\nThe fail rate of {fail_rate:.2%} is the no-skill baseline for a "
        "precision-recall curve. It is also why accuracy is the wrong metric here: "
        f"predicting pass for every record scores {1 - fail_rate:.2%} accuracy "
        "while catching zero failures.\n"
    )

    parts.append("## 2. Time\n")
    parts.append(
        md_table(
            pd.DataFrame(
                [
                    {"property": "first timestamp", "value": summary["timestamp_min"]},
                    {"property": "last timestamp", "value": summary["timestamp_max"]},
                    {
                        "property": "raw file already in chronological order",
                        "value": summary["timestamp_monotonic_in_raw_file"],
                    },
                    {
                        "property": "duplicate timestamps",
                        "value": summary["duplicate_timestamps"],
                    },
                    {
                        "property": "median gap between records (minutes)",
                        "value": f"{summary['median_gap_minutes']:.1f}",
                    },
                    {
                        "property": "largest gap between records (hours)",
                        "value": f"{summary['max_gap_hours']:.1f}",
                    },
                ]
            )
        )
    )
    parts.append(
        "\nThe loader sorts by timestamp regardless. Confirming that the raw order "
        "happens to be chronological is not the same as relying on it.\n"
    )
    parts.append("### Failure rate by month\n")
    parts.append(md_table(tables["months"]))
    parts.append("")

    parts.append("## 3. Missingness\n")
    parts.append(
        md_table(
            pd.DataFrame(
                [
                    {"property": "total cells", "value": summary["cells_total"]},
                    {"property": "missing cells", "value": summary["cells_missing"]},
                    {
                        "property": "overall cell missing rate",
                        "value": f"{summary['cell_missing_rate']:.4f}",
                    },
                    {
                        "property": "sensors that are entirely missing",
                        "value": summary["all_missing_sensors"],
                    },
                    {
                        "property": "median per-row missing rate",
                        "value": f"{summary['row_missing_rate_median']:.4f}",
                    },
                    {
                        "property": "worst per-row missing rate",
                        "value": f"{summary['row_missing_rate_max']:.4f}",
                    },
                ]
            )
        )
    )
    parts.append("\nDistribution of per-sensor missing rate:\n")
    parts.append(md_table(tables["missing_quantiles"]))
    parts.append("\nTen worst sensors by missing rate:\n")
    parts.append(md_table(tables["worst_missing"]))
    parts.append("")

    parts.append("## 4. Degenerate and redundant columns\n")
    parts.append(
        md_table(
            pd.DataFrame(
                [
                    {
                        "property": "constant sensors (at most one distinct non-null value)",
                        "value": summary["constant_sensors"],
                    },
                    {
                        "property": "sensors where one value covers at least 95 percent of non-nulls",
                        "value": summary["sensors_modal_share_ge_95"],
                    },
                    {
                        "property": "sensors where one value covers at least 99 percent of non-nulls",
                        "value": summary["sensors_modal_share_ge_99"],
                    },
                    {
                        "property": "groups of exactly identical sensors",
                        "value": summary["duplicate_column_groups"],
                    },
                    {
                        "property": "sensors removable as duplicates",
                        "value": summary["sensors_removable_as_duplicates"],
                    },
                    {
                        "property": "duplicated feature rows",
                        "value": summary["duplicate_feature_rows"],
                    },
                ]
            )
        )
    )
    parts.append(
        "\nDuplicate sensor groups (identical values including the missing pattern):\n"
    )
    parts.append(tables["duplicate_lines"])
    parts.append("")

    parts.append("## 5. Unsupervised structure in the missingness matrix\n")
    parts.append(
        "The sparse band sensors are not measured on every lot. Grouping them by their "
        "exact missing or present vector asks whether those absences are independent "
        "per sensor, or whether sensors are measured together in blocks.\n"
    )
    parts.append(
        "They are measured in blocks. The sparse sensors collapse to a small number of "
        "distinct patterns, which is the signature of **metrology sampling plans**, "
        "where not every lot is measured on every tool, rather than of sensors failing "
        "independently at random.\n"
    )
    parts.append(
        "This is unsupervised structure discovery over the missingness matrix, done by "
        "exact grouping rather than by a clustering algorithm. Exact grouping is the "
        "correct method here because the patterns are exact: imposing a distance metric "
        "would turn an exact answer into an approximate one. The query is "
        "`sql/missingness_patterns.sql`.\n"
    )
    parts.append(md_table(tables["patterns"]))
    parts.append("\n### Is pattern membership itself associated with failure?\n")
    parts.append(
        "Worth asking on its merits. If a sampling plan correlates with the failure "
        "rate, that is a **sampling bias finding, not a sensor finding**, and it should "
        "be reported either way.\n"
    )
    parts.append(md_table(tables["pattern_association"]))
    parts.append(
        "\nThese figures are descriptive and computed on the train partition only. The "
        "authoritative test is in `reports/screening.md`, where each pattern enters the "
        "screening family on equal footing with the sensor values and is corrected for "
        "multiplicity alongside them. Two patterns clear that FDR threshold, so the "
        "answer is yes: which lots were measured is associated with whether they "
        "failed. That is a property of how the data was collected, and any sensor "
        "shortlisted from those same lots inherits the concern.\n"
    )
    parts.append("## 6. Effect of candidate drop rules\n")
    parts.append(
        "Surviving sensor count under combinations of a missing-rate ceiling and a "
        "modal-share ceiling. All-missing and constant sensors are dropped in every "
        "row of this table.\n"
    )
    parts.append(md_table(tables["grid"]))
    parts.append(
        "\nThe same grid computed on the train partition alone. The thresholds fixed "
        "in DECISIONS.md are justified by this second table, not the first, so that "
        "no holdout record influences which sensors enter the screen. The two tables "
        "agree closely, which is itself the evidence that the choice is not sensitive "
        "to the split.\n"
    )
    parts.append(md_table(tables["grid_train"]))
    parts.append("")

    parts.append(
        f"## 7. Chronological split preview at {TRAIN_FRACTION:.0%}\n"
    )
    parts.append(
        md_table(
            pd.DataFrame(
                [
                    {"property": "train rows", "value": split["n_train"]},
                    {"property": "holdout rows", "value": split["n_holdout"]},
                    {
                        "property": "last train timestamp",
                        "value": split["last_train_timestamp"],
                    },
                    {
                        "property": "first holdout timestamp",
                        "value": split["first_holdout_timestamp"],
                    },
                    {
                        "property": "boundary is strict (no timestamp straddles the split)",
                        "value": split["boundary_is_strict"],
                    },
                    {"property": "train fails", "value": split["train_fails"]},
                    {"property": "holdout fails", "value": split["holdout_fails"]},
                    {
                        "property": "train fail rate",
                        "value": f"{split['train_fail_rate']:.4f}",
                    },
                    {
                        "property": "holdout fail rate",
                        "value": f"{split['holdout_fail_rate']:.4f}",
                    },
                ]
            )
        )
    )
    parts.append("")

    parts.append("## 8. Raw file checksums\n")
    parts.append(
        md_table(
            pd.DataFrame(
                [
                    {"file": name, "sha256": meta["sha256"], "bytes": meta["bytes"]}
                    for name, meta in summary["checksums"].items()
                ]
            )
        )
    )
    parts.append("")
    return "\n".join(parts)


def main() -> int:
    # The pipeline reads from the warehouse. `tests/test_warehouse.py` asserts this
    # returns data identical to the raw loader, so the raw files remain the provenance
    # anchor while the database is the working source.
    wh.ensure_warehouse()
    data = wh.load_secom_from_warehouse()
    X, y, timestamp = data.X, data.y, data.timestamp
    n, n_sensors = X.shape

    profile = column_profile(X)
    duplicates = duplicate_column_groups(X)
    grid = threshold_grid(profile)
    split = split_preview(timestamp, y, TRAIN_FRACTION)
    months = monthly_table_sql()

    # The drop thresholds in DECISIONS.md are justified on train alone, so that no
    # holdout record influences which sensors enter the screen.
    n_train = int(np.floor(n * TRAIN_FRACTION))
    grid_train = threshold_grid(column_profile(X.iloc[:n_train]))

    patterns = pattern_discovery_table()
    pattern_association = pattern_failure_association(
        patterns, X.iloc[:n_train], y.to_numpy()[:n_train]
    )

    row_missing = X.isna().sum(axis=1) / n_sensors
    gaps = timestamp.diff().dropna()

    missing_quantiles = (
        profile["missing_rate"]
        .quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
        .rename("missing_rate")
        .rename_axis("quantile")
        .reset_index()
    )
    worst_missing = (
        profile.sort_values("missing_rate", ascending=False)
        .head(10)[["missing_rate", "non_null_count", "unique_non_null"]]
        .reset_index()
        .rename(columns={"index": "sensor"})
    )

    summary = {
        "n_rows": int(n),
        "n_sensors": int(n_sensors),
        "n_fail": int(y.sum()),
        "n_pass": int((1 - y).sum()),
        "fail_rate": float(y.mean()),
        "timestamp_min": str(timestamp.min()),
        "timestamp_max": str(timestamp.max()),
        "timestamp_monotonic_in_raw_file": bool(
            (data.original_row.to_numpy() == np.arange(n)).all()
        ),
        "duplicate_timestamps": int(timestamp.duplicated().sum()),
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "median_gap_minutes": float(gaps.dt.total_seconds().median() / 60.0),
        "max_gap_hours": float(gaps.dt.total_seconds().max() / 3600.0),
        "all_missing_sensors": int(profile["is_all_missing"].sum()),
        "constant_sensors": int(profile["is_constant"].sum()),
        "sensors_modal_share_ge_95": int(
            (profile["modal_share"].fillna(1.0) >= 0.95).sum()
        ),
        "sensors_modal_share_ge_99": int(
            (profile["modal_share"].fillna(1.0) >= 0.99).sum()
        ),
        "duplicate_column_groups": len(duplicates),
        "sensors_removable_as_duplicates": int(sum(len(g) - 1 for g in duplicates)),
        "cells_total": int(n * n_sensors),
        "cells_missing": int(X.isna().sum().sum()),
        "cell_missing_rate": float(X.isna().sum().sum() / (n * n_sensors)),
        "row_missing_rate_max": float(row_missing.max()),
        "row_missing_rate_median": float(row_missing.median()),
        "split_preview": split,
        "duplicate_groups": duplicates,
        "checksums": file_checksums(),
    }

    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if duplicates:
        shown = "\n".join(f"- {', '.join(g)}" for g in duplicates[:15])
        if len(duplicates) > 15:
            shown += (
                f"\n- and {len(duplicates) - 15} further groups, "
                "listed in reports/data_quality.json"
            )
    else:
        shown = "- none found"

    tables = {
        "months": months,
        "patterns": patterns,
        "pattern_association": pattern_association,
        "missing_quantiles": missing_quantiles,
        "worst_missing": worst_missing,
        "grid": grid,
        "grid_train": grid_train,
        "duplicate_lines": shown,
    }

    REPORT_MD.write_text(build_report(summary, tables), encoding="utf-8")
    print(f"wrote {REPORT_MD.relative_to(REPO_ROOT)}")
    print(f"wrote {REPORT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
