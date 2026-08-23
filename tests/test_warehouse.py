"""The safety net for the SQL layer: it must change nothing.

A warehouse was added to this project after the results were already frozen and
reported. That is only defensible if it is provably a change of source and not a
change of answer, so these tests assert two properties:

1. **Round trip identity.** Data loaded from DuckDB is bit identical to data parsed
   from the checksum verified raw files, values, labels, timestamps and row order.
2. **Rule agreement.** The declared drop rules computed in SQL produce the same
   verdict, the same reason, and the same underlying statistics as the pandas
   `Preprocessor`, sensor by sensor across all 590.

If either fails, a reported number has moved and the SQL layer is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_screening import warehouse as wh
from yield_screening.data_io import load_secom
from yield_screening.preprocess import Preprocessor
from yield_screening.splitting import chronological_split


@pytest.fixture(scope="module")
def raw():
    """The raw file parse, which is the reference the warehouse must reproduce."""
    return load_secom()


@pytest.fixture(scope="module")
def from_db(raw):
    """The warehouse parse, built from the same verified files."""
    wh.build_warehouse(data=raw)
    return wh.load_secom_from_warehouse()


@pytest.fixture(scope="module")
def fitted(raw):
    """The pandas preprocessor fitted on train, the reference for the SQL rules."""
    split = chronological_split(raw.timestamp)
    return Preprocessor().fit(raw.X.iloc[split.train_idx]), split


# ---------------------------------------------------------------------------
# 1. Round trip identity
# ---------------------------------------------------------------------------


def test_warehouse_values_are_bit_identical_to_the_raw_parse(raw, from_db) -> None:
    """Every sensor value survives the round trip through DuckDB unchanged."""
    assert list(from_db.X.columns) == list(raw.X.columns)
    assert from_db.X.shape == raw.X.shape
    assert np.array_equal(
        from_db.X.to_numpy(), raw.X.to_numpy(), equal_nan=True
    ), "sensor values differ between the warehouse and the raw parse"


def test_warehouse_preserves_missing_cells(raw, from_db) -> None:
    """A NULL in the database means the same thing as a NaN in the raw parse."""
    assert np.array_equal(from_db.X.isna().to_numpy(), raw.X.isna().to_numpy())


def test_warehouse_labels_timestamps_and_row_order_match(raw, from_db) -> None:
    """Labels, timestamps and the timestamp sort order all survive the round trip."""
    assert np.array_equal(from_db.y.to_numpy(), raw.y.to_numpy())
    assert np.array_equal(from_db.timestamp.to_numpy(), raw.timestamp.to_numpy())
    assert np.array_equal(from_db.original_row.to_numpy(), raw.original_row.to_numpy())


def test_sql_partition_matches_the_python_split(raw) -> None:
    """The partition expressed as a query selects exactly the frozen split."""
    split = chronological_split(raw.timestamp)
    boundary = wh.partition_boundary().set_index("partition")

    assert int(boundary.loc["train", "n_records"]) == split.n_train == 1096
    assert int(boundary.loc["holdout", "n_records"]) == split.n_holdout == 471
    assert int(boundary.loc["train", "n_fail"]) == 78
    assert int(boundary.loc["holdout", "n_fail"]) == 26
    assert str(boundary.loc["train", "last_measured_at"]) == str(
        split.last_train_timestamp
    )
    assert str(boundary.loc["holdout", "first_measured_at"]) == str(
        split.first_holdout_timestamp
    )


# ---------------------------------------------------------------------------
# 2. Rule agreement
# ---------------------------------------------------------------------------


def test_sql_drop_rules_keep_exactly_the_same_sensors(fitted) -> None:
    """The dense band chosen in SQL is the same set of sensors as in pandas."""
    pre, _ = fitted
    sql = wh.drop_rules().set_index("sensor_id")
    assert set(sql.index[sql["is_dense_band"]]) == set(pre.dense_columns)


def test_sql_drop_rules_give_the_same_reason_for_every_sensor(fitted) -> None:
    """Not just the same verdict: the same declared rule fires for each sensor."""
    pre, _ = fitted
    sql = wh.drop_rules().set_index("sensor_id")
    python_reason = dict(zip(pre.drop_log["sensor"], pre.drop_log["reason"]))

    mismatches = []
    for sensor, row in sql.iterrows():
        in_sql = row["drop_reason"]
        in_sql = None if in_sql is None or pd.isna(in_sql) else in_sql
        if in_sql != python_reason.get(sensor):
            mismatches.append((sensor, in_sql, python_reason.get(sensor)))
    assert not mismatches, f"drop reason disagreements: {mismatches[:5]}"


def test_sql_sensor_statistics_are_exactly_equal_to_pandas(raw, fitted) -> None:
    """Missing rate, distinct count and modal share agree to the last bit.

    Exact equality rather than a tolerance: both compute the same integer ratio in
    IEEE 754 double precision, so any difference would mean a different computation,
    not a rounding difference.
    """
    _, split = fitted
    X_train = raw.X.iloc[split.train_idx]
    sql = wh.sensor_stats("train").set_index("sensor_id")

    expected_missing = X_train.isna().mean().reindex(sql.index)
    expected_distinct = X_train.nunique(dropna=True).reindex(sql.index)

    assert np.array_equal(sql["missing_rate"].to_numpy(), expected_missing.to_numpy())
    assert np.array_equal(
        sql["n_distinct_values"].to_numpy(), expected_distinct.to_numpy()
    )

    expected_modal = {}
    for column in X_train.columns:
        values = X_train[column].dropna()
        expected_modal[column] = (
            float(values.value_counts().iloc[0] / len(values)) if len(values) else np.nan
        )
    expected_modal = pd.Series(expected_modal).reindex(sql.index)
    both_present = sql["modal_share"].notna() & expected_modal.notna()
    assert np.array_equal(
        sql["modal_share"][both_present].to_numpy(),
        expected_modal[both_present].to_numpy(),
    )


def test_sql_missingness_patterns_match_the_python_grouping(fitted) -> None:
    """The 8 discovered patterns, their members and their coverage all agree."""
    pre, _ = fitted
    sql = wh.missingness_patterns()

    assert len(sql) == len(pre.indicator_groups)
    assert list(sql["representative"]) == [g.representative for g in pre.indicator_groups]
    assert [sorted(m) for m in sql["members"]] == [
        sorted(g.members) for g in pre.indicator_groups
    ]
    assert np.array_equal(
        sql["train_missing_rate"].to_numpy(),
        np.array([g.train_missing_rate for g in pre.indicator_groups]),
    )


def test_sql_monthly_failure_rate_matches_pandas(raw) -> None:
    """The monthly aggregation moved to SQL returns the same rates."""
    sql = wh.monthly_failure_rate()
    expected = (
        pd.DataFrame(
            {
                "month": raw.timestamp.dt.to_period("M").astype(str),
                "y": raw.y.to_numpy(),
            }
        )
        .groupby("month")["y"]
        .agg(n_records="size", n_fail="sum")
        .reset_index()
    )
    expected["fail_rate"] = expected["n_fail"] / expected["n_records"]

    assert list(sql["month"]) == list(expected["month"])
    assert np.array_equal(
        sql["n_records"].to_numpy().astype(int), expected["n_records"].to_numpy()
    )
    assert np.array_equal(
        sql["n_fail"].to_numpy().astype(int), expected["n_fail"].to_numpy()
    )
    assert np.array_equal(sql["fail_rate"].to_numpy(), expected["fail_rate"].to_numpy())


# ---------------------------------------------------------------------------
# 3. Guards
# ---------------------------------------------------------------------------


def test_monthly_sensor_means_rejects_unexpected_identifiers() -> None:
    """Sensor ids are interpolated into SQL, so they are validated rather than trusted."""
    with pytest.raises(ValueError, match="refusing to interpolate"):
        wh.monthly_sensor_means(["sensor_059'; DROP TABLE lot; --"])


def test_sensor_stats_rejects_an_unknown_partition() -> None:
    """A typo in the partition name must fail loudly, not silently scan everything."""
    with pytest.raises(ValueError, match="unknown partition"):
        wh.sensor_stats("trian")


def test_drop_rules_are_always_train_scoped() -> None:
    """A fitted rule may never see the holdout, so the query hard codes the train scope."""
    assert "p.partition = 'train'" in wh.drop_rules_sql()
    assert "p.partition = 'holdout'" not in wh.drop_rules_sql()
