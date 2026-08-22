"""Leak guards for the project: temporal ordering and fit scope.

Per PLAN.md's test schedule, these are two of the four required tests, and they
protect the two properties the whole pipeline depends on:

1. Temporal ordering. The chronological split (DECISIONS.md D1) must place every
   holdout record strictly after every train record, with no timestamp straddling
   the boundary. A random or off-by-one split here would let future process state
   leak into train, and the reported metrics would describe interpolation rather
   than prediction.
2. Fit scope. The preprocessor (DECISIONS.md D2, D3) must derive every parameter
   from train rows only, and `transform` must derive nothing from the rows it is
   given. If either property failed, information from holdout rows could reach the
   fitted model through the back door of column selection, medians, or missingness
   indicators, even though the model itself never sees holdout labels directly.

Both groups run against the real SECOM dataset loaded once per module, plus small
synthetic inputs where a real dataset cannot exercise an error path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yield_screening.data_io import SecomData, load_secom
from yield_screening.preprocess import Preprocessor
from yield_screening.splitting import Split, chronological_split

# ---------------------------------------------------------------------------
# Module scoped fixtures. Loading the real dataset takes a moment, so it, the
# split derived from it, and the preprocessor fitted on that split are each
# computed once per module and reused read-only by every test.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def secom_data() -> SecomData:
    """The real, parsed SECOM dataset, sorted ascending by timestamp."""
    return load_secom()


@pytest.fixture(scope="module")
def split(secom_data: SecomData) -> Split:
    """The frozen D1 chronological split of the real dataset."""
    return chronological_split(secom_data.timestamp)


@pytest.fixture(scope="module")
def fitted_preprocessor(secom_data: SecomData, split: Split) -> Preprocessor:
    """A Preprocessor fitted on the real train partition only.

    Never re-fitted by any test. Tests that need a fresh fit (to compare against
    this baseline) construct their own local `Preprocessor()` instead of mutating
    this shared fixture.
    """
    X_train = secom_data.X.iloc[split.train_idx]
    return Preprocessor().fit(X_train)


def _corrupt_holdout_rows(X: pd.DataFrame, holdout_idx: np.ndarray) -> pd.DataFrame:
    """Return a copy of X with holdout rows aggressively perturbed, train rows untouched.

    The holdout rows are split into three blocks, each given a different extreme
    corruption: multiplied by 1000, forced to NaN, and forced to a huge constant.
    Train rows are never touched. Either leak-guard property under test would be
    violated if a fitted parameter, or a train row's transformed output, moved
    because of any of this.
    """
    corrupted = X.copy()
    n_holdout = len(holdout_idx)
    third = max(n_holdout // 3, 1)
    block_a = holdout_idx[:third]
    block_b = holdout_idx[third : 2 * third]
    block_c = holdout_idx[2 * third :]

    corrupted.iloc[block_a] = corrupted.iloc[block_a] * 1000.0
    corrupted.iloc[block_b] = np.nan
    corrupted.iloc[block_c] = 1.0e12
    return corrupted


# ---------------------------------------------------------------------------
# Test group 1: temporal ordering (required test 1).
# ---------------------------------------------------------------------------


def test_every_holdout_timestamp_is_later_than_every_train_timestamp(
    secom_data: SecomData, split: Split
) -> None:
    """Protects: no holdout record's timestamp can be earlier than, or tied with,
    any train record's timestamp, on the real dataset."""
    train_ts = secom_data.timestamp.iloc[split.train_idx]
    holdout_ts = secom_data.timestamp.iloc[split.holdout_idx]
    assert train_ts.max() < holdout_ts.min()


def test_split_sizes_match_decisions(split: Split) -> None:
    """Protects: the frozen D1 split sizes and boundary timestamps cannot silently
    drift. This is a regression guard on the frozen contract, not a design check."""
    assert split.n_train == 1096
    assert split.n_holdout == 471
    assert split.last_train_timestamp == pd.Timestamp("2008-09-26 02:26:00")
    assert split.first_holdout_timestamp == pd.Timestamp("2008-09-26 03:12:00")


def test_split_indices_partition_the_frame(secom_data: SecomData, split: Split) -> None:
    """Protects: train_idx and holdout_idx are disjoint, contiguous, ascending, and
    together cover every row index in the frame exactly once."""
    n = len(secom_data.X)
    train_idx = split.train_idx
    holdout_idx = split.holdout_idx

    assert set(train_idx.tolist()).isdisjoint(holdout_idx.tolist())

    combined = np.sort(np.concatenate([train_idx, holdout_idx]))
    np.testing.assert_array_equal(combined, np.arange(n))

    assert np.all(np.diff(train_idx) == 1), "train_idx must be contiguous and ascending"
    assert np.all(np.diff(holdout_idx) == 1), "holdout_idx must be contiguous and ascending"
    assert holdout_idx[0] == train_idx[-1] + 1


def test_split_rejects_unsorted_timestamps() -> None:
    """Protects: chronological_split refuses to run on a timestamp series that is
    not already sorted ascending, rather than silently splitting by position."""
    shuffled = pd.Series(
        pd.to_datetime(
            ["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-05", "2020-01-04"]
        )
    )
    assert not shuffled.is_monotonic_increasing

    with pytest.raises(ValueError):
        chronological_split(shuffled)


def test_split_rejects_a_straddling_boundary() -> None:
    """Protects: a timestamp value duplicated across the split boundary (so the last
    train record and the first holdout record share a timestamp) must raise, rather
    than silently placing two simultaneous records on opposite sides."""
    base = pd.Timestamp("2020-01-01")
    # Ascending offsets with index 6 and 7 sharing a timestamp.
    offsets = [0, 1, 2, 3, 4, 5, 6, 6, 7, 8]
    timestamps = pd.Series([base + np.timedelta64(o, "D") for o in offsets])
    assert timestamps.is_monotonic_increasing

    fraction = 0.7
    n_train = int(np.floor(len(timestamps) * fraction))
    assert n_train == 7, "test construction check: n_train must land on the duplicated pair"
    assert timestamps.iloc[n_train - 1] == timestamps.iloc[n_train], (
        "test construction check: last train and first holdout must share a timestamp"
    )

    with pytest.raises(ValueError):
        chronological_split(timestamps, fraction=fraction)


# ---------------------------------------------------------------------------
# Test group 2: fit scope (required test 2).
# ---------------------------------------------------------------------------


def test_perturbing_holdout_rows_cannot_change_fitted_parameters(
    secom_data: SecomData, split: Split, fitted_preprocessor: Preprocessor
) -> None:
    """Protects: no perturbation of holdout rows, however extreme, can change any
    parameter a fresh Preprocessor derives by fitting on the (untouched) train rows."""
    original_snapshot = fitted_preprocessor.fitted_parameters()

    corrupted_full_frame = _corrupt_holdout_rows(secom_data.X, split.holdout_idx)
    X_train_from_corrupted = corrupted_full_frame.iloc[split.train_idx]

    # Sanity check on the corruption itself: train rows must be untouched.
    pd.testing.assert_frame_equal(
        X_train_from_corrupted, secom_data.X.iloc[split.train_idx]
    )

    refit = Preprocessor().fit(X_train_from_corrupted)
    assert refit.fitted_parameters() == original_snapshot


def test_transform_output_for_a_train_row_is_unchanged_by_holdout_corruption(
    secom_data: SecomData, split: Split, fitted_preprocessor: Preprocessor
) -> None:
    """Protects: transform derives nothing from the rows it is given, so the
    transformed output for a train row cannot depend on what holdout rows contain."""
    corrupted_full_frame = _corrupt_holdout_rows(secom_data.X, split.holdout_idx)

    original_output = fitted_preprocessor.transform(secom_data.X)
    corrupted_output = fitted_preprocessor.transform(corrupted_full_frame)

    train_idx = split.train_idx
    pd.testing.assert_frame_equal(
        original_output.iloc[train_idx], corrupted_output.iloc[train_idx]
    )


def test_transform_uses_stored_train_medians_not_input_medians(
    secom_data: SecomData, split: Split, fitted_preprocessor: Preprocessor
) -> None:
    """Protects: a missing dense value is filled from the stored train median, not
    from any statistic (such as a median) of the slice actually being transformed."""
    holdout_X = secom_data.X.iloc[split.holdout_idx]

    target_col: str | None = None
    for col in fitted_preprocessor.dense_columns:
        if holdout_X[col].isna().any():
            target_col = col
            break

    if target_col is None:
        pytest.skip("no dense column has a missing value anywhere in holdout")

    missing_mask = holdout_X[target_col].isna()
    tiny_slice = holdout_X.loc[missing_mask]
    assert len(tiny_slice) >= 1

    # Every row of this slice is missing in target_col, so the slice's own column
    # of non-null values is empty: there is no slice statistic to fall back on. If
    # transform used anything but the stored train median here, the result could
    # not be a finite number equal to it.
    transformed = fitted_preprocessor.transform(tiny_slice)
    filled = transformed[target_col]
    stored_median = float(fitted_preprocessor.dense_medians[target_col])

    assert filled.nunique() == 1
    assert float(filled.iloc[0]) == pytest.approx(stored_median)


def test_transform_before_fit_raises() -> None:
    """Protects: transform cannot be called on an unfitted Preprocessor."""
    fresh = Preprocessor()
    with pytest.raises(RuntimeError):
        fresh.transform(pd.DataFrame({"sensor_000": [1.0, 2.0]}))


def test_no_missing_values_survive_transform(
    secom_data: SecomData, fitted_preprocessor: Preprocessor
) -> None:
    """Protects: the transformed output has zero missing values, for every row in
    the full frame, not just the train rows the preprocessor was fitted on."""
    output = fitted_preprocessor.transform(secom_data.X)
    assert output.isna().sum().sum() == 0
