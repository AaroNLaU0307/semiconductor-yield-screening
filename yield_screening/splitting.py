"""Chronological train and holdout split, per DECISIONS.md D1.

There is exactly one split function in this project and it is this one. No random
splitter and no shuffling KFold is imported anywhere, which is the cheapest available
guarantee that the leak this project is built to avoid cannot be reintroduced by a
convenient one line change somewhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TRAIN_FRACTION


@dataclass(frozen=True)
class Split:
    """Index arrays and their descriptive facts.

    Attributes
    ----------
    train_idx, holdout_idx : np.ndarray
        Positional indices into the timestamp sorted frame.
    last_train_timestamp, first_holdout_timestamp : pd.Timestamp
        The boundary, reported so it can be checked against DECISIONS.md D1.
    """

    train_idx: np.ndarray
    holdout_idx: np.ndarray
    last_train_timestamp: pd.Timestamp
    first_holdout_timestamp: pd.Timestamp

    @property
    def n_train(self) -> int:
        return int(len(self.train_idx))

    @property
    def n_holdout(self) -> int:
        return int(len(self.holdout_idx))


def chronological_split(
    timestamp: pd.Series, fraction: float = TRAIN_FRACTION
) -> Split:
    """Split by position after asserting the timestamps are already sorted.

    Parameters
    ----------
    timestamp : pd.Series
        Measurement timestamps, expected ascending. `load_secom` guarantees this.
    fraction : float
        Share of records assigned to train.

    Raises
    ------
    ValueError
        If the timestamps are not ascending, if the split would leave either side
        empty, or if a single timestamp value straddles the boundary. The last case
        would put two records with the same timestamp on opposite sides, which is a
        small but real leak, and it is cheaper to fail loudly than to allow it.
    """
    if not timestamp.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted ascending before splitting")

    n = len(timestamp)
    n_train = int(np.floor(n * fraction))
    if n_train < 1 or n_train >= n:
        raise ValueError(f"fraction {fraction} leaves an empty partition for n={n}")

    last_train = timestamp.iloc[n_train - 1]
    first_holdout = timestamp.iloc[n_train]
    if not first_holdout > last_train:
        raise ValueError(
            "a single timestamp straddles the split boundary: "
            f"last train {last_train}, first holdout {first_holdout}"
        )

    return Split(
        train_idx=np.arange(n_train),
        holdout_idx=np.arange(n_train, n),
        last_train_timestamp=last_train,
        first_holdout_timestamp=first_holdout,
    )
