"""Column filtering, imputation and missingness indicators, per DECISIONS.md D2 and D3.

Everything here is fitted on train rows only. `fit` is the only function that looks at
data to derive a parameter; `transform` applies stored parameters and reads nothing
from its input except the values it is transforming. That separation is what the fit
scope test checks, and it is the reason the class stores plain values rather than
holding a reference to the training frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import (
    DENSE_BAND_CEILING,
    INDICATOR_SUFFIX,
    MISSING_RATE_HARD_CEILING,
    MODAL_SHARE_CEILING,
)

DROP_REASONS = (
    "all missing on train",
    "constant on train",
    "near constant on train",
    "train missing rate above hard ceiling",
    "duplicate of an earlier sensor",
    "sparse band, values dropped, indicator retained",
    "sparse band, indicator degenerate",
)


@dataclass(frozen=True)
class IndicatorGroup:
    """A set of sparse sensors sharing one missingness pattern on train."""

    representative: str
    members: tuple[str, ...]
    train_missing_rate: float

    @property
    def column_name(self) -> str:
        return f"{self.representative}{INDICATOR_SUFFIX}"


@dataclass
class Preprocessor:
    """Fitted column filter, imputer and indicator builder.

    Attributes
    ----------
    dense_columns : list[str]
        Sensors whose values are kept and median imputed.
    dense_medians : pd.Series
        Train medians for `dense_columns`, the only fitted numeric parameters.
    indicator_groups : list[IndicatorGroup]
        Deduplicated sparse band missingness patterns.
    drop_log : pd.DataFrame
        One row per dropped sensor with the reason, so the filter is auditable.
    """

    dense_columns: list[str] = field(default_factory=list)
    dense_medians: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    indicator_groups: list[IndicatorGroup] = field(default_factory=list)
    drop_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    is_fitted: bool = False

    # -- fitting ----------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame) -> "Preprocessor":
        """Derive every parameter from train rows only.

        The rules are applied in the order declared in DECISIONS.md D2, then the
        surviving sensors are assigned to the dense or sparse band per D3.
        """
        n = len(X_train)
        if n == 0:
            raise ValueError("cannot fit on an empty frame")

        dropped: dict[str, str] = {}
        alive = list(X_train.columns)

        missing_rate = X_train.isna().mean()

        # Rule 1 and 2: all missing, then constant.
        n_unique = X_train.nunique(dropna=True)
        for col in list(alive):
            if n_unique[col] == 0:
                dropped[col] = "all missing on train"
            elif n_unique[col] <= 1:
                dropped[col] = "constant on train"
        alive = [c for c in alive if c not in dropped]

        # Rule 3: near constant by modal share of non-null train values.
        for col in list(alive):
            values = X_train[col].dropna()
            share = float(values.value_counts().iloc[0] / len(values))
            if share >= MODAL_SHARE_CEILING:
                dropped[col] = "near constant on train"
        alive = [c for c in alive if c not in dropped]

        # Rule 4: hard missing ceiling.
        for col in list(alive):
            if missing_rate[col] > MISSING_RATE_HARD_CEILING:
                dropped[col] = "train missing rate above hard ceiling"
        alive = [c for c in alive if c not in dropped]

        # Rule 5: exact duplicates, keeping the first in column order.
        seen: dict[bytes, str] = {}
        for col in list(alive):
            key = X_train[col].to_numpy(dtype=np.float64).tobytes()
            if key in seen:
                dropped[col] = "duplicate of an earlier sensor"
            else:
                seen[key] = col
        alive = [c for c in alive if c not in dropped]

        # D3: band assignment.
        dense = [c for c in alive if missing_rate[c] <= DENSE_BAND_CEILING]
        sparse = [c for c in alive if missing_rate[c] > DENSE_BAND_CEILING]

        self.dense_columns = dense
        self.dense_medians = X_train[dense].median(skipna=True)
        if self.dense_medians.isna().any():
            bad = list(self.dense_medians.index[self.dense_medians.isna()])
            raise ValueError(f"dense band sensors have no train median: {bad}")

        # Sparse band: values dropped, missingness pattern retained and deduplicated.
        pattern_to_members: dict[bytes, list[str]] = {}
        for col in sparse:
            key = X_train[col].isna().to_numpy().tobytes()
            pattern_to_members.setdefault(key, []).append(col)
            dropped[col] = "sparse band, values dropped, indicator retained"

        groups: list[IndicatorGroup] = []
        for key, members in pattern_to_members.items():
            pattern = np.frombuffer(key, dtype=bool)
            share = float(max(pattern.mean(), 1.0 - pattern.mean()))
            representative = sorted(members)[0]
            if share >= MODAL_SHARE_CEILING:
                for col in members:
                    dropped[col] = "sparse band, indicator degenerate"
                continue
            groups.append(
                IndicatorGroup(
                    representative=representative,
                    members=tuple(sorted(members)),
                    train_missing_rate=float(pattern.mean()),
                )
            )
        self.indicator_groups = sorted(groups, key=lambda g: g.representative)

        self.drop_log = pd.DataFrame(
            sorted(({"sensor": k, "reason": v} for k, v in dropped.items()),
                   key=lambda r: r["sensor"])
        )
        self.is_fitted = True
        return self

    # -- application ------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored parameters. Derives nothing from `X`.

        The only things read from `X` are the values of the selected columns and
        their missing positions. No statistic of `X` influences the output, which is
        what makes it safe to call this on the holdout.
        """
        if not self.is_fitted:
            raise RuntimeError("transform called before fit")

        # fillna with a stored Series maps column to column by label, so the train
        # median of a sensor is never applied to a different sensor.
        dense = X[self.dense_columns].fillna(self.dense_medians)

        # Built in one concat rather than column by column, which keeps the frame
        # unfragmented and avoids a pandas PerformanceWarning.
        if self.indicator_groups:
            indicators = pd.DataFrame(
                {
                    group.column_name: X[group.representative].isna().astype(np.int8)
                    for group in self.indicator_groups
                },
                index=X.index,
            )
            out = pd.concat([dense, indicators], axis=1)
        else:
            out = dense.copy()

        if out.isna().any().any():
            bad = list(out.columns[out.isna().any()])
            raise ValueError(f"transform left missing values in: {bad}")
        return out

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)

    # -- description ------------------------------------------------------------

    @property
    def indicator_columns(self) -> list[str]:
        return [g.column_name for g in self.indicator_groups]

    @property
    def feature_names(self) -> list[str]:
        return list(self.dense_columns) + self.indicator_columns

    def fitted_parameters(self) -> dict[str, object]:
        """A hashable snapshot of every fitted parameter, for the fit scope test."""
        return {
            "dense_columns": tuple(self.dense_columns),
            "dense_medians": tuple(
                (str(k), float(v)) for k, v in self.dense_medians.items()
            ),
            "indicator_groups": tuple(
                (g.representative, g.members, g.train_missing_rate)
                for g in self.indicator_groups
            ),
        }
