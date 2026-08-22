"""Loading and parsing of the raw UCI SECOM files.

The raw files live in data/raw and are committed to the repository so that the
pipeline runs offline. Two files are used:

  secom.data         1567 rows, 590 space separated sensor readings, "NaN" for missing
  secom_labels.data  1567 rows, a pass/fail flag and a quoted timestamp

The raw label encoding is -1 for pass and +1 for fail. Everything downstream of
this module uses the modelling encoding 0 for pass and 1 for fail, so that the
minority class (fail) is the positive class for precision and recall.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
SENSOR_PREFIX = "sensor_"
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"


@dataclass(frozen=True)
class SecomData:
    """The parsed dataset, sorted ascending by timestamp.

    Attributes
    ----------
    X : pd.DataFrame
        Sensor readings, one column per sensor, named sensor_000 to sensor_589.
    y : pd.Series
        Modelling label, 1 for fail and 0 for pass.
    timestamp : pd.Series
        Parsed measurement timestamp, ascending.
    original_row : pd.Series
        Zero based position of each record in the raw file, retained so that the
        effect of sorting is auditable.
    """

    X: pd.DataFrame
    y: pd.Series
    timestamp: pd.Series
    original_row: pd.Series

    def __post_init__(self) -> None:
        n = len(self.X)
        if not (len(self.y) == len(self.timestamp) == len(self.original_row) == n):
            raise ValueError("X, y, timestamp and original_row must have equal length")
        if not self.timestamp.is_monotonic_increasing:
            raise ValueError("timestamps are not sorted ascending")


def file_checksums() -> dict[str, dict[str, object]]:
    """Return the sha256 and byte count of every file in data/raw."""
    out: dict[str, dict[str, object]] = {}
    for path in sorted(RAW_DIR.iterdir()):
        if path.name == "CHECKSUMS.json" or not path.is_file():
            continue
        payload = path.read_bytes()
        out[path.name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
    return out


def verify_checksums(strict: bool = True) -> bool:
    """Compare the current data/raw files against the recorded CHECKSUMS.json."""
    recorded_path = RAW_DIR / "CHECKSUMS.json"
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    actual = file_checksums()
    ok = recorded == actual
    if strict and not ok:
        raise ValueError(f"checksum mismatch: recorded={recorded} actual={actual}")
    return ok


def load_secom(raw_dir: Path | None = None) -> SecomData:
    """Parse the raw SECOM files and return them sorted ascending by timestamp.

    Sorting matters. The raw file order is close to chronological but is not
    guaranteed to be exactly chronological, and a chronological train/holdout
    split defined on unsorted rows would leak future records into train.
    """
    raw_dir = RAW_DIR if raw_dir is None else Path(raw_dir)

    features = pd.read_csv(
        raw_dir / "secom.data",
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        dtype=np.float64,
        engine="c",
    )
    features.columns = [f"{SENSOR_PREFIX}{i:03d}" for i in range(features.shape[1])]

    # The timestamp is a single double quoted field containing a space, so the
    # whitespace separator yields two columns, not three.
    labels = pd.read_csv(
        raw_dir / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["raw_label", "stamp"],
        quotechar='"',
    )
    if len(labels) != len(features):
        raise ValueError(
            f"row count mismatch: features={len(features)} labels={len(labels)}"
        )

    timestamp = pd.to_datetime(
        labels["stamp"].astype(str).str.strip(),
        format=TIMESTAMP_FORMAT,
        errors="raise",
    )

    raw_label = labels["raw_label"].astype(int)
    unexpected = set(raw_label.unique()) - {-1, 1}
    if unexpected:
        raise ValueError(f"unexpected raw label values: {sorted(unexpected)}")
    y = (raw_label == 1).astype(int)

    frame = features.copy()
    frame["__timestamp__"] = timestamp
    frame["__y__"] = y.to_numpy()
    frame["__original_row__"] = np.arange(len(frame), dtype=int)

    # kind="stable" keeps raw file order among records sharing a timestamp, so the
    # sort is deterministic and reversible.
    frame = frame.sort_values("__timestamp__", kind="stable").reset_index(drop=True)

    sensor_cols = [c for c in frame.columns if c.startswith(SENSOR_PREFIX)]
    return SecomData(
        X=frame[sensor_cols],
        y=frame["__y__"].astype(int),
        timestamp=frame["__timestamp__"],
        original_row=frame["__original_row__"].astype(int),
    )
