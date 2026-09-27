"""Sentences in the generated reports that assert a result must follow from the data.

A claim typed into a report generator does not update when the numbers under it
change, and one such sentence in `reports/spc.md` said the holdout out of control
rate was higher "for every charted sensor" while the table above it showed a sensor
where it was lower. These tests pin the computed replacements: each one is fed a
table where the old wording would be false, and each committed report is checked
against its own table. The same applies to the designed experiment in
`reports/doe_proposal.md`, whose factors must be the features that replicated.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yield_screening.config import INDICATOR_SUFFIX
from yield_screening.spc import ooc_failure_contingency

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a script from scripts/ as a module, without running its main()."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pipeline():
    return _load_script("run_pipeline")


def _write_spc_report(pipeline, tmp_path: Path, monkeypatch, per_sensor: pd.DataFrame) -> str:
    """Render reports/spc.md into tmp_path from a synthetic section 2 table."""
    monkeypatch.setattr(pipeline, "REPORTS", tmp_path)
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.1).astype(int)
    counts = rng.integers(0, len(per_sensor) + 1, size=200)
    contingency = ooc_failure_contingency(counts >= 1, y)
    sweep = pipeline.contingency_sweep(counts, counts, y, len(per_sensor))
    pipeline.write_spc_report(list(per_sensor["sensor"]), per_sensor, contingency, sweep, y)
    return (tmp_path / "spc.md").read_text(encoding="utf-8")


def _per_sensor(train: list[float], holdout: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sensor": [f"sensor_{i:03d}" for i in range(len(train))],
            "effect size": 0.3,
            "train OOC rate": train,
            "holdout OOC rate": holdout,
            "train imputed share": 0.0,
            "holdout imputed share": 0.0,
        }
    )


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start : end if end != -1 else len(text)]


def test_spc_drift_sentence_counts_the_sensors_the_table_shows(
    pipeline, tmp_path, monkeypatch
) -> None:
    """One sensor with a lower holdout rate: the report must say 2 of 3, and name it."""
    per_sensor = _per_sensor([0.10, 0.20, 0.39], [0.25, 0.30, 0.37])
    text = _write_spc_report(pipeline, tmp_path, monkeypatch, per_sensor)
    drift = _section(text, "## 6. Process drift")

    assert "every charted sensor" not in drift
    assert "higher than the train rate for 2 of the 3 charted sensors" in drift
    assert "1 of them by a factor of two or more" in drift
    assert "the exception is sensor_002 (0.3900 on train, 0.3700 on holdout)" in drift


def test_spc_drift_sentence_says_every_sensor_only_when_true(
    pipeline, tmp_path, monkeypatch
) -> None:
    per_sensor = _per_sensor([0.10, 0.20], [0.25, 0.30])
    drift = _section(
        _write_spc_report(pipeline, tmp_path, monkeypatch, per_sensor), "## 6. Process drift"
    )
    assert "higher than the train rate for every charted sensor" in drift
    assert "exception" not in drift
    assert "the process moved" in drift


def test_spc_drift_reading_is_withheld_when_most_sensors_did_not_move(
    pipeline, tmp_path, monkeypatch
) -> None:
    per_sensor = _per_sensor([0.30, 0.20, 0.40], [0.25, 0.30, 0.35])
    drift = _section(
        _write_spc_report(pipeline, tmp_path, monkeypatch, per_sensor), "## 6. Process drift"
    )
    assert "for 1 of the 3 charted sensors" in drift
    assert "the process moved" not in drift
    assert "not a consistent shift" in drift


def test_committed_spc_report_drift_sentence_matches_its_own_table() -> None:
    """reports/spc.md section 6 must state the count its section 2 table implies."""
    text = (REPO_ROOT / "reports" / "spc.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| (sensor_\d+) \| [-\d.]+ \| ([\d.]+) \| ([\d.]+) \|",
        _section(text, "## 2."),
        re.MULTILINE,
    )
    assert rows, "could not parse the charted sensor table"
    n_higher = sum(float(holdout) > float(train) for _, train, holdout in rows)
    drift = _section(text, "## 6. Process drift")
    if n_higher == len(rows):
        assert "for every charted sensor" in drift
    else:
        assert f"for {n_higher} of the {len(rows)} charted sensors" in drift
        assert "every charted sensor" not in drift


# ---------------------------------------------------------------------------
# The designed experiment targets the features that replicated
# ---------------------------------------------------------------------------


def _replicating_sensor_values() -> list[str]:
    replication = json.loads(
        (REPO_ROOT / "dashboard" / "replication_data.json").read_text(encoding="utf-8")
    )
    return [
        row["feature"]
        for row in sorted(replication["features"], key=lambda r: r["holdout BH adjusted p"])
        if row["replicates"] and not row["feature"].endswith(INDICATOR_SUFFIX)
    ]


def test_doe_factor_selection_uses_replication_not_train_rank() -> None:
    """Replicating sensor values become factors, ordered by holdout adjusted p;
    indicators never do; larger train effects that did not replicate are reported
    as passed over rather than chosen."""
    doe = _load_script("build_doe_proposal")
    replication = {
        "summary": {"n_shortlisted": 4},
        "features": [
            {"feature": "sensor_001", "train effect": 0.50, "holdout effect": -0.02,
             "holdout BH adjusted p": 0.90, "replicates": False},
            {"feature": "sensor_002", "train effect": 0.20, "holdout effect": 0.30,
             "holdout BH adjusted p": 0.05, "replicates": True},
            {"feature": "sensor_003", "train effect": -0.25, "holdout effect": -0.40,
             "holdout BH adjusted p": 0.01, "replicates": True},
            {"feature": f"sensor_004{INDICATOR_SUFFIX}", "train effect": 0.60,
             "holdout effect": 0.50, "holdout BH adjusted p": 0.001, "replicates": True},
        ],
    }
    factors, passed_over = doe.select_factors(replication)
    assert list(factors["feature"]) == ["sensor_003", "sensor_002"]
    assert list(passed_over["feature"]) == ["sensor_001"]


def test_committed_doe_proposal_targets_the_replicating_sensors() -> None:
    """reports/doe_proposal.md must propose exactly the sensor values that replicated."""
    expected = _replicating_sensor_values()
    text = (REPO_ROOT / "reports" / "doe_proposal.md").read_text(encoding="utf-8")
    proposed = re.findall(r"^\| F\d+ \| (\S+) \|", text, re.MULTILINE)
    assert proposed == expected
    assert f"| factors | {len(expected)} |" in text
