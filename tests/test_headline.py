"""results/headline.json must match the artifacts it cites.

The file is the one place outside this repository's reports that states its headline,
so every number in it is checked here against the committed artifact it names, and the
time order preserving p-value is recomputed from the committed raw data with the
repository's own functions. Nothing here needs data beyond what is committed.
"""

from __future__ import annotations

import json
import re
from functools import reduce
from pathlib import Path

import pytest

from yield_screening import config
from yield_screening.data_io import load_secom
from yield_screening.preprocess import Preprocessor
from yield_screening.screening import circular_shift_null, mannwhitney_screen
from yield_screening.splitting import chronological_split

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADLINE = json.loads((REPO_ROOT / "results" / "headline.json").read_text(encoding="utf-8"))

TOP_KEYS = {"schema", "repo", "source_commit", "as_of", "rows"}
ROW_KEYS = {
    "id", "section", "hypothesis", "verdict", "verdict_detail", "mechanism", "stats", "caveats",
}
STAT_KEYS = {"label", "display", "value", "artifact", "locator", "provenance"}
PROVENANCE = {"reproduced", "repo-reported, not reproduced", "predates fix; pending re-run"}

STATS = [stat for row in HEADLINE["rows"] for stat in row["stats"]]


def _json_path(document: object, dotted: str) -> object:
    return reduce(lambda node, key: node[key], dotted.split("."), document)


def test_schema_keys_and_values() -> None:
    assert set(HEADLINE) == TOP_KEYS
    assert HEADLINE["schema"] == "headline/v1"
    assert HEADLINE["repo"] == "AaroNLaU0307/semiconductor-yield-screening"
    assert HEADLINE["source_commit"] == "PENDING" or re.fullmatch(
        r"[0-9a-f]{40}", HEADLINE["source_commit"]
    )
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", HEADLINE["as_of"])
    assert HEADLINE["rows"]
    for row in HEADLINE["rows"]:
        assert set(row) == ROW_KEYS
        assert row["section"] in {"archive", "other"}
        assert all(isinstance(c, str) for c in row["caveats"])
        for stat in row["stats"]:
            assert set(stat) == STAT_KEYS
            assert stat["provenance"] in PROVENANCE


@pytest.mark.parametrize("stat", STATS, ids=[s["label"] for s in STATS])
def test_every_artifact_exists_and_holds_the_value(stat: dict) -> None:
    artifact = REPO_ROOT / stat["artifact"]
    assert artifact.is_file(), stat["artifact"]
    if artifact.suffix == ".json":
        # The locator starts with the dotted path of the value inside the file.
        document = json.loads(artifact.read_text(encoding="utf-8"))
        stored = _json_path(document, stat["locator"].split()[0])
        assert stored == pytest.approx(stat["value"], rel=1e-12)
    else:
        assert stat["display"] in artifact.read_text(encoding="utf-8")


def _stat(artifact: str) -> dict:
    return next(s for s in STATS if s["artifact"] == artifact)


def test_replication_count_recomputed_from_its_artifact() -> None:
    stat = _stat("dashboard/replication_data.json")
    replication = json.loads((REPO_ROOT / stat["artifact"]).read_text(encoding="utf-8"))
    n_replicating = sum(bool(f["replicates"]) for f in replication["features"])
    n_shortlisted = len(replication["features"])
    assert n_replicating == stat["value"] == replication["summary"]["n_replicating"]
    assert n_shortlisted == replication["summary"]["n_shortlisted"]
    assert stat["display"] == f"{n_replicating} of {n_shortlisted}"


def test_time_order_p_value_recomputed_from_the_raw_data() -> None:
    """Rerun the train screen and its circular shift null with the repository's code."""
    stat = _stat("dashboard/screening_data.json")
    data = load_secom()
    split = chronological_split(data.timestamp, config.TRAIN_FRACTION)
    X_raw = data.X.iloc[split.train_idx]
    y = data.y.iloc[split.train_idx].to_numpy()
    X = Preprocessor().fit(X_raw).transform(X_raw)
    screen = mannwhitney_screen(X, y)
    null = circular_shift_null(X, y, observed_count=screen.n_survivors, q=config.FDR_Q)

    assert (screen.n_survivors, screen.n_tested) == (32, 424)
    assert null.p_value == pytest.approx(stat["value"], rel=1e-12)
    assert stat["display"] == f"{null.p_value:.3f}"
