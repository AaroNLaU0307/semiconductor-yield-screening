"""Required test 4: seed determinism, DECISIONS.md D13 and PLAN.md's test schedule.

"Seed determinism. Same seed, same run, same reported metrics." is the exact
requirement this file implements.

Two groups of tests live here, because that requirement has two different shapes
depending on how far you zoom out.

Group 1 calls individual library functions twice, in process, on the same inputs
and the same seed. These tests are fast, and when one fails it points directly at
the one function whose output moved: a stack trace that stops at the actual non
deterministic call, rather than at whichever downstream report file happened to
differ. A module scoped fixture loads the real dataset, splits it, and fits and
applies the Preprocessor once, so every test below reuses that instead of paying
the load and fit cost again.

Group 2 runs `python scripts/run_pipeline.py` twice, as two separate operating
system processes, from a clean interpreter each time, and diffs every file it
writes byte for byte. This is the property README.md and PLAN.md actually claim:
not that a function is a pure function of its arguments, but that running the
one declared command twice reproduces every generated report and dashboard file
exactly. Nothing in group 1 exercises a fresh process, a fresh import of every
module, or the interaction of every phase end to end, so group 1 passing is
necessary but not sufficient evidence for the claim group 2 checks directly.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yield_screening.config import RANDOM_SEED
from yield_screening.data_io import REPO_ROOT, SecomData, load_secom
from yield_screening.modeling import bootstrap_pr_auc, make_hgb
from yield_screening.preprocess import Preprocessor
from yield_screening.screening import mannwhitney_screen, permutation_null
from yield_screening.splitting import Split, chronological_split

# ---------------------------------------------------------------------------
# Group 1: fast in-process determinism.
# ---------------------------------------------------------------------------


@dataclass
class _TrainSetup:
    """Everything the fast in-process tests below need, computed once per module."""

    X_train_raw: pd.DataFrame
    y_train: np.ndarray
    preprocessor: Preprocessor
    X_train: pd.DataFrame


@pytest.fixture(scope="module")
def train_setup() -> _TrainSetup:
    """Load the real SECOM dataset, split it, fit the Preprocessor on train, and
    transform train, exactly once for every test in this module.

    Every test that uses this fixture only reads from it; none of them refit or
    mutate `train_setup.preprocessor` or `train_setup.X_train` in place, which is
    what makes sharing this expensive setup across tests safe.
    """
    secom_data: SecomData = load_secom()
    split: Split = chronological_split(secom_data.timestamp)
    X_train_raw = secom_data.X.iloc[split.train_idx]
    y_train = secom_data.y.iloc[split.train_idx].to_numpy()
    preprocessor = Preprocessor().fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    return _TrainSetup(
        X_train_raw=X_train_raw,
        y_train=y_train,
        preprocessor=preprocessor,
        X_train=X_train,
    )


def test_preprocessor_fit_is_deterministic(train_setup: _TrainSetup) -> None:
    """Fitting a fresh Preprocessor on the same train rows twice yields the exact
    same dense columns, train medians and indicator groups both times."""
    first = train_setup.preprocessor.fitted_parameters()
    second = Preprocessor().fit(train_setup.X_train_raw).fitted_parameters()
    assert first == second


def test_screen_is_deterministic(train_setup: _TrainSetup) -> None:
    """Calling mannwhitney_screen twice on the same train input returns identical
    tables, checked column for column and row for row rather than just summarised."""
    first = mannwhitney_screen(train_setup.X_train, train_setup.y_train).table
    second = mannwhitney_screen(train_setup.X_train, train_setup.y_train).table
    pd.testing.assert_frame_equal(first, second)


def test_permutation_null_is_deterministic_for_a_fixed_seed(
    train_setup: _TrainSetup,
) -> None:
    """Two permutation_null calls sharing one seed produce an identical null_counts
    array and an identical p_value. n_permutations is reduced to keep this fast;
    determinism does not depend on how many permutations are drawn."""
    screen = mannwhitney_screen(train_setup.X_train, train_setup.y_train)
    shared_arguments = dict(
        X=train_setup.X_train,
        y=train_setup.y_train,
        observed_count=screen.n_survivors,
        n_permutations=200,
        seed=RANDOM_SEED,
    )
    first = permutation_null(**shared_arguments)
    second = permutation_null(**shared_arguments)

    np.testing.assert_array_equal(first.null_counts, second.null_counts)
    assert first.p_value == second.p_value


def test_permutation_null_differs_for_a_different_seed(
    train_setup: _TrainSetup,
) -> None:
    """Two permutation_null calls with different seeds must not produce identical
    null_counts arrays.

    This is the test the fixed seed test above cannot do on its own: reusing the
    same seed twice would look identical even if `permutation_null` silently
    ignored its `seed` argument and always drew the same fixed sequence. Only a
    changed seed producing a changed result proves the seed is actually wired
    through to the permutation draws.
    """
    screen = mannwhitney_screen(train_setup.X_train, train_setup.y_train)
    common_arguments = dict(
        X=train_setup.X_train,
        y=train_setup.y_train,
        observed_count=screen.n_survivors,
        n_permutations=200,
    )
    first = permutation_null(seed=RANDOM_SEED, **common_arguments)
    second = permutation_null(seed=RANDOM_SEED + 1, **common_arguments)

    assert not np.array_equal(first.null_counts, second.null_counts)


def _synthetic_labels_and_scores(seed: int, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """A small synthetic binary label vector and a correlated score vector.

    bootstrap_pr_auc only needs `y` and `proba` of matching length with both
    classes present, so fitting a real model to build them would be unnecessary
    setup cost for what this test is checking.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    if y.sum() < 5:
        y[:5] = 1
    proba = np.clip(0.5 * y + rng.normal(scale=0.25, size=n), 0.0, 1.0)
    return y, proba


def test_bootstrap_pr_auc_is_deterministic() -> None:
    """bootstrap_pr_auc called twice with the same seed on the same small synthetic
    inputs returns identical interval bounds and an identical resampled score array."""
    y, proba = _synthetic_labels_and_scores(seed=13)

    low_first, high_first, scores_first = bootstrap_pr_auc(
        y, proba, n_resamples=300, seed=RANDOM_SEED
    )
    low_second, high_second, scores_second = bootstrap_pr_auc(
        y, proba, n_resamples=300, seed=RANDOM_SEED
    )

    assert low_first == low_second
    assert high_first == high_second
    np.testing.assert_array_equal(scores_first, scores_second)


def test_model_fit_is_deterministic(train_setup: _TrainSetup) -> None:
    """Fitting make_hgb() twice on the same screened train data gives bit for bit
    identical predict_proba outputs.

    Restricted to the first 15 of the shortlisted features (32 survive on the
    real dataset, per reports/screening.md) so the fit itself stays fast.
    """
    screen = mannwhitney_screen(train_setup.X_train, train_setup.y_train)
    features = list(screen.shortlist["feature"])[:15]
    assert len(features) == 15, "test construction check: shortlist has fewer than 15 features"
    X_small = train_setup.X_train[features]
    y = train_setup.y_train

    first = make_hgb().fit(X_small, y).predict_proba(X_small)
    second = make_hgb().fit(X_small, y).predict_proba(X_small)
    np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# Group 2: end to end reproducibility, the property actually claimed.
# ---------------------------------------------------------------------------

RUN_PIPELINE = REPO_ROOT / "scripts" / "run_pipeline.py"


def _tracked_report_files() -> list[Path]:
    """Every file this test regenerates and compares.

    Every `.md`, `.csv` and `.json` file directly under `reports/`, and every
    `.json` file directly under `dashboard/`. `dashboard/index.html` is
    deliberately excluded (it does not match the `.json` filter above): a
    separate script generates it, and it is out of scope here.
    """
    reports_files = [
        path
        for path in (REPO_ROOT / "reports").iterdir()
        if path.is_file() and path.suffix in (".md", ".csv", ".json")
    ]
    dashboard_files = [
        path
        for path in (REPO_ROOT / "dashboard").iterdir()
        if path.is_file() and path.suffix == ".json"
    ]
    return sorted(reports_files + dashboard_files)


def _sha256_by_relative_path(paths: list[Path]) -> dict[str, str]:
    """Map each path, keyed by its path relative to the repo root, to its sha256."""
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


@pytest.mark.slow
def test_pipeline_rerun_produces_identical_reports() -> None:
    """Running scripts/run_pipeline.py twice, from a clean process each time,
    produces byte identical reports and dashboard data both times.

    This is the property required test 4 actually claims: not that a function is
    pure given its arguments, but that running the one declared command twice
    reproduces every generated report and dashboard file exactly. Each run takes
    roughly 50 seconds, so this test takes roughly 100 seconds; it is marked
    `slow` so a reader can filter it out, but it is not skipped by default.

    This test regenerates every tracked report and dashboard file. Since the
    content should be identical, that is normally harmless, but if the test
    fails partway through, the repository must not be left holding a half
    written report. The bytes of every tracked file are read and saved before
    the first run, and are restored, but only if the test did not complete
    successfully.
    """
    tracked = _tracked_report_files()
    original_bytes = {path: path.read_bytes() for path in tracked}

    succeeded = False
    try:
        subprocess.run(
            [sys.executable, str(RUN_PIPELINE)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            timeout=300,
        )
        first_hashes = _sha256_by_relative_path(_tracked_report_files())

        subprocess.run(
            [sys.executable, str(RUN_PIPELINE)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            timeout=300,
        )
        second_hashes = _sha256_by_relative_path(_tracked_report_files())

        assert set(first_hashes) == set(second_hashes), (
            "the set of generated report/dashboard files changed between runs: "
            f"only after run 1: {sorted(set(first_hashes) - set(second_hashes))}, "
            f"only after run 2: {sorted(set(second_hashes) - set(first_hashes))}"
        )
        differing = sorted(
            name for name in first_hashes if first_hashes[name] != second_hashes[name]
        )
        assert not differing, f"non-deterministic output detected in: {differing}"
        succeeded = True
    finally:
        if not succeeded:
            for path, content in original_bytes.items():
                path.write_bytes(content)
