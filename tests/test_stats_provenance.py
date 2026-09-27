"""Pin the statistics helpers so that a silent edit fails CI.

Every reported p-value, interval and FDR decision in this repository comes from the
helpers below, all written here rather than vendored (see the provenance notes in the
docstrings of `yield_screening/screening.py` and `yield_screening/modeling.py`). The
behaviour tests in `tests/test_screening_math.py` check what they compute; this file
checks that their source has not changed without anyone deciding it should. The hash
is taken over `inspect.getsource`, with trailing whitespace stripped from each line,
so line endings and editor whitespace do not matter but any change to code or
docstring does. An intended edit updates the pinned hash in the same commit.
"""

from __future__ import annotations

import hashlib
import inspect

import pytest

from yield_screening import config, modeling, screening

PINNED_SHA256 = {
    "screening.column_midranks": "9d6ee32caa84af43b2f1d5b8847bce107678a94bd099e5f119149c00f583806d",
    "screening.tie_terms": "2d2da2a6d077fd146068c795eb7056c9994f2764c0810b194a3ed79c49c116ef",
    "screening.u_sigma": "91c772658759f089e1f28c22ad0be6c04f934a30ad9ecd53b0754d029372cd0b",
    "screening.u_from_rank_sums": "4698ea7485dae0ec7e5434a64bdec2fdd9f14743d8daa3c06d0cf68e57bea1d4",
    "screening.p_from_u": "8f2e8e2317a3e0a5b08f61544709404cf97327324ae6e3b4667986b77e2e88ea",
    "screening.rank_biserial": "08b5c60d1376bd3f5e00a900ec2b72f64ebdaa7e56a8674ef6730380a1833786",
    "screening.bootstrap_rank_biserial": "d9445c4fadc0366a3d78b6095f6b1b05d6a621d2324923d0dd6c86258a72d331",
    "screening.benjamini_hochberg": "bae895afd7bfa0a608d90654160c8a9cdb2a20864e9a22b1a10b1994fb9afab9",
    "screening.bh_survivor_counts": "3da2448f96de6fe63bff1691258bab58d8ea299416204e16cd25ca13bbdf2419",
    "screening.PermutationNullResult": "e66009d9f80d294e0c7c6b5800a3fb04d6bd3ba6648d9fc0e45f4de18f4e14e8",
    "screening.permutation_null": "04526bf9ad237f10e7b36c3130cfbe110558d39a7e052970ecf0024b28c85f68",
    "screening.circular_shift_null": "dfff5f49d3a5a6436fc2d893f6674836172ece1e7b5fc54c5e2d0c8e55a83a73",
    "modeling.bootstrap_pr_auc": "f5df1b6d5a28a6a670c812bc27d01beadccc485690aea8afe96977f82e7d6539",
}

MODULES = {"screening": screening, "modeling": modeling}


def normalised_source_sha256(obj: object) -> str:
    source = inspect.getsource(obj)
    normalised = "\n".join(line.rstrip() for line in source.splitlines())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("qualified_name", sorted(PINNED_SHA256))
def test_helper_source_matches_its_pinned_hash(qualified_name: str) -> None:
    module_name, attribute = qualified_name.split(".")
    helper = getattr(MODULES[module_name], attribute)
    assert normalised_source_sha256(helper) == PINNED_SHA256[qualified_name], (
        f"{qualified_name} changed. If the change is intended, update its hash here in "
        "the same commit and say why in the commit message."
    )


def test_declared_defaults_match_the_provenance_notes() -> None:
    """The levels the docstrings state are the ones the helpers actually default to."""
    assert config.FDR_Q == 0.10
    for helper in (
        screening.benjamini_hochberg,
        screening.bh_survivor_counts,
        screening.mannwhitney_screen,
        screening.permutation_null,
        screening.circular_shift_null,
    ):
        assert inspect.signature(helper).parameters["q"].default == 0.10

    permutation = inspect.signature(screening.permutation_null).parameters
    assert permutation["n_permutations"].default == 2000
    assert permutation["seed"].default == 20080719

    for helper in (screening.bootstrap_rank_biserial, modeling.bootstrap_pr_auc):
        parameters = inspect.signature(helper).parameters
        assert parameters["n_resamples"].default == 2000
        assert parameters["seed"].default == 20080719
        assert parameters["interval"].default == (2.5, 97.5)
