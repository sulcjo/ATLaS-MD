"""A retired phase must never be pooled into a feasibility statistic.

`audit_reweighting_feasibility.py` decides whether GaMD reweighting can work on
a run. Pooling an abandoned phase's samples into that decision is the same class
of defect the audit exists to detect: data silently entering a statistic because
of where it happens to sit on disk.

The concrete case. A crashed phase is renamed with a `_CRASHED_<date>` suffix and
a fresh one takes its place, and the audit already skipped those. But a
HAND-ARCHIVED phase is not covered by that name, and whether it gets pooled then
depends only on how deeply someone happened to nest it:

    adaptive_production/_archived_.../final/baseline/samples  ->  */*/*/samples, missed by the glob
    adaptive_production/final_old/baseline/samples            ->  */*/samples,   POOLED SILENTLY

The first is immune by accident, not by design. These tests make the exclusion
name-based so it no longer depends on directory depth.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "_audit_feas",
    pathlib.Path(__file__).resolve().parents[1] / "audit_reweighting_feasibility.py",
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)


@pytest.mark.parametrize("rel", [
    "epoch_000",
    "epoch_001/baseline",
    "epoch_001/topup_001_18080000",
    "final/baseline",
])
def test_live_phases_are_kept(rel):
    assert not _audit._is_retired(rel), f"{rel} is a live phase and must be analysed"


@pytest.mark.parametrize("rel", [
    "final_CRASHED_20260901/baseline",     # the driver's own rename
    "final_old/baseline",                  # archived IN PLACE -- the silent case
    "final_bak/baseline",
    "final_backup/baseline",
    "_archived_final_20260901_163000/final",
    "_scratch/baseline",
])
def test_retired_phases_are_excluded(rel):
    assert _audit._is_retired(rel), (
        f"{rel} would be pooled into the feasibility statistics. A dead phase's "
        "samples shift `a`, the anharmonicity and the CE2 columns with no message."
    )


def test_a_phase_is_retired_if_any_component_is():
    """Depth must not decide it -- that is the whole point of the change."""
    assert _audit._is_retired("_archived_x/final/baseline")
    assert _audit._is_retired("final/_old_baseline")


def test_discovery_on_the_real_tree_finds_live_phases_only():
    """Guards the glob and the filter together, against real directory names."""
    run = pathlib.Path(__file__).resolve().parents[1] / "RUNS" / "chignolin_6"
    if not (run / "adaptive_production").is_dir():
        pytest.skip("RUNS/chignolin_6 not present (gitignored)")
    base = run / "adaptive_production"
    found = [pathlib.Path(d).relative_to(base).as_posix() for d in _audit._phase_dirs(str(run))]
    assert found, "no phases discovered at all -- the glob is broken"
    assert not any(_audit._is_retired(f) for f in found), (
        f"a retired phase slipped through discovery: {found}"
    )
    assert "epoch_000" in found and "final/baseline" in found, found
