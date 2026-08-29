"""Two dashboard defects that each raised a false "the run is broken" alarm.

Both were reported from a live 4-GPU campaign (chignolin_6, 2026-08-29) whose
physics was fine: the operator was told GaMD was dead and that 15 of 32 windows
were BAD, and stopped a healthy run twice to investigate.

1. ``no GaMD boost (plain umbrella run)`` while the integrator was demonstrably
   boosting (``boosted: true``, ForceScalingFactor 0.600). The campaign-wide
   GaMD sidecar is exported to ``<run>/adaptive_production/global_shared_gamd_setup/``,
   but the reader only looked in ``<run>/global_shared_gamd_setup/`` -- which
   exists on a real run as an EMPTY directory, so nothing even hinted at the
   miss.
2. ``BAD dead exchange 0.000`` on 15 of 32 windows while global acceptance was
   3858/4428 = 0.871 and every neighbour pair shown read 0.91-0.94. The window
   inherits the worst acceptance among its own pairs, and 16 of 140 pairs had
   ``attempts: 1, accepted: 0`` -- a single rejected attempt, which is 0.0 and
   finite, so it condemned the window.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from gareus.dashboard.context import acceptance_by_pair, acceptance_by_window
from gareus.dashboard.ranking import BAD, MIN_ATTEMPTS_FOR_DEAD, rank_windows
from gareus.dashboard.sidecar import SidecarCache

_GLOBALS = {
    "joint_envelope": {"Dihedral": {"boosted": True, "k0": 1.0,
                                    "sigma0_kj_mol": 10.46,
                                    "sigmaV_kj_mol": 16.05}},
    "sigma0p_kcal_mol": 2.5,
}


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _run_tree(tmp_path: Path) -> tuple[Path, Path]:
    """The real adaptive-production shape: worker deep under the run root."""
    run = tmp_path / "chignolin_6"
    worker = run / "adaptive_production" / "epoch_000"
    worker.mkdir(parents=True)
    return run, worker


# --------------------------------------------------------------------------
# 1. GaMD sidecar lookup
# --------------------------------------------------------------------------

def test_gamd_sidecar_is_found_at_the_campaign_wide_export_path(tmp_path):
    """The real chignolin_6 shape: exported under adaptive_production/."""
    run, worker = _run_tree(tmp_path)
    # Exactly what a real run leaves behind: the run-root directory exists but
    # is empty, so an existence check on the directory is not enough either.
    (run / "global_shared_gamd_setup").mkdir()
    _write(run / "adaptive_production" / "global_shared_gamd_setup"
           / "shared_gamd_setup_globals.json", _GLOBALS)

    snap = SidecarCache(worker, min_interval_s=0.0).snapshot(now=0.0)
    assert snap.gamd is not None, "campaign-wide GaMD sidecar was not found"
    assert snap.gamd["joint_envelope"]["Dihedral"]["boosted"] is True


def test_gamd_sidecar_still_found_at_the_original_run_root_path(tmp_path):
    """Back-compat: the pre-fix layout must keep working."""
    run, worker = _run_tree(tmp_path)
    _write(run / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json",
           _GLOBALS)

    snap = SidecarCache(worker, min_interval_s=0.0).snapshot(now=0.0)
    assert snap.gamd is not None
    assert snap.gamd["sigma0p_kcal_mol"] == 2.5


def test_gamd_sidecar_is_found_beside_the_worker_itself(tmp_path):
    """A worker writes its own copy before the campaign-wide export exists."""
    run, worker = _run_tree(tmp_path)
    _write(worker / "shared_gamd_setup_globals.json", _GLOBALS)

    snap = SidecarCache(worker, min_interval_s=0.0).snapshot(now=0.0)
    assert snap.gamd is not None


def test_a_run_with_no_gamd_anywhere_still_reports_none(tmp_path):
    """The genuine plain-umbrella case must not become a false positive."""
    _run, worker = _run_tree(tmp_path)
    snap = SidecarCache(worker, min_interval_s=0.0).snapshot(now=0.0)
    assert snap.gamd is None


# --------------------------------------------------------------------------
# 2. "dead exchange" from a single rejected attempt
# --------------------------------------------------------------------------

def _stats(pairs: dict[str, tuple[int, int]]) -> dict:
    return {"pairs": {k: {"attempts": a, "accepted": c}
                      for k, (a, c) in pairs.items()}}


def test_one_rejected_attempt_does_not_condemn_a_window():
    """The real shape: healthy neighbours plus a 0/1 long-range pair."""
    stats = _stats({"0-1": (400, 370), "1-2": (400, 366), "1-11": (1, 0)})
    pairs = acceptance_by_pair(stats)
    assert pairs[(1, 11)] == 0.0, "fixture must reproduce the 0/1 rate"

    per_window = acceptance_by_window(pairs, n_windows=12, attempts=stats)
    ranked = {s.window: s for s in rank_windows(
        n_windows=12, acceptance_by_window=per_window, overlap_by_pair={},
        delta_by_window={}, k_list=[200.0] * 12, temperature_k=300.0,
        centers_a=[0.1 * i for i in range(12)])}
    assert ranked[1].status != BAD, (
        f"window 1 condemned by a single rejected attempt: {ranked[1]}")


def test_a_genuinely_dead_pair_is_still_condemned():
    """Enough attempts to actually mean it -- the alarm must survive."""
    dead = MIN_ATTEMPTS_FOR_DEAD + 10
    stats = _stats({"0-1": (400, 370), "1-2": (dead, 0)})
    pairs = acceptance_by_pair(stats)

    per_window = acceptance_by_window(pairs, n_windows=3, attempts=stats)
    ranked = {s.window: s for s in rank_windows(
        n_windows=3, acceptance_by_window=per_window, overlap_by_pair={},
        delta_by_window={}, k_list=[200.0] * 3, temperature_k=300.0,
        centers_a=[0.0, 0.1, 0.2])}
    assert ranked[1].status == BAD
    assert "dead exchange" in " ".join(ranked[1].reasons)


def test_a_low_attempt_pair_does_not_hide_a_real_neighbour_failure():
    """The low-attempt skip must not mask a well-measured dead neighbour."""
    dead = MIN_ATTEMPTS_FOR_DEAD + 10
    stats = _stats({"0-1": (dead, 0), "1-11": (1, 0)})
    pairs = acceptance_by_pair(stats)

    per_window = acceptance_by_window(pairs, n_windows=12, attempts=stats)
    assert per_window[1] == 0.0
    assert per_window[0] == 0.0


def test_acceptance_by_window_is_unchanged_without_attempt_counts():
    """Back-compat: existing callers pass no attempts and keep old behaviour."""
    pairs = {(0, 1): 0.9, (1, 2): 0.0, (2, 3): float("nan")}
    assert acceptance_by_window(pairs, n_windows=4) == {0: 0.9, 1: 0.0, 2: 0.0}


def test_zero_attempt_pairs_remain_unmeasured_not_dead():
    """The pre-existing NaN guard must survive the new one."""
    pairs = acceptance_by_pair(_stats({"0-1": (0, 0)}))
    assert math.isnan(pairs[(0, 1)])
    assert acceptance_by_window(pairs, n_windows=2) == {}
