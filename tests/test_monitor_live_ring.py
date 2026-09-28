"""gareus_monitor reads the per-phase live_distances ring (spec R2)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import gareus_monitor as gm
from gareus.logger import LIVE_DASHBOARD_EVERY


def _event(step, windows=2, dashboard=False):
    e = {"event": "distances", "step": step, "wall_time_s": 1.0 + step,
         "distances": [{"replica": w, "window": w, "primary_cv_value": 0.1 * w, "secondary_cv": 1.0,
                        "umbrella_bias_kcal_mol": 0.2, "gamd_boost_total_kcal_mol": 1.5}
                       for w in range(windows)]}
    if dashboard:
        e["dashboard"] = {"exchange_stats": {"attempts": step}}
    return e


def _write(p: Path, events):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(e) + "\n" for e in events))


def test_no_ring_returns_empty(tmp_path):
    assert gm.live_ring_files(tmp_path) == []


def test_picks_most_recent_phase_ring_oldest_first(tmp_path):
    old = tmp_path / "adaptive_production" / "final" / "baseline" / "live_distances.jsonl"
    new = tmp_path / "adaptive_production" / "final_extension_001" / "live_distances.jsonl"
    _write(old, [_event(1)])
    _write(new.with_name("live_distances.1.jsonl"), [_event(2)])
    _write(new, [_event(3)])
    os.utime(old, (1, 1))
    assert gm.live_ring_files(tmp_path) == [new.with_name("live_distances.1.jsonl"), new]


def test_boost_samples_span_rotation(tmp_path):
    ring = tmp_path / "adaptive_production" / "final" / "baseline" / "live_distances.jsonl"
    _write(ring.with_name("live_distances.1.jsonl"), [_event(s) for s in range(1, 6)])
    _write(ring, [_event(6)])
    samples = gm.read_ring_boost_samples(gm.live_ring_files(tmp_path), target_per_window=4)
    assert len(samples) >= 8  # 2 windows x >= 4 events, needing the rotated file


def test_ring_entries_newest_last(tmp_path):
    ring = tmp_path / "live_distances.jsonl"
    _write(ring.with_name("live_distances.1.jsonl"), [_event(1)])
    _write(ring, [_event(2, dashboard=True)])
    entries = gm.read_ring_entries(gm.live_ring_files(tmp_path), max_bytes=1 << 20)
    assert [e["step"] for e in entries] == [1, 2]


def test_live_snapshot_dashboard_age_from_ring_when_progress_has_none(tmp_path):
    # progress.jsonl holds only distances_summary/progress lines (Task 5's split:
    # the ring gets every 'distances' event, progress.jsonl only the periodic
    # scalar summary) -- no dashboard payload ever appears there once a ring
    # exists, so live_snapshot must source both the dashboard and its age from
    # the ring, not from progress.jsonl's own tail.
    run_dir = tmp_path / "chignolin_test"
    run_dir.mkdir()
    progress = run_dir / "progress.jsonl"
    _write(progress, [
        {"event": "progress", "phase": "gareus_production", "step": 100, "wall_time_s": 10.0},
        {"event": "distances_summary", "phase": "gareus_production", "step": 200, "wall_time_s": 20.0},
    ])
    ring = run_dir / "adaptive_production" / "final" / "baseline" / "live_distances.jsonl"
    # newest ring line (step=2) has no dashboard; the older one (step=1) does.
    _write(ring, [_event(1, dashboard=True), _event(2, dashboard=False)])

    state = gm.PeptideState.from_rundir(run_dir)
    snap = state.live_snapshot()

    assert snap["_dashboard"] == {"exchange_stats": {"attempts": 1}}
    assert snap["_dashboard_age"] == 1
    assert snap["_dashboard_age_s"] == 18.0
    assert len(snap["_dist_samples"]) == 4


# ---- final-review fix wave (I3): realistic sizes ----------------------------

class _ProgressFileSink:
    """Appends each emitted event to progress.jsonl, stamped like GuiProgressSink."""

    def __init__(self, path: Path):
        self.path = path

    def emit(self, event):
        import time
        with self.path.open("a") as fh:
            fh.write(json.dumps({**event, "wall_time_s": time.time()}) + "\n")


def _big_rows(n=300):
    return [{"replica": i, "window": i, "center_A": 0.5, "k_kcal_mol_A2": 100.0, "cv_A": 0.4,
             "primary_cv_value": 0.4123456789, "secondary_cv": 1.5123456789,
             "gamd_boost_total_kcal_mol": 2.0123456789, "umbrella_bias_kcal_mol": 0.0123456789,
             "gamd_lambda": 0.25, "potential_kj_mol": -5e5} for i in range(n)]


def _big_dash():
    # ~500 kB exchange_stats block, like 236 states' cumulative pair statistics.
    pairs = {f"{i}-{j}": [i * 1000 + j, 0.123456789, 0.987654321]
             for i in range(236) for j in range(i + 1, min(236, i + 60))}
    return {"n_windows": 300, "exchange_stats": {"pairs": pairs}, "secondary_cv": {},
            "secondary_cv_centers": []}


def test_dashboard_always_found_with_realistic_line_sizes(tmp_path):
    from types import SimpleNamespace

    from gareus.logger import DistanceLogger

    run_dir = tmp_path / "chignolin_big"
    phase = run_dir / "adaptive_production" / "final" / "baseline"
    phase.mkdir(parents=True)
    progress = run_dir / "progress.jsonl"
    progress.write_text("")
    dash = _big_dash()
    assert len(json.dumps(dash)) > 450_000
    args = SimpleNamespace(tui_mode="none", distance_output_mode="none", live_distances_max_mb=256)
    lg = DistanceLogger(phase, args, progress=_ProgressFileSink(progress), no_file_persistence=True)
    state = gm.PeptideState.from_rundir(run_dir)
    missing = []
    try:
        for k in range(45):
            lg.log(_big_rows(), "gareus_production", 250 * (k + 1), None, dashboard_info=dash)
            snap = state.live_snapshot()
            if snap.get("_dashboard") is None or snap.get("_dashboard_age_s") is None:
                missing.append(k)
    finally:
        lg.close()
    ring = phase / "live_distances.jsonl"
    slim = [len(x) for x in ring.read_text().splitlines()[1:LIVE_DASHBOARD_EVERY]]
    assert min(slim) > 40_000  # realistic slim-line size, so 20 of them outrun a 1 MB tail
    assert missing == []
    assert snap["_dashboard"]["exchange_stats"] == dash["exchange_stats"]
    assert snap["_dashboard_age_s"] >= 0.0


def test_live_dashboard_json_missing_falls_back_to_ring_scan(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    _write(run_dir / "progress.jsonl", [{"event": "progress", "step": 5, "wall_time_s": 10.0}])
    ring = run_dir / "adaptive_production" / "final" / "baseline" / "live_distances.jsonl"
    _write(ring, [_event(1, dashboard=True)])
    (ring.parent / "live_dashboard.json").write_text("{corrupt")
    snap = gm.PeptideState.from_rundir(run_dir).live_snapshot()
    assert snap["_dashboard"] == {"exchange_stats": {"attempts": 1}}
    assert snap["_dashboard_age_s"] == 8.0


def test_live_dashboard_json_preferred_over_ring_scan(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    _write(run_dir / "progress.jsonl", [{"event": "progress", "step": 5, "wall_time_s": 100.0}])
    ring = run_dir / "adaptive_production" / "final" / "baseline" / "live_distances.jsonl"
    _write(ring, [_event(1, dashboard=True), _event(2)])
    (ring.parent / "live_dashboard.json").write_text(json.dumps(
        {"step": 2, "wall_time_s": 95.0, "dashboard": {"exchange_stats": {"attempts": 99}}}))
    snap = gm.PeptideState.from_rundir(run_dir).live_snapshot()
    assert snap["_dashboard"] == {"exchange_stats": {"attempts": 99}}
    assert snap["_dashboard_age"] is None
    assert snap["_dashboard_age_s"] == 5.0
