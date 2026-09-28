"""gareus_monitor reads the per-phase live_distances ring (spec R2)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import gareus_monitor as gm


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
