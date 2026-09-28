"""Per-phase live_distances.jsonl ring (spec R2)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from gareus.cli import parse_args
from gareus.logger import LIVE_DASHBOARD_EVERY, LIVE_ROW_KEYS, DistanceLogger


class FakeSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(dict(event))


def _args(**kw):
    base = dict(tui_mode="none", distance_output_mode="none", live_distances_max_mb=256)
    base.update(kw)
    return SimpleNamespace(**base)


def _rows(n=3):
    return [{"replica": i, "window": i, "center_A": 0.5, "k_kcal_mol_A2": 100.0, "cv_A": 0.4,
             "primary_cv_value": 0.4, "secondary_cv": 1.5, "gamd_boost_total_kcal_mol": 2.0,
             "potential_kj_mol": -5e5} for i in range(n)]


def _dash():
    return {"n_windows": 3, "exchange_stats": {"pairs": {"0-1": [1, 2]}},
            "secondary_cv": {"k": 1}, "secondary_cv_centers": [0.0, 1.0, 2.0]}


def test_flag_default_and_zero():
    assert parse_args(["--seq", "AA", "--out", "u"]).live_distances_max_mb == 256
    assert parse_args(["--seq", "AA", "--out", "u", "--live-distances-max-mb", "0"]).live_distances_max_mb == 0


def test_ring_gets_slim_rows_and_progress_gets_summary(tmp_path):
    sink = FakeSink()
    lg = DistanceLogger(tmp_path, _args(), progress=sink, no_file_persistence=True)
    lg.log(_rows(), "gareus_production", 250, 1000, dashboard_info=_dash())
    lg.close()
    ring = [json.loads(x) for x in (tmp_path / "live_distances.jsonl").read_text().splitlines()]
    assert len(ring) == 1 and ring[0]["event"] == "distances"
    assert all(set(r) <= set(LIVE_ROW_KEYS) for r in ring[0]["distances"])
    assert ring[0]["distances"][0]["secondary_cv"] == 1.5
    assert "exchange_stats" in ring[0]["dashboard"]
    (summary,) = sink.events
    assert summary["event"] == "distances_summary"
    assert "distances" not in summary and "dashboard" not in summary
    assert summary["step"] == 250 and "cv_mean_A" in summary


def test_dashboard_only_every_nth_event(tmp_path):
    lg = DistanceLogger(tmp_path, _args(), progress=FakeSink(), no_file_persistence=True)
    for k in range(LIVE_DASHBOARD_EVERY + 1):
        lg.log(_rows(), "gareus_production", 250 * (k + 1), None, dashboard_info=_dash())
    lg.close()
    ring = [json.loads(x) for x in (tmp_path / "live_distances.jsonl").read_text().splitlines()]
    with_dash = [i for i, e in enumerate(ring) if "dashboard" in e]
    assert with_dash == [0, LIVE_DASHBOARD_EVERY]


def test_zero_keeps_legacy_full_event(tmp_path):
    sink = FakeSink()
    lg = DistanceLogger(tmp_path, _args(live_distances_max_mb=0), progress=sink, no_file_persistence=True)
    lg.log(_rows(), "gareus_production", 250, 1000, dashboard_info=_dash())
    lg.close()
    assert not (tmp_path / "live_distances.jsonl").exists()
    assert sink.events[0]["event"] == "distances" and len(sink.events[0]["distances"]) == 3
