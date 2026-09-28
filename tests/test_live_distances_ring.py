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


# ---- final-review fix wave (I1, I3, M4, M5) ---------------------------------

import pytest  # noqa: E402

import gareus.logger as logger_mod  # noqa: E402


class _BrokenRing:
    def __init__(self):
        self.calls = 0

    def write_json(self, payload):
        self.calls += 1
        raise OSError(28, "No space left on device")

    def close(self):
        pass


def test_ring_write_failure_falls_back_to_legacy_event_and_warns_once(tmp_path, capsys):
    sink = FakeSink()
    lg = DistanceLogger(tmp_path, _args(), progress=sink, no_file_persistence=True)
    broken = _BrokenRing()
    lg._live_ring = broken
    for k in range(3):
        lg.log(_rows(), "gareus_production", 250 * (k + 1), 1000, dashboard_info=_dash())
    lg.close()
    assert broken.calls == 1
    assert capsys.readouterr().out.count("WARNING") == 1
    assert [e["event"] for e in sink.events] == ["distances"] * 3
    assert all(len(e["distances"]) == 3 and "dashboard" in e for e in sink.events)


def test_ring_construction_failure_falls_back_to_legacy_event(tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise OSError(116, "Stale file handle")

    monkeypatch.setattr(logger_mod, "RotatingJsonlWriter", boom)
    sink = FakeSink()
    lg = DistanceLogger(tmp_path, _args(), progress=sink, no_file_persistence=True)
    lg.log(_rows(), "gareus_production", 250, 1000, dashboard_info=_dash())
    lg.close()
    assert capsys.readouterr().out.count("WARNING") == 1
    assert sink.events[0]["event"] == "distances" and len(sink.events[0]["distances"]) == 3


def test_non_json_dashboard_value_does_not_crash_log(tmp_path, capsys):
    sink = FakeSink()
    lg = DistanceLogger(tmp_path, _args(), progress=sink, no_file_persistence=True)
    dash = _dash()
    dash["exchange_stats"] = {"bad": object()}
    lg.log(_rows(), "gareus_production", 250, 1000, dashboard_info=dash)
    lg.close()
    assert "WARNING" in capsys.readouterr().out
    assert sink.events[-1]["event"] == "distances"


def test_live_dashboard_json_written_atomically_with_wall_time_and_step(tmp_path):
    lg = DistanceLogger(tmp_path, _args(), progress=FakeSink(), no_file_persistence=True)
    for k in range(LIVE_DASHBOARD_EVERY + 1):
        lg.log(_rows(), "gareus_production", 250 * (k + 1), None, dashboard_info=_dash())
        if k == 0:
            first = json.loads((tmp_path / "live_dashboard.json").read_text())
    lg.close()
    assert first["step"] == 250 and "wall_time_s" in first
    last = json.loads((tmp_path / "live_dashboard.json").read_text())
    assert last["step"] == 250 * (LIVE_DASHBOARD_EVERY + 1)
    assert last["dashboard"]["exchange_stats"] == _dash()["exchange_stats"]
    assert [p.name for p in tmp_path.iterdir() if "live_dashboard" in p.name] == ["live_dashboard.json"]


def test_no_live_dashboard_json_when_ring_disabled(tmp_path):
    lg = DistanceLogger(tmp_path, _args(live_distances_max_mb=0), progress=FakeSink(), no_file_persistence=True)
    lg.log(_rows(), "gareus_production", 250, 1000, dashboard_info=_dash())
    lg.close()
    assert not (tmp_path / "live_dashboard.json").exists()


def test_negative_live_distances_max_mb_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--seq", "AA", "--out", "u", "--live-distances-max-mb", "-1"])


def test_live_distances_help_mentions_console_mode():
    import argparse
    import re
    from gareus import cli
    p = argparse.ArgumentParser()
    cli._add_output_args(p)
    text = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", p.format_help()).split())
    assert "--progress-mode console" in text
