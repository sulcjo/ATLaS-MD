"""Optional production phase timers (--production-phase-timers).

Diagnostics only: wall time per production-loop phase, so the cost that the
replica-admission cap created (docs/superpowers/specs/performance-upgrades/
review-2026-09-27-p5-p7.md) can be attributed before any further upgrade.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gareus.cli import parse_args
from gareus.phase_timers import PhaseTimers, aggregate_npt_timings

MINIMAL = ["--seq", "AA", "--out", "unused"]


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_disabled_timers_record_nothing_and_write_nothing(tmp_path):
    clock = FakeClock()
    timers = PhaseTimers(enabled=False, clock=clock)
    timers.begin()
    with timers.phase("md"):
        clock.advance(5.0)
    timers.note_steps(100)
    assert timers.snapshot() == {"enabled": False}
    timers.write(tmp_path / "t.json")
    assert not (tmp_path / "t.json").exists()


def test_phases_accumulate_seconds_calls_and_fractions():
    clock = FakeClock()
    timers = PhaseTimers(enabled=True, clock=clock)
    timers.begin()
    for _ in range(2):
        with timers.phase("md"):
            clock.advance(3.0)
        with timers.phase("sample"):
            with timers.phase("sample.fetch"):
                clock.advance(0.5)
            clock.advance(0.25)
        clock.advance(0.25)  # unattributed loop overhead
        timers.note_steps(250)
    snap = timers.snapshot()
    assert snap["enabled"] is True
    assert snap["wall_s"] == pytest.approx(8.0)
    assert snap["steps"] == 500
    md = snap["phases"]["md"]
    assert md["seconds"] == pytest.approx(6.0) and md["calls"] == 2
    assert md["mean_s"] == pytest.approx(3.0)
    assert md["fraction_of_wall"] == pytest.approx(0.75)
    assert md["seconds_per_2000_steps"] == pytest.approx(24.0)
    assert snap["phases"]["sample.fetch"]["seconds"] == pytest.approx(1.0)
    # nested phases ("a.b") are excluded from the top-level sum
    assert snap["top_level_sum_s"] == pytest.approx(7.5)
    assert snap["unattributed_s"] == pytest.approx(0.5)


def test_phase_records_time_even_when_body_raises():
    clock = FakeClock()
    timers = PhaseTimers(enabled=True, clock=clock)
    timers.begin()
    with pytest.raises(ValueError):
        with timers.phase("exchange"):
            clock.advance(2.0)
            raise ValueError("boom")
    assert timers.snapshot()["phases"]["exchange"]["seconds"] == pytest.approx(2.0)


def test_write_is_atomic_json_with_extra(tmp_path):
    clock = FakeClock()
    timers = PhaseTimers(enabled=True, clock=clock)
    timers.begin()
    with timers.phase("md"):
        clock.advance(1.0)
    timers.note_steps(10)
    path = tmp_path / "production_phase_timers.json"
    timers.write(path, extra={"npt": {"attempts": 3}})
    data = json.loads(path.read_text())
    assert data["phases"]["md"]["calls"] == 1
    assert data["npt"] == {"attempts": 3}
    assert not list(tmp_path.glob("*.tmp"))


def test_summary_line_names_largest_phases():
    clock = FakeClock()
    timers = PhaseTimers(enabled=True, clock=clock)
    timers.begin()
    with timers.phase("md"):
        clock.advance(9.0)
    with timers.phase("sample"):
        clock.advance(1.0)
    timers.note_steps(2000)
    line = timers.summary_line()
    assert line.startswith("[phase-timers]")
    assert "md 90.0%" in line and "sample 10.0%" in line


def test_aggregate_npt_timings_sums_controllers_and_skips_missing():
    ctrl = SimpleNamespace(_timings={"read_s": 1.0, "evaluate_s": 2.0, "attempts": 3})
    drivers = [
        SimpleNamespace(controller=ctrl),
        SimpleNamespace(controller=SimpleNamespace(_timings={"read_s": 0.5, "evaluate_s": 0.5, "attempts": 1})),
        SimpleNamespace(controller=None),
        SimpleNamespace(),
    ]
    agg = aggregate_npt_timings(drivers)
    assert agg == {"read_s": 1.5, "evaluate_s": 2.5, "attempts": 4, "controllers": 2}
    assert aggregate_npt_timings([SimpleNamespace(controller=None)]) == {"controllers": 0}


def test_flag_defaults_off_and_is_reachable_from_cli_and_yaml(tmp_path):
    assert parse_args(MINIMAL).production_phase_timers is False
    assert parse_args(MINIMAL + ["--production-phase-timers"]).production_phase_timers is True
    cfg = tmp_path / "c.yaml"
    cfg.write_text("production_phase_timers: true\n")
    assert parse_args(MINIMAL + ["--config", str(cfg)]).production_phase_timers is True
