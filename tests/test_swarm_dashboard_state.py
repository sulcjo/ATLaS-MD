"""Swarm dashboard live state (spec docs/superpowers/specs/2026-10-04-swarm-dashboard-design.md)."""
from __future__ import annotations

import json
import threading
import time
import types
from pathlib import Path

import pytest

from gareus.swarm import round_progress as RP
from gareus.swarm.round_progress import SwarmRoundProgress

EDGES = {"cv1": [0.0, 0.1, 0.2, 0.3, 0.4], "rg": [0.5, 0.7, 0.9, 1.1], "e2e": [0.5, 1.0, 1.5, 2.0]}


class _Sink:
    def __init__(self, tui_mode="dashboard", mode="both", interval=0.0):
        self.tui_mode, self.mode = tui_mode, mode
        self.args = types.SimpleNamespace(dashboard_render_interval_sec=interval, tui_glyphs="unicode",
                                          tui_mode=tui_mode, tui_clear_mode="never", dashboard_max_height=0)
        self.progress_calls, self.events = [], []
        self.baseline_step, self.phase_start = {}, {}

    def progress(self, phase, step, total, **kw):
        self.progress_calls.append({"phase": phase, "step": step, "total": total, **kw})

    def emit(self, event):
        self.events.append(event)


def _rp(sink=None, n=6, done=(), **kw):
    return SwarmRoundProgress(sink or _Sink(tui_mode="none"), n_members=n, n_prod_steps=1000,
                              n_already_done=len(done), timestep_fs=2.0, round_index=0,
                              member_ids=list(range(n)), done_ids=list(done),
                              cells={i: f"{i % 2}_0_0" for i in range(n)}, edges=EDGES, run_label="c10", **kw)


def test_phases_follow_the_hooks():
    rp = _rp(done=[5])
    assert rp.snapshot().phases == ("queued",) * 5 + ("done",)
    rp.member_started(0, device="1")
    rp.member_phase(0, "grafting")
    rp.member_phase(1, "graft_wait")
    rp.member_phase(2, "equilibrating")
    rp.add_prod_steps(3, 100)
    s = rp.snapshot()
    assert s.phases[:4] == ("grafting", "graft_wait", "equilibrating", "production")
    assert s.production_fraction[3] == pytest.approx(0.1)
    rp.member_finished(3, ok=False, status="md_failed", ns_per_day=0.0)
    rp.member_finished(0, ok=True, status="ok", ns_per_day=120.0)
    s = rp.snapshot()
    assert s.phases[0] == "done" and s.phases[3] == "failed"
    assert [e["member_id"] for e in s.events] == [0, 3] and s.events[1]["status"] == "md_failed"
    assert s.events[0]["cell_id"] == "0_0_0"
    rp.member_phase(0, "production")           # a finished member never goes back
    assert rp.snapshot().phases[0] == "done"


def test_coverage_counters_follow_plan_edges():
    rp = _rp()
    rp.add_frame(0, 0.05, 0.6, 0.7)            # cell (0, 0, 0)
    rp.add_frame(0, 0.35, 1.0, 1.9)            # cell (3, 2, 2)
    rp.add_frame(1, 0.05, 0.6, 0.7)            # same cell again
    rp.add_frame(1, -5.0, 0.6, 0.7)            # far below the cv1 range: counted below, clipped into cell 0
    rp.add_frame(2, float("nan"), 0.6, 0.7)    # incomplete row: no cell
    s = rp.snapshot()
    assert s.cells_total == 4 * 3 * 3 and s.cells_visited == 2
    cv1 = next(h for h in s.histograms if h.name == "cv1")
    assert sum(cv1.counts) == 3 and cv1.below == 1
    assert s.bins == (4, 3, 3)


def test_snapshot_is_detached():
    rp = _rp()
    rp.member_started(0, device="0")
    s = rp.snapshot()
    rp.member_finished(0, ok=True)
    assert s.phases[0] == "graft_wait"


def test_rates_etas_and_stalls(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(RP.time, "time", lambda: clock[0])
    rp = _rp(n=3)
    for m in (0, 1):
        rp.member_started(m, device=str(m))
        rp.member_phase(m, "production")
    for _ in range(10):
        clock[0] += 1.0
        rp.add_prod_steps(0, 20)               # 20 steps/s
        rp.add_prod_steps(1, 5)                # 5 steps/s: the straggler
    s = rp.snapshot()
    assert s.aggregate_ns_per_day == pytest.approx(25 * 2e-6 * 86400, rel=1e-6)
    # mean: (800 + 950 + 1000 queued) steps left at 25 steps/s; last: the straggler, 950 at 5 steps/s
    # (it outlasts member 2, queued, whose full 1000 steps at the 12.5 median take 80 s)
    assert s.eta_mean_s == pytest.approx(2750 / 25, rel=1e-6)
    assert s.eta_last_s == pytest.approx(950 / 5, rel=1e-6) and s.eta_last_s >= s.eta_mean_s
    clock[0] += RP.STALL_S + 1
    assert set(rp.snapshot().stalled) == {0, 1}


def test_device_slow_flag(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(RP.time, "time", lambda: clock[0])
    rp = _rp(n=3)
    for m, dev in ((0, "0"), (1, "1"), (2, "2")):
        rp.member_started(m, device=dev)
        rp.member_phase(m, "production")
    for _ in range(5):
        clock[0] += 1.0
        rp.add_prod_steps(0, 20)
        rp.add_prod_steps(1, 20)
        rp.add_prod_steps(2, 4)
    devs = {d.device: d for d in rp.snapshot().devices}
    assert devs["2"].slow and not devs["0"].slow and not devs["1"].slow


def test_concurrent_hooks_are_consistent():
    rp = _rp(n=60)

    def member(m):
        rp.member_started(m, device=str(m % 4))
        rp.member_phase(m, "grafting")
        rp.member_phase(m, "equilibrating")
        for _ in range(10):
            rp.add_prod_steps(m, 100)
            rp.add_frame(m, 0.1, 0.6, 0.7)
        rp.member_finished(m, ok=True)

    threads = [threading.Thread(target=member, args=(m,)) for m in range(60)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s = rp.snapshot()
    assert s.count("done") == 60 and s.mean_fraction == pytest.approx(1.0)
    assert sum(next(h for h in s.histograms if h.name == "cv1").counts) == 600


def test_frames_are_throttled_and_never_block(monkeypatch):
    frames = []
    monkeypatch.setattr("gareus.tui.write_tui_frame", lambda text, args: frames.append(text))
    clock = [0.0]
    monkeypatch.setattr(RP.time, "time", lambda: clock[0])
    rp = _rp(sink=_Sink(interval=0.0))
    assert rp.frames_enabled and rp.frame_interval_s == RP.DEFAULT_FRAME_INTERVAL_S
    for m in range(6):                         # forced emits inside one interval: one frame
        rp.member_started(m, device="0")
    assert len(frames) == 1
    clock[0] += RP.DEFAULT_FRAME_INTERVAL_S
    rp.add_prod_steps(0, 10)
    assert len(frames) == 2
    # a render in progress is skipped, not waited for
    rp._render_lock.acquire()
    clock[0] += 10.0
    rp.add_prod_steps(0, 10)
    rp._render_lock.release()
    assert len(frames) == 2
    assert "SWARM round 0" in frames[0]
    rp.close()                                 # the final frame is always drawn
    assert len(frames) == 3


def test_no_frames_outside_dashboard_modes(monkeypatch):
    frames = []
    monkeypatch.setattr("gareus.tui.write_tui_frame", lambda text, args: frames.append(text))
    for sink in (_Sink(tui_mode="none"), _Sink(tui_mode="progress"), _Sink(mode="jsonl")):
        rp = _rp(sink=sink)
        rp.member_started(0, device="0")
        rp.close()
        assert not rp.frames_enabled
        assert all(not c["extra"]["swarm_dashboard"] for c in sink.progress_calls)
    assert frames == []


def test_render_error_disables_frames_not_members(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("tty gone")
    monkeypatch.setattr("gareus.dashboard.swarm_view.render_swarm_screen", boom)
    sink = _Sink()
    rp = _rp(sink=sink)
    rp.member_started(0, device="0")
    rp.add_prod_steps(0, 10)
    rp.member_finished(0, ok=True)
    assert rp._render_failed and "swarm dashboard disabled" in capsys.readouterr().err
    assert sink.progress_calls[-1]["extra"]["swarm_dashboard"] is False   # the bar comes back


def test_live_status_written_atomically(tmp_path: Path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(RP.time, "time", lambda: clock[0])
    path = tmp_path / "live_status.json"
    rp = _rp(live_status_path=path)
    rp.member_started(0, device="0")
    assert json.loads(path.read_text())["counts"]["graft_wait"] == 1
    rp.member_finished(0, ok=True)
    assert json.loads(path.read_text())["counts"]["graft_wait"] == 1      # throttled (15 s)
    rp.close()
    rec = json.loads(path.read_text())
    assert rec["counts"]["done"] == 1 and rec["schema_version"] == "swarm_live_status_v1"
    assert not list(tmp_path.glob("*.tmp*"))


def test_sink_drops_its_bar_when_the_swarm_dashboard_owns_the_screen(tmp_path, capsys, monkeypatch):
    from gareus.progress import GuiProgressSink
    monkeypatch.setattr("gareus.progress.write_tui_frame", lambda text, args: print("FRAME:" + text), raising=False)
    args = types.SimpleNamespace(progress_mode="console", tui_mode="dashboard", progress_update_interval_sec=0.0,
                                 tui_clear_mode="never", dashboard_max_height=0)
    sink = GuiProgressSink(tmp_path, args)
    sink.progress("gareus_production", 1, 10, force=True, extra={"swarm": True, "swarm_dashboard": True})
    assert "gareus_production" not in capsys.readouterr().out
    sink.progress("gareus_production", 2, 10, force=True, extra={"swarm": True, "swarm_dashboard": False})
    assert "gareus_production" in capsys.readouterr().out
