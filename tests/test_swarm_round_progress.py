"""The swarm round reports as production epoch 0 (one aggregated stream, not per-member bars)."""
from __future__ import annotations

import json
import threading
import types
from pathlib import Path

from gareus.progress import GuiProgressSink
from gareus.swarm.members import member_step_counts
from gareus.swarm.round_progress import PHASE, make_round_progress


class _Recorder:
    def __init__(self):
        self.calls = []

    def progress(self, phase, step, total, **kw):
        self.calls.append({"phase": phase, "step": step, "total": total, **kw})


def _rp(sink, **kw):
    params = dict(n_members=4, n_prod_steps=1000, n_already_done=1, timestep_fs=2.0, round_index=0)
    params.update(kw)
    rp = make_round_progress(sink, **params)
    rp.min_interval_s = 0.0
    return rp


def test_step_is_mean_member_production_and_failures_are_credited():
    rec = _Recorder()
    rp = _rp(rec)
    rp.emit()
    assert rec.calls[-1]["step"] == 250          # 1 resumed member of 4
    assert rec.calls[-1]["phase"] == PHASE == "gareus_production"
    assert rec.calls[-1]["n_replicas"] == 4
    rp.member_started(7)
    rp.add_prod_steps(7, 400)
    assert rec.calls[-1]["step"] == (1000 + 400) // 4
    rp.member_finished(7, ok=False)              # crashed at 400: the rest is credited
    last = rec.calls[-1]
    assert last["step"] == 2000 // 4
    assert last["extra"]["members_done"] == 2 and last["extra"]["members_failed"] == 1
    assert last["extra"]["swarm"] is True and last["extra"]["epoch"] == 0
    assert last["extra"]["members_running"] == 0


def test_concurrent_members_reach_exactly_100_percent():
    rec = _Recorder()
    rp = _rp(rec, n_members=60, n_already_done=0)

    def member(mid):
        rp.member_started(mid)
        for _ in range(10):
            rp.add_prod_steps(mid, 100)
        rp.member_finished(mid, ok=True)

    threads = [threading.Thread(target=member, args=(m,)) for m in range(60)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert rec.calls[-1]["step"] == rec.calls[-1]["total"] == 1000
    assert rec.calls[-1]["extra"]["members_done"] == 60


def test_no_sink_means_no_aggregator():
    assert make_round_progress(None, n_members=4, n_prod_steps=1, n_already_done=0,
                               timestep_fs=2.0, round_index=0) is None


def test_member_step_counts_whole_frames():
    args = types.SimpleNamespace(timestep_fs=2.0, swarm_output_interval_ps=2.5, swarm_seed_ns=5.0,
                                 swarm_equil_ps=100.0)
    spf, n_prod, n_equil = member_step_counts(args)
    assert (spf, n_prod, n_equil) == (1250, 2_500_000, 50_000)


def test_monitor_shows_swarm_as_epoch0_production_without_pool_warnings(tmp_path: Path):
    import gareus_monitor as M

    run = tmp_path / "c10"
    run.mkdir()
    sink = GuiProgressSink(run, types.SimpleNamespace(progress_mode="jsonl", tui_mode="none"))
    rp = _rp(sink, n_members=174, n_prod_steps=2_500_000, n_already_done=20)
    rp.emit()
    sink.close()
    assert json.loads((run / "progress.jsonl").read_text().splitlines()[-1])["swarm"] is True

    s = M.PeptideState.from_rundir(run)
    s.refresh()
    assert s.phase == "gareus_production"
    assert s.epochs_completed == 0
    assert s.n_replicas == 174
    assert abs(s.percent - 100.0 * 20 / 174) < 0.1
    codes = {d.code for d in M.diagnostics_for_state(s)}
    assert "missing_pool" not in codes and "checkpoint_missing" not in codes


def test_close_resets_the_shared_sink_baseline_for_later_production(tmp_path: Path):
    sink = GuiProgressSink(tmp_path, types.SimpleNamespace(progress_mode="jsonl", tui_mode="none"))
    rp = _rp(sink, n_members=4, n_prod_steps=1000, n_already_done=3)
    rp.emit()
    rp.close()
    assert PHASE not in sink.baseline_step and PHASE not in sink.phase_start
    # epoch 1 on the same sink: its own baseline (step 0), not the swarm's 750
    sink.progress(PHASE, 0, 10_000, timestep_fs=2.0, n_replicas=8, force=True)
    sink.progress(PHASE, 500, 10_000, timestep_fs=2.0, n_replicas=8, force=True)
    sink.close()
    last = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[-1])
    assert last["segment_steps"] == 500
