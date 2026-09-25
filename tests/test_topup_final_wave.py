"""Final-review fix wave for effective top-ups (ruling 33): I2, I3.

Drives ``run_scheduled_adaptive_epoch`` directly with a fake ``run_gareus`` so
the segment calls (name, steps, resume flag) are observable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import gareus.adaptive_production as ap
from gareus.adaptive.topup_allocator import TopupPlan
from gareus.adaptive.topup_state import load_plan, save_plan
from gareus.lifecycle import _graceful_shutdown


@pytest.fixture(autouse=True)
def _clear():
    _graceful_shutdown.clear(); yield; _graceful_shutdown.clear()


def _campaign(tmp_path, *, topups=True, resume=False):
    from gareus.cli import parse_args
    out = tmp_path / "run"
    adaptive = out / "adaptive_production"
    adaptive.mkdir(parents=True)
    reg = ap.WindowStateRegistry()
    for c in (0.0, 0.05, 0.10):
        reg.add_state(c, 200.0, epoch=0, source="seed")
    reg.save(adaptive)
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(out)])
    args.adaptive_production_topups = topups
    args.adaptive_production_resume = resume
    return args, adaptive, reg


def _rows(requested, baseline):
    return [{"state_id": i, "requested_steps": r, "baseline_steps": b, "extra_steps": r - b,
             "score": 0.0, "sample_count": 0, "allocation_reason": "baseline"}
            for i, (r, b) in enumerate(zip(requested, baseline))]


def _run(monkeypatch, args, epoch_dir, reg, schedule, plan, runtime_pool=None, done=None):
    """Run one scheduled phase; returns [(segment, steps, resume)]."""
    calls = []
    done = {} if done is None else done

    def _fake_run_gareus(a, d, *rest, **kw):
        calls.append((Path(d).name, int(a.gamd_production_steps), bool(a.resume)))
        done[Path(d)] = int(a.gamd_production_steps)

    def _plan(a, ed, registry, policy, *, full_steps):
        save_plan(ed, plan)
        return plan

    monkeypatch.setattr(ap, "production_checkpoint_available", lambda d: Path(d) in done)
    monkeypatch.setattr(ap, "_segment_checkpoint_prod_done", lambda d: done.get(Path(d)))
    monkeypatch.setattr(ap, "_topup_plan_for_phase", _plan)
    monkeypatch.setattr(ap, "_seedable_patch", lambda plan, parents: plan)
    monkeypatch.setattr(ap, "_after_topup_update", lambda *a, **k: None)
    ap.run_scheduled_adaptive_epoch(args, epoch_dir, reg, schedule, _fake_run_gareus,
                                    None, None, None, None, None, None,
                                    policy=ap.policy_from_args(args), runtime_pool=runtime_pool)
    return calls


# ---- I2: full_steps from requested_steps, not the old schedule's min baseline ----

def test_full_steps_of_an_old_score_allocator_schedule_is_the_requested_mean():
    old = _rows([10_000, 20_000, 30_000], [1_000, 1_000, 1_000])
    assert ap._schedule_full_steps(old) == 20_000


def test_full_steps_of_a_new_uniform_schedule_is_its_per_state_length():
    assert ap._schedule_full_steps(_rows([12_000] * 3, [12_000] * 3)) == 12_000


def test_topups_on_resume_of_an_old_schedule_budgets_from_the_requested_steps(tmp_path, monkeypatch):
    args, adaptive, reg = _campaign(tmp_path)
    seen = {}

    def _plan(a, ed, registry, policy, *, full_steps):
        seen["full_steps"] = full_steps
        return TopupPlan(reason="healthy")

    monkeypatch.setattr(ap, "_topup_plan_for_phase", _plan)
    calls = []
    ap.run_scheduled_adaptive_epoch(args, adaptive / "epoch_001", reg,
                                    _rows([10_000, 20_000, 30_000], [1_000, 1_000, 1_000]),
                                    lambda a, d, *r, **k: calls.append((Path(d).name, int(a.gamd_production_steps))),
                                    None, None, None, None, None, None, policy=ap.policy_from_args(args))
    assert seen["full_steps"] == 20_000
    assert calls[0] == ("baseline", 14_000)            # 0.7 x 20,000, not max(1000, 0.7 x 1,000)


def test_topups_off_baseline_is_unchanged_for_both_schedule_shapes(tmp_path, monkeypatch):
    args, adaptive, reg = _campaign(tmp_path, topups=False)
    for i, sched in enumerate((_rows([10_000, 20_000, 30_000], [1_000] * 3), _rows([12_345] * 3, [12_345] * 3))):
        calls = []
        ap.run_scheduled_adaptive_epoch(args, adaptive / f"epoch_00{i + 1}", reg, sched,
                                        lambda a, d, *r, **k: calls.append(int(a.gamd_production_steps)),
                                        None, None, None, None, None, None, policy=ap.policy_from_args(args))
        # the pre-fix off-branch formula: max(min baseline, quantized mean requested)
        req = [r["requested_steps"] for r in sched]
        want = max(min(r["baseline_steps"] for r in sched), ap._quantized_extra_steps(round(sum(req) / len(req))))
        assert calls == [want]


# ---- I3: the final phase gives an unspent top-up budget back to its baseline ----

HEALTHY = TopupPlan(reason="healthy")
PLANNED = TopupPlan(state_ids=(0, 1), steps=2000, deficit_state_ids=(0,), partner_state_ids=(1,), reason="planned")


@pytest.mark.parametrize("plan", [HEALTHY, TopupPlan(reason="cap_too_small"), TopupPlan(reason="no_diagnostics")])
def test_final_phase_extends_its_baseline_when_no_topup_ran(tmp_path, monkeypatch, plan):
    args, adaptive, reg = _campaign(tmp_path)          # fresh campaign: resume NOT requested
    calls = _run(monkeypatch, args, adaptive / "final", reg, _rows([20_000] * 3, [20_000] * 3), plan)
    assert calls == [("baseline", 14_000, False), ("baseline", 20_000, True)]


def test_final_phase_extends_after_a_seed_mismatch(tmp_path, monkeypatch):
    from gareus.topup_seeding import SeedMismatchError
    args, adaptive, reg = _campaign(tmp_path)
    calls, done = [], {}

    def _fake(a, d, *rest, **kw):
        calls.append((Path(d).name, int(a.gamd_production_steps), bool(a.resume)))
        done[Path(d)] = int(a.gamd_production_steps)

    def _mismatch(plan, parents):
        raise SeedMismatchError("bad seed")

    monkeypatch.setattr(ap, "production_checkpoint_available", lambda d: Path(d) in done)
    monkeypatch.setattr(ap, "_segment_checkpoint_prod_done", lambda d: done.get(Path(d)))
    monkeypatch.setattr(ap, "_topup_plan_for_phase", lambda a, ed, *r, **k: (save_plan(ed, PLANNED), PLANNED)[1])
    monkeypatch.setattr(ap, "_seedable_patch", _mismatch)
    ap.run_scheduled_adaptive_epoch(args, adaptive / "final", reg, _rows([20_000] * 3, [20_000] * 3), _fake,
                                    None, None, None, None, None, None, policy=ap.policy_from_args(args))
    assert load_plan(adaptive / "final").reason == "seed_mismatch"
    assert calls == [("baseline", 14_000, False), ("baseline", 20_000, True)]


def test_final_phase_does_not_extend_after_a_completed_topup(tmp_path, monkeypatch):
    args, adaptive, reg = _campaign(tmp_path)
    calls = _run(monkeypatch, args, adaptive / "final", reg, _rows([20_000] * 3, [20_000] * 3), PLANNED)
    assert [c[0] for c in calls] == ["baseline", "topup_001_2000"]


def test_a_numbered_epoch_never_extends_its_baseline(tmp_path, monkeypatch):
    args, adaptive, reg = _campaign(tmp_path)
    calls = _run(monkeypatch, args, adaptive / "epoch_001", reg, _rows([20_000] * 3, [20_000] * 3), HEALTHY)
    assert calls == [("baseline", 14_000, False)]


def test_final_extension_needs_the_baseline_checkpoint(tmp_path, monkeypatch):
    args, adaptive, reg = _campaign(tmp_path)
    calls = _run(monkeypatch, args, adaptive / "final", reg, _rows([20_000] * 3, [20_000] * 3), HEALTHY)
    assert len(calls) == 2
    monkeypatch.setattr(ap, "production_checkpoint_available", lambda d: False)
    assert ap._final_phase_strands_topup_budget(adaptive / "final", 14_000, 20_000) is False


def test_a_resumed_final_that_already_extended_charges_nothing_more(tmp_path, monkeypatch):
    """Both baseline calls hit the already-complete fast path: no MD, no pool charge."""
    args, adaptive, reg = _campaign(tmp_path, resume=True)
    final = adaptive / "final"
    done = {final / "baseline": 20_000}
    calls = []
    monkeypatch.setattr(ap, "production_checkpoint_available", lambda d: Path(d) in done)
    monkeypatch.setattr(ap, "_segment_checkpoint_prod_done", lambda d: done.get(Path(d)))
    monkeypatch.setattr(ap, "_topup_plan_for_phase", lambda a, ed, *r, **k: (save_plan(ed, HEALTHY), HEALTHY)[1])
    ap.run_scheduled_adaptive_epoch(args, final, reg, _rows([20_000] * 3, [20_000] * 3),
                                    lambda a, d, *r, **k: calls.append(Path(d).name),
                                    None, None, None, None, None, None, policy=ap.policy_from_args(args))
    assert calls == []


# ---- I1: final_window_states is exported only where a top-up can read it ----

def _segment_export_decisions(tmp_path, *, topups):
    from gareus.topup_seeding import should_export_final_window_states
    args, adaptive, reg = _campaign(tmp_path, topups=topups)
    seen = []
    ap.run_scheduled_adaptive_epoch(
        args, adaptive / "epoch_001", reg, _rows([20_000] * 3, [20_000] * 3),
        lambda a, d, *r, **k: seen.append(should_export_final_window_states(a)),
        None, None, None, None, None, None, policy=ap.AdaptiveDecisionPolicy(topups_enabled=False))
    return seen


def test_a_topups_off_segment_exports_no_final_window_states(tmp_path):
    assert _segment_export_decisions(tmp_path, topups=False) == [False]


def test_a_topups_on_segment_exports_final_window_states(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "_run_phase_topup", lambda *a, **k: None)
    assert _segment_export_decisions(tmp_path, topups=True) == [True]


def test_a_run_outside_adaptive_production_never_exports():
    from types import SimpleNamespace
    from gareus.topup_seeding import should_export_final_window_states
    assert should_export_final_window_states(SimpleNamespace(adaptive_production_topups=True)) is False
    assert should_export_final_window_states(SimpleNamespace(
        adaptive_production_topups=True, _adaptive_phase_info={"is_adaptive_epoch": False})) is False
    assert should_export_final_window_states(SimpleNamespace(
        adaptive_production_topups=False, _adaptive_phase_info={"is_adaptive_epoch": True})) is False


def test_run_gareus_guards_the_export_with_the_predicate():
    """run_gareus itself needs a live OpenMM stack; pin the guard structurally."""
    import ast
    src = (Path(__file__).resolve().parents[1] / "gareus" / "production.py").read_text()
    guarded = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.If)
               and ast.unparse(n.test) == "should_export_final_window_states(args)"]
    assert len(guarded) == 1
    calls = [c for c in ast.walk(guarded[0]) if isinstance(c, ast.Call)
             and ast.unparse(c.func) == "export_final_window_states"]
    assert len(calls) == 1
    everywhere = [c for c in ast.walk(ast.parse(src)) if isinstance(c, ast.Call)
                  and ast.unparse(c.func) == "export_final_window_states"]
    assert len(everywhere) == 1                      # no unguarded second call site


# ---- I4: a tried-out (structural) edge no longer sets its endpoints' sigma ----

def test_an_exhausted_edge_is_dropped_from_both_endpoints_sigma_neighbours():
    nb = {0: [1, 2], 1: [0], 2: [0]}
    rung = {0: [10], 1: [11], 2: [12]}
    got = ap._topup_sigma_neighbours([0, 1, 2], nb, rung, {(0, 1): 2}, 2)
    assert got == {0: [2], 1: [11], 2: [0]}          # 1 lost its only neighbour -> rung fallback


def test_an_edge_below_the_attempt_limit_still_counts():
    nb = {0: [1], 1: [0]}
    got = ap._topup_sigma_neighbours([0, 1], nb, {}, {(1, 0): 1}, 2)
    assert got == {0: [1], 1: [0]}


def test_phase_diagnostics_pass_the_filtered_sigma_neighbours(tmp_path, monkeypatch):
    args, adaptive, reg = _campaign(tmp_path)
    seen = {}
    monkeypatch.setattr(ap, "build_union_state_mbar_inputs", lambda *a, **k: {"arrays_npz": "x"})
    import gareus.adaptive.union_diagnostics as ud
    monkeypatch.setattr(ud, "union_diagnostics_from_npz",
                        lambda *a, **k: seen.setdefault("sigma_neighbours", k["sigma_neighbours"]))
    layout = ([0, 1, 2], {0: [1], 1: [0, 2], 2: [1]}, {0: [], 1: [], 2: []})
    ap._phase_union_diagnostics(args, adaptive, reg, ap.policy_from_args(args), {}, layout,
                                edge_attempts={(1, 2): 5})
    assert seen["sigma_neighbours"] == {0: [1], 1: [0], 2: []}


# ---- M2: the driver logs the plan's structural edges ----

def test_the_driver_logs_structural_edges_and_saves_them(tmp_path, monkeypatch, capsys):
    args, adaptive, reg = _campaign(tmp_path)
    edges = tuple((i, i + 1) for i in range(7))
    plan = TopupPlan(structural_edges=edges, reason="healthy")
    _run(monkeypatch, args, adaptive / "epoch_001", reg, _rows([20_000] * 3, [20_000] * 3), plan)
    out = capsys.readouterr().out
    assert "7 structural edge(s)" in out and "0-1, 1-2, 2-3, 3-4, 4-5 (+2 more)" in out
    assert tuple(tuple(e) for e in load_plan(adaptive / "epoch_001").structural_edges) == edges


# ---- memory guard: an oversized per-epoch union is skipped, not OOM-killed ----

def test_the_peak_estimate_reproduces_the_bench():
    from gareus.adaptive.union_diagnostics import estimate_union_diagnostics_peak_gb
    assert abs(estimate_union_diagnostics_peak_gb(1_000_000, 236) - 14.6) < 0.2
    assert estimate_union_diagnostics_peak_gb(250_000, 236) < 4.06


def test_the_real_builder_calls_the_guard_with_kept_rows_before_the_matrices(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_adaptive_segmented_diagnostics import N_WINDOWS, _write_parquet_epoch_run, _write_window_csv
    registry = ap.registry_from_window_csv(_write_window_csv(tmp_path / "w.csv"), epoch=0, source="t")
    adaptive = tmp_path / "adaptive"
    _write_parquet_epoch_run(adaptive / "final", n_windows=N_WINDOWS, rows_per_window=200)
    seen = []
    meta = ap.build_union_state_mbar_inputs(adaptive, registry, size_guard=lambda n, k: seen.append((n, k)))
    assert seen == [(meta["n_samples"], meta["n_states"])]
    with pytest.raises(ap.TopupDiagnosticsTooLarge):
        ap.build_union_state_mbar_inputs(adaptive, registry, output_prefix="guarded",
                                         size_guard=ap._topup_union_size_guard(1e-9))
    assert not (adaptive / "guarded.npz").exists()


def test_an_oversized_union_skips_the_plan_with_no_diagnostics_and_warns(tmp_path, monkeypatch, caplog):
    import logging
    args, adaptive, reg = _campaign(tmp_path)
    args.adaptive_production_topup_diagnostics_max_gb = 8.0
    policy = ap.policy_from_args(args)
    assert policy.topup_diagnostics_max_gb == 8.0

    def _builder(*a, size_guard=None, **k):
        size_guard(1_000_000, 236)                      # ~14.5 GB at the bench's scale
        raise AssertionError("the guard must stop the build")

    monkeypatch.setattr(ap, "build_union_state_mbar_inputs", _builder)
    (adaptive / "epoch_001").mkdir()
    with caplog.at_level(logging.WARNING):
        plan = ap._topup_plan_for_phase(args, adaptive / "epoch_001", reg, policy, full_steps=20_000)
    assert plan.reason == "no_diagnostics"
    assert "14.5 GB" in caplog.text and "--ap-topup-diagnostics-max-gb 8.0" in caplog.text


def test_the_guard_passes_a_union_under_the_limit():
    ap._topup_union_size_guard(8.0)(250_000, 236)     # ~3.6 GB: no raise


def test_the_diagnostics_limit_is_a_cli_flag():
    from gareus.cli import parse_args
    args = parse_args(["--seq", "AA", "--ap-topup-diagnostics-max-gb", "3.5"])
    assert ap.policy_from_args(args).topup_diagnostics_max_gb == 3.5
    assert ap.policy_from_args(parse_args(["--seq", "AA"])).topup_diagnostics_max_gb == 8.0


# ---- ruling 34: under an enabled MD pool the extension clips the DELTA, not the target ----

def _pool(total_ns):
    return ap.AdaptiveRuntimePool(total_ns=total_ns, timestep_fs=2.0)


def _summary(epoch_dir):
    import json
    return json.loads((Path(epoch_dir) / "scheduled_epoch_summary.json").read_text())


def test_under_a_real_pool_the_final_extension_spends_the_withheld_budget(tmp_path, monkeypatch):
    args, adaptive, reg = _campaign(tmp_path)
    pool = _pool(3 * 20_000 * 2.0 / 1e6)                 # exactly the phase's full MD: 0.12 ns
    final = adaptive / "final"
    calls = _run(monkeypatch, args, final, reg, _rows([20_000] * 3, [20_000] * 3), HEALTHY, runtime_pool=pool)
    assert calls[0] == ("baseline", 14_000, False)
    assert len(calls) == 2 and calls[1][0] == "baseline" and calls[1][2] is True
    assert 19_990 <= calls[1][1] <= 20_000               # the delta ran (floor rounding may drop a step)
    assert pool.remaining_ns() < 3 * 10 * 2.0 / 1e6      # pool ends ~spent
    rows = _summary(final)["segments"]
    assert [r["segment"] for r in rows] == ["baseline", "baseline"]
    assert not any(r.get("already_complete") or r.get("skipped_by_runtime_pool") for r in rows)
    assert _summary(final)["baseline_steps"] == calls[1][1]


def test_an_exactly_exhausted_pool_makes_the_extension_a_silent_noop(tmp_path, monkeypatch, capsys):
    args, adaptive, reg = _campaign(tmp_path)
    pool = _pool(3 * 14_000 * 2.0 / 1e6)                 # only the shortened baseline's MD
    final = adaptive / "final"
    calls = _run(monkeypatch, args, final, reg, _rows([20_000] * 3, [20_000] * 3), HEALTHY, runtime_pool=pool)
    assert calls == [("baseline", 14_000, False)]
    rows = _summary(final)["segments"]
    assert len(rows) == 1 and not rows[0].get("skipped_by_runtime_pool")
    assert _summary(final)["baseline_steps"] == 14_000
    assert capsys.readouterr().out.count("extension skipped, no new steps") == 1


def test_an_ordinary_resume_still_clips_the_full_target(tmp_path, monkeypatch):
    """force_resume=False (chignolin_9's path) is unchanged: the full target is clipped."""
    args, adaptive, reg = _campaign(tmp_path, topups=False, resume=True)
    epoch = adaptive / "epoch_001"
    pool = _pool(3 * 6_000 * 2.0 / 1e6)                  # room for 6,000 more steps/state
    done = {epoch / "baseline": 14_000}
    calls = _run(monkeypatch, args, epoch, reg, _rows([20_000] * 3, [20_000] * 3), HEALTHY,
                 runtime_pool=pool, done=done)
    assert calls == []                                    # 20,000 clipped to 6,000 < 14,000: fast path, as today
    rows = _summary(epoch)["segments"]
    assert len(rows) == 1 and rows[0]["already_complete"] and rows[0]["steps"] <= 6_000
