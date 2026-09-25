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


def _run(monkeypatch, args, epoch_dir, reg, schedule, plan):
    """Run one scheduled phase; returns [(segment, steps, resume)]."""
    calls, done = [], {}

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
                                    policy=ap.policy_from_args(args))
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
