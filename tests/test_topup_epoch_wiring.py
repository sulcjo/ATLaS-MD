from pathlib import Path

import pytest

import gareus.adaptive_production as ap
import gareus.production as prod
from gareus.adaptive.topup_allocator import TopupPlan
from gareus.lifecycle import _graceful_shutdown
from gareus.topup_seeding import SeedMismatchError

from test_scheduled_final_interruption import _scheduled_final_campaign


@pytest.fixture(autouse=True)
def _clear():
    _graceful_shutdown.clear(); yield; _graceful_shutdown.clear()


def _drive(monkeypatch, args, out, plan, *, worker=None, seedable=None):
    calls = []

    def _fake(a, d, *r, **k):
        calls.append((Path(d).name, int(a.gamd_production_steps)))
        if worker:
            worker(Path(d).name)

    monkeypatch.setattr(prod, "run_gareus", _fake)
    monkeypatch.setattr(ap, "_topup_plan_for_phase", lambda *a, **k: plan)
    monkeypatch.setattr(ap, "_seedable_patch", seedable or (lambda plan, parents: plan))
    monkeypatch.setattr(ap, "_after_topup_update", lambda *a, **k: None)
    ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    return calls


def _topups_on(tmp_path):
    args, out = _scheduled_final_campaign(tmp_path)
    args.adaptive_production_topups = True
    return args, out


PLAN = TopupPlan(state_ids=(0, 1), steps=2000, deficit_state_ids=(0,), partner_state_ids=(1,), reason="planned")


def test_schedule_is_uniform_no_score_distribution(tmp_path):
    args, out = _scheduled_final_campaign(tmp_path)
    reg = ap.WindowStateRegistry.load(Path(out) / "adaptive_production")
    rows = ap.build_adaptive_epoch_schedule(reg, {"edges": [{"state_i": 0, "state_j": 1, "overlap": None}]},
                                            ap.AdaptiveDecisionPolicy(), epoch=1, default_steps=10_000)
    assert {r["requested_steps"] for r in rows} == {10_000}


def test_topups_on_runs_a_shortened_baseline_and_exactly_one_topup(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)
    calls = _drive(monkeypatch, args, out, PLAN)
    names = [n for n, _ in calls]
    assert names.count("baseline") == 1 and [n for n in names if n.startswith("topup_")] == ["topup_001_2000"]


def test_the_topup_budget_uses_the_unshortened_default(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)
    seen = {}

    def _capture(a, epoch_dir, registry, policy, *, full_steps):
        seen["full_steps"] = full_steps
        return TopupPlan(reason="healthy")

    monkeypatch.setattr(prod, "run_gareus", lambda a, d, *r, **k: seen.setdefault("baseline", int(a.gamd_production_steps)))
    monkeypatch.setattr(ap, "_topup_plan_for_phase", _capture)
    ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert seen["full_steps"] > seen["baseline"]                 # budget from full, baseline shortened


def test_a_healthy_plan_runs_no_topup(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)
    calls = _drive(monkeypatch, args, out, TopupPlan(reason="healthy"))
    assert not [n for n, _ in calls if n.startswith("topup_")]


def test_a_seed_mismatch_ends_the_topup_not_the_campaign(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)

    def worker(name):
        if name.startswith("topup_"):
            raise SeedMismatchError("state 1 seeded from the wrong window")

    calls = _drive(monkeypatch, args, out, PLAN, worker=worker)
    assert any(n.startswith("topup_") for n, _ in calls)          # it was attempted, the drive returned normally


def test_a_corrupt_parent_seed_index_skips_the_topup_not_the_campaign(tmp_path, monkeypatch):
    """Ruling 19: load_seed_index raising inside _seedable_patch is handled like a runtime mismatch."""
    from gareus.adaptive.topup_state import load_plan
    args, out = _topups_on(tmp_path)

    def _corrupt(plan, parents):
        raise SeedMismatchError("top-up seed index is unreadable")

    calls = _drive(monkeypatch, args, out, PLAN, seedable=_corrupt)
    assert not [n for n, _ in calls if n.startswith("topup_")]
    assert load_plan(Path(out) / "adaptive_production" / "final").reason == "seed_mismatch"


def test_windows_without_a_final_state_are_left_out(tmp_path, monkeypatch):
    args, out = _topups_on(tmp_path)
    shrunk = TopupPlan(state_ids=(0,), steps=2000, deficit_state_ids=(0,), reason="planned")
    calls = _drive(monkeypatch, args, out, PLAN, seedable=lambda plan, parents: shrunk)
    assert [n for n, _ in calls if n.startswith("topup_")] == ["topup_001_2000"]


def test_the_seedable_patch_is_given_the_same_parents_the_segment_loads(tmp_path, monkeypatch):
    from gareus.topup_seeding import topup_parent_dirs_by_creation_order
    args, out = _topups_on(tmp_path)
    seen = {}

    def _record(plan, parents):
        seen["parents"] = [Path(p) for p in parents]
        return plan

    _drive(monkeypatch, args, out, PLAN, seedable=_record)
    final_dir = Path(out) / "adaptive_production" / "final"
    assert seen["parents"] == topup_parent_dirs_by_creation_order(final_dir / "topup_001_2000")
    assert final_dir / "baseline" in seen["parents"]


def test_a_saved_plan_is_discarded_when_the_layout_changed(tmp_path, monkeypatch):
    from gareus.adaptive.topup_state import save_plan
    args, out = _topups_on(tmp_path)
    reg = ap.WindowStateRegistry.load(Path(out) / "adaptive_production")
    epoch_dir = Path(out) / "adaptive_production" / "final"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    save_plan(epoch_dir, TopupPlan(state_ids=(999,), steps=1000, deficit_state_ids=(999,), reason="planned"))
    monkeypatch.setattr(ap, "build_union_state_mbar_inputs", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no data")))
    plan = ap._topup_plan_for_phase(args, epoch_dir, reg, ap.policy_from_args(args), full_steps=10_000)
    assert 999 not in plan.state_ids and plan.reason == "no_diagnostics"


def test_a_saved_planned_plan_is_reused_on_resume(tmp_path, monkeypatch):
    """An interrupted top-up resumes the SAME plan, never a fresh one."""
    from gareus.adaptive.topup_state import save_plan
    args, out = _topups_on(tmp_path)
    reg = ap.WindowStateRegistry.load(Path(out) / "adaptive_production")
    epoch_dir = Path(out) / "adaptive_production" / "final"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    save_plan(epoch_dir, PLAN)
    monkeypatch.setattr(ap, "build_union_state_mbar_inputs",
                        lambda *a, **k: pytest.fail("a reusable saved plan must not be recomputed"))
    plan = ap._topup_plan_for_phase(args, epoch_dir, reg, ap.policy_from_args(args), full_steps=10_000)
    assert plan == PLAN


def test_a_finished_topup_is_never_replanned_even_after_a_layout_change(tmp_path, monkeypatch):
    """A saved plan whose top-up already finished (or was refused) is final for the phase."""
    from gareus.adaptive.topup_state import save_plan
    args, out = _topups_on(tmp_path)
    reg = ap.WindowStateRegistry.load(Path(out) / "adaptive_production")
    epoch_dir = Path(out) / "adaptive_production" / "final"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    done = TopupPlan(state_ids=(999,), steps=1000, deficit_state_ids=(999,), reason="completed")
    save_plan(epoch_dir, done)
    monkeypatch.setattr(ap, "build_union_state_mbar_inputs",
                        lambda *a, **k: pytest.fail("a finished phase must not be replanned"))
    assert ap._topup_plan_for_phase(args, epoch_dir, reg, ap.policy_from_args(args), full_steps=10_000) == done


def test_one_topup_per_phase_across_a_resume(tmp_path, monkeypatch):
    """A phase whose top-up is recorded in topup_state.json plans and runs no second one."""
    from gareus.adaptive.topup_state import load_state, save_state
    args, out = _topups_on(tmp_path)
    adaptive_dir = Path(out) / "adaptive_production"
    st = load_state(adaptive_dir)
    st["wall_time"].append({"segment": "final/topup_001_5000", "n_states": 2, "steps": 5000,
                            "predicted_h": 1.0, "realised_h": 1.0})
    save_state(adaptive_dir, st)
    planned = []
    monkeypatch.setattr(prod, "run_gareus", lambda a, d, *r, **k: planned.append(Path(d).name))
    monkeypatch.setattr(ap, "_topup_plan_for_phase", lambda *a, **k: planned.append("PLANNED") or PLAN)
    ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert "PLANNED" not in planned and not [n for n in planned if n.startswith("topup_")]


def test_a_completed_topup_is_recorded_and_its_update_runs_once(tmp_path, monkeypatch):
    from gareus.adaptive.topup_state import load_plan
    args, out = _topups_on(tmp_path)
    updates = []
    monkeypatch.setattr(prod, "run_gareus", lambda a, d, *r, **k: None)
    monkeypatch.setattr(ap, "_topup_plan_for_phase", lambda *a, **k: PLAN)
    monkeypatch.setattr(ap, "_seedable_patch", lambda plan, parents: plan)
    monkeypatch.setattr(ap, "_after_topup_update", lambda *a, **k: updates.append(a[4]))
    ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert updates == [PLAN]
    assert load_plan(Path(out) / "adaptive_production" / "final").reason == "completed"


def test_union_edge_overlaps_reach_the_epoch_diagnostics():
    policy = ap.AdaptiveDecisionPolicy()
    diagnostics = {"edges": [
        {"state_i": 1, "state_j": 0, "edge_type": "rung", "window_i": 1, "window_j": 0,
         "overlap": None, "mbar_overlap": None, "warnings": ["rung_overlap_unavailable"]},
        {"state_i": 1, "state_j": 2, "edge_type": "geometry", "overlap": 0.4, "warnings": []},
        {"state_i": 2, "state_j": 3, "edge_type": "geometry", "overlap": 0.4, "warnings": []},
    ]}
    low = float(policy.min_rung_overlap) / 2.0
    ap._apply_union_edge_overlap(diagnostics, {(0, 1): low, (1, 2): 0.7}, policy)
    rung, spatial, unmeasured = diagnostics["edges"]
    assert rung["mbar_overlap"] == low and rung["warnings"] == ["low_rung_overlap"]
    assert ap._edge_is_measured_weak(rung, policy)
    assert spatial["mbar_overlap"] == 0.7 and spatial["overlap"] == 0.4
    assert "mbar_overlap" not in unmeasured


def test_the_plan_uses_the_campaign_end_union_source_filters(tmp_path, monkeypatch):
    """The tICA guard must apply mid-campaign too; the campaign-end artifacts are never overwritten."""
    args, out = _topups_on(tmp_path)
    args.tica_cv_version = "tica_v3"
    reg = ap.WindowStateRegistry.load(Path(out) / "adaptive_production")
    epoch_dir = Path(out) / "adaptive_production" / "final"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    seen = {}

    def _capture(*a, **k):
        seen.update(k)
        raise RuntimeError("stop")

    monkeypatch.setattr(ap, "build_union_state_mbar_inputs", _capture)
    ap._topup_plan_for_phase(args, epoch_dir, reg, ap.policy_from_args(args), full_steps=10_000)
    assert seen["tica_cv_version"] == "tica_v3"
    assert seen["output_prefix"] != "adaptive_union_mbar"
