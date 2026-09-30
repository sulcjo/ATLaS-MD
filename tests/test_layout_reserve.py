"""Spec P1: --swarm-adaptive-reserve-fraction leaves part of max_replicas unfilled.

The swarm layout reserves floor(f * max_replicas) replicas, records the reserve in
layout_plan.json (v2) and exposes it to the adaptive side; reserve_allowances splits the
live free slots per epoch (add_rung <= 1/3, resolution <= 1/2 of what remains). f = 0 is
today's layout, byte for byte.
"""
import json

import pytest

from gareus.adaptive.reserve_budget import (ReserveAllowance, reserve_allowances, reserved_replicas,
                                            validate_reserve_fraction)
from gareus.layout_plan import adaptive_reserve, plan_states
from gareus.swarm.ladder_design import (LAYOUT_STATUS_INSUFFICIENT, LAYOUT_STATUS_PROPOSED,
                                        design_exploration_layout, layout_plan_record, layout_rows)


def _plan(**kw):
    base = dict(n_rungs=4, max_replicas=236, region_centre_indices=[0, 9])
    base.update(kw)
    return design_exploration_layout(16, 6, **base)


def _record(plan):
    rows = layout_rows(plan, [0.1 * i for i in range(16)], [100.0] * 16,
                       [-1.0 + 0.4 * j for j in range(6)], [1.2] * 6)
    return layout_plan_record(plan, rows, [0.0, 0.2, 0.5, 1.0])


def test_reserved_replicas_is_the_floor_of_the_fraction():
    assert reserved_replicas(236, 0.15) == 35
    assert reserved_replicas(236, 0.0) == 0
    assert reserved_replicas(0, 0.5) == 0          # unlimited cap: nothing to reserve


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5, float("nan")])
def test_reserve_fraction_outside_zero_one_is_refused(bad):
    with pytest.raises(ValueError):
        validate_reserve_fraction(bad)


def test_zero_reserve_is_todays_layout_byte_for_byte():
    today = _plan()
    zero = _plan(reserve_fraction=0.0)
    assert zero == today
    assert "adaptive_reserve" not in zero
    assert json.dumps(_record(zero), sort_keys=True) == json.dumps(_record(today), sort_keys=True)
    assert "adaptive_reserve" not in _record(zero)


def test_reserve_leaves_slots_empty_and_is_recorded():
    plan = _plan(reserve_fraction=0.15)
    assert plan["status"] == LAYOUT_STATUS_PROPOSED
    assert plan["cap_spatial"] == 236 // 4                      # the full cap is unchanged
    rec = plan["adaptive_reserve"]
    assert rec["fraction"] == 0.15 and rec["max_replicas"] == 236 and rec["n_rungs"] == 4
    assert rec["reserved_replicas_requested"] == 35
    assert rec["fill_cap_spatial"] == (236 - 35) // 4 == plan["spatial_states"]
    assert rec["granted_states"] == plan["spatial_states"] * 4 == 200
    assert rec["free_slots"] == 236 - 200 >= 35 and rec["reserve_shortfall"] == 0
    # the mandatory stacks are still there
    assert plan["state_roles"][:3] == ["unrestrained_anchor", "region_representative", "region_representative"]
    record = _record(plan)
    assert record["adaptive_reserve"] == rec
    assert record["n_states"] == 200


def test_reserve_never_turns_a_feasible_layout_insufficient():
    # cap 6 spatial states; 1 anchor + 5 representatives = 6 mandatory, a 50 % reserve cannot fit
    plan = design_exploration_layout(8, 3, n_rungs=4, max_replicas=24, region_centre_indices=[0, 1, 2, 3, 4],
                                     reserve_fraction=0.5)
    assert plan["status"] == LAYOUT_STATUS_PROPOSED and plan["spatial_states"] == 6
    rec = plan["adaptive_reserve"]
    assert rec["reserved_replicas_requested"] == 12 and rec["free_slots"] == 0
    assert rec["reserve_shortfall"] == 12
    # ...and an infeasible one is still reported as before
    bad = design_exploration_layout(8, 3, n_rungs=4, max_replicas=20, region_centre_indices=[0, 1, 2, 3, 4, 5],
                                    reserve_fraction=0.5)
    assert bad["status"] == LAYOUT_STATUS_INSUFFICIENT


def test_adaptive_reserve_reader_on_v1_v2_and_reserve_plans():
    assert adaptive_reserve({"version": "layout_plan_v1", "states": []}) is None
    assert adaptive_reserve(_record(_plan())) is None
    rec = adaptive_reserve(_record(_plan(reserve_fraction=0.15)))
    assert rec is not None and rec["reserved_replicas_requested"] == 35
    # the reserve record never adds states
    assert len(plan_states(_record(_plan(reserve_fraction=0.15)))) == 200


def test_allowances_split_one_third_add_rung_then_half_of_the_rest():
    reserve = {"fraction": 0.15, "max_replicas": 236, "reserved_replicas_requested": 35}
    out = reserve_allowances(236, 200, reserve, n_rungs=4)
    assert isinstance(out, ReserveAllowance) and out.governed
    assert out.free_slots == 36 and out.add_rung_slots == 12
    assert out.resolution_slots == (36 - 12) // 2 == 12
    assert out.add_rung_centres == 3 and out.resolution_centres == 3
    # add_rung did not fire: resolution gets half of all free slots
    idle = reserve_allowances(236, 200, reserve, n_rungs=4, add_rung_taken=0)
    assert idle.resolution_slots == 18 and idle.add_rung_slots == 12
    # add_rung_taken is clamped to its own allowance
    greedy = reserve_allowances(236, 200, reserve, n_rungs=4, add_rung_taken=30)
    assert greedy.resolution_slots == 12


def test_allowances_without_a_reserve_or_cap():
    unlimited = reserve_allowances(0, 200, {"fraction": 0.15})
    assert not unlimited.governed and unlimited.reason == "unlimited" and unlimited.free_slots is None
    none = reserve_allowances(236, 200, None)
    assert not none.governed and none.reason == "no_reserve"
    assert none.resolution_slots == 0 and none.add_rung_slots is None      # add_rung: today's budget rule
    full = reserve_allowances(236, 236, {"fraction": 0.15, "max_replicas": 236})
    assert full.governed and full.free_slots == 0 and full.add_rung_slots == 0 and full.resolution_slots == 0
    over = reserve_allowances(236, 240, {"fraction": 0.15, "max_replicas": 236})
    assert over.free_slots == 0
    moved = reserve_allowances(300, 200, {"fraction": 0.15, "max_replicas": 236})
    assert moved.reason == "cap_changed" and moved.free_slots == 100


def _analysis_bytes(out):
    an = out / "swarm" / "analysis"
    return {name: (an / name).read_bytes() for name in ("windows_lambda_ladder.csv", "layout_plan.json",
                                                        "cv_selection_report.json")}


def test_swarm_analyze_zero_reserve_and_uniform_layout_are_todays_bytes(synthetic_swarm, swarm_args):
    from gareus.swarm.analyze import analyze_swarm_stage
    out_default = synthetic_swarm(with_features=True, wide_anchor=True)
    analyze_swarm_stage(out_default, swarm_args(secondary_cv="auto"))
    out_explicit = synthetic_swarm(with_features=True, wide_anchor=True)
    analyze_swarm_stage(out_explicit, swarm_args(secondary_cv="auto", swarm_adaptive_reserve_fraction=0.0,
                                                 swarm_cv2_layout="uniform", swarm_cv2_min_mode_members=8))
    assert _analysis_bytes(out_default) == _analysis_bytes(out_explicit)


def test_swarm_analyze_records_the_reserve_in_layout_plan(synthetic_swarm, swarm_args):
    from gareus.swarm.analyze import analyze_swarm_stage
    out = synthetic_swarm(with_features=True, wide_anchor=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto", max_replicas=64,
                                                 swarm_adaptive_reserve_fraction=0.25))
    assert report["status"] == "pass"
    plan = json.loads((out / "swarm" / "analysis" / "layout_plan.json").read_text())
    rec = plan["adaptive_reserve"]
    assert rec["reserved_replicas_requested"] == 16
    assert plan["n_states"] <= 64 - 16 and rec["free_slots"] == 64 - plan["n_states"]


def test_reserve_flags_parse_and_reach_the_swarm_yaml_section(tmp_path):
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "x")])
    assert a.swarm_adaptive_reserve_fraction == 0.0 and a.swarm_cv2_layout == "uniform"
    assert a.swarm_cv2_min_mode_members == 8
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("seq: GYDPETGTWG\nout: /tmp/x\nswarm:\n  adaptive_reserve_fraction: 0.15\n"
                   "  cv2_layout: shape\n  cv2_min_mode_members: 10\n")
    b = parse_args(["--config", str(cfg)])
    assert b.swarm_adaptive_reserve_fraction == 0.15 and b.swarm_cv2_layout == "shape"
    assert b.swarm_cv2_min_mode_members == 10
    with pytest.raises(SystemExit):
        parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "y"), "--swarm-adaptive-reserve-fraction", "1.0"])


def test_reserve_and_layout_flags_are_recorded_in_method_settings():
    import types
    from gareus.provenance import _method_settings
    ms = _method_settings(types.SimpleNamespace(swarm_adaptive_reserve_fraction=0.15, swarm_cv2_layout="shape",
                                                swarm_cv2_min_mode_members=8))
    assert ms["swarm_adaptive_reserve_fraction"] == 0.15 and ms["swarm_cv2_layout"] == "shape"
    assert ms["swarm_cv2_min_mode_members"] == 8
