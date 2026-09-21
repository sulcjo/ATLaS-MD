"""F05/P09: mandatory exploration states survive every replica cap; regions are retained or the
design refuses; zero stiffness round-trips exactly (spec F05; findings I02, I03).
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from fixtures.thermodynamic_repair import cap_transition_case, separated_support_case
from gareus.cv_selection.coverage import RegionPolicy, build_region_inventory, cv1_support_intervals, region_centres
from gareus.swarm.ladder_design import (LAYOUT_STATUS_INSUFFICIENT, LAYOUT_STATUS_PROPOSED, ROLE_JOINT,
                                        ROLE_REGION_REPRESENTATIVE, ROLE_UNRESTRAINED_ANCHOR,
                                        design_exploration_layout, layout_plan_record, layout_rows,
                                        write_ladder_windows_2d_csv)


@pytest.mark.parametrize("cap", [92, 96, 128, 252, 256, 260])
def test_the_unrestrained_stack_survives_every_cap_and_stacks_are_complete(cap):
    c = cap_transition_case()
    plan = design_exploration_layout(c.n1, c.n2, n_rungs=c.n_rungs, max_replicas=cap)
    assert plan["status"] == LAYOUT_STATUS_PROPOSED
    assert plan["cells"][0] == (None, None) and plan["state_roles"][0] == ROLE_UNRESTRAINED_ANCHOR
    assert plan["spatial_states"] * c.n_rungs <= cap                       # never silently over the cap
    assert len(set(plan["cells"])) == plan["spatial_states"]
    rec = layout_plan_record(plan, layout_rows(plan, np.linspace(0, 1, c.n1), np.full(c.n1, 100.0),
                                               np.linspace(-1, 1, c.n2), np.full(c.n2, 0.7)), [0.0, 0.3, 0.6, 1.0])
    assert rec["n_states"] == plan["spatial_states"] * 4                   # a complete rung stack per spatial state
    anchor_states = [s for s in rec["states"] if s["role"] == ROLE_UNRESTRAINED_ANCHOR]
    assert len(anchor_states) == 4 and all(s["k1"] == 0.0 and s["k2"] == 0.0 for s in anchor_states)
    assert any(s["gamd_lambda"] == 0.0 for s in anchor_states)              # an actual k1=k2=lambda=0 target


def test_cap_arithmetic_matches_the_spec_examples():
    # 96 replicas / 4 rungs: one mandatory stack + at most 23 others; 256/4: at most 63 others
    p96 = design_exploration_layout(6, 4, n_rungs=4, max_replicas=96)
    assert p96["spatial_states"] == 24 and p96["kind"] == "sparse"           # 6*4 = 24 > 23 -> sparse, not the old all-joint
    # 16 x 4 = 64 joint cells do NOT fit beside the reserved stack (63 others): the layout is
    # sparse -- the anchor, the 16 + 4 single-axis soft states, then 43 joint cells on the
    # diagonal band -- and never silently exceeds 256 replicas.
    p256 = design_exploration_layout(16, 4, n_rungs=4, max_replicas=256)
    assert p256["spatial_states"] == 64 and p256["kind"] == "sparse" and p256["mandatory_spatial"] == 1
    assert p256["cells"][1:].count((None, None)) == 0
    assert sum(1 for r in p256["state_roles"] if r == ROLE_JOINT) == 43
    # one fewer CV1 centre and the joint grid fits with the stack: 15 x 4 + 1 = 61 <= 64
    p_fit = design_exploration_layout(15, 4, n_rungs=4, max_replicas=256)
    assert p_fit["kind"] == "joint" and p_fit["spatial_states"] == 61


def test_region_representatives_are_reserved_before_joint_cells_and_infeasible_budgets_refuse():
    plan = design_exploration_layout(6, 4, n_rungs=4, max_replicas=96, region_centre_indices=[0, 5])
    assert plan["state_roles"][:3] == [ROLE_UNRESTRAINED_ANCHOR, ROLE_REGION_REPRESENTATIVE, ROLE_REGION_REPRESENTATIVE]
    assert (0, None) in plan["cells"] and (5, None) in plan["cells"]
    tiny = design_exploration_layout(6, 4, n_rungs=4, max_replicas=8, region_centre_indices=[0, 1, 5])
    assert tiny["status"] == LAYOUT_STATUS_INSUFFICIENT
    assert tiny["requested_replicas"] == 16 and tiny["available_replicas"] == 8 and tiny["cells"] == []


def test_all_zero_stiffness_rows_round_trip_through_the_csv_and_loader_exactly(tmp_path):
    plan = design_exploration_layout(3, 2, n_rungs=2, max_replicas=40)
    rows = layout_rows(plan, [0.1, 0.5, 0.9], [200.0, 300.0, 250.0], [-1.0, 1.0], [0.5, 0.7])
    assert rows[0] == {"center1": 0.5, "k1": 0.0, "center2": 0.0, "k2": 0.0}       # finite placeholders, exact zeros
    path = write_ladder_windows_2d_csv(tmp_path / "w.csv", rows, [0.0, 1.0])
    with path.open() as fh:
        recs = list(csv.DictReader(fh))
    assert float(recs[0]["primary_cv_k_kcal"]) == 0.0 and float(recs[0]["secondary_cv_k_kcal_mol"]) == 0.0
    from gareus.windows import load_explicit_2d_window_csv

    class Args:
        primary_cv = "nonlocal-contacts"; secondary_cv = "residual-torsion-pc"; contact_k_kcal = None; secondary_cv_k_kcal = 1.0
    centers, ks, sec_c, sec_k, meta, _ = load_explicit_2d_window_csv(Args(), path)
    assert ks[0] == 0.0 and sec_k[0] == 0.0 and np.isfinite(centers[0]) and np.isfinite(sec_c[0])
    assert len(centers) == len(rows) * 2


def test_separated_support_is_retained_in_both_orientations_and_population_orders():
    c = separated_support_case()
    for values in (c.values, 1.0 - c.values, np.r_[c.values[10000:], c.values[:10000]]):
        inv = build_region_inventory(values, temperature_k=300.0, k_max_kcal=1200.0)
        assert len(inv.cv1_regions) == 2, [(r.lo, r.hi) for r in inv.regions]
        rare = min(inv.cv1_regions, key=lambda r: r.n_members)
        assert rare.n_members == 200 and rare.eligible_seed_rows      # kept, with seeds
        design = region_centres(inv, 6, temperature_k=300.0, k_max_kcal=1200.0)
        owners = set(design["region_of_centre"])
        assert owners == {r.region_id for r in inv.cv1_regions}          # every region has a centre
        assert any(abs(cc - rare.representative) < 1e-12 for cc in design["centres"])
    # swap populations: the dense cluster becomes the rare one and is still retained
    swapped = np.r_[np.linspace(0.1, 0.3, 200), np.linspace(0.8, 0.82, 10000)]
    inv = build_region_inventory(swapped, temperature_k=300.0, k_max_kcal=1200.0)
    assert sorted(r.n_members for r in inv.cv1_regions) == [200, 10000]


def test_invalid_rows_and_duplicates_are_counted_not_binned_or_multiplied():
    x = np.r_[np.linspace(0.1, 0.3, 50), [np.nan, np.inf], np.full(500, 0.85)]
    inv = build_region_inventory(x, temperature_k=300.0, k_max_kcal=1200.0, groups=np.r_[np.arange(50), [0, 0], np.zeros(500, int)],
                                 n_failed_seed_preparations=3)
    assert inv.n_invalid_rows == 2 and inv.n_failed_seed_preparations == 3
    upper = [r for r in inv.cv1_regions if r.lo > 0.5][0]
    assert upper.n_members == 500 and upper.n_distinct_frames == 1 and upper.n_families == 1
    assert cv1_support_intervals(np.array([]), 0.1) == []


def test_a_region_without_an_eligible_seed_is_reported_unresolved_not_dropped():
    x = np.r_[np.linspace(0.1, 0.3, 100), np.linspace(0.8, 0.82, 20)]
    inv = build_region_inventory(x, temperature_k=300.0, k_max_kcal=1200.0, seed_values=x[:100])
    assert len(inv.cv1_regions) == 2
    assert any("cv1_region_01" in u for u in inv.unresolved)
    rec = inv.as_record()
    assert rec["role"] == "discovery_partition" and rec["unresolved"]
