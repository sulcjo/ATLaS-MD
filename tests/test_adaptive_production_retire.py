"""Real-gareus regression tests: adaptive-production DOES remove windows.

These exercise the shipped decision directly (no OpenMM): build a registry +
a diagnostics dict and assert propose_actions_from_diagnostics retires redundant,
well-sampled, non-critical windows down to the min_active_states floor — and that
the gate keys off the RAW sample_count (an ESS-deflated count below the gate must
NOT silently block retirement).
"""
from __future__ import annotations

from collections import Counter

from gareus.adaptive_production import (AdaptiveDecisionPolicy, WindowStateRegistry,
                                        build_geometry_edges,
                                        propose_actions_from_diagnostics)


def _diag(reg, *, overlap=0.70, exch=0.5, sample_count=1000):
    edges = build_geometry_edges(reg)
    return {
        "schema_version": "adaptive_production_epoch_diagnostics_v1",
        "states": [{"state_id": sid, "sample_count": sample_count,
                    "gamd_boost_sd_kcal_mol": 0.0} for sid in reg.active_state_ids()],
        "edges": [{"state_i": a, "state_j": b, "edge_type": t,
                   "overlap": overlap, "exchange_acceptance": exch} for a, b, t, _ in edges],
    }


def _grid_registry(n1, n2):
    reg = WindowStateRegistry()
    for i in range(n1):
        for j in range(n2):
            reg.add_state(primary_center=0.1 + 0.22 * i, primary_k=80.0,
                          secondary_center=-0.6 + 0.6 * j, secondary_k=50.0,
                          epoch=0, source="seed")
    return reg


def test_retire_removes_redundant_2d_windows_to_floor():
    reg = _grid_registry(4, 3)                       # 12 redundant 2D windows
    pol = AdaptiveDecisionPolicy()                   # min_active_states=8
    acts = propose_actions_from_diagnostics(reg, _diag(reg), pol)
    n_retire = sum(1 for a in acts if a[0] == "retire")
    assert n_retire == 12 - pol.min_active_states    # 12 -> 8 (interior removable in 2D)


def test_retire_respects_min_active_floor():
    reg = _grid_registry(3, 3)                       # 9 windows, floor 8
    acts = propose_actions_from_diagnostics(reg, _diag(reg), AdaptiveDecisionPolicy())
    assert sum(1 for a in acts if a[0] == "retire") <= 1   # cannot drop below 8


def test_retire_gate_is_raw_sample_count_not_ess():
    # With raw sample_count below the retire gate, retirement must NOT fire — and
    # raising it above the gate (what a raw count does, but an ESS-deflated count
    # would not) must enable it. This locks the gate to raw counts.
    reg = _grid_registry(4, 3)
    pol = AdaptiveDecisionPolicy()                   # min_samples_for_retire=200
    below = propose_actions_from_diagnostics(reg, _diag(reg, sample_count=150), pol)
    above = propose_actions_from_diagnostics(reg, _diag(reg, sample_count=1000), pol)
    assert sum(1 for a in below if a[0] == "retire") == 0
    assert sum(1 for a in above if a[0] == "retire") >= 1


def test_retire_blocked_when_not_redundant():
    # overlaps below redundant_overlap (0.45): nothing is redundant -> no retire.
    reg = _grid_registry(4, 3)
    acts = propose_actions_from_diagnostics(reg, _diag(reg, overlap=0.35), AdaptiveDecisionPolicy())
    assert sum(1 for a in acts if a[0] == "retire") == 0
