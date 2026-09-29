"""P7b and the bugs that ride with it (spec 2026-09-29 adaptive-CV2, Sections 3.0 P7b, 3.1, 13).

* A: ``build_geometry_edges`` chains only true neighbours (spec-T3 c9 artefacts, T2 wrap-around).
* B: the 3.1 appended graph edges never block ``retire_converged``.
* C: a bridge's reason quotes the metric that made its edge weak.
* D: ``layout_plan.json`` v2 and the consumers switched to the P7a rule behind a gate.
"""
from __future__ import annotations

from collections import Counter

from gareus.adaptive_production import (AdaptiveDecisionPolicy, WindowStateRegistry,
                                        build_geometry_edges, propose_actions_from_diagnostics)


def _line_registry(n=8, spacing=0.04, k=200.0):
    reg = WindowStateRegistry()
    for i in range(n):
        reg.add_state(primary_center=0.3 + spacing * i, primary_k=k, epoch=0, source="seed")
    return reg


def _diag(reg, edges, sample_count=1000):
    return {"states": [{"state_id": s, "sample_count": sample_count, "gamd_boost_sd_kcal_mol": 0.0}
                       for s in reg.active_state_ids()],
            "edges": edges}


# ---- B: graded-only graph edges do not block retirement --------------------------------------

def test_appended_graph_edges_with_no_marginal_overlap_do_not_block_retirement():
    reg = _line_registry()
    pol = AdaptiveDecisionPolicy(min_active_states=2, edge_metric="pairwise-mbar")
    geom = [{"state_i": a, "state_j": b, "edge_type": t, "overlap": 0.9, "exchange_acceptance": 0.5}
            for a, b, t, _ in build_geometry_edges(reg)]
    base = Counter(a[0] for a in propose_actions_from_diagnostics(reg, _diag(reg, geom), pol))
    assert base["retire"] > 0
    appended = [{"state_i": i, "state_j": i + 2, "edge_type": kind, "overlap": None,
                 "exchange_acceptance": None} for i, kind in ((0, "neighbour"), (3, "spanning"))]
    got = Counter(a[0] for a in propose_actions_from_diagnostics(reg, _diag(reg, geom + appended), pol))
    assert got["retire"] == base["retire"]


def test_an_unmeasured_geometry_edge_still_blocks_retirement_of_its_ends():
    # Deliberate: only the graded-only 3.1 edges are skipped. A geometry edge with no marginal
    # overlap (a state without samples) keeps protecting its endpoints, as before P7b.
    reg = _line_registry()
    pol = AdaptiveDecisionPolicy(min_active_states=2)
    geom = [{"state_i": a, "state_j": b, "edge_type": t, "overlap": (None if (a, b) == (3, 4) else 0.9),
             "exchange_acceptance": 0.5} for a, b, t, _ in build_geometry_edges(reg)]
    retired = {a[1] for a in propose_actions_from_diagnostics(reg, _diag(reg, geom), pol) if a[0] == "retire"}
    assert retired and not retired & {3, 4}
