"""Task 7: combined window-economy safety invariants.

Verifies that an economy pass (retire_converged + redundancy gate from Tasks
3-4) never violates the hard safety invariants: it must keep the active window
graph connected and must not retire articulation/bridge windows, while still
reclaiming at least one genuinely-redundant replica.
"""
from gareus.adaptive_production import (
    WindowStateRegistry,
    AdaptiveDecisionPolicy,
    propose_actions_from_diagnostics,
    _apply_registry_actions,
    active_graph_connected,
    build_geometry_edges,
)


def _registry(n):
    reg = WindowStateRegistry()
    for i in range(n):
        reg.add_state(primary_center=float(i), primary_k=10.0,
                      secondary_center=None, secondary_k=None,
                      epoch=0, source="seed")
    return reg


def test_retirement_keeps_graph_connected_and_reclaims():
    # Geometry chain 0-1-2-3-4 (build_geometry_edges = sorted nearest-neighbour).
    # Endpoint state 4 is redundant: edge 3-4 overlap 0.60 (>= reclaim 0.45) and
    # all edges are healthy (>= target 0.30, acc >= 0.08) so nothing is bad-touching.
    # Interior states 1,2,3 are articulation points and must survive.
    reg = _registry(5)
    ids = reg.active_state_ids()
    states = [{"state_id": s, "sample_count": 500, "gamd_boost_sd_kcal_mol": 1.0} for s in ids]
    edges = [
        {"state_i": ids[0], "state_j": ids[1], "overlap": 0.30, "exchange_acceptance": 0.5},
        {"state_i": ids[1], "state_j": ids[2], "overlap": 0.40, "exchange_acceptance": 0.5},
        {"state_i": ids[2], "state_j": ids[3], "overlap": 0.40, "exchange_acceptance": 0.5},
        {"state_i": ids[3], "state_j": ids[4], "overlap": 0.60, "exchange_acceptance": 0.6},
    ]
    pol = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=0)
    actions = propose_actions_from_diagnostics(reg, {"states": states, "edges": edges}, pol)

    retires = [a for a in actions if a[0] == "retire"]
    assert retires, "economy should retire the redundant non-articulation endpoint"
    # only the redundant endpoint, never an interior articulation window
    retired_ids = {a[1] for a in retires}
    assert ids[4] in retired_ids
    assert ids[1] not in retired_ids and ids[2] not in retired_ids and ids[3] not in retired_ids

    _apply_registry_actions(reg, retires, epoch=1)

    assert active_graph_connected(reg)               # invariant: graph stays connected
    assert len(reg.active_state_ids()) < len(ids)    # economy reclaimed >= 1 replica
    assert ids[1] in reg.active_state_ids()          # articulation interiors survive
    assert ids[2] in reg.active_state_ids()


def test_non_redundant_chain_is_not_thinned():
    # All overlaps below the 0.45 reclaim trigger -> nothing redundant -> no retires.
    reg = _registry(4)
    ids = reg.active_state_ids()
    states = [{"state_id": s, "sample_count": 500, "gamd_boost_sd_kcal_mol": 1.0} for s in ids]
    edges = [
        {"state_i": ids[0], "state_j": ids[1], "overlap": 0.32, "exchange_acceptance": 0.5},
        {"state_i": ids[1], "state_j": ids[2], "overlap": 0.35, "exchange_acceptance": 0.5},
        {"state_i": ids[2], "state_j": ids[3], "overlap": 0.33, "exchange_acceptance": 0.5},
    ]
    pol = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=0)
    actions = propose_actions_from_diagnostics(reg, {"states": states, "edges": edges}, pol)
    assert not any(a[0] == "retire" for a in actions)
    assert active_graph_connected(reg)
    assert len(reg.active_state_ids()) == len(ids)


def test_2d_multi_retire_never_disconnects():
    # 2D square: every window redundant + healthy. In a cyclic geometry graph,
    # co-retiring opposite corners could disconnect it. The greedy connectivity-safe
    # gate must refuse enough drops to keep the active graph connected and non-empty,
    # even though each node is individually a non-articulation, redundant candidate.
    reg = WindowStateRegistry()
    for (p, s) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]:
        reg.add_state(primary_center=p, primary_k=10.0,
                      secondary_center=s, secondary_k=10.0, epoch=0, source="seed")
    ids = reg.active_state_ids()
    states = [{"state_id": s, "sample_count": 500, "gamd_boost_sd_kcal_mol": 1.0} for s in ids]
    # mark every actual geometry edge as highly redundant + healthy
    edges = [{"state_i": a, "state_j": b, "overlap": 0.60, "exchange_acceptance": 0.6}
             for (a, b, _t, _n) in build_geometry_edges(reg)]
    pol = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=0)
    actions = propose_actions_from_diagnostics(reg, {"states": states, "edges": edges}, pol)

    retires = [a for a in actions if a[0] == "retire"]
    _apply_registry_actions(reg, retires, epoch=1)

    assert active_graph_connected(reg)            # INVARIANT: never disconnect
    assert len(reg.active_state_ids()) >= 1       # never empties the graph


def _redundant_chain(n):
    # 1D chain of n windows where every neighbor edge is over-overlapped (>=0.45)
    reg = _registry(n)
    ids = reg.active_state_ids()
    states = [{"state_id": s, "sample_count": 500, "gamd_boost_sd_kcal_mol": 1.0} for s in ids]
    edges = [{"state_i": ids[i], "state_j": ids[i + 1], "overlap": 0.60, "exchange_acceptance": 0.6}
             for i in range(n - 1)]
    return reg, ids, {"states": states, "edges": edges}


def test_min_active_states_floor_blocks_retirement_at_floor():
    # floor == current count -> nothing may retire even though all windows are redundant
    reg, ids, diag = _redundant_chain(5)
    pol = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=5)
    actions = propose_actions_from_diagnostics(reg, diag, pol)
    assert not any(a[0] == "retire" for a in actions)
    _apply_registry_actions(reg, [a for a in actions if a[0] == "retire"], epoch=1)
    assert len(reg.active_state_ids()) == 5


def test_min_active_states_floor_caps_retirement():
    # floor = count-1 -> at most one redundant window may be retired
    reg, ids, diag = _redundant_chain(5)
    pol = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=4)
    actions = propose_actions_from_diagnostics(reg, diag, pol)
    _apply_registry_actions(reg, [a for a in actions if a[0] == "retire"], epoch=1)
    assert len(reg.active_state_ids()) >= 4
    assert active_graph_connected(reg)
