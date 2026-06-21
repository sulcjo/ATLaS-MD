from gareus.adaptive_production import (
    WindowStateRegistry, AdaptiveDecisionPolicy, propose_actions_from_diagnostics,
)


def _registry(n):
    reg = WindowStateRegistry()
    for i in range(n):
        reg.add_state(primary_center=float(i), primary_k=10.0,
                      secondary_center=None, secondary_k=None,
                      epoch=0, source="seed")
    return reg


def _diag(states, edges):
    return {"states": states, "edges": edges}


def test_non_redundant_state_is_not_retired():
    reg = _registry(3)
    ids = reg.active_state_ids()
    # middle state well-sampled but only modest overlap (0.30 < 0.45) -> keep
    states = [{"state_id": s, "sample_count": 500, "gamd_boost_sd_kcal_mol": 1.0} for s in ids]
    edges = [
        {"state_i": ids[0], "state_j": ids[1], "overlap": 0.30, "exchange_acceptance": 0.5},
        {"state_i": ids[1], "state_j": ids[2], "overlap": 0.30, "exchange_acceptance": 0.5},
    ]
    pol = AdaptiveDecisionPolicy(retire_converged=True)
    actions = propose_actions_from_diagnostics(reg, _diag(states, edges), pol)
    assert not any(a[0] == "retire" for a in actions)


def test_redundant_non_articulation_endpoint_is_retired():
    # With 3 states (centers 0, 1, 2) the geometry chain is 0-1-2.
    # The endpoint at center=2 (ids[2]) is NOT an articulation point:
    # removing it leaves 0-1 connected.
    # ids[1]-ids[2] has high overlap (0.60 >= 0.45) so ids[2] qualifies as redundant.
    reg = _registry(3)
    ids = reg.active_state_ids()
    states = [{"state_id": s, "sample_count": 500, "gamd_boost_sd_kcal_mol": 1.0} for s in ids]
    edges = [
        {"state_i": ids[0], "state_j": ids[1], "overlap": 0.30, "exchange_acceptance": 0.4},
        {"state_i": ids[1], "state_j": ids[2], "overlap": 0.60, "exchange_acceptance": 0.6},
    ]
    pol = AdaptiveDecisionPolicy(retire_converged=True)
    actions = propose_actions_from_diagnostics(reg, _diag(states, edges), pol)
    assert any(a[0] == "retire" and a[1] == ids[2] for a in actions)
    # The non-redundant endpoint (ids[0], overlap 0.30) must NOT be retired
    assert not any(a[0] == "retire" and a[1] == ids[0] for a in actions)
