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


def _diag(states, edges, non_neighbor_redundancies=None):
    diag = {"states": states, "edges": edges}
    if non_neighbor_redundancies is not None:
        diag["non_neighbor_redundancies"] = non_neighbor_redundancies
    return diag


def test_non_redundant_state_is_not_retired():
    reg = _registry(3)
    ids = reg.active_state_ids()
    # middle state well-sampled but only modest overlap (0.30 < 0.45) -> keep
    states = [{"state_id": s, "sample_count": 500, "gamd_boost_sd_kcal_mol": 1.0} for s in ids]
    edges = [
        {"state_i": ids[0], "state_j": ids[1], "overlap": 0.30, "exchange_acceptance": 0.5},
        {"state_i": ids[1], "state_j": ids[2], "overlap": 0.30, "exchange_acceptance": 0.5},
    ]
    pol = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=0)
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
    pol = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=0)
    actions = propose_actions_from_diagnostics(reg, _diag(states, edges), pol)
    assert any(a[0] == "retire" and a[1] == ids[2] for a in actions)
    # The non-redundant endpoint (ids[0], overlap 0.30) must NOT be retired
    assert not any(a[0] == "retire" and a[1] == ids[0] for a in actions)


def test_hidden_non_neighbor_redundancy_retires_only_the_off_target_interloper():
    # Same 3-state chain (0-1-2) as test_non_redundant_state_is_not_retired:
    # both geometry edges sit exactly at target_overlap (0.30), neither weak
    # nor redundant on their own, so nothing would normally be retired.
    #
    # Endpoints 0 and 2 are NOT a geometry edge in a 3-node chain, so a
    # window collapse between them (achieved-sample overlap 0.70, e.g. window
    # 2 drifted back onto window 0's basin instead of reaching its own
    # target) is invisible to `edges` -- it only shows up via
    # `non_neighbor_redundancies`.
    #
    # _hist_overlap is symmetric, so the alert alone can't tell which of 0/2
    # is the healthy anchor and which is the interloper that failed to reach
    # its target. ids[2] carries the "off_target_primary" warning (as
    # collect_epoch_diagnostics would set it) -- only THAT state may be
    # retired via this signal; the on-target anchor (ids[0]) must survive.
    reg = _registry(3)
    ids = reg.active_state_ids()
    states = [{"state_id": s, "sample_count": 500, "gamd_boost_sd_kcal_mol": 1.0, "warnings": []} for s in ids]
    states[2]["warnings"] = ["off_target_primary"]
    edges = [
        {"state_i": ids[0], "state_j": ids[1], "overlap": 0.30, "exchange_acceptance": 0.5},
        {"state_i": ids[1], "state_j": ids[2], "overlap": 0.30, "exchange_acceptance": 0.5},
    ]
    non_neighbor_redundancies = [
        {"state_i": ids[0], "state_j": ids[2], "overlap": 0.70},
    ]
    pol = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=0)
    actions = propose_actions_from_diagnostics(reg, _diag(states, edges, non_neighbor_redundancies), pol)
    assert any(a[0] == "retire" and a[1] == ids[2] for a in actions)
    assert not any(a[0] == "retire" and a[1] == ids[0] for a in actions)


def test_ambiguous_non_neighbor_redundancy_retires_neither_side():
    # Identical setup, but NEITHER side carries an off-target warning (e.g.
    # both windows genuinely reached their own targets and just happen to
    # look similar, or the flag simply isn't available). Without a signal
    # to tell anchor from interloper, the non-neighbor overlap must not be
    # used to single out either state for retirement -- guessing risks
    # dropping the wrong (correct) window and corrupting downstream MBAR
    # coverage.
    reg = _registry(3)
    ids = reg.active_state_ids()
    states = [{"state_id": s, "sample_count": 500, "gamd_boost_sd_kcal_mol": 1.0, "warnings": []} for s in ids]
    edges = [
        {"state_i": ids[0], "state_j": ids[1], "overlap": 0.30, "exchange_acceptance": 0.5},
        {"state_i": ids[1], "state_j": ids[2], "overlap": 0.30, "exchange_acceptance": 0.5},
    ]
    non_neighbor_redundancies = [
        {"state_i": ids[0], "state_j": ids[2], "overlap": 0.70},
    ]
    pol = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=0)
    actions = propose_actions_from_diagnostics(reg, _diag(states, edges, non_neighbor_redundancies), pol)
    assert not any(a[0] == "retire" for a in actions)
