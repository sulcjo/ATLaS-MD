"""Atomic ``respace_ladder`` registry action: whole rungs, endpoints fixed, budgeted, new state ids."""
from gareus.adaptive_production import (AdaptiveDecisionPolicy, AdaptiveProductionController,
                                        WindowStateRegistry)


def _registry(centers=(0.05, 0.15, 0.25), rungs=(0.0, 0.5, 1.0)):
    reg = WindowStateRegistry()
    for c in centers:
        for lam in rungs:
            reg.add_state(primary_center=c, primary_k=800.0, gamd_lambda=lam, epoch=0, source="initial", reason="seed")
    return reg


def _rungs_by_centre(reg):
    out = {}
    for s in reg.active_states():
        out.setdefault(round(s.primary_center, 6), []).append(round(float(s.gamd_lambda or 0.0), 6))
    return {k: sorted(v) for k, v in out.items()}


def test_respace_adds_then_retires_whole_rungs():
    reg = _registry()
    before = {s.state_id for s in reg.active_states()}
    AdaptiveProductionController(reg).apply_actions(3, [("respace_ladder", (0.5,), (0.3, 0.7), "t")])
    assert reg.rung_lambdas() == [0.0, 0.3, 0.7, 1.0]
    assert all(v == [0.0, 0.3, 0.7, 1.0] for v in _rungs_by_centre(reg).values())
    retired = [reg.get_state(i) for i in before if not reg.get_state(i).active]
    assert len(retired) == 3 and all(abs(s.gamd_lambda - 0.5) < 1e-9 and s.retired_epoch == 4 for s in retired)


def test_endpoints_are_never_dropped(capsys):
    for drop in ((0.0,), (1.0,)):
        reg = _registry()
        AdaptiveProductionController(reg).apply_actions(0, [("respace_ladder", drop, (), "t")])
        assert reg.rung_lambdas() == [0.0, 0.5, 1.0]
    assert "refusing" in capsys.readouterr().out


def test_respace_refused_over_cap(capsys):
    reg = _registry()
    policy = AdaptiveDecisionPolicy(max_replicas_budget=9)
    ctl = AdaptiveProductionController(reg, policy=policy)
    ctl.apply_actions(0, [("respace_ladder", (), (0.3,), "t")])          # +3 states over a 9-state cap
    assert reg.rung_lambdas() == [0.0, 0.5, 1.0]
    assert "cap" in capsys.readouterr().out
    ctl.apply_actions(0, [("respace_ladder", (0.5,), (0.3,), "t")])      # net zero: applies
    assert reg.rung_lambdas() == [0.0, 0.3, 1.0]


def test_new_rung_states_are_new_ids_and_existing_states_keep_their_lambda():
    reg = _registry()
    old = {s.state_id: float(s.gamd_lambda) for s in reg.active_states()}
    AdaptiveProductionController(reg).apply_actions(0, [("respace_ladder", (0.5,), (0.3,), "t")])
    for sid, lam in old.items():
        assert float(reg.get_state(sid).gamd_lambda) == lam
    new = [s for s in reg.active_states() if s.state_id not in old]
    assert len(new) == 3 and all(abs(s.gamd_lambda - 0.3) < 1e-9 for s in new)
    reps = {s.state_id for s in reg.active_states() if abs(float(s.gamd_lambda or 0.0)) < 1e-9}
    assert all(s.parent_state_id in reps for s in new)


def test_duplicate_or_existing_rung_add_is_a_no_op():
    reg = _registry()
    AdaptiveProductionController(reg).apply_actions(0, [("respace_ladder", (), (0.5,), "t")])
    assert len(reg.active_states()) == 9
