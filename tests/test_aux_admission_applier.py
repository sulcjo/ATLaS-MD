from gareus.adaptive_production import (WindowStateRegistry, AdaptiveDecisionPolicy, AdaptiveProductionController,
                                        is_auxiliary_state, aux_params, _assert_hamiltonians_unchanged)
from gareus.adaptive.reserve_budget import reserve_allowances

SHA = "c" * 64
RESERVE = {"fraction": 0.10, "max_replicas": 236}


def _reg():
    r = WindowStateRegistry()
    r.add_state(0.0, 10.0, 0.5, 2.0, gamd_lambda=0.0)
    r.add_state(0.0, 10.0, 0.5, 2.0, gamd_lambda=0.2)
    return r


def _ctl(r, **kw):
    return AdaptiveProductionController(r, policy=AdaptiveDecisionPolicy(aux_discovery=True, **kw))


def _act(parent=0, c3=1.2, k3=2.0, sha=SHA, prov=None):
    return ("admit_aux", parent, {"aux_center": c3, "aux_k_kcal_mol": k3, "aux_model_sha256": sha,
                                  "burnin_steps": 3000}, "aux_discovery epoch 1", {"aux": prov or {"epoch": 1}})


def _workers(r):
    return [s for s in r.active_states() if is_auxiliary_state(s)]


def test_admit_adds_one_lambda0_worker_with_parent_restraints():
    r = _reg(); c = _ctl(r)
    c.apply_actions(1, [_act()])
    w = _workers(r)
    assert len(w) == 1 and not c.refused_actions
    w = w[0]; p = r.get_state(0)
    assert (w.primary_center, w.primary_k, w.secondary_center, w.secondary_k, w.gamd_lambda) == \
           (p.primary_center, p.primary_k, p.secondary_center, p.secondary_k, 0.0)
    assert aux_params(w)["aux_center"] == 1.2 and w.burnin_steps == 3000 and w.parent_state_id == 0
    assert set(aux_params(w)) == {"role", "aux_center", "aux_k_kcal_mol", "aux_model_sha256",
                                  "state_instance_id", "spawn_parent_state_id", "admitted_epoch", "epoch"}


def test_provenance_cannot_override_core_keys():
    r = _reg(); c = _ctl(r)
    c.apply_actions(1, [_act(prov={"aux_center": 99.0, "role": "x", "note": "n"})])
    a = aux_params(_workers(r)[0])
    assert a["aux_center"] == 1.2 and a["role"] == "auxiliary" and a["note"] == "n"


def test_refuses_parent_on_boosted_rung():
    r = _reg(); c = _ctl(r)
    c.apply_actions(1, [_act(parent=1)])
    assert c.refused_actions[0]["reason"] == "aux_parent_not_lambda0" and not _workers(r)


def test_refuses_aux_parent_and_unknown_parent():
    r = _reg(); c = _ctl(r, aux_reserve_slots=4)
    c.apply_actions(1, [_act()])
    wid = _workers(r)[0].state_id
    c.apply_actions(2, [_act(parent=wid, c3=3.0), _act(parent=77, c3=4.0)])
    assert [x["reason"] for x in c.refused_actions] == ["aux_parent_is_aux", "unknown_state"]


def test_refuses_second_model_sha():
    r = _reg(); c = _ctl(r)
    c.apply_actions(1, [_act(), _act(c3=-1.0, sha="d" * 64)])
    assert [x["reason"] for x in c.refused_actions] == ["aux_model_mismatch"]
    assert len(_workers(r)) == 1


def test_refuses_duplicate_worker():
    r = _reg(); c = _ctl(r)
    c.apply_actions(1, [_act(), _act()])
    assert [x["reason"] for x in c.refused_actions] == ["aux_duplicate"]


def test_refuses_over_aux_slice():
    r = _reg(); c = _ctl(r, aux_reserve_slots=1)
    c.apply_actions(1, [_act(c3=1.0), _act(c3=-1.0)])
    assert [x["reason"] for x in c.refused_actions] == ["aux_budget"]
    assert len(_workers(r)) == 1


def test_global_replica_cap_still_applies_after_aux_slice():
    r = _reg(); c = _ctl(r, max_replicas_budget=2)
    c.apply_actions(1, [_act()])
    assert c.refused_actions[0]["reason"] == "max_replicas_budget" and not _workers(r)


def test_validate_refusal_mutates_nothing():
    r = _reg(); c = _ctl(r)
    n = len(r.all_states())
    plan, refusal = c._validate_action(_act(parent=1))
    assert plan is None and refusal[0] == "aux_parent_not_lambda0" and len(r.all_states()) == n


def test_hamiltonians_of_existing_states_unchanged():
    r = _reg(); before = {s.state_id: s.to_dict() for s in r.all_states()}
    _ctl(r).apply_actions(1, [_act()])
    for sid, d in before.items():
        assert r.get_state(sid).to_dict() == d


def test_reserve_aux_slice_taken_first():
    a = reserve_allowances(236, 212, RESERVE, n_rungs=3, aux_slots=4)
    assert a.aux_slots == 4 and a.free_slots == 20
    b = reserve_allowances(236, 212, RESERVE, n_rungs=3)
    assert b.aux_slots == 0 and b.free_slots == 24
    c = reserve_allowances(236, 234, RESERVE, n_rungs=3, aux_slots=4)
    assert c.aux_slots == 2 and c.free_slots == 0
