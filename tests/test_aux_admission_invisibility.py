from gareus.adaptive_production import (WindowStateRegistry, AdaptiveDecisionPolicy, AUX_METADATA_KEY,
                                        ordinary_active_states, build_geometry_edges,
                                        propose_actions_from_diagnostics, _seed_source_restriction)
from gareus.adaptive import cv2_resolution, ladder_adapt, edge_metric

SHA = "b" * 64


def _registry_with_worker():
    r = WindowStateRegistry()
    for c1 in (0.0, 0.5, 1.0):
        for lam in (0.0, 0.2):
            r.add_state(c1, 10.0, 0.5, 2.0, gamd_lambda=lam)
    meta = {AUX_METADATA_KEY: {"role": "auxiliary", "aux_center": 1.0, "aux_k_kcal_mol": 2.0,
                               "aux_model_sha256": SHA, "state_instance_id": None,
                               "spawn_parent_state_id": 0, "admitted_epoch": 1}}
    w = r.add_state(0.0, 10.0, 0.5, 2.0, gamd_lambda=0.0, parent_state_id=0, epoch=1,
                    source="adaptive_production_aux", reason="aux", metadata=meta)
    return r, w


def test_ordinary_active_states_excludes_worker():
    r, w = _registry_with_worker()
    assert w.state_id not in {s.state_id for s in ordinary_active_states(r)}
    assert w.state_id in {s.state_id for s in r.active_states()}


def test_geometry_edges_never_touch_worker():
    r, w = _registry_with_worker()
    edges = build_geometry_edges(r, AdaptiveDecisionPolicy())
    ids = {int(e[0]) for e in edges} | {int(e[1]) for e in edges}
    assert ids and w.state_id not in ids


def test_retirement_never_proposes_worker():
    r, w = _registry_with_worker()
    diag = {"states": [{"state_id": s.state_id, "sample_count": 10 ** 6, "warnings": []}
                       for s in r.active_states()], "edges": []}
    acts = propose_actions_from_diagnostics(r, diag, policy=AdaptiveDecisionPolicy(retire_converged=True))
    assert all(not (len(a) > 1 and a[0] in ("retire", "extend") and int(a[1]) == w.state_id) for a in acts)


def test_cv2_resolution_views_skip_worker():
    r, w = _registry_with_worker()
    views = cv2_resolution.state_views(r, {"states": []})
    assert w.state_id not in {int(k) for k in views}


def test_ladder_centre_samples_skip_worker():
    r, w = _registry_with_worker()
    states = {int(s.state_id): s.to_dict() for s in r.all_states()}
    kept = ladder_adapt.registry_states_for_ladder(states)
    assert set(kept) == {k for k in states if k != w.state_id}


def test_rung_lambdas_and_centre_members_ignore_worker():
    r, w = _registry_with_worker()
    # a worker on its own rung must not create a ladder rung
    meta = {AUX_METADATA_KEY: {"role": "auxiliary", "aux_center": 1.0, "aux_k_kcal_mol": 2.0,
                               "aux_model_sha256": SHA, "state_instance_id": None,
                               "spawn_parent_state_id": 0, "admitted_epoch": 1}}
    r.add_state(0.5, 10.0, 0.5, 2.0, gamd_lambda=0.9, metadata=meta)
    assert r.rung_lambdas() == [0.0, 0.2]


def test_seed_source_restriction_for_worker_and_resolution():
    r, w = _registry_with_worker()
    assert _seed_source_restriction(w) == 0
    assert _seed_source_restriction(r.get_state(1)) is None


def test_edge_metric_nodes_skip_auxiliary_rows():
    rec = {"primary_center": 0.0, "primary_k": 10.0, "secondary_center": 0.5, "secondary_k": 2.0,
           "gamd_lambda": 0.0, "cv1_restrained": True, "cv2_restrained": True}
    row = lambda sid, **kw: {"state_id": sid, "epoch_window": sid, **kw, "paired_cv": {
        "restraint": rec, "cv1": {"mean": 0.0, "var": 1.0}, "cv2": {"mean": 0.0, "var": 1.0}, "n_pairs": 5}}
    nodes = edge_metric._state_nodes({"states": [row(1), row(2, auxiliary=True)]})
    assert set(nodes) == {1}
