"""Spec 3.3 CV2 resolution actions R1-R3 (``--ap-cv2-resolution``): decisions, springs, budget.

Synthetic registries/payloads only; the driver wiring and the off-path identity are in
``test_cv2_resolution_wiring.py``.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gareus.adaptive_production as ap  # noqa: E402
from gareus.adaptive import cv2_coverage as cov  # noqa: E402
from gareus.adaptive import cv2_resolution as cr  # noqa: E402
from gareus.adaptive import cv2_resolution_io as cio  # noqa: E402
from gareus.swarm.ladder_design import R_KCAL_MOL_K  # noqa: E402

T = 300.0
RT = R_KCAL_MOL_K * T
K1, K2 = 800.0, 5.0
SETTINGS = cr.ResolutionSettings(temperature_k=T)


def _registry(lambdas=(0.0,), c1s=(0.2, 0.4), c2s=(-1.0, 0.0, 1.0), k2=K2):
    reg = ap.WindowStateRegistry()
    for c1 in c1s:
        for c2 in c2s:
            for lam in lambdas:
                reg.add_state(c1, K1, secondary_center=c2, secondary_k=k2, gamd_lambda=lam, epoch=0, source="seed")
    return reg


def _sid(reg, c1, c2, lam=0.0):
    return next(s.state_id for s in reg.active_states() if abs(s.primary_center - c1) < 1e-9
                and s.secondary_center is not None and abs(s.secondary_center - c2) < 1e-9
                and abs(s.gamd_lambda - lam) < 1e-9)


def _moments(z):
    z = np.asarray(z, float)
    m, v = float(z.mean()), float(z.var())
    return {"n": int(z.size), "mean": m, "var": v, "skewness": float(((z - m) ** 3).mean() / v ** 1.5),
            "kurtosis_excess": float(((z - m) ** 4).mean() / v ** 2 - 3.0)}


def _payload(reg, *, edges=(), components=None, samples=None, edge_metric=True):
    samples = samples or {}
    rows = []
    for s in reg.active_states():
        z = samples.get(s.state_id)
        m2 = _moments(z) if z is not None else {"n": 2000, "mean": s.secondary_center, "var": 0.1,
                                                 "skewness": 0.0, "kurtosis_excess": 0.0}
        rows.append({"state_id": s.state_id, "sample_count": 5000,
                     "paired_cv": {"n_pairs": 2000, "cv2": m2}})
    pay = {"states": rows, "edges": list(edges)}
    if edge_metric:
        ids = [s.state_id for s in reg.active_states() if abs(s.gamd_lambda) < 1e-9]
        pay["edge_metric"] = {"status": "ok", "metric": "pairwise-mbar", "stage": "pre_union",
                              "components": {"components": components or [ids]}}
    return pay


def _edge(i, j, *, status="ok", upper=0.40, etype="secondary_chain"):
    pm = {"status": status, "overlap": None if status != "ok" else upper - 0.02,
          "overlap_lower": None if status != "ok" else upper - 0.04,
          "overlap_upper": None if status != "ok" else upper, "pattern_pair": "same", "n_eff": [500, 500]}
    return {"state_i": i, "state_j": j, "edge_type": etype, "normalized_distance": 1.0, "overlap": 0.8,
            "mbar_overlap": None, "pairwise_mbar": pm, "warnings": []}


def _policy(cap=10, **kw):
    return ap.AdaptiveDecisionPolicy(cv2_resolution=True, edge_metric="pairwise-mbar", max_replicas_budget=cap, **kw)


RESERVE = {"fraction": 0.2, "max_replicas": 10}


def _propose(reg, payload, *, actions=(), epoch=0, history=None, subsamples=None, runs=None, reserve=RESERVE,
             policy=None, union=None, ignore_budget=False, settings=SETTINGS):
    runs_for = (lambda ids: {i: runs[i] for i in ids if i in runs}) if runs is not None else None
    return cio.propose_cv2_resolution(reg, payload, list(actions), settings, policy or _policy(), epoch=epoch,
                                      history=history or {}, subsamples=subsamples or {}, runs_for=runs_for,
                                      union=union, reserve=reserve, ignore_budget=ignore_budget)


def _decisions(report, rule):
    return [c for c in report["candidates"] if c["rule"] == rule]


# ---- springs -----------------------------------------------------------------------------

def test_f2_from_biased_samples_subtracts_the_windows_own_spring():
    f2_true, k2 = 3.0, K2
    var = RT / (k2 + f2_true)                         # harmonic bowl under the window's spring
    assert cr.f2_under_bias(var, k2, T) == pytest.approx(f2_true, rel=1e-9)
    assert cr.f2_under_bias(RT / 2.0, k2, T) == 0.0    # wider than the spring alone: floored at 0


def test_child_spring_shape_rule_cap_and_min_sigma():
    out = cr.child_spring(0.3, 1.0, K2, SETTINGS)
    assert out["k2"] == pytest.approx(RT / 0.09 - 1.0) and out["refusal"] is None
    capped = cr.child_spring(0.05, 0.0, K2, cr.ResolutionSettings(temperature_k=T, refine_min_sigma=0.01))
    assert capped["k2"] == pytest.approx(4.0 * K2)     # child k2 <= 4 x parent k2
    wide = cr.child_spring(0.01, 0.0, K2, SETTINGS)     # never below refine_min_sigma (0.1)
    assert wide["sigma_used"] == pytest.approx(0.1)


def test_k2_at_floor_is_a_refusal_not_a_state():
    s = cr.ResolutionSettings(temperature_k=T, k2_min=1.0)
    out = cr.child_spring(1.0, 0.0, K2, s)              # RT / sigma^2 = 0.60 < k_min: floored
    assert out["k2"] == pytest.approx(1.0) and out["refusal"] == "k2_at_floor"


def test_cap_below_the_mean_compression_floor_is_a_refusal():
    out = cr.child_spring(0.3, 30.0, K2, SETTINGS)       # floor k2 >= F'' = 30 > cap 4 x 5 = 20
    assert out["k2"] == pytest.approx(20.0) and out["refusal"] == "k2_capped_below_compression"
    assert out["mean_compression"] == pytest.approx(20.0 / 50.0)
    ok = cr.child_spring(1.0, 3.0, K2, SETTINGS)         # width rule negative: raised to the floor
    assert ok["refusal"] is None and ok["at_compression_floor"] and ok["mean_compression"] == pytest.approx(0.5)


class _Gate:
    def __init__(self, k2):
        self.k2 = k2

    def gate(self, c1, k1, c2, k2, context=""):
        return self.k2, "lowered"


def test_a_gate_that_lowers_k2_below_the_compression_floor_refuses():
    child = cr.child_spring(1.0, 3.0, K2, SETTINGS)
    out = cr.gate_child(child, 0.2, K1, 0.0, _Gate(1.0), "t", SETTINGS)
    assert out["refusal"] == "k2_capped_below_compression" and out["mean_compression"] == pytest.approx(0.25)


def test_k2_at_a_zero_floor_is_a_refusal(monkeypatch):
    # With cv2_k_min = 0 a shape rule that floors returns 0 (whichever floor rule cv2_shape uses).
    monkeypatch.setattr(cr, "shape_rule_k2", lambda sigma, f2, t, k_min, k_max, **kw: k_min)
    out = cr.child_spring(0.5, 50.0, K2, SETTINGS)
    assert out["k2"] == 0.0 and out["refusal"] == "k2_at_floor"


# ---- R3 ----------------------------------------------------------------------------------

def _bimodal(n=2000, sep=0.66, sd=0.33, seed=1):
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 2, n)
    return np.where(lab == 0, -sep, sep) + rng.normal(0.0, sd, n)


def _alternating_runs(n_trans=40, sep=0.66):
    return [np.array([(-sep if i % 2 == 0 else sep) for i in range(n_trans + 1)])]


def _r3_case(runs_kind, samples=None):
    reg = _registry()
    parent = _sid(reg, 0.2, 0.0)
    z = _bimodal() if samples is None else samples
    sub = {parent: {"cv2": z, "cv1": np.full(z.size, 0.2), "source_index": np.zeros(z.size, int)}}
    runs = {"with": {parent: {"source": "parquet", "replica_runs": _alternating_runs(), "state_runs": []}},
            "without": {parent: {"source": "parquet", "replica_runs": [np.full(60, -0.66), np.full(60, 0.66)],
                                 "state_runs": _alternating_runs()}}}[runs_kind]
    new, report, _h = _propose(reg, _payload(reg, samples={parent: z}), subsamples=sub, runs=runs)
    return reg, parent, new, report


def test_r3_bimodal_with_transitions_inserts_two_children_and_keeps_the_parent():
    reg, parent, new, report = _r3_case("with")
    (cand,) = [c for c in _decisions(report, "R3") if c["state_ids"] == [parent]]
    assert cand["decision"] == "proposed", cand["reason"]
    inserts = [a for a in new if a[0] == "insert"]
    assert len(inserts) == 1 and inserts[0][1] == parent
    kids = inserts[0][2]
    assert len(kids) == 2
    modes = sorted(c["mean"] for c in cand["metrics"]["modes"])
    assert [k[2] for k in kids] == pytest.approx(modes)            # centred on the modes, not the nominal 0.0
    assert all(k[3] <= 4 * K2 for k in kids)
    n0 = len(reg.active_states())
    refused = ap._apply_registry_actions(reg, new, 0, policy=_policy())
    assert refused == []
    assert reg.get_state(parent).active                             # parent kept
    assert len(reg.active_states()) == n0 + 2
    kids_states = [s for s in reg.active_states() if s.source == cr.SOURCE]
    assert all(s.parent_state_id == parent for s in kids_states)
    assert all(s.metadata[cr.METADATA_KEY]["seed_source_state_id"] == parent for s in kids_states)


def test_r3_bimodal_without_transitions_flags_and_inserts_nothing():
    _reg, parent, new, report = _r3_case("without")
    (cand,) = [c for c in _decisions(report, "R3") if c["state_ids"] == [parent]]
    assert cand["decision"] == "flagged" and cand["metrics"]["trapped_or_orthogonal"] is True
    assert cand["metrics"]["transitions"] == 0 and cand["metrics"]["transitions_state_series"] > 0
    assert not [a for a in new if a[0] == "insert"]
    assert report["summary"]["n_trapped_or_orthogonal"] == 1


@pytest.mark.parametrize("shape", ["plateau_between_walls", "stiff_bowl"])
def test_r3_plateau_and_stiff_bowl_produce_zero_actions(shape):
    rng = np.random.default_rng(3)
    z = rng.uniform(-0.6, 0.6, 2000) if shape == "plateau_between_walls" else rng.normal(0.0, 0.03, 2000)
    reg, parent, new, report = _r3_case("with", samples=z)
    assert new == []
    assert report["summary"]["n_proposed"] == 0
    (cand,) = [c for c in _decisions(report, "R3") if c["state_ids"] == [parent]]
    assert cand["decision"] == "no_action"
    assert cand["metrics"]["diagnostics_trigger"] is False and cand["metrics"]["sd_over_sigma_w"] is not None


def test_transition_counting_uses_cores_with_hysteresis():
    runs = [np.array([-1.0, -0.05, 0.05, -0.05, 0.05, 1.0, 0.9, -1.0])]
    assert cr.count_transitions(runs, -0.5, 0.5) == 2               # barrier jitter is not a transition


def test_r3_on_subsample_only_records_a_lower_bound_and_never_inserts():
    reg = _registry()
    parent = _sid(reg, 0.2, 0.0)
    z = _bimodal()
    sub = {parent: {"cv2": z, "cv1": np.full(z.size, 0.2), "source_index": np.zeros(z.size, int)}}
    runs = {parent: {"source": "no_replica_column", "replica_runs": [], "state_runs": [z]}}
    new, report, _h = _propose(reg, _payload(reg, samples={parent: z}), subsamples=sub, runs=runs)
    (cand,) = [c for c in _decisions(report, "R3") if c["state_ids"] == [parent]]
    assert cand["decision"] == "flagged" and cand["metrics"]["transitions_lower_bound"] > 0
    assert new == []


# ---- R1 ----------------------------------------------------------------------------------

def test_r1_structural_component_split_bridges_immediately_at_the_sampled_midpoint():
    reg = _registry()
    a, b = _sid(reg, 0.2, -1.0), _sid(reg, 0.2, 0.0)
    low = [_sid(reg, c1, -1.0) for c1 in (0.2, 0.4)]
    rest = [s.state_id for s in reg.active_states() if s.state_id not in low]
    edges = [_edge(a, b, upper=0.05)]
    new, report, hist = _propose(reg, _payload(reg, edges=edges, components=[low, rest]))
    (cand,) = _decisions(report, "R1")
    assert cand["class"] == "structural" and cand["decision"] == "proposed"
    (add,) = [x for x in new if x[0] == "add"]
    assert add[2][2] == pytest.approx(-0.5) and add[4][cr.METADATA_KEY]["rule"] == "R1"
    assert hist["edges"][cr.edge_key(a, b)]["structural"] == [0]


def test_r1_weak_bridges_and_replaces_the_midpoint_bridger_on_that_edge():
    reg = _registry()
    a, b = _sid(reg, 0.2, -1.0), _sid(reg, 0.2, 0.0)
    main = [("add", a, (0.2, K1, -0.5, K2), f"weak edge {a}-{b}: pairwise"), ("extend", a, "x")]
    new, report, _h = _propose(reg, _payload(reg, edges=[_edge(a, b, upper=0.10)]), actions=main)
    (cand,) = _decisions(report, "R1")
    assert cand["class"] == "weak" and cand["decision"] == "proposed"
    adds = [x for x in new if x[0] == "add"]
    assert len(adds) == 1 and str(adds[0][3]).startswith("cv2_resolution R1/weak")
    assert report["actions_removed"][0]["reason"].startswith("edge bridged by cv2_resolution")


def test_r1_refused_bridge_leaves_the_midpoint_bridger_in_place():
    reg = _registry()
    a, b = _sid(reg, 0.2, -1.0), _sid(reg, 0.2, 0.0)
    main = [("add", a, (0.2, K1, -0.5, K2), f"weak edge {a}-{b}: pairwise")]
    new, report, _h = _propose(reg, _payload(reg, edges=[_edge(a, b, upper=0.10)]), actions=main, reserve=None)
    assert _decisions(report, "R1")[0]["refusal"] == "no_reserve"
    assert new == main


def test_r1_ignores_edges_that_differ_mainly_in_cv1():
    reg = _registry()
    a, b = _sid(reg, 0.2, 0.0), _sid(reg, 0.4, 0.0)
    _new, report, _h = _propose(reg, _payload(reg, edges=[_edge(a, b, upper=0.05, etype="primary_chain")]))
    assert _decisions(report, "R1") == [] and report["rules"]["R1"]["skipped_edges"]["not_mainly_cv2"] == 1


def test_r1_unmeasured_waits_two_epochs_and_the_history_survives_a_resume(tmp_path):
    reg = _registry()
    a, b = _sid(reg, 0.2, -1.0), _sid(reg, 0.2, 0.0)
    pay = _payload(reg, edges=[_edge(a, b, status="unmeasured")])
    decisions = []
    for epoch in (0, 1, 1, 2):                      # epoch 1 re-proposed (killed before the ledger)
        hist = cio.load_history(tmp_path)           # a fresh job each time: history from disk
        new, report, hist = _propose(reg, pay, epoch=epoch, history=hist)
        cio.save_history(tmp_path, hist)
        decisions.append(_decisions(report, "R1")[0]["decision"])
        if epoch < 2:
            assert sorted(x[1] for x in new if x[0] == "extend") == sorted([a, b])
    assert decisions == ["extend", "extend", "extend", "proposed"]
    assert cio.load_history(tmp_path)["edges"][cr.edge_key(a, b)]["unmeasured"] == [0, 1, 2]


def test_r1_unavailable_without_the_pairwise_edge_metric():
    reg = _registry()
    _new, report, _h = _propose(reg, _payload(reg, edge_metric=False))
    assert report["rules"]["R1"]["status"] == "unavailable"


# ---- R2 ----------------------------------------------------------------------------------

def test_r2_unavailable_without_union_diagnostics(tmp_path):
    npz, why = cio.phase_union_npz(tmp_path, tmp_path / "epoch_000", _policy())   # top-ups off
    assert npz is None and "post-union" in why
    reg = _registry()
    _new, report, _h = _propose(reg, _payload(reg), union=None)
    assert report["rules"]["R2"]["status"] == "unavailable" and _decisions(report, "R2") == []


def test_r2_proposes_a_window_where_one_centre_carries_the_weight():
    reg = _registry(c1s=(0.2,), c2s=(-1.0, 0.0))
    views = cr.state_views(reg, _payload(reg))
    ids = sorted(views)
    rng = np.random.default_rng(5)
    cv2 = np.concatenate([rng.normal(-1.0, 0.3, 4000), rng.normal(0.0, 0.3, 4000), rng.normal(1.1, 0.25, 4000)])
    state_idx = np.repeat([0, 1, 1], 4000)         # the +1.1 lobe is sampled only by the c2 = 0 window
    cv1 = np.full(cv2.size, 0.2)
    w = np.full(cv2.size, 1.0 / cv2.size)
    cands = cov.coverage_holes(cv1, cv2, state_idx, w, [views[i] for i in ids], views, SETTINGS)
    prop = [c for c in cands if c["decision"] == "proposed"]
    assert prop and all(c["metrics"]["n_contributing_windows"] < 2 for c in prop)
    assert any(c["proposal"]["children"][0]["secondary_center"] > 0.4 for c in prop)


def test_mbar_fixed_point_recovers_harmonic_free_energies():
    rng = np.random.default_rng(0)
    views = [cr.StateView(i, 0.0, None, c, K2, 0.0) for i, c in enumerate((-0.5, 0.0, 0.5))]
    sd = math.sqrt(RT / K2)
    cv2 = np.concatenate([rng.normal(v.c2, sd, 3000) for v in views])
    u = cov.reduced_umbrella(np.zeros(cv2.size), cv2, views, 1.0 / RT)
    f = cov.solve_mbar(u, np.full(3, 3000.0))
    assert np.allclose(f, 0.0, atol=0.05)            # flat landscape: equal-width windows, equal f


# ---- budget --------------------------------------------------------------------------------

def _structural_setup(reg):
    a, b = _sid(reg, 0.2, -1.0), _sid(reg, 0.2, 0.0)
    low = [_sid(reg, c1, -1.0) for c1 in (0.2, 0.4)]
    rest = [s.state_id for s in reg.active_states() if s.state_id not in low]
    return _payload(reg, edges=[_edge(a, b, upper=0.05)], components=[low, rest])


@pytest.mark.parametrize("reserve,cap,code", [(None, 10, "no_reserve"), (RESERVE, 0, "no_reserve")])
def test_no_reserve_refuses_with_a_recorded_reason(reserve, cap, code):
    reg = _registry()
    new, report, _h = _propose(reg, _structural_setup(reg), reserve=reserve, policy=_policy(cap=cap))
    (cand,) = _decisions(report, "R1")
    assert cand["decision"] == "refused" and cand["refusal"] == code
    assert report["summary"]["n_blocking"] == 1 and not [a for a in new if a[0] == "add"]


def test_budget_split_add_rung_third_and_resolution_half_of_the_rest():
    reg = _registry(lambdas=(0.0, 1.0))             # 12 states, 6 centres x 2 rungs
    policy = _policy(cap=24)                        # 12 free slots
    rung = ("add_rung", 0.5, "gap")                 # +6 states > floor(12/3) = 4
    new, report, _h = _propose(reg, _structural_setup(reg), actions=[rung], policy=policy,
                               reserve={"fraction": 0.5, "max_replicas": 24})
    b = report["budget"]
    assert b["add_rung"]["refused"] and b["add_rung"]["reason"] == "add_rung_reserve_share"
    assert not [a for a in new if a[0] == "add_rung"]
    assert b["resolution_slots"] == 6               # floor(0.5 * (12 - 0))
    (cand,) = _decisions(report, "R1")
    assert cand["decision"] == "proposed" and cand["cost_states"] == 2


def test_resolution_budget_refuses_what_does_not_fit_in_priority_order():
    reg = _registry()
    cands = [cr.new_candidate("R3", "state", [1], "proposed", "r3"),
             cr.new_candidate("R1", "edge", [1, 2], "proposed", "r1", cls="weak")]
    from gareus.adaptive.reserve_budget import reserve_allowances
    allowance = reserve_allowances(10, 7, RESERVE, n_rungs=1)   # free 3 -> resolution floor(1.5) = 1
    out, rec = cr.allocate(cands, allowance, 1)
    by = {c["rule"]: c for c in out}
    assert by["R1"]["decision"] == "proposed" and by["R3"]["refusal"] == "resolution_budget"
    assert rec["spent_states"] == 1


def test_ignore_budget_dry_run_proposes_without_a_reserve():
    reg = _registry()
    _new, report, _h = _propose(reg, _structural_setup(reg), reserve=None, ignore_budget=True)
    assert report["budget"]["mode"] == "ignored" and _decisions(report, "R1")[0]["decision"] == "proposed"


# ---- protection, convergence ----------------------------------------------------------------

def test_new_windows_are_protected_from_retirement_and_refinement():
    reg = _registry()
    a, b = _sid(reg, 0.2, -1.0), _sid(reg, 0.2, 0.0)
    ap._apply_registry_actions(reg, [("add", a, (0.2, K1, -0.5, 4.0), "cv2_resolution R1: x",
                                      cr.action_metadata("R1", "weak", 0, a))], 0, policy=_policy(cap=0))
    child = next(s for s in reg.active_states() if cr.METADATA_KEY in s.metadata)
    assert child.created_epoch == 1
    new, report, _h = _propose(reg, _payload(reg), actions=[("retire", child.state_id, "redundant")], epoch=1)
    assert not [x for x in new if x[0] == "retire"]
    assert report["actions_removed"][0]["reason"] == "protected by cv2_resolution"
    new3, _r, _h = _propose(reg, _payload(reg), actions=[("retire", child.state_id, "redundant")], epoch=3)
    assert [x for x in new3 if x[0] == "retire"]    # protection lasts refine_protect_epochs (2)


def test_pending_resolution_blocks_convergence(tmp_path):
    policy = _policy()
    assert ap._adaptive_production_converged([("insert", 1, [], "r")], {"edges": []}, policy) is False
    reg = _registry()
    epoch_dir = tmp_path / "epoch_000"
    epoch_dir.mkdir()
    (epoch_dir / cr.REPORT_NAME).write_text(json.dumps({"summary": {"n_blocking": 2}}))
    gate = ap.evaluate_adaptive_convergence_gate(epoch_dir, 0, reg, {"states": [], "edges": []}, [], policy)
    assert any("CV2-resolution" in r for r in gate["continue_reasons"])
    assert gate["status"] != "converged"
