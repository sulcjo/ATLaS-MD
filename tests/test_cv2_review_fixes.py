"""Review-board fixes on the adaptive-CV2 branch (2026-09-30): convergence with budget
refusals, the no-reserve warning, the transition-count naming, the spring-cap CAUTION, the R3
gate record, the de-regularised curvature variance, policy validation and the ``_step`` rule.
"""
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gareus.adaptive_production as ap  # noqa: E402
from gareus.adaptive import cv2_resolution as cr  # noqa: E402
from gareus.adaptive import cv2_resolution_grade as grade  # noqa: E402
from gareus.adaptive import cv2_resolution_io as cio  # noqa: E402
from gareus.adaptive import cv2_resolution_rules as rules  # noqa: E402
from gareus.adaptive import cv2_resolution_summary as crs  # noqa: E402
from gareus.adaptive import cv2_shape as shp  # noqa: E402
from gareus.swarm import cv2_shape_layout as lay  # noqa: E402
from gareus.swarm.ladder_design import R_KCAL_MOL_K  # noqa: E402

T = 300.0
RT = R_KCAL_MOL_K * T


def _registry():
    reg = ap.WindowStateRegistry()
    for c1 in (0.2, 0.4):
        for c2 in (-1.0, 0.0, 1.0):
            reg.add_state(c1, 800.0, secondary_center=c2, secondary_k=5.0, gamd_lambda=0.0, epoch=0, source="seed")
    return reg


def _cand(decision, refusal=None, rule="R1"):
    return cr.new_candidate(rule, "edge", [1, 2], decision, "why", refusal=refusal)


# ---- condition 2: budget refusals never block convergence --------------------------------------

def test_budget_refusals_are_recorded_but_not_blocking():
    s = cr.summarise([_cand("refused", "no_reserve"), _cand("refused", "resolution_budget"),
                      _cand("proposed")])
    assert s["n_blocking"] == 1
    assert s["n_refused_budget"] == 2


def test_is_blocking_recounts_proposed_candidates_not_a_stale_summary(tmp_path):
    rep = {"summary": {"n_blocking": 2}, "candidates": [_cand("refused", "no_reserve")]}
    (tmp_path / cr.REPORT_NAME).write_text(json.dumps(rep))
    assert cio.is_blocking(tmp_path) == 0
    rep["candidates"].append(_cand("proposed"))
    (tmp_path / cr.REPORT_NAME).write_text(json.dumps(rep))
    assert cio.is_blocking(tmp_path) == 1


def test_no_reserve_refusals_let_the_convergence_gate_converge(tmp_path):
    policy = ap.AdaptiveDecisionPolicy(cv2_resolution=True, edge_metric="pairwise-mbar")
    epoch_dir = tmp_path / "epoch_000"
    epoch_dir.mkdir()
    rep = {"candidates": [_cand("refused", "no_reserve")],
           "summary": cr.summarise([_cand("refused", "no_reserve")])}
    (epoch_dir / cr.REPORT_NAME).write_text(json.dumps(rep))
    gate = ap.evaluate_adaptive_convergence_gate(epoch_dir, 0, _registry(), {"states": [], "edges": []}, [], policy)
    assert not any("CV2-resolution" in r for r in gate["continue_reasons"])
    assert gate["status"] == "converged"
    assert any("refused for budget" in r for r in gate["recommendations"])


def test_no_reserve_warning_is_printed_once_per_campaign_job(tmp_path, capsys):
    cio.reset_no_reserve_warnings()
    reg = _registry()
    policy = ap.AdaptiveDecisionPolicy(cv2_resolution=True, edge_metric="pairwise-mbar", max_replicas_budget=24)
    for epoch in (0, 1, 2):
        epoch_dir = tmp_path / f"epoch_{epoch:03d}"
        epoch_dir.mkdir()
        cio.run_epoch_cv2_resolution(adaptive_dir=tmp_path, epoch_dir=epoch_dir, epoch=epoch, registry=reg,
                                     diagnostics={"states": [], "edges": []}, actions=[], policy=policy,
                                     args=Namespace(temperature_k=T), out_dir=tmp_path, phase_dirs=[epoch_dir])
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("WARNING") and "--swarm-adaptive-reserve-fraction" in ln]
    assert len(lines) == 1, out
    assert "no_reserve" in lines[0]
    cio.reset_no_reserve_warnings()


# ---- condition 3: transition-count naming ------------------------------------------------------

def test_fallback_field_is_named_a_state_series_lower_bound():
    rec = {"source": "no_replica_column", "state_runs": [np.array([-1.0, 1.0, -1.0])]}
    out = rules._transitions(rec, (-0.5, 0.5))
    assert out["transitions"] is None and out["transitions_state_series_lower_bound"] == 2
    assert "transitions_lower_bound" not in out
    assert cr.SCHEMA_VERSION == "cv2_resolution_report_v3"


@pytest.mark.parametrize("key", ["transitions_lower_bound", "transitions_state_series_lower_bound"])
def test_summary_reads_v1_and_v2_fallback_keys(key):
    n, est = crs._transitions({"transitions": None, key: 7})
    assert (n, est) == (7, "state_series_lower_bound")


def test_state_series_count_is_never_below_the_replica_resolved_count():
    rng = np.random.default_rng(11)
    for trial in range(200):
        n = int(rng.integers(20, 200))
        steps = np.arange(n) * 10
        replica = np.cumsum(rng.random(n) < 0.3)          # residences: random swap points
        cv2 = np.where(rng.random(n) < 0.5, -1.0, 1.0) + rng.normal(0, 0.3, n)
        if trial % 3 == 0:                                  # step gaps break runs too
            steps = steps + np.cumsum(rng.random(n) < 0.05) * 7
        data = {"step": steps, "window_id": np.zeros(n, int), "replica": replica, "cv2": cv2}
        rec = cio._runs_one_source(data, {0: 5}, {5})[5]
        ss = cr.count_transitions(rec["state_runs"], -0.5, 0.5)
        rr = cr.count_transitions(rec["replica_runs"], -0.5, 0.5)
        assert ss >= rr


# ---- condition 4: spring-cap refusals are counted and CAUTION-graded ------------------------------

def _summary_with(cands):
    payload = {"states": [], "edges": [], "edge_metric": None}
    report = {"status": "ok", "candidates": cands, "summary": cr.summarise(cands), "settings": {"temperature_k": T}}
    return crs.build_summary(payload, report, label="epoch_002")


def test_spring_cap_refusals_are_counted_in_the_summary():
    s = _summary_with([_cand("refused", "k2_capped_below_compression", rule="R3"), _cand("proposed")])
    assert s["counts"]["n_refused_spring_cap"] == 1
    assert crs.build_summary({"states": [], "edges": []}, None, label="x")["counts"]["n_refused_spring_cap"] is None


def test_spring_cap_refusals_grade_caution_not_fail():
    c = {"n_states": 10, "n_cv2_restrained": 10, "n_graded_edges": 5, "n_weak": 0, "n_unmeasured": 0,
         "n_components": 1, "edge_metric_status": "ok", "report_status": "ok", "n_refused_budget": 0,
         "n_trapped_or_orthogonal": 0, "n_refused_spring_cap": 3,
         "evaluation_metadata": "ok", "evaluation_complete": True, "rules_incomplete": []}
    row = grade.check_cv2_resolution({"cv2_resolution": {"label": "epoch_002", "counts": c}})
    assert row["status"] == "caution"
    assert "k2_capped_below_compression" in row["detail"] and "spring cap" in grade.RULE_TEXT
    c["n_refused_spring_cap"] = 0
    assert grade.check_cv2_resolution({"cv2_resolution": {"label": "e", "counts": c}})["status"] == "pass"


# ---- condition 5: which R3 test decided -----------------------------------------------------------

def _unimodal(n=2000, seed=2):
    return np.random.default_rng(seed).normal(0.0, 0.3, n)


def _r3_propose(z, runs=None):
    reg = _registry()
    parent = next(s.state_id for s in reg.active_states() if s.primary_center == 0.2 and s.secondary_center == 0.0)
    rows = [{"state_id": s.state_id, "sample_count": 5000,
             "paired_cv": {"n_pairs": 2000, "cv2": {"n": 2000, "mean": 0.0, "var": 0.1, "skewness": 0.0,
                                                    "kurtosis_excess": 0.0}}} for s in reg.active_states()]
    sub = {parent: {"cv2": z, "cv1": np.full(z.size, 0.2), "source_index": np.zeros(z.size, int)}}
    runs_for = (lambda ids: {i: runs for i in ids}) if runs is not None else None
    policy = ap.AdaptiveDecisionPolicy(cv2_resolution=True, edge_metric="pairwise-mbar", max_replicas_budget=10)
    _new, report, _h = cio.propose_cv2_resolution(
        reg, {"states": rows, "edges": []}, [], cr.ResolutionSettings(temperature_k=T, refine_transition_count="replica", refine_r3_mode="insert", coverage_count="any",
                               refine_pmf_sigma_kT=0.5), policy,
        epoch=0,
        history={}, subsamples=sub, runs_for=runs_for, reserve={"fraction": 0.2, "max_replicas": 10})
    return next(c for c in report["candidates"] if c["rule"] == "R3" and c["state_ids"] == [parent])


def test_no_action_records_the_deciding_r3_gate_and_its_values():
    cand = _r3_propose(_unimodal())
    assert cand["decision"] == "no_action"
    m = cand["metrics"]
    assert m["r3_gate"] in cr.R3_GATES
    assert m["r3_gate"] in ("single_component", "member_support", "no_density_minimum", "depth_below_1kT",
                            "mode_weight_below_10pct")
    v = m["r3_gate_values"]
    assert v["min_depth_kT"] == cr.R3_MIN_DEPTH_KT and v["min_weight"] == cr.R3_MIN_WEIGHT
    assert v["min_mode_members"] == shp.DEFAULT_MIN_MODE_MEMBERS and v["n_components"] >= 1


def test_mode_gate_order_member_support_depth_weight():
    def comp(mean, var, w, n, ok=True):
        return shp.MixtureComponent(mean, var, w, n, ok, "accepted" if ok else "min_mode_members")
    one = cr.r3_mode_gate([comp(0.0, 0.1, 1.0, 20)])
    assert one["gate"] == "single_component"
    unsupported = cr.r3_mode_gate([comp(-1.0, 0.05, 0.5, 20), comp(1.0, 0.05, 0.5, 3, ok=False)])
    assert unsupported["gate"] == "member_support" and unsupported["values"]["component_members"] == [20, 3]
    shallow = cr.r3_mode_gate([comp(-0.4, 0.09, 0.5, 20), comp(0.4, 0.09, 0.5, 20)])
    assert shallow["gate"] in ("depth_below_1kT", "no_density_minimum")
    light = cr.r3_mode_gate([comp(-1.0, 0.02, 0.95, 20), comp(1.0, 0.02, 0.05, 20)])
    assert light["gate"] == "mode_weight_below_10pct" and light["values"]["deepest_pair"]["depth_kT"] >= 1.0
    good = cr.r3_mode_gate([comp(-1.0, 0.02, 0.5, 20), comp(1.0, 0.02, 0.5, 20)])
    assert good["gate"] == "passed" and good["pair"] == (0, 1)


def test_flagged_records_transitions_gate_and_summary_carries_it():
    rng = np.random.default_rng(1)
    z = np.where(rng.integers(0, 2, 2000) == 0, -0.66, 0.66) + rng.normal(0.0, 0.33, 2000)
    runs = {"source": "parquet", "replica_runs": [np.full(60, -0.66), np.full(60, 0.66)], "state_runs": []}
    cand = _r3_propose(z, runs)
    assert cand["decision"] == "flagged" and cand["metrics"]["r3_gate"] == "transitions_below_min"
    assert cand["metrics"]["r3_gate_values"]["min_transitions"] == 10
    flagged = _r3_propose(z, {"source": "no_replica_column", "replica_runs": [], "state_runs": []})
    assert flagged["metrics"]["r3_gate"] == "no_replica_series"
    row = crs.state_record({"state_id": cand["state_ids"][0]}, None, cand, T, True)
    assert row["r3_gate"] == "transitions_below_min" and "r3_gate" in crs._STATE_COLS


# ---- condition 6: de-regularised curvature variance --------------------------------------------

def _narrow_plus_broad(seed=4):
    rng = np.random.default_rng(seed)
    z, ids = [], []
    for m in range(60):                                     # 60 members, each in one mode
        narrow = m < 20
        z.append(rng.normal(-1.5, 0.2, 80) if narrow else rng.normal(0.3, 0.7, 80))
        ids.append(np.full(80, m))
    return np.concatenate(z), np.concatenate(ids)


def test_curvature_variance_removes_the_em_regularisation():
    z, ids = _narrow_plus_broad()
    fit = shp.fit_cv2_mixture(z, ids, reg_grid=(3e-2,), max_components=2)
    narrow = min(fit.components, key=lambda c: c.mean)
    reg_var = 3e-2 * fit.pooled_variance
    assert narrow.reg_variance == pytest.approx(reg_var)
    assert narrow.variance_curvature < narrow.variance - reg_var     # beyond plain subtraction
    true_var = 0.2 ** 2
    assert narrow.variance_curvature == pytest.approx(true_var, rel=0.10)
    assert narrow.variance > 1.5 * true_var                  # the bias the fix removes
    f2_true = shp.estimate_f2(true_var, fit.pooled_variance, narrow.n_members, T)
    f2_fixed = shp.estimate_f2(narrow.variance_curvature, fit.pooled_variance, narrow.n_members, T)
    f2_old = shp.estimate_f2(narrow.variance, fit.pooled_variance, narrow.n_members, T)
    assert f2_fixed == pytest.approx(f2_true, rel=0.10)      # stated tolerance: 10 %
    assert f2_old < 0.8 * f2_true                            # regularised: biased low by > 20 %
    rec = narrow.as_record()
    assert rec["variance_curvature"] == pytest.approx(narrow.variance_curvature)


def test_curvature_variance_floor_and_default():
    c = shp.MixtureComponent(0.0, 0.1, 1.0, 10, True, "accepted")
    assert c.variance_curvature == 0.1 and c.reg_variance == 0.0
    assert shp.curvature_variance(0.01, 0.0099, 1.0) == pytest.approx(shp.CURVATURE_VARIANCE_FLOOR * 1.0)
    assert shp.curvature_variance(0.2, 0.05, 1.0) == pytest.approx(0.15)
    assert shp.curvature_variance(0.2, 0.05, 1.0, polished=0.12) == pytest.approx(0.12)
    assert shp.curvature_variance(0.2, 0.05, 1.0, polished=0.5) == pytest.approx(0.2)    # never above variance


def test_every_f2_consumer_uses_the_curvature_variance():
    comp = shp.MixtureComponent(0.0, 0.10, 1.0, 40, True, "accepted", reg_variance=0.06, variance_curvature=0.04)
    fit = shp.CV2MixtureFit((comp,), 1, {}, 0.03, {}, 0, 1000, 40, 1000.0, 0.0, 2.0, 8)
    want = shp.estimate_f2(0.04, 2.0, 40, T)
    model = shp._ShapeModel(fit, T, 0.7, 1e-3, 1000.0, shp.DEFAULT_PRIOR_MEMBERS)
    assert model.f2(0.0)[0] == pytest.approx(want)
    pl = shp.place_cv2_centres(fit, (-1.0, 1.0), sigma_w_target=0.7, temperature_k=T, k_min=1e-3, k_max=1000.0)
    assert pl.f2[pl.kinds.index("mode")] == pytest.approx(want)
    col = SimpleNamespace(placement=SimpleNamespace(envelope=(-1.0, 1.0)), fit=fit, index=0, weight_share=1.0)
    (x7,) = lay.mode_axis_windows([col], [5.0], temperature_k=T, sigma_w_target=0.7, k_min=1e-3, k_max=1000.0)
    assert x7["f2"] == pytest.approx(want)
    view = cr.StateView(0, 0.2, 800.0, 0.0, 5.0, 0.0)
    modes = {"pair": [{**comp.as_record(), "mean": -0.5}, {**comp.as_record(), "mean": 0.5}],
             "fit": {"pooled_variance": 2.0}}
    kids = rules._r3_children(view, modes, cr.ResolutionSettings(temperature_k=T), None)
    assert kids[0]["f2_est"] == pytest.approx(cr.f2_under_bias(0.04, 5.0, T, pooled_var=2.0, n_members=40))
    old = {k: v for k, v in comp.as_record().items() if k != "variance_curvature"}   # a v1 record
    kids_v1 = rules._r3_children(view, {"pair": [{**old, "mean": -0.5}, {**old, "mean": 0.5}],
                                        "fit": {"pooled_variance": 2.0}}, cr.ResolutionSettings(temperature_k=T), None)
    assert kids_v1[0]["f2_est"] == pytest.approx(cr.f2_under_bias(0.10, 5.0, T, pooled_var=2.0, n_members=40))


# ---- condition 7: policy validation and the _step rule --------------------------------------------

def test_bad_transition_count_fails_at_policy_construction():
    with pytest.raises(ValueError, match="refine_transition_count"):
        ap.AdaptiveDecisionPolicy(refine_transition_count="bogus")
    with pytest.raises(ValueError, match="refine_transition_count"):
        ap.policy_from_args(Namespace(adaptive_production_refine_transition_count="bogus"))
    assert ap.policy_from_args(Namespace()).refine_transition_count == "replica-path"


def test_bad_recorded_transition_count_fails_at_load_with_the_file_named(tmp_path):
    policy = ap.AdaptiveDecisionPolicy()
    ap._resolve_decision_settings(tmp_path, policy)
    path = tmp_path / ap.DECISION_SETTINGS_FILENAME
    rec = json.loads(path.read_text())
    rec["settings"]["refine_transition_count"] = "swaps"
    path.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="decision_settings.json") as exc:
        ap._resolve_decision_settings(tmp_path, policy)
    assert "--ap-decision-settings-override" in str(exc.value)
    ap._resolve_decision_settings(tmp_path, policy, override=True)      # the remedy works


def test_every_placement_gap_is_within_the_spacing_rule():
    z, ids = _narrow_plus_broad()
    fit = shp.fit_cv2_mixture(z, ids, max_components=2)
    kw = dict(sigma_w_target=0.3, temperature_k=T, k_min=1e-3, k_max=1000.0)
    pl = shp.place_cv2_centres(fit, (-2.5, 2.5), **kw)
    model = shp._ShapeModel(fit, T, 0.3, 1e-3, 1000.0, shp.DEFAULT_PRIOR_MEMBERS)
    c = np.array(pl.centres)
    for a, b, kind in zip(c[:-1], c[1:], pl.kinds[1:]):
        if kind == "edge":
            continue
        min_sigma = float(np.min(model.sigma(np.linspace(a, b, 257))))
        assert b - a <= shp.DEFAULT_SPACING_SIGMA * min_sigma * (1.0 + 2e-2), (a, b, kind)
