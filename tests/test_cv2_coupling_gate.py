"""Spec 3.4 coupling gate (cv2_coupling_fraction) and P5 pair-model threading for the driver."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import gareus.adaptive_production as ap
from gareus.adaptive.pair_runtime import (AdaptiveCouplingGate, DriverPair, gate_from_args,
                                          load_driver_pair)
from gareus.cv_selection.coupling import (R_KCAL_MOL_K, CouplingWindow, cv2_coupling_fraction,
                                          gate_layout_rows, largest_passing_k2)
from gareus.cv_selection.models import (coupling_curvature_kcal, fit_residual_components,
                                        from_candidate_set, to_candidate_set)
from gareus.cv_selection.residual_runtime import compile_component
from gareus.swarm.gates import evaluate_gates

from test_cv_selection_conditional_tica import slow_selection  # noqa: F401  (module fixture)
from test_residual_model_v2 import _anchor, _dataset, _pair_v2, _schema8

T = 300.0
RT = R_KCAL_MOL_K * T


def _fit(degree):
    X, a = _dataset(degree)
    return fit_residual_components(X, a, degree=degree), X, a


def _c_of(fit, a_std):
    return fit.anchor_mean + a_std * fit.anchor_std


def _fd_curvature(comp, x_row, c, z0, k2, h):
    """d2/dc2 of k2 (z(c) - z0)^2 / 2 by central differences on the compiled coordinate."""
    def u(cc):
        z = comp.evaluate_features(x_row[None, :], np.array([cc]))[0]
        return 0.5 * k2 * (z - z0) ** 2
    return (u(c + h) - 2.0 * u(c) + u(c - h)) / (h * h)


# ---- the curvature formula ----------------------------------------------------------------

def test_curvature_formula_matches_finite_differences_of_the_compiled_coordinate():
    fit, X, _ = _fit(2)
    comp = compile_component(fit, 1, norm=1.0)
    assert comp.k2 != 0.0, "fixture must be genuinely quadratic"
    k2 = 3.0
    lo, hi = fit.anchor_clamp
    for a_std in np.linspace(lo + 0.2, hi - 0.2, 7):
        c = _c_of(fit, a_std)
        z_here = comp.evaluate_features(X[:1], np.array([c]))[0]
        _, dz, d2, _ = comp.anchor_partials(X[:1], np.array([c]))
        for offset in (-1.3, 0.0, 0.7):
            z0 = z_here - offset
            analytic = k2 * (dz[0] ** 2 + (z_here - z0) * d2[0])
            fd = _fd_curvature(comp, X[0], c, z0, k2, h=1e-4 * fit.anchor_std)
            assert fd == pytest.approx(analytic, rel=1e-5, abs=1e-6)


def test_box_maximum_equals_a_brute_force_finite_difference_grid():
    fit, X, _ = _fit(2)
    comp = compile_component(fit, 1, norm=1.0)
    k1, k2 = 2000.0, 5.0
    c1 = _c_of(fit, 0.3)
    window = CouplingWindow(primary_center=c1, primary_k=k1, temperature_k=T)
    out = cv2_coupling_fraction(fit, 1, k2, window)
    assert out["status"] == "ok" and not out["anchor_clamped"]
    s1, s2 = math.sqrt(RT / k1), math.sqrt(RT / k2)
    grid = []
    for c in np.linspace(c1 - 2 * s1, c1 + 2 * s1, 81):
        z_here = comp.evaluate_features(X[:1], np.array([c]))[0]
        for dz0 in (-2 * s2, 2 * s2):                       # |z - z0| <= 2 sigma_w2, extremes
            grid.append(_fd_curvature(comp, X[0], c, z_here - dz0, k2, h=1e-4 * fit.anchor_std))
    assert out["curvature_kcal"] == pytest.approx(max(grid), rel=1e-4)
    assert out["fraction"] == pytest.approx(out["curvature_kcal"] / (RT / s1 ** 2), rel=1e-12)


def test_degree_one_reduces_to_the_selection_certificate_value(slow_selection):  # noqa: F811
    _, sel = slow_selection
    fit = from_candidate_set(sel.candidate_set)
    j = int(sel.pair_model.selected_component_index)
    for k2 in (0.5, 1.0, 7.0):
        out = cv2_coupling_fraction(fit, j, k2, {"primary_center": fit.anchor_mean, "primary_k": 100.0})
        assert out["curvature_kcal"] == pytest.approx(coupling_curvature_kcal(fit, j, k2), rel=1e-12)
        assert out["fraction"] == pytest.approx(out["curvature_kcal"] / 100.0, rel=1e-12)


def test_designed_curvature_includes_f_second_and_falls_back_to_k1_when_not_positive():
    fit, _, _ = _fit(1)
    c1 = fit.anchor_mean
    with_f = cv2_coupling_fraction(fit, 1, 2.0, CouplingWindow(c1, 50.0, cv1_curvature_kcal=30.0))
    assert with_f["designed_cv1_curvature_kcal"] == pytest.approx(80.0)
    neg = cv2_coupling_fraction(fit, 1, 2.0, CouplingWindow(c1, 50.0, cv1_curvature_kcal=-70.0))
    assert neg["designed_cv1_curvature_kcal"] == pytest.approx(50.0)
    assert "k1 + F'' <= 0" in neg["sigma_w1_source"]


def test_window_at_the_training_clamp_is_evaluated_at_the_clamp_and_recorded():
    fit, _, _ = _fit(2)
    comp = compile_component(fit, 1, norm=1.0)
    hi = fit.anchor_clamp[1]
    out = cv2_coupling_fraction(fit, 1, 4.0, CouplingWindow(_c_of(fit, hi), 60.0))
    assert out["anchor_clamped"] is True
    # the one-sided derivative at the clamp, never the zero a hard clip gives beyond it
    at_clamp = abs(comp.k1 + 2 * comp.k2 * hi) / (comp.sigma_j * comp.anchor_std)
    assert out["dz_dc_max"] >= at_clamp > 0.0


def test_largest_passing_k2_lands_on_the_threshold_and_is_not_a_linear_rescale():
    fit, _, _ = _fit(2)
    window = CouplingWindow(_c_of(fit, 0.2), 8.0)
    k2 = 50.0
    out = largest_passing_k2(fit, 1, k2, window, max_fraction=0.25)
    assert out["lowered"] and out["before"]["fraction"] > 0.25
    assert out["after"]["fraction"] == pytest.approx(0.25, rel=1e-9)
    assert cv2_coupling_fraction(fit, 1, out["k2"] * 1.01, window)["fraction"] > 0.25
    linear = k2 * 0.25 / out["before"]["fraction"]
    assert abs(linear - out["k2"]) / out["k2"] > 1e-3     # the sigma_w2 term makes it nonlinear
    ok = largest_passing_k2(fit, 1, out["k2"] * 0.5, window, max_fraction=0.25)
    assert not ok["lowered"] and ok["k2"] == pytest.approx(out["k2"] * 0.5)


def test_k1_zero_window_needs_a_sampled_width_and_bounds_the_mean_shift():
    fit, _, _ = _fit(2)
    na = cv2_coupling_fraction(fit, 1, 2.0, CouplingWindow(fit.anchor_mean, 0.0))
    assert na["status"] == "NA" and na["fraction"] is None
    sd = 0.5 * fit.anchor_std
    w = CouplingWindow(fit.anchor_mean, 0.0, sigma_w1=sd)
    out = cv2_coupling_fraction(fit, 1, 2.0, w)
    assert out["mode"] == "k1_unrestrained"
    assert out["mean_shift_sigma"] == pytest.approx(out["tilt_kcal_per_cv"] * sd / RT)
    assert out["fraction"] == pytest.approx(max(out["curvature_ratio"], out["mean_shift_sigma"]))
    assert 0.0 < out["width_change_fraction"] < 1.0
    gated = largest_passing_k2(fit, 1, 500.0, w, max_fraction=0.25)
    assert gated["lowered"] and gated["after"]["fraction"] == pytest.approx(0.25, rel=1e-9)


def test_zero_k2_passes_trivially():
    fit, _, _ = _fit(1)
    out = cv2_coupling_fraction(fit, 1, 0.0, CouplingWindow(fit.anchor_mean, 10.0))
    assert out["status"] == "ok" and out["fraction"] == 0.0


# ---- swarm layout ------------------------------------------------------------------------

def test_layout_gate_lowers_cells_without_mutating_rows_and_fails_below_k_min():
    fit, _, _ = _fit(2)
    rows = [{"center1": _c_of(fit, 0.0), "k1": 8.0, "center2": 0.0, "k2": 50.0},
            {"center1": _c_of(fit, 0.0), "k1": 8.0, "center2": 1.0, "k2": 0.01},
            {"center1": _c_of(fit, 0.0), "k1": 0.0, "center2": 0.5, "k2": 20.0},
            {"center1": _c_of(fit, 0.5), "k1": 8.0, "center2": 0.0, "k2": 0.0}]
    cells = [(0, 0), (0, 1), (None, 2), (1, None)]
    snapshot = json.loads(json.dumps(rows))
    gated, rep = gate_layout_rows(fit, 1, rows, cells, [0.0, 0.0], temperature_k=T,
                                  pooled_cv1_sd=fit.anchor_std, max_fraction=0.25, k_min=1e-3)
    assert rows == snapshot
    assert gated[0]["k2"] < 50.0 and gated[1]["k2"] == 0.01 and gated[3]["k2"] == 0.0
    assert rep["ok"] and rep["n_lowered"] >= 1 and rep["gated_k2"][0] == gated[0]["k2"]
    _, strict = gate_layout_rows(fit, 1, rows, cells, [0.0, 0.0], temperature_k=T,
                                 pooled_cv1_sd=fit.anchor_std, max_fraction=0.25, k_min=40.0)
    assert not strict["ok"] and 0 in [d["row"] for d in strict["decisions"] if d["decision"] == "below_k_min"]
    assert "cv2_k_min" in strict["reasons"][0]


def test_swarm_gate_carries_the_coupling_verdict_only_when_given():
    base = evaluate_gates([], set(), {}, 0, {"lambdas": [0.0]}, [])
    assert "cv2_coupling" not in base["gates"]
    failed = evaluate_gates([], set(), {}, 0, {"lambdas": [0.0]}, [],
                            cv2_coupling={"ok": False, "reasons": ["cv2 coupling: x"], "n_lowered": 0,
                                          "n_below_k_min": 1})
    assert failed["gates"]["cv2_coupling"]["ok"] is False
    assert "cv2 coupling: x" in failed["reasons"] and failed["status"] == "fail"


# ---- P5: the frozen pair for the driver --------------------------------------------------

def _write_artifacts(tmp_path, *, deployable=True, degree=1):
    X, a = _dataset(degree)
    fit = fit_residual_components(X, a, degree=degree)
    schema = _schema8()
    cs = to_candidate_set(fit, schema, _anchor(), "a" * 64, "b" * 64, {"numpy": np.__version__})
    pair = _pair_v2(cs, schema, deployable=deployable)
    paths = (tmp_path / "cv_pair_model.json", tmp_path / "cv_candidate_set.json", tmp_path / "cv_feature_schema.json")
    paths[0].write_bytes(pair.to_json_bytes())
    paths[1].write_bytes(cs.to_json_bytes())
    paths[2].write_bytes(schema.to_json_bytes())
    return fit, pair, [str(p) for p in paths]


def _args(paths=(None, None, None), **kw):
    return SimpleNamespace(secondary_cv="residual-torsion-pc", secondary_cv_model=paths[0],
                           secondary_cv_candidate_set=paths[1], secondary_cv_feature_schema=paths[2], **kw)


def test_driver_pair_binds_a_verified_residual_model(tmp_path):
    fit, pair, paths = _write_artifacts(tmp_path)
    got = load_driver_pair(_args(paths))
    assert got.bound and got.j == 1 and got.pair_sha256 == pair.sha256 and got.paths_source == "args"
    np.testing.assert_array_equal(got.fit.right_vectors, fit.right_vectors)
    np.testing.assert_array_equal(got.fit.coefficients, fit.coefficients)


def test_driver_pair_restores_paths_from_the_campaign_manifest_on_resume(tmp_path):
    _, pair, paths = _write_artifacts(tmp_path)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {
        "secondary_cv_model": paths[0], "secondary_cv_candidate_set": paths[1],
        "secondary_cv_feature_schema": paths[2], "cv_pair_model_sha256": pair.sha256}}))
    got = load_driver_pair(_args(), tmp_path)
    assert got.bound and got.paths_source == "run_manifest"


def test_driver_pair_falls_back_to_a_phase_manifest_when_the_root_lacks_the_pair(tmp_path):
    """A swarm campaign's root manifest is written before the sidecar: only phases record the pair."""
    _, pair, paths = _write_artifacts(tmp_path)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {"secondary_cv": "auto"}}))
    phase = tmp_path / "adaptive_production" / "epoch_000"
    phase.mkdir(parents=True)
    (phase / "run_manifest.json").write_text(json.dumps({"method_settings": {
        "secondary_cv_model": paths[0], "secondary_cv_candidate_set": paths[1],
        "secondary_cv_feature_schema": paths[2], "cv_pair_model_sha256": pair.sha256}}))
    got = load_driver_pair(_args(), tmp_path)
    assert got.bound and got.paths_source == "run_manifest"
    (phase / "run_manifest.json").write_text(json.dumps({"method_settings": {
        "secondary_cv_model": paths[0], "secondary_cv_candidate_set": paths[1],
        "secondary_cv_feature_schema": paths[2], "cv_pair_model_sha256": "e" * 64}}))
    assert load_driver_pair(_args(paths), tmp_path).status == "NA"      # digest checked against the phase record


@pytest.mark.parametrize("case", ["other_cv2", "missing_path", "tampered", "undeployable", "digest"])
def test_driver_pair_is_na_and_never_raises(tmp_path, case):
    _, _, paths = _write_artifacts(tmp_path, deployable=(case != "undeployable"))
    args = _args(paths)
    if case == "other_cv2":
        args.secondary_cv = "tica-linear"
    elif case == "missing_path":
        args.secondary_cv_feature_schema = None
    elif case == "tampered":
        raw = json.loads(open(paths[1]).read())
        raw["components"][0]["projection_std"] = raw["components"][0]["projection_std"] * 1.0001
        open(paths[1], "w").write(json.dumps(raw))
    elif case == "digest":
        (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {"cv_pair_model_sha256": "f" * 64}}))
    got = load_driver_pair(args, tmp_path)
    assert got.status == "NA" and got.reason


# ---- the adaptive gate at the weak-edge bridge ---------------------------------------------

def _bridge_case(fit, k1=8.0, k2=60.0):
    reg = ap.WindowStateRegistry()
    reg.add_state(_c_of(fit, -0.2), k1, -0.5, k2, epoch=0, source="epoch0_windows")
    reg.add_state(_c_of(fit, 0.2), k1, 0.5, k2, epoch=0, source="epoch0_windows")
    diag = {"states": [{"state_id": 0, "sample_count": 100_000}, {"state_id": 1, "sample_count": 100_000}],
            "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.05}]}
    return reg, diag, ap.AdaptiveDecisionPolicy(max_new_windows_per_epoch=1, bridge_multi_window=False)


def _adds(reg, diag, pol, gate=None):
    return [a for a in ap.propose_actions_from_diagnostics(reg, diag, pol, temperature_K=T, coupling_gate=gate)
            if a[0] == "add"]


def test_gate_lowers_a_bridge_k2_and_records_why():
    fit, _, _ = _fit(2)
    reg, diag, pol = _bridge_case(fit)
    plain = _adds(reg, diag, pol)
    assert len(plain) == 1
    gate = AdaptiveCouplingGate(DriverPair("bound", fit=fit, j=1), temperature_k=T)
    gated = _adds(reg, diag, pol, gate)
    assert len(gated) == 1 and gated[0][2][3] < plain[0][2][3]
    assert gated[0][2][:3] == plain[0][2][:3]
    assert "CV2 coupling gate" in gated[0][3]
    rec = gate.report()
    assert rec["counts"] == {"lowered": 1} and rec["decisions"][0]["fraction_after"] == pytest.approx(0.25, rel=1e-9)


def test_gate_refuses_a_bridge_whose_passing_k2_is_below_k_min():
    fit, _, _ = _fit(2)
    reg, diag, pol = _bridge_case(fit)
    gate = AdaptiveCouplingGate(DriverPair("bound", fit=fit, j=1), temperature_k=T, k_min=55.0)
    assert _adds(reg, diag, pol, gate) == []
    assert gate.report()["counts"] == {"refused_below_k_min": 1}


def test_na_gate_blocks_nothing_and_changes_no_action():
    fit, _, _ = _fit(2)
    reg, diag, pol = _bridge_case(fit)
    gate = AdaptiveCouplingGate(DriverPair("NA", "cv2 is 'tica-linear'"), temperature_k=T)
    assert _adds(reg, diag, pol, gate) == _adds(reg, diag, pol)
    assert gate.report()["counts"] == {"NA": 1}


def test_gate_is_off_by_default_and_frozen_as_a_decision_rule(tmp_path):
    pol = ap.policy_from_args(SimpleNamespace())
    assert pol.cv2_coupling_gate is False and pol.max_coupling_fraction == 0.25
    assert {"cv2_coupling_gate", "max_coupling_fraction"} <= set(ap.DECISION_SETTINGS_FIELDS)
    assert gate_from_args(_args(), tmp_path, pol, temperature_k=T) is None
    on = ap.policy_from_args(SimpleNamespace(adaptive_production_cv2_coupling_gate=True,
                                             adaptive_production_max_coupling_fraction=0.2))
    gate = gate_from_args(_args(), tmp_path, on, temperature_k=T)
    assert gate is not None and gate.max_fraction == 0.2 and not gate.pair.bound


def test_cli_flags_default_off_and_reach_the_policy():
    from gareus.cli import parse_args

    base = ["--seq", "GA", "--out", "x"]
    off = parse_args(base)
    assert off.adaptive_production_cv2_coupling_gate is False
    assert off.adaptive_production_max_coupling_fraction == 0.25
    assert off.swarm_cv2_coupling_gate is False and off.swarm_cv2_max_coupling_fraction == 0.25
    on = parse_args(base + ["--ap-cv2-coupling-gate", "--ap-max-coupling-fraction", "0.2",
                            "--swarm-cv2-coupling-gate"])
    assert on.adaptive_production_cv2_coupling_gate is True and on.swarm_cv2_coupling_gate is True
    assert ap.policy_from_args(on).max_coupling_fraction == 0.2


# ---- the real epoch loop hands the gate to the proposer and writes its report ---------------

@pytest.mark.parametrize("on", [False, True])
def test_epoch_loop_threads_the_gate_and_writes_its_report(tmp_path, monkeypatch, on):
    from test_ladder_adapt_resume_e2e import _ReachedFinal, _campaign, _stub

    args, out, adaptive = _campaign(tmp_path)
    args.adaptive_production_cv2_coupling_gate = on
    calls = {"propose": 0, "diag_states": []}
    _stub(monkeypatch, calls, {"armed": False})
    seen = []

    def fake_propose(registry, diagnostics, **kw):
        seen.append(kw.get("coupling_gate"))
        return []

    monkeypatch.setattr(ap, "propose_actions_from_diagnostics", fake_propose)
    with pytest.raises(_ReachedFinal):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    report = adaptive / "epoch_000" / "cv2_coupling_gate.json"
    if not on:
        assert seen == [None] and not report.exists()
        return
    assert isinstance(seen[0], AdaptiveCouplingGate) and not seen[0].pair.bound
    rec = json.loads(report.read_text())
    assert rec["pair"]["status"] == "NA" and rec["decisions"] == []
    settings = json.loads((adaptive / "decision_settings.json").read_text())["settings"]
    assert settings["cv2_coupling_gate"] is True and settings["max_coupling_fraction"] == 0.25
