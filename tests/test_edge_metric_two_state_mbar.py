"""Spec 3.1: two-state MBAR edge metric behind --ap-edge-metric (gareus/adaptive/edge_metric.py).

OpenMM-free. Independent oracle: harmonic windows on a flat landscape sample
exact Gaussians, whose pairwise-MBAR-scale overlap
sqrt(N_a N_b) * int p_a p_b / (N_a p_a + N_b p_b) is integrated numerically.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gareus.adaptive_production as ap
from gareus.adaptive import edge_metric as em
from gareus.adaptive.neighbour_rule import NeighbourPoint, components
from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    collect_epoch_diagnostics,
    collect_final_combined_diagnostics,
    collect_segmented_epoch_diagnostics,
    registry_from_window_csv,
)

T = 300.0
BETA = em.beta_mol_per_kcal(T)
PAIRWISE = AdaptiveDecisionPolicy(edge_metric="pairwise-mbar")


def _sigma(k):
    return 1.0 / math.sqrt(BETA * k)


def _analytic_overlap_1d(ca, sa, cb, sb, na=1.0, nb=1.0):
    lo, hi = min(ca - 12 * sa, cb - 12 * sb), max(ca + 12 * sa, cb + 12 * sb)
    x = np.linspace(lo, hi, 400001)
    pa = np.exp(-0.5 * ((x - ca) / sa) ** 2) / (sa * math.sqrt(2 * math.pi))
    pb = np.exp(-0.5 * ((x - cb) / sb) ** 2) / (sb * math.sqrt(2 * math.pi))
    den = na * pa + nb * pb
    integrand = np.where(den > 0, pa * pb / np.where(den > 0, den, 1.0), 0.0)
    return math.sqrt(na * nb) * float(np.trapezoid(integrand, x))


def _rest(c1, k1, c2=None, k2=None, lam=0.0):
    return em.Restraint(primary_center=c1, primary_k=k1, secondary_center=c2, secondary_k=k2, gamd_lambda=lam)


def _gauss_state(sid, rest, n, rng, cv2_sd=None):
    cv1 = rng.normal(rest.primary_center, _sigma(rest.primary_k), n) if em._axis_restrained(rest.primary_k) \
        else rng.normal(0.0, 1.0, n)
    if rest.pattern[1]:
        cv2 = rng.normal(rest.secondary_center, _sigma(rest.secondary_k), n)
    else:
        cv2 = np.full(n, np.nan) if cv2_sd is None else rng.normal(0.0, cv2_sd, n)
    return em.StateSamples(sid, rest, cv1, cv2, np.zeros(n, dtype=np.int32))


# ---------------------------------------------------------------------------
# estimator: independent oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dc,ka,kb", [(0.10, 60.0, 60.0), (0.20, 60.0, 60.0), (0.15, 40.0, 160.0)])
def test_two_state_overlap_matches_analytic_gaussian_1d(dc, ka, kb):
    rng = np.random.default_rng(1)
    ra, rb = _rest(0.3, ka), _rest(0.3 + dc, kb)
    a, b = _gauss_state(0, ra, 40000, rng), _gauss_state(1, rb, 40000, rng)
    delta_a = rb.reduced_bias(a.cv1, a.cv2, BETA) - ra.reduced_bias(a.cv1, a.cv2, BETA)
    delta_b = rb.reduced_bias(b.cv1, b.cv2, BETA) - ra.reduced_bias(b.cv1, b.cv2, BETA)
    ov, df = em.two_state_overlap(delta_a, delta_b)
    truth = _analytic_overlap_1d(0.3, _sigma(ka), 0.3 + dc, _sigma(kb))
    assert ov == pytest.approx(truth, abs=0.01)
    # f_b - f_a = -ln(Z_b/Z_a) = 0.5 ln(k_b/k_a) for a harmonic window on a flat landscape.
    assert df == pytest.approx(0.5 * math.log(kb / ka), abs=0.03)


def test_two_state_overlap_matches_analytic_gaussian_2d():
    rng = np.random.default_rng(2)
    ra, rb = _rest(0.3, 80.0, -0.2, 20.0), _rest(0.38, 80.0, 0.1, 20.0)
    a, b = _gauss_state(0, ra, 40000, rng), _gauss_state(1, rb, 40000, rng)
    d = lambda s: rb.reduced_bias(s.cv1, s.cv2, BETA) - ra.reduced_bias(s.cv1, s.cv2, BETA)  # noqa: E731
    ov, _ = em.two_state_overlap(d(a), d(b))
    s1, s2 = _sigma(80.0), _sigma(20.0)
    x = np.linspace(0.3 - 8 * s1, 0.38 + 8 * s1, 801)
    y = np.linspace(-0.2 - 8 * s2, 0.1 + 8 * s2, 801)
    X, Y = np.meshgrid(x, y)
    pa = np.exp(-0.5 * (((X - 0.3) / s1) ** 2 + ((Y + 0.2) / s2) ** 2)) / (2 * math.pi * s1 * s2)
    pb = np.exp(-0.5 * (((X - 0.38) / s1) ** 2 + ((Y - 0.1) / s2) ** 2)) / (2 * math.pi * s1 * s2)
    truth = float(np.trapezoid(np.trapezoid(pa * pb / (pa + pb + 1e-300), x, axis=1), y))
    assert ov == pytest.approx(truth, abs=0.01)


def test_two_state_equals_union_pairwise_with_exact_f():
    """Spec T1: the two-state edge equals the union pairwise value on a synthetic two-state set."""
    from gareus.mbar_analysis.ladder import pairwise_state_overlap
    rng = np.random.default_rng(3)
    ra, rb = _rest(0.3, 60.0), _rest(0.42, 90.0)
    a, b = _gauss_state(0, ra, 20000, rng), _gauss_state(1, rb, 20000, rng)
    cv1 = np.concatenate([a.cv1, b.cv1])
    u = np.column_stack([ra.reduced_bias(cv1, cv1, BETA), rb.reduced_bias(cv1, cv1, BETA)])
    window = np.r_[np.zeros(a.cv1.size, int), np.ones(b.cv1.size, int)]
    exact_f = np.array([0.0, 0.5 * math.log(90.0 / 60.0)])
    union = pairwise_state_overlap(u, window, exact_f, np.array([a.cv1.size, b.cv1.size], float), 0, 1)
    ov, _ = em.two_state_overlap(u[:a.cv1.size, 1] - u[:a.cv1.size, 0], u[a.cv1.size:, 1] - u[a.cv1.size:, 0])
    assert ov == pytest.approx(union, abs=2e-3)


def test_unequal_sample_counts_symmetric_and_n_weighted():
    rng = np.random.default_rng(4)
    ra, rb = _rest(0.3, 60.0), _rest(0.44, 60.0)
    a, b = _gauss_state(0, ra, 8000, rng), _gauss_state(1, rb, 32000, rng)
    res_ab = em.evaluate_edge(a, b, BETA, n_bootstrap=20)
    res_ba = em.evaluate_edge(b, a, BETA, n_bootstrap=20)
    assert res_ab["overlap"] == pytest.approx(res_ba["overlap"], rel=1e-8)
    assert res_ab["delta_f_kT"] == pytest.approx(-res_ba["delta_f_kT"], abs=1e-8)
    truth = _analytic_overlap_1d(0.3, _sigma(60.0), 0.44, _sigma(60.0), 8000.0, 32000.0)
    assert res_ab["overlap"] == pytest.approx(truth, abs=0.01)
    assert res_ab["n"] == [8000, 32000]


def test_no_overlap_is_measured_near_zero_not_an_error():
    rng = np.random.default_rng(5)
    ra, rb = _rest(0.1, 400.0), _rest(0.9, 400.0)
    res = em.evaluate_edge(_gauss_state(0, ra, 3000, rng), _gauss_state(1, rb, 3000, rng), BETA, n_bootstrap=10)
    assert res["status"] == "ok" and res["overlap"] < 1e-6
    assert em.edge_is_weak_pairwise({"edge_type": "primary_chain", "pairwise_mbar": res}, 0.15)


# ---------------------------------------------------------------------------
# blocking, bootstrap, sufficiency
# ---------------------------------------------------------------------------


def _ar1(n, phi, rng):
    x = np.empty(n)
    x[0] = rng.normal()
    e = rng.normal(size=n) * math.sqrt(1 - phi * phi)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


@pytest.mark.parametrize("phi", [0.0, 0.6, 0.9])
def test_blocking_recovers_ar1_inefficiency(phi):
    rng = np.random.default_rng(6)
    x = _ar1(1 << 16, phi, rng)
    res = em.blocking_inefficiency(x)
    g_true = (1 + phi) / (1 - phi)
    assert res["g"] == pytest.approx(g_true, rel=0.25)
    assert res["tau"] == pytest.approx(0.5 * (g_true - 1), abs=max(0.3, 0.25 * g_true))
    assert res["n_eff"] == pytest.approx(x.size / res["g"])
    assert res["tau_err"] >= 0.0


def test_blocking_never_crosses_a_source():
    """Two sources with different means: blocks crossing the join would inflate g."""
    rng = np.random.default_rng(7)
    x = np.r_[rng.normal(size=4096), rng.normal(size=4096) + 0.0]
    src = np.r_[np.zeros(4096, int), np.ones(4096, int)]
    assert em.blocking_inefficiency(x, src)["g"] == pytest.approx(1.0, abs=0.3)
    assert em._segments(src, src.size) == [(0, 4096), (4096, 8192)]


def test_correlated_samples_below_min_neff_are_unmeasured_never_weak():
    rng = np.random.default_rng(8)
    ra, rb = _rest(0.1, 400.0), _rest(0.9, 400.0)   # no overlap at all
    n = 2000
    a = em.StateSamples(0, ra, 0.1 + _sigma(400.0) * _ar1(n, 0.99, rng), np.full(n, np.nan))
    b = em.StateSamples(1, rb, 0.9 + _sigma(400.0) * _ar1(n, 0.99, rng), np.full(n, np.nan))
    res = em.evaluate_edge(a, b, BETA, min_neff=200)
    assert res["status"] == "unmeasured" and res["reason"] == "low_neff"
    assert min(res["n_eff"]) < 200 and res["overlap_point"] < 1e-6
    assert not em.edge_is_weak_pairwise({"edge_type": "neighbour", "pairwise_mbar": res}, 0.15)


def test_bootstrap_bounds_bracket_the_point_and_decide_on_the_upper_bound():
    rng = np.random.default_rng(9)
    ra, rb = _rest(0.3, 60.0), _rest(0.3 + 1.9 * math.sqrt(2) * _sigma(60.0), 60.0)
    res = em.evaluate_edge(_gauss_state(0, ra, 1500, rng), _gauss_state(1, rb, 1500, rng), BETA, n_bootstrap=200)
    assert res["status"] == "ok"
    assert res["overlap_lower"] <= res["overlap"] <= res["overlap_upper"]
    edge = {"edge_type": "primary_chain", "pairwise_mbar": dict(res)}
    edge["pairwise_mbar"]["overlap_upper"] = 0.149
    assert em.edge_is_weak_pairwise(edge, 0.15)
    edge["pairwise_mbar"]["overlap_upper"] = 0.151   # point may be below; not CONFIDENTLY below
    edge["pairwise_mbar"]["overlap"] = 0.10
    assert not em.edge_is_weak_pairwise(edge, 0.15)


def test_bootstrap_width_matches_spread_over_repeated_datasets():
    """iid calibration: q10-q90 bootstrap width ~ 2.563 sd of the estimate over fresh datasets."""
    ra = _rest(0.3, 60.0)
    rb = _rest(0.3 + 1.5 * math.sqrt(2) * _sigma(60.0), 60.0)
    rng = np.random.default_rng(14)
    pts, widths = [], []
    for _ in range(20):
        res = em.evaluate_edge(_gauss_state(0, ra, 500, rng), _gauss_state(1, rb, 500, rng), BETA,
                               min_neff=10, n_bootstrap=100)
        pts.append(res["overlap"])
        widths.append(res["overlap_upper"] - res["overlap_lower"])
    assert np.mean(widths) == pytest.approx(2.563 * np.std(pts), rel=0.5)
    assert np.mean(pts) == pytest.approx(_analytic_overlap_1d(0.3, _sigma(60.0), rb.primary_center, _sigma(60.0)),
                                         abs=0.01)


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(10)
    ra, rb = _rest(0.3, 60.0), _rest(0.45, 60.0)
    a, b = _gauss_state(3, ra, 1000, rng), _gauss_state(7, rb, 1000, rng)
    assert em.evaluate_edge(a, b, BETA, n_bootstrap=30) == em.evaluate_edge(a, b, BETA, n_bootstrap=30)


# ---------------------------------------------------------------------------
# restraint patterns
# ---------------------------------------------------------------------------


def test_unrestrained_axis_pair_ignores_cv2():
    rng = np.random.default_rng(11)
    ra, rb = _rest(0.3, 60.0, -0.5, 0.0), _rest(0.42, 60.0, 0.7, 0.0)   # CV1-only: placeholders on CV2
    a, b = _gauss_state(0, ra, 20000, rng), _gauss_state(1, rb, 20000, rng)   # cv2 all NaN
    res = em.evaluate_edge(a, b, BETA, n_bootstrap=10)
    assert res["status"] == "ok" and res["n_dropped"] == [0, 0] and res["pattern_pair"] == "same"
    truth = _analytic_overlap_1d(0.3, _sigma(60.0), 0.42, _sigma(60.0))
    assert res["overlap"] == pytest.approx(truth, abs=0.01)


def test_cross_pattern_edge_drops_missing_cv2_and_is_never_weak():
    rng = np.random.default_rng(12)
    ra = _rest(0.3, 60.0, 0.0, 0.0)                 # CV1-only
    rb = _rest(0.3, 60.0, 5.0, 200.0)               # 2D, far off in CV2
    a = _gauss_state(0, ra, 3000, rng, cv2_sd=0.2)
    a.cv2[::3] = np.nan
    b = _gauss_state(1, rb, 3000, rng)
    res = em.evaluate_edge(a, b, BETA, n_bootstrap=10)
    assert res["pattern_pair"] == "cross" and res["n_dropped"][0] == 1000
    assert res["overlap"] < 0.01
    assert not em.edge_is_weak_pairwise({"edge_type": "nearest_2d", "pairwise_mbar": res}, 0.15)
    assert not em.edge_is_weak_pairwise({"edge_type": "spanning", "pairwise_mbar": res}, 0.15)


def test_unknown_k_is_unmeasured():
    rng = np.random.default_rng(13)
    a = _gauss_state(0, _rest(0.3, 60.0), 500, rng)
    b = em.StateSamples(1, _rest(0.4, None), a.cv1 + 0.1, a.cv2)
    res = em.evaluate_edge(a, b, BETA)
    assert res["status"] == "unmeasured" and res["reason"] == "restraint_k_unknown"


# ---------------------------------------------------------------------------
# weak predicate, proposer, default path
# ---------------------------------------------------------------------------


def test_unmeasured_is_never_weak_through_the_driver_predicate():
    for pm in (None, {"status": "unmeasured", "overlap_point": 0.0}, {"status": "ok", "overlap_upper": None}):
        edge = {"edge_type": "neighbour", "overlap": 0.01, "exchange_acceptance": 0.0}
        if pm is not None:
            edge["pairwise_mbar"] = pm
        assert not ap._edge_is_measured_weak(edge, PAIRWISE)
        assert ap._edge_is_measured_weak(edge, AdaptiveDecisionPolicy())   # marginal: unchanged


def test_radius_graph_edges_are_measured_but_never_weak():
    pm = {"status": "ok", "pattern_pair": "same", "overlap": 0.04, "overlap_upper": 0.05}
    for etype in ("neighbour", "spanning"):
        assert not ap._edge_is_measured_weak({"edge_type": etype, "pairwise_mbar": dict(pm)}, PAIRWISE)
        assert em.edge_below_threshold({"edge_type": etype, "pairwise_mbar": dict(pm)}, 0.15)
    for etype in ("primary_chain", "nearest_2d", "geometry", "segmented"):
        assert ap._edge_is_measured_weak({"edge_type": etype, "pairwise_mbar": dict(pm)}, PAIRWISE)


def test_components_keep_unmeasured_edges_and_cut_confident_gaps():
    ids = [0, 1, 2, 3]
    graded = [(0, 1, 0.3, False), (1, 2, None, False), (2, 3, 0.05, True)]
    out = em._components_summary(ids, graded, 0.15, [100, 100, 100, 100])
    assert out["n_components"] == 2 and out["components"] == [[0, 1, 2], [3]]
    assert out["n_graph_components"] == 1
    assert out["n_measured_components"] == 3          # the unmeasured 1-2 edge is a hole there


def test_union_value_overrides_the_pre_union_value():
    edge = {"edge_type": "primary_chain", "mbar_overlap": 0.10,
            "pairwise_mbar": {"status": "ok", "pattern_pair": "same", "overlap": 0.3, "overlap_upper": 0.35}}
    assert ap._edge_is_measured_weak(edge, PAIRWISE)
    edge["mbar_overlap"] = 0.2
    assert not ap._edge_is_measured_weak(edge, PAIRWISE)
    assert em.edge_sort_overlap(edge, PAIRWISE) == 0.2
    assert em.edge_sort_overlap(edge, AdaptiveDecisionPolicy()) is None   # marginal reads "overlap"


def _windows_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["primary_cv_center", "primary_cv_k_kcal",
                                           "secondary_cv_center", "secondary_cv_k_kcal_mol"])
        w.writeheader()
        for c1, k1, c2, k2 in rows:
            w.writerow({"primary_cv_center": c1, "primary_cv_k_kcal": k1,
                        "secondary_cv_center": c2, "secondary_cv_k_kcal_mol": k2})
    return path


def test_a_weak_spanning_edge_is_never_bridged(tmp_path):
    reg = registry_from_window_csv(_windows_csv(tmp_path / "w.csv", [(0.3, 80.0, 0.0, 0.0), (0.3, 80.0, 1.0, 100.0)]),
                                   epoch=0, source="t")
    weak = {"status": "ok", "pattern_pair": "cross", "overlap": 0.0, "overlap_upper": 0.0}
    diag = {"states": [{"state_id": 0, "sample_count": 5000}, {"state_id": 1, "sample_count": 5000}],
            "edges": [{"state_i": 0, "state_j": 1, "edge_type": "spanning", "overlap": None,
                       "pairwise_mbar": weak, "warnings": []}]}
    actions = ap.propose_actions_from_diagnostics(reg, diag, PAIRWISE, temperature_K=T)
    assert not [a for a in actions if a and a[0] == "add"]


def _flat_epoch(tmp_path, n=800, seed=21, temperature=True):
    """Three 2D states on one CV2 row plus one CV1-only state; Gaussian samples in each restraint."""
    rows = [(0.30, 80.0, 0.0, 20.0), (0.40, 80.0, 0.0, 20.0), (0.70, 80.0, 0.0, 20.0), (0.35, 80.0, 0.0, 0.0)]
    reg = registry_from_window_csv(_windows_csv(tmp_path / "windows.csv", rows), epoch=0, source="t")
    epoch = tmp_path / "epoch_000"
    epoch.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    out = []
    for wi, (c1, k1, c2, k2) in enumerate(rows):
        x = rng.normal(c1, _sigma(k1), n)
        y = rng.normal(c2, _sigma(k2), n) if k2 > 0 else rng.normal(0.05, 0.3, n)
        for t, (u, v) in enumerate(zip(x, y)):
            out.append({"window": wi, "step": t * 100, "cv_A": float(u), "secondary_cv": float(v),
                        "gamd_boost_total_kcal_mol": 1.0})
    with (epoch / "samples.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    if temperature:
        (epoch / "run_manifest.json").write_text(json.dumps({"resolved_args": {"temperature_k": T}}))
    return reg, epoch


def _strip_policy(payload):
    out = copy.deepcopy(payload)
    for key in ("edge_metric", "min_edge_neff"):
        out.get("policy", {}).pop(key, None)
    return out


def test_default_metric_leaves_diagnostics_unchanged(tmp_path, monkeypatch):
    reg, epoch = _flat_epoch(tmp_path)
    default = collect_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    assert "edge_metric" not in default
    assert not any("pairwise_mbar" in e for e in default["edges"])
    assert default["policy"]["edge_metric"] == "marginal"
    monkeypatch.setattr(ap, "attach_edge_metric", lambda payload, policy, run_dir: payload)
    hookless = collect_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    assert json.dumps(default, sort_keys=True) == json.dumps(hookless, sort_keys=True)


def test_pairwise_metric_grades_edges_and_keeps_existing_keys(tmp_path):
    reg, epoch = _flat_epoch(tmp_path)
    default = collect_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    on = collect_epoch_diagnostics(epoch, reg, PAIRWISE)
    rec = on["edge_metric"]
    assert rec["status"] == "ok" and rec["metric"] == "pairwise-mbar" and rec["temperature_k"] == T
    assert on["policy"]["edge_metric"] == "pairwise-mbar"          # P8 stamp in the report
    # every pre-existing edge keeps its keys and values; graph edges are appended after them
    n_old = len(default["edges"])
    for old, new in zip(default["edges"], on["edges"][:n_old]):
        new = {k: v for k, v in new.items() if k != "pairwise_mbar"}
        new["warnings"] = [w for w in new["warnings"] if w not in ("low_pairwise_mbar_overlap", "pairwise_mbar_unmeasured")]
        assert new == old
    by_pair = {tuple(sorted((e["state_i"], e["state_j"]))): e for e in on["edges"]}
    near = by_pair[(0, 1)]["pairwise_mbar"]          # 0.30 vs 0.40 at k1 = 80: d ~ 0.82 widths
    assert near["status"] == "ok" and near["overlap"] > 0.2
    assert by_pair[(0, 3)]["pairwise_mbar"]["pattern_pair"] == "cross"
    assert by_pair[(0, 3)]["pairwise_mbar"]["graph_kind"] in ("spanning", None)
    assert any(e["pairwise_mbar"].get("graph_kind") == "spanning" for e in on["edges"])
    # 0.40 -> 0.70 at k1 = 80 is a confident CV1 gap: the graph stays one piece, the verdict splits
    assert rec["components"]["n_graph_components"] == 1 and rec["components"]["n_components"] == 2
    assert any("components" in w for w in rec["warnings"])
    assert rec["stage"] == "pre_union"


def test_missing_temperature_is_an_error_record_not_a_crash(tmp_path):
    reg, epoch = _flat_epoch(tmp_path, temperature=False)
    on = collect_epoch_diagnostics(epoch, reg, PAIRWISE)
    assert on["edge_metric"]["status"] == "error"
    assert all(e["pairwise_mbar"]["status"] == "unmeasured" for e in on["edges"] if e["edge_type"] != "rung")
    assert not any(ap._edge_is_measured_weak(e, PAIRWISE) for e in on["edges"])


def test_segmented_and_final_collectors_grade_the_pooled_payload(tmp_path, monkeypatch):
    reg, epoch = _flat_epoch(tmp_path)
    seg_epoch = tmp_path / "epoch_001"
    for seg in ("baseline", "topup_001"):
        (seg_epoch / seg).mkdir(parents=True)
        for f in ("samples.csv", "run_manifest.json"):
            (seg_epoch / seg / f).write_bytes((epoch / f).read_bytes())
    calls = []
    real = ap.attach_edge_metric
    monkeypatch.setattr(ap, "attach_edge_metric", lambda p, pol, d: (calls.append(Path(d).name), real(p, pol, d))[1])
    seg = collect_segmented_epoch_diagnostics(seg_epoch, reg, PAIRWISE)
    assert calls == ["epoch_001"]                    # not once per segment
    assert seg["edge_metric"]["status"] == "ok"
    near = next(e for e in seg["edges"] if {e["state_i"], e["state_j"]} == {0, 1})["pairwise_mbar"]
    assert near["n"] == [1600, 1600]           # pooled over both segments
    monkeypatch.setattr(ap, "_final_sample_dirs", lambda d: [("baseline", seg_epoch / "baseline"),
                                                              ("topup_001", seg_epoch / "topup_001")])
    final = collect_final_combined_diagnostics(tmp_path, reg, policy=PAIRWISE)
    assert final["edge_metric"]["status"] == "ok"
    assert any(e.get("pairwise_mbar", {}).get("status") == "ok" for e in final["edges"])


def test_cli_flag_and_frozen_decision_setting(tmp_path):
    from gareus.cli import parse_args
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path)])
    assert args.adaptive_production_edge_metric == "marginal" and args.adaptive_production_min_edge_neff == 200.0
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path), "--ap-edge-metric", "pairwise-mbar",
                       "--ap-min-edge-neff", "150"])
    policy = ap.policy_from_args(args)
    assert policy.edge_metric == "pairwise-mbar" and policy.min_edge_neff == 150.0
    assert "edge_metric" in ap.DECISION_SETTINGS_FIELDS and "min_edge_neff" in ap.DECISION_SETTINGS_FIELDS
    frozen, _ = ap._resolve_decision_settings(tmp_path, policy)
    later, rec = ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy())
    assert later.edge_metric == "pairwise-mbar" and rec["settings"]["min_edge_neff"] == 150.0


# ---------------------------------------------------------------------------
# graph: spanning guarantee on chignolin_9's registry (skipped without RUNS/)
# ---------------------------------------------------------------------------

C9 = Path(__file__).resolve().parents[1] / "RUNS" / "chignolin_9" / "adaptive_production"
C9_MAIN = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_9/adaptive_production")


def _c9_dir():
    for d in (C9, C9_MAIN):
        if (d / "state_registry.json").exists() and (d / "adaptive_final_combined_diagnostics.json").exists():
            return d
    return None


@pytest.mark.skipif(_c9_dir() is None, reason="chignolin_9 run artefacts not present")
def test_chignolin_9_registry_graph_is_connected_with_spanning_edges():
    d = _c9_dir()
    states = json.loads((d / "state_registry.json").read_text())["states"]
    states = states.values() if isinstance(states, dict) else states
    diag = json.loads((d / "adaptive_final_combined_diagnostics.json").read_text())
    rows = {int(r["state_id"]): r for r in diag["states"]}
    lam0 = [s for s in states if s.get("active", True) and abs(float(s.get("gamd_lambda") or 0.0)) < 1e-9]
    points, nodes = [], []
    for s in lam0:
        r = rows.get(int(s["state_id"]), {})
        mean = (r.get("cv_mean"), r.get("secondary_mean"))
        p = NeighbourPoint(s["primary_center"], s["primary_k"], s["secondary_center"], s["secondary_k"],
                           0.0, mean)
        points.append(p)
        std = (r.get("cv_std"), r.get("secondary_std"))
        nodes.append((em.Restraint(s["primary_center"], s["primary_k"], s["secondary_center"], s["secondary_k"]),
                      tuple(None if v is None else float(v) ** 2 for v in std)))
    pooled = em.pooled_free_axis_sd(nodes)
    same = em.build_edge_graph(points, T, pooled_sd=pooled, n_spanning=0)
    assert len(components(len(points), [(i, j) for i, j, _k, _d in same])) >= 4
    graph = em.build_edge_graph(points, T, pooled_sd=pooled)
    assert len(components(len(points), [(i, j) for i, j, _k, _d in graph])) == 1
    spanned = {i for e in graph if e[2] == "spanning" for i in e[:2]}
    for i, p in enumerate(points):
        if p.pattern != (True, True):
            assert i in spanned
