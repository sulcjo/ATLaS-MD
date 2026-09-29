"""Spec 2026-09-29 adaptive-CV2, T2: the synth harness drives the REAL collectors.

OpenMM-free. The adapter (gareus/synth/collector_adapter.py) writes harness samples in the
on-disk layout of a production phase; the shipped ``collect_epoch_diagnostics`` (P4 paired
CV, spec 3.1 pairwise-MBAR grading) and ``propose_actions_from_diagnostics`` run on it.
Independent oracle: the analytic two-state overlap by quadrature.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import AdaptiveDecisionPolicy, _edge_is_measured_weak
from gareus.synth import collector_adapter as A
from gareus.synth.landscapes import (LANDSCAPES, LANDSCAPES_3D, Landscape, coupled_tilt,
                                     shoulder_conditional_cv2_density)
from gareus.synth.sampler import Window, sample_window_exact
from gareus.synth.vec_langevin import run_chains

PAIRWISE = AdaptiveDecisionPolicy(edge_metric="pairwise-mbar")
FLAT = Landscape("flat", lambda a, b: 0.0 * np.asarray(a) * np.asarray(b), (-5.0, 5.0), (-5.0, 5.0))


def _gauss_samples(w: Window, n: int, rng) -> np.ndarray:
    return np.column_stack([rng.normal(w.center1, 1 / math.sqrt(w.k1), n),
                            rng.normal(w.center2, 1 / math.sqrt(w.k2), n)])


def _analytic_1d(d_sigma: float) -> float:
    x = np.linspace(-20, 20, 400001)
    pa = np.exp(-0.5 * x ** 2); pb = np.exp(-0.5 * (x - d_sigma) ** 2)
    pa /= np.trapezoid(pa, x); pb /= np.trapezoid(pb, x)
    return float(np.trapezoid(pa * pb / (pa + pb), x))


def _edge(diag, a, b):
    return next(e for e in diag["edges"] if {int(e["state_i"]), int(e["state_j"])} == {a, b})


# ---------------------------------------------------------------------------
# truth quadrature and units
# ---------------------------------------------------------------------------


def test_exact_pair_overlap_matches_closed_form_on_a_flat_surface():
    wa, wb = Window(0.0, 400.0, 0.0, 25.0), Window(0.1, 400.0, 0.0, 25.0)   # 2 sigma_w on CV1
    out = A.exact_pair_overlap(FLAT, wa, wb)
    assert out["overlap"] == pytest.approx(_analytic_1d(2.0), abs=2e-4)
    assert out["delta_f"] == pytest.approx(0.0, abs=1e-6)
    wc = Window(0.0, 900.0, 0.3, 100.0)     # f_c - f_a = 0.5 ln(k1c/k1a) + 0.5 ln(k2c/k2a)
    assert A.exact_pair_overlap(FLAT, wa, wc)["delta_f"] == pytest.approx(
        0.5 * math.log(900 / 400) + 0.5 * math.log(100 / 25), abs=1e-4)


def test_units_round_trip_through_the_registry():
    ws = [Window(0.3, 150.0, -0.2, 12.0), Window(0.5, 150.0, None, None), Window(0.4, 0.0, 0.1, 9.0)]
    reg = A.registry_from_windows(ws)
    back = [A.window_of_state(s) for s in reg.active_states()]
    assert back[0].k1 == pytest.approx(150.0) and back[0].center1 == pytest.approx(0.3)
    assert back[0].k2 == pytest.approx(12.0) and back[0].center2 == pytest.approx(-0.2)
    assert back[1].center2 is None and back[1].k2 is None           # CV1-only stays CV1-only
    assert back[2].k1 == 0.0 and back[2].k2 == pytest.approx(9.0)   # CV2-only
    assert reg.active_states()[0].primary_k == pytest.approx(150.0 / A.beta_real())   # kcal/mol/CV^2


# ---------------------------------------------------------------------------
# adapter -> real collectors
# ---------------------------------------------------------------------------


def _three_windows():
    # 0-1: CV1 neighbours; 0-2: CV2 neighbours with a gap (3.2 sigma_w)
    return [Window(0.30, 400.0, 0.0, 25.0), Window(0.38, 400.0, 0.0, 25.0), Window(0.30, 400.0, 0.64, 25.0)]


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_adapter_writes_a_collector_valid_phase(tmp_path, fmt):
    ws = _three_windows()
    rng = np.random.default_rng(0)
    samples = {i: _gauss_samples(w, 3000, rng) for i, w in enumerate(ws)}
    reg = A.registry_from_windows(ws)
    A.write_phase_dir(tmp_path / "epoch_000", samples, dict(enumerate(ws)),
                      exchange_pairs=A.geometry_pairs(reg), rng=rng, fmt=fmt)
    for name in ("epoch_window_map.csv", "umbrella_explicit_windows.csv", "run_manifest.json"):
        assert (tmp_path / "epoch_000" / name).exists()
    diag = A.collect(tmp_path / "epoch_000", reg, PAIRWISE)
    assert diag["n_samples_rows"] == 9000 and diag["n_exchange_rows"] > 0
    assert {int(s["state_id"]): int(s["sample_count"]) for s in diag["states"]} == {0: 3000, 1: 3000, 2: 3000}
    assert diag["paired_cv"]["status"] == "ok"
    assert diag["edge_metric"]["status"] == "ok" and diag["edge_metric"]["temperature_k"] == A.TEMPERATURE_K
    for sid, w in enumerate(ws):
        rec = next(s for s in diag["states"] if int(s["state_id"]) == sid)["paired_cv"]["restraint"]
        assert rec["primary_k"] == pytest.approx(A.k_to_kcal(w.k1)) and rec["cv2_restrained"]


def test_parquet_and_csv_layouts_grade_identically(tmp_path):
    ws = _three_windows()
    rng = np.random.default_rng(1)
    samples = {i: _gauss_samples(w, 2500, rng) for i, w in enumerate(ws)}
    reg = A.registry_from_windows(ws)
    got = {}
    for fmt in ("parquet", "csv"):
        A.write_phase_dir(tmp_path / fmt, samples, dict(enumerate(ws)), rng=np.random.default_rng(2), fmt=fmt)
        d = A.collect(tmp_path / fmt, reg, PAIRWISE)
        got[fmt] = {(int(e["state_i"]), int(e["state_j"])): e["pairwise_mbar"].get("overlap") for e in d["edges"]}
    for key, v in got["csv"].items():
        assert got["parquet"][key] == pytest.approx(v, abs=1e-4)      # parquet stores CVs as float32


def test_analytic_overlap_vs_the_3_1_estimate_on_a_gaussian_pair(tmp_path):
    ls = LANDSCAPES["harmonic-bowl"]
    ws = _three_windows()
    rng = np.random.default_rng(3)
    samples = {i: sample_window_exact(ls, w, 4000, rng=rng, res=600) for i, w in enumerate(ws)}
    reg = A.registry_from_windows(ws)
    A.write_phase_dir(tmp_path / "e", samples, dict(enumerate(ws)), rng=rng, fmt="csv")
    diag = A.collect(tmp_path / "e", reg, PAIRWISE)
    for a, b in ((0, 1), (0, 2)):
        pm = _edge(diag, a, b)["pairwise_mbar"]
        truth = A.exact_pair_overlap(ls, ws[a], ws[b])["overlap"]
        assert pm["status"] == "ok"
        assert pm["overlap"] == pytest.approx(truth, abs=0.02)
    assert _edge_is_measured_weak(_edge(diag, 0, 2), PAIRWISE)          # truth ~0.07: a CV2 gap
    assert not _edge_is_measured_weak(_edge(diag, 0, 1), PAIRWISE)      # truth ~0.25


def test_calibration_evaluator_equals_the_collector_path(tmp_path):
    """t2_calibration grades edges directly; it must give the collector's exact record."""
    from gareus.synth.t2_calibration import _eval_job
    ws = [Window(0.30, 400.0, 0.0, 25.0), Window(0.38, 400.0, 0.1, 30.0)]
    rng = np.random.default_rng(4)
    run = run_chains(FLAT, np.array([[0.3, 0.0], [0.38, 0.1]]), 5000, c1=[0.3, 0.38], k1=400.0,
                     c2=[0.0, 0.1], k2=[25.0, 30.0], D=(1.0, 0.05), dt=5e-4, steps_per_sample=5, rng=rng)
    samples = {0: run.samples[0], 1: run.samples[1]}
    reg = A.registry_from_windows(ws)
    A.write_phase_dir(tmp_path / "e", samples, dict(enumerate(ws)), rng=rng, fmt="csv")
    pm = _edge(A.collect(tmp_path / "e", reg, PAIRWISE), 0, 1)["pairwise_mbar"]
    direct = _eval_job({"wa": ws[0], "wb": ws[1], "xa": samples[0], "xb": samples[1]})
    for key in ("overlap", "overlap_lower", "overlap_upper", "delta_f_kT"):
        assert direct[key] == pytest.approx(pm[key], abs=1e-9), key
    assert direct["n_eff"] == pytest.approx(pm["n_eff"], rel=1e-9)
    assert min(pm["n_eff"]) < 2000        # correlated chain: blocking sees it


def test_real_proposer_bridges_the_cv2_gap_only_under_pairwise(tmp_path):
    ls = LANDSCAPES["harmonic-bowl"]
    ws = _three_windows()
    rng = np.random.default_rng(5)
    samples = {i: sample_window_exact(ls, w, 3000, rng=rng, res=600) for i, w in enumerate(ws)}
    reg = A.registry_from_windows(ws)
    A.write_phase_dir(tmp_path / "e", samples, dict(enumerate(ws)), rng=rng, fmt="csv", exchange_pairs=())
    marg = A.collect(tmp_path / "e", reg, AdaptiveDecisionPolicy())
    assert A.action_counts(A.propose(reg, marg, AdaptiveDecisionPolicy())).get("add", 0) == 0
    pw = A.collect(tmp_path / "e", reg, PAIRWISE)
    adds = [a for a in A.propose(reg, pw, PAIRWISE) if a[0] == "add"]
    # the bridge sits between the CV2 rows (which geometry edge carries the gap -- the CV2
    # neighbour pair or the primary chain's diagonal -- is build_geometry_edges' choice)
    assert adds and all(0.0 < a[2][2] < 0.64 for a in adds)


@pytest.mark.xfail(strict=True, reason="production bug found by T2: the 3.1 grading appends neighbour/"
                   "spanning edges with overlap None, and the retire_converged loop marks both "
                   "endpoints of any edge with overlap None as bad_touching, so --ap-edge-metric "
                   "pairwise-mbar disables retirement entirely")
def test_pairwise_metric_does_not_disable_retirement(tmp_path):
    ls = LANDSCAPES["harmonic-bowl"]
    ws = [Window(float(c), 200.0, 0.0, 10.0) for c in np.linspace(0.3, 0.58, 8)]   # 0.04 apart, sigma_w 0.07
    rng = np.random.default_rng(7)
    samples = {i: sample_window_exact(ls, w, 1500, rng=rng, res=400) for i, w in enumerate(ws)}
    reg = A.registry_from_windows(ws)
    A.write_phase_dir(tmp_path / "e", samples, dict(enumerate(ws)), exchange_pairs=A.geometry_pairs(reg),
                      rng=rng, fmt="csv")
    marg = AdaptiveDecisionPolicy(min_active_states=2)
    pw = AdaptiveDecisionPolicy(min_active_states=2, edge_metric="pairwise-mbar")
    n_marg = A.action_counts(A.propose(reg, A.collect(tmp_path / "e", reg, marg), marg)).get("retire", 0)
    assert n_marg > 0                                   # redundant windows exist and marginal retires them
    diag = A.collect(tmp_path / "e", reg, pw)
    stripped = dict(diag, edges=[e for e in diag["edges"] if e["edge_type"] not in ("neighbour", "spanning")])
    assert A.action_counts(A.propose(reg, stripped, pw)).get("retire", 0) > 0   # cause: the appended edges
    assert A.action_counts(A.propose(reg, diag, pw)).get("retire", 0) > 0       # fails today: 0 retirements


# ---------------------------------------------------------------------------
# samplers and landscapes
# ---------------------------------------------------------------------------


def test_mala_chains_sample_the_exact_biased_distribution():
    ls = LANDSCAPES["harmonic-bowl"]
    w = Window(0.4, 200.0, 0.3, 30.0)
    run = run_chains(ls, np.tile([[0.4, 0.3]], (100, 1)), 300, c1=0.4, k1=200.0, c2=0.3, k2=30.0,
                     D=(1.0, 0.05), dt=2e-3, steps_per_sample=10, burn_in=300, rng=np.random.default_rng(6))
    x = run.samples.reshape(-1, 2)
    m = A.exact_window_moments(ls, w)
    assert x[:, 0].std() == pytest.approx(m["sd1"], rel=0.03)      # Euler at D k dt = 0.4 would be +11 %
    assert x[:, 0].mean() == pytest.approx(m["mean1"], abs=0.003)
    assert 0.3 < run.acceptance <= 1.0


def test_new_landscapes_have_the_stated_shape():
    for name in ("stiff-bowl", "narrow-cv2-band-at-high-cv1", "plateau-walls"):
        _c1, _c2, f = LANDSCAPES[name].grid(res=60)
        assert np.isfinite(f).all() and f.min() == 0.0
    y = np.linspace(-6, 6, 20001)
    for c1, narrow in ((0.2, 0.0), (1.0, 0.2)):
        p = shoulder_conditional_cv2_density(np.full_like(y, c1), y)
        assert np.trapezoid(p, y) == pytest.approx(1.0, abs=1e-6)
        assert np.trapezoid(p[y > 1.1], y[y > 1.1]) == pytest.approx(narrow + 0.8 * 0.061 if narrow else 0.061,
                                                                    abs=0.02)
    tilt = coupled_tilt(0.8)
    e = tilt.energy(np.array([0.7]), np.array([0.8 * 0.2]))
    assert float(e[0]) == pytest.approx(0.5 * 16.0 * 0.2 ** 2)     # on the ridge only the CV1 term
    ax1, ax2, f2 = LANDSCAPES_3D["hidden-slow-cv3"].marginal_2d(res=41)
    row = f2[20]                                                    # cv1 = 0.5: bimodal in CV2
    assert row[np.argmin(np.abs(ax2))] > row.min() + 1.0


# ---------------------------------------------------------------------------
# end-to-end resolution: switched on once spec 3.3 (R1-R3) exists
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="spec 3.3 resolution actions (R1-R3) are not implemented; today's proposer "
                         "can only place a midpoint bridge at the saddle, which does not resolve it")
def test_slow_cv2_double_branch_is_resolved_end_to_end():
    from gareus.synth.t2_campaign import RESOLUTION_TOLERANCE, is_resolved, run_arm
    for seed in (0, 1, 2):
        out = run_arm("slow-cv2-double-branch-asym:k2x4", "pairwise-mbar", seed)
        assert not is_resolved(out["epochs"][0])          # the starting layout really is unresolved
        final = out["epochs"][-1]
        assert final["branch_delta_f_err_kT"] <= RESOLUTION_TOLERANCE["branch_delta_f_err_kT"]
        assert final["cv2_pmf_rmse_lowf_kT"] <= RESOLUTION_TOLERANCE["cv2_pmf_rmse_lowf_kT"]
        assert final["components_truth"] == 1
