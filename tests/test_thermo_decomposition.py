"""Tests for gareus.mbar_analysis.thermo (spec: docs/superpowers/specs/2026-09-23-thermo-energy-decomposition/spec.md)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gareus.mbar_analysis import thermo
from gareus.mbar_analysis.data import Data
from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj

BETA = 1.0 / (0.00831446261815324 * 300.0)  # 1/(kJ/mol) at 300 K


def _env():
    # Total channel: v_pep ~ -3000; dihedral channel: v_dih ~ 100.
    return PepGamdEnvelope(vmax_total=-2800.0, vmin_total=-3200.0, threshold_total=-2800.0, k0max_total=0.8,
                           vmax_dih=140.0, vmin_dih=60.0, threshold_dih=140.0, k0max_dih=0.6)


def _data(tmp_path, *, cv, u_own_reduced, window, K, potential, v_pep, v_dih, replica, epoch_src=None,
          boost=None, state_lambdas=None, ladder=False):
    n = cv.size
    u_nk = np.zeros((n, K))
    u_nk[np.arange(n), window] = u_own_reduced
    meta = {'gamd_ladder': bool(ladder)}
    if epoch_src is not None:
        meta['_epoch_source'] = list(np.asarray(epoch_src, int))
    return Data(prod_dir=Path(tmp_path), out_dir=Path(tmp_path) / 'pmf_analysis', cv=cv, cv2=np.full(n, np.nan),
                rg_A=np.full(n, np.nan), window=np.asarray(window, int), replica=np.asarray(replica, int),
                step=np.arange(n), u_nk=u_nk, centers=np.zeros(K), k_kcal=np.ones(K), beta=BETA,
                temp=300.0, boost_kj=np.zeros(n) if boost is None else boost, potential_kj=potential,
                source='synthetic', meta=meta, v_pep_kj=v_pep, v_dih_kj=v_dih, state_lambdas=state_lambdas)


# --- own-window umbrella recovery -------------------------------------------------------------

def test_own_umbrella_off_ladder_is_own_column_over_beta():
    W = np.array([0.0, 1.5, 3.0, 0.2])
    u = np.zeros((4, 3)); win = np.array([0, 2, 1, 2])
    u[np.arange(4), win] = BETA * W
    got = thermo.own_state_umbrella_kj(u, win, BETA, v_pep=None, v_dih=None, state_lambdas=None, envelope=None)
    np.testing.assert_allclose(got, W, rtol=0, atol=1e-12)


def test_own_umbrella_ladder_subtracts_own_rung_boost_exactly():
    rng = np.random.default_rng(1)
    n, lams = 400, np.array([0.0, 0.5, 1.0])
    env = _env()
    win = rng.integers(0, 3, n)
    v_pep = rng.uniform(-3200, -2800, n); v_dih = rng.uniform(60, 140, n)
    W = rng.exponential(2.0, n)
    boost_own = np.array([pep_gamd_boost_kj(v_pep[i], v_dih[i], lams[win[i]], env) for i in range(n)])
    assert boost_own.max() > 1.0  # the subtraction is actually exercised
    u = np.zeros((n, 3)); u[np.arange(n), win] = BETA * (W + boost_own)
    got = thermo.own_state_umbrella_kj(u, win, BETA, v_pep=v_pep, v_dih=v_dih, state_lambdas=lams, envelope=env)
    np.testing.assert_allclose(got, W, atol=1e-9)
    # Mutations that must NOT reproduce W: wrong rung, boost added twice, wrong sign.
    wrong_rung = np.array([pep_gamd_boost_kj(v_pep[i], v_dih[i], lams[(win[i] + 1) % 3], env) for i in range(n)])
    assert np.max(np.abs((W + boost_own - wrong_rung) - W)) > 1.0
    assert np.max(np.abs(got - (W - boost_own))) > 1.0


def test_own_umbrella_ladder_without_envelope_refuses():
    u = np.zeros((2, 2)); win = np.array([0, 1])
    with pytest.raises(ValueError, match='envelope'):
        thermo.own_state_umbrella_kj(u, win, BETA, v_pep=np.zeros(2), v_dih=np.zeros(2),
                                     state_lambdas=np.array([0.0, 0.5]), envelope=None)


def test_negative_own_umbrella_is_flagged():
    diag = thermo.own_umbrella_diagnostics(np.array([0.5, -0.2, 1.0]), np.array([0, 1, 1]), BETA)
    assert diag['n_negative'] == 1 and diag['ok'] is False
    diag = thermo.own_umbrella_diagnostics(np.array([0.5, 0.0, 1.0]), np.array([0, 1, 1]), BETA)
    assert diag['ok'] is True and diag['n_negative'] == 0


# --- boost type / weights ---------------------------------------------------------------------

def test_resolve_boost_type_reads_manifest_from_parent(tmp_path):
    (tmp_path / 'run_manifest.json').write_text(json.dumps({'method_settings': {'gamd_boost_type': 'pep-gamd-lower-dual'}}))
    sub = tmp_path / 'adaptive_production'; sub.mkdir()
    assert thermo.resolve_boost_type(sub) == 'pep-gamd-lower-dual'
    assert thermo.resolve_boost_type(tmp_path / 'a' / 'b' / 'c') is None  # manifest is 3 levels up: out of reach


@pytest.mark.parametrize('bt,split', [('pep-gamd-lower-dual', True), ('pep-gamd-lower-dual-rs', False),
                                      ('lower-dual', False), (None, False)])
def test_split_requires_exact_kernel_name(bt, split):
    assert thermo.split_supported(bt) is split


def test_weight_policy(tmp_path):
    n = 10
    base = dict(cv=np.linspace(0, 1, n), u_own_reduced=np.zeros(n), window=np.zeros(n, int), K=1,
                potential=np.zeros(n), v_pep=np.zeros(n), v_dih=np.zeros(n), replica=np.zeros(n, int))
    logw = np.zeros(n)
    w, method, _ = thermo.observable_weights(_data(tmp_path, **base), logw)
    assert method == 'mbar_unboosted' and w is not None
    w, method, _ = thermo.observable_weights(_data(tmp_path, ladder=True, boost=np.ones(n), **base), logw)
    assert method == 'mbar_ladder_exact' and w is not None
    w, method, reason = thermo.observable_weights(_data(tmp_path, boost=np.linspace(0, 5, n), **base), logw)
    assert w is None and 'stock GaMD' in reason


# --- basins -----------------------------------------------------------------------------------

def test_parse_basins():
    assert thermo.parse_basin_specs(['folded:0.5:1.0', 'open:0:0.2']) == [('folded', 0.5, 1.0), ('open', 0.0, 0.2)]
    for bad in (['x:1:0'], ['x:0:1', 'x:1:2'], ['nope']):
        with pytest.raises(ValueError):
            thermo.parse_basin_specs(bad)


def test_auto_basins_split_at_barrier():
    x = np.linspace(-2, 2, 4001)
    F = (x ** 2 - 1) ** 2  # minima at +-1, barrier at 0 (kT units below)
    w = np.exp(-4.0 * F); w /= w.sum()
    basins = thermo.auto_basins(x, w, 40, kbt_kcal=1.0)
    assert [b[0] for b in basins] == ['auto_low_cv', 'auto_high_cv']
    assert abs(basins[0][2]) < 0.15 and basins[0][2] == basins[1][1]
    assert thermo.auto_basins(x, np.exp(-x ** 2) / np.exp(-x ** 2).sum(), 40, kbt_kcal=1.0) == []


# --- estimator: independent quadrature oracle ---------------------------------------------------

def _two_region_system(n=200_000, seed=0):
    """Samples from a uniform proposal on [0,1] with importance weights exp(-beta*U_phys).

    U_phys = V_pep + U_ee with V_pep = a*x and U_ee = b*sin(3x) + noise-free, so every
    basin average and free energy has an exact quadrature answer.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    a, b = 12.0, 7.0
    v_pep = a * x - 3000.0
    u_ee = b * np.sin(3 * x) - 262000.0
    U = v_pep + u_ee
    logw = -BETA * (U - U.min())
    return x, v_pep, u_ee, U, logw, (a, b)


def _quad(lo, hi, a, b):
    xs = np.linspace(lo, hi, 200_001)
    vp = a * xs - 3000.0; ue = b * np.sin(3 * xs) - 262000.0; U = vp + ue
    p = np.exp(-BETA * (U - U.min()))
    Z = np.trapezoid(p, xs)
    return Z, np.trapezoid(p * U, xs) / Z, np.trapezoid(p * vp, xs) / Z, np.trapezoid(p * ue, xs) / Z, U.min()


def test_basin_thermodynamics_match_quadrature(tmp_path):
    x, v_pep, u_ee, U, logw, (a, b) = _two_region_system()
    n = x.size
    W = np.full(n, 0.7)  # a constant umbrella that must be removed from potential
    d = _data(tmp_path, cv=x, u_own_reduced=BETA * W, window=np.zeros(n, int), K=1,
              potential=U + W, v_pep=v_pep, v_dih=np.full(n, 100.0), replica=np.arange(n) % 40)
    res = thermo.compute_thermo(d, logw, boost_type='pep-gamd-lower-dual',
                                basins=[('A', 0.0, 0.4), ('B', 0.6, 1.0)], n_bins=20, n_boot=50, seed=3,
                                min_basin_ess=100, min_basin_blocks=8, envelope_loader=lambda p: None)
    assert res['available'] and res['split_available']
    ZA, HA, VA, EA, m1 = _quad(0.0, 0.4, a, b)
    ZB, HB, VB, EB, m2 = _quad(0.6, 1.0, a, b)
    # shift both Z to a common energy origin: _quad normalizes by its own U.min()
    dG = -np.log((ZB * np.exp(-BETA * (m2 - m1))) / ZA) / BETA
    diff = res['differences'][0]
    assert diff['pair'] == 'A->B' and diff['status'] == 'ok'
    q = {k: diff[k]['estimate'] for k in ('dG_kj', 'dH_kj', 'dV_pep_kj', 'dU_ee_kj', 'minus_TdS_kj', 'minus_TdS_prime_kj')}
    assert q['dG_kj'] == pytest.approx(dG, abs=0.05)
    assert q['dH_kj'] == pytest.approx(HB - HA, abs=0.05)
    assert q['dV_pep_kj'] == pytest.approx(VB - VA, abs=0.05)
    assert q['dU_ee_kj'] == pytest.approx(EB - EA, abs=0.05)
    # exact algebraic identities of the split
    assert q['dH_kj'] == pytest.approx(q['dV_pep_kj'] + q['dU_ee_kj'], abs=1e-6)
    assert q['minus_TdS_kj'] == pytest.approx(q['dG_kj'] - q['dH_kj'], abs=1e-9)
    assert q['minus_TdS_prime_kj'] == pytest.approx(q['dG_kj'] - q['dV_pep_kj'], abs=1e-9)
    assert diff['dH_kj']['se'] > 0 and np.isfinite(diff['dH_kj']['se'])
    # bootstrap is paired: -TdS replicates equal dG - dH replicates exactly
    reps = res['_replicates']['A->B']
    np.testing.assert_allclose(reps['minus_TdS_kj'], reps['dG_kj'] - reps['dH_kj'], atol=1e-9)


def test_umbrella_not_removed_is_a_detectable_error(tmp_path):
    """Mutation: a window-dependent W left inside U must shift dH (the channel is not blind to W)."""
    x, v_pep, u_ee, U, logw, _ = _two_region_system(n=50_000)
    n = x.size
    W = 5.0 * x
    common = dict(cv=x, window=np.zeros(n, int), K=1, v_pep=v_pep, v_dih=np.full(n, 100.0), replica=np.arange(n) % 40)
    good = thermo.compute_thermo(_data(tmp_path, u_own_reduced=BETA * W, potential=U + W, **common), logw,
                                 boost_type='pep-gamd-lower-dual', basins=[('A', 0, .4), ('B', .6, 1)], n_bins=10,
                                 n_boot=0, seed=0, min_basin_ess=10, min_basin_blocks=2, envelope_loader=lambda p: None)
    bad = thermo.compute_thermo(_data(tmp_path, u_own_reduced=np.zeros(n), potential=U + W, **common), logw,
                                boost_type='pep-gamd-lower-dual', basins=[('A', 0, .4), ('B', .6, 1)], n_bins=10,
                                n_boot=0, seed=0, min_basin_ess=10, min_basin_blocks=2, envelope_loader=lambda p: None)
    g = good['differences'][0]['dH_kj']['estimate']; b = bad['differences'][0]['dH_kj']['estimate']
    assert abs(b - g) > 2.0


def test_split_refused_for_non_exact_kernel_but_totals_kept(tmp_path):
    x, v_pep, u_ee, U, logw, _ = _two_region_system(n=20_000)
    n = x.size
    d = _data(tmp_path, cv=x, u_own_reduced=np.zeros(n), window=np.zeros(n, int), K=1, potential=U,
              v_pep=v_pep, v_dih=np.full(n, 100.0), replica=np.arange(n) % 20)
    res = thermo.compute_thermo(d, logw, boost_type='pep-gamd-lower-dual-rs', basins=[('A', 0, .4), ('B', .6, 1)],
                                n_bins=10, n_boot=0, seed=0, min_basin_ess=10, min_basin_blocks=2,
                                envelope_loader=lambda p: None)
    assert res['available'] and not res['split_available'] and 'pep-gamd-lower-dual-rs' in res['split_reason']
    diff = res['differences'][0]
    assert np.isfinite(diff['dH_kj']['estimate']) and 'dU_ee_kj' not in diff


def test_basin_with_few_blocks_is_inconclusive(tmp_path):
    x, v_pep, u_ee, U, logw, _ = _two_region_system(n=20_000)
    n = x.size
    replica = np.where(x > 0.6, 0, np.arange(n) % 20)  # basin B lives in a single block
    d = _data(tmp_path, cv=x, u_own_reduced=np.zeros(n), window=np.zeros(n, int), K=1, potential=U,
              v_pep=v_pep, v_dih=np.full(n, 100.0), replica=replica)
    res = thermo.compute_thermo(d, logw, boost_type='pep-gamd-lower-dual', basins=[('A', 0, .4), ('B', .6, 1)],
                                n_bins=10, n_boot=20, seed=0, min_basin_ess=10, min_basin_blocks=8,
                                envelope_loader=lambda p: None)
    assert res['differences'][0]['status'] == 'inconclusive'
    assert res['basins'][1]['n_blocks'] == 1


def test_missing_potential_is_unavailable(tmp_path):
    n = 100
    d = _data(tmp_path, cv=np.linspace(0, 1, n), u_own_reduced=np.zeros(n), window=np.zeros(n, int), K=1,
              potential=None, v_pep=np.zeros(n), v_dih=np.zeros(n), replica=np.zeros(n, int))
    res = thermo.compute_thermo(d, np.zeros(n), boost_type='pep-gamd-lower-dual', basins=[], n_bins=10, n_boot=0,
                                seed=0, min_basin_ess=10, min_basin_blocks=2, envelope_loader=lambda p: None)
    assert res['available'] is False and 'potential' in res['reason']


# --- driver / outputs -------------------------------------------------------------------------

def test_analyze_writes_outputs(tmp_path):
    x, v_pep, u_ee, U, logw, _ = _two_region_system(n=30_000)
    n = x.size
    (tmp_path / 'run_manifest.json').write_text(json.dumps({'method_settings': {'gamd_boost_type': 'pep-gamd-lower-dual'}}))
    d = _data(tmp_path, cv=x, u_own_reduced=np.zeros(n), window=np.zeros(n, int), K=1, potential=U,
              v_pep=v_pep, v_dih=np.full(n, 100.0), replica=np.arange(n) % 20)
    args = SimpleNamespace(thermo_basin=['A:0:0.4', 'B:0.6:1'], thermo_bins=12, thermo_bootstrap=10, thermo_seed=1,
                           thermo_min_basin_ess=10, thermo_min_basin_blocks=2, no_thermo_decomposition=False, bins=30)
    warns: list = []
    info = thermo.analyze_thermo_decomposition(d, args, {'logw': logw}, kbt_kcal=0.596, out=tmp_path / 'out', warnings=warns)
    assert info['available']
    for f in ('thermo_basins.csv', 'thermo_differences.csv', 'thermo_cv_profiles.csv', 'thermo_summary.json'):
        assert (tmp_path / 'out' / 'thermo_decomposition' / f).exists(), f
    summary = json.loads((tmp_path / 'out' / 'thermo_decomposition' / 'thermo_summary.json').read_text())
    assert '_replicates' not in summary and summary['basin_source'] == 'explicit'
    assert json.dumps(info)  # JSON-serializable, merged into pmf_summary.json


def test_ladder_run_recovers_umbrella_through_compute_thermo(tmp_path):
    """End to end on a ladder run: u_nk own column = beta*(W + boost_own); dH must not see the boost."""
    x, v_pep, u_ee, U, logw, _ = _two_region_system(n=40_000, seed=5)
    n = x.size
    env = _env()
    lams = np.array([0.0, 0.5, 1.0])
    rng = np.random.default_rng(2)
    win = rng.integers(0, 3, n)
    v_pep_l = rng.uniform(-3200, -2800, n)  # boost channel inputs
    v_dih = rng.uniform(60, 140, n)
    W = 3.0 * x
    boost_own = np.zeros(n)
    for k, lam in enumerate(lams):
        sel = win == k
        boost_own[sel] = pep_gamd_boost_kj(v_pep_l[sel], v_dih[sel], lam, env)
    assert boost_own.max() > 1.0
    common = dict(cv=x, window=win, K=3, potential=U + W, v_pep=v_pep_l, v_dih=v_dih, replica=np.arange(n) % 30,
                  ladder=True, boost=boost_own)
    kw = dict(boost_type='pep-gamd-lower-dual', basins=[('A', 0, .4), ('B', .6, 1)], n_bins=10, n_boot=0, seed=0,
              min_basin_ess=10, min_basin_blocks=2)
    good = thermo.compute_thermo(_data(tmp_path, u_own_reduced=BETA * (W + boost_own), state_lambdas=lams, **common),
                                 logw, envelope_loader=lambda p: env, **kw)
    ref = thermo.compute_thermo(_data(tmp_path, u_own_reduced=BETA * W, state_lambdas=None, **common),
                                logw, envelope_loader=lambda p: None, **kw)
    assert good['available'] and good['own_umbrella']['ok']
    assert good['differences'][0]['dH_kj']['estimate'] == pytest.approx(ref['differences'][0]['dH_kj']['estimate'], abs=1e-8)
    # without the envelope the channel is refused, never silently zero-boosted
    miss = thermo.compute_thermo(_data(tmp_path, u_own_reduced=BETA * (W + boost_own), state_lambdas=lams, **common),
                                 logw, envelope_loader=lambda p: None, **kw)
    assert miss['available'] is False and 'envelope' in miss['reason']


def test_auto_basins_ignore_noise_wiggle_in_single_well():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 0.5, 20_000)  # one well; finite-sample histogram noise makes spurious local minima
    w = np.full(x.size, 1.0 / x.size)
    assert thermo.auto_basins(x, w, 60, kbt_kcal=0.596) == []


def test_all_nan_v_dih_keeps_totals_and_refuses_split(tmp_path):
    x, v_pep, u_ee, U, logw, _ = _two_region_system(n=20_000)
    n = x.size
    d = _data(tmp_path, cv=x, u_own_reduced=np.zeros(n), window=np.zeros(n, int), K=1, potential=U,
              v_pep=v_pep, v_dih=np.full(n, np.nan), replica=np.arange(n) % 20)
    res = thermo.compute_thermo(d, logw, boost_type='pep-gamd-lower-dual', basins=[('A', 0, .4), ('B', .6, 1)],
                                n_bins=10, n_boot=0, seed=0, min_basin_ess=10, min_basin_blocks=2,
                                envelope_loader=lambda p: None)
    assert res['available'] and not res['split_available'] and res['n_used'] == n
    assert np.isfinite(res['differences'][0]['dH_kj']['estimate'])


def test_basins_are_half_open_so_a_shared_edge_counts_once(tmp_path):
    n = 1000
    cv = np.concatenate([np.full(500, 0.25), np.full(500, 0.5)])  # half the samples sit exactly on the split
    d = _data(tmp_path, cv=cv, u_own_reduced=np.zeros(n), window=np.zeros(n, int), K=1, potential=np.zeros(n),
              v_pep=np.zeros(n), v_dih=np.zeros(n), replica=np.arange(n) % 10)
    res = thermo.compute_thermo(d, np.zeros(n), boost_type='pep-gamd-lower-dual',
                                basins=[('lo', 0.0, 0.5), ('hi', 0.5, 1.0)], n_bins=5, n_boot=0, seed=0,
                                min_basin_ess=1, min_basin_blocks=1, envelope_loader=lambda p: None)
    assert [b['n_samples'] for b in res['basins']] == [500, 500]


def test_mie_entropy_analytic():
    from gareus.mbar_analysis.thermo_entropy import mie_entropy, torsion_bins
    nb = 12
    theta = np.linspace(-np.pi, np.pi, 12_000, endpoint=False) + 1e-6
    b = torsion_bins(np.stack([theta, theta], axis=1), nb)
    r = mie_entropy(b, np.ones(theta.size), nb)
    assert r["S1"] == pytest.approx(2 * np.log(nb), abs=1e-9)   # two uniform torsions
    assert r["MI"] == pytest.approx(np.log(nb), abs=1e-9)       # identical -> I = H
    assert r["S2"] == pytest.approx(np.log(nb), abs=1e-9)
    # weights reshape the histogram: all weight on one bin -> zero entropy
    w = (b[:, 0] == 3).astype(float)
    assert mie_entropy(b, w, nb)["S1"] == pytest.approx(0.0, abs=1e-12)


def test_frame_channels_and_entropy_in_basin_differences(tmp_path):
    x, v_pep, u_ee, U, logw, _ = _two_region_system(n=30_000, seed=7)
    n = x.size
    rng = np.random.default_rng(8)
    # basin A: torsions uniform; basin B: torsion concentrated near 0; frames only for 2/3 of samples
    theta = np.where(x[:, None] < 0.5, rng.uniform(-np.pi, np.pi, (n, 2)), rng.normal(np.pi / 12, 0.05, (n, 2)))  # centre of one 12-bin cell
    has_frame = rng.random(n) < 2 / 3
    theta[~has_frame] = np.nan
    v_pp = np.where(has_frame, v_pep - 50.0 - 10.0 * x, np.nan)
    d = _data(tmp_path, cv=x, u_own_reduced=np.zeros(n), window=np.zeros(n, int), K=1, potential=U,
              v_pep=v_pep, v_dih=np.full(n, 100.0), replica=np.arange(n) % 20)
    res = thermo.compute_thermo(d, logw, boost_type='pep-gamd-lower-dual', basins=[('A', 0, .4), ('B', .6, 1)],
                                n_bins=10, n_boot=30, seed=1, min_basin_ess=10, min_basin_blocks=2,
                                envelope_loader=lambda p: None, extra_channels={'V_pp': v_pp}, torsions=theta,
                                entropy_bins=12)
    assert res['n_used'] == int(has_frame.sum())  # one population for every quantity
    diff = res['differences'][0]
    q = {k: v['estimate'] for k, v in diff.items() if isinstance(v, dict) and 'estimate' in v}
    assert q['dV_pe_kj'] == pytest.approx(q['dV_pep_kj'] - q['dV_pp_kj'], abs=1e-9)
    kT = 1.0 / BETA
    # A uniform -> S1 = 2 ln 12; B inside one-to-two bins -> ~0: TdS_conf ~ -2 kT ln 12
    assert q['TdS_conf_S1_kj'] == pytest.approx(-2 * kT * np.log(12), rel=0.1)
    assert q['minus_TdS_solv_kj'] == pytest.approx(q['minus_TdS_kj'] + q['TdS_conf_S2_kj'], abs=1e-9)
    reps = res['_replicates']['A->B']
    np.testing.assert_allclose(reps['minus_TdS_solv_kj'], reps['minus_TdS_kj'] + reps['TdS_conf_S2_kj'], atol=1e-9)
    assert diff['TdS_conf_S2_kj']['se'] > 0


def test_analyze_disabled(tmp_path):
    args = SimpleNamespace(no_thermo_decomposition=True)
    info = thermo.analyze_thermo_decomposition(None, args, {}, kbt_kcal=0.6, out=tmp_path, warnings=[])
    assert info == {'available': False, 'reason': 'disabled via --no-thermo-decomposition'}
