"""R2's lambda = 0 MBAR on gareus-analyze's solver, and the re-solved-f bootstrap
(``--ap-coverage-bootstrap``): agreement with the pre-swap NumPy fixed point, the (K, N) wrapper's
contract, determinism, the resolve-f bootstrap against a hand-rolled reference and against the
true sampling spread, and the knob's wiring (settings, policy, CLI, YAML, frozen fields, help).
"""
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gareus.adaptive_production as ap  # noqa: E402
from gareus.adaptive import cv2_coverage as cov  # noqa: E402
from gareus.adaptive import cv2_resolution as cr  # noqa: E402
from gareus.cli import parse_args  # noqa: E402
from gareus.swarm.ladder_design import R_KCAL_MOL_K  # noqa: E402

T = 300.0
RT = R_KCAL_MOL_K * T


def _lse(a, axis):
    top = np.max(a, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return np.squeeze(top, axis=axis) + np.log(np.sum(np.exp(a - top), axis=axis))


def _old_fixed_point(u_kn, n_k, tol=1e-13, max_iter=200000):
    """The pre-swap cv2_coverage.solve_mbar, verbatim, run to a tight tolerance."""
    log_n = np.log(np.maximum(n_k, 1e-300))
    f = np.zeros(u_kn.shape[0])
    for _ in range(max_iter):
        log_den = _lse(log_n[:, None] + f[:, None] - u_kn, axis=0)
        f_new = -_lse(-u_kn - log_den[None, :], axis=1)
        f_new -= f_new[0]
        delta = float(np.max(np.abs(f_new - f)))
        f = f_new
        if delta < tol:
            break
    return f


def _grid_union(seed=0, n=600, c1s=(0.2, 0.24), c2s=(-1.0, 0.0, 1.0), k1=800.0, k2=5.0):
    """Gaussian samples of a 2-D harmonic grid (flat landscape): (cv1, cv2, state_idx, views)."""
    rng = np.random.default_rng(seed)
    views, cv1, cv2, idx = [], [], [], []
    for c1 in c1s:
        for c2 in c2s:
            k = len(views)
            views.append(cr.StateView(k, c1, k1, c2, k2, 0.0))
            cv1.append(rng.normal(c1, math.sqrt(RT / k1), n))
            cv2.append(rng.normal(c2 + 0.1 * c1, math.sqrt(RT / k2), n))
            idx.append(np.full(n, k))
    return np.concatenate(cv1), np.concatenate(cv2), np.concatenate(idx), views


def test_gareus_analyze_solver_matches_the_old_fixed_point_to_1e8():
    cv1, cv2, idx, views = _grid_union()
    u = cov.reduced_umbrella(cv1, cv2, views, 1.0 / RT)
    n_k = np.bincount(idx).astype(float)
    ref = _old_fixed_point(u, n_k)
    info = {}
    f = cov.solve_mbar(u, n_k, info=info)
    assert info["converged"] and info["backend"] in (cov.MBAR_BACKEND, cov.MBAR_FALLBACK_BACKEND)
    assert f[0] == 0.0 and np.max(np.abs(f - ref)) < 1e-8
    # the per-row API with the real (interleaved) state index gives the same f
    perm = np.random.default_rng(1).permutation(idx.size)
    f_rows, logw = cov.solve_rows(np.ascontiguousarray(u.T[perm]), idx[perm])
    assert np.max(np.abs(f_rows - ref)) < 1e-8
    lw_old = cov.log_weights(u[:, perm], n_k, ref)
    assert np.allclose(logw - logw.max(), lw_old - lw_old.max(), atol=1e-8)
    # and the NumPy fallback backend agrees too
    res = cov._mbar.solve_mbar(np.ascontiguousarray(u.T), idx, backend=cov.MBAR_FALLBACK_BACKEND, tol=1e-10)
    assert np.max(np.abs(res["f_k"] - res["f_k"][0] - ref)) < 1e-8


def test_solver_is_deterministic():
    cv1, cv2, idx, views = _grid_union(seed=3)
    u = cov.reduced_umbrella(cv1, cv2, views, 1.0 / RT)
    n_k = np.bincount(idx).astype(float)
    assert np.array_equal(cov.solve_mbar(u, n_k), cov.solve_mbar(u, n_k))


def test_wrapper_rejects_counts_that_do_not_describe_the_rows():
    u = np.zeros((2, 10))
    for bad in (np.array([5.0, 4.0]), np.array([5.5, 4.5]), np.array([10.0, 0.0]), np.array([10.0])):
        with pytest.raises(ValueError, match="n_k"):
            cov.solve_mbar(u, bad)


def test_warm_start_converges_to_the_same_f_in_fewer_iterations():
    cv1, cv2, idx, views = _grid_union(seed=4)
    u_nk = np.ascontiguousarray(cov.reduced_umbrella(cv1, cv2, views, 1.0 / RT).T)
    cold, warm = {}, {}
    f, _ = cov.solve_rows(u_nk, idx, info=cold)
    f2, _ = cov.solve_rows(u_nk, idx, f_init=f + 1e-3, info=warm)
    assert np.max(np.abs(f - f2)) < 1e-8 and warm["iterations"] < cold["iterations"]


# ---- resolve-f bootstrap ----------------------------------------------------------------

def _one_column(views, cv1, cv2, w, settings):
    cols = [cov._extend_to_weight(c, cv1, cv2, w) for c in cov.columns({v.state_id: v for v in views},
                                                                        [v.state_id for v in views], settings)]
    return cols


def test_resolve_f_sigma_equals_a_hand_rolled_reference():
    cv1, cv2, idx, views = _grid_union(seed=5, n=400)
    st = cr.ResolutionSettings(temperature_k=T)
    u_nk = np.ascontiguousarray(cov.reduced_umbrella(cv1, cv2, views, 1.0 / RT).T)
    f, lw = cov.solve_rows(u_nk, idx)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    cols = _one_column(views, cv1, cv2, w, st)
    data = [cov.column_bins(c, cv1, cv2) for c in cols]
    blocks = cov._block_ids(idx, 10)
    got = cov.resolve_f_sigma(u_nk, idx, blocks, data, f, np.random.default_rng(9), n_boot=30)
    # reference: same draws, the OLD NumPy fixed point per replicate
    rng = np.random.default_rng(9)
    uniq = np.unique(blocks)
    sob = uniq // 10 ** 7
    strata = [np.flatnonzero(sob == k) for k in np.unique(sob)]
    reps = [[] for _ in data]
    for _ in range(30):
        picked = np.concatenate([rng.choice(s, size=s.size) for s in strata])
        pick = np.concatenate([np.flatnonzero(blocks == uniq[b]) for b in picked])
        nk = np.bincount(idx[pick], minlength=len(views)).astype(float)
        fr = _old_fixed_point(u_nk[pick].T, nk)
        lwr = -_lse(np.log(nk)[:, None] + fr[:, None] - u_nk[pick].T, axis=0)
        wr = np.exp(lwr - lwr.max())
        for c, (slab, bins, inside, n_int) in enumerate(data):
            part = np.bincount(bins[pick][inside[pick]], weights=wr[inside[pick]], minlength=n_int)
            with np.errstate(divide="ignore"):
                reps[c].append(-np.log(part / wr[slab[pick]].sum()))
    for g, r in zip(got, reps):
        r = np.asarray(r)
        fin = np.isfinite(r)
        with np.errstate(invalid="ignore"):
            ref = np.where(fin.mean(axis=0) >= 0.9, np.nanstd(np.where(fin, r, np.nan), axis=0), np.inf)
        both = np.isfinite(ref)
        assert np.array_equal(np.isfinite(g), both)
        assert np.max(np.abs(g[both] - ref[both])) < 1e-6


def test_resolve_f_tracks_the_true_spread_where_fixed_f_does_not():
    """iid samples of two windows (spacing ~2.6 sampled sd) on a harmonic landscape, one column:
    an interval's F depends on the two windows' relative f, which the fixed-f bootstrap never
    resamples. Over 30 independent data sets the resolve-f sigma matches the empirical spread of
    F in the heavy intervals; the fixed-f sigma reads low."""
    st = cr.ResolutionSettings(temperature_k=T)
    k2, big_k = 20.0, 20.0                       # umbrella k2 on an underlying 0.5 * big_k * z^2
    views = [cr.StateView(k, 0.2, 800.0, c2, k2, 0.0) for k, c2 in enumerate((-0.4, 0.4))]
    sd = math.sqrt(RT / (k2 + big_k))
    col = cov.columns({v.state_id: v for v in views}, [0, 1], st)[0]
    est, s_fix, s_res = [], [], []
    for seed in range(30):
        rng = np.random.default_rng(100 + seed)
        cv2 = np.concatenate([rng.normal(v.c2 * k2 / (k2 + big_k), sd, 300) for v in views])
        cv1 = np.full(cv2.size, 0.2)
        idx = np.repeat([0, 1], 300)
        u_nk = np.ascontiguousarray(cov.reduced_umbrella(cv1, cv2, views, 1.0 / RT).T)
        f, lw = cov.solve_rows(u_nk, idx)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        slab, bins, inside, n_int = data = cov.column_bins(col, cv1, cv2)
        part = np.bincount(bins[inside], weights=w[inside], minlength=n_int)
        with np.errstate(divide="ignore"):
            est.append(-np.log(part / w[slab].sum()))
        blocks = cov._block_ids(idx, 20)
        s_res.append(cov.resolve_f_sigma(u_nk, idx, blocks, [data], f, np.random.default_rng(seed), n_boot=60)[0])
        w_int = np.stack([np.where(inside & (bins == i), w, 0.0) for i in range(n_int)])
        s_fix.append(cov._boot_sigma(w_int, np.where(slab, w, 0.0), blocks, np.random.default_rng(seed)))
    est, s_res, s_fix = np.asarray(est), np.asarray(s_res), np.asarray(s_fix)
    heavy = np.exp(-est).mean(axis=0) >= 0.05
    true = est[:, heavy].std(axis=0, ddof=1)
    r_res = np.median(s_res[:, heavy], axis=0) / true
    r_fix = np.median(s_fix[:, heavy], axis=0) / true
    assert heavy.sum() >= 4 and 0.75 < float(np.median(r_res)) < 1.3
    assert float(np.median(r_fix)) < 0.8 and np.all(r_fix <= r_res + 0.05)


def test_coverage_holes_records_the_mode_and_fixed_f_is_the_old_path():
    cv1, cv2, idx, views = _grid_union(seed=6, n=500, c1s=(0.2,), c2s=(-1.0, 0.0))
    u_nk = np.ascontiguousarray(cov.reduced_umbrella(cv1, cv2, views, 1.0 / RT).T)
    f, lw = cov.solve_rows(u_nk, idx)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    vmap = {v.state_id: v for v in views}
    base = dict(temperature_k=T, coverage_count="any", refine_pmf_sigma_kT=0.5)
    info_a, info_b, info_r = {}, {}, {}
    a = cov.coverage_holes(cv1, cv2, idx, w, views, vmap, cr.ResolutionSettings(**base), info=info_a)
    b = cov.coverage_holes(cv1, cv2, idx, w, views, vmap, cr.ResolutionSettings(**base, coverage_bootstrap="fixed-f"),
                           info=info_b)
    assert a == b and info_a["bootstrap"]["mode"] == "fixed-f" and "resolve_f" not in info_a["bootstrap"]
    r = cov.coverage_holes(cv1, cv2, idx, w, views, vmap,
                           cr.ResolutionSettings(**base, coverage_bootstrap="resolve-f"), info=info_r)
    rec = info_r["bootstrap"]["resolve_f"]
    assert info_r["bootstrap"]["mode"] == "resolve-f" and rec["n_replicates"] == cov.N_BOOT
    assert rec["n_not_converged"] == 0 and rec["iterations_q50_max"][1] >= 1
    # u_nk / f_point passed in give the identical answer to recomputing them
    r2 = cov.coverage_holes(cv1, cv2, idx, w, views, vmap,
                            cr.ResolutionSettings(**base, coverage_bootstrap="resolve-f"), u_nk=u_nk, f_point=f)
    assert r == r2


# ---- the knob ---------------------------------------------------------------------------

def test_coverage_bootstrap_knob_is_mirrored_validated_and_frozen():
    assert ap.COVERAGE_BOOTSTRAPS == cr.COVERAGE_BOOTSTRAPS
    assert "coverage_bootstrap" in ap.DECISION_SETTINGS_FIELDS and "coverage_bootstrap" in cr.DEFAULTS
    assert ap.AdaptiveDecisionPolicy().coverage_bootstrap == cr.DEFAULTS["coverage_bootstrap"]
    assert ap.policy_from_args(Namespace()).coverage_bootstrap == cr.DEFAULTS["coverage_bootstrap"]
    assert cr.ResolutionSettings.from_policy(ap.AdaptiveDecisionPolicy(coverage_bootstrap="resolve-f")
                                             ).coverage_bootstrap == "resolve-f"
    with pytest.raises(ValueError, match="coverage_bootstrap"):
        ap.AdaptiveDecisionPolicy(coverage_bootstrap="bogus")
    with pytest.raises(ValueError, match="coverage_bootstrap"):
        cr.ResolutionSettings(coverage_bootstrap="bogus")
    with pytest.raises(ValueError, match="coverage_bootstrap"):
        ap.policy_from_args(Namespace(adaptive_production_coverage_bootstrap="bogus"))


def test_coverage_bootstrap_cli_yaml_and_help(tmp_path):
    other = "fixed-f" if cr.DEFAULTS["coverage_bootstrap"] == "resolve-f" else "resolve-f"
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "r"), "--ap-coverage-bootstrap", other])
    assert ap.policy_from_args(args).coverage_bootstrap == other
    assert ap.policy_from_args(parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path / "d")])
                               ).coverage_bootstrap == cr.DEFAULTS["coverage_bootstrap"]
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"adaptive_production:\n  ap_coverage_bootstrap: {other}\n")
    p = ap.policy_from_args(parse_args(["--config", str(cfg), "--seq", "GYDPETGTWG", "--out", str(tmp_path / "y")]))
    assert p.coverage_bootstrap == other
    from gareus.helptext import _METHOD_ENCYCLOPEDIA
    assert "--ap-coverage-bootstrap" in _METHOD_ENCYCLOPEDIA


def test_a_record_without_the_key_takes_this_jobs_value(tmp_path):
    import json
    policy = ap.AdaptiveDecisionPolicy(coverage_bootstrap="resolve-f")
    rec = {f: getattr(policy, f) for f in ap.DECISION_SETTINGS_FIELDS if f != "coverage_bootstrap"}
    (tmp_path / ap.DECISION_SETTINGS_FILENAME).write_text(json.dumps({"settings": rec}))
    resolved, record = ap._resolve_decision_settings(tmp_path, policy)
    assert resolved.coverage_bootstrap == "resolve-f" and record["settings"]["coverage_bootstrap"] == "resolve-f"
    resolved, _ = ap._resolve_decision_settings(tmp_path, ap.AdaptiveDecisionPolicy(coverage_bootstrap="fixed-f"))
    assert resolved.coverage_bootstrap == "resolve-f"        # frozen once recorded
