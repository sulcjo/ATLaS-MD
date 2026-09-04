"""Regression tests for two related MBAR-reweighting bugs found by a
physics/math correctness audit and confirmed against real production data.

Bug 1 -- masked-global-logw subset PMF carries a systematic tilt
------------------------------------------------------------------
``run_pmf_and_gamd_boost_report`` and ``run_secondary_cv_analyses`` both used
to build a subset's per-sample weights by slicing the GLOBAL MBAR ``logw``
to the subset's mask and renormalizing (``norm_logw(logw[mask])``). That only
corrects for the subset's overall size -- it silently keeps using the
GLOBAL per-state sample counts N_k inside every sample's MBAR
self-consistency denominator, which is wrong whenever different states lose
different *fractions* of their samples to the split (e.g. the epoch_000/rest
split, or a secondary-CV regime split). This produces a real,
direction-consistent tilt in the resulting PMF (confirmed on real data: a
~0.11 kcal/mol shift in ``pmf_span_kcal_mol`` on a ~13 kcal/mol span).

The fix, ``_subset_logw_from_global_fk``, reuses the already-solved GLOBAL
free energies ``f_k`` (a property of the whole population, not the subset)
but recomputes the denominator with the SUBSET's own per-state N_k:

    logw_S[n] = -logsumexp_k( log(N_k^subset[k]) + f_k[k] - u_nk[n, k] )

Bug 2 -- convergence checks compare mismatched populations
------------------------------------------------------------
When the epoch_000/rest split is active, ``analyze()``'s reference PMF
(``sel``) is built from ``d_main`` (epoch_000 excluded). The convergence
checks (``run_pmf_convergence`` / ``run_epoch_pmf_convergence``) used to be
run against the FULL ``d`` (epoch_000 included) instead -- so even the
100%-of-data checkpoint could never match a reference built from a smaller,
different population, forcing a spurious ``converged=False`` purely from
population mismatch. The fix passes ``d_main`` consistently to both the
reference-PMF build and the convergence checks.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

import analyze_gareus_mbar as A
from analyze_gareus_mbar import (
    Data,
    KJ_PER_KCAL,
    _epoch_zero_split_masks,
    _masked_data,
    _subset_logw_from_global_fk,
    make_bins,
    norm_logw,
    pmf_from_weights,
    run_pmf_and_gamd_boost_report,
    run_pmf_convergence,
    run_secondary_cv_analyses,
    solve_mbar,
)


# --- shared synthetic-data builder -------------------------------------------

def _harmonic_windows_data(rng, centers, k_spring, beta, counts_per_window,
                            step_offset=0, prod_dir=Path("."), boost=None):
    """Build a real multi-window harmonic-umbrella Data.

    Samples for window k are drawn directly from its own biased Gaussian
    (mean=centers[k], var=1/(beta*k_spring[k])), and u_nk[n, k] is the exact
    analytic reduced harmonic bias -- a real, non-degenerate MBAR problem
    with a known-good self-consistent solution, not an all-zero-bias stub.
    """
    K = len(centers)
    cv_parts, win_parts = [], []
    for k in range(K):
        n = counts_per_window[k]
        sigma = 1.0 / np.sqrt(beta * k_spring[k])
        cv_parts.append(rng.normal(centers[k], sigma, n))
        win_parts.append(np.full(n, k, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(win_parts)
    N = cv.size
    u_nk = beta * 0.5 * np.asarray(k_spring)[None, :] * (cv[:, None] - np.asarray(centers)[None, :]) ** 2
    boost_kj = np.zeros(N) if boost is None else np.asarray(boost, dtype=np.float64)
    step = np.arange(step_offset, step_offset + N, dtype=np.int64)
    d = Data(
        prod_dir=prod_dir, out_dir=prod_dir / "out",
        cv=cv, cv2=np.full(N, np.nan), rg_A=np.full(N, np.nan),
        window=window, replica=window.copy(), step=step,
        u_nk=u_nk, centers=np.asarray(centers, float), k_kcal=np.asarray(k_spring, float) / KJ_PER_KCAL,
        beta=beta, temp=1.0 / (A.K_B_KJ_PER_MOL_K * beta), boost_kj=boost_kj, potential_kj=None,
        source="synthetic", meta={},
    )
    return d


def _pmf_error(pmf_a: dict, pmf_ref: dict, min_ref_count: int = 5) -> float:
    """Mean absolute PMF difference over bins well-populated in the reference."""
    fa = np.asarray(pmf_a["pmf"], dtype=float)
    fr = np.asarray(pmf_ref["pmf"], dtype=float)
    mask = np.isfinite(fa) & np.isfinite(fr) & (np.asarray(pmf_ref["counts"]) > min_ref_count)
    assert np.any(mask), "no comparable bins between the two PMFs"
    return float(np.mean(np.abs(fa[mask] - fr[mask])))


# --- Bug 1, requirement 1: corrected subset logw beats naive mask+renorm ----

def test_corrected_subset_logw_closer_to_true_subset_resolve_than_naive_mask():
    """Construct a subset that excludes a disproportionate share of one
    window's samples relative to the others (window 0: keep 10%; windows 1,
    2: keep 90%). The corrected reweight (_subset_logw_from_global_fk, reusing
    the global f_k with the subset's own N_k) must land substantially closer
    to an independent from-scratch MBAR resolve of the subset alone than the
    old naive mask-and-renormalize approach.
    """
    rng = np.random.default_rng(42)
    centers = [0.0, 1.5, 3.0]
    k_spring = [40.0, 40.0, 40.0]
    beta = 0.4
    d = _harmonic_windows_data(rng, centers, k_spring, beta, counts_per_window=[2000, 2000, 2000])

    m_global = solve_mbar(d.u_nk, d.window, backend="numpy", tol=1e-12, maxiter=20000)
    assert m_global["converged"]

    keep_frac = [0.1, 0.9, 0.9]
    mask_subset = np.zeros(d.cv.size, dtype=bool)
    for k, frac in enumerate(keep_frac):
        idx = np.where(d.window == k)[0]
        mask_subset[idx] = rng.random(idx.size) < frac
    counts = np.bincount(d.window[mask_subset], minlength=3)
    # Sanity: the exclusion really is disproportionate across windows.
    assert counts[0] < 0.3 * counts[1]
    assert counts[0] < 0.3 * counts[2]

    d_subset = _masked_data(d, mask_subset)

    naive_logw = m_global["logw"][mask_subset]  # the old buggy behavior
    corrected_logw = _subset_logw_from_global_fk(d_subset, m_global["f_k"])

    m_true_subset = solve_mbar(d_subset.u_nk, d_subset.window, backend="numpy", tol=1e-12, maxiter=20000)
    assert m_true_subset["converged"]

    bins = make_bins(d.cv, 30, float(d.cv.min()), float(d.cv.max()))
    kbt_kcal = (1.0 / beta) / KJ_PER_KCAL

    pmf_naive = pmf_from_weights(d_subset.cv, norm_logw(naive_logw), bins, kbt_kcal)
    pmf_corrected = pmf_from_weights(d_subset.cv, norm_logw(corrected_logw), bins, kbt_kcal)
    pmf_true = pmf_from_weights(d_subset.cv, norm_logw(m_true_subset["logw"]), bins, kbt_kcal)

    err_naive = _pmf_error(pmf_naive, pmf_true)
    err_corrected = _pmf_error(pmf_corrected, pmf_true)

    assert err_naive > 0.15, f"expected the naive approach to show a real tilt, got {err_naive:.4f} kcal/mol"
    assert err_corrected < 0.5 * err_naive, (
        f"corrected error ({err_corrected:.4f}) should be well under half the naive "
        f"error ({err_naive:.4f}) vs the true from-scratch subset resolve"
    )


# --- Bug 1, requirement 2: degenerate case (subset == full population) -----

def test_corrected_subset_logw_matches_global_logw_when_subset_is_everything():
    rng = np.random.default_rng(7)
    centers = [0.0, 2.0]
    k_spring = [30.0, 30.0]
    beta = 0.5
    d = _harmonic_windows_data(rng, centers, k_spring, beta, counts_per_window=[500, 500])

    m_global = solve_mbar(d.u_nk, d.window, backend="numpy", tol=1e-12, maxiter=20000)

    full_mask = np.ones(d.cv.size, dtype=bool)
    d_full = _masked_data(d, full_mask)
    recomputed = _subset_logw_from_global_fk(d_full, m_global["f_k"])

    np.testing.assert_allclose(recomputed, m_global["logw"], atol=1e-9)


# --- Bug 1 (SUPERSEDED): run_secondary_cv_analyses used to reweight each
# regime out of the pooled f_k (_subset_logw_from_global_fk). That fixed the
# binning axis but not the estimator: across a CV2 regime change the pooled
# solve splices two Hamiltonians into every column, so its f_k is not the
# whole-population property that reweight assumes. Each regime now gets its
# own independent solve and f_k_global is deliberately unused. The tests
# below assert the replacement behaviour.
#
# The _subset_logw_from_global_fk tests above remain valid: they exercise the
# function itself on a subset of ONE Hamiltonian, which is what it is for.
# ---------------------------------------------------------------------------

def _two_regime_data(rng, centers, k_spring, beta, counts_regime0, counts_regime1, tmp_path,
                      centers_regime1=None):
    """Two epochs with different ``secondary_cv`` modes in their manifests.

    ``centers_regime1`` makes the split a REAL regime change rather than only
    a metadata one: the same window ids get different restraint centres after
    the switch, so the pooled u_nk splices two Hamiltonians into every column
    -- which is the situation on the motivating run (shared states keep their
    primary params but every secondary centre moves). Left as None, both
    regimes share one Hamiltonian and the pooled f_k is legitimately valid.
    """
    d0 = _harmonic_windows_data(rng, centers, k_spring, beta, counts_regime0, step_offset=0)
    d1 = _harmonic_windows_data(rng, centers if centers_regime1 is None else centers_regime1,
                                 k_spring, beta, counts_regime1, step_offset=d0.cv.size)
    N0, N1 = d0.cv.size, d1.cv.size
    cv = np.concatenate([d0.cv, d1.cv])
    cv2 = rng.normal(0, 1, N0 + N1)
    window = np.concatenate([d0.window, d1.window])
    step = np.concatenate([d0.step, d1.step])
    u_nk = np.concatenate([d0.u_nk, d1.u_nk], axis=0)
    boost_kj = np.zeros(N0 + N1)

    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text('{"resolved_args": {"secondary_cv": "torsion-pca"}}')
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    (e1 / "run_manifest.json").write_text('{"resolved_args": {"secondary_cv": "tica-linear"}}')

    meta = {
        "_epoch_source": [0] * N0 + [1] * N1,
        "adaptive_epoch_run_dirs": [str(e0), str(e1)],
    }
    d = Data(
        prod_dir=tmp_path, out_dir=tmp_path / "out",
        cv=cv, cv2=cv2, rg_A=np.full(N0 + N1, np.nan),
        window=window, replica=window.copy(), step=step,
        u_nk=u_nk, centers=np.asarray(centers, float), k_kcal=np.asarray(k_spring, float) / KJ_PER_KCAL,
        beta=beta, temp=1.0 / (A.K_B_KJ_PER_MOL_K * beta), boost_kj=boost_kj, potential_kj=None,
        source="synthetic", meta=meta,
    )
    return d


def _test_args():
    from analyze_gareus_mbar import parse_args
    args = parse_args(["dummy_input", "--bins", "16", "--no-convergence"])
    args.no_poincare_map = True
    return args


def _capture_regime_calls(d, args, base_logw, out, **kw):
    """Run run_secondary_cv_analyses, capturing (d_regime, logw) per regime."""
    import gareus.mbar_analysis.pmf as pmfmod
    real_analyze = pmfmod.analyze_secondary_cv_pmf
    seen: list[tuple] = []

    def _spy(d_regime, args_, logw_arg, *a, **kwargs):
        seen.append((d_regime, np.array(logw_arg, dtype=np.float64)))
        return real_analyze(d_regime, args_, logw_arg, *a, **kwargs)

    pmfmod.analyze_secondary_cv_pmf = _spy
    try:
        result = run_secondary_cv_analyses(d, args, base_logw, "umbrella_only", False, 0.6,
                                            out, kw.pop('warnings', []), None, **kw)
    finally:
        pmfmod.analyze_secondary_cv_pmf = real_analyze
    return seen, result


def test_each_regime_logw_is_its_own_independent_solve(tmp_path):
    """Each regime's weights must come from a solve over that regime's rows
    alone -- not from the pooled f_k, whose state free energies are not a
    single Hamiltonian's property once the CV2 definition changes."""
    rng = np.random.default_rng(11)
    centers = [0.0, 3.0]
    k_spring = [45.0, 45.0]
    beta = 0.4
    # A genuine regime change: the same window ids are restrained to different
    # centres after the switch, so the pooled u_nk is two spliced Hamiltonians.
    d = _two_regime_data(rng, centers, k_spring, beta,
                          counts_regime0=[350, 50], counts_regime1=[200, 200], tmp_path=tmp_path,
                          centers_regime1=[1.5, 4.5])

    m_global = solve_mbar(d.u_nk, d.window, backend="numpy", tol=1e-12, maxiter=20000)
    assert m_global["converged"]

    seen, _ = _capture_regime_calls(d, _test_args(), m_global["logw"], tmp_path / "out_a")
    assert len(seen) == 2

    for d_regime, logw_used in seen:
        own = solve_mbar(d_regime.u_nk, d_regime.window, backend="auto",
                         tol=float(getattr(_test_args(), 'mbar_tol', 1e-10)))
        assert own["converged"]
        np.testing.assert_allclose(logw_used, own["logw"], rtol=0, atol=1e-9)

    # Deliberately NOT asserted here: that this differs numerically from the
    # pooled-f_k reweight. logw is invariant to f_k + const, so on a two-window
    # fixture there is a single free difference f_1 - f_0 and the pooled value
    # can legitimately coincide with each regime's own. How far the pooled f_k
    # is distorted is a property of the run's data, not of this code, so
    # asserting a gap here would only measure how hard the fixture was tuned
    # to produce one. The invariant that defines this change is the equality
    # above; f_k_global's irrelevance is pinned separately below.


def test_f_k_global_no_longer_influences_regime_logw(tmp_path):
    """f_k_global is retained in the signature but deliberately unused: there
    is no correct way to derive a regime's weights from a cross-regime solve."""
    rng = np.random.default_rng(11)
    centers = [0.0, 3.0]
    k_spring = [45.0, 45.0]
    beta = 0.4
    d = _two_regime_data(rng, centers, k_spring, beta,
                          counts_regime0=[350, 50], counts_regime1=[200, 200], tmp_path=tmp_path)
    m_global = solve_mbar(d.u_nk, d.window, backend="numpy", tol=1e-12, maxiter=20000)

    without, _ = _capture_regime_calls(d, _test_args(), m_global["logw"], tmp_path / "out_b")
    with_fk, _ = _capture_regime_calls(d, _test_args(), m_global["logw"], tmp_path / "out_c",
                                        f_k_global=m_global["f_k"])
    assert len(without) == len(with_fk) == 2
    for (_, a), (_, b) in zip(without, with_fk):
        np.testing.assert_array_equal(a, b)


def test_regime_provenance_is_recorded(tmp_path):
    rng = np.random.default_rng(11)
    d = _two_regime_data(rng, [0.0, 3.0], [45.0, 45.0], 0.4,
                          counts_regime0=[350, 50], counts_regime1=[200, 200], tmp_path=tmp_path)
    m_global = solve_mbar(d.u_nk, d.window, backend="numpy", tol=1e-12, maxiter=20000)
    _, (pmf_info, _fes_info) = _capture_regime_calls(d, _test_args(), m_global["logw"],
                                                      tmp_path / "out_d")
    assert pmf_info.get('secondary_cv_logw_source') == 'per_regime_solve'
    breakdown = pmf_info.get('regime_breakdown') or {}
    assert breakdown, "expected a regime breakdown for a two-regime run"
    for regime, entry in breakdown.items():
        assert entry['secondary_cv_logw_source'] == 'per_regime_solve', regime


def test_non_converging_regime_emits_nothing_rather_than_pooled_fk(tmp_path, monkeypatch):
    """A regime is a small minority of samples, so its own solve can fail to
    converge. That must yield an ABSENT result, never a pooled-f_k fallback:
    a silently wrong curve is worse than a missing one."""
    import gareus.mbar_analysis.pmf as pmfmod
    rng = np.random.default_rng(11)
    d = _two_regime_data(rng, [0.0, 3.0], [45.0, 45.0], 0.4,
                          counts_regime0=[350, 50], counts_regime1=[200, 200], tmp_path=tmp_path)
    m_global = solve_mbar(d.u_nk, d.window, backend="numpy", tol=1e-12, maxiter=20000)

    _agm = pmfmod._bridge()
    real_solve = _agm.solve_mbar

    def _never_converges(u_nk, window, **kw):
        out = dict(real_solve(u_nk, window, **kw))
        out['converged'] = False
        out['max_delta'] = 1.234e-3
        return out

    monkeypatch.setattr(_agm, 'solve_mbar', _never_converges)
    warnings: list[str] = []
    seen, (pmf_info, fes_info) = _capture_regime_calls(
        d, _test_args(), m_global["logw"], tmp_path / "out_e", warnings=warnings)

    # analyze_secondary_cv_pmf must never be reached for a regime with no estimator.
    assert seen == [], "a regime with no usable estimator still produced a PMF"
    assert pmf_info.get('available') is False
    assert str(pmf_info.get('secondary_cv_logw_source', '')).startswith('unavailable:not_converged')
    assert fes_info.get('available') is False
    assert any('refusing to reweight' in w for w in warnings), warnings


# --- Bug 2: convergence checks must use the same population as the reference

def test_run_pmf_convergence_top_checkpoint_matches_reference_population(tmp_path):
    """When epoch_000 is split out, run_pmf_convergence must be called with
    d_main (the same population the reference PMF `sel` was built from) --
    not the full d. This regression-checks both directions: the FIXED call
    (d_main) reaches near-zero JS/RMSE at its own 100% checkpoint, while the
    OLD buggy call (full d) spuriously fails there purely because its
    checkpoint population (epoch_000 + rest) never matches the population
    the reference was built from (rest only).

    Windows use DELIBERATELY ASYMMETRIC spring constants (55 vs 25) so the
    true relative free energy Delta-f is genuinely non-zero and estimating it
    is real work, not a coincidental Delta-f=0 degeneracy that any subset
    would recover trivially -- confirmed below via the checkpoint's own
    mbar_iterations (the fresh top-checkpoint resolve of d_main actually runs
    to the iteration cap, it is not a 1-shot warm-start bypass).
    """
    rng = np.random.default_rng(7)
    centers = [0.0, 2.2]
    k_spring = [55.0, 25.0]
    beta = 0.4
    # epoch_000: heavily window-0. rest: balanced. A real, disproportionate
    # per-state exclusion, same shape as the real chignolin_5 finding.
    d0 = _harmonic_windows_data(rng, centers, k_spring, beta, counts_per_window=[350, 50],
                                 step_offset=0, prod_dir=tmp_path)
    d_rest = _harmonic_windows_data(rng, centers, k_spring, beta, counts_per_window=[200, 200],
                                     step_offset=d0.cv.size, prod_dir=tmp_path)
    N0, Nr = d0.cv.size, d_rest.cv.size
    cv = np.concatenate([d0.cv, d_rest.cv])
    window = np.concatenate([d0.window, d_rest.window])
    step = np.concatenate([d0.step, d_rest.step])
    u_nk = np.concatenate([d0.u_nk, d_rest.u_nk], axis=0)
    boost_kj = np.zeros(N0 + Nr)
    meta = {
        "_epoch_source": [0] * N0 + [1] * Nr,
        "adaptive_epoch_run_dirs": ["/x/epoch_000", "/x/epoch_001/baseline"],
        "timestep_fs": 4.0,
    }
    d = Data(
        prod_dir=tmp_path, out_dir=tmp_path / "out",
        cv=cv, cv2=np.full(N0 + Nr, np.nan), rg_A=np.full(N0 + Nr, np.nan),
        window=window, replica=window.copy(), step=step,
        u_nk=u_nk, centers=np.asarray(centers, float), k_kcal=np.asarray(k_spring, float) / KJ_PER_KCAL,
        beta=beta, temp=1.0 / (A.K_B_KJ_PER_MOL_K * beta), boost_kj=boost_kj, potential_kj=None,
        source="synthetic", meta=meta,
    )

    m_global = solve_mbar(d.u_nk, d.window, backend="numpy", tol=1e-12, maxiter=20000)
    split = _epoch_zero_split_masks(d)
    assert split is not None
    _mask0, mask_rest = split
    d_main = _masked_data(d, mask_rest)
    logw_main = _subset_logw_from_global_fk(d_main, m_global["f_k"])

    # Sanity: the true Delta-f is genuinely non-zero and genuinely differs
    # between the global fit and a from-scratch fit of d_main alone -- this
    # is a real (if small, given modest N) estimation problem, not a trivial
    # symmetric-window degeneracy any subset would recover identically.
    m_true_main = solve_mbar(d_main.u_nk, d_main.window, backend="numpy", tol=1e-12, maxiter=20000)
    assert abs(m_global["f_k"][1] - m_true_main["f_k"][1]) > 1e-4

    bins = make_bins(d.cv, 20, float(d.cv.min()), float(d.cv.max()))
    kbt_kcal = (1.0 / beta) / KJ_PER_KCAL

    args = SimpleNamespace(
        bins=20, min_neighbor_overlap=0.0, selected_method="auto",
        gamd_smooth_sigma=0.0, pmf_smooth_sigma=0.0, smooth_sigma=0.0,
        no_convergence=False, convergence_timepoints=8,
        convergence_mbar_backend="numpy", convergence_mbar_tol=1e-8,
        convergence_mbar_maxiter=5000, mbar_threads=0,
        no_convergence_mbar_cache=False,
        convergence_js_threshold=0.01, convergence_rmse_threshold=0.10,
        convergence_dir="convergence", no_basin_tracking=True,
    )

    main_info = run_pmf_and_gamd_boost_report(d_main, args, logw_main, bins, kbt_kcal,
                                               tmp_path / "main_report", [], None)
    selected = main_info["selected"]
    sel = main_info["pmfs"][selected]

    fixed_out = tmp_path / "fixed_conv"
    buggy_out = tmp_path / "buggy_conv"
    fixed_res = run_pmf_convergence(d_main, args, bins, selected, sel, fixed_out,
                                     progress=None, f_init_hint=m_global.get("f_k"))
    buggy_res = run_pmf_convergence(d, args, bins, selected, sel, buggy_out,
                                     progress=None, f_init_hint=m_global.get("f_k"))

    fixed_summary = fixed_res["summary"]
    buggy_summary = buggy_res["summary"]

    # Confirm the FIXED top checkpoint really did an independent, substantive
    # fresh MBAR resolve of d_main (not a trivial 0/1-iteration bypass from
    # the warm start) -- addresses whether the near-match below is genuine.
    import csv as _csv
    fixed_rows = list(_csv.DictReader(open(fixed_out / "convergence" / "total_pmf_convergence.csv")))
    last_fixed_row = fixed_rows[-1]
    assert float(last_fixed_row["frac_total"]) == 1.0
    assert int(last_fixed_row["n_samples"]) == d_main.cv.size
    assert int(last_fixed_row["mbar_iterations"]) > 100

    # FIXED: the top checkpoint (100% of d_main) IS d_main's own population,
    # landing very close to the reference (small residual only from sel
    # using the approximate global-f_k reweight vs an independently-solved
    # fresh resolve of the exact same samples).
    assert fixed_summary["last_eval_JS"] < 1e-3
    assert fixed_summary["last_eval_RMSE_F_kcal_mol"] < 0.02

    # BUGGY: the top checkpoint of the full d always includes epoch_000's
    # disproportionately window-0-heavy samples, which the reference (built
    # from the balanced d_main alone) never saw -- a real, spurious mismatch,
    # not a genuine convergence failure. Confirm it actually breaches the
    # user-configured RMSE threshold outright, and by a wide margin over fixed.
    assert buggy_summary["last_eval_RMSE_F_kcal_mol"] > args.convergence_rmse_threshold
    assert buggy_summary["last_eval_RMSE_F_kcal_mol"] > 10 * fixed_summary["last_eval_RMSE_F_kcal_mol"]
