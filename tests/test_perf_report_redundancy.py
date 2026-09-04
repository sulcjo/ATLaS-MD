"""Regression tests for three small, verified-safe performance fixes in
analyze_gareus_mbar.py, found by a performance audit. Every fix is required
to produce identical (or, where explicitly noted below, numerically
equivalent but not bit-for-bit identical) output to the pre-fix code -- these
tests exist to prove that, not just that the code runs.

1. `run_pmf_and_gamd_boost_report`'s new `precomputed_base_w` parameter lets
   `analyze()` skip a redundant `norm_logw(logw)` recompute in the common
   no-split case, where it already computed the identical value for the same
   full-population `logw`. The genuinely-different-population (epoch_000/
   rest split) case must still always recompute internally.

2. `_window_cv_mean_std` (extracted from window_diagnostics.csv's writer in
   run_pmf_and_gamd_boost_report) replaces an O(N*K) Python loop that rebuilt
   a fresh boolean mask (`cv[window == k]`) once per window with a single
   vectorized bincount-based reduction. This one is confirmed NOT bit-
   identical to the original np.mean/np.std loop (see the test below for
   why) -- verified numerically equivalent instead, and documented as such
   here and in the commit message per the task's own guidance.

3. `_get_checkpoint_steps_cache`/`_checkpoint_steps_cache_key` memoize
   `checkpoint_steps_from_data`'s np.unique(step) call (an O(N log N) sort)
   across the ~20+ observable-convergence calls in one analyze() run that
   share the same production `d` and the same finite-sample mask -- confirmed
   by tracing analyze_extra_observable_pmfs: phi/psi/SASA/contacts/secondary-
   structure fractions are all filled from the same per-chunk `sample_idx`,
   so their NaN patterns coincide. The cache is keyed on mask *content*
   (blake2b digest, same idiom as the existing `_convergence_mbar_cache`),
   not object identity, since every call site filters a fresh boolean-indexed
   copy of `step` -- a naive `id(step)`-keyed cache would never hit.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

import analyze_gareus_mbar as A
from analyze_gareus_mbar import (
    Data,
    KJ_PER_KCAL,
    _checkpoint_steps_cache_key,
    _get_checkpoint_steps_cache,
    _window_cv_mean_std,
    checkpoint_steps_from_data,
    norm_logw,
    parse_args,
    run_observable_pmf_convergence,
    run_pmf_and_gamd_boost_report,
)


def _test_args():
    args = parse_args(["dummy_input", "--bins", "20", "--no-convergence"])
    args.no_poincare_map = True
    return args


def _full_data(cv, boost, k=3, window=None, prod_dir=None):
    n = len(cv)
    window = np.zeros(n, dtype=np.int32) if window is None else np.asarray(window, dtype=np.int32)
    return Data(
        prod_dir=prod_dir or Path("."), out_dir=Path("."),
        cv=np.asarray(cv, dtype=np.float64), cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=window, replica=np.zeros(n, dtype=np.int32),
        step=np.arange(n, dtype=np.int64), u_nk=np.zeros((n, k)),
        centers=np.zeros(k), k_kcal=np.zeros(k), beta=0.4, temp=300.0,
        boost_kj=np.asarray(boost, dtype=np.float64), potential_kj=None, source="test",
        meta={},
    )


def _harmonic_windows_data(rng, centers, k_spring, beta, counts_per_window, prod_dir):
    """Real, non-degenerate multi-window harmonic-umbrella Data (same
    construction as tests/test_masked_logw_subset_pmf.py's helper of the same
    name) -- gives run_observable_pmf_convergence's internal per-checkpoint
    MBAR solves a real self-consistent problem instead of an all-zero-bias
    stub.
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
    step = np.arange(N, dtype=np.int64)
    return Data(
        prod_dir=prod_dir, out_dir=prod_dir / "out",
        cv=cv, cv2=np.full(N, np.nan), rg_A=np.full(N, np.nan),
        window=window, replica=window.copy(), step=step,
        u_nk=u_nk, centers=np.asarray(centers, float), k_kcal=np.asarray(k_spring, float) / KJ_PER_KCAL,
        beta=beta, temp=1.0 / (A.K_B_KJ_PER_MOL_K * beta), boost_kj=np.zeros(N), potential_kj=None,
        source="synthetic", meta={},
    )


# =============================================================================
# Fix 1: run_pmf_and_gamd_boost_report(precomputed_base_w=...)
# =============================================================================

def test_precomputed_base_w_matches_recompute_bit_identical(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    cv = rng.normal(0, 1, n)
    boost = np.abs(rng.normal(2.0, 0.5, n)) * 4.184
    d = _full_data(cv, boost)
    logw = rng.normal(0, 0.1, n)  # non-trivial logw, not all-zero
    bins = np.linspace(-3, 3, 21)

    out_recompute = tmp_path / "recompute"; out_recompute.mkdir()
    out_precomputed = tmp_path / "precomputed"; out_precomputed.mkdir()

    info_recompute = run_pmf_and_gamd_boost_report(
        d, _test_args(), logw, bins, 0.6, out_recompute, [], None,
        precomputed_base_w=None,
    )
    # Simulates exactly what analyze() does for the no-split case: base_w was
    # already computed once from this same full-population logw.
    already_computed_base_w = norm_logw(logw)
    info_precomputed = run_pmf_and_gamd_boost_report(
        d, _test_args(), logw, bins, 0.6, out_precomputed, [], None,
        precomputed_base_w=already_computed_base_w,
    )

    assert info_recompute["selected"] == info_precomputed["selected"]
    for method in info_recompute["pmfs"]:
        a = info_recompute["pmfs"][method]
        b = info_precomputed["pmfs"][method]
        for key in ("pmf", "prob", "counts", "cv_A"):
            np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]))
    assert info_recompute["pmf_span_kcal_mol"] == info_precomputed["pmf_span_kcal_mol"]
    assert info_recompute["boost"] == info_precomputed["boost"]


def test_precomputed_base_w_is_actually_used_not_silently_ignored(tmp_path):
    """If precomputed_base_w were accidentally ignored inside the function,
    these two calls (identical logw, but one given a deliberately WRONG
    precomputed value) would produce identical output. They must differ --
    otherwise the parameter isn't wired to anything.
    """
    rng = np.random.default_rng(1)
    n = 200
    cv = rng.normal(0, 1, n)
    boost = np.zeros(n)
    d = _full_data(cv, boost)
    logw = np.zeros(n)
    bins = np.linspace(-3, 3, 21)

    correct = norm_logw(logw)
    wrong = norm_logw(rng.normal(0, 5, n))  # unrelated weights, same shape

    out_correct = tmp_path / "correct"; out_correct.mkdir()
    out_wrong = tmp_path / "wrong"; out_wrong.mkdir()

    info_correct = run_pmf_and_gamd_boost_report(
        d, _test_args(), logw, bins, 0.6, out_correct, [], None, precomputed_base_w=correct,
    )
    info_wrong = run_pmf_and_gamd_boost_report(
        d, _test_args(), logw, bins, 0.6, out_wrong, [], None, precomputed_base_w=wrong,
    )
    prob_correct = np.asarray(info_correct["pmfs"]["umbrella_only"]["prob"])
    prob_wrong = np.asarray(info_wrong["pmfs"]["umbrella_only"]["prob"])
    assert not np.array_equal(prob_correct, prob_wrong)


def test_precomputed_base_w_array_not_mutated_by_report_call(tmp_path):
    """Regression guard for the aliasing this fix introduces: analyze() keeps
    using its own `base_w` after passing it into run_pmf_and_gamd_boost_report
    (Rg analysis, `base_ess` in pmf_summary.json). Nothing inside the report
    (directly, or via pmf_from_weights/cumulant2/cumulant3) may mutate the
    caller's array in place -- verified here rather than assumed, since a
    report-output comparison alone (the tests above) can't see a corruption
    that only shows up in a value the report itself never reads back.
    """
    rng = np.random.default_rng(11)
    n = 250
    cv = rng.normal(0, 1, n)
    boost = np.abs(rng.normal(2.0, 0.5, n)) * 4.184  # boost_ok path: exercises cumulant2/cumulant3 too
    d = _full_data(cv, boost)
    logw = rng.normal(0, 0.2, n)
    bins = np.linspace(-3, 3, 21)

    base_w = norm_logw(logw)
    base_w_snapshot = base_w.copy()
    out = tmp_path / "mutation_check"; out.mkdir()

    run_pmf_and_gamd_boost_report(
        d, _test_args(), logw, bins, 0.6, out, [], None, precomputed_base_w=base_w,
    )
    np.testing.assert_array_equal(base_w, base_w_snapshot)


def test_analyze_call_site_recomputes_for_split_reuses_for_no_split():
    """Structural guard on analyze()'s own call-site contract (the part that
    actually matters for correctness, since run_pmf_and_gamd_boost_report
    itself trusts whatever it's given): the epoch_000/rest split branch must
    set main_precomputed_base_w=None (forcing a fresh recompute for that
    genuinely different, renormalized population), and only the no-split
    branch may thread the already-computed, full-population base_w through.
    Uses source-position ordering rather than exact-text matching so it
    survives incidental reformatting while still enforcing the control-flow
    contract.

    Reads ``_analyze_population``, not ``analyze``: the single-population
    analysis (which is where this contract lives) was extracted out of
    ``analyze`` when the latter became the per-secondary-CV-regime
    orchestrator. The contract itself is unchanged -- only the function
    holding it moved.
    """
    import inspect
    src = inspect.getsource(A._analyze_population)

    pos_if = src.find("if epoch0_split is not None:")
    pos_none_assign = src.find("main_precomputed_base_w=None")
    pos_else = src.find("else:", pos_if)
    pos_basew_assign = src.find("main_precomputed_base_w=base_w", pos_else)
    pos_call = src.find("precomputed_base_w=main_precomputed_base_w")

    assert pos_if != -1 and pos_none_assign != -1 and pos_else != -1
    assert pos_basew_assign != -1 and pos_call != -1
    # None is assigned inside the split (if) branch, before the else.
    assert pos_if < pos_none_assign < pos_else
    # base_w is only threaded through in the else (no-split) branch.
    assert pos_else < pos_basew_assign < pos_call


# =============================================================================
# Fix 2: _window_cv_mean_std
# =============================================================================

def _reference_window_mean_std(window, cv, K):
    """The ORIGINAL per-window loop this fix replaced, reproduced verbatim
    for comparison: `vals = cv[window == k]; np.mean(vals)/np.std(vals) if
    vals.size else ''`.
    """
    window = np.asarray(window)
    cv = np.asarray(cv, dtype=np.float64)
    n_k = np.zeros(K, dtype=np.int64)
    has_samples = np.zeros(K, dtype=bool)
    mean = np.full(K, np.nan)
    std = np.full(K, np.nan)
    for k in range(K):
        vals = cv[window == k]
        n_k[k] = vals.size
        if vals.size:
            has_samples[k] = True
            mean[k] = np.mean(vals)
            std[k] = np.std(vals)
    return n_k, has_samples, mean, std


def test_window_cv_mean_std_matches_original_loop_incl_empty_and_nan_window():
    rng = np.random.default_rng(42)
    K = 5
    # window 0: 50 samples, real spread; window 1: EMPTY; window 2: 1 sample
    # (std must be exactly 0); window 3: 30 samples; window 4: 10 samples
    # with one NaN cv value (mean/std for the WHOLE window must come out NaN,
    # matching np.mean/np.std's own NaN-propagation on that slice).
    window = np.concatenate([
        np.zeros(50, dtype=np.int32),
        np.full(1, 2, dtype=np.int32),
        np.full(30, 3, dtype=np.int32),
        np.full(10, 4, dtype=np.int32),
    ])
    cv = np.concatenate([
        rng.normal(10.0, 1.5, 50),
        rng.normal(-5.0, 0.1, 1),
        rng.normal(3.0, 0.3, 30),
        rng.normal(20.0, 0.05, 10),
    ])
    cv[-1] = np.nan  # last sample belongs to window 4

    n_k_ref, has_samples_ref, mean_ref, std_ref = _reference_window_mean_std(window, cv, K)
    n_k_new, mean_new, std_new = _window_cv_mean_std(window, cv, K)
    has_samples_new = n_k_new > 0

    np.testing.assert_array_equal(n_k_new, n_k_ref)
    np.testing.assert_array_equal(has_samples_new, has_samples_ref)
    assert not has_samples_new[1]  # window 1 really is empty in both

    for k in range(K):
        if not has_samples_ref[k]:
            # Caller writes '' for this case in both old and new code --
            # here we just confirm the underlying arrays agree it's NaN.
            assert np.isnan(mean_new[k]) and np.isnan(std_new[k])
            continue
        if np.isnan(mean_ref[k]):
            assert np.isnan(mean_new[k]), f"window {k}: NaN propagation mismatch (mean)"
            assert np.isnan(std_new[k]), f"window {k}: NaN propagation mismatch (std)"
            continue
        # NOT bit-identical by design: the vectorized two-pass bincount
        # reduction accumulates in linear (bincount) order, while np.mean/
        # np.std use pairwise summation internally -- different floating-
        # point summation order, so results can differ in the last few ULPs
        # for real (non-trivial) data. Verified empirically here to land
        # well within 1e-9 relative / 1e-12 absolute, not exact equality.
        assert np.isclose(mean_new[k], mean_ref[k], rtol=1e-9, atol=1e-12), k
        assert np.isclose(std_new[k], std_ref[k], rtol=1e-9, atol=1e-12), k


def test_window_cv_mean_std_single_sample_window_std_is_exactly_zero():
    window = np.array([0], dtype=np.int32)
    cv = np.array([7.5])
    n_k, mean, std = _window_cv_mean_std(window, cv, 1)
    assert n_k[0] == 1
    assert mean[0] == 7.5
    assert std[0] == 0.0


def test_window_cv_mean_std_ignores_out_of_range_window_indices():
    # window index -1 and >= K must be excluded, same as the original loop
    # (which could only ever match k in range(K), so out-of-range window
    # values never contributed to any vals slice).
    window = np.array([0, 0, -1, 5, 0], dtype=np.int32)
    cv = np.array([1.0, 2.0, 999.0, -999.0, 3.0])
    n_k, mean, std = _window_cv_mean_std(window, cv, K=3)
    assert n_k[0] == 3
    assert mean[0] == pytest.approx(2.0)
    assert n_k[1] == 0 and n_k[2] == 0


def test_report_window_diagnostics_csv_matches_reference_loop_end_to_end(tmp_path):
    """Exercises the actual writer inside run_pmf_and_gamd_boost_report (not
    just the extracted helper in isolation), confirming the CSV it produces
    matches the original per-window loop's numbers.
    """
    import csv as _csv

    rng = np.random.default_rng(5)
    window = np.array([0] * 40 + [1] * 0 + [2] * 20, dtype=np.int32)  # window 1 empty
    cv = np.concatenate([rng.normal(0, 1, 40), rng.normal(5, 0.2, 20)])
    boost = np.zeros(cv.size)
    d = _full_data(cv, boost, k=3, window=window)
    logw = np.zeros(cv.size)
    bins = np.linspace(-3, 8, 21)

    run_pmf_and_gamd_boost_report(d, _test_args(), logw, bins, 0.6, tmp_path, [], None)

    n_k_ref, has_samples_ref, mean_ref, std_ref = _reference_window_mean_std(window, cv, 3)
    with (tmp_path / "window_diagnostics.csv").open() as f:
        rows = {int(r["window"]): r for r in _csv.DictReader(f)}
    for k in range(3):
        assert int(rows[k]["samples"]) == int(n_k_ref[k])
        if not has_samples_ref[k]:
            assert rows[k]["cv_mean_A"] == "" and rows[k]["cv_std_A"] == ""
        else:
            assert float(rows[k]["cv_mean_A"]) == pytest.approx(mean_ref[k], rel=1e-9, abs=1e-12)
            assert float(rows[k]["cv_std_A"]) == pytest.approx(std_ref[k], rel=1e-9, abs=1e-12)


# =============================================================================
# Fix 3: checkpoint_steps_from_data memoization
# =============================================================================

def test_checkpoint_steps_cache_key_content_based_not_shape_based():
    """Two masks of the same length but different content must get different
    cache keys -- otherwise the cache would silently collide two genuinely
    different finite-sample populations that happen to share a size.
    """
    d = _full_data(np.zeros(10), np.zeros(10), k=1, prod_dir=Path("/x/run"))
    mask_a = np.array([True] * 5 + [False] * 5)
    mask_b = np.array([False] * 5 + [True] * 5)
    key_a = _checkpoint_steps_cache_key(d, mask_a, 10)
    key_b = _checkpoint_steps_cache_key(d, mask_b, 10)
    assert key_a != key_b


def test_checkpoint_steps_cache_key_identical_for_identical_content():
    d = _full_data(np.zeros(10), np.zeros(10), k=1, prod_dir=Path("/x/run"))
    mask = np.array([True, False] * 5)
    key1 = _checkpoint_steps_cache_key(d, mask, 10)
    key2 = _checkpoint_steps_cache_key(d, mask.copy(), 10)  # different object, same content
    assert key1 == key2


def test_checkpoint_steps_cache_key_differs_on_population_size_and_timepoints():
    d1 = _full_data(np.zeros(10), np.zeros(10), k=1, prod_dir=Path("/x/run"))
    d2 = _full_data(np.zeros(20), np.zeros(20), k=1, prod_dir=Path("/x/run"))
    mask10 = np.ones(10, dtype=bool)
    mask20 = np.ones(20, dtype=bool)
    assert _checkpoint_steps_cache_key(d1, mask10, 10) != _checkpoint_steps_cache_key(d2, mask20, 10)
    assert _checkpoint_steps_cache_key(d1, mask10, 10) != _checkpoint_steps_cache_key(d1, mask10, 12)


def test_get_checkpoint_steps_cache_scoped_per_args_and_persists():
    args1 = SimpleNamespace()
    args2 = SimpleNamespace()
    cache1a = _get_checkpoint_steps_cache(args1)
    cache1b = _get_checkpoint_steps_cache(args1)
    cache2 = _get_checkpoint_steps_cache(args2)
    assert cache1a is cache1b  # same args -> same cache object across calls
    assert cache1a is not cache2  # different args -> independent caches
    cache1a["k"] = "v"
    assert _get_checkpoint_steps_cache(args1)["k"] == "v"


def test_checkpoint_steps_from_data_same_array_gives_equal_results():
    """Baseline sanity the task asked for directly: checkpoint_steps_from_data
    itself is a pure, deterministic function of its input -- two calls on the
    exact same array give identical results (this holds regardless of the
    caching layer built around it at the call site).
    """
    rng = np.random.default_rng(9)
    step = np.sort(rng.choice(1_000_000, size=200_000, replace=False)).astype(np.int64)
    result1 = checkpoint_steps_from_data(step, 10)
    result2 = checkpoint_steps_from_data(step, 10)
    np.testing.assert_array_equal(result1, result2)


def _lookup_or_compute_checkpoint_steps(d, finite_global, n_timepoints, cache):
    """Mirrors the exact cache block in run_observable_pmf_convergence
    (content-based key + stored-array identity check on hit) so this test
    exercises the real design, not a simplified stand-in for it.
    """
    key = _checkpoint_steps_cache_key(d, finite_global, n_timepoints)
    hit = cache.get(key)
    if hit is not None and hit[0] is d.step:
        return hit[1]
    steps = checkpoint_steps_from_data(d.step[finite_global], n_timepoints)
    cache[key] = (d.step, steps)
    return steps


def test_checkpoint_steps_cache_hit_reuses_result_and_is_measurably_faster():
    """The actual perf claim: for a large step array, a cache hit (digest +
    dict lookup + identity check) must be substantially cheaper than a cache
    miss (digest + the full np.unique(step) sort this fix targets).
    """
    n = 3_000_000
    step = np.arange(n, dtype=np.int64)  # every value unique -> real sort cost
    finite_global = np.ones(n, dtype=bool)
    d = _full_data(np.zeros(1), np.zeros(1), k=1, prod_dir=Path("/x/run"))
    d.step = step
    args = SimpleNamespace()
    n_timepoints = 10

    cache = _get_checkpoint_steps_cache(args)
    assert cache == {}

    t0 = time.perf_counter()
    steps1 = _lookup_or_compute_checkpoint_steps(d, finite_global, n_timepoints, cache)
    t_cold = time.perf_counter() - t0

    assert len(cache) == 1

    t0 = time.perf_counter()
    steps2 = _lookup_or_compute_checkpoint_steps(d, finite_global, n_timepoints, cache)
    t_warm = time.perf_counter() - t0

    np.testing.assert_array_equal(steps1, steps2)
    assert len(cache) == 1, "second call must reuse the same cache entry, not add a new one"
    assert t_warm < t_cold, f"cache hit ({t_warm:.4f}s) should be faster than the first, uncached call ({t_cold:.4f}s)"


def test_checkpoint_steps_cache_rejects_key_collision_across_different_step_arrays():
    """The hazard the content-based key alone can't rule out: two different
    Data objects (different underlying `step` values) that happen to share
    (prod_dir, population size, mask digest, n_timepoints) exactly. The
    stored-array identity check must force a real recompute for the second
    one rather than silently returning the first one's answer.
    """
    n_timepoints = 5
    finite_global = np.ones(4, dtype=bool)
    d1 = _full_data(np.zeros(4), np.zeros(4), k=1, prod_dir=Path("/x/run"))
    d1.step = np.array([10, 20, 30, 40], dtype=np.int64)
    d2 = _full_data(np.zeros(4), np.zeros(4), k=1, prod_dir=Path("/x/run"))
    d2.step = np.array([100, 200, 300, 400], dtype=np.int64)  # same shape/prod_dir/mask -> same key
    assert _checkpoint_steps_cache_key(d1, finite_global, n_timepoints) == _checkpoint_steps_cache_key(d2, finite_global, n_timepoints)

    cache = {}
    steps1 = _lookup_or_compute_checkpoint_steps(d1, finite_global, n_timepoints, cache)
    steps2 = _lookup_or_compute_checkpoint_steps(d2, finite_global, n_timepoints, cache)

    np.testing.assert_array_equal(steps1, d1.step)
    np.testing.assert_array_equal(steps2, d2.step)
    assert not np.array_equal(steps1, steps2)


def test_run_observable_pmf_convergence_reuses_checkpoint_cache_across_observables(tmp_path):
    """Integration-level check of the actual wiring inside
    run_observable_pmf_convergence (not just the standalone cache helpers):
    two DIFFERENT observables computed from the SAME Data (mirroring e.g. Rg
    vs SASA both being fully sampled with no missing trajectory frames) must
    share one checkpoint_steps_from_data cache entry rather than each
    recomputing/storing their own.
    """
    rng = np.random.default_rng(7)
    d = _harmonic_windows_data(
        rng, centers=[0.0, 2.0], k_spring=[20.0, 20.0], beta=0.5,
        counts_per_window=[60, 60], prod_dir=tmp_path,
    )
    obs_a = rng.normal(5.0, 1.0, d.cv.size)  # fully finite, unrelated observable
    obs_b = rng.normal(-2.0, 0.5, d.cv.size)  # fully finite, different unrelated observable
    bins = np.linspace(-4, 6, 11)
    final_pmf = {"cv_A": bins[:-1], "pmf": np.zeros(10), "prob": np.full(10, 0.1), "counts": np.full(10, 8)}

    args = SimpleNamespace(
        bins=20, min_neighbor_overlap=0.0, selected_method="auto",
        gamd_smooth_sigma=0.0, pmf_smooth_sigma=0.0, smooth_sigma=0.0,
        no_convergence=False, convergence_timepoints=4,
        convergence_mbar_backend="numpy", convergence_mbar_tol=1e-8,
        convergence_mbar_maxiter=2000, mbar_threads=0,
        no_convergence_mbar_cache=False,
        convergence_js_threshold=0.01, convergence_rmse_threshold=0.10,
        convergence_dir="convergence", no_basin_tracking=True,
    )

    cache = _get_checkpoint_steps_cache(args)
    assert cache == {}

    run_observable_pmf_convergence(
        d, args, obs_a, bins, "umbrella_only", final_pmf, tmp_path / "a",
        metric_name="obs_a", metric_label="A", x_label="a",
        out_dir_name="conv_a", file_prefix="a", progress=None,
    )
    assert len(cache) == 1

    run_observable_pmf_convergence(
        d, args, obs_b, bins, "umbrella_only", final_pmf, tmp_path / "b",
        metric_name="obs_b", metric_label="B", x_label="b",
        out_dir_name="conv_b", file_prefix="b", progress=None,
    )
    # obs_a and obs_b share the same `d` and are both fully finite -> same
    # finite-sample mask -> same cache key -> the second call must be a
    # cache HIT (still 1 entry), not a second, redundant computation.
    assert len(cache) == 1
