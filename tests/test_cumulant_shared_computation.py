"""Regression tests for the cumulant2+cumulant3 shared-computation
performance fix.

Background: `cumulant2(...)` and `cumulant3(...)` (and their 2D twins) are
called back-to-back on IDENTICAL inputs at 7 call sites in
analyze_gareus_mbar.py to produce both estimators for a report. Since
order=3 is a strict superset of order=2's work (same histogram/bin-index
assignment/mean/variance, plus one extra kappa3 bincount), doing both calls
separately redid the expensive O(N) part twice. The fix introduces
`_cumulant_shared_stats`/`_cumulant_from_shared` (and `_cumulant_shared_stats_2d`/
`_cumulant_from_shared_2d`) so that shared work happens once, plus
`_cumulant_expansion_both`/`_cumulant_expansion_2d_both` for callers that need
both orders. `cumulant2`/`cumulant3`/`_cumulant_expansion` (and 2D
equivalents) are kept as fully independent, unchanged-behavior single-order
entry points (there is a genuine single-order caller elsewhere in the file,
`_observable_pmf_from_logw`).

This file's most important job is proving the new combined path is
bit-for-bit identical to calling the OLD (pre-fix) `cumulant2`+`cumulant3`
separately -- not merely close. `np.array_equal`/byte-equality is used
throughout; `np.allclose` is deliberately never used for the equivalence
checks.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import (
    Data,
    _bin_indices,
    _cumulant_expansion,
    _cumulant_expansion_2d,
    _cumulant_expansion_2d_both,
    _cumulant_expansion_both,
    cumulant2,
    cumulant2_2d,
    cumulant3,
    cumulant3_2d,
    parse_args,
    pmf_from_weights,
    run_pmf_and_gamd_boost_report,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BETA = 0.0387  # ~1/(kJ/mol) at 300 K, realistic magnitude
KBT_KCAL = 0.596


def _load_old_module():
    """Load the pre-fix analyze_gareus_mbar.py (the current git HEAD, i.e.
    the version before this session's edits) as an independent module, so
    its cumulant2/cumulant3 can be called and compared against the new
    combined path without reimplementing the old logic by hand (which would
    just test that the reimplementation matches itself)."""
    old_src = subprocess.run(
        ["git", "show", "HEAD:analyze_gareus_mbar.py"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    old_path = Path("/tmp/_old_ref_analyze_gareus_mbar_for_tests.py")
    old_path.write_text(old_src)
    spec = importlib.util.spec_from_file_location("old_agm_ref", old_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["old_agm_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def old_agm():
    return _load_old_module()


def _assert_float_array_bytes_equal(a, b, label):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, f"{label}: shape mismatch {a.shape} vs {b.shape}"
    # np.array_equal treats NaN != NaN, but these arrays are EXPECTED to
    # contain NaN by design (empty/undefined bins) -- tobytes() comparison
    # is the true bitwise check (also catches +0.0 vs -0.0) without needing
    # a NaN-aware wrapper, and never silently degrades to a tolerance check.
    assert a.tobytes() == b.tobytes(), f"{label}: bit-identical comparison failed"


def _assert_pmf_dict_bit_identical(d1, d2, keys_float, keys_int, label):
    for k in keys_float:
        _assert_float_array_bytes_equal(d1[k], d2[k], f"{label}[{k}]")
    for k in keys_int:
        assert np.array_equal(np.asarray(d1[k]), np.asarray(d2[k])), f"{label}[{k}] (int) mismatch"


PMF_FLOAT_KEYS_1D = ("cv_A", "prob", "pmf")
PMF_INT_KEYS = ("counts",)
DIAG_FLOAT_KEYS = ("boost_mean_kj", "boost_var_kj2", "boost_kappa3_kj3", "log_reweight_factor")
PMF_FLOAT_KEYS_2D = ("cv_A", "rg_A", "cv_edges_A", "rg_edges_A", "prob", "pmf")


def _rng():
    return np.random.default_rng(20260812)


def _make_1d_case(rng, n=4000, n_bins=17, with_nan_boost_bin=False, with_zero_weight=False):
    cv = rng.normal(0.0, 1.5, n)
    # Force some samples exactly on interior/edge bin boundaries and a few
    # out-of-range values, since that's exactly where a hand-rolled
    # searchsorted-based bin-index reimplementation could diverge from
    # np.histogram's own binning.
    edges = np.linspace(cv.min() - 0.3, cv.max() + 0.3, n_bins + 1)
    cv[0] = edges[-1]           # exactly the right edge (must land in last bin)
    cv[1] = edges[0]            # exactly the left edge
    cv[2] = edges[3]            # exactly an interior edge
    cv = np.concatenate([cv, [edges[0] - 5.0, edges[-1] + 5.0, np.nan, np.inf, -np.inf]])
    boost = rng.normal(40.0, 12.0, cv.size)
    boost[10:15] = np.nan  # a few individually-NaN boost samples scattered in
    base_w = rng.uniform(0.1, 1.0, cv.size)
    if with_zero_weight:
        base_w[20:30] = 0.0
    if with_nan_boost_bin:
        # Push a whole bin's worth of samples to have all-NaN boost, so that
        # bin exercises the "samples present, correction unknown" NaN path.
        mask = (cv >= edges[5]) & (cv < edges[6])
        boost = boost.copy()
        boost[mask] = np.nan
    return cv, base_w, boost, edges


def _make_2d_case(rng, n=4000, nx=11, ny=9):
    x = rng.normal(0.0, 1.2, n)
    y = rng.normal(0.0, 0.8, n)
    xedges = np.linspace(x.min() - 0.2, x.max() + 0.2, nx + 1)
    yedges = np.linspace(y.min() - 0.2, y.max() + 0.2, ny + 1)
    x[0] = xedges[-1]; y[0] = yedges[-1]
    x[1] = xedges[0]; y[1] = yedges[0]
    boost = rng.normal(60.0, 20.0, n)
    boost[5:9] = np.nan
    base_w = rng.uniform(0.2, 1.0, n)
    return x, y, base_w, boost, xedges, yedges


# ---------------------------------------------------------------------------
# 1. Bit-identical verification: NEW combined path vs OLD (pre-fix) separate
#    cumulant2()+cumulant3() calls. This is the single most important check
#    in this file.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smooth_sigma", [0.0, 1.5])
@pytest.mark.parametrize("with_nan_boost_bin", [False, True])
@pytest.mark.parametrize("with_zero_weight", [False, True])
def test_1d_combined_matches_old_separate_calls_bit_identical(
    old_agm, smooth_sigma, with_nan_boost_bin, with_zero_weight
):
    rng = _rng()
    cv, base_w, boost, edges = _make_1d_case(
        rng, with_nan_boost_bin=with_nan_boost_bin, with_zero_weight=with_zero_weight
    )

    old_pmf2, old_diag2 = old_agm.cumulant2(cv, base_w, boost, edges, BETA, KBT_KCAL, smooth_logfac_sigma=smooth_sigma)
    old_pmf3, old_diag3 = old_agm.cumulant3(cv, base_w, boost, edges, BETA, KBT_KCAL, smooth_logfac_sigma=smooth_sigma)

    (new_pmf2, new_diag2), (new_pmf3, new_diag3) = _cumulant_expansion_both(
        cv, base_w, boost, edges, BETA, KBT_KCAL, smooth_logfac_sigma=smooth_sigma
    )

    _assert_pmf_dict_bit_identical(old_pmf2, new_pmf2, PMF_FLOAT_KEYS_1D, PMF_INT_KEYS, "pmf2")
    _assert_pmf_dict_bit_identical(old_diag2, new_diag2, DIAG_FLOAT_KEYS, (), "diag2")
    _assert_pmf_dict_bit_identical(old_pmf3, new_pmf3, PMF_FLOAT_KEYS_1D, PMF_INT_KEYS, "pmf3")
    _assert_pmf_dict_bit_identical(old_diag3, new_diag3, DIAG_FLOAT_KEYS, (), "diag3")


def test_1d_combined_matches_old_separate_calls_int_bins(old_agm):
    """Same check with `bins` passed as a plain int (uniform-bin fast path
    inside np.histogram) rather than an explicit edge array -- both forms
    are used across the codebase and must both stay bit-identical."""
    rng = _rng()
    n = 3000
    cv = rng.normal(0.0, 2.0, n)
    boost = rng.normal(50.0, 15.0, n)
    base_w = rng.uniform(0.1, 1.0, n)
    n_bins = 12

    old_pmf2, old_diag2 = old_agm.cumulant2(cv, base_w, boost, n_bins, BETA, KBT_KCAL)
    old_pmf3, old_diag3 = old_agm.cumulant3(cv, base_w, boost, n_bins, BETA, KBT_KCAL)
    (new_pmf2, new_diag2), (new_pmf3, new_diag3) = _cumulant_expansion_both(
        cv, base_w, boost, n_bins, BETA, KBT_KCAL
    )

    _assert_pmf_dict_bit_identical(old_pmf2, new_pmf2, PMF_FLOAT_KEYS_1D, PMF_INT_KEYS, "pmf2(int bins)")
    _assert_pmf_dict_bit_identical(old_diag2, new_diag2, DIAG_FLOAT_KEYS, (), "diag2(int bins)")
    _assert_pmf_dict_bit_identical(old_pmf3, new_pmf3, PMF_FLOAT_KEYS_1D, PMF_INT_KEYS, "pmf3(int bins)")
    _assert_pmf_dict_bit_identical(old_diag3, new_diag3, DIAG_FLOAT_KEYS, (), "diag3(int bins)")


@pytest.mark.parametrize("smooth_sigma", [0.0, 1.0])
def test_2d_combined_matches_old_separate_calls_bit_identical(old_agm, smooth_sigma):
    rng = _rng()
    x, y, base_w, boost, xedges, yedges = _make_2d_case(rng)

    old_fes2, old_diag2 = old_agm.cumulant2_2d(x, y, base_w, boost, xedges, yedges, BETA, KBT_KCAL, smooth_logfac_sigma=smooth_sigma)
    old_fes3, old_diag3 = old_agm.cumulant3_2d(x, y, base_w, boost, xedges, yedges, BETA, KBT_KCAL, smooth_logfac_sigma=smooth_sigma)

    (new_fes2, new_diag2), (new_fes3, new_diag3) = _cumulant_expansion_2d_both(
        x, y, base_w, boost, xedges, yedges, BETA, KBT_KCAL, smooth_logfac_sigma=smooth_sigma
    )

    _assert_pmf_dict_bit_identical(old_fes2, new_fes2, PMF_FLOAT_KEYS_2D, PMF_INT_KEYS, "fes2")
    _assert_pmf_dict_bit_identical(old_diag2, new_diag2, DIAG_FLOAT_KEYS, (), "diag2_2d")
    _assert_pmf_dict_bit_identical(old_fes3, new_fes3, PMF_FLOAT_KEYS_2D, PMF_INT_KEYS, "fes3")
    _assert_pmf_dict_bit_identical(old_diag3, new_diag3, DIAG_FLOAT_KEYS, (), "diag3_2d")


def test_pmf_from_weights_matches_old_bit_identical(old_agm):
    """pmf_from_weights got the same (separate, smaller) bincount-for-counts
    optimization; verify it independently of the cumulant restructuring."""
    rng = _rng()
    cv, base_w, boost, edges = _make_1d_case(rng, with_zero_weight=True)
    del boost
    w = base_w.copy()
    # Also scatter some NaN weights, mirroring norm_logw's real output when
    # upstream logw/boost has NaNs.
    w[3:7] = np.nan

    old_pmf = old_agm.pmf_from_weights(cv, w, edges, KBT_KCAL)
    new_pmf = pmf_from_weights(cv, w, edges, KBT_KCAL)
    _assert_pmf_dict_bit_identical(old_pmf, new_pmf, ("cv_A", "prob", "pmf"), ("counts",), "pmf_from_weights")


# ---------------------------------------------------------------------------
# 2. Standalone single-order path must be untouched: cumulant2/cumulant3
#    (and 2D) alone must still match the old implementation bit-identically.
#    This is the genuine-single-order-caller guarantee
#    (_observable_pmf_from_logw calls exactly one of cumulant2/cumulant3,
#    never both).
# ---------------------------------------------------------------------------

def test_standalone_cumulant2_matches_old(old_agm):
    rng = _rng()
    cv, base_w, boost, edges = _make_1d_case(rng, with_nan_boost_bin=True)
    old_pmf, old_diag = old_agm.cumulant2(cv, base_w, boost, edges, BETA, KBT_KCAL)
    new_pmf, new_diag = cumulant2(cv, base_w, boost, edges, BETA, KBT_KCAL)
    _assert_pmf_dict_bit_identical(old_pmf, new_pmf, PMF_FLOAT_KEYS_1D, PMF_INT_KEYS, "standalone cumulant2 pmf")
    _assert_pmf_dict_bit_identical(old_diag, new_diag, DIAG_FLOAT_KEYS, (), "standalone cumulant2 diag")


def test_standalone_cumulant3_matches_old(old_agm):
    rng = _rng()
    cv, base_w, boost, edges = _make_1d_case(rng, with_nan_boost_bin=True)
    old_pmf, old_diag = old_agm.cumulant3(cv, base_w, boost, edges, BETA, KBT_KCAL)
    new_pmf, new_diag = cumulant3(cv, base_w, boost, edges, BETA, KBT_KCAL)
    _assert_pmf_dict_bit_identical(old_pmf, new_pmf, PMF_FLOAT_KEYS_1D, PMF_INT_KEYS, "standalone cumulant3 pmf")
    _assert_pmf_dict_bit_identical(old_diag, new_diag, DIAG_FLOAT_KEYS, (), "standalone cumulant3 diag")


def test_standalone_cumulant2_2d_and_cumulant3_2d_match_old(old_agm):
    rng = _rng()
    x, y, base_w, boost, xedges, yedges = _make_2d_case(rng)
    old_fes2, old_diag2 = old_agm.cumulant2_2d(x, y, base_w, boost, xedges, yedges, BETA, KBT_KCAL)
    new_fes2, new_diag2 = cumulant2_2d(x, y, base_w, boost, xedges, yedges, BETA, KBT_KCAL)
    _assert_pmf_dict_bit_identical(old_fes2, new_fes2, PMF_FLOAT_KEYS_2D, PMF_INT_KEYS, "standalone cumulant2_2d fes")
    _assert_pmf_dict_bit_identical(old_diag2, new_diag2, DIAG_FLOAT_KEYS, (), "standalone cumulant2_2d diag")

    old_fes3, old_diag3 = old_agm.cumulant3_2d(x, y, base_w, boost, xedges, yedges, BETA, KBT_KCAL)
    new_fes3, new_diag3 = cumulant3_2d(x, y, base_w, boost, xedges, yedges, BETA, KBT_KCAL)
    _assert_pmf_dict_bit_identical(old_fes3, new_fes3, PMF_FLOAT_KEYS_2D, PMF_INT_KEYS, "standalone cumulant3_2d fes")
    _assert_pmf_dict_bit_identical(old_diag3, new_diag3, DIAG_FLOAT_KEYS, (), "standalone cumulant3_2d diag")


def test_cumulant_expansion_bad_order_still_raises():
    with pytest.raises(ValueError):
        _cumulant_expansion(np.zeros(3), np.ones(3), np.ones(3), np.array([-1.0, 1.0]), BETA, KBT_KCAL, order=4)
    with pytest.raises(ValueError):
        _cumulant_expansion_2d(
            np.zeros(3), np.zeros(3), np.ones(3), np.ones(3),
            np.array([-1.0, 1.0]), np.array([-1.0, 1.0]), BETA, KBT_KCAL, order=5,
        )


# ---------------------------------------------------------------------------
# 3. Result-object independence: the order-2 and order-3 results from one
#    `_cumulant_expansion_both` call must not alias each other's arrays.
# ---------------------------------------------------------------------------

def test_combined_results_do_not_alias_between_orders():
    rng = _rng()
    cv, base_w, boost, edges = _make_1d_case(rng)
    (pmf2, diag2), (pmf3, diag3) = _cumulant_expansion_both(cv, base_w, boost, edges, BETA, KBT_KCAL)
    for key in ("cv_A", "prob", "pmf", "counts"):
        assert pmf2[key] is not pmf3[key], f"pmf dict key {key!r} aliased between orders"
    for key in ("boost_mean_kj", "boost_var_kj2", "boost_kappa3_kj3", "log_reweight_factor"):
        assert diag2[key] is not diag3[key], f"diag dict key {key!r} aliased between orders"
    # Mutating one order's result must not corrupt the other's.
    pmf2["prob"][:] = -999.0
    assert not np.any(pmf3["prob"] == -999.0)


def test_combined_results_2d_do_not_alias_between_orders():
    rng = _rng()
    x, y, base_w, boost, xedges, yedges = _make_2d_case(rng)
    (fes2, diag2), (fes3, diag3) = _cumulant_expansion_2d_both(x, y, base_w, boost, xedges, yedges, BETA, KBT_KCAL)
    for key in ("cv_A", "rg_A", "cv_edges_A", "rg_edges_A", "prob", "pmf", "counts"):
        assert fes2[key] is not fes3[key], f"fes dict key {key!r} aliased between orders"
    for key in ("boost_mean_kj", "boost_var_kj2", "boost_kappa3_kj3", "log_reweight_factor"):
        assert diag2[key] is not diag3[key], f"diag dict key {key!r} aliased between orders"


# ---------------------------------------------------------------------------
# 4. The bincount-for-counts optimization, verified in isolation.
# ---------------------------------------------------------------------------

def test_bincount_counts_match_histogram_unweighted_counts_nontrivial_distribution():
    rng = _rng()
    n = 5000
    cv = rng.normal(2.0, 3.0, n)
    edges = np.linspace(-8.0, 12.0, 26)  # deliberately non-trivial, non-uniform occupancy
    # Force some exact-edge and out-of-range values too.
    cv = np.concatenate([cv, [edges[0], edges[-1], edges[7], edges[0] - 1.0, edges[-1] + 1.0]])

    hist_counts, hist_edges = np.histogram(cv, bins=edges)
    bi = _bin_indices(cv, hist_edges)
    B = len(hist_edges) - 1
    inrange = (bi >= 0) & (bi < B)
    bincount_counts = np.bincount(bi[inrange], minlength=B)

    assert np.array_equal(hist_counts, bincount_counts)


def test_bincount_counts_match_histogram_with_nan_and_inf_values():
    cv = np.array([0.1, 0.5, 0.9, np.nan, np.inf, -np.inf, 1.5, -0.5])
    edges = np.linspace(0.0, 2.0, 5)
    hist_counts, hist_edges = np.histogram(cv, bins=edges)
    bi = _bin_indices(cv, hist_edges)
    B = len(hist_edges) - 1
    inrange = (bi >= 0) & (bi < B)
    bincount_counts = np.bincount(bi[inrange], minlength=B)
    assert np.array_equal(hist_counts, bincount_counts)


def test_bincount_counts_2d_match_histogram2d_unweighted_counts():
    rng = _rng()
    n = 4000
    x = rng.normal(0.0, 1.0, n)
    y = rng.normal(0.0, 1.0, n)
    xedges = np.linspace(-4.0, 4.0, 13)
    yedges = np.linspace(-4.0, 4.0, 9)
    x = np.concatenate([x, [xedges[0], xedges[-1], xedges[0] - 1.0]])
    y = np.concatenate([y, [yedges[0], yedges[-1], yedges[-1] + 1.0]])

    hist_counts, hx, hy = np.histogram2d(x, y, bins=[xedges, yedges])
    xi = _bin_indices(x, hx)
    yi = _bin_indices(y, hy)
    Bx = len(hx) - 1
    By = len(hy) - 1
    inrange = (xi >= 0) & (xi < Bx) & (yi >= 0) & (yi < By)
    idx_all = xi[inrange].astype(np.int64) * By + yi[inrange].astype(np.int64)
    bincount_counts = np.bincount(idx_all, minlength=Bx * By).reshape(Bx, By)

    assert np.array_equal(hist_counts, bincount_counts)


# ---------------------------------------------------------------------------
# 5. Negative test: the WEIGHTED probability histogram must remain
#    np.histogram(..., weights=...), never bincount -- proving the risky
#    half of the optimization was deliberately NOT applied.
# ---------------------------------------------------------------------------

def test_weighted_probability_histogram_still_uses_np_histogram_not_bincount():
    """np.bincount(bin_indices, weights=w) differs from
    np.histogram(..., weights=w) by ~1e-12 due to different internal
    floating-point summation order. The task explicitly forbids applying
    the bincount optimization to the weighted probability histogram. This
    test proves, non-tautologically, that the actual prob values in
    pmf_from_weights and the cumulant p0 computation match plain
    np.histogram(weights=...) and NOT the bincount alternative -- by
    computing the forbidden bincount alternative directly and asserting the
    real output differs from it (while still agreeing with it under
    np.allclose, confirming the difference is exactly the kind an
    allclose-based check would miss).
    """
    rng = _rng()
    n = 2000
    cv = rng.uniform(-5.0, 5.0, n)
    edges = np.linspace(-5.0, 5.0, 21)
    # Deliberately adversarial weights: huge dynamic range so that summation
    # order actually matters at float64 precision.
    w = rng.uniform(1e-12, 1.0, n)

    hist_prob, _ = np.histogram(cv, bins=edges, weights=w)

    # The FORBIDDEN alternative computed directly on the same data.
    bi = _bin_indices(cv, edges)
    B = len(edges) - 1
    inrange = (bi >= 0) & (bi < B)
    bincount_prob = np.bincount(bi[inrange], weights=w[inrange], minlength=B)

    # Confirm the test data actually discriminates: histogram and bincount
    # must differ at the bit level here (else this test would pass no
    # matter which one the code used).
    assert not np.array_equal(hist_prob, bincount_prob), (
        "test data does not discriminate np.histogram from np.bincount "
        "weighted sums -- strengthen the adversarial weights"
    )
    assert np.allclose(hist_prob, bincount_prob), (
        "sanity: both are the same mathematical sum, just different "
        "floating-point summation order -- they should still be close"
    )

    # pmf_from_weights's prob must match the np.histogram value, not the
    # bincount alternative.
    pmf = pmf_from_weights(cv, w, edges, KBT_KCAL)
    total = float(np.sum(hist_prob))
    hist_norm = hist_prob / total if total > 0 else hist_prob
    actual_prob = np.asarray(pmf["prob"], dtype=np.float64)
    assert np.array_equal(actual_prob, hist_norm.astype(np.float64))
    if total > 0:
        bincount_norm = bincount_prob / float(np.sum(bincount_prob))
        assert not np.array_equal(actual_prob, bincount_norm.astype(np.float64))

    # Same proof for the cumulant p0 path: reconstruct against BOTH the
    # known-correct np.histogram p0 and the forbidden bincount p0, and
    # confirm the real output matches only the former.
    boost = rng.normal(30.0, 5.0, n)
    _pmf2, diag2 = cumulant2(cv, w, boost, edges, BETA, KBT_KCAL)
    logfac = np.asarray(diag2["log_reweight_factor"], dtype=np.float64)
    factor = np.exp(np.clip(logfac, -700, 700))

    def _normalize(p0):
        p = p0 * factor
        s = float(np.nansum(p))
        return p / s if s > 0 else p

    actual_p2_prob = np.asarray(_pmf2["prob"], dtype=np.float64)
    assert np.array_equal(_normalize(hist_prob), actual_p2_prob)
    assert not np.array_equal(_normalize(bincount_prob), actual_p2_prob)


# ---------------------------------------------------------------------------
# 6. End-to-end smoke: exercise (as closely as feasible without a full real
#    MD run) the 7 call-site owning functions, confirming they still run
#    and produce internally-consistent gamd_cumulant2/gamd_cumulant3 outputs
#    after the refactor. Full integration is covered by the existing
#    end-to-end test suites (test_gareus_report.py,
#    test_epoch0_pmf_gamd_split.py, test_pmf_gamd_nan_bin_handling.py,
#    test_validate_ala_dipeptide.py); this focuses narrowly on "did the
#    unpacking at each call site come out right" for the two functions that
#    are exercised directly by unit tests already
#    (_observable_pmf_from_logw's sibling code path via run_pmf_and_gamd_boost_report
#    is exercised in test_epoch0_pmf_gamd_split.py / test_gareus_report.py).
# ---------------------------------------------------------------------------

def _full_data(cv, boost, k=3, window=None):
    n = len(cv)
    window = np.zeros(n, dtype=np.int32) if window is None else np.asarray(window, dtype=np.int32)
    return Data(
        prod_dir=Path("."), out_dir=Path("."),
        cv=np.asarray(cv, dtype=np.float64), cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=window, replica=np.zeros(n, dtype=np.int32),
        step=np.arange(n, dtype=np.int64), u_nk=np.zeros((n, k)),
        centers=np.zeros(k), k_kcal=np.zeros(k), beta=0.4, temp=300.0,
        boost_kj=np.asarray(boost, dtype=np.float64), potential_kj=None, source="test",
        meta={},
    )


def _test_args():
    args = parse_args(["dummy_input", "--bins", "20", "--no-convergence"])
    args.no_poincare_map = True
    return args


def test_run_pmf_and_gamd_boost_report_call_site_end_to_end(tmp_path):
    """Exercise the main CV1 PMF + GaMD boost report call site (one of the 7
    converted sites) end-to-end via the real `Data`/`run_pmf_and_gamd_boost_report`
    fixture pattern used by test_epoch0_pmf_gamd_split.py, confirming
    gamd_cumulant2/gamd_cumulant3 both come back finite and (with injected
    boost skew) distinct after the
    (cum_pmf,cdiag),(cum3_pmf,cdiag3) = _cumulant_expansion_both(...) refactor."""
    rng = _rng()
    n = 2000
    cv = rng.normal(0.0, 1.0, n)
    boost = rng.normal(45.0, 15.0, n)
    boost[50:55] = 150.0  # a touch of skew so cumulant2 vs cumulant3 actually differ
    # 3 real windows (not the degenerate K=1 case, where the neighbor-overlap
    # branch -- range(K-1) -- is skipped entirely) so the real overlap-matrix
    # path also executes alongside the cumulant unpacking.
    window = np.digitize(cv, [-0.4, 0.4]).astype(np.int32)
    d = _full_data(cv, boost, k=3, window=window)
    logw = np.zeros(n)
    bins = np.linspace(cv.min() - 0.2, cv.max() + 0.2, 21)

    info = run_pmf_and_gamd_boost_report(d, _test_args(), logw, bins, KBT_KCAL, tmp_path, [], None)

    assert info["boost_ok"]
    cum2 = info["pmfs"]["gamd_cumulant2"]
    cum3 = info["pmfs"]["gamd_cumulant3"]
    assert np.any(np.isfinite(cum2["pmf"]))
    assert np.any(np.isfinite(cum3["pmf"]))
    # With injected skew, cumulant3's PMF should differ from cumulant2's
    # somewhere (kappa3 correction is non-trivial).
    finite_both = np.isfinite(cum2["pmf"]) & np.isfinite(cum3["pmf"])
    assert np.any(cum2["pmf"][finite_both] != cum3["pmf"][finite_both])
    for fname in ("pmf_gamd_cumulant2.csv", "pmf_gamd_cumulant3.csv"):
        assert (tmp_path / fname).exists()


def test_run_pmf_and_gamd_boost_report_uses_combined_path_exactly_once(tmp_path, monkeypatch):
    """Pins the actual intent of this performance fix, not just its
    arithmetic: run_pmf_and_gamd_boost_report must call
    `_cumulant_expansion_both` exactly once (not zero, not twice) and must
    NOT call the standalone `cumulant2`/`cumulant3` at all. Without this,
    every bit-identical-output test above would still pass even if this
    call site were reverted to two separate cumulant2()+cumulant3() calls
    -- silently reintroducing the exact duplicated O(N) work this fix
    removes.
    """
    import analyze_gareus_mbar as agm
    import gareus.mbar_analysis.pmf as pmfmod

    rng = _rng()
    n = 500
    cv = rng.normal(0.0, 1.0, n)
    boost = rng.normal(45.0, 15.0, n)
    d = _full_data(cv, boost, k=1)
    logw = np.zeros(n)
    bins = np.linspace(cv.min() - 0.2, cv.max() + 0.2, 11)

    calls = {"both": 0, "single": 0}
    real_both = pmfmod._cumulant_expansion_both

    def _counting_both(*args, **kwargs):
        calls["both"] += 1
        return real_both(*args, **kwargs)

    def _fail_if_called(*args, **kwargs):
        calls["single"] += 1
        raise AssertionError("standalone cumulant2/cumulant3 must not be called from this site")

    monkeypatch.setattr(pmfmod, "_cumulant_expansion_both", _counting_both)
    monkeypatch.setattr(pmfmod, "cumulant2", _fail_if_called)
    monkeypatch.setattr(pmfmod, "cumulant3", _fail_if_called)

    agm.run_pmf_and_gamd_boost_report(d, _test_args(), logw, bins, KBT_KCAL, tmp_path, [], None)

    assert calls["both"] == 1, f"expected exactly 1 call to _cumulant_expansion_both, got {calls['both']}"
    assert calls["single"] == 0
