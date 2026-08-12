"""Regression tests for two related PMF/GaMD-reweighting correctness bugs:

1. ``_cumulant_expansion`` / ``_cumulant_expansion_2d`` used to zero-initialize
   the per-bin correction exponent ``logfac``, so a CV bin whose samples all
   had non-finite (NaN) GaMD boost -- e.g. a whole segment/epoch missing the
   ``gamd_boost_total_kj_mol`` column -- silently fell back to "no correction
   needed" (``p = p0*exp(0) = p0``, i.e. looked like the raw/unbiased-looking
   histogram) instead of "correction unknown". Fixed by NaN-flagging exactly
   those bins (real weighted samples present, zero with finite boost) while
   leaving genuinely-empty bins (no samples at all) untouched at logfac=0.

2. ``pmf_from_weights``'s ``counts`` field was an unconditional raw histogram
   over all CV values, regardless of whether each sample's weight was
   finite/positive. A bin populated only by zero/NaN-weight samples (samples
   that contribute nothing to the reweighted PMF) still showed counts>0, so
   the ``occupied_bins`` convergence diagnostic treated it as "covered" even
   though its true reweighted probability is 0. Fixed by histogramming only
   samples with finite, positive weight.
"""

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import (
    _cumulant_expansion,
    _cumulant_expansion_2d,
    norm_logw,
    pmf_from_weights,
    pmf_probability,
    js_divergence_1d,
    pmf_rmse_1d,
    barrier_error_1d,
)


BETA = 0.1  # 1/(kJ/mol)
KBT_KCAL = 0.6
# 3 bins: [-1,0) has real samples with finite boost, [0,1) has real samples
# whose boost is entirely NaN, [1,2) is genuinely empty (no samples at all).
BINS_3 = np.array([-1.0, 0.0, 1.0, 2.0])


def _mixed_finite_and_all_nan_boost_fixture():
    cv_good = np.full(20, -0.5)   # bin 0: finite boost throughout
    cv_bad = np.full(15, 0.5)     # bin 1: boost entirely NaN
    cv = np.concatenate([cv_good, cv_bad])
    base_w = np.ones_like(cv)
    boost = np.concatenate([
        np.linspace(10.0, 30.0, cv_good.size),
        np.full(cv_bad.size, np.nan),
    ])
    return cv, base_w, boost


def test_cumulant_expansion_nan_boost_bin_is_nan_not_umbrella_fallback():
    cv, base_w, boost = _mixed_finite_and_all_nan_boost_fixture()
    pmf, diag = _cumulant_expansion(cv, base_w, boost, BINS_3, BETA, KBT_KCAL, order=2)

    # Bin 1 (samples present, but all-NaN boost): correction is unknown ->
    # NaN probability and NaN PMF, not a silently-plausible finite number.
    assert np.isnan(pmf["prob"][1])
    assert not np.isfinite(pmf["pmf"][1])
    assert np.isnan(diag["log_reweight_factor"][1])

    # Bin 2 is genuinely empty (no samples at all) -- must be unaffected by
    # the NaN-flagging change: prob stays exactly 0, pmf stays non-finite
    # (regression guard pinning the "p0>0" restriction).
    assert pmf["prob"][2] == 0.0
    assert not np.isfinite(pmf["pmf"][2])
    assert diag["log_reweight_factor"][2] == 0.0

    # Bin 0 (real correction) must still be finite and properly normalized:
    # bins 1 and 2 contribute nothing (NaN / zero) to the normalizing sum, so
    # bin 0 -- the only bin with real finite positive probability -- must end
    # up carrying ~all of the normalized probability mass. This only holds if
    # the normalizing sum ignores NaN (np.nansum); with a plain np.sum the
    # whole array stays unnormalized once any bin is NaN, and this assertion
    # would fail.
    assert np.isfinite(pmf["pmf"][0])
    assert pmf["prob"][0] == pytest.approx(1.0, abs=1e-9)


def test_cumulant_expansion_order3_nan_boost_bin_is_nan():
    cv, base_w, boost = _mixed_finite_and_all_nan_boost_fixture()
    pmf3, diag3 = _cumulant_expansion(cv, base_w, boost, BINS_3, BETA, KBT_KCAL, order=3)
    assert np.isnan(pmf3["prob"][1])
    assert not np.isfinite(pmf3["pmf"][1])
    assert np.isnan(diag3["log_reweight_factor"][1])
    assert pmf3["prob"][2] == 0.0
    assert np.isfinite(pmf3["pmf"][0])


def test_cumulant_expansion_2d_nan_boost_bin_is_nan_not_umbrella_fallback():
    xbins = BINS_3
    ybins = np.array([-1.0, 1.0])  # single y-bin; everything falls in it
    cv, base_w, boost = _mixed_finite_and_all_nan_boost_fixture()
    y = np.zeros_like(cv)

    fes, diag = _cumulant_expansion_2d(cv, y, base_w, boost, xbins, ybins, BETA, KBT_KCAL, order=2)

    # (x-bin 1, y-bin 0): real samples, all-NaN boost -> unknown correction.
    assert np.isnan(fes["prob"][1, 0])
    assert not np.isfinite(fes["pmf"][1, 0])
    assert np.isnan(diag["log_reweight_factor"][1, 0])

    # (x-bin 2, y-bin 0): genuinely empty -> unaffected, still exactly 0.
    assert fes["prob"][2, 0] == 0.0
    assert not np.isfinite(fes["pmf"][2, 0])

    # (x-bin 0, y-bin 0): real correction, must still be finite and properly
    # normalized despite the NaN bin elsewhere in the grid (nansum check).
    assert np.isfinite(fes["pmf"][0, 0])
    assert fes["prob"][0, 0] == pytest.approx(1.0, abs=1e-9)


def test_pmf_from_weights_counts_ignore_invalid_weight_samples():
    # bin 0: 5 samples, ALL zero weight -> true reweighted probability is 0;
    #        counts must be 0, not the raw sample count (5).
    # bin 1: 6 samples, 4 with a valid positive weight and 2 with zero weight
    #        -> counts must be 4, not the raw sample count (6).
    # bin 2: empty, as a baseline (both old and new code give counts==0 here).
    cv = np.concatenate([
        np.full(5, -0.5),
        np.full(6, 0.5),
    ])
    w = np.concatenate([
        np.zeros(5),
        np.array([0.3, 0.3, 0.3, 0.3, 0.0, 0.0]),
    ])
    pmf = pmf_from_weights(cv, w, BINS_3, KBT_KCAL)

    assert pmf["counts"][0] == 0
    assert pmf["counts"][1] == 4
    assert pmf["counts"][2] == 0
    # Bin 0 contributes nothing to the weighted histogram either.
    assert pmf["prob"][0] == 0.0
    assert pmf["prob"][1] > 0.0


def test_pmf_from_weights_counts_drop_nan_weight_samples():
    """NaN weights (defensive case, e.g. an upstream bug reintroducing them)
    must also be excluded from counts, not just zero weights."""
    cv = np.concatenate([np.full(3, -0.5), np.full(4, 0.5)])
    w = np.concatenate([np.full(3, np.nan), np.array([1.0, 1.0, np.nan, np.nan])])
    pmf = pmf_from_weights(cv, w, BINS_3, KBT_KCAL)
    assert pmf["counts"][0] == 0
    assert pmf["counts"][1] == 2


def test_pmf_from_weights_real_shaped_nan_boost_segment_gives_zero_weight_and_count():
    """Reproduces the real Bug-1/Bug-2 interaction: a segment with NaN GaMD
    boost feeds exp_w = norm_logw(logw + beta*boost) in
    run_pmf_and_gamd_boost_report / _observable_pmf_from_logw. norm_logw
    zeroes out any non-finite input, so those samples get exactly w=0 and
    must not count as "occupied" in the gamd_exponential PMF's counts."""
    n_good = 30
    n_bad = 20
    cv = np.concatenate([np.full(n_good, -0.5), np.full(n_bad, 0.5)])
    logw = np.zeros(n_good + n_bad)
    boost = np.concatenate([
        np.linspace(5.0, 15.0, n_good),
        np.full(n_bad, np.nan),  # whole segment missing gamd_boost_total_kj_mol
    ])
    exp_w = norm_logw(logw + BETA * boost)
    assert np.all(exp_w[n_good:] == 0.0)  # sanity: norm_logw zeroes NaN input

    pmf = pmf_from_weights(cv, exp_w, BINS_3, KBT_KCAL)
    assert pmf["counts"][1] == 0  # all-NaN-boost bin must read as unoccupied
    assert pmf["prob"][1] == 0.0
    assert pmf["counts"][0] == n_good
    assert pmf["prob"][0] > 0.0


def test_nan_cumulant_bin_does_not_poison_convergence_metrics():
    """The NaN-flagged bin from _cumulant_expansion must not leak into the
    epoch/frame convergence diagnostics (run_pmf_convergence /
    run_epoch_pmf_convergence), which feed pmf_probability's output into
    js_divergence_1d / pmf_rmse_1d / barrier_error_1d. pmf_probability
    already nansum-normalizes and zero-fills non-finite entries, and the
    three metric helpers all mask on np.isfinite/P>0 -- so a NaN bin must
    read as "no probability mass" everywhere, not propagate NaN through the
    whole metric."""
    cv, base_w, boost = _mixed_finite_and_all_nan_boost_fixture()
    pmf, _diag = _cumulant_expansion(cv, base_w, boost, BINS_3, BETA, KBT_KCAL, order=2)
    assert np.isnan(pmf["prob"][1])  # sanity: the bug-case bin is really NaN

    prob = pmf_probability(pmf)
    assert np.all(np.isfinite(prob))
    assert prob[1] == 0.0
    assert prob[0] == pytest.approx(1.0, abs=1e-9)

    ref_prob = np.array([0.6, 0.3, 0.1])
    js = js_divergence_1d(prob, ref_prob)
    assert np.isfinite(js)

    F = np.asarray(pmf["pmf"], dtype=float)
    Fref = np.array([0.0, 1.0, 2.0])
    rmse = pmf_rmse_1d(F, Fref, prob, ref_prob)
    assert np.isfinite(rmse)

    barrier = barrier_error_1d(F, Fref, prob, ref_prob)
    assert np.isfinite(barrier)
