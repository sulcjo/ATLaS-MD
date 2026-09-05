"""Whether the GaMD cumulant expansion may be used at all on a given boost.

A truncated cumulant expansion approximates ``ln<exp(beta*dV)>`` by its first
few cumulants. That is only meaningful when successive terms SHRINK. For a
Gaussian boost of width sigma the terms go like ``(beta*sigma)^n / n!``, so the
series is only useful while the boost is at most a couple of kT wide.

Measured on a real 29M-sample chignolin run with ``beta*sigma = 3.57``:

    1st  beta*<dV>        8.81 kT
    2nd  0.5*beta^2*var   6.38 kT
    3rd  beta^3*kappa3/6  9.57 kT

Flat, not shrinking. The truncation error is the size of the terms retained,
so neither the 2nd- nor the 3rd-order curve estimates the unbiased free energy
-- yet ``gamd_cumulant2`` was selected unconditionally whenever a boost was
present. On that run one bin's variance excursion also made it the PMF minimum
and, because the PMF is zeroed at its minimum, re-referenced the whole curve,
moving the reported free-energy minimum by a full bin.

The remedy here is a convergence gate on the estimator, NOT smoothing. The
excursions on that run were 50-150 sigma against their own neighbours given
their sample counts, i.e. real structure; smoothing would have erased signal
and produced a tidier wrong answer. Smoothing remains available behind
``--gamd-smooth-sigma`` and is never applied on its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gareus.mbar_analysis.pmf import (  # noqa: E402
    CUMULANT_CONVERGENCE_MAX_RATIO,
    _cumulant_expansion,
    cumulant_series_verdict,
)

KT_KJ = 0.0083145 * 300.0
BETA = 1.0 / KT_KJ
KBT_KCAL = KT_KJ / 4.184


def _diag(t1, t2, t3, n=1):
    return {'cumulant_term1_kT': np.full(n, t1),
            'cumulant_term2_kT': np.full(n, t2),
            'cumulant_term3_kT': np.full(n, t3)}


# --- the convergence verdict -------------------------------------------------

def test_real_chignolin_boost_is_judged_non_converging():
    """The motivating case: terms 8.81 / 6.38 / 9.57 kT are flat, not
    shrinking, so the expansion must be rejected."""
    v = cumulant_series_verdict(_diag(8.81, 6.38, 9.57))
    assert v['converging'] is False
    assert v['ratio_2_over_1'] == pytest.approx(6.38 / 8.81, rel=1e-6)
    assert v['ratio_3_over_2'] == pytest.approx(9.57 / 6.38, rel=1e-6)
    assert v['max_ratio'] > CUMULANT_CONVERGENCE_MAX_RATIO


def test_a_narrow_boost_is_judged_converging():
    v = cumulant_series_verdict(_diag(2.0, 0.4, 0.05))
    assert v['converging'] is True
    assert v['max_ratio'] <= CUMULANT_CONVERGENCE_MAX_RATIO


def test_a_late_order_blow_up_is_caught_even_if_the_first_ratio_is_fine():
    """2nd/1st can look healthy while 3rd/2nd diverges; both ratios must pass."""
    v = cumulant_series_verdict(_diag(10.0, 1.0, 5.0))
    assert v['ratio_2_over_1'] == pytest.approx(0.1)   # looks fine, and is irrelevant
    assert v['ratio_3_over_2'] == pytest.approx(5.0)
    assert v['converging'] is False


def test_near_empty_bins_cannot_swing_the_verdict():
    """Bins are weighted by sample count, so a couple of ten-sample bins with
    wild moments neither condemn nor rescue an otherwise well-behaved run."""
    t1 = np.concatenate([np.full(40, 2.0), np.full(2, 50.0)])
    t2 = np.concatenate([np.full(40, 0.4), np.full(2, 90.0)])
    t3 = np.concatenate([np.full(40, 0.05), np.full(2, 90.0)])
    counts = np.concatenate([np.full(40, 1e6), np.full(2, 10.0)])
    d = {'cumulant_term1_kT': t1, 'cumulant_term2_kT': t2, 'cumulant_term3_kT': t3}
    assert cumulant_series_verdict(d, counts=counts)['converging'] is True
    # unweighted, the two rogue bins would dominate
    assert cumulant_series_verdict(d)['converging'] is False


def test_verdict_survives_missing_third_order():
    """order-2 diagnostics carry no kappa3; the verdict must fall back to the
    ratio it can compute rather than crashing or silently passing."""
    d = _diag(8.81, 6.38, np.nan)
    v = cumulant_series_verdict(d)
    assert not np.isfinite(v['ratio_3_over_2'])
    assert v['basis'].startswith('beta_sigma')
    # term2 = 0.5*(beta*sigma)^2 = 6.38 -> beta*sigma = 3.57, far above 1
    assert v['beta_sigma'] == pytest.approx(3.57, rel=0.02)
    assert v['converging'] is False


# --- term magnitudes are actually produced -----------------------------------

def _run(sigma_kj, n=200_000, nbins=41, seed=4, order=3, smooth=0.0):
    rng = np.random.default_rng(seed)
    cv = rng.uniform(0.0, 1.0, n)
    bins = np.linspace(0.0, 1.0, nbins)
    boost = rng.normal(22.0, sigma_kj, n)
    w = np.full(n, 1.0 / n)
    return _cumulant_expansion(cv, w, boost, bins, BETA, KBT_KCAL,
                               order=order, smooth_logfac_sigma=smooth)


def test_term_magnitudes_are_reported_and_scale_as_predicted():
    """For a Gaussian boost the terms go like (beta*sigma)^n/n!, so
    term2/term1 = beta*sigma/2. This pins that the recorded terms are the real
    ones and not, say, mislabelled or off by a factor."""
    sigma = 8.9
    _pmf, d = _run(sigma)
    v = cumulant_series_verdict(d, counts=_pmf['counts'])
    assert np.isfinite(v['term1_kT']) and np.isfinite(v['term2_kT'])
    # term2/term1 = (0.5*beta^2*sigma^2)/(beta*mean) = beta*sigma^2/(2*mean):
    # it depends on the arbitrary boost MEAN, which is exactly why the verdict
    # does not gate on it.
    assert v['ratio_2_over_1'] == pytest.approx(BETA * sigma**2 / (2.0 * 22.0), rel=0.05)
    assert v['beta_sigma'] == pytest.approx(BETA * sigma, rel=0.05)
    assert v['basis'] == 'ratio_3_over_2'
    # a Gaussian has kappa3 = 0, so the third term must be negligible
    assert v['term3_kT'] < 0.05 * v['term2_kT']


def test_a_wide_but_GAUSSIAN_boost_is_accepted():
    """Width alone does not invalidate CE2.

    For a Gaussian dV every cumulant above the second is exactly zero, so the
    expansion TERMINATES at order 2 and is exact no matter how wide the boost
    is. Rejecting on width would throw away a valid estimator. What breaks CE2
    is non-Gaussianity, which is what term3/term2 measures -- hence the gate.
    """
    wide, dw = _run(8.9)      # beta*sigma = 3.57, but perfectly Gaussian
    v = cumulant_series_verdict(dw, counts=wide['counts'])
    assert v['beta_sigma'] == pytest.approx(3.57, rel=0.05)
    assert v['converging'] is True


def test_a_skewed_boost_is_rejected():
    """A non-Gaussian boost is what actually breaks the truncation, and it is
    what the real run had: kappa3 worth 9.57 kT against a 6.38 kT second-order
    term."""
    rng = np.random.default_rng(3)
    n, nbins = 200_000, 41
    cv = rng.uniform(0.0, 1.0, n)
    bins = np.linspace(0.0, 1.0, nbins)
    boost = rng.gamma(shape=2.0, scale=6.0, size=n)   # strongly right-skewed
    w = np.full(n, 1.0 / n)
    pmf, d = _cumulant_expansion(cv, w, boost, bins, BETA, KBT_KCAL,
                                 order=3, smooth_logfac_sigma=0.0)
    v = cumulant_series_verdict(d, counts=pmf['counts'])
    assert v['term3_kT'] > 0.5 * v['term2_kT'], 'fixture is not skewed enough'
    assert v['converging'] is False


# --- smoothing is available but never automatic ------------------------------

def test_no_smoothing_is_applied_unless_asked():
    """The fix must not quietly smooth: where the bin-to-bin variation is real
    structure (it was, at 50-150 sigma, on the motivating run), smoothing
    erases signal."""
    sig = np.full(40, 8.9); sig[25] = 10.6
    rng = np.random.default_rng(4)
    n, nbins = 200_000, 41
    cv = rng.uniform(0.0, 1.0, n)
    bins = np.linspace(0.0, 1.0, nbins)
    idx = np.clip(np.digitize(cv, bins) - 1, 0, nbins - 2)
    boost = rng.normal(22.0, sig[idx])
    w = np.full(n, 1.0 / n)
    a, da = _cumulant_expansion(cv, w, boost, bins, BETA, KBT_KCAL, order=2, smooth_logfac_sigma=0.0)
    b, db = _cumulant_expansion(cv, w, boost, bins, BETA, KBT_KCAL, order=2, smooth_logfac_sigma=2.0)
    assert da['logfac_smooth_sigma_requested'] == 0.0
    assert 'logfac_auto_smoothed' not in da, 'auto-smoothing must be gone'
    np.testing.assert_allclose(np.asarray(a['pmf'], float)[np.isfinite(a['pmf'])],
                               np.asarray(a['pmf'], float)[np.isfinite(a['pmf'])])
    # asking for smoothing must actually change the curve
    fa = np.asarray(a['pmf'], float); fb = np.asarray(b['pmf'], float)
    m = np.isfinite(fa) & np.isfinite(fb)
    assert np.max(np.abs(fa[m] - fb[m])) > 0.05, 'requested smoothing had no effect'


def test_requested_smoothing_does_not_launder_undefined_bins():
    """A bin with samples but no finite boost has an UNKNOWN correction and is
    flagged NaN. Smoothing, when asked for, must neither turn that into a
    number nor spread it across neighbours."""
    rng = np.random.default_rng(7)
    n, nbins = 60_000, 41
    cv = rng.uniform(0.0, 1.0, n)
    bins = np.linspace(0.0, 1.0, nbins)
    idx = np.clip(np.digitize(cv, bins) - 1, 0, nbins - 2)
    boost = rng.normal(22.0, 8.9, n)
    boost[idx == 12] = np.nan
    w = np.full(n, 1.0 / n)
    _pmf, diag = _cumulant_expansion(cv, w, boost, bins, BETA, KBT_KCAL,
                                     order=2, smooth_logfac_sigma=2.0)
    lf = np.asarray(diag['log_reweight_factor'], dtype=float)
    assert not np.isfinite(lf[12]), 'the undefined bin must stay NaN'
    others = np.delete(lf, 12)
    assert np.isfinite(others).sum() >= others.size - 1, (
        'NaN leaked into neighbours through the smoothing filter')
