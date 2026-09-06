"""Which GaMD estimator gets selected, and why it is not always CE2.

Exponential reweighting of the MBAR umbrella weights,
``norm_logw(logw + beta*dV)``, is not an approximation. Because this package
calibrates ONE shared GaMD setup per campaign (see
``global_shared_gamd_setup_policy.json``: the first adaptive-production worker
calibrates it and every later worker reuses it), dV is the same function of
configuration for every state. Reweighting by ``exp(beta*dV)`` is then
algebraically identical to putting dV into the reduced potential and adding an
unbiased target state with u = 0: the boost is a per-row constant across state
columns, so it cancels from the f_k solve and survives only in the target
state's weight. So ``gamd_exponential`` IS the exact estimator, and CE2 is the
approximation to it.

What makes CE2 the right choice on a wide boost is not correctness but variance:
the exact estimator's effective sample size falls off like exp(-(beta*sigma)^2).
Measured on chignolin_6 (beta*sigma = 3.57) that is 3.0e-6 of N, which estimates
nothing, so CE2's truncation bias is the lesser evil there and stays selected.

That trade reverses once the boost is narrow. At the sigma0 = 1.0 kcal/mol this
campaign is being rebuilt with, beta*sigma = 1.43 and ESS/N = 0.13 -- an exact
estimator with an eighth of the samples, against a biased one with all of them.
Taking CE2's truncation error there buys nothing.

Note the direction of the gate. An earlier change that switched estimator on a
NON-CONVERGING CUMULANT VERDICT was reverted, correctly: that fires exactly when
the boost is wide, which is exactly when the exponential estimator is worthless.
This gates on the exponential estimator's OWN effective sample size instead, so
it can only fire when the exact estimator is actually affordable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gareus.mbar_analysis.pmf import (  # noqa: E402
    EXACT_REWEIGHT_MIN_ESS,
    EXACT_REWEIGHT_MIN_ESS_FRACTION,
    select_unbiased_method,
)


# --- the two measured campaigns --------------------------------------------

def test_chignolin_6_boost_keeps_ce2():
    """beta*sigma = 3.57 gives ESS/N = 3.0e-6. The exact estimator is
    unusable, so the biased one stays selected."""
    n = 940_788
    method, reason = select_unbiased_method(exp_ess=3.0e-6 * n, n_samples=n)
    assert method == 'gamd_cumulant2'
    assert 'ess' in reason.lower()


def test_the_rebuilt_narrow_boost_selects_the_exact_estimator():
    """sigma0 = 1.0 kcal/mol gives beta*sigma = 1.43 and ESS/N = 0.13."""
    n = 940_788
    method, _reason = select_unbiased_method(exp_ess=0.131 * n, n_samples=n)
    assert method == 'gamd_exponential'


# --- the gate itself --------------------------------------------------------

def test_both_a_fraction_and_an_absolute_floor_must_be_met():
    """A good fraction of a tiny sample is still a tiny sample."""
    n = 120
    ess = 0.5 * n                      # fraction is excellent, count is not
    assert ess < EXACT_REWEIGHT_MIN_ESS
    method, reason = select_unbiased_method(exp_ess=ess, n_samples=n)
    assert method == 'gamd_cumulant2'
    assert str(int(EXACT_REWEIGHT_MIN_ESS)) in reason


def test_a_huge_count_at_a_bad_fraction_is_still_rejected():
    """The fraction is what measures overlap between the boosted and unbiased
    ensembles; a big N can put many samples behind a hopeless fraction."""
    n = 100_000_000
    ess = 0.001 * n                    # 100k effective samples, but 0.1%
    assert ess > EXACT_REWEIGHT_MIN_ESS
    method, _r = select_unbiased_method(exp_ess=ess, n_samples=n)
    assert method == 'gamd_cumulant2'


def test_the_threshold_is_inclusive_at_the_boundary():
    n = 1_000_000
    ess = EXACT_REWEIGHT_MIN_ESS_FRACTION * n
    assert select_unbiased_method(exp_ess=ess, n_samples=n)[0] == 'gamd_exponential'
    method, _r = select_unbiased_method(exp_ess=ess * 0.999, n_samples=n)
    assert method == 'gamd_cumulant2'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.0])
def test_a_non_finite_or_negative_ess_falls_back_to_ce2(bad):
    """Never select the exact estimator on an ESS we could not compute."""
    assert select_unbiased_method(exp_ess=bad, n_samples=1000)[0] == 'gamd_cumulant2'


def test_zero_samples_does_not_divide_by_zero():
    method, _r = select_unbiased_method(exp_ess=0.0, n_samples=0)
    assert method == 'gamd_cumulant2'


def test_the_reason_is_always_populated():
    for ess, n in ((0.9e6, 1_000_000), (1.0, 1_000_000)):
        _m, reason = select_unbiased_method(exp_ess=ess, n_samples=n)
        assert isinstance(reason, str) and reason.strip()


def test_thresholds_are_overridable_for_callers_that_know_better():
    n = 1000
    ess = 0.5 * n
    assert select_unbiased_method(exp_ess=ess, n_samples=n,
                                  min_ess=10.0)[0] == 'gamd_exponential'
    assert select_unbiased_method(exp_ess=ess, n_samples=n, min_ess=10.0,
                                  min_ess_fraction=0.9)[0] == 'gamd_cumulant2'


def test_the_default_fraction_matches_the_existing_low_ess_warning():
    """pmf.py already warns below ESS/N = 0.05; the gate must use the same
    number so a run cannot be told its ESS is fine and still be given CE2."""
    assert EXACT_REWEIGHT_MIN_ESS_FRACTION == pytest.approx(0.05)
