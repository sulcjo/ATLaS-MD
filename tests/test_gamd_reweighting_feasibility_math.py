"""The closed-form GaMD reweighting-feasibility math, pinned against sampling.

`audit_reweighting_feasibility.py` decides whether either reweighting estimator
can work on a run, and every one of its verdicts rests on modelling beta*dV as a
scaled noncentral chi-square with one degree of freedom. For a lower-bound GaMD
boost dV = 0.5*k*(E-V)^2 with V approximately Gaussian, that is exact.

The verdicts are strong ("infinite weight variance", "not resolved"), so the
algebra behind them should not be taken on trust. These tests generate from a
KNOWN noncentral chi-square and check the recovery, the cumulants, and -- most
importantly -- that the empirical exponential average really does fail where the
theory says it must.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest

from scipy.stats import ncx2

_spec = importlib.util.spec_from_file_location(
    "_audit_feas", pathlib.Path(__file__).resolve().parents[1] / "audit_reweighting_feasibility.py"
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)

A_TRUE, LAM_TRUE = 0.377, 22.0          # fitted from real chignolin_6 data


def _draw(a=A_TRUE, lam=LAM_TRUE, n=2_000_000, seed=0):
    return a * ncx2.rvs(df=1, nc=lam, size=n, random_state=np.random.default_rng(seed))


def _kappa(n, a=A_TRUE, lam=LAM_TRUE):
    """kappa_n = 2^(n-1) * a^n * (n-1)! * (1 + n*lambda)."""
    return 2 ** (n - 1) * a ** n * math.factorial(n - 1) * (1 + n * lam)


def test_moment_matching_recovers_the_scale_it_was_generated_with():
    """`a = m - sqrt(m^2 - v/2)` is the whole basis of the exponential verdict."""
    x = _draw()
    a = _audit._chi2_scale(x)
    assert a == pytest.approx(A_TRUE, rel=0.01), f"recovered a={a:.4f}, generated {A_TRUE}"
    lam = float(x.mean()) / a - 1.0
    assert lam == pytest.approx(LAM_TRUE, rel=0.02)


def test_the_minus_root_is_the_right_one():
    """The plus root also solves the quadratic and is physically wrong.

    a = m +/- sqrt(m^2 - v/2). The plus root gives lambda = m/a - 1 < 0, which is
    not a valid noncentrality, and would put `a` above 0.5 for this data --
    flipping the verdict from "infinite variance" to "estimator undefined".
    """
    x = _draw()
    m, v = float(x.mean()), float(x.var(ddof=1))
    plus = m + math.sqrt(m * m - v / 2.0)
    assert plus / A_TRUE > 10, "the two roots should be far apart here"
    assert (m / plus - 1.0) < 0.0, "the plus root implies a negative noncentrality"
    assert _audit._chi2_scale(x) < m, "the helper must take the minus root"


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_the_cumulant_formula_matches_sampled_cumulants(n):
    x = _draw()
    c = x - x.mean()
    m2 = float((c ** 2).mean())
    sampled = {
        1: float(x.mean()),
        2: m2,
        3: float((c ** 3).mean()),
        4: float((c ** 4).mean()) - 3.0 * m2 * m2,
    }[n]
    assert sampled == pytest.approx(_kappa(n), rel=0.02)


def test_successive_cumulant_terms_shrink_by_two_a():
    """The ratio that an earlier version wrongly extrapolated a bin-SPREAD with.

    The ratio is real -- it just governs successive terms at FIXED (a, lambda),
    not the spread across bins where (a, lambda) themselves vary. Pinned here so
    the true statement stays available and the false use stays out.
    """
    terms = [_kappa(n) / math.factorial(n) for n in range(1, 9)]
    ratios = [terms[i + 1] / terms[i] for i in range(2, 7)]
    assert all(r < 1.0 for r in ratios), "series must converge for a < 0.5"
    assert ratios[-1] == pytest.approx(2 * A_TRUE, rel=0.02), (
        f"late-order ratio {ratios[-1]:.4f} should approach 2a = {2 * A_TRUE}"
    )


def test_predicted_skew_and_kurtosis_match_so_the_family_assumption_is_checkable():
    """The parametric CE2 estimate is only as good as the family fit."""
    x = _draw()
    k2, k3, k4 = _kappa(2), _kappa(3), _kappa(4)
    c = x - x.mean()
    sd = float(c.std())
    assert float((c ** 3).mean()) / sd ** 3 == pytest.approx(k3 / k2 ** 1.5, rel=0.03)
    assert float((c ** 4).mean()) / sd ** 4 - 3.0 == pytest.approx(k4 / k2 ** 2, rel=0.06)


def test_the_empirical_exponential_average_fails_where_the_theory_says_it_must():
    """The evidence for the "NOT RESOLVED" verdict, on data with a known answer.

    At a = 0.377 the weights exp(+X) have infinite variance, so the sample mean
    of exp(X) has no finite standard error and does not converge. Here the true
    value is known in closed form, ln<e^X> = lambda*a/(1-2a) - 0.5*ln(1-2a), so
    the failure is measurable rather than argued: two million draws from the
    exact distribution land far below it, and land somewhere different for each
    seed.

    This is why the audit refuses to report an empirical CE2 truncation error for
    this run instead of quoting the number it happens to compute.
    """
    closed = LAM_TRUE * A_TRUE / (1 - 2 * A_TRUE) - 0.5 * math.log(1 - 2 * A_TRUE)

    def _ln_mean_exp(x):
        mx = float(x.max())
        return mx + math.log(float(np.exp(x - mx).mean()))

    estimates = [_ln_mean_exp(_draw(seed=s)) for s in range(3)]
    assert all(e < closed - 1.0 for e in estimates), (
        f"MC estimates {estimates} should fall well short of the closed form "
        f"{closed:.2f} at a = {A_TRUE}"
    )
    assert max(estimates) - min(estimates) > 0.5, (
        f"seed-to-seed spread {max(estimates) - min(estimates):.2f} -- if this "
        "were small the estimator would look convergent and the audit's refusal "
        "to quote it would be unjustified"
    )


def test_a_narrow_boost_is_estimable_so_the_refusal_is_not_unconditional():
    """Control: at a well below 0.25 the same machinery converges nicely."""
    a = 0.05
    closed = LAM_TRUE * a / (1 - 2 * a) - 0.5 * math.log(1 - 2 * a)

    def _ln_mean_exp(x):
        mx = float(x.max())
        return mx + math.log(float(np.exp(x - mx).mean()))

    est = [_ln_mean_exp(_draw(a=a, seed=s)) for s in range(3)]
    assert all(abs(e - closed) < 0.05 for e in est), f"{est} vs closed {closed:.3f}"
