"""The Crooks/Bennett slope identity and, critically, its DIRECTION.

`audit_crooks_slope.py` reads a fitted slope as a scale on the recorded force
constant, and converts it to a temperature as `T / |slope|`. Both readings are
directional: if they are inverted, a run that sampled too HOT is reported as too
cold, and a force constant that was applied too STRONG is reported as too weak.
Every interpretive sentence in the Tier 2 report depends on getting this right,
and the algebra alone is easy to convince yourself of in either direction.

So it is pinned by simulation against a known answer rather than by derivation:
draw exact samples from a nontrivial biased distribution, deliberately
mis-calibrate one thing at a time, and check which way the slope moves.

Measured here and reproduced independently:

    calibrated                        slope -0.999   (the exact null, -1)
    k_applied = 1.10 * k_recorded     slope -1.100   (NOT -0.909)
    sampled at 330 K, analysed 300 K  slope -0.907   (= 300/330)

so |slope| = k_applied / k_recorded, and T_true = T_assumed / |slope|.

Mutation battery against audit_crooks_slope.py (applied, run, reverted):

    CAUGHT  bias sign flipped                     2 failed
    CAUGHT  beta dropped from the reduced bias    3 failed
    CAUGHT  the 0.5 factor dropped from 0.5*k*x^2 3 failed
    passes  explicit separation guard removed     redundancy, not a gap

The last is not a miss. With the explicit support-overlap check removed, the
Newton loop's own weight-underflow branch still returns "separated", so the
observable behaviour is unchanged -- the module protects that path twice.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "_audit_crooks", pathlib.Path(__file__).resolve().parents[1] / "audit_crooks_slope.py"
)
_crooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_crooks)

_R_KJ = 0.008314462618
T_REC = 300.0
BETA_REC = 1.0 / (_R_KJ * T_REC)
GRID = np.linspace(-3.0, 3.0, 20001)


def _u0(x):
    """A tilted double well -- nontrivial, so the slope cannot be right by accident."""
    return 12.0 * (x ** 2 - 1.0) ** 2 + 2.0 * x


def _draw(beta_sample, k_sample, centre, n, seed):
    """Exact samples from exp(-beta*(U0 + 0.5*k*(x-c)^2)) by inverse-CDF on a grid."""
    logp = -beta_sample * (_u0(GRID) + 0.5 * k_sample * (GRID - centre) ** 2)
    p = np.exp(logp - logp.max())
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    u = np.random.default_rng(seed).random(n)
    return GRID[np.searchsorted(cdf, u)]


def _slope(k_sample, beta_sample, k_recorded, n=120_000, seed=0):
    """Fit the shipped estimator to two umbrellas, analysed at the recorded params."""
    ck, cl = -0.45, 0.45
    xk = _draw(beta_sample, k_sample, ck, n, seed)
    xl = _draw(beta_sample, k_sample, cl, n, seed + 1)
    x = np.concatenate([xk, xl])
    wk = _crooks.Win(ck, k_recorded, float("nan"), float("nan"))
    wl = _crooks.Win(cl, k_recorded, float("nan"), float("nan"))
    du = (_crooks._bias_kt(x, x, wk, BETA_REC) - _crooks._bias_kt(x, x, wl, BETA_REC))
    lab = np.concatenate([np.ones(n), np.zeros(n)])
    slope, se, status = _crooks._logistic_slope(du, lab)
    assert status == "fitted", status
    return slope, se


def test_a_correctly_calibrated_pair_gives_the_exact_null():
    slope, se = _slope(k_sample=10.0, beta_sample=BETA_REC, k_recorded=10.0)
    assert slope == pytest.approx(_crooks.NULL_SLOPE, abs=4 * se + 0.01), (
        f"slope {slope:.4f} +/- {se:.4f} against the exact null of -1"
    )


def test_an_over_strong_force_constant_makes_the_slope_MORE_negative():
    """|slope| = k_applied / k_recorded, not its reciprocal.

    If this is inverted, the audit reports an over-stiff restraint as too soft.
    """
    slope, se = _slope(k_sample=11.0, beta_sample=BETA_REC, k_recorded=10.0)
    assert slope < -1.05, f"slope {slope:.4f} should move toward -1.10, not toward -0.909"
    assert abs(slope) == pytest.approx(1.10, rel=0.03)


def test_sampling_hotter_than_assumed_makes_the_slope_LESS_negative():
    """|slope| = beta_true / beta_recorded = T_recorded / T_true.

    So the recovered temperature is T_assumed / |slope| -- the mapping the audit
    prints. Sampling at 330 K while analysing at 300 K must give 300/330.
    """
    beta_hot = 1.0 / (_R_KJ * 330.0)
    slope, se = _slope(k_sample=10.0, beta_sample=beta_hot, k_recorded=10.0)
    assert abs(slope) == pytest.approx(300.0 / 330.0, rel=0.03), (
        f"slope {slope:.4f}; expected {300.0 / 330.0:.4f}"
    )
    recovered = T_REC / abs(slope)
    assert recovered == pytest.approx(330.0, rel=0.03), (
        f"T/|slope| recovered {recovered:.1f} K from a 330 K sample -- the audit's "
        "temperature mapping is inverted"
    )


def test_perfectly_separated_states_are_reported_not_silently_skipped():
    """Widely spaced umbrellas share no support; the MLE does not exist.

    The original script returned NaN here and the caller skipped it with no
    message, so three real pairs vanished from a '28 pairs fitted' headline.
    """
    n = 5000
    far_k = _draw(BETA_REC, 400.0, -2.0, n, 7)
    far_l = _draw(BETA_REC, 400.0, +2.0, n, 8)
    x = np.concatenate([far_k, far_l])
    wk = _crooks.Win(-2.0, 400.0, float("nan"), float("nan"))
    wl = _crooks.Win(+2.0, 400.0, float("nan"), float("nan"))
    du = _crooks._bias_kt(x, x, wk, BETA_REC) - _crooks._bias_kt(x, x, wl, BETA_REC)
    lab = np.concatenate([np.ones(n), np.zeros(n)])
    _slope_v, _se, status = _crooks._logistic_slope(du, lab)
    assert status == "separated", f"status was {status!r}, so the failure is silent again"
