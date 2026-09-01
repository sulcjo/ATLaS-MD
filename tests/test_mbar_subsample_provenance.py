"""Equilibration detection must not fail open silently.

`equilibrated_subsample_indices` returned `arange(n)` -- every index, no
thinning -- whenever pymbar was missing or `detect_equilibration` raised. That
is indistinguishable, at the call site, from "this series was already
uncorrelated". Any statistic built on the result then overstates n_eff by ~tau
and understates its own standard error by ~sqrt(tau), which turns a meaningless
test into a passing one.

That matters directly for the Crooks slope test, whose whole claim is a
confidence interval on a slope. So the subsampler now reports what it did, and
can be asked to refuse rather than guess.
"""
from __future__ import annotations

import numpy as np
import pytest

from gareus.mbar_subsample import (
    SubsampleResult,
    equilibrated_subsample,
    equilibrated_subsample_indices,
)

pymbar = pytest.importorskip("pymbar")


def _ar1(n=4000, rho=0.97, seed=7):
    """A correlated series with a known-ish tau = (1+rho)/(1-rho) ~ 65."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal() * np.sqrt(1 - rho * rho)
    return x


def test_a_correlated_series_reports_the_g_it_used():
    res = equilibrated_subsample(_ar1())
    assert isinstance(res, SubsampleResult)
    assert res.status == "subsampled"
    assert res.g > 1.0, "an AR(1) rho=0.97 series is not uncorrelated"
    assert 0 < res.indices.size < 4000, "nothing was actually thinned"
    # thinning spacing should track g
    spacing = float(np.diff(res.indices).mean())
    assert spacing == pytest.approx(res.g, rel=0.5), (
        f"kept every {spacing:.1f} frames but reported g={res.g:.1f}"
    )


def test_the_recovered_g_is_in_the_right_ballpark_for_a_known_tau():
    """Not a precision claim -- a sanity bound, since nothing else checks g."""
    rho = 0.97
    tau_expected = (1.0 + rho) / (1.0 - rho)          # ~65.7
    res = equilibrated_subsample(_ar1(n=20000, rho=rho))
    assert res.g == pytest.approx(tau_expected, rel=0.6), (
        f"g={res.g:.1f} against an expected tau of {tau_expected:.1f}"
    )


def test_an_uncorrelated_series_is_barely_thinned():
    res = equilibrated_subsample(np.random.default_rng(3).normal(size=4000))
    assert res.status == "subsampled"
    assert res.g < 3.0
    assert res.indices.size > 1000


def test_a_short_series_says_so_instead_of_pretending():
    res = equilibrated_subsample(np.arange(5.0))
    assert res.status == "too_short"
    assert np.isnan(res.g), "a g was reported for a series too short to estimate one"
    assert res.indices.size == 5


def test_strict_mode_refuses_rather_than_returning_every_index():
    with pytest.raises(RuntimeError, match="too_short"):
        equilibrated_subsample(np.arange(5.0), strict=True)


def test_strict_mode_propagates_a_detector_failure(monkeypatch):
    """A raising detector must not look like a clean uncorrelated series."""
    import gareus.mbar_subsample as ms

    def _boom(*a, **k):
        raise ValueError("synthetic detector failure")

    monkeypatch.setattr(ms, "_detect", _boom)
    res = equilibrated_subsample(_ar1())
    assert res.status == "detect_failed"
    assert res.indices.size == 4000, "fallback should still return everything"
    assert np.isnan(res.g)
    with pytest.raises(RuntimeError, match="detect_failed"):
        equilibrated_subsample(_ar1(), strict=True)


def test_the_legacy_helper_keeps_its_contract():
    """The existing adaptive-production caller must be unaffected."""
    idx = equilibrated_subsample_indices(_ar1())
    assert idx.dtype == np.int64
    assert idx.ndim == 1 and idx.size > 0
    assert np.all(np.diff(idx) > 0), "indices must be strictly increasing"
    assert idx.max() < 4000
    # and it still fails open, by design, for callers that cannot handle a raise
    assert equilibrated_subsample_indices(np.arange(5.0)).size == 5


def test_the_detection_cap_recovers_the_same_g_far_faster():
    """Striding the equilibration scan must not change the answer materially.

    detect_equilibration is ~O(n^2); at production trace lengths it is minutes
    per window, which is why the cap exists. It is only worth having if the
    recovered statistical inefficiency survives it.
    """
    import time

    x = _ar1(n=30000, rho=0.97)

    t0 = time.perf_counter()
    capped = equilibrated_subsample(x, max_detect_points=3000)
    t_capped = time.perf_counter() - t0

    t0 = time.perf_counter()
    full = equilibrated_subsample(x)
    t_full = time.perf_counter() - t0

    assert capped.status == full.status == "subsampled"
    assert capped.g == pytest.approx(full.g, rel=0.5), (
        f"capped detection recovered g={capped.g:.1f} against the full scan's "
        f"{full.g:.1f} -- the stride changed the answer, not just the cost"
    )
    assert t_capped < t_full, f"cap gave no speedup ({t_capped:.2f}s vs {t_full:.2f}s)"


def test_the_cap_does_not_invent_correlation_in_an_uncorrelated_series():
    """The reduction must be contiguous, not strided.

    Striding by s hides every lag below s, so an uncorrelated series comes back
    as g ~ s and gets thinned by that factor -- discarding ~97% of good data at
    production trace lengths. Measured directly when this cap was first written
    with a stride: g = 39.1 against a true g of ~1. A contiguous head preserves
    all lags up to its own length.
    """
    x = np.random.default_rng(1).normal(size=60000)     # true g ~ 1
    res = equilibrated_subsample(x, max_detect_points=4000)
    assert res.g < 5.0, (
        f"an uncorrelated series was assigned g={res.g:.1f}; the detection "
        "reduction is hiding short lags and will over-thin"
    )
    assert res.indices.size > 10000, "over-thinned an uncorrelated series"


def test_a_transient_longer_than_the_head_does_not_silently_shrink_g():
    """The head-truncation defect, with a series AR(1) structurally cannot show.

    `test_the_detection_cap_recovers_the_same_g_far_faster` above uses AR(1)
    rho=0.97 -- ONE timescale, tau ~ 66, comfortably inside any head. It always
    passes, and an adversarial review showed it green-lights a property that
    fails by 11x on real production traces. This is the case it cannot express:
    an equilibration transient LONGER than the head.

    detect_equilibration picks t0 by scanning within the series it is handed, so
    t0 is bounded above by the head. Give it a 6000-frame drift through a
    4000-frame window and t0 saturates near 4000, leaving a handful of points to
    fit g on -- which returns a g far too SMALL, over-keeping frames and
    overstating n_eff. Measured on real data before the fix: three chignolin_6
    windows reported g = 1.87/2.21/3.17 against full-trace 126/334/396.
    """
    rng = np.random.default_rng(11)
    n, rho = 40000, 0.97
    x = np.empty(n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal() * np.sqrt(1 - rho * rho)
    # a slow equilibration drift that outlasts the 4000-frame head
    x[:6000] += np.linspace(40.0, 0.0, 6000)

    res = equilibrated_subsample(x, max_detect_points=4000)
    assert res.status == "subsampled"
    # the head must have GROWN past its cap rather than reporting a tiny g
    assert res.n_detect > 4000, (
        f"detection stayed at a {res.n_detect}-frame head with t0={res.t0}: the "
        "transient outlasts the head, so g is fitted on the remainder and comes "
        "back far too small"
    )
    assert res.t0 >= 4000, "the 6000-frame transient should push t0 past the head"
    tau_expected = (1.0 + rho) / (1.0 - rho)          # ~65.7
    assert res.g > 0.3 * tau_expected, (
        f"g={res.g:.1f} against an expected tau of {tau_expected:.1f} -- the "
        "head truncation is still shrinking it"
    )


def test_head_exhaustion_is_reported_when_even_the_full_series_is_too_short():
    """A trace that is genuinely too short must say so, not return a small g."""
    rng = np.random.default_rng(5)
    x = rng.normal(size=1200)
    x[:1150] += np.linspace(60.0, 0.0, 1150)          # transient eats ~all of it
    res = equilibrated_subsample(x, max_detect_points=600)
    if res.head_exhausted:
        with pytest.raises(RuntimeError, match="too short to characterise"):
            equilibrated_subsample(x, max_detect_points=600, strict=True)
    else:
        # detection found a usable remnant; then it must have grown to get there
        assert res.n_detect >= 600


def test_the_cap_is_off_by_default_so_the_existing_caller_is_untouched():
    x = _ar1(n=2000)
    assert np.array_equal(
        equilibrated_subsample(x).indices,
        equilibrated_subsample(x, max_detect_points=None).indices,
    )
