"""Equilibration-discard + autocorrelation subsampling for MBAR pooling.

pyMBAR assumes each state's samples are uncorrelated and drawn from
equilibrium. Pilot and short-epoch traces violate that, so we discard the
pre-equilibration prefix and subsample to the statistical-inefficiency
spacing before pooling samples across states.

Two entry points, because the honest answer and the convenient one differ:

``equilibrated_subsample``
    Returns what it did -- the indices, the discarded prefix ``t0``, the
    statistical inefficiency ``g`` actually used, and a ``status``. Use this
    anywhere the result feeds a confidence interval, and record the ``g``.

``equilibrated_subsample_indices``
    The legacy shape: indices only, failing open to "keep everything" when
    equilibration cannot be estimated. Kept for the adaptive-production loader,
    which has no way to handle a refusal mid-campaign.

Why the split: failing open is indistinguishable, at the call site, from "this
series was already uncorrelated". A statistic built on that overstates n_eff by
roughly tau and understates its own standard error by roughly sqrt(tau) -- which
converts a meaningless test into a passing one.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# Guard to warn once per process when pymbar is unusable
_warned_pymbar_missing = False


class SubsampleResult(NamedTuple):
    """Indices plus the provenance needed to judge whether they mean anything."""
    indices: np.ndarray
    t0: int
    g: float          # statistical inefficiency actually used; NaN if not estimated
    status: str       # subsampled | too_short | pymbar_missing | detect_failed
    n_detect: int = 0        # frames detection finally ran on, after any head growth
    head_exhausted: bool = False   # t0 ate the probe: g is unreliable and too small


def _detect(arr: np.ndarray, max_detect_points: "int | None" = None,
            min_remnant: int = 500):
    """Thin seam over pymbar so tests can drive the failure path.

    ``detect_equilibration`` scans candidate t0 values and costs roughly O(n^2),
    which is minutes per window at production trace lengths. When
    ``max_detect_points`` is set, the scan runs on a **contiguous** head of the
    series rather than the whole thing.

    Contiguous, not strided, and the distinction matters: striding by s makes
    every lag below s invisible, so g can never be resolved below s and a
    weakly-correlated window gets reported as g ~ s and over-thinned by that
    factor. A contiguous head preserves every lag up to its own length and
    yields g directly in units of the sampling interval -- no rescaling, no
    resolution floor.

    **The head is NOT the conservative direction.** An earlier version of this
    docstring claimed it was, on the reasoning that an equilibration transient
    lives at the start. That is wrong, and measurably so. ``detect_equilibration``
    chooses ``t0`` by scanning candidates *within the series it is given*, so t0
    is bounded above by the head length. A transient longer than the head is
    silently truncated, g is then fitted on whatever few points remain after t0,
    and it comes back far too SMALL -- which over-keeps frames and overstates
    n_eff. Anti-conservative, in exactly the direction that turns a meaningless
    confidence interval into a narrow one.

    Measured on real data (`RUNS/chignolin_6/adaptive_production/epoch_000`, a
    4000-frame head): 9 of 32 windows pressed against the ceiling, three of them
    at t0 = 3760/3860/3808, leaving 240/140/192 points to fit g. They reported
    g = 1.87/2.21/3.17 against full-trace values of 126/334/396 -- over-keeping
    by 62x/139x/115x.

    So the head now GROWS when it is exhausted: if fewer than ``min_remnant``
    points survive t0, detection reruns on a head twice as long, up to the full
    series. ``head_exhausted`` in the result says whether even the full series
    left too little, which is a real "this trace is too short to characterise"
    signal rather than a silently small g.

    The thinning itself always runs on the full series, so no frames are lost
    merely because detection was made cheap.
    """
    from pymbar import timeseries

    n = int(arr.size)
    cap = int(max_detect_points) if max_detect_points else n
    cap = max(int(min_remnant) * 2, min(cap, n))
    while True:
        probe = arr[:cap] if cap < n else arr
        t0, g, _neff = timeseries.detect_equilibration(probe)
        remnant = int(probe.size) - int(t0)
        if remnant >= int(min_remnant) or cap >= n:
            break
        cap = min(n, cap * 2)
    kept = timeseries.subsample_correlated_data(arr[int(t0):], g=float(g))
    exhausted = (int(probe.size) - int(t0)) < int(min_remnant)
    return int(t0), float(g), np.asarray(kept, dtype=np.int64), int(probe.size), bool(exhausted)


def equilibrated_subsample(series, *, min_samples: int = 10,
                           strict: bool = False,
                           max_detect_points: "int | None" = None) -> SubsampleResult:
    """Equilibration discard + autocorrelation thinning, with provenance.

    ``strict=True`` raises instead of falling back, for callers whose result is
    only meaningful if the thinning actually happened.

    ``max_detect_points`` caps the cost of the equilibration scan (see
    :func:`_detect`); ``None`` keeps the exact pre-existing behaviour, so the
    adaptive-production caller is unaffected.
    """
    arr = np.asarray(series, dtype=float).ravel()
    n = arr.size

    def _fallback(status: str) -> SubsampleResult:
        if strict:
            raise RuntimeError(
                f"equilibrated_subsample: {status} -- refusing to return every "
                "index, because that is indistinguishable from an uncorrelated "
                "series and would overstate n_eff by ~tau"
            )
        return SubsampleResult(np.arange(n, dtype=np.int64), 0, float("nan"), status,
                               int(n), False)

    if n < int(min_samples):
        return _fallback("too_short")
    try:
        import pymbar  # noqa: F401
    except Exception:
        global _warned_pymbar_missing
        if not _warned_pymbar_missing:
            from gareus.pymbar_check import warn_if_pymbar_unusable  # noqa: PLC0415
            warn_if_pymbar_unusable("equilibration subsampling")
            _warned_pymbar_missing = True
        return _fallback("pymbar_missing")
    try:
        t0, g, kept, n_detect, exhausted = _detect(arr, max_detect_points)
    except Exception:
        return _fallback("detect_failed")
    if exhausted and strict:
        raise RuntimeError(
            f"equilibrated_subsample: equilibration detection consumed the probe "
            f"(t0={t0} of {n_detect}); g={g:.3g} is fitted on too few points and "
            "will be far too small. This trace is too short to characterise."
        )
    return SubsampleResult(kept + int(t0), int(t0), float(g), "subsampled",
                           int(n_detect), bool(exhausted))


def equilibrated_subsample_indices(series, *, min_samples: int = 10):
    """Increasing integer indices after equilibration discard + thinning.

    Fails open to all indices when equilibration cannot be estimated. Prefer
    :func:`equilibrated_subsample` when the caller can record or act on that.
    """
    return equilibrated_subsample(series, min_samples=min_samples).indices
