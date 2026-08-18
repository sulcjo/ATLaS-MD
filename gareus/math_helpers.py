"""
Mathematical helper functions.

This module contains low‑level mathematical routines extracted from
``gareus_peptide.py``.  They include histogram overlap metrics and
vectorised torsion angle computations.  These functions are shared
between CV definition, diagnostics and adaptive feedback modules.

Note that the names are prefixed with underscores to reflect their
original internal use; they are not part of the public API.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


def _hist_overlap(a: List[float], b: List[float], lo: float, hi: float, bins: int = 24) -> float:
    """Compute the histogram overlap between two one‑dimensional samples.

    The overlap is defined as the sum of the minimum of the normalised
    histograms of ``a`` and ``b`` over the specified range ``[lo, hi]``.

    Args:
        a: List of samples for the first distribution.
        b: List of samples for the second distribution.
        lo: Lower bound of the histogram range.
        hi: Upper bound of the histogram range.
        bins: Number of bins to use for both histograms (default 24).

    Returns:
        The histogram overlap as a float between 0 and 1, or ``nan``
        if there are too few finite samples or the range is invalid.
    """
    av = np.asarray([x for x in a if math.isfinite(float(x))], dtype=float)
    bv = np.asarray([x for x in b if math.isfinite(float(x))], dtype=float)
    if av.size < 5 or bv.size < 5 or hi <= lo:
        return float("nan")
    ha, _ = np.histogram(av, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(bv, bins=bins, range=(lo, hi), density=False)
    sa, sb = ha.sum(), hb.sum()
    if sa <= 0 or sb <= 0:
        return float("nan")
    pa = ha.astype(float) / float(sa)
    pb = hb.astype(float) / float(sb)
    return float(np.minimum(pa, pb).sum())


def _adaptive_hist_overlap(values_a: List[float] | np.ndarray, values_b: List[float] | np.ndarray, lo: float, hi: float, bins: int = 80, min_samples: int = 0) -> float:
    """Histogram overlap sum(min(P_i, P_j)) for two 1D CV samples.

    Accepts lists or already-sliced NumPy arrays.  Keeping bootstrap resamples as
    arrays avoids thousands of temporary Python-list conversions in adaptive
    feedback without changing the histogram definition.

    ``min_samples`` is an optional post-finite-filter floor on each side's
    sample count (default 0, i.e. only the empty-array case returns NaN --
    unchanged behavior for every existing call site in adaptive_feedback.py).
    gareus.diagnostics._hist_overlap_np delegates here with min_samples=5,
    its own pre-existing floor.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    floor = max(1, int(min_samples))
    if a.size < floor or b.size < floor:
        return float("nan")
    lo = float(lo)
    hi = float(hi)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        vals = np.concatenate([a, b])
        if vals.size < 2:
            return float("nan")
        lo, hi = float(np.min(vals)), float(np.max(vals))
    pad = max(0.1, 0.02 * (hi - lo))
    hist_a, _ = np.histogram(a, bins=max(8, int(bins)), range=(lo - pad, hi + pad))
    hist_b, _ = np.histogram(b, bins=max(8, int(bins)), range=(lo - pad, hi + pad))
    sa = float(np.sum(hist_a))
    sb = float(np.sum(hist_b))
    if sa <= 0.0 or sb <= 0.0:
        return float("nan")
    pa = hist_a.astype(float) / sa
    pb = hist_b.astype(float) / sb
    return float(np.sum(np.minimum(pa, pb)))


def ess(w: np.ndarray) -> float:
    """Effective sample size (Kish estimator) from an array of importance weights.

    ``sum(w)**2 / sum(w**2)``. Non-finite and negative entries are dropped
    before computing; an all-dropped or all-non-positive input returns 0.0.

    Kept without this module's usual leading underscore because it is
    already public API on ``analyze_gareus_mbar`` (``agm.ess``), called
    directly by three real-MD physics-oracle test files
    (tests/test_thermodynamic_validity_2d.py,
    tests/test_thermodynamic_validity_2d_rough.py,
    tests/test_thermodynamic_validity_real_md.py).
    """
    w = np.asarray(w, dtype=np.float64)
    w = w[np.isfinite(w) & (w >= 0)]
    if w.size == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return 0.0 if s2 <= 0 else s1 * s1 / s2


def _batch_torsion_angles_rad(positions_nm: np.ndarray, torsion_idx: np.ndarray) -> np.ndarray:
    """Compute torsion angles for all atom quadruples in ``torsion_idx``.

    Args:
        positions_nm: Array of shape (N, 3) containing positions in nanometers.
        torsion_idx: Integer array of shape (M, 4) with atom indices.

    Returns:
        Array of M torsion angles in radians.  Entries are NaN where
        the b1 bond length is degenerate (zero length).
    """
    if len(torsion_idx) == 0:
        return np.empty(0, dtype=np.float64)
    pos = np.asarray(positions_nm, dtype=np.float64)
    p0 = pos[torsion_idx[:, 0]]
    p1 = pos[torsion_idx[:, 1]]
    p2 = pos[torsion_idx[:, 2]]
    p3 = pos[torsion_idx[:, 3]]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    n = np.linalg.norm(b1, axis=1, keepdims=True)
    valid = (n > 1.0e-12).flatten()
    n_safe = np.where(n > 1.0e-12, n, 1.0)
    b1n = b1 / n_safe
    v = b0 - (b0 * b1n).sum(axis=1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(axis=1, keepdims=True) * b1n
    x = (v * w).sum(axis=1)
    y = (np.cross(b1n, v) * w).sum(axis=1)
    angles = np.arctan2(y, x)
    angles[~valid] = np.nan
    return angles


def _mean_torsion_score_from_angles(angles: np.ndarray, target_rad: float, sigma_rad: float) -> float:
    """Compute a mean Gaussian torsion score from an array of angles.

    The score uses a Gaussian kernel centred at ``target_rad`` with
    width ``sigma_rad``.
    """
    if len(angles) == 0:
        return 0.0
    scores = np.exp(-(1.0 - np.cos(angles - target_rad)) / max(1.0e-12, sigma_rad * sigma_rad))
    valid = np.isfinite(scores)
    if not np.any(valid):
        return 0.0
    return float(np.mean(scores[valid]))


def _torsion_angle_rad_from_positions(positions_nm: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """Compute a single torsion angle from four atom positions.

    Returns NaN if the bond length between atoms b and c is zero.
    """
    p0 = np.asarray(positions_nm[int(a)], dtype=float)
    p1 = np.asarray(positions_nm[int(b)], dtype=float)
    p2 = np.asarray(positions_nm[int(c)], dtype=float)
    p3 = np.asarray(positions_nm[int(d)], dtype=float)
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    n = np.linalg.norm(b1)
    if not math.isfinite(float(n)) or n <= 1.0e-12:
        return float("nan")
    b1n = b1 / n
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1n, v), w))
    return float(np.arctan2(y, x))


__all__ = [
    "_hist_overlap",
    "_adaptive_hist_overlap",
    "ess",
    "_batch_torsion_angles_rad",
    "_mean_torsion_score_from_angles",
    "_torsion_angle_rad_from_positions",
]
