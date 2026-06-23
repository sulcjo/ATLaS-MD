"""The known answer, computed analytically from the exact free-energy grid.

The oracle never inspects what the adaptive algorithm did; it derives, purely
from ``F``, what a good window layout and the recovered PMF *should* look like:

* ``grid_overlap``      -- exact Bhattacharyya overlap of two windows' biased
  marginals along a CV axis (the quantity the adaptive logic estimates noisily).
* ``low_f_mask``        -- the region windows should cover (``F <= min + 5 kBT``).
* ``ideal_axis_ladder`` -- a minimal monotone center chain along one axis that
  keeps every neighbor overlap at/above the target and stays connected.
* ``reference_pmf``     -- the true marginal free energy along an axis.

Metrics compare the adaptive output to these.
"""
from __future__ import annotations

import numpy as np

from .landscapes import Landscape
from .sampler import Window, BIAS

LOW_F_THRESHOLD_KBT = 5.0


def _axis_biased_marginal(landscape: Landscape, window: Window, *,
                          beta: float = 1.0, res: int = 120, axis: str = "cv1"):
    c1, c2, f = landscape.grid(res=res)
    g1, g2 = np.meshgrid(c1, c2, indexing="ij")
    logp = -beta * (f + BIAS(window, g1, g2))
    logp = logp - logp.max()
    p = np.exp(logp)
    if axis == "cv1":
        m = p.sum(axis=1)
        x = c1
    else:
        m = p.sum(axis=0)
        x = c2
    s = m.sum()
    return x, (m / s if s > 0 else m)


def grid_overlap(landscape, wa, wb, *, beta: float = 1.0, res: int = 120,
                 axis: str = "cv1") -> float:
    """Bhattacharyya coefficient of two windows' biased axis marginals in [0,1]."""
    _, pa = _axis_biased_marginal(landscape, wa, beta=beta, res=res, axis=axis)
    _, pb = _axis_biased_marginal(landscape, wb, beta=beta, res=res, axis=axis)
    return float(np.sum(np.sqrt(pa * pb)))


def grid_overlap_2d(landscape, wa, wb, *, beta: float = 1.0, res: int = 120) -> float:
    """JOINT Bhattacharyya overlap of two windows' full 2D biased distributions.

    Unlike ``grid_overlap`` (an axis marginal), this is sensitive to separation
    in EITHER CV: two windows sharing a CV1 center but differing in CV2 correctly
    score near zero here even though their CV1 marginals coincide.
    """
    c1, c2, f = landscape.grid(res=res)
    g1, g2 = np.meshgrid(c1, c2, indexing="ij")
    def _dist(w):
        logp = -beta * (f + BIAS(w, g1, g2))
        logp = logp - logp.max()
        p = np.exp(logp)
        s = p.sum()
        return p / s if s > 0 else p
    pa, pb = _dist(wa), _dist(wb)
    return float(np.sum(np.sqrt(pa * pb)))


def low_f_mask(landscape, *, res: int = 120, threshold_kbt: float = LOW_F_THRESHOLD_KBT):
    """Return ``(cv1_axis, cv2_axis, mask)`` of the low-F region."""
    c1, c2, f = landscape.grid(res=res)
    return c1, c2, (f <= (f.min() + float(threshold_kbt)))


def reference_pmf(landscape, *, axis: str = "cv1", res: int = 120):
    """True marginal free energy along an axis (min-shifted to 0), in kBT."""
    c1, c2, f = landscape.grid(res=res)
    if axis == "cv1":
        g = -np.log(np.exp(-f).sum(axis=1))
        x = c1
    else:
        g = -np.log(np.exp(-f).sum(axis=0))
        x = c2
    g = g - g.min()
    return x, g


def ideal_axis_ladder(landscape, *, axis: str = "cv1", beta: float = 1.0,
                      res: int = 120, target_overlap: float = 0.30,
                      k: float = 50.0) -> list:
    """Greedy minimal monotone center chain meeting target neighbor overlap.

    Starts at the low-F projection minimum and, at each step, advances to the
    farthest next center whose neighbor overlap still meets ``target_overlap``,
    yielding a near-minimal connected ladder over the relevant region.
    """
    x, pmf = reference_pmf(landscape, axis=axis, res=res)
    relevant = x[pmf <= (pmf.min() + LOW_F_THRESHOLD_KBT)]
    lo, hi = float(relevant.min()), float(relevant.max())
    if hi <= lo + 1e-9:
        return [lo]
    centers = [lo]
    cur = lo
    step_grid = np.linspace(0.0, hi - lo, 400)[1:]
    guard = 0
    while cur < hi - 1e-9 and guard < 500:
        guard += 1
        best = None
        for ds in step_grid:
            cand = min(hi, cur + float(ds))
            o = grid_overlap(landscape, Window(cur, k), Window(cand, k),
                             beta=beta, res=res, axis=axis)
            if o >= target_overlap:
                best = cand
            else:
                break
        if best is None or best <= cur + 1e-9:
            best = min(hi, cur + float(step_grid[0]))  # forced minimal step
        centers.append(float(best))
        cur = float(best)
        if len(centers) > 200:
            break
    out = [centers[0]]
    for c in centers[1:]:
        if c - out[-1] > 1e-6:
            out.append(c)
    return out
