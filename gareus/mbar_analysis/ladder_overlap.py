"""Overlap along each axis of a (CV1 centre, λ rung) state grid.

A ladder run is two-dimensional even when cv2 is "none": states differ by CV1
centre and by rung. Collapsing both into one CV1-marginal number reports
whichever axis happens to dominate and hides the other -- chignolin_7 epoch 0
measured 0.93-0.98 along λ and 0.35-0.69 across CV1 boundaries, and only the
second is a problem.
"""
from __future__ import annotations

import numpy as np

from gareus.mbar_analysis.ladder import symmetric_state_overlap

_TOL = 1.0e-9


def _summarise(pairs):
    if not pairs:
        return {"pairs": [], "worst": None, "worst_pair": None, "n_pairs": 0}
    worst = min(pairs, key=lambda p: p[2])
    return {
        "pairs": [(int(a), int(b), float(v)) for a, b, v in pairs],
        "worst": float(worst[2]),
        "worst_pair": (int(worst[0]), int(worst[1])),
        "n_pairs": len(pairs),
    }


def ladder_overlap_by_axis(overlap, state_lambdas, centers) -> dict:
    """Split neighbour overlaps into the λ direction and the CV1 direction.

    λ-direction pairs share a CV1 centre and are adjacent in sorted λ.
    CV1-direction pairs share a rung and are adjacent in sorted centre.

    Each pair's value is ``symmetric_state_overlap(overlap, a, b)`` -- i.e.
    ``sqrt(O_ab * O_ba)`` -- never the raw ``overlap[a, b]``. ``O`` is
    asymmetric whenever the two states' sample counts differ (unequal
    ``n_k`` is the normal case under adaptive extension, not the exception;
    see ``gareus.mbar_analysis.ladder.symmetric_state_overlap``), so the raw
    entry would make a reported "worst" pair depend on which state happened
    to come first in the pair -- not on anything physical.
    """
    overlap = np.asarray(overlap, dtype=np.float64)
    lam = np.asarray(state_lambdas, dtype=np.float64)
    cen = np.asarray(centers, dtype=np.float64)

    lam_pairs, cv1_pairs = [], []
    for value in np.unique(np.round(cen, 9)):
        idx = np.flatnonzero(np.abs(cen - value) < _TOL)
        order = idx[np.argsort(lam[idx])]
        for a, b in zip(order[:-1], order[1:]):
            if abs(lam[b] - lam[a]) > _TOL:
                v = symmetric_state_overlap(overlap, a, b)
                if v is not None:
                    lam_pairs.append((a, b, v))
    for value in np.unique(np.round(lam, 9)):
        idx = np.flatnonzero(np.abs(lam - value) < _TOL)
        order = idx[np.argsort(cen[idx])]
        for a, b in zip(order[:-1], order[1:]):
            if abs(cen[b] - cen[a]) > _TOL:
                v = symmetric_state_overlap(overlap, a, b)
                if v is not None:
                    cv1_pairs.append((a, b, v))
    return {"lambda_direction": _summarise(lam_pairs),
            "cv1_direction": _summarise(cv1_pairs)}
