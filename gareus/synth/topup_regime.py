"""Ruling 28 regime calibration for the synthetic top-up study (arm A only, no arm B ever runs here)."""
from __future__ import annotations

import numpy as np

from .landscapes import Landscape

DEFICIT_BAND = (0.10, 0.40)  # fraction of states above target in arm A's final union solve
MIN_ROWS = 40                # median decorrelated rows per state over the campaign
CAL_SEEDS = (100, 101, 102)  # calibration seeds, disjoint from the study's seeds 0..n-1


def _regime_point(landscape: Landscape, hours: float, seeds, run_arm) -> dict:
    r = [run_arm(landscape, arm="uniform", seed=s, wall_hours_budget=hours) for s in seeds]
    return {"hours": hours, "deficit_fraction": float(np.median([x.deficit_fraction for x in r])),
            "median_rows": float(np.median([x.median_rows for x in r]))}


def calibrate_regime(landscape: Landscape, run_arm, *, seeds=CAL_SEEDS, band=DEFICIT_BAND, min_rows=MIN_ROWS,
                     start_hours: float = 1.0, iters: int = 8):
    """Ruling 28, arm A only: a budget where ``band`` of states are above target AND median rows >= ``min_rows``.

    Rows grow and the deficit fraction falls with hours, so the rows floor sets the smallest budget
    (h_rows, by bisection) and with it the largest reachable deficit fraction. If that is below the
    band there is no targetable regime (returns None); if above, bisect upward to the band's middle.
    Returns (hours or None, trace).
    """
    trace = []

    def point(h):
        trace.append(_regime_point(landscape, h, seeds, run_arm))
        return trace[-1]

    lo, hi = 0.0, float(start_hours)
    while point(hi)["median_rows"] < min_rows:
        lo, hi = hi, 2.0 * hi
    for _ in range(iters):                       # smallest hours meeting the rows floor
        mid = 0.5 * (lo + hi)
        (lo, hi) = (lo, mid) if point(mid)["median_rows"] >= min_rows else (mid, hi)
    at_floor = point(hi)
    if at_floor["deficit_fraction"] < band[0]:
        return None, trace
    if at_floor["deficit_fraction"] <= band[1]:
        return hi, trace
    lo, hi, target = hi, 2.0 * hi, 0.5 * (band[0] + band[1])
    while point(hi)["deficit_fraction"] > target:
        lo, hi = hi, 2.0 * hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q = point(mid)
        if band[0] <= q["deficit_fraction"] <= band[1]:
            return mid, trace
        (lo, hi) = (mid, hi) if q["deficit_fraction"] > target else (lo, mid)
    raise RuntimeError(f"regime calibration did not converge for {landscape.name}")


def bridge_budget(trace) -> float:
    """Smallest calibrated budget meeting the rows floor (the bridge scenario needs rows, not deficits)."""
    ok = [t["hours"] for t in trace if t["median_rows"] >= MIN_ROWS]
    if not ok:
        raise RuntimeError("no calibration point meets the rows floor")
    return float(min(ok))
