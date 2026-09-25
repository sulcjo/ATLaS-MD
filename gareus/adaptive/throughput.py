"""Wall-clock cost of a lockstep segment from a measured node throughput table.

The table maps contexts per GPU -> aggregate ns/day for the node. Between points
it interpolates linearly; outside them it stays flat (no extrapolated gain),
because the low-occupancy end is unmeasured (spec section 4.3.5).
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def node_ns_per_day(n_states: int, n_gpus: int, table: Sequence[Tuple[float, float]]) -> float:
    if not table:
        raise ValueError("throughput table is empty")
    pts = sorted((float(c), float(v)) for c, v in table)
    x = max(1.0, float(n_states)) / max(1, int(n_gpus))
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    return float(np.interp(x, xs, ys, left=ys[0], right=ys[-1]))


def wall_hours(steps: int, n_states: int, timestep_fs: float, n_gpus: int,
               table: Sequence[Tuple[float, float]]) -> float:
    aggregate_ns = float(steps) * float(timestep_fs) * 1e-6 * float(n_states)
    return aggregate_ns / node_ns_per_day(n_states, n_gpus, table) * 24.0
