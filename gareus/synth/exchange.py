"""Synthetic replica-exchange statistics for the adaptive-window harness.

Produces an ``exchange_stats`` dict in the exact schema the real adaptive
dispatcher consumes (see ``gareus.adaptive_feedback._adaptive_feedback_exchange_acceptance``):

    {"attempts": int, "accepted": int,
     "pairs": {"i-j": {"attempts": int, "accepted": int}},
     "jump_bins": {}, "mode": str}

with pair key ``f"{min(i,j)}-{max(i,j)}"``.

For each adjacent window pair we draw sample pairs and accept with the
Metropolis criterion on the *bias* energies only -- the underlying surface F
cancels in a swap, so acceptance depends purely on how much the two windows'
sampled regions overlap.  This is meaningful even in exact-equilibrium mode as
an overlap/exchange-stat generator (not as a sampling accelerator).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .sampler import Window, BIAS


def _grid_neighbor_pairs(n_windows: int, n_primary, n_secondary):
    """Adjacent window-index pairs.

    With ``n_primary``/``n_secondary`` given (rectangular grid, window index =
    ``ip*n_secondary + js``), enumerate BOTH axis topologies: secondary-axis
    neighbors within each primary column AND primary-axis neighbors across
    columns at the same secondary row.  Otherwise fall back to the 1D
    stride-1 chain.
    """
    if not n_primary or not n_secondary or n_primary * n_secondary != n_windows:
        return [(i, i + 1) for i in range(n_windows - 1)]
    pairs = []
    for ip in range(n_primary):
        for js in range(n_secondary - 1):
            w = ip * n_secondary + js
            pairs.append((w, w + 1))                       # secondary-axis neighbor
    for ip in range(n_primary - 1):
        for js in range(n_secondary):
            w = ip * n_secondary + js
            pairs.append((w, w + n_secondary))             # primary-axis neighbor
    return pairs


def build_exchange_stats(windows, samples_by_window, *, beta: float = 1.0,
                         n_swaps_per_pair: int = 200,
                         rng: Optional[np.random.Generator] = None,
                         mode: str = "neighbor",
                         n_primary=None, n_secondary=None) -> dict:
    """Build ``exchange_stats`` from per-window samples.

    Pass ``n_primary``/``n_secondary`` for a 2D grid so BOTH primary- and
    secondary-axis neighbor pairs are populated (the dispatcher reads per-axis
    exchange acceptance; flat stride-1 pairs alone leave the primary axis blind).
    """
    rng = rng if rng is not None else np.random.default_rng()
    stats = {"attempts": 0, "accepted": 0, "pairs": {}, "jump_bins": {}, "mode": mode}
    for i, j in _grid_neighbor_pairs(len(windows), n_primary, n_secondary):
        si = np.asarray(samples_by_window.get(i, np.empty((0, 2))), float)
        sj = np.asarray(samples_by_window.get(j, np.empty((0, 2))), float)
        if si.shape[0] == 0 or sj.shape[0] == 0:
            continue
        wi, wj = windows[i], windows[j]
        n = int(min(n_swaps_per_pair, si.shape[0], sj.shape[0]))
        ai = rng.integers(0, si.shape[0], n)
        aj = rng.integers(0, sj.shape[0], n)
        xi, xj = si[ai], sj[aj]
        # F cancels: only the four umbrella-bias terms remain.
        d = (BIAS(wi, xj[:, 0], xj[:, 1]) + BIAS(wj, xi[:, 0], xi[:, 1])
             - BIAS(wi, xi[:, 0], xi[:, 1]) - BIAS(wj, xj[:, 0], xj[:, 1]))
        acc = np.minimum(1.0, np.exp(-beta * d))
        accepted = int(np.sum(rng.uniform(0.0, 1.0, n) < acc))
        key = f"{min(i, j)}-{max(i, j)}"
        stats["pairs"][key] = {"attempts": int(n), "accepted": accepted}
        stats["attempts"] += int(n)
        stats["accepted"] += accepted
    return stats
