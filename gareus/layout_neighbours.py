"""Spatial neighbour pairs for the dashboard's overlap check on a 2D layout.

The 1D check (`context.overlap_by_pair`) compares window w with w+1. On a 2D
lambda-ladder layout that pairing is meaningless: consecutive indices sit in
different cells or on different rungs, so their CV1 histograms barely touch and
every such pair looked "dead" (a real-shaped chignolin_9 render flagged 78 of 236
windows on overlap; with the pairs below, none).

Here each window is paired with its nearest windows ON THE SAME RUNG, measured
in restraint widths on each axis (sigma = sqrt(kT/k)): the nearest one, plus any
within `NEIGHBOUR_SLACK` x that distance. That covers dense rows, sparse rows
and diagonal layouts alike -- a row/column rule would compare a diagonal
window with a far same-row one and flag a healthy chain.

  * Rung-to-rung pairs are left out on purpose: two rungs at one centre overlap
    ~1 in CV space by construction, so a CV histogram cannot see a rung gap
    (that needs the energy-space MBAR overlap, gareus.mbar_analysis.ladder).
  * Windows pair only with windows restrained on the same axes; an axis without
    restraint (recorded k = 0) is ignored. Fully unrestrained windows are left out.
  * Pair overlap is the product of the per-axis histogram overlaps.
  * A window is then judged by its BEST neighbour (`best_neighbour_overlaps`):
    in 2D one weak edge does not cut a window off, being isolated does.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

from .math_helpers import _hist_overlap
from .dashboard.ranking import restraint_sigma

NEIGHBOUR_SLACK = 1.5
_CENTRE_DECIMALS = 6          # same grouping tolerance as the 2D window map


def _finite(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _restrained_mask(k: Sequence[float], n: int) -> np.ndarray:
    """Unknown k counts as restrained: only a recorded 0 takes a window off an axis."""
    out = np.ones(n, dtype=bool)
    for w in range(min(n, len(k))):
        value = _finite(k[w])
        if value is not None and value == 0.0:
            out[w] = False
    return out


def _axis_scale(pos: np.ndarray, k: Sequence[float], restrained: np.ndarray, temperature_k: float) -> float:
    """Typical restraint width on an axis; the centre spacing if no k is known."""
    widths = []
    for w in np.flatnonzero(restrained):
        value = _finite(k[w]) if w < len(k) else None
        if value is not None and value > 0.0:
            sigma = restraint_sigma(value, temperature_k)
            if math.isfinite(sigma):
                widths.append(sigma)
    if widths:
        return float(np.median(widths))
    distinct = np.unique(np.round(pos[restrained], _CENTRE_DECIMALS))
    gaps = np.diff(distinct)
    return float(np.median(gaps)) if gaps.size else 1.0


def _layout(centers, secondary_centers, lambdas, k1, k2, temperature_k):
    n = min(len(centers), len(secondary_centers))
    pos = np.array([[float(centers[w]), float(secondary_centers[w])] for w in range(n)], dtype=float)
    rung = []
    for w in range(n):
        lam = _finite(lambdas[w]) if w < len(lambdas) else None
        rung.append(None if lam is None else round(lam, _CENTRE_DECIMALS))
    restrained = np.stack([_restrained_mask(k1, n), _restrained_mask(k2, n)], axis=1)
    scale = np.array([_axis_scale(pos[:, a], (k1, k2)[a], restrained[:, a], temperature_k)
                      for a in range(2)])
    return n, pos, rung, restrained, scale


def _pairs_from_layout(n, pos, rung, restrained, scale) -> list[tuple[int, int]]:
    active = restrained.any(axis=1)
    z = pos / np.where(scale > 0, scale, 1.0)
    pairs: set[tuple[int, int]] = set()
    for i in np.flatnonzero(active):
        same = np.array([j != i and active[j] and rung[j] == rung[i] for j in range(n)])
        if not same.any():
            continue
        # Same restraint pattern only: a CV2-only window (parking CV1 value) would
        # otherwise sit at zero CV1 distance from everyone and hide CV1 gaps.
        axes = restrained[i] & restrained          # (n, 2): axes both windows are restrained on
        usable = same & (restrained == restrained[i]).all(axis=1)
        if not usable.any():
            continue
        d = np.sqrt(((z - z[i]) ** 2 * axes).sum(axis=1))
        positive = d[usable & (d > 0)]
        cutoff = NEIGHBOUR_SLACK * positive.min() if positive.size else 0.0
        for j in np.flatnonzero(usable & (d <= cutoff)):
            pairs.add((min(i, j), max(i, j)))
    return sorted((int(a), int(b)) for a, b in pairs)


def spatial_neighbour_pairs(
    centers: Sequence[float],
    secondary_centers: Sequence[float],
    lambdas: Sequence[float],
    k1: Sequence[float],
    k2: Sequence[float],
    temperature_k: float,
) -> list[tuple[int, int]]:
    """`(a, b)` with a < b for every nearest-neighbour pair on the same rung."""
    return _pairs_from_layout(*_layout(centers, secondary_centers, lambdas, k1, k2, temperature_k))


def overlap_by_pair_2d(
    history_cv1: Mapping[int, Sequence[float]],
    history_cv2: Mapping[int, Sequence[float]],
    centers: Sequence[float],
    secondary_centers: Sequence[float],
    lambdas: Sequence[float],
    k1: Sequence[float],
    k2: Sequence[float],
    temperature_k: float,
) -> dict[tuple[int, int], float]:
    """Overlap of each neighbour pair: product of per-axis histogram overlaps.

    Each axis histogram range is local to the pair (both centres +/- their
    spacing, or +/- 4 restraint widths when the centres coincide on that axis),
    so resolution follows the neighbour spacing rather than the whole axis span.
    """
    n, pos, rung, restrained, scale = _layout(centers, secondary_centers, lambdas, k1, k2, temperature_k)
    histories = (history_cv1, history_cv2)
    out: dict[tuple[int, int], float] = {}
    for a, b in _pairs_from_layout(n, pos, rung, restrained, scale):
        product = 1.0
        for axis in range(2):
            if not (restrained[a, axis] and restrained[b, axis]):
                continue
            ca, cb = float(pos[a, axis]), float(pos[b, axis])
            pad = abs(cb - ca) or 4.0 * float(scale[axis])
            ov = _hist_overlap(list(histories[axis].get(a, ())), list(histories[axis].get(b, ())),
                               min(ca, cb) - pad, max(ca, cb) + pad)
            if not math.isfinite(ov):
                product = math.nan
                break
            product *= ov
        if math.isfinite(product):
            out[(a, b)] = float(product)
    return out


def best_neighbour_overlaps(
    pair_overlaps: Mapping[tuple[int, int], float], n_windows: int
) -> tuple[dict[tuple[int, int], float], dict[int, float]]:
    """Each window's best neighbour overlap, and the edges that achieve it.

    Returns ``(edges, per_window)``: ``per_window[w]`` is the largest overlap
    over w's neighbour pairs (a low value means w is isolated); ``edges`` keeps
    only those best links, so a listed pair is always some window's strongest.
    """
    best: dict[int, tuple[float, tuple[int, int]]] = {}
    for pair, ov in pair_overlaps.items():
        for w in pair:
            if 0 <= w < n_windows and (w not in best or ov > best[w][0]):
                best[w] = (float(ov), pair)
    per_window = {w: ov for w, (ov, _pair) in sorted(best.items())}
    edges = {pair: ov for ov, pair in best.values()}
    return dict(sorted(edges.items())), per_window


def same_rung_neighbours(pairs, lambdas) -> dict:
    """window -> sorted list of its spatial-neighbour windows (pairs are same-rung already)."""
    out: dict = {}
    for a, b in pairs:
        out.setdefault(int(a), []).append(int(b))
        out.setdefault(int(b), []).append(int(a))
    return {w: sorted(set(v)) for w, v in out.items()}


def other_rung_same_centre(centers, secondary_centers, lambdas) -> dict:
    """window -> other windows at the same (CV1, CV2) centre on a different, known rung."""
    n = min(len(centers), len(secondary_centers))
    key = [(round(float(centers[w]), _CENTRE_DECIMALS), round(float(secondary_centers[w]), _CENTRE_DECIMALS))
           for w in range(n)]
    lam = [(_finite(lambdas[w]) if w < len(lambdas) else None) for w in range(n)]
    out = {}
    for w in range(n):
        out[w] = [] if lam[w] is None else [
            j for j in range(n)
            if j != w and key[j] == key[w] and lam[j] is not None and lam[j] != lam[w]]
    return out


__all__ = ["NEIGHBOUR_SLACK", "best_neighbour_overlaps", "overlap_by_pair_2d",
           "spatial_neighbour_pairs", "same_rung_neighbours", "other_rung_same_centre"]
