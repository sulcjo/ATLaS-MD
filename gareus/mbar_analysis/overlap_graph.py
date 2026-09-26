"""Pairwise MBAR overlap graph over the state layout (CV1 centre, CV2 centre, lambda).

The K x K overlap heatmap is laid out in state-index order, which on a 2D
lambda-ladder grid puts CV neighbours and rung partners far apart and hides
where the gaps are. Here every state sits at its restraint centres and rung:

  * edges join neighbouring states -- same-rung spatial neighbours
    (``gareus.layout_neighbours.spatial_neighbour_pairs``) and, for each
    centre, adjacent rungs (lambda_a <-> the next lambda up);
  * each edge carries the PAIRWISE symmetric MBAR state overlap
    ``sqrt(O_ij O_ji)`` (``gareus.mbar_analysis.ladder.pairwise_state_overlap``),
    the energy-space number that also sees rung gaps, which a CV histogram
    cannot (two rungs at one centre overlap ~1 in CV space by construction);
  * the per-sample integrand of that overlap,
    ``sqrt(N_i N_j) * W_ni * W_nj``, is binned on the CV grid, so the density
    shows WHERE in CV space the pair shares configurations and integrates
    back to the edge's overlap exactly (minus samples outside the grid).

Same-rung pairs are drawn on their rung's lambda layer; rung pairs on the
layer halfway between the two rungs (``layer_value``).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from gareus.layout_neighbours import spatial_neighbour_pairs
from gareus.math_helpers import restraint_sigma
from gareus.mbar_analysis.ladder import pair_log_weights

PAIR_KIND_CV = "cv"
PAIR_KIND_RUNG = "rung"
PAIR_KIND_GAP = "gap"     # closest pair joining two same-rung groups the neighbour rule left disconnected
_CENTRE_DECIMALS = 6


def _as_float_array(values: Optional[Sequence[float]], n: int, fill: float) -> np.ndarray:
    if values is None:
        return np.full(n, fill, dtype=float)
    out = np.full(n, fill, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)[:n]
    out[:arr.size] = arr
    return out


def graph_pairs(
    centers: Sequence[float],
    secondary_centers: Sequence[float],
    lambdas: Optional[Sequence[float]],
    k1: Sequence[float],
    k2: Sequence[float],
    temperature_k: float,
) -> list[tuple[int, int, str]]:
    """``(i, j, kind)`` with i < j: same-rung spatial neighbours plus adjacent-rung partners.

    The neighbour rule only joins a state to its nearest states, so a CV gap
    between two groups on a rung produces no edge at all. Each rung's groups
    are therefore joined by their closest pair (kind ``gap``) until the rung is
    one component, so a break in the layout is drawn with its measured overlap
    instead of disappearing.

    A missing secondary centre/k (CV1-only run) is treated as "no CV2
    restraint" (k2 = 0 on that axis), and ``lambdas=None`` as a single rung.
    """
    n = len(centers)
    c1 = _as_float_array(centers, n, math.nan)
    c2 = _as_float_array(secondary_centers, n, math.nan)
    kk2 = _as_float_array(k2, n, math.nan)
    no_cv2 = ~np.isfinite(c2) | ~np.isfinite(kk2)
    c2 = np.where(no_cv2, 0.0, c2)
    kk2 = np.where(no_cv2, 0.0, kk2)
    lam = _as_float_array(lambdas, n, 0.0)
    lam = np.where(np.isfinite(lam), lam, 0.0)

    pairs = {(a, b, PAIR_KIND_CV)
             for a, b in spatial_neighbour_pairs(c1, c2, lam, list(k1), list(kk2), temperature_k)}

    pairs |= _gap_pairs(c1, c2, lam, k1, kk2, temperature_k, pairs)

    by_centre: dict[tuple[float, float], dict[float, list[int]]] = {}
    for w in range(n):
        if not math.isfinite(c1[w]):
            continue
        key = (round(float(c1[w]), _CENTRE_DECIMALS), round(float(c2[w]), _CENTRE_DECIMALS))
        by_centre.setdefault(key, {}).setdefault(round(float(lam[w]), _CENTRE_DECIMALS), []).append(w)
    for rungs in by_centre.values():
        levels = sorted(rungs)
        for lo, hi in zip(levels, levels[1:]):
            for a in rungs[lo]:
                for b in rungs[hi]:
                    pairs.add((min(a, b), max(a, b), PAIR_KIND_RUNG))
    return sorted(pairs)


def _axis_width(k: np.ndarray, pos: np.ndarray, temperature_k: float) -> float:
    widths = [restraint_sigma(float(v), temperature_k) for v in k if math.isfinite(v) and v > 0]
    widths = [w for w in widths if math.isfinite(w)]
    if widths:
        return float(np.median(widths))
    spread = float(np.ptp(pos[np.isfinite(pos)])) if np.any(np.isfinite(pos)) else 0.0
    return spread or 1.0


def _gap_pairs(c1, c2, lam, k1, k2, temperature_k, pairs) -> set:
    """Closest pair between disconnected same-rung groups, one per join, per rung."""
    n = c1.size
    kk1 = _as_float_array(k1, n, math.nan)
    z = np.column_stack([c1 / _axis_width(kk1, c1, temperature_k),
                         c2 / _axis_width(k2, c2, temperature_k)])
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, _kind in pairs:
        parent[find(a)] = find(b)
    out = set()
    rungs = np.round(lam, _CENTRE_DECIMALS)
    for rung in np.unique(rungs):
        members = [w for w in np.flatnonzero(rungs == rung) if math.isfinite(c1[w])]
        while len({find(w) for w in members}) > 1:
            best = None
            for ia, a in enumerate(members):
                for b in members[ia + 1:]:
                    if find(a) == find(b):
                        continue
                    d = float(np.hypot(*(z[a] - z[b])))
                    if best is None or d < best[0]:
                        best = (d, a, b)
            _d, a, b = best
            out.add((int(min(a, b)), int(max(a, b)), PAIR_KIND_GAP))
            parent[find(a)] = find(b)
    return out


def layer_value(lambda_i: float, lambda_j: float) -> float:
    """Lambda height a pair's density is drawn at: its rung, or halfway between two rungs."""
    return 0.5 * (float(lambda_i) + float(lambda_j))


def _rows_by_state(window: np.ndarray):
    order = np.argsort(window, kind="stable")
    sorted_w = window[order]

    def rows_of(state: int) -> np.ndarray:
        lo, hi = np.searchsorted(sorted_w, [state, state + 1])
        return order[lo:hi]

    return rows_of


def pair_overlap_densities(
    u_nk: np.ndarray,
    window: np.ndarray,
    f_k: np.ndarray,
    n_k: np.ndarray,
    cv1: np.ndarray,
    cv2: Optional[np.ndarray],
    pairs: Iterable[tuple[int, int, str]],
    edges1: np.ndarray,
    edges2: Optional[np.ndarray],
) -> list[dict]:
    """Pairwise overlap and its CV-space density for every pair.

    ``overlap`` equals ``pairwise_state_overlap(u_nk, window, f_k, n_k, i, j)``
    (NaN when either state has no samples); ``density`` is the same integrand
    histogrammed on ``edges1`` (x ``edges2`` when ``cv2`` is given).
    Rows are grouped by state once, so the cost is one pass over each pair's
    own samples rather than a full-length mask per pair.
    """
    u_nk = np.asarray(u_nk, dtype=np.float64)
    window = np.asarray(window)
    f_k = np.asarray(f_k, dtype=np.float64).reshape(-1)
    n_k = np.asarray(n_k, dtype=np.float64).reshape(-1)
    cv1 = np.asarray(cv1, dtype=np.float64)
    two_d = cv2 is not None and edges2 is not None
    cv2_arr = np.asarray(cv2, dtype=np.float64) if two_d else None
    shape = (len(edges1) - 1, len(edges2) - 1) if two_d else (len(edges1) - 1,)
    rows_of = _rows_by_state(window)

    out = []
    for i, j, kind in pairs:
        rows = np.concatenate([rows_of(i), rows_of(j)])
        result = {"i": int(i), "j": int(j), "kind": kind, "overlap": math.nan,
                  "density": np.zeros(shape)}
        out.append(result)
        if rows.size == 0 or not (n_k[i] > 0 and n_k[j] > 0):
            continue
        cols = np.array([i, j])
        log_w = pair_log_weights(u_nk[rows[:, None], cols[None, :]], f_k[cols], n_k[cols])
        term = np.exp(log_w[:, 0] + log_w[:, 1]) * math.sqrt(n_k[i] * n_k[j])
        result["overlap"] = float(term.sum())
        keep = np.isfinite(term) & np.isfinite(cv1[rows])
        if two_d:
            keep &= np.isfinite(cv2_arr[rows])
            result["density"] = np.histogram2d(cv1[rows][keep], cv2_arr[rows][keep],
                                               bins=(edges1, edges2), weights=term[keep])[0]
        else:
            result["density"] = np.histogram(cv1[rows][keep], bins=edges1, weights=term[keep])[0]
    return out


def write_pairs_csv(rows: Iterable[dict], path: Path, threshold: float) -> None:
    """One row per edge: i, j, kind (cv|rung|gap), layer lambda, overlap, below_threshold (1/0, blank if NaN)."""
    with Path(path).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["i", "j", "kind", "layer_lambda", "overlap", "below_threshold"])
        for r in rows:
            ov = float(r["overlap"])
            below = "" if not math.isfinite(ov) else str(int(ov < threshold))
            writer.writerow([r["i"], r["j"], r["kind"], f"{float(r['layer']):.6g}",
                             "" if not math.isfinite(ov) else f"{ov:.6g}", below])


__all__ = ["PAIR_KIND_CV", "PAIR_KIND_GAP", "PAIR_KIND_RUNG", "graph_pairs", "layer_value",
           "pair_overlap_densities", "write_pairs_csv"]
