"""Connectivity of the UNION window ladder that MBAR is actually asked to solve.

Both axes already guard their own spacing at design time
(:func:`gareus.windows.adaptive_contact_force_constants_kcal` for CV1,
:func:`gareus.cv.adaptive_secondary_force_constants_kcal` for CV2). Each of those
checks one axis, with that epoch's local spacing, at the moment the windows are
created. Neither can see the ladder MBAR is finally handed.

That is the gap chignolin_6 fell through. Its CV2 was redefined mid-campaign, so
states created before the switch carry secondary centres written in one
coordinate and states created after carry centres written in another. Every epoch
looked well spaced on its own. The union did not: the pooled ladder split into
two groups separated by 3.74 sigma, an adjacent overlap of 0.059, and MBAR then
fixed free energies only up to a separate additive constant per component --
which is not a free-energy surface.

The same failure follows from anything that edits centres across epochs: window
recentring, adaptive insertion, state retirement. So the check belongs on the
final registry, not on any single epoch's windows.

WHAT THIS CANNOT SEE, stated plainly because it is the motivating case. Run on
chignolin_6's own registry this module reports ONE connected component, on both
nominal and measured inputs). It does not detect that campaign's disconnection
and is not a
substitute for the per-regime split that does.

The reason is that chignolin_6's split was not geometric. Its CV2 was redefined
mid-campaign, so a secondary centre of +0.380 written under `torsion-pca` and one
of +1.393 written under `tica-linear` are coordinates on two DIFFERENT axes. Any
model built from centres and widths sees two numbers on one line and computes a
perfectly ordinary separation; the mismatch is in the units, not the distance.
Detecting it needs a comparison of the CV MODEL each state was written under -- a
provenance check -- not geometry.

So: use this to validate a ladder whose states all share one CV definition, where
it catches a genuine spacing or force-constant blunder before any GPU time is
spent. Do not read a pass as evidence that a pooled solve is sound. The 0.030 it
reports for chignolin_6 is itself worth noting -- that ladder sat right on the
disconnection floor even by this generous measure.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .cv import MBAR_DISCONNECT_OVERLAP_FLOOR, _implied_neighbor_overlap_from_k

_R_KCAL_MOL_K = 0.00198720425864083

__all__ = [
    "pair_overlap",
    "assess_ladder_connectivity",
    "format_connectivity_report",
]


def pair_overlap(
    c_a: Sequence[float],
    s_a: Sequence[float],
    c_b: Sequence[float],
    s_b: Sequence[float],
    rt_kcal_mol: float,
) -> float:
    """Analytic overlap between two multi-dimensional harmonic windows.

    Positions and WIDTHS are passed directly rather than force constants, because
    the distinction decides whether this catches anything. See
    :func:`assess_ladder_connectivity` for how the widths are obtained.

    Each axis contributes its own 1-D flat-PMF overlap; the axes are treated as
    separable, so the window overlap is their product. Separability is exact for
    a product of independent harmonic restraints and approximate once the
    underlying PMF correlates the axes -- it errs towards declaring MORE overlap
    than a correlated PMF would give, so an edge this rejects is genuinely
    disconnected.

    An axis with a non-positive or non-finite width is unrestrained and
    contributes no separation, i.e. an overlap of 1 on that axis.
    """
    total = 1.0
    for ca, sa, cb, sb in zip(c_a, s_a, c_b, s_b):
        if not all(math.isfinite(v) for v in (ca, cb)):
            return float("nan")
        d = abs(float(ca) - float(cb))
        sa_f, sb_f = float(sa), float(sb)
        if not (math.isfinite(sa_f) and sa_f > 0.0) or not (math.isfinite(sb_f) and sb_f > 0.0):
            continue                      # unrestrained axis: no separation
        if d <= 0.0:
            continue                      # identical centres: full overlap
        sig = 0.5 * (sa_f + sb_f)
        k_eff = rt_kcal_mol / (sig * sig)
        ov = _implied_neighbor_overlap_from_k(k_eff, d, rt_kcal_mol)
        if not math.isfinite(ov):
            return float("nan")
        total *= ov
    return total


def assess_ladder_connectivity(
    states: Iterable[Dict[str, Any]],
    temperature_k: float = 300.0,
    floor: float = MBAR_DISCONNECT_OVERLAP_FLOOR,
    axes: Sequence[Tuple[str, str, str, str]] = (
        ("primary_center", "primary_k", "primary_mean", "primary_sigma"),
        ("secondary_center", "secondary_k", "secondary_mean", "secondary_sigma"),
    ),
) -> Dict[str, Any]:
    """Partition the union ladder into MBAR-connected components.

    Each axis is described by ``(center_key, k_key, mean_key, sigma_key)``. When a
    state carries a measured ``mean``/``sigma`` those are used; otherwise the
    nominal centre and the harmonic width ``sqrt(RT/k)`` stand in.

    Prefer measured widths. Run on chignolin_6's nominal registry this check
    reports ONE connected component with a weakest holding overlap of 0.0308 --
    it passes, on a campaign whose overlap graph demonstrably split. Nominal
    centres are where windows were ASKED to sit: that run's windows drifted up to
    +0.067 off centre, realised widths ran 0.59-1.03x harmonic, and its CV2 was
    redefined mid-campaign so pre- and post-switch centres are not even on a
    common axis. A design-time check on nominal numbers catches gross errors and
    nothing subtler, so it is the floor of what this does, not the point of it.

    Two states are joined when their analytic overlap reaches ``floor``. The
    ladder is solvable as one free-energy surface only if that graph has a single
    component: MBAR determines state free energies up to one additive constant
    PER COMPONENT, so a split ladder yields pieces that cannot be placed on a
    common scale, however well sampled each piece is.

    Returns a dict with ``connected``, ``n_components``, ``components`` (lists of
    state ids), ``weakest_link`` and ``best_between``.

    ``weakest_link`` is the bottleneck: for every state take its STRONGEST edge
    to any other state, then take the smallest of those. It answers "how well
    attached is the most isolated window", which is what decides whether a ladder
    is safe. It is deliberately NOT the smallest edge in the graph -- in any
    ladder of more than a few windows that is some far-apart pair sitting just
    above the floor, which says nothing about connectivity.

    ``best_between`` is the largest overlap bridging two components, i.e. how
    close a disconnected ladder came to holding together.
    """
    rows = list(states)
    n = len(rows)
    rt = _R_KCAL_MOL_K * float(temperature_k)
    ids = [r.get("state_id", i) for i, r in enumerate(rows)]

    used_measured = {"center": 0, "width": 0}

    def centers(r):
        out = []
        for ckey, _k, mkey, _s in axes:
            if r.get(mkey) is not None:
                used_measured["center"] += 1
                out.append(float(r[mkey]))
            else:
                out.append(float(r.get(ckey, 0.0) or 0.0))
        return out

    def widths(r):
        out = []
        for _c, kkey, _m, skey in axes:
            if r.get(skey) is not None:
                used_measured["width"] += 1
                out.append(float(r[skey]))
                continue
            kv = float(r.get(kkey, 0.0) or 0.0)
            out.append(math.sqrt(rt / kv) if kv > 0.0 else 0.0)
        return out

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ov = pair_overlap(centers(rows[i]), widths(rows[i]),
                              centers(rows[j]), widths(rows[j]), rt)
            if math.isfinite(ov):
                edges.append((ov, i, j))
                if ov >= floor:
                    union(i, j)

    groups: Dict[int, List[Any]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(ids[i])
    comps = sorted(groups.values(), key=lambda g: (-len(g), str(g[0])))

    comp_of = {}
    for gi, g in enumerate(comps):
        for sid in g:
            comp_of[sid] = gi

    best_edge = [0.0] * n
    for ov, i, j in edges:
        best_edge[i] = max(best_edge[i], ov)
        best_edge[j] = max(best_edge[j], ov)
    between = [ov for ov, i, j in edges if comp_of[ids[i]] != comp_of[ids[j]]]

    return {
        "n_states": n,
        "connected": len(comps) <= 1,
        "n_components": len(comps),
        "components": comps,
        "floor": float(floor),
        "weakest_link": min(best_edge) if n > 1 else float("nan"),
        "best_between": max(between) if between else float("nan"),
        "used_measured": used_measured["center"] > 0 or used_measured["width"] > 0,
    }


def format_connectivity_report(res: Dict[str, Any]) -> str:
    """One-paragraph human summary, suitable for a preflight log or a warning."""
    basis = "measured" if res.get("used_measured") else "NOMINAL (design-time only)"
    if res["connected"]:
        return (f"ladder OK [{basis}]: {res['n_states']} states form 1 MBAR-connected "
                f"component; most isolated window attaches at overlap "
                f"{res['weakest_link']:.4f} (floor {res['floor']:g})")
    sizes = ", ".join(str(len(c)) for c in res["components"])
    return (f"LADDER DISCONNECTED [{basis}]: {res['n_states']} states split into "
            f"{res['n_components']} components (sizes {sizes}) at overlap floor "
            f"{res['floor']:g}. The best bridge between components is "
            f"{res['best_between']:.4f}. MBAR fixes free energies only up to one "
            f"additive constant per component, so these cannot be placed on a "
            f"common scale. Add bridging windows or widen the offending axis.")
