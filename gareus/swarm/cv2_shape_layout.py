"""Shape-based 2-D swarm layout (spec 3.2 + X7; ``--swarm-cv2-layout shape``).

Per CV1 window the CV2 samples of the swarm (weighted by a CV1 kernel of that window's own
width) are fitted by ``gareus.adaptive.cv2_shape.fit_cv2_mixture`` and CV2 centres are placed
by ``place_cv2_centres`` inside the column's rung-reweighted support, clipped to today's
global envelope (``ladder_design.reweighted_cv2_centers``). The swarm is a design measure,
not an equilibrium conditional: it is used only as an upper bound on resolution and to
locate candidate structure.

Budget (sparse fill, in this order): the mandatory stacks (unrestrained anchor, one CV1-axis
representative per region -- judged against the full cap), then the P1 reserve is held back,
then ranked requests:

1. mode cells (a joint cell at every accepted mode of every column) and X7 CV1-free windows,
   one per accepted CV2 mode (deduplicated across columns and against the uniform CV2 rows);
2. the connectivity states of today's sparse layout: the remaining CV1-axis windows, then
   the uniform CV2-axis rows;
3. fill cells, greedily: a cell is eligible once it is adjacent (in its column's CV2 order)
   to a granted cell of that column (any cell of a column with none), and the eligible cell
   owning the largest share of the design measure goes next.

The share a cell "owns" (swarm frames in its CV2 Voronoi interval x its column's share of the
kernel mass) stands in for its contribution to the PMF variance; adjacency stands in for its
contribution to connectivity; narrowness plays no part. When every mode and fill request fits,
the layout is ``joint`` (all cells, plus the X7 windows, no other axis states), as the uniform
joint grid. Roles stay the layout's own (``joint``, ``axis``); provenance -- per-column fits,
placements, every request with rank, granted flag and reason -- is the plan's ``cv2_shape``
record. Dropped requests never become states.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gareus.adaptive.cv2_shape import (CV2MixtureFit, CV2Placement, MixtureComponent, estimate_f2, fit_cv2_mixture,
                                       internal_barriers_kT, place_cv2_centres, predicted_overlaps,
                                       predicted_sampled_sigma, shape_rule_k2)
from gareus.adaptive.reserve_budget import layout_reserve_record
from gareus.swarm.ladder_design import (LAYOUT_PLAN_VERSION, LAYOUT_STATUS_INSUFFICIENT, LAYOUT_STATUS_PROPOSED,
                                        R_KJ_PER_MOL_K, ROLE_AXIS, ROLE_JOINT, ROLE_REGION_REPRESENTATIVE,
                                        ROLE_UNRESTRAINED_ANCHOR, _check_layout_invariants, _weighted_quantile,
                                        reserve_fill_cap, window_sigma_cv)

KERNEL_CUTOFF_SIGMA = 3.0      # CV1 kernel truncated beyond this many window widths
MODE_DEDUP_SIGMA = 0.5         # an X7 mode window within this x its sampled sigma of another is a duplicate
BRIDGE_MIN_PREDICTED_OVERLAP = 0.15   # bridge / chain below this predicted adjacent overlap: granted, but warned
BRIDGE_MAX_INTERNAL_BARRIER_KT = 1.0  # a granted fill straddling a barrier above this inside its own window: warned
CV1_LOOSE_FACTOR = 2.0                # a column whose CV1 window is this x wider than the median is CV1-loose
CHAIN_MIN_GAIN = 0.05                 # the chain is granted only if it raises the CV1-free bottleneck by this
PREDICTED_OVERLAP_NOTE = ("predicted overlaps are an UPPER bound: equilibrium overlap on the swarm design profile, "
                          "blind to within-window trapping (internal_barrier_kT) and to swarm-vs-production "
                          "differences (2026-10-03 review: production CV2 sd ~0.72x the swarm's)")
SHAPE_LAYOUT_NOTE = ("the swarm is a design measure (short seeded runs), not an equilibrium conditional; "
                     "its mixture only locates structure and bounds resolution from above")


@dataclass(frozen=True)
class ShapeColumn:
    index: int
    centre1: float
    k1: float
    sigma_w1: float
    weight_share: float
    support: Tuple[float, float]
    fit: CV2MixtureFit
    placement: CV2Placement
    owned_mass: Tuple[float, ...]

    def as_record(self) -> dict:
        return {"index": self.index, "centre1": self.centre1, "k1": self.k1, "sigma_w1": self.sigma_w1,
                "weight_share": self.weight_share, "support": list(self.support), "fit": self.fit.as_record(),
                "placement": self.placement.as_record(), "owned_mass": list(self.owned_mass)}


def column_kernel(cv1, centre1: float, sigma_w1: float, *, cutoff: float = KERNEL_CUTOFF_SIGMA) -> np.ndarray:
    """Gaussian CV1 kernel of the window's own width, zero beyond ``cutoff`` widths."""
    x = (np.asarray(cv1, dtype=float) - float(centre1)) / float(sigma_w1)
    w = np.exp(-0.5 * x ** 2)
    w[~np.isfinite(w) | (np.abs(x) > float(cutoff))] = 0.0
    return w


def column_support(z2, deltav_kj, weights, temperature_k: float, lambdas, *, lo_q: float = 0.02,
                   hi_q: float = 0.98) -> Optional[Tuple[float, float]]:
    """Union over rungs of the kernel- and boost-weighted CV2 quantiles (``reweighted_cv2_centers``
    per column); None when the column has too little data."""
    z, dv, w = (np.asarray(a, dtype=float) for a in (z2, deltav_kj, weights))
    keep = np.isfinite(z) & np.isfinite(dv) & (w > 0)
    if int(keep.sum()) < 2:
        return None
    z, dv, w = z[keep], dv[keep], w[keep]
    beta = 1.0 / (R_KJ_PER_MOL_K * float(temperature_k))
    bounds = []
    for lam in lambdas:
        wl = w * np.exp(-beta * float(lam) * (dv - dv.min()))
        bounds.append((_weighted_quantile(z, wl, lo_q), _weighted_quantile(z, wl, hi_q)))
    return min(b[0] for b in bounds), max(b[1] for b in bounds)


def owned_mass(centres: Sequence[float], z2, weights) -> Tuple[float, ...]:
    """Share of the column's kernel-weighted frames in each centre's CV2 Voronoi interval."""
    c = np.asarray(centres, dtype=float)
    z, w = np.asarray(z2, dtype=float), np.asarray(weights, dtype=float)
    keep = np.isfinite(z) & (w > 0)
    total = float(w[keep].sum())
    if c.size == 0 or total <= 0:
        return tuple(0.0 for _ in c)
    cuts = 0.5 * (c[:-1] + c[1:])
    owner = np.searchsorted(cuts, z[keep])
    return tuple(float(x) / total for x in np.bincount(owner, weights=w[keep], minlength=c.size))


def design_columns(*, cv1, z2, member_ids, deltav_kj, centers1, ks1, lambdas, temperature_k: float,
                   envelope: Tuple[float, float], sigma_w_target: float, k_min: float, k_max: float,
                   spacing_sigma: float, min_mode_members: int, max_components: int = 3) -> List[ShapeColumn]:
    """One ``ShapeColumn`` (fit + placement) per CV1 window centre."""
    lo, hi = float(min(envelope)), float(max(envelope))
    kernels = [column_kernel(cv1, c, window_sigma_cv(float(k), temperature_k)) for c, k in zip(centers1, ks1)]
    mass = np.array([float(w.sum()) for w in kernels])
    shares = mass / mass.sum() if mass.sum() > 0 else np.zeros_like(mass)
    columns = []
    for i, (c1, k1, w) in enumerate(zip(centers1, ks1, kernels)):
        fit = fit_cv2_mixture(z2, member_ids, w, max_components=max_components, min_mode_members=min_mode_members)
        sup = column_support(z2, deltav_kj, w, temperature_k, lambdas)
        sup = (max(lo, sup[0]), min(hi, sup[1])) if sup is not None and min(hi, sup[1]) > max(lo, sup[0]) else (lo, hi)
        pl = place_cv2_centres(fit, sup, sigma_w_target=sigma_w_target, temperature_k=temperature_k, k_min=k_min,
                               k_max=k_max, spacing_sigma=spacing_sigma)
        columns.append(ShapeColumn(i, float(c1), float(k1), window_sigma_cv(float(k1), temperature_k),
                                   float(shares[i]), sup, fit, pl, owned_mass(pl.centres, z2, w)))
    return columns


def mode_axis_windows(columns: Sequence[ShapeColumn], uniform_centres2, *, temperature_k: float,
                      sigma_w_target: float, k_min: float, k_max: float) -> List[dict]:
    """X7: one CV1-free window per accepted CV2 mode, deduplicated (within ``MODE_DEDUP_SIGMA`` x
    its predicted sampled sigma of a uniform CV2 row or an already kept mode window)."""
    cands = []
    for col in columns:
        lo, hi = col.placement.envelope
        for comp in col.fit.accepted_components:
            if lo <= comp.mean <= hi:
                f2 = estimate_f2(comp.variance_curvature, col.fit.pooled_variance, comp.n_members, temperature_k)
                k2 = shape_rule_k2(sigma_w_target, f2, temperature_k, k_min, k_max)
                cands.append({"center2": float(comp.mean), "k2": k2, "f2": f2, "source_column": col.index,
                              "sampled_sigma": predicted_sampled_sigma(k2, f2, temperature_k),
                              "score": float(comp.weight * col.weight_share)})
    cands.sort(key=lambda r: (-r["score"], r["center2"]))
    kept: List[dict] = []
    taken = [float(z) for z in uniform_centres2]
    for r in cands:
        if all(abs(r["center2"] - z) >= MODE_DEDUP_SIGMA * r["sampled_sigma"] for z in taken):
            kept.append(r)
            taken.append(r["center2"])
    return kept


# --- bridge sets (spec 2026-10-03-cv2-bridge-sets-design) --------------------------------

def _near_duplicate(z: float, anchor: float, sigma: float) -> bool:
    return abs(float(z) - float(anchor)) < MODE_DEDUP_SIGMA * float(sigma)


def _placement_bridges(placement: CV2Placement, fit: CV2MixtureFit, temperature_k: float) -> List[dict]:
    """The runs of fills strictly between consecutive ``mode`` anchors of one placement. A fill
    within ``MODE_DEDUP_SIGMA`` x its sampled sigma of either anchor (the step rule's last fill can
    land on the next anchor) is a near duplicate: excluded from the set and never granted."""
    kinds = list(placement.kinds)
    anchors = [i for i, k in enumerate(kinds) if k == "mode"]
    out = []
    for i, j in zip(anchors[:-1], anchors[1:]):
        inner = list(range(i + 1, j))
        dup = [q for q in inner if _near_duplicate(placement.centres[q], placement.centres[i], placement.sampled_sigma[q])
               or _near_duplicate(placement.centres[q], placement.centres[j], placement.sampled_sigma[q])]
        fills = [q for q in inner if q not in dup]
        if not fills:
            continue
        cs = [placement.centres[i], *[placement.centres[q] for q in fills], placement.centres[j]]
        ks = [placement.k2[i], *[placement.k2[q] for q in fills], placement.k2[j]]
        ov = predicted_overlaps(fit, cs, ks, temperature_k)
        barriers = internal_barriers_kT(fit, cs[1:-1], ks[1:-1], temperature_k)
        out.append({"lower_position": i, "upper_position": j, "fill_positions": fills, "near_duplicate_positions": dup,
                    "centres": [float(placement.centres[q]) for q in fills],
                    "k2": [float(placement.k2[q]) for q in fills], "n_fills": len(fills),
                    "predicted_overlaps": ov, "predicted_min_overlap": float(min(ov)) if ov else None,
                    "internal_barrier_kT": barriers, "max_internal_barrier_kT": float(max(barriers)) if barriers else 0.0})
    return out


def column_bridge_sets(columns: Sequence[ShapeColumn], temperature_k: float) -> List[dict]:
    """Every column's bridge sets. Score = the column's design-measure density (weight_share /
    sigma_w1: a wide CV1 kernel collects mass by width alone) per fill. A CV1-loose column (CV1
    window wider than ``CV1_LOOSE_FACTOR`` x the median, e.g. at the CV1 k floor) is not eligible:
    its windows barely restrain CV1 and duplicate the CV1-free chain (2026-10-03 review, C2)."""
    widths = [float(c.sigma_w1) for c in columns]
    median = float(np.median(widths)) if widths else 0.0
    sets = []
    for col in columns:
        loose = median > 0 and float(col.sigma_w1) > CV1_LOOSE_FACTOR * median
        density = float(col.weight_share) / float(col.sigma_w1) if col.sigma_w1 > 0 else 0.0
        for b in _placement_bridges(col.placement, col.fit, temperature_k):
            sets.append({"index": len(sets), "column": col.index, "centre1": col.centre1,
                         "score": density / b["n_fills"], "eligible": not loose,
                         "ineligible_reason": "cv1_loose_column" if loose else None, **b})
    return sets


def pooled_fit(columns: Sequence[ShapeColumn]) -> CV2MixtureFit:
    """The columns' mixtures, each weighted by its column's ``weight_share``, as one fit (no new
    EM; replayable from a layout plan): the CV1-free design profile of the barrier chain. It is the
    swarm's stratified design marginal, not the equilibrium CV1 marginal."""
    shares = [max(float(c.weight_share), 0.0) for c in columns]
    total = sum(shares)
    if total <= 0:
        shares, total = [1.0] * len(columns), float(len(columns) or 1)
    comps = tuple(MixtureComponent(c.mean, c.variance, c.weight * sh / total, c.n_members, c.accepted, c.reason,
                                   c.reg_variance, c.variance_curvature)
                  for col, sh in zip(columns, shares) for c in col.fit.components if c.weight * sh > 0)

    def _avg(attr):
        vals = [(sh, float(getattr(col.fit, attr))) for col, sh in zip(columns, shares)
                if math.isfinite(float(getattr(col.fit, attr)))]
        w = sum(v[0] for v in vals)
        return sum(a * b for a, b in vals) / w if w > 0 else float("nan")
    return CV2MixtureFit(comps, len(comps), {}, 0.0, {}, 0, sum(c.fit.n_frames for c in columns),
                         max((c.fit.n_members for c in columns), default=0),
                         sum(c.fit.weight_total for c in columns), _avg("pooled_mean"), _avg("pooled_variance"),
                         max((c.fit.min_mode_members for c in columns), default=0), ("pooled over columns",))


def _cv1_free_bottleneck(fit, windows: Sequence[Tuple[float, float]], temperature_k: float) -> Optional[float]:
    """Min adjacent predicted overlap along the CV1-free windows sorted by centre (1-D: the widest path)."""
    ws = sorted((float(z), float(k)) for z, k in windows)
    if len(ws) < 2:
        return None
    ov = predicted_overlaps(fit, [w[0] for w in ws], [w[1] for w in ws], temperature_k)
    return float(min(ov)) if ov else None


def barrier_chain(columns: Sequence[ShapeColumn], envelope: Tuple[float, float], *, temperature_k: float,
                  sigma_w_target: float, k_min: float, k_max: float, spacing_sigma: float,
                  existing: Sequence[Tuple[float, float]] = ()) -> dict:
    """The CV1-free chain across the pooled profile's barriers: the fills between consecutive pooled
    anchors, minus any within ``MODE_DEDUP_SIGMA`` sampled sigma of an existing CV1-free window
    (``existing`` = uniform rows + X7 as (centre2, k2)). Proposed only if it raises the CV1-free
    bottleneck (rows + X7 [+ chain], predicted on the pooled profile) by >= ``CHAIN_MIN_GAIN``."""
    fit = pooled_fit(columns)
    pl = place_cv2_centres(fit, envelope, sigma_w_target=sigma_w_target, temperature_k=temperature_k, k_min=k_min,
                           k_max=k_max, spacing_sigma=spacing_sigma)
    bridges = _placement_bridges(pl, fit, temperature_k)
    cands = [(z, k, sig) for b in bridges for z, k in zip(b["centres"], b["k2"])
             for sig in [float(pl.sampled_sigma[list(pl.centres).index(z)])]]
    kept = [(z, k) for z, k, sig in cands if not any(_near_duplicate(z, e, sig) for e, _k in existing)]
    before = _cv1_free_bottleneck(fit, existing, temperature_k)
    after = _cv1_free_bottleneck(fit, [*existing, *kept], temperature_k)
    gain = (after - before) if after is not None and before is not None else after
    useful = bool(kept) and (before is None or (gain is not None and gain >= CHAIN_MIN_GAIN))
    barriers = internal_barriers_kT(fit, [z for z, _ in kept], [k for _, k in kept], temperature_k) if kept else []
    rec = {"anchors": [float(pl.centres[i]) for i, k in enumerate(pl.kinds) if k == "mode"],
           "candidates": [float(z) for z, _k, _s in cands], "deduplicated": len(cands) - len(kept),
           "cv1_free_bottleneck_without": before, "cv1_free_bottleneck_with": after, "gain": gain,
           "min_gain": CHAIN_MIN_GAIN, "internal_barrier_kT": barriers,
           "centres": [float(z) for z, _ in kept] if useful else [], "k2": [float(k) for _, k in kept] if useful else [],
           "predicted_min_overlap": after if useful else None}
    if not useful:
        rec["not_proposed"] = "no_barrier" if not cands else ("all_duplicates" if not kept else "no_gain")
    return rec


# --- requests and the budget -------------------------------------------------------------

def _cv2_table(columns, uniform_centres2, uniform_ks2, modes, chain=None) -> Tuple[List[float], List[float], dict]:
    """One global CV2 (centre, k2) list: uniform rows, X7 mode windows, then every column's
    centres. Cells index it; returns (centres2, ks2, index) with index[(kind, a, b)] -> j."""
    centres2 = [float(z) for z in uniform_centres2]
    ks2 = [float(k) for k in uniform_ks2]
    index: Dict[tuple, int] = {("row", j, None): j for j in range(len(centres2))}
    for m, r in enumerate(modes):
        index[("mode_axis", m, None)] = len(centres2)
        centres2.append(r["center2"]); ks2.append(r["k2"])
    for col in columns:
        for p, (z, k) in enumerate(zip(col.placement.centres, col.placement.k2)):
            index[("cell", col.index, p)] = len(centres2)
            centres2.append(float(z)); ks2.append(float(k))
    for m, (z, k) in enumerate(zip((chain or {}).get("centres", []), (chain or {}).get("k2", []))):
        index[("barrier_axis", m, None)] = len(centres2)
        centres2.append(float(z)); ks2.append(float(k))
    return centres2, ks2, index


def _requests(columns, modes, index, n_uniform: int, reps: Sequence[int], n1: int) -> Tuple[list, list, list]:
    """(mode-tier requests, connectivity requests, fill requests); each a dict with ``cell``."""
    tier_mode, tier_fill = [], []
    for col in columns:
        pl = col.placement
        for p, (z, kind) in enumerate(zip(pl.centres, pl.kinds)):
            req = {"kind": "mode" if kind == "mode" else "fill", "placement_kind": kind, "column": col.index,
                   "position": p, "cell": (col.index, index[("cell", col.index, p)]), "center1": col.centre1,
                   "center2": float(z), "k2": float(pl.k2[p]), "score": float(col.owned_mass[p] * col.weight_share)}
            (tier_mode if kind == "mode" else tier_fill).append(req)
    for m, r in enumerate(modes):
        tier_mode.append({"kind": "axis2_mode", "column": r["source_column"], "position": None,
                          "cell": (None, index[("mode_axis", m, None)]), "center1": None, "center2": r["center2"],
                          "k2": r["k2"], "score": r["score"]})
    tier_mode.sort(key=lambda q: (-q["score"], q["cell"][0] is None, q["cell"][1]))
    connect = [{"kind": "axis1", "column": i, "position": None, "cell": (i, None), "center1": None, "center2": None,
                "k2": 0.0, "score": None} for i in range(n1) if i not in set(reps)]
    connect += [{"kind": "axis2_row", "column": None, "position": None, "cell": (None, j), "center1": None,
                 "center2": None, "k2": None, "score": None} for j in range(n_uniform)]
    return tier_mode, connect, tier_fill


def _grant(req: dict, rank: int, tier: str, cells: list, roles: list) -> int:
    req.update({"granted": True, "rank": rank, "tier": tier, "reason": "granted"})
    cells.append(req["cell"])
    roles.append(ROLE_JOINT if req["cell"][0] is not None and req["cell"][1] is not None else ROLE_AXIS)
    return rank + 1


def _greedy_fill(fill: list, granted_positions: Dict[int, set], budget: int, rank: int, cells, roles) -> int:
    """Grant fill cells by design-mass share, each adjacent to a granted cell of its column."""
    pending = list(fill)
    while budget > 0 and pending:
        eligible = [q for q in pending if not granted_positions.get(q["column"])
                    or {q["position"] - 1, q["position"] + 1} & granted_positions[q["column"]]]
        if not eligible:
            break
        best = max(eligible, key=lambda q: (q["score"], -q["column"], -q["position"]))
        rank = _grant(best, rank, "fill", cells, roles)
        granted_positions.setdefault(best["column"], set()).add(best["position"])
        pending.remove(best)
        budget -= 1
    return rank


def _grant_bridge_tiers(tier_mode, connect, fill, chain_reqs, sets, fill_cap, cells, roles) -> int:
    """The sparse fill with bridge sets: modes, the barrier chain (atomic), connectivity, whole
    bridge sets by score (a set that does not fit is skipped for the next), then tails."""
    rank = 0
    granted_positions: Dict[int, set] = {}
    for q in tier_mode:
        if len(cells) >= fill_cap:
            break
        rank = _grant(q, rank, "mode", cells, roles)
        if q["kind"] == "mode":
            granted_positions.setdefault(q["column"], set()).add(q["position"])
    if chain_reqs:
        if len(chain_reqs) <= fill_cap - len(cells):
            for q in chain_reqs:
                rank = _grant(q, rank, "barrier_chain", cells, roles)
        else:
            for q in chain_reqs:
                q.update({"granted": False, "rank": None, "tier": None, "reason": "barrier_chain_does_not_fit"})
    for q in connect:
        if len(cells) >= fill_cap:
            break
        rank = _grant(q, rank, "connectivity", cells, roles)
    by_pos = {(q["column"], q["position"]): q for q in fill}
    for bs in sorted(sets, key=lambda b: (-b["score"], b["column"], b["lower_position"])):
        members = [by_pos[(bs["column"], p)] for p in bs["fill_positions"]]
        if not bs.get("eligible", True):
            bs.update({"granted": False, "reason": bs.get("ineligible_reason") or "ineligible"})
            for q in members:
                q.update({"granted": False, "rank": None, "tier": None, "reason": bs["reason"]})
            continue
        if len(members) <= fill_cap - len(cells):
            for q in members:
                rank = _grant(q, rank, "bridge_set", cells, roles)
                granted_positions.setdefault(q["column"], set()).add(q["position"])
            bs.update({"granted": True, "reason": "granted"})
        else:
            bs.update({"granted": False, "reason": "bridge_set_does_not_fit"})
            for q in members:
                q.update({"granted": False, "rank": None, "tier": None, "reason": "bridge_set_does_not_fit"})
    tails = [q for q in fill if q.get("bridge_set") is None]
    return _greedy_fill(tails, granted_positions, fill_cap - len(cells), rank, cells, roles)


def design_shape_layout(columns: Sequence[ShapeColumn], modes: Sequence[dict], uniform_centres2, uniform_ks2, *,
                        n_rungs: int, max_replicas: int, region_centre_indices=None,
                        reserve_fraction: float = 0.0, bridge_sets: Optional[List[dict]] = None,
                        chain: Optional[dict] = None) -> Tuple[dict, List[float], List[float]]:
    """``(plan, centres2, ks2)``: a ``design_exploration_layout``-shaped plan (cells index
    ``centres2``/``ks2``; the CV1 index is the column) plus its ``cv2_shape`` record."""
    if int(max_replicas) <= 0:
        raise ValueError("max_replicas must be set and positive; the default 0 cannot size a 2-D ladder")
    n1 = len(columns)
    cap = int(max_replicas) // int(n_rungs)
    reps = sorted({int(i) for i in (region_centre_indices or [])})
    mandatory = 1 + len(reps)
    bridges_on = bridge_sets is not None
    centres2, ks2, index = _cv2_table(columns, uniform_centres2, uniform_ks2, modes, chain if bridges_on else None)
    base = {"version": LAYOUT_PLAN_VERSION, "n1": n1, "n2": len(centres2), "n_rungs": int(n_rungs), "cap_spatial": cap,
            "mandatory_spatial": mandatory}
    if mandatory > cap:
        return ({**base, "status": LAYOUT_STATUS_INSUFFICIENT, "kind": None, "spatial_states": 0, "cells": [],
                 "state_roles": [], "requested_replicas": mandatory * int(n_rungs),
                 "available_replicas": int(max_replicas),
                 "reason": f"{mandatory} mandatory rung stacks ({mandatory * int(n_rungs)} replicas) exceed the "
                           f"cap of {int(max_replicas)} replicas / {cap} spatial states"}, centres2, ks2)
    cells: list = [(None, None)] + [(i, None) for i in reps]
    roles: list = [ROLE_UNRESTRAINED_ANCHOR] + [ROLE_REGION_REPRESENTATIVE] * len(reps)
    fill_cap = reserve_fill_cap(cap, mandatory, max_replicas, n_rungs, reserve_fraction)
    tier_mode, connect, fill = _requests(columns, modes, index, len(uniform_centres2), reps, n1)
    chain_reqs: list = []
    if bridges_on:
        member = {(bs["column"], p): bs["index"] for bs in bridge_sets for p in bs["fill_positions"]}
        dups = {(bs["column"], p) for bs in bridge_sets for p in bs.get("near_duplicate_positions", ())}
        for q in fill:
            q["bridge_set"] = member.get((q["column"], q["position"]))
            if (q["column"], q["position"]) in dups:
                q.update({"bridge_set": "near_duplicate", "granted": False, "rank": None, "tier": None,
                          "reason": "near_duplicate"})
        chain_reqs = [{"kind": "barrier_axis", "column": None, "position": m,
                       "cell": (None, index[("barrier_axis", m, None)]), "center1": None, "center2": float(z),
                       "k2": float(k), "score": None, "bridge_set": None}
                      for m, (z, k) in enumerate(zip(chain.get("centres", []), chain.get("k2", [])))]
    budget = fill_cap - len(cells)
    rank = 0
    if len(tier_mode) + len(fill) + len(chain_reqs) <= budget:
        kind, requests = "joint", tier_mode + fill + chain_reqs
        for q in sorted(tier_mode + fill, key=lambda q: (q["cell"][0] is None, q["column"], q["position"] or 0)):
            rank = _grant(q, rank, "joint", cells, roles)
        for q in chain_reqs:
            rank = _grant(q, rank, "joint", cells, roles)
        for bs in bridge_sets or []:
            bs.update({"granted": True, "reason": "granted"})
    elif bridges_on:
        kind, requests = "sparse", tier_mode + chain_reqs + connect + fill
        rank = _grant_bridge_tiers(tier_mode, connect, fill, chain_reqs, bridge_sets, fill_cap, cells, roles)
    else:
        kind, requests = "sparse", tier_mode + connect + fill
        granted_positions: Dict[int, set] = {}
        for tier, group in (("mode", tier_mode), ("connectivity", connect)):
            for q in group:
                if len(cells) >= fill_cap:
                    break
                rank = _grant(q, rank, tier, cells, roles)
                if q["kind"] == "mode":
                    granted_positions.setdefault(q["column"], set()).add(q["position"])
        rank = _greedy_fill(fill, granted_positions, fill_cap - len(cells), rank, cells, roles)
    for q in requests:
        q.setdefault("granted", False)
        if not q["granted"]:
            q.update({"rank": None, "tier": None, "reason": q.get("reason") or "cap"})
        q["cell"] = [q["cell"][0], q["cell"][1]]
    plan = {**base, "status": LAYOUT_STATUS_PROPOSED, "kind": kind, "spatial_states": len(cells), "cells": cells,
            "state_roles": roles, "mandatory_spatial_indices": list(range(mandatory)),
            "region_representative_centre_indices": reps,
            "cv2_shape": {"layout": "shape", "note": SHAPE_LAYOUT_NOTE, "columns": [c.as_record() for c in columns],
                          "mode_axis_windows": list(modes), "requests": requests,
                          "n_granted": sum(1 for q in requests if q["granted"]),
                          "n_dropped": sum(1 for q in requests if not q["granted"])}}
    if bridges_on:
        granted_chain = bool(chain_reqs) and all(q["granted"] for q in chain_reqs)
        for bs in bridge_sets:
            bs.setdefault("granted", False)
            bs.setdefault("reason", "cap")
        plan["cv2_shape"].update({
            "bridge_sets_enabled": True, "predicted_overlap_note": PREDICTED_OVERLAP_NOTE,
            "bridge_sets": list(bridge_sets),
            "barrier_chain": {**chain, "granted": granted_chain,
                              "reason": ("granted" if granted_chain else
                                         (chain.get("not_proposed", "no_barrier") if not chain_reqs
                                          else chain_reqs[0].get("reason") or "cap"))}})
    if float(reserve_fraction) > 0.0:
        plan["adaptive_reserve"] = layout_reserve_record(float(reserve_fraction), int(max_replicas), int(n_rungs),
                                                         fill_cap_spatial=fill_cap, spatial_states=len(cells))
    _check_layout_invariants(plan)
    return plan, centres2, ks2


def build_shape_pair_layout(*, cv1, z2, member_ids, deltav_kj, centers1, ks1, lambdas, temperature_k: float,
                            uniform_centres2, uniform_ks2, overlap_sigma: float, k_min: float, k_max: float,
                            max_replicas: int, region_centre_indices=None, reserve_fraction: float = 0.0,
                            min_mode_members: int = 8, max_components: int = 3,
                            bridge_sets: bool = False) -> Dict[str, Any]:
    """The one call ``swarm.analyze`` makes under ``--swarm-cv2-layout shape``.

    The base design width is the uniform layout's (spacing / overlap_sigma over the same
    rung-reweighted envelope), so a single-Gaussian column reproduces a uniform-equivalent
    grid. Returns {"layout", "centres2", "ks2", "record", "summary"}; the 3.4 coupling gate
    is applied by the caller, per cell, as for the uniform layout."""
    u = np.asarray(uniform_centres2, dtype=float)
    envelope = (float(u.min()), float(u.max()))
    sigma_t = (envelope[1] - envelope[0]) / max(u.size - 1, 1) / float(overlap_sigma)
    columns = design_columns(cv1=cv1, z2=z2, member_ids=member_ids, deltav_kj=deltav_kj, centers1=centers1,
                             ks1=ks1, lambdas=lambdas, temperature_k=temperature_k, envelope=envelope,
                             sigma_w_target=sigma_t, k_min=k_min, k_max=k_max, spacing_sigma=overlap_sigma,
                             min_mode_members=min_mode_members, max_components=max_components)
    modes = mode_axis_windows(columns, u, temperature_k=temperature_k, sigma_w_target=sigma_t, k_min=k_min,
                              k_max=k_max)
    extra = {}
    if bridge_sets:
        extra = {"bridge_sets": column_bridge_sets(columns, temperature_k),
                 "chain": barrier_chain(columns, envelope, temperature_k=temperature_k, sigma_w_target=sigma_t,
                                        k_min=k_min, k_max=k_max, spacing_sigma=overlap_sigma,
                                        existing=[*zip((float(z) for z in u), (float(k) for k in uniform_ks2)),
                                                  *((float(m["center2"]), float(m["k2"])) for m in modes)])}
    layout, centres2, ks2 = design_shape_layout(columns, modes, u, uniform_ks2, n_rungs=len(lambdas),
                                                max_replicas=max_replicas, region_centre_indices=region_centre_indices,
                                                reserve_fraction=reserve_fraction, **extra)
    record = layout.get("cv2_shape", {})
    record.update({"sigma_w_target": sigma_t, "spacing_sigma": float(overlap_sigma), "envelope": list(envelope),
                   "min_mode_members": int(min_mode_members), "cv2_k_min": float(k_min), "cv2_k_max": float(k_max)})
    summary = {"n_columns": len(columns), "n_accepted_modes": [len(c.fit.accepted_components) for c in columns],
               "n_centres_per_column": [len(c.placement.centres) for c in columns],
               "n_at_k_floor": [c.placement.n_at_k_floor for c in columns], "n_at_compression_floor": [c.placement.n_at_compression_floor for c in columns], "n_mode_axis_windows": len(modes),
               "n_granted": record.get("n_granted"), "n_dropped": record.get("n_dropped"),
               "sigma_w_target": sigma_t, "kind": layout.get("kind"), "warnings": _floor_warnings(columns, k_min)}
    if bridge_sets:
        summary["warnings"] += _bridge_warnings(record)
        summary.update({"n_bridge_sets": len(record.get("bridge_sets", [])),
                        "n_bridge_sets_granted": sum(1 for b in record.get("bridge_sets", []) if b.get("granted")),
                        "barrier_chain_granted": bool((record.get("barrier_chain") or {}).get("granted"))})
    return {"layout": layout, "centres2": centres2, "ks2": ks2, "record": record, "summary": summary}


def _bridge_warnings(record: dict) -> List[str]:
    """Granted bridges / chain whose predicted adjacent overlap on the design profile is weak."""
    out = []
    for bs in record.get("bridge_sets", []):
        m = bs.get("predicted_min_overlap")
        if bs.get("granted") and m is not None and m < BRIDGE_MIN_PREDICTED_OVERLAP:
            out.append(f"cv2 bridge set {bs['index']} (column {bs['column']}) predicted min overlap {m:.3f} < "
                       f"{BRIDGE_MIN_PREDICTED_OVERLAP}")
        b = bs.get("max_internal_barrier_kT") or 0.0
        if bs.get("granted") and b > BRIDGE_MAX_INTERNAL_BARRIER_KT:
            out.append(f"cv2 bridge set {bs['index']} (column {bs['column']}): a fill straddles a {b:.1f} kT barrier "
                       f"inside its own window (> {BRIDGE_MAX_INTERNAL_BARRIER_KT} kT): expect trapping")
    ch = record.get("barrier_chain") or {}
    m = ch.get("predicted_min_overlap")
    if ch.get("granted") and m is not None and m < BRIDGE_MIN_PREDICTED_OVERLAP:
        out.append(f"cv2 barrier chain predicted min overlap {m:.3f} < {BRIDGE_MIN_PREDICTED_OVERLAP}")
    return out


def _floor_warnings(columns: Sequence[ShapeColumn], k_min: float) -> List[str]:
    """Where the landscape is already as narrow as the target width the rule floors k2 at
    cv2_k_min: such windows are restrained only nominally (their means follow the mode)."""
    n = sum(len(c.placement.centres) for c in columns)
    floored = sum(c.placement.n_at_k_floor for c in columns)
    if n and floored > 0.5 * n:
        return [f"cv2 shape layout: {floored} of {n} CV2 centres have k2 at the cv2_k_min floor ({k_min:g}); the "
                "swarm's CV2 is already as narrow as the target width there, so those windows barely restrain CV2 "
                "(see mean_compression in layout_plan.json)"]
    return []


# --- dispatch from swarm.analyze ----------------------------------------------------------

def select_pair_layout(args, n_win: int, n2: int, lambdas, region_rep_indices, uniform_centres2, uniform_ks2,
                       **shape_inputs):
    """``(layout, centres2, ks2, shape)`` for ``swarm.analyze``: today's exploration layout
    (``--swarm-cv2-layout uniform``, the default) or the shape layout; both honour
    ``--swarm-adaptive-reserve-fraction`` (P1). ``shape`` is None for uniform, whose output is
    today's byte for byte at reserve 0 (the reserve keyword is not even passed then)."""
    from gareus.swarm.ladder_design import design_exploration_layout
    max_replicas = int(getattr(args, "max_replicas", 0) or 0)
    reserve = float(getattr(args, "swarm_adaptive_reserve_fraction", 0.0) or 0.0)
    if str(getattr(args, "swarm_cv2_layout", "uniform") or "uniform") == "shape":
        shaped = build_shape_pair_layout(
            lambdas=lambdas, uniform_centres2=uniform_centres2, uniform_ks2=uniform_ks2,
            k_max=float(getattr(args, "cv2_k_max", 1000.0)), max_replicas=max_replicas,
            region_centre_indices=region_rep_indices, reserve_fraction=reserve,
            min_mode_members=int(getattr(args, "swarm_cv2_min_mode_members", 8)),
            **({"bridge_sets": True} if bool(getattr(args, "swarm_cv2_bridge_sets", False)) else {}), **shape_inputs)
        return shaped["layout"], shaped["centres2"], shaped["ks2"], shaped
    layout = design_exploration_layout(int(n_win), n2, n_rungs=len(lambdas), max_replicas=max_replicas,
                                       region_centre_indices=region_rep_indices,
                                       **({"reserve_fraction": reserve} if reserve > 0 else {}))
    return layout, uniform_centres2, uniform_ks2, None


def one_dimensional_reserve(args, n_states: int, warnings: List[str]) -> dict:
    """P1 on a CV1-only ladder (which writes no layout plan and never fills the cap itself):
    {} at reserve 0; else ``{"adaptive_reserve": record}`` in the swarm report, with a warning
    when the ladder leaves fewer free replicas than the reserve asks for."""
    fraction = float(getattr(args, "swarm_adaptive_reserve_fraction", 0.0) or 0.0)
    max_replicas = int(getattr(args, "max_replicas", 0) or 0)
    if fraction <= 0.0:
        return {}
    if max_replicas <= 0:
        warnings.append("adaptive reserve: --max-replicas is 0 (unlimited), so there is no cap to reserve from")
        return {"adaptive_reserve": {"fraction": fraction, "max_replicas": 0, "reserved_replicas_requested": 0}}
    from gareus.adaptive.reserve_budget import reserved_replicas
    requested = reserved_replicas(max_replicas, fraction)
    free = max_replicas - int(n_states)
    if free < requested:
        warnings.append(f"adaptive reserve: the CV1 ladder has {int(n_states)} states, leaving {free} of "
                        f"{max_replicas} replicas free; the reserve asks for {requested}")
    return {"adaptive_reserve": {"fraction": fraction, "max_replicas": max_replicas, "granted_states": int(n_states),
                                 "reserved_replicas_requested": requested, "free_slots": free,
                                 "reserve_shortfall": max(0, requested - free)}}
