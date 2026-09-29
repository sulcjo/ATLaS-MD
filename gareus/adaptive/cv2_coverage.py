"""Spec 3.3 R2 (coverage hole), post-union only. Pure NumPy.

Inputs are one union's pooled samples (``build_union_state_mbar_inputs`` NPZ: ``cv_A``,
``secondary_cv``, ``sampled_state_ids``) restricted to the representative rung (lambda = 0):
there the Hamiltonians differ only in their umbrella terms, so the reduced energies are
recomputed exactly from the restraints (no boost term), and a small MBAR over those states
alone gives unbiased weights w_n = 1 / sum_k N_k exp(f_k - u_k(x_n)).

A CV1 column = the eligible 2-D states sharing one CV1 centre. Its slab is
|cv1 - c1| <= min(half the gap to the nearest other column, 2 sigma_w1); its CV2 range is
[min(c2 - 2 sigma_w2), max(c2 + 2 sigma_w2)] over the column's windows, widened to the slab's
weighted 1-99 % CV2 range, cut into intervals one median sigma_w2 wide. An interval holding
>= MIN_INTERVAL_WEIGHT (2 %) of the slab's unbiased weight is a hole when EITHER
  * the number of CENTRES contributing >= CONTRIBUTOR_SHARE (10 %) of its weight (p_c = share
    of the interval's weight from samples of centre c; here only lambda = 0 states) is
    < ``coverage_min_windows`` (1 / sum p_c^2 is reported too); or
  * the block-bootstrap sigma of its free energy F = -ln(W_interval / W_slab), f held fixed,
    blocks = BOOT_BLOCKS contiguous row blocks per state, exceeds ``refine_pmf_sigma_kT``.
Adjacent hole intervals form one hole, split at every window centre of the column (a window's
own neighbourhood is never a hole of its own); the new window sits at a piece's centre unless
an existing window is within NEAR_CENTRE_SIGMA (1.5, the healthy spacing) of its own sigma_w2
of it. The outer half of an edge window is therefore not a hole: a piece there has its centre
within 1.5 sigma of that window unless weight extends well beyond it.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gareus.adaptive import cv2_resolution as cr

MIN_INTERVAL_WEIGHT = 0.02
WEIGHT_RANGE = (0.01, 0.99)
NEAR_CENTRE_SIGMA = cr.SPACING_SIGMA
CONTRIBUTOR_SHARE = 0.10
BOOT_BLOCKS = 20
N_BOOT = 100
_BOOT_SEED = 3303


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    top = np.max(a, axis=axis, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return np.squeeze(top, axis=axis) + np.log(np.sum(np.exp(a - top), axis=axis))


def reduced_umbrella(cv1: np.ndarray, cv2: np.ndarray, views: Sequence[cr.StateView], beta: float) -> np.ndarray:
    """(K, N) reduced umbrella energies of ``views`` on every sample (restrained axes only)."""
    u = np.zeros((len(views), cv1.size))
    for k, v in enumerate(views):
        if v.k1 is not None and v.k1 > 0:
            u[k] += 0.5 * v.k1 * (cv1 - v.c1) ** 2
        if v.c2 is not None and v.k2 is not None and v.k2 > 0:
            u[k] += 0.5 * v.k2 * (cv2 - v.c2) ** 2
    return float(beta) * u


def solve_mbar(u_kn: np.ndarray, n_k: np.ndarray, *, tol: float = 1e-8, max_iter: int = 20000) -> np.ndarray:
    """Self-consistent MBAR free energies (f_0 = 0) of the sampled states."""
    log_n = np.log(np.maximum(n_k, 1e-300))
    f = np.zeros(u_kn.shape[0])
    for _ in range(int(max_iter)):
        log_den = _logsumexp(log_n[:, None] + f[:, None] - u_kn, axis=0)
        f_new = -_logsumexp(-u_kn - log_den[None, :], axis=1)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            return f_new
        f = f_new
    return f


def log_weights(u_kn: np.ndarray, n_k: np.ndarray, f: np.ndarray) -> np.ndarray:
    return -_logsumexp(np.log(np.maximum(n_k, 1e-300))[:, None] + f[:, None] - u_kn, axis=0)


def columns(views: Mapping[int, cr.StateView], ids: Sequence[int], settings: cr.ResolutionSettings
            ) -> List[Dict[str, Any]]:
    """Eligible 2-D states grouped by CV1 centre, with slab and CV2 intervals."""
    groups: Dict[float, List[cr.StateView]] = {}
    for s in ids:
        if cr.eligibility(views[s], settings)[0]:
            groups.setdefault(round(views[s].c1, 6), []).append(views[s])
    centres = sorted(groups)
    out = []
    for c in centres:
        members = sorted(groups[c], key=lambda v: float(v.c2))
        sw1 = float(np.median([v.sigma_w(1, settings.rt) for v in members]))
        others = [abs(c - o) for o in centres if o != c]
        half = min(0.5 * min(others), 2.0 * sw1) if others else 2.0 * sw1
        sw2 = [float(v.sigma_w(2, settings.rt)) for v in members]
        lo = min(float(v.c2) - 2 * s for v, s in zip(members, sw2))
        hi = max(float(v.c2) + 2 * s for v, s in zip(members, sw2))
        width = float(np.median(sw2))
        n_int = max(1, int(math.ceil((hi - lo) / width)))
        out.append({"c1": float(members[0].c1), "members": members, "slab_half_width": float(half),
                    "edges": np.linspace(lo, hi, n_int + 1), "sigma_w2": sw2})
    return out


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cdf = np.cumsum(w[order])
    return float(x[order][min(int(np.searchsorted(cdf, q * cdf[-1])), x.size - 1)])


def _extend_to_weight(col: Dict[str, Any], cv1: np.ndarray, cv2: np.ndarray, weights: np.ndarray) -> Dict[str, Any]:
    """Widen the column's CV2 range to its slab's weighted 1-99 % range (same interval width),
    so weight beyond the outermost windows is inspected too."""
    slab = (np.abs(cv1 - col["c1"]) <= col["slab_half_width"]) & (weights > 0)
    edges = col["edges"]
    if not slab.any():
        return col
    width = float(edges[1] - edges[0])
    lo = min(float(edges[0]), _weighted_quantile(cv2[slab], weights[slab], WEIGHT_RANGE[0]))
    hi = max(float(edges[-1]), _weighted_quantile(cv2[slab], weights[slab], WEIGHT_RANGE[1]))
    n_lo = int(math.ceil((float(edges[0]) - lo) / width - 1e-9))
    n_hi = int(math.ceil((hi - float(edges[-1])) / width - 1e-9))
    new = np.concatenate([edges[0] - width * np.arange(n_lo, 0, -1), edges, edges[-1] + width * np.arange(1, n_hi + 1)])
    return {**col, "edges": new}


def _block_ids(state_idx: np.ndarray, n_blocks: int) -> np.ndarray:
    out = np.zeros(state_idx.size, dtype=np.int64)
    for k in np.unique(state_idx):
        idx = np.flatnonzero(state_idx == k)
        length = max(1, int(math.ceil(idx.size / float(n_blocks))))
        out[idx] = k * 10 ** 7 + np.arange(idx.size) // length
    return out


def _boot_sigma(w_int: np.ndarray, w_slab: np.ndarray, block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """sigma of -ln(W_i / W_slab) per interval over a block bootstrap stratified by state
    (``block`` = state * 10**7 + block index, see ``_block_ids``); inf when > 10 % of the
    replicates leave the interval empty."""
    uniq, inv = np.unique(block, return_inverse=True)
    bi = np.zeros((uniq.size, w_int.shape[0]))
    np.add.at(bi, inv, w_int.T)
    bs = np.bincount(inv, weights=w_slab, minlength=uniq.size)
    sob = uniq // 10 ** 7
    strata = [np.flatnonzero(sob == k) for k in np.unique(sob)]
    reps = []
    for _ in range(N_BOOT):
        pick = np.concatenate([rng.choice(idx, size=idx.size) for idx in strata])
        tot, part = bs[pick].sum(), bi[pick].sum(axis=0)
        with np.errstate(divide="ignore"):
            reps.append(-np.log(part / tot) if tot > 0 else np.full(part.shape, np.inf))
    reps = np.asarray(reps)
    finite = np.isfinite(reps)
    with np.errstate(invalid="ignore"):
        spread = np.nanstd(np.where(finite, reps, np.nan), axis=0)
    return np.where(finite.mean(axis=0) >= 0.9, spread, np.inf)


def _interval_stats(col, cv1, cv2, state_idx, weights, centre_of_state, rng):
    slab = np.abs(cv1 - col["c1"]) <= col["slab_half_width"]
    edges = col["edges"]
    bins = np.clip(np.searchsorted(edges, cv2, side="right") - 1, 0, edges.size - 2)
    inside = slab & (cv2 >= edges[0]) & (cv2 <= edges[-1])
    w_slab = np.where(slab, weights, 0.0)
    n_int = edges.size - 1
    w_int = np.zeros((n_int, cv1.size))
    for i in range(n_int):
        w_int[i] = np.where(inside & (bins == i), weights, 0.0)
    total = w_slab.sum()
    frac = w_int.sum(axis=1) / total if total > 0 else np.zeros(n_int)
    n_eff, n_contrib = np.zeros(n_int), np.zeros(n_int)
    for i in range(n_int):
        wi = w_int[i]
        if wi.sum() <= 0:
            continue
        share = np.bincount(centre_of_state[state_idx], weights=wi) / wi.sum()
        n_eff[i] = 1.0 / float(np.sum(share ** 2))
        n_contrib[i] = float(np.sum(share >= CONTRIBUTOR_SHARE))
    sigma = _boot_sigma(w_int, w_slab, _block_ids(state_idx, BOOT_BLOCKS), rng)
    return frac, n_eff, n_contrib, sigma


def _hole_runs(flags: np.ndarray, edges: np.ndarray, centres: Sequence[float]) -> List[Tuple[int, int]]:
    """Runs of flagged intervals; an interval holding a window centre ends a run and is
    never part of one."""
    holds = [any(edges[i] <= c < edges[i + 1] for c in centres) for i in range(len(flags))]
    runs, start = [], None
    for i, flag in enumerate(list(flags) + [False]):
        usable = bool(flag) and (i >= len(holds) or not holds[i])
        if usable and start is None:
            start = i
        elif not usable and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def _hole_candidate(col, i0, i1, frac, n_eff, n_contrib, sigma, settings) -> Dict[str, Any]:
    edges = col["edges"]
    z = 0.5 * (edges[i0] + edges[i1 + 1])
    dist = [abs(float(v.c2) - z) for v in col["members"]]
    j = int(np.argmin(dist))
    parent, sw = col["members"][j], col["sigma_w2"][j]
    metrics = {"c1": col["c1"], "interval": [float(edges[i0]), float(edges[i1 + 1])],
               "weight_fraction": float(frac[i0:i1 + 1].sum()),
               "n_eff_windows": float(np.min(n_eff[i0:i1 + 1])),
               "n_contributing_windows": float(np.min(n_contrib[i0:i1 + 1])), "pmf_sigma_kT": float(np.max(sigma[i0:i1 + 1])),
               "nearest_window": parent.state_id, "nearest_distance": dist[j]}
    ids = [parent.state_id]
    if dist[j] < NEAR_CENTRE_SIGMA * sw:
        return cr.new_candidate("R2", "interval", ids, "no_action", "existing_window_near", metrics=metrics)
    f2 = cr.f2_under_bias(parent.var2, parent.k2, settings.temperature_k)
    child = {"primary_center": parent.c1, "primary_k": float(parent.k1), "secondary_center": float(z),
             "centred_on": "interval_centre", **cr.child_spring(dist[j] / cr.SPACING_SIGMA, f2, float(parent.k2),
                                                                 settings)}
    proposal = {"parent_state_id": parent.state_id, "children": [child]}
    reason = (f"CV2 interval {metrics['interval'][0]:.4g}..{metrics['interval'][1]:.4g} at CV1 {col['c1']:.4g}: "
              f"{metrics['n_contributing_windows']:.0f} contributing window(s), PMF sigma {metrics['pmf_sigma_kT']:.3g} kT")
    if child["refusal"]:
        return cr.new_candidate("R2", "interval", ids, "refused", f"{child['refusal']}: {reason}",
                                refusal=child["refusal"], metrics=metrics, proposal=proposal)
    return cr.new_candidate("R2", "interval", ids, "proposed", reason, metrics=metrics, proposal=proposal)


def coverage_holes(cv1: np.ndarray, cv2: np.ndarray, state_idx: np.ndarray, weights: np.ndarray,
                   sample_views: Sequence[cr.StateView], column_views: Mapping[int, cr.StateView],
                   settings: cr.ResolutionSettings, *, epoch: int = 0) -> List[Dict[str, Any]]:
    """R2 candidates over every CV1 column of ``column_views`` (active representative-rung
    states with their P4 moments). ``state_idx`` indexes ``sample_views`` (the states of the
    MBAR that produced ``weights``, retired ones included); ``weights`` sum to 1."""
    keys = [(round(v.c1, 6), None if v.c2 is None else round(v.c2, 6)) for v in sample_views]
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    centre_of_state = np.asarray([uniq[k] for k in keys], dtype=np.int64)
    rng = np.random.default_rng([_BOOT_SEED, int(epoch)])
    out: List[Dict[str, Any]] = []
    for col in columns(column_views, sorted(column_views), settings):
        col = _extend_to_weight(col, cv1, cv2, weights)
        frac, n_eff, n_contrib, sigma = _interval_stats(col, cv1, cv2, state_idx, weights, centre_of_state, rng)
        flags = (frac >= MIN_INTERVAL_WEIGHT) & ((n_contrib < float(settings.coverage_min_windows))
                                                 | (sigma > float(settings.refine_pmf_sigma_kT)))
        for i0, i1 in _hole_runs(flags, col["edges"], [float(v.c2) for v in col["members"]]):
            out.append(_hole_candidate(col, i0, i1, frac, n_eff, n_contrib, sigma, settings))
    return out


__all__ = ["columns", "coverage_holes", "log_weights", "reduced_umbrella", "solve_mbar"]
