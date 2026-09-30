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
    of the interval's TOTAL weight from samples of centre c, never renormalised; here only
    lambda = 0 states) is < ``coverage_min_windows``. With ``coverage_count`` "same-column"
    (default) only centres of the interval's own column count: a sample state is in column c1
    iff it restrains CV1 above the floor (k1 > max(0, cv1_k_min), as ``columns`` builds them)
    at round(c1, 6) == the column's; retired states count (they produced samples); a
    CV1-unrestrained state (k1 <= 0, placeholder c1, P6) counts in NO column. "any" counts
    every centre, neighbouring columns included (T2 9.6: near-inert on a 2-D grid). Both counts
    are reported (``n_contributing_windows`` = the one used, ``n_contributing_windows_any`` /
    ``_same_column``), and 1 / sum p_c^2 over all centres; or
  * the block-bootstrap sigma of its free energy F = -ln(W_interval / W_slab), f held fixed,
    exceeds ``refine_pmf_sigma_kT``. Blocks (``autocorrelation_block_ids``): per state, the
    statistical inefficiency g (``effective_samples.pooled_inefficiency``, Geyer's initial
    monotone sequence over that state's rows in each source, max over its restrained axes;
    g = 1 when it cannot be estimated, recorded) sets a block length of
    ceil(BOOT_BLOCK_G_MULTIPLE x g) rows, blocks never cross a source, and the length is cut to
    give >= BOOT_MIN_BLOCKS blocks per state when the rows are too few (recorded). Rows are the
    union NPZ's, already thinned by the union builder, so g is in those rows. Replaces the
    fixed BOOT_BLOCKS blocks per state (``_block_ids``, kept for replays).
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
from gareus.adaptive.effective_samples import pooled_inefficiency

MIN_INTERVAL_WEIGHT = 0.02
WEIGHT_RANGE = (0.01, 0.99)
NEAR_CENTRE_SIGMA = cr.SPACING_SIGMA
CONTRIBUTOR_SHARE = 0.10
BOOT_BLOCKS = 20                 # the pre-v3 fixed block count (``_block_ids``; replays only)
BOOT_BLOCK_G_MULTIPLE = cr.BOOT_BLOCK_G_MULTIPLE
BOOT_MIN_BLOCKS = cr.BOOT_MIN_BLOCKS
G_MIN_RUN = 20                   # shortest per-source run used to estimate g (pooled_inefficiency)
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


def solve_mbar(u_kn: np.ndarray, n_k: np.ndarray, *, tol: float = 1e-8, max_iter: int = 20000,
               info: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Self-consistent MBAR free energies (f_0 = 0) of the sampled states. ``info``, when
    given, receives ``converged``, ``iterations`` and the last ``max_delta_f``."""
    log_n = np.log(np.maximum(n_k, 1e-300))
    f = np.zeros(u_kn.shape[0])
    delta, it = float("inf"), 0
    for it in range(1, int(max_iter) + 1):
        log_den = _logsumexp(log_n[:, None] + f[:, None] - u_kn, axis=0)
        f_new = -_logsumexp(-u_kn - log_den[None, :], axis=1)
        f_new -= f_new[0]
        delta = float(np.max(np.abs(f_new - f)))
        f = f_new
        if delta < tol:
            break
    if info is not None:
        info.update(converged=bool(delta < tol), iterations=int(it), max_delta_f=delta)
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


def autocorrelation_block_ids(state_idx: np.ndarray, series: Sequence[np.ndarray],
                              source_idx: Optional[np.ndarray] = None, *,
                              restrained: Optional[Sequence[Sequence[bool]]] = None,
                              multiple: float = BOOT_BLOCK_G_MULTIPLE, min_blocks: int = BOOT_MIN_BLOCKS
                              ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Bootstrap block ids (state * 10**7 + block index, as ``_block_ids``) with the block
    length set by each state's own autocorrelation, and a record per state.

    ``series``: the observables (CV1, CV2 arrays over all rows); ``restrained[k][a]`` says
    whether state k restrains axis a (default: all). Per state: g = max over its restrained
    axes of ``pooled_inefficiency`` over its rows in each source (in row order; runs shorter
    than G_MIN_RUN skipped); L = ceil(multiple x g); blocks are cut within each source; when the
    state would get fewer than ``min_blocks`` blocks, L = ceil(n / min_blocks) (never below 1)
    and ``min_blocks_bound`` is recorded."""
    state_idx = np.asarray(state_idx, dtype=np.int64)
    src = np.zeros(state_idx.size, dtype=np.int64) if source_idx is None else np.asarray(source_idx, dtype=np.int64)
    out = np.zeros(state_idx.size, dtype=np.int64)
    per_state: Dict[str, Any] = {}
    for k in np.unique(state_idx):
        rows = np.flatnonzero(state_idx == k)
        pieces = [rows[src[rows] == s] for s in np.unique(src[rows])]
        gs, statuses = [], []
        for a, x in enumerate(series):
            if restrained is not None and int(k) < len(restrained) and not bool(restrained[int(k)][a]):
                continue
            est = pooled_inefficiency([np.asarray(x)[p] for p in pieces], min_segment=G_MIN_RUN)
            statuses.append(est.status)
            if est.status == "ok":
                gs.append(float(est.g))
        g = max(gs) if gs else 1.0
        length = max(1, int(math.ceil(float(multiple) * g)))
        n_blocks = sum(int(math.ceil(p.size / float(length))) for p in pieces)
        bound = n_blocks < int(min_blocks) and rows.size > 0
        if bound:
            length = max(1, int(math.ceil(rows.size / float(min_blocks))))
        counter = 0
        for p in pieces:
            b = np.arange(p.size) // length
            out[p] = int(k) * 10 ** 7 + counter + b
            counter += int(b.max()) + 1 if p.size else 0
        per_state[str(int(k))] = {"g": g, "g_status": "ok" if gs else (statuses[0] if statuses else "no_axis"),
                                  "block_rows": length, "n_blocks": counter, "n_rows": int(rows.size),
                                  "n_sources": len(pieces), "min_blocks_bound": bool(bound)}
    info = {"method": "autocorrelation", "multiple": float(multiple), "min_blocks": int(min_blocks),
            "n_states": len(per_state),
            "n_min_blocks_bound": sum(1 for r in per_state.values() if r["min_blocks_bound"]),
            "n_g_fallback": sum(1 for r in per_state.values() if r["g_status"] != "ok"),
            "g_q10_50_90": (np.quantile([r["g"] for r in per_state.values()], [0.1, 0.5, 0.9]).tolist()
                            if per_state else None),
            "states": per_state}
    return out, info


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


def _interval_stats(col, cv1, cv2, state_idx, weights, centre_of_state, rng, *, block_ids=None,
                    centre_column=None, info=None):
    """(frac, n_eff, n_contrib [any column], sigma) per interval. ``block_ids``: bootstrap
    blocks (default ``autocorrelation_block_ids`` over this call's rows, one source);
    ``centre_column``: per centre, its CV1 column key or None (CV1-unrestrained), from which
    ``info["n_contrib_same_column"]`` is counted (the centres of THIS column only)."""
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
    n_eff, n_contrib, n_same = np.zeros(n_int), np.zeros(n_int), np.zeros(n_int)
    in_col = None
    if centre_column is not None:
        key = round(float(col["c1"]), 6)
        in_col = np.asarray([c is not None and c == key for c in centre_column], dtype=bool)
    for i in range(n_int):
        wi = w_int[i]
        if wi.sum() <= 0:
            continue
        share = np.bincount(centre_of_state[state_idx], weights=wi) / wi.sum()
        n_eff[i] = 1.0 / float(np.sum(share ** 2))
        n_contrib[i] = float(np.sum(share >= CONTRIBUTOR_SHARE))
        if in_col is not None:
            n_same[i] = float(np.sum((share >= CONTRIBUTOR_SHARE) & in_col[:share.size]))
    if block_ids is None:
        block_ids, _binfo = autocorrelation_block_ids(state_idx, [cv1, cv2])
    sigma = _boot_sigma(w_int, w_slab, block_ids, rng)
    if info is not None and in_col is not None:
        info["n_contrib_same_column"] = n_same
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
               "n_contributing_windows": float(np.min(n_contrib[i0:i1 + 1])),
               "coverage_count": str(settings.coverage_count), "pmf_sigma_kT": float(np.max(sigma[i0:i1 + 1])),
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


def centre_columns(sample_views: Sequence[cr.StateView], settings: cr.ResolutionSettings
                   ) -> Tuple[np.ndarray, List[Optional[float]]]:
    """(centre index of each sample state, CV1 column key of each centre or None). A centre
    is (round(c1, 6), round(c2, 6)) with an unrestrained axis's placeholder replaced by None
    (P6; before v3 the placeholder was part of the key); its column is round(c1, 6) iff it
    restrains CV1 above the floor, else None (in no column)."""
    k1_floor = max(0.0, float(settings.k1_min))
    keys, cols = [], []
    for v in sample_views:
        r1 = v.k1 is not None and float(v.k1) > k1_floor
        r2 = v.c2 is not None and v.k2 is not None and float(v.k2) > 0
        keys.append((round(v.c1, 6) if r1 else None, round(v.c2, 6) if r2 else None))
    uniq = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    cols = [k[0] for k in uniq]
    return np.asarray([uniq[k] for k in keys], dtype=np.int64), cols


def coverage_holes(cv1: np.ndarray, cv2: np.ndarray, state_idx: np.ndarray, weights: np.ndarray,
                   sample_views: Sequence[cr.StateView], column_views: Mapping[int, cr.StateView],
                   settings: cr.ResolutionSettings, *, epoch: int = 0, source_idx: Optional[np.ndarray] = None,
                   info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """R2 candidates over every CV1 column of ``column_views`` (active representative-rung
    states with their P4 moments). ``state_idx`` indexes ``sample_views`` (the states of the
    MBAR that produced ``weights``, retired ones included); ``weights`` sum to 1;
    ``source_idx`` (optional) = each row's sample source, so bootstrap blocks never cross one.
    ``info``, when given, receives the bootstrap block record (``bootstrap``)."""
    centre_of_state, centre_column = centre_columns(sample_views, settings)
    restrained = [(v.k1 is not None and v.k1 > 0, v.c2 is not None and v.k2 is not None and v.k2 > 0)
                  for v in sample_views]
    blocks, binfo = autocorrelation_block_ids(state_idx, [cv1, cv2], source_idx, restrained=restrained)
    binfo["n_sources"] = 1 if source_idx is None else int(np.unique(source_idx).size)
    if info is not None:
        info["bootstrap"] = {k: v for k, v in binfo.items() if k != "states"}
    same = str(settings.coverage_count) == "same-column"
    rng = np.random.default_rng([_BOOT_SEED, int(epoch)])
    out: List[Dict[str, Any]] = []
    for col in columns(column_views, sorted(column_views), settings):
        col = _extend_to_weight(col, cv1, cv2, weights)
        extra: Dict[str, Any] = {}
        frac, n_eff, n_any, sigma = _interval_stats(col, cv1, cv2, state_idx, weights, centre_of_state, rng,
                                                    block_ids=blocks, centre_column=centre_column, info=extra)
        n_same = extra["n_contrib_same_column"]
        n_contrib = n_same if same else n_any
        flags = (frac >= MIN_INTERVAL_WEIGHT) & ((n_contrib < float(settings.coverage_min_windows))
                                                 | (sigma > float(settings.refine_pmf_sigma_kT)))
        for i0, i1 in _hole_runs(flags, col["edges"], [float(v.c2) for v in col["members"]]):
            cand = _hole_candidate(col, i0, i1, frac, n_eff, n_contrib, sigma, settings)
            cand["metrics"].update(n_contributing_windows_any=float(np.min(n_any[i0:i1 + 1])),
                                   n_contributing_windows_same_column=float(np.min(n_same[i0:i1 + 1])))
            out.append(cand)
    return out


__all__ = ["autocorrelation_block_ids", "centre_columns", "columns", "coverage_holes", "log_weights", "reduced_umbrella", "solve_mbar"]
