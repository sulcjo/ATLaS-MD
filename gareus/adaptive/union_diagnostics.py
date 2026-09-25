"""Per-epoch union MBAR: local sigma, per-state local split-halves, edge overlap.

Reads the union NPZ written by adaptive_production.build_union_state_mbar_inputs.
Its samples are ALREADY decorrelated per state (equilibration discard + thinning),
so no second autocorrelation pass runs here: the inefficiency comes from the
builder's meta (subsample_counts_per_state). Never raises: any failure returns
None and the epoch runs no top-up.

Split-halves: per-state LOCAL test on each state's own reduced bias (first vs second
half of chronological rows). Welch z-test, Bonferroni-corrected over tested states,
plus effect floor. Local by construction: drift in one state cannot flag neighbours.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.stats import norm

MIN_HALF = 20


@dataclass(frozen=True)
class UnionDiagnostics:
    state_ids: tuple
    n_k: dict
    sigma_kcal: dict
    unconverged: frozenset
    inefficiency: dict
    edge_overlap: dict
    f_kT: dict


def _solve(u_nk: np.ndarray, window: np.ndarray, n_states: int, f_init: Optional[np.ndarray] = None):
    """f (reduced, vs first active state), pairwise uncertainty matrix (K x K, nan where unsampled), n_k.

    Initialised from zeros or a warm start, not BAR: pymbar's BAR initialisation chains
    consecutive states, and the union's state order (centres x rungs) is not overlap order.
    Warm start is partial: non-finite entries filled with 0.0 after shifting.
    """
    from pymbar import MBAR  # noqa: PLC0415
    n_k = np.bincount(window, minlength=n_states)
    active = np.flatnonzero(n_k > 0)
    order = np.argsort(window, kind="stable")                 # pymbar wants rows grouped by state
    u_kn = u_nk[order][:, active].T
    kwargs = {"initialize": "zeros", "solver_protocol": "robust"}
    if f_init is not None:
        f_active = f_init[active].copy()
        finite = np.isfinite(f_active)
        if np.any(finite):
            shift = f_active[finite][0]
            f_active[finite] -= shift
            f_active[~finite] = 0.0
            kwargs["initial_f_k"] = f_active
    mbar = MBAR(u_kn, n_k[active], **kwargs)
    res = mbar.compute_free_energy_differences(compute_uncertainty=True)
    f = np.full(n_states, np.nan); dmat = np.full((n_states, n_states), np.nan)
    f[active] = res["Delta_f"][0]
    dmat[np.ix_(active, active)] = res["dDelta_f"]
    return f, dmat, n_k


def _local_sigma(dmat: np.ndarray, k: int, neighbours) -> float:
    """min over edge-neighbours j of dDelta_f[j, k]; min of the finite off-diagonal row when k has no measured neighbour."""
    vals = [dmat[j, k] for j in neighbours if np.isfinite(dmat[j, k])]
    if vals:
        return float(min(vals))
    row = dmat[k][np.isfinite(dmat[k]) & (np.arange(len(dmat)) != k)]
    return float(np.min(row)) if row.size else float("nan")


def _inefficiency(ids, subsample_counts) -> Dict[int, float]:
    out = {}
    for k, sid in enumerate(ids):
        rec = (subsample_counts or {}).get(str(sid)) or {}
        g = rec.get("g")
        if not (isinstance(g, (int, float)) and math.isfinite(float(g)) and float(g) >= 1.0):
            raw, t0, kept = rec.get("raw"), rec.get("t0", 0), rec.get("kept")
            g = max(1.0, (raw - t0) / max(1, kept)) if raw is not None and kept else 1.0
        out[sid] = float(g)
    return out


def _split_halves(u, window, K, min_effect_kT, alpha):
    """Per-state local drift test: a state's own reduced bias, first vs second half of its own rows.

    Rows are chronological within a state (the union builder keeps source order). Welch z on the
    difference of means, Bonferroni over the tested states, plus an effect floor in kT. Local by
    construction: another state's samples never enter a state's statistic.
    """
    own = u[np.arange(len(window)), window]
    tested = []
    for k in range(K):
        x = own[window == k]
        if len(x) < 2 * MIN_HALF:
            continue
        h = len(x) // 2
        a, b = x[:h], x[h:]
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        tested.append((k, abs(float(a.mean() - b.mean())), se))
    if not tested:
        return set()
    z_star = float(norm.ppf(1.0 - alpha / (2.0 * len(tested))))
    return {k for k, d, se in tested
            if math.isfinite(d) and math.isfinite(se) and d > max(z_star * se, min_effect_kT)}


def union_diagnostics_from_npz(npz_path, edges: Iterable[Tuple[int, int]], *, kt_kcal: float,
                               subsample_counts: Optional[dict] = None, min_effect_kcal: float = 0.05,
                               alpha: float = 0.05, f_init: Optional[dict] = None,
                               split_halves: bool = True) -> Optional[UnionDiagnostics]:
    try:
        npz_path = Path(npz_path)
        if not npz_path.exists():
            logging.warning("top-up diagnostics: union NPZ not found; top-ups off for this epoch (%s)", npz_path)
            return None
        with np.load(npz_path, allow_pickle=False) as z:
            u = np.asarray(z["umbrella_reduced_bias_nk"], dtype=float)
            ids = [int(s) for s in z["state_ids"].tolist()]
            sampled = np.asarray(z["sampled_state_ids"], dtype=np.int64)
        idx = {s: i for i, s in enumerate(ids)}
        window = np.asarray([idx.get(int(s), -1) for s in sampled], dtype=np.int64)
        keep = (window >= 0) & np.isfinite(u).all(axis=1)       # NaN rows = missing energies, excluded by design
        excluded_count = int((~keep).sum())
        total_count = len(keep)
        if excluded_count:
            pct = 100.0 * excluded_count / total_count if total_count > 0 else 100.0
            level = logging.WARNING if pct > 10.0 else logging.INFO
            logging.log(level, "top-up diagnostics: %d of %d rows excluded (non-finite, %.1f%%)", excluded_count, total_count, pct)
        u, window = u[keep], window[keep]
        if u.shape[0] == 0:
            logging.warning("top-up diagnostics: no finite rows remain; top-ups off for this epoch")
            return None
        K = len(ids)
        f0 = None
        if f_init:
            f0 = np.asarray([float(f_init.get(s, np.nan)) for s in ids])
        f, dmat, n_k = _solve(u, window, K, f0)
        nbrs = {k: [] for k in range(K)}
        pairs = set()
        for a, b in edges:
            ia, ib = idx.get(int(a)), idx.get(int(b))
            if ia is not None and ib is not None and ia != ib:
                nbrs[ia].append(ib); nbrs[ib].append(ia)
                pairs.add((min(ia, ib), max(ia, ib)))
        sigma = np.array([_local_sigma(dmat, k, nbrs[k]) if n_k[k] > 0 else np.nan for k in range(K)])
        flagged = (_split_halves(u, window, K, min_effect_kcal / kt_kcal, alpha)
                   if split_halves else set())
        from ..mbar_analysis.ladder import mbar_state_overlap  # noqa: PLC0415
        O = mbar_state_overlap(u, np.where(np.isfinite(f), f, 0.0), n_k)
        edge_overlap = {}
        for ia, ib in sorted(pairs):
            if n_k[ia] == 0 or n_k[ib] == 0:
                continue
            x, y = float(O[ia, ib]), float(O[ib, ia])
            if math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0:
                edge_overlap[(min(ids[ia], ids[ib]), max(ids[ia], ids[ib]))] = math.sqrt(x * y)
        return UnionDiagnostics(
            state_ids=tuple(ids),
            n_k={ids[k]: int(n_k[k]) for k in range(K)},
            sigma_kcal={ids[k]: (float(sigma[k]) * kt_kcal if math.isfinite(sigma[k]) else math.nan)
                        for k in range(K)},
            unconverged=frozenset(ids[k] for k in flagged),
            inefficiency=_inefficiency(ids, subsample_counts),
            edge_overlap=edge_overlap,
            f_kT={ids[k]: float(f[k]) for k in range(K) if math.isfinite(f[k])},
        )
    except Exception as exc:  # the epoch then runs no top-up
        logging.warning("top-up union diagnostics unavailable (%s)", exc, exc_info=True)
        return None
