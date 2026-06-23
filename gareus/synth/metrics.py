"""Success metrics comparing adaptive-campaign output to the grid oracle.

Core: action accuracy vs oracle, final overlap vs target, low-F coverage, window
economy.  Extended: PMF recovery (proper per-sample MBAR reweighting vs the true
marginal), overlap-graph connectivity, center churn.

All metrics are computed against quantities the oracle derives from the exact
free-energy grid, so "did the adaptive logic do the right thing" has an absolute
reference.  Metrics that depend on window width use the *actual* per-window force
constants from the record (``record.k1``), not a hardcoded default, so a campaign
that runs at k=40 is not judged as if it ran at k=50.
"""
from __future__ import annotations

import numpy as np

from .landscapes import Landscape
from .oracle import grid_overlap, ideal_axis_ladder, low_f_mask, reference_pmf
from .sampler import BIAS, Window

# Fallback width only when a record carries no usable k (e.g. hand-built test records).
_FALLBACK_K = 50.0


def _distinct_centers_k(centers, ks, tol: float = 1e-3):
    """Return aligned distinct (center, k) lists, sorted by center.

    Collapses windows that share a CV1 center (the 2D grid repeats each CV1
    center across CV2 rows), keeping the first k seen for that center.
    """
    ks = list(ks) if ks is not None and len(ks) == len(centers) else [None] * len(centers)
    pairs = sorted(zip((float(c) for c in centers), ks), key=lambda p: p[0])
    out_c, out_k = [], []
    for c, k in pairs:
        if not out_c or c - out_c[-1] > tol:
            out_c.append(c)
            out_k.append(float(k) if (k is not None and np.isfinite(k) and k > 0) else _FALLBACK_K)
    return out_c, out_k


def _representative_k(record) -> float:
    ks = [float(k) for k in (record.k1 or []) if np.isfinite(k) and k > 0]
    return float(np.median(ks)) if ks else _FALLBACK_K


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def final_overlap_vs_target(landscape, record, *, target: float = 0.30,
                            res: int = 120) -> dict:
    cs, ks = _distinct_centers_k(record.centers1, record.k1)
    if len(cs) < 2:
        return {"mean": float("nan"), "min": float("nan"),
                "frac_pairs_below_target": float("nan")}
    ovs = np.asarray([
        grid_overlap(landscape, Window(cs[i], ks[i]), Window(cs[i + 1], ks[i + 1]), res=res)
        for i in range(len(cs) - 1)], float)
    return {"mean": float(ovs.mean()), "min": float(ovs.min()),
            "frac_pairs_below_target": float(np.mean(ovs < target))}


def low_f_coverage(landscape, record, *, res: int = 120,
                   threshold_kbt: float = 5.0) -> float:
    c1, c2, mask = low_f_mask(landscape, res=res, threshold_kbt=threshold_kbt)
    g1, _g2 = np.meshgrid(c1, c2, indexing="ij")
    pts1 = g1[mask]
    if pts1.size == 0:
        return float("nan")
    covered = np.zeros(pts1.shape, bool)
    cs, ks = _distinct_centers_k(record.centers1, record.k1)
    for c, k in zip(cs, ks):
        sigma = 1.0 / np.sqrt(k)
        covered |= (np.abs(pts1 - c) <= 2.0 * sigma)
    return float(np.mean(covered))


def window_economy(landscape, record, *, target: float = 0.30, res: int = 120) -> dict:
    k = _representative_k(record)
    ladder = ideal_axis_ladder(landscape, axis="cv1", target_overlap=target,
                               k=k, res=res)
    n_ideal = len(ladder)
    n = len(_distinct_centers_k(record.centers1, record.k1)[0])
    return {"n_windows": int(n), "n_ideal": int(n_ideal),
            "ratio": (n / n_ideal) if n_ideal else float("nan"),
            "k_used": float(k)}


def action_accuracy(landscape, records, *, target: float = 0.30, res: int = 120) -> dict:
    if not records:
        return {"converged_toward_ideal": False, "final_gap": None}
    k = _representative_k(records[-1])
    ladder = ideal_axis_ladder(landscape, axis="cv1", target_overlap=target,
                               k=k, res=res)
    n_ideal = len(ladder)
    n0 = len(_distinct_centers_k(records[0].centers1, records[0].k1)[0])
    nf = len(_distinct_centers_k(records[-1].centers1, records[-1].k1)[0])
    gap0, gapf = abs(n0 - n_ideal), abs(nf - n_ideal)
    return {"converged_toward_ideal": bool(gapf <= gap0),
            "initial_gap": int(gap0), "final_gap": int(gapf),
            "n_ideal": int(n_ideal), "final_n": int(nf), "initial_n": int(n0)}


# ---------------------------------------------------------------------------
# Extended metrics
# ---------------------------------------------------------------------------

def overlap_graph_connectivity(landscape, record, *, target: float = 0.30,
                               res: int = 120) -> dict:
    cs, ks = _distinct_centers_k(record.centers1, record.k1)
    if len(cs) < 2:
        return {"min_edge": float("nan"), "connected": False, "spectral_gap": float("nan")}
    edges = np.asarray([
        grid_overlap(landscape, Window(cs[i], ks[i]), Window(cs[i + 1], ks[i + 1]), res=res)
        for i in range(len(cs) - 1)], float)
    min_bridge = max(0.04, 0.2 * target)
    connected = bool(np.all(edges >= min_bridge))
    n = len(cs)
    w = np.clip(edges, 0.0, None)
    lap = np.zeros((n, n))
    for i in range(n - 1):
        lap[i, i] += w[i]; lap[i + 1, i + 1] += w[i]
        lap[i, i + 1] -= w[i]; lap[i + 1, i] -= w[i]
    evals = np.sort(np.linalg.eigvalsh(lap))
    gap = float(evals[1]) if n >= 2 else float("nan")
    return {"min_edge": float(edges.min()), "connected": connected, "spectral_gap": gap}


def center_churn(records) -> float:
    if len(records) < 2:
        return 0.0
    moves = []
    for a, b in zip(records[:-1], records[1:]):
        ca = np.asarray(_distinct_centers_k(a.centers1, a.k1)[0])
        cb = np.asarray(_distinct_centers_k(b.centers1, b.k1)[0])
        if ca.size == 0 or cb.size == 0:
            continue
        d = np.min(np.abs(cb[:, None] - ca[None, :]), axis=1)
        moves.append(float(np.mean(d)))
    return float(np.mean(moves)) if moves else 0.0


def _mbar_weights(windows, samples_by_window, *, beta: float = 1.0,
                  max_iter: int = 400, tol: float = 1e-7):
    """Self-consistent MBAR: return pooled samples and per-sample log weights.

    Removes the umbrella bias of EVERY window from the pooled biased samples,
    yielding weights for the unbiased ensemble.  ``u_ij = beta * BIAS(window_i,
    x_j)`` over the full 2D bias.  Returns ``(pooled (M,2), log_w (M,))``.
    """
    from scipy.special import logsumexp

    ids = [i for i in sorted(samples_by_window) if len(samples_by_window[i])]
    if not ids:
        return np.empty((0, 2)), np.empty((0,))
    parts = [np.asarray(samples_by_window[i], float) for i in ids]
    counts = np.array([p.shape[0] for p in parts], float)
    pooled = np.concatenate(parts, axis=0)
    K = len(ids)
    # u[i, j] = beta * bias of window i evaluated at sample j  (K x M)
    u = np.empty((K, pooled.shape[0]))
    for a, i in enumerate(ids):
        u[a] = beta * BIAS(windows[i], pooled[:, 0], pooled[:, 1])
    log_N = np.log(counts)
    f = np.zeros(K)
    for _ in range(max_iter):
        # log_denom_j = logsumexp_i( log_N_i + f_i - u_ij )
        log_denom = logsumexp(log_N[:, None] + f[:, None] - u, axis=0)  # (M,)
        f_new = -logsumexp(-u - log_denom[None, :], axis=1)  # (K,)
        f_new = f_new - f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    log_denom = logsumexp(log_N[:, None] + f[:, None] - u, axis=0)
    log_w = -log_denom
    log_w = log_w - logsumexp(log_w)  # normalize
    return pooled, log_w


def pmf_recovery(landscape, samples_by_window, windows, *, axis: str = "cv1",
                 res: int = 120, beta: float = 1.0) -> dict:
    """Per-sample MBAR-reweighted PMF estimate vs the true marginal PMF.

    ``windows`` must be a mapping ``{window_index: Window}`` aligned with
    ``samples_by_window`` so the applied umbrella bias can be removed.  Returns
    ``rmse_kbt`` / ``js`` / ``rmse_lowf_weighted`` on the explored support.
    """
    x_ref, pmf_ref = reference_pmf(landscape, axis=axis, res=res)
    pooled, log_w = _mbar_weights(windows, samples_by_window, beta=beta)
    if pooled.shape[0] == 0:
        return {"rmse_kbt": float("nan"), "js": float("nan"),
                "rmse_lowf_weighted": float("nan")}
    coord = pooled[:, 0] if axis == "cv1" else pooled[:, 1]
    lo, hi = (landscape.cv1_bounds if axis == "cv1" else landscape.cv2_bounds)
    bins = np.linspace(lo, hi, res + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    w = np.exp(log_w)
    dens, _ = np.histogram(coord, bins=bins, weights=w, density=True)
    dens = np.clip(dens, 1e-12, None)
    pmf_est = -np.log(dens)
    pmf_est -= pmf_est.min()
    pmf_ref_i = np.interp(centers, x_ref, pmf_ref)
    # explored support: bins that received real (unweighted) samples
    raw_counts, _ = np.histogram(coord, bins=bins)
    support = raw_counts > max(1, coord.size // (5 * res))
    if support.sum() < 3:
        support = raw_counts > 0
    diff = (pmf_est - pmf_ref_i)[support]
    diff = diff - np.mean(diff)  # PMFs defined up to a constant
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    wgt = np.exp(-pmf_ref_i)[support]
    wgt = wgt / wgt.sum()
    rmse_lowf = float(np.sqrt(np.sum(wgt * diff ** 2)))
    p = np.exp(-pmf_est)[support]; p /= p.sum()
    q = np.exp(-pmf_ref_i)[support]; q /= q.sum()
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * np.log(np.clip(a, 1e-12, None) / np.clip(b, 1e-12, None))))

    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return {"rmse_kbt": rmse, "js": float(js), "rmse_lowf_weighted": rmse_lowf}


def dispatcher_overlap_from_samples(samples_by_window, windows, *, bins: int = 80) -> dict:
    """Neighbor CV1 overlap using the SHIPPED dispatcher functional.

    Groups final-layout windows into CV1 columns (by primary center) and computes
    the histogram-intersection overlap (``gareus.math_helpers._adaptive_hist_overlap``,
    the same functional the real dispatcher targets at 0.30) between adjacent
    columns' pooled CV1 samples.  This is directly comparable to the 0.30 target,
    unlike the oracle's Bhattacharyya ``final_overlap``.
    """
    from gareus.math_helpers import _adaptive_hist_overlap

    cols = {}
    for wi, w in windows.items():
        s = samples_by_window.get(wi)
        if s is None or len(s) == 0:
            continue
        key = round(float(w.center1), 6)
        cols.setdefault(key, []).extend(np.asarray(s, float)[:, 0].tolist())
    centers = sorted(cols)
    if len(centers) < 2:
        return {"mean": float("nan"), "min": float("nan"),
                "frac_pairs_below_target": float("nan")}
    ovs = []
    for a, b in zip(centers[:-1], centers[1:]):
        va = np.asarray(cols[a], float)
        vb = np.asarray(cols[b], float)
        lo = float(min(va.min(), vb.min()))
        hi = float(max(va.max(), vb.max()))
        ovs.append(float(_adaptive_hist_overlap(va, vb, lo, hi, bins)))
    ovs = np.asarray(ovs, float)
    return {"mean": float(np.nanmean(ovs)), "min": float(np.nanmin(ovs)),
            "frac_pairs_below_target": float(np.mean(ovs < 0.30))}


def campaign_metrics(landscape, records, *, target: float = 0.30, res: int = 120,
                     samples_by_window=None, windows=None) -> dict:
    last = records[-1]
    out = {
        "final_overlap": final_overlap_vs_target(landscape, last, target=target, res=res),
        "low_f_coverage": low_f_coverage(landscape, last, res=res),
        "economy": window_economy(landscape, last, target=target, res=res),
        "action_accuracy": action_accuracy(landscape, records, target=target, res=res),
        "connectivity": overlap_graph_connectivity(landscape, last, target=target, res=res),
        "center_churn": center_churn(records),
        "rounds": len(records),
        "converged": bool(last.converged),
    }
    if samples_by_window is not None and windows is not None:
        out["pmf_recovery"] = pmf_recovery(landscape, samples_by_window, windows, res=res)
        out["dispatcher_overlap"] = dispatcher_overlap_from_samples(samples_by_window, windows)
    return out
