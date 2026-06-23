"""Replica-count experiment: few-deep vs many-shallow at fixed total budget.

At a fixed total sample budget B, place R umbrella windows along CV1 (R = 1..12,
B/R raw samples each), recover the CV1 marginal PMF by MBAR, and measure the
error. The spring constant is calibrated PER R (the fair-k rule) so every R is a
same-overlap, connected ladder — otherwise small R fails trivially on fixed k.

The ESS/burn-in cost model (gareus.synth.ess) makes the trade-off physical: more
windows pay more total burn-in (R*burn_in) and get fewer raw samples each, while
too few windows cannot bridge a tall barrier even at the tightest connected k.
"""
from __future__ import annotations

import numpy as np

from .ess import BURN_IN, effective_count, tau_int, thin_to_ess
from .landscapes import Landscape
from .oracle import grid_overlap, low_f_mask, reference_pmf
from .sampler import Window, sample_window_exact

CV2_CENTER = 0.0
CV2_K = 15.0          # mild CV2 restraint (nuisance coordinate)


def low_f_support(landscape: Landscape, *, threshold_kbt: float = 5.0, res: int = 200):
    """CV1 span [lo, hi] of the low-F region the window ladder must tile."""
    x, g = reference_pmf(landscape, axis="cv1", res=res)
    sel = x[g <= (g.min() + threshold_kbt)]
    if sel.size == 0:
        return float(x.min()), float(x.max())
    return float(sel.min()), float(sel.max())


def _min_adjacent_overlap(landscape, centers, k, *, res):
    ov = [grid_overlap(landscape, Window(centers[i], k, CV2_CENTER, CV2_K),
                       Window(centers[i + 1], k, CV2_CENTER, CV2_K), res=res, axis="cv1")
          for i in range(len(centers) - 1)]
    return float(min(ov)) if ov else 1.0


def calibrate_k(landscape: Landscape, R: int, *, overlap_target: float = 0.30,
                res: int = 120, k_lo: float = 8.0, k_hi: float = 4000.0):
    """Largest spring k whose min adjacent CV1 overlap >= target (the fair-k rule).

    Returns (centers, k, min_overlap, valid). min_adjacent_overlap decreases with
    k (tighter windows overlap less), so we bisect for the largest k still meeting
    the target. If even the widest allowed k (k_lo) cannot reach the target, the
    ladder cannot be made connected at this R -> valid=False (best-effort k_lo).
    R=1 is the unbiased single-window baseline (k=0).
    """
    lo, hi = low_f_support(landscape)
    if R <= 1:
        return [0.5 * (lo + hi)], 0.0, 1.0, False
    centers = list(np.linspace(lo, hi, R))
    if _min_adjacent_overlap(landscape, centers, k_lo, res=res) < overlap_target:
        return centers, k_lo, _min_adjacent_overlap(landscape, centers, k_lo, res=res), False
    a, b = k_lo, k_hi
    for _ in range(28):
        m = 0.5 * (a + b)
        if _min_adjacent_overlap(landscape, centers, m, res=res) >= overlap_target:
            a = m            # still connected -> can go tighter
        else:
            b = m
    return centers, float(a), _min_adjacent_overlap(landscape, centers, a, res=res), True


def sample_replica_layout(landscape: Landscape, R: int, budget: int, *, seed: int = 0,
                          beta: float = 1.0, res: int = 120, overlap_target: float = 0.30):
    """Place R fair-k windows and draw B/R ESS-thinned samples each.

    Returns ``(centers, k, min_ov_grid, valid_k, n_raw, windows, samples, ess)``
    where ``windows``/``samples``/``ess`` are dicts keyed by window index.
    """
    rng = np.random.default_rng(seed)
    centers, k, min_ov_grid, valid_k = calibrate_k(landscape, R, overlap_target=overlap_target, res=res)
    n_raw = int(budget // max(1, R))
    windows, samples, ess = {}, {}, {}
    for i, c in enumerate(centers):
        w = Window(float(c), float(k), CV2_CENTER, CV2_K)
        windows[i] = w
        raw = sample_window_exact(landscape, w, n_raw, beta=beta, res=res, rng=rng)
        tau = tau_int(landscape, w)
        ne = int(effective_count(n_raw, tau, burn_in=BURN_IN, is_new=True))   # each window equilibrates once
        ess[i] = ne
        samples[i] = thin_to_ess(raw, ne)
    return centers, k, min_ov_grid, valid_k, n_raw, windows, samples, ess


def replica_run(landscape: Landscape, R: int, budget: int, *, seed: int = 0,
                beta: float = 1.0, res: int = 120, overlap_target: float = 0.30,
                coverage_min: float = 0.80) -> dict:
    """One (R, budget) experiment: place R fair-k windows, sample B/R each (ESS),
    MBAR-recover the CV1 marginal PMF, return error + gates + cost diagnostics."""
    from .metrics import pmf_recovery
    (centers, k, min_ov_grid, valid_k, n_raw,
     windows, samples, ess) = sample_replica_layout(
        landscape, R, budget, seed=seed, beta=beta, res=res, overlap_target=overlap_target)

    nonempty = {i: s for i, s in samples.items() if len(s) > 0}
    wins_ne = {i: windows[i] for i in nonempty}

    # coverage of the low-F CV1 region by the pooled samples
    c1, c2, mask = low_f_mask(landscape, res=res)
    lowf_cv1 = np.unique(np.where(mask.any(axis=1))[0])
    if nonempty:
        allcv1 = np.concatenate([np.asarray(s)[:, 0] for s in nonempty.values()])
        hit = np.clip(np.searchsorted(c1, allcv1) - 1, 0, len(c1) - 1)
        coverage = float(len(set(hit) & set(lowf_cv1.tolist())) / max(1, len(lowf_cv1)))
    else:
        coverage = 0.0

    if R <= 1:
        # unbiased single-window baseline: valid as a data point if it covers the
        # low-F region (it generally cannot bridge a tall barrier -> high error).
        connected = bool(coverage >= coverage_min)
        gated_ok = bool(coverage >= coverage_min and len(nonempty) >= 1)
    else:
        connected = bool(valid_k and len(nonempty) == len(centers) and min_ov_grid >= overlap_target)
        gated_ok = bool(connected and coverage >= coverage_min and len(nonempty) >= 2)

    if len(nonempty) >= 1:
        pmf = pmf_recovery(landscape, nonempty, wins_ne, axis="cv1", res=res)["rmse_lowf_weighted"]
    else:
        pmf = float("nan")

    ess_vals = [v for v in ess.values() if v > 0]
    return {
        "R": int(R), "budget": int(budget), "seed": int(seed),
        "k": float(k), "n_raw_per_window": n_raw,
        "min_overlap_grid": float(min_ov_grid), "valid_ladder": bool(valid_k),
        "connected": connected, "coverage": coverage, "gated_ok": gated_ok,
        "pmf_rmse_lowf": float(pmf),
        "min_per_window_ess": int(min(ess_vals)) if ess_vals else 0,
        "total_ess": int(sum(ess.values())),
        "ess_efficiency": float(sum(ess.values()) / max(1, budget)),
        "burnin_fraction": float(R * BURN_IN / max(1, budget)),
        "n_windows": int(len(centers)),
    }


# ---------------------------------------------------------------------------
# 2D replica placement: windows tile CV1xCV2 (for landscapes rugged along both)
# ---------------------------------------------------------------------------

def low_f_box(landscape: Landscape, *, threshold_kbt: float = 5.0, res: int = 160):
    """Bounding box (lo1, hi1, lo2, hi2) of the low-F region to tile in 2D."""
    c1, c2, f = landscape.grid(res=res)
    mask = f <= (f.min() + threshold_kbt)
    i1 = np.where(mask.any(axis=1))[0]
    i2 = np.where(mask.any(axis=0))[0]
    if i1.size == 0 or i2.size == 0:
        return landscape.cv1_bounds[0], landscape.cv1_bounds[1], landscape.cv2_bounds[0], landscape.cv2_bounds[1]
    return float(c1[i1.min()]), float(c1[i1.max()]), float(c2[i2.min()]), float(c2[i2.max()])


def place_windows_2d(landscape: Landscape, R: int, *, threshold_kbt: float = 5.0,
                     res: int = 120, scale1: float = 1.0, scale2: float = 2.0):
    """R window centers spread over the low-F region by farthest-point sampling.

    Data-driven (no hardcoded basin knowledge): seeds at the global minimum, then
    greedily adds the low-F cell farthest (in normalized CV) from the current set.
    Tracks the low-F bands instead of a naive box grid that would drop windows
    onto high-F barrier regions.
    """
    c1, c2, f = landscape.grid(res=res)
    mask = f <= (f.min() + threshold_kbt)
    g1, g2 = np.meshgrid(c1, c2, indexing="ij")
    pts = np.column_stack([g1[mask], g2[mask]])
    fvals = f[mask]
    if pts.shape[0] == 0:
        return [(0.5 * sum(landscape.cv1_bounds), 0.5 * sum(landscape.cv2_bounds))]
    if R >= pts.shape[0]:
        return [(float(a), float(b)) for a, b in pts]
    P = pts / np.array([scale1, scale2])
    chosen = [int(np.argmin(fvals))]                 # seed at the deepest basin
    d = np.sqrt(((P - P[chosen[0]]) ** 2).sum(axis=1))
    for _ in range(R - 1):
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, np.sqrt(((P - P[j]) ** 2).sum(axis=1)))
    return [(float(pts[i, 0]), float(pts[i, 1])) for i in chosen]


def _nn_pairs(centers, *, scale1=1.0, scale2=2.0):
    """Each point's nearest neighbor (normalized), as unique index pairs."""
    P = np.array([[c[0] / scale1, c[1] / scale2] for c in centers])
    pairs = set()
    for i in range(len(P)):
        d = np.sqrt(((P - P[i]) ** 2).sum(axis=1)); d[i] = np.inf
        j = int(np.argmin(d))
        pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def calibrate_k_2d(landscape, centers, *, overlap_target=0.30, res=90,
                   k_lo=8.0, k_hi=4000.0):
    """Largest isotropic k whose min nearest-neighbour 2D overlap >= target."""
    from .oracle import grid_overlap_2d
    if len(centers) <= 1:
        return 0.0, 1.0, False
    pairs = _nn_pairs(centers)

    def min_ov(k):
        ov = [grid_overlap_2d(landscape, Window(centers[i][0], k, centers[i][1], k),
                              Window(centers[j][0], k, centers[j][1], k), res=res)
              for i, j in pairs]
        return float(min(ov)) if ov else 1.0

    if min_ov(k_lo) < overlap_target:
        return k_lo, min_ov(k_lo), False
    a, b = k_lo, k_hi
    for _ in range(26):
        m = 0.5 * (a + b)
        if min_ov(m) >= overlap_target:
            a = m
        else:
            b = m
    return float(a), min_ov(a), True


def replica_run_2d(landscape: Landscape, R: int, budget: int, *, seed: int = 0,
                   beta: float = 1.0, res: int = 90, overlap_target: float = 0.30,
                   coverage_min: float = 0.50, max_per_window: int = 3000,
                   return_layout: bool = False):
    """2D replica experiment: R fair-k windows tiling CV1xCV2, B/R ESS samples each,
    MBAR-recover the 2D PMF over the low-F region."""
    from .metrics import pmf_recovery_2d
    rng = np.random.default_rng(seed)
    centers = place_windows_2d(landscape, R)
    k, min_ov, valid_k = calibrate_k_2d(landscape, centers, overlap_target=overlap_target, res=res)
    n_raw = int(budget // max(1, R))
    windows, samples, ess = {}, {}, {}
    for i, (c1, c2) in enumerate(centers):
        w = Window(float(c1), float(k), float(c2), float(k))
        windows[i] = w
        raw = sample_window_exact(landscape, w, n_raw, beta=beta, res=res, rng=rng)
        tau = tau_int(landscape, w)
        ne = int(effective_count(n_raw, tau, burn_in=BURN_IN, is_new=True))
        ess[i] = ne
        samples[i] = thin_to_ess(raw, ne)
    nonempty = {i: s for i, s in samples.items() if len(s) > 0}
    wins_ne = {i: windows[i] for i in nonempty}

    rec = pmf_recovery_2d(landscape, nonempty, wins_ne, res=res, max_per_window=max_per_window) \
        if len(nonempty) >= 1 else {"rmse_lowf": float("nan"), "coverage": 0.0}
    coverage = rec["coverage"]
    if R <= 1:
        connected = bool(coverage >= coverage_min)
        gated_ok = bool(coverage >= coverage_min and len(nonempty) >= 1)
    else:
        connected = bool(valid_k and len(nonempty) == len(centers) and min_ov >= overlap_target)
        gated_ok = bool(connected and coverage >= coverage_min and len(nonempty) >= 2)
    ess_vals = [v for v in ess.values() if v > 0]
    out = {
        "R": int(R), "budget": int(budget), "seed": int(seed), "k": float(k),
        "n_raw_per_window": n_raw, "min_overlap_grid": float(min_ov), "valid_ladder": bool(valid_k),
        "connected": connected, "coverage": float(coverage), "gated_ok": gated_ok,
        "pmf_rmse_lowf": float(rec["rmse_lowf"]),
        "min_per_window_ess": int(min(ess_vals)) if ess_vals else 0,
        "total_ess": int(sum(ess.values())), "ess_efficiency": float(sum(ess.values()) / max(1, budget)),
        "burnin_fraction": float(R * BURN_IN / max(1, budget)), "n_windows": int(len(centers)),
    }
    if return_layout:
        return out, nonempty, wins_ne
    return out
