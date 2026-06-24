"""The known answer, computed analytically from the exact free-energy grid.

The oracle never inspects what the adaptive algorithm did; it derives, purely
from ``F``, what a good window layout and the recovered PMF *should* look like:

* ``grid_overlap``              -- exact Bhattacharyya overlap of two windows'
  biased marginals along a CV axis.
* ``low_f_mask``                -- the region windows should cover (F <= min+5 kBT).
* ``ideal_axis_ladder``         -- greedy minimal center chain that maintains
  constant neighbor overlap (constant-overlap rule = correct staging criterion).
* ``reference_pmf``             -- true marginal free energy along an axis.
* ``oracle_k_for_r``            -- bisect k to find the spring constant making
  the greedy walk yield >= R centers (R-window oracle).
* ``oracle_optimal_centers_1d`` -- R oracle-optimal 1D centers (dense at barriers).
* ``oracle_optimal_centers_2d`` -- R oracle-optimal 2D centers via greedy
  farthest-point in Bhattacharyya-overlap-distance space.
* ``placement_comparison_1d``   -- benchmark oracle vs linspace MBAR-PMF RMSE.

Metrics compare the adaptive output to these.
"""
from __future__ import annotations

import numpy as np

from .landscapes import Landscape
from .sampler import Window, BIAS

LOW_F_THRESHOLD_KBT = 5.0


def _axis_biased_marginal(landscape: Landscape, window: Window, *,
                          beta: float = 1.0, res: int = 120, axis: str = "cv1"):
    c1, c2, f = landscape.grid(res=res)
    g1, g2 = np.meshgrid(c1, c2, indexing="ij")
    logp = -beta * (f + BIAS(window, g1, g2))
    logp = logp - logp.max()
    p = np.exp(logp)
    if axis == "cv1":
        m = p.sum(axis=1)
        x = c1
    else:
        m = p.sum(axis=0)
        x = c2
    s = m.sum()
    return x, (m / s if s > 0 else m)


def grid_overlap(landscape, wa, wb, *, beta: float = 1.0, res: int = 120,
                 axis: str = "cv1") -> float:
    """Bhattacharyya coefficient of two windows' biased axis marginals in [0,1]."""
    _, pa = _axis_biased_marginal(landscape, wa, beta=beta, res=res, axis=axis)
    _, pb = _axis_biased_marginal(landscape, wb, beta=beta, res=res, axis=axis)
    return float(np.sum(np.sqrt(pa * pb)))


def grid_overlap_2d(landscape, wa, wb, *, beta: float = 1.0, res: int = 120) -> float:
    """JOINT Bhattacharyya overlap of two windows' full 2D biased distributions.

    Unlike ``grid_overlap`` (an axis marginal), this is sensitive to separation
    in EITHER CV: two windows sharing a CV1 center but differing in CV2 correctly
    score near zero here even though their CV1 marginals coincide.
    """
    c1, c2, f = landscape.grid(res=res)
    g1, g2 = np.meshgrid(c1, c2, indexing="ij")
    def _dist(w):
        logp = -beta * (f + BIAS(w, g1, g2))
        logp = logp - logp.max()
        p = np.exp(logp)
        s = p.sum()
        return p / s if s > 0 else p
    pa, pb = _dist(wa), _dist(wb)
    return float(np.sum(np.sqrt(pa * pb)))


def low_f_mask(landscape, *, res: int = 120, threshold_kbt: float = LOW_F_THRESHOLD_KBT):
    """Return ``(cv1_axis, cv2_axis, mask)`` of the low-F region."""
    c1, c2, f = landscape.grid(res=res)
    return c1, c2, (f <= (f.min() + float(threshold_kbt)))


def reference_pmf(landscape, *, axis: str = "cv1", res: int = 120):
    """True marginal free energy along an axis (min-shifted to 0), in kBT."""
    c1, c2, f = landscape.grid(res=res)
    if axis == "cv1":
        g = -np.log(np.exp(-f).sum(axis=1))
        x = c1
    else:
        g = -np.log(np.exp(-f).sum(axis=0))
        x = c2
    g = g - g.min()
    return x, g


def ideal_axis_ladder(landscape, *, axis: str = "cv1", beta: float = 1.0,
                      res: int = 120, target_overlap: float = 0.30,
                      k: float = 50.0) -> list:
    """Greedy minimal monotone center chain meeting target neighbor overlap.

    Starts at the low-F projection minimum and, at each step, advances to the
    farthest next center whose neighbor overlap still meets ``target_overlap``,
    yielding a near-minimal connected ladder over the relevant region.
    """
    x, pmf = reference_pmf(landscape, axis=axis, res=res)
    relevant = x[pmf <= (pmf.min() + LOW_F_THRESHOLD_KBT)]
    lo, hi = float(relevant.min()), float(relevant.max())
    if hi <= lo + 1e-9:
        return [lo]
    centers = [lo]
    cur = lo
    step_grid = np.linspace(0.0, hi - lo, 400)[1:]
    guard = 0
    while cur < hi - 1e-9 and guard < 500:
        guard += 1
        best = None
        for ds in step_grid:
            cand = min(hi, cur + float(ds))
            o = grid_overlap(landscape, Window(cur, k), Window(cand, k),
                             beta=beta, res=res, axis=axis)
            if o >= target_overlap:
                best = cand
            else:
                break
        if best is None or best <= cur + 1e-9:
            best = min(hi, cur + float(step_grid[0]))  # forced minimal step
        centers.append(float(best))
        cur = float(best)
        if len(centers) > 200:
            break
    out = [centers[0]]
    for c in centers[1:]:
        if c - out[-1] > 1e-6:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Oracle-optimal placement: R windows at greedy constant-overlap positions
# ---------------------------------------------------------------------------

def oracle_k_for_r(landscape: Landscape, R: int, *, axis: str = "cv1",
                   beta: float = 1.0, res: int = 120,
                   target_overlap: float = 0.30,
                   k_lo: float = 8.0, k_hi: float = 4000.0) -> float:
    """Spring constant k* such that the greedy constant-overlap walk yields >= R centers.

    The walk count is non-decreasing in k (tighter spring = narrower biased
    distribution = more windows to tile the range), so bisection is exact.
    Returns k_lo if count(k_lo) >= R (already enough at softest spring), k_hi
    if count(k_hi) < R (R exceeds what's achievable -- caller gets best-effort).
    Returns 0.0 for R <= 1 (unbiased single-window baseline).
    """
    if R <= 1:
        return 0.0

    def count(k: float) -> int:
        return len(ideal_axis_ladder(landscape, axis=axis, beta=beta, res=res,
                                     target_overlap=target_overlap, k=k))

    if count(k_hi) < R:
        return float(k_hi)
    if count(k_lo) >= R:
        return float(k_lo)
    # Binary search for smallest k with count >= R
    a, b = float(k_lo), float(k_hi)
    for _ in range(52):
        m = 0.5 * (a + b)
        if count(m) < R:
            a = m   # too few centers — tighten spring
        else:
            b = m   # count >= R — can still loosen
    return float(b)


def oracle_optimal_centers_1d(landscape: Landscape, R: int, *,
                               axis: str = "cv1", beta: float = 1.0,
                               res: int = 120,
                               target_overlap: float = 0.30) -> tuple:
    """R oracle-optimal 1D centers from the greedy constant-overlap walk.

    Finds k* such that the greedy walk generates >= R centers, then subsamples
    evenly to exactly R (preserving the barrier-dense structure).  Centers are
    dense near PMF barriers (where biased distributions are narrow) and sparse
    in basins (where distributions are wide) -- the correct staging geometry for
    umbrella sampling.

    Returns ``(centers: list[float], k: float)``.  For R=1, returns the midpoint
    of the low-F region with k=0 (unbiased baseline).
    """
    from .replica import low_f_support
    if R <= 1:
        lo, hi = low_f_support(landscape, res=res)
        return [0.5 * (lo + hi)], 0.0

    k = oracle_k_for_r(landscape, R, axis=axis, beta=beta, res=res,
                        target_overlap=target_overlap)
    ladder = ideal_axis_ladder(landscape, axis=axis, beta=beta, res=res,
                                target_overlap=target_overlap, k=k)
    n = len(ladder)
    if n == R:
        return list(ladder), float(k)
    if n < R:
        # k_hi was hit — best effort, return what we have
        return list(ladder), float(k)
    # Subsample to R evenly-spaced ladder positions (preserves barrier density)
    indices = np.round(np.linspace(0, n - 1, R)).astype(int)
    centers = [float(ladder[i]) for i in indices]
    return centers, float(k)


def oracle_optimal_centers_2d(landscape: Landscape, R: int, *,
                               n_iter: int = 30, res: int = 60,
                               scale1: float = 1.0, scale2: float = 2.0) -> list:
    """R oracle-optimal 2D centers via Lloyd CVT on the low-F region.

    Algorithm:
      1. Build the low-F candidate grid (F <= min + 5 kBT).
      2. Initialise with greedy Euclidean farthest-point sampling seeded at the
         global minimum, giving a well-spread starting layout.
      3. Run Lloyd iterations: assign each low-F grid point to the nearest
         center (Voronoi partition), then move each center to the centroid of
         its assigned grid points clamped to the nearest low-F candidate.
      4. Return the converged centers.

    Lloyd CVT minimises within-cell mean squared displacement (equivalent to
    uniform-density CVT), which is a robust proxy for equal MBAR coverage of the
    low-F region.  Unlike overlap-based farthest-point, it is numerically stable
    for well-separated basins (no BC≈0 degenerate case).

    Returns list of (cv1, cv2) float tuples.
    """
    c1, c2, f = landscape.grid(res=res)
    mask = f <= (f.min() + LOW_F_THRESHOLD_KBT)
    g1, g2 = np.meshgrid(c1, c2, indexing="ij")
    pts = np.column_stack([g1[mask].ravel(), g2[mask].ravel()])
    fvals = f[mask].ravel()

    if pts.shape[0] == 0:
        return [(0.5 * sum(landscape.cv1_bounds), 0.5 * sum(landscape.cv2_bounds))]
    if R >= pts.shape[0]:
        return [(float(p[0]), float(p[1])) for p in pts]

    # ---- Step 1: farthest-point initialisation (normalised CV space) ----
    P = pts / np.array([scale1, scale2])
    chosen = [int(np.argmin(fvals))]                    # seed: global minimum
    d = np.sqrt(((P - P[chosen[0]]) ** 2).sum(axis=1))
    for _ in range(R - 1):
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, np.sqrt(((P - P[j]) ** 2).sum(axis=1)))
    centers = pts[np.array(chosen)].copy()              # (R, 2)

    # ---- Step 2: Lloyd iterations ----
    for _ in range(n_iter):
        diffs = pts[:, None, :] - centers[None, :, :]  # (N, R, 2)
        dists2 = (diffs ** 2).sum(axis=2)               # (N, R)
        labels = dists2.argmin(axis=1)                  # (N,) cluster id
        new_centers = centers.copy()
        for r in range(R):
            idx = np.where(labels == r)[0]
            if idx.size == 0:
                continue
            mean = pts[idx].mean(axis=0)
            # Clamp centroid to nearest low-F grid point
            d_to_mean = ((pts - mean) ** 2).sum(axis=1)
            new_centers[r] = pts[int(np.argmin(d_to_mean))]
        if np.allclose(centers, new_centers, atol=1e-9):
            break
        centers = new_centers

    return [(float(c[0]), float(c[1])) for c in centers]


# ---------------------------------------------------------------------------
# Benchmark: oracle-optimal vs linspace PMF RMSE at fixed R, budget
# ---------------------------------------------------------------------------

def placement_comparison_1d(landscape: Landscape, R: int, budget: int, *,
                             seeds: list | None = None,
                             beta: float = 1.0, res: int = 120,
                             target_overlap: float = 0.30) -> dict:
    """Compare oracle-optimal vs linspace 1D window placement by MBAR PMF RMSE.

    For each placement strategy, samples ``budget // R`` raw samples per window,
    runs MBAR, and reports the mean PMF RMSE over ``seeds``.

    Returns dict with keys ``'oracle'`` and ``'linspace'``, each containing
    ``{'pmf_rmse': float, 'n_windows': int, 'k': float, 'centers': list}``.
    """
    from .replica import low_f_support, calibrate_k
    from .sampler import sample_window_exact, Window
    from .ess import tau_int, effective_count, thin_to_ess, BURN_IN
    from .metrics import pmf_recovery

    seeds = seeds or [0]
    n_per = max(1, budget // R)

    def _run_placement(centers_list, k_val):
        """Sample and MBAR-recover PMF for the given centers/k. Return mean RMSE."""
        rmses = []
        for s in seeds:
            rng = np.random.default_rng(s)
            samples, windows = {}, {}
            for i, c in enumerate(centers_list):
                w = Window(float(c), float(k_val), 0.0, 15.0)
                windows[i] = w
                raw = sample_window_exact(landscape, w, n_per, beta=beta, res=res, rng=rng)
                tau = tau_int(landscape, w)
                ne = int(effective_count(n_per, tau, burn_in=BURN_IN, is_new=True))
                samples[i] = thin_to_ess(raw, ne)
            nonempty = {i: s for i, s in samples.items() if len(s) > 0}
            wins_ne = {i: windows[i] for i in nonempty}
            if len(nonempty) >= 1:
                r = pmf_recovery(landscape, nonempty, wins_ne, axis="cv1", res=res)
                rmses.append(r["rmse_lowf_weighted"])
        return float(np.mean(rmses)) if rmses else float("nan")

    # Oracle-optimal placement
    oracle_centers, k_oracle = oracle_optimal_centers_1d(
        landscape, R, beta=beta, res=res, target_overlap=target_overlap)
    oracle_rmse = _run_placement(oracle_centers, k_oracle)

    # Linspace placement with calibrated fair-k
    lo, hi = low_f_support(landscape, res=res)
    lin_centers = list(np.linspace(lo, hi, R))
    _, k_lin, _, _ = calibrate_k(landscape, R, overlap_target=target_overlap, res=res)
    lin_rmse = _run_placement(lin_centers, k_lin)

    return {
        "oracle": {"pmf_rmse": oracle_rmse, "n_windows": R,
                   "k": float(k_oracle), "centers": oracle_centers},
        "linspace": {"pmf_rmse": lin_rmse, "n_windows": R,
                     "k": float(k_lin), "centers": lin_centers},
    }
