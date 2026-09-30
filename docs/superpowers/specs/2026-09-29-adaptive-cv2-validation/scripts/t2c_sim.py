"""T2 calibration of the 3.3 knobs: scenarios, exact truth and the two sampling regimes.

Read-only use of gareus (no gareus/ file is edited). Units are the harness's (F in kBT,
k in kBT/CV^2); the 3.3 code is driven with temperature_k = 1 / R_KCAL_MOL_K so RT = 1.

Regimes:
  indep -- one MALA chain per window (today's harness);
  re    -- Hamiltonian replica exchange: one walker per window, after every recorded sample
           ``attempts`` random neighbour-pair swap attempts (Metropolis on the two bias
           energies; the landscape cancels), so each window keeps exactly one walker and N_k is
           fixed. Replica identity is recorded per (sample, window), as the production Parquet
           ``replica`` column is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gareus.synth.landscapes import LANDSCAPES, LANDSCAPES_3D, Landscape


def _c9_like_narrow() -> Landscape:
    """chignolin_9-like CV2 shape at every CV1 (epoch_002 state 84's fit: a broad mode
    N(0.25, 0.30) with 70 % and a narrow mode N(1.2, 0.12) with 30 %, ~1.7 kBT between them),
    plus a gentle CV1 well. CV2 domain [-1.5, 2.5]."""
    def f(cv1, cv2):
        cv1 = np.asarray(cv1, float)
        z = np.asarray(cv2, float)
        broad = 0.7 * np.exp(-0.5 * ((z - 0.25) / 0.30) ** 2) / 0.30
        narrow = 0.3 * np.exp(-0.5 * ((z - 1.2) / 0.12) ** 2) / 0.12
        return -np.log(broad + narrow + 1e-300) + 0.5 * ((cv1 - 0.5) / 0.35) ** 2
    return Landscape("c9-like-narrow", f, (0.0, 1.0), (-1.5, 2.5), basins=((0.5, 0.25), (0.5, 1.2)))


CUSTOM = {"c9-like-narrow": _c9_like_narrow()}
from gareus.synth.sampler import Window
from gareus.synth.vec_langevin import run_chains


def k_for_spacing(spacing: float) -> float:
    """Today's per-gap rule: sigma_w = spacing / 1.5, k = 1 / sigma_w^2."""
    return 1.0 / (spacing / 1.5) ** 2


def grid(c1s, k1, c2s, k2, drop=()) -> List[Window]:
    out = []
    for i, a in enumerate(c1s):
        for j, b in enumerate(c2s):
            if (i, j) in drop:
                continue
            out.append(Window(float(a), float(k1), float(b), float(k2)))
    return out


@dataclass
class Scenario:
    name: str
    surface_name: str
    windows: List[Window]
    D: Tuple[float, ...]
    expect_r3: str            # "none" | "resolve" | "hidden" | "exact" (decide from the exact density)
    expect_r2: str            # "none" | "hole"
    hole: Optional[Tuple[float, Tuple[float, float]]] = None   # (c1, (z_lo, z_hi)) of the pre-registered hole
    wall: float = 0.0         # extra curvature for the dt rule (stiff / walls)
    start_cv3: Optional[Sequence[float]] = None
    notes: str = ""

    @property
    def surface(self):
        if self.surface_name in LANDSCAPES_3D:
            return LANDSCAPES_3D[self.surface_name]
        return CUSTOM[self.surface_name] if self.surface_name in CUSTOM else LANDSCAPES[self.surface_name]

    @property
    def dim(self) -> int:
        return 3 if self.surface_name in LANDSCAPES_3D else 2


def scenarios() -> Dict[str, Scenario]:
    """Pre-registered (before any run): expected R3 / R2 outcome per scenario."""
    D2, D3 = (1.0, 0.05), (1.0, 0.05, 0.05)
    hb_c1 = np.linspace(0.15, 0.85, 6)
    out = {}
    out["harmonic-bowl"] = Scenario("harmonic-bowl", "harmonic-bowl",
                                    grid(hb_c1, k_for_spacing(0.14), [-0.5, 0.5], k_for_spacing(1.0)), D2,
                                    "none", "none", notes="t2_campaign layout")
    out["stiff-bowl"] = Scenario("stiff-bowl", "stiff-bowl",
                                 grid(hb_c1, k_for_spacing(0.14), [-0.5, 0.5], k_for_spacing(1.0)), D2,
                                 "none", "none", wall=400.0, notes="t2_campaign layout; F''2 = 400")
    out["plateau-walls"] = Scenario("plateau-walls", "plateau-walls",
                                    grid(hb_c1, k_for_spacing(0.14), [-0.6, -0.2, 0.2, 0.6], k_for_spacing(0.4)),
                                    D2, "none", "none", wall=2000.0, notes="t2_campaign layout")
    # pre-registered as holes, measured NOT to be holes (smoke run, PMF error <= 0.15 kT there): the
    # adjacent CV1 columns' windows (1.5 sigma_w1 apart) cover a single dropped cell -> controls
    out["harmonic-drop1"] = Scenario("harmonic-drop1", "harmonic-bowl",
                                     grid(hb_c1, k_for_spacing(0.14), [-0.8, 0.0, 0.8], k_for_spacing(0.8),
                                          drop={(2, 1)}), D2, "none", "none",
                                     notes="3 CV2 rows, the centre window of column 2 removed (CV1 neighbours cover it)")
    out["plateau-drop1"] = Scenario("plateau-drop1", "plateau-walls",
                                    grid(hb_c1, k_for_spacing(0.14), [-0.6, -0.2, 0.2, 0.6], k_for_spacing(0.4),
                                         drop={(2, 1), (2, 2)}), D2, "none", "none", wall=2000.0,
                                    notes="the two inner windows of column 2 removed (CV1 neighbours cover it)")
    ncol = len(hb_c1)
    out["harmonic-rowgap"] = Scenario("harmonic-rowgap", "harmonic-bowl",
                                      grid(hb_c1, k_for_spacing(0.14), [-0.8, 0.0, 0.8], k_for_spacing(0.8),
                                           drop={(i, 1) for i in range(ncol)}), D2, "none", "hole",
                                      hole=(None, (-0.3, 0.3)),
                                      notes="CV2 row 0 removed in every column (window means 2.6 sampled sd apart)")
    out["plateau-rowgap"] = Scenario("plateau-rowgap", "plateau-walls",
                                     grid(hb_c1, k_for_spacing(0.14), [-0.6, -0.2, 0.2, 0.6], k_for_spacing(0.4),
                                          drop={(i, j) for i in range(ncol) for j in (1, 2)}), D2, "none", "hole",
                                     hole=(None, (-0.35, 0.35)), wall=2000.0,
                                     notes="the two inner CV2 rows removed in every column (4.5 sigma_w gap)")
    # added after the first sweep showed both row gaps are NOT holes (the +-rows' tails cover the
    # gap: PMF error there <= 0.4 kT at N = 2000): the same gap with stiff rows, so the gap centre
    # sits 3.6 / 5.4 / 8.1 kT up each window's spring (sampled, but thinly)
    for k2h in (20.0, 30.0, 45.0):
        nm = f"plateau-hardgap-k{int(k2h)}"
        out[nm] = Scenario(nm, "plateau-walls", grid(hb_c1, k_for_spacing(0.14), [-0.6, 0.6], k2h), D2, "none",
                           "hole", hole=(None, (-0.35, 0.35)), wall=2000.0,
                           notes=f"CV2 rows +-0.6 only, k2 = {k2h:g} (gap centre {0.5 * k2h * 0.36:.1f} kT up each spring)")
    db_c1 = np.linspace(0.2, 0.8, 5)
    out["dbl-saddle"] = Scenario("dbl-saddle", "slow-cv2-double-branch-asym",
                                 grid(db_c1, k_for_spacing(0.15), [-0.6, 0.0, 0.6], k_for_spacing(0.6)), D2,
                                 "exact", "none", notes="a CV2 row on the 6 kBT saddle: bimodal with crossings")
    out["dbl-gap"] = Scenario("dbl-gap", "slow-cv2-double-branch-asym",
                              grid(db_c1, k_for_spacing(0.15), [-0.4, 0.4], 4.0 * k_for_spacing(0.8)), D2,
                              "none", "none", notes="t2 k2x4: branches disconnected (an R1 case, not R2/R3)")
    out["gated-barrier"] = Scenario("gated-barrier", "gated-barrier",
                                    grid(np.linspace(0.1, 0.9, 7), k_for_spacing(0.8 / 6), [-0.6, 0.0, 0.6],
                                         k_for_spacing(0.6)), D2, "exact", "none", notes="t2_campaign layout")
    out["narrow-band"] = Scenario("narrow-band", "narrow-cv2-band-at-high-cv1",
                                  grid(np.linspace(0.1, 0.95, 6), k_for_spacing(0.17), [-1.4, 0.0, 1.4],
                                       k_for_spacing(1.4)), D2, "exact", "none",
                                  notes="t2_campaign layout; c9-like narrow upper mode under a soft k2")
    # c9-matched soft CV2 spring: 1.1795 kcal/mol/CV^2 at 300 K = 1.98 kBT/CV^2; rows at c9's CV2 centres
    out["c9-like-narrow"] = Scenario("c9-like-narrow", "c9-like-narrow",
                                     grid(np.linspace(0.15, 0.85, 5), k_for_spacing(0.175), [-0.785, 0.281, 1.348],
                                          1.1795 / (0.0019872041 * 300.0)), D2, "exact", "none",
                                     notes="chignolin_9-like: soft k2 over a broad + narrow CV2 mode pair")
    hc_c1 = np.round(np.linspace(0.15, 0.85, 5), 6)
    hw = grid(hc_c1, k_for_spacing(hc_c1[1] - hc_c1[0]), [-0.45, 0.0, 0.45], k_for_spacing(0.45))
    alt = [(-1.0 if (i % 2) else 1.0) for i in range(len(hw))]
    out["hidden-cv3"] = Scenario("hidden-cv3", "hidden-slow-cv3", hw, D3, "hidden", "none", start_cv3=alt,
                                 notes="t2_reseed layout; cv3 starts alternate +-1 across windows")
    out["hidden-cv3-weak"] = Scenario("hidden-cv3-weak", "hidden-slow-cv3-weak", hw, D3, "hidden", "none",
                                      start_cv3=alt, notes="weaker projection 0.2")
    return out


# ---- exact truth -------------------------------------------------------------------------

def _log_density_grid(sc: Scenario, w: Window, nz: int = 1201):
    """(z grid, log p(z) under window w), cv1 (and cv3) integrated out exactly on a grid."""
    surf = sc.surface
    lo1, hi1 = surf.cv1_bounds
    lo2, hi2 = surf.cv2_bounds
    s1 = 1.0 / math.sqrt(w.k1)
    x = np.linspace(max(lo1, w.center1 - 7 * s1), min(hi1, w.center1 + 7 * s1), 121)
    z = np.linspace(lo2, hi2, nz)
    from scipy.special import logsumexp  # noqa: PLC0415
    if sc.dim == 2:
        g1, g2 = np.meshgrid(x, z, indexing="ij")
        e = surf.energy(g1, g2) + 0.5 * w.k1 * (g1 - w.center1) ** 2 + 0.5 * w.k2 * (g2 - w.center2) ** 2
        lp = logsumexp(-e, axis=0)
    else:
        lo3, hi3 = surf.bounds[2]
        y = np.linspace(lo3, hi3, 241)
        lp = np.empty(z.size)
        for i in range(0, z.size, 200):
            zz = z[i:i + 200]
            g1, g2, g3 = np.meshgrid(x, zz, y, indexing="ij")
            e = (surf.energy(g1, g2, g3) + 0.5 * w.k1 * (g1 - w.center1) ** 2
                 + 0.5 * w.k2 * (g2 - w.center2) ** 2)
            lp[i:i + 200] = logsumexp(-e, axis=(0, 2))
    return z, lp - lp.max()


def exact_window_truth(sc: Scenario, w: Window) -> Dict[str, float]:
    """Exact within-window CV2 density: mean, var, and the deepest resolvable mode pair
    (depth = ln(min peak / valley) >= 1 kT, basin masses >= 10 %)."""
    z, lp = _log_density_grid(sc, w)
    p = np.exp(lp)
    p /= p.sum()
    m = float(np.sum(p * z))
    v = float(np.sum(p * (z - m) ** 2))
    peaks = [i for i in range(1, z.size - 1) if lp[i] > lp[i - 1] and lp[i] >= lp[i + 1] and lp[i] > -20]
    best = {"depth_kT": 0.0, "resolvable": False, "modes": None}
    if len(peaks) >= 2:
        # basins between valleys
        vals = [int(peaks[k] + np.argmin(lp[peaks[k]:peaks[k + 1] + 1])) for k in range(len(peaks) - 1)]
        bounds = [0] + vals + [z.size - 1]
        mass = [float(p[bounds[k]:bounds[k + 1] + 1].sum()) for k in range(len(peaks))]
        for k in range(len(peaks) - 1):
            depth = float(min(lp[peaks[k]], lp[peaks[k + 1]]) - lp[vals[k]])
            ok = depth >= 1.0 and min(mass[k], mass[k + 1]) >= 0.10
            if (ok, depth) > (best["resolvable"], best["depth_kT"]):
                best = {"depth_kT": depth, "resolvable": bool(ok),
                        "modes": [float(z[peaks[k]]), float(z[peaks[k + 1]])],
                        "masses": [mass[k], mass[k + 1]], "valley": float(z[vals[k]])}
    return {"mean": m, "var": v, **best}


def exact_slab_cv2(sc: Scenario, c1: float, half: float, edges: np.ndarray) -> np.ndarray:
    """Exact unbiased F(z-interval) = -ln P(interval | |cv1 - c1| <= half), per interval."""
    surf = sc.surface
    x = np.linspace(c1 - half, c1 + half, 41)
    z = np.linspace(edges[0], edges[-1], 40 * (edges.size - 1) + 1)
    from scipy.special import logsumexp  # noqa: PLC0415
    if sc.dim == 2:
        g1, g2 = np.meshgrid(x, z, indexing="ij")
        lp = logsumexp(-surf.energy(g1, g2), axis=0)
    else:
        lo3, hi3 = surf.bounds[2]
        y = np.linspace(lo3, hi3, 241)
        g1, g2, g3 = np.meshgrid(x, z, y, indexing="ij")
        lp = logsumexp(-surf.energy(g1, g2, g3), axis=(0, 2))
    p = np.exp(lp - lp.max())
    idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, edges.size - 2)
    mass = np.bincount(idx, weights=p, minlength=edges.size - 1)
    with np.errstate(divide="ignore"):
        return -np.log(mass / mass.sum())


# ---- sampling ----------------------------------------------------------------------------

def _dt(sc: Scenario) -> float:
    k1 = max(w.k1 for w in sc.windows)
    k2 = max(w.k2 or 0.0 for w in sc.windows)
    return float(min(2.0e-3, 0.3 / max(sc.D[0] * max(k1, 1.0), sc.D[1] * (k2 + sc.wall))))


def neighbour_pairs(windows: Sequence[Window]) -> List[Tuple[int, int]]:
    """Adjacent CV2 rows within a CV1 column and adjacent CV1 columns within a CV2 row."""
    c1s = sorted({round(w.center1, 6) for w in windows})
    c2s = sorted({round(w.center2, 6) for w in windows})
    pos = {(round(w.center1, 6), round(w.center2, 6)): i for i, w in enumerate(windows)}
    pairs = []
    for a in c1s:
        col = [pos[(a, b)] for b in c2s if (a, b) in pos]
        pairs += list(zip(col[:-1], col[1:]))
    for b in c2s:
        row = [pos[(a, b)] for a in c1s if (a, b) in pos]
        pairs += list(zip(row[:-1], row[1:]))
    return pairs


def _bias(w: Window, x: np.ndarray) -> np.ndarray:
    return 0.5 * w.k1 * (x[..., 0] - w.center1) ** 2 + 0.5 * w.k2 * (x[..., 1] - w.center2) ** 2


@dataclass
class Sim:
    x: np.ndarray            # (n_samples, n_windows, dim), state-indexed
    rep: np.ndarray          # (n_samples, n_windows) replica id at each window
    acceptance: float
    swap_acceptance: float
    mean_residence: float    # samples per (replica, window) stay
    wall_s: float
    meta: dict = field(default_factory=dict)


def simulate(sc: Scenario, regime: str, seed: int, n_samples: int, steps_per_sample: int = 20,
             attempts_per_window: float = 2.0, burn_in: int = 500) -> Sim:
    import time  # noqa: PLC0415
    t0 = time.time()
    rng = np.random.default_rng([int(seed), 7717])
    ws = sc.windows
    n = len(ws)
    x0 = np.zeros((n, sc.dim))
    x0[:, 0] = [w.center1 for w in ws]
    x0[:, 1] = [w.center2 for w in ws]
    if sc.dim == 3:
        x0[:, 2] = np.asarray(sc.start_cv3 if sc.start_cv3 is not None else np.ones(n), float)
    dt = _dt(sc)
    c1 = np.array([w.center1 for w in ws]); k1 = np.array([w.k1 for w in ws])
    c2 = np.array([w.center2 for w in ws]); k2 = np.array([w.k2 for w in ws])
    kw = dict(D=sc.D, dt=dt, steps_per_sample=steps_per_sample)
    # burn-in in own window (the US pull stand-in), not recorded
    x = run_chains(sc.surface, x0, 1, c1=c1, k1=k1, c2=c2, k2=k2, burn_in=burn_in, rng=rng, **kw).final
    X = np.empty((n_samples, n, sc.dim))
    R = np.empty((n_samples, n), dtype=np.int64)
    win_of_rep = np.arange(n)          # replica r sits at window win_of_rep[r]
    acc_sum, tries = 0.0, 0
    s_acc, s_try = 0, 0
    if regime == "indep":
        run = run_chains(sc.surface, x, n_samples, c1=c1, k1=k1, c2=c2, k2=k2, rng=rng, **kw)
        X[:] = np.transpose(run.samples, (1, 0, 2))
        R[:] = np.arange(n)[None, :]
        acc_sum, tries = run.acceptance, 1
    elif regime == "re":
        pairs = neighbour_pairs(ws)
        n_att = max(1, int(round(attempts_per_window * n)))
        for t in range(n_samples):
            wr = win_of_rep
            run = run_chains(sc.surface, x, 1, c1=c1[wr], k1=k1[wr], c2=c2[wr], k2=k2[wr], rng=rng, **kw)
            x = run.final
            acc_sum += run.acceptance; tries += 1
            rep_at = np.empty(n, dtype=np.int64)
            rep_at[wr] = np.arange(n)
            X[t] = x[rep_at]
            R[t] = rep_at
            for _ in range(n_att):
                i, j = pairs[int(rng.integers(len(pairs)))]
                ri, rj = rep_at[i], rep_at[j]
                d = (_bias(ws[i], x[rj]) + _bias(ws[j], x[ri]) - _bias(ws[i], x[ri]) - _bias(ws[j], x[rj]))
                s_try += 1
                if d <= 0 or rng.uniform() < math.exp(-d):
                    s_acc += 1
                    rep_at[i], rep_at[j] = rj, ri
                    win_of_rep[ri], win_of_rep[rj] = j, i
    else:
        raise ValueError(regime)
    # residence: mean length of runs of equal replica id per window
    lengths = []
    for w in range(n):
        r = R[:, w]
        brk = np.flatnonzero(np.diff(r) != 0)
        lengths.append((n_samples) / (brk.size + 1))
    return Sim(X, R, float(acc_sum / max(tries, 1)), float(s_acc / s_try) if s_try else float("nan"),
               float(np.mean(lengths)), time.time() - t0, {"dt": dt, "steps_per_sample": steps_per_sample})
