"""S1 -> S2 design inputs, all from the unbiased swarm (spec Sec.2 S1 products (2)(3)(4), Sec.3.5, Sec.8 item 8).

lambda rungs: DeltaV_lambda(x) = lambda * DeltaV_max(x) for the lower-bound boost, so adjacent-rung
acceptance is set by Delta_lambda * beta * sigma_lambda(DeltaV_max). sigma_lambda is estimated from
the lambda=0 swarm by reweighting w ~ exp(-beta*lambda*DeltaV_max); ESS = (sum w)^2 / sum(w^2) bounds
how far that estimate can be trusted. Grow the ladder from 0 with Delta_lambda = target/(beta*sigma_lambda)
until 1.

k(CV1): sigma_w^2 = kT/(k + F'') with sigma_w = spacing/overlap_sigma -> k = kT/sigma_w^2 - F'',
clamped to [k_min, k_max].

Coverage, not [0,1] (measured on r7, 2026-09-07): the accessible heavy-CV1 range of the seed library
is ~0-0.069 and a restrained pull cannot move the CV. Every quantity here is evaluated against the
observed coverage range of the swarm's/library's CV1 samples, never against a fixed [0,1] grid.
"""
from __future__ import annotations

import csv

from gareus.io import write_csv_atomic
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj

R_KJ_MOL_K = 0.008314462618
R_KCAL_MOL_K = 0.0019872041


def deltav_max_kj(v_pep_kj, v_dih_kj, env: PepGamdEnvelope) -> np.ndarray:
    """The boost each frame would receive at rung lambda = 1 (the top of the ladder)."""
    return np.asarray(pep_gamd_boost_kj(np.asarray(v_pep_kj, dtype=float), np.asarray(v_dih_kj, dtype=float), 1.0, env), dtype=float)


def _reweighted_sigma_and_ess(dv: np.ndarray, beta: float, lam: float) -> tuple[float, float]:
    """Reweight lambda=0 swarm frames to the ensemble at rung lambda via w ~ exp(-beta*lambda*dv)."""
    logw = -beta * lam * dv
    logw = logw - logw.max()
    w = np.exp(logw)
    w = w / w.sum()
    mean = float((w * dv).sum())
    var = float((w * (dv - mean) ** 2).sum())
    ess = float(1.0 / (w ** 2).sum())
    return math.sqrt(max(var, 0.0)), ess


def design_lambda_ladder(deltav_kj, temperature_k: float, *, target_beta_sigma: float = 1.0,
                          min_rungs: int = 3, max_rungs: int = 12, ess_floor: int = 50) -> dict:
    """Grow lambda from 0 to 1 with adjacent-rung spacing set by target_beta_sigma * (beta*sigma_lambda)^-1.

    sigma_lambda(DeltaV_max) is the reweighted standard deviation of DeltaV_max under the ensemble at
    rung lambda (reweighting swarm frames sampled at lambda=0). Rungs whose reweighted ESS falls below
    ess_floor are flagged via extrapolated_from_rung: everything from that rung on is extrapolated
    beyond what the lambda=0 swarm can support and must be confirmed by the S3 stage.
    """
    dv = np.asarray(deltav_kj, dtype=float)
    dv = dv[np.isfinite(dv)]
    if dv.size < 2:
        raise ValueError("need at least two finite DeltaV_max samples")
    beta = 1.0 / (R_KJ_MOL_K * float(temperature_k))

    lambdas: list[float] = [0.0]
    while lambdas[-1] < 1.0 and len(lambdas) < int(max_rungs):
        lam = lambdas[-1]
        sig, _ess = _reweighted_sigma_and_ess(dv, beta, lam)
        step = float(target_beta_sigma) / (beta * sig) if sig > 0 else 1.0
        lambdas.append(min(1.0, lam + step))
    if lambdas[-1] < 1.0:
        # max_rungs reached before lambda=1: force the top rung so production always has one.
        lambdas[-1] = 1.0

    while len(lambdas) < int(min_rungs):
        # Too few rungs for the floor: bisect the currently-widest gap.
        gaps = np.diff(lambdas)
        i = int(np.argmax(gaps))
        lambdas.insert(i + 1, 0.5 * (lambdas[i] + lambdas[i + 1]))

    sigmas: list[float] = []
    esss: list[float] = []
    for lam in lambdas:
        sig, ess = _reweighted_sigma_and_ess(dv, beta, lam)
        sigmas.append(sig)
        esss.append(ess)

    extrapolated: Optional[int] = None
    for i, ess in enumerate(esss):
        if i > 0 and ess < ess_floor:
            extrapolated = i
            break

    return {
        "lambdas": [float(x) for x in lambdas],
        "sigma_kj_per_rung": [float(x) for x in sigmas],
        "ess_per_rung": [float(x) for x in esss],
        "extrapolated_from_rung": extrapolated,
        "target_beta_sigma": float(target_beta_sigma),
        "beta_sigma_lambda0": float(beta * sigmas[0]),
        "n_samples": int(dv.size),
    }


def fsf_floor_per_rung(lambdas, env: PepGamdEnvelope, *, warn_threshold: float = 0.5) -> dict:
    """ForceScalingFactor at V = Vmin for each rung.

    gamd-openmm's lower-bound FSF is 1 - k0*(E - V)/(Vmax - Vmin) (gamd/langevin/base_integrator.py:395),
    unclamped: at V = Vmin this is 1 - k0. Under the ladder, k0 = lambda * k0max per channel, so the
    floor is 1 - lambda*k0max. Pep-GaMD leaves water-water at full strength, so a floor near 0 lets the
    solvent collapse into the peptide unopposed (S3 attempts 6-7: NaN coordinates at k0_Total = 1). The
    plan does NOT clamp the FSF here -- that would be a method change and is left as a user decision.

    A dihedral-only envelope (``env.has_total is False``, per the optional field noted in this task's
    brief) has no meaningful Total channel: ``Total``/``top_rung_floor_total`` are reported as ``None``
    and ``warn`` is driven off the Dihedral channel's top-rung floor instead.
    """
    lam = [float(x) for x in lambdas]
    has_total = bool(getattr(env, "has_total", True))
    dih = [1.0 - l * float(env.k0max_dih) for l in lam]
    top_dih = dih[-1] if dih else 1.0
    if has_total:
        tot = [1.0 - l * float(env.k0max_total) for l in lam]
        top_tot = tot[-1] if tot else 1.0
        warn = bool(top_tot < float(warn_threshold))
    else:
        tot = None
        top_tot = None
        warn = bool(top_dih < float(warn_threshold))
    return {
        "Total": tot,
        "Dihedral": dih,
        "top_rung_floor_total": top_tot,
        "warn": warn,
        "warn_threshold": float(warn_threshold),
        "note": "FSF floor = 1 - lambda*k0max at V = Vmin (unclamped lower-bound formula); "
                "a floor near 0 on the top rung is the NaN mechanism seen in S3 attempts 6-7",
    }


def window_sigma_cv(k_kcal: float, temperature_k: float) -> float:
    """sigma_w = sqrt(kT/k) in CV units; at 300 K, k=250 -> 0.049, k=800 -> 0.027 (r7's ~0.07-wide range)."""
    return math.sqrt(R_KCAL_MOL_K * float(temperature_k) / float(k_kcal))


def n_resolvable_windows(coverage_range: float, temperature_k: float, *,
                          k_max_kcal: float = 1200.0, overlap_sigma: float = 1.5) -> int:
    """floor(coverage_range / (overlap_sigma * sigma_w(k_max))): the true window-count ceiling.

    Spec S2's "16 centres" is only reachable when the accessible CV1 range is wide enough to hold 16
    non-overlapping windows at the stiffest allowed force constant. With r7's measured coverage
    (~0.069) and k_max=1200 kcal/mol/CV^2 at 300 K, this resolves to 2, not 16 -- the plan writes
    min(requested_n_windows, n_resolvable_windows(...)) and records why.
    """
    sigma_w_min = window_sigma_cv(float(k_max_kcal), float(temperature_k))
    return int(math.floor(float(coverage_range) / (float(overlap_sigma) * sigma_w_min)))


# Upper-bound probing walks DOWN the observed seed values. A real swarm pools tens of
# thousands of frames, so probing every unique value is O(n_seeds^2) work for a bound that
# a few hundred evenly-spaced observed candidates locate just as well.
MAX_UPPER_BOUND_PROBES = 256


def probe_centre_seed_support(centres, seed_cv1) -> np.ndarray:
    """|cv1_seed - centre| of the nearest available seed, one entry per centre."""
    c = np.asarray(centres, dtype=float)
    s = np.asarray(seed_cv1, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.full(c.shape, np.inf)
    return np.min(np.abs(s[None, :] - c[:, None]), axis=1)


def autotune_cv1_upper_bound(cv1, seed_cv1, *, n_windows: int, lo_q: float = 0.005,
                             hi_q: float = 0.995, temperature_k: float = 300.0,
                             k_max_kcal: float = 1200.0,
                             max_seed_gap_sigma: float = 0.5) -> dict:
    """Highest upper bound whose every centre has a seed within tolerance.

    tol = max_seed_gap_sigma * window_sigma_cv(k_max_kcal, temperature_k) (the tightest window
    the design may use, so a seed inside tol is on-centre for every window at least that soft).

    Start from hi = quantile(cv1, hi_q). Probe the centres it implies. If every centre is supported,
    accept it unchanged. Otherwise walk DOWN the observed seed values (descending unique seed_cv1
    at or below the initial hi -- autofound from data, never a fixed grid) and accept the FIRST
    (highest) candidate whose centres are all supported. Raise ValueError only when no candidate is
    supported, and put the measured worst gap and the tolerance in the message.

    Returns {"lo", "hi", "hi_initial", "centres", "nearest_seed_gap", "tol",
             "autotuned": bool, "n_probes": int}.
    """
    v = np.asarray(cv1, dtype=float)
    v = v[np.isfinite(v)]
    seeds = np.asarray(seed_cv1, dtype=float)
    seeds = seeds[np.isfinite(seeds)]

    lo = max(0.0, float(np.quantile(v, lo_q)))
    hi_initial = min(1.0, float(np.quantile(v, hi_q)))
    if hi_initial <= lo:
        raise ValueError("degenerate CV1 coverage: quantile range collapsed to a point")

    tol = float(max_seed_gap_sigma) * window_sigma_cv(float(k_max_kcal), float(temperature_k))

    def _probe(hi_candidate: float):
        centres = np.linspace(lo, hi_candidate, int(n_windows))
        return centres, probe_centre_seed_support(centres, seeds)

    centres, gaps = _probe(hi_initial)
    n_probes = 1
    best_hi, best_centres, best_gaps = hi_initial, centres, gaps
    best_worst_gap = float(np.max(gaps)) if gaps.size else float("inf")

    if best_worst_gap <= tol:
        return {
            "lo": lo, "hi": hi_initial, "hi_initial": hi_initial,
            "centres": centres, "nearest_seed_gap": gaps, "tol": tol,
            "autotuned": False, "n_probes": n_probes,
        }

    candidates = sorted({float(x) for x in seeds if lo < x <= hi_initial}, reverse=True)
    if len(candidates) > MAX_UPPER_BOUND_PROBES:
        # Subsample the OBSERVED values (never a synthetic grid): evenly spaced indices keep
        # the walk descending and bounded, and always retain the highest candidate.
        idx = np.unique(np.linspace(0, len(candidates) - 1, MAX_UPPER_BOUND_PROBES).astype(int))
        candidates = [candidates[i] for i in idx]
    for candidate_hi in candidates:
        centres, gaps = _probe(candidate_hi)
        n_probes += 1
        worst_gap = float(np.max(gaps)) if gaps.size else float("inf")
        if worst_gap < best_worst_gap:
            best_hi, best_centres, best_gaps, best_worst_gap = candidate_hi, centres, gaps, worst_gap
        if worst_gap <= tol:
            return {
                "lo": lo, "hi": candidate_hi, "hi_initial": hi_initial,
                "centres": centres, "nearest_seed_gap": gaps, "tol": tol,
                "autotuned": True, "n_probes": n_probes,
            }

    raise ValueError(
        f"no CV1 upper bound has seed support within tol={tol:.4f}: the best candidate probed "
        f"(hi={best_hi:.4f}, out of {n_probes} probed) still has a worst-case nearest-seed gap "
        f"of {best_worst_gap:.4f} > tol={tol:.4f} -- extend the swarm or widen "
        "--swarm-max-seed-gap-sigma"
    )


def cv1_centers_from_samples(cv1, n_windows: int = 16, lo_q: float = 0.005, hi_q: float = 0.995, *,
                              seed_cv1=None, temperature_k: float = 300.0, k_max_kcal: float = 1200.0,
                              max_seed_gap_sigma: float = 0.5, probe_out: Optional[dict] = None) -> np.ndarray:
    """Centres spanning the swarm's OBSERVED CV1 coverage, never a fixed [0,1] grid.

    The static library-quantile veto that used to live here (library_cv1/library_q, raising
    when a centre exceeded the GENPEPT library's q99) is REMOVED -- measured on a real
    99-member swarm run (2026-09-08) it was checking the wrong pool and its own advice was
    inert:
      - Round-0 windows are seeded from the swarm's OWN frames (select_window_seed_frames ->
        seed bank, seed_selection_mode: active-cv), never pulled from the GENPEPT library.
      - Every proposed centre -- including the one that tripped the old veto at 0.0595 --
        had 3 distinct-member swarm frames within 3e-4 of it, so a pull restrained to that
        centre had real support the library-based cap could not see.
      - centres = np.linspace(lo, hi, n) always ends at hi, so lowering n_windows can never
        lower the top centre (n = 2..16 all gave the identical top centre); "lower
        --swarm-n-windows" was never actionable advice.
    The replacement (autotune_cv1_upper_bound, wired in via seed_cv1) probes the SAME seed
    pool the windows will actually start from and only lowers hi when a centre genuinely has
    no nearby seed.
    """
    v = np.asarray(cv1, dtype=float)
    v = v[np.isfinite(v)]
    if seed_cv1 is not None:
        result = autotune_cv1_upper_bound(
            v, seed_cv1, n_windows=n_windows, lo_q=lo_q, hi_q=hi_q, temperature_k=temperature_k,
            k_max_kcal=k_max_kcal, max_seed_gap_sigma=max_seed_gap_sigma,
        )
        if probe_out is not None:
            probe_out.update(result)
        return np.asarray(result["centres"], dtype=float)

    lo, hi = float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q))
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    if hi <= lo:
        raise ValueError("degenerate CV1 coverage: quantile range collapsed to a point")
    return np.linspace(lo, hi, int(n_windows))


def cv1_curvature_kcal(cv1, centers, temperature_k: float, *, n_hist: int = 60, smooth_bins: int = 2) -> np.ndarray:
    """F'' in kcal/mol/CV^2 at each centre, from a smoothed histogram spanning [min(cv1), max(cv1)]
    (the observed coverage range, NOT [0,1]). 0 where the histogram is empty at a centre."""
    v = np.asarray(cv1, dtype=float)
    v = v[np.isfinite(v)]
    kT = R_KCAL_MOL_K * float(temperature_k)
    hist, edges = np.histogram(v, bins=int(n_hist), range=(float(v.min()), float(v.max()) + 1e-12))
    p = hist.astype(float)
    if smooth_bins > 0:
        kern = np.exp(-0.5 * (np.arange(-3 * smooth_bins, 3 * smooth_bins + 1) / smooth_bins) ** 2)
        kern = kern / kern.sum()
        p = np.convolve(p, kern, mode="same")
    mid = 0.5 * (edges[:-1] + edges[1:])
    h = mid[1] - mid[0]
    with np.errstate(divide="ignore"):
        F = np.where(p > 0, -kT * np.log(np.where(p > 0, p, 1.0)), np.nan)
    F2 = np.full_like(F, np.nan)
    F2[1:-1] = (F[2:] - 2 * F[1:-1] + F[:-2]) / h ** 2
    out = np.zeros(len(centers), dtype=float)
    for i, c in enumerate(centers):
        if c < edges[0] or c > edges[-1]:
            continue  # centre lies outside the observed swarm coverage: no data, F'' = 0
        j = int(np.clip(np.searchsorted(mid, c), 1, len(mid) - 2))
        out[i] = F2[j] if np.isfinite(F2[j]) else 0.0
    return out


def cv1_force_constants_from_curvature(centers, curvature_kcal, temperature_k: float, *, overlap_sigma: float = 1.5,
                                        k_min_kcal: float = 5.0, k_max_kcal: float = 1200.0,
                                        coverage_range: Optional[float] = None,
                                        warnings_out: Optional[list] = None) -> List[float]:
    """k from local spacing and curvature: k = kT/sigma_w^2 - F'', clamped to [k_min, k_max].

    sigma_w is judged against coverage_range (the swarm's observed CV1 span, e.g. ~0.07 for r7), never
    against [0,1]. A window whose sigma_w exceeds half the coverage range (the whole range is then
    effectively one window) is appended to warnings_out when both coverage_range and warnings_out are
    given -- callers write these into ladder_design.json["k_warnings"].
    """
    c = np.asarray(centers, dtype=float)
    f2 = np.asarray(curvature_kcal, dtype=float)
    kT = R_KCAL_MOL_K * float(temperature_k)
    if c.size < 2:
        return [float(k_max_kcal)]
    sp = np.diff(c)
    local = np.empty_like(c)
    local[0] = sp[0]
    local[-1] = sp[-1]
    if c.size > 2:
        local[1:-1] = 0.5 * (sp[:-1] + sp[1:])
    ks: list[float] = []
    for i, (spacing, curv) in enumerate(zip(local, f2)):
        sigma_w = max(1e-5, float(spacing) / float(overlap_sigma))
        k = kT / sigma_w ** 2 - (float(curv) if math.isfinite(curv) else 0.0)
        k_cl = float(min(float(k_max_kcal), max(float(k_min_kcal), k)))
        ks.append(k_cl)
        if coverage_range and warnings_out is not None:
            sw = window_sigma_cv(k_cl, temperature_k)
            if sw > 0.5 * float(coverage_range):
                warnings_out.append({
                    "window": i,
                    "center": float(c[i]),
                    "k_kcal": k_cl,
                    "sigma_w": sw,
                    "sigma_w_over_range": sw / float(coverage_range),
                    "note": "window width exceeds half the accessible CV1 range",
                })
    return ks


def write_ladder_windows_csv(path, centers, ks_kcal, lambdas) -> Path:
    """Full cross product of CV1 centres x lambda rungs (never truncate the exchange graph)."""
    path = Path(path)
    lambdas = list(lambdas)

    def _rows():
        n = 0
        for c, k in zip(centers, ks_kcal):
            for lam in lambdas:
                yield [n, "contacts", f"{float(c):.6f}", f"{float(k):.4f}", f"{float(lam):.6f}"]
                n += 1

    # Staged and renamed: this file IS the production state space, and a
    # truncated copy silently yields fewer (window, rung) states rather than
    # an error. The rename also means the previous ladder stays readable
    # right up to the instant the new one replaces it.
    return write_csv_atomic(
        path,
        ["window", "primary_cv_mode", "primary_cv_center", "primary_cv_k_kcal", "gamd_lambda"],
        _rows(),
    )


# ---------------------------------------------------------------------------
# Two-dimensional layout for an automatically selected (CV1, CV2) pair
# ---------------------------------------------------------------------------

R_KCAL_PER_MOL_K = 0.0019872041
R_KJ_PER_MOL_K = 0.0083144626


def design_2d_layout(n1: int, n2: int, *, n_rungs: int, max_replicas: int) -> dict:
    """Spatial states for a 2-D ladder under the replica cap.

    ``n1 * n2`` cells fit -> a joint grid. Otherwise a sparse layout: one bridge
    (both umbrellas off), every axis-1 window with ``k2 = 0``, every axis-2 window
    with ``k1 = 0``, then joint patches filling the diagonal band outward until
    the cap. Cells are ``(i1 | None, i2 | None)``; ``None`` = that axis unrestrained.
    """
    if int(max_replicas) <= 0:
        raise ValueError("max_replicas must be set and positive; the default 0 cannot size a 2-D ladder")
    if int(n_rungs) <= 0 or int(n1) <= 0 or int(n2) <= 0:
        raise ValueError("n1, n2 and n_rungs must be positive")
    cap = int(max_replicas) // int(n_rungs)
    if n1 * n2 <= cap:
        return {"kind": "joint", "spatial_states": n1 * n2, "n1": int(n1), "n2": int(n2),
                "cells": [(i, j) for i in range(n1) for j in range(n2)]}
    cells: list = [(None, None)] + [(i, None) for i in range(n1)] + [(None, j) for j in range(n2)]
    budget = cap - len(cells)
    if budget < 0:
        raise ValueError(f"{n1} + {n2} + 1 axis states exceed the {cap}-state cap; reduce windows")
    order = sorted(((i, j) for i in range(n1) for j in range(n2)),
                   key=lambda ij: (abs(ij[0] / max(n1 - 1, 1) - ij[1] / max(n2 - 1, 1)), ij))
    cells += order[:budget]
    return {"kind": "sparse", "spatial_states": len(cells), "n1": int(n1), "n2": int(n2), "cells": cells}


# ---------------------------------------------------------------------------------------------
# Exploration-preserving layout (repair spec F05; findings I02, I03)
# ---------------------------------------------------------------------------------------------

ROLE_UNRESTRAINED_ANCHOR = "unrestrained_anchor"      # k1 = k2 = 0 on every rung, including lambda = 0
ROLE_REGION_REPRESENTATIVE = "region_representative"  # CV1 axis state at a discovered region's representative
ROLE_AXIS = "axis"                                      # single-axis soft bridge state
ROLE_JOINT = "joint"
LAYOUT_STATUS_PROPOSED = "PROPOSED"
LAYOUT_STATUS_INSUFFICIENT = "INSUFFICIENT_REGION_COVERAGE"
LAYOUT_PLAN_VERSION = "layout_plan_v1"


def design_exploration_layout(n1: int, n2: int, *, n_rungs: int, max_replicas: int,
                              region_centre_indices=None) -> dict:
    """Spatial states for a 2-D ladder with the exploration invariants of spec F05.

    Builder order, at ANY replica cap:
      1. one spatial state with k1 = k2 = 0 (an actual unbiased target at lambda = 0) -- one
         complete rung stack, reserved before anything else;
      2. a CV1-axis representative (k2 = 0) for every discovered region (``region_centre_indices``
         are the CV1 centre indices that represent regions);
      3. if the whole n1 x n2 grid still fits, every joint cell; otherwise the remaining CV1 axis
         states, the CV2 axis states and joint cells filling the diagonal band outward.
    Every spatial state gets a complete rung stack. If (1) + (2) alone exceed the cap the plan
    is INSUFFICIENT_REGION_COVERAGE with requested/available counts; nothing is silently dropped.

    Cells are ``(i1 | None, i2 | None)``; ``None`` = that axis unrestrained (k = 0, finite
    placeholder centre). Roles live in the returned plan, never in the physics rows.
    """
    if int(max_replicas) <= 0:
        raise ValueError("max_replicas must be set and positive; the default 0 cannot size a 2-D ladder")
    if int(n_rungs) <= 0 or int(n1) <= 0 or int(n2) <= 0:
        raise ValueError("n1, n2 and n_rungs must be positive")
    cap = int(max_replicas) // int(n_rungs)
    reps = sorted({int(i) for i in (region_centre_indices or [])})
    if any(i < 0 or i >= n1 for i in reps):
        raise ValueError("region centre indices must index the CV1 centres")
    cells = [(None, None)]
    roles = [ROLE_UNRESTRAINED_ANCHOR]
    mandatory = 1 + len(reps)
    if mandatory > cap:
        return {"version": LAYOUT_PLAN_VERSION, "status": LAYOUT_STATUS_INSUFFICIENT, "kind": None,
                "spatial_states": 0, "n1": int(n1), "n2": int(n2), "n_rungs": int(n_rungs), "cap_spatial": cap,
                "mandatory_spatial": mandatory, "requested_replicas": mandatory * int(n_rungs),
                "available_replicas": int(max_replicas), "cells": [], "state_roles": [],
                "reason": f"{mandatory} mandatory rung stacks ({mandatory * int(n_rungs)} replicas) exceed the "
                          f"cap of {int(max_replicas)} replicas / {cap} spatial states"}
    for i in reps:
        cells.append((i, None)); roles.append(ROLE_REGION_REPRESENTATIVE)
    remaining = cap - len(cells)
    if n1 * n2 <= remaining:
        kind = "joint"
        for i in range(n1):
            for j in range(n2):
                cells.append((i, j)); roles.append(ROLE_JOINT)
    else:
        kind = "sparse"
        axis1 = [(i, None) for i in range(n1) if i not in reps]
        axis2 = [(None, j) for j in range(n2)]
        soft = axis1 + axis2
        take = soft[:remaining]
        cells += take; roles += [ROLE_AXIS] * len(take)
        remaining -= len(take)
        if remaining > 0:
            order = sorted(((i, j) for i in range(n1) for j in range(n2)),
                           key=lambda ij: (abs(ij[0] / max(n1 - 1, 1) - ij[1] / max(n2 - 1, 1)), ij))
            cells += order[:remaining]; roles += [ROLE_JOINT] * min(remaining, len(order))
    plan = {"version": LAYOUT_PLAN_VERSION, "status": LAYOUT_STATUS_PROPOSED, "kind": kind,
            "spatial_states": len(cells), "n1": int(n1), "n2": int(n2), "n_rungs": int(n_rungs),
            "cap_spatial": cap, "mandatory_spatial": mandatory, "cells": cells, "state_roles": roles,
            "mandatory_spatial_indices": list(range(mandatory)),
            "region_representative_centre_indices": reps}
    _check_layout_invariants(plan)
    return plan


def _check_layout_invariants(plan: dict) -> None:
    cells, roles = plan["cells"], plan["state_roles"]
    if len(cells) != len(roles):
        raise AssertionError("layout roles and cells disagree in length")
    if cells[0] != (None, None) or roles[0] != ROLE_UNRESTRAINED_ANCHOR:
        raise AssertionError("the unrestrained anchor stack must be the first spatial state")
    if len(set(cells)) != len(cells):
        raise AssertionError("duplicate spatial states in the layout")
    if plan["spatial_states"] > plan["cap_spatial"]:
        raise AssertionError("layout exceeds the replica cap")


def layout_rows(plan: dict, centers1, ks1, centers2, ks2) -> list:
    """One physics row per spatial cell (the canonical fields only). An unrestrained axis gets
    k = 0 exactly and a finite placeholder centre (the axis midpoint) so no reader ever meets
    ``0 * nan``; roles stay in the companion plan (spec F05)."""
    c1 = np.asarray(centers1, dtype=float); c2 = np.asarray(centers2, dtype=float) if centers2 is not None and len(centers2) else np.array([0.0])
    mid1 = float(0.5 * (c1.min() + c1.max())) if c1.size else 0.0
    mid2 = float(0.5 * (c2.min() + c2.max())) if c2.size else 0.0
    rows = []
    for i1, i2 in plan["cells"]:
        rows.append({"center1": float(c1[i1]) if i1 is not None else mid1,
                     "k1": float(ks1[i1]) if i1 is not None else 0.0,
                     "center2": float(c2[i2]) if i2 is not None else mid2,
                     "k2": float(ks2[i2]) if i2 is not None else 0.0})
    return rows


def layout_plan_record(plan: dict, rows: list, lambdas, *, region_inventory: Optional[dict] = None,
                       region_of_centre=None) -> dict:
    """The companion artifact (``layout_plan.json``): per-state roles, region coverage, mandatory
    state ids and unresolved regions -- digested separately from the physics rows."""
    lambdas = [float(l) for l in lambdas]
    states = []
    state_id = 0
    for cell_index, (cell, role) in enumerate(zip(plan["cells"], plan["state_roles"])):
        for lam in lambdas:
            states.append({"state_id": state_id, "spatial_index": cell_index, "cell": [cell[0], cell[1]],
                           "role": role, "gamd_lambda": lam, "mandatory": role in (ROLE_UNRESTRAINED_ANCHOR, ROLE_REGION_REPRESENTATIVE),
                           "center1": rows[cell_index]["center1"], "k1": rows[cell_index]["k1"],
                           "center2": rows[cell_index]["center2"], "k2": rows[cell_index]["k2"]})
            state_id += 1
    coverage = None
    if region_inventory is not None and region_of_centre is not None:
        covered = {}
        for cell, role in zip(plan["cells"], plan["state_roles"]):
            if cell[0] is not None and role in (ROLE_REGION_REPRESENTATIVE, ROLE_AXIS, ROLE_JOINT):
                covered.setdefault(region_of_centre[cell[0]], []).append(role)
        coverage = {r["region_id"]: {"covered_by": covered.get(r["region_id"], []),
                                     "unresolved": r["region_id"] not in covered}
                    for r in region_inventory.get("regions", [])}
    return {"version": plan["version"], "status": plan["status"], "kind": plan["kind"], "n_rungs": len(lambdas),
            "spatial_states": plan["spatial_states"], "n_states": len(states), "cap_spatial": plan["cap_spatial"],
            "mandatory_spatial": plan["mandatory_spatial"], "mandatory_state_ids": [s["state_id"] for s in states if s["mandatory"]],
            "states": states, "region_coverage": coverage, "region_inventory": region_inventory,
            "unresolved": (region_inventory or {}).get("unresolved", []) if region_inventory else []}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cdf, v))


def reweighted_cv2_centers(z2, deltav_kj, temperature_k: float, lambdas, n_windows: int, *,
                           lo_q: float = 0.02, hi_q: float = 0.98) -> dict:
    """CV2 centres spanning the union of the lambda = 0 and top-rung z2 distributions.

    The Pep-GaMD dihedral channel boosts exactly the torsion energies z2 is built
    from, so rungs above lambda = 0 sample a broader z2 than the unbiased swarm.
    Reweighting the swarm frames by exp(-beta lambda DeltaV_max) -- the same weights
    ladder_design uses for the rung ESS -- predicts each rung's quantiles; the
    centres cover the union so no rung starts outside the grid. The ESS per rung
    says how much to trust each prediction.
    """
    z = np.asarray(z2, dtype=float)
    dv = np.asarray(deltav_kj, dtype=float)
    if z.shape != dv.shape or z.ndim != 1:
        raise ValueError("z2 and deltav_kj must be 1-D of equal length")
    finite = np.isfinite(z) & np.isfinite(dv)
    z, dv = z[finite], dv[finite]
    if z.size < 2:
        raise ValueError("need at least two finite frames to place CV2 centres")
    beta = 1.0 / (R_KJ_PER_MOL_K * float(temperature_k))
    quantiles: dict = {}
    ess: dict = {}
    for lam in lambdas:
        w = np.exp(-beta * float(lam) * (dv - dv.min()))
        w = w / w.sum()
        quantiles[float(lam)] = (_weighted_quantile(z, w, lo_q), _weighted_quantile(z, w, hi_q))
        ess[float(lam)] = float(1.0 / np.sum(w ** 2))
    lo = min(q[0] for q in quantiles.values())
    hi = max(q[1] for q in quantiles.values())
    return {"centers": np.linspace(lo, hi, int(n_windows)), "per_rung_quantiles": quantiles,
            "per_rung_ess": ess, "lo_q": float(lo_q), "hi_q": float(hi_q)}


def cv2_force_constants_per_gap(centers, temperature_k: float, *, overlap_sigma: float = 1.5,
                                k_min_kcal: float, k_max_kcal: float) -> list:
    """k = RT / (spacing / overlap_sigma)^2 per centre, spacing = the smaller adjacent gap."""
    c = np.asarray(centers, dtype=float)
    if c.size < 2 or np.any(np.diff(c) <= 0):
        raise ValueError("CV2 centres must be at least two and strictly increasing")
    gaps = np.diff(c)
    rt = R_KCAL_PER_MOL_K * float(temperature_k)
    out = []
    for i in range(c.size):
        left = gaps[i - 1] if i > 0 else np.inf
        right = gaps[i] if i < gaps.size else np.inf
        spacing = float(min(left, right))
        sigma = spacing / float(overlap_sigma)
        out.append(float(min(max(rt / sigma ** 2, k_min_kcal), k_max_kcal)))
    return out


def write_ladder_windows_2d_csv(path, rows, lambdas) -> Path:
    """(center1, k1, center2, k2) rows x lambda rungs; an unrestrained axis has k = 0.

    A ``None`` centre on an unrestrained axis is written empty. The production loader
    accepts that only together with ``k = 0`` on the same axis.
    """
    lambdas = list(lambdas)

    def _rows():
        n = 0
        for r in rows:
            for lam in lambdas:
                c2 = "" if r.get("center2") is None else f"{float(r['center2']):.6f}"
                yield [n, "contacts", f"{float(r['center1']):.6f}", f"{float(r['k1']):.4f}",
                       c2, f"{float(r['k2']):.4f}", f"{float(lam):.6f}"]
                n += 1

    return write_csv_atomic(
        Path(path),
        ["window", "primary_cv_mode", "primary_cv_center", "primary_cv_k_kcal",
         "secondary_cv_center", "secondary_cv_k_kcal_mol", "gamd_lambda"],
        _rows(),
    )
