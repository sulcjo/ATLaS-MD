"""Adapter: harness samples -> the on-disk layout the REAL adaptive collectors read.

Spec 2026-09-29 adaptive-CV2, T2 (Section 5; review finding A5: "synth harness never
calls the real collectors"). Everything decision-relevant here is shipped code:

* ``gareus.adaptive_production.collect_epoch_diagnostics`` /
  ``collect_segmented_epoch_diagnostics`` (P4 ``paired_cv`` and, with
  ``policy.edge_metric == "pairwise-mbar"``, the spec 3.1 two-state MBAR grading),
* ``propose_actions_from_diagnostics`` and ``_apply_registry_actions``,
* ``_edge_is_measured_weak`` for every weak/ok verdict.

The adapter only writes files. Per phase directory (flat epoch, or one segment of a
scheduled epoch) it writes what a production phase leaves behind and the collectors read:

* ``samples/seg_001/*.parquet`` through ``gareus.store.ParquetSampleWriter`` (the
  canonical layout; ``fmt="csv"`` writes the legacy ``samples.csv`` instead). Rows are
  written in time order, ``step`` increasing, so the P4 stride subsample and the
  blocking estimate see each state's series in order;
* ``exchanges`` (Parquet or ``exchanges.csv``): synthetic Metropolis swap attempts on the
  bias energies of each geometry edge's two states (F cancels in a swap), so the marginal
  arm's exchange-acceptance test (< 0.08) is not artificially blind;
* ``epoch_window_map.csv`` (explicit; never the active-order fallback),
* ``umbrella_explicit_windows.csv`` (the phase's window table),
* ``run_manifest.json`` with ``resolved_args.temperature_k``.

Units: the harness keeps F in kBT and k in kBT per CV^2 (beta = 1). The registry and
the collectors use kcal/mol per CV^2 with beta = 1/(R T): ``k_kcal = k_reduced / beta``
at ``TEMPERATURE_K`` (300 K). Reduced energies are then identical on both sides.

Analytic truth (:func:`exact_pair_overlap`): for Gaussian-restrained windows on a known
surface, the pairwise-MBAR-scale overlap sqrt(N_a N_b) * int p_a p_b / (N_a p_a + N_b p_b)
and df = f_b - f_a by quadrature on an adaptive grid. For a 3D surface whose windows
restrain only (cv1, cv2) the (cv1, cv2) marginal is exact (p_a / p_b depends on the CVs
only).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .sampler import BIAS, Window

TEMPERATURE_K = 300.0
EXCHANGE_ATTEMPTS_PER_EDGE = 200
STEPS_PER_SAMPLE_MD = 250           # the "MD step" spacing written into the step column


def beta_real() -> float:
    from gareus.adaptive.edge_metric import beta_mol_per_kcal  # noqa: PLC0415
    return beta_mol_per_kcal(TEMPERATURE_K)


def k_to_kcal(k_reduced: Optional[float]) -> float:
    """Harness k (kBT/CV^2) -> kcal/mol/CV^2 at TEMPERATURE_K; None/0 -> 0 (axis free)."""
    if k_reduced is None or not float(k_reduced) > 0.0:
        return 0.0
    return float(k_reduced) / beta_real()


def k_to_reduced(k_kcal: Optional[float]) -> float:
    if k_kcal is None or not math.isfinite(float(k_kcal)) or not float(k_kcal) > 0.0:
        return 0.0
    return float(k_kcal) * beta_real()


# ---------------------------------------------------------------------------------------
# registry <-> harness windows
# ---------------------------------------------------------------------------------------

def window_of_state(state) -> Window:
    """Harness Window (reduced k) for a registry state; k2 = 0 / None -> CV2 unrestrained."""
    k2 = k_to_reduced(state.secondary_k) if state.secondary_center is not None else 0.0
    c2 = float(state.secondary_center) if (state.secondary_center is not None and k2 > 0.0) else None
    return Window(float(state.primary_center), k_to_reduced(state.primary_k), c2, k2 if c2 is not None else None,
                  lam=float(getattr(state, "gamd_lambda", 0.0) or 0.0))


def registry_from_windows(windows: Sequence[Window], *, epoch: int = 0, source: str = "synth"):
    """A real ``WindowStateRegistry`` with one state per harness window (ids 0..n-1)."""
    from gareus.adaptive_production import WindowStateRegistry  # noqa: PLC0415
    reg = WindowStateRegistry()
    for w in windows:
        has2 = w.center2 is not None and w.k2 is not None and float(w.k2) > 0.0
        reg.add_state(primary_center=float(w.center1), primary_k=k_to_kcal(w.k1),
                      secondary_center=float(w.center2) if w.center2 is not None else 0.0,
                      secondary_k=k_to_kcal(w.k2) if has2 else 0.0,
                      gamd_lambda=float(w.lam), epoch=epoch, source=source)
    return reg


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


# ---------------------------------------------------------------------------------------
# phase directory writer
# ---------------------------------------------------------------------------------------

@dataclass
class PhaseWrite:
    phase_dir: Path
    window_of_state: Dict[int, int]
    n_rows: int


def _exchange_rows(pairs, samples, windows, rng, attempts):
    rows = []
    for (si, sj) in pairs:
        a, b = samples.get(si), samples.get(sj)
        if a is None or b is None or len(a) == 0 or len(b) == 0:
            continue
        wi, wj = windows[si], windows[sj]
        ia = rng.integers(0, len(a), attempts)
        ib = rng.integers(0, len(b), attempts)
        xa, xb = a[ia], b[ib]
        delta = (BIAS(wi, xb[:, 0], xb[:, 1]) + BIAS(wj, xa[:, 0], xa[:, 1])
                 - BIAS(wi, xa[:, 0], xa[:, 1]) - BIAS(wj, xb[:, 0], xb[:, 1]))
        acc = rng.uniform(size=attempts) < np.minimum(1.0, np.exp(-np.clip(delta, -50, 700)))
        rows.extend((int(si), int(sj), float(d), bool(ok)) for d, ok in zip(delta, acc))
    return rows


def write_phase_dir(phase_dir: Path, samples: Mapping[int, np.ndarray], windows: Mapping[int, Window], *,
                    exchange_pairs: Sequence[Tuple[int, int]] = (), rng: Optional[np.random.Generator] = None,
                    fmt: str = "parquet", step_offset: int = 0,
                    exchange_attempts: int = EXCHANGE_ATTEMPTS_PER_EDGE) -> PhaseWrite:
    """Write one phase directory the collectors can read.

    ``samples[state_id]`` is that state's ``(n, >=2)`` time-ordered array (only cv1, cv2
    are written); ``windows[state_id]`` its harness Window. Local window indices follow
    sorted state ids and are recorded in ``epoch_window_map.csv``.
    """
    phase_dir = Path(phase_dir)
    phase_dir.mkdir(parents=True, exist_ok=True)
    rng = rng if rng is not None else np.random.default_rng(0)
    sids = sorted(int(s) for s in samples)
    local = {sid: i for i, sid in enumerate(sids)}
    _write_csv(phase_dir / "epoch_window_map.csv", ["epoch_window", "state_id"],
               ({"epoch_window": local[s], "state_id": s} for s in sids))
    _write_csv(phase_dir / "umbrella_explicit_windows.csv",
               ["window", "state_id", "primary_cv_center", "primary_cv_k_kcal", "secondary_cv_center",
                "secondary_cv_k_kcal_mol", "gamd_lambda"],
               ({"window": local[s], "state_id": s, "primary_cv_center": windows[s].center1,
                 "primary_cv_k_kcal": k_to_kcal(windows[s].k1),
                 "secondary_cv_center": "" if windows[s].center2 is None else windows[s].center2,
                 "secondary_cv_k_kcal_mol": k_to_kcal(windows[s].k2), "gamd_lambda": windows[s].lam}
                for s in sids))
    (phase_dir / "run_manifest.json").write_text(json.dumps(
        {"resolved_args": {"temperature_k": TEMPERATURE_K}, "method_settings": {"temperature_k": TEMPERATURE_K}}))

    # interleave states in time: row t of every state is written at step (t+1)*spacing
    n_max = max((len(samples[s]) for s in sids), default=0)
    ex = _exchange_rows(exchange_pairs, samples, windows, rng, int(exchange_attempts))
    n_rows = int(sum(len(samples[s]) for s in sids))
    if fmt == "csv":
        rows = []
        for t in range(n_max):
            for s in sids:
                arr = samples[s]
                if t < len(arr):
                    rows.append({"window": local[s], "step": step_offset + (t + 1) * STEPS_PER_SAMPLE_MD,
                                 "replica": local[s], "cv_A": float(arr[t, 0]), "secondary_cv": float(arr[t, 1]),
                                 "gamd_boost_total_kcal_mol": 0.0})
        _write_csv(phase_dir / "samples.csv",
                   ["window", "step", "replica", "cv_A", "secondary_cv", "gamd_boost_total_kcal_mol"], rows)
        _write_csv(phase_dir / "exchanges.csv", ["step", "window_i", "window_j", "delta_e", "accepted"],
                   ({"step": step_offset + (i + 1) * STEPS_PER_SAMPLE_MD, "window_i": local[a],
                     "window_j": local[b], "delta_e": d, "accepted": int(ok)}
                    for i, (a, b, d, ok) in enumerate(ex)))
    elif fmt == "parquet":
        from gareus.store import ParquetExchangeWriter, ParquetSampleWriter, SegmentRegistry  # noqa: PLC0415
        reg = SegmentRegistry(phase_dir)
        seg = reg.open_segment("synth", None, 1)
        sw = ParquetSampleWriter(phase_dir / "samples" / seg, flush_rows=200_000)
        for t in range(n_max):
            step = step_offset + (t + 1) * STEPS_PER_SAMPLE_MD
            for s in sids:
                arr = samples[s]
                if t < len(arr):
                    sw.write_sample(step, local[s], local[s], float(arr[t, 0]), float(arr[t, 1]),
                                    0.0, 0.0, 0.0, 0.0, gamd_lambda=float(windows[s].lam))
        sw.close()
        xw = ParquetExchangeWriter(phase_dir / "exchanges" / seg, flush_rows=200_000)
        for i, (a, b, d, ok) in enumerate(ex):
            xw.write_exchange(step=step_offset + (i + 1) * STEPS_PER_SAMPLE_MD, replica_i=local[a],
                              replica_j=local[b], window_i=local[a], window_j=local[b], delta_e=d, accepted=ok)
        xw.close()
        reg.close_segment(seg, end_step=step_offset + n_max * STEPS_PER_SAMPLE_MD)
    else:
        raise ValueError(f"fmt must be 'parquet' or 'csv', not {fmt!r}")
    return PhaseWrite(phase_dir, local, n_rows)


def geometry_pairs(registry, policy=None) -> List[Tuple[int, int]]:
    from gareus.adaptive_production import build_geometry_edges  # noqa: PLC0415
    return [(int(a), int(b)) for a, b, _t, _d in build_geometry_edges(registry, policy)]


def collect(epoch_dir: Path, registry, policy, *, segmented: bool = False) -> Dict[str, Any]:
    """Run the real collector on a written phase (flat) or epoch of segments."""
    from gareus.adaptive_production import (collect_epoch_diagnostics,  # noqa: PLC0415
                                            collect_segmented_epoch_diagnostics)
    if segmented:
        return collect_segmented_epoch_diagnostics(Path(epoch_dir), registry, policy)
    return collect_epoch_diagnostics(Path(epoch_dir), registry, policy)


def propose(registry, diagnostics, policy, *, secondary_k_max: Optional[float] = None):
    from gareus.adaptive_production import propose_actions_from_diagnostics  # noqa: PLC0415
    return propose_actions_from_diagnostics(registry, diagnostics, policy, temperature_K=TEMPERATURE_K,
                                            secondary_k_max=secondary_k_max)


def weak_edges(diagnostics, policy) -> List[Dict[str, Any]]:
    from gareus.adaptive_production import _edge_is_measured_weak  # noqa: PLC0415
    return [e for e in diagnostics.get("edges", []) if str(e.get("edge_type")) != "rung"
            and _edge_is_measured_weak(e, policy)]


def action_counts(actions) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for a in actions:
        if a:
            out[str(a[0])] = out.get(str(a[0]), 0) + 1
    return out


# ---------------------------------------------------------------------------------------
# analytic truth
# ---------------------------------------------------------------------------------------

def _sigma(k_reduced: Optional[float]) -> Optional[float]:
    if k_reduced is None or not float(k_reduced) > 0.0:
        return None
    return 1.0 / math.sqrt(float(k_reduced))


def _axis_grid(lo, hi, centres, sigmas, *, pad=8.0, per_sigma=10.0, min_n=241, max_n=1601):
    sig = [s for s in sigmas if s is not None]
    if sig and all(c is not None for c in centres):
        a = max(lo, min(centres) - pad * max(sig))
        b = min(hi, max(centres) + pad * max(sig))
        if not b > a:
            a, b = lo, hi
        n = int(min(max_n, max(min_n, math.ceil((b - a) / (min(sig) / per_sigma)) + 1)))
    else:
        a, b, n = lo, hi, min_n * 2
    return np.linspace(a, b, n)


def _log_marginal(surface, g1, g2, res3: int = 121):
    if hasattr(surface, "bounds") and len(surface.bounds) == 3:
        from scipy.special import logsumexp  # noqa: PLC0415
        z = np.linspace(*surface.bounds[2], res3)
        dz = z[1] - z[0]
        out = np.empty(g1.shape)
        for i in range(g1.shape[0]):          # row by row (memory)
            e = surface.energy(g1[i][:, None], g2[i][:, None], z[None, :])
            out[i] = logsumexp(-e, axis=1) + math.log(dz)
        return out
    return -surface.energy(g1, g2)


def exact_pair_overlap(surface, wa: Window, wb: Window, *, na: float = 1.0, nb: float = 1.0) -> Dict[str, float]:
    """Exact pairwise-MBAR-scale overlap and df = f_b - f_a (kT) of two windows on ``surface``.

    Quadrature on a grid local to both windows (8 sigma_w past each restrained centre,
    >= 10 points per the narrowest sigma_w; an axis neither restrains spans its bounds).
    """
    lo1, hi1 = surface.cv1_bounds
    lo2, hi2 = surface.cv2_bounds
    s1 = [_sigma(wa.k1), _sigma(wb.k1)]
    x = _axis_grid(lo1, hi1, [wa.center1, wb.center1], s1)
    r2a = wa.center2 is not None and (wa.k2 or 0) > 0
    r2b = wb.center2 is not None and (wb.k2 or 0) > 0
    if r2a and r2b:
        y = _axis_grid(lo2, hi2, [wa.center2, wb.center2], [_sigma(wa.k2), _sigma(wb.k2)])
    else:
        y = np.linspace(lo2, hi2, 801)
    if not (wa.k1 > 0 and wb.k1 > 0):
        x = np.linspace(lo1, hi1, 801)
    g1, g2 = np.meshgrid(x, y, indexing="ij")
    logm = _log_marginal(surface, g1, g2)
    la = logm - BIAS(wa, g1, g2)
    lb = logm - BIAS(wb, g1, g2)
    dx, dy = x[1] - x[0], y[1] - y[0]
    from scipy.special import logsumexp  # noqa: PLC0415
    lza = logsumexp(la) + math.log(dx * dy)
    lzb = logsumexp(lb) + math.log(dx * dy)
    pa = np.exp(la - lza)
    pb = np.exp(lb - lzb)
    den = na * pa + nb * pb
    integrand = np.where(den > 0, pa * pb / np.where(den > 0, den, 1.0), 0.0)
    ov = math.sqrt(na * nb) * float(np.sum(integrand) * dx * dy)
    return {"overlap": ov, "delta_f": float(-(lzb - lza)), "grid": [int(x.size), int(y.size)]}


def exact_window_moments(surface, w: Window) -> Dict[str, float]:
    """Exact mean / sd of cv1 and cv2 under one window (same quadrature as the overlap)."""
    lo1, hi1 = surface.cv1_bounds
    lo2, hi2 = surface.cv2_bounds
    x = _axis_grid(lo1, hi1, [w.center1], [_sigma(w.k1)]) if w.k1 > 0 else np.linspace(lo1, hi1, 801)
    y = (_axis_grid(lo2, hi2, [w.center2], [_sigma(w.k2)]) if (w.center2 is not None and (w.k2 or 0) > 0)
         else np.linspace(lo2, hi2, 801))
    g1, g2 = np.meshgrid(x, y, indexing="ij")
    lp = _log_marginal(surface, g1, g2) - BIAS(w, g1, g2)
    p = np.exp(lp - lp.max()); p /= p.sum()
    m1 = float(np.sum(p * g1)); m2 = float(np.sum(p * g2))
    return {"mean1": m1, "sd1": float(np.sqrt(np.sum(p * (g1 - m1) ** 2))),
            "mean2": m2, "sd2": float(np.sqrt(np.sum(p * (g2 - m2) ** 2)))}


# ---------------------------------------------------------------------------------------
# union MBAR (post-union verdicts, PMF error)
# ---------------------------------------------------------------------------------------

def union_mbar(samples: Mapping[int, np.ndarray], windows: Mapping[int, Window], *, max_iter: int = 5000,
               tol: float = 1e-8):
    """Self-consistent MBAR over every state: (sids, pooled (M, 2), u_nk (M, K), window (M,), f (K,), n_k)."""
    from scipy.special import logsumexp  # noqa: PLC0415
    sids = [s for s in sorted(samples) if len(samples[s])]
    parts = [np.asarray(samples[s], float)[:, :2] for s in sids]
    n_k = np.array([p.shape[0] for p in parts], float)
    pooled = np.concatenate(parts)
    window = np.concatenate([np.full(p.shape[0], i) for i, p in enumerate(parts)])
    u = np.column_stack([BIAS(windows[s], pooled[:, 0], pooled[:, 1]) for s in sids])
    log_n = np.log(n_k)
    f = np.zeros(len(sids))
    for _ in range(max_iter):
        log_den = logsumexp(log_n[None, :] + f[None, :] - u, axis=1)
        f_new = -logsumexp(-u - log_den[:, None], axis=0)
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new
    return sids, pooled, u, window, f, n_k


def union_pairwise_overlap(u, window, f, n_k, i: int, j: int) -> float:
    from gareus.mbar_analysis.ladder import pairwise_state_overlap  # noqa: PLC0415
    return float(pairwise_state_overlap(u, window, f, n_k, i, j))
