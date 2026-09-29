"""Per-state statistical inefficiency and effective sample counts (spec X5, shared with 3.1).

Spec: docs/superpowers/specs/2026-09-29-adaptive-cv2-resolution-design.md, Section 12 X5.

What is estimated
-----------------
For each state, g = the integrated autocorrelation time (statistical inefficiency,
in units of the sample spacing) of the state's own CV series, and N_eff = N / g.

* **Series = state-indexed, not replica-indexed.** All rows logged for one state,
  ordered by step, within one sample segment (``segment_id``) of one sample
  directory. Under replica exchange a replica stays at a state for about two
  samples (chignolin_9: exchange every 400 steps, samples every 250), so
  replica-indexed runs are far too short to estimate anything. The variance of a
  state average depends only on the autocovariance of the series of that state's
  samples, whichever replica produced each one, so the state-indexed series is
  the right one for N_eff.
* **Never stitched.** A series is broken at every sample directory (phase,
  baseline/top-up segment), every ``segment_id`` (resume), every step gap and
  every non-finite value. Each contiguous run is centred on its own mean, and the
  runs' autocovariances are pooled (sum over runs, biased 1/N normalisation), so
  offsets between phases (nonstationarity) do not inflate g. The union builder's
  own g (``equilibrated_subsample`` on one stitched CV1 trace per state) does
  not have this property.
* **Observable = CV1 and CV2, the slowest restrained axis wins.** The pairwise
  MBAR energy differences a state's samples enter are linear in the CVs; the
  state's own reduced bias is quadratic in the displacement, and for a Gaussian
  process its autocorrelation is rho**2, which roughly halves the integrated
  time. An axis the state does not restrain (k <= 0) is ignored.
* **Estimator = Geyer's initial monotone sequence** over the pooled normalised
  autocovariance (FFT). Robust without a lag cutoff; exact on AR(1) up to noise.
  Runs shorter than ``MIN_SEGMENT_SAMPLES`` are skipped for the estimate (their
  samples still count in N).

Failure never raises: a state without a usable series gets g = 1 and a
``status``/``reason``. The allocator applies no correction to such a state (it
keeps the union builder's g), so a failed estimate can never make a state look
better sampled than the builder thinks it is.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MIN_SEGMENT_SAMPLES = 50
REPORT_NAME = "effective_samples.json"
_REL_VAR_FLOOR = 1e-12


@dataclass(frozen=True)
class InefficiencyEstimate:
    g: float
    n_samples: int
    n_segments: int
    n_segments_used: int
    status: str            # ok | no_samples | too_short | zero_variance | failed
    reason: str = ""


@dataclass(frozen=True)
class StateEffectiveSamples:
    state_id: int
    n_samples: int
    g: float
    g_cv1: Optional[float]
    g_cv2: Optional[float]
    n_eff: float
    n_segments: int
    status: str
    reason: str = ""


def _fallback(status: str, reason: str, n: int, n_seg: int, used: int = 0) -> InefficiencyEstimate:
    return InefficiencyEstimate(g=1.0, n_samples=int(n), n_segments=int(n_seg), n_segments_used=int(used),
                                status=status, reason=reason)


def _autocov_sum(d: np.ndarray) -> np.ndarray:
    """Unnormalised autocovariance sums sum_i d_i d_{i+t}, t = 0..n-1 (FFT, zero-padded)."""
    n = d.size
    size = 1 << int(2 * n - 1).bit_length()
    f = np.fft.rfft(d, size)
    return np.fft.irfft(f * np.conj(f), size)[:n]


def _initial_monotone_tau(rho: np.ndarray) -> float:
    """Geyer (1992) initial monotone sequence estimate of 1 + 2 sum_{t>=1} rho_t."""
    n_pairs = rho.size // 2
    if n_pairs == 0:
        return 1.0
    gam = rho[: 2 * n_pairs].reshape(n_pairs, 2).sum(axis=1)
    positive = gam > 0.0
    stop = int(np.argmin(positive)) if not positive.all() else n_pairs
    gam = np.minimum.accumulate(gam[:stop]) if stop > 0 else gam[:0]
    return float(-1.0 + 2.0 * gam.sum())


def pooled_inefficiency(segments: Iterable[Sequence[float]], *,
                        min_segment: int = MIN_SEGMENT_SAMPLES) -> InefficiencyEstimate:
    """Statistical inefficiency of a set of independent contiguous runs of one observable."""
    try:
        segs = [np.asarray(s, dtype=float).ravel() for s in segments]
        n_total = int(sum(s.size for s in segs))
        if n_total == 0:
            return _fallback("no_samples", "no samples", 0, len(segs))
        used = [s for s in segs if s.size >= int(min_segment) and np.isfinite(s).all()]
        if not used:
            return _fallback("too_short", f"no contiguous run of >= {int(min_segment)} samples "
                             f"(longest {max(s.size for s in segs)})", n_total, len(segs))
        acov = np.zeros(max(s.size for s in used))
        denom = 0
        scale = 0.0
        for s in used:
            d = s - s.mean()
            acov[: s.size] += _autocov_sum(d)
            denom += s.size
            scale = max(scale, float(np.abs(s).max()))
        acov /= denom
        if not (acov[0] > _REL_VAR_FLOOR * max(scale * scale, 1e-300)):
            return _fallback("zero_variance", "series has no variance", n_total, len(segs), len(used))
        g = _initial_monotone_tau(acov / acov[0])
        if not math.isfinite(g):
            return _fallback("failed", "non-finite estimate", n_total, len(segs), len(used))
        return InefficiencyEstimate(g=max(1.0, g), n_samples=n_total, n_segments=len(segs),
                                    n_segments_used=len(used), status="ok")
    except Exception as exc:                  # never raise into an epoch
        return _fallback("failed", f"{type(exc).__name__}: {exc}", 0, 0)


def contiguous_runs(steps: np.ndarray, values: np.ndarray, stride: Optional[int] = None) -> List[np.ndarray]:
    """Split a step-sorted series into runs with a constant step stride and finite values.

    ``stride`` defaults to the most common positive step difference.
    """
    steps = np.asarray(steps, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if steps.size == 0:
        return []
    if stride is None:
        diffs = np.diff(steps)
        diffs = diffs[diffs > 0]
        if diffs.size:
            vals, cnt = np.unique(diffs, return_counts=True)
            stride = int(vals[int(np.argmax(cnt))])
        else:
            stride = 1
    finite = np.isfinite(values)
    brk = np.ones(steps.size, dtype=bool)
    brk[1:] = (np.diff(steps) != stride) | ~finite[1:] | ~finite[:-1]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], steps.size]
    return [values[a:b] for a, b in zip(starts, ends) if finite[a]]


def _default_load(sample_dir: Path) -> dict:
    from gareus.query import load_samples
    return load_samples(Path(sample_dir))


def _combine(sid: int, n: int, n_seg: int, per_axis: Dict[str, Optional[InefficiencyEstimate]]) -> StateEffectiveSamples:
    ok = {a: e for a, e in per_axis.items() if e is not None and e.status == "ok"}
    g1 = ok["cv1"].g if "cv1" in ok else None
    g2 = ok["cv2"].g if "cv2" in ok else None
    if not ok:
        bad = [f"{a}: {e.status} ({e.reason})" for a, e in per_axis.items() if e is not None]
        return StateEffectiveSamples(sid, n, 1.0, None, None, float(n), n_seg,
                                     "no_samples" if n == 0 else "failed",
                                     "; ".join(bad) or "no restrained axis with data")
    g = max(e.g for e in ok.values())
    notes = [f"{a}: {e.status} ({e.reason})" for a, e in per_axis.items() if e is not None and e.status != "ok"]
    return StateEffectiveSamples(sid, n, float(g), g1, g2, float(n) / float(g), n_seg, "ok", "; ".join(notes))


def state_effective_samples(sources: Sequence[Tuple[str, Path]], *,
                            window_map_for: Callable[[Path], Mapping[int, int]],
                            restrained_axes: Mapping[int, Tuple[bool, bool]],
                            load: Optional[Callable[[Path], dict]] = None,
                            min_segment: int = MIN_SEGMENT_SAMPLES) -> Dict[int, StateEffectiveSamples]:
    """g and N_eff for every state in ``restrained_axes`` that has samples in ``sources``.

    ``sources``: ``(label, sample_dir)`` as ``_epoch_sample_sources`` returns them.
    ``window_map_for(sample_dir)``: local window -> state_id (the phase's own map).
    ``restrained_axes[state_id] = (cv1 restrained, cv2 restrained)``; with neither, both
    axes with data are used. A source that cannot be read is skipped with a warning.
    """
    load = load or _default_load
    runs: Dict[int, Dict[str, List[np.ndarray]]] = {}
    counts: Dict[int, int] = {}
    n_seg: Dict[int, int] = {}
    for label, sample_dir in sources:
        try:
            data = load(Path(sample_dir)) or {}
            n = int(len(data.get("step", [])))
            if n == 0:
                continue
            steps = np.asarray(data["step"], dtype=np.int64)
            win = np.asarray(data["window_id"], dtype=np.int64)
            seg_raw = data.get("segment_id")
            seg = np.asarray(seg_raw if seg_raw is not None else np.zeros(n), dtype=object).astype(str)
            _seg_codes, seg_idx = np.unique(seg, return_inverse=True)
            cvs = {}
            for axis, key in (("cv1", "cv1"), ("cv2", "cv2")):
                col = data.get(key)
                if col is None:
                    continue
                arr = np.ma.asarray(col).astype(float).filled(np.nan)
                cvs[axis] = np.asarray(arr, dtype=float)
            wmap = dict(window_map_for(Path(sample_dir)) or {})
            state = np.asarray([int(wmap.get(int(w), int(w))) for w in win], dtype=np.int64)
            # drop exact duplicates of (segment, step, window): leftover chunks re-log rows
            order = np.lexsort((win, steps, seg_idx))
            key = np.stack([seg_idx[order], steps[order], win[order]], axis=1)
            keep = np.ones(order.size, dtype=bool)
            keep[1:] = np.any(key[1:] != key[:-1], axis=1)
            order = order[keep]
            # group by (segment, state), step-sorted within the group
            order = order[np.lexsort((steps[order], state[order], seg_idx[order]))]
            s_o, g_o = state[order], seg_idx[order]
            cut = np.flatnonzero(np.r_[True, (s_o[1:] != s_o[:-1]) | (g_o[1:] != g_o[:-1]), True])
            for a, b in zip(cut[:-1], cut[1:]):
                sid = int(s_o[a])
                if sid not in restrained_axes:
                    continue
                idx = order[a:b]
                counts[sid] = counts.get(sid, 0) + int(idx.size)
                n_seg[sid] = n_seg.get(sid, 0) + 1
                r1, r2 = restrained_axes[sid]
                use = [ax for ax, r in (("cv1", r1), ("cv2", r2)) if r] or ["cv1", "cv2"]
                for ax in use:
                    if ax in cvs:
                        runs.setdefault(sid, {}).setdefault(ax, []).extend(
                            contiguous_runs(steps[idx], cvs[ax][idx]))
        except Exception as exc:
            print(f"      effective samples: skipping {label} ({sample_dir}): {type(exc).__name__}: {exc}")
    out: Dict[int, StateEffectiveSamples] = {}
    for sid, n in counts.items():
        r1, r2 = restrained_axes[sid]
        use = [ax for ax, r in (("cv1", r1), ("cv2", r2)) if r] or ["cv1", "cv2"]
        per_axis: Dict[str, Optional[InefficiencyEstimate]] = {}
        for ax in use:
            segs = runs.get(sid, {}).get(ax)
            per_axis[ax] = pooled_inefficiency(segs, min_segment=min_segment) if segs is not None else None
        out[sid] = _combine(sid, n, n_seg[sid], per_axis)
    return out


def effective_g_for_states(state_ids: Iterable[int],
                           estimates: Mapping[int, StateEffectiveSamples]) -> Dict[int, StateEffectiveSamples]:
    """``estimates`` restricted to ``state_ids``, a g = 1 fallback for every state without one."""
    out = {}
    for s in state_ids:
        s = int(s)
        out[s] = estimates.get(s) or StateEffectiveSamples(s, 0, 1.0, None, None, 0.0, 0, "no_samples",
                                                           "no samples found for this state")
    return out


def _restrained_axes(registry, state_ids) -> Dict[int, Tuple[bool, bool]]:
    out = {}
    for s in state_ids:
        st = registry.get_state(int(s))
        if st is None:
            continue
        k1 = float(st.primary_k or 0.0)
        k2 = float(st.secondary_k or 0.0) if st.secondary_center is not None else 0.0
        out[int(s)] = (k1 > 0.0, k2 > 0.0)
    return out


def campaign_effective_samples(adaptive_dir: Path, registry, state_ids: Sequence[int], *,
                               pilot_dirs=None, tica_cv_version=None) -> Dict[int, StateEffectiveSamples]:
    """g and N_eff for ``state_ids`` over the same sample sources the per-epoch union build reads."""
    from gareus.adaptive_production import _epoch_sample_sources, _load_epoch_window_map
    sources = _epoch_sample_sources(Path(adaptive_dir), include_epochs=True, pilot_dirs=pilot_dirs,
                                    tica_cv_version=tica_cv_version)
    est = state_effective_samples(sources, window_map_for=lambda d: _load_epoch_window_map(d, registry),
                                  restrained_axes=_restrained_axes(registry, state_ids))
    return effective_g_for_states(state_ids, est)


def allocation_shift(plan_a, plan_b) -> float:
    """Fraction of top-up state-steps that moved between two plans: 0.5 * sum |share_a - share_b|.

    A top-up runs its states in lockstep, so a plan's per-state steps are ``steps`` for
    each of its ``state_ids``. One empty plan against a non-empty one moved everything (1.0).
    """
    def shares(p):
        ids = tuple(getattr(p, "state_ids", ()) or ())
        steps = int(getattr(p, "steps", 0) or 0)
        total = steps * len(ids)
        return {int(s): 1.0 / len(ids) for s in ids} if total > 0 else {}
    a, b = shares(plan_a), shares(plan_b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 0.5 * sum(abs(a.get(s, 0.0) - b.get(s, 0.0)) for s in set(a) | set(b))


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _plan_summary(p) -> dict:
    return {"reason": p.reason, "steps": int(p.steps), "state_ids": [int(s) for s in p.state_ids],
            "deficit_state_ids": [int(s) for s in p.deficit_state_ids],
            "partner_state_ids": [int(s) for s in p.partner_state_ids]}


def write_report(epoch_dir: Path, payload: dict) -> Path:
    path = Path(epoch_dir) / REPORT_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)
    return path


def ess_topup_plan(plan_fn: Callable[[Optional[Dict[int, float]]], object], raw_plan, diag, *,
                   epoch_dir: Path, adaptive_dir: Path, registry, state_ids: Sequence[int],
                   pilot_dirs=None, tica_cv_version=None):
    """The top-up plan under ``allocation_weight = "ess"``, with its audit written.

    ``plan_fn(effective_g)`` re-runs ``plan_topup`` on the SAME diagnostics; ``raw_plan``
    is ``plan_fn(None)``. Any failure returns ``raw_plan`` and records why.
    """
    from .topup_allocator import decision_sigma
    base = {"schema_version": "effective_samples_v1", "allocation_weight": "ess",
            "estimator": "state-indexed CV1/CV2 series, per contiguous run, pooled autocovariance, "
                         "Geyer initial monotone sequence; g in units of the sample spacing",
            "min_segment_samples": MIN_SEGMENT_SAMPLES}
    if diag is None:
        write_report(epoch_dir, {**base, "status": "no_diagnostics", "reason": "no union diagnostics"})
        return raw_plan
    try:
        est = campaign_effective_samples(adaptive_dir, registry, state_ids, pilot_dirs=pilot_dirs,
                                         tica_cv_version=tica_cv_version)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        print(f"      effective samples unavailable ({reason}); top-up planned on raw counts")
        write_report(epoch_dir, {**base, "status": "failed", "reason": reason,
                                 "raw_plan": _plan_summary(raw_plan), "plan": _plan_summary(raw_plan)})
        return raw_plan
    eff = {s: e.g for s, e in est.items() if e.status == "ok"}
    plan = plan_fn(eff)
    rows = {}
    for s, e in est.items():
        g_b = max(1.0, float(diag.inefficiency.get(s, 1.0)))
        sig = diag.sigma_kcal.get(s, math.nan)
        rows[str(s)] = {**{k: (_num(v) if isinstance(v, float) else v) for k, v in asdict(e).items()},
                        "g_builder": _num(g_b), "n_kept_builder": int(diag.n_k.get(s, 0)),
                        "sigma_raw_kcal": _num(sig),
                        "sigma_decision_kcal": _num(decision_sigma(sig, g_b, eff.get(s)))}
    shift = allocation_shift(raw_plan, plan)
    n_ok = len(eff)
    print(f"      effective samples: g estimated for {n_ok}/{len(est)} states "
          f"(median {np.median(list(eff.values())) if eff else float('nan'):.1f}); "
          f"top-up {raw_plan.reason}/{len(raw_plan.state_ids)} states (raw) -> "
          f"{plan.reason}/{len(plan.state_ids)} states (ess), {shift:.0%} of state-steps moved")
    write_report(epoch_dir, {**base, "status": "ok", "reason": "", "states": rows,
                             "raw_plan": _plan_summary(raw_plan), "plan": _plan_summary(plan),
                             "allocation_shift": float(shift)})
    return plan
