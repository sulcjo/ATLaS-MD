"""Immutable snapshot every view is a pure function of.

Views run on the render thread. Handing them the live ``DistanceLogger`` would
let them observe deques mutating mid-frame, and would make every view test
require a live logger. Copying into tuples here costs one pass over histories
that are already capped by ``distance_history_limit``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..math_helpers import _hist_overlap
from .sidecar import SidecarSnapshot

DEFAULT_TEMPERATURE_K = 300.0


def acceptance_by_pair(exchange_stats: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    """``{"0-1": {"attempts": n, "accepted": m}}`` -> ``{(0, 1): m / n}``.

    Zero attempts yields NaN, not 0.0: "not tried yet" and "tried and always
    rejected" must not rank the same.
    """
    out: dict[tuple[int, int], float] = {}
    for key, stats in (exchange_stats or {}).items():
        try:
            a_str, b_str = str(key).split("-")[:2]
            a, b = int(a_str), int(b_str)
            attempts = float((stats or {}).get("attempts", 0) or 0)
            accepted = float((stats or {}).get("accepted", 0) or 0)
        except Exception:
            continue
        out[(a, b)] = (accepted / attempts) if attempts > 0 else float("nan")
    return out


def acceptance_by_window(
    pairs: Mapping[tuple[int, int], float], n_windows: int
) -> dict[int, float]:
    """Each window inherits the worst finite acceptance among its own pairs."""
    out: dict[int, float] = {}
    for (a, b), rate in pairs.items():
        if not math.isfinite(rate):
            continue
        for w in (a, b):
            if 0 <= w < int(n_windows):
                out[w] = min(out.get(w, math.inf), float(rate))
    return out


def overlap_by_pair(
    history_by_window: Mapping[int, Sequence[float]], centers_a: Sequence[float]
) -> dict[tuple[int, int], float]:
    """Neighbour histogram overlap on one shared axis spanning all centres."""
    centers = [float(c) for c in centers_a]
    if len(centers) < 2:
        return {}
    lo, hi = min(centers), max(centers)
    if hi <= lo:
        hi = lo + 1.0
    out: dict[tuple[int, int], float] = {}
    for a in range(len(centers) - 1):
        ov = _hist_overlap(
            list(history_by_window.get(a, ())), list(history_by_window.get(a + 1, ())), lo, hi
        )
        if math.isfinite(ov):
            out[(a, a + 1)] = float(ov)
    return out


def delta_by_window(rows: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """Signed CV offset from each window's restraint centre, latest sample."""
    out: dict[int, float] = {}
    for row in rows:
        try:
            out[int(row["window"])] = float(row["cv_A"]) - float(row["center_A"])
        except Exception:
            continue
    return out


@dataclass(frozen=True)
class DashboardContext:
    # identity / phase
    run_label: str
    phase: str
    segment_name: str
    epoch_index: Optional[int]
    epoch_total: Any
    # progress
    step: int
    total_steps: Optional[int]
    display_step: int
    display_total_steps: Optional[int]
    elapsed_s: float
    eta_s: Optional[float]
    timestep_fs: float
    n_replicas: int
    now_wall: float
    # geometry
    term_w: int
    term_h: int
    # window layout
    n_windows: int
    centers_a: tuple[float, ...]
    k_list: tuple[float, ...]
    secondary_centers: tuple[float, ...]
    secondary_k: tuple[float, ...]
    is_2d: bool
    topology_label: str
    secondary_cv_type: str
    temperature_k: float
    # live data (all copies)
    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    cv_history_by_window: Mapping[int, tuple[float, ...]]
    cv_history_by_replica: Mapping[int, tuple[float, ...]]
    secondary_history_by_window: Mapping[int, tuple[float, ...]]
    pe_history_by_replica: Mapping[int, tuple[float, ...]]
    boost_history_all: tuple[float, ...]
    window_trace_by_replica: Mapping[int, tuple[int, ...]]
    exchange_stats: Mapping[str, Any]
    acceptance_pairs: Mapping[tuple[int, int], float]
    acceptance_windows: Mapping[int, float]
    overlap_pairs: Mapping[tuple[int, int], float]
    deltas: Mapping[int, float]
    decision: Mapping[str, Any]
    # labels + config
    primary_cv_label: str
    primary_cv_units: str
    primary_k_units: str
    ascii_mode: str = "hist3d"
    ascii_max_replicas: int = 32
    sidecar: SidecarSnapshot = field(default_factory=SidecarSnapshot)
    view: str = "progress"
    glyphs: str = "unicode"


def _tuple_map(source: Mapping[int, Any]) -> dict[int, tuple[float, ...]]:
    return {int(k): tuple(float(x) for x in v) for k, v in (source or {}).items()}


def _resolve_temperature(args: Any, sidecar: SidecarSnapshot) -> float:
    for value in (getattr(args, "temperature_k", None),
                  (sidecar.gamd or {}).get("temperature_K")):
        try:
            t = float(value)
            if math.isfinite(t) and t > 0.0:
                return t
        except (TypeError, ValueError):
            continue
    return DEFAULT_TEMPERATURE_K


def build_context(
    *,
    logger: Any,
    rows: Sequence[Mapping[str, Any]],
    phase: str,
    step: int,
    total_steps: Optional[int],
    summary: Mapping[str, Any],
    dashboard_info: Optional[Mapping[str, Any]],
    sidecar: SidecarSnapshot,
    term_w: int,
    term_h: int,
    now: float,
    view: str,
    glyphs: str,
) -> DashboardContext:
    info = dict(dashboard_info or {})
    centers = tuple(float(x) for x in info.get("centers_a", ()))
    n_windows = int(info.get("n_windows", len(centers)) or len(centers))
    sec_centers = tuple(
        float(x) for x in info.get("secondary_cv_centers", ())
        if str(x) not in {"", "None", "nan"}
    )
    sec_meta = dict(info.get("secondary_cv", {}) or {})
    explicit_2d = bool(sec_meta.get("explicit_2d_windows", info.get("explicit_2d", False)))
    n_sec_targets = len(set(round(x, 4) for x in sec_centers))
    is_2d = bool(explicit_2d or n_sec_targets > 1)
    if explicit_2d and not bool(sec_meta.get("grid", False)):
        topology = "sparse explicit 2D"
    elif is_2d:
        topology = "rectangular 2D"
    else:
        topology = "1D/secondary-fixed"

    adaptive = dict(info.get("adaptive_phase", {}) or {})
    display_step = int(info.get("display_step", step) or 0)
    display_total = info.get("display_total_steps", total_steps)
    display_total = int(display_total) if display_total is not None else None
    frac = (float(display_step) / float(display_total)) if display_total else 0.0
    frac = max(0.0, min(1.0, frac))
    eta_start = float(info.get("eta_start_wall", getattr(logger, "start_wall", now)) or now)
    elapsed = max(0.0, float(now) - eta_start)
    eta = (elapsed * (1.0 - frac) / frac) if frac > 0.0 and display_total else None

    hist_windows = _tuple_map(getattr(logger, "history_by_window", {}))
    pairs = acceptance_by_pair(info.get("exchange_stats", {}) or {})
    try:
        decision = logger._dashboard_decision_state(
            list(rows), info.get("exchange_stats", {}) or {}, list(centers), dict(summary),
            str(phase), info,
        )
    except Exception:
        decision = {}

    return DashboardContext(
        run_label=str(getattr(logger.out_dir, "name", "") or ""),
        phase=str(phase),
        segment_name=str(adaptive.get("segment_name", "") or ""),
        epoch_index=adaptive.get("epoch_index"),
        epoch_total=adaptive.get("epoch_total"),
        step=int(step),
        total_steps=int(total_steps) if total_steps is not None else None,
        display_step=display_step,
        display_total_steps=display_total,
        elapsed_s=elapsed,
        eta_s=eta,
        timestep_fs=float(getattr(logger.args, "timestep_fs", 0.0) or 0.0),
        n_replicas=max(1, len(rows)),
        now_wall=float(now),
        term_w=int(term_w),
        term_h=int(term_h),
        n_windows=n_windows,
        centers_a=centers,
        k_list=tuple(float(x) for x in info.get("k_list", ())) or tuple(
            float(r.get("k_kcal_mol_A2", float("nan"))) for r in rows
        ),
        secondary_centers=sec_centers,
        secondary_k=tuple(float(x) for x in info.get("secondary_cv_k_kcal_mol", ())),
        is_2d=is_2d,
        topology_label=topology,
        secondary_cv_type=str(sec_meta.get("type", "") or ""),
        temperature_k=_resolve_temperature(logger.args, sidecar),
        rows=tuple(dict(r) for r in rows),
        summary=dict(summary),
        cv_history_by_window=hist_windows,
        cv_history_by_replica=_tuple_map(getattr(logger, "history_by_replica", {})),
        secondary_history_by_window=_tuple_map(
            getattr(logger, "secondary_history_by_window", {})),
        pe_history_by_replica=_tuple_map(getattr(logger, "potential_history_by_replica", {})),
        boost_history_all=tuple(float(x) for x in getattr(logger, "boost_history_all", ())),
        window_trace_by_replica={
            int(k): tuple(int(x) for x in v)
            for k, v in (getattr(logger, "window_trace_by_replica", {}) or {}).items()
        },
        exchange_stats=dict(info.get("exchange_stats", {}) or {}),
        acceptance_pairs=pairs,
        acceptance_windows=acceptance_by_window(pairs, n_windows),
        overlap_pairs=overlap_by_pair(hist_windows, centers),
        deltas=delta_by_window(rows),
        decision=dict(decision or {}),
        primary_cv_label=str(info.get("primary_cv_label", "primary CV")),
        primary_cv_units=str(info.get("primary_cv_units", "")),
        primary_k_units=str(info.get("primary_k_units", "")),
        ascii_mode=str(getattr(logger, "ascii_mode", "hist3d") or "hist3d"),
        ascii_max_replicas=int(getattr(logger, "ascii_max_replicas", 32) or 32),
        sidecar=sidecar,
        view=str(view),
        glyphs=str(glyphs),
    )


__all__ = [
    "DEFAULT_TEMPERATURE_K", "DashboardContext", "acceptance_by_pair",
    "acceptance_by_window", "build_context", "delta_by_window", "overlap_by_pair",
]
