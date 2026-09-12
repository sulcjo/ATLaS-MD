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
from .ranking import MIN_ATTEMPTS_FOR_DEAD
from .sidecar import SidecarSnapshot

DEFAULT_TEMPERATURE_K = 300.0


def acceptance_by_pair(exchange_stats: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    """Per-neighbour-pair acceptance rate, keyed by ``(i, j)``.

    The real payload nests per-pair counters under ``"pairs"`` alongside
    run-level totals::

        {"attempts": 120, "accepted": 40, "mode": "neighbor",
         "pairs": {"0-1": {"attempts": 40, "accepted": 12}, ...},
         "jump_bins": {...}}

    Reading the top level instead returns nothing on real runs -- and nothing is
    indistinguishable from "no exchanges attempted", which would leave every
    window unranked, the exchange strip permanently empty, and the union-find
    connectivity check reporting every window as its own component, i.e. a false
    claim of total MBAR disconnection on a perfectly healthy run.
    ``gareus/production.py``'s own reader takes the same ``"pairs"`` sub-mapping.

    A flat mapping is still accepted, so a hand-built fixture or any future
    flat producer keeps working.

    Zero attempts yields NaN, not 0.0: "not tried yet" and "tried and always
    rejected" must not rank the same.
    """
    payload = exchange_stats or {}
    nested = payload.get("pairs")
    source = nested if isinstance(nested, Mapping) else payload
    out: dict[tuple[int, int], float] = {}
    for key, stats in source.items():
        if not isinstance(stats, Mapping):
            continue
        try:
            a_str, b_str = str(key).split("-")[:2]
            a, b = int(a_str), int(b_str)
            attempts = float(stats.get("attempts", 0) or 0)
            accepted = float(stats.get("accepted", 0) or 0)
        except Exception:
            continue
        out[(a, b)] = (accepted / attempts) if attempts > 0 else float("nan")
    return out


def attempts_by_pair(exchange_stats: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    """Per-pair exchange attempt counts, keyed like ``acceptance_by_pair``.

    Kept separate from the rates so the displayed acceptance stays exactly what
    was measured; the counts only gate whether a rate is allowed to *condemn* a
    window (see ``acceptance_by_window``).
    """
    payload = exchange_stats or {}
    nested = payload.get("pairs")
    source = nested if isinstance(nested, Mapping) else payload
    out: dict[tuple[int, int], float] = {}
    for key, stats in source.items():
        if not isinstance(stats, Mapping):
            continue
        try:
            a_str, b_str = str(key).split("-")[:2]
            out[(int(a_str), int(b_str))] = float(stats.get("attempts", 0) or 0)
        except Exception:
            continue
    return out


def acceptance_by_window(
    pairs: Mapping[tuple[int, int], float],
    n_windows: int,
    attempts: Optional[Mapping[Any, Any]] = None,
    min_attempts: int = MIN_ATTEMPTS_FOR_DEAD,
) -> dict[int, float]:
    """Each window inherits the worst *sufficiently measured* acceptance.

    ``attempts`` takes the raw ``exchange_stats`` payload (or an
    already-extracted ``{(i, j): attempts}`` mapping). Pairs with fewer than
    ``min_attempts`` attempts are skipped rather than ranked: a pair tried once
    and rejected reads 0.0, which is finite and below ``DEAD_ACCEPTANCE``, so
    without this it condemns both its windows on a single coin flip. Skipping
    only removes a pair from the *worst-of* comparison -- a window whose other
    pairs are well measured still inherits those, and a window left with no
    measured pair at all simply goes unranked (NaN), which the ranker already
    treats as "unmeasured" rather than "dead".

    Passing no counts keeps the original behaviour, so existing callers and
    hand-built fixtures are unaffected.
    """
    counts: Mapping[tuple[int, int], float] = {}
    if attempts is not None:
        # Accept either the raw exchange_stats payload or a prepared mapping.
        keys = list(attempts.keys())
        if keys and isinstance(keys[0], tuple):
            counts = attempts                    # type: ignore[assignment]
        else:
            counts = attempts_by_pair(attempts)

    out: dict[int, float] = {}
    for (a, b), rate in pairs.items():
        if not math.isfinite(rate):
            continue
        if counts and float(counts.get((a, b), 0.0)) < float(min_attempts):
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
    roundtrip_state: Mapping[int, Mapping[str, Any]]
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
    # `final` is the terminal scheduled stage: it has no position in the epoch
    # sequence, so it carries this flag instead of an epoch_index. Defaulted so
    # every existing construction site keeps working unchanged.
    is_final_stage: bool = False


def _tuple_map(source: Mapping[int, Any]) -> dict[int, tuple[float, ...]]:
    return {int(k): tuple(float(x) for x in v) for k, v in (source or {}).items()}


def _copy_exchange_stats(source: Any) -> dict[str, Any]:
    """Copy the exchange payload one level deeper than a plain ``dict()``.

    ``gareus/production.py`` keeps ONE long-lived ``exchange_stats`` dict and
    mutates it in place for the whole run -- ``setdefault(pair, {...})`` then
    incrementing ``attempts``/``accepted`` (production.py:4914-4922). A shallow
    copy leaves those per-pair counters aliased to live state, so the render
    thread can read ``attempts`` after an increment but ``accepted`` before its
    own -- a torn read that reports an acceptance rate above 1.0.
    """
    payload = dict(source or {})
    for key in ("pairs", "jump_bins", "pair_metadata"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            payload[key] = {
                k: (dict(v) if isinstance(v, Mapping) else v) for k, v in nested.items()
            }
    return payload


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
        # An absence must not read as the best case. `_dashboard_decision_state`
        # always returns `{"health", "issues", "reasons", "actions"}` with
        # `health` one of "OK"/"WATCH"/"BAD" -- falling back to a bare `{}` here
        # made `_verdict`'s (gareus/dashboard/spine.py) own `.get("health", "OK")`
        # default silently render a swallowed failure as "✓ OK" in green. The
        # explicit "UNAVAILABLE" sentinel here is never returned by a real,
        # successful call, so `_verdict` can render it as its own distinct,
        # muted "verdict unavailable" state instead.
        decision = {"health": "UNAVAILABLE", "issues": [], "reasons": [], "actions": []}

    # `logger.out_dir` is the *segment* directory DistanceLogger was actually
    # constructed with -- on a real adaptive run that is e.g.
    # ".../adaptive_production/epoch_001/baseline", whose basename
    # ("baseline") is not the run's identity and duplicates `segment_name` two
    # fields later. `sidecar.run_root` (populated by `SidecarCache.snapshot`
    # from `find_run_root`) already resolves the true run directory; fall back
    # to `out_dir` only when no sidecar/run_root is available (most unit-test
    # fixtures, and any caller that built `SidecarSnapshot` by hand).
    run_root_name = str(getattr(sidecar.run_root, "name", "") or "")
    return DashboardContext(
        run_label=run_root_name or str(getattr(logger.out_dir, "name", "") or ""),
        phase=str(phase),
        segment_name=str(adaptive.get("segment_name", "") or ""),
        epoch_index=adaptive.get("epoch_index"),
        epoch_total=adaptive.get("epoch_total"),
        is_final_stage=bool(adaptive.get("is_final_stage", False)),
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
        exchange_stats=_copy_exchange_stats(info.get("exchange_stats")),
        # One level deep, for the same reason as exchange_stats: `_update_history`
        # mutates these dicts in place all run (gareus/logger.py:525-537).
        roundtrip_state={
            int(k): dict(v) for k, v in
            (getattr(logger, "roundtrip_state", {}) or {}).items()
            if isinstance(v, Mapping)
        },
        acceptance_pairs=pairs,
        acceptance_windows=acceptance_by_window(
            pairs, n_windows, attempts=info.get("exchange_stats")),
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
    "acceptance_by_window", "attempts_by_pair", "build_context", "delta_by_window",
    "overlap_by_pair",
]
