"""Full-screen in-job dashboard for the swarm round (spec 2026-10-04-swarm-dashboard-design).

The umbrella dashboard (``screen.render_screen``) is window-centric: windows, exchange, overlap,
boosts. A swarm round has none of them -- many short unbiased members -- so this module renders
its own spine and panels from a plain-data ``SwarmSnapshot`` (built by
``gareus.swarm.round_progress.SwarmRoundProgress.snapshot``), reusing the domain-free layout layer
(``gareus.tui_screen``) and therefore the same contract: a frame fits ``term_w - 2`` columns and
``term_h - 1`` lines. Pure and deterministic: the wall clock comes from the snapshot.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from gareus.tui import _ansi_truncate, _sparkline, format_duration, make_progress_bar
from gareus.tui_screen import FOOTER_LINES, Panel, Row, allocate_rows, compose_rows, frame_tiers

PHASES = ("queued", "graft_wait", "grafting", "equilibrating", "production", "done", "failed")
ACTIVE_PHASES = ("graft_wait", "grafting", "equilibrating", "production")
GLYPHS_UNICODE = {"queued": "·", "graft_wait": "g", "grafting": "G", "equilibrating": "e", "done": "✓",
                  "failed": "✗", "production": "▁▂▃▅▇"}
GLYPHS_ASCII = {"queued": ".", "graft_wait": "g", "grafting": "G", "equilibrating": "e", "done": "+",
                "failed": "x", "production": "12345"}
SPINE_FULL, SPINE_COMPACT = 4, 2
SLOW_DEVICE_FRACTION = 0.5


@dataclass(frozen=True)
class Histogram:
    """Live CV histogram over a fixed range; ``below``/``above`` count frames outside it."""
    name: str
    lo: float
    hi: float
    counts: Tuple[int, ...]
    below: int = 0
    above: int = 0


@dataclass(frozen=True)
class DeviceStat:
    device: str
    n_active: int
    ns_per_day: float
    production_fraction: float
    slow: bool = False


@dataclass(frozen=True)
class SwarmSnapshot:
    now: float
    run_label: str
    round_index: int
    n_members: int
    timestep_fs: float
    n_prod_steps: int
    bins: Tuple[int, ...]
    member_ids: Tuple[int, ...]
    phases: Tuple[str, ...]                 # per member, PHASES
    production_fraction: Tuple[float, ...]  # per member, 0..1 of its production steps
    elapsed_s: float
    mean_fraction: float                    # mean member production fraction (the round's %)
    aggregate_ns_per_day: float
    median_member_ns_per_day: float
    slowest_member_ns_per_day: float
    eta_mean_s: Optional[float]
    eta_last_s: Optional[float]
    stalled: Tuple[int, ...] = ()
    devices: Tuple[DeviceStat, ...] = ()
    histograms: Tuple[Histogram, ...] = ()
    cells_visited: int = 0
    cells_total: int = 0
    completions_per_bin: Tuple[int, ...] = ()   # completions per COMPLETION_BIN_S, oldest first
    completion_bin_s: float = 600.0
    events: Tuple[Dict[str, object], ...] = ()  # newest first

    def count(self, phase: str) -> int:
        return sum(1 for p in self.phases if p == phase)


# --- panels ---------------------------------------------------------------------------------

def _panel(key, title, lines, *, min_lines, want_lines, priority, weight=1.0, statuses=()) -> Panel:
    return Panel(key=key, title=title, lines=tuple(str(x) for x in lines), min_lines=int(min_lines),
                 want_lines=int(want_lines), priority=int(priority), weight=float(weight),
                 line_statuses=tuple(statuses))


def _fmt_rate(x: float) -> str:
    return f"{x:,.0f}" if math.isfinite(x) else "?"


def _member_glyph(phase: str, frac: float, glyphs: Dict[str, str]) -> str:
    if phase == "production":
        ramp = glyphs["production"]
        return ramp[min(len(ramp) - 1, max(0, int(frac * len(ramp))))]
    return glyphs.get(phase, "?")


def members_panel(s: SwarmSnapshot, width: int, glyphs: Dict[str, str]) -> Panel:
    w = max(10, width - 4)
    marks = "".join(_member_glyph(p, f, glyphs) for p, f in zip(s.phases, s.production_fraction))
    grid = [marks[i:i + w] for i in range(0, len(marks), w)] or ["(no members)"]
    legend = (f"{glyphs['queued']} queued  {glyphs['graft_wait']}/{glyphs['grafting']} graft  "
              f"{glyphs['equilibrating']} equil  {glyphs['production']} production  "
              f"{glyphs['done']} done  {glyphs['failed']} failed")
    lines = [*grid, legend]
    return _panel("members", f"members ({s.n_members})", lines, min_lines=min(len(lines), 2),
                  want_lines=len(lines), priority=1, weight=1.4)


def projection_panel(s: SwarmSnapshot) -> Panel:
    lines = [f"ETA mean   {format_duration(s.eta_mean_s)}   (mean member)",
             f"ETA last   {format_duration(s.eta_last_s)}   (slowest member: the round ends with it)",
             f"rate       median {_fmt_rate(s.median_member_ns_per_day)} / slowest "
             f"{_fmt_rate(s.slowest_member_ns_per_day)} ns/day per member",
             f"done/10min {_sparkline([float(x) for x in s.completions_per_bin], 24) if s.completions_per_bin else '-'}"]
    if s.stalled:
        lines.append(f"stalled    {len(s.stalled)}: " + " ".join(str(m) for m in s.stalled[:12]))
    return _panel("projection", "projection", lines, min_lines=2, want_lines=len(lines), priority=2)


def devices_panel(s: SwarmSnapshot) -> Panel:
    lines, statuses = [], []
    for d in s.devices:
        lines.append(f"gpu {d.device:>3}  {d.n_active:3d} active  {_fmt_rate(d.ns_per_day):>8} ns/day  "
                     f"{100.0 * d.production_fraction:5.1f}%" + ("  SLOW" if d.slow else ""))
        statuses.append("WARN" if d.slow else "")
    if not lines:
        lines = ["(no device data yet)"]
    return _panel("devices", "devices", lines, min_lines=1, want_lines=len(lines), priority=3, statuses=statuses)


def _histogram_line(h: Histogram, width: int) -> str:
    ramp = " ▁▂▃▄▅▆▇█"
    n = max(1, len(h.counts))
    w = max(8, min(width, n))
    # rebin to w columns
    cols = [0] * w
    for i, c in enumerate(h.counts):
        cols[min(w - 1, i * w // n)] += int(c)
    top = max(cols) or 1
    bar = "".join(ramp[min(len(ramp) - 1, int(round(c / top * (len(ramp) - 1))))] for c in cols)
    out = f"<{h.below}" if h.below else ""
    over = f">{h.above}" if h.above else ""
    return f"{h.name:>4} {h.lo:+.2f} |{bar}| {h.hi:+.2f} {out}{over}".rstrip()


def coverage_panel(s: SwarmSnapshot, width: int) -> Panel:
    w = max(8, width - 26)
    lines = [_histogram_line(h, w) for h in s.histograms] or ["(no frames yet)"]
    if s.cells_total:
        lines.append(f"strata cells visited {s.cells_visited}/{s.cells_total}")
    return _panel("coverage", "coverage (live frames)", lines, min_lines=1, want_lines=len(lines), priority=4)


def events_panel(s: SwarmSnapshot) -> Panel:
    lines, statuses = [], []
    for e in s.events:
        status = str(e.get("status", "?"))
        wall = time.strftime("%H:%M:%S", time.localtime(float(e.get("wall", 0.0))))
        rate = e.get("ns_per_day")
        rate_s = f"{float(rate):.0f} ns/day" if isinstance(rate, (int, float)) and math.isfinite(float(rate)) else ""
        lines.append(f"{wall}  member {int(e.get('member_id', -1)):04d}  {status:<12} cell {e.get('cell_id', '?')}  {rate_s}")
        statuses.append("" if status == "ok" else "BAD")
    if not lines:
        lines = ["(no member finished yet)"]
    return _panel("events", "events", lines, min_lines=1, want_lines=min(len(lines), 12), priority=5,
                  statuses=statuses)


# --- spine, footer, screen -------------------------------------------------------------------

def spine_lines(s: SwarmSnapshot, budget: int, term_w: int) -> List[str]:
    w = max(10, term_w - 2)
    cells = "x".join(str(b) for b in s.bins) if s.bins else "?"
    head = (f"SWARM round {s.round_index} · {s.run_label} · {s.n_members} members · {s.timestep_fs:g} fs · "
            f"{cells} cells")
    bar = make_progress_bar(s.mean_fraction, max(8, min(36, w - 70)))
    prog = (f"{bar} {100.0 * s.mean_fraction:5.1f}%  elapsed {format_duration(s.elapsed_s)}  "
            f"ETA mean {format_duration(s.eta_mean_s)}  ETA last {format_duration(s.eta_last_s)}  "
            f"{_fmt_rate(s.aggregate_ns_per_day)} ns/day")
    counts = "  ".join(f"{p} {s.count(p)}" for p in PHASES if s.count(p))
    counts += f"  | graft gate {s.count('grafting')} (+{s.count('graft_wait')} waiting)"
    alerts = []
    if s.count("failed"):
        alerts.append(f"{s.count('failed')} failed")
    if s.stalled:
        alerts.append(f"{len(s.stalled)} stalled > 10 min")
    slow = [d.device for d in s.devices if d.slow]
    if slow:
        alerts.append("slow gpu " + ",".join(slow))
    alert = "alerts: " + ("; ".join(alerts) if alerts else "none")
    lines = [head, prog, counts, alert] if budget >= SPINE_FULL else [head, prog]
    return [_ansi_truncate(x, w) for x in lines[:max(1, budget)]]


def footer_line(s: SwarmSnapshot, dropped: Sequence[str], term_w: int) -> str:
    parts = [f"[swarm] round {s.round_index}"]
    if dropped:
        parts.append("dropped: " + ",".join(dropped))
    parts.append(time.strftime("%H:%M:%S", time.localtime(s.now)))
    return _ansi_truncate("──── " + "   ".join(parts), max(1, term_w - 2))


def render_swarm_screen(s: SwarmSnapshot, *, term_w: int, term_h: int, glyphs: str = "unicode") -> str:
    """One frame, guaranteed to fit ``term_h - 1`` lines and ``term_w - 2`` columns."""
    g = GLYPHS_ASCII if glyphs == "ascii" else GLYPHS_UNICODE
    total_budget = max(1, int(term_h) - 1)
    spine_budget, body_budget = frame_tiers(term_h)
    spine = spine_lines(s, min(spine_budget, SPINE_FULL), term_w)
    reclaimed = body_budget + max(0, spine_budget - len(spine))
    body_budget = min(reclaimed, max(0, total_budget - len(spine) - FOOTER_LINES))
    if body_budget <= 0:
        return "\n".join(spine[:total_budget])
    half = max(30, (term_w - 4) // 2)
    rows = [Row((members_panel(s, term_w, g),)),
            Row((projection_panel(s), devices_panel(s))),
            Row((coverage_panel(s, half), events_panel(s)))]
    allocated, dropped = allocate_rows(rows, body_budget, term_w)
    body = compose_rows(allocated, term_w)
    return "\n".join((*spine, *body, footer_line(s, dropped, term_w)))


__all__ = ["ACTIVE_PHASES", "DeviceStat", "Histogram", "PHASES", "SwarmSnapshot", "render_swarm_screen"]
