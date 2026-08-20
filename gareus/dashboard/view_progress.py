"""PROGRESS view: where the campaign is, and how long the rest will take.

Everything here comes from the pool ledger on disk, which is the only record of
the quality-gate extension rounds -- they were previously invisible live.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..colors import color_text
from ..tui import format_duration
from ..tui_screen import Panel, Row
from .context import DashboardContext
from .panels import _num, panel
from .spine import _ns_per_day


def collapse_extension_rounds(
    events: Sequence[Mapping[str, object]]
) -> tuple[tuple[str, int, float, int, bool], ...]:
    """Group ledger events by segment label, in first-appearance order.

    Returns ``(label, occurrences, total_ns, n_states, is_current)`` per label.

    Why grouping rather than a chronological list with the repeats renamed: on a
    real ledger the quality-gate extension loop does not repeat one segment
    consecutively, it cycles through several. The chignolin_5 ledger holds 27
    events over just 6 distinct labels -- ``epoch_001/baseline``,
    ``final/baseline`` and ``final/topup_001_375000`` each appear 8 times,
    interleaved. Numbering repeats in place would put "round 5" of one segment
    directly under a different segment's row, and 27 rows in a 12-line panel
    means 16 of them get trimmed away regardless.

    Grouped, the same ledger is 6 rows that fit, and they answer the question the
    panel exists for -- where the MD budget actually went. Chronology is not lost
    entirely: the group holding the final event is marked current.
    """
    order: list[str] = []
    counts: dict[str, int] = {}
    totals: dict[str, float] = {}
    states: dict[str, int] = {}
    last_label = ""
    for event in events:
        # A wrong-shaped sidecar (e.g. a stray non-dict entry in "events") must
        # not raise -- partial/malformed JSON is a display state, not a crash.
        if not isinstance(event, Mapping):
            continue
        label = str(event.get("label", "") or "")
        if label not in counts:
            order.append(label)
            counts[label] = 0
            totals[label] = 0.0
        counts[label] += 1
        totals[label] += float(event.get("consumed_ns", 0.0) or 0.0)
        states[label] = int(event.get("n_states", 0) or 0)
        last_label = label
    return tuple(
        (label, counts[label], totals[label], states[label], label == last_label)
        for label in order
    )


def _ledger_age(ctx: DashboardContext) -> str:
    """How stale the ledger is. It is written at phase boundaries, so early in a
    segment it legitimately lags reality -- say so rather than implying it is live."""
    if ctx.sidecar.pool_mtime is None:
        return ""
    age_s = max(0.0, ctx.now_wall - float(ctx.sidecar.pool_mtime))
    return f" (ledger {format_duration(age_s)} old)"


def _fit_label(label: str, width: int) -> str:
    """Shorten a segment label without inventing a plausible-looking one.

    A plain `f"{label:<26.26}"` cuts mid-number, turning
    `epoch_001/topup_001_3388000` into `epoch_001/topup_001_338800` -- a step
    count that reads as complete and is wrong, and which two different segments
    can collide on. Drop the trailing step-count suffix first, since it is the
    least informative part of the name, and only then ellipsize.
    """
    text = str(label)
    if len(text) <= width:
        return text
    head, sep, tail = text.rpartition("_")
    if sep and tail.isdigit() and len(head) <= width:
        return head
    return text[: max(1, width - 1)] + "…"


def timeline_panel(ctx: DashboardContext) -> Panel:
    pool = ctx.sidecar.pool or {}
    events = pool.get("events") or []
    if not events:
        return panel("timeline", "campaign timeline",
                     [color_text("unavailable: no adaptive_runtime_pool.json yet "
                                 "(non-adaptive run, or first epoch still open)", "dim")],
                     min_lines=1, want_lines=2, priority=1)
    rows = collapse_extension_rounds(events)
    peak = max((ns for _l, _n, ns, _s, _c in rows), default=1.0) or 1.0
    label_w = 26
    bar_w = max(10, (ctx.term_w - 2) - 64)
    lines = []
    for label, occurrences, consumed, states, is_current in rows:
        filled = int(round(bar_w * consumed / peak)) if peak > 0 else 0
        shown = f"{label} x{occurrences}" if occurrences > 1 else label
        mark = "▶ running" if is_current else "✓ done"
        lines.append(f" {_fit_label(shown, label_w):<{label_w}} "
                     + "█" * filled + " " * (bar_w - filled)
                     + f" {consumed:9.1f} ns  {states:2d} st  {mark}")
    remaining = float(pool.get("remaining_ns", 0.0) or 0.0)
    if remaining > 0:
        # Same label field width as the rows above: this panel exists so bar
        # lengths can be compared by eye, and a reserve bar starting four columns
        # left of the others defeats that.
        lines.append(f" {_fit_label('final (reserve)', label_w):<{label_w}} "
                     + "░" * bar_w
                     + f" {remaining:9.1f} ns       ○ scheduled")
    # `want_lines` must never fall below `min_lines=6`: a short `lines` (e.g. a
    # single unavailable-ledger message, or very few real events) made
    # `min(12, len(lines))` land below 6, handing the allocator a row whose
    # `want - min` "room" (gareus/tui_screen.py's `allocate_rows`) went
    # negative -- clamp so the panel never claims to want less than its own
    # declared minimum.
    return panel("timeline", "campaign timeline" + _ledger_age(ctx), lines,
                 min_lines=6, want_lines=max(6, min(12, len(lines))), priority=1)


def projection_panel(ctx: DashboardContext) -> Panel:
    pool = ctx.sidecar.pool or {}
    remaining = float(pool.get("remaining_ns", 0.0) or 0.0)
    ns_day = _ns_per_day(ctx)
    aggregate = ns_day * ctx.n_replicas if math.isfinite(ns_day) else float("nan")
    # `_ns_per_day` is nan whenever timestep_fs/elapsed_s aren't measurable yet
    # (e.g. a real logger built before the first frame has any wall-clock
    # elapsed) -- a bare `{ns_day:.0f}` printed the literal token "nan" here,
    # inconsistently with the very next branch below, which already has its
    # own correct not-yet-measured wording.
    lines = [f" perf now        {_num(ns_day, 0, 0)} ns/day/rep   "
             f"({_num(aggregate, 0, 0)} aggregate)"]
    if remaining > 0 and math.isfinite(aggregate) and aggregate > 0:
        lines.append(f" pool remaining  {remaining:.0f} ns = "
                     f"{remaining / aggregate:.1f} GPU-days at this rate")
    elif remaining > 0:
        lines.append(f" pool remaining  {remaining:.0f} ns (GPU-days unknown until "
                     f"throughput is measured)")
    lines.append(f" this segment    eta {format_duration(ctx.eta_s)}")
    lines.append(f" elapsed         {format_duration(ctx.elapsed_s)}")
    return panel("projection", "projection", lines,
                 min_lines=4, want_lines=7, priority=2, weight=1.0)


def throughput_panel(ctx: DashboardContext) -> Panel:
    total_samples = sum(len(v) for v in ctx.cv_history_by_window.values())
    lines = [
        f" replicas        {ctx.n_replicas}",
        f" cv samples      {total_samples} in live history",
        f" step            {ctx.display_step} / {ctx.display_total_steps or '?'}",
    ]
    if ctx.timestep_fs > 0:
        lines.append(f" sim time        {ctx.display_step * ctx.timestep_fs / 1e6:.3g} ns/rep")
    return panel("throughput", "throughput", lines,
                 min_lines=3, want_lines=6, priority=3, weight=1.0)


def build(ctx: DashboardContext) -> tuple[Row, ...]:
    return (
        Row(panels=(timeline_panel(ctx),)),
        Row(panels=(projection_panel(ctx), throughput_panel(ctx))),
    )


__all__ = ["build", "collapse_extension_rounds", "projection_panel", "throughput_panel",
           "timeline_panel"]
