"""Domain-free TUI screen assembly: panels, row allocation, frame composition.

The live dashboard used to render every panel blind and let the frame writer
clip whatever did not fit, which discarded 33-52% of the frame by *position*
rather than by importance. This module inverts that: the screen hands each row
a line budget, so the frame fits by construction.

Nothing here knows about molecular dynamics. Domain panels live in
``gareus.dashboard``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .tui import (
    ABSOLUTE_MIN_PANEL_WIDTH,
    MIN_PANEL_WIDTH,
    _join_columns,
    _panel_lines,
    _weighted_panel_widths,
    dashboard_row_gap,
)

# Top border, title, separator, bottom border -- see gareus.tui._panel_lines.
PANEL_CHROME_LINES = 4

FULL_SPINE_LINES = 10
COMPACT_SPINE_LINES = 5
FOOTER_LINES = 1
_FULL_SPINE_MIN_USABLE = 27   # a 10-line spine past this point is >36% of the screen
_COMPACT_SPINE_MIN_USABLE = 19


@dataclass(frozen=True)
class Panel:
    """One boxed panel's content plus its line-budget declaration.

    ``priority`` is ordinal with 1 as the most important; rows are dropped from
    the least important end when the budget cannot hold every minimum.
    ``line_statuses`` is a parallel array of severity labels, one per line
    (empty string for a header or a line with no severity). The allocator later
    calls ``trim_panel`` to cut content to fit the budget; that function composes
    a severity-aware truncation notice ("+9 more (1 bad, 2 warn)") from the labels
    of the lines it hides, since a panel cannot know in advance how many lines
    the allocator will hide.
    """

    key: str
    title: str
    lines: tuple[str, ...]
    min_lines: int = 1
    want_lines: int = 1
    priority: int = 5
    weight: float = 1.0
    line_statuses: tuple[str, ...] = ()


@dataclass(frozen=True)
class Row:
    """Panels rendered side by side. They share one set of border lines."""

    panels: tuple[Panel, ...]


def row_priority(row: Row) -> int:
    """A row is as important as its most important panel."""
    return min((p.priority for p in row.panels), default=9)


def row_min(row: Row) -> int:
    """Total lines a row occupies at its minimum, chrome included."""
    return max((p.min_lines for p in row.panels), default=1) + PANEL_CHROME_LINES


def row_want(row: Row) -> int:
    """Total lines a row would occupy if fully satisfied, chrome included."""
    return max((p.want_lines for p in row.panels), default=1) + PANEL_CHROME_LINES


def allocate_rows(
    rows: Sequence[Row], budget: int
) -> tuple[tuple[tuple[Row, int], ...], tuple[str, ...]]:
    """Fit ``rows`` into ``budget`` lines.

    Returns ``((row, body_lines), ...)`` in declaration order, plus the keys of
    panels that were dropped entirely. Dropped panels are reported so the caller
    can name them in the footer -- silence is what made the old behaviour a bug
    rather than a limitation.
    """
    kept = list(rows)
    dropped: list[str] = []
    while kept and sum(row_min(r) for r in kept) > int(budget):
        victim = max(range(len(kept)), key=lambda i: (row_priority(kept[i]), i))
        dropped.extend(p.key for p in kept.pop(victim).panels)
    if not kept:
        return (), tuple(dropped)

    body = {i: row_min(r) - PANEL_CHROME_LINES for i, r in enumerate(kept)}
    spare = int(budget) - sum(row_min(r) for r in kept)
    for i in sorted(range(len(kept)), key=lambda i: (row_priority(kept[i]), i)):
        if spare <= 0:
            break
        room = row_want(kept[i]) - row_min(kept[i])
        give = min(max(0, room), spare)
        body[i] += give
        spare -= give
    return tuple((r, body[i]) for i, r in enumerate(kept)), tuple(dropped)


def _composition_tail(hidden_statuses: Sequence[str], hidden: int) -> str:
    """Describe a truncated tail by severity, not just by count.

    "+9 more" tells a reader nothing about whether the tail mattered;
    "+9 more (1 bad, 2 warn)" tells them to widen the terminal.
    """
    labels = [str(s).upper() for s in hidden_statuses]
    bad = labels.count("BAD")
    warn = labels.count("WARN")
    if not labels:
        return f"… {hidden} more"
    if not bad and not warn:
        return f"+{hidden} more (all ok)"
    parts = [f"{bad} bad"] if bad else []
    if warn:
        parts.append(f"{warn} warn")
    return f"+{hidden} more ({', '.join(parts)})"


def trim_panel(panel: Panel, body_lines: int) -> Panel:
    """Cut a panel's content to ``body_lines``, keeping the first (worst) lines.

    Panel content arrives already ranked worst-first, so a positional cut here
    is a severity cut. The last kept line becomes the truncation notice, whose
    wording depends on what was hidden.
    """
    limit = max(0, int(body_lines))
    if len(panel.lines) <= limit:
        return panel
    keep = max(0, limit - 1)
    hidden = len(panel.lines) - keep
    tail = _composition_tail(panel.line_statuses[keep:], hidden)
    return replace(panel, lines=tuple(panel.lines[:keep]) + (tail,))


def frame_tiers(term_h: int) -> tuple[int, int]:
    """Split the terminal height into ``(spine_lines, body_budget)``.

    One line is reserved to match ``gareus.tui._safe_tui_frame_text``'s own
    reserve, and one more for the footer. Below 20 rows the view body is
    dropped entirely and only the spine renders.
    """
    usable = max(1, int(term_h) - 1)
    if usable >= _FULL_SPINE_MIN_USABLE:
        return FULL_SPINE_LINES, max(0, usable - FULL_SPINE_LINES - FOOTER_LINES)
    if usable >= _COMPACT_SPINE_MIN_USABLE:
        return COMPACT_SPINE_LINES, max(0, usable - COMPACT_SPINE_LINES - FOOTER_LINES)
    return usable, 0


def compose_rows(
    allocated: Sequence[tuple[Row, int]], term_w: int, gap: int | None = None
) -> tuple[str, ...]:
    """Render allocated rows to text lines, stacking when too narrow for columns."""
    gap_v = dashboard_row_gap(term_w) if gap is None else int(gap)
    usable = max(40, int(term_w) - 2)
    out: list[str] = []
    for row, body in allocated:
        panels = [trim_panel(p, body) for p in row.panels]
        n = len(panels)
        min_w = max(ABSOLUTE_MIN_PANEL_WIDTH, MIN_PANEL_WIDTH)
        if n > 1 and usable < n * min_w + (n - 1) * gap_v:
            for p in panels:
                out.extend(_panel_lines(p.title, list(p.lines), usable))
            continue
        widths = _weighted_panel_widths(
            [p.weight for p in panels], term_w=term_w, gap=gap_v, min_panel_width=min_w
        )
        cols = [_panel_lines(p.title, list(p.lines), w) for p, w in zip(panels, widths)]
        out.extend(_join_columns(cols, gap=gap_v))
    return tuple(out)


__all__ = [
    "PANEL_CHROME_LINES",
    "Panel",
    "Row",
    "allocate_rows",
    "compose_rows",
    "frame_tiers",
    "row_min",
    "row_priority",
    "row_want",
    "trim_panel",
]
