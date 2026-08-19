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


def _would_stack_on_narrow_terminal(row: Row, term_w: int) -> bool:
    """Predicate shared between allocator and composer to detect narrow-terminal stacking.

    Returns True if the row's panels would be stacked vertically instead of
    side-by-side, using the exact same threshold arithmetic as compose_rows.
    """
    gap_v = dashboard_row_gap(term_w)
    usable = max(40, int(term_w) - 2)
    n = len(row.panels)
    min_w = max(ABSOLUTE_MIN_PANEL_WIDTH, MIN_PANEL_WIDTH)
    return n > 1 and usable < n * min_w + (n - 1) * gap_v


def row_min(row: Row, term_w: int | None = None) -> int:
    """Total lines a row occupies at its minimum, chrome included.

    If term_w is provided, prices the row for the layout it would use on that
    terminal width (stacked vs side-by-side). Otherwise uses side-by-side pricing.
    """
    if term_w is not None and _would_stack_on_narrow_terminal(row, term_w):
        return sum(p.min_lines for p in row.panels) + PANEL_CHROME_LINES * len(row.panels)
    return max((p.min_lines for p in row.panels), default=1) + PANEL_CHROME_LINES


def row_want(row: Row, term_w: int | None = None) -> int:
    """Total lines a row would occupy if fully satisfied, chrome included.

    If term_w is provided, prices the row for the layout it would use on that
    terminal width (stacked vs side-by-side). Otherwise uses side-by-side pricing.
    """
    if term_w is not None and _would_stack_on_narrow_terminal(row, term_w):
        return sum(p.want_lines for p in row.panels) + PANEL_CHROME_LINES * len(row.panels)
    return max((p.want_lines for p in row.panels), default=1) + PANEL_CHROME_LINES


def allocate_rows(
    rows: Sequence[Row], budget: int, term_w: int | None = None
) -> tuple[tuple[tuple[Row, int], ...], tuple[str, ...]]:
    """Fit ``rows`` into ``budget`` lines.

    Returns ``((row, body_lines), ...)`` in declaration order, plus the keys of
    panels that were dropped entirely. Dropped panels are reported so the caller
    can name them in the footer -- silence is what made the old behaviour a bug
    rather than a limitation.

    When term_w is provided, rows are priced according to whether they would
    stack vertically (ensuring "fits by construction" even on narrow terminals).
    """
    kept = list(rows)
    dropped: list[str] = []
    while kept and sum(row_min(r, term_w) for r in kept) > int(budget):
        victim = max(range(len(kept)), key=lambda i: (row_priority(kept[i]), i))
        dropped.extend(p.key for p in kept.pop(victim).panels)
    if not kept:
        return (), tuple(dropped)

    # Compute body lines: row_min includes chrome, so subtract it.
    # Works for both stacking and side-by-side since row_min accounts for layout.
    body = {i: row_min(r, term_w) - PANEL_CHROME_LINES for i, r in enumerate(kept)}

    spare = int(budget) - sum(row_min(r, term_w) for r in kept)
    for i in sorted(range(len(kept)), key=lambda i: (row_priority(kept[i]), i)):
        if spare <= 0:
            break
        room = row_want(kept[i], term_w) - row_min(kept[i], term_w)
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
    allocated: Sequence[tuple[Row, int]], term_w: int
) -> tuple[str, ...]:
    """Render allocated rows to text lines, stacking when too narrow for columns."""
    gap_v = dashboard_row_gap(term_w)
    out: list[str] = []
    for row, body in allocated:
        min_w = max(ABSOLUTE_MIN_PANEL_WIDTH, MIN_PANEL_WIDTH)
        if _would_stack_on_narrow_terminal(row, term_w):
            # Stacked layout: split the body budget across panels.
            # Row was priced at: body + PANEL_CHROME_LINES (one shared border set)
            # Stacked needs: PANEL_CHROME_LINES * n (n separate border sets)
            # Available body lines to divide: body + PANEL_CHROME_LINES - PANEL_CHROME_LINES * n
            n = len(row.panels)
            available = body + PANEL_CHROME_LINES - PANEL_CHROME_LINES * n
            # Sanity check: at body = sum(min_lines) + 4n - 4, available = sum(min_lines)
            # so every panel can get at least min_lines.

            # Distribute: give each panel its min_lines first, then remaining toward want_lines
            panel_bodies: dict[int, int] = {}
            for i, p in enumerate(row.panels):
                panel_bodies[i] = max(1, p.min_lines)

            spare = available - sum(panel_bodies.values())
            for i, p in enumerate(row.panels):
                if spare <= 0:
                    break
                room = p.want_lines - p.min_lines
                give = min(max(0, room), spare)
                panel_bodies[i] += give
                spare -= give

            # Trim each panel to its own budget and render
            render_w = max(18, int(term_w) - 2)
            for i, p in enumerate(row.panels):
                trimmed = trim_panel(p, panel_bodies[i])
                out.extend(_panel_lines(trimmed.title, list(trimmed.lines), render_w))
            continue

        # Side-by-side layout: all panels get the same body height
        panels = [trim_panel(p, body) for p in row.panels]
        widths = _weighted_panel_widths(
            [p.weight for p in panels], term_w=term_w, gap=gap_v, min_panel_width=min_w
        )
        # Clamp widths to actual terminal usable width
        clamped_widths = [max(18, min(w, int(term_w) - 2)) for w in widths]
        cols = [_panel_lines(p.title, list(p.lines), w) for p, w in zip(panels, clamped_widths)]
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
