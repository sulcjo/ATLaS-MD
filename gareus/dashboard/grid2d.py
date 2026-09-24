"""2D window map: the (CV1, CV2) umbrella layout as a grid of lambda rungs.

A sparse pseudo-2D layout (chignolin_9: 16 CV1 x 5 CV2 centres, 59 occupied
cells, 4 lambda rungs each = 236 windows) is unreadable as the flat ranked table
alone -- "w173 BAD" does not say *where* in CV space the trouble is. Here every
occupied cell shows one glyph per rung, lowest lambda first, carrying the same
status `rank_windows` gives the table, so the two can never disagree.

Status is readable without colour (distinct glyphs); colour only reinforces it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ..colors import ROLE_BAD, ROLE_GOOD, ROLE_MUTED, ROLE_WARN, color_text, role_text
from ..tui import _ansi_truncate, strip_ansi_len
from .context import DashboardContext, per_window_from_rows
from .panels import panel
from .ranking import BAD, WARN, WindowStatus

# Centres are recorded as floats computed from the same layout table, so equal
# centres differ at most by float noise; 6 decimals groups them without merging
# genuinely distinct centres (the closest real pair is ~0.05 apart).
_CENTRE_DECIMALS = 6
_ROW_LABEL_W = 7          # "+1.348 " -- CV2 label column
_MIN_CELL_W = 2           # one glyph + one space: the compact (worst-rung) mode

_GLYPHS = {
    "unicode": {"ok": "■", WARN: "!", BAD: "✗", "none": "·"},
    "ascii": {"ok": "#", WARN: "!", BAD: "x", "none": "."},
}
_ROLE = {"ok": ROLE_GOOD, WARN: ROLE_WARN, BAD: ROLE_BAD, "none": ROLE_MUTED}
_SEVERITY = {"none": 0, "ok": 1, WARN: 2, BAD: 3}


@dataclass(frozen=True)
class Grid:
    cv1_values: tuple[float, ...]                    # columns, ascending
    cv2_values: tuple[float, ...]                    # rows, descending (top = highest)
    cells: Mapping[tuple[int, int], tuple[int, ...]]  # (col, row) -> windows, lambda ascending


def layout_grid(
    centers_a: Sequence[float],
    secondary_centers: Sequence[float],
    lambdas: Sequence[float],
) -> Grid:
    """Group windows by rounded (CV1, CV2) centre; order each cell by lambda."""
    n = min(len(centers_a), len(secondary_centers))
    keys = [(round(float(centers_a[w]), _CENTRE_DECIMALS),
             round(float(secondary_centers[w]), _CENTRE_DECIMALS)) for w in range(n)]
    cols = sorted({k[0] for k in keys})
    rows = sorted({k[1] for k in keys}, reverse=True)
    col_of = {v: i for i, v in enumerate(cols)}
    row_of = {v: i for i, v in enumerate(rows)}

    def _lam(w: int) -> float:
        value = float(lambdas[w]) if w < len(lambdas) else math.nan
        return value if math.isfinite(value) else math.inf

    cells: dict[tuple[int, int], list[int]] = {}
    for w, (c1, c2) in enumerate(keys):
        cells.setdefault((col_of[c1], row_of[c2]), []).append(w)
    ordered = {cell: tuple(sorted(ws, key=lambda w: (_lam(w), w))) for cell, ws in cells.items()}
    return Grid(tuple(cols), tuple(rows), ordered)


def _state(w: int, status_by_window: Mapping[int, str], sampled: set[int]) -> str:
    if w not in sampled:
        return "none"
    status = status_by_window.get(w)
    return status if status in (WARN, BAD) else "ok"


def _cell_text(states: Sequence[str], glyphs: Mapping[str, str], width: int) -> str:
    text = "".join(role_text(glyphs[s], _ROLE[s], bold=s != "none") for s in states)
    return text + " " * max(0, width - len(states))


def _column_header(grid: Grid, cell_w: int, cv1_free_columns: frozenset = frozenset()) -> str:
    """CV1 centre labels, thinned so no two overlap.

    A column whose windows carry no CV1 restraint is labelled ``free``: its
    centre is a parking value, not a place in CV1 space.
    """
    labels = ["free" if i in cv1_free_columns else f"{v:.2f}"
              for i, v in enumerate(grid.cv1_values)]
    label_w = max((len(s) for s in labels), default=4) + 1
    every = max(1, math.ceil(label_w / cell_w))
    out = " " * _ROW_LABEL_W
    for i, label in enumerate(labels):
        slot = cell_w * every if i % every == 0 else 0
        if slot:
            out += label.ljust(slot)[:slot]
    return color_text(out.rstrip(), "dim")


def render_grid(
    grid: Grid,
    status_by_window: Mapping[int, str],
    sampled: set[int],
    *,
    width: int,
    glyphs: str = "unicode",
    free_windows: Sequence[int] = (),
    cv1_free_columns: frozenset = frozenset(),
) -> Optional[list[str]]:
    """Header, one line per CV2 row, legend -- or None if it cannot fit `width`.

    ``free_windows`` are unrestrained (k = 0) states: they sit at no point in CV
    space, so they get their own "free" row instead of a phantom grid column.
    """
    table = _GLYPHS.get(glyphs, _GLYPHS["unicode"])
    n_rungs = max((len(ws) for ws in grid.cells.values()), default=1)
    ncols = len(grid.cv1_values)
    full_w = n_rungs + 1
    if _ROW_LABEL_W + ncols * full_w <= width:
        cell_w, per_rung = full_w, True
    elif _ROW_LABEL_W + ncols * _MIN_CELL_W <= width:
        cell_w, per_rung = _MIN_CELL_W, False
    else:
        return None

    counts = {"ok": 0, WARN: 0, BAD: 0, "none": 0}
    lines = [_column_header(grid, cell_w, frozenset(cv1_free_columns))]
    for r, cv2 in enumerate(grid.cv2_values):
        line = color_text(f"{cv2:+.3f}".rjust(_ROW_LABEL_W - 1), "dim") + " "
        for c in range(ncols):
            windows = grid.cells.get((c, r), ())
            states = [_state(w, status_by_window, sampled) for w in windows]
            for s in states:
                counts[s] += 1
            if not per_rung and states:
                states = [max(states, key=_SEVERITY.__getitem__)]
            line += _cell_text(states, table, cell_w)
        lines.append(line.rstrip())
    if free_windows:
        states = [_state(w, status_by_window, sampled) for w in free_windows]
        for s in states:
            counts[s] += 1
        lines.append("free".ljust(_ROW_LABEL_W) + _cell_text(states, table, len(states))
                     + color_text("   unrestrained (k=0)", "dim"))
    legend = "   ".join(
        role_text(table[k], _ROLE[k]) + f" {name} {counts[k]}"
        for k, name in (("ok", "ok"), (WARN, "warn"), (BAD, "bad"), ("none", "no samples"))
    )
    rung_note = "cell = λ rungs, low→high" if per_rung else "cell = worst rung (narrow terminal)"
    full = legend + "   " + color_text(rung_note, "dim")
    lines.append(full if strip_ansi_len(full) <= width else _ansi_truncate(legend, width))
    return lines


def _lambdas_by_window(ctx: DashboardContext) -> list[float]:
    """Each window's lambda, read from the replica currently occupying it."""
    return per_window_from_rows(ctx.rows, "gamd_lambda", ctx.n_windows)


def _sort_lam(value: float) -> float:
    return value if math.isfinite(value) else math.inf


def _secondary_k_by_window(ctx: DashboardContext) -> list[float]:
    """CV2 force constant per window: the layout's list, else the live rows; NaN = unknown."""
    if len(ctx.secondary_k) >= ctx.n_windows:
        return [float(x) for x in ctx.secondary_k[:ctx.n_windows]]
    return per_window_from_rows(ctx.rows, "secondary_cv_k_kcal_mol", ctx.n_windows)


def _unrestrained_windows(ctx: DashboardContext) -> list[int]:
    """Windows with k = 0 on BOTH CVs: the free/zero-k stack.

    A window restrained on CV2 only also has primary k = 0 (chignolin_9 has 20
    of those against 4 truly free ones), so an unknown CV2 k must count as
    restrained -- otherwise every CV2-only window would leave the grid.
    """
    k2 = _secondary_k_by_window(ctx)
    free = []
    for w in range(min(ctx.n_windows, len(ctx.k_list))):
        try:
            k1 = float(ctx.k_list[w])
        except (TypeError, ValueError):
            continue
        if k1 == 0.0 and k2[w] == 0.0:
            free.append(w)
    return free


def _cv1_free_columns(ctx: DashboardContext, grid: Grid) -> frozenset:
    """Grid columns in which every window has CV1 force constant 0."""
    def _k1(w: int) -> float:
        try:
            return float(ctx.k_list[w])
        except (IndexError, TypeError, ValueError):
            return math.nan
    by_col: dict[int, list[int]] = {}
    for (col, _row), windows in grid.cells.items():
        by_col.setdefault(col, []).extend(windows)
    return frozenset(col for col, ws in by_col.items() if ws and all(_k1(w) == 0.0 for w in ws))


def _layout_subset(ctx: DashboardContext, windows: Sequence[int], lam: Sequence[float]) -> Grid:
    """layout_grid over `windows` only, keyed by their real window indices."""
    sub_grid = layout_grid([ctx.centers_a[w] for w in windows],
                           [ctx.secondary_centers[w] for w in windows],
                           [lam[w] for w in windows])
    cells = {cell: tuple(windows[i] for i in ws) for cell, ws in sub_grid.cells.items()}
    return Grid(sub_grid.cv1_values, sub_grid.cv2_values, cells)


def window_grid_panel(ctx: DashboardContext, statuses: Sequence[WindowStatus], *, priority: int = 2):
    """The 2D window map panel, or None for a 1D run / a terminal too narrow for it."""
    if (not ctx.is_2d or len(ctx.secondary_centers) < ctx.n_windows
            or len(ctx.centers_a) < ctx.n_windows or not ctx.centers_a):
        return None
    lam = _lambdas_by_window(ctx)
    free = set(_unrestrained_windows(ctx))
    grid = _layout_subset(ctx, [w for w in range(ctx.n_windows) if w not in free], lam)
    status_by_window = {int(s.window): s.status for s in statuses}
    sampled = {w for w in range(ctx.n_windows) if ctx.cv_history_by_window.get(w)}
    # Panel border (2) plus the row gap the compositor may add either side.
    lines = render_grid(grid, status_by_window, sampled, width=max(1, ctx.term_w - 6),
                        glyphs=ctx.glyphs, free_windows=sorted(free, key=lambda w: (_sort_lam(lam[w]), w)),
                        cv1_free_columns=_cv1_free_columns(ctx, grid))
    if lines is None:
        return None
    cv2_name = ctx.secondary_cv_type or "cv2"
    title = f"2D window map   cv1 → {ctx.primary_cv_label}   cv2 ↑ {cv2_name}"
    return panel("window_grid", title, lines, min_lines=len(lines),
                 want_lines=len(lines), priority=int(priority), weight=1.0)


__all__ = ["Grid", "layout_grid", "render_grid", "window_grid_panel"]
