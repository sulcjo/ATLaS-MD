"""WINDOWS view: which window is misbehaving, and why.

With no tty there are no keys, so the detail panel auto-selects the worst-ranked
window: a SLURM log tail then always shows expanded exactly the window someone
would have pressed `j` to reach.
"""

from __future__ import annotations

from typing import Sequence

from ..tui import _weighted_panel_widths, dashboard_row_gap
from ..tui_screen import Row
from .context import DashboardContext
from .panels import cv_map_panel, pe_map_panel, window_detail_panel, window_table_panel
from .ranking import OK, WindowStatus, rank_windows

CV_WEIGHT = 2.4
PE_WEIGHT = 0.9
_PE_LABEL_OVERHEAD = 24     # pinned by tests/test_dashboard_scaling_integration.py
_CV_LABEL_OVERHEAD = 72     # measured prefix + "] nw=<n>" suffix, see gareus/logger.py history


def select_window(ctx: DashboardContext, statuses: Sequence[WindowStatus]) -> int:
    """Worst-ranked window, or window 0 when everything is healthy."""
    for status in statuses:
        if status.status != OK:
            return int(status.window)
    return int(statuses[0].window) if statuses else 0


def _map_bar_widths(ctx: DashboardContext) -> tuple[int, int]:
    gap = dashboard_row_gap(ctx.term_w)
    cv_w, pe_w = _weighted_panel_widths([CV_WEIGHT, PE_WEIGHT], term_w=ctx.term_w, gap=gap)
    return max(24, cv_w - _CV_LABEL_OVERHEAD), max(10, pe_w - _PE_LABEL_OVERHEAD)


def build(ctx: DashboardContext) -> tuple[Row, ...]:
    statuses = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k,
    )
    cv_bar_w, pe_bar_w = _map_bar_widths(ctx)
    return (
        Row(panels=(window_table_panel(ctx, statuses),)),
        Row(panels=(cv_map_panel(ctx, cv_bar_w), pe_map_panel(ctx, pe_bar_w))),
        Row(panels=(window_detail_panel(ctx, select_window(ctx, statuses)),)),
    )


__all__ = ["CV_WEIGHT", "PE_WEIGHT", "build", "select_window"]
