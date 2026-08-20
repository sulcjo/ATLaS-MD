"""Assemble the frame: spine, one view's allocated rows, footer.

The frame is built to fit the terminal, so gareus.tui._safe_tui_frame_text's clip
becomes a safety net that should never fire. Panels that could not be given their
minimum are named in the footer -- silence about them is what made the old
overflow a bug rather than a limitation.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Mapping, Sequence

from ..colors import ROLE_SECTION, color_text, role_text
from ..math_helpers import boost_anharmonicity
from ..tui import _ansi_truncate
from ..tui_screen import FOOTER_LINES, allocate_rows, compose_rows, frame_tiers
from . import view_physics, view_progress, view_windows
from .context import DashboardContext
from .ranking import BAD, rank_windows
from .spine import spine_lines
from .view_physics import connectivity_verdict

VIEWS: Mapping[str, Callable[[DashboardContext], tuple]] = {
    "progress": view_progress.build,
    "physics": view_physics.build,
    "windows": view_windows.build,
}
DEFAULT_VIEW = "progress"
ANHARMONICITY_PROMOTE = 1.0


def promotion_reasons(ctx: DashboardContext) -> tuple[str, ...]:
    """Conditions that justify taking over the screen, worst first."""
    reasons: list[str] = []
    ranked = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k,
    )
    for status in ranked:
        if status.status == BAD:
            reasons.append(f"windows: w{status.window:02d} " + ", ".join(status.reasons))
            break
    # Promote on a PROVEN split only. `connected_components` alone cannot tell a
    # disconnected graph from an unmeasured one, so checking `largest < n_windows`
    # here would fire on every run's first frames -- before any exchange has been
    # attempted every window is its own component -- and pin `auto` to PHYSICS
    # until exchanges accumulate. That is the exact false alarm `connectivity_verdict`
    # exists to prevent; the promotion path has to honour it too.
    graph_state, largest, _measured = connectivity_verdict(ctx)
    if graph_state == "split":
        reasons.append(f"physics: graph {largest}/{ctx.n_windows} connected")
    boosts = [b for b in ctx.boost_history_all if math.isfinite(b)]
    if boosts:
        # `boost_anharmonicity` (gareus/math_helpers.py) returns the score under
        # the key "score", not "anharmonicity_score" -- that other key name
        # belongs to the unrelated post-hoc MBAR-analysis anharmonicity
        # computation in gareus/mbar_analysis/pmf.py. Reading the wrong key here
        # made this rule permanently unreachable: `.get("anharmonicity_score", nan)`
        # always fell through to nan, so `math.isfinite(score)` was always False
        # and no run's boost distribution -- however non-Gaussian -- could ever
        # promote the auto view to PHYSICS through this branch. See
        # gareus/dashboard/panels.py's `boost_envelope_panel`, which reads the
        # correct key and carries the same warning.
        score = float(boost_anharmonicity(boosts).get("score", float("nan")))
        if math.isfinite(score) and score > ANHARMONICITY_PROMOTE:
            reasons.append(f"physics: anharmonicity {score:.2f}")
    return tuple(reasons)


def resolve_view(ctx: DashboardContext, requested: str) -> str:
    """Explicit request wins; `auto` promotes only while a rule holds."""
    name = str(requested or "").lower()
    if name in VIEWS:
        return name
    if name != "auto":
        return DEFAULT_VIEW
    for reason in promotion_reasons(ctx):
        target = reason.split(":", 1)[0].strip()
        if target in VIEWS:
            return target
    return DEFAULT_VIEW


def footer_line(ctx: DashboardContext, dropped: Sequence[str], view: str) -> str:
    tabs = "   ".join(
        (role_text(f"[{i}] {name.upper()}", ROLE_SECTION) if name == view
         else color_text(f"[{i}] {name}", "dim"))
        for i, name in enumerate(VIEWS, start=1)
    )
    clock = time.strftime("%H:%M:%S", time.localtime(ctx.now_wall))
    parts = [tabs]
    if dropped:
        parts.append(color_text("dropped: " + ", ".join(dict.fromkeys(dropped)), "dim"))
    parts.append(color_text(clock, "dim"))
    # No floor above the real budget here either -- see the identical fix and
    # rationale in gareus/dashboard/spine.py's spine_lines.
    return _ansi_truncate("──── " + "   ".join(parts), max(1, ctx.term_w - 2))


def render_screen(ctx: DashboardContext) -> str:
    """Full frame as one string, guaranteed to fit `ctx.term_h - 1` lines."""
    total_budget = max(1, int(ctx.term_h) - 1)
    spine_budget, body_budget = frame_tiers(ctx.term_h)
    spine = spine_lines(ctx, spine_budget)
    # Hand any unused spine reservation to the view: the spine has nine real
    # line-kinds (eight for 1D), so a full-tier frame reclaims one or two rows
    # here instead of rendering them blank. But never reclaim past what the
    # real terminal height can still hold once the spine's own true length and
    # the footer's one line are set aside -- frame_tiers' own short-terminal
    # tier hands back body_budget=0 specifically because there is no room left
    # for a view body *or* a footer once the spine fills it; naively adding
    # (spine_budget - len(spine)) back on top of that, with no ceiling, was
    # manufacturing view+footer room that was never really there (reproduced
    # at term_h=18: physics/windows views rendered 18 lines into a 17-line
    # budget -- 1 line over, every time, regardless of terminal width).
    reclaimed = body_budget + max(0, spine_budget - len(spine))
    room_for_body_and_footer = max(0, total_budget - len(spine) - FOOTER_LINES)
    body_budget = min(reclaimed, room_for_body_and_footer)
    if body_budget <= 0:
        return "\n".join(spine)
    view = resolve_view(ctx, ctx.view)
    rows = VIEWS[view](ctx)
    # term_w must match between allocate_rows and compose_rows: the allocator
    # prices a row differently depending on whether it will stack at this
    # width (stacked panels each carry their own chrome). Omitting term_w here
    # defaults to side-by-side pricing regardless of what compose_rows below
    # actually does, silently under-costing any row that stacks on a narrow
    # terminal.
    allocated, dropped = allocate_rows(rows, body_budget, ctx.term_w)
    body = compose_rows(allocated, ctx.term_w)
    return "\n".join((*spine, *body, footer_line(ctx, dropped, view)))


__all__ = ["ANHARMONICITY_PROMOTE", "DEFAULT_VIEW", "VIEWS", "footer_line",
           "promotion_reasons", "render_screen", "resolve_view"]
