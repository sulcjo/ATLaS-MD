"""PHYSICS view: will MBAR work on what is being collected right now?

Both derived analyses here already exist post-run in analyze_gareus_mbar.py. The
cheap live versions are what let a mistuned k or a disconnected state graph be
fixed on day 1 instead of discovered after a week of MD.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ..colors import ROLE_BAD, ROLE_GOOD, ROLE_WARN, color_text, role_text
from ..tui_screen import Panel, Row
from .context import DashboardContext
from .panels import boost_envelope_panel, overlap_panel, panel
from .ranking import DEAD_ACCEPTANCE, OK, restraint_sigma

TARGET_OVERLAP = 0.30


def connected_components(
    n_windows: int,
    acceptance_pairs: Mapping[tuple[int, int], float],
    min_acceptance: float = DEAD_ACCEPTANCE,
) -> tuple[frozenset[int], ...]:
    """Union-find over pairs that actually exchange.

    A NaN rate means "not attempted yet", which is not evidence of a link, so it
    does not join two components.
    """
    parent = list(range(int(n_windows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), rate in acceptance_pairs.items():
        if not (0 <= a < n_windows and 0 <= b < n_windows):
            continue
        if math.isfinite(rate) and rate >= float(min_acceptance):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups: dict[int, set[int]] = {}
    for w in range(int(n_windows)):
        groups.setdefault(find(w), set()).add(w)
    return tuple(sorted((frozenset(g) for g in groups.values()), key=lambda s: (-len(s), min(s))))


def sigma_spacing_panel(ctx: DashboardContext) -> Panel:
    """Is k tuned for the window spacing actually in use?"""
    if len(ctx.centers_a) < 2 or not ctx.k_list:
        return panel("sigma", "σ-vs-spacing check",
                     [color_text("needs at least two windows", "dim")],
                     min_lines=1, want_lines=2, priority=3)
    spacings = np.diff(np.asarray(ctx.centers_a, dtype=float))
    median_spacing = float(np.median(np.abs(spacings)))
    k_median = float(np.median(np.asarray(ctx.k_list, dtype=float)))
    sigma = restraint_sigma(k_median, ctx.temperature_k)
    observed = [v for v in ctx.overlap_pairs.values() if math.isfinite(v)]
    obs_median = float(np.median(observed)) if observed else float("nan")
    ratio = (sigma / median_spacing) if median_spacing else float("nan")
    if not math.isfinite(ratio):
        verdict = color_text("not yet measurable", "dim")
    elif ratio < 0.5:
        verdict = role_text(f"k likely too stiff (σ/spacing {ratio:.2f})", ROLE_WARN)
    elif ratio > 1.5:
        verdict = role_text(f"k likely too soft (σ/spacing {ratio:.2f})", ROLE_WARN)
    else:
        verdict = role_text(f"σ/spacing {ratio:.2f} {OK}", ROLE_GOOD)
    return panel("sigma", "σ-vs-spacing check", [
        f" k median {k_median:.2f} {ctx.primary_k_units} → σ {sigma:.2f} "
        f"{ctx.primary_cv_units}   spacing {median_spacing:.2f}",
        f" observed median overlap {obs_median:.2f}   target {TARGET_OVERLAP:.2f}",
        " " + verdict,
    ], min_lines=3, want_lines=3, priority=2)


def connectivity_verdict(ctx: DashboardContext) -> tuple[str, int, int]:
    """``(state, largest_component, pairs_measured)`` for the state graph.

    ``state`` is ``"unmeasured"``, ``"partial"``, ``"connected"`` or ``"split"``.

    Why this is not simply "count the components": an unattempted pair carries a
    `nan` rate, and an unlinked pair is indistinguishable from a disconnected one
    by topology alone. On a run's first frames NO pair has been attempted, so a
    naive count finds every window in its own component and would announce
    "1/29 connected" -- a confident claim of total MBAR disconnection on a
    perfectly healthy run, which would also promote the auto view to PHYSICS and
    keep it there until exchanges accumulate.

    So the verdict holds judgement until the evidence exists: no measured pair at
    all is ``unmeasured``; some but not all measured is ``partial`` (report the
    coverage, do not claim a split); only with every pair measured is a split
    asserted.
    """
    total_pairs = max(0, int(ctx.n_windows) - 1)
    measured = sum(1 for r in ctx.acceptance_pairs.values() if math.isfinite(r))
    comps = connected_components(ctx.n_windows, ctx.acceptance_pairs)
    largest = len(comps[0]) if comps else 0
    if measured == 0:
        return "unmeasured", largest, measured
    if measured < total_pairs:
        return "partial", largest, measured
    if largest >= int(ctx.n_windows) and ctx.n_windows > 0:
        return "connected", largest, measured
    return "split", largest, measured


def connectivity_panel(ctx: DashboardContext) -> Panel:
    """MBAR readiness: is the state graph one piece?"""
    state, largest, measured = connectivity_verdict(ctx)
    total_pairs = max(0, int(ctx.n_windows) - 1)
    comps = connected_components(ctx.n_windows, ctx.acceptance_pairs)
    lines = []
    if ctx.secondary_cv_type:
        lines.append(f" cv2 {ctx.secondary_cv_type}   "
                     f"{len(set(round(x, 4) for x in ctx.secondary_centers))} rows")
    if state == "unmeasured":
        lines.append(" graph " + color_text("no exchange attempts yet — connectivity "
                                            "not yet measurable", "dim"))
    elif state == "partial":
        lines.append(" graph " + color_text(
            f"{measured}/{total_pairs} pairs measured — holding judgement", "dim"))
    elif state == "connected":
        lines.append(" graph " + role_text(f"{largest}/{ctx.n_windows} connected {OK}", ROLE_GOOD))
    else:
        isolated = sorted(w for comp in comps[1:] for w in comp)
        shown = ", ".join(f"w{w:02d}" for w in isolated[:8])
        lines.append(" graph " + role_text(
            f"{largest}/{ctx.n_windows} connected", ROLE_BAD)
            + f"   isolated: {shown}"
            + ("" if len(isolated) <= 8 else f" (+{len(isolated) - 8} more)"))
        lines.append(role_text(
            f" → union MBAR drops {ctx.n_windows - largest} states unless a bridge is added",
            ROLE_WARN))
    return panel("connectivity", "CV2 regime + graph connectivity", lines,
                 min_lines=2, want_lines=4, priority=1)


def build(ctx: DashboardContext) -> tuple[Row, ...]:
    return (
        Row(panels=(overlap_panel(ctx), boost_envelope_panel(ctx))),
        Row(panels=(sigma_spacing_panel(ctx),)),
        Row(panels=(connectivity_panel(ctx),)),
    )


__all__ = ["TARGET_OVERLAP", "build", "connected_components", "connectivity_panel",
           "sigma_spacing_panel"]
