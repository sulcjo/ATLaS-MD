"""The always-visible header: the real lines that answer all three operator questions.

Nine line-kinds exist (eight for a 1D run, which has no CV2 line): identity
and verdict, run progress, pool budget, CV1 coverage, CV2 coverage (2D only),
window sample strip, neighbour-exchange strip, GaMD boost envelope, and top
alerts. `FULL_SPINE_LINES = 10` is the budget `render_screen` reserves for
the spine, not a line count this module pads out to -- the design mockup's
tenth row is the view-tab/clock divider that `render_screen` renders as its
own footer. `spine_lines` returns only real content; whatever's left of the
reservation goes back to the view body instead of sitting here as blank rows.

Every line is `label + elastic middle + fixed right summary`; the middle is sized
from the measured width of the fixed parts -- bracket characters included, never
guessed -- so a normal-width terminal never has to fall back on the final
`_ansi_truncate` safety net to fit. The window strip and exchange strip share an
x-axis on purpose: the exchange strip's first cell starts one column to
the right of the window strip's first cell, so pair *i* (between windows *i* and
*i+1*) reads as a seam between two window columns rather than sitting directly
under either window's own glyph.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..colors import ROLE_BAD, ROLE_GOOD, ROLE_WARN, color_text, role_text
from ..tui import _ansi_truncate, _coverage_bar, format_duration, make_progress_bar
from .context import DashboardContext
from .panels import live_gamd_envelope
from .ranking import BAD, DEAD_ACCEPTANCE, OK, WARN, WindowStatus, rank_windows

FULL_SPINE_LINES = 10
COMPACT_SPINE_LINES = 5

_DENSITY_GLYPHS = " ▁▂▃▄▅▆▇█"
_ASCII_GLYPHS = " .:-=+*#%"
_K0_SATURATED = 0.999


def bucket_strip(
    values: Sequence[float], statuses: Sequence[str], cells: int, glyphs: str = "unicode"
) -> str:
    """One glyph per value, or per aggregated bucket when values outnumber cells.

    Bucket height is the mean; bucket status is the worst member. Real 364-window
    runs cannot fit one cell each, and dropping the tail would hide exactly the
    windows worth seeing.
    """
    vals = [float(v) for v in values]
    if not vals:
        return ""
    n_cells = max(1, min(int(cells), len(vals)))
    ramp = _DENSITY_GLYPHS if glyphs != "ascii" else _ASCII_GLYPHS
    edges = np.linspace(0, len(vals), n_cells + 1).astype(int)
    finite = [v for v in vals if math.isfinite(v)]
    hi = max(finite) if finite else 1.0
    lo = min(finite) if finite else 0.0
    span = (hi - lo) or 1.0
    out: list[str] = []
    for i in range(n_cells):
        chunk = vals[edges[i]:edges[i + 1]] or [lo]
        chunk_status = [str(statuses[j]) for j in range(edges[i], edges[i + 1])
                        if j < len(statuses)]
        level = int(round((float(np.mean(chunk)) - lo) / span * (len(ramp) - 1)))
        glyph = ramp[max(0, min(len(ramp) - 1, level))]
        if BAD in chunk_status:
            out.append(role_text("X" if glyphs == "ascii" else glyph, ROLE_BAD))
        elif WARN in chunk_status:
            out.append(role_text("!" if glyphs == "ascii" else glyph, ROLE_WARN))
        else:
            out.append(glyph)
    return "".join(out)


def _ranked_windows(ctx: DashboardContext) -> tuple[WindowStatus, ...]:
    return rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k,
    )


def _statuses_by_window(ranked: Sequence[WindowStatus], n_windows: int) -> list[str]:
    by_window = {s.window: s.status for s in ranked}
    return [by_window.get(w, OK) for w in range(n_windows)]


def _verdict(ctx: DashboardContext) -> str:
    """Top-level health, read from the same decision state the run itself uses.

    `_dashboard_decision_state` (gareus/logger.py) returns
    `{"health", "issues", "reasons", "actions"}` -- there is no `"status"` or
    `"issue_count"` key. Reading those instead would always miss, silently
    rendering every run "OK" regardless of what `ctx.decision` actually holds.
    """
    issues = list(ctx.decision.get("issues", []) or [])
    health = str(ctx.decision.get("health", "OK") or "OK").upper()
    if health == "BAD":
        return role_text(f"✗ BAD {len(issues)}", ROLE_BAD)
    if health == "WATCH" or issues:
        return role_text(f"⚠ CAUTION {len(issues)}", ROLE_WARN)
    return role_text("✓ OK", ROLE_GOOD)


def _ns_per_day(ctx: DashboardContext) -> float:
    if ctx.timestep_fs <= 0.0 or ctx.elapsed_s <= 0.0:
        return float("nan")
    sim_ns = ctx.display_step * ctx.timestep_fs / 1.0e6
    return sim_ns / ctx.elapsed_s * 86400.0


def _pool_line(ctx: DashboardContext, width: int) -> str:
    pool = ctx.sidecar.pool or {}
    total = float(pool.get("total_ns", 0.0) or 0.0)
    used = float(pool.get("used_ns", 0.0) or 0.0)
    if total <= 0.0:
        return "pool  " + color_text("unavailable (no adaptive_runtime_pool.json yet)", "dim")
    ns_day = _ns_per_day(ctx)
    # `prefix`/`suffix` include the bar's own brackets, so the width budget is
    # measured against exactly what gets printed -- not just the label/summary
    # text either side of them (that omission previously left every populated
    # pool line 2 columns over budget, rescued only by truncation at render time).
    prefix = "pool  ["
    suffix = f"]  {used:.0f}/{total:.0f} ns   perf {ns_day:.0f} ns/d/rep"
    bar_w = max(12, width - len(prefix) - len(suffix))
    return prefix + make_progress_bar(used / total, bar_w) + suffix


def _run_line(ctx: DashboardContext, width: int) -> str:
    total = ctx.display_total_steps or 0
    frac = (ctx.display_step / total) if total else 0.0
    prefix = "run   ["
    suffix = (f"]  {100.0 * frac:5.1f}%   {ctx.display_step / 1e6:.2f}/{total / 1e6:.2f} Msteps"
              f"   wall {format_duration(ctx.elapsed_s)}  eta {format_duration(ctx.eta_s)}")
    bar_w = max(12, width - len(prefix) - len(suffix))
    return prefix + make_progress_bar(frac, bar_w) + suffix


def _gamd_line(ctx: DashboardContext) -> str:
    # Same helper the boost panel uses: never re-read `joint_envelope` directly,
    # or the spine shows the pre-recalibration envelope for most of the run.
    group = live_gamd_envelope(ctx.sidecar.gamd)
    if not group:
        return "gamd  " + color_text("no GaMD boost (plain umbrella run)", "dim")
    sigma_v = float(group.get("sigmaV_kj_mol", float("nan")))
    sigma_0 = float(group.get("sigma0_kj_mol", float("nan")))
    k0 = float(group.get("k0", float("nan")))
    pct = (100.0 * sigma_v / sigma_0) if sigma_0 else float("nan")
    k0_txt = role_text("SATURATED", ROLE_BAD) if k0 >= _K0_SATURATED else role_text(OK, ROLE_GOOD)
    return (f"gamd  σΔV {sigma_v:.2f} / σ0 {sigma_0:.2f} kJ ({pct:.0f}%)   "
            f"k0 {k0:.2f} {k0_txt}")


def _alert_line(ranked: Sequence[WindowStatus]) -> str:
    # Each entry leads with its own status word (not just the reason text) so
    # BAD/WARN survive here as literal text too, the same as SATURATED/ok do
    # on the gamd line -- status is never colour-only anywhere on the spine.
    alerts = [f"w{s.window:02d} {s.status} " + ", ".join(s.reasons) for s in ranked
              if s.status in {BAD, WARN}][:2]
    if not alerts:
        return "alert " + color_text("none", "dim")
    return "alert " + "   |   ".join(alerts)


def spine_lines(ctx: DashboardContext, lines_budget: int) -> tuple[str, ...]:
    """Render the spine's real content lines (9 full/2D, 8 full/1D, 5 compact).

    `lines_budget` selects a tier (>= `FULL_SPINE_LINES` for the full tier,
    else the 5-line compact tier) -- it is not a target length the return
    value is padded to reach.
    """
    width = max(40, ctx.term_w - 2)
    ranked = _ranked_windows(ctx)
    statuses = _statuses_by_window(ranked, ctx.n_windows)
    counts = [float(len(ctx.cv_history_by_window.get(w, ()))) for w in range(ctx.n_windows)]
    epoch = ""
    if ctx.epoch_index is not None:
        epoch = f"ep {ctx.epoch_index}/{ctx.epoch_total}  "
    identity = (f"GaREUS  {ctx.run_label}   {ctx.phase}  {epoch}{ctx.segment_name}   "
                f"{ctx.n_windows} win  {ctx.topology_label}")
    if ctx.secondary_cv_type:
        identity += f"  cv2 {ctx.secondary_cv_type}"
    identity += "      " + _verdict(ctx)

    cells = max(8, width - 60)
    win_strip = bucket_strip(counts, statuses, cells=cells, glyphs=ctx.glyphs)
    accept = [ctx.acceptance_pairs.get((w, w + 1), float("nan")) for w in range(ctx.n_windows - 1)]
    pair_status = [BAD if (math.isfinite(a) and a < DEAD_ACCEPTANCE) else OK for a in accept]
    exch_strip = bucket_strip(accept, pair_status, cells=cells, glyphs=ctx.glyphs)
    finite_acc = [a for a in accept if math.isfinite(a)]

    lines = [identity, _run_line(ctx, width), _pool_line(ctx, width)]
    cv_vals = [v for w in range(ctx.n_windows) for v in ctx.cv_history_by_window.get(w, ())]
    if cv_vals and ctx.centers_a:
        lo, hi = min(ctx.centers_a), max(ctx.centers_a)
        lines.append(f"cv1 {lo:6.1f} |{_coverage_bar(cv_vals, lo, hi, max(12, width - 46))}| "
                     f"{hi:6.1f} {ctx.primary_cv_units}   span {hi - lo:.1f}")
    else:
        lines.append("cv1   " + color_text("no samples yet", "dim"))
    if ctx.is_2d and ctx.secondary_centers:
        sec_vals = [v for w in range(ctx.n_windows)
                    for v in ctx.secondary_history_by_window.get(w, ())]
        s_lo, s_hi = min(ctx.secondary_centers), max(ctx.secondary_centers)
        bar = _coverage_bar(sec_vals, s_lo, s_hi, max(12, width - 46)) if sec_vals else ""
        lines.append(f"cv2 {s_lo:+6.1f} |{bar}| {s_hi:+6.1f}   "
                     f"{len(set(round(x, 4) for x in ctx.secondary_centers))} rows")
    # "win   |" is 7 columns before the strip starts; "exch    " is 8 -- a
    # deliberate one-column shift so pair i's cell sits under the seam between
    # window i's and window i+1's cells rather than directly under either one.
    lines.append(f"win   |{win_strip}|  samples {min(counts) if counts else 0:.0f}-"
                 f"{max(counts) if counts else 0:.0f}/win")
    lines.append(f"exch    {exch_strip}   accept "
                 + (f"{np.mean(finite_acc):.2f} mean, {min(finite_acc):.2f} min"
                    if finite_acc else "no attempts yet"))
    lines.append(_gamd_line(ctx))
    lines.append(_alert_line(ranked))

    budget = max(1, int(lines_budget))
    if budget >= FULL_SPINE_LINES:
        # No blank padding. There are nine real line-kinds (eight for a 1D run,
        # which has no CV2 line) -- the design mockup's tenth line is the
        # view-tab/clock divider that `render_screen` renders as its own
        # footer, not a spine line. `FULL_SPINE_LINES` is still the budget
        # `render_screen` reserves for the spine (frame_tiers' contract is
        # unchanged); the spine simply doesn't need to fill all of it, and
        # `render_screen` hands the unused reservation to the view body
        # instead of the spine wasting it as blank rows.
        chosen = lines[:FULL_SPINE_LINES]
    else:
        # Compact tier: identity, run, pool, window strip, top alert -- all
        # five are always real content, so no padding is needed here either.
        chosen = [lines[0], lines[1], lines[2],
                  next((l for l in lines if l.startswith("win   ")), ""),
                  lines[-1]][:budget]
    return tuple(_ansi_truncate(l, width) for l in chosen)


__all__ = ["COMPACT_SPINE_LINES", "FULL_SPINE_LINES", "bucket_strip", "spine_lines"]
