"""Content panels for the live dashboard, as pure functions of DashboardContext.

Ported from DistanceLogger's _render_* methods. Two changes from the originals:
they take `ctx` rather than `self`, and they drop their own leading title line
because the title now lives on the Panel (which is what lets the allocator
account for chrome).
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..colors import ROLE_BAD, ROLE_GOOD, ROLE_WARN, color_text, role_text
from ..cv import format_primary_cv_value, format_primary_delta_value
from ..math_helpers import boost_anharmonicity
from ..tui import _join_columns, _mini_bar, _sparkline, render_distance_ascii, replica_marker
from ..tui_screen import Panel
from .context import DashboardContext
from .ranking import BAD, DEAD_OVERLAP, OK, WARN, WindowStatus, restraint_sigma

_K0_SATURATED = 0.999


def panel(
    key: str,
    title: str,
    lines: Iterable[str],
    *,
    min_lines: int,
    want_lines: int,
    priority: int,
    weight: float = 1.0,
    line_statuses: Iterable[str] = (),
) -> Panel:
    return Panel(
        key=key, title=title, lines=tuple(str(x) for x in lines), min_lines=int(min_lines),
        want_lines=int(want_lines), priority=int(priority), weight=float(weight),
        line_statuses=tuple(str(s) for s in line_statuses),
    )


def _cv_fmt_mode(ctx: DashboardContext) -> dict:
    """A minimal args-shaped mapping so `cv.py`'s format helpers read units off
    `ctx` instead of a real argparse namespace.

    `format_primary_cv_value`/`format_primary_delta_value` resolve units via
    `primary_cv_units(args_or_mode)`, which special-cases a dict with a
    `"units"` key -- any other input (a plain object with no `primary_cv`
    attribute, e.g. `DashboardContext` itself) falls through to that
    function's own contact-mode detection and silently defaults to `"A"`.
    `ctx.primary_cv_units` already holds the correct, precomputed unit string,
    so route through the dict branch instead of re-deriving it.
    """
    return {"units": ctx.primary_cv_units}


def overlap_panel(ctx: DashboardContext) -> Panel:
    """Neighbour histogram overlap, worst pair first.

    Ordering is the whole point: the previous version emitted pairs in window
    order and truncated the tail, so a disconnected pair past the visible cut
    was invisible.
    """
    pairs = sorted(ctx.overlap_pairs.items(), key=lambda kv: (kv[1], kv[0]))
    if not pairs:
        return panel("overlap", "neighbour overlap — worst first",
                     [color_text("insufficient samples", "dim")],
                     min_lines=5, want_lines=12, priority=1, weight=1.3)
    lines = [color_text("pair        overlap   target 0.30", "white", bold=True)]
    statuses = [""]                      # the header line carries no severity
    for (a, b), ov in pairs:
        if ov < DEAD_OVERLAP:
            role, label = ROLE_BAD, "DEAD  → MBAR graph breaks" if ov < 0.05 else "BAD"
        elif ov < 0.30:
            role, label = ROLE_WARN, "WARN"
        else:
            role, label = ROLE_GOOD, OK
        lines.append(
            f"  w{a:02d}-w{b:02d}   {ov:6.2f}   {_mini_bar(ov / 0.5, 10)}  "
            + role_text(label, role)
        )
        statuses.append(BAD if ov < DEAD_OVERLAP else (WARN if ov < 0.30 else OK))
    return panel("overlap", "neighbour overlap — worst first", lines,
                 min_lines=5, want_lines=12, priority=1, weight=1.3,
                 line_statuses=statuses)


def live_gamd_envelope(gamd: "Mapping[str, object] | None") -> dict:
    """The GaMD envelope actually in force — not the one first calibrated.

    ``joint_envelope`` records the calibration frozen before production. If the
    run later recalibrated from real epoch-0 sampling (``recalibration_history``,
    which fires at most once), the ``*_after`` values are what the integrator is
    actually running. Verified against a real run: ``joint_envelope`` held
    ``sigmaV 10.7737`` while both the live integrator globals and the
    recalibration record held ``11.0394`` — so reading ``joint_envelope`` alone
    displays a stale envelope for the entire post-epoch-0 run, which is most of it.

    ``sigma0`` is the calibration *target* and is not changed by recalibration, so
    it always comes from ``joint_envelope``.
    """
    payload = dict(gamd or {})
    envelope = payload.get("joint_envelope") or {}
    if not isinstance(envelope, dict) or not envelope:
        return {}
    group_name, group = next(iter(envelope.items()))
    out = dict(group if isinstance(group, dict) else {})
    out["group"] = str(group_name)
    history = payload.get("recalibration_history") or []
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        last = history[-1]
        recal = ((last.get("groups") or {}).get(group_name)) or {}
        for src, dst in (
            ("sigmaV_after_kj_mol", "sigmaV_kj_mol"),
            ("k0_after", "k0"),
            ("threshold_energy_after_kj_mol", "threshold_energy_kj_mol"),
        ):
            if src in recal:
                out[dst] = recal[src]
        if "sigmaV_before_kj_mol" in recal:
            out["sigmaV_before_kj_mol"] = recal["sigmaV_before_kj_mol"]
        out["recalibrated_from_epoch"] = last.get("recalibrated_from_epoch")
    return out


def boost_envelope_panel(ctx: DashboardContext) -> Panel:
    """GaMD envelope: achieved sigma vs calibration target, k0 saturation, shape.

    The sigma/k0 half comes from the sidecar calibration file, the shape half
    from live boost samples. A run with no GaMD says so rather than rendering
    an empty box.
    """
    group = live_gamd_envelope(ctx.sidecar.gamd)
    if not group:
        # The sidecar's GaMD envelope, not a non-empty `boost_history_all`, is
        # the authoritative "is this actually a GaMD run" signal: a plain
        # umbrella run's boost history should always be empty in practice, but
        # nothing here should have to assume that -- the sidecar is written
        # only when a real GaMD calibration ran.
        return panel("boost", "GaMD boost envelope",
                     [color_text("no GaMD boost (plain umbrella run)", "dim")],
                     min_lines=5, want_lines=11, priority=1)

    boosts = [b for b in ctx.boost_history_all if math.isfinite(b)]
    lines: list[str] = []
    if boosts:
        shape = boost_anharmonicity(boosts)
        # `boost_anharmonicity` (moved out of logger.py in Step 0) returns the
        # score under the key "score", not "anharmonicity_score" -- that other
        # key name belongs to the unrelated post-hoc MBAR-analysis anharmonicity
        # computation in gareus/mbar_analysis/pmf.py.
        score = float(shape.get("score", float("nan")))
        role = ROLE_BAD if score > 1.5 else (ROLE_WARN if score > 1.0 else ROLE_GOOD)
        lines.append(
            f"  boost {np.mean(boosts):.2f} ± {np.std(boosts):.2f} kcal/mol   "
            f"skew {shape.get('skew', float('nan')):+.2f}  "
            f"kurt {shape.get('excess_kurtosis', float('nan')):+.2f}"
        )
        lines.append("  anharmonicity " + role_text(
            f"{score:.2f} " + ("HIGH" if score > 1.0 else OK), role))
    if group:
        sigma_v = float(group.get("sigmaV_kj_mol", float("nan")))
        sigma_0 = float(group.get("sigma0_kj_mol", float("nan")))
        k0 = float(group.get("k0", float("nan")))
        pct = (100.0 * sigma_v / sigma_0) if sigma_0 else float("nan")
        lines.append(f"  σΔV {sigma_v:.2f} / σ0 {sigma_0:.2f} kJ  ({pct:.0f}%)")
        if math.isfinite(k0):
            k0_label = "SATURATED at ceiling" if k0 >= _K0_SATURATED else OK
            role = ROLE_BAD if k0 >= _K0_SATURATED else ROLE_GOOD
            lines.append(f"  k0 {k0:.2f}  " + role_text(k0_label, role))
        epoch = group.get("recalibrated_from_epoch")
        if epoch is not None:
            before = float(group.get("sigmaV_before_kj_mol", float("nan")))
            lines.append(
                f"  recalibrated after epoch {epoch}: σΔV {before:.2f} → {sigma_v:.2f} kJ"
            )
    return panel("boost", "GaMD boost envelope", lines,
                 min_lines=5, want_lines=11, priority=1)


def cv_map_panel(ctx: DashboardContext, bar_width: int) -> Panel:
    """Per-window CV distributions: the densest panel on the dashboard."""
    if ctx.ascii_mode == "none":
        return panel("cv_map", "per-window CV distributions",
                     [color_text("disabled (--distance-ascii-mode none)", "dim")],
                     min_lines=1, want_lines=2, priority=2, weight=2.4)
    block = render_distance_ascii(
        [dict(r) for r in ctx.rows], phase=ctx.phase, step=ctx.step,
        total_steps=ctx.total_steps, width=int(bar_width),
        max_replicas=ctx.ascii_max_replicas, mode=ctx.ascii_mode,
        history_by_replica={k: list(v) for k, v in ctx.cv_history_by_replica.items()},
        history_by_window={k: list(v) for k, v in ctx.cv_history_by_window.items()},
        histogram_source="window", primary_label=ctx.primary_cv_label,
        primary_units=ctx.primary_cv_units, primary_k_unit_label=ctx.primary_k_units,
    )
    lines = block.splitlines()
    if len(lines) > 4:
        lines = lines[3:]                 # drop its own border/title/separator
    return panel("cv_map", "per-window CV distributions",
                 lines or [color_text("no samples yet", "dim")],
                 min_lines=6, want_lines=18, priority=2, weight=2.4)


def pe_map_panel(ctx: DashboardContext, bar_width: int) -> Panel:
    """Per-replica potential-energy histograms, current value marked."""
    width = max(10, int(bar_width or 24))
    all_vals: list[float] = []
    for vals in ctx.pe_history_by_replica.values():
        all_vals.extend([float(v) for v in vals if math.isfinite(float(v))])
    if not all_vals:
        return panel("pe_map", "potential energy histograms",
                     [color_text("PE unavailable/not sampled yet", "dim")],
                     min_lines=4, want_lines=10, priority=3, weight=0.9)
    lo = float(np.nanpercentile(np.asarray(all_vals, dtype=float), 2.0))
    hi = float(np.nanpercentile(np.asarray(all_vals, dtype=float), 98.0))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(all_vals))
        hi = float(np.nanmax(all_vals))
    if hi <= lo:
        hi = lo + 1.0
    lines = [color_text(
        f"  PE range {lo:.0f} to {hi:.0f} kJ/mol; shaded history, ● current", "dim")]
    max_reps = max(1, min(len(ctx.rows), int(ctx.ascii_max_replicas)))
    for r in ctx.rows[:max_reps]:
        rep = int(r.get("replica", 0))
        try:
            pe = float(r.get("potential_kj_mol", "nan"))
        except Exception:
            pe = float("nan")
        hist = ctx.pe_history_by_replica.get(rep, ())
        if hist:
            counts, _edges = np.histogram(np.asarray(hist, dtype=float), bins=width, range=(lo, hi))
            mx = max(1, int(np.nanmax(counts)))
            chars = [" ", "·", "░", "▒", "▓", "█"]
            bar = "".join(chars[min(len(chars) - 1, int(round(c / mx * (len(chars) - 1))))] for c in counts)
        else:
            bar = "·" * width
        if math.isfinite(pe):
            pos = int(round((pe - lo) / (hi - lo) * (width - 1)))
            pos = max(0, min(width - 1, pos))
            marker = replica_marker("●", rep, bg256=236, bold=True)
            bar = bar[:pos] + marker + bar[pos + 1:]
            pe_txt = f"{pe:9.0f}"
        else:
            pe_txt = "      n/a"
        lines.append(f"  {replica_marker(f'r{rep:02d}', rep)} {pe_txt} kJ |{bar}|")
    if len(ctx.rows) > max_reps:
        lines.append(color_text(f"  … {len(ctx.rows) - max_reps} more replicas", "dim"))
    return panel("pe_map", "potential energy histograms", lines,
                 min_lines=4, want_lines=10, priority=3, weight=0.9)


def exchange_panel(ctx: DashboardContext) -> Panel:
    """Exchange / window-walk diagnostics: overall rate, then per-pair detail.

    Per-pair acceptance for display is read from ``ctx.acceptance_pairs``
    (already resolved against the payload's nested ``"pairs"`` key, ``nan`` for
    a never-attempted pair) rather than re-deriving accepted/attempts here.
    ``ctx.exchange_stats`` is only used for the run-level totals/mode/gibbs/jump
    counters that ``acceptance_pairs`` does not carry.
    """
    stats = ctx.exchange_stats if isinstance(ctx.exchange_stats, dict) else {}
    mode = str(stats.get("mode", "neighbor"))
    attempts = int(stats.get("attempts", 0) or 0)
    accepted = int(stats.get("accepted", 0) or 0)
    if attempts <= 0 and int(stats.get("gibbs_choices", 0) or 0) <= 0:
        return panel("exchange", "exchange / window-walk diagnostics",
                     [f"  mode {mode}; no exchange attempts yet"],
                     min_lines=4, want_lines=10, priority=2, weight=1.15)

    lines: list[str] = []
    frac_total = accepted / max(1, attempts)
    role = ROLE_GOOD if frac_total >= 0.20 else (ROLE_WARN if frac_total >= 0.08 else ROLE_BAD)
    lines.append(
        f"  mode {mode:<14} total {100*frac_total:5.1f}% {_mini_bar(frac_total, 10)} "
        f"{accepted}/{attempts} " + role_text(OK if frac_total >= 0.08 else "LOW", role)
    )
    if mode == "gibbs-walk" or int(stats.get("gibbs_choices", 0) or 0) > 0:
        choices = int(stats.get("gibbs_choices", 0) or 0)
        moves = int(stats.get("gibbs_moves", 0) or 0)
        stays = int(stats.get("gibbs_stays", 0) or 0)
        move_frac = moves / max(1, choices)
        lines.append(
            f"  Gibbs choices: moves {moves}/{choices} ({100*move_frac:4.1f}%), stays {stays}; "
            "p shown in exchanges.csv is heat-bath choice prob"
        )

    jumps = stats.get("jump_bins", {}) or {}
    if jumps:
        lines.append(color_text("  by window jump distance", "white", bold=True))

        def _jump_sort_key(item):
            key, _ = item
            try:
                return int(str(key).replace("dw", ""))
            except Exception:
                return 999

        for key, st in sorted(jumps.items(), key=_jump_sort_key)[:6]:
            att = int(st.get("attempts", 0) or 0)
            acc = int(st.get("accepted", 0) or 0)
            if att <= 0:
                continue
            frac = acc / max(1, att)
            role = ROLE_GOOD if frac >= 0.20 else (ROLE_WARN if frac >= 0.08 else ROLE_BAD)
            label = str(key).replace("dw", "Δw=")
            lines.append(
                f"  {label:<6} {100*frac:5.1f}% {_mini_bar(frac, 10)} {acc}/{att} "
                + role_text("LOW" if frac < 0.08 else OK, role)
            )

    raw_pairs = stats.get("pairs", {}) or {}
    if raw_pairs:
        if mode == "neighbor":
            for a in range(max(0, ctx.n_windows - 1)):
                st = raw_pairs.get(f"{a}-{a+1}", {})
                att = int(st.get("attempts", 0) or 0)
                acc = int(st.get("accepted", 0) or 0)
                rate = ctx.acceptance_pairs.get((a, a + 1), float("nan"))
                if not math.isfinite(rate):
                    lines.append(f"  w{a:02d}-w{a+1:02d}   --   " + color_text("not tried", "dim"))
                    continue
                role = ROLE_GOOD if rate >= 0.20 else (ROLE_WARN if rate >= 0.08 else ROLE_BAD)
                lines.append(
                    f"  w{a:02d}-w{a+1:02d} {100*rate:5.1f}% {_mini_bar(rate, 10)} {acc}/{att} "
                    + role_text(BAD if rate < 0.08 else OK, role)
                )
        else:
            lines.append(color_text("  busiest specific window pairs", "white", bold=True))
            items = sorted(raw_pairs.items(), key=lambda kv: int(kv[1].get("attempts", 0) or 0), reverse=True)
            for key, st in items[:6]:
                att = int(st.get("attempts", 0) or 0)
                if att <= 0:
                    continue
                acc = int(st.get("accepted", 0) or 0)
                try:
                    a_str, b_str = str(key).split("-")[:2]
                    rate = ctx.acceptance_pairs.get((int(a_str), int(b_str)), float("nan"))
                except Exception:
                    rate = float("nan")
                if not math.isfinite(rate):
                    continue
                role = ROLE_GOOD if rate >= 0.20 else (ROLE_WARN if rate >= 0.08 else ROLE_BAD)
                lines.append(
                    f"  w{key:<5} {100*rate:5.1f}% {_mini_bar(rate, 10)} {acc}/{att} "
                    + role_text("LOW" if rate < 0.08 else OK, role)
                )
    return panel("exchange", "exchange / window-walk diagnostics", lines,
                 min_lines=4, want_lines=10, priority=2, weight=1.15)


def pull_panel(ctx: DashboardContext) -> Panel:
    """Umbrella restoring-force gauge, one fixed-height line per window.

    Positive pull extends the CV, negative compacts it. ``primary_cv_is_contacts``
    is not a field on ``ctx``, so contact-vs-distance mode is read off
    ``ctx.primary_cv_label`` ("nonlocal contact ..." vs "terminal distance")
    instead -- the one primary-CV field this needs that ``ctx`` does carry.
    """
    by_w: dict[int, list[float]] = {}
    for r in ctx.rows:
        try:
            by_w.setdefault(int(r["window"]), []).append(float(r.get("umbrella_pull_kcal_mol_A", 0.0)))
        except Exception:
            continue

    finite_vals = [abs(float(v)) for vals in by_w.values() for v in vals if math.isfinite(float(v))]
    contact_mode = "contact" in ctx.primary_cv_label.lower()
    pull_units = "kcal/mol/CV" if contact_mode else "kcal/mol/A"
    left_label = "lower" if contact_mode else "compact"
    right_label = "higher" if contact_mode else "extend"
    scale = max(0.5, float(np.nanpercentile(finite_vals, 90)) if finite_vals else 1.0)
    half = 7
    empty_l = " " * half
    empty_r = " " * half
    header = f"  {'win':>3} {left_label:>{half}}│{right_label:<{half}}  pull {pull_units}"
    lines = [color_text(header, "white", bold=True)]

    for w in range(int(ctx.n_windows)):
        vals = [float(v) for v in by_w.get(w, []) if math.isfinite(float(v))]
        if not vals:
            lines.append(color_text(f"  w{w:02d} {empty_l}│{empty_r}     no sample", "dim"))
            continue

        p = float(np.nanmean(vals))
        frac = min(1.0, abs(p) / scale) if scale > 0 else 0.0
        n = max(1 if abs(p) > 0.02 else 0, int(round(frac * half)))
        if abs(p) < 0.02:
            left = " " * half
            right = " " * half
            label = "neutral"
        elif p < 0.0:
            left_raw = ("<" * n).rjust(half)
            left = role_text(left_raw, ROLE_WARN if abs(p) < 2.0 else ROLE_BAD)
            right = " " * half
            label = left_label
        else:
            left = " " * half
            right_raw = (">" * n).ljust(half)
            right = role_text(right_raw, ROLE_WARN if abs(p) < 2.0 else ROLE_BAD)
            label = right_label

        if abs(p) < 0.5:
            role = ROLE_GOOD
        elif abs(p) < 2.0:
            role = ROLE_WARN
        else:
            role = ROLE_BAD
        lines.append(f"  w{w:02d} {left}│{right} {p:+7.3f} " + role_text(label, role))

    lines.append(color_text(f"  scale: full bar ≈ {scale:.2f} {pull_units}; sign from k(center-CV)", "dim"))
    return panel("pull", "umbrella pull field", lines, min_lines=4, want_lines=10, priority=3, weight=1.0)


def replica_table_panel(ctx: DashboardContext, ncols: int = 1) -> Panel:
    """All-replica table: health, CV/delta, PE, boost, sparkline, window trail.

    Phase 1 drops the original method's optional 2D CV2 column: it needed
    per-replica secondary-CV history, which ``ctx`` only carries per-window
    (``secondary_history_by_window``), not per-replica. Round-trip/span comes
    straight from ``ctx.roundtrip_state`` (added to ``DashboardContext``
    specifically for this), the same lifetime `min`/`max`/`roundtrips` counter
    ``self.roundtrip_state`` was -- not an approximation from the bounded
    (200-entry) ``window_trace_by_replica``, which would silently undercount a
    long-running replica's real lifetime total.
    """
    if not ctx.rows:
        return panel("replicas", "replica table",
                     [color_text("no replica data", "dim")],
                     min_lines=4, want_lines=16, priority=2, weight=1.5)
    mode = _cv_fmt_mode(ctx)
    max_span = max(0, ctx.n_windows - 1)
    rep_lines: list[str] = []
    for r in sorted(ctx.rows, key=lambda x: int(x.get("replica", 0))):
        rep = int(r.get("replica", 0))
        win = int(r.get("window", 0))
        cv = float(r.get("cv_A", float("nan")))
        center = float(r.get("center_A", float("nan")))
        delta = cv - center if math.isfinite(cv) and math.isfinite(center) else float("nan")
        try:
            pe = float(r.get("potential_kj_mol", float("nan")))
        except Exception:
            pe = float("nan")
        try:
            b_kcal = float(r.get("gamd_boost_total_kcal_mol", float("nan")))
        except Exception:
            b_kcal = float("nan")

        h = ctx.cv_history_by_replica.get(rep, ())
        spark = _sparkline(list(h)[-20:], 10) if h else " " * 10
        recent10 = list(h)[-10:]
        recent_span = (max(recent10) - min(recent10)) if len(recent10) >= 2 else float("nan")
        cv_stuck = len(recent10) >= 10 and math.isfinite(recent_span) and recent_span < 0.05

        tr = list(ctx.window_trace_by_replica.get(rep, ()))
        rt = ctx.roundtrip_state.get(rep, {})
        span_val = int(rt.get("max", 0)) - int(rt.get("min", 0))
        trips = int(rt.get("roundtrips", 0))
        win_stuck = len(set(tr[-5:])) <= 1 and len(tr) >= 5
        trail = "→".join(f"w{x:02d}" for x in tr[-4:])

        if not math.isfinite(pe) and not math.isfinite(cv):
            status, role = BAD, ROLE_BAD
        elif win_stuck or cv_stuck:
            status, role = "STUCK", ROLE_WARN
        else:
            status, role = OK, ROLE_GOOD

        cv_str = (format_primary_cv_value(cv, mode, precision=3).rjust(9) if math.isfinite(cv) else "      n/a")
        delta_str = ("Δ" + format_primary_delta_value(delta, mode, precision=3).rjust(7)) if math.isfinite(delta) else "Δ   n/a"
        pe_str = f"{pe:8.1f}kJ" if math.isfinite(pe) else "     n/akJ"
        b_str = f"b{b_kcal:5.2f}kc" if math.isfinite(b_kcal) else "b  n/akc"
        status_txt = role_text(status, role)
        cov_pct = int(100 * span_val / max_span) if max_span > 0 else 0
        rep_lines.append(
            f"  {replica_marker(f'r{rep:02d}', rep)} w{win:02d}  {cv_str}  {delta_str}  {pe_str}  {b_str}"
            f"  {spark}  {trail:<12}  {cov_pct:3d}%  {trips:2d}t  {status_txt}"
        )

    if int(ncols) > 1 and len(rep_lines) > 1:
        per_col = math.ceil(len(rep_lines) / int(ncols))
        columns = [rep_lines[c * per_col: (c + 1) * per_col] for c in range(int(ncols))]
        rep_lines = list(_join_columns(columns, gap=3))

    return panel("replicas", "replica table", rep_lines,
                 min_lines=4, want_lines=16, priority=2, weight=1.5)


def window_table_panel(ctx: DashboardContext, statuses: Sequence[WindowStatus]) -> Panel:
    """One row per window, worst first, with a severity-composition tail."""
    def _num(value: float, width: int, places: int) -> str:
        """Render a measurement, or an em dash if there is nothing to render.

        A literal `nan` in a table cell is this design's own rule broken in the
        display layer: `nan` means "not measured", and printing it puts a token
        in a numeric column that reads as data. Boundary windows genuinely have
        no left or right neighbour, so their acceptance is absent rather than bad.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "—".rjust(width)
        return f"{v:{width}.{places}f}" if math.isfinite(v) else "—".rjust(width)

    # A 1D run has no secondary CV at all, so the column is dropped rather than
    # filled with placeholders down its whole length.
    show_cv2 = bool(ctx.secondary_centers)
    cv2_head = "  cv2 ctr " if show_cv2 else ""
    header = color_text(
        f"win   cv1 ctr {cv2_head}   accL   accR   |Δ|max   status", "white", bold=True)
    lines = [header]
    line_statuses = [""]                 # the header line carries no severity
    for s in statuses:
        w = s.window
        centre = ctx.centers_a[w] if w < len(ctx.centers_a) else float("nan")
        role = {BAD: ROLE_BAD, WARN: ROLE_WARN}.get(s.status, ROLE_GOOD)
        cv2_cell = ""
        if show_cv2:
            sec = ctx.secondary_centers[w] if w < len(ctx.secondary_centers) else float("nan")
            cv2_cell = "  " + _num(sec, 8, 2) + " "
        lines.append(
            f"  w{w:02d}  {_num(centre, 8, 2)} {cv2_cell}  "
            f"{_num(ctx.acceptance_pairs.get((w - 1, w), float('nan')), 5, 2)}  "
            f"{_num(ctx.acceptance_pairs.get((w, w + 1), float('nan')), 5, 2)}   "
            f"{_num(abs(float(ctx.deltas.get(w, float('nan')))), 6, 2)}   "
            + role_text(s.status, role)
            + ("  " + ", ".join(s.reasons) if s.reasons else "")
        )
        line_statuses.append(s.status)
    return panel("windows", "windows — worst first", lines, min_lines=4, want_lines=16,
                 priority=1, line_statuses=line_statuses)


def window_detail_panel(ctx: DashboardContext, window: int) -> Panel:
    """Everything known about one window, for the auto- or key-selected row."""
    w = int(window)
    centre = ctx.centers_a[w] if w < len(ctx.centers_a) else float("nan")
    k = ctx.k_list[w] if w < len(ctx.k_list) else float("nan")
    sigma = restraint_sigma(k, ctx.temperature_k)
    samples = ctx.cv_history_by_window.get(w, ())
    mean_cv = float(np.mean(samples)) if samples else float("nan")
    lines = [
        f"  restraint   cv1 {centre:.2f} {ctx.primary_cv_units}  "
        f"k {k:.2f} {ctx.primary_k_units}   σ {sigma:.2f}",
        f"  sampled     n {len(samples)} in history   mean {mean_cv:.2f}   "
        f"Δ {mean_cv - centre:+.2f}",
    ]
    for (a, b), rate in sorted(ctx.acceptance_pairs.items()):
        if w in (a, b):
            other = b if a == w else a
            label = "DEAD" if math.isfinite(rate) and rate < 0.02 else OK
            lines.append(f"  exchange    w{other:02d} {rate:5.2f}  " + label)
    # `window_trace_by_replica` is keyed by REPLICA (gareus/logger.py:520), so it
    # must be indexed by whichever replica currently occupies this window -- not
    # by the window index. Every fixture built replica == window, which is why no
    # test could tell the two apart; on a real run after the first swap, indexing
    # by window shows a different replica's trail.
    replica = next((int(r["replica"]) for r in ctx.rows
                    if int(r.get("window", -1)) == w and r.get("replica") is not None), None)
    trace = ctx.window_trace_by_replica.get(replica, ()) if replica is not None else ()
    if trace:
        lines.append(f"  occupancy   replica r{replica:02d}, windows visited: "
                     + ", ".join(f"w{t:02d}" for t in trace[-6:]))
    return panel(f"detail-w{w:02d}", f"w{w:02d} detail", lines,
                 min_lines=6, want_lines=11, priority=1)


__all__ = [
    "boost_envelope_panel",
    "cv_map_panel",
    "exchange_panel",
    "live_gamd_envelope",
    "overlap_panel",
    "panel",
    "pe_map_panel",
    "pull_panel",
    "replica_table_panel",
    "window_detail_panel",
    "window_table_panel",
]
