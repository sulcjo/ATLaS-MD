"""
Terminal user‑interface rendering utilities.

This module contains functions for rendering textual dashboards and
progress information in a terminal.  Functions are extracted from the
monolithic ``gareus_peptide.py`` script to improve reusability and
testability.  They handle ANSI coloring, progress bars, histogram
rendering and panel layouts used by the GaREUS dashboard.
"""

from __future__ import annotations

import math
import re
import shutil
import sys
from typing import Optional, Iterable, List

import numpy as np

from .colors import ASCII_COLOR_ENABLED, color_text, style_text

__all__ = [
    "tui_clear_enabled",
    "clear_tui_screen",
    "write_tui_frame",
    "restore_tui_cursor",
    "strip_ansi",
    "strip_ansi_len",
    "make_progress_bar",
    "format_duration",
    "render_distance_ascii",
    "dashboard_body_budget",
    # Internal helpers that may be useful elsewhere
    "replica_marker",
    "replica_fg256",
]


def tui_clear_enabled(args) -> bool:
    """Return whether full‑frame TUI redraws should clear the visible terminal.

    Dashboard‑style redraws must not depend on ``sys.stdout.isatty()``:
    MPI launchers, ``tee``, SLURM wrappers and some IDE terminals often
    report non‑TTY even when ANSI cursor controls still work.  Clearing
    the screen by default for full‑frame TUI modes avoids leaving
    artefacts behind unless the user explicitly requests ``--tui-clear-mode never``.
    """
    mode = str(getattr(args, "tui_clear_mode", "always") or "always").lower()
    if mode == "never":
        return False
    tui_mode = str(getattr(args, "tui_mode", "dashboard") or "dashboard").lower()
    if tui_mode not in {"dashboard", "interactive"}:
        return False
    return True


def clear_tui_screen(args) -> None:
    """Move home and clear the visible TUI area without touching scrollback."""
    global _TUI_LAST_FRAME_LINES
    if not tui_clear_enabled(args):
        return
    try:
        # Hide cursor, move to home, clear from cursor down.  Using ``J``
        # rather than printing a newline avoids the subtle one‑line
        # scroll/jump some terminals produce when a full‑screen frame ends
        # on the last row.
        sys.stdout.write("\033[?25l\033[H\033[J")
        sys.stdout.flush()
        _TUI_LAST_FRAME_LINES = 0
    except Exception:
        pass


def _safe_tui_frame_text(text: str, max_width: Optional[int] = None, max_height: Optional[int] = None) -> str:
    """Prepare a terminal frame that will not wrap and scroll by one row.

    The previous dashboard sometimes jumped because a long ANSI‑decorated
    line wrapped at the terminal edge.  This function keeps every
    physical line below the terminal width.  If truncation is needed,
    it strips ANSI for that line; ugly but stable beats
    beautiful‑but‑sentient terminal confetti.
    """
    term_size = shutil.get_terminal_size((160, 40))
    term_w = int(max_width or term_size.columns or 160)
    term_h = int(term_size.lines or 40)
    usable = max(20, term_w - 2)
    usable_h = max(8, term_h - 1)
    if max_height is not None and int(max_height) > 0:
        usable_h = max(8, min(usable_h, int(max_height)))
    out_lines: List[str] = []
    raw_lines = str(text).rstrip("\n").splitlines()
    truncated = len(raw_lines) > usable_h
    for line in raw_lines[:usable_h]:
        if strip_ansi_len(line) <= usable:
            out_lines.append(line)
            continue
        raw = strip_ansi(line)
        out_lines.append(raw[: max(8, usable - 1)] + "…")
    if truncated and out_lines:
        msg = f"… dashboard truncated to terminal height ({usable_h} lines); enlarge terminal or reduce shown panels"
        out_lines[-1] = color_text(msg[:usable], "dim")
    return "\n".join(out_lines)


_TUI_LAST_FRAME_LINES = 0


def _write_stdout_best_effort(payload: str) -> bool:
    """Best‑effort terminal write used only for nonessential TUI repaints."""
    try:
        sys.stdout.write(payload)
        sys.stdout.flush()
        return True
    except (BlockingIOError, BrokenPipeError):
        return False
    except OSError:
        return False
    except Exception:
        return False


def write_tui_frame(text: str, args) -> None:
    """Write a full‑frame TUI repaint without appending frames.

    Dashboard and interactive modes are full‑frame modes.  They should
    never degrade into newline‑style progress output during setup phases
    (NVT, NPT, adaptive prescan, starting‑structure pulling, replica
    construction) because that turns the TUI into a log waterfall.
    Use an absolute home+clear repaint on every update; if stdout is
    temporarily nonblocking, drop this repaint and try again on the
    next update rather than printing a fallback line.
    """
    global _TUI_LAST_FRAME_LINES

    frame = _safe_tui_frame_text(text, max_height=int(getattr(args, "dashboard_max_height", 0) or 0))

    if tui_clear_enabled(args):
        payload = "\033[?25l\033[H\033[2J\033[3J" + frame + "\033[0m\033[J"
        if _write_stdout_best_effort(payload):
            _TUI_LAST_FRAME_LINES = len(frame.splitlines() or [""])
        return

    # Explicit log‑style mode: append by design.
    if _write_stdout_best_effort(frame + "\n"):
        _TUI_LAST_FRAME_LINES = 0


def restore_tui_cursor() -> None:
    """Show the cursor again if it has been hidden by the TUI."""
    try:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
    except Exception:
        pass


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI colour codes from a string."""
    return _ANSI_RE.sub("", str(text))


def strip_ansi_len(text: str) -> int:
    """Return the visible length of a string after stripping ANSI codes."""
    return len(strip_ansi(text))


def _ansi_truncate(text: str, width: int) -> str:
    """Return a string no longer than ``width`` visible characters.

    If truncation is needed, ANSI styling is stripped for that one line.
    That is intentional: stable dashboard geometry matters more than
    preserving colour on a clipped edge.
    """
    width = max(1, int(width))
    if strip_ansi_len(text) <= width:
        return text
    raw = strip_ansi(text)
    if width <= 1:
        return raw[:width]
    return raw[: width - 1] + "…"


def _ansi_pad(text: str, width: int) -> str:
    """Pad or truncate a string to ``width`` visible characters."""
    text = _ansi_truncate(str(text), width)
    return text + " " * max(0, int(width) - strip_ansi_len(text))


def _panel_lines(title: str, body: List[str], width: int, max_body_lines: Optional[int] = None) -> List[str]:
    """Format a compact dashboard panel with fixed visible width."""
    width = max(18, int(width))
    inner = max(8, width - 2)
    body = [str(x) for x in body]
    if max_body_lines is not None and len(body) > max_body_lines:
        keep = max(0, int(max_body_lines) - 1)
        body = body[:keep] + [color_text(f"… {len(body) - keep} more", "dim")]
    top = "┌" + "─" * inner + "┐"
    title_line = "│" + _ansi_pad(color_text(title, "cyan", bold=True), inner) + "│"
    sep = "├" + "─" * inner + "┤"
    out: List[str] = [top, title_line, sep]
    if not body:
        body = [color_text("empty", "dim")]
    for line in body:
        out.append("│" + _ansi_pad(line, inner) + "│")
    out.append("└" + "─" * inner + "┘")
    return out


def _join_columns(columns: List[List[str]], gap: int = 2) -> List[str]:
    """Join already formatted fixed‑width panels side by side."""
    if not columns:
        return []
    heights = [len(c) for c in columns]
    widths = [max(strip_ansi_len(x) for x in c) if c else 0 for c in columns]
    hmax = max(heights)
    pad = " " * max(1, int(gap))
    rows: List[str] = []
    for i in range(hmax):
        parts: List[str] = []
        for col, w in zip(columns, widths):
            parts.append(_ansi_pad(col[i] if i < len(col) else "", w))
        rows.append(pad.join(parts).rstrip())
    return rows


def _dashboard_row(
    title: str,
    panels: List[tuple[str, List[str]]],
    term_w: Optional[int] = None,
    gap: int = 2,
    max_panel_body_lines: Optional[int] = None,
) -> List[str]:
    """A themed row made from panels laid out side by side when width allows.

    Narrow terminals now stack panels vertically instead of forcing tiny
    unreadable boxes.  This preserves all dashboard content while
    using the available width more honestly.
    """
    term_w_val = int(term_w or shutil.get_terminal_size((160, 40)).columns or 160)
    usable = max(40, term_w_val - 2)
    n = max(1, len(panels))
    gap_total = max(0, n - 1) * gap
    min_panel_w = 28
    if n > 1 and usable < n * min_panel_w + gap_total:
        out: List[str] = [color_text(title, "magenta", bold=True)]
        for name, body in panels:
            out.extend(_panel_lines(name, body, usable, max_panel_body_lines))
        return out
    panel_w = max(min_panel_w, (usable - gap_total) // n)
    cols = [_panel_lines(name, body, panel_w, max_panel_body_lines) for name, body in panels]
    return [color_text(title, "magenta", bold=True)] + _join_columns(cols, gap=gap)


def _dashboard_weighted_row(
    title: str,
    panels: List[tuple[str, List[str] | tuple]],
    term_w: Optional[int] = None,
    gap: int = 2,
    max_panel_body_lines: Optional[int] = None,
    min_panel_width: int = 30,
) -> List[str]:
    """A themed row with proportional panel widths and safe narrow stacking.

    Each panel may be ``(name, body)`` or ``(name, body, weight)``.  Weighted
    rows make wide terminals useful instead of giving every diagnostic
    the same cramped column even when one panel is visually dominant.
    """
    term_w_val = int(term_w or shutil.get_terminal_size((160, 40)).columns or 160)
    usable = max(40, term_w_val - 2)
    clean: List[tuple[str, List[str], float]] = []
    for item in panels:
        if len(item) >= 3:
            name, body, weight = item[0], item[1], float(item[2])
        else:
            name, body, weight = item[0], item[1], 1.0
        clean.append((str(name), list(body), max(0.05, weight)))
    n = len(clean)
    if n <= 0:
        return [color_text(title, "magenta", bold=True)]
    gap_total = max(0, n - 1) * gap
    min_w = max(22, int(min_panel_width))
    if n > 1 and usable < n * min_w + gap_total:
        out: List[str] = [color_text(title, "magenta", bold=True)]
        for name, body, _weight in clean:
            out.extend(_panel_lines(name, body, usable, max_panel_body_lines))
        return out
    available = max(n * min_w, usable - gap_total)
    total_weight = sum(w for _name, _body, w in clean) or 1.0
    raw_widths = [max(min_w, int(round(available * w / total_weight))) for _name, _body, w in clean]
    delta = available - sum(raw_widths)
    # Adjust the widest panel first so total visible width stays within terminal.
    order = sorted(range(n), key=lambda i: raw_widths[i], reverse=True)
    idx = 0
    while delta != 0 and order:
        i = order[idx % len(order)]
        if delta > 0:
            raw_widths[i] += 1
            delta -= 1
        elif raw_widths[i] > min_w:
            raw_widths[i] -= 1
            delta += 1
        idx += 1
        if idx > 10000:
            break
    cols = [
        _panel_lines(name, body, width, max_panel_body_lines)
        for (name, body, _w), width in zip(clean, raw_widths)
    ]
    return [color_text(title, "magenta", bold=True)] + _join_columns(cols, gap=gap)


def _dashboard_full_width_panel(
    title: str,
    panel_title: str,
    body: List[str],
    term_w: Optional[int] = None,
    max_body_lines: Optional[int] = None,
) -> List[str]:
    """A themed dashboard row containing one full‑width panel."""
    term_w_val = int(term_w or shutil.get_terminal_size((160, 40)).columns or 160)
    usable = max(40, term_w_val - 2)
    prefix = [color_text(title, "magenta", bold=True)] if title else []
    return prefix + _panel_lines(panel_title, body, usable, max_body_lines)


def _dashboard_density(args, term_w: Optional[int] = None, term_h: Optional[int] = None) -> str:
    """Infer dashboard density from terminal dimensions and args settings."""
    mode = str(getattr(args, "dashboard_density", "auto") or "auto").lower()
    if mode in {"compact", "normal", "full"}:
        return mode
    term_size = shutil.get_terminal_size((160, 40))
    term_w_val = int(term_w or term_size.columns or 160)
    term_h_val = int(term_h or term_size.lines or 40)
    if term_w_val < 110 or term_h_val < 34:
        return "compact"
    if term_w_val >= 150 and term_h_val >= 48:
        return "full"
    return "normal"


def dashboard_body_budget(term_h: int, fraction: float, floor: int = 10) -> int:
    """Proportional body-line budget for one dashboard row/panel.

    Scales continuously with terminal height instead of a fixed ceiling tied
    to a coarse density bucket, so a tall terminal shows more content rather
    than the same capped amount plus blank margin. `_safe_tui_frame_text`
    already truncates a frame that ends up taller than the terminal, so this
    does not need to sum exactly across rows to stay safe.
    """
    return max(int(floor), int(round(float(term_h) * float(fraction))))


def format_duration(seconds: Optional[float]) -> str:
    """Format a duration in seconds into a human‑readable string."""
    if seconds is None or not math.isfinite(float(seconds)):
        return "?"
    seconds_int = max(0, int(seconds))
    h, rem = divmod(seconds_int, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def make_progress_bar(fraction: float, width: int = 36) -> str:
    """Draw a simple shaded progress bar using ANSI colours when enabled."""
    width_int = max(8, int(width))
    frac = max(0.0, min(1.0, float(fraction) if math.isfinite(float(fraction)) else 0.0))
    filled = int(round(frac * width_int))
    bar = "#" * filled + "-" * (width_int - filled)
    if ASCII_COLOR_ENABLED:
        return color_text("#" * filled, "green", bold=True) + color_text("-" * (width_int - filled), "dim")
    return bar


def replica_fg256(replica: int) -> int:
    """Stable bright colour for a replica identity, so exchanges are visually trackable."""
    palette = [46, 51, 226, 201, 208, 117, 171, 82, 214, 159, 99, 197, 45, 220, 141, 33]
    try:
        return palette[int(replica) % len(palette)]
    except Exception:
        return 231


def replica_marker(text: str, replica: int, bg256: int = 236, bold: bool = True) -> str:
    """Return a coloured marker string for a given replica index."""
    return style_text(text, fg256=replica_fg256(replica), bg256=bg256, bold=bold)


def _ascii_position(value: float, lo: float, hi: float, width: int) -> int:
    """Map a continuous axis value to the same bin convention used by ``np.histogram``.

    Earlier dashboard builds used ``round((width‑1)*fraction)``, while the
    shaded histogram itself uses width equal bins.  That can put the yellow
    umbrella centre marker one cell away from the histogram/current‑CV bin
    near bin boundaries.  Use ``floor(fraction*width)``, capped at
    ``width‑1``, so centre markers, current markers, and histogram
    cells all share one distance axis.
    """
    width_int = max(1, int(width))
    if not math.isfinite(float(value)) or hi <= lo:
        return 0
    frac = (float(value) - float(lo)) / (float(hi) - float(lo))
    if frac <= 0.0:
        return 0
    if frac >= 1.0:
        return width_int - 1
    return max(0, min(width_int - 1, int(math.floor(frac * width_int))))


def _hist3d_cell(level: int) -> str:
    level_clamped = max(0, min(8, int(level)))
    if level_clamped <= 0:
        return color_text("·", "dim")
    chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    fg_palette = [None, 246, 249, 81, 87, 117, 123, 159, 231]
    bg_palette = [None, 236, 236, 237, 238, 239, 24, 25, 31]
    return style_text(chars[level_clamped], fg256=fg_palette[level_clamped], bg256=bg_palette[level_clamped], bold=level_clamped >= 6)


def _render_histogram_row(
    hist_values: List[float],
    current_value: float,
    center_value: float,
    lo: float,
    hi: float,
    width: int,
    replica: int = 0,
) -> str:
    width_int = max(20, int(width))
    values = np.asarray([float(v) for v in hist_values if math.isfinite(float(v))], dtype=float)
    if values.size == 0:
        values = np.asarray([float(current_value)], dtype=float)
    counts, _ = np.histogram(values, bins=width_int, range=(lo, hi))
    max_count = int(np.max(counts)) if counts.size > 0 else 0
    chars: List[str] = []
    for count in counts:
        if max_count <= 0:
            chars.append(color_text("·", "dim"))
        else:
            chars.append(_hist3d_cell(int(round(float(count) / float(max_count) * 8.0))))
    cpos = _ascii_position(center_value, lo, hi, width_int)
    dpos = _ascii_position(current_value, lo, hi, width_int)
    if dpos == cpos:
        chars[dpos] = replica_marker("◉", replica, bg256=53, bold=True)
    else:
        chars[cpos] = style_text("│", fg256=226, bg256=236, bold=True)
        chars[dpos] = replica_marker("●", replica, bg256=236, bold=True)
    return "".join(chars)


def render_distance_ascii(
    rows: List[dict],
    phase: str,
    step: int,
    total_steps: Optional[int] = None,
    width: int = 54,
    max_replicas: int = 32,
    mode: str = "hist3d",
    history_by_replica: Optional[dict[int, List[float]]] = None,
    history_by_window: Optional[dict[int, List[float]]] = None,
    histogram_source: str = "window",
    primary_label: str = "terminal distance",
    primary_units: str = "A",
    primary_k_unit_label: str = "kcal/mol/A^2",
) -> str:
    """Render an ASCII histogram or table of replica CV values for the dashboard."""
    if not rows:
        return ""
    mode_val = str(mode or "none").lower()
    if mode_val == "none":
        return ""
    width_int = max(20, int(width or 54))
    max_rep = max(1, int(max_replicas or 32))
    history_by_replica = history_by_replica or {}
    history_by_window = history_by_window or {}
    histogram_source_val = str(histogram_source or "window").lower()
    if histogram_source_val not in {"window", "replica"}:
        histogram_source_val = "window"
    active_history: dict[int, List[float]] = history_by_window if histogram_source_val == "window" else history_by_replica
    primary_label_val = str(primary_label or "primary CV")
    primary_units_val = str(primary_units or "")
    primary_k_unit_label_val = str(primary_k_unit_label or "kcal/mol/CV^2")

    def _unit_text(value: float) -> str:
        if primary_units_val in {"", "dimensionless"}:
            return f"{float(value):6.2f}"
        return f"{float(value):6.2f} {primary_units_val}"

    clean: List[dict] = []
    for row in rows:
        try:
            clean.append({
                "replica": int(row["replica"]),
                "window": int(row["window"]),
                "center_A": float(row["center_A"]),
                "k_kcal_mol_A2": float(row["k_kcal_mol_A2"]),
                "cv_A": float(row["cv_A"]),
                "umbrella_bias_kcal_mol": float(
                    row.get(
                        "umbrella_bias_kcal_mol",
                        0.5
                        * float(row["k_kcal_mol_A2"])
                        * (float(row["cv_A"]) - float(row["center_A"])) ** 2,
                    )
                ),
                "umbrella_pull_kcal_mol_A": float(
                    row.get(
                        "umbrella_pull_kcal_mol_A",
                        float(row["k_kcal_mol_A2"]) * (float(row["center_A"]) - float(row["cv_A"]))
                    )
                ),
            })
        except Exception:
            continue
    if not clean:
        return ""

    values = np.asarray([r["cv_A"] for r in clean], dtype=float)
    centers = np.asarray([r["center_A"] for r in clean], dtype=float)
    k_values = np.asarray([r["k_kcal_mol_A2"] for r in clean], dtype=float)
    finite_parts: List[np.ndarray] = [values[np.isfinite(values)], centers[np.isfinite(centers)]]
    hist_minmax: List[float] = []
    for vals in active_history.values():
        arr = np.asarray([float(v) for v in vals if math.isfinite(float(v))], dtype=float)
        if arr.size:
            hist_minmax.extend([float(np.nanmin(arr)), float(np.nanmax(arr))])
    if hist_minmax:
        finite_parts.append(np.asarray(hist_minmax, dtype=float))
    finite = np.concatenate([p for p in finite_parts if p.size > 0])
    if finite.size == 0:
        return ""
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    span = max(1.0, hi - lo)
    lo -= 0.08 * span
    hi += 0.08 * span
    summary = {
        "min": float(np.nanmin(values)),
        "mean": float(np.nanmean(values)),
        "max": float(np.nanmax(values)),
    }
    total_txt = f"/{int(total_steps)}" if total_steps is not None and int(total_steps) > 0 else ""
    finite_k = k_values[np.isfinite(k_values)]
    k_summary = ""
    if finite_k.size:
        if abs(float(np.nanmax(finite_k)) - float(np.nanmin(finite_k))) < 1.0e-12:
            k_summary = f"  {color_text('k', 'dim')} {float(np.nanmean(finite_k)):6.3f} {primary_k_unit_label_val}"
        else:
            k_summary = (
                f"  {color_text('k', 'dim')} {float(np.nanmin(finite_k)):6.3f}"
                f"..{float(np.nanmax(finite_k)):6.3f} {primary_k_unit_label_val}"
            )
    header = (
        "\n"
        + color_text(f"+-- CV {primary_label_val} coverage ", "cyan", bold=True)
        + color_text("-" * 58, "cyan")
        + "\n"
        + f"| {color_text('phase', 'dim')} {color_text(phase, 'cyan', bold=True)}  "
        + f"{color_text('step', 'dim')} {color_text(str(int(step)) + total_txt, 'white', bold=True)}\n"
        + f"| {color_text('range', 'dim')} {_unit_text(lo)} .. {_unit_text(hi)}  "
        + f"{color_text('min', 'dim')} {summary['min']:6.2f}  "
        + f"{color_text('mean', 'dim')} {summary['mean']:6.2f}  "
        + f"{color_text('max', 'dim')} {summary['max']:6.2f}"
        + k_summary
    )
    if mode_val == "compact":
        compact: List[str] = [color_text(".", "dim") for _ in range(width_int)]
        for cval in centers:
            compact[_ascii_position(cval, lo, hi, width_int)] = color_text("|", "yellow", bold=True)
        for rr, val in zip(clean, values):
            pos = _ascii_position(val, lo, hi, width_int)
            compact[pos] = replica_marker("*", int(rr["replica"]))
        return header + "\n" + f"{_unit_text(lo)} [{''.join(compact)}] {_unit_text(hi)}"
    shown = sorted(clean, key=lambda r: (r["window"], r["replica"]))[:max_rep]
    if mode_val in {"hist", "hist3d"}:
        lines: List[str] = [
            header,
            color_text("legend:", "dim") + " "
            + color_text("●", "white", bold=True) + f" current {primary_label_val} colored by replica id, "
            + style_text("│", fg256=226, bg256=236, bold=True) + " umbrella center, "
            + style_text("◉", fg256=231, bg256=53, bold=True) + " overlap, "
            + color_text(
                "shaded bars = sampled CV distribution by " + ("umbrella window" if histogram_source_val == "window" else "replica"),
                "dim",
            ),
            color_text(
                "rep win      cv_A  center       k     bias     pull   delta  "
                + ("window-CV distribution" if histogram_source_val == "window" else "replica-CV history"),
                "white",
                bold=True,
            ),
        ]
        for r in shown:
            hist_key = int(r["window"] if histogram_source_val == "window" else r["replica"])
            hist_values = active_history.get(hist_key, [r["cv_A"]])
            if not hist_values:
                hist_values = [r["cv_A"]]
            bar = _render_histogram_row(hist_values, r["cv_A"], r["center_A"], lo, hi, width_int, replica=int(r["replica"]))
            delta = r["cv_A"] - r["center_A"]
            delta_col = "green" if abs(delta) <= 0.5 else "yellow" if abs(delta) <= 1.5 else "red"
            lines.append(
                f"r{r['replica']:02d} w{r['window']:02d}  "
                + f"{r['cv_A']:8.2f} {r['center_A']:7.2f} {r['k_kcal_mol_A2']:7.3f} "
                + f"{r['umbrella_bias_kcal_mol']:8.2f} {r['umbrella_pull_kcal_mol_A']:+8.3f} "
                + f"{color_text(f'{delta:+7.2f}', delta_col, bold=True)}  "
                + f"[{bar}] "
                + color_text(("nw=" if histogram_source_val == "window" else "nr=") + str(len(hist_values)), "dim")
            )
        omitted = len(clean) - len(shown)
        if omitted > 0:
            lines.append(color_text(f"... {omitted} more replicas omitted; raise --distance-ascii-max-replicas to show them.", "dim"))
        return "\n".join(lines)
    lines: List[str] = [
        header,
        color_text("legend:", "dim") + " " + color_text("C", "yellow", bold=True) + " center, "
        + color_text("*", "white", bold=True) + " current colored by replica id, " + color_text("@", "magenta", bold=True) + " overlap",
        color_text("rep win      cv_A  center       k     bias     pull   delta  visual", "white", bold=True),
    ]
    for r in shown:
        hist_key = int(r["window"] if histogram_source_val == "window" else r["replica"])
        hist_values = active_history.get(hist_key, [r["cv_A"]])
        if not hist_values:
            hist_values = [r["cv_A"]]
        bar = _render_histogram_row(hist_values, r["cv_A"], r["center_A"], lo, hi, width_int, replica=int(r["replica"]))
        delta = r["cv_A"] - r["center_A"]
        delta_col = "green" if abs(delta) <= 0.5 else "yellow" if abs(delta) <= 1.5 else "red"
        lines.append(
            f"r{r['replica']:02d} w{r['window']:02d}  "
            + f"{r['cv_A']:8.2f} {r['center_A']:7.2f} {r['k_kcal_mol_A2']:7.3f} "
            + f"{r['umbrella_bias_kcal_mol']:8.2f} {r['umbrella_pull_kcal_mol_A']:+8.3f} "
            + f"{color_text(f'{delta:+7.2f}', delta_col, bold=True)}  "
            + f"{bar} "
        )
    omitted = len(clean) - len(shown)
    if omitted > 0:
        lines.append(color_text(f"... {omitted} more replicas omitted; raise --distance-ascii-max-replicas to show them.", "dim"))
    return "\n".join(lines)