"""
ANSI colour handling for TUI output.

This module centralises the colour configuration used by the console
interface.  It exposes a small set of helpers to apply ANSI colour
sequences and to toggle colour output on or off.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# Define a mapping of simple colour names to ANSI escape codes.
ANSI_COLORS: dict[str, str] = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

ROLE_TITLE = "title"
ROLE_SECTION = "section"
ROLE_GOOD = "good"
ROLE_WARN = "warn"
ROLE_BAD = "bad"
ROLE_MUTED = "muted"
ROLE_VALUE = "value"

_ROLE_STYLE: dict[str, tuple[str, bool]] = {
    ROLE_TITLE: ("cyan", True),
    ROLE_SECTION: ("magenta", True),
    ROLE_GOOD: ("green", True),
    ROLE_WARN: ("yellow", True),
    ROLE_BAD: ("red", True),
    ROLE_MUTED: ("dim", False),
    ROLE_VALUE: ("white", True),
}

# This flag controls whether colour escapes are emitted.  It defaults to
# ``False`` because typical HPC launchers and log parsers may not
# handle ANSI escapes well.  Use :func:`configure_color` to change it.
ASCII_COLOR_ENABLED: bool = False


def configure_color(mode: str, *, tui_mode: Optional[str] = None) -> None:
    """Configure ANSI colour handling for console output.

    ``mode`` may be ``"always"``, ``"never"``, or ``"auto"`` (the default).
    ``auto`` enables colour when ``sys.stdout`` is a TTY, and also for the live
    dashboard (``tui_mode="dashboard"``) even when stdout is a SLURM log: the
    dashboard already writes cursor/clear escapes there, so a log reader must
    handle ANSI anyway, and without colour the OK/WARN/BAD status is lost.
    The ``NO_COLOR`` convention (https://no-color.org) switches ``auto`` off;
    an explicit ``always`` still wins, since it was asked for by name.
    """
    global ASCII_COLOR_ENABLED
    mode = str(mode or "auto").lower()
    if mode == "always":
        ASCII_COLOR_ENABLED = True
    elif mode == "never":
        ASCII_COLOR_ENABLED = False
    elif os.environ.get("NO_COLOR"):
        ASCII_COLOR_ENABLED = False
    else:
        is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        ASCII_COLOR_ENABLED = is_tty or str(tui_mode or "").lower() == "dashboard"


def style_text(
    text: object,
    color: Optional[str] = None,
    bold: bool = False,
    dim: bool = False,
    fg256: Optional[int] = None,
    bg256: Optional[int] = None,
) -> str:
    """Apply ANSI styles to ``text`` if colour output is enabled.

    Args:
        text: The text to style.
        color: A key from :data:`ANSI_COLORS`.
        bold: Whether to apply the bold style.
        dim: Whether to apply the dim style.
        fg256: A 0–255 colour index for 256‑colour foreground.
        bg256: A 0–255 colour index for 256‑colour background.

    Returns:
        The styled text as a string.  If colour is disabled, returns
        ``str(text)`` unchanged.
    """
    s = str(text)
    if not ASCII_COLOR_ENABLED:
        return s
    prefix = ""
    if bold:
        prefix += ANSI_COLORS["bold"]
    if dim:
        prefix += ANSI_COLORS["dim"]
    if color:
        prefix += ANSI_COLORS.get(color, "")
    if fg256 is not None:
        prefix += f"\033[38;5;{int(fg256)}m"
    if bg256 is not None:
        prefix += f"\033[48;5;{int(bg256)}m"
    return prefix + s + ANSI_COLORS["reset"]


def color_text(
    text: object,
    color: Optional[str] = None,
    bold: bool = False,
    dim: bool = False,
) -> str:
    """Convenience wrapper around :func:`style_text` for simple colour use."""
    return style_text(text, color=color, bold=bold, dim=dim)


def role_text(text: object, role: str, bold: Optional[bool] = None) -> str:
    """Style `text` using a named semantic role instead of a raw color name.

    Centralizing role -> color mapping here means every dashboard panel
    that asks for "a title" or "a warning" gets the same color, instead of
    each call site picking cyan/magenta/yellow ad hoc.
    """
    color, default_bold = _ROLE_STYLE.get(role, _ROLE_STYLE[ROLE_MUTED])
    use_bold = default_bold if bold is None else bool(bold)
    if color == "dim":
        return style_text(text, dim=True, bold=use_bold)
    return style_text(text, color=color, bold=use_bold)


def severity_role(value: float, warn_at: float, bad_at: float) -> str:
    """Map a magnitude to ROLE_GOOD/ROLE_WARN/ROLE_BAD by absolute thresholds."""
    magnitude = abs(float(value))
    if magnitude <= float(warn_at):
        return ROLE_GOOD
    if magnitude <= float(bad_at):
        return ROLE_WARN
    return ROLE_BAD


__all__ = [
    "ANSI_COLORS",
    "ASCII_COLOR_ENABLED",
    "configure_color",
    "style_text",
    "color_text",
    "role_text",
    "severity_role",
    "ROLE_TITLE",
    "ROLE_SECTION",
    "ROLE_GOOD",
    "ROLE_WARN",
    "ROLE_BAD",
    "ROLE_MUTED",
    "ROLE_VALUE",
]
