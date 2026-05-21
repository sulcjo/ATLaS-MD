"""
ANSI colour handling for TUI output.

This module centralises the colour configuration used by the console
interface.  It exposes a small set of helpers to apply ANSI colour
sequences and to toggle colour output on or off.
"""

from __future__ import annotations

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

# This flag controls whether colour escapes are emitted.  It defaults to
# ``False`` because typical HPC launchers and log parsers may not
# handle ANSI escapes well.  Use :func:`configure_color` to change it.
ASCII_COLOR_ENABLED: bool = False


def configure_color(mode: str) -> None:
    """Configure ANSI colour handling for console output.

    ``mode`` may be ``"always"``, ``"never"``, or ``"auto"`` (the
    default).  In ``auto`` mode, colour is enabled if ``sys.stdout``
    appears to be a TTY.
    """
    global ASCII_COLOR_ENABLED
    mode = str(mode or "auto").lower()
    if mode == "always":
        ASCII_COLOR_ENABLED = True
    elif mode == "never":
        ASCII_COLOR_ENABLED = False
    else:
        # Fall back to auto detection.  Some environments may not set
        # isatty properly, but this provides a reasonable default.
        ASCII_COLOR_ENABLED = bool(getattr(sys.stdout, "isatty", lambda: False)())


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


__all__ = [
    "ANSI_COLORS",
    "ASCII_COLOR_ENABLED",
    "configure_color",
    "style_text",
    "color_text",
]