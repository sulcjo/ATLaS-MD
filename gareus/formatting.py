"""
Formatting helpers for collective variable (CV) values.

These functions provide common string representations for primary
collective variables.  They were extracted from the monolithic
``gareus_peptide.py`` script to improve reuse across modules.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from . import colors  # re‑use colour helpers if needed


def primary_cv_unit_suffix(args_or_mode: Any) -> str:
    """Return the units suffix for the primary CV.

    ``args_or_mode`` may be an argument namespace or a string
    identifier.  When no units are defined or the CV is dimensionless,
    an empty string is returned.  This helper delegates to
    ``primary_cv_units`` if it exists in the caller; otherwise it
    attempts to read an attribute from ``args_or_mode``.
    """
    units = ""
    # Try to introspect a units attribute or call a function.  This is a
    # lightweight implementation; more exhaustive logic lives in
    # gareus.core until it is refactored.
    if hasattr(args_or_mode, "primary_cv_units"):
        try:
            units = args_or_mode.primary_cv_units
        except Exception:
            units = ""
    elif hasattr(args_or_mode, "primary_cv"):
        # Fallback for legacy args namespaces: use a canonical unit.
        units = "nm"
    if not units or units == "dimensionless":
        return ""
    return f" {units}"


def format_primary_cv_value(
    value: float, args_or_mode: Any, precision: int = 3
) -> str:
    """Format a primary CV value with units and precision.

    If ``value`` is not finite, returns ``"n/a"``.
    """
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    suffix = primary_cv_unit_suffix(args_or_mode)
    return f"{v:.{int(precision)}f}{suffix}"


def primary_cv_format_value(
    value: float, args_or_mode: Any, precision: int = 3
) -> str:
    """Backward‑compatible alias for :func:`format_primary_cv_value`.

    Several patches used the name ``primary_cv_format_value`` while the
    canonical helper is named :func:`format_primary_cv_value`.  Retain
    this alias to preserve backwards compatibility.
    """
    return format_primary_cv_value(value, args_or_mode, precision=precision)


def format_primary_delta_value(
    value: float, args_or_mode: Any, precision: int = 3
) -> str:
    """Format a delta value for the primary CV.

    This is currently an alias for :func:`format_primary_cv_value`,
    included for API completeness.
    """
    return format_primary_cv_value(value, args_or_mode, precision=precision)


__all__ = [
    "primary_cv_unit_suffix",
    "format_primary_cv_value",
    "primary_cv_format_value",
    "format_primary_delta_value",
]