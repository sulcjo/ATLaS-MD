"""Shared house style for GaREUS analysis plots (categorical estimator curves).

Presentation-only. Centralises the colour + linestyle decisions so the many
1D-PMF / estimator-overlay plots read as one system instead of drifting per
call-site. 2D FES heatmaps keep viridis (correct sequential choice) and are NOT
touched by this module.

Palette: the Okabe-Ito colourblind-safe qualitative set, reordered so the
strongest-contrast hues lead (validated with dataviz `validate_palette.js`:
lightness/chroma PASS, adjacent CVD ΔE 37.2 PASS). Yellow (#F0E442) is dropped
for line work — it is a fill hue and washes out as a thin line on white.

Because the three light hues fall just under the 3:1 line-contrast bar on white,
identity is carried by a SECOND channel as well — each estimator gets a fixed
linestyle. Hue + linestyle + legend means a curve is identifiable under CVD, in
grayscale print, and at low contrast, not by colour alone.
"""
from __future__ import annotations

from typing import Optional

# Full Okabe-Ito set (reference).
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

# Validated line-safe order (yellow + pure black dropped; strong hues first).
OKABE_ITO_LINES = [
    "#0072B2",  # blue        (strong)
    "#009E73",  # bluish green(strong)
    "#D55E00",  # vermillion  (strong)
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
]

_SOLID, _DASH, _DASHDOT, _DOT = "-", "--", "-.", ":"

# Fixed (colour, linestyle) per known estimator. Strong hues + solid lines go to
# the estimators an operator reads first; weaker hues get linestyle relief.
METHOD_STYLE: dict[str, tuple[str, str]] = {
    "gamd_cumulant2":  ("#0072B2", _SOLID),    # default selected estimator
    "umbrella_only":   ("#009E73", _SOLID),    # the honest unbiased reference
    "gamd_exponential":("#D55E00", _DASH),
    "gamd_cumulant3":  ("#CC79A7", _DASHDOT),
    "dtram":           ("#E69F00", _DASH),
}

_PRETTY: dict[str, str] = {
    "gamd_cumulant2":  "GaMD cumulant-2",
    "gamd_cumulant3":  "GaMD cumulant-3",
    "gamd_exponential":"GaMD exponential",
    "umbrella_only":   "umbrella (unbiased)",
    "dtram":           "dTRAM",
}

# Deterministic fallback for observables/methods with no fixed slot.
_FALLBACK_LS = [_SOLID, _DASH, _DASHDOT, _DOT]

BAR_COLOR = "#0072B2"        # uniform bars (window counts etc.)
REFERENCE_GREY = "#4D4D4D"   # neutral guide lines / reference curves


def method_style(name: str, idx: int = 0) -> tuple[str, str]:
    """Stable (colour, linestyle) for an estimator. Unknown names cycle the
    line-safe palette by ``idx`` so repeated observables stay distinct."""
    if name in METHOD_STYLE:
        return METHOD_STYLE[name]
    color = OKABE_ITO_LINES[idx % len(OKABE_ITO_LINES)]
    ls = _FALLBACK_LS[(idx // len(OKABE_ITO_LINES)) % len(_FALLBACK_LS)]
    return color, ls


def method_color(name: str, idx: int = 0) -> str:
    return method_style(name, idx)[0]


def pretty_method(name: str) -> str:
    return _PRETTY.get(name, name)


def plot_method_curve(ax, x, y, method: str, selected: Optional[str] = None,
                      idx: int = 0, **kw):
    """Plot one estimator curve with its fixed house style. The selected
    estimator is drawn thicker and on top. Returns the Line2D list from ax.plot."""
    color, ls = method_style(method, idx)
    is_sel = (method == selected)
    kw.setdefault("linewidth", 2.6 if is_sel else 1.5)
    kw.setdefault("zorder", 3 if is_sel else 2)
    kw.setdefault("alpha", 1.0 if is_sel else 0.9)
    return ax.plot(x, y, label=pretty_method(method), color=color, linestyle=ls, **kw)


def style_line_axes(ax, xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                    title: Optional[str] = None, legend: bool = True) -> None:
    """Apply consistent grid, spines, labels and legend to a 1D line/bar axis."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        if side in ax.spines:
            ax.spines[side].set_visible(False)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(frameon=False, fontsize=9)


def annotate_minimum(ax, x, y, label: Optional[str] = None,
                     color: str = REFERENCE_GREY) -> None:
    """Mark the PMF minimum with a light vertical guide + point. No-op if all-NaN."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        return
    i = int(np.nanargmin(np.where(finite, y, np.inf)))
    xm, ym = float(x[i]), float(y[i])
    ax.axvline(xm, color=color, linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)
    ax.plot([xm], [ym], marker="o", markersize=6, color=color, zorder=5)
    txt = label if label is not None else f"min @ {xm:.3g}"
    ax.annotate(txt, xy=(xm, ym), xytext=(4, 6), textcoords="offset points",
                fontsize=8, color=color)
