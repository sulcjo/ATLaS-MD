"""Tests for gareus_plotstyle: shared house style for GaREUS analysis plots.

Locks the categorical-palette contract (stable per-estimator hue + linestyle,
colorblind-safe order, deterministic fallback) and the axis-styling helpers.
Rendering helpers are smoke-tested on the headless Agg backend.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import gareus_plotstyle as ps  # noqa: E402


# --- palette / mapping (pure) -----------------------------------------------
def test_okabe_ito_line_palette_is_the_validated_order():
    # Reordered Okabe-Ito subset that passed the CVD validator (ΔE 37.2),
    # yellow dropped for line contrast on white.
    assert ps.OKABE_ITO_LINES[0] == "#0072B2"        # strong blue leads
    assert "#F0E442" not in ps.OKABE_ITO_LINES       # yellow excluded for lines
    assert len(ps.OKABE_ITO_LINES) == len(set(ps.OKABE_ITO_LINES))  # no dups


def test_method_style_is_stable_per_estimator():
    # Same method -> same (color, linestyle) every call (color follows entity).
    assert ps.method_style("gamd_cumulant2") == ps.method_style("gamd_cumulant2")
    c2, _ = ps.method_style("gamd_cumulant2")
    assert c2 == "#0072B2"                            # selected default gets strong blue


def test_method_style_gives_distinct_colors_to_core_estimators():
    core = ["umbrella_only", "gamd_exponential", "gamd_cumulant2", "gamd_cumulant3"]
    colors = [ps.method_style(m)[0] for m in core]
    assert len(set(colors)) == len(core)              # all distinct


def test_method_style_weak_hues_get_linestyle_relief():
    # Light Okabe-Ito hues (contrast WARN) must carry a non-solid linestyle so
    # identity survives low contrast / CVD / print.
    for m in ("gamd_exponential", "gamd_cumulant3"):
        _, ls = ps.method_style(m)
        assert ls != "-"


def test_method_style_unknown_name_cycles_deterministically():
    a = ps.method_style("mystery_obs", idx=0)
    b = ps.method_style("other_obs", idx=1)
    assert a[0] != b[0]                               # different slot -> different hue
    assert ps.method_style("mystery_obs", idx=0) == a  # deterministic


def test_pretty_method_maps_known_and_falls_back():
    assert ps.pretty_method("gamd_cumulant2") == "GaMD cumulant-2"
    assert ps.pretty_method("umbrella_only") == "umbrella (unbiased)"
    assert ps.pretty_method("weird_name") == "weird_name"  # fallback = raw


# --- rendering helpers (Agg smoke) ------------------------------------------
def test_style_line_axes_sets_grid_labels_and_trims_spines():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="x")
    ps.style_line_axes(ax, xlabel="CV (Å)", ylabel="PMF", title="T", legend=True)
    assert ax.get_xlabel() == "CV (Å)"
    assert ax.get_ylabel() == "PMF"
    assert ax.get_title() == "T"
    assert ax.xaxis._major_tick_kw is not None
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    assert ax.get_legend() is not None
    plt.close(fig)


def test_style_line_axes_no_legend_when_disabled():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ps.style_line_axes(ax, legend=False)
    assert ax.get_legend() is None
    plt.close(fig)


def test_plot_method_curve_applies_color_and_emphasizes_selected():
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 10)
    ln_sel = ps.plot_method_curve(ax, x, x, "gamd_cumulant2", selected="gamd_cumulant2")[0]
    ln_oth = ps.plot_method_curve(ax, x, x, "gamd_cumulant3", selected="gamd_cumulant2")[0]
    # matplotlib normalizes hex to rgba; compare via to_rgba
    from matplotlib.colors import to_rgba
    assert to_rgba(ln_sel.get_color()) == to_rgba("#0072B2")
    assert ln_sel.get_linewidth() > ln_oth.get_linewidth()   # selected is thicker
    assert ln_sel.get_label() == "GaMD cumulant-2"           # pretty label
    plt.close(fig)


def test_annotate_minimum_adds_marker_without_error():
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 50)
    y = (x - 0.4) ** 2
    n_before = len(ax.lines)
    ps.annotate_minimum(ax, x, y)
    assert len(ax.lines) > n_before                          # added a marker/vline
    plt.close(fig)


def test_annotate_minimum_safe_on_all_nan():
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 5)
    y = np.full(5, np.nan)
    ps.annotate_minimum(ax, x, y)   # must not raise
    plt.close(fig)
