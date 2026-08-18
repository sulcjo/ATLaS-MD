import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_gareus_mbar as agm

_NAMES = [
    "_secondary_cv_label", "_secondary_cv_regions", "_regime_slug",
    "_primary_cv_label", "_primary_cv_units", "_primary_cv_axis_label",
    "_smooth_masked_grid", "_smooth_pmf_1d", "_eff_smooth", "FES_PLOT_VMAX_VALUES",
    "_fes_range_tag", "_fes_range_label", "_fes_variant_path", "_plot_2d_fes_range",
    "_plot_2d_fes_multirange", "plot_2d_fes", "plot_cv1_cv2_2d_fes", "plot_pca_2d_fes",
    "_per_window_gamd_boost_stats", "plot_gamd_boost", "plot_outputs",
    "plot_rg_outputs", "_OPT_IN_GAMD_METHODS", "_want_gamd_method", "_visible_pmfs",
]


def test_plotting_module_is_importable():
    import gareus.mbar_analysis.plotting  # noqa: F401


def test_every_relocated_name_is_the_same_object_via_reexport():
    import gareus.mbar_analysis.plotting as plotting

    for name in _NAMES:
        assert hasattr(plotting, name), f"plotting.py missing {name}"
        assert hasattr(agm, name), f"analyze_gareus_mbar re-export missing {name}"
        assert getattr(agm, name) is getattr(plotting, name), (
            f"{name}: analyze_gareus_mbar copy is not the same object as "
            "gareus.mbar_analysis.plotting's — re-export is broken"
        )


def test_signatures_unchanged():
    import gareus.mbar_analysis.plotting as plotting

    assert list(inspect.signature(plotting.plot_outputs).parameters) == [
        "d", "pmfs", "selected", "O", "out", "warnings", "smooth_sigma", "args",
    ]
    assert list(inspect.signature(plotting.plot_gamd_boost).parameters) == ["d", "out", "warnings"]
    assert list(inspect.signature(plotting._smooth_pmf_1d).parameters) == ["pmf", "sigma"]


def test_fes_range_helpers_unchanged_behavior():
    import gareus.mbar_analysis.plotting as plotting

    assert plotting._fes_range_tag(None) == "0_all"
    assert plotting._fes_range_tag(5.0) == "0_5"
    assert plotting.FES_PLOT_VMAX_VALUES == (2.0, 5.0, 10.0, 20.0, None)


def test_want_gamd_method_resolves_its_module_constant_after_move():
    # Regression guard for the L6386 boundary: _want_gamd_method reads
    # _OPT_IN_GAMD_METHODS from its own module's globals. If the constant
    # were left behind in analyze_gareus_mbar.py, this would raise
    # NameError instead of returning a bool.
    import gareus.mbar_analysis.plotting as plotting

    assert plotting._OPT_IN_GAMD_METHODS == {
        "gamd_exponential": "plot_gamd_exponential",
        "gamd_cumulant3": "plot_gamd_cumulant3",
    }

    class _Args:
        plot_gamd_exponential = False

    assert plotting._want_gamd_method("gamd_exponential", "gamd_exponential", _Args()) is True
    assert plotting._want_gamd_method("gamd_exponential", "umbrella_only", _Args()) is False
    assert plotting._want_gamd_method("umbrella_only", "umbrella_only", _Args()) is True


def test_cv_label_helpers_unchanged_behavior():
    # The six CV-label helpers are pure meta-dict -> str/list formatters with
    # no I/O; a spot-check of each branch-selecting input is enough to prove
    # the relocation copied bodies rather than paraphrasing them.
    # _regime_slug additionally exercises the interim `from
    # analyze_gareus_mbar import _slug` local import (Plan A6d's symbol) --
    # if that import were placed at module level instead, importing
    # gareus.mbar_analysis.plotting at all would already have failed above.
    import gareus.mbar_analysis.plotting as plotting

    assert plotting._secondary_cv_label({"secondary_cv": "torsion-pca"}) == "Bootstrap torsion PC1"
    assert plotting._secondary_cv_label({"secondary_cv": {"mode": "tica-linear"}}) == "tIC1 torsion CV"
    assert plotting._secondary_cv_label({}) == "Secondary CV"

    assert plotting._secondary_cv_regions(
        {"secondary_cv": {"regions": [{"value": 1.5, "label": "alpha"}, {"name": "no-value"}]}}
    ) == [{"value": 1.5, "label": "alpha"}]
    assert plotting._secondary_cv_regions({"secondary_cv": "tica-linear"}) == []

    assert plotting._regime_slug("") == "unknown"
    # A non-empty regime must route through the interim
    # `from analyze_gareus_mbar import _slug` local import. Assert only that
    # the import resolves and yields a usable slug -- do NOT pin _slug's own
    # output format here, that belongs to Plan A6d, which owns _slug.
    slug = plotting._regime_slug("torsion-pca")
    assert isinstance(slug, str) and slug and slug != "unknown"

    assert plotting._primary_cv_label({"primary_cv": "nonlocal-contacts"}) == "nonlocal contact fraction"
    assert plotting._primary_cv_units({"primary_cv": "nonlocal-contacts"}) == "dimensionless"
    assert plotting._primary_cv_axis_label({"primary_cv": "nonlocal-contacts"}) == "nonlocal contact fraction"
    assert plotting._primary_cv_axis_label({}) == "CV distance (A)"
