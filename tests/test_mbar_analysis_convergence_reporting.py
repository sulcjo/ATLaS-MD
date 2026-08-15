import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_gareus_mbar as agm

_NAMES = [
    "_add_epoch_annotations_to_axes", "_write_epoch_ess_plot", "write_convergence_plots",
    "_add_aggregate_ns_secondary_axis", "write_observable_convergence_plots",
    "_plot_basin_population_convergence", "write_observable_convergence_report",
]


def test_convergence_reporting_module_is_importable():
    import gareus.mbar_analysis.convergence_reporting  # noqa: F401


def test_every_relocated_name_is_the_same_object_via_reexport():
    import gareus.mbar_analysis.convergence_reporting as cr

    for name in _NAMES:
        assert hasattr(cr, name), f"convergence_reporting.py missing {name}"
        assert getattr(agm, name) is getattr(cr, name), f"{name}: re-export is broken"


def test_write_convergence_report_is_fully_gone():
    assert not hasattr(agm, "write_convergence_report")
    import gareus.mbar_analysis.convergence_reporting as cr
    assert not hasattr(cr, "write_convergence_report")


def test_convergence_reporting_functions_a6b_still_owns_stayed_behind():
    # checkpoint_steps_from_data and its cache-key siblings are Plan A6b's,
    # not this plan's -- confirm they were NOT accidentally swept up here.
    assert hasattr(agm, "checkpoint_steps_from_data")
    assert hasattr(agm, "_get_convergence_mbar_cache")
    assert hasattr(agm, "_get_checkpoint_steps_cache")
    import gareus.mbar_analysis.convergence_reporting as cr
    assert not hasattr(cr, "checkpoint_steps_from_data")
    assert not hasattr(cr, "_get_convergence_mbar_cache")
