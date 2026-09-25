import math
import subprocess
import sys

from gareus.layout_neighbours import (
    other_rung_same_centre, same_rung_neighbours, spatial_neighbour_pairs,
)
from gareus.math_helpers import restraint_sigma as restraint_sigma_from_math_helpers
from gareus.dashboard.ranking import restraint_sigma as restraint_sigma_from_ranking


def _layout():
    # 2 centres x 2 rungs; windows 0,1 at centre A (lambda 0, 1); 2,3 at centre B.
    centers = [0.1, 0.1, 0.2, 0.2]
    sec = [0.0, 0.0, 0.0, 0.0]
    lam = [0.0, 1.0, 0.0, 1.0]
    return centers, sec, lam


def test_the_dashboard_module_reexports_the_same_function():
    from gareus.dashboard import neighbours as dash

    assert dash.spatial_neighbour_pairs is spatial_neighbour_pairs


def test_same_rung_neighbours_maps_each_window_to_its_rung_partners():
    c, s, lam = _layout()
    pairs = spatial_neighbour_pairs(c, s, lam, [300.0] * 4, [1.0] * 4, 300.0)
    nb = same_rung_neighbours(pairs, lam)
    assert nb[0] == [2] and nb[2] == [0] and nb[1] == [3] and nb[3] == [1]


def test_other_rung_same_centre_groups_rungs_of_one_centre():
    c, s, lam = _layout()
    other = other_rung_same_centre(c, s, lam)
    assert other == {0: [1], 1: [0], 2: [3], 3: [2]}


def test_unknown_lambda_windows_have_no_rung_partners():
    other = other_rung_same_centre([0.1, 0.1], [0.0, 0.0], [math.nan, math.nan])
    assert other == {0: [], 1: []}


def test_core_module_imports_without_dashboard():
    """Verify layout_neighbours imports with no circular dependency on dashboard."""
    result = subprocess.run(
        [sys.executable, "-c", "import gareus.layout_neighbours as m; print(m.spatial_neighbour_pairs)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "spatial_neighbour_pairs" in result.stdout


def test_restraint_sigma_is_same_object_in_both_modules():
    """Verify restraint_sigma from ranking and math_helpers is the same object."""
    assert restraint_sigma_from_ranking is restraint_sigma_from_math_helpers


def test_other_rung_same_centre_rounds_lambdas_consistently():
    """Verify that nearly-equal lambdas (0.5 and 0.5+1e-9) are treated as the same rung."""
    # With rounding to 6 decimals, 0.5 and 0.5+1e-9 should round to the same value
    # and thus not appear as neighbours (they are the same rung).
    other = other_rung_same_centre([0.1, 0.1], [0.0, 0.0], [0.5, 0.5 + 1e-9])
    assert other == {0: [], 1: []}
