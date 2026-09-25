import math

from gareus.layout_neighbours import (
    other_rung_same_centre, same_rung_neighbours, spatial_neighbour_pairs,
)


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
