import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.diagnostics as gd
import analyze_gareus_mbar as agm


def test_relocated_functions_are_the_same_object_not_a_copy():
    assert agm.identify_basins_1d is gd.identify_basins_1d
    assert agm._compute_basin_populations is gd._compute_basin_populations


def test_scipy_signal_optional_import_flag_exists():
    # Whichever value it resolves to in this environment, the flag and the
    # (possibly-None) function reference must both exist as module attrs.
    assert hasattr(gd, "SCIPY_SIGNAL_AVAILABLE")
    assert hasattr(gd, "_scipy_find_peaks")


def _symmetric_double_well():
    cv_A = np.arange(7.0)
    F = np.array([4.0, 2.0, 0.0, 3.0, 0.0, 2.0, 4.0])
    return cv_A, F


def test_identify_basins_finds_exactly_two_basins_with_correct_boundaries():
    cv_A, F = _symmetric_double_well()
    basins = gd.identify_basins_1d(cv_A, F, min_depth_kcal=1.0)
    assert len(basins) == 2
    assert basins[0]['left_bin'] == 0
    assert basins[0]['right_bin'] == 3
    assert basins[0]['center_cv_A'] == pytest.approx(2.0)
    assert basins[0]['min_F_kcal'] == pytest.approx(0.0)
    assert basins[1]['left_bin'] == 4
    assert basins[1]['right_bin'] == 6
    assert basins[1]['center_cv_A'] == pytest.approx(4.0)


def test_identify_basins_partition_covers_full_range_with_no_gap_or_overlap():
    cv_A, F = _symmetric_double_well()
    basins = gd.identify_basins_1d(cv_A, F, min_depth_kcal=1.0)
    basins = sorted(basins, key=lambda b: b['left_bin'])
    assert basins[0]['left_bin'] == 0
    assert basins[-1]['right_bin'] == len(F) - 1
    for a, b in zip(basins, basins[1:]):
        assert b['left_bin'] == a['right_bin'] + 1  # no gap, no overlap


def test_identify_basins_too_few_points_returns_empty():
    assert gd.identify_basins_1d(np.array([0.0, 1.0]), np.array([1.0, 2.0])) == []


def test_compute_basin_populations_sums_to_one_and_matches_known_split():
    cv_A, F = _symmetric_double_well()
    basins = gd.identify_basins_1d(cv_A, F, min_depth_kcal=1.0)
    prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
    pops = gd._compute_basin_populations(prob, basins)
    assert pops == pytest.approx([0.4, 0.6], abs=1e-9)
    assert sum(pops) == pytest.approx(1.0, abs=1e-9)
