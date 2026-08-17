import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.diagnostics as gd
import analyze_gareus_mbar as agm


def test_relocated_functions_are_the_same_object_not_a_copy():
    assert agm._weighted_mean_std is gd._weighted_mean_std
    assert agm._pmf_distribution_mean_std is gd._pmf_distribution_mean_std


def test_weighted_mean_std_known_answer():
    mean, std = gd._weighted_mean_std([1, 2, 3], [1, 1, 1])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx((2.0 / 3.0) ** 0.5)


def test_pmf_distribution_mean_std_agrees_with_weighted_mean_std_for_uniform_prob():
    # Uniform prob over [1,2,3] is the same distribution as uniform weights
    # over [1,2,3] -- the two functions must agree exactly.
    pmf = {'cv_A': [1, 2, 3], 'prob': [1, 1, 1]}
    mean, std = gd._pmf_distribution_mean_std(pmf)
    mean_w, std_w = gd._weighted_mean_std([1, 2, 3], [1, 1, 1])
    assert mean == pytest.approx(mean_w)
    assert std == pytest.approx(std_w)


def test_weighted_mean_std_all_zero_weight_is_nan():
    mean, std = gd._weighted_mean_std([1, 2, 3], [0, 0, 0])
    assert np.isnan(mean)
    assert np.isnan(std)


def test_pmf_distribution_mean_std_empty_is_nan():
    mean, std = gd._pmf_distribution_mean_std({'cv_A': [], 'prob': []})
    assert np.isnan(mean)
    assert np.isnan(std)
