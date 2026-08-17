import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.diagnostics as gd
import analyze_gareus_mbar as agm


def test_relocated_functions_are_the_same_object_not_a_copy():
    # `is`, not `==` -- proves the import actually replaced the local
    # definition rather than merely producing a coincidentally-equal copy.
    assert agm.pmf_probability is gd.pmf_probability
    assert agm.js_divergence_1d is gd.js_divergence_1d
    assert agm.pmf_rmse_1d is gd.pmf_rmse_1d
    assert agm.barrier_error_1d is gd.barrier_error_1d


def test_pmf_probability_normalizes_and_zero_fills_non_finite():
    out = gd.pmf_probability({'prob': [1, 2, 3, float('nan'), -1]})
    np.testing.assert_allclose(out, [0.2, 0.4, 0.6, 0.0, 0.0])


def test_js_divergence_identical_distributions_is_zero():
    assert gd.js_divergence_1d([1, 2, 3], [1, 2, 3]) == 0.0


def test_js_divergence_disjoint_distributions_is_log2():
    # P=[1,0] and Q=[0,1] normalize to two disjoint point masses; JS
    # divergence between disjoint distributions is exactly ln(2).
    js = gd.js_divergence_1d([1, 0], [0, 1])
    assert js == pytest.approx(np.log(2), abs=1e-12)


def test_pmf_rmse_constant_offset_gives_exact_rmse():
    F = np.array([1.0, 3.0])
    Fref = F + 2.0
    P = np.array([0.5, 0.5])
    assert gd.pmf_rmse_1d(F, Fref, P, P) == pytest.approx(2.0, abs=1e-12)


def test_pmf_rmse_empty_after_masking_is_nan():
    F = np.array([1.0, 3.0])
    P = np.array([0.0, 0.0])  # nothing passes P > min_prob
    assert np.isnan(gd.pmf_rmse_1d(F, F, P, P))


def test_barrier_error_is_abs_difference_of_masked_maxima():
    F = np.array([5.0])
    Fref = np.array([3.0])
    P = np.array([1.0])
    assert gd.barrier_error_1d(F, Fref, P, P) == pytest.approx(2.0, abs=1e-12)
