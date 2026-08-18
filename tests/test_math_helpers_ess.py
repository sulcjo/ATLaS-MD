import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.math_helpers as gmh
import analyze_gareus_mbar as agm


def test_relocated_ess_is_the_same_object_not_a_copy():
    assert agm.ess is gmh.ess


def test_ess_uniform_weights_equals_sample_count():
    assert gmh.ess(np.ones(5)) == pytest.approx(5.0)


def test_ess_single_dominant_weight_is_one():
    assert gmh.ess(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_ess_empty_after_filtering_is_zero():
    assert gmh.ess(np.array([-1.0, float('nan'), -5.0])) == 0.0


def test_ess_used_by_thermodynamic_validity_tests_still_resolves():
    # Sanity check for the exact call pattern
    # tests/test_thermodynamic_validity_2d.py and friends use directly.
    w = np.array([0.5, 0.5, 0.5, 0.5])
    assert agm.ess(w) / w.size == pytest.approx(1.0)
