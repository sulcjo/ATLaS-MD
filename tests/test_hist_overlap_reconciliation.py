import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gareus.diagnostics import _hist_overlap_np
from gareus.math_helpers import _adaptive_hist_overlap


def test_diagnostics_floor_still_returns_nan_below_five_samples():
    # 3 identical-valued samples per side: below _hist_overlap_np's
    # pre-existing 5-sample floor -> NaN, even though the underlying
    # histograms would show perfect (1.0) overlap.
    a = [1.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0]
    assert np.isnan(_hist_overlap_np(a, b, 0.0, 2.0))


def test_math_helpers_default_has_no_five_sample_floor():
    # Same 3-sample input, default min_samples=0 (unchanged
    # adaptive_feedback.py behavior): full overlap, not NaN.
    a = [1.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0]
    assert _adaptive_hist_overlap(a, b, 0.0, 2.0) == pytest.approx(1.0)


def test_math_helpers_explicit_min_samples_matches_diagnostics_floor():
    a = [1.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0]
    assert np.isnan(_adaptive_hist_overlap(a, b, 0.0, 2.0, min_samples=5))


def test_both_entry_points_agree_bit_for_bit_above_the_floor():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50)
    b = rng.normal(0.2, 1, 50)
    via_diagnostics = _hist_overlap_np(a, b, -3, 3, bins=40)
    via_math_helpers = _adaptive_hist_overlap(a, b, -3, 3, bins=40, min_samples=5)
    assert via_diagnostics == via_math_helpers


def test_adaptive_feedback_default_call_shape_unaffected():
    # Mirrors gareus/adaptive_feedback.py's real call shape: positional
    # a, b, lo, hi, no min_samples argument at all.
    a = [0.1, 0.2, 0.3, 0.4, 0.5]
    b = [0.15, 0.25, 0.35, 0.45, 0.55]
    result = _adaptive_hist_overlap(a, b, 0.0, 1.0)
    assert np.isfinite(result)
