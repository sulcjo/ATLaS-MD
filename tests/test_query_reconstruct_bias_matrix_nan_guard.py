"""Regression tests for a NaN-guard bug in gareus.query.reconstruct_bias_matrix,
found by direct comparison against analyze_gareus_mbar.py's own two
bias-reconstruction copies (_compute_u_nk_analytical,
_reconstruct_union_bias_block), which a prior audit this session already
fixed for exactly this bug class -- see tests/test_bias_reconstruction_nan_handling.py.

Bug (window-side): the `if cv2 is not None and "center2" in w and "k2" in w`
guard only checked dict-key *presence*, never that the values were finite and
k2 > 0. A window with a real "center2"/"k2" key holding NaN (or a
non-positive k2) was completely unguarded and could NaN-poison, or wrongly
apply, a secondary term that should have been skipped entirely.

Bug (sample-side, the primary one): for a window that DOES have a real
secondary restraint, `d2 = np.where(np.isfinite(c2), c2-center2, 0.0)`
fabricated a deviation of exactly 0.0 -- "assume this sample was perfectly
on-target" -- for any sample whose secondary CV was never measured (NaN),
instead of letting NaN propagate so downstream code (e.g.
analyze_gareus_mbar.py's clean()) can exclude just that sample. This
silently kept a should-be-excluded sample in the pool with a fabricated,
too-low bias energy.

Fixed: window-side guard now checks isfinite(center2) and isfinite(k2) and
k2 > 0 before adding any secondary term at all; the np.where(..., 0.0)
sample-side fallback is removed so NaN propagates correctly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("gareus.query")

from gareus.query import reconstruct_bias_matrix


# --- window-side guard: a window with no REAL secondary restraint must
#     never poison or wrongly restrain, regardless of what the keys hold ---

def test_nan_k2_produces_finite_cv1_only_bias():
    cv1 = np.array([0.0, 1.0, 2.0])
    cv2 = np.array([5.0, -3.0, np.nan])  # secondary values irrelevant/unmeasured
    windows = [{"center1": 0.0, "k1": 10.0, "center2": -1.0, "k2": np.nan}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1 = cv1 - 0.0
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(nk[:, 0], expected)


def test_zero_k2_produces_finite_cv1_only_bias():
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([100.0, -100.0])  # would blow up if the secondary term applied
    windows = [{"center1": 0.0, "k1": 10.0, "center2": 0.0, "k2": 0.0}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1 = cv1 - 0.0
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(nk[:, 0], expected)


def test_negative_k2_produces_finite_cv1_only_bias():
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([2.0, -2.0])
    windows = [{"center1": 0.0, "k1": 10.0, "center2": 0.0, "k2": -5.0}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1 = cv1 - 0.0
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(nk[:, 0], expected)


def test_nan_center2_with_positive_finite_k2_produces_finite_cv1_only_bias():
    """Deliberate edge case: a real, positive k2 but a NaN/missing center is
    the OPPOSITE half of the guard from the k2 checks above -- both isfinite
    checks are required, neither alone is sufficient.
    """
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([5.0, -3.0])  # finite -- would previously NaN-poison via c2-NaN
    windows = [{"center1": 0.0, "k1": 10.0, "center2": np.nan, "k2": 11.87}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1 = cv1 - 0.0
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(nk[:, 0], expected)


# --- positive control: a real secondary restraint must still apply ---------

def test_real_secondary_restraint_still_applies_correctly():
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([-2.0, -2.0])
    windows = [{"center1": 0.0, "k1": 10.0, "center2": -2.1337, "k2": 11.87}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    d1 = cv1 - 0.0
    d2 = cv2 - (-2.1337)
    expected = (0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
                + 0.4 * 4.184 * 0.5 * 11.87 * d2 ** 2)
    np.testing.assert_allclose(nk[:, 0], expected)


# --- sample-side exclusion: the primary bug ---------------------------------

def test_nan_cv2_sample_under_restrained_window_produces_nan_for_that_sample_only():
    cv1 = np.array([0.0, 0.0])
    cv2 = np.array([np.nan, -2.0])  # sample 0 unmeasured, sample 1 valid
    windows = [{"center1": 0.0, "k1": 10.0, "center2": -2.1337, "k2": 11.87}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert math.isnan(nk[0, 0])
    assert math.isfinite(nk[1, 0])
    d1 = cv1[1] - 0.0
    d2 = cv2[1] - (-2.1337)
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 * d1 + 0.4 * 4.184 * 0.5 * 11.87 * d2 * d2
    assert nk[1, 0] == pytest.approx(expected)


def test_all_nan_cv2_under_restrained_window_yields_all_nan_column():
    cv1 = np.array([0.0, 0.1, -0.1])
    cv2 = np.array([np.nan, np.nan, np.nan])
    windows = [{"center1": 0.0, "k1": 10.0, "center2": -2.1337, "k2": 11.87}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isnan(nk[:, 0]))


def test_mixed_windows_one_restrained_one_not_isolated_correctly():
    """A no-restraint window's NaN k2 must not affect another window's
    column in the same call (regression against a cross-window broadcast bug)."""
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([-2.0, 3.0])
    windows = [
        {"center1": 0.0, "k1": 10.0, "center2": np.nan, "k2": np.nan},
        {"center1": 0.5, "k1": 20.0, "center2": 1.0, "k2": 5.0},
    ]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1_0 = cv1 - 0.0
    expected_col0 = 0.4 * 4.184 * 0.5 * 10.0 * d1_0 ** 2
    np.testing.assert_allclose(nk[:, 0], expected_col0)

    d1_1 = cv1 - 0.5
    d2_1 = cv2 - 1.0
    expected_col1 = (0.4 * 4.184 * 0.5 * 20.0 * d1_1 ** 2
                     + 0.4 * 4.184 * 0.5 * 5.0 * d2_1 ** 2)
    np.testing.assert_allclose(nk[:, 1], expected_col1)


def test_1d_window_with_no_center2_k2_keys_at_all_is_unaffected():
    """Baseline: a window dict with no secondary keys at all (the ordinary 1D
    case) must be completely untouched by the guard changes."""
    cv1 = np.array([0.1, 0.2, 0.3])
    windows = [{"center1": 0.15, "k1": 100.0}]

    nk = reconstruct_bias_matrix(cv1, None, windows, beta=1.0 / (8.314462618e-3 * 300.0))

    beta = 1.0 / (8.314462618e-3 * 300.0)
    expected = beta * 4.184 * 0.5 * 100.0 * (cv1 - 0.15) ** 2
    np.testing.assert_allclose(nk[:, 0], expected, rtol=1e-6)
