"""Regression tests for two NaN-handling bugs in bias reconstruction, found by
a physics/math correctness audit of `analyze_gareus_mbar.py`.

Bug 1 -- `_compute_u_nk_analytical` (legacy `adaptive_feedback_round_*`
augmentation path): the secondary-restraint term
`0.5 * w['secondary_k_kcal'] * dc2 ** 2` was computed unconditionally. A
window with no secondary restraint stores `secondary_k_kcal` as NaN (this
file's own convention for "no restraint" -- see `load_csv`'s
`has_secondary`/`isfinite` guard), so `0.5 * NaN * dc2 ** 2 == NaN` poisoned
the *entire* bias row for every sample of that window. `clean()` drops any
row with a NaN anywhere in `u_nk`, so one no-secondary-restraint window could
silently wipe the whole augmented dataset. Fixed by guarding the secondary
term with the same `isfinite(center) and isfinite(k) and k > 0` pattern
already used by `load_csv` and `_reconstruct_union_bias_block`.

Bug 2 -- `_reconstruct_union_bias_block` (per-epoch union-MBAR bias
reconstruction): `d2 = np.where(np.isfinite(cv2), cv2 - sec_centers[k], 0.0)`
fabricated a deviation of exactly 0.0 -- "assume the sample is exactly
on-target" -- for any sample whose secondary CV was never measured (NaN),
instead of letting the NaN propagate so `clean()` excludes just that sample.
Every other bias-reconstruction site in this file (`load_csv`,
`load_epoch_csv_adaptive`, `_compute_u_nk_analytical`) lets NaN propagate.
Fixed by removing the per-sample `np.where` fallback; the correct
per-window guard (`math.isfinite(sec_centers[k]) and sec_ks[k] > 0`, which
governs whether the window has a secondary restraint AT ALL) is untouched.

Also: the hardcoded literal `4.184` in `_reconstruct_union_bias_block` was
replaced with the module constant `KJ_PER_KCAL` for consistency (numerically
identical today; removes a latent drift risk).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import (
    KJ_PER_KCAL,
    _compute_u_nk_analytical,
    _reconstruct_union_bias_block,
)


# --- Bug 1: _compute_u_nk_analytical ----------------------------------------

def test_no_secondary_restraint_window_produces_finite_secondary_free_bias():
    """A window with NaN secondary_k_kcal (this file's 'no restraint' sentinel)
    must not poison the whole row with NaN -- the reconstructed bias should
    equal the primary-only harmonic term.
    """
    beta = 0.4
    cv1 = np.array([0.0, 1.0, 2.0])
    cv2 = np.array([5.0, -3.0, np.nan])  # secondary values irrelevant/unmeasured
    union_windows = [{
        'primary_center': 0.0, 'primary_k_kcal': 10.0,
        'secondary_cv_center': np.nan, 'secondary_k_kcal': np.nan,
    }]

    u = _compute_u_nk_analytical(cv1, cv2, union_windows, beta)

    assert np.all(np.isfinite(u))
    d1 = cv1 - 0.0
    expected = beta * KJ_PER_KCAL * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(u[:, 0], expected)


def test_zero_secondary_k_also_produces_finite_secondary_free_bias():
    """secondary_k_kcal == 0.0 (finite, but non-positive) must also skip the
    secondary term rather than just guarding against NaN specifically.
    """
    beta = 0.4
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([100.0, -100.0])  # would blow up if secondary term applied
    union_windows = [{
        'primary_center': 0.0, 'primary_k_kcal': 10.0,
        'secondary_cv_center': 0.0, 'secondary_k_kcal': 0.0,
    }]

    u = _compute_u_nk_analytical(cv1, cv2, union_windows, beta)

    assert np.all(np.isfinite(u))
    d1 = cv1 - 0.0
    expected = beta * KJ_PER_KCAL * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(u[:, 0], expected)


def test_real_secondary_restraint_still_applies_the_term():
    """A window that DOES have a real secondary restraint must still get the
    secondary term applied correctly -- the fix must not disable it globally.
    """
    beta = 0.4
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([-2.0, -2.0])
    union_windows = [{
        'primary_center': 0.0, 'primary_k_kcal': 10.0,
        'secondary_cv_center': -2.1337, 'secondary_k_kcal': 11.87,
    }]

    u = _compute_u_nk_analytical(cv1, cv2, union_windows, beta)

    d1 = cv1 - 0.0
    d2 = cv2 - (-2.1337)
    expected = (beta * KJ_PER_KCAL * 0.5 * 10.0 * d1 ** 2
                + beta * KJ_PER_KCAL * 0.5 * 11.87 * d2 ** 2)
    np.testing.assert_allclose(u[:, 0], expected)


def test_finite_positive_k_with_nan_center_still_produces_finite_bias():
    """Deliberate deviation from the audit's suggested fix: the guard checks
    isfinite(secondary_cv_center) in addition to isfinite/positive
    secondary_k_kcal (mirroring `load_csv`'s full guard, which checks both
    fields). A window with a real, positive k but a NaN/missing center would
    otherwise still poison the row via `dc2 = cv2 - nan == nan`.
    """
    beta = 0.4
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([5.0, -3.0])
    union_windows = [{
        'primary_center': 0.0, 'primary_k_kcal': 10.0,
        'secondary_cv_center': np.nan, 'secondary_k_kcal': 11.87,
    }]

    u = _compute_u_nk_analytical(cv1, cv2, union_windows, beta)

    assert np.all(np.isfinite(u))
    d1 = cv1 - 0.0
    expected = beta * KJ_PER_KCAL * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(u[:, 0], expected)


def test_mixed_windows_one_restrained_one_not_isolated_correctly():
    """A NaN secondary_k_kcal in one union window must not affect another
    window's column in the same call (regression against any accidental
    cross-window mutation/broadcast bug).
    """
    beta = 0.4
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([-2.0, 3.0])
    union_windows = [
        {'primary_center': 0.0, 'primary_k_kcal': 10.0,
         'secondary_cv_center': np.nan, 'secondary_k_kcal': np.nan},
        {'primary_center': 0.5, 'primary_k_kcal': 20.0,
         'secondary_cv_center': 1.0, 'secondary_k_kcal': 5.0},
    ]

    u = _compute_u_nk_analytical(cv1, cv2, union_windows, beta)

    assert np.all(np.isfinite(u))
    d1_0 = cv1 - 0.0
    expected_col0 = beta * KJ_PER_KCAL * 0.5 * 10.0 * d1_0 ** 2
    np.testing.assert_allclose(u[:, 0], expected_col0)

    d1_1 = cv1 - 0.5
    d2_1 = cv2 - 1.0
    expected_col1 = (beta * KJ_PER_KCAL * 0.5 * 20.0 * d1_1 ** 2
                      + beta * KJ_PER_KCAL * 0.5 * 5.0 * d2_1 ** 2)
    np.testing.assert_allclose(u[:, 1], expected_col1)


# --- Bug 2: _reconstruct_union_bias_block -----------------------------------

def test_nan_cv2_under_restrained_window_produces_nan_for_that_sample_only():
    """A sample with an unmeasured (NaN) secondary CV, under a window that DOES
    restrain the secondary CV, must get NaN bias for THAT SAMPLE (so `clean()`
    excludes it) -- not a fabricated zero deviation. A neighboring sample with
    a valid cv2 value must be completely unaffected.
    """
    beta = 0.4
    cv = np.array([0.0, 0.0])
    cv2 = np.array([np.nan, -2.0])  # sample 0 unmeasured, sample 1 valid
    pc = np.array([0.0]); pk = np.array([10.0])
    sc = np.array([-2.1337]); sk = np.array([11.87])

    u = _reconstruct_union_bias_block(cv, cv2, beta, pc, pk, sc, sk)

    assert math.isnan(u[0, 0])
    assert math.isfinite(u[1, 0])
    d1 = cv[1] - pc[0]
    d2 = cv2[1] - sc[0]
    expected = beta * KJ_PER_KCAL * 0.5 * pk[0] * d1 * d1 + beta * KJ_PER_KCAL * 0.5 * sk[0] * d2 * d2
    assert u[1, 0] == pytest.approx(expected)


def test_nan_cv2_under_unrestrained_window_still_contributes_zero_secondary_bias():
    """A window with a real, finite secondary_cv_center but a non-restraining
    (zero) secondary_k -- the actual shape `_build_union_window_table`/the
    registry produce for an unrestrained state (center recorded, k defaults
    to 0.0, not NaN) -- must still produce finite, secondary-term-free bias
    regardless of cv2. This is the pre-existing, correct per-WINDOW guard
    (`math.isfinite(sec_centers[k]) and sec_ks[k] > 0`) and must be untouched
    by the per-sample NaN-propagation fix. Using a NaN center here instead
    would let the guard's `isfinite(center)` clause short-circuit before ever
    exercising the `k > 0` half.
    """
    beta = 0.4
    cv = np.array([0.0, 1.0])
    cv2 = np.array([np.nan, np.nan])
    pc = np.array([0.0]); pk = np.array([10.0])
    sc = np.array([-2.1337]); sk = np.array([0.0])

    u = _reconstruct_union_bias_block(cv, cv2, beta, pc, pk, sc, sk)

    assert np.all(np.isfinite(u))
    d1 = cv - pc[0]
    expected = beta * KJ_PER_KCAL * 0.5 * pk[0] * d1 * d1
    np.testing.assert_allclose(u[:, 0], expected)


def test_all_nan_cv2_under_restrained_window_yields_all_nan_column():
    """Aggregate/edge case: if EVERY sample's cv2 is NaN under a window that
    does restrain the secondary CV, the whole column is NaN (every sample in
    that state gets excluded by `clean()`) -- this is the correct, loud
    failure mode replacing the old silent wrong-zero-bias behavior. Documented
    here so the aggregate consequence (potential large sample-count drop for
    a state with poor secondary-CV coverage) is explicit and pinned by a test,
    not just the single-sample case above.
    """
    beta = 0.4
    cv = np.array([0.0, 0.1, -0.1])
    cv2 = np.array([np.nan, np.nan, np.nan])
    pc = np.array([0.0]); pk = np.array([10.0])
    sc = np.array([-2.1337]); sk = np.array([11.87])

    u = _reconstruct_union_bias_block(cv, cv2, beta, pc, pk, sc, sk)

    assert np.all(np.isnan(u[:, 0]))
