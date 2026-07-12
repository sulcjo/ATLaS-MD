"""Contact CV1 effective-range assembly (fix C: windows collapse to unfolded band).

The CV1 boundary pull runs two short (~ps) pulls to estimate the achievable
contact-fraction range.  The extended (min) direction equilibrates fast, but
folding (the max direction) is slow: a short pull under-reaches the native
contact fraction, and trusting it collapses every umbrella window into the
unfolded band so the folded basin is never sampled.

The effective max must therefore be floored at the user-configured
contact_adaptive_max (a short fold-pull may EXPAND beyond it, never SHRINK
below it), mirroring the existing lower-bound protection.
"""

import pytest

from gareus.windows import _contact_effective_range


def test_short_fold_pull_does_not_collapse_range_below_user_max():
    # Boundary pull only reached 0.113 (5 ps can't fold), user wants up to 0.80.
    lo, hi = _contact_effective_range(raw_lo=0.02, raw_hi=0.113, user_min=0.0, user_max=0.80)
    assert lo == pytest.approx(0.0)
    assert hi >= 0.80  # native-ward range preserved, not collapsed to 0.113


def test_pull_may_expand_beyond_user_max():
    # If the system genuinely reaches higher contacts, keep the larger range.
    lo, hi = _contact_effective_range(raw_lo=0.0, raw_hi=0.95, user_min=0.0, user_max=0.80)
    assert hi >= 0.95


def test_lower_bound_still_floored_at_user_min():
    lo, hi = _contact_effective_range(raw_lo=0.30, raw_hi=0.90, user_min=0.05, user_max=0.80)
    assert lo <= 0.05  # never push the lower bound above user config


def test_range_clamped_to_unit_interval():
    lo, hi = _contact_effective_range(raw_lo=-0.1, raw_hi=1.5, user_min=0.0, user_max=0.80, margin=0.1)
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0


def test_minimum_width_enforced():
    lo, hi = _contact_effective_range(raw_lo=0.40, raw_hi=0.40, user_min=0.40, user_max=0.40)
    assert hi - lo >= 0.05
