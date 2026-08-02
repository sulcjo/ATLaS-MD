"""Contact CV1 empirical-frontier range and confirmation tests."""

import pytest

from gareus.windows import _contact_effective_range, _frontier_probe_confirmed


def test_empirical_fold_pull_sets_ceiling_not_config_guess():
    lo, hi = _contact_effective_range(raw_lo=0.02, raw_hi=0.45, user_min=0.0, user_max=0.80)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.45)


def test_pull_may_expand_beyond_optional_safety_cap():
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


def test_frontier_probe_requires_sustained_hit_not_single_touch():
    held = _frontier_probe_confirmed([0.48, 0.51, 0.52, 0.47, 0.51], 0.50,
                                     min_hit_fraction=0.20, unreachable_deficit=0.08)
    touched = _frontier_probe_confirmed([0.30, 0.31, 0.50, 0.29, 0.30], 0.50,
                                        min_hit_fraction=0.40, unreachable_deficit=0.08)
    failed = _frontier_probe_confirmed([0.30, 0.31, 0.32], 0.50,
                                       min_hit_fraction=0.02, unreachable_deficit=0.08)
    assert held["confirmed"] is True
    assert touched["confirmed"] is False
    assert touched["reason"] == "insufficient_hold"
    assert failed["reason"] == "unreachable"


def test_frontier_probe_needs_two_hits_even_when_fraction_rounds_to_one_sample():
    verdict = _frontier_probe_confirmed(
        [0.30] * 49 + [0.50], 0.50,
        min_hit_fraction=0.02, unreachable_deficit=0.08,
    )
    assert verdict["confirmed"] is False
    assert verdict["reason"] == "insufficient_hold"
    assert verdict["required_hits"] == 2


def test_contact_frontier_cli_maps_and_validates_probe_knobs():
    from gareus.cli import parse_args

    args = parse_args([
        "--seq", "AAAAAA", "--cv1", "contacts", "--window-mode", "adaptive",
        "--cv1-frontier", "--cv1-frontier-probe-spacing", "0.03",
        "--cv1-frontier-confirm-rounds", "3", "--cv1-frontier-min-hit-fraction", "0.10",
        "--cv1-frontier-unreachable-deficit", "0.06",
    ])
    assert args.contact_frontier_enabled is True
    assert args.contact_frontier_probe_spacing == pytest.approx(0.03)
    assert args.contact_frontier_confirm_rounds == 3
    assert args.contact_frontier_min_hit_fraction == pytest.approx(0.10)
    assert args.contact_frontier_unreachable_deficit == pytest.approx(0.06)

    with pytest.raises(ValueError, match="frontier-probe-spacing"):
        parse_args([
            "--seq", "AAAAAA", "--cv1", "contacts", "--window-mode", "adaptive",
            "--cv1-frontier", "--cv1-frontier-probe-spacing", "0",
        ])
