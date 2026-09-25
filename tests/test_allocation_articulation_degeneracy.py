"""Allocation-bonus policy knobs still load from config.

The per-state score allocator these knobs fed was removed (effective top-ups);
the fields stay readable from old configs and warn once if set non-default.
The helper that decided whether the articulation bonus could discriminate
(``_articulation_is_degenerate``) had no production caller left and was
deleted together with its own tests.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# --- the knobs themselves -----------------------------------------------------
# All four allocation bonuses were hardcoded dataclass defaults. policy_from_args
# mapped 42 other fields to config keys and not these, so there was no way to
# retune the scheduler without editing source.

import argparse

from gareus.adaptive_production import AdaptiveDecisionPolicy, policy_from_args


def _args(**kw):
    return argparse.Namespace(**kw)


def test_bonuses_default_to_the_previous_hardcoded_values():
    """Plumbing must not silently change behaviour for anyone not setting them."""
    p = policy_from_args(_args())
    assert p.frontier_bonus == 1.5
    assert p.weak_edge_bonus == 3.0
    assert p.low_sample_bonus == 2.0
    assert p.high_boost_bonus == 1.0


def test_each_bonus_is_settable_from_config():
    p = policy_from_args(_args(
        adaptive_production_frontier_bonus=0.25,
        adaptive_production_weak_edge_bonus=4.0,
        adaptive_production_low_sample_bonus=2.5,
        adaptive_production_high_boost_bonus=0.5,
    ))
    assert p.frontier_bonus == 0.25
    assert p.weak_edge_bonus == 4.0
    assert p.low_sample_bonus == 2.5
    assert p.high_boost_bonus == 0.5


def test_degenerate_fraction_is_settable_and_defaults_to_080():
    assert policy_from_args(_args()).articulation_degenerate_fraction == 0.80
    p = policy_from_args(_args(adaptive_production_articulation_degenerate_fraction=1.0))
    assert p.articulation_degenerate_fraction == 1.0


def test_policy_dataclass_still_constructs_with_no_arguments():
    """The dataclass is instantiated bare in several synth drivers."""
    p = AdaptiveDecisionPolicy()
    assert p.articulation_degenerate_fraction == 0.80
