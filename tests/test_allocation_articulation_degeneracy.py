"""The articulation bonus must not fire when it cannot discriminate.

`frontier_bonus` exists to protect states that are load-bearing for MBAR
overlap connectivity -- an articulation point is a node whose removal splits
the exchange graph.  That is a real signal on a graph with structure.

On a *path* graph it is not a signal at all: every interior node is an
articulation point and only the two endpoints are not.  So on chignolin_7's
112-state ladder the term fired for exactly 110 states every epoch, before any
run data existed, and the two endpoints took a ~36% smaller top-up purely for
sitting at the ends of the chain.

Measured on chignolin_7 epoch_001 -- the two groups differed by nothing except
this term:

    2 states   score 2.660   extra 1,539,102   baseline; low_effective_sample_proxy
  110 states   score 4.160   extra 2,407,082   baseline; low_effective_sample_proxy;
                                               graph_bridge_state

and 4.160 - 2.660 = 1.500 = frontier_bonus exactly.  In epoch_002 the endpoints
had *fewer* samples than the interior (9,345 vs 12,817), so the one term that
does respond to run data ranked them higher -- and was overridden anyway.

A scoring term that partitions the same 112 states into the same 110 and 2
every epoch, regardless of the data, is not prioritising.  It is a fixed
penalty on whichever two states end the chain.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import _articulation_is_degenerate


def test_path_graph_articulation_is_degenerate():
    """chignolin_7's real shape: 110 of 112 are articulation points."""
    active = list(range(112))
    articulation = set(range(1, 111))
    assert _articulation_is_degenerate(articulation, active, 0.80) is True


def test_a_graph_with_real_structure_is_not_degenerate():
    """A few genuine bridges out of many states: the term still discriminates."""
    active = list(range(50))
    articulation = {7, 21, 34}
    assert _articulation_is_degenerate(articulation, active, 0.80) is False


def test_threshold_is_inclusive_at_the_configured_fraction():
    active = list(range(10))
    assert _articulation_is_degenerate(set(range(8)), active, 0.80) is True
    assert _articulation_is_degenerate(set(range(7)), active, 0.80) is False


def test_empty_articulation_set_is_never_degenerate():
    assert _articulation_is_degenerate(set(), list(range(10)), 0.80) is False


def test_no_active_states_is_never_degenerate():
    """Guards the division; an empty run must not raise here."""
    assert _articulation_is_degenerate(set(), [], 0.80) is False


def test_fraction_of_one_disables_the_guard_entirely():
    """1.0 means 'only skip if literally every state is an articulation point',
    which is the opt-out for anyone who wants the old behaviour back."""
    active = list(range(112))
    assert _articulation_is_degenerate(set(range(1, 111)), active, 1.0) is False
    assert _articulation_is_degenerate(set(active), active, 1.0) is True


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
