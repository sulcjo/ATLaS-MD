"""Post-switch tICA coverage-extension regression tests."""

import numpy as np
import pytest

from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    WindowStateRegistry,
    _apply_registry_actions,
    _propose_tica_coverage_actions,
)


def _registry():
    registry = WindowStateRegistry()
    registry.add_state(0.0, 40.0, -0.25, 20.0, epoch=0, source="seed")
    registry.add_state(1.0, 40.0, 0.25, 20.0, epoch=0, source="seed")
    return registry


def test_populated_uncovered_tica_cluster_adds_home_window_despite_normal_action_cap():
    """Catch populated tIC1 basin outside every secondary restraint center."""
    registry = _registry()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=0, min_samples_for_add=10_000)
    primary = np.concatenate([np.full(990, 0.5), np.full(20, 0.6)])
    tic1 = np.concatenate([np.full(990, 0.0), np.full(20, 3.0)])

    actions = _propose_tica_coverage_actions(registry, primary, tic1, policy)

    assert len(actions) == 1
    kind, parent, params, reason, metadata = actions[0]
    assert kind == "tica_coverage_add"
    assert parent in registry.active_state_ids()
    assert params[0] == 0.6
    assert params[2] == 3.0
    assert "uncovered" in reason
    assert metadata["population_count"] == 20


def test_covered_or_sparse_tica_samples_add_no_coverage_window():
    """Do not spend replicas on an existing home region or sparse outliers."""
    registry = _registry()
    policy = AdaptiveDecisionPolicy(min_samples_for_add=1)
    covered_primary = np.full(990, 0.5)
    covered_tic1 = np.zeros(990)
    sparse_primary = np.full(2, 0.6)
    sparse_tic1 = np.array([2.9, 3.1])

    assert _propose_tica_coverage_actions(
        registry, covered_primary, covered_tic1, policy) == []
    assert _propose_tica_coverage_actions(
        registry, np.concatenate([covered_primary, sparse_primary]),
        np.concatenate([covered_tic1, sparse_tic1]), policy) == []


def test_sparse_bridge_does_not_merge_two_dense_uncovered_basins():
    """Avoid placing one restraint in middle of two populated uncovered basins."""
    registry = _registry()
    primary = np.concatenate([
        np.full(990, 0.5), np.full(20, 0.4), np.full(12, 0.5), np.full(20, 0.6),
    ])
    tic1 = np.concatenate([
        np.zeros(990), np.full(20, -3.0), np.linspace(-2.8, -2.2, 12), np.full(20, -2.0),
    ])

    actions = _propose_tica_coverage_actions(registry, primary, tic1)

    assert len(actions) == 2
    assert sorted(action[2][2] for action in actions) == [-3.0, -2.0]


def test_wide_dense_uncovered_band_gets_multiple_home_windows():
    """One coverage-width state cannot home a band wider than two widths."""
    registry = _registry()
    primary = np.concatenate([np.full(990, 0.5), np.full(80, 0.5)])
    tic1 = np.concatenate([np.zeros(990), np.linspace(-3.2, -2.0, 80)])

    actions = _propose_tica_coverage_actions(registry, primary, tic1)

    assert len(actions) >= 2


def test_tica_coverage_action_records_distinct_auditable_state_source():
    """Keep coverage additions distinguishable from normal weak-edge actions."""
    registry = _registry()
    actions = _propose_tica_coverage_actions(
        registry, np.concatenate([np.full(990, 0.5), np.full(20, 0.6)]),
        np.concatenate([np.zeros(990), np.full(20, 3.0)]),
    )

    _apply_registry_actions(registry, actions, epoch=0)

    added = registry.active_states()[-1]
    assert added.source == "tica_coverage"
    assert added.metadata["population_count"] == 20


_KBT_KCAL_298K = 1.987204e-3 * 298.0  # matches _propose_tica_coverage_actions' default temperature_K


def test_tica_coverage_k_derived_from_observed_spread_via_equipartition():
    """Stiffening is data-derived (k = kT/Var of the new population), not a flat guess.

    Regression for chignolin_5 window 27: a tica_coverage window inherited its
    parent's secondary_k unchanged, was too soft to hold the newly-discovered
    (previously uncovered) target, and relaxed >10 sigma back into the
    parent's own basin instead.
    """
    registry = _registry()
    policy = AdaptiveDecisionPolicy()
    primary = np.concatenate([np.full(990, 0.5), np.full(20, 0.6)])
    cluster = np.linspace(2.85, 3.15, 20)
    tic1 = np.concatenate([np.full(990, 0.0), cluster])

    actions = _propose_tica_coverage_actions(registry, primary, tic1, policy)

    assert len(actions) == 1
    _, parent_id, params, _, metadata = actions[0]
    parent = registry.get_state(parent_id)
    _target_primary, primary_k, _target_secondary, secondary_k = params

    assert metadata["population_count"] == 20  # whole cluster formed one group, none dropped
    expected_k = _KBT_KCAL_298K / float(np.std(cluster, ddof=1)) ** 2
    assert secondary_k == pytest.approx(expected_k)
    assert secondary_k != parent.secondary_k  # actually moved, not a no-op
    assert primary_k == parent.primary_k  # primary axis untouched (it wasn't the failure mode)
    assert secondary_k >= parent.secondary_k  # never loosens below the parent's own k


def test_tica_coverage_k_capped_for_tight_but_nonzero_spread():
    """A very tight (but non-degenerate) cluster must not blow secondary_k up unbounded."""
    registry = _registry()
    policy = AdaptiveDecisionPolicy()
    primary = np.concatenate([np.full(990, 0.5), np.full(20, 0.6)])
    tic1 = np.concatenate([np.full(990, 0.0), np.linspace(2.98, 3.02, 20)])

    actions = _propose_tica_coverage_actions(registry, primary, tic1, policy)

    assert len(actions) == 1
    parent = registry.get_state(actions[0][1])
    secondary_k = actions[0][2][3]
    assert secondary_k == pytest.approx(parent.secondary_k * policy.coverage_k_stiffen_cap)


def test_tica_coverage_k_falls_back_to_parent_for_degenerate_zero_spread():
    """A perfectly repeated (zero-variance) sample carries no spread information.

    Must fall back to the parent's own k, not divide by zero / blow up to inf.
    """
    registry = _registry()
    primary = np.concatenate([np.full(990, 0.5), np.full(20, 0.6)])
    tic1 = np.concatenate([np.full(990, 0.0), np.full(20, 3.0)])

    actions = _propose_tica_coverage_actions(registry, primary, tic1)

    assert len(actions) == 1
    parent = registry.get_state(actions[0][1])
    secondary_k = actions[0][2][3]
    assert secondary_k == parent.secondary_k
    assert np.isfinite(secondary_k)


def test_tica_coverage_k_stiffen_cap_is_configurable():
    """The safety ceiling on the data-derived estimate is a policy knob."""
    registry = _registry()
    primary = np.concatenate([np.full(990, 0.5), np.full(20, 0.6)])
    tic1 = np.concatenate([np.full(990, 0.0), np.linspace(2.98, 3.02, 20)])

    tight_cap = AdaptiveDecisionPolicy(coverage_k_stiffen_cap=1.0)
    actions = _propose_tica_coverage_actions(registry, primary, tic1, tight_cap)
    parent = registry.get_state(actions[0][1])
    assert actions[0][2][3] == pytest.approx(parent.secondary_k)  # cap=1x -> no stiffening allowed

    registry = _registry()
    loose_cap = AdaptiveDecisionPolicy(coverage_k_stiffen_cap=1000.0)
    actions = _propose_tica_coverage_actions(registry, primary, tic1, loose_cap)
    parent = registry.get_state(actions[0][1])
    # cap=1000x is so loose the raw equipartition estimate wins uncapped, and
    # that raw estimate is nowhere near the 1000x ceiling for this cluster.
    assert parent.secondary_k < actions[0][2][3] < parent.secondary_k * 1000.0


def test_tica_coverage_k_stiffen_cap_below_one_never_loosens_below_parent():
    """Regression: np.clip(x, lo, hi) with hi < lo silently returns hi, not lo.

    A cap < 1.0 must not be able to push secondary_k below the parent's own k
    -- the docstring's invariant is "only ever tighten, never loosen."
    """
    registry = _registry()
    primary = np.concatenate([np.full(990, 0.5), np.full(20, 0.6)])
    tic1 = np.concatenate([np.full(990, 0.0), np.linspace(2.98, 3.02, 20)])

    sub_one_cap = AdaptiveDecisionPolicy(coverage_k_stiffen_cap=0.5)
    actions = _propose_tica_coverage_actions(registry, primary, tic1, sub_one_cap)
    parent = registry.get_state(actions[0][1])
    assert actions[0][2][3] == pytest.approx(parent.secondary_k)
