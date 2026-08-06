"""Post-switch tICA coverage-extension regression tests."""

import numpy as np

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


def test_tica_coverage_add_stiffens_secondary_k_relative_to_parent():
    """Extrapolated coverage windows must not just inherit the parent's secondary_k.

    Regression for chignolin_5 window 27: a tica_coverage window inherited its
    parent's secondary_k unchanged, was too soft to hold the newly-discovered
    (previously uncovered) target, and relaxed >10 sigma back into the
    parent's own basin instead.
    """
    registry = _registry()
    policy = AdaptiveDecisionPolicy()
    primary = np.concatenate([np.full(990, 0.5), np.full(20, 0.6)])
    tic1 = np.concatenate([np.full(990, 0.0), np.full(20, 3.0)])

    actions = _propose_tica_coverage_actions(registry, primary, tic1, policy)

    assert len(actions) == 1
    _, parent_id, params, _, _ = actions[0]
    parent = registry.get_state(parent_id)
    _target_primary, primary_k, _target_secondary, secondary_k = params

    assert primary_k == parent.primary_k
    assert secondary_k == parent.secondary_k * policy.coverage_k_stiffen_factor


def test_tica_coverage_k_stiffen_factor_is_configurable():
    """The stiffening amount is a policy knob, not a hardcoded constant."""
    registry = _registry()
    primary = np.concatenate([np.full(990, 0.5), np.full(20, 0.6)])
    tic1 = np.concatenate([np.full(990, 0.0), np.full(20, 3.0)])

    unstiffened = AdaptiveDecisionPolicy(coverage_k_stiffen_factor=1.0)
    actions = _propose_tica_coverage_actions(registry, primary, tic1, unstiffened)
    parent = registry.get_state(actions[0][1])
    assert actions[0][2][3] == parent.secondary_k

    registry = _registry()
    stiffened = AdaptiveDecisionPolicy(coverage_k_stiffen_factor=5.0)
    actions = _propose_tica_coverage_actions(registry, primary, tic1, stiffened)
    parent = registry.get_state(actions[0][1])
    assert actions[0][2][3] == parent.secondary_k * 5.0
