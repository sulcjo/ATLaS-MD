import types
from gareus.adaptive_production import AdaptiveDecisionPolicy, policy_from_args


def test_dataclass_target_overlap_unified():
    assert AdaptiveDecisionPolicy().target_overlap == 0.30


def test_policy_from_args_defaults_to_030_when_unset():
    p = policy_from_args(types.SimpleNamespace())
    assert p.target_overlap == 0.30


def test_retire_converged_on_by_default():
    assert AdaptiveDecisionPolicy().retire_converged is True
    p = policy_from_args(types.SimpleNamespace())
    assert p.retire_converged is True


def test_min_active_states_default_is_8():
    assert AdaptiveDecisionPolicy().min_active_states == 8
    p = policy_from_args(types.SimpleNamespace())
    assert p.min_active_states == 8


def test_min_active_states_honors_min_total_windows():
    # an explicit total-window floor raises the retirement floor accordingly
    p = policy_from_args(types.SimpleNamespace(min_total_windows=10))
    assert p.min_active_states == 10


def test_min_active_states_explicit_override_can_disable():
    p = policy_from_args(types.SimpleNamespace(adaptive_production_min_active_states=0))
    assert p.min_active_states == 0


def test_max_target_deviation_sigma_default_is_3():
    assert AdaptiveDecisionPolicy().max_target_deviation_sigma == 3.0
    p = policy_from_args(types.SimpleNamespace())
    assert p.max_target_deviation_sigma == 3.0


def test_max_target_deviation_sigma_explicit_override():
    p = policy_from_args(types.SimpleNamespace(adaptive_production_max_target_deviation_sigma=5.0))
    assert p.max_target_deviation_sigma == 5.0
