import types
from gareus.adaptive_production import AdaptiveDecisionPolicy, policy_from_args


def test_dataclass_target_overlap_unified():
    assert AdaptiveDecisionPolicy().target_overlap == 0.30


def test_policy_from_args_defaults_to_030_when_unset():
    p = policy_from_args(types.SimpleNamespace())
    assert p.target_overlap == 0.30
