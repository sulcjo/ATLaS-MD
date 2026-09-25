from gareus.adaptive_production import AdaptiveDecisionPolicy, policy_from_args
from gareus.cli import build_gareus_parser, parse_args


def test_topups_are_off_by_default_and_opt_in():
    p = build_gareus_parser()
    assert p.parse_args(["--seq", "AA"]).ap_topups is False
    assert p.parse_args(["--seq", "AA", "--ap-topups"]).ap_topups is True


def test_topup_knobs_reach_the_policy():
    args = parse_args(["--seq", "AA", "--ap-topups", "--ap-topup-target-sigma", "0.2",
                       "--ap-topup-weak-overlap", "0.1", "--ap-topup-max-fraction", "0.25",
                       "--ap-topup-min-effect", "0.03", "--ap-topup-max-edge-attempts", "3"])
    pol = policy_from_args(args)
    assert pol.topups_enabled is True
    assert (pol.topup_target_sigma, pol.topup_weak_overlap, pol.topup_max_fraction) == (0.2, 0.1, 0.25)
    assert (pol.topup_min_effect, pol.topup_max_edge_attempts) == (0.03, 3)
    assert pol.topup_throughput_table == ((16.0, 3154.0), (59.0, 2300.0))


def test_policy_defaults_match_the_spec():
    pol = AdaptiveDecisionPolicy()
    assert pol.topups_enabled is False
    assert (pol.topup_target_sigma, pol.topup_weak_overlap, pol.topup_max_fraction) == (0.10, 0.15, 0.3)
    assert (pol.topup_min_effect, pol.topup_max_edge_attempts) == (0.05, 2)


def test_a_malformed_throughput_table_is_a_clear_error():
    import pytest
    with pytest.raises(SystemExit):
        parse_args(["--seq", "AA", "--ap-topup-throughput-table", "16-3154"])
