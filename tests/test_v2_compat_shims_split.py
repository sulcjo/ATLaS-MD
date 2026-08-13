"""Tests for the _apply_v2_compat_shims decomposition.

_apply_v2_compat_shims used to be one 340-line function; it's now an orchestrator
calling one _shim_* helper per subsystem plus _validate_config_sanity. These tests
pin two things:
  1. The split is behavior-preserving: parse_args on a known argv still produces the
     same full set of shimmed attribute values as before the refactor.
  2. The bug the split surfaced is actually fixed: --flush-every-log used to be
     unconditionally forced to True after argparse ran, silently defeating both the
     flag's own documented default (False) and --no-flush-every-log.

genpept_prior_enabled is asserted to stay False - it is deliberately retired
(see _shim_genpept_prior_disabled's docstring), not something this test expects
to ever become reachable via CLI.

No OpenMM / PeptideBuilder imports (unit-test rule).
"""
import argparse

import pytest

import gareus.cli as cli

BASE = ["--seq", "AAAAA", "--cv1", "contacts", "--cv2", "rama-map"]


def _parsed(extra_argv=()):
    return cli.parse_args([*BASE, *extra_argv])


def test_shim_helpers_exist_and_are_independently_callable():
    names = [
        "_shim_cv",
        "_shim_cv1_legacy_names",
        "_shim_us_pulling",
        "_shim_windows",
        "_shim_seeding",
        "_shim_genpept_prescan",
        "_shim_genpept_prior_disabled",
        "_shim_gamd",
        "_shim_adaptive_production",
        "_shim_output",
        "_shim_system",
        "_shim_misc",
        "_validate_config_sanity",
    ]
    for name in names:
        fn = getattr(cli, name)
        assert callable(fn)
        # Re-running a shim helper on an already-fully-shimmed args must be
        # idempotent (no raise) - proves each one is a standalone unit, not a
        # fragment that only works mid-way through the original inline sequence.
        args = _parsed()
        fn(args)


def test_flush_every_log_default_is_false_not_forced_true():
    args = _parsed()
    assert args.flush_every_log is False


def test_flush_every_log_explicit_true_is_honoured():
    args = _parsed(["--flush-every-log"])
    assert args.flush_every_log is True


def test_flush_every_log_explicit_false_is_honoured():
    args = _parsed(["--no-flush-every-log"])
    assert args.flush_every_log is False


def test_us_2d_distance_fraction_default_matches_historical_value():
    # --us-2d-distance-fraction did not exist as a CLI flag until this test was
    # added: _shim_us_pulling unconditionally hardcoded
    # args.us_2d_start_distance_fraction = 0.50 regardless of any config,
    # silently defeating the --us-2d-start-distance-fraction tuning advice
    # cv_discovery.py's own diagnostics have long recommended. This pins the
    # historical default so introducing the real flag doesn't change behavior
    # for anyone who never sets it.
    args = _parsed()
    assert args.us_2d_start_distance_fraction == 0.50


def test_us_2d_distance_fraction_is_actually_configurable():
    args = _parsed(["--us-2d-distance-fraction", "0.05"])
    assert args.us_2d_start_distance_fraction == pytest.approx(0.05)


def test_genpept_prior_stays_deliberately_disabled():
    args = _parsed()
    assert args.genpept_prior_enabled is False
    # No CLI flag exists to flip this - confirm the parser has none.
    assert not any(
        "genpept-prior" in opt
        for action in cli.build_gareus_parser()._actions
        for opt in action.option_strings
    )


@pytest.mark.parametrize(
    "attr,expected",
    [
        # CV
        ("secondary_cv", "rama-map"),
        ("secondary_cv_force_group", 29),
        # CV1 legacy names (distance-mode + contact-mode + dropped tuning)
        ("adaptive_min_windows", 4),
        ("contact_adaptive_min_sigma", 0.02),
        ("umbrella_force_group", 31),
        # US pulling
        ("contact_us_pull_timestep_fs", 1.0),
        # Windows
        ("explicit_2d_exchange_slots", 4),
        # Seeding
        ("seed_secondary_weight", 1.0),
        # GENPEPT prescan (live feature)
        ("genpept_prescan_output_prefix", "genpept_prescan"),
        # GaMD
        ("gamd_cmd_prep_steps", 5000),
        # Adaptive production
        ("adaptive_production_allocation_scheduler", True),
        # Output
        ("csv_flush_rows", 1000),
        # System
        ("nvt_start_temperature_k", 50.0),
        # Misc
        ("cv_mode", "terminal-ca"),
    ],
)
def test_full_shim_attribute_set_preserved(attr, expected):
    args = _parsed()
    assert getattr(args, attr) == expected
