"""CLI wiring for the unbiased swarm stage: --swarm-* flags, --swarm-stage dispatch,
the public --shared-gamd-setup-dir flag, and the chignolin example config.

Fixture-free: every test is a plain zero-argument function (see
.superpowers/sdd/2026-09-07-swarm-stage/global-constraints.md's runner note).
"""


def test_swarm_flags_parse_with_defaults_and_budget():
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GYDPETGTWG", "--out", "/tmp/x", "--swarm-stage", "run", "--swarm-budget-ns", "120"])
    assert a.swarm_stage == "run" and a.swarm_seed_ns == 1.0 and a.swarm_budget_ns == 120.0
    assert a.swarm_bins == "4,3,3" and a.swarm_output_interval_ps == 2.0 and a.shared_gamd_setup_dir is None


def test_swarm_example_config_flattens_into_swarm_dests():
    from gareus.cli import parse_args
    a = parse_args(["--config", "examples/chignolin_swarm_stage.yaml"])
    assert a.swarm_stage == "run" and a.seq == "GYDPETGTWG" and a.gamd_boost_type == "pep-gamd-lower-dual"


def test_shared_gamd_setup_dir_survives_explicit_flag():
    """cli.py's _shim_gamd used to force args.shared_gamd_setup_dir = "" unconditionally,
    which would silently discard an explicitly-supplied --shared-gamd-setup-dir. The fix
    only defaults it to "" when the attribute is genuinely absent (a hand-built Namespace
    predating this flag); a real parse_args() call always has the attribute, so an
    explicit value survives untouched."""
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GYDPETGTWG", "--out", "/tmp/x", "--shared-gamd-setup-dir", "/x"])
    assert a.shared_gamd_setup_dir == "/x"


def test_swarm_stage_off_by_default():
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GYDPETGTWG", "--out", "/tmp/x"])
    assert a.swarm_stage == "off"


def test_swarm_all_flags_are_known_config_dests():
    """Every --swarm-* flag (and --shared-gamd-setup-dir) must be a real argparse dest so the
    YAML config loader's flattening (_flatten_config_mapping) accepts a matching section key --
    global-constraints.md's binding anchor on gareus/config.py."""
    from gareus.cli import build_gareus_parser
    from gareus.config import _build_known_config_dests
    known = _build_known_config_dests(build_gareus_parser())
    expected = {
        "swarm_stage", "swarm_seed_ns", "swarm_replicates_per_cell", "swarm_budget_ns",
        "swarm_bins", "swarm_equil_ps", "swarm_output_interval_ps", "swarm_seed_frame_interval_ps",
        "swarm_member_range", "swarm_round", "swarm_seed_source", "swarm_production_seed_csv",
        "swarm_n_windows", "swarm_overlap_sigma", "swarm_target_beta_sigma", "swarm_min_rungs",
        "swarm_max_rungs", "swarm_ess_floor", "swarm_seeds_per_window", "swarm_discard_block_frames",
        "swarm_min_discard_ps", "swarm_fsf_floor_warn", "swarm_pilot_globals", "shared_gamd_setup_dir",
    }
    missing = expected - known
    assert not missing, f"missing argparse dests: {sorted(missing)}"
