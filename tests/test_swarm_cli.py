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
        "swarm_graft_minimize_iters", "swarm_max_seed_gap_sigma",
        "swarm_stability_sigma_rel_tol", "swarm_stability_extrema_sigma_tol",
        "swarm_min_done_fraction", "swarm_max_graft_fallback_fraction",
    }
    missing = expected - known
    assert not missing, f"missing argparse dests: {sorted(missing)}"


def test_swarm_graft_minimize_iters_and_max_seed_gap_sigma_flags_parse_with_defaults():
    """Neither flag existed before this fix -- --swarm-graft-minimize-iters was unreachable
    from config or command line (members.py's getattr fell straight to the module constant),
    and --swarm-max-seed-gap-sigma is new for the autotuned CV1 upper-bound probe tolerance."""
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GYDPETGTWG", "--out", "/tmp/x"])
    assert a.swarm_graft_minimize_iters == 500
    assert a.swarm_max_seed_gap_sigma == 0.5


def test_swarm_graft_minimize_iters_and_max_seed_gap_sigma_override_from_cli():
    from gareus.cli import parse_args
    a = parse_args([
        "--seq", "GYDPETGTWG", "--out", "/tmp/x",
        "--swarm-graft-minimize-iters", "750", "--swarm-max-seed-gap-sigma", "0.25",
    ])
    assert a.swarm_graft_minimize_iters == 750
    assert a.swarm_max_seed_gap_sigma == 0.25


def test_swarm_graft_minimize_iters_and_max_seed_gap_sigma_settable_from_swarm_yaml_section():
    """Schema-v2 config loading is dest-name based, not section-name based (config.py's
    _flatten_config_mapping ignores nesting), so a ``swarm:`` YAML section reaches these
    dests automatically once the flags exist -- verified here, no extra plumbing needed."""
    import pathlib
    import tempfile
    from gareus.cli import parse_args
    cfg = pathlib.Path(tempfile.mkdtemp()) / "cfg.yaml"
    cfg.write_text(
        "seq: GYDPETGTWG\n"
        "out: /tmp/x\n"
        "swarm:\n"
        "  swarm_graft_minimize_iters: 800\n"
        "  swarm_max_seed_gap_sigma: 0.3\n"
    )
    a = parse_args(["--config", str(cfg)])
    assert a.swarm_graft_minimize_iters == 800
    assert a.swarm_max_seed_gap_sigma == 0.3


def test_swarm_gate_tolerance_flags_parse_with_documented_defaults():
    """analyze.py's evaluate_gates(...) call reads these four thresholds via
    getattr(args, "<dest>", <default>) -- until this fix, three of them had no
    argparse flag at all (same latent-bug class as swarm_graft_minimize_iters
    before its fix), so they were unreachable from the CLI or the schema-v2
    swarm: YAML section and silently stuck at their getattr default forever.
    swarm_ess_floor already had a flag and is untouched here."""
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GYDPETGTWG", "--out", "/tmp/x"])
    assert a.swarm_stability_sigma_rel_tol == 0.10
    assert a.swarm_stability_extrema_sigma_tol == 1.0
    assert a.swarm_min_done_fraction == 0.9
    assert a.swarm_max_graft_fallback_fraction == 0.10


def test_swarm_gate_tolerance_flags_override_from_cli():
    from gareus.cli import parse_args
    a = parse_args([
        "--seq", "GYDPETGTWG", "--out", "/tmp/x",
        "--swarm-stability-sigma-rel-tol", "0.2",
        "--swarm-stability-extrema-sigma-tol", "1.5",
        "--swarm-min-done-fraction", "0.8",
        "--swarm-max-graft-fallback-fraction", "0.15",
    ])
    assert a.swarm_stability_sigma_rel_tol == 0.2
    assert a.swarm_stability_extrema_sigma_tol == 1.5
    assert a.swarm_min_done_fraction == 0.8
    assert a.swarm_max_graft_fallback_fraction == 0.15


def test_swarm_gate_tolerance_flags_settable_from_swarm_yaml_section():
    """Schema-v2 config loading is dest-name based (config.py's _flatten_config_mapping
    ignores nesting), so a swarm: YAML section reaches these dests automatically once
    the flags exist -- verified here, no extra plumbing needed."""
    import pathlib
    import tempfile
    from gareus.cli import parse_args
    cfg = pathlib.Path(tempfile.mkdtemp()) / "cfg.yaml"
    cfg.write_text(
        "seq: GYDPETGTWG\n"
        "out: /tmp/x\n"
        "swarm:\n"
        "  swarm_stability_sigma_rel_tol: 0.25\n"
        "  swarm_stability_extrema_sigma_tol: 2.0\n"
        "  swarm_min_done_fraction: 0.75\n"
        "  swarm_max_graft_fallback_fraction: 0.2\n"
    )
    a = parse_args(["--config", str(cfg)])
    assert a.swarm_stability_sigma_rel_tol == 0.25
    assert a.swarm_stability_extrema_sigma_tol == 2.0
    assert a.swarm_min_done_fraction == 0.75
    assert a.swarm_max_graft_fallback_fraction == 0.2


def test_chignolin_swarm_stage_example_carries_the_validated_contact_cv_block():
    """Measured 2026-09-08 on 1500 real swarm frames: the CLI-default contact CV
    (r0=4.5 A, beta=6.0 /A) gives median 0.008, IQR 0.017, entropy 0.037, only 2
    resolvable windows (every k clamped to k_min). heavy atom-pairs, sequence
    separation >= 4, r0=12 A, beta=3.0 /A gives median 0.558, IQR 0.334, entropy
    0.907, span 0.065-0.905, and 25 resolvable windows at cv1_k_max=1200 -- this
    is the definition the example must pin explicitly (CLI defaults are not
    enough), because any production config used with the swarm's handoff must
    carry an IDENTICAL contact_cv block."""
    from gareus.cli import parse_args
    a = parse_args(["--config", "examples/chignolin_swarm_stage.yaml"])
    assert a.contact_scheme == "atom-pairs"
    assert a.contact_atom_selection == "heavy"
    assert a.contact_min_sequence_separation == 4
    assert a.contact_r0_a == 12.0
    assert a.contact_beta_a_inv == 3.0
    assert a.contact_normalize is True
    assert a.cv1_k_max == 1200.0


def test_helptext_has_a_swarm_stage_section():
    """Task 11's helptext section 18 renders as a real, jumpable encyclopedia topic.
    Matched by title text with a "<some number>." prefix, not a pinned literal number
    (gareus.helptext._parse_encyclopedia assigns topic numbers sequentially at render
    time from whatever headings are present, immune to unrelated heading edits -- see
    test_package_smoke.py's test_python_m_gareus_heavy_help for the same convention)."""
    import re
    from gareus.cli import build_gareus_parser
    from gareus.helptext import heavy_help_text
    text = heavy_help_text(build_gareus_parser())
    assert re.search(r"\d+\.\s+Unbiased swarm stage \(S0/S1\)", text)
    assert "--windows-2d-csv swarm/analysis/windows_lambda_ladder.csv" in text
    assert "--seed-conformers-dir swarm/analysis/seed_bank" in text
    assert "--shared-gamd-setup-dir swarm/analysis/shared_gamd_setup" in text
    assert "No native reference or folded-state label is used anywhere in this stage." in text
    # gates + extension rule
    for gate_name in ("coverage", "envelope_stability", "ladder_ess", "graft"):
        assert gate_name in text
    # pilot comparison tolerance
    assert "<= 25%" in text
    assert "[0.7, 1.4]" in text


def test_helptext_swarm_section_topic_lookup_renders_body():
    """`-hh <topic>` (keyword form) reaches the section body, not just the TOC entry --
    the controller's binding check ("python -m gareus -hh 18 ... renders it") verified
    via the keyword form since the encyclopedia's topic *numbers* are position-derived
    (see test above) and not stable across unrelated heading edits."""
    from gareus.cli import build_gareus_parser
    from gareus.helptext import heavy_help_text
    text = heavy_help_text(build_gareus_parser(), topic="Unbiased swarm stage")
    assert "ab initio exploration stage" in text
    assert "frozen Pep-GaMD" in text
