"""Default GaMD boost type must be C1-safe (group boost, not total boost).

Scientific-audit finding C1: the total-boost portion of `lower-dual` reads the
full system potential, which includes the umbrella restraint (gamd-openmm's
set_all_forces_to_group(system, 0) folds the umbrella into the boosted group).
That makes the boost window-dependent and misspecifies MBAR + breaks REUS
detailed balance. `lower-dual-nonbonded-dihedral` boosts only the nonbonded and
dihedral groups, leaving the umbrella (and secondary restraint) unboosted, so
u_nk = beta*w_k is a valid MBAR spec and umbrella-only exchange is correct.

Therefore the DEFAULT must be the group-boost variant, so configs that don't set
`gamd_boost_type` (e.g. the flagship) are C1-safe.
"""

SAFE = "lower-dual-nonbonded-dihedral"


def test_argparse_default_gamd_boost_type_is_c1_safe():
    from gareus.cli import build_gareus_parser
    assert build_gareus_parser().get_default("gamd_boost_type") == SAFE


def test_config_template_gamd_boost_type_is_c1_safe():
    from gareus.config import _basic_chignolin_config
    assert _basic_chignolin_config()["gamd"]["gamd_boost_type"] == SAFE


def test_gamd_multiwindow_recon_flags_have_expected_defaults():
    from gareus.cli import build_gareus_parser
    parser = build_gareus_parser()
    assert parser.get_default("gamd_multiwindow_recon_prep_steps") == 2000
    assert parser.get_default("gamd_multiwindow_recon_cmd_steps") == 20000
    assert parser.get_default("gamd_multiwindow_recon_steps") == 20000
    assert parser.get_default("gamd_multiwindow_recon_report_interval") == 0


def test_gamd_boosted_calibration_flags_default_on():
    from gareus.cli import build_gareus_parser
    parser = build_gareus_parser()
    # Boosted self-consistent calibration is on by default.
    assert parser.get_default("gamd_recon_boosted_iters") == 4
    assert parser.get_default("gamd_recon_boosted_tol") == 0.05


def test_gamd_multiwindow_recon_flags_are_overridable():
    from gareus.cli import build_gareus_parser
    parser = build_gareus_parser()
    args = parser.parse_args([
        "--seq", "GYDPETGTWG",
        "--gamd-multiwindow-recon-prep-steps", "500",
        "--gamd-multiwindow-recon-cmd-steps", "3000",
        "--gamd-multiwindow-recon-steps", "5000",
        "--gamd-multiwindow-recon-report-interval", "100",
        "--gamd-recon-boosted-iters", "2",
        "--gamd-recon-boosted-tol", "0.1",
    ])
    assert args.gamd_multiwindow_recon_prep_steps == 500
    assert args.gamd_multiwindow_recon_cmd_steps == 3000
    assert args.gamd_multiwindow_recon_steps == 5000
    assert args.gamd_multiwindow_recon_report_interval == 100
    assert args.gamd_recon_boosted_iters == 2
    assert args.gamd_recon_boosted_tol == 0.1


def test_sigma0d_warns_for_single_boost_types(capsys):
    """sigma0d only reaches the upstream gamd package's dual-boost integrators
    (integrator_factory.get_integrator forwards sigma0p alone to every
    single-boost create_*_integrator function) -- setting it for a
    single-boost type is always a no-op, so this should warn, not silently
    calibrate off the (possibly wrong) sigma0p default instead."""
    from gareus.cli import parse_args

    for boost_type in (
        "gamd-cmd-base", "lower-total", "upper-total",
        "lower-dihedral", "upper-dihedral",
        "lower-nonbonded", "upper-nonbonded",
    ):
        parse_args(["--seq", "DPETG", "--gamd-boost-type", boost_type, "--sigma0d", "2.0"])
        out = capsys.readouterr().out
        assert "sigma0d" in out and "silently ignored" in out, boost_type


def test_sigma0d_no_warning_for_dual_boost_types(capsys):
    from gareus.cli import parse_args

    for boost_type in (
        "lower-dual", "upper-dual",
        "lower-dual-nonbonded-dihedral", "upper-dual-nonbonded-dihedral",
    ):
        parse_args(["--seq", "DPETG", "--gamd-boost-type", boost_type, "--sigma0d", "2.0"])
        out = capsys.readouterr().out
        assert "sigma0d" not in out, boost_type


def test_sigma0d_no_warning_when_left_at_default(capsys):
    from gareus.cli import parse_args

    parse_args(["--seq", "DPETG", "--gamd-boost-type", "lower-dihedral"])
    out = capsys.readouterr().out
    assert "sigma0d" not in out
