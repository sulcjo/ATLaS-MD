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
    assert parser.get_default("gamd_multiwindow_recon_steps") == 20000
    assert parser.get_default("gamd_multiwindow_recon_report_interval") == 0


def test_gamd_multiwindow_recon_flags_are_overridable():
    from gareus.cli import build_gareus_parser
    parser = build_gareus_parser()
    args = parser.parse_args([
        "--seq", "GYDPETGTWG",
        "--gamd-multiwindow-recon-prep-steps", "500",
        "--gamd-multiwindow-recon-steps", "5000",
        "--gamd-multiwindow-recon-report-interval", "100",
    ])
    assert args.gamd_multiwindow_recon_prep_steps == 500
    assert args.gamd_multiwindow_recon_steps == 5000
    assert args.gamd_multiwindow_recon_report_interval == 100
