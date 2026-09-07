"""gamd-openmm refuses an integrator whose ntcmd/nteb constructor arguments are not multiples of
the averaging window (``ntcmd must be greater than and a multiple of ntave``).  That check fires
only after system build, pulling and seeding -- the S3 pilot lost a 7-minute pull to it.  The CLI
must reject such a config up front, with the offending pair named."""
from types import SimpleNamespace

from gareus.cli import validate_gamd_stage_multiples


def _args(**over):
    base = dict(
        gamd_cmd_steps=25000,
        gamd_equil_steps=25000,
        gamd_averaging_window=2500,
        gamd_multiwindow_recon_cmd_steps=5000,
        gamd_multiwindow_recon_steps=10000,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_valid_multiples_pass():
    validate_gamd_stage_multiples(_args())


def test_cmd_steps_not_multiple_of_window_is_rejected_naming_the_pair():
    try:
        validate_gamd_stage_multiples(_args(gamd_averaging_window=2000))
    except ValueError as exc:
        msg = str(exc)
        assert "gamd_cmd_steps" in msg and "25000" in msg and "2000" in msg
    else:
        raise AssertionError("25000 is not a multiple of 2000; must raise")


def test_recon_stages_are_not_constrained():
    # The recon stages step an integrator that already exists (plain cMD, or the
    # GaMD integrator seeded with copied globals); ntave only constrains the
    # constructor arguments ntcmd/nteb.  The tiny end-to-end test runs recon cMD
    # with 2 steps at ntave 50 on real MD -- that must stay legal.
    validate_gamd_stage_multiples(_args(gamd_multiwindow_recon_cmd_steps=2, gamd_multiwindow_recon_steps=7))


def test_equil_steps_not_multiple_is_rejected():
    try:
        validate_gamd_stage_multiples(_args(gamd_equil_steps=26000))
    except ValueError as exc:
        assert "gamd_equil_steps" in str(exc)
    else:
        raise AssertionError("26000 is not a multiple of 2500; must raise")


def test_zero_recon_stage_is_skipped():
    # A disabled recon stage (0 steps) is not a stage the integrator sees.
    validate_gamd_stage_multiples(_args(gamd_multiwindow_recon_cmd_steps=0, gamd_multiwindow_recon_steps=0))
