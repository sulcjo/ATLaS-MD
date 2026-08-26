"""Manifest checks for the per-system validation script bundles.

Everything these tests look at lives under ``RUNS/``, which ``.gitignore``
excludes wholesale: line 20 is a bare ``RUNS/`` and the file carries no
re-including negation for anything beneath it, so no path under
``RUNS/validation/`` is ever committed.  (A helper script or two sitting
directly under ``RUNS/`` has been force-added past that rule over time; the
validation bundles never have.  Deliberately not enumerated here - naming them
is what made the previous version of this docstring go stale.)  So the bundles
are local working-tree artifacts, not committed code, and a *committed* test that
asserts their existence unconditionally passes or fails per machine and per
checkout - which is exactly what happened: on a fresh clone all four of these
failed, while in the author's tree two passed only because those two
directories happened to be present.

Hence: skip cleanly when a bundle's directory is absent (nothing to check on
this machine), and check it thoroughly when it is present.  Please do not
"fix" these skips by re-adding unconditional asserts - the skip is the point.
"""

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROOT / "RUNS" / "validation"


def _validation_dir_or_skip(system: str) -> Path:
    """Return ``RUNS/validation/<system>``, or skip if it is not present here."""
    validation_dir = VALIDATION_ROOT / system
    if not validation_dir.is_dir():
        pytest.skip(
            f"{validation_dir.relative_to(ROOT)} is absent - the whole RUNS/ tree "
            "is gitignored, so this bundle only exists on machines where it was "
            "generated. Nothing to validate here."
        )
    return validation_dir


def _assert_validation_bundle(system: str, scripts: set[str]) -> None:
    """Every named script exists, the launcher runs each one, checklist present."""
    validation_dir = _validation_dir_or_skip(system)
    launcher = validation_dir / "run_all_validations.sh"

    assert launcher.exists()
    launcher_text = launcher.read_text()
    for script in sorted(scripts):
        assert (validation_dir / script).exists()
        assert script in launcher_text

    # The checklist is a sibling of the per-system directory, not a child of
    # RUNS/ itself - i.e. RUNS/validation/<system>_validation_checklist.md.
    # These tests used to look one level too high (RUNS/<system>_...md), which
    # never matched anything on any machine.
    checklist = VALIDATION_ROOT / f"{system}_validation_checklist.md"
    assert checklist.exists(), f"missing {checklist.relative_to(ROOT)}"


def test_cln025_validation_scripts_exist_and_launcher_references_them():
    _assert_validation_bundle(
        "cln025",
        {
            "pmf_cv_summary.py",
            "aromatic_cluster_distances.py",
            "hairpin_hbonds.py",
            "rmsd_to_reference.py",
            "sidechain_chi1.py",
            "rama_sanity.py",
        },
    )


def test_flag_tag_validation_scripts_exist_and_launcher_references_them():
    _assert_validation_bundle(
        "flag_tag",
        {
            "idp_ensemble_geometry.py",
            "saltbridge_occupancy.py",
            "rama_idp_content.py",
            "tyr2_chi1.py",
            "cv_coverage_2d.py",
            "charge_check.py",
        },
    )


def test_ggkgmgfgl_validation_scripts_exist_and_launcher_references_them():
    _assert_validation_bundle(
        "ggkgmgfgl",
        {
            "cv_ensemble_summary.py",
            "ensemble_geometry.py",
            "hydrophobic_contacts.py",
            "glycine_rama.py",
            "phe7_chi1.py",
            "sequence_chemistry.py",
        },
    )


def test_iggfm_validation_scripts_exist_and_launcher_references_them():
    _assert_validation_bundle(
        "iggfm",
        {
            "cv_ensemble_summary.py",
            "ensemble_geometry.py",
            "hydrophobic_contacts.py",
            "glycine_rama.py",
            "phe4_chi1.py",
            "sequence_chemistry.py",
        },
    )
