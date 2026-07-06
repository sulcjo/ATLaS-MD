from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cln025_validation_scripts_exist_and_launcher_references_them():
    scripts = {
        "pmf_cv_summary.py",
        "aromatic_cluster_distances.py",
        "hairpin_hbonds.py",
        "rmsd_to_reference.py",
        "sidechain_chi1.py",
        "rama_sanity.py",
    }
    validation_dir = ROOT / "RUNS" / "validation" / "cln025"
    launcher = validation_dir / "run_all_validations.sh"

    assert launcher.exists()
    launcher_text = launcher.read_text()
    for script in scripts:
        assert (validation_dir / script).exists()
        assert script in launcher_text


def test_flag_tag_validation_scripts_exist_and_launcher_references_them():
    scripts = {
        "idp_ensemble_geometry.py",
        "saltbridge_occupancy.py",
        "rama_idp_content.py",
        "tyr2_chi1.py",
        "cv_coverage_2d.py",
        "charge_check.py",
    }
    validation_dir = ROOT / "RUNS" / "validation" / "flag_tag"
    launcher = validation_dir / "run_all_validations.sh"

    assert launcher.exists()
    launcher_text = launcher.read_text()
    for script in scripts:
        assert (validation_dir / script).exists()
        assert script in launcher_text


def test_ggkgmgfgl_validation_scripts_exist_and_launcher_references_them():
    scripts = {
        "cv_ensemble_summary.py",
        "ensemble_geometry.py",
        "hydrophobic_contacts.py",
        "glycine_rama.py",
        "phe7_chi1.py",
        "sequence_chemistry.py",
    }
    validation_dir = ROOT / "RUNS" / "validation" / "ggkgmgfgl"
    launcher = validation_dir / "run_all_validations.sh"

    assert launcher.exists()
    launcher_text = launcher.read_text()
    for script in scripts:
        assert (validation_dir / script).exists()
        assert script in launcher_text
    assert (ROOT / "RUNS" / "ggkgmgfgl_validation_checklist.md").exists()


def test_iggfm_validation_scripts_exist_and_launcher_references_them():
    scripts = {
        "cv_ensemble_summary.py",
        "ensemble_geometry.py",
        "hydrophobic_contacts.py",
        "glycine_rama.py",
        "phe4_chi1.py",
        "sequence_chemistry.py",
    }
    validation_dir = ROOT / "RUNS" / "validation" / "iggfm"
    launcher = validation_dir / "run_all_validations.sh"

    assert launcher.exists()
    launcher_text = launcher.read_text()
    for script in scripts:
        assert (validation_dir / script).exists()
        assert script in launcher_text
    assert (ROOT / "RUNS" / "iggfm_validation_checklist.md").exists()
