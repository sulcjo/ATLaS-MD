from pathlib import Path
import sys

import numpy as np
import pytest


VALIDATION_ROOT = Path(__file__).resolve().parents[1] / "RUNS" / "validation"
sys.path.insert(0, str(VALIDATION_ROOT))

pv = pytest.importorskip("common.peptide_validation", reason="RUNS/validation is a local gitignored artifact")


def test_basin_masks_contacts_use_folded_and_unfolded_thresholds():
    cv = np.array([0.00, 0.02, 0.12, 0.20, 0.45])
    secondary = np.array([-0.1, -0.3, -0.4, 0.1, -0.5])

    folded, unfolded = pv.make_basin_masks(
        cv,
        secondary,
        "contacts",
        fold_cv1_lo=0.10,
        fold_cv2_hi=-0.20,
        unfold_cv1_hi=0.03,
    )

    assert folded.tolist() == [False, False, True, False, True]
    assert unfolded.tolist() == [True, True, False, False, False]


def test_window_or_replica_key_prefers_replica_then_falls_back_to_window():
    assert pv.window_or_replica_key({"replica": np.array([2, 1])}).tolist() == [2, 1]
    assert pv.window_or_replica_key({"window": np.array([4, 3])}).tolist() == [4, 3]


def test_dihedral_and_rotamer_classification():
    assert pv.classify_rotamer(-65.0) == "g-"
    assert pv.classify_rotamer(60.0) == "g+"
    assert pv.classify_rotamer(175.0) == "t"
    assert pv.classify_rotamer(-175.0) == "t"
    assert pv.classify_rotamer(0.0) == "other"


def test_rama_masks_classify_ppii_helix_and_beta_regions():
    phi = np.array([-65.0, -60.0, -130.0, 30.0])
    psi = np.array([145.0, -40.0, 130.0, 20.0])

    assert pv.ppii_mask(phi, psi).tolist() == [True, False, False, False]
    assert pv.helix_like_mask(phi, psi).tolist() == [False, True, False, False]
    assert pv.beta_like_mask(phi, psi).tolist() == [False, False, True, False]


def test_net_charge_from_sequence_handles_flag_tag():
    assert pv.formal_charge_from_sequence("DYKDDDDK") == -3
    assert pv.formal_charge_from_sequence("YYDPETGTWY") == -2
    assert pv.formal_charge_from_sequence("GGKGMGFGL") == 1
    assert pv.formal_charge_from_sequence("IGGFM") == 0


def test_kyte_doolittle_hydropathy_for_hydrophobic_run_peptides():
    ggk = pv.kyte_doolittle_summary("GGKGMGFGL")
    iggfm = pv.kyte_doolittle_summary("IGGFM")

    assert ggk["n_residues"] == 9
    assert ggk["hydropathy_sum"] == 2.6
    assert round(ggk["hydropathy_mean"], 3) == 0.289
    assert iggfm["n_residues"] == 5
    assert round(iggfm["hydropathy_mean"], 2) == 1.68


def test_ca_contour_distance_upper_bound():
    assert pv.ca_contour_distance_upper_bound_A(9) == 30.4
    assert pv.ca_contour_distance_upper_bound_A(5) == 15.2
