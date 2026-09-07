# tests/test_box_audit_guard.py
import json, pathlib, tempfile, types
import numpy as np


def _audit(box_nm, contour_nm, cutoff_nm):
    from gareus.system_setup import _write_box_audit
    d = pathlib.Path(tempfile.mkdtemp())
    # pos: minimal all-atom array; ca: 10 positions for 10-residue peptide GYDPETGTWG
    # This gives sequence_contour_estimate_nm = (10-1)*0.38 + 0.40 = 3.82 nm
    pos = np.zeros((3, 3))
    ca = np.zeros((10, 3))
    args = types.SimpleNamespace(padding_nm=1.0, nonbonded_cutoff_nm=cutoff_nm, seq="GYDPETGTWG")
    _write_box_audit(d, pos, ca, contour_nm, box_nm, args)
    return json.loads((d / "box_audit.json").read_text())


def test_guard_is_silent_for_the_chignolin_box():
    a = _audit(box_nm=5.82, contour_nm=3.82, cutoff_nm=0.9)
    assert a["pbc_self_contact_warning"] is False
    assert abs(a["min_image_gap_nm"] - 2.0) < 1e-9


def test_guard_fires_when_the_gap_is_below_the_cutoff():
    a = _audit(box_nm=4.5, contour_nm=3.82, cutoff_nm=0.9)     # gap 0.68 < 0.9
    assert a["pbc_self_contact_warning"] is True
