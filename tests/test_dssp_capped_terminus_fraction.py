"""Regression tests for capped-terminus (ACE/NME) residues being counted as
"coil" in `analyze_gareus_mbar.py`'s DSSP secondary-structure fractions.

mdtraj's `compute_dssp` assigns the special code 'NA' to any topology
"residue" that lacks a full CA/N/C/O backbone quad -- notably capping groups
like ACE/NME, which this pipeline explicitly supports (see the CLI's own
docs, "For capped/non-standard structures, e.g. Ace-Ala-Nme"). The buggy code
computed helix/strand/coil fractions as

    helix = mean(arr == 'H', axis=1)
    coil  = mean((arr != 'H') & (arr != 'E'), axis=1)

which (a) counts 'NA' entries as coil (they're neither 'H' nor 'E') and
(b) keeps them in the denominator (the *full* residue count) -- so any run
with capping groups reports helix/strand fractions biased LOW and coil
biased HIGH by 1-2 non-structural cap "residues" per frame. A synthetic
fully-helical 6-real-residue peptide with ACE/NME caps reported
helix_fraction=0.75 (6/8) instead of the physically correct 1.0 (6/6).

The fix excludes 'NA' entries from both the numerator and the denominator,
via a shared `_dssp_valid_counts(arr, axis)` helper used by both the
per-frame aggregate path (`_dssp_fraction_arrays`) and the per-residue
breakdown (`_dssp_residue_probability_rows`, which additionally drops any
residue index that is 'NA' in every frame -- i.e. a cap -- instead of
reporting a spurious 0/0 probability for it).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import (
    _dssp_fraction_arrays,
    _dssp_residue_probability_rows,
    _dssp_valid_counts,
)


class _FakeMD:
    """Stand-in for the `md` module parameter of `_dssp_fraction_arrays`.

    `_dssp_fraction_arrays(md, traj)` only ever calls `md.compute_dssp(traj,
    simplified=True)`, so the DSSP code array can be injected directly --
    exercising the exact fraction-computation logic under test without any
    real mdtraj/trajectory I/O.
    """

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def compute_dssp(self, traj, simplified=True):
        assert simplified is True
        return self._arr


# ---------------------------------------------------------------------------
# _dssp_fraction_arrays: aggregate per-frame fractions
# ---------------------------------------------------------------------------

def test_fully_helical_capped_peptide_reports_full_helix_fraction():
    # 1 frame, 8 "residues": ACE cap, 6 real helical residues, NME cap.
    arr = np.array([["NA", "H", "H", "H", "H", "H", "H", "NA"]])
    result = _dssp_fraction_arrays(_FakeMD(arr), traj=None)
    assert result is not None
    _, helix, strand, coil = result
    assert helix[0] == pytest.approx(1.0)  # not 6/8 == 0.75
    assert strand[0] == pytest.approx(0.0)
    assert coil[0] == pytest.approx(0.0)


def test_uncapped_peptide_fraction_unaffected_by_fix():
    # No 'NA' entries at all -- fix must be a no-op here (denominator was
    # already the real residue count).
    arr = np.array([["H", "H", "H", "H", "H", "H"]])
    result = _dssp_fraction_arrays(_FakeMD(arr), traj=None)
    assert result is not None
    _, helix, strand, coil = result
    assert helix[0] == pytest.approx(1.0)
    assert strand[0] == pytest.approx(0.0)
    assert coil[0] == pytest.approx(0.0)


def test_uncapped_mixed_ss_fraction_matches_plain_mean():
    # Mixed H/E/C, no caps: fixed formula must reduce exactly to the old
    # np.mean behavior when there's nothing to exclude.
    arr = np.array([["H", "H", "E", "C", "C", "C"]])
    result = _dssp_fraction_arrays(_FakeMD(arr), traj=None)
    assert result is not None
    _, helix, strand, coil = result
    assert helix[0] == pytest.approx(2 / 6)
    assert strand[0] == pytest.approx(1 / 6)
    assert coil[0] == pytest.approx(3 / 6)


def test_capped_peptide_partial_ss_denominator_excludes_caps():
    # 6 real residues (2 helix, 1 strand, 3 coil) + 2 NA caps. Denominator
    # must be 6 (real residues only), not 8.
    arr = np.array([["NA", "H", "H", "E", "C", "C", "C", "NA"]])
    result = _dssp_fraction_arrays(_FakeMD(arr), traj=None)
    assert result is not None
    _, helix, strand, coil = result
    assert helix[0] == pytest.approx(2 / 6)
    assert strand[0] == pytest.approx(1 / 6)
    assert coil[0] == pytest.approx(3 / 6)
    assert helix[0] + strand[0] + coil[0] == pytest.approx(1.0)


def test_all_na_frame_does_not_divide_by_zero():
    # Pathological: an entire frame is 'NA' (e.g. topology mismatch). Must
    # not raise / produce NaN or Inf -- falls back to a safe all-zero row.
    arr = np.array([["NA", "NA", "NA"]])
    result = _dssp_fraction_arrays(_FakeMD(arr), traj=None)
    assert result is not None
    _, helix, strand, coil = result
    assert math.isfinite(helix[0]) and helix[0] == 0.0
    assert math.isfinite(strand[0]) and strand[0] == 0.0
    assert math.isfinite(coil[0]) and coil[0] == 0.0


def test_none_returned_when_compute_dssp_raises():
    class _Raising:
        def compute_dssp(self, traj, simplified=True):
            raise RuntimeError("boom")

    assert _dssp_fraction_arrays(_Raising(), traj=None) is None


# ---------------------------------------------------------------------------
# _dssp_valid_counts: shared counting helper (used along both axes)
# ---------------------------------------------------------------------------

def test_dssp_valid_counts_axis1_per_frame():
    arr = np.array([
        ["NA", "H", "H", "E", "C", "NA"],
        ["NA", "H", "C", "E", "C", "NA"],
    ])
    n_h, n_e, n_coil, n_valid = _dssp_valid_counts(arr, axis=1)
    np.testing.assert_array_equal(n_h, [2, 1])
    np.testing.assert_array_equal(n_e, [1, 1])
    np.testing.assert_array_equal(n_coil, [1, 2])
    np.testing.assert_array_equal(n_valid, [4, 4])


def test_dssp_valid_counts_axis0_per_residue_across_frames():
    # Rows = frames, columns = residues (the shape used by the per-residue
    # accumulator in analyze_extra_observable_pmfs). Residue 0 is a cap
    # ('NA' every frame); residues 1-2 are real.
    arr = np.array([
        ["NA", "H", "C"],
        ["NA", "H", "E"],
        ["NA", "C", "E"],
    ])
    n_h, n_e, n_coil, n_valid = _dssp_valid_counts(arr, axis=0)
    np.testing.assert_array_equal(n_h, [0, 2, 0])
    np.testing.assert_array_equal(n_e, [0, 0, 2])
    np.testing.assert_array_equal(n_coil, [0, 1, 1])
    np.testing.assert_array_equal(n_valid, [0, 3, 3])


# ---------------------------------------------------------------------------
# _dssp_residue_probability_rows: per-residue CSV row building
# ---------------------------------------------------------------------------

def test_residue_rows_excludes_fully_capped_residue_index():
    dssp_labels = ["ACE-cap", "ALA2", "ALA3", "NME-cap"]
    dssp_counts = {
        "H": np.array([0, 3, 0, 0]),
        "E": np.array([0, 0, 3, 0]),
        "C": np.array([0, 0, 0, 0]),
    }
    # Cap residue indices (0, 3) never had a valid (non-'NA') frame.
    dssp_valid_counts = np.array([0, 3, 3, 0])
    rows = _dssp_residue_probability_rows(dssp_labels, dssp_counts, dssp_valid_counts, dssp_total=3)
    labels_out = [r["residue"] for r in rows]
    assert labels_out == ["ALA2", "ALA3"]
    row_by_label = {r["residue"]: r for r in rows}
    assert row_by_label["ALA2"]["helix_probability"] == pytest.approx(1.0)
    assert row_by_label["ALA2"]["n_frames"] == 3
    assert row_by_label["ALA3"]["strand_probability"] == pytest.approx(1.0)


def test_residue_rows_falls_back_to_dssp_total_when_valid_counts_missing():
    # Backward-compat path: dssp_valid_counts is None (shouldn't happen from
    # the current caller, but keep the fallback honest).
    dssp_labels = ["ALA1"]
    dssp_counts = {"H": np.array([4]), "E": np.array([0]), "C": np.array([0])}
    rows = _dssp_residue_probability_rows(dssp_labels, dssp_counts, None, dssp_total=4)
    assert len(rows) == 1
    assert rows[0]["helix_probability"] == pytest.approx(1.0)
    assert rows[0]["n_frames"] == 4


# ---------------------------------------------------------------------------
# Live mdtraj confirmation: real compute_dssp really emits 'NA' for ACE/NME
# ---------------------------------------------------------------------------

def _build_capped_helix_pdb(tmp_path):
    """Write a tiny synthetic ACE-(ALA)x6-NME structure and return its path.

    Backbone internal coordinates reuse GENPEPT's own `place_internal_atom`
    NeRF placement and standard bond-length/angle constants, with ideal
    alpha-helix phi/psi dihedrals for the real residues. Cap placements
    aren't chemically precise (DSSP's 'NA' assignment is purely a check for
    the presence of CA/N/C/O atom *names* in a residue, not 3D geometry), but
    every atom gets a distinct, non-degenerate position.
    """
    from GENPEPT import (
        BACKBONE_ANGLE_C_N_CA,
        BACKBONE_ANGLE_CA_C_N,
        BACKBONE_ANGLE_N_CA_C,
        BACKBONE_BOND_C_N,
        BACKBONE_BOND_CA_C,
        BACKBONE_BOND_N_CA,
        DEFAULT_OMEGA,
        place_internal_atom,
    )

    n_res = 6
    phi, psi = -57.0, -47.0

    N = [None] * n_res
    CA = [None] * n_res
    C = [None] * n_res
    O = [None] * n_res
    N[0] = np.array([0.0, 0.0, 0.0])
    CA[0] = np.array([BACKBONE_BOND_N_CA, 0.0, 0.0])
    theta = math.radians(BACKBONE_ANGLE_N_CA_C)
    C[0] = CA[0] + np.array(
        [BACKBONE_BOND_CA_C * math.cos(math.pi - theta), BACKBONE_BOND_CA_C * math.sin(math.pi - theta), 0.0]
    )
    for i in range(n_res - 1):
        Nn = place_internal_atom(N[i], CA[i], C[i], BACKBONE_BOND_C_N, BACKBONE_ANGLE_CA_C_N, psi)
        CAn = place_internal_atom(CA[i], C[i], Nn, BACKBONE_BOND_N_CA, BACKBONE_ANGLE_C_N_CA, DEFAULT_OMEGA)
        Cn = place_internal_atom(C[i], Nn, CAn, BACKBONE_BOND_CA_C, BACKBONE_ANGLE_N_CA_C, phi)
        N[i + 1], CA[i + 1], C[i + 1] = Nn, CAn, Cn
    for i in range(n_res):
        O[i] = place_internal_atom(N[i], CA[i], C[i], 1.231, 120.8, 180.0)

    ace_c = place_internal_atom(C[0], CA[0], N[0], BACKBONE_BOND_C_N, BACKBONE_ANGLE_C_N_CA, DEFAULT_OMEGA)
    ace_o = place_internal_atom(CA[0], N[0], ace_c, 1.231, 120.0, 0.0)
    ace_ch3 = place_internal_atom(CA[0], N[0], ace_c, 1.51, 120.0, 180.0)

    last = n_res - 1
    nme_n = place_internal_atom(N[last], CA[last], C[last], BACKBONE_BOND_C_N, BACKBONE_ANGLE_CA_C_N, psi)
    nme_ch3 = place_internal_atom(CA[last], C[last], nme_n, BACKBONE_BOND_N_CA, BACKBONE_ANGLE_C_N_CA, DEFAULT_OMEGA)

    def fmt(serial, name, resname, chain, resseq, xyz, element):
        x, y, z = xyz
        namefield = f" {name:<3s}" if len(name) < 4 else f"{name:<4s}"
        return (
            f"ATOM  {serial:5d} {namefield}{'':1s}{resname:>3s} {chain:1s}{resseq:4d}{'':1s}   "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}      {'':4s}{element:>2s}"
        )

    lines = []
    serial = 1
    resseq = 1
    lines.append(fmt(serial, "CH3", "ACE", "A", resseq, ace_ch3, "C")); serial += 1
    lines.append(fmt(serial, "C", "ACE", "A", resseq, ace_c, "C")); serial += 1
    lines.append(fmt(serial, "O", "ACE", "A", resseq, ace_o, "O")); serial += 1
    resseq += 1
    for i in range(n_res):
        lines.append(fmt(serial, "N", "ALA", "A", resseq, N[i], "N")); serial += 1
        lines.append(fmt(serial, "CA", "ALA", "A", resseq, CA[i], "C")); serial += 1
        lines.append(fmt(serial, "C", "ALA", "A", resseq, C[i], "C")); serial += 1
        lines.append(fmt(serial, "O", "ALA", "A", resseq, O[i], "O")); serial += 1
        resseq += 1
    lines.append(fmt(serial, "N", "NME", "A", resseq, nme_n, "N")); serial += 1
    lines.append(fmt(serial, "CH3", "NME", "A", resseq, nme_ch3, "C")); serial += 1
    lines.append("TER")
    lines.append("END")

    pdb_path = tmp_path / "capped_hexala.pdb"
    pdb_path.write_text("\n".join(lines) + "\n")
    return pdb_path, n_res


def test_real_mdtraj_marks_capped_termini_na_and_fix_excludes_them(tmp_path):
    md = pytest.importorskip("mdtraj")
    pdb_path, n_res = _build_capped_helix_pdb(tmp_path)
    traj = md.load_pdb(str(pdb_path))
    assert traj.topology.n_residues == n_res + 2  # + ACE + NME

    ss = md.compute_dssp(traj, simplified=True)
    assert ss.shape == (1, n_res + 2)
    # Real mdtraj assigns 'NA' to the cap "residues" (no CA/N/C/O quad)...
    assert ss[0, 0] == "NA"
    assert ss[0, -1] == "NA"
    # ...and a real simplified DSSP code (never 'NA') to every real residue.
    for i in range(1, n_res + 1):
        assert ss[0, i] != "NA"

    result = _dssp_fraction_arrays(md, traj)
    assert result is not None
    arr, helix, strand, coil = result
    np.testing.assert_array_equal(arr, ss)
    # Fractions must sum to 1 and be computed over the 6 real residues only.
    assert helix[0] + strand[0] + coil[0] == pytest.approx(1.0)
    _, _, _, n_valid = _dssp_valid_counts(ss, axis=1)
    assert n_valid[0] == n_res  # denominator excludes the 2 NA caps (would be 8 pre-fix)
