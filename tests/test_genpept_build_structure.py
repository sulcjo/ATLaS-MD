"""GENPEPT real-sequence conformer construction (fix D root cause).

PeptideBuilder's ``Geometry.geometry()`` takes a ONE-letter amino-acid code and
silently falls back to glycine for anything it does not recognize.  GENPEPT
passed the THREE-letter code (``AA3[aa]``, e.g. "TYR"), so every residue built as
glycine and every survivor conformer was a poly-glycine backbone.  That poly-Gly
Ramachandran does not match the real sequence, which misregistered the CV2
torsion-PCA basis at runtime (bug D) and also broke grafting (A) and seed contact
scoring (B).

build_structure must construct the real residues for the given sequence.
"""

import pytest

pytest.importorskip("PeptideBuilder")

import GENPEPT


def _flat_angles(n, value):
    return [float(value)] * n


def test_build_structure_produces_real_residue_names():
    seq = "GYDPETGTWG"  # chignolin
    n = len(seq)
    struct = GENPEPT.build_structure(seq, _flat_angles(n, -60.0), _flat_angles(n, -45.0))
    resnames = [r.get_resname() for r in struct.get_residues()]
    assert resnames == ["GLY", "TYR", "ASP", "PRO", "GLU", "THR", "GLY", "THR", "TRP", "GLY"]


def test_build_structure_builds_sidechains_not_backbone_only_glycine():
    seq = "GYDPETGTWG"
    n = len(seq)
    struct = GENPEPT.build_structure(seq, _flat_angles(n, -60.0), _flat_angles(n, -45.0))
    natoms = sum(1 for _ in struct.get_atoms())
    # Backbone-only glycine would be ~4 atoms/residue; real chignolin sidechains
    # (Tyr, Trp, Pro, ...) push this well above that.
    assert natoms > 6 * n


def test_geometry_lookup_rejects_silent_glycine_substitution():
    # A non-glycine residue must not silently build as glycine.
    seq = "AW"  # Ala-Trp: neither is glycine
    struct = GENPEPT.build_structure(seq, [-60.0, -60.0], [-45.0, -45.0])
    resnames = [r.get_resname() for r in struct.get_residues()]
    assert "GLY" not in resnames
