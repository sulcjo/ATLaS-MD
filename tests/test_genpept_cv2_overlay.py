"""Tests for the GENPEPT pseudo-FES CV2 overlay (analyze feature)."""
from pathlib import Path

import numpy as np
import pytest

from gareus.genpept_cv2_overlay import (
    _ca_coords_angstrom,
    collect_seed_pdbs,
    fit_cv2_field,
    locate_genpept_seed_dir,
    overlay_from_pdbs,
    plot_overlay,
    project_conformer_cv2,
)
from gareus.tica import TICAResult, backbone_dihedral_features


# --- fake OpenMM-style topology (duck-typed for peptide_residues / atom map) ---

class _Atom:
    def __init__(self, index, name):
        self.index = index
        self.name = name


class _Res:
    def __init__(self, name, atoms):
        self.name = name
        self._atoms = atoms

    def atoms(self):
        return iter(self._atoms)


class _Top:
    def __init__(self, residues):
        self._res = residues

    def residues(self):
        return iter(self._res)


def _three_residue_topology():
    residues = []
    idx = 0
    for r in range(3):
        atoms = [_Atom(idx + 0, "N"), _Atom(idx + 1, "CA"), _Atom(idx + 2, "C")]
        residues.append(_Res("ALA", atoms))
        idx += 3
    return _Top(residues)


def _conformer_atoms_matching_topology():
    atoms = []
    idx = 0
    for r in range(3):
        for name in ("N", "CA", "C"):
            atoms.append({"index": idx, "name": name, "residue_ordinal": r})
            idx += 1
    return atoms


def test_fit_cv2_field_recovers_exact_linear_relation():
    rng = np.random.default_rng(0)
    pc1 = rng.normal(size=200)
    pc2 = rng.normal(size=200)
    cv2 = 2.0 * pc1 - 3.0 * pc2 + 1.0  # exact linear, no noise

    field = fit_cv2_field(pc1, pc2, cv2, order=1)

    assert field.usable is True
    assert field.r2 == pytest.approx(1.0, abs=1e-8)
    # evaluate reproduces the known field
    got = field.evaluate(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert got == pytest.approx(np.array([1.0, 0.0]), abs=1e-6)


def test_fit_cv2_field_reports_low_r2_for_uncorrelated_noise():
    rng = np.random.default_rng(1)
    pc1 = rng.normal(size=300)
    pc2 = rng.normal(size=300)
    cv2 = rng.normal(size=300)  # independent of pc1/pc2

    field = fit_cv2_field(pc1, pc2, cv2, order=1)

    assert field.r2 < 0.3


def test_fit_cv2_field_handles_collinear_input_without_crashing():
    pc1 = np.linspace(-1.0, 1.0, 50)
    pc2 = pc1.copy()  # perfectly collinear -> rank-deficient design
    cv2 = 0.5 * pc1

    field = fit_cv2_field(pc1, pc2, cv2, order=1)

    assert np.isfinite(field.r2)
    # a rank-deficient 2D fit cannot be trusted for iso-contours
    assert field.usable is False


def test_fit_cv2_field_marks_unusable_when_too_few_points():
    field = fit_cv2_field(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), order=1)
    assert field.usable is False


def test_fit_cv2_field_constant_target_is_unusable():
    rng = np.random.default_rng(11)
    pc1 = rng.normal(size=30)  # independent, full-rank design
    pc2 = rng.normal(size=30)
    cv2 = np.full(30, 3.0)  # degenerate: zero variance in the target
    field = fit_cv2_field(pc1, pc2, cv2, order=1)
    assert field.r2 == 0.0
    assert field.usable is False


def test_fit_cv2_field_ignores_nonfinite_samples():
    pc1 = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    pc2 = np.array([0.0, 1.0, 0.0, 1.0, 5.0])
    cv2 = np.array([1.0, 0.0, 3.0, 2.0, np.inf])

    field = fit_cv2_field(pc1, pc2, cv2, order=1)

    # the 4 finite rows are exactly cv2 = 2*pc1 - 3*pc2 + 1
    assert field.r2 == pytest.approx(1.0, abs=1e-8)


def test_locate_genpept_seed_dir_prefers_absolute_existing(tmp_path):
    seed = tmp_path / "genpept_r3"
    seed.mkdir()
    got = locate_genpept_seed_dir(tmp_path / "run", {"seed_conformers_dir": str(seed)})
    assert got == seed


def test_locate_genpept_seed_dir_resolves_relative_sibling(tmp_path):
    # run dir and genpept dir are siblings; run_args stores a relative name
    (tmp_path / "genpept_r3").mkdir()
    run_root = tmp_path / "run3"
    run_root.mkdir()
    got = locate_genpept_seed_dir(run_root, {"seed_conformers_dir": "genpept_r3"})
    assert got == tmp_path / "genpept_r3"


def test_locate_genpept_seed_dir_missing_returns_none(tmp_path):
    assert locate_genpept_seed_dir(tmp_path, {"seed_conformers_dir": "nope"}) is None
    assert locate_genpept_seed_dir(tmp_path, {}) is None


def test_collect_seed_pdbs_expands_run_root_subdirs(tmp_path):
    sub = tmp_path / "basin_hop_minima"
    sub.mkdir()
    (sub / "a.pdb").write_text("ATOM\n")
    (sub / "b.pdb").write_text("ATOM\n")
    got = {p.name for p in collect_seed_pdbs(tmp_path)}
    assert got == {"a.pdb", "b.pdb"}


def test_ca_coords_angstrom_extracts_ca_in_residue_order():
    atoms = _conformer_atoms_matching_topology()
    positions_nm = np.arange(9 * 3, dtype=float).reshape(9, 3) / 10.0
    ca = _ca_coords_angstrom(atoms, positions_nm)
    # CA atoms are indices 1, 4, 7 -> positions_nm rows * 10 (back to Angstrom)
    expected = positions_nm[[1, 4, 7]] * 10.0
    assert ca.shape == (3, 3)
    assert ca == pytest.approx(expected)


def test_project_conformer_cv2_matches_direct_projection():
    topology = _three_residue_topology()
    atoms = _conformer_atoms_matching_topology()
    rng = np.random.default_rng(3)
    positions_nm = rng.normal(size=(9, 3))

    phi = [(2, 3, 4, 5), (5, 6, 7, 8)]
    psi = [(0, 1, 2, 3), (3, 4, 5, 6)]
    weights = rng.normal(size=2 * len(phi) + 2 * len(psi))
    result = TICAResult(
        weights=weights,
        eigenvalue=1.0,
        mean=np.zeros_like(weights),
        offset=0.75,
        lag=0,
        phi_torsion_indices=phi,
        psi_torsion_indices=psi,
        method="pca",
        explained_variance_ratio=0.5,
    )

    # identity atom map -> mapped torsions equal topology torsions
    expected_feat = backbone_dihedral_features(positions_nm, phi, psi)
    expected_cv2 = float(expected_feat @ weights + result.offset)

    got = project_conformer_cv2(positions_nm, atoms, topology, result)
    assert got == pytest.approx(expected_cv2, abs=1e-9)


def _write_pdb(path: Path, positions_a: np.ndarray):
    names = ["N", "CA", "C"] * 3
    lines = []
    for i, (name, (x, y, z)) in enumerate(zip(names, positions_a)):
        resseq = i // 3 + 1
        lines.append(
            f"ATOM  {i + 1:>5} {name:<4}ALA A{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        )
    lines.append("END\n")
    path.write_text("".join(lines))


def _make_result():
    phi = [(2, 3, 4, 5), (5, 6, 7, 8)]
    psi = [(0, 1, 2, 3), (3, 4, 5, 6)]
    weights = np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.25, 0.15, -0.35], dtype=float)
    return TICAResult(
        weights=weights,
        eigenvalue=1.0,
        mean=np.zeros_like(weights),
        offset=0.1,
        lag=0,
        phi_torsion_indices=phi,
        psi_torsion_indices=psi,
        method="pca",
        explained_variance_ratio=0.5,
    )


def test_overlay_from_pdbs_builds_scores_and_cv2(tmp_path):
    topology = _three_residue_topology()
    result = _make_result()
    rng = np.random.default_rng(7)
    pdbs = []
    for k in range(5):
        pos_a = rng.normal(scale=3.0, size=(9, 3)) + k  # Angstrom
        p = tmp_path / f"seed_{k}.pdb"
        _write_pdb(p, pos_a)
        pdbs.append(p)

    data = overlay_from_pdbs(pdbs, topology, result, bins=8)

    assert data.pc1.shape == (5,)
    assert data.pc2.shape == (5,)
    assert data.cv2.shape == (5,)
    assert np.isfinite(data.pc1).all()
    assert np.isfinite(data.cv2).all()
    assert data.grid_dE.shape == (data.pc1_centers.size, data.pc2_centers.size)
    assert np.isfinite(data.grid_dE).any()


def test_plot_overlay_writes_png(tmp_path):
    rng = np.random.default_rng(9)
    pc1 = rng.normal(size=40)
    pc2 = rng.normal(size=40)
    cv2 = 1.5 * pc1 - 0.5 * pc2
    field = fit_cv2_field(pc1, pc2, cv2, order=1)

    class _Data:
        pass

    data = _Data()
    data.pc1, data.pc2, data.cv2 = pc1, pc2, cv2
    data.pc1_centers = np.linspace(pc1.min(), pc1.max(), 8)
    data.pc2_centers = np.linspace(pc2.min(), pc2.max(), 8)
    data.grid_dE = rng.random((8, 8))

    out = tmp_path / "overlay.png"
    info = plot_overlay(data, field, out, labels={"title": "t", "cv2": "CV2"})
    assert out.exists() and out.stat().st_size > 0
    assert info["contours_drawn"] is True  # R2 is high for this linear field


def test_plot_overlay_degrades_on_all_nan_input(tmp_path):
    class _Data:
        pass

    data = _Data()
    data.pc1 = np.full(5, np.nan)
    data.pc2 = np.full(5, np.nan)
    data.cv2 = np.full(5, np.nan)
    data.pc1_centers = np.array([np.nan, np.nan])
    data.pc2_centers = np.array([np.nan, np.nan])
    data.grid_dE = np.full((2, 2), np.nan)
    field = fit_cv2_field(data.pc1, data.pc2, data.cv2, order=1)

    out = tmp_path / "nan_overlay.png"
    info = plot_overlay(data, field, out, labels={})  # must not raise
    assert out.exists()
    assert info["contours_drawn"] is False
