"""Backbone-subset graft assembly (fix A: poly-glycine GENPEPT seeds).

GENPEPT survivor seeds are frequently poly-glycine backbone-only conformers with
fewer atoms than the real sidechain-bearing peptide.  The historical graft did a
wholesale position overwrite and rejected every such seed with
``atom_count_mismatch``.  The backbone-subset graft instead overwrites only the
mapped (backbone) atoms and keeps the real sidechains for minimization to relax.

These tests cover the pure NumPy assembly helper (no OpenMM context required).
"""

import numpy as np
import pytest

from gareus.seeding import (
    _assemble_grafted_positions_nm,
    _MIN_GRAFT_BACKBONE_ATOMS,
)


def _rigid_transform(points: np.ndarray, seed: int = 0) -> np.ndarray:
    """Apply a fixed proper rotation + translation to a point set."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-np.pi, np.pi, size=3)
    Rx = np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
    Ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
    Rz = np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    t = rng.uniform(-2.0, 2.0, size=3)
    return points @ R.T + t


def test_unmapped_atoms_are_preserved():
    # 2 residues x (N, CA, C, O backbone + 1 sidechain) = 10 atoms.
    full = np.arange(30, dtype=float).reshape(10, 3)
    backbone_topo = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=int)
    sidechain_topo = np.array([4, 9], dtype=int)
    # Seed carries only the 8 backbone atoms, in some other frame.
    seed = _rigid_transform(full[backbone_topo], seed=1)
    seed_idx = np.arange(8, dtype=int)

    out = _assemble_grafted_positions_nm(full, seed, backbone_topo, seed_idx)

    assert out.shape == full.shape
    # Sidechain / unmapped atoms must be untouched.
    np.testing.assert_allclose(out[sidechain_topo], full[sidechain_topo])


def test_rigidly_transformed_seed_reproduces_target_backbone():
    # If the seed is a pure rigid transform of the target backbone, Kabsch
    # alignment inverts it and the grafted backbone must match the original.
    full = np.random.default_rng(7).uniform(-1.0, 1.0, size=(12, 3))
    backbone_topo = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
    seed = _rigid_transform(full[backbone_topo], seed=3)
    seed_idx = np.arange(backbone_topo.size, dtype=int)

    out = _assemble_grafted_positions_nm(full, seed, backbone_topo, seed_idx)

    np.testing.assert_allclose(out[backbone_topo], full[backbone_topo], atol=1e-9)


def test_non_rigid_seed_actually_moves_backbone_atoms():
    full = np.zeros((9, 3), dtype=float)
    full[:, 0] = np.arange(9)  # collinear-ish target, distinct points
    full[:, 1] = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
    backbone_topo = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    seed_idx = np.arange(6, dtype=int)
    # Seed backbone is a genuinely different conformation (not a rigid image).
    seed = full[backbone_topo].copy()
    seed[:, 2] += np.array([0.0, 0.5, -0.5, 0.7, -0.3, 0.2])

    out = _assemble_grafted_positions_nm(full, seed, backbone_topo, seed_idx)

    # At least some mapped atoms moved (the graft changed the backbone).
    assert not np.allclose(out[backbone_topo], full[backbone_topo])


def test_min_backbone_atoms_constant_is_reasonable():
    # Need enough mapped atoms for a stable Kabsch fit (>= 2 residues worth).
    assert _MIN_GRAFT_BACKBONE_ATOMS >= 4
