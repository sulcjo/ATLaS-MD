"""Dihedral features of an auxiliary-CV model, in OpenMM's ``theta`` convention.

theta = OpenMM CustomTorsionForce ``theta`` = -(gareus.tica._dihedral_rad). A "negated" feature is
trig(-theta) (the tica / swarm convention), a "direct" feature trig(theta). Coordinates are used
as given (``periodic_imaging: "none"``): no minimum-image is applied to torsion atoms.
"""
from __future__ import annotations

import numpy as np

from ..correctness._io import IntegrityError
from .model import AuxModel

#: |b1 x b2|^2 or |b2 x b3|^2 below this (nm^4) makes the torsion undefined (collinear atoms).
DEGENERATE_CROSS2_NM4 = 1e-12


class AuxGeometryError(IntegrityError):
    """A configuration where an auxiliary CV is undefined (degenerate torsion)."""


def openmm_dihedrals(xyz_nm, quads) -> np.ndarray:
    """(n_frames, n_torsions) OpenMM-convention dihedrals in radians; NaN where degenerate."""
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    if xyz.ndim == 2:
        xyz = xyz[None]
    q = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    p0, p1, p2, p3 = (xyz[:, q[:, k], :] for k in range(4))
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    nn1, nn2 = np.sum(n1 * n1, axis=-1), np.sum(n2 * n2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b2n = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
        m1 = np.cross(n1, b2n)
        theta = -np.arctan2(np.sum(m1 * n2, axis=-1), np.sum(n1 * n2, axis=-1))
    theta[(nn1 < DEGENERATE_CROSS2_NM4) | (nn2 < DEGENERATE_CROSS2_NM4)] = np.nan
    return theta


def unique_torsions(model: AuxModel):
    quads: list[tuple[int, int, int, int]] = []
    where: dict[tuple[int, int, int, int], int] = {}
    idx = np.empty(model.feature_schema.width, dtype=np.int64)
    for j, feature in enumerate(model.feature_schema.features):
        quad = tuple(int(a) for a in feature.atom_indices)
        if quad not in where:
            where[quad] = len(quads)
            quads.append(quad)
        idx[j] = where[quad]
    return tuple(quads), idx


def active_feature_mask(model: AuxModel) -> np.ndarray:
    """Features with a nonzero coefficient: the only ones the force builds and z depends on."""
    return np.asarray(model.coefficients, dtype=np.float64) != 0.0


def feature_signs(model: AuxModel) -> np.ndarray:
    return np.array([-1.0 if f.dihedral_sign_convention == "negated" else 1.0
                     for f in model.feature_schema.features])


def feature_values(theta, model: AuxModel) -> np.ndarray:
    """(n_frames, width) feature matrix from per-unique-torsion theta (n_frames, n_unique)."""
    theta = np.asarray(theta, dtype=np.float64)
    _quads, idx = unique_torsions(model)
    arg = theta[:, idx] * feature_signs(model)[None, :]
    is_sin = np.array([f.trig == "sin" for f in model.feature_schema.features])
    return np.where(is_sin[None, :], np.sin(arg), np.cos(arg))


_EXPECTED = {"phi": ("C", "N", "CA", "C"), "psi": ("N", "CA", "C", "N")}


def check_feature_atoms(model: AuxModel, topology, *, topology_sha256: str | None = None) -> None:
    """Refuse a model whose torsions are not the backbone phi/psi they claim on this topology."""
    if topology_sha256 is not None and topology_sha256 != model.feature_schema.topology_sha256:
        raise IntegrityError(f"aux model was built for topology {model.feature_schema.topology_sha256}, "
                             f"this run's topology is {topology_sha256}")
    atoms = list(topology.atoms())
    for feature in model.feature_schema.features:
        block = feature.torsion_name.split("-")[0]
        if block not in _EXPECTED:
            raise IntegrityError(f"feature {feature.name}: torsion_name must start with phi or psi")
        quad = feature.atom_indices
        if max(quad) >= len(atoms):
            raise IntegrityError(f"feature {feature.name}: atom index beyond topology ({len(atoms)} atoms)")
        names = tuple(atoms[i].name for i in quad)
        if names != _EXPECTED[block]:
            raise IntegrityError(f"feature {feature.name}: atom names {names} are not a {block} "
                                 f"{_EXPECTED[block]}")
        res = [atoms[i].residue.index for i in quad]
        ok = (res[0] + 1 == res[1] == res[2] == res[3]) if block == "phi" else (res[0] == res[1] == res[2] == res[3] - 1)
        if not ok:
            raise IntegrityError(f"feature {feature.name}: atoms span residues {res}, not one {block}")
