"""Exact auxiliary-CV evaluator: z, analytic dz/dx and the harmonic restraint 0.5 k (z - c)^2.

z = (offset + sum_j c_j f_j) / scale. Units: positions nm, energies kJ/mol, forces kJ/mol/nm,
k given in kcal/mol per z^2 (converted exactly once by KJ_PER_KCAL). This module is the numerical
reference for the OpenMM force. Only features with a nonzero coefficient enter z, exactly as in
the force (``build_aux_force`` skips zero-coefficient features), so a degenerate torsion whose
features all have zero coefficients does not make z undefined; a degenerate torsion that carries
weight does (NaN offline, AuxGeometryError per structure).
"""
from __future__ import annotations

import numpy as np

from ..correctness._io import IntegrityError
from ..correctness.bias import KJ_PER_KCAL, finite_number
from .features import (AuxGeometryError, active_feature_mask, feature_signs, feature_values,
                       openmm_dihedrals, unique_torsions)
from .model import AuxModel


def z_from_dihedrals(theta, model: AuxModel) -> np.ndarray:
    """z per frame from OpenMM theta (radians), shape (n_frames, n_unique).

    Columns are the model's unique torsions in the first-appearance order of
    ``unique_torsions(model)``; any other shape is refused.
    """
    theta = np.asarray(theta, dtype=np.float64)
    if getattr(model, "schema_version", 1) == 2:
        n_active = len(model.projection.quads)
        if theta.ndim != 2 or theta.shape[1] != n_active or not np.isfinite(theta).all():
            raise IntegrityError(f"theta must have shape (n_frames, {n_active}) for active v2 torsions, "
                                 f"got {theta.shape}")
        return np.asarray(model.projection.from_angles(theta), dtype=np.float64)
    n_unique = len(unique_torsions(model)[0])
    if theta.ndim != 2 or theta.shape[1] != n_unique:
        raise IntegrityError(f"theta must have shape (n_frames, {n_unique}) (one column per unique "
                             f"torsion), got {theta.shape}")
    active = active_feature_mask(model)
    feats = feature_values(theta, model)[:, active]
    coeffs = np.asarray(model.coefficients, dtype=np.float64)[active]
    return (model.offset + feats @ coeffs) / model.scale


def z_from_positions(xyz_nm, model: AuxModel) -> np.ndarray:
    if getattr(model, "schema_version", 1) == 2:
        return model.values(xyz_nm)
    quads, _idx = unique_torsions(model)
    return z_from_dihedrals(openmm_dihedrals(xyz_nm, quads), model)


def _dtheta_dx(p: np.ndarray) -> np.ndarray:
    """Blondel-Karplus gradient of OpenMM theta w.r.t. the 4 atoms, shape (4, 3).

    Verified against finite differences of openmm_dihedrals (2e-10) on random geometries.
    """
    p0, p1, p2, p3 = p
    F, G, H = p0 - p1, p1 - p2, p3 - p2
    A, B = np.cross(F, G), np.cross(H, G)
    A2, B2, Gn = A @ A, B @ B, np.linalg.norm(G)
    g0 = -Gn / A2 * A
    g3 = Gn / B2 * B
    g1 = Gn / A2 * A + (F @ G) / (A2 * Gn) * A - (H @ G) / (B2 * Gn) * B
    g2 = -Gn / B2 * B - (F @ G) / (A2 * Gn) * A + (H @ G) / (B2 * Gn) * B
    return np.array([g0, g1, g2, g3])


def z_and_gradient(xyz_nm, model: AuxModel) -> tuple[float, np.ndarray]:
    x = np.asarray(xyz_nm, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3:
        raise IntegrityError("z_and_gradient takes one configuration of shape (n_atoms, 3)")
    if getattr(model, "schema_version", 1) == 2:
        return model.value_gradient(x)
    quads, idx = unique_torsions(model)
    active = active_feature_mask(model)
    used = np.zeros(len(quads), dtype=bool)
    used[idx[active]] = True                    # torsions that carry weight in z (and in the force)
    theta = openmm_dihedrals(x, quads)[0]
    degenerate = np.isnan(theta) & used
    if degenerate.any():
        bad = [quads[t] for t in np.flatnonzero(degenerate)]
        raise AuxGeometryError(f"auxiliary CV undefined: degenerate torsion(s) {bad}")
    signs = feature_signs(model)[active]
    coeffs = np.asarray(model.coefficients, dtype=np.float64)[active]
    is_sin = np.array([f.trig == "sin" for f in model.feature_schema.features])[active]
    arg = theta[idx[active]] * signs
    # d trig(s*theta)/d theta: sin -> s cos(s theta); cos -> -s sin(s theta)
    dfeat = np.where(is_sin, signs * np.cos(arg), -signs * np.sin(arg))
    dz_dtheta = np.zeros(len(quads))
    np.add.at(dz_dtheta, idx[active], coeffs * dfeat / model.scale)
    grad = np.zeros_like(x)
    for t, quad in enumerate(quads):
        if dz_dtheta[t] != 0.0:
            grad[list(quad)] += dz_dtheta[t] * _dtheta_dx(x[list(quad)])
    z = float(z_from_dihedrals(theta[None, :], model)[0])
    return z, grad


def _strength(k_kcal) -> float:
    return finite_number(k_kcal, "aux_k", minimum=0)


def aux_energy_kj(z, center, k_kcal) -> np.ndarray:
    k = _strength(k_kcal)
    z = np.asarray(z, dtype=np.float64)
    if k == 0.0:
        return np.zeros_like(z)          # exact zero; never 0 * (NaN - c)^2
    c = finite_number(center, "aux center")
    return 0.5 * k * KJ_PER_KCAL * (z - c) ** 2


def aux_forces_kj_nm(xyz_nm, model: AuxModel, center, k_kcal) -> np.ndarray:
    k = _strength(k_kcal)
    x = np.asarray(xyz_nm, dtype=np.float64)
    if k == 0.0:
        return np.zeros_like(x)
    c = finite_number(center, "aux center")
    z, grad = z_and_gradient(x, model)
    return -k * KJ_PER_KCAL * (z - c) * grad
