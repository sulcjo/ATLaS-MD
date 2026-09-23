"""OpenMM force-construction helpers for GAREUS.

This module owns small reusable force helpers that were previously
embedded in the historical monolith.
"""

from __future__ import annotations

import gc
import math

import numpy as np

from .imports import import_openmm
from .units import kcal_to_kj
from .cv import (
    _contact_pair_indices,
    _contact_pair_weight,
    contact_normalization_denominator,
    contact_scheme,
    nonlocal_contact_cv_from_positions_nm,
)

__all__ = [
    "validate_openmm_force_group",
    "add_umbrella_force",
    "add_contact_umbrella_force",
    "contact_switch_constants_nm",
    "self_test_nonlocal_contact_force",
]

def contact_switch_constants_nm(args) -> tuple[float, float]:
    """(r0 in nm, beta in 1/nm) of the contact switching function, validated positive finite."""
    r0_nm = float(getattr(args, "contact_r0_a", 4.5) or 4.5) * 0.1
    beta_nm_inv = float(getattr(args, "contact_beta_a_inv", 6.0) or 6.0) * 10.0
    if not math.isfinite(r0_nm) or r0_nm <= 0.0:
        raise ValueError("--contact-r0-a must be a positive finite value")
    if not math.isfinite(beta_nm_inv) or beta_nm_inv <= 0.0:
        raise ValueError("--contact-beta-a-inv must be a positive finite value")
    return r0_nm, beta_nm_inv


def validate_openmm_force_group(force_group: int, label: str = "force group") -> int:
    """OpenMM force groups are limited to integers 0..31."""
    try:
        group = int(force_group)
    except Exception as exc:
        raise ValueError(f"{label} must be an integer between 0 and 31; got {force_group!r}") from exc
    if group < 0 or group > 31:
        raise ValueError(f"{label} must be between 0 and 31 for OpenMM; got {group}. Use e.g. 29 for --secondary-cv-force-group.")
    return group

def add_umbrella_force(openmm, system, atom1: int, atom2: int, force_group: int = 31):
    # Historical distance umbrella: r is in nm; k is kJ/mol/nm^2; r0 is nm.
    force = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    force.addGlobalParameter("k", 0.0)
    force.addGlobalParameter("r0", 0.0)
    force.addBond(int(atom1), int(atom2), [])
    force.setForceGroup(int(force_group))
    system.addForce(force)
    return force

def add_contact_umbrella_force(openmm, system, contact_pairs: list[tuple[int, int, float]], args, force_group: int = 31):
    """Add a smooth weighted nonlocal-contact primary umbrella.

    OpenMM implementation is intentionally one CustomCVForce deep: the collective
    variable is the weighted smooth contact sum from a CustomBondForce, and the
    harmonic umbrella divides by contact_norm directly in the outer expression.
    Avoiding nested CustomCVForce objects is more portable across OpenMM versions.

    The outer umbrella intentionally uses the same global parameter names as the
    historical distance force (r0 and k) so existing set_window()/exchange code can
    assign thermodynamic states without caring which primary CV is active.  The
    contact switching threshold is embedded as a fixed constant, while per-bond
    contact_weight supports residue-balanced contact definitions.
    """
    if not contact_pairs:
        raise ValueError("Cannot add nonlocal-contact umbrella without contact pairs")
    # Embed the fixed switching constants directly in the child force expression
    # instead of adding child-force global parameters.  This keeps the Context
    # parameter namespace limited to the actual thermodynamic-state parameters
    # (r0 and k) plus contact_norm.
    contact_switch_r0_nm, contact_beta_nm_inv = contact_switch_constants_nm(args)
    # Numerically stable logistic contact switch.  This is algebraically
    # identical to 1/(1+exp(beta*(r-r0))) but avoids exponential overflow
    # on GPU kernels/fast math paths when the argument becomes large during
    # aggressive contact-CV pulling.
    contact_expr = f"contact_weight*0.5*(1-tanh(0.5*{contact_beta_nm_inv:.17g}*(r-{contact_switch_r0_nm:.17g})))"
    contact_sum = openmm.CustomBondForce(contact_expr)
    contact_sum.addPerBondParameter("contact_weight")
    for pair in contact_pairs:
        i, j = _contact_pair_indices(pair)
        w = _contact_pair_weight(pair)
        contact_sum.addBond(int(i), int(j), [float(w)])

    force = openmm.CustomCVForce("0.5*k*((contacts/contact_norm)-r0)^2")
    force.addCollectiveVariable("contacts", contact_sum)
    force.addGlobalParameter("contact_norm", contact_normalization_denominator(contact_pairs, args))
    force.addGlobalParameter("k", 0.0)
    force.addGlobalParameter("r0", 0.0)
    force.setForceGroup(int(force_group))
    system.addForce(force)
    return force

def self_test_nonlocal_contact_force(args) -> dict:
    """Construct and evaluate the contact umbrella in a tiny OpenMM Context.

    This is a dedicated runtime test for the part that cannot be validated by
    py_compile: a CustomBondForce used as the collective variable of a
    CustomCVForce.  It does not depend on PeptideBuilder, solvation, GaMD, or the
    full GaREUS workflow; it only proves that the selected OpenMM installation can
    instantiate the contact force, set r0/k, and evaluate a finite energy.
    """
    openmm, _app, unit = import_openmm()
    system = openmm.System()
    for _ in range(8):
        system.addParticle(12.0)
    if contact_scheme(args) == "residue-balanced":
        # Four atom terms grouped into two effective residue-pair contacts.
        pairs = [(0, 4, 0.5), (1, 5, 0.5), (2, 6, 0.5), (3, 7, 0.5)]
    else:
        pairs = [(0, 4, 1.0), (1, 5, 1.0), (2, 6, 1.0), (3, 7, 1.0)]
    force = add_contact_umbrella_force(openmm, system, pairs, args, force_group=0)
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    try:
        platform = openmm.Platform.getPlatformByName("Reference")
        context = openmm.Context(system, integrator, platform)
        platform_name = "Reference"
    except Exception:
        context = openmm.Context(system, integrator)
        platform_name = context.getPlatform().getName()

    positions_nm = np.asarray([
        [0.00, 0.00, 0.00],
        [0.10, 0.00, 0.00],
        [0.20, 0.00, 0.00],
        [0.30, 0.00, 0.00],
        [0.35, 0.00, 0.00],
        [0.70, 0.00, 0.00],
        [1.00, 0.00, 0.00],
        [1.20, 0.00, 0.00],
    ], dtype=float)
    context.setPositions(positions_nm * unit.nanometer)
    center = 0.50 if bool(getattr(args, "contact_normalize", True)) else 2.0
    context.setParameter("r0", float(center))
    context.setParameter("k", float(kcal_to_kj(20.0)))
    state = context.getState(getEnergy=True)
    energy_kj = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    try:
        cv_values = [float(x) for x in force.getCollectiveVariableValues(context)]
    except Exception:
        cv_values = []
    manual_cv = nonlocal_contact_cv_from_positions_nm(positions_nm, pairs, args)
    if not math.isfinite(energy_kj):
        raise RuntimeError("contact umbrella self-test produced non-finite energy")
    if not math.isfinite(manual_cv):
        raise RuntimeError("manual contact-CV evaluator produced non-finite value")
    try:
        del context
        del integrator
        gc.collect()
    except Exception:
        pass
    return {
        "ok": True,
        "platform": platform_name,
        "n_particles": 8,
        "n_contact_pairs": len(pairs),
        "contact_scheme": contact_scheme(args),
        "contact_normalize": bool(getattr(args, "contact_normalize", True)),
        "contact_norm": contact_normalization_denominator(pairs, args),
        "manual_primary_cv": float(manual_cv),
        "customcv_collective_values": cv_values,
        "umbrella_center": float(center),
        "umbrella_k_kcal_mol_CV2": 20.0,
        "energy_kj_mol": float(energy_kj),
    }
