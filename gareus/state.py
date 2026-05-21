"""
State and collective variable helpers.

This module contains a handful of utility functions for extracting
distance and energy values from OpenMM contexts and states.  These
functions were originally defined in ``gareus_peptide.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Any

import numpy as np


def cv_distance_nm(context: Any, atom1: int, atom2: int, unit) -> float:
    """Compute the distance between two atoms in an OpenMM context (nm).

    Args:
        context: An OpenMM ``Context`` with positions.
        atom1: Index of the first atom.
        atom2: Index of the second atom.
        unit: The OpenMM ``unit`` module; only used for the conversion constant.

    Returns:
        The distance between ``atom1`` and ``atom2`` in nanometres.
    """
    state = context.getState(getPositions=True, enforcePeriodicBox=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return cv_distance_from_positions_nm(pos, atom1, atom2)


def cv_distance_and_potential_from_state(
    context: Any, atom1: int, atom2: int, unit
) -> Tuple[float, float]:
    """Return CV distance and potential energy from one OpenMM state read.

    Reading both the positions and energy in a single call avoids
    synchronisation overhead when running on a GPU.  The returned
    potential energy is in kJ/mol.
    """
    state = context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    cv_nm = cv_distance_from_positions_nm(pos, atom1, atom2)
    potential_kj = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    return cv_nm, potential_kj


def cv_distance_from_positions_nm(positions_nm: np.ndarray, atom1: int, atom2: int) -> float:
    """Compute the distance between two atoms from an array of positions in nm."""
    return float(np.linalg.norm(positions_nm[int(atom1)] - positions_nm[int(atom2)]))


def _scalar_to_float(value: Any) -> Optional[float]:
    """Convert plain scalars or OpenMM unit quantities to a Python float.

    If ``value`` is an OpenMM ``Quantity``, its internal unit is used.  If
    conversion fails, ``None`` is returned.
    """
    if value is None:
        return None
    # Try the direct float conversion first.
    try:
        return float(value)
    except Exception:
        pass
    # Fall back to the ``_value`` attribute used by OpenMM quantities.
    try:
        return float(value._value)  # type: ignore[attr-defined]
    except Exception:
        return None


def _energy_to_kj_mol(value: Any, unit) -> Optional[float]:
    """Convert a possible OpenMM energy quantity to kJ/mol, else a plain float."""
    if value is None:
        return None
    try:
        return float(value.value_in_unit(unit.kilojoule_per_mole))
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        pass
    try:
        return float(value._value)  # type: ignore[attr-defined]
    except Exception:
        return None


def get_energy_kj(state: Any, unit) -> Optional[float]:
    """Helper to extract potential energy from a state in kJ/mol."""
    try:
        return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    except Exception:
        return None


__all__ = [
    "cv_distance_nm",
    "cv_distance_and_potential_from_state",
    "cv_distance_from_positions_nm",
    "_scalar_to_float",
    "_energy_to_kj_mol",
    "get_energy_kj",
]