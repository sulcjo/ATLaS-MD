"""
Unit conversion helpers.

These functions convert between the unconventional kcal/mol/Å² units used
by some umbrella sampling inputs and the canonical kJ/mol/nm² units.
They were originally defined in ``gareus_peptide.py``.
"""

KJ_PER_KCAL = 4.184
K_B_KJ_PER_MOL_K = 0.00831446261815324  # Molar gas constant R = k_B * N_A, kJ/(mol*K)


def kcal_a2_to_kj_nm2(k_kcal_a2: float) -> float:
    """Convert a force constant from kcal/mol/Å² to kJ/mol/nm².

    KJ_PER_KCAL (4.184) converts kcal to kJ, and 0.01 converts Å² to nm².

    Args:
        k_kcal_a2: Force constant in kcal/mol/Å².

    Returns:
        The corresponding force constant in kJ/mol/nm².
    """
    return float(k_kcal_a2) * KJ_PER_KCAL / 0.01


def kj_nm2_to_kcal_a2(k_kj_nm2: float) -> float:
    """Convert a force constant from kJ/mol/nm² to kcal/mol/Å²."""
    return float(k_kj_nm2) * 0.01 / KJ_PER_KCAL


def kcal_to_kj(k_kcal: float) -> float:
    """Convert an energy from kcal/mol to kJ/mol."""
    return float(k_kcal) * KJ_PER_KCAL


def bias_energy_kj(bias_kcal: float) -> float:
    """Convert a bias energy from kcal/mol to kJ/mol.

    Bias energies are often specified in kcal/mol.  Converting to kJ/mol
    ensures consistency with OpenMM, which uses kJ by default.
    """
    return kcal_to_kj(bias_kcal)


__all__ = [
    "KJ_PER_KCAL",
    "K_B_KJ_PER_MOL_K",
    "kcal_a2_to_kj_nm2",
    "kj_nm2_to_kcal_a2",
    "kcal_to_kj",
    "bias_energy_kj",
]
