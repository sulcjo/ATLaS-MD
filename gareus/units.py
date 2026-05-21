"""
Unit conversion helpers.

These functions convert between the unconventional kcal/mol/Å² units used
by some umbrella sampling inputs and the canonical kJ/mol/nm² units.
They were originally defined in ``gareus_peptide.py``.
"""

def kcal_a2_to_kj_nm2(k_kcal_a2: float) -> float:
    """Convert a force constant from kcal/mol/Å² to kJ/mol/nm².

    The factor 4.184 converts kcal to kJ, and 0.01 converts Å² to nm².

    Args:
        k_kcal_a2: Force constant in kcal/mol/Å².

    Returns:
        The corresponding force constant in kJ/mol/nm².
    """
    return float(k_kcal_a2) * 4.184 / 0.01


def kj_nm2_to_kcal_a2(k_kj_nm2: float) -> float:
    """Convert a force constant from kJ/mol/nm² to kcal/mol/Å²."""
    return float(k_kj_nm2) * 0.01 / 4.184


def kcal_to_kj(k_kcal: float) -> float:
    """Convert an energy from kcal/mol to kJ/mol."""
    return float(k_kcal) * 4.184


def bias_energy_kj(bias_kcal: float) -> float:
    """Convert a bias energy from kcal/mol to kJ/mol.

    Bias energies are often specified in kcal/mol.  Converting to kJ/mol
    ensures consistency with OpenMM, which uses kJ by default.
    """
    return kcal_to_kj(bias_kcal)


__all__ = [
    "kcal_a2_to_kj_nm2",
    "kj_nm2_to_kcal_a2",
    "kcal_to_kj",
    "bias_energy_kj",
]