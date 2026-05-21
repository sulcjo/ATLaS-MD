"""
Constants used throughout the Gareus codebase.

These values were originally defined at the top of
``gareus_peptide.py``.  They are grouped here to provide
a single point of reference for residue naming and solvent/ion
identification.
"""

# Mapping of one‑letter amino acid codes to three‑letter codes.
AA3: dict[str, str] = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}

# Water residue names supported by the code.
WATER_RESNAMES: set[str] = {"HOH", "WAT", "SOL"}

# Ion residue names supported by the code.
ION_RESNAMES: set[str] = {"NA", "CL", "K", "MG", "CA", "ZN", "Na+", "Cl-"}

__all__ = ["AA3", "WATER_RESNAMES", "ION_RESNAMES"]