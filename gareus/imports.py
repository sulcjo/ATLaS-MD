"""
Lazy import helpers.

OpenMM and other optional dependencies are expensive to import.  These
functions wrap the imports in try/except blocks and provide clear
error messages if a dependency is missing.  They were originally
defined in ``gareus_peptide.py``.
"""

def import_openmm():
    """Import and return the OpenMM packages.

    Returns:
        A tuple ``(openmm, app, unit)`` upon success.

    Raises:
        RuntimeError: If OpenMM is not installed.
    """
    try:
        import openmm  # type: ignore[import-not-found]
        import openmm.app as app  # type: ignore[import-not-found]
        import openmm.unit as unit  # type: ignore[import-not-found]
        return openmm, app, unit
    except Exception as exc:  # pragma: no cover - runtime error path
        raise RuntimeError("This script needs OpenMM installed.") from exc


def import_gamd_factory():
    """Import the GaMD integrator factory from ``gamd-openmm``.

    Returns:
        The :class:`gamd.integrator_factory.GamdIntegratorFactory`.

    Raises:
        RuntimeError: If the package cannot be imported.
    """
    try:
        from gamd.integrator_factory import GamdIntegratorFactory  # type: ignore[import-not-found]
        return GamdIntegratorFactory
    except Exception as exc:  # pragma: no cover - runtime error path
        raise RuntimeError(
            "Could not import gamd-openmm. Install it from https://github.com/MiaoLab20/gamd-openmm."
        ) from exc


def import_peptidebuilder():
    """Import the PeptideBuilder package and return common utilities.

    Returns:
        A tuple ``(PeptideBuilder, Geometry, PDBIO)`` upon success.

    Raises:
        RuntimeError: If PeptideBuilder or its BioPython dependency is missing.
    """
    try:
        import PeptideBuilder  # type: ignore[import-not-found]
        from PeptideBuilder import Geometry  # type: ignore[import-not-found]
        from Bio.PDB import PDBIO  # type: ignore[import-not-found]
        return PeptideBuilder, Geometry, PDBIO
    except Exception as exc:  # pragma: no cover - runtime error path
        raise RuntimeError(
            "Could not import PeptideBuilder and its BioPython dependency."
        ) from exc


__all__ = ["import_openmm", "import_gamd_factory", "import_peptidebuilder"]