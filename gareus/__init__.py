"""
gareus package

This package modularizes the original ``gareus_peptide.py`` script into
discrete submodules.  The intent is to improve navigability and
testability while preserving the original command‑line interface.

Only a small subset of the original functionality has been extracted
into standalone modules at this stage.  Additional functions and
classes will be migrated over time.
"""

__version__ = "0.9.0"

# Re‑export commonly used helpers for convenience
from . import constants  # noqa: F401
from . import units  # noqa: F401
from . import colors  # noqa: F401
from . import formatting  # noqa: F401
from . import state  # noqa: F401
from . import imports  # noqa: F401
from . import lifecycle  # noqa: F401
from . import math_helpers  # noqa: F401
from . import io  # noqa: F401
from . import progress  # noqa: F401
from . import tui  # noqa: F401
from . import logger  # noqa: F401
from . import system_setup  # noqa: F401
from . import seeding  # noqa: F401
from . import cv  # noqa: F401
from . import windows  # noqa: F401
from . import diagnostics  # noqa: F401
from . import production  # noqa: F401
from . import adaptive_feedback  # noqa: F401
from . import adaptive_production  # noqa: F401
from . import cli  # noqa: F401
from . import checkpoints  # noqa: F401
from . import analysis  # noqa: F401
from . import forces  # noqa: F401
from . import provenance  # noqa: F401
from . import store  # noqa: F401
from . import query  # noqa: F401

__all__ = [
    "__version__",
    "constants",
    "units",
    "colors",
    "formatting",
    "state",
    "imports",
    "lifecycle",
    "math_helpers",
    "io",
    "progress",
    "tui",
    "logger",
    "system_setup",
    "seeding",
    "cv",
    "windows",
    "diagnostics",
    "production",
    "adaptive_feedback",
    "cli",
    "checkpoints",
    "analysis",
    "forces",
    "provenance",
    "adaptive_production",
    "store",
    "query",
]
