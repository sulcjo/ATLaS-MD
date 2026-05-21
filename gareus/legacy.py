"""Compatibility namespace for the historical monolithic GAREUS script.

The old implementation has been split into focused modules.  This module keeps
legacy imports such as ``from gareus_peptide import run_gareus`` working by
re-exporting the modular implementations; it intentionally contains no copied
monolithic function bodies or final rebinding hacks.
"""

from __future__ import annotations

from .constants import *  # noqa: F401,F403
from .units import *  # noqa: F401,F403
from .colors import *  # noqa: F401,F403
from .formatting import *  # noqa: F401,F403
from .state import *  # noqa: F401,F403
from .imports import *  # noqa: F401,F403
from .lifecycle import *  # noqa: F401,F403
from .math_helpers import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .progress import *  # noqa: F401,F403
from .tui import *  # noqa: F401,F403
from .logger import *  # noqa: F401,F403
from .system_setup import *  # noqa: F401,F403
from .seeding import *  # noqa: F401,F403
from .cv import *  # noqa: F401,F403
from .windows import *  # noqa: F401,F403
from .diagnostics import *  # noqa: F401,F403
from .analysis import *  # noqa: F401,F403
from .forces import *  # noqa: F401,F403
from .production import *  # noqa: F401,F403
from .adaptive_feedback import *  # noqa: F401,F403
from .checkpoints import *  # noqa: F401,F403
from .config import *  # noqa: F401,F403
from .cli import parse_args, main  # noqa: F401

__all__ = [name for name in globals() if not name.startswith("__")]

if __name__ == "__main__":
    from .tui import restore_tui_cursor
    try:
        main()
    finally:
        restore_tui_cursor()
