"""Core command-line entrypoint for the modular GAREUS package."""

from __future__ import annotations

from typing import Iterable, Optional

__all__ = ["main"]


def main(argv: Optional[Iterable[str]] = None) -> None:
    """Run the GAREUS command-line workflow.

    This wrapper is used by console-script entry points, ``python -m gareus``,
    and the legacy top-level ``gareus_peptide.py`` shim.  Keep it small and
    import the heavy CLI lazily so importing :mod:`gareus.core` stays cheap.
    """
    from .cli import main as _main
    from .tui import restore_tui_cursor

    try:
        return _main(argv)
    finally:
        restore_tui_cursor()
