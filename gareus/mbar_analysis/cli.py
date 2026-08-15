"""Delegating CLI entry point for `gareus-analyze`.

Strangler-fig placeholder: owns no analysis logic yet. Later plans in this
modularization sequence progressively move analyze_gareus_mbar.py's real
logic into this subpackage; until then this module's only job is to make
`gareus-analyze` behave identically to `python analyze_gareus_mbar.py`.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    import analyze_gareus_mbar
    # Normalize sys.argv[0] so argparse's help output matches the old script.
    # When invoked as a delegating entry point, sys.argv[0] might be "-c" (python -c),
    # a console-script name, or other wrapper — but the help should show "analyze_gareus_mbar.py"
    # for consistency with the old invocation path.
    old_argv0 = sys.argv[0]
    try:
        sys.argv[0] = "analyze_gareus_mbar.py"
        analyze_gareus_mbar.main(argv)
    finally:
        sys.argv[0] = old_argv0
