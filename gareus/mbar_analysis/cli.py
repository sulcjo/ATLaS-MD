"""Delegating CLI entry point for `gareus-analyze`.

Strangler-fig placeholder: owns no analysis logic yet. Later plans in this
modularization sequence progressively move analyze_gareus_mbar.py's real
logic into this subpackage; until then this module's only job is to make
`gareus-analyze` behave identically to `python analyze_gareus_mbar.py`.

Requires the repository root on `sys.path` (so `import analyze_gareus_mbar`
resolves) until Plan A6e retires the old direct-invocation script entirely.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> int:
    # This import is deliberately function-local, not module-level. gareus/units.py
    # is imported at module scope by analyze_gareus_mbar.py (script -> package), so a
    # module-level `import analyze_gareus_mbar` here would close a cycle
    # (analyze_gareus_mbar -> gareus.units -> gareus/__init__ -> ... -> mbar_analysis
    # -> analyze_gareus_mbar) the moment any later plan hoists a package -> script
    # import to module scope elsewhere in gareus/mbar_analysis/. Keep this import
    # function-local until A6e retires the old script.
    import analyze_gareus_mbar
    # Normalize sys.argv[0] so argparse's help output matches the old script.
    # When invoked as a delegating entry point, sys.argv[0] might be "-c" (python -c),
    # a console-script name, or other wrapper — but the help should show "analyze_gareus_mbar.py"
    # for consistency with the old invocation path.
    # TODO(A6e): remove this normalization once analyze_gareus_mbar.py's direct
    # invocation is retired and this entry point can report its own real name.
    old_argv0 = sys.argv[0]
    try:
        sys.argv[0] = "analyze_gareus_mbar.py"
        return analyze_gareus_mbar.main(argv)
    finally:
        sys.argv[0] = old_argv0
