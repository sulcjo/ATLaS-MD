"""Backward-compatible command-line shim for the modular :mod:`gareus` package.

The implementation that used to live in this file has been moved into the
package namespace as ``gareus.legacy`` while the actively maintained pieces are
split across ``gareus.*`` modules.  This shim intentionally stays tiny so old
commands such as ``python gareus_peptide.py ...`` and old imports from
``gareus_peptide`` keep working during the transition.
"""

from __future__ import annotations

# Re-export the legacy namespace for backwards compatibility with scripts that
# import helper functions from gareus_peptide.  The legacy module itself now re-exports
# modular implementations without copied monolithic bodies or rebinding hacks.
from gareus.legacy import *  # noqa: F401,F403


def main(argv=None):
    """Run the modular command-line entrypoint."""
    from gareus.core import main as _main
    return _main(argv)


if __name__ == "__main__":
    from gareus.tui import restore_tui_cursor
    try:
        main()
    finally:
        restore_tui_cursor()
