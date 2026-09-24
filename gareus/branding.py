"""User-facing product name and version label.

The package, its console commands (``gareus``, ``gareus-analyze``, ...) and its
on-disk artifacts (``gareus_metadata.json``, ``GAREUS_*`` env vars) keep their
legacy names; only the name *shown to people* is ATLaS-MD.  Every visible
"ATLaS-MD vX.Y.Z" string is built here from ``gareus.__version__`` (never from
installed dist metadata, which can lag an editable checkout).
"""

from __future__ import annotations

from . import __version__

PRODUCT_NAME = "ATLaS-MD"


def product_label() -> str:
    """Return the versioned product label, e.g. ``"ATLaS-MD v0.8.3"``."""
    return f"{PRODUCT_NAME} v{__version__}"


def versioned_title(title: str) -> str:
    """Insert the version after the product name in a heading.

    ``"ATLaS-MD peptide workflow"`` -> ``"ATLaS-MD v0.8.3 peptide workflow"``;
    a title without the product name is prefixed instead.  Idempotent.
    """
    label = product_label()
    if label in title:
        return title
    if PRODUCT_NAME in title:
        return title.replace(PRODUCT_NAME, label, 1)
    return f"{label} -- {title}"
