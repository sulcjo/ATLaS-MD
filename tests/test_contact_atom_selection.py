"""Unit tests for gareus.cv.contact_atom_allowed atom-selection modes.

No OpenMM import: atoms are plain mock objects exposing only ``.name``.
"""

from types import SimpleNamespace

import pytest

from gareus.cv import contact_atom_allowed


def atom(name):
    return SimpleNamespace(name=name)


BACKBONE_HEAVY = ["N", "CA", "C", "O", "OXT"]
BACKBONE_HYDROGENS = ["H", "H1", "H2", "H3", "HA", "HA2", "HA3"]
SIDECHAIN_HEAVY_EXAMPLES = ["CB", "CG", "CG1", "CG2", "OG", "OG1", "SD", "NZ", "CZ", "NE2"]
SIDECHAIN_HYDROGEN_EXAMPLES = ["HB", "HB1", "HB2", "HG1", "HD1", "HE1", "HZ", "HH11"]


@pytest.mark.parametrize("name", BACKBONE_HEAVY)
def test_sidechain_heavy_excludes_backbone_heavy(name):
    assert contact_atom_allowed(atom(name), "sidechain-heavy") is False


@pytest.mark.parametrize("name", SIDECHAIN_HEAVY_EXAMPLES)
def test_sidechain_heavy_includes_sidechain_heavy(name):
    assert contact_atom_allowed(atom(name), "sidechain-heavy") is True


@pytest.mark.parametrize("name", SIDECHAIN_HYDROGEN_EXAMPLES)
def test_sidechain_heavy_excludes_hydrogens(name):
    assert contact_atom_allowed(atom(name), "sidechain-heavy") is False


@pytest.mark.parametrize("name", BACKBONE_HEAVY)
def test_sidechain_all_excludes_backbone_heavy(name):
    assert contact_atom_allowed(atom(name), "sidechain-all") is False


@pytest.mark.parametrize("name", BACKBONE_HYDROGENS)
def test_sidechain_all_excludes_backbone_hydrogens(name):
    assert contact_atom_allowed(atom(name), "sidechain-all") is False


@pytest.mark.parametrize("name", SIDECHAIN_HEAVY_EXAMPLES)
def test_sidechain_all_includes_sidechain_heavy(name):
    assert contact_atom_allowed(atom(name), "sidechain-all") is True


@pytest.mark.parametrize("name", SIDECHAIN_HYDROGEN_EXAMPLES)
def test_sidechain_all_includes_sidechain_hydrogens(name):
    assert contact_atom_allowed(atom(name), "sidechain-all") is True


def test_glycine_has_no_sidechain_all_atoms():
    # Glycine's only atoms beyond backbone heavy are the alpha hydrogens,
    # which are backbone hydrogens (HA2/HA3), not sidechain.
    glycine_atom_names = ["N", "CA", "C", "O", "H", "HA2", "HA3"]
    assert not any(contact_atom_allowed(atom(n), "sidechain-all") for n in glycine_atom_names)
    assert not any(contact_atom_allowed(atom(n), "sidechain-heavy") for n in glycine_atom_names)


def test_unsupported_mode_raises():
    with pytest.raises(ValueError):
        contact_atom_allowed(atom("CB"), "not-a-real-mode")
