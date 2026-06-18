import types
from pathlib import Path

import pytest

from gareus.system_setup import resolve_input_pdb, validate_sequence


def test_validate_sequence_min_two_default_rejects_single():
    with pytest.raises(ValueError):
        validate_sequence("A")  # default require_min_two=True


def test_validate_sequence_allows_single_when_min_two_disabled():
    # input_pdb runs pass a nominal seq; length gate must be skippable
    assert validate_sequence("A", require_min_two=False) == "A"


def test_validate_sequence_still_rejects_empty_and_bad_when_relaxed():
    with pytest.raises(ValueError):
        validate_sequence("", require_min_two=False)
    with pytest.raises(ValueError):
        validate_sequence("Z", require_min_two=False)  # non-canonical residue


def _args(**kw):
    return types.SimpleNamespace(**kw)


def test_resolve_returns_none_when_unset():
    assert resolve_input_pdb(_args(input_pdb=None)) is None
    assert resolve_input_pdb(_args(input_pdb="")) is None
    assert resolve_input_pdb(_args()) is None  # attribute absent


def test_resolve_returns_absolute_path_for_existing_file(tmp_path):
    pdb = tmp_path / "fix.pdb"
    pdb.write_text("END\n")
    got = resolve_input_pdb(_args(input_pdb="fix.pdb"), base_dir=tmp_path)
    assert got == pdb
    assert got.is_absolute()


def test_resolve_absolute_input_is_returned_as_is(tmp_path):
    pdb = tmp_path / "fix.pdb"
    pdb.write_text("END\n")
    got = resolve_input_pdb(_args(input_pdb=str(pdb)), base_dir=Path("/nonexistent"))
    assert got == pdb


def test_resolve_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_input_pdb(_args(input_pdb="nope.pdb"), base_dir=tmp_path)
