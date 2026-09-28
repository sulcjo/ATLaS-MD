from __future__ import annotations

from gareus.cli import parse_args
from gareus.production import prune_us_starting_pdbs


def test_flag_default_off():
    assert parse_args(["--seq", "AA", "--out", "u"]).prune_us_starting_structures is False
    assert parse_args(["--seq", "AA", "--out", "u", "--prune-us-starting-structures"]).prune_us_starting_structures


def test_removes_only_pdbs(tmp_path):
    d = tmp_path / "us_starting_structures"
    d.mkdir()
    for name in ("window_000_start.pdb", "window_001_CRASHFALLBACK_start.pdb"):
        (d / name).write_text("ATOM\n")
    for name in ("us_starting_structure_quality.json", "graft_report.json", "us_pulling_starting_structures.csv"):
        (d / name).write_text("{}")
    assert prune_us_starting_pdbs(tmp_path) == 2
    assert sorted(p.name for p in d.iterdir()) == [
        "graft_report.json", "us_pulling_starting_structures.csv", "us_starting_structure_quality.json"]
    assert prune_us_starting_pdbs(tmp_path) == 0


def test_missing_dir_is_fine(tmp_path):
    assert prune_us_starting_pdbs(tmp_path) == 0
