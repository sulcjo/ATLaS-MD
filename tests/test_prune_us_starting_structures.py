from __future__ import annotations

from gareus.cli import parse_args
from gareus.production import prune_us_starting_pdbs, _maybe_prune_us_starting


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


def test_helper_flag_off(tmp_path):
    """Helper does nothing when flag is off."""
    scratch = tmp_path / "scratch"
    main = tmp_path / "main"
    for d in [scratch, main]:
        us_dir = d / "us_starting_structures"
        us_dir.mkdir(parents=True)
        (us_dir / "window_000_start.pdb").write_text("ATOM\n")

    args = parse_args(["--seq", "AA", "--out", "u"])
    _maybe_prune_us_starting(args, scratch, main)

    # Nothing removed
    assert (scratch / "us_starting_structures" / "window_000_start.pdb").exists()
    assert (main / "us_starting_structures" / "window_000_start.pdb").exists()


def test_helper_flag_on_multiple_dirs(tmp_path):
    """Helper prunes PDBs from both scratch and main directories when flag is on."""
    scratch = tmp_path / "scratch"
    main = tmp_path / "main"
    for d in [scratch, main]:
        us_dir = d / "us_starting_structures"
        us_dir.mkdir(parents=True)
        (us_dir / "window_000_start.pdb").write_text("ATOM\n")
        (us_dir / "window_001_CRASHFALLBACK_start.pdb").write_text("ATOM\n")
        (us_dir / "us_starting_structure_quality.json").write_text("{}")

    args = parse_args(["--seq", "AA", "--out", "u", "--prune-us-starting-structures"])
    _maybe_prune_us_starting(args, scratch, main)

    # PDBs removed from both directories, reports kept
    for d in [scratch, main]:
        us_dir = d / "us_starting_structures"
        assert not (us_dir / "window_000_start.pdb").exists()
        assert not (us_dir / "window_001_CRASHFALLBACK_start.pdb").exists()
        assert (us_dir / "us_starting_structure_quality.json").exists()
