from __future__ import annotations

import io
import json
import tarfile

import pytest

zstandard = pytest.importorskip("zstandard")

import gareus.retention as retention  # noqa: E402


def _campaign(tmp_path, status="completed"):
    run = tmp_path / "run"
    ap = run / "adaptive_production"
    (ap / "final" / "baseline" / "final_pdbs").mkdir(parents=True)
    (ap / "final" / "baseline" / "final_pdbs" / "replica_000_window_000.pdb").write_text("ATOM 1\n")
    (ap / "seed_bank_final" / "pdbs").mkdir(parents=True)
    (ap / "seed_bank_final" / "pdbs" / "s.pdb").write_text("ATOM 2\n")
    (ap / "seed_bank_final" / "final_survivor_seeds.csv").write_text("a\n1\n")
    (ap / "final" / "baseline" / "samples").mkdir()
    (ap / "final" / "baseline" / "samples" / "data.parquet").write_bytes(b"PAR1")
    (ap / "adaptive_production_driver_summary.json").write_text(json.dumps({"status": status}))
    return run


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_refuses_unfinished_campaign(tmp_path):
    run = _campaign(tmp_path, status="interrupted_after_checkpoint")
    with pytest.raises(SystemExit):
        retention.archive_run(run, apply=True)


def test_dry_run_changes_nothing(tmp_path):
    run = _campaign(tmp_path)
    before = _snapshot(run)
    rows = retention.archive_run(run, apply=False)
    assert {r["status"] for r in rows} == {"dry_run"} and _snapshot(run) == before


def test_round_trip_is_byte_identical(tmp_path):
    run = _campaign(tmp_path)
    before = _snapshot(run)
    retention.archive_run(run, apply=True)
    assert not (run / "adaptive_production" / "seed_bank_final").exists()
    assert (run / "adaptive_production" / "seed_bank_final.tar.zst").exists()
    assert (run / "adaptive_production" / "final" / "baseline" / "samples" / "data.parquet").exists()
    retention.restore_run(run, apply=True)
    assert _snapshot(run) == before


def test_corrupt_archive_keeps_originals(tmp_path, monkeypatch):
    run = _campaign(tmp_path)
    monkeypatch.setattr(retention, "_verify_archive", lambda archive, index: False)
    rows = retention.archive_run(run, apply=True)
    assert all(r["status"] == "verify_failed" for r in rows)
    assert (run / "adaptive_production" / "seed_bank_final" / "pdbs" / "s.pdb").exists()
    assert not list(run.rglob("*.tar.zst"))


def test_restore_refuses_when_destination_already_exists(tmp_path):
    run = _campaign(tmp_path)
    retention.archive_run(run, apply=True)
    target_dir = run / "adaptive_production" / "seed_bank_final"
    # Simulate the directory having reappeared after archiving (e.g. a re-run).
    target_dir.mkdir(parents=True)
    (target_dir / "sentinel.txt").write_text("do not touch")

    rows = retention.restore_run(run, apply=True)

    row = next(r for r in rows if r["archive"].endswith("seed_bank_final.tar.zst"))
    assert row["status"] == "verify_failed"
    # Archive/index still present (nothing was consumed); destination untouched.
    assert (run / "adaptive_production" / "seed_bank_final.tar.zst").exists()
    assert (run / "adaptive_production" / "seed_bank_final.tar.zst.index.json").exists()
    assert (target_dir / "sentinel.txt").read_text() == "do not touch"
    assert [p.name for p in target_dir.iterdir()] == ["sentinel.txt"]
    # No leftover staging directory.
    assert not (run / "adaptive_production" / ".seed_bank_final.restore").exists()


def test_intra_directory_hard_links_round_trip(tmp_path):
    """Task 8 hard-links PDBs into filtered_seed_bank/; two archived files sharing
    an inode must not be collapsed into a tarfile hard-link reference (that would
    make restore's regular-files-only guard refuse its own archive)."""
    run = _campaign(tmp_path)
    bank = run / "adaptive_production" / "filtered_seed_bank"
    bank.mkdir(parents=True)
    original = bank / "a.pdb"
    original.write_text("ATOM 1\n")
    linked = bank / "b.pdb"
    import os as _os
    _os.link(original, linked)
    assert original.stat().st_ino == linked.stat().st_ino

    before = _snapshot(run)
    rows = retention.archive_run(run, apply=True)
    assert {r["status"] for r in rows} == {"archived"}
    assert not bank.exists()
    retention.restore_run(run, apply=True)
    assert _snapshot(run) == before


def _write_hostile_archive(archive_path, member_name, symlink=False):
    """Hand-craft a tar.zst whose one member is a path-traversal or symlink attack."""
    cctx = zstandard.ZstdCompressor(level=3)
    with archive_path.open("wb") as fh:
        with cctx.stream_writer(fh) as zw:
            with tarfile.open(fileobj=zw, mode="w|") as tar:
                info = tarfile.TarInfo(name=member_name)
                if symlink:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "/etc/passwd"
                    tar.addfile(info)
                else:
                    data = b"pwned"
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize(
    "member_name,symlink",
    [
        ("/etc/passwd", False),
        ("../../escaped.txt", False),
        ("evil_link", True),
    ],
    ids=["absolute-path", "dotdot-path", "symlink-member"],
)
def test_restore_refuses_hostile_archive_member(tmp_path, member_name, symlink):
    run = tmp_path / "run"
    run.mkdir()
    archive = run / "evil.tar.zst"
    index_path = run / "evil.tar.zst.index.json"
    _write_hostile_archive(archive, member_name, symlink=symlink)
    index_path.write_text(json.dumps({"members": {}}))
    dest = run / "evil"

    rows = retention.restore_run(run, apply=True)

    assert len(rows) == 1
    assert rows[0]["status"] == "verify_failed"
    # Nothing was written to the intended destination or its staging area...
    assert not dest.exists()
    assert not (run / ".evil.restore").exists()
    # ...and nothing escaped outside the run directory tree.
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path.parent / "escaped.txt").exists()
