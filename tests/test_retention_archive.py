from __future__ import annotations

import io
import json
import os
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
    os.link(original, linked)
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


# --- Fix round 1 (Controller Ruling 9) -------------------------------------


def test_main_archive_returns_1_when_a_row_is_verify_failed(tmp_path, monkeypatch, capsys):
    run = _campaign(tmp_path)
    monkeypatch.setattr(retention, "_verify_archive", lambda archive, index: False)

    rc = retention.main(["archive", str(run), "--apply"])

    assert rc == 1
    out = capsys.readouterr().out
    assert "verify_failed" in out


def test_main_archive_returns_0_when_every_row_archives_cleanly(tmp_path):
    run = _campaign(tmp_path)

    rc = retention.main(["archive", str(run), "--apply"])

    assert rc == 0


def test_main_restore_returns_1_when_one_archives_index_is_corrupt(tmp_path):
    run = _campaign(tmp_path)
    retention.archive_run(run, apply=True)
    archives = sorted(run.rglob("*" + retention.ARCHIVE_SUFFIX))
    assert len(archives) == 2  # seed_bank_final + final/baseline/final_pdbs, per _campaign()
    bad, good = archives
    bad_index = bad.with_name(bad.name + ".index.json")
    bad_index.write_text("{not valid json")
    good_dest = good.with_name(good.name[: -len(retention.ARCHIVE_SUFFIX)])
    bad_dest = bad.with_name(bad.name[: -len(retention.ARCHIVE_SUFFIX)])

    rc = retention.main(["restore", str(run), "--apply"])

    assert rc == 1
    # The good archive restored and was consumed.
    assert good_dest.exists()
    assert not good.exists()
    # The bad archive/index are left exactly as they were; nothing was extracted.
    assert bad.exists()
    assert bad_index.read_text() == "{not valid json"
    assert not bad_dest.exists()


def test_restore_missing_index_yields_verify_failed_and_leaves_archive(tmp_path):
    run = _campaign(tmp_path)
    retention.archive_run(run, apply=True)
    archive = next(run.rglob("seed_bank_final" + retention.ARCHIVE_SUFFIX))
    index_path = archive.with_name(archive.name + ".index.json")
    index_path.unlink()

    rows = retention.restore_run(run, apply=True)

    row = next(r for r in rows if r["archive"] == str(archive))
    assert row["status"] == "verify_failed"
    assert archive.exists()
    dest = archive.with_name(archive.name[: -len(retention.ARCHIVE_SUFFIX)])
    assert not dest.exists()


@pytest.mark.parametrize("which", ["archive", "index"])
def test_archive_run_never_overwrites_a_preexisting_archive_or_index(tmp_path, which):
    run = _campaign(tmp_path)
    target_dir = run / "adaptive_production" / "seed_bank_final"
    archive_path = target_dir.with_name(target_dir.name + retention.ARCHIVE_SUFFIX)
    index_path = target_dir.with_name(target_dir.name + retention.ARCHIVE_SUFFIX + ".index.json")
    if which == "archive":
        archive_path.write_bytes(b"PRE-EXISTING-ARCHIVE")
    else:
        index_path.write_text('{"members": {}}')

    rows = retention.archive_run(run, apply=True)

    row = next(r for r in rows if r["dir"] == str(target_dir))
    assert row["status"] == "exists"
    # Directory untouched - never opened, never removed.
    assert target_dir.exists()
    assert (target_dir / "pdbs" / "s.pdb").read_text() == "ATOM 2\n"
    if which == "archive":
        assert archive_path.read_bytes() == b"PRE-EXISTING-ARCHIVE"
        assert not index_path.exists()
    else:
        assert index_path.read_text() == '{"members": {}}'
        assert not archive_path.exists()


def test_archive_skips_symlinks_counts_them_and_warns(tmp_path, capsys):
    run = _campaign(tmp_path)
    target_dir = run / "adaptive_production" / "seed_bank_final"
    real_file = run / "adaptive_production" / "outside_the_bank.pdb"
    real_file.write_text("ATOM 9\n")
    link = target_dir / "pdbs" / "link.pdb"
    os.symlink(real_file, link)

    rows = retention.archive_run(run, apply=True)

    row = next(r for r in rows if r["dir"] == str(target_dir))
    assert row["status"] == "archived"
    assert row["skipped_symlinks"] == 1
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert str(target_dir) in out
    # The unaffected target's row is not warned about and carries no skip count.
    other_row = next(r for r in rows if r["dir"] != str(target_dir))
    assert other_row["skipped_symlinks"] == 0

    retention.restore_run(run, apply=True)
    restored_names = {p.name for p in (target_dir / "pdbs").iterdir()}
    assert restored_names == {"s.pdb"}  # the symlink itself was never archived


@pytest.mark.parametrize("index_text", ['{"other": 1}', '[1, 2]', '{"members": [1]}'])
def test_restore_index_without_members_is_verify_failed_and_others_continue(tmp_path, index_text):
    run = _campaign(tmp_path)
    retention.archive_run(run, apply=True)
    bad, good = sorted(run.rglob("*" + retention.ARCHIVE_SUFFIX))
    bad_index = bad.with_name(bad.name + ".index.json")
    bad_index.write_text(index_text)

    rows = retention.restore_run(run, apply=True)

    status = {r["archive"]: r["status"] for r in rows}
    assert status[str(bad)] == "verify_failed"
    assert status[str(good)] == "restored"
    assert bad.exists() and bad_index.read_text() == index_text
    assert not bad.with_name(bad.name[: -len(retention.ARCHIVE_SUFFIX)]).exists()
