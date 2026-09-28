"""Output retention tools (spec docs/superpowers/specs/2026-09-28-output-retention-design.md).

    python -m gareus.retention prune-checkpoints <run_dir> [--keep 4] [--apply]
    python -m gareus.retention split-progress <progress.jsonl> [--apply]
    python -m gareus.retention archive <run_dir> [--apply] [--force]
    python -m gareus.retention restore <run_dir> [--apply]

Every subcommand is a dry run unless --apply is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Iterable, Optional

from .correctness.checkpoint_store import MANIFEST_NAME, prune_generations

DEFAULT_KEEP = 4


def find_checkpoint_phases(run_dir: Path) -> list[Path]:
    """Every directory under run_dir that owns checkpoints/production_checkpoint_manifest.json."""
    run_dir = Path(run_dir)
    return sorted(p.parent.parent for p in run_dir.rglob(f"checkpoints/{MANIFEST_NAME}"))


def _lock_is_live(phase: Path) -> bool:
    lock = phase / ".gareus_run.lock"
    try:
        pid = int(lock.read_text().strip())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _generation_count(phase: Path) -> int:
    gens_dir = phase / "checkpoints" / "generations"
    if not gens_dir.is_dir():
        return 0
    return sum(1 for e in gens_dir.iterdir() if e.is_dir())


def prune_run_checkpoints(run_dir: Path, keep: int, apply: bool) -> list[dict]:
    """One row per checkpoint phase; a per-phase failure never aborts the others.

    status is one of "locked" (a live run owns the phase; never pruned),
    "error" (prune_generations raised; nothing in that phase was touched or, if
    it raised mid-delete, is reported as if it had not been), "dry_run", or
    "pruned". "skipped" (unreadable/orphaned generations prune_generations itself
    declined to touch) is only present on "dry_run"/"pruned" rows.
    """
    rows = []
    for phase in find_checkpoint_phases(run_dir):
        n_gen = _generation_count(phase)
        if _lock_is_live(phase):
            rows.append({"phase": str(phase), "status": "locked",
                         "generations": n_gen, "would_delete": 0, "bytes": 0})
            continue
        try:
            report = prune_generations(phase, keep, dry_run=not apply)
        except Exception as exc:  # noqa: BLE001 - isolate one bad phase from the rest
            rows.append({"phase": str(phase), "status": "error",
                         "generations": n_gen, "would_delete": 0, "bytes": 0,
                         "error": str(exc)})
            continue
        status = "pruned" if apply else "dry_run"
        rows.append({"phase": str(phase), "status": status, "generations": n_gen,
                     "would_delete": len(report["deleted"]), "bytes": report["bytes_freed"],
                     "skipped": len(report["skipped"])})
    return rows


def _print_rows(rows: Iterable[dict], apply: bool) -> None:
    total = 0
    for r in rows:
        if r["status"] == "error":
            print(f"error    {r['phase']}: {r['error']}")
            continue
        if r["status"] != "locked":
            total += r["bytes"]
        print(f"{r['status']:8s} {r['would_delete']:5d}/{r['generations']:<5d} "
              f"skip={r.get('skipped', 0):<3d} "
              f"{r['bytes'] / 1e9:9.2f} GB  {r['phase']}")
    verb = "freed" if apply else "would free (dry run; add --apply)"
    print(f"total {verb}: {total / 1e9:.2f} GB")


def _rewrite_lines(src: Path, dst: Path) -> tuple[int, int]:
    """Copy src to dst, replacing each distances event by its scalar summary."""
    n_lines = n_dist = 0
    with src.open("rb") as fin, dst.open("wb") as fout:
        for raw in fin:
            n_lines += 1
            if b'"event": "distances"' not in raw and b'"event":"distances"' not in raw:
                fout.write(raw)
                continue
            try:
                e = json.loads(raw)
            except ValueError:
                fout.write(raw)
                continue
            if e.get("event") != "distances":
                fout.write(raw)
                continue
            n_dist += 1
            summary = {k: v for k, v in e.items() if k not in ("distances", "dashboard")}
            summary["event"] = "distances_summary"
            fout.write((json.dumps(summary, sort_keys=True) + "\n").encode("utf-8"))
        fout.flush()
        os.fsync(fout.fileno())
    return n_lines, n_dist


def split_progress_jsonl(path: Path, apply: bool) -> dict:
    """Rewrite an existing progress.jsonl, replacing each distances event with its
    small distances_summary (the same event name production now writes live).

    Aborts without touching the original file if it changes size/mtime during the
    rewrite (a live run still appending to it), reporting status
    "changed_during_rewrite" rather than risking a lost concurrent write.
    """
    path = Path(path)
    before = path.stat()
    tmp = path.with_name(path.name + ".split.tmp")
    try:
        n_lines, n_dist = _rewrite_lines(path, tmp)
        result = {"lines": n_lines, "distances_lines": n_dist, "bytes_before": before.st_size,
                  "bytes_after": tmp.stat().st_size}
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            result["status"] = "changed_during_rewrite"
            return result
        if not apply:
            result["status"] = "dry_run"
            return result
        os.replace(tmp, path)
        result["status"] = "rewritten"
        return result
    finally:
        tmp.unlink(missing_ok=True)


ARCHIVE_DIR_NAMES = ("final_pdbs", "final_window_states", "us_starting_structures", "filtered_seed_bank")
ARCHIVE_SUFFIX = ".tar.zst"


def _zstd():
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("archive/restore need the 'zstandard' Python package") from exc
    return zstandard


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def archive_targets(run_dir: Path) -> list[Path]:
    """Every archivable directory under run_dir, deduped so a nested match (e.g. a
    seed_bank_* dir containing another archivable name) is only reported once, at its
    outermost level."""
    out: list[Path] = []
    for p in sorted(Path(run_dir).rglob("*")):
        if p.is_dir() and not p.is_symlink() and (p.name in ARCHIVE_DIR_NAMES or p.name.startswith("seed_bank_")):
            if not any(parent in out for parent in p.parents):
                out.append(p)
    return out


def _campaign_completed(run_dir: Path) -> bool:
    summary = Path(run_dir) / "adaptive_production" / "adaptive_production_driver_summary.json"
    try:
        return json.loads(summary.read_text()).get("status") == "completed"
    except (OSError, ValueError):
        return False


def _write_archive(src: Path, archive: Path) -> dict:
    """Stream every regular file under src into a new tar.zst, hashing as it goes.

    Symlinks are deliberately never archived (skipped, not followed - counted in the
    returned "skipped_symlinks" so callers can warn) so an archive this code writes
    can never itself carry a member restore must refuse. Every member is forced to
    REGTYPE with full content, rather than using tar.add()'s default same-session
    hard-link collapsing (a second file sharing an inode with one already added
    would otherwise be written as an LNKTYPE reference, which restore's
    isfile()-only guard would then have to refuse) - archived seed banks genuinely
    contain hard-linked PDBs (Task 8, filtered_seed_bank).
    """
    members = {}
    skipped_symlinks = 0
    zstandard = _zstd()
    cctx = zstandard.ZstdCompressor(level=19, write_checksum=True)
    with archive.open("wb") as fh, cctx.stream_writer(fh) as zw, tarfile.open(fileobj=zw, mode="w|") as tar:
        for p in sorted(src.rglob("*")):
            if p.is_symlink():
                skipped_symlinks += 1
                continue
            if not p.is_file():
                continue
            f = p
            rel = f.relative_to(src).as_posix()
            members[rel] = {"size": f.stat().st_size, "sha256": _sha256(f)}
            info = tar.gettarinfo(str(f), arcname=rel)
            # gettarinfo() zeroes .size (as well as setting .type) for a file it
            # detects shares an inode with one already added in this session - undo
            # both, since we always want the real bytes stored for every member.
            info.type = tarfile.REGTYPE
            info.linkname = ""
            info.size = members[rel]["size"]
            with f.open("rb") as data:
                tar.addfile(info, data)
    return {"members": members, "skipped_symlinks": skipped_symlinks}


def _iter_archive(archive: Path):
    """Yield (relpath, data) for every member of archive, refusing (SystemExit) any
    member that is not a plain regular file at a safe relative path - no absolute
    path, no '..' component, no symlink/hardlink/device/directory entry."""
    zstandard = _zstd()
    with archive.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr, \
            tarfile.open(fileobj=zr, mode="r|") as tar:
        for member in tar:
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts or not member.isfile():
                raise SystemExit(f"refusing archive member {name!r} in {archive}")
            yield name, tar.extractfile(member).read()


def _verify_archive(archive: Path, index: dict) -> bool:
    seen = {}
    try:
        for name, data in _iter_archive(archive):
            seen[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    except SystemExit:
        return False
    return seen == index["members"]


def archive_run(run_dir: Path, apply: bool, force: bool = False) -> list[dict]:
    run_dir = Path(run_dir)
    if not force and not _campaign_completed(run_dir):
        raise SystemExit(f"{run_dir}: campaign is not 'completed'; refusing to archive (use --force)")
    rows = []
    for d in archive_targets(run_dir):
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        row = {"dir": str(d), "bytes": size}
        if not apply:
            rows.append({**row, "status": "dry_run"})
            continue
        archive = d.with_name(d.name + ARCHIVE_SUFFIX)
        index_path = d.with_name(d.name + ARCHIVE_SUFFIX + ".index.json")
        if archive.exists() or index_path.exists():
            # Never open/overwrite a pre-existing archive or index - a re-run over a
            # directory already archived (or one that collides with a leftover file)
            # must leave both the directory and the existing archive untouched.
            rows.append({**row, "status": "exists"})
            continue
        result = _write_archive(d, archive)
        skipped = result.get("skipped_symlinks", 0)
        if skipped:
            print(f"WARNING: skipped {skipped} symlink(s) while archiving {d}", flush=True)
        index = {"members": result["members"]}
        if not _verify_archive(archive, index):
            archive.unlink(missing_ok=True)
            rows.append({**row, "status": "verify_failed", "skipped_symlinks": skipped})
            continue
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
        shutil.rmtree(d)
        rows.append({**row, "status": "archived", "archive_bytes": archive.stat().st_size,
                     "skipped_symlinks": skipped})
    return rows


def restore_run(run_dir: Path, apply: bool) -> list[dict]:
    """Restore every <dir>.tar.zst under run_dir back to <dir>, byte for byte.

    Refuses (status verify_failed, nothing written to dest) if dest already exists,
    if extraction hits a hostile/odd member (see _iter_archive), or if the restored
    file set doesn't match the index exactly - in every refusal case the original
    archive/index are left in place and any staging directory is cleaned up.
    """
    rows = []
    for archive in sorted(Path(run_dir).rglob("*" + ARCHIVE_SUFFIX)):
        index_path = archive.with_name(archive.name + ".index.json")
        dest = archive.with_name(archive.name[: -len(ARCHIVE_SUFFIX)])
        if not apply:
            rows.append({"archive": str(archive), "status": "dry_run"})
            continue
        if dest.exists():
            rows.append({"archive": str(archive), "status": "verify_failed",
                         "error": f"destination already exists: {dest}"})
            continue
        try:
            index = json.loads(index_path.read_text())
        except (OSError, ValueError) as exc:
            rows.append({"archive": str(archive), "status": "verify_failed",
                         "error": f"unreadable index {index_path}: {exc}"})
            continue
        staging = dest.with_name("." + dest.name + ".restore")
        shutil.rmtree(staging, ignore_errors=True)
        try:
            for name, data in _iter_archive(archive):
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        except SystemExit:
            shutil.rmtree(staging, ignore_errors=True)
            rows.append({"archive": str(archive), "status": "verify_failed"})
            continue
        restored = {f.relative_to(staging).as_posix(): {"size": f.stat().st_size, "sha256": _sha256(f)}
                    for f in staging.rglob("*") if f.is_file()}
        if restored != index["members"] or dest.exists():
            shutil.rmtree(staging, ignore_errors=True)
            rows.append({"archive": str(archive), "status": "verify_failed"})
            continue
        os.replace(staging, dest)
        archive.unlink()
        index_path.unlink()
        rows.append({"archive": str(archive), "status": "restored"})
    return rows


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gareus.retention", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pc = sub.add_parser("prune-checkpoints", help="keep the newest N checkpoint generations per phase")
    pc.add_argument("run_dir", type=Path)
    pc.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    pc.add_argument("--apply", action="store_true")
    sp = sub.add_parser("split-progress", help="drop per-replica distances dumps from a progress.jsonl")
    sp.add_argument("path", type=Path)
    sp.add_argument("--apply", action="store_true")
    ar = sub.add_parser("archive", help="pack PDB sets and final_window_states of a finished campaign")
    ar.add_argument("run_dir", type=Path)
    ar.add_argument("--apply", action="store_true")
    ar.add_argument("--force", action="store_true", help="archive even if the campaign is not completed")
    rs = sub.add_parser("restore", help="unpack every archive made by 'archive'")
    rs.add_argument("run_dir", type=Path)
    rs.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prune-checkpoints":
        if args.keep < 1:
            parser.error("--keep must be >= 1 for retro-pruning")
        rows = prune_run_checkpoints(args.run_dir, args.keep, args.apply)
        _print_rows(rows, args.apply)
        return 1 if any(r["status"] == "error" for r in rows) else 0
    if args.command == "split-progress":
        r = split_progress_jsonl(args.path, args.apply)
        print(f"{r['status']}: {r['distances_lines']}/{r['lines']} distances lines, "
              f"{r['bytes_before'] / 1e9:.2f} GB -> {r['bytes_after'] / 1e9:.2f} GB")
        return 0 if r["status"] in ("dry_run", "rewritten") else 1
    if args.command == "archive":
        rows = archive_run(args.run_dir, args.apply, args.force)
        for r in rows:
            print(f"{r['status']:13s} {r['bytes'] / 1e9:8.2f} GB  {r['dir']}")
        return 1 if any(r["status"] == "verify_failed" for r in rows) else 0
    if args.command == "restore":
        rows = restore_run(args.run_dir, args.apply)
        for r in rows:
            print(f"{r['status']:13s} {r['archive']}")
        return 0 if all(r["status"] != "verify_failed" for r in rows) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
