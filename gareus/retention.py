"""Output retention tools (spec docs/superpowers/specs/2026-09-28-output-retention-design.md).

    python -m gareus.retention prune-checkpoints <run_dir> [--keep 4] [--apply]

Every subcommand is a dry run unless --apply is given.
"""
from __future__ import annotations

import argparse
import os
import sys
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


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gareus.retention", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pc = sub.add_parser("prune-checkpoints", help="keep the newest N checkpoint generations per phase")
    pc.add_argument("run_dir", type=Path)
    pc.add_argument("--keep", type=int, default=DEFAULT_KEEP)
    pc.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prune-checkpoints":
        if args.keep < 1:
            parser.error("--keep must be >= 1 for retro-pruning")
        rows = prune_run_checkpoints(args.run_dir, args.keep, args.apply)
        _print_rows(rows, args.apply)
        return 1 if any(r["status"] == "error" for r in rows) else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
