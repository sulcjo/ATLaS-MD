# Output Retention and Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut a campaign's disk footprint (chignolin_9: 688 GB) by pruning superseded checkpoint generations, moving the per-replica `distances` dumps out of `progress.jsonl` into a bounded per-phase ring, hard-linking filtered seed banks, and adding opt-in tools to prune/archive finished outputs — without changing anything resume, the adaptive driver, or analysis reads.

**Architecture:** R1 adds generation pruning inside `gareus/correctness/checkpoint_store.py` (called under the existing publication lock after a publish or copy, opt-in via `--checkpoint-keep-generations`). R2 adds a two-file `RotatingJsonlWriter` (`gareus/io.py`) that `DistanceLogger` uses per phase directory, while `progress.jsonl` keeps a ~1 kB scalar summary event; `gareus_monitor.py` learns to read the ring with a legacy fallback. R3/R4 are small changes plus one tool module `gareus/retention.py` (`python -m gareus.retention {prune-checkpoints,split-progress,archive,restore}`).

**Tech Stack:** Python (production runs on aurum2's calc env, **Python 3.9**), OpenMM checkpoint store (pure Python part only), `zstandard` 0.23 (optional import), `tarfile`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-output-retention-design.md` (reviewed 2026-09-28; decisions in its §7).

## Global Constraints

- Code must run on **Python 3.9** (aurum2 calc env): no `match`, no `tarfile` `filter=` argument, no `X | Y` type unions outside `from __future__ import annotations` modules.
- `--checkpoint-keep-generations` default **0** = keep every generation (today's behaviour); recommended value 4. Never delete the generation the root manifest names; never delete any phase's root `production_checkpoint_manifest.json`.
- `--live-distances-max-mb` default **256**; `0` restores the legacy behaviour (full `distances` event into `progress.jsonl`, no ring).
- `--prune-us-starting-structures` default **off**; the archive/restore and split tools are opt-in and default to a dry run (`--apply` to act).
- Parquet samples/exchanges, trajectories, `tica_obs`, window/registry/manifests, `final_window_states/` and `final_pdbs/` are never modified by R1-R3.
- Tests run through opencode (a hook blocks direct pytest): `opencode run "run: pytest -q <files>"`. Run only the tests named in each task (user rule: targeted tests only).
- Commit messages: conventional commits (`feat:`, `test:`, `docs:`), ending with the session's attribution lines.

## Review Focus

1. **Live run pruning races a publish.** A retro-prune (Task 3) run while a job is publishing must never delete the newest generation or the one being written: only generations reachable backwards from the root manifest beyond the first `keep` are candidates, and orphan generations newer than the root (`absolute_step` > root's) are kept. Test in Task 1 (orphan kept) and Task 3 (locked phase skipped).
2. **Scratch → main copy keeps accumulating.** With `--scratchdir`, main receives one copied generation per checkpoint; if only scratch prunes, main grows forever. Task 2 tests that `sync_run_tree_quiescent(..., keep_generations=N)` prunes the destination.
3. **Monitor on an old run or a phase with no ring yet.** Must fall back to `progress.jsonl` exactly as today. Task 6 tests legacy-only and ring-present layouts.
4. **Ring rotation mid-read.** The monitor may read `live_distances.jsonl` just after rotation (tiny current file). It must still reach depth via `live_distances.1.jsonl`. Task 6 test.
5. **Split tool on a live `progress.jsonl`.** If the file grows while being rewritten, the tool must abort without replacing it. Task 7 test.

---

### Task 1: Checkpoint generation pruning in the store

**Files:**
- Modify: `gareus/correctness/checkpoint_store.py` (module docstring lines 8-10; `publish_generation` ~line 134; `copy_committed_generation` ~line 336; new functions after `require_available`)
- Test: `tests/test_checkpoint_generation_pruning.py` (create)

**Interfaces:**
- Produces: `prune_generations(out_dir, keep: int) -> dict` (takes the writer lock itself; returns `{"kept": [ids], "deleted": [ids], "skipped": [(id, reason)], "bytes_freed": int}`); `_prune_locked(root: Path, keep: int) -> dict` (caller holds `writer_lock(root)`); new keyword `keep_generations: int = 0` on `publish_generation` and `copy_committed_generation`.

- [ ] **Step 1: Write the failing tests**

```python
"""Checkpoint generation retention (spec 2026-09-28-output-retention-design, R1)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gareus.correctness.checkpoint_store import (
    MANIFEST_NAME, copy_committed_generation, prune_generations, publish_generation,
    read_validated_generation,
)


def _fields(step: int) -> dict:
    return {
        "assignments": [0, 1], "prod_done": step, "absolute_step": step, "parity": 0,
        "attempt": 0, "next_exchange": step, "next_log": step, "exchange_stats": {},
        "rng_bit_generator": "PCG64", "rng_state": {},
    }


def _publish(out: Path, step: int, keep: int = 0) -> str:
    m = publish_generation(out, [b"a" * 64, b"b" * 64], _fields(step), keep_generations=keep)
    return m["generation_id"]


def _gens(out: Path) -> set:
    return {p.name for p in (out / "checkpoints" / "generations").iterdir()}


def test_default_keep_zero_deletes_nothing(tmp_path):
    ids = [_publish(tmp_path, s) for s in (100, 200, 300)]
    assert _gens(tmp_path) == set(ids)


def test_publish_keeps_newest_n_by_chain(tmp_path):
    ids = [_publish(tmp_path, s, keep=2) for s in (100, 200, 300, 400)]
    assert _gens(tmp_path) == set(ids[-2:])
    assert read_validated_generation(tmp_path).manifest["generation_id"] == ids[-1]


def test_order_follows_previous_chain_not_mtime(tmp_path):
    ids = [_publish(tmp_path, s) for s in (100, 200, 300)]
    gens = tmp_path / "checkpoints" / "generations"
    os.utime(gens / ids[0], (4_000_000_000, 4_000_000_000))  # oldest gets the newest mtime
    report = prune_generations(tmp_path, keep=1)
    assert _gens(tmp_path) == {ids[2]}
    assert set(report["deleted"]) == {ids[0], ids[1]}


def test_orphan_newer_than_root_is_kept(tmp_path):
    ids = [_publish(tmp_path, s) for s in (100, 200)]
    root_bytes = (tmp_path / "checkpoints" / MANIFEST_NAME).read_bytes()
    orphan = _publish(tmp_path, 300)
    # Simulate a crash after the generation rename but before the root was replaced.
    (tmp_path / "checkpoints" / MANIFEST_NAME).write_bytes(root_bytes)
    prune_generations(tmp_path, keep=1)
    assert _gens(tmp_path) == {ids[1], orphan}


def test_unreadable_generation_manifest_is_skipped(tmp_path):
    ids = [_publish(tmp_path, s) for s in (100, 200, 300)]
    (tmp_path / "checkpoints" / "generations" / ids[0] / "manifest.json").write_text("{not json")
    report = prune_generations(tmp_path, keep=1)
    assert ids[0] in _gens(tmp_path)
    assert any(gid == ids[0] for gid, _ in report["skipped"])


def test_keep_below_zero_rejected(tmp_path):
    _publish(tmp_path, 100)
    with pytest.raises(ValueError):
        prune_generations(tmp_path, keep=-1)


def test_copy_prunes_destination(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "main"
    for step in (100, 200, 300):
        _publish(src, step)
        copy_committed_generation(src, dst, keep_generations=2)
    assert len(_gens(dst)) == 2
    assert read_validated_generation(dst).manifest["absolute_step"] == 300


def test_absent_checkpoint_prunes_nothing(tmp_path):
    assert prune_generations(tmp_path, keep=2)["deleted"] == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `opencode run "run: pytest -q tests/test_checkpoint_generation_pruning.py"`
Expected: FAIL with `ImportError: cannot import name 'prune_generations'`.

- [ ] **Step 3: Implement**

Replace the module docstring's last paragraph (lines 8-10) with:

```python
Generation retention is opt-in: ``publish_generation``/``copy_committed_generation``
accept ``keep_generations`` (0 = keep all). Pruning follows the
``previous_generation_id`` chain back from the root manifest, never directory
mtimes, and never touches the root's generation or orphans newer than it.
Filesystems must support POSIX locks, same-FS rename, and directory fsync.
Network-storage durability must be validated locally.
"""
```

Add `keep_generations: int = 0,` as the last keyword of `publish_generation` and, inside the `with writer_lock(root):` block, replace `return fields` with:

```python
            if keep_generations:
                _prune_locked(root, keep_generations)
            return fields
```

Add `keep_generations: int = 0,` as the last keyword of `copy_committed_generation` and, after `_event(fault_hook, "copy_root_published")` (still inside its `with writer_lock(dest_root):`), add:

```python
        if keep_generations:
            _prune_locked(dest_root, keep_generations)
```

Append after `require_available`:

```python
def _generation_manifest(generations: Path, generation_id: str) -> dict[str, Any] | None:
    try:
        data = json_loads((generations / generation_id / "manifest.json").read_bytes())
    except (OSError, IntegrityError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("generation_id") != generation_id:
        return None
    return data


def _dir_bytes(path: Path) -> int:
    total = 0
    for directory, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(directory) / name).stat().st_size
            except OSError:
                pass
    return total


def _prune_locked(root: Path, keep: int) -> dict[str, Any]:
    """Delete superseded generations; caller holds ``writer_lock(root)``."""
    keep = int(keep)
    if keep < 0:
        raise ValueError(f"keep must be >= 0, got {keep}")
    report: dict[str, Any] = {"kept": [], "deleted": [], "skipped": [], "bytes_freed": 0}
    encoded = _root_bytes(root)
    generations = root / "generations"
    if keep == 0 or encoded is None or not generations.is_dir():
        return report
    root_manifest = json_loads(encoded)
    if not isinstance(root_manifest, dict) or not isinstance(root_manifest.get("generation_id"), str):
        return report
    root_step = root_manifest.get("absolute_step")
    # The newest `keep` generations, following previous_generation_id from the root.
    kept: list[str] = []
    current = root_manifest["generation_id"]
    while current and len(kept) < keep:
        kept.append(current)
        data = _generation_manifest(generations, current)
        if data is None:
            break
        current = data.get("previous_generation_id")
    report["kept"] = list(kept)
    for entry in sorted(generations.iterdir()):
        if not entry.is_dir() or entry.name in kept:
            continue
        data = _generation_manifest(generations, entry.name)
        if data is None:
            report["skipped"].append((entry.name, "unreadable or mismatched manifest.json"))
            continue
        step = data.get("absolute_step")
        if not isinstance(step, int) or not isinstance(root_step, int) or step > root_step:
            report["skipped"].append((entry.name, "newer than the root manifest (unpublished orphan)"))
            continue
        size = _dir_bytes(entry)
        shutil.rmtree(entry)
        report["deleted"].append(entry.name)
        report["bytes_freed"] += size
    if report["deleted"]:
        fsync_directory(generations)
    return report


def prune_generations(out_dir: Path | str, keep: int) -> dict[str, Any]:
    """Keep the newest ``keep`` generations of one phase; ``keep=0`` deletes nothing."""
    if int(keep) < 0:
        raise ValueError(f"keep must be >= 0, got {keep}")
    root = _storage_root(out_dir)
    if not root.is_dir():
        return {"kept": [], "deleted": [], "skipped": [], "bytes_freed": 0}
    with writer_lock(root):
        return _prune_locked(root, keep)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `opencode run "run: pytest -q tests/test_checkpoint_generation_pruning.py tests/test_adaptive_new_state_guards.py"`
Expected: PASS (the second file exercises `publish_generation` with default arguments).

- [ ] **Step 5: Commit**

```bash
git add gareus/correctness/checkpoint_store.py tests/test_checkpoint_generation_pruning.py
git commit -m "feat: opt-in checkpoint generation pruning in the checkpoint store"
```

---

### Task 2: Wire `--checkpoint-keep-generations` into production and scratch sync

**Files:**
- Modify: `gareus/cli.py` (new argument next to `--production-phase-timers`, ~line 898; final sync call ~line 2278)
- Modify: `gareus/production.py` (`save_production_checkpoint` signature ~line 5125 and its `publish_generation` call ~line 5239; both call sites ~lines 8441 and 8541; `sync_scratch_to_main` ~line 5241 and its three callers at ~8456, ~8560)
- Modify: `gareus/correctness/repo_adapters.py` (`sync_run_tree_quiescent`, ~line 56)
- Test: `tests/test_checkpoint_keep_generations_wiring.py` (create)

**Interfaces:**
- Consumes: `publish_generation(..., keep_generations=)`, `copy_committed_generation(..., keep_generations=)` from Task 1.
- Produces: `args.checkpoint_keep_generations: int`; `save_production_checkpoint(..., keep_generations: int = 0)`; `sync_scratch_to_main(scratch_dir, main_dir, keep_generations: int = 0)`; `sync_run_tree_quiescent(source, destination, keep_generations: int = 0)`.

- [ ] **Step 1: Write the failing tests**

```python
"""--checkpoint-keep-generations wiring (spec 2026-09-28-output-retention-design, R1)."""
from __future__ import annotations

import inspect

import pytest

from gareus.cli import parse_args
from gareus.correctness.checkpoint_store import publish_generation
from gareus.correctness.repo_adapters import sync_run_tree_quiescent

MINIMAL = ["--seq", "AA", "--out", "unused"]


def _fields(step):
    return {"assignments": [0], "prod_done": step, "absolute_step": step, "parity": 0,
            "attempt": 0, "next_exchange": step, "next_log": step, "exchange_stats": {},
            "rng_bit_generator": "PCG64", "rng_state": {}}


def test_flag_defaults_to_zero_and_reads_cli_and_yaml(tmp_path):
    assert parse_args(MINIMAL).checkpoint_keep_generations == 0
    assert parse_args(MINIMAL + ["--checkpoint-keep-generations", "4"]).checkpoint_keep_generations == 4
    cfg = tmp_path / "c.yaml"
    cfg.write_text("checkpoint_keep_generations: 4\n")
    assert parse_args(MINIMAL + ["--config", str(cfg)]).checkpoint_keep_generations == 4


def test_negative_value_rejected():
    with pytest.raises(SystemExit):
        parse_args(MINIMAL + ["--checkpoint-keep-generations", "-1"])


def test_save_and_sync_accept_keep_generations():
    from gareus import production
    assert "keep_generations" in inspect.signature(production.save_production_checkpoint).parameters
    assert "keep_generations" in inspect.signature(production.sync_scratch_to_main).parameters


def test_quiescent_sync_prunes_destination(tmp_path):
    src, dst = tmp_path / "scratch", tmp_path / "main"
    src.mkdir()
    (src / "note.txt").write_text("x")
    for step in (100, 200, 300, 400, 500):
        publish_generation(src, [b"z" * 32], _fields(step))
        sync_run_tree_quiescent(src, dst, keep_generations=2)
    assert len(list((dst / "checkpoints" / "generations").iterdir())) == 2
    assert (dst / "note.txt").read_text() == "x"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `opencode run "run: pytest -q tests/test_checkpoint_keep_generations_wiring.py"`
Expected: FAIL (`AttributeError: ... checkpoint_keep_generations`).

- [ ] **Step 3: Implement**

In `gareus/cli.py`, after the `--production-phase-timers` argument:

```python
    p.add_argument("--checkpoint-keep-generations", type=int, default=0,
                   help="Keep only the newest N production checkpoint generations per phase and "
                        "delete older ones after each checkpoint (and on the main copy after a "
                        "--scratchdir sync). 0 = keep every generation (default). Recommended 4. "
                        "Only the newest generation is ever read on resume. "
                        "Spec 2026-09-28-output-retention-design.")
```

In the same file, find the function where `_validate_replica_admission_args` is called from `parse_args` and add next to it:

```python
    if int(getattr(args, "checkpoint_keep_generations", 0) or 0) < 0:
        parser.error("--checkpoint-keep-generations must be >= 0")
```

(use the local names that function already has for the parser and namespace).

In `gareus/production.py`, add `keep_generations: int = 0` as the last parameter of `save_production_checkpoint` and change its call to `publish_generation(out_dir, replica_payloads, manifest, keep_generations=int(keep_generations or 0))`. Change `sync_scratch_to_main`:

```python
def sync_scratch_to_main(scratch_dir: Path, main_dir: Path, keep_generations: int = 0) -> None:
    """Checkpoint-safe quiescent copy; mutable sample files are individually atomic.

    This is not yet a generation-index transaction across all Parquet output.
    Any failure remains visible instead of claiming a successful backup.
    ``keep_generations`` prunes the main copy's generations the same way.
    """
    from .correctness.repo_adapters import sync_run_tree_quiescent
    sync_run_tree_quiescent(scratch_dir, main_dir, keep_generations=keep_generations)
```

At both `save_production_checkpoint(` call sites add `keep_generations=int(getattr(args, "checkpoint_keep_generations", 0) or 0),` after `npt_runtime=npt_runtime,`, and change both `sync_scratch_to_main(out_dir, Path(_scratch_main))` calls in that loop to `sync_scratch_to_main(out_dir, Path(_scratch_main), keep_generations=int(getattr(args, "checkpoint_keep_generations", 0) or 0))`. In `gareus/cli.py` ~line 2278 do the same for `sync_scratch_to_main(out_dir, Path(_final_main))`.

In `gareus/correctness/repo_adapters.py`, change the signature to `def sync_run_tree_quiescent(source, destination, keep_generations: int = 0) -> None:` and its last line to `copy_committed_generation(source, destination, keep_generations=keep_generations)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `opencode run "run: pytest -q tests/test_checkpoint_keep_generations_wiring.py tests/test_checkpoint_generation_pruning.py tests/test_phase_timers.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/cli.py gareus/production.py gareus/correctness/repo_adapters.py tests/test_checkpoint_keep_generations_wiring.py
git commit -m "feat: --checkpoint-keep-generations for production checkpoints and scratch sync"
```

---

### Task 3: `python -m gareus.retention prune-checkpoints` for existing runs

**Files:**
- Create: `gareus/retention.py`
- Test: `tests/test_retention_prune_checkpoints.py` (create)

**Interfaces:**
- Consumes: `prune_generations`, `checkpoint_manifest_path` from Task 1 / existing store.
- Produces: `find_checkpoint_phases(run_dir: Path) -> list[Path]`; `prune_run_checkpoints(run_dir: Path, keep: int, apply: bool) -> list[dict]` (one row per phase: `phase`, `status` in `{"pruned","dry_run","locked"}`, `generations`, `would_delete`, `bytes`); `main(argv=None) -> int`. Later tasks add subcommands to the same `main`.

- [ ] **Step 1: Write the failing tests**

```python
"""Retro-pruning of existing runs (spec R1c)."""
from __future__ import annotations

import os
from pathlib import Path

from gareus.correctness.checkpoint_store import publish_generation
from gareus.retention import find_checkpoint_phases, main, prune_run_checkpoints


def _fields(step):
    return {"assignments": [0], "prod_done": step, "absolute_step": step, "parity": 0,
            "attempt": 0, "next_exchange": step, "next_log": step, "exchange_stats": {},
            "rng_bit_generator": "PCG64", "rng_state": {}}


def _phase(run: Path, rel: str, n: int) -> Path:
    d = run / rel
    for step in range(1, n + 1):
        publish_generation(d, [b"q" * 128], _fields(step * 100))
    return d


def _ngen(d: Path) -> int:
    return len(list((d / "checkpoints" / "generations").iterdir()))


def test_finds_every_phase(tmp_path):
    a = _phase(tmp_path, "adaptive_production/epoch_000", 2)
    b = _phase(tmp_path, "adaptive_production/final/baseline", 2)
    assert set(find_checkpoint_phases(tmp_path)) == {a, b}


def test_dry_run_deletes_nothing_and_reports(tmp_path):
    d = _phase(tmp_path, "adaptive_production/final/baseline", 6)
    rows = prune_run_checkpoints(tmp_path, keep=4, apply=False)
    assert _ngen(d) == 6
    assert rows[0]["status"] == "dry_run" and rows[0]["would_delete"] == 2 and rows[0]["bytes"] > 0


def test_apply_prunes_each_phase(tmp_path):
    a = _phase(tmp_path, "adaptive_production/epoch_000", 6)
    b = _phase(tmp_path, "adaptive_production/final/baseline", 3)
    prune_run_checkpoints(tmp_path, keep=4, apply=True)
    assert _ngen(a) == 4 and _ngen(b) == 3


def test_live_lock_skips_phase(tmp_path):
    d = _phase(tmp_path, "adaptive_production/final/baseline", 6)
    (d / ".gareus_run.lock").write_text(str(os.getpid()))
    rows = prune_run_checkpoints(tmp_path, keep=4, apply=True)
    assert rows[0]["status"] == "locked" and _ngen(d) == 6


def test_cli_defaults_to_dry_run(tmp_path, capsys):
    d = _phase(tmp_path, "adaptive_production/epoch_000", 6)
    assert main(["prune-checkpoints", str(tmp_path)]) == 0
    assert _ngen(d) == 6
    assert "dry run" in capsys.readouterr().out
    assert main(["prune-checkpoints", str(tmp_path), "--keep", "4", "--apply"]) == 0
    assert _ngen(d) == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `opencode run "run: pytest -q tests/test_retention_prune_checkpoints.py"`
Expected: FAIL (`ModuleNotFoundError: No module named 'gareus.retention'`).

- [ ] **Step 3: Implement `gareus/retention.py`**

```python
"""Output retention tools (spec docs/superpowers/specs/2026-09-28-output-retention-design.md).

    python -m gareus.retention prune-checkpoints <run_dir> [--keep 4] [--apply]

Every subcommand is a dry run unless --apply is given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

from .correctness.checkpoint_store import MANIFEST_NAME, _generation_manifest, prune_generations

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


def _would_delete(phase: Path, keep: int) -> tuple[int, int, int]:
    """(generations, deletable, bytes) without deleting, mirroring _prune_locked's rule."""
    gens_dir = phase / "checkpoints" / "generations"
    root = json.loads((phase / "checkpoints" / MANIFEST_NAME).read_text())
    kept, current = [], root.get("generation_id")
    while current and len(kept) < keep:
        kept.append(current)
        data = _generation_manifest(gens_dir, current)
        if data is None:
            break
        current = data.get("previous_generation_id")
    entries = [e for e in gens_dir.iterdir() if e.is_dir()] if gens_dir.is_dir() else []
    n_del, n_bytes = 0, 0
    for e in entries:
        if e.name in kept:
            continue
        data = _generation_manifest(gens_dir, e.name)
        step = None if data is None else data.get("absolute_step")
        if not isinstance(step, int) or step > int(root.get("absolute_step", -1)):
            continue
        n_del += 1
        n_bytes += sum(f.stat().st_size for f in e.rglob("*") if f.is_file())
    return len(entries), n_del, n_bytes


def prune_run_checkpoints(run_dir: Path, keep: int, apply: bool) -> list[dict]:
    rows = []
    for phase in find_checkpoint_phases(run_dir):
        n_gen, n_del, n_bytes = _would_delete(phase, keep)
        row = {"phase": str(phase), "generations": n_gen, "would_delete": n_del, "bytes": n_bytes}
        if _lock_is_live(phase):
            row["status"] = "locked"
        elif apply:
            report = prune_generations(phase, keep)
            row.update(status="pruned", would_delete=len(report["deleted"]), bytes=report["bytes_freed"])
        else:
            row["status"] = "dry_run"
        rows.append(row)
    return rows


def _print_rows(rows: Iterable[dict], apply: bool) -> None:
    total = 0
    for r in rows:
        total += r["bytes"] if r["status"] != "locked" else 0
        print(f"{r['status']:8s} {r['would_delete']:5d}/{r['generations']:<5d} "
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
        _print_rows(prune_run_checkpoints(args.run_dir, args.keep, args.apply), args.apply)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `opencode run "run: pytest -q tests/test_retention_prune_checkpoints.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/retention.py tests/test_retention_prune_checkpoints.py
git commit -m "feat: gareus.retention prune-checkpoints for existing runs"
```

---

### Task 4: Two-file JSONL ring writer

**Files:**
- Modify: `gareus/io.py` (new class after `BufferedJsonlWriter`, ~line 377; add to `__all__`)
- Test: `tests/test_rotating_jsonl_writer.py` (create)

**Interfaces:**
- Produces: `RotatingJsonlWriter(path: Path, max_bytes: int)` with `write_json(payload: dict) -> None`, `close() -> None`, attribute `previous_path` (`<stem>.1<suffix>`).

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json

from gareus.io import RotatingJsonlWriter


def _lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_writes_compact_lines_and_flushes(tmp_path):
    w = RotatingJsonlWriter(tmp_path / "live.jsonl", max_bytes=10_000)
    w.write_json({"b": 1, "a": 2})
    assert (tmp_path / "live.jsonl").read_text() == '{"a":2,"b":1}\n'
    w.close()


def test_rotates_into_single_previous_file(tmp_path):
    p = tmp_path / "live.jsonl"
    w = RotatingJsonlWriter(p, max_bytes=200)
    for i in range(40):
        w.write_json({"i": i, "pad": "x" * 20})
    w.close()
    prev = tmp_path / "live.1.jsonl"
    assert w.previous_path == prev
    assert p.stat().st_size <= 200 and prev.stat().st_size <= 200
    seq = [e["i"] for e in _lines(prev) + _lines(p)]
    assert seq == list(range(seq[0], 40))  # contiguous, newest last
    assert sorted(x.name for x in tmp_path.iterdir()) == ["live.1.jsonl", "live.jsonl"]


def test_appends_across_reopen(tmp_path):
    p = tmp_path / "live.jsonl"
    RotatingJsonlWriter(p, max_bytes=10_000).write_json({"i": 0})
    w = RotatingJsonlWriter(p, max_bytes=10_000)
    w.write_json({"i": 1})
    w.close()
    assert [e["i"] for e in _lines(p)] == [0, 1]


def test_single_line_larger_than_limit_is_still_written(tmp_path):
    p = tmp_path / "live.jsonl"
    w = RotatingJsonlWriter(p, max_bytes=10)
    w.write_json({"big": "y" * 100})
    w.write_json({"big": "z" * 100})
    w.close()
    assert len(_lines(p)) == 1 and len(_lines(tmp_path / "live.1.jsonl")) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "run: pytest -q tests/test_rotating_jsonl_writer.py"`
Expected: FAIL (`ImportError: cannot import name 'RotatingJsonlWriter'`).

- [ ] **Step 3: Implement** (append after `BufferedJsonlWriter` in `gareus/io.py`; add `"RotatingJsonlWriter"` to `__all__`; `os` and `json` are already imported there — add `import os` if not)

```python
class RotatingJsonlWriter:
    """Two-file JSONL ring: ``<path>`` (current) and ``<stem>.1<suffix>`` (previous).

    Each line is flushed immediately (a monitor tails the file). When the current
    file would pass ``max_bytes`` it replaces the previous one and a new current
    file starts, so the pair never holds more than ~2 x max_bytes.
    """

    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = Path(path)
        self.previous_path = self.path.with_name(self.path.stem + ".1" + self.path.suffix)
        self.max_bytes = max(1, int(max_bytes))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        self.size = self.path.stat().st_size

    def write_json(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        n = len(line.encode("utf-8"))
        if self.size > 0 and self.size + n > self.max_bytes:
            self._rotate()
        self.handle.write(line)
        self.handle.flush()
        self.size += n

    def _rotate(self) -> None:
        self.handle.close()
        os.replace(self.path, self.previous_path)
        self.handle = self.path.open("a", encoding="utf-8")
        self.size = 0

    def close(self) -> None:
        try:
            self.handle.close()
        except Exception:
            pass
```

- [ ] **Step 4: Run to verify pass**

Run: `opencode run "run: pytest -q tests/test_rotating_jsonl_writer.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/io.py tests/test_rotating_jsonl_writer.py
git commit -m "feat: RotatingJsonlWriter two-file JSONL ring"
```

---

### Task 5: `DistanceLogger` writes a per-phase `live_distances.jsonl` ring

**Files:**
- Modify: `gareus/cli.py` (new argument next to `--distance-output-mode`, ~line 711)
- Modify: `gareus/logger.py` (`DistanceLogger.__init__` ~line 83, before the `if no_file_persistence: return`; `log()` event emission ~lines 1668-1693; `close()` ~line 152)
- Test: `tests/test_live_distances_ring.py` (create)

**Interfaces:**
- Consumes: `RotatingJsonlWriter` (Task 4).
- Produces: `args.live_distances_max_mb: int` (default 256; 0 = legacy); ring file `<phase out_dir>/live_distances.jsonl` whose lines are `{"event": "distances", "phase", "step", "total_steps", "wall_time_s", "distances": [slim rows], "dashboard"?}`; slim row keys ⊆ `LIVE_ROW_KEYS`; `progress.jsonl` receives `{"event": "distances_summary", ...scalar summary fields...}` instead of the full event. Constants `LIVE_ROW_KEYS`, `LIVE_DASHBOARD_EVERY = 20` in `gareus/logger.py`.

- [ ] **Step 1: Write the failing tests**

```python
"""Per-phase live_distances.jsonl ring (spec R2)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from gareus.cli import parse_args
from gareus.logger import LIVE_DASHBOARD_EVERY, LIVE_ROW_KEYS, DistanceLogger


class FakeSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(dict(event))


def _args(**kw):
    base = dict(tui_mode="none", distance_output_mode="none", live_distances_max_mb=256)
    base.update(kw)
    return SimpleNamespace(**base)


def _rows(n=3):
    return [{"replica": i, "window": i, "center_A": 0.5, "k_kcal_mol_A2": 100.0, "cv_A": 0.4,
             "primary_cv_value": 0.4, "secondary_cv": 1.5, "gamd_boost_total_kcal_mol": 2.0,
             "potential_kj_mol": -5e5} for i in range(n)]


def _dash():
    return {"n_windows": 3, "exchange_stats": {"pairs": {"0-1": [1, 2]}},
            "secondary_cv": {"k": 1}, "secondary_cv_centers": [0.0, 1.0, 2.0]}


def test_flag_default_and_zero():
    assert parse_args(["--seq", "AA", "--out", "u"]).live_distances_max_mb == 256
    assert parse_args(["--seq", "AA", "--out", "u", "--live-distances-max-mb", "0"]).live_distances_max_mb == 0


def test_ring_gets_slim_rows_and_progress_gets_summary(tmp_path):
    sink = FakeSink()
    lg = DistanceLogger(tmp_path, _args(), progress=sink, no_file_persistence=True)
    lg.log(_rows(), "gareus_production", 250, 1000, dashboard_info=_dash())
    lg.close()
    ring = [json.loads(x) for x in (tmp_path / "live_distances.jsonl").read_text().splitlines()]
    assert len(ring) == 1 and ring[0]["event"] == "distances"
    assert all(set(r) <= set(LIVE_ROW_KEYS) for r in ring[0]["distances"])
    assert ring[0]["distances"][0]["secondary_cv"] == 1.5
    assert "exchange_stats" in ring[0]["dashboard"]
    (summary,) = sink.events
    assert summary["event"] == "distances_summary"
    assert "distances" not in summary and "dashboard" not in summary
    assert summary["step"] == 250 and "cv_mean_A" in summary


def test_dashboard_only_every_nth_event(tmp_path):
    lg = DistanceLogger(tmp_path, _args(), progress=FakeSink(), no_file_persistence=True)
    for k in range(LIVE_DASHBOARD_EVERY + 1):
        lg.log(_rows(), "gareus_production", 250 * (k + 1), None, dashboard_info=_dash())
    lg.close()
    ring = [json.loads(x) for x in (tmp_path / "live_distances.jsonl").read_text().splitlines()]
    with_dash = [i for i, e in enumerate(ring) if "dashboard" in e]
    assert with_dash == [0, LIVE_DASHBOARD_EVERY]


def test_zero_keeps_legacy_full_event(tmp_path):
    sink = FakeSink()
    lg = DistanceLogger(tmp_path, _args(live_distances_max_mb=0), progress=sink, no_file_persistence=True)
    lg.log(_rows(), "gareus_production", 250, 1000, dashboard_info=_dash())
    lg.close()
    assert not (tmp_path / "live_distances.jsonl").exists()
    assert sink.events[0]["event"] == "distances" and len(sink.events[0]["distances"]) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "run: pytest -q tests/test_live_distances_ring.py"`
Expected: FAIL (`ImportError: cannot import name 'LIVE_DASHBOARD_EVERY'`).

- [ ] **Step 3: Implement**

`gareus/cli.py`, after `--distance-output-interval`:

```python
    p.add_argument("--live-distances-max-mb", type=int, default=256,
                   help="Per-phase live_distances.jsonl ring for the monitor: per-replica CV/bias/boost "
                        "rows every sample, at most 2 x this many MB on disk; progress.jsonl then gets "
                        "only a small summary event. 0 = legacy: the full distances event goes into "
                        "progress.jsonl. Spec 2026-09-28-output-retention-design.")
```

`gareus/logger.py`, module level near the other constants (import `RotatingJsonlWriter` from `.io` next to the existing `BufferedJsonlWriter`/`BufferedCsvDictWriter` imports):

```python
# Fields of a distances row the monitor reads (gareus_monitor.parse_distances_samples).
LIVE_ROW_KEYS = ("replica", "window", "window_index", "state_id", "primary_cv_value", "cv_A",
                 "secondary_cv", "umbrella_bias_kcal_mol", "gamd_boost_total_kcal_mol", "gamd_lambda")
# The dashboard block (cumulative exchange stats, ~0.5 MB at 236 states) goes into
# every Nth ring line only; the monitor reads the newest one.
LIVE_DASHBOARD_EVERY = 20
```

In `DistanceLogger.__init__`, immediately before `if no_file_persistence:` add:

```python
        self._live_ring = None
        self._live_count = 0
        _live_mb = int(getattr(args, "live_distances_max_mb", 256) or 0)
        if _live_mb > 0:
            self._live_ring = RotatingJsonlWriter(self.out_dir / "live_distances.jsonl",
                                                  max_bytes=_live_mb * 1024 * 1024)
```

In `log()`, replace

```python
        if self.progress is not None and not self.no_gui:
            self.progress.emit(event)
```

with

```python
        if self._live_ring is not None:
            live = {k: event[k] for k in ("event", "phase", "step", "total_steps") if k in event}
            live["wall_time_s"] = time.time()
            live["distances"] = [{k: r[k] for k in LIVE_ROW_KEYS if k in r} for r in clean_rows]
            if "dashboard" in event and self._live_count % LIVE_DASHBOARD_EVERY == 0:
                live["dashboard"] = event["dashboard"]
            self._live_ring.write_json(live)
            self._live_count += 1
        if self.progress is not None and not self.no_gui:
            if self._live_ring is not None:
                self.progress.emit({**{k: v for k, v in event.items() if k not in ("distances", "dashboard")},
                                    "event": "distances_summary"})
            else:
                self.progress.emit(event)
```

In `close()`, add after the `for h in (...)` loop:

```python
        if getattr(self, "_live_ring", None) is not None:
            self._live_ring.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `opencode run "run: pytest -q tests/test_live_distances_ring.py tests/test_dashboard_panels.py tests/test_dashboard_grid2d.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/cli.py gareus/logger.py tests/test_live_distances_ring.py
git commit -m "feat: per-phase live_distances.jsonl ring; progress.jsonl gets a summary event"
```

---

### Task 6: Monitor reads the ring, falls back to `progress.jsonl`

**Files:**
- Modify: `gareus_monitor.py` (new helpers after `read_boost_samples` ~line 1333; `live_boost_samples` ~line 2311; `live_snapshot` ~line 2363)
- Test: `tests/test_monitor_live_ring.py` (create)

**Interfaces:**
- Consumes: ring layout from Task 5 (`live_distances.jsonl`, `live_distances.1.jsonl`; `event == "distances"` lines, optional `dashboard`).
- Produces: `live_ring_files(run_dir: Path) -> list[Path]` (oldest first, from the most recently written phase ring; `[]` if none); `read_ring_entries(paths: list[Path], max_bytes: int) -> list[dict]`; `read_ring_boost_samples(paths, target_per_window=..., max_bytes=...) -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

```python
"""gareus_monitor reads the per-phase live_distances ring (spec R2)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import gareus_monitor as gm


def _event(step, windows=2, dashboard=False):
    e = {"event": "distances", "step": step, "wall_time_s": 1.0 + step,
         "distances": [{"replica": w, "window": w, "primary_cv_value": 0.1 * w, "secondary_cv": 1.0,
                        "umbrella_bias_kcal_mol": 0.2, "gamd_boost_total_kcal_mol": 1.5}
                       for w in range(windows)]}
    if dashboard:
        e["dashboard"] = {"exchange_stats": {"attempts": step}}
    return e


def _write(p: Path, events):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(e) + "\n" for e in events))


def test_no_ring_returns_empty(tmp_path):
    assert gm.live_ring_files(tmp_path) == []


def test_picks_most_recent_phase_ring_oldest_first(tmp_path):
    old = tmp_path / "adaptive_production" / "final" / "baseline" / "live_distances.jsonl"
    new = tmp_path / "adaptive_production" / "final_extension_001" / "live_distances.jsonl"
    _write(old, [_event(1)])
    _write(new.with_name("live_distances.1.jsonl"), [_event(2)])
    _write(new, [_event(3)])
    os.utime(old, (1, 1))
    assert gm.live_ring_files(tmp_path) == [new.with_name("live_distances.1.jsonl"), new]


def test_boost_samples_span_rotation(tmp_path):
    ring = tmp_path / "adaptive_production" / "final" / "baseline" / "live_distances.jsonl"
    _write(ring.with_name("live_distances.1.jsonl"), [_event(s) for s in range(1, 6)])
    _write(ring, [_event(6)])
    samples = gm.read_ring_boost_samples(gm.live_ring_files(tmp_path), target_per_window=4)
    assert len(samples) >= 8  # 2 windows x >= 4 events, needing the rotated file


def test_ring_entries_newest_last(tmp_path):
    ring = tmp_path / "live_distances.jsonl"
    _write(ring.with_name("live_distances.1.jsonl"), [_event(1)])
    _write(ring, [_event(2, dashboard=True)])
    entries = gm.read_ring_entries(gm.live_ring_files(tmp_path), max_bytes=1 << 20)
    assert [e["step"] for e in entries] == [1, 2]
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "run: pytest -q tests/test_monitor_live_ring.py"`
Expected: FAIL (`AttributeError: module 'gareus_monitor' has no attribute 'live_ring_files'`).

- [ ] **Step 3: Implement**

After `read_boost_samples` in `gareus_monitor.py`:

```python
LIVE_RING_NAME = "live_distances.jsonl"
LIVE_RING_PREV = "live_distances.1.jsonl"


def live_ring_files(run_dir: Path) -> list:
    """The most recently written phase's live_distances ring, oldest file first.

    DistanceLogger writes one ring per phase directory (spec
    2026-09-28-output-retention-design, R2); a campaign root has none of its own.
    Phases sit at most three levels below the run root.
    """
    run_dir = Path(run_dir)
    candidates = []
    for pattern in (LIVE_RING_NAME, f"*/{LIVE_RING_NAME}", f"*/*/{LIVE_RING_NAME}",
                    f"*/*/*/{LIVE_RING_NAME}"):
        candidates.extend(run_dir.glob(pattern))
    if not candidates:
        return []
    try:
        current = max(candidates, key=lambda p: p.stat().st_mtime)
    except OSError:
        return []
    prev = current.with_name(LIVE_RING_PREV)
    return [prev, current] if prev.exists() else [current]


def read_ring_entries(paths: list, max_bytes: int = BOOST_READ_MAX_BYTES) -> list:
    """Tail entries across the ring files (newest last), reading at most max_bytes."""
    entries: list = []
    budget = int(max_bytes)
    for p in reversed(list(paths)):
        if budget <= 0:
            break
        try:
            size = p.stat().st_size
        except OSError:
            continue
        take = min(size, budget)
        if take <= 0:
            continue
        entries = tail_jsonl(p, 1_000_000, chunk_bytes=take) + entries
        budget -= take
    return entries


def read_ring_boost_samples(paths: list,
                            target_per_window: int = BOOST_TARGET_PER_WINDOW,
                            max_bytes: int = BOOST_READ_MAX_BYTES) -> list:
    """Like read_boost_samples, but over the ring: newest file first, older only if short."""
    entries: list = []
    budget = int(max_bytes)
    samples: list = []
    for p in reversed(list(paths)):
        if budget <= 0:
            break
        try:
            size = p.stat().st_size
        except OSError:
            continue
        take = min(size, budget)
        entries = tail_jsonl(p, 1_000_000, chunk_bytes=take) + entries
        budget -= take
        samples = parse_distances_samples(entries)
        if _min_boost_samples_per_window(samples) >= target_per_window:
            break
    return samples
```

In `live_boost_samples`, right after the `_progress_has_uncommitted_live_ns()` early return, insert:

```python
        ring = live_ring_files(self.run_dir)
        if ring:
            try:
                st = ring[-1].stat()
                key = ("ring", str(ring[-1]), st.st_mtime, st.st_size, target_per_window)
            except OSError:
                key = None
            if key is not None and self._boost_samples_cache is not None \
                    and self._boost_samples_cache[0] == key:
                return self._boost_samples_cache[1]
            samples = read_ring_boost_samples(ring, target_per_window=target_per_window,
                                              max_bytes=max_bytes)
            if key is not None:
                self._boost_samples_cache = (key, samples)
            return samples
```

In `live_snapshot`, after the line `snap["_dist_samples"] = parse_distances_samples(entries)[-6000:]` insert:

```python
        ring = live_ring_files(self.run_dir)
        if ring:
            ring_entries = read_ring_entries(ring, max_bytes=chunk_bytes)
            snap["_dist_samples"] = parse_distances_samples(ring_entries)[-6000:]
            if dash is None:
                for e in reversed(ring_entries):
                    if e.get("dashboard"):
                        dash, dash_wall = e["dashboard"], e.get("wall_time_s")
                        break
```

(Check that `snap["_dashboard"] = dash` is assigned after this insertion; if it is assigned before, move the insertion above it so the ring dashboard is used.)

- [ ] **Step 4: Run to verify pass**

Run: `opencode run "run: pytest -q tests/test_monitor_live_ring.py"`
Then smoke the monitor against the local chignolin_9 mirror (legacy layout, no ring yet): `opencode run "run: timeout 60 python gareus_monitor.py --help"` and, if the monitor has a one-shot/print mode, run it once on `RUNS/chignolin_9` and confirm it still renders boost samples.
Expected: PASS; monitor unchanged on the legacy run.

- [ ] **Step 5: Commit**

```bash
git add gareus_monitor.py tests/test_monitor_live_ring.py
git commit -m "feat: monitor reads the per-phase live_distances ring with legacy fallback"
```

---

### Task 7: `split-progress` for existing `progress.jsonl`

**Files:**
- Modify: `gareus/retention.py` (new functions + subcommand)
- Test: `tests/test_retention_split_progress.py` (create)

**Interfaces:**
- Produces: `split_progress_jsonl(path: Path, apply: bool) -> dict` returning `{"lines", "distances_lines", "bytes_before", "bytes_after", "status"}` with `status` in `{"dry_run", "rewritten", "changed_during_rewrite"}`; subcommand `split-progress <progress.jsonl> [--apply]`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json

import gareus.retention as retention


def _write(p, events):
    p.write_text("".join(json.dumps(e) + "\n" for e in events))


def _sample():
    return [
        {"event": "progress", "step": 1, "aggregate_sim_time_ns": 0.1},
        {"event": "distances", "step": 250, "cv_mean_A": 0.4, "distances": [{"replica": 0}] * 50,
         "dashboard": {"exchange_stats": {"pairs": {"a": 1}}}},
        {"event": "run_complete"},
    ]


def test_dry_run_reports_and_keeps_file(tmp_path):
    p = tmp_path / "progress.jsonl"
    _write(p, _sample())
    before = p.read_bytes()
    r = retention.split_progress_jsonl(p, apply=False)
    assert r["status"] == "dry_run" and r["distances_lines"] == 1 and r["bytes_after"] < r["bytes_before"]
    assert p.read_bytes() == before


def test_apply_rewrites_distances_as_summary_and_keeps_others_byte_for_byte(tmp_path):
    p = tmp_path / "progress.jsonl"
    _write(p, _sample())
    lines_before = p.read_text().splitlines()
    retention.split_progress_jsonl(p, apply=True)
    lines_after = p.read_text().splitlines()
    assert lines_after[0] == lines_before[0] and lines_after[2] == lines_before[2]
    e = json.loads(lines_after[1])
    assert e["event"] == "distances_summary" and e["cv_mean_A"] == 0.4
    assert "distances" not in e and "dashboard" not in e


def test_aborts_if_file_grows_during_rewrite(tmp_path, monkeypatch):
    p = tmp_path / "progress.jsonl"
    _write(p, _sample())
    real = retention._rewrite_lines

    def growing(src, dst):
        out = real(src, dst)
        with p.open("a") as fh:
            fh.write(json.dumps({"event": "progress"}) + "\n")
        return out
    monkeypatch.setattr(retention, "_rewrite_lines", growing)
    r = retention.split_progress_jsonl(p, apply=True)
    assert r["status"] == "changed_during_rewrite"
    assert json.loads(p.read_text().splitlines()[1])["event"] == "distances"
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "run: pytest -q tests/test_retention_split_progress.py"`
Expected: FAIL (`AttributeError: module 'gareus.retention' has no attribute 'split_progress_jsonl'`).

- [ ] **Step 3: Implement** (add to `gareus/retention.py`, above `main`)

```python
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
```

Register the subcommand in `main` (before `args = parser.parse_args(argv)`):

```python
    sp = sub.add_parser("split-progress", help="drop per-replica distances dumps from a progress.jsonl")
    sp.add_argument("path", type=Path)
    sp.add_argument("--apply", action="store_true")
```

and handle it (before `return 2`):

```python
    if args.command == "split-progress":
        r = split_progress_jsonl(args.path, args.apply)
        print(f"{r['status']}: {r['distances_lines']}/{r['lines']} distances lines, "
              f"{r['bytes_before'] / 1e9:.2f} GB -> {r['bytes_after'] / 1e9:.2f} GB")
        return 0 if r["status"] in ("dry_run", "rewritten") else 1
```

Add `split-progress <progress.jsonl> [--apply]` to the module docstring's usage lines.

- [ ] **Step 4: Run to verify pass**

Run: `opencode run "run: pytest -q tests/test_retention_split_progress.py tests/test_retention_prune_checkpoints.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/retention.py tests/test_retention_split_progress.py
git commit -m "feat: gareus.retention split-progress for existing progress.jsonl"
```

---

### Task 8: Hard-linked filtered seed banks (R3a)

**Files:**
- Modify: `gareus/adaptive_production.py` (`filter_seed_bank_for_state_ids`, the `shutil.copy2(src, dst)` inside `_copy_row`; new helper `_link_or_copy` above the function)
- Test: `tests/test_seed_bank_hardlinks.py` (create)

**Interfaces:**
- Produces: `_link_or_copy(src: Path, dst: Path) -> str` returning `"link"` or `"copy"`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import csv
import os

import gareus.adaptive_production as ap


def _bank(tmp_path):
    bank = tmp_path / "seed_bank_final"
    (bank / "pdbs").mkdir(parents=True)
    rows = []
    for sid in (0, 1):
        pdb = bank / "pdbs" / f"seed_{sid}.pdb"
        pdb.write_text(f"ATOM {sid}\nEND\n")
        rows.append({"seed_name": f"seed_{sid}", "survivor_pdb_path": str(pdb),
                     "seed_source_state_id": sid})
    with (bank / "final_survivor_seeds.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return bank


def test_filtered_bank_uses_hard_links(tmp_path):
    bank = _bank(tmp_path)
    out = tmp_path / "seg" / "filtered_seed_bank"
    ap.filter_seed_bank_for_state_ids(bank, [0, 1], out)
    for sid in (0, 1):
        src, dst = bank / "pdbs" / f"seed_{sid}.pdb", out / "pdbs" / f"seed_{sid}.pdb"
        assert dst.read_text() == src.read_text()
        assert os.stat(dst).st_ino == os.stat(src).st_ino


def test_falls_back_to_copy_when_link_fails(tmp_path, monkeypatch):
    bank = _bank(tmp_path)

    def no_link(*a, **k):
        raise OSError("cross-device")
    monkeypatch.setattr(ap.os, "link", no_link)
    out = tmp_path / "seg" / "filtered_seed_bank"
    ap.filter_seed_bank_for_state_ids(bank, [0], out)
    dst = out / "pdbs" / "seed_0.pdb"
    assert dst.read_text() == "ATOM 0\nEND\n"
    assert os.stat(dst).st_ino != os.stat(bank / "pdbs" / "seed_0.pdb").st_ino
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "run: pytest -q tests/test_seed_bank_hardlinks.py"`
Expected: FAIL (inode assertion; and `ap.os` missing if `os` is not imported in the module).

- [ ] **Step 3: Implement**

Add `import os` to the imports of `gareus/adaptive_production.py` if absent. Above `filter_seed_bank_for_state_ids`:

```python
def _link_or_copy(src: Path, dst: Path) -> str:
    """Hard-link a seed PDB into a filtered bank; copy when linking is impossible.

    Seed PDBs are never modified after they are written, so a link is
    indistinguishable to every reader and saves the whole copy (chignolin_9:
    7.1 GB of filtered_seed_bank). Spec 2026-09-28-output-retention-design R3a.
    """
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"
```

In `_copy_row`, replace `shutil.copy2(src, dst)` with `_link_or_copy(src, dst)`.

- [ ] **Step 4: Run to verify pass**

Run: `opencode run "run: pytest -q tests/test_seed_bank_hardlinks.py tests/test_resume_seed_bank_discovery.py tests/test_seed_bank_rescore.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/adaptive_production.py tests/test_seed_bank_hardlinks.py
git commit -m "feat: hard-link filtered seed banks instead of copying PDBs"
```

---

### Task 9: Opt-in `--prune-us-starting-structures` (R3b)

**Files:**
- Modify: `gareus/cli.py` (new argument next to `--checkpoint-keep-generations`)
- Modify: `gareus/production.py` (new helper near `sync_scratch_to_main`; call after the regular `save_production_checkpoint` at ~line 8541)
- Test: `tests/test_prune_us_starting_structures.py` (create)

**Interfaces:**
- Produces: `args.prune_us_starting_structures: bool`; `prune_us_starting_pdbs(out_dir: Path) -> int` (number of `.pdb` files removed; reports `*.csv`/`*.json` untouched).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "run: pytest -q tests/test_prune_us_starting_structures.py"`
Expected: FAIL (`ImportError: cannot import name 'prune_us_starting_pdbs'`).

- [ ] **Step 3: Implement**

`gareus/cli.py`:

```python
    p.add_argument("--prune-us-starting-structures", action="store_true", default=False,
                   help="After a phase's first production checkpoint, delete the pulled window PDBs in "
                        "<phase>/us_starting_structures/ (reports are kept). A resumed phase never "
                        "pulls again. Off by default: the PDBs help debug a bad pull.")
```

`gareus/production.py`, after `sync_scratch_to_main`:

```python
def prune_us_starting_pdbs(out_dir: Path) -> int:
    """Delete us_starting_structures/*.pdb (the pull's window structures), keep its reports.

    Only the pull that wrote them reads them; a resumed phase skips pulling.
    Spec 2026-09-28-output-retention-design R3b (opt-in).
    """
    d = Path(out_dir) / "us_starting_structures"
    if not d.is_dir():
        return 0
    n = 0
    for pdb in d.glob("*.pdb"):
        try:
            pdb.unlink()
            n += 1
        except OSError:
            pass
    return n
```

After the regular checkpoint's `save_production_checkpoint(...)` block (inside `with _phase_timers.phase("checkpoint_save"):` is fine, or right after it), add:

```python
                if bool(getattr(args, "prune_us_starting_structures", False)):
                    _n_pruned = prune_us_starting_pdbs(out_dir)
                    if _n_pruned:
                        print(f"[retention] removed {_n_pruned} pulled window PDBs from "
                              f"{out_dir / 'us_starting_structures'}", flush=True)
```

- [ ] **Step 4: Run to verify pass**

Run: `opencode run "run: pytest -q tests/test_prune_us_starting_structures.py tests/test_checkpoint_keep_generations_wiring.py"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add gareus/cli.py gareus/production.py tests/test_prune_us_starting_structures.py
git commit -m "feat: opt-in --prune-us-starting-structures after the first checkpoint"
```

---

### Task 10: Cold archive and restore for finished campaigns (R4)

**Files:**
- Modify: `gareus/retention.py` (archive/restore functions + subcommands)
- Test: `tests/test_retention_archive.py` (create)

**Interfaces:**
- Produces: `ARCHIVE_DIR_NAMES = ("final_pdbs", "final_window_states", "us_starting_structures", "filtered_seed_bank")` plus any directory named `seed_bank_*`; `archive_targets(run_dir) -> list[Path]`; `archive_run(run_dir, apply: bool, force: bool = False) -> list[dict]`; `restore_run(run_dir, apply: bool) -> list[dict]`; archive file `<dir>.tar.zst` and index `<dir>.tar.zst.index.json` (`{"members": {relpath: {"size", "sha256"}}}`); subcommands `archive <run_dir> [--apply] [--force]`, `restore <run_dir> [--apply]`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

import json

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
```

- [ ] **Step 2: Run to verify failure**

Run: `opencode run "run: pytest -q tests/test_retention_archive.py"`
Expected: FAIL (`AttributeError: ... archive_run`).

- [ ] **Step 3: Implement** (add to `gareus/retention.py`; `import hashlib, shutil, tarfile` at the top)

```python
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
    out = []
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
    members = {}
    zstandard = _zstd()
    cctx = zstandard.ZstdCompressor(level=19, write_checksum=True)
    with archive.open("wb") as fh, cctx.stream_writer(fh) as zw, tarfile.open(fileobj=zw, mode="w|") as tar:
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = f.relative_to(src).as_posix()
            members[rel] = {"size": f.stat().st_size, "sha256": _sha256(f)}
            tar.add(str(f), arcname=rel, recursive=False)
    return {"members": members}


def _iter_archive(archive: Path):
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
    for name, data in _iter_archive(archive):
        seen[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
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
        index = _write_archive(d, archive)
        if not _verify_archive(archive, index):
            archive.unlink(missing_ok=True)
            rows.append({**row, "status": "verify_failed"})
            continue
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
        shutil.rmtree(d)
        rows.append({**row, "status": "archived", "archive_bytes": archive.stat().st_size})
    return rows


def restore_run(run_dir: Path, apply: bool) -> list[dict]:
    rows = []
    for archive in sorted(Path(run_dir).rglob("*" + ARCHIVE_SUFFIX)):
        index_path = archive.with_name(archive.name + ".index.json")
        dest = archive.with_name(archive.name[: -len(ARCHIVE_SUFFIX)])
        if not apply:
            rows.append({"archive": str(archive), "status": "dry_run"})
            continue
        index = json.loads(index_path.read_text())
        staging = dest.with_name("." + dest.name + ".restore")
        shutil.rmtree(staging, ignore_errors=True)
        for name, data in _iter_archive(archive):
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
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
```

Subcommands in `main`:

```python
    ar = sub.add_parser("archive", help="pack PDB sets and final_window_states of a finished campaign")
    ar.add_argument("run_dir", type=Path)
    ar.add_argument("--apply", action="store_true")
    ar.add_argument("--force", action="store_true", help="archive even if the campaign is not completed")
    rs = sub.add_parser("restore", help="unpack every archive made by 'archive'")
    rs.add_argument("run_dir", type=Path)
    rs.add_argument("--apply", action="store_true")
```

```python
    if args.command == "archive":
        for r in archive_run(args.run_dir, args.apply, args.force):
            print(f"{r['status']:13s} {r['bytes'] / 1e9:8.2f} GB  {r['dir']}")
        return 0
    if args.command == "restore":
        rows = restore_run(args.run_dir, args.apply)
        for r in rows:
            print(f"{r['status']:13s} {r['archive']}")
        return 0 if all(r["status"] != "verify_failed" for r in rows) else 1
```

Note on hard links: a `filtered_seed_bank` archived after Task 8 stores its PDBs as regular files (each archive is independent), so restore yields regular files, byte-identical.

- [ ] **Step 4: Run to verify pass**

Run: `opencode run "run: pytest -q tests/test_retention_archive.py tests/test_retention_split_progress.py tests/test_retention_prune_checkpoints.py"`
Expected: PASS (archive tests skip only if `zstandard` is missing locally; it must be present on aurum2's calc env).

- [ ] **Step 5: Commit**

```bash
git add gareus/retention.py tests/test_retention_archive.py
git commit -m "feat: gareus.retention archive/restore for finished campaigns"
```

---

### Task 11: Documentation and roll-out notes

**Files:**
- Modify: `CLAUDE.md` (new section after "Production phase timers")
- Modify: `docs/superpowers/specs/2026-09-28-output-retention-design.md` (status line)
- Modify: `docs/atlas-md/developer/topups-todo.md` (T7 status)

- [ ] **Step 1: Add to `CLAUDE.md`** after the "Production phase timers" section:

```markdown
## Output retention (`--checkpoint-keep-generations`, live_distances ring, `gareus.retention`)

- Spec `docs/superpowers/specs/2026-09-28-output-retention-design.md`; chignolin_9 was 688 GB, 92 % superseded checkpoint generations.
- `--checkpoint-keep-generations N` (opt-in, default 0 = keep all, recommended 4): `checkpoint_store._prune_locked` runs under the publication lock after `publish_generation` and after `copy_committed_generation` (so a `--scratchdir` main copy is pruned too). Order is the `previous_generation_id` chain from the root manifest, never mtimes; orphans newer than the root are kept. Only the root's generation is ever read; every phase's root manifest must stay (pool reconciliation, extension seeding).
- `DistanceLogger` writes per-replica rows to `<phase>/live_distances.jsonl` (two-file ring, `--live-distances-max-mb` 256; 0 = legacy) with the dashboard block every 20th line; `progress.jsonl` gets a scalar `distances_summary` event. `gareus_monitor.py` reads the newest phase ring and falls back to `progress.jsonl`.
- `filtered_seed_bank/` PDBs are hard links to the source bank (`_link_or_copy`). `--prune-us-starting-structures` (opt-in) deletes pulled window PDBs after a phase's first checkpoint; reports stay.
- Tools (dry run unless `--apply`): `python -m gareus.retention prune-checkpoints <run> [--keep 4]`, `split-progress <progress.jsonl>`, `archive <run>` / `restore <run>` (completed campaigns only; `.tar.zst` + sha256 index, byte-identical restore).
- Tests: `tests/test_checkpoint_generation_pruning.py`, `test_checkpoint_keep_generations_wiring.py`, `test_retention_*.py`, `test_rotating_jsonl_writer.py`, `test_live_distances_ring.py`, `test_monitor_live_ring.py`, `test_seed_bank_hardlinks.py`, `test_prune_us_starting_structures.py`.
```

- [ ] **Step 2: Update statuses.** In the spec, change `Status: reviewed 2026-09-28 (decisions in §7).` to `Status: implemented 2026-09-28 (plan docs/superpowers/plans/2026-09-28-output-retention.md); roll-out on chignolin_9 pending.` In `topups-todo.md`, change the T7 heading suffix `(spec written)` to `(implemented; roll-out pending)`.

- [ ] **Step 3: Run the full targeted set once**

Run: `opencode run "run: pytest -q tests/test_checkpoint_generation_pruning.py tests/test_checkpoint_keep_generations_wiring.py tests/test_retention_prune_checkpoints.py tests/test_rotating_jsonl_writer.py tests/test_live_distances_ring.py tests/test_monitor_live_ring.py tests/test_retention_split_progress.py tests/test_seed_bank_hardlinks.py tests/test_prune_us_starting_structures.py tests/test_retention_archive.py tests/test_phase_timers.py tests/test_adaptive_new_state_guards.py tests/test_resume_seed_bank_discovery.py tests/test_dashboard_panels.py"`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/atlas-md/developer/topups-todo.md
git add -f docs/superpowers/specs/2026-09-28-output-retention-design.md docs/superpowers/plans/2026-09-28-output-retention.md
git commit -m "docs: output retention (checkpoint pruning, live distances ring, retention tools)"
```

**Roll-out (not part of this plan's code; needs the user's go-ahead at the time):** merge, deploy to aurum2 with the usual rsync to both trees (never `--delete`) while the chignolin_9 chain is held between jobs; add `checkpoint_keep_generations: 4` to `chignolin_9.yaml`; run `python -m gareus.retention prune-checkpoints ~/gareus/chignolin/chignolin_9 --keep 4` as a dry run, then `--apply` with the chain held; record bytes before/after; prune the local mirror with the same tool.
