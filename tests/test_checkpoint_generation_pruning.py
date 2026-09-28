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


def test_dry_run_reports_same_ids_but_deletes_nothing(tmp_path):
    ids = [_publish(tmp_path, s) for s in (100, 200, 300)]
    before = _gens(tmp_path)
    dry_report = prune_generations(tmp_path, keep=1, dry_run=True)
    assert dry_report["dry_run"] is True
    assert set(dry_report["deleted"]) == {ids[0], ids[1]}
    # Nothing was actually removed.
    assert _gens(tmp_path) == before

    real_report = prune_generations(tmp_path, keep=1)
    assert real_report["dry_run"] is False
    assert set(real_report["deleted"]) == set(dry_report["deleted"])
    assert real_report["bytes_freed"] == dry_report["bytes_freed"]
    assert _gens(tmp_path) == {ids[2]}


# ---- final-review fix wave (I2, I4, M2) -------------------------------------

import shutil  # noqa: E402

import gareus.correctness.checkpoint_store as cs  # noqa: E402


def _real_gens(out: Path) -> set:
    return {p.name for p in (out / "checkpoints" / "generations").iterdir()
            if p.is_dir() and not p.name.startswith(".")}


def _raise_rmtree(*a, **k):
    raise PermissionError("simulated rmtree failure")


def test_publish_prune_failure_warns_and_returns(tmp_path, monkeypatch, capsys):
    for s in (100, 200, 300):
        _publish(tmp_path, s)
    monkeypatch.setattr(cs.shutil, "rmtree", _raise_rmtree)
    m = publish_generation(tmp_path, [b"a" * 64, b"b" * 64], _fields(400), keep_generations=1)
    monkeypatch.undo()
    assert m["absolute_step"] == 400
    assert read_validated_generation(tmp_path).manifest["generation_id"] == m["generation_id"]
    assert "WARNING" in capsys.readouterr().out


def test_copy_prune_failure_warns_and_returns(tmp_path, monkeypatch, capsys):
    src, dst = tmp_path / "scratch", tmp_path / "main"
    for step in (100, 200):
        _publish(src, step)
        copy_committed_generation(src, dst)
    _publish(src, 300)
    monkeypatch.setattr(cs.shutil, "rmtree", _raise_rmtree)
    m = copy_committed_generation(src, dst, keep_generations=1)
    monkeypatch.undo()
    assert m["absolute_step"] == 300
    assert read_validated_generation(dst).manifest["absolute_step"] == 300
    assert "WARNING" in capsys.readouterr().out


def test_broken_chain_middle_fills_from_step_sorted_fallbacks(tmp_path):
    ids = [_publish(tmp_path, s) for s in (100, 200, 300, 400, 500)]
    # The chain from the root (500) breaks at 400: its manifest becomes unreadable.
    (tmp_path / "checkpoints" / "generations" / ids[3] / "manifest.json").write_text("{not json")
    prune_generations(tmp_path, keep=3)
    # Only 500 is a readable chain member; the newest readable fallbacks by step
    # (300, 200) fill the remaining slots. 100 goes; unreadable 400 is never deleted.
    remaining = _real_gens(tmp_path)
    assert {ids[4], ids[3], ids[2], ids[1]} == remaining


def test_chain_to_missing_generation_keeps_n_readable(tmp_path):
    # Main copy missing an unsynced scratch generation: the chain reaches a
    # previous_generation_id that does not exist on disk.
    ids = [_publish(tmp_path, s) for s in (100, 200, 300, 400)]
    shutil.rmtree(tmp_path / "checkpoints" / "generations" / ids[2])
    prune_generations(tmp_path, keep=2)
    assert _real_gens(tmp_path) == {ids[3], ids[1]}


def test_equal_step_non_chain_generation_is_kept(tmp_path):
    ids = [_publish(tmp_path, s) for s in (100, 200)]
    root_bytes = (tmp_path / "checkpoints" / MANIFEST_NAME).read_bytes()
    orphan = _publish(tmp_path, 200)  # same absolute_step as the root, not on its chain
    (tmp_path / "checkpoints" / MANIFEST_NAME).write_bytes(root_bytes)
    prune_generations(tmp_path, keep=1)
    assert _real_gens(tmp_path) == {ids[1], orphan}


def test_delete_renames_to_deleting_first(tmp_path, monkeypatch):
    ids = [_publish(tmp_path, s) for s in (100, 200)]
    gens = tmp_path / "checkpoints" / "generations"
    seen = []

    def crash_rmtree(path, *a, **k):
        seen.append(Path(path).name)
        raise OSError("crash mid-rmtree")

    monkeypatch.setattr(cs.shutil, "rmtree", crash_rmtree)
    with pytest.raises(OSError):
        prune_generations(tmp_path, keep=1)
    monkeypatch.undo()
    assert seen and seen[0].startswith(".deleting-" + ids[0])
    assert ids[0] not in {p.name for p in gens.iterdir()}
    assert len([p for p in gens.iterdir() if p.name.startswith(".deleting-")]) == 1
    # The next prune sweeps the leftover; it is never counted as a generation.
    report = prune_generations(tmp_path, keep=1)
    assert not [p for p in gens.iterdir() if p.name.startswith(".deleting-")]
    assert _real_gens(tmp_path) == {ids[1]}
    assert not any(n.startswith(".deleting-") for n in report["kept"] + report["deleted"])


def test_deleting_leftover_not_counted_by_retention(tmp_path):
    from gareus.retention import _generation_count
    _publish(tmp_path, 100)
    (tmp_path / "checkpoints" / "generations" / ".deleting-abc").mkdir()
    assert _generation_count(tmp_path) == 1


def test_undeletable_deleting_leftover_does_not_block_prune(tmp_path, monkeypatch):
    ids = [_publish(tmp_path, s) for s in (100, 200, 300)]
    gens = tmp_path / "checkpoints" / "generations"
    (gens / ".deleting-stuck").mkdir()
    real_rmtree = shutil.rmtree

    def picky(path, *a, **k):
        if Path(path).name == ".deleting-stuck":
            raise PermissionError("stuck")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(cs.shutil, "rmtree", picky)
    report = prune_generations(tmp_path, keep=1)
    assert _real_gens(tmp_path) == {ids[2]}
    assert any(name == ".deleting-stuck" for name, _ in report["skipped"])
