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
