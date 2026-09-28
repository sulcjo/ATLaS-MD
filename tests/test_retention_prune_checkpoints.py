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
