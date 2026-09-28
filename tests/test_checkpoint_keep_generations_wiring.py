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
