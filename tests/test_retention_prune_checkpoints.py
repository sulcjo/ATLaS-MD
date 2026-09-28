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


def test_error_isolates_phase_and_apply_continues(tmp_path, monkeypatch):
    import gareus.retention as retention

    a = _phase(tmp_path, "adaptive_production/epoch_000", 6)
    b = _phase(tmp_path, "adaptive_production/final/baseline", 6)
    real_prune = retention.prune_generations

    def _flaky(phase, keep, dry_run=False):
        if phase == a:
            raise RuntimeError("boom")
        return real_prune(phase, keep, dry_run=dry_run)

    monkeypatch.setattr(retention, "prune_generations", _flaky)

    rows = retention.prune_run_checkpoints(tmp_path, keep=4, apply=True)
    by_phase = {r["phase"]: r for r in rows}
    assert by_phase[str(a)]["status"] == "error"
    assert "boom" in by_phase[str(a)]["error"]
    assert by_phase[str(b)]["status"] == "pruned"
    assert _ngen(a) == 6
    assert _ngen(b) == 4

    assert retention.main(["prune-checkpoints", str(tmp_path), "--keep", "4", "--apply"]) == 1


def _subcommand_help(capsys, name):
    import pytest
    with pytest.raises(SystemExit):
        main([name, "--help"])
    import re
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out).split())


def test_help_states_host_local_lock_and_stopped_campaigns_only(capsys):
    assert "host-local" in _subcommand_help(capsys, "prune-checkpoints")
    sp = _subcommand_help(capsys, "split-progress")
    assert "stopped campaigns only" in sp and "--force" in sp
