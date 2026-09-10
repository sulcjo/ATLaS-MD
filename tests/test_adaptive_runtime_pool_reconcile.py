"""Runtime-pool ledger durability and reconciliation against delivered MD.

The ledger used to be persisted only at coarse boundaries (end of a scheduled
epoch, or a graceful shutdown).  A hard kill -- SIGSEGV/SIGABRT, i.e. the exit
139/134 seen repeatedly on the chignolin_7 chain -- discarded every charge
accrued since the last flush.  The resumed run then took ``run_segment``'s
"already complete ... no pool charge" fast path, so the lost charge was never
re-booked and ``used_ns`` drifted permanently low, once per crash.

Measured on chignolin_7 before this fix: 247.59 ns of delivered MD missing from
a ledger that claimed 652.68 ns used.

These tests are intentionally OpenMM-free.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.io import read_json_file, write_json
from gareus.adaptive_production import (
    AdaptiveRuntimePool,
    _adaptive_runtime_pool_validate,
    reconcile_runtime_pool_with_delivered_md,
    stage_delivered_md_steps,
)


def _make_stage(adaptive_dir: Path, rel: str, prod_done: int, n_states: int) -> Path:
    """Write the two artifacts a stage exposes: its checkpoint and its window map."""
    stage = adaptive_dir / rel
    (stage / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_json(
        stage / "checkpoints" / "production_checkpoint_manifest.json",
        {"prod_done": int(prod_done), "absolute_step": int(prod_done) + 1_010_000},
    )
    rows = ["epoch_window,state_id,primary_center,secondary_center"]
    rows += [f"{i},{i},0.1," for i in range(n_states)]
    (stage / "epoch_window_map.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return stage


def test_consume_flushes_the_ledger_immediately_when_bound(tmp_path: Path):
    # Arrange
    pool = AdaptiveRuntimePool(total_ns=100.0, timestep_fs=3.5)
    pool.bind_ledger(tmp_path)

    # Act -- one charge, no explicit report write
    pool.consume(label="epoch_001/baseline", kind="scheduled_epoch", n_states=4, steps=1000)

    # Assert -- the charge is on disk already, not just in memory
    on_disk = read_json_file(tmp_path / "adaptive_runtime_pool.json", {})
    assert [e["label"] for e in on_disk["events"]] == ["epoch_001/baseline"]
    assert abs(on_disk["used_ns"] - pool.used_ns) < 1.0e-12


def test_unbound_pool_does_not_write_a_ledger(tmp_path: Path):
    pool = AdaptiveRuntimePool(total_ns=100.0, timestep_fs=3.5)
    pool.consume(label="epoch_000", kind="adaptive_epoch", n_states=4, steps=1000)
    assert not (tmp_path / "adaptive_runtime_pool.json").exists()


def test_stage_delivered_md_steps_reads_checkpoint_and_window_map(tmp_path: Path):
    _make_stage(tmp_path, "epoch_000", prod_done=1_275_510, n_states=112)
    _make_stage(tmp_path, "epoch_001/topup_001_1539000", prod_done=1_539_000, n_states=2)

    delivered = stage_delivered_md_steps(tmp_path)

    assert delivered["epoch_000"] == (1_275_510, 112)
    assert delivered["epoch_001/topup_001_1539000"] == (1_539_000, 2)


def test_reconcile_books_md_that_a_hard_kill_lost(tmp_path: Path):
    # Arrange -- chignolin_7's real shape: baseline half-charged, both top-ups unbooked
    _make_stage(tmp_path, "epoch_000", prod_done=1_275_510, n_states=112)
    _make_stage(tmp_path, "epoch_001/baseline", prod_done=797_193, n_states=112)
    _make_stage(tmp_path, "epoch_001/topup_001_1539000", prod_done=1_539_000, n_states=2)
    pool = AdaptiveRuntimePool(total_ns=6000.0, timestep_fs=3.5)
    pool.consume(label="epoch_000", kind="adaptive_epoch", n_states=112, steps=1_275_510)
    pool.consume(label="epoch_001/baseline", kind="scheduled_epoch", n_states=112, steps=389_500)
    used_before = pool.used_ns

    # Act
    report = reconcile_runtime_pool_with_delivered_md(tmp_path, pool, apply=True)

    # Assert -- epoch_000 was already exact and must not be touched
    by_label = {r["label"]: r for r in report["stages"]}
    assert by_label["epoch_000"]["missing_steps"] == 0
    assert by_label["epoch_001/baseline"]["missing_steps"] == 797_193 - 389_500
    assert by_label["epoch_001/topup_001_1539000"]["missing_steps"] == 1_539_000

    expected_ns = (112 * 407_693 + 2 * 1_539_000) * 3.5 / 1.0e6
    assert abs(pool.used_ns - (used_before + expected_ns)) < 1.0e-9
    assert abs(report["recovered_ns"] - expected_ns) < 1.0e-9

    recovered = [e for e in pool.events if e["kind"] == "reconciliation"]
    assert {e["label"] for e in recovered} == {"epoch_001/baseline", "epoch_001/topup_001_1539000"}


def test_reconcile_is_idempotent(tmp_path: Path):
    _make_stage(tmp_path, "epoch_001/baseline", prod_done=797_193, n_states=112)
    pool = AdaptiveRuntimePool(total_ns=6000.0, timestep_fs=3.5)
    pool.consume(label="epoch_001/baseline", kind="scheduled_epoch", n_states=112, steps=389_500)

    first = reconcile_runtime_pool_with_delivered_md(tmp_path, pool, apply=True)
    used_after_first = pool.used_ns
    n_events_after_first = len(pool.events)

    second = reconcile_runtime_pool_with_delivered_md(tmp_path, pool, apply=True)

    assert first["recovered_ns"] > 0.0
    assert second["recovered_ns"] == 0.0
    assert pool.used_ns == used_after_first
    assert len(pool.events) == n_events_after_first


def test_reconcile_dry_run_reports_without_charging(tmp_path: Path):
    _make_stage(tmp_path, "epoch_001/baseline", prod_done=797_193, n_states=112)
    pool = AdaptiveRuntimePool(total_ns=6000.0, timestep_fs=3.5)
    pool.consume(label="epoch_001/baseline", kind="scheduled_epoch", n_states=112, steps=389_500)
    used_before = pool.used_ns

    report = reconcile_runtime_pool_with_delivered_md(tmp_path, pool, apply=False)

    assert report["recovered_ns"] > 0.0
    assert pool.used_ns == used_before
    assert not [e for e in pool.events if e["kind"] == "reconciliation"]


def test_reconcile_warns_but_never_lowers_used_ns_on_overcharge(tmp_path: Path):
    # A ledger claiming MORE than the checkpoints delivered is a different defect
    # (double-charging).  Silently shrinking a budget is not auditable, so the
    # reconciler reports it and leaves used_ns alone.
    _make_stage(tmp_path, "epoch_001/baseline", prod_done=100_000, n_states=8)
    pool = AdaptiveRuntimePool(total_ns=6000.0, timestep_fs=3.5)
    pool.consume(label="epoch_001/baseline", kind="scheduled_epoch", n_states=8, steps=250_000)
    used_before = pool.used_ns

    report = reconcile_runtime_pool_with_delivered_md(tmp_path, pool, apply=True)

    assert report["recovered_ns"] == 0.0
    assert pool.used_ns == used_before
    assert any("overcharge" in w for w in pool.warnings)
    assert report["overcharged_stages"] == ["epoch_001/baseline"]


def test_reconcile_ignores_stages_with_no_checkpoint_yet(tmp_path: Path):
    stage = tmp_path / "epoch_001" / "topup_002_2407000"
    stage.mkdir(parents=True)
    (stage / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n", encoding="utf-8")
    pool = AdaptiveRuntimePool(total_ns=6000.0, timestep_fs=3.5)

    report = reconcile_runtime_pool_with_delivered_md(tmp_path, pool, apply=True)

    assert report["stages"] == []
    assert pool.used_ns == 0.0


def test_validate_flags_a_ledger_that_understates_delivered_md(tmp_path: Path):
    _make_stage(tmp_path, "epoch_001/baseline", prod_done=797_193, n_states=112)
    pool = AdaptiveRuntimePool(total_ns=6000.0, timestep_fs=3.5)
    pool.consume(label="epoch_001/baseline", kind="scheduled_epoch", n_states=112, steps=389_500)

    payload = _adaptive_runtime_pool_validate(tmp_path, pool, label="runtime_pool_test")

    assert payload["status"] == "warning"
    assert payload["unbooked_md_ns"] > 0.0
    assert "epoch_001/baseline" in payload["unbooked_stages"]
