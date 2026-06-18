"""Tests for segmented epoch diagnostics and zero-sample guard.

These tests are intentionally OpenMM-free.  They validate the diagnostic
collection path used by the adaptive production controller and the guard helper
that converts a silent controller-freeze into a loud RuntimeError.

TDD note on RED/GREEN status
-----------------------------
Test A is a characterisation regression test: both collectors already exist, so
the test passes before *and* after the production fix.  Its purpose is to
document and protect the behavioural difference between the two collectors; the
actual production call-site swaps are inside the OpenMM-driven loop and are not
unit-testable here.

Test B is a genuine RED→GREEN test: ``_assert_epoch_has_samples`` did not exist
before this patch, so the test file itself could not even be imported before the
fix was applied.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    _assert_epoch_has_samples,
    build_union_state_mbar_inputs,
    collect_epoch_diagnostics,
    collect_segmented_epoch_diagnostics,
    evaluate_adaptive_convergence_gate,
    propose_actions_from_diagnostics,
    registry_from_window_csv,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

N_WINDOWS = 3
# Each window gets this many rows per segment; two segments per epoch.
ROWS_PER_WINDOW_PER_SEGMENT = 150
# Total per window: 150 * 2 = 300, which exceeds min_samples_for_retire (200)
# and convergence_min_samples_per_state (50).
ROWS_PER_WINDOW_TOTAL = ROWS_PER_WINDOW_PER_SEGMENT * 2


def _write_window_csv(path: Path, n_windows: int = N_WINDOWS) -> Path:
    """Write a minimal window CSV accepted by registry_from_window_csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "primary_cv_center",
                "primary_cv_k_kcal",
                "secondary_cv_center",
                "secondary_cv_k_kcal_mol",
            ],
        )
        writer.writeheader()
        for i in range(n_windows):
            writer.writerow(
                {
                    "primary_cv_center": round(0.2 + 0.3 * i, 3),
                    "primary_cv_k_kcal": 10.0,
                    "secondary_cv_center": round(-0.5 + 0.5 * i, 3),
                    "secondary_cv_k_kcal_mol": 5.0,
                }
            )
    return path


def _write_samples_csv(path: Path, n_windows: int, rows_per_window: int) -> int:
    """Write a synthetic samples.csv with ``rows_per_window`` rows per window.

    The ``window`` column uses 0-based indices matching the fallback window_map
    produced by ``_load_epoch_window_map`` (state_ids assigned sequentially from 0).
    """
    rng = np.random.default_rng(12345)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in range(n_windows):
        cv_vals = rng.normal(0.2 + 0.3 * w, 0.02, rows_per_window).clip(0.0, 1.0)
        sec_vals = rng.normal(-0.5 + 0.5 * w, 0.05, rows_per_window).clip(-1.0, 1.0)
        boost_vals = rng.normal(1.0, 0.3, rows_per_window)
        for cv, sec, boost in zip(cv_vals, sec_vals, boost_vals):
            rows.append(
                {
                    "window": w,
                    "cv_A": round(float(cv), 6),
                    "secondary_cv": round(float(sec), 6),
                    "gamd_boost_total_kcal_mol": round(float(boost), 6),
                }
            )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["window", "cv_A", "secondary_cv", "gamd_boost_total_kcal_mol"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _make_segmented_epoch(
    epoch_dir: Path,
    n_windows: int = N_WINDOWS,
    rows_per_window_per_segment: int = ROWS_PER_WINDOW_PER_SEGMENT,
) -> Tuple[int, int]:
    """Create epoch_dir with baseline/ and topup_001/ subdirs, each with samples.csv.

    Returns (rows_in_baseline, rows_in_topup).
    """
    baseline_dir = epoch_dir / "baseline"
    topup_dir = epoch_dir / "topup_001"
    n_baseline = _write_samples_csv(
        baseline_dir / "samples.csv", n_windows, rows_per_window_per_segment
    )
    n_topup = _write_samples_csv(
        topup_dir / "samples.csv", n_windows, rows_per_window_per_segment
    )
    return n_baseline, n_topup


def _write_parquet_epoch_run(run_dir: Path, n_windows: int = N_WINDOWS, rows_per_window: int = 24) -> int:
    """Write current canonical Parquet production outputs under one run dir."""
    from gareus.store import ParquetExchangeWriter, ParquetSampleWriter, SegmentRegistry

    rng = np.random.default_rng(2468)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gareus_metadata.json").write_text(json.dumps({"temperature_K": 300.0}), encoding="utf-8")

    reg = SegmentRegistry(run_dir)
    seg_id = reg.open_segment("run_001", None, 1)
    samples = ParquetSampleWriter(run_dir / "samples" / seg_id, flush_rows=10000)
    rows = 0
    for w in range(n_windows):
        center = 0.2 + 0.3 * w
        for i in range(rows_per_window):
            step = 100 * (rows + 1)
            cv = float(rng.normal(center, 0.02))
            sec = float(rng.normal(-0.5 + 0.5 * w, 0.05))
            boost = 4.184 * float(abs(rng.normal(1.0, 0.1)))
            samples.write_sample(step, i % 2, w, cv, sec, -100.0, boost, boost * 0.5, boost * 0.5)
            rows += 1
    samples.close()

    exchanges = ParquetExchangeWriter(run_dir / "exchanges" / seg_id, flush_rows=1000)
    exchanges.write_exchange(step=1000, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.0, accepted=True)
    exchanges.close()

    reg.close_segment(seg_id, end_step=100 * rows)
    return rows


# ---------------------------------------------------------------------------
# Test A — regression for the freeze
# ---------------------------------------------------------------------------


def test_segmented_vs_flat_collector(tmp_path: Path) -> None:
    """Flat collector sees zero samples on a segmented epoch; segmented collector
    sees the pooled total.  This regression test protects the behavioural
    difference that the production call-site swap relies on.

    Additionally asserts that propose_actions_from_diagnostics does NOT emit an
    'extend' action for a well-sampled segmented epoch, and that
    evaluate_adaptive_convergence_gate does NOT report those windows in
    low_sample_states.
    """
    window_csv = _write_window_csv(tmp_path / "windows.csv")
    registry = registry_from_window_csv(window_csv, epoch=0, source="test")
    policy = AdaptiveDecisionPolicy()

    epoch_dir = tmp_path / "epoch_000"
    _make_segmented_epoch(epoch_dir)

    # The flat collector reads epoch_dir/samples.csv which does not exist here.
    flat_diag = collect_epoch_diagnostics(epoch_dir, registry, policy=policy)
    flat_total = sum(
        int(s.get("sample_count", 0) or 0) for s in flat_diag.get("states", [])
    )
    assert flat_total == 0, (
        f"Flat collector on segmented epoch root should report 0 samples "
        f"(demonstrates the freeze bug), got {flat_total}"
    )

    # The segmented collector sums across baseline/ and topup_001/.
    seg_diag = collect_segmented_epoch_diagnostics(epoch_dir, registry, policy)
    seg_total = sum(
        int(s.get("sample_count", 0) or 0) for s in seg_diag.get("states", [])
    )
    expected_total = N_WINDOWS * ROWS_PER_WINDOW_TOTAL
    assert seg_total == expected_total, (
        f"Segmented collector should report {expected_total} samples, got {seg_total}"
    )
    # Per-state counts must all be > 0
    for state in seg_diag.get("states", []):
        sid = state.get("state_id")
        count = int(state.get("sample_count", 0) or 0)
        assert count > 0, f"State {sid} has zero samples in segmented diagnostics"

    # propose_actions_from_diagnostics: no 'extend' for any well-sampled state.
    actions = propose_actions_from_diagnostics(registry, seg_diag, policy=policy)
    extend_actions = [a for a in actions if a[0] == "extend"]
    # ROWS_PER_WINDOW_TOTAL=300 >= min_samples_for_retire=200 so no extend expected
    assert extend_actions == [], (
        f"Expected no 'extend' actions for well-sampled states (300 samples/window >= "
        f"min_samples_for_retire={policy.min_samples_for_retire}), got: {extend_actions}"
    )

    # evaluate_adaptive_convergence_gate: no low_sample_states
    gate = evaluate_adaptive_convergence_gate(
        epoch_dir, epoch=1, registry=registry,
        diagnostics=seg_diag, actions=actions, policy=policy,
    )
    low_sample_states = gate.get("low_sample_states", [])
    assert low_sample_states == [], (
        f"Expected no low_sample_states from segmented diagnostics, got: {low_sample_states}"
    )


def test_collect_epoch_diagnostics_reads_parquet_outputs(tmp_path: Path) -> None:
    """Canonical production output is Parquet; diagnostics must not require samples.csv."""
    window_csv = _write_window_csv(tmp_path / "windows.csv", n_windows=N_WINDOWS)
    registry = registry_from_window_csv(window_csv, epoch=0, source="test")
    policy = AdaptiveDecisionPolicy()

    epoch_dir = tmp_path / "epoch_parquet"
    expected_rows = _write_parquet_epoch_run(epoch_dir, n_windows=N_WINDOWS, rows_per_window=12)

    diag = collect_epoch_diagnostics(epoch_dir, registry, policy=policy)

    assert int(diag.get("n_samples_rows", 0)) == expected_rows
    assert int(diag.get("n_exchange_rows", 0)) == 1
    counts = {int(s["state_id"]): int(s["sample_count"]) for s in diag.get("states", [])}
    assert counts == {0: 12, 1: 12, 2: 12}
    edge_01 = next(
        e for e in diag.get("edges", [])
        if {int(e["state_i"]), int(e["state_j"])} == {0, 1}
    )
    assert int(edge_01["exchange_attempts"]) == 1
    assert int(edge_01["exchange_accepted"]) == 1


def test_build_union_state_mbar_inputs_reads_parquet_final(tmp_path: Path) -> None:
    """Union-state MBAR input builder must consume final/samples/*.parquet."""
    window_csv = _write_window_csv(tmp_path / "windows.csv", n_windows=N_WINDOWS)
    registry = registry_from_window_csv(window_csv, epoch=0, source="test")

    adaptive_dir = tmp_path / "adaptive"
    expected_rows = _write_parquet_epoch_run(adaptive_dir / "final", n_windows=N_WINDOWS, rows_per_window=10)

    meta = build_union_state_mbar_inputs(adaptive_dir, registry)

    assert int(meta["n_samples"]) == expected_rows
    assert int(meta["n_states"]) == N_WINDOWS
    with np.load(meta["arrays_npz"], allow_pickle=False) as data:
        assert data["cv_A"].shape == (expected_rows,)
        assert data["umbrella_reduced_bias_nk"].shape == (expected_rows, N_WINDOWS)
        assert np.isfinite(data["umbrella_reduced_bias_nk"]).all()


# ---------------------------------------------------------------------------
# Test B — zero-sample guard
# ---------------------------------------------------------------------------


def test_assert_epoch_has_samples_raises_on_zero_with_steps(tmp_path: Path) -> None:
    """Guard raises RuntimeError when epoch ran (steps > 0) but all states have
    zero samples (as would happen with the flat collector on a segmented epoch).
    """
    window_csv = _write_window_csv(tmp_path / "windows.csv")
    registry = registry_from_window_csv(window_csv, epoch=0, source="test")
    policy = AdaptiveDecisionPolicy()

    # Build a segmented epoch and deliberately collect flat diagnostics to
    # reproduce the all-zeros condition.
    epoch_dir = tmp_path / "epoch_bad"
    _make_segmented_epoch(epoch_dir)
    flat_diag = collect_epoch_diagnostics(epoch_dir, registry, policy=policy)

    # Sanity: confirm all zeros (otherwise the test premise is wrong).
    total = sum(int(s.get("sample_count", 0) or 0) for s in flat_diag.get("states", []))
    assert total == 0, f"Test premise broken: flat_diag has {total} samples"

    # Guard MUST raise when steps > 0 and samples == 0.
    with pytest.raises(RuntimeError, match="Zero samples"):
        _assert_epoch_has_samples(flat_diag, registry, epoch_dir, steps=1000)


def test_assert_epoch_has_samples_silent_when_steps_zero(tmp_path: Path) -> None:
    """Guard must NOT raise when steps == 0 (legitimately empty epoch)."""
    window_csv = _write_window_csv(tmp_path / "windows.csv")
    registry = registry_from_window_csv(window_csv, epoch=0, source="test")
    policy = AdaptiveDecisionPolicy()

    epoch_dir = tmp_path / "epoch_empty"
    _make_segmented_epoch(epoch_dir)
    flat_diag = collect_epoch_diagnostics(epoch_dir, registry, policy=policy)

    # Should not raise when steps == 0 regardless of sample count.
    _assert_epoch_has_samples(flat_diag, registry, epoch_dir, steps=0)


def test_assert_epoch_has_samples_silent_when_samples_exist(tmp_path: Path) -> None:
    """Guard must NOT raise when samples are present (healthy epoch)."""
    window_csv = _write_window_csv(tmp_path / "windows.csv")
    registry = registry_from_window_csv(window_csv, epoch=0, source="test")
    policy = AdaptiveDecisionPolicy()

    epoch_dir = tmp_path / "epoch_ok"
    _make_segmented_epoch(epoch_dir)
    seg_diag = collect_segmented_epoch_diagnostics(epoch_dir, registry, policy)

    # Should not raise when samples are present.
    _assert_epoch_has_samples(seg_diag, registry, epoch_dir, steps=5000)


def test_assert_epoch_has_samples_raises_on_missing_active_state(tmp_path: Path) -> None:
    """Guard raises RuntimeError when diagnostics have total_samples > 0 but one
    active state is missing from the states list entirely.

    This validates the ``missing_ids`` branch of the guard independently of the
    zero-samples branch.
    """
    window_csv = _write_window_csv(tmp_path / "windows.csv", n_windows=N_WINDOWS)
    registry = registry_from_window_csv(window_csv, epoch=0, source="test")

    # Collect real diagnostics from a segmented epoch.
    epoch_dir = tmp_path / "epoch_missing_state"
    _make_segmented_epoch(epoch_dir)
    policy = AdaptiveDecisionPolicy()
    seg_diag = collect_segmented_epoch_diagnostics(epoch_dir, registry, policy)

    # Confirm there ARE samples so the test exercises the missing-state branch only.
    total = sum(int(s.get("sample_count", 0) or 0) for s in seg_diag.get("states", []))
    assert total > 0, f"Test premise broken: segmented diag has {total} samples"

    # Drop the last state entry to simulate a missing active state.
    states_list = list(seg_diag.get("states", []))
    assert len(states_list) == N_WINDOWS, (
        f"Expected {N_WINDOWS} state entries, got {len(states_list)}"
    )
    truncated_diag = dict(seg_diag)
    truncated_diag["states"] = states_list[:-1]  # drop one state

    with pytest.raises(RuntimeError, match="missing from diagnostics"):
        _assert_epoch_has_samples(truncated_diag, registry, epoch_dir, steps=5000)


# ---------------------------------------------------------------------------
# Test C — pooled overlap replaces nanmin-of-per-segment aggregation
# ---------------------------------------------------------------------------


def _write_samples_csv_good_overlap(path: Path, n_windows: int, rows_per_window: int = 300) -> None:
    """Write samples.csv where adjacent windows genuinely overlap (wide std, close centers).

    Centers: 0.3, 0.42, 0.54 with std=0.10 → pooled overlap for edge (0,1) ~0.49
    with seed=42 (margin ~0.24 above target_overlap=0.25, robust across numpy versions).
    """
    rng = np.random.default_rng(42)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for w in range(n_windows):
        center = 0.3 + 0.12 * w
        cv_vals = rng.normal(center, 0.10, rows_per_window).clip(0.0, 1.0)
        sec_vals = rng.normal(-0.5 + 0.5 * w, 0.05, rows_per_window).clip(-1.0, 1.0)
        boost_vals = rng.normal(1.0, 0.3, rows_per_window)
        for cv, sec, boost in zip(cv_vals, sec_vals, boost_vals):
            rows.append(
                {
                    "window": w,
                    "cv_A": round(float(cv), 6),
                    "secondary_cv": round(float(sec), 6),
                    "gamd_boost_total_kcal_mol": round(float(boost), 6),
                }
            )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["window", "cv_A", "secondary_cv", "gamd_boost_total_kcal_mol"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_samples_csv_single_window(path: Path, window_idx: int = 1, rows: int = 300) -> None:
    """Write samples.csv for a topup that only sampled one window.

    This simulates a topup segment that ran only for window ``window_idx``.
    For any edge where the other endpoint has no samples, the per-segment
    overlap is degenerate (0 / None), but the pooled overlap across both
    segments should remain healthy.
    """
    rng = np.random.default_rng(99)
    path.parent.mkdir(parents=True, exist_ok=True)
    center = 0.3 + 0.12 * window_idx
    cv_vals = rng.normal(center, 0.10, rows).clip(0.0, 1.0)
    rows_data = [
        {
            "window": window_idx,
            "cv_A": round(float(cv), 6),
            "secondary_cv": round(0.0, 6),
            "gamd_boost_total_kcal_mol": round(1.0, 6),
        }
        for cv in cv_vals
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["window", "cv_A", "secondary_cv", "gamd_boost_total_kcal_mol"],
        )
        writer.writeheader()
        writer.writerows(rows_data)


def test_pooled_overlap_not_dragged_by_topup_segment(tmp_path: Path) -> None:
    """Test C — pooled overlap aggregation.

    Scenario: 3-window registry; segment 1 (baseline) covers all 3 windows
    with good neighbor overlap; segment 2 (topup) covers ONLY window 1.

    The old nanmin-of-per-segment approach would union the ``low_or_missing_overlap``
    warning from the topup into the aggregate edge (because the topup produces None
    overlap for edges 0-1 and 1-2, which the collector treated as missing/low).

    The new pooled approach must:
    1. Compute aggregate edge overlap from the pooled cv_A arrays (both segments
       contribute window-1 data; only segment 1 contributes windows 0 and 2).
    2. Edge 0-1 pooled overlap must be >= target_overlap (good, not dragged).
    3. ``segment_min_overlap`` records the degenerate per-segment value (0.0) from
       the topup segment that only saw one endpoint of the edge.
    4. ``nonstationary_overlap`` warning is present because
       pooled_overlap - segment_min_overlap > NONSTATIONARY_OVERLAP_DELTA.
    5. ``low_or_missing_overlap`` is NOT in the aggregate edge warnings (the
       pooled overlap is genuinely good; the warning must come from pooled data
       only, not be blindly unioned from per-segment warnings).
    """
    window_csv = _write_window_csv(tmp_path / "windows.csv", n_windows=3)
    registry = registry_from_window_csv(window_csv, epoch=0, source="test")
    policy = AdaptiveDecisionPolicy()

    epoch_dir = tmp_path / "epoch_pooled"

    # Segment 1: all 3 windows, good overlap
    _write_samples_csv_good_overlap(
        epoch_dir / "baseline" / "samples.csv", n_windows=3
    )
    # Segment 2 (topup): ONLY window 1 — degenerate for edges 0-1 and 1-2
    _write_samples_csv_single_window(
        epoch_dir / "topup_001" / "samples.csv", window_idx=1
    )

    seg_diag = collect_segmented_epoch_diagnostics(epoch_dir, registry, policy)

    edges_by_pair = {
        (int(e["state_i"]), int(e["state_j"])): e
        for e in seg_diag.get("edges", [])
    }

    # Edge (0, 1): the critical edge with a degenerate-topup per-segment overlap
    assert (0, 1) in edges_by_pair, "Edge (0,1) must appear in diagnostics"
    edge_01 = edges_by_pair[(0, 1)]

    # 1. Pooled overlap must be >= target_overlap (good data from segment 1 must dominate)
    pooled_overlap = edge_01.get("overlap")
    assert pooled_overlap is not None, "Edge (0,1) must have a non-None pooled overlap"
    assert float(pooled_overlap) >= float(policy.target_overlap), (
        f"Pooled overlap {pooled_overlap:.4f} must be >= target_overlap "
        f"{policy.target_overlap} — baseline segment has good overlap, "
        "topup should not drag it down"
    )

    # 2. segment_min_overlap must exist and be low (0.0 — topup saw only window 1)
    seg_min = edge_01.get("segment_min_overlap")
    assert seg_min is not None, (
        "Edge (0,1) must have 'segment_min_overlap' field (from topup degenerate segment)"
    )
    assert float(seg_min) == 0.0, (
        f"segment_min_overlap for edge (0,1) must be 0.0 (topup only sampled window 1), "
        f"got {seg_min}"
    )

    # 3. nonstationary_overlap warning must be present
    warnings_01 = edge_01.get("warnings", [])
    assert "nonstationary_overlap" in warnings_01, (
        f"Edge (0,1) must have 'nonstationary_overlap' warning "
        f"(pooled={pooled_overlap:.3f} >> segment_min={seg_min}), "
        f"got warnings: {warnings_01}"
    )

    # 4. low_or_missing_overlap must NOT be in aggregate warnings (pooled overlap is good)
    assert "low_or_missing_overlap" not in warnings_01, (
        f"Edge (0,1) must NOT have spurious 'low_or_missing_overlap' warning when "
        f"pooled overlap {pooled_overlap:.3f} >= target {policy.target_overlap}. "
        f"Got warnings: {warnings_01}"
    )


def test_assert_epoch_has_samples_malformed_state_id_raises_runtime_error(tmp_path: Path) -> None:
    """Guard must raise RuntimeError (not TypeError) when a diagnostics entry has
    state_id=None (absent key).

    This validates Finding 1: the old ``int(s.get("state_id"))`` call would raise
    ``TypeError`` for a None value; the hardened version must funnel that into the
    guard's clean RuntimeError.
    """
    window_csv = _write_window_csv(tmp_path / "windows.csv", n_windows=2)
    registry = registry_from_window_csv(window_csv, epoch=0, source="test")

    epoch_dir = tmp_path / "epoch_malformed"

    # Build a diagnostics dict: one well-formed state with samples, one entry
    # whose state_id is None (missing key).  total_samples > 0 but the None-id
    # entry will not appear in reported_ids, so the missing-state branch fires
    # as a RuntimeError rather than a TypeError.
    malformed_diag: dict = {
        "states": [
            {"state_id": 0, "sample_count": 100},   # well-formed
            {"sample_count": 50},                     # state_id absent (None from .get)
        ]
    }

    # Must raise RuntimeError, not TypeError.
    with pytest.raises(RuntimeError):
        _assert_epoch_has_samples(malformed_diag, registry, epoch_dir, steps=1000)

    # Explicitly confirm it is NOT a TypeError bubbling up.
    try:
        _assert_epoch_has_samples(malformed_diag, registry, epoch_dir, steps=1000)
    except RuntimeError:
        pass  # expected
    except TypeError as exc:
        raise AssertionError(
            f"Guard raised TypeError instead of RuntimeError for None state_id: {exc}"
        ) from exc
