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
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    _assert_epoch_has_samples,
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
