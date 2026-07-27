"""The off-target-window check (achieved sample mean vs nominal restraint
target, in units of the window's own sample std) has run reliably during the
adaptive-discovery epoch loop since it was added (collect_epoch_diagnostics),
but collect_final_combined_diagnostics -- which aggregates the frozen final
phase's samples, i.e. exactly what feeds the PMF -- never called it: every
state's primary_target_deviation_sigma/secondary_target_deviation_sigma
stayed None regardless of how far off-target the pooled final samples were.

Confirmed on a real run (chignolin_sigma3_2d): epoch_002/baseline flagged
states 20-24 with off_target_primary (dev_sigma 3.1-4.3) every single round,
but adaptive_final_combined_diagnostics.json showed primary_target_deviation_sigma:
null for all 25 states, and evaluate_adaptive_quality_gate never surfaced it.

These tests are OpenMM-free; they exercise collect_final_combined_diagnostics
and evaluate_adaptive_quality_gate directly against synthetic samples.csv
fixtures, following the same pattern as test_window_target_deviation.py.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    collect_final_combined_diagnostics,
    evaluate_adaptive_quality_gate,
    quality_gate_fixable_by_more_final_sampling,
    registry_from_window_csv,
)

RNG = np.random.default_rng(2026)


def _write_window_csv(path: Path, centers) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["primary_cv_center", "primary_cv_k_kcal", "secondary_cv_center", "secondary_cv_k_kcal_mol"],
        )
        writer.writeheader()
        for primary, secondary in centers:
            writer.writerow(
                {
                    "primary_cv_center": primary,
                    "primary_cv_k_kcal": 80.0,
                    "secondary_cv_center": secondary,
                    "secondary_cv_k_kcal_mol": 100.0,
                }
            )
    return path


def _write_final_baseline_csv(adaptive_dir: Path, per_window_cv, per_window_secondary, rows_per_window: int) -> None:
    run_dir = adaptive_dir / "final" / "baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for w, (cv_mean, cv_std) in enumerate(per_window_cv):
        sec_mean, sec_std = per_window_secondary[w]
        cv_vals = RNG.normal(cv_mean, cv_std, rows_per_window)
        sec_vals = RNG.normal(sec_mean, sec_std, rows_per_window)
        for cv, sec in zip(cv_vals, sec_vals):
            rows.append({"window": w, "cv_A": round(float(cv), 6), "secondary_cv": round(float(sec), 6)})
    with (run_dir / "samples.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["window", "cv_A", "secondary_cv"])
        writer.writeheader()
        writer.writerows(rows)


def _two_window_fixture(tmp_path, rows_per_window=300, collapsed_secondary_state1=False):
    """State 0 target=0.0, tracks it fine. State 1 target=0.9, but its pooled
    final samples collapse back onto ~0.0 (restraint too soft at that end) --
    the exact chignolin_sigma3_2d windows-20-24 pattern, just 1D/2-window.
    """
    centers = [(0.0, 0.0), (0.9, 0.0)]
    window_csv = _write_window_csv(tmp_path / "umbrella_windows.csv", centers)
    registry = registry_from_window_csv(window_csv)
    # State 1's achieved mean sits right on top of state 0's -- heavy
    # histogram overlap with its neighbor (no incidental weak_edges finding),
    # while still miles from its own 0.9 target in its own tiny std (a real
    # off_target_primary finding). collapsed_secondary_state1 is unused here
    # but kept in the signature for future secondary-axis variants.
    per_window_cv = [(0.0, 0.02), (0.01, 0.02)]
    per_window_secondary = [(0.0, 0.02), (0.0, 0.02)]
    _write_final_baseline_csv(tmp_path, per_window_cv, per_window_secondary, rows_per_window)
    return registry


def test_final_off_target_state_is_flagged(tmp_path):
    registry = _two_window_fixture(tmp_path, rows_per_window=300)
    policy = AdaptiveDecisionPolicy()
    diag = collect_final_combined_diagnostics(tmp_path, registry, policy)
    states = {s["state_id"]: s for s in diag["states"]}

    assert states[1]["primary_target_deviation_sigma"] is not None
    assert states[1]["primary_target_deviation_sigma"] >= policy.max_target_deviation_sigma
    assert "off_target_primary" in states[1]["warnings"]

    assert "off_target_primary" not in states[0]["warnings"]


def test_final_low_sample_count_skips_deviation_check(tmp_path):
    """Below final_min_samples_per_state, deviation must stay unset rather
    than fire off a noisy small-n estimate (mirrors the real run's
    final/baseline snapshot, where n=500/window with 0 samples for two
    straggler states produced spurious/undefined moments)."""
    policy = AdaptiveDecisionPolicy(final_min_samples_per_state=1000)
    registry = _two_window_fixture(tmp_path, rows_per_window=300)
    diag = collect_final_combined_diagnostics(tmp_path, registry, policy)
    states = {s["state_id"]: s for s in diag["states"]}

    assert states[1]["primary_target_deviation_sigma"] is None
    assert "off_target_primary" not in states[1]["warnings"]
    assert "low_final_sample_count" in states[1]["warnings"]


def test_quality_gate_surfaces_off_target_final_states(tmp_path):
    registry = _two_window_fixture(tmp_path, rows_per_window=300)
    policy = AdaptiveDecisionPolicy()
    final_diag = collect_final_combined_diagnostics(tmp_path, registry, policy)
    gate = evaluate_adaptive_quality_gate(tmp_path, registry, final_diag, policy)

    off_target = {row["state_id"]: row for row in gate["off_target_states"]}
    assert 1 in off_target
    assert off_target[1]["primary_target_deviation_sigma"] >= policy.max_target_deviation_sigma
    assert 0 not in off_target

    assert gate["status"] == "needs_more_sampling"
    assert any("sigma away from their restraint target" in msg for msg in gate["needs_more_sampling"])


def test_quality_gate_does_not_flag_well_behaved_windows(tmp_path):
    centers = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)]
    window_csv = _write_window_csv(tmp_path / "umbrella_windows.csv", centers)
    registry = registry_from_window_csv(window_csv)
    per_window_cv = [(0.0, 0.02), (0.5, 0.02), (1.0, 0.02)]
    per_window_secondary = [(0.0, 0.02)] * 3
    _write_final_baseline_csv(tmp_path, per_window_cv, per_window_secondary, rows_per_window=300)

    policy = AdaptiveDecisionPolicy()
    final_diag = collect_final_combined_diagnostics(tmp_path, registry, policy)
    gate = evaluate_adaptive_quality_gate(tmp_path, registry, final_diag, policy)

    assert gate["off_target_states"] == []


def test_extension_loop_must_not_spin_on_off_target_alone(tmp_path):
    """status == "needs_more_sampling" also covers off_target_states, but
    more frozen-final sampling at the same restraint center can't fix a
    collapsed/pinned window -- it can even inflate the pooled std enough to
    shrink dev_sigma below threshold on a later round, silently clearing the
    flag with no physical change. The extension loop must stop here, not
    treat an off-target-only finding as a reason to keep running."""
    registry = _two_window_fixture(tmp_path, rows_per_window=300)
    policy = AdaptiveDecisionPolicy()
    final_diag = collect_final_combined_diagnostics(tmp_path, registry, policy)
    gate = evaluate_adaptive_quality_gate(tmp_path, registry, final_diag, policy)

    assert gate["status"] == "needs_more_sampling"
    assert gate["off_target_states"]
    assert gate["low_sample_states"] == []
    assert gate["weak_edges"] == []
    assert quality_gate_fixable_by_more_final_sampling(gate) is False


@pytest.mark.parametrize(
    "low_sample_states,weak_edges,expected",
    [
        ([], [], False),
        ([{"state_id": 1}], [], True),
        ([], [{"state_i": 0, "state_j": 1}], True),
        ([{"state_id": 1}], [{"state_i": 0, "state_j": 1}], True),
    ],
)
def test_quality_gate_fixable_by_more_final_sampling_truth_table(low_sample_states, weak_edges, expected):
    gate = {
        "low_sample_states": low_sample_states,
        "weak_edges": weak_edges,
        "off_target_states": [{"state_id": 99}],
        "high_boost_states": [{"state_id": 98}],
    }
    assert quality_gate_fixable_by_more_final_sampling(gate) is expected
