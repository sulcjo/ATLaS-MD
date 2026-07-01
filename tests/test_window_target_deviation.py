"""Tests for the achieved-vs-nominal window diagnostics.

Motivating bug (IGGFM run, analysis_IGGFM): windows nominally spaced across
CV1 (e.g. targets 0.0, 0.104, 0.2079, 0.3396, 0.4713) but pinned at the same
secondary_cv_center (extended/beta rama basin) never reached their assigned
CV1 target -- the achieved sample mean stayed near the CV1=0 window's basin
regardless of nominal separation. ``build_geometry_edges`` only wires up
nominal near-neighbors (sorted-primary chain + one nearest-2D neighbor per
state), so this kind of collapse between *non-neighbor* windows was invisible
to both the per-state diagnostics and the redundant-overlap retirement logic.

These tests are OpenMM-free; they exercise ``collect_epoch_diagnostics``
directly against synthetic samples.csv fixtures.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    collect_epoch_diagnostics,
    propose_actions_from_diagnostics,
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


def _write_samples_csv(path: Path, per_window_cv, per_window_secondary, rows_per_window: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for w, (cv_mean, cv_std) in enumerate(per_window_cv):
        sec_mean, sec_std = per_window_secondary[w]
        cv_vals = RNG.normal(cv_mean, cv_std, rows_per_window)
        sec_vals = RNG.normal(sec_mean, sec_std, rows_per_window)
        for cv, sec in zip(cv_vals, sec_vals):
            rows.append({"window": w, "cv_A": round(float(cv), 6), "secondary_cv": round(float(sec), 6)})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["window", "cv_A", "secondary_cv"])
        writer.writeheader()
        writer.writerows(rows)


def _iggfm_like_fixture(tmp_path: Path):
    """5 windows: 0,1,2,3 form a normal, well-behaved primary chain (all at
    secondary=-0.33). Window 4 is nominally far out on the primary axis
    (target 0.9) but ALSO sits at secondary=-1.0 (like the IGGFM beta-basin
    windows) and its achieved samples collapse onto window 0's basin instead
    of reaching its own target -- window 4 is not a geometry neighbor of
    window 0 (its primary-chain neighbor is window 3, its nearest-2D neighbor
    is also window 3 since window 0 is far away on the primary axis), so this
    collapse is only visible via achieved-sample overlap, not via nominal
    geometry.
    """
    centers = [
        (0.0, -0.33),
        (0.3, -0.33),
        (0.6, -0.33),
        (0.85, -0.33),
        (0.9, -1.0),  # nominally near window 3, but behaves like window 0
    ]
    window_csv = _write_window_csv(tmp_path / "umbrella_windows.csv", centers)
    registry = registry_from_window_csv(window_csv)

    # Windows 0-3: achieved mean tracks the nominal target closely (healthy).
    per_window_cv = [
        (0.01, 0.02),
        (0.3, 0.02),
        (0.6, 0.02),
        (0.85, 0.02),
        (0.02, 0.02),  # window 4: collapsed back onto window 0's basin, not its own 0.9 target
    ]
    per_window_secondary = [(-0.33, 0.05)] * 4 + [(-1.0, 0.02)]
    epoch_dir = tmp_path / "epoch_000"
    _write_samples_csv(epoch_dir / "samples.csv", per_window_cv, per_window_secondary)
    return registry, epoch_dir


def test_off_target_window_flagged_by_deviation_sigma(tmp_path):
    registry, epoch_dir = _iggfm_like_fixture(tmp_path)
    policy = AdaptiveDecisionPolicy()
    diag = collect_epoch_diagnostics(epoch_dir, registry, policy)
    states = {s["state_id"]: s for s in diag["states"]}

    # Window 4 (state_id 4) never reached its target (0.9 vs achieved ~0.01) --
    # must be flagged.
    assert states[4]["primary_target_deviation_sigma"] is not None
    assert states[4]["primary_target_deviation_sigma"] >= policy.max_target_deviation_sigma
    assert "off_target_primary" in states[4]["warnings"]

    # A healthy window (state_id 1) tracks its target within normal thermal
    # noise and must NOT be flagged.
    assert "off_target_primary" not in states[1]["warnings"]


def test_non_neighbor_redundancy_detected_for_collapsed_window(tmp_path):
    registry, epoch_dir = _iggfm_like_fixture(tmp_path)
    policy = AdaptiveDecisionPolicy()
    diag = collect_epoch_diagnostics(epoch_dir, registry, policy)

    edge_pairs = {tuple(sorted((e["state_i"], e["state_j"]))) for e in diag["edges"]}
    assert (0, 4) not in edge_pairs, "0 and 4 must not already be nominal geometry neighbors"

    redundancies = diag.get("non_neighbor_redundancies", [])
    pairs = {tuple(sorted((r["state_i"], r["state_j"]))): r for r in redundancies}
    assert (0, 4) in pairs
    assert pairs[(0, 4)]["overlap"] >= policy.redundant_overlap


def test_well_separated_windows_have_no_non_neighbor_redundancy(tmp_path):
    centers = [(0.0, None), (1.0, None), (2.0, None)]
    window_csv = _write_window_csv(tmp_path / "umbrella_windows.csv", centers)
    registry = registry_from_window_csv(window_csv)
    per_window_cv = [(0.0, 0.05), (1.0, 0.05), (2.0, 0.05)]
    per_window_secondary = [(0.0, 0.0)] * 3
    epoch_dir = tmp_path / "epoch_000"
    _write_samples_csv(epoch_dir / "samples.csv", per_window_cv, per_window_secondary)

    diag = collect_epoch_diagnostics(epoch_dir, registry, AdaptiveDecisionPolicy())
    assert diag.get("non_neighbor_redundancies", []) == []


def test_off_target_secondary_flagged_by_deviation_sigma(tmp_path):
    # State 1's primary target is reached fine, but it never reaches its own
    # secondary_cv_center (1.0) and instead sits near state 0's secondary
    # value (0.0) -- the secondary-axis analogue of the IGGFM CV1 collapse.
    centers = [(0.0, 0.0), (0.0, 1.0)]
    window_csv = _write_window_csv(tmp_path / "umbrella_windows.csv", centers)
    registry = registry_from_window_csv(window_csv)
    per_window_cv = [(0.0, 0.02), (0.0, 0.02)]
    per_window_secondary = [(0.0, 0.05), (0.02, 0.02)]
    epoch_dir = tmp_path / "epoch_000"
    _write_samples_csv(epoch_dir / "samples.csv", per_window_cv, per_window_secondary)

    policy = AdaptiveDecisionPolicy()
    diag = collect_epoch_diagnostics(epoch_dir, registry, policy)
    states = {s["state_id"]: s for s in diag["states"]}

    assert states[1]["secondary_target_deviation_sigma"] is not None
    assert states[1]["secondary_target_deviation_sigma"] >= policy.max_target_deviation_sigma
    assert "off_target_secondary" in states[1]["warnings"]
    assert "off_target_primary" not in states[1]["warnings"]


def test_full_pipeline_never_retires_the_on_target_anchor(tmp_path):
    """Full pipeline: collect_epoch_diagnostics -> propose_actions_from_diagnostics.

    Window 4 collapsed onto window 0's basin (see _iggfm_like_fixture) and is
    flagged off_target_primary by collect_epoch_diagnostics; window 0 is not.
    Whatever the geometry graph and pre-existing safety gates (bad_touching,
    articulation points, min_active_states) end up deciding about window 4,
    the non-neighbor-redundancy signal must never make the healthy,
    on-target anchor (window 0) retirement-eligible -- _hist_overlap is
    symmetric, so without the off-target gate it would be exactly as
    "redundant" as window 4 and could be dropped instead of it.
    """
    registry, epoch_dir = _iggfm_like_fixture(tmp_path)
    policy = AdaptiveDecisionPolicy(retire_converged=True, min_active_states=0)
    diag = collect_epoch_diagnostics(epoch_dir, registry, policy)

    redundancies = diag.get("non_neighbor_redundancies", [])
    assert any({r["state_i"], r["state_j"]} == {0, 4} for r in redundancies)
    state_warnings = {s["state_id"]: s["warnings"] for s in diag["states"]}
    assert "off_target_primary" in state_warnings[4]
    assert "off_target_primary" not in state_warnings[0]

    actions = propose_actions_from_diagnostics(registry, diag, policy)
    assert not any(a[0] == "retire" and a[1] == 0 for a in actions)
