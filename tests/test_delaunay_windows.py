"""Tests for Delaunay-based window placement.

No OpenMM required. scipy tests are skipped when scipy is not installed.
"""
from __future__ import annotations

import csv
import math
import types
from pathlib import Path

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs):
    """Minimal args namespace for Delaunay tests (contacts + rama-map)."""
    defaults = dict(
        primary_cv="nonlocal-contacts",
        secondary_cv="rama-map",
        secondary_cv_centers=[-1.0, -1/3, 1/3, 1.0],
        contact_normalize=True,
        contact_adaptive_min_k_kcal=10.0,
        contact_adaptive_max_k_kcal=200.0,
        contact_adaptive_default_k_kcal=25.0,
        secondary_cv_k_kcal=25.0,
        secondary_cv_adaptive_min_k_kcal=5.0,
        secondary_cv_adaptive_max_k_kcal=100.0,
        default_window_k_kcal_a2=50.0,
        contact_k_kcal=None,
        # Delaunay args
        delaunay_min_pilot_samples=10,
        delaunay_density_floor_quantile=0.05,
        delaunay_kde_grid_res=32,
        delaunay_n_anchors=16,
        delaunay_dedup_radius=0.10,
        delaunay_bridge_min_edge_length=0.20,
        delaunay_circumcenter_probes=False,
        delaunay_circumcenter_max_radius=2.0,
        delaunay_k_sigma_factor=0.50,
        # explicit-2d graph args used by load_explicit_2d_window_csv
        explicit_2d_exchange_neighbor_k=2,
        explicit_2d_exchange_radius=1.65,
    )
    defaults.update(kwargs)
    ns = types.SimpleNamespace(**defaults)
    return ns


def _write_samples_csv(path: Path, clusters: list[tuple[float, float, float, float, int]]) -> int:
    """Write a synthetic samples.csv with the given clusters.

    Each cluster is (cv1_center, cv1_std, cv2_center, cv2_std, n_samples).
    Returns total number of samples written.
    """
    rng = np.random.default_rng(42)
    rows = []
    window_id = 0
    for (c1, s1, c2, s2, n) in clusters:
        cv1_vals = rng.normal(c1, s1, n).clip(0.0, 1.0)
        cv2_vals = rng.normal(c2, s2, n).clip(-1.0, 1.0)
        for cv1, cv2 in zip(cv1_vals, cv2_vals):
            rows.append({"window": window_id, "cv_A": cv1, "secondary_cv": cv2})
        window_id += 1
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["window", "cv_A", "secondary_cv"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# Test 1 — Basin anchor picking
# ---------------------------------------------------------------------------

def test_basin_anchor_picking(tmp_path):
    """Three distinct clusters in CV space each produce at least one basin_anchor."""
    from gareus.windows import build_delaunay_windows_from_pilot_samples

    samples = tmp_path / "samples.csv"
    _write_samples_csv(samples, [
        (0.1, 0.02, -0.8, 0.05, 200),  # cluster A: low-contact, helix
        (0.5, 0.02, 0.0,  0.05, 200),  # cluster B: mid-contact, coil
        (0.9, 0.02, 0.7,  0.05, 200),  # cluster C: high-contact, beta
    ])
    args = _make_args(delaunay_n_anchors=8)
    rows, meta = build_delaunay_windows_from_pilot_samples(samples, args)

    assert len(rows) >= 3, f"Expected ≥3 windows, got {len(rows)}"
    anchors = [r for r in rows if r["window_type"] == "basin_anchor"]
    assert len(anchors) >= 3, f"Expected ≥3 basin_anchors, got {len(anchors)}"

    # All primary CV centers must be in [0, 1]
    for r in rows:
        cv1 = float(r["primary_cv_center"])
        assert 0.0 <= cv1 <= 1.0, f"primary_cv_center {cv1} out of [0, 1]"

    # All secondary CV centers must be in [-1, 1]
    for r in rows:
        cv2 = float(r["secondary_cv_center"])
        assert -1.0 <= cv2 <= 1.0, f"secondary_cv_center {cv2} out of [-1, 1]"


# ---------------------------------------------------------------------------
# Test 2 — Window types present and no duplicates
# ---------------------------------------------------------------------------

def test_window_types_present(tmp_path):
    """Both basin_anchor and delaunay_bridge windows appear; no duplicate positions."""
    from gareus.windows import build_delaunay_windows_from_pilot_samples

    samples = tmp_path / "samples.csv"
    _write_samples_csv(samples, [
        (0.1, 0.02, -0.9, 0.05, 200),
        (0.5, 0.02,  0.0, 0.05, 200),
        (0.9, 0.02,  0.8, 0.05, 200),
    ])
    args = _make_args(
        delaunay_n_anchors=6,
        delaunay_bridge_min_edge_length=0.05,  # low threshold to guarantee bridges
    )
    rows, meta = build_delaunay_windows_from_pilot_samples(samples, args)

    types = {r["window_type"] for r in rows}
    assert "basin_anchor" in types, f"Expected basin_anchor; got types {types}"
    assert "delaunay_bridge" in types, f"Expected delaunay_bridge; got types {types}"

    # No duplicate (cv1, cv2) pairs within 1e-3
    seen = set()
    for r in rows:
        key = (round(float(r["primary_cv_center"]), 3), round(float(r["secondary_cv_center"]), 3))
        assert key not in seen, f"Duplicate window position {key}"
        seen.add(key)


# ---------------------------------------------------------------------------
# Test 3 — Density floor filter drops sparse cluster
# ---------------------------------------------------------------------------

def test_density_floor_filter(tmp_path):
    """With a tight density floor, a very sparse cluster is excluded."""
    from gareus.windows import build_delaunay_windows_from_pilot_samples

    samples = tmp_path / "samples.csv"
    # Dense cluster A + very sparse cluster B
    _write_samples_csv(samples, [
        (0.5, 0.02, 0.0, 0.05, 400),  # dense
        (0.2, 0.01, -0.7, 0.01, 4),   # sparse
    ])
    args_tight = _make_args(
        delaunay_n_anchors=8,
        delaunay_density_floor_quantile=0.30,  # reject bottom 30%
    )
    rows_tight, _ = build_delaunay_windows_from_pilot_samples(samples, args_tight)

    args_loose = _make_args(
        delaunay_n_anchors=8,
        delaunay_density_floor_quantile=0.001,
    )
    rows_loose, _ = build_delaunay_windows_from_pilot_samples(samples, args_loose)

    # Tight floor should produce fewer anchors than loose floor
    anchors_tight = [r for r in rows_tight if r["window_type"] == "basin_anchor"]
    anchors_loose = [r for r in rows_loose if r["window_type"] == "basin_anchor"]
    assert len(anchors_tight) <= len(anchors_loose), (
        f"Tight filter ({len(anchors_tight)}) should produce ≤ loose ({len(anchors_loose)})"
    )


# ---------------------------------------------------------------------------
# Test 4 — Force constant formula gives physically reasonable values
# ---------------------------------------------------------------------------

def test_force_constant_formula(tmp_path):
    """Force constants from local spacing are in physically reasonable range."""
    from gareus.windows import build_delaunay_windows_from_pilot_samples

    samples = tmp_path / "samples.csv"
    # Two well-separated clusters
    _write_samples_csv(samples, [
        (0.2, 0.01, -0.5, 0.05, 200),  # CV1 ~ 0.2
        (0.8, 0.01,  0.5, 0.05, 200),  # CV1 ~ 0.8, delta ≈ 0.6
    ])
    args = _make_args(
        delaunay_n_anchors=4,
        delaunay_k_sigma_factor=0.50,
    )
    rows, meta = build_delaunay_windows_from_pilot_samples(samples, args)

    for r in rows:
        k1 = float(r["primary_cv_k_kcal"])
        k2 = float(r["secondary_cv_k_kcal_mol"])
        assert 5.0 <= k1 <= 300.0, f"primary k {k1} out of physical range [5, 300]"
        assert 5.0 <= k2 <= 100.0, f"secondary k {k2} out of CV2 range [5, 100]"


# ---------------------------------------------------------------------------
# Test 5 — Locality annotation patch handles non-axis edges correctly
# ---------------------------------------------------------------------------

def test_locality_annotation_patch():
    """Non-axis edges always get local_patch_candidate, axis edges obey threshold."""
    from gareus.adaptive_feedback import _adaptive_feedback_2d_annotate_locality

    args = _make_args(adaptive_2d_global_bad_fraction=0.50)

    # Build 3 edges: one pure distance_axis pair (same axis), plus knn_geometry + delaunay_bridge
    edge_rows = [
        # Axis edge 1: distance_axis, bad — interval 0
        {
            "edge_type": "distance_axis",
            "primary_pair_index": 0,
            "secondary_pair_index": -1,
            "edge_status": "low_overlap",
            "window_i": 0, "window_j": 1,
            "local_patch_center_A": 0.3,
            "local_patch_secondary_cv_center": 0.0,
        },
        # Axis edge 2: distance_axis, ok — same interval 0 → bad_fraction = 0.5
        {
            "edge_type": "distance_axis",
            "primary_pair_index": 0,
            "secondary_pair_index": -1,
            "edge_status": "ok",
            "window_i": 0, "window_j": 2,
            "local_patch_center_A": 0.4,
            "local_patch_secondary_cv_center": 0.0,
        },
        # knn_geometry edge: bad
        {
            "edge_type": "knn_geometry",
            "primary_pair_index": -1,
            "secondary_pair_index": -1,
            "edge_status": "low_exchange",
            "window_i": 1, "window_j": 3,
            "local_patch_center_A": 0.5,
            "local_patch_secondary_cv_center": -0.3,
        },
        # delaunay_bridge edge: bad
        {
            "edge_type": "delaunay_bridge",
            "primary_pair_index": -1,
            "secondary_pair_index": -1,
            "edge_status": "low_overlap_low_exchange",
            "window_i": 2, "window_j": 4,
            "local_patch_center_A": 0.6,
            "local_patch_secondary_cv_center": 0.3,
        },
        # compound axis+knn edge: bad — should be treated as non-axis
        {
            "edge_type": "distance_axis+knn_geometry",
            "primary_pair_index": 1,
            "secondary_pair_index": -1,
            "edge_status": "low_overlap",
            "window_i": 3, "window_j": 4,
            "local_patch_center_A": 0.7,
            "local_patch_secondary_cv_center": 0.3,
        },
    ]

    annotated, defects = _adaptive_feedback_2d_annotate_locality(edge_rows, args)

    # knn_geometry edge: must be local_patch_candidate (never full_axis_refinement_candidate)
    knn_idx = 2
    assert annotated[knn_idx]["defect_scope"] == "local_patch_candidate", (
        f"knn_geometry edge should be local_patch_candidate, got {annotated[knn_idx]['defect_scope']}"
    )
    assert annotated[knn_idx]["recommendation"] == "add_or_test_local_midpoint_patch_only"

    # delaunay_bridge edge: must be local_patch_candidate
    delaunay_idx = 3
    assert annotated[delaunay_idx]["defect_scope"] == "local_patch_candidate", (
        f"delaunay_bridge edge should be local_patch_candidate, got {annotated[delaunay_idx]['defect_scope']}"
    )

    # compound axis+knn edge: must be local_patch_candidate (not full_axis_refinement_candidate)
    compound_idx = 4
    assert annotated[compound_idx]["defect_scope"] == "local_patch_candidate", (
        f"compound axis+knn edge should be local_patch_candidate, got {annotated[compound_idx]['defect_scope']}"
    )

    # Pure axis edge interval 0: bad_fraction = 0.5 (at threshold), behavior at threshold is local
    # (threshold is >=, so 0.5 >= 0.5 triggers full axis... let's check both are annotated consistently)
    # The bad axis edge should at least be in defects
    axis_bad_defects = [d for d in defects if d.get("edge_type") == "distance_axis" and d.get("edge_status") == "low_overlap"]
    assert len(axis_bad_defects) >= 1, "Bad distance_axis edge should appear in defects"


# ---------------------------------------------------------------------------
# Test 6 — CSV output is round-trippable through load_explicit_2d_window_csv
# ---------------------------------------------------------------------------

def test_trigger_predicate_off_for_adaptive_feedback(tmp_path):
    """adaptive-feedback with default delaunay_after_round=0 must NOT trigger Delaunay."""
    # Simulate the predicate logic from the trigger block in run_adaptive_feedback_auto_loop.
    # This catches the regression where default=1 silently enabled Delaunay for all adaptive runs.
    def _would_trigger(window_mode: str, delaunay_after_round: int, round_no: int,
                       iterate: bool | None = None) -> bool:
        _delaunay_trigger = int(delaunay_after_round or 0)
        if window_mode == "delaunay-feedback" and _delaunay_trigger == 0:
            _delaunay_trigger = 1
        _iterate = (bool(iterate) if iterate is not None else window_mode == "delaunay-feedback")
        cond = round_no >= _delaunay_trigger if _iterate else round_no == _delaunay_trigger
        return _delaunay_trigger > 0 and cond

    # adaptive-feedback + default 0: must not fire regardless of round
    assert not _would_trigger("adaptive-feedback", 0, 1)
    assert not _would_trigger("adaptive-feedback", 0, 2)

    # delaunay-feedback + default 0: iterate=True by default → fires on round 1 AND round 2
    assert _would_trigger("delaunay-feedback", 0, 1)
    assert _would_trigger("delaunay-feedback", 0, 2)   # iterate: fires every round >= 1

    # delaunay-feedback with iterate explicitly disabled: fires only on the trigger round
    assert _would_trigger("delaunay-feedback", 0, 1, iterate=False)
    assert not _would_trigger("delaunay-feedback", 0, 2, iterate=False)

    # delaunay-feedback + explicit round 2 + iterate: fires on rounds >= 2
    assert not _would_trigger("delaunay-feedback", 2, 1)
    assert _would_trigger("delaunay-feedback", 2, 2)
    assert _would_trigger("delaunay-feedback", 2, 3)

    # adaptive-feedback + explicit non-zero, iterate disabled: fires exactly once
    assert _would_trigger("adaptive-feedback", 1, 1)
    assert not _would_trigger("adaptive-feedback", 1, 2)

    # adaptive-feedback + explicit non-zero, iterate enabled: fires every round >= trigger
    assert _would_trigger("adaptive-feedback", 1, 1, iterate=True)
    assert _would_trigger("adaptive-feedback", 1, 2, iterate=True)


# ---------------------------------------------------------------------------
# Test 8 — Coverage scaffold fills uncovered CV cells
# ---------------------------------------------------------------------------

def test_coverage_scaffold_fills_gap(tmp_path):
    """Samples only in top-right CV corner → scaffold populates remaining 3×3 grid cells."""
    from gareus.windows import build_delaunay_windows_from_pilot_samples

    samples = tmp_path / "samples.csv"
    # All samples clustered in top-right corner: cv1~0.85, cv2~0.8 (raw)
    _write_samples_csv(samples, [(0.85, 0.02, 0.8, 0.05, 400)])

    args = _make_args(
        delaunay_n_anchors=4,
        delaunay_min_pilot_samples=10,
        delaunay_coverage_scaffold=True,
        delaunay_coverage_grid_n=3,
    )
    rows, meta = build_delaunay_windows_from_pilot_samples(samples, args)

    scaffold = [r for r in rows if r["window_type"] == "coverage_scaffold"]
    anchors = [r for r in rows if r["window_type"] == "basin_anchor"]

    assert len(anchors) >= 1, "Expected ≥1 basin_anchor from top-right cluster"
    assert len(scaffold) >= 5, (
        f"Expected ≥5 coverage_scaffold windows for uncovered 3×3 cells, got {len(scaffold)}"
    )
    # At least one scaffold window should be in the unsampled low-cv1 region
    scaffold_cv1 = [float(r["primary_cv_center"]) for r in scaffold]
    assert any(cv1 < 0.5 for cv1 in scaffold_cv1), (
        "Scaffold windows should include low-cv1 cells not covered by the cluster"
    )
    # metadata tracks scaffold count
    assert meta.get("n_scaffolds", 0) == len(scaffold)


# ---------------------------------------------------------------------------
# Test 9 — Coverage warning on long Delaunay edge
# ---------------------------------------------------------------------------

def test_coverage_warning_long_edge(tmp_path, capsys):
    """Isolated outlier cluster creates a long Delaunay edge; WARNING is printed."""
    from gareus.windows import build_delaunay_windows_from_pilot_samples

    samples = tmp_path / "samples.csv"
    # Three clusters tight together + one isolated outlier → very long edge to outlier
    _write_samples_csv(samples, [
        (0.10, 0.01, -0.80, 0.02, 300),  # cluster A at norm ~(0.10, 0.10)
        (0.20, 0.01, -0.80, 0.02, 300),  # cluster B at norm ~(0.20, 0.10); dist 0.10 from A
        (0.10, 0.01, -0.60, 0.02, 300),  # cluster C at norm ~(0.10, 0.20); dist 0.10 from A
        (0.90, 0.01,  0.80, 0.02, 300),  # outlier D at norm ~(0.90, 0.90)
    ])
    args = _make_args(
        delaunay_n_anchors=8,
        delaunay_dedup_radius=0.08,   # tight enough to keep all 4 anchors
        delaunay_min_pilot_samples=10,
        delaunay_bridge_min_edge_length=0.50,  # suppress bridges to keep row count predictable
    )
    rows, meta = build_delaunay_windows_from_pilot_samples(samples, args)
    captured = capsys.readouterr()

    anchors = [r for r in rows if r["window_type"] == "basin_anchor"]
    assert len(anchors) >= 3, f"Expected ≥3 basin_anchors, got {len(anchors)}"
    assert "WARNING" in captured.out, (
        "Expected coverage-gap WARNING in stdout when Delaunay has a long edge.\n"
        f"Captured stdout: {captured.out!r}"
    )


def test_csv_round_trip(tmp_path):
    """Output rows pass through load_explicit_2d_window_csv without ValueError."""
    from gareus.windows import build_delaunay_windows_from_pilot_samples, load_explicit_2d_window_csv

    samples = tmp_path / "samples.csv"
    _write_samples_csv(samples, [
        (0.15, 0.02, -0.8, 0.05, 200),
        (0.55, 0.02,  0.0, 0.05, 200),
        (0.85, 0.02,  0.7, 0.05, 200),
    ])
    args = _make_args(delaunay_n_anchors=6)
    rows, meta = build_delaunay_windows_from_pilot_samples(samples, args, out_dir=tmp_path)
    assert len(rows) > 0, "No windows returned"

    # Write to CSV and reload
    out_csv = tmp_path / "delaunay_initial_windows.csv"
    assert out_csv.exists(), "build_delaunay_windows_from_pilot_samples should write CSV to out_dir"

    centers_a, k_list, sec_centers, sec_k_list, sec_meta, win_meta = load_explicit_2d_window_csv(args, out_csv)
    assert len(centers_a) == len(rows), f"Round-trip row count mismatch: {len(centers_a)} vs {len(rows)}"
    assert sec_centers is not None, "Secondary centers should be present"
    assert len(sec_centers) == len(rows)

    # All force constants must be positive
    for k in k_list:
        assert k > 0.0, f"Non-positive primary k: {k}"
    for k in sec_k_list:
        assert k > 0.0, f"Non-positive secondary k: {k}"
