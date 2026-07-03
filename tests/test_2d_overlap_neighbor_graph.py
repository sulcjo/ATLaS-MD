"""Sparse-2D-safe overlap validation for diagnostics.py and cv_discovery.py.

GAREUS windows are stored in a flat list; on sparse explicit-2D grids, flat
index ``i``/``i+1`` are NOT necessarily neighbors in CV space -- true adjacency
is recorded in ``explicit_2d_neighbor_graph.csv`` edges. ``gareus/analysis.py``
already validates sparse 2D runs via that graph (see
``validate_analysis_metadata_readiness``). These tests pin the same behavior
for ``gareus/diagnostics.py::validate_us_mbar_inputs`` and
``gareus/cv_discovery.py::suggest_cvs``, which historically assumed flat
adjacency == CV adjacency and could emit spurious "disconnected"/"weak
overlap" reports on sparse 2D (or miss real gaps).

No OpenMM import: both modules under test are pure numpy/csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from gareus.diagnostics import _hist_overlap_np, validate_us_mbar_inputs
from gareus.cv_discovery import suggest_cvs


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sparse_2d_fixture(tmp_path: Path, *, n_per_window: int = 60) -> dict:
    """Four sparse-2D windows where flat i,i+1 are NOT CV neighbors.

    Windows 0 and 2 sit close together in CV space (true neighbors); windows
    1 and 3 sit close together elsewhere in CV space (true neighbors). But
    flat order is 0,1,2,3, so old flat-adjacency code compares (0,1), (1,2),
    (2,3) -- all of which are far apart and would show ~zero overlap.
    """
    centers = {0: 0.10, 1: 0.90, 2: 0.13, 3: 0.87}
    rng = np.random.default_rng(7)
    cv_vals = []
    win_vals = []
    for w, c in centers.items():
        samples = rng.normal(c, 0.03, n_per_window).clip(0.0, 1.0)
        cv_vals.append(samples)
        win_vals.append(np.full(n_per_window, w, dtype=np.int64))
    cv = np.concatenate(cv_vals)
    win = np.concatenate(win_vals)
    n_samples = cv.size
    n_windows = len(centers)

    np.savez(
        tmp_path / "analysis_arrays.npz",
        cv_A=cv,
        window=win,
        umbrella_reduced_bias_nk=np.zeros((n_samples, n_windows), dtype=float),
    )
    _write_csv(
        tmp_path / "umbrella_windows.csv",
        [{"window": w, "center_A": centers[w]} for w in sorted(centers)],
        ["window", "center_A"],
    )
    _write_csv(
        tmp_path / "umbrella_explicit_windows.csv",
        [
            {"window": w, "primary_center": centers[w], "explicit_2d": 1, "rectangular_grid": 0}
            for w in sorted(centers)
        ],
        ["window", "primary_center", "explicit_2d", "rectangular_grid"],
    )
    # True CV-space neighbors: (0,2) and (1,3). Flat neighbors (0,1),(1,2),(2,3)
    # are deliberately absent from the graph.
    _write_csv(
        tmp_path / "explicit_2d_neighbor_graph.csv",
        [
            {"window_i": 0, "window_j": 2, "edge_type": "primary_neighbor"},
            {"window_i": 1, "window_j": 3, "edge_type": "primary_neighbor"},
        ],
        ["window_i", "window_j", "edge_type"],
    )
    return {"cv": cv, "win": win, "centers": centers}


def _rectangular_1d_fixture_with_real_gap(tmp_path: Path, *, n_per_window: int = 60) -> None:
    """Three 1D windows where flat i,i+1 ARE the true neighbors and window 1->2
    has a genuine sampling gap (should still be flagged)."""
    centers = {0: 0.0, 1: 0.5, 2: 5.0}
    rng = np.random.default_rng(11)
    cv_vals = []
    win_vals = []
    for w, c in centers.items():
        samples = rng.normal(c, 0.15, n_per_window)
        cv_vals.append(samples)
        win_vals.append(np.full(n_per_window, w, dtype=np.int64))
    cv = np.concatenate(cv_vals)
    win = np.concatenate(win_vals)
    n_samples = cv.size
    n_windows = len(centers)

    np.savez(
        tmp_path / "analysis_arrays.npz",
        cv_A=cv,
        window=win,
        umbrella_reduced_bias_nk=np.zeros((n_samples, n_windows), dtype=float),
    )
    _write_csv(
        tmp_path / "umbrella_windows.csv",
        [{"window": w, "center_A": centers[w]} for w in sorted(centers)],
        ["window", "center_A"],
    )
    # No umbrella_explicit_windows.csv / neighbor graph: this is a plain 1D run.


# ---------------------------------------------------------------------------
# gareus/diagnostics.py::validate_us_mbar_inputs
# ---------------------------------------------------------------------------

def test_flat_adjacency_would_be_spurious_control_check(tmp_path):
    """Sanity control: prove the sparse-2D fixture really would fool flat i,i+1."""
    data = _sparse_2d_fixture(tmp_path)
    cv, win, centers = data["cv"], data["win"], data["centers"]
    lo, hi = min(centers.values()), max(centers.values())
    flat_overlaps = []
    for i in range(3):
        a = cv[win == i]
        b = cv[win == i + 1]
        flat_overlaps.append(_hist_overlap_np(a, b, lo, hi, bins=80))
    # All flat neighbor pairs are far apart in CV space -> ~zero overlap.
    assert all(ov < 0.03 for ov in flat_overlaps), flat_overlaps


def test_validate_us_mbar_inputs_sparse_2d_uses_graph_neighbors_not_flat(tmp_path):
    _sparse_2d_fixture(tmp_path)

    result = validate_us_mbar_inputs(tmp_path, temperature_k=300.0, target_overlap=0.30)

    assert result["sparse_2d"] is True
    # No spurious disconnection from comparing non-adjacent flat windows.
    assert not any("disconnected" in e for e in result["errors"]), result["errors"]
    assert result["status"] != "error", result

    pairs = {(r["left_window"], r["right_window"]) for r in result["neighbor_overlaps"]}
    # Must compare graph-true neighbors...
    assert (0, 2) in pairs
    assert (1, 3) in pairs
    # ...and must NOT compare flat non-neighbors.
    assert (0, 1) not in pairs
    assert (1, 2) not in pairs
    assert (2, 3) not in pairs

    # The true graph-neighbor pairs overlap well, so no weak-overlap warning.
    weak_pairs = {(r["left_window"], r["right_window"]) for r in result["neighbor_pairs_below_target_overlap"]}
    assert not weak_pairs, result["neighbor_pairs_below_target_overlap"]


def test_validate_us_mbar_inputs_1d_flat_neighbors_unchanged_real_gap_detected(tmp_path):
    _rectangular_1d_fixture_with_real_gap(tmp_path)

    result = validate_us_mbar_inputs(tmp_path, temperature_k=300.0, target_overlap=0.30)

    assert result["sparse_2d"] is False
    pairs = {(r["left_window"], r["right_window"]) for r in result["neighbor_overlaps"]}
    assert pairs == {(0, 1), (1, 2)}
    # Real gap between window 1 (center 0.5) and window 2 (center 5.0) must
    # still be caught -- 1D/rectangular behavior is unchanged.
    assert any("disconnected" in e for e in result["errors"]), result["errors"]
    assert result["overlap_connected_at_0p03"] is False


# ---------------------------------------------------------------------------
# gareus/cv_discovery.py::suggest_cvs
# ---------------------------------------------------------------------------

def test_suggest_cvs_sparse_2d_no_spurious_weak_overlap_suggestion(tmp_path):
    _sparse_2d_fixture(tmp_path)

    payload = suggest_cvs(tmp_path, target_overlap=0.30)

    titles = [s["title"] for s in payload["suggestions"]]
    assert not any("neighbor overlap is weak" in t.lower() for t in titles), titles


def test_suggest_cvs_1d_flat_neighbors_still_flagged(tmp_path):
    _rectangular_1d_fixture_with_real_gap(tmp_path)

    payload = suggest_cvs(tmp_path, target_overlap=0.30)

    titles = [s["title"] for s in payload["suggestions"]]
    assert any("neighbor overlap is weak" in t.lower() for t in titles), titles
