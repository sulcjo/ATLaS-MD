"""Tests for gareus.query — Parquet-based data loading and bias reconstruction."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest


# --- Helpers to build fixture data ---

def _write_segment(tmp_path: Path, seg_id: str, n_samples: int = 20,
                   n_windows: int = 4, with_cv2: bool = False) -> tuple[list, list]:
    """Write Parquet samples + window snapshot. Returns (windows, samples_list)."""
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    rng = np.random.default_rng(42)
    centers1 = [float(i) * 0.1 for i in range(n_windows)]
    k1 = 200.0
    centers2 = [float(i) * 0.5 - 0.75 for i in range(n_windows)] if with_cv2 else None
    k2 = 50.0

    windows = [
        {
            "window_id": w,
            "center1": centers1[w],
            "k1": k1,
            **({"center2": centers2[w], "k2": k2} if with_cv2 else {}),
        }
        for w in range(n_windows)
    ]

    reg = SegmentRegistry(tmp_path)
    seg_id_created = reg.open_segment("run_001", None, 1)
    assert seg_id_created == seg_id

    cv2_type = "rama-map" if with_cv2 else None
    WindowSnapshot(tmp_path).snapshot(seg_id, windows, cv1_type="contacts", cv2_type=cv2_type)

    writer = ParquetSampleWriter(tmp_path / "samples" / seg_id, flush_rows=1000)
    samples = []
    for i in range(n_samples):
        w = i % n_windows
        cv1 = centers1[w] + rng.normal(0, 0.02)
        cv2 = (centers2[w] + rng.normal(0, 0.1)) if with_cv2 else None
        pe = -100.0 + rng.normal(0, 5)
        bt = abs(rng.normal(1.0, 0.5))
        bd = bt * 0.6
        bn = bt * 0.4
        writer.write_sample(i * 100, i % 2, w, cv1, cv2, pe, bt, bd, bn)
        samples.append({"step": i * 100, "replica": i % 2, "window_id": w,
                        "cv1": cv1, "cv2": cv2, "potential": pe,
                        "gamd_boost_total": bt})
    writer.close()
    reg.close_segment(seg_id, end_step=(n_samples - 1) * 100)

    return windows, samples


# --- load_samples ---

def test_load_samples_returns_expected_columns(tmp_path):
    from gareus.query import load_samples
    _write_segment(tmp_path, "seg_001")

    result = load_samples(tmp_path)
    expected_cols = {"step", "replica", "window_id", "cv1", "cv2",
                     "potential", "gamd_boost_total", "gamd_boost_dihedral", "gamd_boost_nonbonded"}
    assert expected_cols.issubset(set(result.keys()))


def test_load_samples_correct_row_count(tmp_path):
    from gareus.query import load_samples
    _write_segment(tmp_path, "seg_001", n_samples=30)

    result = load_samples(tmp_path)
    assert len(result["step"]) == 30


def test_load_samples_ordered_by_step(tmp_path):
    from gareus.query import load_samples
    _write_segment(tmp_path, "seg_001", n_samples=20)

    result = load_samples(tmp_path)
    steps = result["step"]
    assert np.all(steps[1:] >= steps[:-1])


def test_load_samples_multi_segment(tmp_path):
    from gareus.query import load_samples
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    # Manually add a second segment
    _write_segment(tmp_path, "seg_001", n_samples=10)

    reg = SegmentRegistry(tmp_path)
    seg2 = reg.open_segment("run_001", "seg_001", 1)
    WindowSnapshot(tmp_path).snapshot(seg2, [{"window_id": 0, "center1": 0.1, "k1": 200.0}],
                                       cv1_type="contacts", cv2_type=None)
    writer = ParquetSampleWriter(tmp_path / "samples" / seg2, flush_rows=1000)
    for i in range(5):
        writer.write_sample(i * 1000, 0, 0, 0.1, None, -100.0, 1.0, 0.6, 0.4)
    writer.close()
    reg.close_segment(seg2, end_step=4000)

    result = load_samples(tmp_path)
    assert len(result["step"]) == 15


def test_load_samples_empty_dir(tmp_path):
    from gareus.query import load_samples
    result = load_samples(tmp_path)
    assert result == {}


# --- load_windows ---

def test_load_windows_returns_correct_count(tmp_path):
    from gareus.query import load_windows
    _write_segment(tmp_path, "seg_001", n_windows=6)

    windows = load_windows(tmp_path)
    assert len(windows) == 6


def test_load_windows_latest_segment(tmp_path):
    from gareus.query import load_windows
    from gareus.store import SegmentRegistry, WindowSnapshot

    _write_segment(tmp_path, "seg_001", n_windows=4)
    reg = SegmentRegistry(tmp_path)
    seg2 = reg.open_segment("run_001", "seg_001", 2)
    WindowSnapshot(tmp_path).snapshot(seg2,
        [{"window_id": i, "center1": float(i)*0.2, "k1": 300.0} for i in range(8)],
        cv1_type="contacts", cv2_type=None)

    windows = load_windows(tmp_path)  # should return seg_002's windows (8 windows)
    assert len(windows) == 8


def test_load_windows_no_data_returns_empty(tmp_path):
    from gareus.query import load_windows
    assert load_windows(tmp_path) == []


def test_load_windows_explicit_segment_id(tmp_path):
    from gareus.query import load_windows
    _write_segment(tmp_path, "seg_001", n_windows=4)

    windows = load_windows(tmp_path, segment_id="seg_001")
    assert len(windows) == 4
    assert windows[0]["window_id"] == 0


# --- reconstruct_bias_matrix ---

def test_reconstruct_bias_matrix_shape(tmp_path):
    from gareus.query import reconstruct_bias_matrix

    N, K = 50, 4
    cv_A = np.linspace(0.0, 0.3, N)
    windows = [{"window_id": k, "center1": k * 0.1, "k1": 200.0} for k in range(K)]
    beta = 1.0 / (8.314462618e-3 * 300.0)

    nk = reconstruct_bias_matrix(cv_A, None, windows, beta)
    assert nk.shape == (N, K)


def test_reconstruct_bias_matrix_1d_formula():
    from gareus.query import reconstruct_bias_matrix

    cv_A = np.array([0.1, 0.2, 0.3])
    windows = [{"window_id": 0, "center1": 0.15, "k1": 100.0}]
    beta = 1.0 / (8.314462618e-3 * 300.0)

    nk = reconstruct_bias_matrix(cv_A, None, windows, beta)
    # Manual: U_0(cv) = 0.5 * 100.0 * (cv - 0.15)^2 kcal/mol → * 4.184 * beta
    expected = beta * 4.184 * 0.5 * 100.0 * (cv_A - 0.15) ** 2
    np.testing.assert_allclose(nk[:, 0], expected, rtol=1e-6)


def test_reconstruct_bias_matrix_2d_formula():
    from gareus.query import reconstruct_bias_matrix

    cv1 = np.array([0.1, 0.2])
    cv2 = np.array([-1.0, -0.5])
    windows = [{"window_id": 0, "center1": 0.15, "k1": 100.0, "center2": -0.8, "k2": 50.0}]
    beta = 1.0 / (8.314462618e-3 * 300.0)

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta)
    u1 = 0.5 * 100.0 * (cv1 - 0.15) ** 2
    u2 = 0.5 * 50.0 * (cv2 - (-0.8)) ** 2
    expected = beta * 4.184 * (u1 + u2)
    np.testing.assert_allclose(nk[:, 0], expected, rtol=1e-6)


def test_reconstruct_bias_matrix_zero_at_center():
    from gareus.query import reconstruct_bias_matrix

    center = 0.25
    cv_A = np.array([center])
    windows = [{"window_id": 0, "center1": center, "k1": 200.0}]
    beta = 1.0 / (8.314462618e-3 * 300.0)

    nk = reconstruct_bias_matrix(cv_A, None, windows, beta)
    assert abs(nk[0, 0]) < 1e-10


def test_reconstruct_bias_matrix_nonnegative():
    from gareus.query import reconstruct_bias_matrix

    cv_A = np.random.default_rng(0).normal(0.1, 0.05, 100)
    windows = [{"window_id": k, "center1": k * 0.1, "k1": 200.0} for k in range(5)]
    beta = 1.0 / (8.314462618e-3 * 300.0)

    nk = reconstruct_bias_matrix(cv_A, None, windows, beta)
    assert np.all(nk >= 0)


# --- export_analysis_arrays_npz ---

def test_export_npz_creates_file(tmp_path):
    from gareus.query import export_analysis_arrays_npz
    _write_segment(tmp_path, "seg_001", n_samples=20, n_windows=4)

    beta = 1.0 / (8.314462618e-3 * 300.0)
    out = export_analysis_arrays_npz(tmp_path, beta)
    assert out.exists()


def test_export_npz_correct_arrays(tmp_path):
    from gareus.query import export_analysis_arrays_npz
    _write_segment(tmp_path, "seg_001", n_samples=20, n_windows=4)

    beta = 1.0 / (8.314462618e-3 * 300.0)
    out = export_analysis_arrays_npz(tmp_path, beta)

    data = np.load(out, allow_pickle=False)
    assert "cv_A" in data.files
    assert "window" in data.files
    assert "umbrella_reduced_bias_nk" in data.files
    assert data["cv_A"].shape == (20,)
    assert data["window"].shape == (20,)
    assert data["umbrella_reduced_bias_nk"].shape == (20, 4)


def test_export_npz_bias_matrix_nonnegative(tmp_path):
    from gareus.query import export_analysis_arrays_npz
    _write_segment(tmp_path, "seg_001", n_samples=30, n_windows=4)

    beta = 1.0 / (8.314462618e-3 * 300.0)
    out = export_analysis_arrays_npz(tmp_path, beta)
    data = np.load(out, allow_pickle=False)
    assert np.all(data["umbrella_reduced_bias_nk"] >= 0)


def test_export_npz_custom_out_path(tmp_path):
    from gareus.query import export_analysis_arrays_npz
    _write_segment(tmp_path, "seg_001", n_samples=10, n_windows=3)

    beta = 1.0 / (8.314462618e-3 * 300.0)
    custom_out = tmp_path / "custom_arrays.npz"
    out = export_analysis_arrays_npz(tmp_path, beta, out_path=custom_out)
    assert out == custom_out
    assert custom_out.exists()


# --- validate_analysis_metadata_readiness Parquet fallback ---

def test_validate_readiness_no_npz_error_when_parquet_present(tmp_path):
    """With Parquet data + gareus_metadata.json, NPZ reconstructed; no 'missing' error."""
    import json
    from gareus.analysis import validate_analysis_metadata_readiness

    _write_segment(tmp_path, "seg_001", n_samples=40, n_windows=4)

    # Write minimal gareus_metadata.json so _ensure_analysis_arrays_npz can get temperature
    meta = {"temperature_K": 300.0}
    (tmp_path / "gareus_metadata.json").write_text(json.dumps(meta))

    # Write minimal umbrella_windows.csv for window validation
    with (tmp_path / "umbrella_windows.csv").open("w") as f:
        f.write("window,distance_center_A,distance_k_kcal_mol_A2\n")
        for i in range(4):
            f.write(f"{i},{i * 0.1},{200.0}\n")

    result = validate_analysis_metadata_readiness(tmp_path)

    # NPZ should have been reconstructed
    assert (tmp_path / "analysis_arrays.npz").exists()
    # No "analysis_arrays.npz missing" error
    npz_errors = [e for e in result.get("errors", []) if "analysis_arrays.npz missing" in e]
    assert npz_errors == [], f"Got unexpected NPZ errors: {npz_errors}"
    assert result.get("n_samples", 0) == 40
