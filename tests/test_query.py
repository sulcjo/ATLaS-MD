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


def _write_parquet_rows(seg_dir, steps):
    """Write one Parquet file with rows at the given step values (no registry update)."""
    from gareus.store import ParquetSampleWriter
    writer = ParquetSampleWriter(seg_dir, flush_rows=10000)
    for s in steps:
        writer.write_sample(s, 0, 0, 0.1, None, -100.0, 1.0, 0.5, 0.5)
    writer.close()


def test_load_samples_skips_abandoned_segment(tmp_path):
    """Abandoned segments (crashed before first checkpoint) must not appear in results."""
    from gareus.query import load_samples
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg1 = reg.open_segment("run_001", None, 1)
    _write_parquet_rows(tmp_path / "samples" / seg1, [100, 200, 300])
    reg.seal_segment(seg1, absolute_end_step=-1, status="abandoned")

    result = load_samples(tmp_path)
    assert result == {}


def test_load_samples_filters_interrupted_rows(tmp_path):
    """Interrupted segment: only rows with step <= end_step are included."""
    from gareus.query import load_samples
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg1 = reg.open_segment("run_001", None, 1)
    # steps 100..1000; checkpoint was at absolute_step=600
    _write_parquet_rows(tmp_path / "samples" / seg1, list(range(100, 1100, 100)))
    reg.seal_segment(seg1, absolute_end_step=600, status="interrupted")

    result = load_samples(tmp_path)
    assert set(result["step"].tolist()) == {100, 200, 300, 400, 500, 600}


def test_load_samples_includes_last_running_segment(tmp_path):
    """The last (and only) running segment is an active run — all its rows must appear."""
    from gareus.query import load_samples
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg1 = reg.open_segment("run_001", None, 1)
    _write_parquet_rows(tmp_path / "samples" / seg1, [100, 200, 300])
    # No close — segment stays "running"

    result = load_samples(tmp_path)
    assert len(result["step"]) == 3


def test_load_samples_skips_non_last_running_segment(tmp_path):
    """A running segment that is NOT the last is a crashed run — skip it."""
    from gareus.query import load_samples
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    # seg_001: running (crashed, not sealed)
    seg1 = reg.open_segment("run_001", None, 1)
    _write_parquet_rows(tmp_path / "samples" / seg1, [100, 200, 300])
    # seg_002: complete (new run after crash)
    seg2 = reg.open_segment("run_001", seg1, 1)
    _write_parquet_rows(tmp_path / "samples" / seg2, [700, 800, 900])
    reg.close_segment(seg2, end_step=900)

    result = load_samples(tmp_path)
    assert set(result["step"].tolist()) == {700, 800, 900}


def test_load_samples_no_duplicates_after_crash_restart(tmp_path):
    """Crash-restart scenario: seg_001 interrupted at step 500; seg_002 starts at 600.
    Combined result must have no rows with step > 500 from seg_001."""
    from gareus.query import load_samples
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    # seg_001 wrote steps 100–900 but checkpoint was at absolute_step=500
    seg1 = reg.open_segment("run_001", None, 1)
    _write_parquet_rows(tmp_path / "samples" / seg1, list(range(100, 1000, 100)))
    reg.seal_segment(seg1, absolute_end_step=500, status="interrupted")

    # seg_002 continues from step 600 (next log interval after checkpoint 500)
    seg2 = reg.open_segment("run_001", seg1, 1)
    _write_parquet_rows(tmp_path / "samples" / seg2, list(range(600, 1600, 100)))
    reg.close_segment(seg2, end_step=1500)

    result = load_samples(tmp_path)
    steps = sorted(result["step"].tolist())
    # seg_001 contributes 100–500; seg_002 contributes 600–1500
    assert max(result["step"][result["step"] <= 500]) <= 500
    assert steps == list(range(100, 600, 100)) + list(range(600, 1600, 100))
    # Monotonically increasing — no duplicates
    assert steps == sorted(set(steps))


def test_load_samples_no_segments_json_fallback(tmp_path):
    """Without segments.json, all Parquet files are loaded (legacy behavior)."""
    from gareus.query import load_samples

    seg_dir = tmp_path / "samples" / "seg_001"
    _write_parquet_rows(seg_dir, [100, 200, 300])

    result = load_samples(tmp_path)
    assert len(result["step"]) == 3


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


def test_export_npz_uses_each_segment_window_snapshot(tmp_path):
    """Rows from old segments must use old window centers, not latest snapshot."""
    from gareus.query import export_analysis_arrays_npz
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    reg = SegmentRegistry(tmp_path)
    snap = WindowSnapshot(tmp_path)

    seg1 = reg.open_segment("run_001", None, 1)
    snap.snapshot(seg1, [{"window_id": 0, "center1": 0.0, "k1": 100.0}], cv1_type="contacts", cv2_type=None)
    writer1 = ParquetSampleWriter(tmp_path / "samples" / seg1, flush_rows=10)
    writer1.write_sample(100, 0, 0, 0.0, None, -100.0, 0.0, 0.0, 0.0)
    writer1.close()
    reg.close_segment(seg1, end_step=100)

    seg2 = reg.open_segment("run_001", seg1, 1)
    snap.snapshot(seg2, [{"window_id": 0, "center1": 1.0, "k1": 100.0}], cv1_type="contacts", cv2_type=None)
    writer2 = ParquetSampleWriter(tmp_path / "samples" / seg2, flush_rows=10)
    writer2.write_sample(200, 0, 0, 1.0, None, -100.0, 0.0, 0.0, 0.0)
    writer2.close()
    reg.close_segment(seg2, end_step=200)

    beta = 1.0 / (8.314462618e-3 * 300.0)
    out = export_analysis_arrays_npz(tmp_path, beta)

    with np.load(out, allow_pickle=False) as data:
        np.testing.assert_allclose(data["umbrella_reduced_bias_nk"][:, 0], [0.0, 0.0], atol=1.0e-12)


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


# --- export_analysis_arrays_npz: lambda-ladder envelope resolution ----------
# Regression pair for the collateral defect the fix wave introduced and then
# fixed: the envelope must be resolved ONLY when the loaded snapshot actually
# carries a lambda>0 window.

def _ladder_globals_payload(env):
    return {"all_globals": {
        "Vmax_Total": env.vmax_total, "Vmin_Total": env.vmin_total,
        "threshold_energy_Total": env.threshold_total, "k0_Total": env.k0max_total,
        "Vmax_Dihedral": env.vmax_dih, "Vmin_Dihedral": env.vmin_dih,
        "threshold_energy_Dihedral": env.threshold_dih, "k0_Dihedral": env.k0max_dih}}


def _write_ladder_segment(prod, lambdas, with_globals):
    """Parquet run whose window snapshot carries `lambdas` and whose samples
    carry real v_pep/v_dih. `with_globals` is the JSON payload to write as
    shared_gamd_setup_globals.json, or None to write no file at all."""
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    prod.mkdir(parents=True, exist_ok=True)
    windows = [{"window_id": i, "center1": 5.0, "k1": 10.0, "gamd_lambda": float(lam)}
               for i, lam in enumerate(lambdas)]
    reg = SegmentRegistry(prod)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(prod).snapshot(seg_id, windows, cv1_type="distance", cv2_type=None)

    writer = ParquetSampleWriter(prod / "samples" / seg_id, flush_rows=1000)
    for i in range(6):
        writer.write_sample(i * 10, 0, i % len(windows), 5.0 + 0.01 * i, None,
                            -100.0, 1.0, 0.6, 0.4,
                            v_pep=10.0 + i, v_dih=3.0 + 0.5 * i,
                            gamd_lambda=float(lambdas[i % len(lambdas)]))
    writer.close()
    reg.close_segment(seg_id, end_step=60)
    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    if with_globals is not None:
        (prod / "shared_gamd_setup_globals.json").write_text(json.dumps(with_globals))
    return windows


def test_export_npz_non_gamd_run_with_empty_globals_still_exports(tmp_path):
    """A GaMD-DISABLED run still writes shared_gamd_setup_globals.json, with
    an EMPTY "all_globals" dict (gareus/production.py's disabled-run writer),
    and PepGamdEnvelope.from_json raises KeyError when no nested dict holds
    k0_Total. Resolving the envelope unconditionally therefore broke every
    plain non-GaMD parquet run -- and gareus/analysis.py wraps this call in
    `except Exception: pass`, so the auto-generation path degraded silently to
    'analysis_arrays.npz still absent'."""
    from gareus.query import export_analysis_arrays_npz

    prod = tmp_path / "final_production"
    _write_ladder_segment(prod, [0.0, 0.0], with_globals={
        "mode": "disabled_cmd", "all_globals": {}, "interesting_globals": {}})

    beta = 1.0 / (8.314462618e-3 * 300.0)
    out = export_analysis_arrays_npz(prod, beta)
    assert out.exists()
    with np.load(out, allow_pickle=False) as data:
        nk = np.asarray(data["umbrella_reduced_bias_nk"], dtype=float)
    assert nk.shape[1] == 2
    # Both windows share centre/k and neither carries a rung, so the two
    # columns are the pure-umbrella bias and identical.
    assert np.allclose(nk[:, 0], nk[:, 1])


def test_export_npz_ladder_run_applies_the_boost(tmp_path):
    """The other half: with a real lambda>0 rung and a valid envelope, the
    boost must actually be applied -- the guard must not have turned into
    'never resolve the envelope'."""
    from gareus.query import export_analysis_arrays_npz
    from gareus.pep_gamd import PepGamdEnvelope, pep_gamd_boost_kj

    env = PepGamdEnvelope(50.0, -50.0, 50.0, 0.8, 50.0, -50.0, 50.0, 0.6)
    prod = tmp_path / "final_production"
    _write_ladder_segment(prod, [0.0, 1.0], with_globals=_ladder_globals_payload(env))

    beta = 1.0 / (8.314462618e-3 * 300.0)
    out = export_analysis_arrays_npz(prod, beta)
    with np.load(out, allow_pickle=False) as data:
        nk = np.asarray(data["umbrella_reduced_bias_nk"], dtype=float)

    v_pep = np.array([10.0 + i for i in range(nk.shape[0])])
    v_dih = np.array([3.0 + 0.5 * i for i in range(nk.shape[0])])
    expected = np.array([beta * pep_gamd_boost_kj(vp, vd, 1.0, env)
                         for vp, vd in zip(v_pep, v_dih)])
    assert np.any(expected > 0.0), expected
    assert np.allclose(nk[:, 1] - nk[:, 0], expected), (nk[:, 1] - nk[:, 0], expected)


def test_export_npz_ladder_run_without_an_envelope_still_raises(tmp_path):
    """The guard must NOT have become a bare try/except: on a real ladder run
    a missing envelope has to surface, not silently produce an umbrella-only
    matrix."""
    from gareus.query import export_analysis_arrays_npz

    prod = tmp_path / "final_production"
    _write_ladder_segment(prod, [0.0, 1.0], with_globals=None)

    beta = 1.0 / (8.314462618e-3 * 300.0)
    try:
        export_analysis_arrays_npz(prod, beta)
    except ValueError as exc:
        assert "v_pep" in str(exc) or "envelope" in str(exc), str(exc)
    else:
        raise AssertionError("a lambda>0 run with no envelope must not export silently")
