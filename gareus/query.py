"""
Parquet-based data loading and MBAR bias reconstruction for GAREUS.

The N×K bias matrix (umbrella_reduced_bias_nk) is reconstructed analytically
from stored CV values and window parameters rather than being persisted:

    U_k(cv) = 0.5 * k1_k * (cv1 - center1_k)^2  [kcal/mol]
            + 0.5 * k2_k * (cv2 - center2_k)^2  [kcal/mol, 2D only]
    reduced_bias[n,k] = beta * 4.184 * U_k(cv[n])

This is 10-100x smaller than storing the full matrix.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .io import read_json_file
from .store import SegmentRegistry


def _concat_numpy_dicts(results: list) -> dict:
    """Concatenate a list of numpy column-dicts into one, sorted by (step, replica)."""
    non_empty = [r for r in results if r and "step" in r and len(r["step"]) > 0]
    if not non_empty:
        return {}
    keys = list(non_empty[0].keys())
    combined = {k: np.concatenate([r[k] for r in non_empty]) for k in keys}
    order = np.lexsort((combined["replica"], combined["step"]))
    return {k: v[order] for k, v in combined.items()}


def load_samples(
    run_dir: Path,
    segment_ids: Optional[list] = None,
) -> dict:
    """Load production samples from Parquet files via DuckDB.

    Uses segments.json to determine which rows are valid:
    - complete: all rows included
    - running (last segment only): all rows included (active run)
    - interrupted: rows with step <= end_step included (end_step = last checkpoint
      absolute_step; rows beyond it are phantom frames from a rolled-back state)
    - abandoned / running (non-last): skipped entirely
    - No segments.json: falls back to reading all Parquet files (legacy)

    segment_ids overrides the registry-based selection when provided.
    """
    import duckdb

    run_dir = Path(run_dir)
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        return {}

    # Explicit override: legacy path or caller-specified selection
    if segment_ids is not None:
        files = sorted(f for f in samples_dir.glob("**/*.parquet")
                       if f.parent.name in segment_ids)
        if not files:
            return {}
        conn = duckdb.connect()
        result = conn.execute(
            "SELECT * FROM read_parquet(?) ORDER BY step, replica",
            [[str(f) for f in files]],
        ).fetchnumpy()
        conn.close()
        return result

    # Registry-aware path
    seg_json = run_dir / "segments.json"
    if not seg_json.exists():
        # Legacy fallback: no registry, read everything
        files = sorted(samples_dir.glob("**/*.parquet"))
        if not files:
            return {}
        conn = duckdb.connect()
        result = conn.execute(
            "SELECT * FROM read_parquet(?) ORDER BY step, replica",
            [[str(f) for f in files]],
        ).fetchnumpy()
        conn.close()
        return result

    segs = json.loads(seg_json.read_text(encoding="utf-8"))
    if not segs:
        return {}

    last_seg_id = segs[-1]["segment_id"]
    unfiltered_files: list = []
    filtered_groups: list = []  # list of (end_step: int, files: list[str])

    for seg in segs:
        seg_id = seg["segment_id"]
        status = seg.get("status", "running")
        seg_dir = samples_dir / seg_id
        if not seg_dir.exists():
            continue
        files = sorted(str(f) for f in seg_dir.glob("*.parquet"))
        if not files:
            continue

        if status == "complete":
            unfiltered_files.extend(files)
        elif status == "running" and seg_id == last_seg_id:
            # Active run — include all rows written so far
            unfiltered_files.extend(files)
        elif status == "interrupted":
            end_step = seg.get("end_step", -1)
            if end_step is not None and int(end_step) >= 0:
                filtered_groups.append((int(end_step), files))
            # end_step=-1 or None: skip (no valid boundary known)
        # abandoned / running non-last: skip

    if not unfiltered_files and not filtered_groups:
        return {}

    conn = duckdb.connect()
    results = []
    if unfiltered_files:
        r = conn.execute(
            "SELECT * FROM read_parquet(?)",
            [unfiltered_files],
        ).fetchnumpy()
        results.append(r)
    for end_step, files in filtered_groups:
        r = conn.execute(
            "SELECT * FROM read_parquet(?) WHERE step <= ?",
            [files, end_step],
        ).fetchnumpy()
        results.append(r)
    conn.close()
    return _concat_numpy_dicts(results)


def load_windows(
    run_dir: Path,
    segment_id: Optional[str] = None,
) -> list:
    """Load window definitions from windows/<segment_id>.json.

    If segment_id is None, uses the latest segment in segments.json.
    Returns list of window dicts sorted by window_id.
    """
    run_dir = Path(run_dir)

    if segment_id is None:
        reg = SegmentRegistry(run_dir)
        latest = reg.get_latest_segment()
        if latest:
            segment_id = latest["segment_id"]

    if segment_id is None:
        return []

    win_path = run_dir / "windows" / f"{segment_id}.json"
    if not win_path.exists():
        return []

    data = json.loads(win_path.read_text(encoding="utf-8"))
    windows = data.get("windows", [])
    return sorted(windows, key=lambda w: int(w.get("window_id", 0)))


def load_windows_metadata(
    run_dir: Path,
    segment_id: Optional[str] = None,
) -> dict:
    """Load full window snapshot (includes cv1_type, cv2_type, segment_id)."""
    run_dir = Path(run_dir)

    if segment_id is None:
        reg = SegmentRegistry(run_dir)
        latest = reg.get_latest_segment()
        if latest:
            segment_id = latest["segment_id"]

    if segment_id is None:
        return {}

    win_path = run_dir / "windows" / f"{segment_id}.json"
    if not win_path.exists():
        return {}

    return json.loads(win_path.read_text(encoding="utf-8"))


def reconstruct_bias_matrix(
    cv_A: np.ndarray,
    cv2: Optional[np.ndarray],
    windows: list,
    beta: float,
) -> np.ndarray:
    """Reconstruct umbrella_reduced_bias_nk analytically.

    Parameters
    ----------
    cv_A : (N,) array of primary CV values
    cv2  : (N,) array of secondary CV values, or None for 1D runs
    windows : list of window dicts with center1, k1 (and optionally center2, k2)
    beta : 1/(kB*T) in mol/kJ (e.g. 1 / (8.314462618e-3 * T_K))

    Returns
    -------
    nk : (N, K) float64 array of dimensionless reduced umbrella biases
    """
    cv_A = np.asarray(cv_A, dtype=np.float64)
    N = len(cv_A)
    K = len(windows)
    nk = np.zeros((N, K), dtype=np.float64)

    for k, w in enumerate(windows):
        d1 = cv_A - float(w["center1"])
        nk[:, k] = 4.184 * 0.5 * float(w["k1"]) * d1 * d1  # kcal → kJ
        if cv2 is not None and "center2" in w and "k2" in w:
            c2 = np.asarray(cv2, dtype=np.float64)
            valid = np.isfinite(c2)
            d2 = np.where(valid, c2 - float(w["center2"]), 0.0)
            nk[:, k] += 4.184 * 0.5 * float(w["k2"]) * d2 * d2

    return beta * nk


def export_analysis_arrays_npz(
    run_dir: Path,
    beta: float,
    out_path: Optional[Path] = None,
) -> Path:
    """Reconstruct and write analysis_arrays.npz from Parquet sample data.

    Provides backward compatibility for downstream tools (gareus analyze,
    pymbar scripts) that expect the legacy NPZ format.

    Parameters
    ----------
    run_dir  : run output directory containing samples/ and windows/
    beta     : 1/(kB*T) in mol/kJ
    out_path : destination path (default: run_dir/analysis_arrays.npz)
    """
    samples = load_samples(run_dir)
    if not samples or "cv1" not in samples:
        raise ValueError(f"No Parquet sample data found in {run_dir}/samples/")

    windows = load_windows(run_dir)
    if not windows:
        raise ValueError(f"No window snapshot found in {run_dir}/windows/")

    cv_A = samples["cv1"].astype(np.float64)
    window = samples["window_id"].astype(np.int32)
    cv2_raw = samples.get("cv2")
    cv2: Optional[np.ndarray] = cv2_raw.astype(np.float64) if cv2_raw is not None else None

    nk = reconstruct_bias_matrix(cv_A, cv2, windows, beta)

    save_kwargs: dict = {
        "cv_A": cv_A,
        "window": window,
        "umbrella_reduced_bias_nk": nk,
    }
    if cv2 is not None:
        save_kwargs["secondary_cv"] = cv2

    out_path = Path(out_path) if out_path is not None else (Path(run_dir) / "analysis_arrays.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save_kwargs)
    return out_path


__all__ = [
    "load_samples",
    "load_windows",
    "load_windows_metadata",
    "reconstruct_bias_matrix",
    "export_analysis_arrays_npz",
]
