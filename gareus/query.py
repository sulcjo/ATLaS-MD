"""
Parquet-based data loading and MBAR bias reconstruction for GAREUS.

The N×K bias matrix (umbrella_reduced_bias_nk) is reconstructed analytically
from stored CV values and window parameters rather than being persisted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .parquet_manifest import committed_files
from .store import SegmentRegistry


def _missing_column_placeholder(reference, length: int):
    """Build a fully-masked stand-in for a column absent from one segment."""
    ref = np.ma.asarray(reference)
    dtype = ref.dtype
    if np.issubdtype(dtype, np.floating):
        data = np.full(length, np.nan, dtype=dtype)
    elif np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        data = np.zeros(length, dtype=dtype)
    else:
        data = np.full(length, None, dtype=object)
    return np.ma.array(data, mask=np.ones(length, dtype=bool))


def _concat_numpy_dicts(results: list) -> dict:
    """Concatenate column dictionaries, preserving masks and schema evolution."""
    non_empty = [r for r in results if r and "step" in r and len(r["step"]) > 0]
    if not non_empty:
        return {}

    for r in non_empty:
        seg_len = len(r["step"])
        for k, v in r.items():
            if len(v) != seg_len:
                raise ValueError(
                    f"segment column {k!r} has length {len(v)}, expected {seg_len} "
                    "(inferred from 'step'); ragged columns within one segment"
                )

    keys: list = []
    seen: set = set()
    first_col_for_key: dict = {}
    for r in non_empty:
        for k, v in r.items():
            if k not in seen:
                seen.add(k)
                keys.append(k)
                first_col_for_key[k] = v

    combined = {}
    for k in keys:
        cols = [
            r[k] if k in r else _missing_column_placeholder(first_col_for_key[k], len(r["step"]))
            for r in non_empty
        ]
        if any(np.ma.isMaskedArray(c) for c in cols):
            combined[k] = np.ma.concatenate(cols)
        else:
            combined[k] = np.concatenate(cols)
    if "replica" in combined:
        order = np.lexsort((combined["replica"], combined["step"]))
    elif "replica_i" in combined:
        order = np.lexsort((combined["replica_i"], combined["step"]))
    else:
        order = np.argsort(combined["step"])
    return {k: v[order] for k, v in combined.items()}


def _result_len(result: dict) -> int:
    if not result or "step" not in result:
        return 0
    return int(len(result["step"]))


def _files_for_segment(segment_dir: Path, kind: str) -> list[str]:
    """Return the authoritative Parquet file set for one segment.

    New-format segments are manifest-backed.  Legacy segments without a
    manifest retain the historical glob behavior for read-only compatibility.
    A present-but-invalid manifest raises in ``committed_files`` and MUST NOT
    fall back to directory globbing.
    """
    files = committed_files(segment_dir, expected_kind=kind, verify_hashes=False)
    if files is not None:
        return files
    return sorted(str(path) for path in Path(segment_dir).glob("*.parquet"))


def _group_parquet_files_by_segment(
    data_dir: Path,
    allowed_segments: Optional[set[str]] = None,
    *,
    kind: Optional[str] = None,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    data_dir = Path(data_dir)
    expected_kind = str(kind or data_dir.name)
    for segment_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        seg_id = str(segment_dir.name)
        if allowed_segments is not None and seg_id not in allowed_segments:
            continue
        files = _files_for_segment(segment_dir, expected_kind)
        if files:
            groups[seg_id] = files
    return groups


def _read_parquet_segment(conn, files: list[str], segment_id: str, end_step: Optional[int] = None) -> dict:
    if not files:
        return {}
    if end_step is None:
        result = conn.execute(
            "SELECT * FROM read_parquet(?)",
            [files],
        ).fetchnumpy()
    else:
        result = conn.execute(
            "SELECT * FROM read_parquet(?) WHERE step <= ?",
            [files, int(end_step)],
        ).fetchnumpy()
    n = _result_len(result)
    if n <= 0:
        return {}
    result["segment_id"] = np.asarray([str(segment_id)] * n, dtype=object)
    return result


def _load_segmented_parquet(
    run_dir: Path,
    dirname: str,
    segment_ids: Optional[list] = None,
    n_threads: int = 0,
) -> dict:
    import duckdb

    run_dir = Path(run_dir)
    data_dir = run_dir / dirname
    if not data_dir.exists():
        return {}

    conn = duckdb.connect()
    if n_threads > 0:
        conn.execute(f"SET threads={n_threads}")
    results = []
    try:
        if segment_ids is not None:
            allowed = {str(x) for x in segment_ids}
            groups = _group_parquet_files_by_segment(
                data_dir, allowed, kind=dirname
            )
            for seg_id, files in sorted(groups.items()):
                results.append(_read_parquet_segment(conn, files, seg_id))
            return _concat_numpy_dicts(results)

        seg_json = run_dir / "segments.json"
        if not seg_json.exists():
            groups = _group_parquet_files_by_segment(data_dir, kind=dirname)
            for seg_id, files in sorted(groups.items()):
                results.append(_read_parquet_segment(conn, files, seg_id))
            return _concat_numpy_dicts(results)

        segs = json.loads(seg_json.read_text(encoding="utf-8"))
        if not segs:
            return {}
        last_seg_id = str(segs[-1]["segment_id"])
        for seg in segs:
            seg_id = str(seg["segment_id"])
            seg_dir = data_dir / seg_id
            if not seg_dir.exists():
                continue
            files = _files_for_segment(seg_dir, dirname)
            if not files:
                continue
            status = str(seg.get("status", "running"))
            if status == "complete":
                results.append(_read_parquet_segment(conn, files, seg_id))
            elif status == "running" and seg_id == last_seg_id:
                results.append(_read_parquet_segment(conn, files, seg_id))
            elif status == "interrupted":
                end_step = seg.get("end_step", -1)
                if end_step is not None and int(end_step) >= 0:
                    results.append(
                        _read_parquet_segment(conn, files, seg_id, int(end_step))
                    )
        return _concat_numpy_dicts(results)
    finally:
        conn.close()


def load_samples(
    run_dir: Path,
    segment_ids: Optional[list] = None,
    n_threads: int = 0,
) -> dict:
    """Load production samples from Parquet files via DuckDB.

    Manifest-backed segments use only their atomically committed files. Legacy
    segments without a manifest retain glob-based read compatibility.
    """
    run_dir = Path(run_dir)
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        return {}
    return _load_segmented_parquet(
        run_dir, "samples", segment_ids=segment_ids, n_threads=n_threads
    )


def load_exchanges(
    run_dir: Path,
    segment_ids: Optional[list] = None,
    n_threads: int = 0,
) -> dict:
    """Load exchange events from Parquet with same segment filtering as samples."""
    run_dir = Path(run_dir)
    exchanges_dir = run_dir / "exchanges"
    if not exchanges_dir.exists():
        return {}
    return _load_segmented_parquet(
        run_dir, "exchanges", segment_ids=segment_ids, n_threads=n_threads
    )


def load_windows(
    run_dir: Path,
    segment_id: Optional[str] = None,
) -> list:
    """Load window definitions from windows/<segment_id>.json."""
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
    *,
    v_pep: Optional[np.ndarray] = None,
    v_dih: Optional[np.ndarray] = None,
    envelope=None,
    meta: Optional[dict] = None,
) -> np.ndarray:
    """Strict reduced umbrella-plus-ladder bias; missing required coordinates raise.

    k1/k2 stay in kcal/mol per squared CV unit. beta stays in mol/kJ.
    The existing ladder helper remains the only boost implementation.
    """
    from .correctness.bias import reconstruct_bias_matrix as _strict_bias
    return _strict_bias(
        cv_A,
        cv2,
        windows,
        beta,
        v_pep=v_pep,
        v_dih=v_dih,
        envelope=envelope,
        meta=meta,
    )


def export_analysis_arrays_npz(
    run_dir: Path,
    beta: float,
    out_path: Optional[Path] = None,
) -> Path:
    """Reconstruct and write analysis_arrays.npz from Parquet sample data."""
    samples = load_samples(run_dir)
    if not samples or "cv1" not in samples:
        raise ValueError(f"No Parquet sample data found in {run_dir}/samples/")

    cv_A = samples["cv1"].astype(np.float64)
    window = samples["window_id"].astype(np.int32)
    cv2_raw = samples.get("cv2")
    if cv2_raw is not None:
        if np.ma.isMaskedArray(cv2_raw):
            cv2 = np.asarray(cv2_raw.filled(np.nan), dtype=np.float64)
        else:
            cv2 = np.asarray(cv2_raw, dtype=np.float64)
    else:
        cv2 = None

    def _energy_col(name):
        raw = samples.get(name)
        if raw is None:
            return None
        arr = np.ma.filled(np.ma.asarray(raw).astype(np.float64), np.nan)
        return np.asarray(arr, dtype=np.float64)

    v_pep_all = _energy_col("v_pep_kj_mol")
    v_dih_all = _energy_col("v_dih_kj_mol")
    _envelope_cache: list = []

    def _envelope_for(windows):
        if not any(
            float(w.get("gamd_lambda", 0.0) or 0.0) > 0.0 for w in windows
        ):
            return None
        if not _envelope_cache:
            from .mbar_analysis.ladder import load_pep_gamd_envelope
            _envelope_cache.append(load_pep_gamd_envelope(run_dir))
        return _envelope_cache[0]

    seg_raw = samples.get("segment_id")
    if seg_raw is not None:
        seg_ids = np.asarray(seg_raw).astype(str)
        first_windows = load_windows(run_dir, segment_id=str(seg_ids[0]))
        if not first_windows:
            raise ValueError(f"No window snapshot found in {run_dir}/windows/")
        nk = np.empty((len(cv_A), len(first_windows)), dtype=np.float64)
        expected_ids = [
            int(w.get("window_id", i)) for i, w in enumerate(first_windows)
        ]
        for seg_id in np.unique(seg_ids):
            seg_windows = load_windows(run_dir, segment_id=str(seg_id))
            if not seg_windows:
                raise ValueError(
                    f"No window snapshot found for segment {seg_id} in {run_dir}/windows/"
                )
            seg_ids_list = [
                int(w.get("window_id", i)) for i, w in enumerate(seg_windows)
            ]
            if len(seg_windows) != len(first_windows) or seg_ids_list != expected_ids:
                raise ValueError(
                    "Cannot write one legacy analysis_arrays.npz for segments with "
                    "different window IDs/counts; use segment-specific or union-state analysis."
                )
            mask = seg_ids == str(seg_id)
            nk[mask, :] = reconstruct_bias_matrix(
                cv_A[mask],
                cv2[mask] if cv2 is not None else None,
                seg_windows,
                beta,
                v_pep=v_pep_all[mask] if v_pep_all is not None else None,
                v_dih=v_dih_all[mask] if v_dih_all is not None else None,
                envelope=_envelope_for(seg_windows),
            )
        windows = first_windows
    else:
        windows = load_windows(run_dir)
        if not windows:
            raise ValueError(f"No window snapshot found in {run_dir}/windows/")
        nk = reconstruct_bias_matrix(
            cv_A,
            cv2,
            windows,
            beta,
            v_pep=v_pep_all,
            v_dih=v_dih_all,
            envelope=_envelope_for(windows),
        )

    save_kwargs: dict = {
        "cv_A": cv_A,
        "window": window,
        "umbrella_reduced_bias_nk": nk,
    }
    if seg_raw is not None:
        save_kwargs["segment_id"] = np.asarray(seg_raw).astype(str)
    if cv2 is not None:
        save_kwargs["secondary_cv"] = cv2

    out_path = (
        Path(out_path)
        if out_path is not None
        else (Path(run_dir) / "analysis_arrays.npz")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **save_kwargs)
    return out_path


__all__ = [
    "load_samples",
    "load_exchanges",
    "load_windows",
    "load_windows_metadata",
    "reconstruct_bias_matrix",
    "export_analysis_arrays_npz",
]
