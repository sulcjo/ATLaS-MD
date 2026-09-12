"""
Parquet-based data loading and MBAR bias reconstruction for GAREUS.

The N×K bias matrix (umbrella_reduced_bias_nk) is reconstructed analytically
from stored CV values and window parameters rather than being persisted:

    U_k(cv) = 0.5 * k1_k * (cv1 - center1_k)^2  [kcal/mol]
            + 0.5 * k2_k * (cv2 - center2_k)^2  [kcal/mol, only when window k
                                                  has isfinite(center2_k) and
                                                  isfinite(k2_k) and k2_k > 0]
    reduced_bias[n,k] = beta * KJ_PER_KCAL * U_k(cv[n])

A sample whose own cv2 is non-finite gets NaN for any window that DOES
restrain CV2 (exclusion-by-propagation), rather than a fabricated zero
deviation -- see reconstruct_bias_matrix's docstring.

This is 10-100x smaller than storing the full matrix.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .io import read_json_file
from .store import SegmentRegistry
from .units import KJ_PER_KCAL


def _missing_column_placeholder(reference, length: int):
    """Build a fully-masked stand-in for a column absent from one segment.

    A production campaign can span a schema change (e.g. a new sample column
    added mid-campaign, such as v_pep_kj_mol/gamd_lambda) -- an older segment's
    Parquet files simply lack the column, so it is absent from that segment's
    dict entirely (not merely null-valued). Masked here rather than fabricating
    a real-looking value (0, "", etc.), mirroring how DuckDB itself represents
    a real SQL NULL and how downstream consumers already expect to unmask it
    (see gareus/mbar_analysis/data.py's _fill_masked_nan).
    """
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
    """Concatenate a list of numpy column-dicts into one, sorted by (step, replica).

    DuckDB's fetchnumpy() returns a numpy.ma.MaskedArray for any column with a
    real SQL NULL (e.g. an unmeasured secondary CV). Plain np.concatenate
    preserves the MaskedArray *subclass* on its output but silently drops the
    *mask itself* -- a well-known numpy.ma gotcha, reproduced directly even
    for a single-element input list (i.e. every single-segment run hits this,
    not just multi-segment concatenation). That corrupted every null entry
    into its arbitrary underlying fill value with mask=False ("not null"),
    upstream of and regardless of any unmasking callers do afterwards (see
    gareus/mbar_analysis/data.py's _fill_masked_nan and this module's own
    export_analysis_arrays_npz). Use np.ma.concatenate for any column DuckDB
    actually returned as masked; plain columns keep the cheaper np.concatenate
    (avoids allocating a mask array for step/window_id/replica/segment_id on
    the multi-million-row hot path).

    The key set is the UNION of keys across all segments, not just the first
    segment's keys: a campaign resumed across a schema change (a new sample
    column, e.g. the lambda-ladder's v_pep_kj_mol/v_dih_kj_mol/gamd_lambda)
    mixes older segments that lack the column with newer ones that have it.
    Taking only results[0]'s keys either silently dropped the new column
    (older-segment-first, the typical oldest-first order out of segments.json)
    or crashed with a bare KeyError (newer-segment-first) -- neither is
    acceptable. A segment missing a key present elsewhere gets a fully-masked
    placeholder of its own length instead, so the column is present end-to-end
    and every row from the segment that lacks it reads as NaN/null, never a
    fabricated real value.
    """
    non_empty = [r for r in results if r and "step" in r and len(r["step"]) > 0]
    if not non_empty:
        return {}

    # Every column actually present in a segment must share that segment's
    # sample count (the 'step' column's length) -- a mismatch means a real
    # write/read bug produced ragged columns within one segment, which must
    # fail loudly rather than propagate into a corrupted concatenation.
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


def _group_parquet_files_by_segment(data_dir: Path, allowed_segments: Optional[set[str]] = None) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in sorted(Path(data_dir).glob("**/*.parquet")):
        seg_id = str(path.parent.name)
        if allowed_segments is not None and seg_id not in allowed_segments:
            continue
        groups.setdefault(seg_id, []).append(str(path))
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
            groups = _group_parquet_files_by_segment(data_dir, allowed)
            for seg_id, files in sorted(groups.items()):
                results.append(_read_parquet_segment(conn, files, seg_id))
            return _concat_numpy_dicts(results)

        seg_json = run_dir / "segments.json"
        if not seg_json.exists():
            groups = _group_parquet_files_by_segment(data_dir)
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
            files = sorted(str(f) for f in seg_dir.glob("*.parquet"))
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
                    results.append(_read_parquet_segment(conn, files, seg_id, int(end_step)))
        return _concat_numpy_dicts(results)
    finally:
        conn.close()


def load_samples(
    run_dir: Path,
    segment_ids: Optional[list] = None,
    n_threads: int = 0,
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
    run_dir = Path(run_dir)
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        return {}
    return _load_segmented_parquet(run_dir, "samples", segment_ids=segment_ids, n_threads=n_threads)


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
    return _load_segmented_parquet(run_dir, "exchanges", segment_ids=segment_ids, n_threads=n_threads)


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
    return _strict_bias(cv_A, cv2, windows, beta,
                        v_pep=v_pep, v_dih=v_dih, envelope=envelope, meta=meta)


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

    # λ-ladder plumbing: window snapshots carry gamd_lambda since
    # production.snapshot_window_rows, and reconstruct_bias_matrix REFUSES a
    # λ>0 window without the raw channel energies + envelope. Supply them from
    # the samples themselves (both columns are written for every parquet
    # sample) so the legacy npz carries the same total bias MBAR uses rather
    # than an umbrella-only matrix -- or a ValueError.
    def _energy_col(name):
        raw = samples.get(name)
        if raw is None:
            return None
        arr = np.ma.filled(np.ma.asarray(raw).astype(np.float64), np.nan)
        return np.asarray(arr, dtype=np.float64)

    v_pep_all = _energy_col("v_pep_kj_mol")
    v_dih_all = _energy_col("v_dih_kj_mol")

    # Resolve the envelope ONLY when the snapshot actually carries a rung,
    # mirroring every other call site. A GaMD-DISABLED run still writes
    # shared_gamd_setup_globals.json, with "all_globals": {} (production.py's
    # disabled-run writer), and PepGamdEnvelope.from_json raises KeyError when
    # no nested dict holds k0_Total -- so resolving unconditionally broke every
    # plain non-GaMD parquet run. Deliberately NOT a bare try/except: on a real
    # ladder run a missing/degenerate envelope must still surface, and
    # gareus/analysis.py wraps this whole call in `except Exception: pass`, so
    # anything swallowed here degrades silently to "analysis_arrays.npz absent".
    _envelope_cache: list = []

    def _envelope_for(windows):
        if not any(float(w.get("gamd_lambda", 0.0) or 0.0) > 0.0 for w in windows):
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
        expected_ids = [int(w.get("window_id", i)) for i, w in enumerate(first_windows)]
        for seg_id in np.unique(seg_ids):
            seg_windows = load_windows(run_dir, segment_id=str(seg_id))
            if not seg_windows:
                raise ValueError(f"No window snapshot found for segment {seg_id} in {run_dir}/windows/")
            seg_ids_list = [int(w.get("window_id", i)) for i, w in enumerate(seg_windows)]
            if len(seg_windows) != len(first_windows) or seg_ids_list != expected_ids:
                raise ValueError(
                    "Cannot write one legacy analysis_arrays.npz for segments with different window IDs/counts; "
                    "use segment-specific or union-state analysis."
                )
            mask = seg_ids == str(seg_id)
            nk[mask, :] = reconstruct_bias_matrix(
                cv_A[mask], cv2[mask] if cv2 is not None else None, seg_windows, beta,
                v_pep=v_pep_all[mask] if v_pep_all is not None else None,
                v_dih=v_dih_all[mask] if v_dih_all is not None else None,
                envelope=_envelope_for(seg_windows))
        windows = first_windows
    else:
        windows = load_windows(run_dir)
        if not windows:
            raise ValueError(f"No window snapshot found in {run_dir}/windows/")
        nk = reconstruct_bias_matrix(cv_A, cv2, windows, beta,
                                     v_pep=v_pep_all, v_dih=v_dih_all,
                                     envelope=_envelope_for(windows))

    save_kwargs: dict = {
        "cv_A": cv_A,
        "window": window,
        "umbrella_reduced_bias_nk": nk,
    }
    if seg_raw is not None:
        save_kwargs["segment_id"] = np.asarray(seg_raw).astype(str)
    if cv2 is not None:
        save_kwargs["secondary_cv"] = cv2

    out_path = Path(out_path) if out_path is not None else (Path(run_dir) / "analysis_arrays.npz")
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
