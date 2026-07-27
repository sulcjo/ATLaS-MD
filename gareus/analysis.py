"""Analysis-readiness and sparse-aware validation helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json_file, resolve_run_temperature_k, _json_ready
from .diagnostics import _read_csv_dicts, _safe_float, _sample_counts_by_window, _hist_overlap_np

__all__ = [
    "_read_window_table_for_validation",
    "validate_analysis_metadata_readiness",
]

def _read_window_table_for_validation(out_dir: Path) -> list[dict]:
    out_dir = Path(out_dir)
    preferred = out_dir / "umbrella_explicit_windows.csv"
    if preferred.exists() and preferred.stat().st_size > 0:
        return _read_csv_dicts(preferred)
    return _read_csv_dicts(out_dir / "umbrella_windows.csv")

def _ensure_analysis_arrays_npz(out_dir: Path) -> None:
    """Reconstruct analysis_arrays.npz from Parquet data if NPZ is absent."""
    npz_path = out_dir / "analysis_arrays.npz"
    if npz_path.exists():
        return
    samples_dir = out_dir / "samples"
    if not samples_dir.exists() or not any(samples_dir.glob("**/*.parquet")):
        return
    try:
        from .query import export_analysis_arrays_npz
        temp_k = resolve_run_temperature_k(out_dir)
        if temp_k is None:
            # gareus_metadata.json/run_manifest.json have no usable
            # temperature; guessing 300 K here would silently corrupt the
            # reduced-bias matrix for any run not actually at 300 K.
            print(f"WARNING: could not resolve simulation temperature for {out_dir}; "
                  f"defaulting to 300.0 K when reconstructing analysis_arrays.npz")
            temp_k = 300.0
        beta = 1.0 / (8.314462618e-3 * temp_k)
        export_analysis_arrays_npz(out_dir, beta, npz_path)
    except Exception as exc:
        pass  # errors reported below when NPZ still absent


def validate_analysis_metadata_readiness(out_dir: Path, target_overlap: float = 0.30) -> dict:
    """Sparse-aware MBAR/readiness validation for final production outputs.

    Unlike the legacy overlap check, this validator never assumes flattened
    window order is a meaningful neighbor relation for explicit sparse 2D grids.
    It validates window-index consistency, matrix shapes, optional graph edges,
    and the presence of bias-component arrays needed by downstream MBAR/PMF code.
    """
    out_dir = Path(out_dir)
    _ensure_analysis_arrays_npz(out_dir)
    warnings: list[str] = []
    errors: list[str] = []
    table_path = out_dir / "umbrella_explicit_windows.csv"
    legacy_table_path = out_dir / "umbrella_windows.csv"
    npz_path = out_dir / "analysis_arrays.npz"
    meta_path = out_dir / "umbrella_pymbar_metadata.json"
    graph_path = out_dir / "explicit_2d_neighbor_graph.csv"
    windows = _read_window_table_for_validation(out_dir)
    n_windows = int(len(windows))
    result = {
        "schema_version": "2.0-sparse-2d-safe",
        "status": "ok",
        "window_table_csv": str(table_path if table_path.exists() else legacy_table_path),
        "umbrella_windows_csv": str(legacy_table_path),
        "analysis_arrays_npz": str(npz_path),
        "umbrella_pymbar_metadata_json": str(meta_path),
        "neighbor_graph_csv": str(graph_path) if graph_path.exists() else "",
        "n_windows": n_windows,
        "warnings": warnings,
        "errors": errors,
    }
    if n_windows <= 0:
        errors.append("no umbrella window table found; expected umbrella_explicit_windows.csv or umbrella_windows.csv")
    explicit_2d = any(str(r.get("explicit_2d", "0")) in {"1", "True", "true"} for r in windows)
    rectangular = all(str(r.get("rectangular_grid", "0")) in {"1", "True", "true"} for r in windows) if explicit_2d and windows else False
    sparse_2d = bool(explicit_2d and not rectangular)
    result.update({"explicit_2d": bool(explicit_2d), "rectangular_grid": bool(rectangular), "sparse_2d": bool(sparse_2d)})
    if sparse_2d:
        result["neighbor_overlap_mode"] = "graph_edges_only; flattened window-order overlap intentionally skipped"
    else:
        result["neighbor_overlap_mode"] = "flat adjacent windows for 1D/rectangular fallback"

    # Validate window table fields.
    seen_windows = set()
    for row_idx, row in enumerate(windows, start=2):
        try:
            w = int(row.get("window", row_idx - 2))
        except Exception:
            errors.append(f"window table row {row_idx} has invalid window index {row.get('window')!r}")
            continue
        if w in seen_windows:
            errors.append(f"window table has duplicate window index {w}")
        seen_windows.add(w)
        if str(row.get("primary_cv", "distance")) == "nonlocal-contacts":
            for key in ("primary_center", "primary_k"):
                val = _safe_float(row.get(key))
                if not math.isfinite(val):
                    errors.append(f"window {w} missing finite {key}")
        else:
            for key in ("distance_center_A", "distance_k_kcal_mol_A2"):
                val = _safe_float(row.get(key))
                if not math.isfinite(val):
                    errors.append(f"window {w} missing finite {key}")
        if str(row.get("secondary_cv_center", "")) not in {"", "nan", "None"}:
            if not math.isfinite(_safe_float(row.get("secondary_cv_k_kcal_mol"))):
                errors.append(f"window {w} has secondary center but missing finite secondary_cv_k_kcal_mol")
    if seen_windows and seen_windows != set(range(n_windows)):
        errors.append("window table indices are not contiguous 0..n_windows-1")

    if not npz_path.exists():
        errors.append("analysis_arrays.npz missing")
    data_shapes = {}
    counts = []
    neighbor_rows = []
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                keys = set(data.files)
                required = {"cv_A", "window", "umbrella_reduced_bias_nk"}
                missing = sorted(required - keys)
                if missing:
                    errors.append("analysis_arrays.npz missing required arrays: " + ", ".join(missing))
                    n_samples = 0
                    win = np.asarray([], dtype=np.int64)
                    cv = np.asarray([], dtype=float)
                    ub = np.asarray([], dtype=float)
                else:
                    cv = np.asarray(data["cv_A"], dtype=float)
                    win = np.asarray(data["window"], dtype=np.int64)
                    ub = np.asarray(data["umbrella_reduced_bias_nk"], dtype=float)
                    n_samples = int(cv.size)
                result["n_samples"] = int(n_samples)
                for key in sorted(keys):
                    try:
                        data_shapes[key] = list(np.asarray(data[key]).shape)
                    except Exception:
                        pass
                matrix_keys = [
                    "primary_umbrella_bias_kcal_mol_nk",
                    "distance_umbrella_bias_kcal_mol_nk",
                    "secondary_cv_bias_kcal_mol_nk",
                    "umbrella_bias_kcal_mol_nk",
                    "umbrella_bias_kj_mol_nk",
                    "umbrella_reduced_bias_nk",
                ]
                result["available_matrix_arrays"] = {k: data_shapes.get(k) for k in matrix_keys if k in data_shapes}
                for key in matrix_keys:
                    if key not in keys:
                        if key == "umbrella_reduced_bias_nk":
                            errors.append(f"analysis_arrays.npz missing matrix array {key}")
                        elif key == "distance_umbrella_bias_kcal_mol_nk" and "primary_umbrella_bias_kcal_mol_nk" in keys:
                            # New generic primary-CV matrix supersedes the legacy distance-named alias.
                            pass
                        else:
                            warnings.append(f"analysis_arrays.npz missing recommended bias-component matrix {key}; samples.csv JSON columns may still exist")
                    else:
                        arr = np.asarray(data[key])
                        if arr.ndim != 2 or arr.shape[0] != n_samples or arr.shape[1] != n_windows:
                            errors.append(f"{key} has shape {arr.shape}, expected ({n_samples}, {n_windows})")
                if "umbrella_bias_kj_mol_nk" in keys and "umbrella_reduced_bias_nk" in keys:
                    total_kj = np.asarray(data["umbrella_bias_kj_mol_nk"], dtype=float)
                    red = np.asarray(data["umbrella_reduced_bias_nk"], dtype=float)
                    if total_kj.shape == red.shape and total_kj.size:
                        finite = np.isfinite(total_kj) & np.isfinite(red)
                        if finite.any():
                            # beta can be inferred from any nonzero finite pair if needed; this is only a consistency smoke check.
                            result["bias_matrix_finite_fraction"] = float(np.mean(finite))
                if n_samples <= 0:
                    errors.append("no samples in analysis_arrays.npz")
                if win.size != n_samples:
                    errors.append("window array length does not match cv_A length")
                if win.size:
                    bad = win[(win < 0) | (win >= n_windows)]
                    if bad.size:
                        errors.append(f"sample window indices outside 0..{n_windows-1}: {sorted(set(int(x) for x in bad[:20]))}")
                    counts = _sample_counts_by_window(win, n_windows)
                    zero = [i for i, c in enumerate(counts) if c <= 0]
                    low = [i for i, c in enumerate(counts) if 0 < c < 10]
                    if zero:
                        errors.append("windows with zero production samples: " + ", ".join(map(str, zero)))
                    if low:
                        warnings.append("windows with fewer than 10 production samples: " + ", ".join(map(str, low)))
                    result["sample_counts_by_window"] = counts
                if not np.all(np.isfinite(cv)):
                    warnings.append("cv_A contains non-finite values")
                if required <= keys and ub.ndim == 2 and not np.all(np.isfinite(ub)):
                    warnings.append("umbrella_reduced_bias_nk contains non-finite values")
        except Exception as exc:
            errors.append(f"could not read analysis_arrays.npz: {exc}")
    result["analysis_array_shapes"] = data_shapes

    # Graph validation for sparse explicit 2D windows.
    if sparse_2d:
        if not graph_path.exists():
            warnings.append("sparse explicit 2D run has no explicit_2d_neighbor_graph.csv; exchange may still have run, but graph diagnostics are missing")
        else:
            graph = _read_csv_dicts(graph_path)
            result["neighbor_graph_edges"] = int(len(graph))
            bad_edges = []
            for row in graph:
                wi = int(_safe_float(row.get("window_i"), -999999))
                wj = int(_safe_float(row.get("window_j"), -999999))
                if wi < 0 or wj < 0 or wi >= n_windows or wj >= n_windows or wi == wj:
                    bad_edges.append(row.get("edge", f"{wi}-{wj}"))
            if bad_edges:
                errors.append("neighbor graph edges reference invalid windows: " + ", ".join(map(str, bad_edges[:20])))
            if not graph:
                warnings.append("sparse explicit 2D neighbor graph is empty")
            # Use graph edges for optional overlap diagnostics if arrays exist.
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    cv = np.asarray(data["cv_A"], dtype=float)
                    win = np.asarray(data["window"], dtype=np.int64)
                    sec = np.asarray(data["secondary_cv"], dtype=float) if "secondary_cv" in data.files else np.full_like(cv, np.nan)
                for row in graph:
                    wi = int(_safe_float(row.get("window_i"), -1))
                    wj = int(_safe_float(row.get("window_j"), -1))
                    if not (0 <= wi < n_windows and 0 <= wj < n_windows):
                        continue
                    edge_type = str(row.get("edge_type", ""))
                    if "secondary" in edge_type and np.isfinite(sec).any():
                        a = sec[win == wi]
                        b = sec[win == wj]
                        lo, hi = -1.0, 1.0
                    else:
                        a = cv[win == wi]
                        b = cv[win == wj]
                        finite_centers = [_safe_float(r.get("primary_center", r.get("distance_center_A"))) for r in windows]
                        finite_centers = [x for x in finite_centers if math.isfinite(x)]
                        lo = min(finite_centers) if finite_centers else float(np.nanmin(cv))
                        hi = max(finite_centers) if finite_centers else float(np.nanmax(cv))
                    ov = _hist_overlap_np(a, b, lo, hi, bins=80)
                    neighbor_rows.append({"window_i": wi, "window_j": wj, "edge_type": edge_type, "overlap": ov, "left_count": int(a.size), "right_count": int(b.size)})
                weak = [r for r in neighbor_rows if math.isfinite(float(r.get("overlap", float("nan")))) and float(r["overlap"]) < float(target_overlap)]
                result["neighbor_overlaps"] = neighbor_rows
                result["neighbor_pairs_below_target_overlap"] = weak
                if weak:
                    warnings.append(f"{len(weak)} graph neighbor edge(s) below target overlap {target_overlap:.2f}")
            except Exception as exc:
                warnings.append(f"could not compute sparse graph overlap diagnostics: {exc}")
    else:
        # Legacy 1D/rectangular fallback: adjacent window overlaps are still useful.
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                cv = np.asarray(data["cv_A"], dtype=float)
                win = np.asarray(data["window"], dtype=np.int64)
            centers = [_safe_float(r.get("primary_center", r.get("distance_center_A", r.get("center_A")))) for r in windows]
            finite_centers = [c for c in centers if math.isfinite(c)]
            lo = min(finite_centers) if finite_centers else float(np.nanmin(cv))
            hi = max(finite_centers) if finite_centers else float(np.nanmax(cv))
            connected = True
            for i in range(max(0, n_windows - 1)):
                a = cv[win == i]
                b = cv[win == i + 1]
                ov = _hist_overlap_np(a, b, lo, hi, bins=80)
                neighbor_rows.append({"left_window": i, "right_window": i + 1, "overlap": ov, "left_count": int(a.size), "right_count": int(b.size)})
                if (not math.isfinite(ov)) or ov < 0.03:
                    connected = False
            result["neighbor_overlaps"] = neighbor_rows
            result["overlap_connected_at_0p03"] = bool(connected)
            weak = [r for r in neighbor_rows if math.isfinite(float(r.get("overlap", float("nan")))) and float(r["overlap"]) < float(target_overlap)]
            result["neighbor_pairs_below_target_overlap"] = weak
            if neighbor_rows and not connected:
                errors.append("neighbor CV histogram overlap graph appears disconnected at threshold 0.03")
            if weak:
                warnings.append(f"{len(weak)} neighbor pair(s) below target overlap {target_overlap:.2f}")
        except Exception as exc:
            if npz_path.exists() and n_windows:
                warnings.append(f"could not compute legacy neighbor overlap diagnostics: {exc}")

    # Metadata file checks.
    if meta_path.exists():
        meta = read_json_file(meta_path, {}) or {}
        result["pymbar_metadata_schema_version"] = meta.get("schema_version", "legacy")
        meta_n = int(meta.get("n_windows", meta.get("window_count", n_windows)) or n_windows)
        if meta_n != n_windows:
            errors.append(f"umbrella_pymbar_metadata.json says n_windows={meta_n}, but window table has {n_windows}")
        if sparse_2d and not meta.get("sparse_2d_safe", False):
            warnings.append("sparse 2D run metadata is missing sparse_2d_safe=true")
    else:
        warnings.append("umbrella_pymbar_metadata.json missing")

    if errors:
        result["status"] = "error"
    elif warnings:
        result["status"] = "warning"
    else:
        result["status"] = "ok"
    return _json_ready(result)
