"""Fixed-state NPZ export from an explicitly pinned, eligible sample view.

This is the hard export core, not a new adaptive union-state loader. The caller
owns Parquet selection and must supply a completed/quiescent or immutable view.
We hash the actual arrays, definitions and boundary token; file existence or
mtime is never a freshness certificate.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping
import numpy as np

from ._io import IntegrityError, digest, fsync_directory, json_bytes, json_loads
from .bias import finite_number, numeric_vector, reconstruct_bias_matrix
from .state_identity import state_definition_hash, validate_fixed_state_segments

EXPORT_SCHEMA = "atlas-fixed-state-npz-v1"
R_KJ_MOL_K = 8.314462618e-3


def _text_vector(raw, label: str, length: int | None = None) -> np.ndarray:
    values = np.ma.asarray(raw)
    if values.ndim != 1 or (length is not None and len(values) != length):
        raise IntegrityError(f"{label} must have shape ({length or 'N'},)")
    if np.ma.getmaskarray(values).any():
        raise IntegrityError(f"{label} contains missing identifiers")
    plain = np.asarray(values)
    if any(not isinstance(v, str) or not v for v in plain.tolist()):
        raise IntegrityError(f"{label} needs nonempty strings")
    return plain.astype(str)


def _integer_vector(raw, label: str, length: int) -> np.ndarray:
    array = np.ma.asarray(raw)
    if array.ndim != 1 or len(array) != length or np.ma.getmaskarray(array).any():
        raise IntegrityError(f"{label} must be an unmasked ({length},) integer array")
    values = np.asarray(array)
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise IntegrityError(f"{label} must contain nonnegative integer IDs")
    if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise IntegrityError(f"{label} overflows int64")
    return values.astype(np.int64)


def array_hash(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise IntegrityError("Object arrays are forbidden in scientific NPZ exports")
    hasher = hashlib.sha256()
    hasher.update(json_bytes({"shape": list(array.shape), "dtype": array.dtype.str}))
    if array.dtype.kind in "US":
        hasher.update(json_bytes(array.tolist()))
    else:
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def build_export_arrays(
    samples: Mapping[str, Any], snapshots: Mapping[str, Mapping[str, Any]], beta: float,
    *, sample_view: Mapping[str, Any], envelope_factory: Callable | None = None,
    reconstruct: Callable = reconstruct_bias_matrix,
) -> dict[str, np.ndarray]:
    """Build one coherent matrix; never change a column according to row origin.

    sample_view must declare immutable/quiescent ownership and contain an explicit
    nonempty boundary ID. This records the caller's guarantee, not proof of a lock.
    All required observations must be finite for export. The kernel independently
    supports NaN propagation; this exporter refuses implicit sample deletion.
    """
    view = json_loads(json_bytes(dict(sample_view)))
    if view.get("kind") not in {"immutable", "quiescent"} or not view.get("boundary_id"):
        raise IntegrityError("Export requires an immutable/quiescent sample view with boundary_id")
    beta = finite_number(beta, "beta", positive=True)
    for name in ("cv1", "window_id", "segment_id"):
        if name not in samples:
            raise IntegrityError(f"Missing required sample column: {name}")
    cv1 = numeric_vector(samples["cv1"], "cv1")
    n = len(cv1)
    if n == 0:
        raise IntegrityError("Cannot export an empty sample set")
    segments = _text_vector(samples["segment_id"], "segment_id", n)
    selected_ids = sorted(set(segments.tolist()))
    missing = set(selected_ids) - set(snapshots)
    if missing:
        raise IntegrityError(f"Missing snapshots for {sorted(missing)}")
    selected = {key: snapshots[key] for key in selected_ids}
    table = validate_fixed_state_segments(selected)
    expected_beta = 1.0 / (R_KJ_MOL_K * table.definition["temperature_k"])
    if not math.isclose(beta, expected_beta, rel_tol=1e-10, abs_tol=0.0):
        raise IntegrityError(f"beta={beta} conflicts with frozen temperature; expected {expected_beta}")
    origin_ids = _integer_vector(samples["window_id"], "window_id", n)
    to_column = {state_id: i for i, state_id in enumerate(table.column_window_ids)}
    unknown = set(origin_ids.tolist()) - set(to_column)
    if unknown:
        raise IntegrityError(f"Samples have unknown origin window IDs: {sorted(unknown)}")
    origins = np.asarray([to_column[v] for v in origin_ids], dtype=np.int64)
    observed_lambdas = None
    if "gamd_lambda" in samples:
        observed_lambdas = numeric_vector(samples["gamd_lambda"], "gamd_lambda", n)
        assigned_lambdas = np.asarray([table.windows[i]["gamd_lambda"] for i in origins])
        if (not np.isfinite(observed_lambdas).all()
                or not np.array_equal(observed_lambdas, assigned_lambdas)):
            raise IntegrityError("Sample gamd_lambda values disagree with their frozen origin states")
    cv2 = None if "cv2" not in samples else numeric_vector(samples["cv2"], "cv2", n)
    pep = None if "v_pep_kj_mol" not in samples else numeric_vector(samples["v_pep_kj_mol"], "v_pep", n)
    dih = None if "v_dih_kj_mol" not in samples else numeric_vector(samples["v_dih_kj_mol"], "v_dih", n)
    has_ladder = any(w["gamd_lambda"] > 0 for w in table.windows)
    if has_ladder and envelope_factory is None:
        raise IntegrityError("Ladder export requires a factory consuming the frozen boost definition")
    envelope = envelope_factory(table.definition["boost"]) if has_ladder else None
    # ONE call with ONE state table for ALL rows; never per-segment states.
    matrix = np.asarray(reconstruct(cv1, cv2, list(table.windows), beta,
                        v_pep=pep, v_dih=dih, envelope=envelope), dtype=np.float64)
    if matrix.shape != (n, len(table.windows)):
        raise IntegrityError(f"Reconstructor returned an invalid matrix shape: {matrix.shape}")
    complete = np.all(np.isfinite(matrix), axis=1)
    if not complete.all():
        counts = np.bincount(origins[~complete], minlength=len(table.windows)).tolist()
        raise IntegrityError(
            f"Incomplete cross-state energies for {int((~complete).sum())} samples; "
            f"counts by origin column={counts}. No rows were silently discarded."
        )
    arrays = {
        "cv_A": cv1, "window": origins, "original_window_id": origin_ids,
        "segment_id": segments, "column_window_ids": np.asarray(table.column_window_ids, dtype=np.int64),
        "umbrella_reduced_bias_nk": matrix,
        "N_k": np.bincount(origins, minlength=len(table.windows)).astype(np.int64),
    }
    if cv2 is not None:
        arrays["secondary_cv"] = cv2
    for name in ("step", "replica"):
        if name in samples:
            arrays[name] = _integer_vector(samples[name], name, n)
    # Include source channels even if they do not all enter the output: changing
    # a raw energy must invalidate a cached ladder export.
    source_arrays = {"cv1": cv1, "origins": origin_ids, "segments": segments}
    for name, value in (("cv2", cv2), ("v_pep", pep), ("v_dih", dih), ("gamd_lambda", observed_lambdas)):
        if value is not None:
            source_arrays[name] = value
    for name in ("step", "replica"):
        if name in arrays:
            source_arrays[name] = arrays[name]
    signature_payload = {
        "export_schema": EXPORT_SCHEMA, "beta": beta,
        "state_definition_sha256": table.definition_sha256,
        "selected_snapshots": {key: digest(json_bytes(selected[key])) for key in selected_ids},
        "sample_view": view,
        "source_arrays": {key: array_hash(value) for key, value in sorted(source_arrays.items())},
    }
    manifest = {
        "schema": EXPORT_SCHEMA, "input_signature": digest(json_bytes(signature_payload)),
        "inputs": signature_payload, "state_definition": table.definition,
        "n_samples": n, "n_states": len(table.windows),
        "quantity": "reduced umbrella-plus-ladder bias; NOT full potential",
        "array_hashes": {key: array_hash(value) for key, value in sorted(arrays.items())},
    }
    arrays["export_manifest_json"] = np.asarray(json_bytes(manifest).decode("utf-8"))
    return arrays


def write_npz_atomic(out_path: Path | str, arrays: Mapping[str, np.ndarray]) -> Path:
    """Publish data and its provenance as one file; no companion-file transaction."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.is_symlink():
        raise IntegrityError("Refusing to replace an NPZ symlink")
    for key, value in arrays.items():
        if not isinstance(key, str) or np.asarray(value).dtype.kind == "O":
            raise IntegrityError("NPZ requires named, non-object arrays")
    fd, name = tempfile.mkstemp(prefix=f".{out_path.name}.", dir=out_path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, out_path)
        fsync_directory(out_path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return out_path


def export_fixed_state_npz(out_path, samples, snapshots, beta, *, sample_view,
                           envelope_factory=None, reconstruct=reconstruct_bias_matrix) -> Path:
    arrays = build_export_arrays(samples, snapshots, beta, sample_view=sample_view,
                                 envelope_factory=envelope_factory, reconstruct=reconstruct)
    return write_npz_atomic(out_path, arrays)


def read_current_export(path: Path | str, expected_input_signature: str) -> dict[str, np.ndarray]:
    """Refuse legacy, stale, incomplete or modified output; allow_pickle=False."""
    with np.load(path, allow_pickle=False) as saved:
        if "export_manifest_json" not in saved.files:
            raise IntegrityError("NPZ has no verified freshness metadata")
        arrays = {key: saved[key] for key in saved.files}
    manifest = json_loads(str(arrays.pop("export_manifest_json").item()))
    if not isinstance(manifest, dict):
        raise IntegrityError("NPZ manifest is not an object")
    if manifest.get("schema") != EXPORT_SCHEMA or manifest.get("input_signature") != expected_input_signature:
        raise IntegrityError("NPZ is stale or uses an unsupported export schema")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or digest(json_bytes(inputs)) != expected_input_signature:
        raise IntegrityError("NPZ input-signature metadata is internally inconsistent")
    if state_definition_hash(manifest.get("state_definition", {})) != inputs.get("state_definition_sha256"):
        raise IntegrityError("NPZ frozen state definition does not match its signature")
    hashes = manifest.get("array_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(arrays):
        raise IntegrityError("NPZ array manifest is incomplete")
    for key, array in arrays.items():
        if array_hash(array) != hashes[key]:
            raise IntegrityError(f"NPZ array checksum mismatch: {key}")
    return arrays
