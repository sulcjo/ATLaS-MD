"""Offline auxiliary energies from stored features, parity, exclusion audits and pooling guards."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..correctness._io import IntegrityError
from ..correctness.bias import KJ_PER_KCAL
from .duplicates import merge_identical_hamiltonians  # noqa: F401  (re-export, final fix wave I4d)
from .evaluate import z_from_dihedrals
from .model import AuxModel
from .sample_schema import AUX_SAMPLES_SCHEMA, AuxSampleSchema

AUDIT_THRESHOLD_FRACTION = 1e-3


def _float_column(samples, name, n) -> np.ndarray:
    raw = samples[name]
    arr = np.ma.filled(np.ma.asarray(raw, dtype=np.float64), np.nan) if np.ma.isMaskedArray(raw) \
        else np.asarray(raw, dtype=np.float64)
    if arr.shape != (n,):
        raise IntegrityError(f"sample column {name} has shape {arr.shape}, expected ({n},)")
    return arr


def registry_model(definition: Mapping[str, Any], sha: str, *, where: str) -> AuxModel:
    """The frozen registry's model for ``sha``; a diagnostic IntegrityError, never a KeyError (board 7)."""
    registry = (definition or {}).get("aux_models") or {}
    if sha not in registry:
        raise IntegrityError(f"{where}: auxiliary model {sha} is recorded in the sample schema but missing "
                             f"from the frozen state definition's aux_models registry {sorted(registry)}")
    return AuxModel.from_mapping(registry[sha])


def parity_context(windows, beta: float) -> dict[str, Any]:
    """beta, the strongest active k per model and that model's active centres, from a state table."""
    k_max: dict[str, float] = {}
    centers: dict[str, list[float]] = {}
    for w in windows:
        if float(w.get("aux_k", 0.0)) > 0:
            sha = w["aux_model_sha256"]
            k_max[sha] = max(k_max.get(sha, 0.0), float(w["aux_k"]))
            centers.setdefault(sha, []).append(float(w["aux_center"]))
    return {"beta": float(beta), "k_max_kcal": k_max, "centers": centers}


def parity_bound(parity: Mapping[str, Any], sha: str) -> tuple[float, list[float]]:
    """(k_max, active centres) of model ``sha`` for the runtime parity bound -- strict (final fix wave I4a).

    No active state anywhere (sham-only population): (0.0, []) explicitly. An active state exists but
    ``sha`` has no k_max: refused, never silently read as 0 (parity must never switch itself off).
    """
    k_max = parity["k_max_kcal"]
    if not k_max:
        return 0.0, []
    if sha not in k_max:
        raise IntegrityError(f"parity bound: model {sha} has no k_max although the state table holds active "
                             f"auxiliary states (models {sorted(k_max)})")
    return float(k_max[sha]), [float(c) for c in parity["centers"][sha]]


def parity_violation(stored, recomputed, *, beta: float, k_max_kcal: float, centers: Sequence[float]) -> np.ndarray:
    """Conservative reduced-energy disagreement per row (exact bound uses |dz|/2; this uses |dz|)."""
    stored = np.asarray(stored, dtype=np.float64)
    recomputed = np.asarray(recomputed, dtype=np.float64)
    dz = np.abs(stored - recomputed)
    cs = np.asarray(list(centers) or [0.0], dtype=np.float64)
    dev = np.max(np.abs(recomputed[:, None] - cs[None, :]), axis=1)
    return float(beta) * KJ_PER_KCAL * float(k_max_kcal) * dz * (dev + dz)


def aux_z_from_samples(samples: Mapping[str, Any], schema: AuxSampleSchema,
                       models: Mapping[str, AuxModel], *, beta: float,
                       k_max_kcal: Mapping[str, float], centers: Mapping[str, Sequence[float]],
                       parity_reduced_tol) -> dict[str, np.ndarray]:
    n = len(np.asarray(samples["cv1"]))
    missing = [c for c in schema.torsion_columns if c not in samples]
    if missing:
        raise IntegrityError(f"historical rows lack auxiliary features (columns {missing[:3]}...); "
                             "they cannot enter a pool with active auxiliary states")
    tol = np.broadcast_to(np.asarray(parity_reduced_tol, dtype=np.float64), (n,))
    theta = np.stack([_float_column(samples, c, n) for c in schema.torsion_columns], axis=1)
    out: dict[str, np.ndarray] = {}
    for col, sha in zip(schema.z_columns, schema.model_shas):
        if sha not in models:
            raise IntegrityError(f"no aux model loaded for schema model {sha}")
        z = z_from_dihedrals(theta[:, schema.model_basis_index(models[sha])], models[sha]).astype(np.float64)
        if col in samples:
            stored = _float_column(samples, col, n)
            if np.any(np.isfinite(stored) != np.isfinite(z)):
                raise IntegrityError(f"stored {col} fails offline parity against model {sha} (finite mismatch)")
            k = float(k_max_kcal.get(sha, 0.0))
            both = np.isfinite(stored) & np.isfinite(z)
            if k > 0 and np.any(both):
                du = parity_violation(stored[both], z[both], beta=beta, k_max_kcal=k, centers=centers[sha])
                bad = du > tol[both]
                if np.any(bad):
                    raise IntegrityError(f"stored {col} fails offline parity against model {sha}: "
                                         f"max reduced-energy disagreement {du.max():.3g} > tolerance "
                                         f"{float(tol[both][bad].min()):g} on {int(bad.sum())} rows")
        out[sha] = z
    return out


def exclusion_report(excluded, *, origin_ids, replicas, steps, segment_ids, aux_z: Mapping[str, np.ndarray],
                     time_block_steps: int, z_bins: int = 10) -> dict[str, Any]:
    excluded = np.asarray(excluded, dtype=bool)
    if int(time_block_steps) <= 0:
        raise IntegrityError("exclusion_report needs a positive time_block_steps")
    def counts(values):
        vals, cnt = np.unique(np.asarray(values)[excluded].astype(str), return_counts=True)
        return {str(v): int(c) for v, c in zip(vals, cnt)}
    blocks = np.asarray([f"{s}:{int(t) // int(time_block_steps)}" for s, t in zip(segment_ids, steps)])
    by_z = {}
    for sha, z in aux_z.items():
        z = np.asarray(z, dtype=np.float64)
        finite_all = z[np.isfinite(z)]
        lo, hi = (float(finite_all.min()), float(finite_all.max())) if finite_all.size else (0.0, 1.0)
        edges = np.linspace(lo, hi if hi > lo else lo + 1.0, int(z_bins) + 1)
        sel = z[excluded]
        hist, _ = np.histogram(sel[np.isfinite(sel)], bins=edges)
        by_z[sha] = {"edges": edges.tolist(), "counts": hist.astype(int).tolist(),
                     "nonfinite": int(np.count_nonzero(~np.isfinite(sel)))}
    n, k = int(excluded.size), int(excluded.sum())
    return {"n_total": n, "n_excluded": k, "fraction": (k / n) if n else 0.0,
            "by_origin_state": counts(origin_ids), "by_carrier": counts(replicas),
            "by_time_block": counts(blocks), "by_z_range": by_z,
            "by_structural_group": "unavailable: no frozen structural labels before Stage D",
            "audit_threshold_fraction": AUDIT_THRESHOLD_FRACTION,
            "above_audit_threshold": bool(n and k / n > AUDIT_THRESHOLD_FRACTION)}


def _segment_payloads(run_dir) -> dict[str, "dict | None"]:
    """``payload_schema`` of every ``samples/<segment>/`` manifest (hashes not verified here)."""
    from pathlib import Path
    from ..parquet_manifest import load_manifest
    out: dict[str, "dict | None"] = {}
    root = Path(run_dir) / "samples"
    if not root.exists():
        return out
    for seg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = load_manifest(seg_dir, expected_kind="samples")
        payload = (manifest or {}).get("payload_schema")
        # Only atlas-aux-samples-v1 payloads are auxiliary; anything else (or none) reads as None.
        out[seg_dir.name] = payload if isinstance(payload, dict) and payload.get("schema") == AUX_SAMPLES_SCHEMA else None
    return out


def segment_aux_schemas(run_dir) -> dict[str, "AuxSampleSchema | None"]:
    return {seg: (AuxSampleSchema.from_payload(p) if p else None) for seg, p in _segment_payloads(run_dir).items()}


def segment_aux_runtime(run_dir) -> dict[str, "dict | None"]:
    """The ``runtime`` block (platform, precision) of every segment's sample payload."""
    from .sample_schema import runtime_from_payload
    return {seg: (runtime_from_payload(p) if p else None) for seg, p in _segment_payloads(run_dir).items()}


POOL_CLASSES = ("aux", "unpersisted", "ineligible_kernel", "no_snapshot", "legacy")
_ALWAYS_REFUSED = (
    ("legacy", "ran without auxiliary states: a campaign directory mixing v2 and v3_aux segments cannot be "
               "pooled (ruling H3); split the run directory"),
    ("no_snapshot", "have no window snapshot in an auxiliary run: their state table is unknown (ruling H2); "
                    "repair the run"),
    ("ineligible_kernel", "have an ineligible kernel (segment_eligibility); they cannot enter an equilibrium pool"),
)


def classify_pool_segments(run_dir, segment_ids) -> dict[str, list[str]]:
    """Pool class of each segment of an auxiliary run (see POOL_CLASSES)."""
    from ..kernel_identity import ELIGIBLE_AUX_UNPERSISTED, ELIGIBLE_VERIFIED, snapshot_has_aux
    from ..query import load_windows_metadata, segment_eligibility
    eligibility = segment_eligibility(run_dir)
    out: dict[str, list[str]] = {key: [] for key in POOL_CLASSES}
    for seg in sorted(str(s) for s in segment_ids):
        snap = load_windows_metadata(run_dir, seg)
        status = (eligibility.get(seg) or {}).get("eligibility")
        if not snap:
            out["no_snapshot"].append(seg)
        elif not snapshot_has_aux(snap):
            out["legacy"].append(seg)
        elif status == ELIGIBLE_VERIFIED:
            out["aux"].append(seg)
        elif status == ELIGIBLE_AUX_UNPERSISTED:
            out["unpersisted"].append(seg)
        else:
            out["ineligible_kernel"].append(seg)
    return out


def refuse_duplicate_sample_keys(samples: Mapping[str, Any]) -> None:
    """Refuse an auxiliary sample pool holding one observation key (step, replica) twice.

    Task 13 carry-over 3: an interrupted auxiliary parent that never checkpointed stays pooled up to its
    crash step while its restarted child re-runs the same steps; the two would count one carrier's
    observation twice.
    """
    from .ledger import _first_duplicate, _segments_of
    if not samples or "step" not in samples or "replica" not in samples:
        return
    step = np.asarray(np.ma.getdata(samples["step"])).astype(np.int64)
    replica = np.asarray(np.ma.getdata(samples["replica"])).astype(np.int64)
    dup = _first_duplicate(np.stack([step, replica], axis=1))
    if dup is not None:
        a, b = dup
        raise IntegrityError(f"duplicate (step, replica) observation in the auxiliary sample pool: step {int(step[a])}, "
                             f"replica {int(replica[a])} in segment(s) {_segments_of(samples, (a, b))}; a crashed "
                             "parent without a checkpoint and its restarted child overlap -- repair the segment "
                             "registry (seal the parent abandoned) before pooling")


def pool_aux_segments(prod, samples, beta, meta, *, exclude_segments_without_aux_features: bool = False,
                      allow_ineligible_aux_segments: bool = False):
    """load_parquet's auxiliary branch: refuse or exclude segment classes, one eligible fixed state,
    one sample schema, offline z with per-segment parity. Returns (samples, windows, aux_z or None)."""
    from ..correctness.state_identity import validate_fixed_state_segments
    from ..query import load_windows_metadata
    from .sample_schema import PARITY_TOLERANCE
    seg_col = np.asarray(samples["segment_id"]).astype(str)
    groups = classify_pool_segments(prod, set(seg_col.tolist()))
    for key, why in _ALWAYS_REFUSED:
        if groups[key]:
            raise IntegrityError(f"segments {groups[key]} {why}")
    if groups["unpersisted"]:
        if not exclude_segments_without_aux_features:
            raise IntegrityError(f"segments {groups['unpersisted']} ran auxiliary states without stored features "
                                 "(Stage B era) and cannot be evaluated; pass exclude_segments_without_aux_features=True "
                                 "to drop them with a note")
        n = seg_col.size
        keep = ~np.isin(seg_col, groups["unpersisted"])
        dropped = {s: int(np.count_nonzero(seg_col == s)) for s in groups["unpersisted"]}
        # v[keep] keeps masked arrays masked (no np.asarray): placeholders stay visible as NaN downstream.
        samples = {k: (v[keep] if hasattr(v, "__len__") and len(v) == n else v) for k, v in samples.items()}
        seg_col = seg_col[keep]
        meta.setdefault("load_notes", []).append(
            f"excluded auxiliary segments without stored features (rows): {dropped}")
    refuse_duplicate_sample_keys(samples)
    aux_segs = groups["aux"]
    if not aux_segs or seg_col.size == 0:
        raise IntegrityError(f"{prod}: no auxiliary segment with stored features remains to pool")
    snaps = {s: load_windows_metadata(prod, s) for s in aux_segs}
    try:
        table = validate_fixed_state_segments(snaps, require_eligible=not allow_ineligible_aux_segments)
    except IntegrityError as exc:
        raise IntegrityError(f"auxiliary segments are not one eligible fixed state: {exc}") from exc
    if allow_ineligible_aux_segments:
        meta.setdefault("load_notes", []).append(
            "engineering analysis: ineligible auxiliary segments allowed, not an equilibrium estimate")
    # load_parquet indexes u_nk columns by the raw window_id: the frozen columns must be 0..K-1 in order and
    # every sample origin one of them, else rows would be scored against the wrong state (cf. chignolin_6).
    columns = tuple(int(c) for c in table.column_window_ids)
    if columns != tuple(range(len(columns))):
        raise IntegrityError(f"frozen state table window_id columns {list(columns)[:8]} are not 0..K-1 in order; "
                             "load_parquet cannot index them by window_id")
    origins = np.asarray(np.ma.filled(np.ma.asarray(samples["window_id"]), -1)).astype(np.int64)
    unknown = sorted(set(origins[(origins < 0) | (origins >= len(columns))].tolist()))
    if unknown:
        raise IntegrityError(f"sample window_id(s) {unknown[:8]} are not states of the frozen table "
                             f"(window_id 0..{len(columns) - 1})")
    schemas, runtimes = segment_aux_schemas(prod), segment_aux_runtime(prod)
    distinct = {schemas.get(s) for s in aux_segs}
    if None in distinct or len(distinct) != 1:
        raise IntegrityError(f"auxiliary sample schema differs between (or is missing from) segments {aux_segs}")
    schema = distinct.pop()
    bad_rt = [s for s in aux_segs if (runtimes.get(s) or {}).get("precision") not in PARITY_TOLERANCE]
    if bad_rt:
        raise IntegrityError(f"platform precision not recorded for segment(s) {bad_rt}; "
                             "parity tolerance cannot be chosen")
    row_tol = np.asarray([PARITY_TOLERANCE[runtimes[s]["precision"]] for s in seg_col], dtype=np.float64)
    # Every schema model must be in the frozen registry: a silent filter would drop an active term.
    models = {sha: registry_model(table.definition, sha, where=f"load_parquet ({prod})") for sha in schema.model_shas}
    windows = [dict(w) for w in table.windows]
    aux_z = None
    if any(float(w.get("aux_k", 0.0)) > 0 for w in windows):
        aux_z = aux_z_from_samples(samples, schema, models, parity_reduced_tol=row_tol, **parity_context(windows, beta))
    meta["aux_models"] = list(schema.model_shas)
    meta["aux_feature_segments"] = list(aux_segs)
    return samples, windows, aux_z
