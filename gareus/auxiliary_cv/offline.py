"""Offline auxiliary energies from stored features, parity, exclusion audits and pooling guards."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ..correctness._io import IntegrityError
from ..correctness.bias import KJ_PER_KCAL
from .evaluate import z_from_dihedrals
from .model import AuxModel
from .sample_schema import AuxSampleSchema

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
