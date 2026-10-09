"""Shared pieces of union-MBAR pooling for admitted auxiliary workers (driver builder and analyzer loader).

z of every sample: the recorded ``aux_z_00`` of a post-admission phase, else the phase's
``aux_z_backfill.parquet`` (pre-admission), joined on exactly (replica, step). A row of ANY state without
z refuses the pooling (the worker restraint is evaluated for every sample under every worker state);
z is never filled. Worker burn-in = the worker's samples of the phase(s) of epoch ``burnin_phase_epoch``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from gareus.adaptive.aux_backfill import BACKFILL_FILENAME, read_phase_backfill
from gareus.adaptive.aux_discovery.frames import phase_epoch
from gareus.kernel_identity import AuxPoolingRefused

Z_COLUMN = "aux_z_00"
KJ_PER_KCAL = 4.184


def worker_table(states: Iterable[Tuple[int, dict]]) -> Dict[int, dict]:
    """{state_id: aux params} for the auxiliary workers among (state_id, metadata) pairs."""
    out = {}
    for sid, metadata in states:
        rec = (metadata or {}).get("aux")
        if isinstance(rec, dict) and rec.get("role") == "auxiliary":
            out[int(sid)] = rec
    return out


def _worker_key_matches(rec: dict, w: dict) -> bool:
    return (rec.get("spawn_parent_state_id") is not None
            and int(rec["spawn_parent_state_id"]) == int(w["parent_state_id"])
            and abs(float(rec["aux_center"]) - float(w["aux_center"])) < 1e-9
            and abs(float(rec["aux_k_kcal_mol"]) - float(w["aux_k_kcal_mol"])) < 1e-9)


def require_admitted_workers(admission: dict, registry_workers: Dict[int, dict], where: str, *,
                             pooled: Optional[Dict[int, dict]] = None) -> None:
    """The admission record's workers and the registry's workers must be the SAME set (parent + centre + k,
    one-to-one, both directions); with ``pooled`` (the workers actually in the union), every one of them must be
    pooled too (an admitted worker that is not usable refuses: fail closed).

    A missing or stale registry would otherwise silently turn workers into ordinary states and drop the aux
    term; a registry worker the record does not list (e.g. after a partial applier refusal) would pool under a
    record that does not describe it."""
    recorded = list(admission.get("workers") or [])
    unmatched = dict(registry_workers)
    missing = []
    for w in recorded:
        hit = next((sid for sid, rec in unmatched.items() if _worker_key_matches(rec, w)), None)
        if hit is None:
            missing.append((w["parent_state_id"], w["aux_center"], w["aux_k_kcal_mol"]))
        else:
            unmatched.pop(hit)
    if missing:
        raise AuxPoolingRefused(f"{where}: admitted aux worker(s) (parent, centre, k) {missing} have no matching "
                                "worker state in the registry (state_registry.json missing, stale or unusable)")
    if unmatched:
        raise AuxPoolingRefused(f"{where}: registry worker state(s) {sorted(unmatched)} are not in aux_admission.json "
                                "(the admission record and the registry disagree)")
    if pooled is not None:
        absent = sorted(int(sid) for sid in registry_workers if int(sid) not in {int(k) for k in pooled})
        if absent:
            raise AuxPoolingRefused(f"{where}: admitted aux worker state(s) {absent} are not usable in the union "
                                    "(an admitted worker must pool)")


def phase_runtime_model_shas(phase_dir: Path) -> set:
    """Aux model digests a phase ran with, from its own window snapshots (``windows/*.json``: kernel identity,
    the frozen state definition's ``aux_models``, any row's ``aux_model_sha256``). Empty = no aux runtime."""
    import json
    shas = set()
    for snap in sorted(Path(phase_dir).glob("windows/*.json")):
        try:
            payload = json.loads(snap.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        ki = payload.get("kernel_identity")
        if isinstance(ki, dict) and ki.get("aux_model_sha256"):
            shas.add(str(ki["aux_model_sha256"]))
        state = payload.get("state_definition") if isinstance(payload.get("state_definition"), dict) else {}
        shas.update(str(k) for k in (state.get("aux_models") or {}))
        for row in list(payload.get("windows") or []) + list(state.get("windows") or []):
            if isinstance(row, dict) and row.get("aux_model_sha256"):
                shas.add(str(row["aux_model_sha256"]))
    return shas


def phase_z(label: str, phase_dir: Path, replica, step, model_sha256: str, recorded=None) -> np.ndarray:
    """z (float64) for the given samples of one phase; raises AuxPoolingRefused when any row lacks it, or when
    recorded z comes from a phase whose runtime aux model (its window snapshots) is not the admission's."""
    replica = np.asarray(replica, dtype=np.int64)
    step = np.asarray(step, dtype=np.int64)
    n = step.size
    if recorded is not None:
        shas = phase_runtime_model_shas(phase_dir)
        if shas != {str(model_sha256)}:
            raise AuxPoolingRefused(f"{label}: recorded {Z_COLUMN} comes from runtime aux model(s) "
                                    f"{sorted(x[:12] for x in shas) or 'none recorded'} in {phase_dir}/windows, "
                                    f"not the admission's {str(model_sha256)[:12]}")
        z = np.asarray(np.ma.asarray(recorded, dtype=np.float64).filled(np.nan), dtype=np.float64)
        missing = int((~np.isfinite(z)).sum())
        if missing:
            raise AuxPoolingRefused(f"{label}: {missing} rows without aux z (recorded {Z_COLUMN} not finite)")
        return z
    if not (Path(phase_dir) / BACKFILL_FILENAME).exists():
        raise AuxPoolingRefused(f"{label}: {n} rows without aux z (no recorded {Z_COLUMN} and no "
                                f"{BACKFILL_FILENAME} backfill in {phase_dir})")
    try:
        bf = read_phase_backfill(Path(phase_dir), model_sha256)
    except ValueError as exc:
        raise AuxPoolingRefused(f"{label}: {exc}") from exc
    import pandas as pd
    index = pd.MultiIndex.from_arrays([bf["replica"].to_numpy(np.int64), bf["step"].to_numpy(np.int64)])
    if not index.is_unique:
        raise AuxPoolingRefused(f"{label}: backfill has duplicate (replica, step) keys")
    pos = index.get_indexer(pd.MultiIndex.from_arrays([replica, step]))
    z = np.full(n, np.nan)
    hit = pos >= 0
    z[hit] = bf["aux_z"].to_numpy(np.float64)[pos[hit]]
    missing = int((~np.isfinite(z)).sum())
    if missing:
        raise AuxPoolingRefused(f"{label}: {missing} rows without aux z (backfill missing or incomplete)")
    return z


def burnin_keep(state_ids: np.ndarray, label_epoch: Optional[int], workers: Dict[int, dict]) -> np.ndarray:
    """Keep mask over rows of ONE phase: False for worker-state rows when the phase is the worker's burn-in epoch."""
    keep = np.ones(np.asarray(state_ids).shape[0], dtype=bool)
    if label_epoch is None:
        return keep
    for sid, rec in workers.items():
        if rec.get("burnin_phase_epoch") is not None and int(rec["burnin_phase_epoch"]) == int(label_epoch):
            keep &= np.asarray(state_ids) != int(sid)
    return keep


def aux_term_kcal(z: np.ndarray, center: float, k_kcal: float) -> np.ndarray:
    return 0.5 * float(k_kcal) * (np.asarray(z, dtype=np.float64) - float(center)) ** 2


__all__ = ["phase_z", "phase_runtime_model_shas", "burnin_keep", "worker_table", "aux_term_kcal", "phase_epoch", "Z_COLUMN", "KJ_PER_KCAL"]
