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


def phase_z(label: str, phase_dir: Path, replica, step, model_sha256: str, recorded=None) -> np.ndarray:
    """z (float64) for the given samples of one phase; raises AuxPoolingRefused when any row lacks it."""
    replica = np.asarray(replica, dtype=np.int64)
    step = np.asarray(step, dtype=np.int64)
    n = step.size
    if recorded is not None:
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


__all__ = ["phase_z", "burnin_keep", "worker_table", "aux_term_kcal", "phase_epoch", "Z_COLUMN", "KJ_PER_KCAL"]
