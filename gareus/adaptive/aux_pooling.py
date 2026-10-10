"""Shared pieces of union-MBAR pooling for admitted auxiliary workers (driver builder and analyzer loader).

z of every sample: the recorded ``aux_z_00`` of a phase that ran with the admitted model (required there: no
backfill fallback), else the phase's ``aux_z_backfill.parquet`` (a phase run without the model), joined on
exactly (replica, step), after its bytes match the sha256 the admission record holds for that phase. A row of
ANY state without z refuses the pooling (the worker restraint is evaluated for every sample under every worker
state); z is never filled. Worker burn-in = the worker's samples of the phase(s) of epoch ``burnin_phase_epoch``.
Pooling also needs a passing post-admission spot check once one is due (``require_spot_check``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from gareus.adaptive.aux_backfill import BACKFILL_FILENAME, phase_label, read_phase_backfill
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


def spot_check_due_phases(adaptive_dir, admission: dict) -> list:
    """Labels of the campaign's phases after the admission epoch (final phases included) that ran with the
    admitted model (window snapshots): the spot check is due once one exists."""
    from gareus.adaptive import discovery_census as census
    sha = str(admission.get("model_sha256"))
    try:
        adm_epoch = int(admission["epoch"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuxPoolingRefused(f"{adaptive_dir}: aux_admission.json records no admission epoch ({exc})") from exc
    phases, _ = census.ordered_phases(Path(adaptive_dir))
    out = []
    for label, phase_dir in phases:
        ep = phase_epoch(label)
        if (ep is None or ep > adm_epoch) and sha in phase_runtime_model_shas(phase_dir):
            out.append(label)
    return out


def require_spot_check(adaptive_dir, admission: dict) -> None:
    """Refuse pooling unless the post-admission spot check (recorded aux_z_00 vs the frame-derived z the
    backfill uses) passed, or is not due yet. Failed, errored or not-run-when-due = AuxPoolingRefused."""
    sc = admission.get("spot_check")
    if sc is not None:
        if isinstance(sc, dict) and sc.get("ok") is True:
            return
        sc = sc if isinstance(sc, dict) else {"record": sc}
        detail = ", ".join(f"{k}={sc[k]}" for k in ("phase", "max_energy_err_kt", "tol_kt", "max_abs_dev",
                                                     "status", "error", "errors") if sc.get(k) is not None)
        raise AuxPoolingRefused(f"{adaptive_dir}: the post-admission aux z spot check did not pass "
                                f"({detail or 'ok is not true'}): the backfilled z is not validated, pooling refused")
    due = spot_check_due_phases(adaptive_dir, admission)
    if due:
        raise AuxPoolingRefused(f"{adaptive_dir}: the post-admission aux z spot check is due (phase(s) {due[:3]} ran "
                                "with the admitted model) but aux_admission.json has no spot_check record")


def expected_backfill_sha256(admission: dict, adaptive_dir, phase_dir, where: str) -> str:
    """The sha256 the admission record holds for this phase's backfill (matched by campaign-relative label;
    an entry without one by its resolved path). Missing or ambiguous = AuxPoolingRefused."""
    label = phase_label(phase_dir, adaptive_dir)
    if label is None:
        raise AuxPoolingRefused(f"{where}: {phase_dir} is outside the campaign {adaptive_dir}: its backfill "
                                "cannot be matched to the admission record")
    hits = []
    for e in (admission or {}).get("backfill") or []:
        if not isinstance(e, dict):
            continue
        if e.get("label") is not None:
            if str(e["label"]) == label:
                hits.append(e)
        elif e.get("phase") is not None and Path(str(e["phase"])).resolve() == Path(phase_dir).resolve():
            hits.append(e)
    shas = {str(e.get("sha256")) for e in hits}
    if not hits:
        raise AuxPoolingRefused(f"{where}: aux_admission.json has no backfill entry for phase {label!r}: its "
                                f"{BACKFILL_FILENAME} is not the one the admission recorded")
    if len(shas) != 1:
        raise AuxPoolingRefused(f"{where}: aux_admission.json has conflicting backfill entries for phase {label!r}")
    sha = shas.pop()
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise AuxPoolingRefused(f"{where}: aux_admission.json backfill entry for phase {label!r} has no sha256")
    return sha


def phase_z(label: str, phase_dir: Path, replica, step, model_sha256: str, recorded=None, *,
            admission: dict, adaptive_dir) -> np.ndarray:
    """z (float64) for the given samples of one phase; raises AuxPoolingRefused when any row lacks it, when
    recorded z comes from a phase whose runtime aux model (its window snapshots) is not the admission's, when a
    phase that ran with the admitted model has no recorded z (never backfilled), or when the backfill's bytes
    are not the ones ``admission`` (the aux_admission.json record) hashed for the phase."""
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
    if str(model_sha256) in phase_runtime_model_shas(phase_dir):
        raise AuxPoolingRefused(f"{label}: phase ran with the admitted aux model {str(model_sha256)[:12]} (its window "
                                f"snapshots) but its samples carry no recorded {Z_COLUMN}: integrity failure, a "
                                "post-admission phase is never backfilled")
    if not (Path(phase_dir) / BACKFILL_FILENAME).exists():
        raise AuxPoolingRefused(f"{label}: {n} rows without aux z (no recorded {Z_COLUMN} and no "
                                f"{BACKFILL_FILENAME} backfill in {phase_dir})")
    expected = expected_backfill_sha256(admission, adaptive_dir, phase_dir, label)
    try:
        bf = read_phase_backfill(Path(phase_dir), model_sha256, expected_sha256=expected)
    except (OSError, ValueError) as exc:
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


__all__ = ["phase_z", "phase_runtime_model_shas", "require_spot_check", "spot_check_due_phases",
           "expected_backfill_sha256", "burnin_keep", "worker_table", "aux_term_kcal", "phase_epoch", "Z_COLUMN", "KJ_PER_KCAL"]
