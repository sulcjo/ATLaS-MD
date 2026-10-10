"""Shared pieces of union-MBAR pooling for admitted auxiliary workers (driver builder and analyzer loader).

z of every sample: the recorded ``aux_z_00`` of a phase that ran with the admitted model (required there: no
backfill fallback), else the phase's ``aux_z_backfill.parquet`` (a phase run without the model), joined on
exactly (replica, step), after its bytes match the sha256 the admission record holds for that phase. A row of
ANY state without z refuses the pooling (the worker restraint is evaluated for every sample under every worker
state); z is never filled. Worker burn-in (F01) = per carrier, inside each phase whose seeding record says the
worker was US-pulled there: ``burnin_selection`` (``burnin_phase_epoch`` is reporting only, never a filter).
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


def _snapshot_model_shas(snap: Path) -> set:
    """Aux model digests one window snapshot names (kernel identity, the frozen state definition's
    ``aux_models``, any row's ``aux_model_sha256``)."""
    import json
    shas = set()
    try:
        payload = json.loads(Path(snap).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return shas
    if isinstance(payload, dict):
        ki = payload.get("kernel_identity")
        if isinstance(ki, dict) and ki.get("aux_model_sha256"):
            shas.add(str(ki["aux_model_sha256"]))
        state = payload.get("state_definition") if isinstance(payload.get("state_definition"), dict) else {}
        shas.update(str(k) for k in (state.get("aux_models") or {}))
        for row in list(payload.get("windows") or []) + list(state.get("windows") or []):
            if isinstance(row, dict) and row.get("aux_model_sha256"):
                shas.add(str(row["aux_model_sha256"]))
    return shas


def phase_runtime_model_shas(phase_dir: Path) -> set:
    """Aux model digests a phase ran with, from its own window snapshots (``windows/*.json``: kernel identity,
    the frozen state definition's ``aux_models``, any row's ``aux_model_sha256``). Empty = no aux runtime."""
    shas = set()
    for snap in sorted(Path(phase_dir).glob("windows/*.json")):
        shas |= _snapshot_model_shas(snap)
    return shas


def require_persisted_model_segments(phase_dir: Path, model_sha256: str, where: str) -> None:
    """In an admitted campaign, a segment that ran with the admitted model but whose samples cannot be
    reconstructed (kernel eligibility ``aux_unpersisted``: no atlas-aux-samples-v1 payload, or no window
    snapshot in a phase run with the model) refuses pooling instead of being dropped. Segments not run with the
    admitted model (pre-admission) are untouched: they pool through the backfill."""
    from gareus.kernel_identity import ELIGIBLE_AUX_UNPERSISTED
    from gareus.query import segment_eligibility
    phase_dir = Path(phase_dir)
    sha = str(model_sha256)
    phase_ran_with_model = None
    for seg_id, info in segment_eligibility(phase_dir).items():
        if info.get("eligibility") != ELIGIBLE_AUX_UNPERSISTED:
            continue
        snap = phase_dir / "windows" / f"{seg_id}.json"
        if snap.exists():
            ran = sha in _snapshot_model_shas(snap)
        else:
            if phase_ran_with_model is None:
                phase_ran_with_model = sha in phase_runtime_model_shas(phase_dir)
            ran = phase_ran_with_model
        if ran:
            raise AuxPoolingRefused(f"{where}: segment {seg_id} ran with the admitted aux model {sha[:12]} but is "
                                    f"{ELIGIBLE_AUX_UNPERSISTED} ({info.get('reason')}): its recorded z is missing, "
                                    "pooling refused (never dropped, never backfilled)")


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
        cause = ("spot check never ran: --ap-aux-discovery is off for this campaign (decision_settings.json "
                 "aux_discovery false; the check runs only inside the discovery hook)"
                 if _aux_discovery_recorded_off(adaptive_dir) else "aux_admission.json has no spot_check record")
        raise AuxPoolingRefused(f"{adaptive_dir}: the post-admission aux z spot check is due (phase(s) {due[:3]} ran "
                                f"with the admitted model) but {cause}")


def spot_check_is_final(sc) -> bool:
    """A recorded spot check that stands: a measured verdict (pass or over-tolerance). A record of a check that
    could not run (``status`` error: exception, setup failure) is re-run at the next boundary."""
    return isinstance(sc, dict) and sc.get("status") != "error"


def _aux_discovery_recorded_off(adaptive_dir) -> bool:
    import json
    try:
        rec = json.loads((Path(adaptive_dir) / "decision_settings.json").read_text())
        return (rec.get("settings") or {}).get("aux_discovery") is False
    except (OSError, ValueError, AttributeError):
        return False


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


BURNIN_RULE = "aux_burnin_carrier_v1"
REASON_CARRIER = "aux_burnin_carrier"
REASON_NO_REPLICA = "aux_burnin_no_replica"


def exclusion_records(phase, replica, state_id, excluded, reason_of_phase: Mapping[str, str]) -> list:
    """[{phase, state_id, n_replicas, rows_excluded, reason}] over the excluded rows, by (phase, sampled state)."""
    excluded = np.asarray(excluded, dtype=bool)
    if not excluded.any():
        return []
    labels = np.asarray(phase, dtype=object).astype(str)[excluded]
    sid = np.asarray(state_id, dtype=np.int64)[excluded]
    rep = None if replica is None else np.asarray(replica, dtype=np.int64)[excluded]
    out = []
    for label in sorted(set(labels.tolist())):
        in_phase = labels == label
        for s in sorted(set(sid[in_phase].tolist())):
            rows = in_phase & (sid == s)
            out.append({"phase": label, "state_id": int(s),
                        "n_replicas": (0 if rep is None or reason_of_phase[label] == REASON_NO_REPLICA
                                       else int(np.unique(rep[rows]).size)),
                        "rows_excluded": int(rows.sum()), "reason": reason_of_phase[label]})
    return out


def burnin_selection(phase, replica, step, state_id, burnin_workers: Mapping[str, Iterable[int]], *,
                     evidence: Optional[Mapping[str, str]] = None, where: str = "aux burn-in"):
    """Per-carrier worker burn-in exclusion (F01); the one rule both union paths use.

    Rows are observations keyed by (phase, replica, step), phase-local. ``burnin_workers[phase]`` = the worker
    states started by a US pull in that phase (``aux_seeding_record.read_burnin_workers``). Within such a
    phase, every replica trajectory loses its rows from its first visit (smallest step, never row order) to a
    pulled worker to the end of the phase; rows before it, other replicas and every other phase are kept.
    ``replica`` None (no replica column) in a phase that needs the exclusion excludes that whole phase.

    Returns ``(keep, record)``: a keep mask over the rows and ``{"rule", "phases": [{phase, burnin_workers,
    evidence, carriers: [{replica, first_excluded_step}], reason}], "records": exclusion_records(...)}``.
    """
    from gareus.correctness.observation_keys import integer_key, validate_observation_keys
    labels = np.asarray(phase, dtype=object).astype(str)
    n = labels.shape[0]
    sid = integer_key(np.asarray(state_id), "state_id", n)
    if replica is None:
        step_a = integer_key(step, "step", n)
        rep = None
    else:
        _, rep, step_a = validate_observation_keys(labels, replica, step, where=where)
    keep = np.ones(n, dtype=bool)
    phases, reasons = [], {}
    for label in sorted(set(labels.tolist())):
        workers = sorted({int(w) for w in burnin_workers.get(label, ())})
        if not workers:
            continue
        in_phase = np.flatnonzero(labels == label)
        at_worker = np.isin(sid[in_phase], workers)
        if not at_worker.any():
            continue
        entry = {"phase": label, "burnin_workers": workers, "evidence": (evidence or {}).get(label)}
        if rep is None:
            keep[in_phase] = False
            entry.update(carriers=None, reason=REASON_NO_REPLICA)
        else:
            reps, inv = np.unique(rep[in_phase], return_inverse=True)
            first = np.full(reps.size, np.iinfo(np.int64).max, dtype=np.int64)
            np.minimum.at(first, inv[at_worker], step_a[in_phase][at_worker])
            keep[in_phase] = step_a[in_phase] < first[inv]
            hit = first < np.iinfo(np.int64).max
            entry.update(carriers=[{"replica": int(r), "first_excluded_step": int(f)}
                                   for r, f in zip(reps[hit], first[hit])], reason=REASON_CARRIER)
        reasons[label] = entry["reason"]
        phases.append(entry)
    return keep, {"rule": BURNIN_RULE, "phases": phases,
                  "records": exclusion_records(labels, rep, sid, ~keep, reasons)}


def merge_burnin_records(records: Iterable[dict]) -> dict:
    """One record over several per-phase ``burnin_selection`` records (phases are disjoint)."""
    out = {"rule": BURNIN_RULE, "phases": [], "records": [], "superseded": []}
    for rec in records:
        out["phases"].extend(rec["phases"])
        out["records"].extend(rec["records"])
        out["superseded"].extend(rec.get("superseded") or [])
    out["phases"].sort(key=lambda e: e["phase"])
    out["records"].sort(key=lambda e: (e["phase"], e["state_id"]))
    out["superseded"].sort(key=lambda e: (e["phase"], e["segment"]))
    return out


def burnin_dropped_by_state(record: dict) -> Dict[str, int]:
    """{state_id: rows excluded over all phases} from a burn-in record."""
    out: Dict[str, int] = {}
    for r in record["records"]:
        out[str(r["state_id"])] = out.get(str(r["state_id"]), 0) + int(r["rows_excluded"])
    return out


REASON_SUPERSEDED = "superseded_by_resume"
_KEY_SHIFT = 40                    # (replica, step) -> replica << 40 | step for overlap tests


def segment_order(phase_dir) -> Dict[str, int]:
    """Durable segment order of one phase: the position in its append-only ``segments.json`` (the segment
    registry appends every gareus invocation). Empty when the phase has no readable registry."""
    import json
    try:
        segs = json.loads((Path(phase_dir) / "segments.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(segs, list):
        return {}
    return {str(seg.get("segment_id")): i for i, seg in enumerate(segs) if isinstance(seg, dict)}


def resume_supersession(phase, replica, step, segment_id, segment_orders: Mapping[str, Mapping[str, int]], *,
                        where: str = "aux union"):
    """Exclude whole segments a later restart superseded (one phase at a time; aux union paths only).

    Within a phase, segments are taken in their durable ``segments.json`` order. When a later segment L's
    (replica, step) keys overlap an earlier, not yet superseded segment E's and L starts at or before E's
    first step, L is a restarted history: every row of E is excluded (reason ``superseded_by_resume``).
    Segments whose keys do not overlap (a continuation starts after the earlier one's last step) are kept.
    An overlap where L starts inside E (not a restart from E's start), or involving a segment with no
    durable order or no segment id, refuses (AuxPoolingRefused). Returns ``(keep, records)``,
    records = [{phase, segment, superseded_by, rows_excluded, reason}].
    """
    from gareus.correctness.observation_keys import integer_key
    from gareus.kernel_identity import AuxPoolingRefused
    labels = np.asarray(phase, dtype=object).astype(str)
    n = labels.shape[0]
    keep = np.ones(n, dtype=bool)
    records: list = []
    if replica is None or n == 0:
        return keep, records
    rep = integer_key(replica, "replica", n)
    st = integer_key(step, "step", n)
    if st.size and int(st.max()) >= (1 << _KEY_SHIFT):
        raise AuxPoolingRefused(f"{where}: step {int(st.max())} too large for the supersession key")
    key = (rep << _KEY_SHIFT) | st
    seg = None if segment_id is None else np.asarray(segment_id, dtype=object).astype(str)
    for label in sorted(set(labels.tolist())):
        idx = np.flatnonzero(labels == label)
        if seg is None:
            continue                              # one unlabelled pool: duplicate keys refuse downstream
        segs = sorted(set(seg[idx].tolist()))
        if len(segs) < 2:
            continue
        order = segment_orders.get(label) or {}
        rows = {s_: idx[seg[idx] == s_] for s_ in segs}
        keys = {s_: np.unique(key[r]) for s_, r in rows.items()}
        unordered = [s_ for s_ in segs if s_ not in order]
        ranked = sorted((s_ for s_ in segs if s_ in order), key=lambda s_: order[s_])
        for u in unordered:
            if any(np.intersect1d(keys[u], keys[o], assume_unique=True).size for o in segs if o != u):
                raise AuxPoolingRefused(f"{where}: {label}: segment {u!r} overlaps another segment's (replica, "
                                        "step) keys but has no durable order in segments.json: refusing")
        superseded = set()
        for j, later in enumerate(ranked):
            for earlier in ranked[:j]:
                if earlier in superseded:
                    continue
                if not np.intersect1d(keys[earlier], keys[later], assume_unique=True).size:
                    continue
                e_first = int(st[rows[earlier]].min())
                l_first = int(st[rows[later]].min())
                if l_first > e_first:
                    raise AuxPoolingRefused(
                        f"{where}: {label}: segment {later!r} overlaps {earlier!r} but starts at step {l_first}, "
                        f"inside it (first step {e_first}): neither a restart nor a continuation, refusing")
                superseded.add(earlier)
                keep[rows[earlier]] = False
                records.append({"phase": label, "segment": earlier, "superseded_by": later,
                                "rows_excluded": int(rows[earlier].size), "reason": REASON_SUPERSEDED})
    records.sort(key=lambda e: (e["phase"], e["segment"]))
    return keep, records


def union_aux_selection(phase, replica, step, segment_id, state_id, burnin_workers: Mapping[str, Iterable[int]], *,
                        segment_orders: Mapping[str, Mapping[str, int]], evidence: Optional[Mapping[str, str]] = None,
                        where: str = "aux union"):
    """The row selection both aux union paths apply: resume supersession, then (on the surviving rows, keys
    validated) per-carrier worker burn-in. Returns ``(keep, record, not_superseded)``: ``record`` is the burn-in
    record plus ``superseded`` (the supersession records); ``not_superseded`` the rows supersession kept."""
    not_superseded, superseded = resume_supersession(phase, replica, step, segment_id, segment_orders, where=where)
    keep = not_superseded.copy()
    sub = np.flatnonzero(keep)
    labels = np.asarray(phase, dtype=object)
    bk, record = burnin_selection(labels[sub], None if replica is None else np.asarray(replica)[sub],
                                  np.asarray(step)[sub], np.asarray(state_id)[sub], burnin_workers,
                                  evidence=evidence, where=where)
    keep[sub] = bk
    record["superseded"] = superseded
    return keep, record, not_superseded


def aux_term_kcal(z: np.ndarray, center: float, k_kcal: float) -> np.ndarray:
    return 0.5 * float(k_kcal) * (np.asarray(z, dtype=np.float64) - float(center)) ** 2


__all__ = ["phase_z", "phase_runtime_model_shas", "require_spot_check", "spot_check_due_phases",
           "spot_check_is_final", "require_persisted_model_segments",
           "expected_backfill_sha256", "burnin_selection", "merge_burnin_records", "burnin_dropped_by_state",
           "exclusion_records", "resume_supersession", "segment_order", "union_aux_selection", "worker_table", "aux_term_kcal", "phase_epoch", "Z_COLUMN", "KJ_PER_KCAL"]
