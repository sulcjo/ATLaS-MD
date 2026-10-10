"""Per-phase seeding evidence for auxiliary workers (CVaux repair F01).

Production writes ``<phase>/aux_seeding_record.json`` at phase setup, before any MD, in every phase that runs
the auxiliary runtime: per aux worker state, ``"pulled"`` (started by a US pull, i.e. a fresh transient) or
``"continued"`` (continued from an earlier end state), and the source. A phase is a worker's burn-in phase iff
the record says it was pulled there. A phase without a record (data older than this record) counts every
worker as pulled; a worker the record does not list counts as pulled too (conservative). A record that cannot
be read or does not follow the schema refuses pooling: missing evidence is conservative, corrupt evidence is
never guessed at.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Optional, Set, Tuple

SEEDING_RECORD_FILENAME = "aux_seeding_record.json"
SEEDING_RECORD_SCHEMA = "atlas-aux-seeding-record-v1"
PULLED = "pulled"
CONTINUED = "continued"
_SEEDING_VALUES = (PULLED, CONTINUED)
EVIDENCE_RECORD = "aux_seeding_record"
EVIDENCE_NONE = "no_seeding_record"


def seeding_record_payload(*, worker_windows: Iterable[int], state_of_window: Mapping[int, int],
                           pulled_windows: Iterable[int], continued_source: Mapping[int, str], branch: str,
                           fallback: Optional[str] = None) -> dict:
    """Record of one phase's worker seeding. ``worker_windows`` = the phase's aux worker windows (aux k > 0);
    ``state_of_window`` = the phase's window -> state map (identity where it has no entry, as the driver's own
    union build reads it); ``pulled_windows`` = windows started by a US pull; ``continued_source`` = window ->
    source of a continued window (``state_export``, ``final_pdb``, ``topup_state_export``); ``branch`` = the
    seeding path taken; ``fallback`` names why a continuation fell back to pulling (``seed_mismatch``)."""
    pulled = {int(w) for w in pulled_windows}
    workers = []
    for w in sorted({int(x) for x in worker_windows}):
        is_pulled = w in pulled
        workers.append({"state_id": int(state_of_window.get(w, w)), "window": w,
                        "seeding": PULLED if is_pulled else CONTINUED,
                        "source": "us_pull" if is_pulled else str(continued_source.get(w, "unknown"))})
    return {"schema": SEEDING_RECORD_SCHEMA, "branch": str(branch), "fallback": fallback, "workers": workers}


def _load(path: Path, where: str) -> dict:
    from gareus.kernel_identity import AuxPoolingRefused
    try:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AuxPoolingRefused(f"{where}: {path} is not readable JSON ({exc})") from exc
    if not isinstance(rec, dict) or rec.get("schema") != SEEDING_RECORD_SCHEMA or not isinstance(rec.get("workers"), list):
        raise AuxPoolingRefused(f"{where}: {path} is not an {SEEDING_RECORD_SCHEMA} record")
    for e in rec["workers"]:
        sid = e.get("state_id") if isinstance(e, dict) else None
        if (not isinstance(sid, int) or isinstance(sid, bool) or sid < 0
                or e.get("seeding") not in _SEEDING_VALUES):
            raise AuxPoolingRefused(f"{where}: {path} has a malformed worker entry {e!r}")
    return rec


def write_seeding_record(phase_dir, payload: dict) -> Path:
    """Write the record atomically. A worker an existing record of this phase lists as pulled stays pulled
    (a re-run setup never hides samples an earlier pulled start may have written)."""
    path = Path(phase_dir) / SEEDING_RECORD_FILENAME
    out = json.loads(json.dumps(payload))
    if path.exists():
        prev = _load(path, f"{phase_dir}: existing seeding record")
        earlier = {int(e["state_id"]) for e in prev["workers"] if e["seeding"] == PULLED}
        for e in out["workers"]:
            if int(e["state_id"]) in earlier and e["seeding"] != PULLED:
                e["seeding"], e["source"] = PULLED, "us_pull (earlier setup of this phase)"
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_burnin_workers(phase_dir, worker_ids: Iterable[int], where: str = "aux burn-in") -> Tuple[Set[int], str]:
    """(worker states pulled in this phase, evidence). No record = every worker in ``worker_ids``."""
    workers = {int(w) for w in worker_ids}
    path = Path(phase_dir) / SEEDING_RECORD_FILENAME
    if not path.exists():
        return workers, EVIDENCE_NONE
    rec = _load(path, where)
    seeding = {int(e["state_id"]): e["seeding"] for e in rec["workers"]}
    return {w for w in workers if seeding.get(w, PULLED) == PULLED}, EVIDENCE_RECORD


__all__ = ["SEEDING_RECORD_FILENAME", "SEEDING_RECORD_SCHEMA", "PULLED", "CONTINUED", "EVIDENCE_RECORD",
           "EVIDENCE_NONE", "seeding_record_payload", "write_seeding_record", "read_burnin_workers"]
