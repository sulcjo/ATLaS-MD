"""Ordered exchange-event ledger: schema, assignment checksums and replay (spec Section 7)."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..correctness._io import IntegrityError, digest, json_bytes

EXCHANGE_EVENT_SCHEMA = "atlas-exchange-events-v1"
#: skip = a proposal the swap kernel declined to attempt (no holder / no outcome).
EVENT_KINDS = ("swap", "stay", "no_candidates", "skip")


def assignment_sha256(assignments: Sequence[int]) -> str:
    """Checksum of the replica -> window assignment list (index = replica)."""
    return digest(json_bytes([int(x) for x in assignments]))


_EVENT_COLUMNS = ("attempt_seq", "kind", "assignment_sha256_after", "selected_replica")


def _col(events, name, rows=None):
    value = events[name]
    if np.ma.isMaskedArray(value):
        mask = np.ma.getmaskarray(value)
        if rows is not None and name in _EVENT_COLUMNS and mask[rows].any():
            raise IntegrityError(f"legacy exchange rows mixed into an event ledger (column {name} has "
                                 f"{int(mask[rows].sum())} missing values in the replay range)")
        value = np.ma.getdata(value)
    arr = np.asarray(value)
    if rows is not None and name in _EVENT_COLUMNS:
        # Board condition 2: a float column (pandas/NumPy upcast of a missing int) carries NaN, and
        # NaN.astype(int64) is garbage. Detect it before any cast, deterministically.
        if arr.dtype.kind == "f" and np.isnan(arr[rows]).any():
            raise IntegrityError(f"legacy exchange rows mixed into an event ledger (column {name} has NaN)")
        if arr.dtype == object and any(x is None or (isinstance(x, float) and x != x) for x in arr[rows]):
            raise IntegrityError(f"legacy exchange rows mixed into an event ledger (column {name} has None/NaN)")
    return arr


def replay_assignments(events: Mapping[str, np.ndarray], start_assignments: Sequence[int], *,
                       after_step: int, up_to_step: int) -> list[int]:
    """Replay the replica -> window assignment through the ordered event ledger.

    Rows with ``after_step < step <= up_to_step`` are applied in (step, attempt_seq) order; an
    accepted ``swap`` must find its replicas holding the recorded windows, and every row's
    assignment checksum must match. Duplicate (step, attempt_seq) and legacy (non-event) rows in
    the range raise :class:`IntegrityError`.
    """
    if "attempt_seq" not in events or "assignment_sha256_after" not in events:
        raise IntegrityError("exchange data has no ordered event columns; replay needs an event-mode ledger")
    step = np.asarray(np.ma.getdata(events["step"])).astype(np.int64)
    rows = np.flatnonzero((step > int(after_step)) & (step <= int(up_to_step)))
    seq_all = _col(events, "attempt_seq", rows)
    seq = seq_all.astype(np.int64)
    idx = rows[np.lexsort((seq[rows], step[rows]))]
    pairs = list(zip(step[idx].tolist(), seq[idx].tolist()))
    if len(set(pairs)) != len(pairs):
        raise IntegrityError("duplicate (step, attempt_seq) in the exchange ledger")
    kind = _col(events, "kind", rows)
    accepted = _col(events, "accepted").astype(bool)
    ri = _col(events, "replica_i").astype(np.int64)
    rj = _col(events, "replica_j").astype(np.int64)
    wi = _col(events, "window_i").astype(np.int64)
    wj = _col(events, "window_j").astype(np.int64)
    sha = _col(events, "assignment_sha256_after", rows)
    current = [int(x) for x in start_assignments]
    for n in idx:
        label = f"step {int(step[n])} seq {int(seq[n])}"
        if kind[n] == "swap" and accepted[n]:
            a, b = int(ri[n]), int(rj[n])
            if current[a] != int(wi[n]) or current[b] != int(wj[n]):
                raise IntegrityError(f"{label}: ledger window pair ({int(wi[n])},{int(wj[n])}) does not match "
                                     f"replicas {a},{b} holding ({current[a]},{current[b]})")
            current[a], current[b] = current[b], current[a]
        if assignment_sha256(current) != str(sha[n]):
            raise IntegrityError(f"{label}: assignment checksum after the event does not match the ledger")
    return current
