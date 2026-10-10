"""Ordered exchange-event ledger: schema, assignment checksums and replay (spec Section 7)."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..correctness._io import IntegrityError, digest, json_bytes

#: v2 (F09) adds ``log_p_accept`` and ``proposal_algorithm``. A v1 segment directory is never appended to
#: (the writer refuses a payload-schema change); v1 segments still load, their rows read as the legacy algorithm.
EXCHANGE_EVENT_SCHEMA = "atlas-exchange-events-v2"
#: Gibbs proposal algorithm of a record without ``proposal_algorithm`` (written before F09): clipped weights,
#: logs of rounded probabilities, -inf where a probability underflowed (the true log q is not recoverable).
LEGACY_PROPOSAL_ALGORITHM = "gibbs_softmax_clipped_v1"
LOG_SPACE_PROPOSAL_ALGORITHM = "gibbs_softmax_log_v2"
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


def _first_duplicate(keys: np.ndarray):
    """Index pair (first, second) of the first repeated row of an (n, 2) int key array, else None."""
    if keys.shape[0] < 2:
        return None
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    sk = keys[order]
    same = np.flatnonzero(np.all(sk[1:] == sk[:-1], axis=1))
    if same.size == 0:
        return None
    a, b = order[same[0]], order[same[0] + 1]
    return int(min(a, b)), int(max(a, b))


def _segments_of(data, rows) -> list[str]:
    seg = data.get("segment_id")
    if seg is None:
        return []
    return sorted({str(np.asarray(seg, dtype=object)[r]) for r in rows})


def refuse_duplicate_event_keys(events: Mapping[str, np.ndarray]) -> None:
    """Refuse a pooled exchange ledger holding one (step, attempt_seq) twice (Task 13 carry-over 3).

    Rows without an ``attempt_seq`` (legacy, non-event rows) carry no ordered key and are not compared.
    A duplicate means two segments replay the same exchange decision (e.g. a pooled crashed parent and
    its restarted child), so no permutation chain can be rebuilt from the ledger.
    """
    if not events or "attempt_seq" not in events or "step" not in events:
        return
    seq = events["attempt_seq"]
    present = ~np.ma.getmaskarray(seq) if np.ma.isMaskedArray(seq) else np.ones(len(seq), dtype=bool)
    seq_f = np.asarray(np.ma.getdata(seq), dtype=np.float64)
    present &= np.isfinite(seq_f)
    rows = np.flatnonzero(present)
    if rows.size < 2:
        return
    step = np.asarray(np.ma.getdata(events["step"])).astype(np.int64)
    keys = np.stack([step[rows], seq_f[rows].astype(np.int64)], axis=1)
    dup = _first_duplicate(keys)
    if dup is not None:
        a, b = rows[dup[0]], rows[dup[1]]
        raise IntegrityError(f"duplicate (step, attempt_seq) in the exchange ledger: step {int(step[a])}, "
                             f"attempt_seq {int(keys[dup[0], 1])} in segment(s) {_segments_of(events, (a, b))}; "
                             "two segments record the same exchange decision (a pooled crashed parent and its "
                             "restarted child?) -- repair the segment registry before resuming or pooling")


def _str_column(events, name, n):
    if name not in events:
        return np.full(n, None, dtype=object)
    value = events[name]
    mask = np.ma.getmaskarray(value) if np.ma.isMaskedArray(value) else np.zeros(n, dtype=bool)
    data = np.asarray(np.ma.getdata(value), dtype=object)
    return np.array([None if (m or x is None or (isinstance(x, float) and x != x)) else str(x)
                     for x, m in zip(data, mask)], dtype=object)


def _float_column(events, name, n):
    if name not in events:
        return np.full(n, np.nan)
    value = events[name]
    if np.ma.isMaskedArray(value):
        return np.ma.filled(value.astype(np.float64), np.nan)
    return np.asarray(value, dtype=np.float64)


def proposal_logs(events: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-row proposal algorithm and the log q / log alpha a reader may rely on (F09).

    A row without ``proposal_algorithm`` (v1 ledger, or a null in a v2 one written by a Gibbs move)
    is the legacy algorithm, unless it carries no proposal at all (pair modes: NaN log q's), which
    stays ``None``. A legacy row's -inf log q / log alpha means a rounded probability underflowed, not
    that the probability was zero: it becomes NaN (unrecoverable), never a finite or -inf claim.
    Legacy rows never recorded ``log_p_accept``: NaN.
    """
    n = len(np.asarray(np.ma.getdata(events["step"]))) if "step" in events else 0
    algorithm = _str_column(events, "proposal_algorithm", n)
    lqf = _float_column(events, "log_q_forward", n)
    lqr = _float_column(events, "log_q_reverse", n)
    lpa = _float_column(events, "log_p_accept", n)
    unknown = np.array([a is None for a in algorithm], dtype=bool)
    has_proposal = ~(np.isnan(lqf) & np.isnan(lqr))
    algorithm[unknown & has_proposal] = LEGACY_PROPOSAL_ALGORITHM
    legacy = np.array([a == LEGACY_PROPOSAL_ALGORITHM for a in algorithm], dtype=bool)
    for arr in (lqf, lqr):
        arr[legacy & np.isneginf(arr)] = np.nan
    lpa[legacy] = np.nan
    return {"proposal_algorithm": algorithm, "log_q_forward": lqf, "log_q_reverse": lqr, "log_p_accept": lpa}
