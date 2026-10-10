"""Observation identity (phase/source, replica, step) and stored-dtype lambda comparison (CVaux repair F10).

One observation is identified by ``(phase/source id, replica, step)``; the segment id is provenance, never
identity, so a crashed parent and its restarted child (same phase, overlapping steps) collide while equal
``(replica, step)`` pairs of different legitimate phases do not. Used by the auxiliary pooling path and the
strict exporter; the plain (non-auxiliary) loaders never call it.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ._io import IntegrityError

_INT64_LIMIT = float(2 ** 63)   # exact in float64; valid integers are strictly below it


class ObservationKeyRefusal(IntegrityError):
    """A missing, masked, fractional, out-of-range, negative or duplicated observation key."""


def _int_key(raw: Any, label: str, n: Optional[int]) -> np.ndarray:
    if raw is None:
        raise ObservationKeyRefusal(f"observation key column {label!r} is missing")
    masked = np.ma.asarray(raw)
    if masked.ndim != 1 or (n is not None and masked.shape[0] != n):
        raise ObservationKeyRefusal(f"observation key {label!r} must be a one-dimensional ({n},) column, "
                                    f"got shape {masked.shape}")
    if np.ma.getmaskarray(masked).any():
        raise ObservationKeyRefusal(f"observation key {label!r} has masked (missing) values")
    values = np.asarray(masked.data if np.ma.isMaskedArray(masked) else masked)
    kind = values.dtype.kind
    if kind == "b" or kind not in "iuf":
        raise ObservationKeyRefusal(f"observation key {label!r} must be integer-valued, got dtype {values.dtype}")
    if kind == "f":
        if not np.isfinite(values).all():
            raise ObservationKeyRefusal(f"observation key {label!r} has non-finite values")
        if np.any(values != np.floor(values)):
            raise ObservationKeyRefusal(f"observation key {label!r} has fractional values; refusing to truncate")
        if np.any(np.abs(values) >= _INT64_LIMIT):
            raise ObservationKeyRefusal(f"observation key {label!r} is outside the int64 range")
    elif kind == "u" and values.size and int(values.max()) > np.iinfo(np.int64).max:
        raise ObservationKeyRefusal(f"observation key {label!r} is outside the int64 range")
    out = values.astype(np.int64)
    if np.any(out < 0):
        raise ObservationKeyRefusal(f"observation key {label!r} has negative values")
    return out


def validate_observation_keys(phase: Any, replica: Any, step: Any, *, segment_ids: Any = None,
                              where: str = "sample pool"):
    """Validate (phase/source id, replica, step) columns; return ``(phase_code, replica, step)`` int64 arrays.

    ``phase`` is a per-row phase/source identifier (any hashable, nonempty; None = the whole pool is one
    phase) and is returned as dense int64 codes (order of first sort, stable within a call). ``replica`` and
    ``step`` must be present, unmasked, integer-valued (fractional values refuse, never truncate), within
    int64 and non-negative. A repeated key refuses -- identical or conflicting, there is no silent dedupe.
    ``segment_ids`` is provenance only, named in the refusal message. Raises ObservationKeyRefusal.
    """
    replica_a = _int_key(replica, "replica", None)
    step_a = _int_key(step, "step", replica_a.shape[0])
    n = step_a.shape[0]
    if phase is None:
        phase_code = np.zeros(n, dtype=np.int64)
    else:
        masked = np.ma.asarray(phase)
        if masked.ndim != 1 or masked.shape[0] != n:
            raise ObservationKeyRefusal(f"phase/source id must be a ({n},) column, got shape {masked.shape}")
        if np.ma.getmaskarray(masked).any():
            raise ObservationKeyRefusal("phase/source id has masked (missing) values")
        labels = [str(v) for v in np.asarray(masked.data if np.ma.isMaskedArray(masked) else masked).tolist()]
        if any(not v for v in labels):
            raise ObservationKeyRefusal("phase/source id has empty values")
        _, phase_code = np.unique(np.asarray(labels, dtype=object).astype(str), return_inverse=True)
        phase_code = np.asarray(phase_code, dtype=np.int64).reshape(-1)
    if n > 1:
        keys = np.stack([phase_code, replica_a, step_a], axis=1)
        order = np.lexsort((step_a, replica_a, phase_code))
        sk = keys[order]
        same = np.flatnonzero(np.all(sk[1:] == sk[:-1], axis=1))
        if same.size:
            a, b = sorted((int(order[same[0]]), int(order[same[0] + 1])))
            segs = ""
            if segment_ids is not None:
                seg = np.asarray(segment_ids, dtype=object)
                segs = f" in segment(s) {sorted({str(seg[a]), str(seg[b])})}"
            raise ObservationKeyRefusal(
                f"duplicate (phase, replica, step) observation in the {where}: step {int(step_a[a])}, replica "
                f"{int(replica_a[a])}{segs}; a crashed parent without a checkpoint and its restarted child "
                "overlap -- repair the segment registry (seal the parent abandoned) before pooling")
    return phase_code, replica_a, step_a


def lambda_equals_frozen(stored: Any, frozen: Any) -> np.ndarray:
    """Row-wise: does each stored lambda equal its frozen float64 value rounded to the STORED dtype?

    A float32 column holds float32(frozen), so exact float64 equality refuses 0.1; comparing
    ``np.float32(frozen) == stored`` is exact for the stored precision and keeps adjacent distinct
    float32 rungs distinct. float64 (or any other) columns compare exactly. The caller keeps using
    the frozen float64 value afterwards.
    """
    arr = np.asarray(np.ma.getdata(stored))
    frozen_a = np.asarray(frozen, dtype=np.float64)
    if arr.dtype.kind == "f" and arr.dtype.itemsize < 8:
        return arr == frozen_a.astype(arr.dtype)
    return arr.astype(np.float64) == frozen_a
