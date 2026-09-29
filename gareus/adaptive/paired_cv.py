"""Row-paired (CV1, CV2) data for the adaptive-production diagnostics (spec P4).

The epoch, segmented-epoch and final-combined collectors summarise each state
with independent per-axis statistics (``cv_mean``/``secondary_mean`` ... each
over its own finite values), so CV1 and CV2 are not row-paired, the segmented
state rows carry no spread at all, and every edge ``overlap`` is the CV1
marginal.  This module adds, next to those unchanged keys:

Per state (``states[i]["paired_cv"]``, a dict):

* ``restraint``: ``primary_center``, ``primary_k``, ``secondary_center``,
  ``secondary_k``, ``gamd_lambda`` from the registry, plus ``cv1_restrained`` /
  ``cv2_restrained`` (the P6 rule: k None = restrained, CV2 needs a centre).
  Units and form are ``gareus.query.reconstruct_bias_matrix``'s:
  ``U = 0.5*k1*(cv1-c1)^2 + 0.5*k2*(cv2-c2)^2`` in kcal/mol, k in kcal/mol per
  squared CV unit.  The λ boost is NOT a function of the CVs, so between two
  states on one rung it cancels (spec 3.1).
* ``n_rows``: rows mapped to the state; ``n_pairs``: rows kept as pairs (finite
  CV1, and finite CV2 when the state restrains CV2); ``n_cv1_nonfinite_dropped``,
  ``n_cv2_nonfinite_dropped`` (the latter only counts for a CV2-restrained state)
  and ``retained_fraction`` = n_pairs / n_rows.
* ``cv1`` / ``cv2``: moments over the kept pairs, each ``{"n", "mean", "var",
  "skewness", "kurtosis_excess"}`` -- population moments (var ddof=0, skewness
  m3/m2^1.5, excess kurtosis m4/m2^2 - 3), None where undefined (n < 2 or var 0).
  ``cv2`` is None for a state with no finite CV2 among its pairs (CV1-only):
  a missing CV2 is never read as 0.0.  ``cov_cv1_cv2`` / ``corr_cv1_cv2`` over
  the pairs with both finite.
* ``subsample``: ``{"n_total", "n_kept", "stride", "method"}`` for the bounded
  pairs written to the sidecar NPZ.

Per edge (``edges[i]``): ``overlap_joint_2d`` -- histogram overlap
sum(min(p_i, p_j)) over a PAIR-LOCAL (CV1, CV2) grid of
``JOINT_OVERLAP_BINS`` x ``JOINT_OVERLAP_BINS`` cells spanning both states'
finite pairs (padded 2 %), computed from ALL pooled pairs (never from the
subsample, never from per-segment values).  None with ``overlap_joint_2d_reason``
for a rung edge (``rung_edge``: CV overlap ~1 by construction), a pair where
either state does not restrain CV2 (``cv2_unrestrained``), or fewer than
``MIN_PAIRS_FOR_OVERLAP`` pairs with finite CV2 on either side
(``too_few_pairs``).  The existing ``overlap`` key (CV1 marginal) is untouched.

Payload (``payload["paired_cv"]``): ``{"status": "ok"|"error", "error",
"npz", "max_pairs_per_state", "subsample_method", "n_states"}``.  The NPZ
(``<json stem>_paired_cv.npz`` next to the diagnostics JSON) holds flat arrays
``state_id`` (int64, per pair), ``cv1``, ``cv2`` (float64, NaN = not measured),
``source_index`` (int32, index into ``sources``), ``row_index`` (int64,
position in that state's pooled kept rows) and ``step`` (int64, the row's MD
step, -1 if not recorded).  Rows are pooled source by source in the
collector's source order, each source sorted by step (stable) if it arrived
out of order; the constant stride keeps that order and the source boundaries,
so spec 3.1 can block the pairs for tau within a source.  Read it
with :func:`load_paired_subsamples`.

Everything here is diagnostics: any failure is caught and reported as
``status: "error"``; it never raises into a collector, so it can never kill a
completed MD epoch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MAX_PAIRS_PER_STATE = 2000
SUBSAMPLE_METHOD = "even_stride_time_order_v1"
JOINT_OVERLAP_BINS = 30
MIN_PAIRS_FOR_OVERLAP = 5


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _axis_restrained(k: Any) -> bool:
    """k None = not recorded = the run's default restraint (same rule as P6)."""
    if k is None:
        return True
    try:
        k = float(k)
    except (TypeError, ValueError):
        return True
    return bool(math.isfinite(k) and k > 0.0)


def axis_moments(values: np.ndarray) -> Optional[Dict[str, Any]]:
    """Population moments of the finite entries; None when there are none."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return None
    mean = float(np.mean(x))
    d = x - mean
    m2 = float(np.mean(d * d))
    out: Dict[str, Any] = {"n": n, "mean": mean, "var": m2, "skewness": None, "kurtosis_excess": None}
    if n >= 2 and m2 > 0.0:
        out["skewness"] = float(np.mean(d ** 3) / m2 ** 1.5)
        out["kurtosis_excess"] = float(np.mean(d ** 4) / (m2 * m2) - 3.0)
    return out


def stride_indices(n: int, max_kept: int) -> Tuple[np.ndarray, int]:
    """Every ``stride``-th index from 0, at most ``max_kept`` of them, time order kept."""
    if n <= 0 or max_kept <= 0:
        return np.zeros(0, dtype=np.int64), 0
    stride = max(1, int(math.ceil(n / float(max_kept))))
    return np.arange(0, n, stride, dtype=np.int64)[:max_kept], stride


def joint_overlap_2d(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                     bins: int = JOINT_OVERLAP_BINS) -> Optional[float]:
    """sum(min(p_a, p_b)) over a pair-local 2D grid; None below MIN_PAIRS_FOR_OVERLAP."""
    ma = np.isfinite(a1) & np.isfinite(a2)
    mb = np.isfinite(b1) & np.isfinite(b2)
    if int(ma.sum()) < MIN_PAIRS_FOR_OVERLAP or int(mb.sum()) < MIN_PAIRS_FOR_OVERLAP:
        return None
    a1, a2, b1, b2 = a1[ma], a2[ma], b1[mb], b2[mb]
    ranges = []
    for u, v in ((a1, b1), (a2, b2)):
        lo = float(min(u.min(), v.min()))
        hi = float(max(u.max(), v.max()))
        if hi <= lo:
            # One axis collapsed to a single value for both states: every
            # sample shares that coordinate, so it does not separate them.
            hi = lo + 1.0
        pad = 0.02 * (hi - lo)
        ranges.append((lo - pad, hi + pad))
    ha, _, _ = np.histogram2d(a1, a2, bins=int(bins), range=ranges)
    hb, _, _ = np.histogram2d(b1, b2, bins=int(bins), range=ranges)
    return float(np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


class PairedCVCollector:
    """Accumulate row-paired (CV1, CV2) per state over one or more sources.

    ``add_rows(state_id, rows, source)`` takes the collectors' row dicts
    (``cv_A``/``primary_cv_value`` and ``secondary_cv``) in time order; call it
    once per (source, state) in a deterministic source order.  Holds the full
    pairs only for the lifetime of one collector call (the edge overlap needs
    them); only the bounded stride subsample is persisted.
    """

    def __init__(self, registry: Any, max_pairs_per_state: Optional[int] = None) -> None:
        self.registry = registry
        self.max_pairs = int(MAX_PAIRS_PER_STATE if max_pairs_per_state is None else max_pairs_per_state)
        self.sources: List[str] = []
        self._chunks: Dict[int, List[Tuple[np.ndarray, np.ndarray, int]]] = {}
        self._n_rows: Dict[int, int] = {}
        self._pooled: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]] = {}
        self._steps: Dict[int, np.ndarray] = {}
        self.error: Optional[str] = None

    def _source_index(self, source: str) -> int:
        if source not in self.sources:
            self.sources.append(str(source))
        return self.sources.index(str(source))

    def add_rows(self, state_id: int, rows: Iterable[Mapping[str, Any]], source: str = "") -> None:
        if self.error is not None:
            return
        try:
            rows = list(rows)
            sid = int(state_id)
            cv1 = np.fromiter((_to_float(r.get("cv_A", r.get("primary_cv_value"))) for r in rows),
                              dtype=float, count=len(rows))
            cv2 = np.fromiter((_to_float(r.get("secondary_cv")) for r in rows), dtype=float, count=len(rows))
            step = np.fromiter((_to_float(r.get("step")) for r in rows), dtype=float, count=len(rows))
            if step.size and np.all(np.isfinite(step)) and np.any(np.diff(step) < 0):
                # Keep each source time-ordered so the stride subsample can be
                # blocked (spec 3.1); stable, so equal steps keep read order.
                order = np.argsort(step, kind="stable")
                cv1, cv2, step = cv1[order], cv2[order], step[order]
            self._n_rows[sid] = self._n_rows.get(sid, 0) + len(rows)
            self._chunks.setdefault(sid, []).append((cv1, cv2, self._source_index(source), step))
            self._pooled.pop(sid, None)
        except Exception as exc:  # diagnostics never kill a collector
            self.error = f"{type(exc).__name__}: {exc}"

    def _restraint(self, sid: int) -> Dict[str, Any]:
        state = self.registry.get_state(int(sid)) if self.registry is not None else None
        if state is None:
            return {"primary_center": None, "primary_k": None, "secondary_center": None,
                    "secondary_k": None, "gamd_lambda": None,
                    "cv1_restrained": None, "cv2_restrained": None}
        sc = state.secondary_center
        return {
            "primary_center": float(state.primary_center),
            "primary_k": None if state.primary_k is None else float(state.primary_k),
            "secondary_center": None if sc is None else float(sc),
            "secondary_k": None if state.secondary_k is None else float(state.secondary_k),
            "gamd_lambda": float(state.gamd_lambda or 0.0),
            "cv1_restrained": _axis_restrained(state.primary_k),
            "cv2_restrained": bool(sc is not None and _axis_restrained(state.secondary_k)),
        }

    def _pairs(self, sid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """Kept pairs (cv1, cv2, source index) in time order, plus drop counts.

        The kept rows' steps are stored in ``self._steps[sid]`` (NaN = not recorded).
        """
        cached = self._pooled.get(int(sid))
        if cached is not None:
            return cached
        chunks = self._chunks.get(int(sid), [])
        if not chunks:
            empty = np.zeros(0, dtype=float)
            self._steps[int(sid)] = empty
            return empty, empty, np.zeros(0, dtype=np.int32), {"cv1": 0, "cv2": 0}
        cv1 = np.concatenate([c[0] for c in chunks])
        cv2 = np.concatenate([c[1] for c in chunks])
        src = np.concatenate([np.full(c[0].size, c[2], dtype=np.int32) for c in chunks])
        steps = np.concatenate([c[3] for c in chunks])
        keep = np.isfinite(cv1)
        drops = {"cv1": int((~keep).sum()), "cv2": 0}
        if self._restraint(sid).get("cv2_restrained"):
            # A CV2-restrained state's bias needs CV2: a pair without it would
            # be a fabricated on-target sample (the 2026-08-15 chain), so drop it.
            bad2 = keep & ~np.isfinite(cv2)
            drops["cv2"] = int(bad2.sum())
            keep &= ~bad2
        self._steps[int(sid)] = steps[keep]
        out = (cv1[keep], cv2[keep], src[keep], drops)
        self._pooled[int(sid)] = out
        return out

    def state_summary(self, sid: int) -> Dict[str, Any]:
        restraint = self._restraint(sid)
        cv1, cv2, _src, drops = self._pairs(sid)
        n_rows = int(self._n_rows.get(int(sid), 0))
        both = np.isfinite(cv1) & np.isfinite(cv2)
        cov = corr = None
        if int(both.sum()) >= 2:
            x, y = cv1[both], cv2[both]
            cov = float(np.mean((x - x.mean()) * (y - y.mean())))
            sx, sy = float(np.std(x)), float(np.std(y))
            corr = float(cov / (sx * sy)) if sx > 0.0 and sy > 0.0 else None
        idx, stride = stride_indices(int(cv1.size), self.max_pairs)
        return {
            "restraint": restraint,
            "n_rows": n_rows,
            "n_pairs": int(cv1.size),
            "n_cv1_nonfinite_dropped": drops["cv1"],
            "n_cv2_nonfinite_dropped": drops["cv2"],
            "retained_fraction": (float(cv1.size) / n_rows) if n_rows > 0 else None,
            "cv1": axis_moments(cv1),
            "cv2": axis_moments(cv2),
            "cov_cv1_cv2": cov,
            "corr_cv1_cv2": corr,
            "subsample": {"n_total": int(cv1.size), "n_kept": int(idx.size),
                          "stride": int(stride), "method": SUBSAMPLE_METHOD},
        }

    def edge_overlap(self, si: int, sj: int, edge_type: str) -> Tuple[Optional[float], Optional[str]]:
        if str(edge_type) == "rung":
            return None, "rung_edge"
        if not (self._restraint(si).get("cv2_restrained") and self._restraint(sj).get("cv2_restrained")):
            return None, "cv2_unrestrained"
        a1, a2, _, _ = self._pairs(si)
        b1, b2, _, _ = self._pairs(sj)
        value = joint_overlap_2d(a1, a2, b1, b2)
        return (value, None) if value is not None else (None, "too_few_pairs")

    def write_npz(self, path: Path, state_ids: Sequence[int]) -> None:
        sid_parts, c1_parts, c2_parts, src_parts, row_parts, step_parts = [], [], [], [], [], []
        for sid in state_ids:
            cv1, cv2, src, _ = self._pairs(int(sid))
            idx, _stride = stride_indices(int(cv1.size), self.max_pairs)
            sid_parts.append(np.full(idx.size, int(sid), dtype=np.int64))
            c1_parts.append(cv1[idx])
            c2_parts.append(cv2[idx])
            src_parts.append(src[idx])
            row_parts.append(idx)
            steps = self._steps[int(sid)][idx]
            step_parts.append(np.where(np.isfinite(steps), steps, -1).astype(np.int64))

        def _cat(parts, dtype):
            return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

        path = Path(path)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            state_id=_cat(sid_parts, np.int64), cv1=_cat(c1_parts, np.float64),
            cv2=_cat(c2_parts, np.float64), source_index=_cat(src_parts, np.int32),
            row_index=_cat(row_parts, np.int64), step=_cat(step_parts, np.int64), sources=np.asarray(self.sources, dtype=str),
            max_pairs_per_state=np.asarray(self.max_pairs), method=np.asarray(SUBSAMPLE_METHOD),
        )
        tmp.replace(path)


def attach_paired_cv(payload: Dict[str, Any], collector: Optional[PairedCVCollector],
                     json_path: Path) -> Dict[str, Any]:
    """Add ``paired_cv`` to every state row, ``overlap_joint_2d`` to every edge,
    and the payload-level ``paired_cv`` record; write the sidecar NPZ.

    Only ever ADDS keys.  Any failure leaves the existing keys as computed and
    records ``payload["paired_cv"] = {"status": "error", ...}``.
    """
    npz_path = Path(json_path).with_name(Path(json_path).stem + "_paired_cv.npz")
    record: Dict[str, Any] = {"status": "ok", "error": None, "npz": str(npz_path),
                              "max_pairs_per_state": MAX_PAIRS_PER_STATE,
                              "subsample_method": SUBSAMPLE_METHOD, "n_states": 0}
    try:
        if collector is None:
            raise RuntimeError("paired-CV collector was not built")
        if collector.error is not None:
            raise RuntimeError(collector.error)
        record["max_pairs_per_state"] = int(collector.max_pairs)
        summaries: Dict[int, Dict[str, Any]] = {}
        for row in payload.get("states", []) or []:
            sid = int(row.get("state_id"))
            summaries[sid] = collector.state_summary(sid)
        for edge in payload.get("edges", []) or []:
            value, reason = collector.edge_overlap(int(edge.get("state_i")), int(edge.get("state_j")),
                                                   str(edge.get("edge_type", "")))
            edge["overlap_joint_2d"] = value
            edge["overlap_joint_2d_reason"] = reason
        collector.write_npz(npz_path, sorted(summaries))
        # States are written last so a failure above leaves them untouched.
        for row in payload.get("states", []) or []:
            row["paired_cv"] = summaries[int(row.get("state_id"))]
        record["n_states"] = len(summaries)
    except Exception as exc:  # diagnostics never kill a completed epoch
        record.update(status="error", error=f"{type(exc).__name__}: {exc}", npz=None)
    payload["paired_cv"] = record
    return payload


def load_paired_subsamples(npz_path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """``{state_id: {"cv1", "cv2", "source_index", "row_index", "step"}}`` from a sidecar NPZ."""
    with np.load(Path(npz_path), allow_pickle=False) as data:
        sid = np.asarray(data["state_id"])
        cols = {k: np.asarray(data[k]) for k in ("cv1", "cv2", "source_index", "row_index", "step")}
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for s in np.unique(sid):
        mask = sid == s
        out[int(s)] = {k: v[mask] for k, v in cols.items()}
    return out
