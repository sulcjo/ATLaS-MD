"""Descriptive table of auxiliary workers for the analysis report (CVaux adaptive, Task 15).

Purely descriptive: no sham states exist, so nothing here is an attribution of benefit to a worker.
Functions take explicit arrays (or a Data-alike) so they are testable without the loaders.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np

OVERLAP_FLOOR = 0.15
ATTRIBUTION = "none (no shams)"
LABEL_WINDOW_FRAMES = 5


def positive_residence_episodes(series, residence, aux_states: Iterable[int]) -> dict:
    """Entries/exits of aux states on one replica's state series (exchange grid, time order).

    ``residence[i]`` is the MD time after assignment i. Zero-residence aux visits (swapped in and out
    with no MD in between) are removed from the series and counted separately in ``zero_time_visits``;
    an entry is an ordinary -> aux transition (of the collapsed series) that is followed by positive
    residence, an exit the aux -> ordinary transition that follows."""
    aux = {int(a) for a in aux_states}
    series = np.asarray(series).astype(np.int64)
    residence = np.asarray(residence, dtype=float)
    keep = np.ones(series.size, bool)
    zero = 0
    for i in range(series.size):
        if residence[i] <= 0.0:
            keep[i] = False
            if int(series[i]) in aux:
                zero += 1
    s = series[keep]
    entries = exits = 0
    for a, b in zip(s[:-1], s[1:]):
        a_aux, b_aux = int(a) in aux, int(b) in aux
        if not a_aux and b_aux:
            entries += 1
        elif a_aux and not b_aux:
            exits += 1
    return {"entries": entries, "exits": exits, "zero_time_visits": zero}


def _majority(labels: np.ndarray) -> Optional[int]:
    if labels.size == 0:
        return None
    return int(np.bincount(labels).argmax())


def return_label_change_fraction(frames, eval_partition, worker_state: int,
                                 n_frames: int = LABEL_WINDOW_FRAMES) -> dict:
    """Fraction of worker episodes after which the (frozen evaluation partition) majority label of the
    frames around the episode differs, before entry vs after exit, within one (phase, replica) lineage."""
    labels = np.asarray(eval_partition.predict(np.hstack([frames.hc, frames.hb]),
                                               np.c_[frames.cv1, frames.cv2]))
    lineage = frames.lineage
    n_ep = n_change = 0
    for lin in np.unique(lineage):
        idx = np.where(lineage == lin)[0]
        idx = idx[np.argsort(frames.step[idx], kind="stable")]
        in_w = np.asarray(frames.state_id)[idx] == worker_state
        j = 0
        while j < idx.size:
            if not in_w[j]:
                j += 1
                continue
            k = j
            while k < idx.size and in_w[k]:
                k += 1
            pre = _majority(labels[idx[max(0, j - n_frames):j]])
            post = _majority(labels[idx[k:k + n_frames]])
            if pre is not None and post is not None:
                n_ep += 1
                n_change += int(pre != post)
            j = k
    return {"fraction": (n_change / n_ep) if n_ep else None, "n_episodes": n_ep}


def worker_table(d: Any, f_k, *, aux_states, ordinary_states, eval_partition=None, frames=None,
                 state_lambdas=None) -> List[dict]:
    from gareus.mbar_analysis.ladder import pairwise_state_overlap

    window = np.asarray(d.window, dtype=np.int64)
    replica = np.asarray(d.replica)
    step = np.asarray(d.step)
    u_nk = np.asarray(d.u_nk)
    f_k = np.asarray(f_k, float)
    n_k = np.bincount(window, minlength=u_nk.shape[1])
    z = getattr(d, "aux_z", None)
    if z is None:
        z = (getattr(d, "meta", None) or {}).get("aux_z")
    z = None if z is None else np.asarray(z, float)
    meta = getattr(d, "meta", None) or {}
    parents = meta.get("aux_parent_state", {}) or {}
    forecasts = meta.get("aux_forecast_O", {}) or {}
    aux_set = {int(a) for a in aux_states}
    ordinary = [int(o) for o in ordinary_states]
    if state_lambdas is None:
        state_lambdas = getattr(d, "state_lambdas", None)
    if state_lambdas is not None:
        lam = np.asarray(state_lambdas, float)
        ordinary = [o for o in ordinary if o < lam.size and lam[o] == 0.0]
    by_replica = []
    for r in np.unique(replica):
        idx = np.where(replica == r)[0]
        by_replica.append(idx[np.argsort(step[idx], kind="stable")])
    rows: List[dict] = []
    for w in sorted(aux_set):
        mine = window == w
        best, best_o = None, None
        for o in ordinary:
            if n_k[o] == 0 or n_k[w] == 0:
                continue
            ov = float(pairwise_state_overlap(u_nk, window, f_k, n_k, int(w), o))
            if best_o is None or ov > best_o:
                best, best_o = o, ov
        entries = exits = 0
        for idx in by_replica:
            ep = positive_residence_episodes(window[idx], np.ones(idx.size), {w})
            entries += ep["entries"]; exits += ep["exits"]
        zw = z[mine] if z is not None else np.array([])
        zw = zw[np.isfinite(zw)]
        row = {
            "state_id": int(w),
            "parent_state_id": parents.get(int(w), parents.get(str(w))),
            "best_partner": best,
            "best_partner_overlap": best_o,
            "overlap_floor_ok": bool(best_o is not None and best_o >= OVERLAP_FLOOR),
            "n_samples": int(mine.sum()),
            "occupancy_fraction": float(mine.sum() / max(1, window.size)),
            "entries": int(entries),
            "exits": int(exits),
            "carrier_diversity": int(np.unique(replica[mine]).size),
            "z_mean": float(zw.mean()) if zw.size else None,
            "z_sd": float(zw.std()) if zw.size else None,
            "forecast_O": forecasts.get(int(w), forecasts.get(str(w))),
            "attribution": ATTRIBUTION,
        }
        if frames is not None and eval_partition is not None:
            rc = return_label_change_fraction(frames, eval_partition, int(w))
            row["return_label_change_fraction"] = rc["fraction"]
            row["return_label_episodes"] = rc["n_episodes"]
        else:
            row["return_label_change_fraction"] = None
            row["return_label_status"] = "frames_unavailable"
        rows.append(row)
    return rows
