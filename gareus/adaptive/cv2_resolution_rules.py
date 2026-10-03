"""Per-rule candidate builders for spec 3.3: R1 (CV2-gap bridge) and R3 (mode resolution).

Pure: they read a diagnostics payload, the P4 subsamples, a transition-run provider and the
edge history, and return report candidates (``cv2_resolution.new_candidate`` records) with
their proposals. Budget, actions and I/O live in ``cv2_resolution_io``. See the
``cv2_resolution`` module docstring for the rules themselves.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gareus.adaptive import cv2_resolution as cr

RunsProvider = Callable[[Sequence[int]], Mapping[int, Mapping[str, Any]]]


# ---- R3 ----------------------------------------------------------------------------------

def _r3_metrics(view: cr.StateView, settings: cr.ResolutionSettings) -> Dict[str, Any]:
    sd = None if view.var2 is None else float(np.sqrt(max(view.var2, 0.0)))
    sw = view.sigma_w(2, settings.rt)
    return {"n_pairs": view.n_pairs, "cv2_mean": view.mean2, "cv2_sd": sd, "sigma_w2": sw,
            "sd_over_sigma_w": None if sd is None or not sw else sd / sw,
            "sarle_bimodality": cr.sarle_bimodality(view.moments2), "diagnostics_trigger": False}


def _r3_children(view: cr.StateView, modes: Mapping[str, Any], settings: cr.ResolutionSettings,
                 gate: Any) -> List[Dict[str, Any]]:
    lo, hi = modes["pair"]
    pooled = float(modes["fit"]["pooled_variance"])
    sigma_target = 2.0 * (0.5 * (float(hi["mean"]) - float(lo["mean"]))) / cr.SPACING_SIGMA
    kids = []
    for comp in (lo, hi):
        # The de-regularised variance (cv2_shape.curvature_variance); a v1 record has only "variance".
        var_c = comp.get("variance_curvature")
        f2 = cr.f2_under_bias(comp["variance"] if var_c is None else var_c, view.k2, settings.temperature_k,
                              pooled_var=pooled, n_members=comp["n_members"])
        child = {"primary_center": view.c1, "primary_k": float(view.k1), "secondary_center": float(comp["mean"]),
                 "centred_on": "mixture_mode", **cr.child_spring(sigma_target, f2, float(view.k2), settings)}
        kids.append(cr.gate_child(child, view.c1, float(view.k1), float(comp["mean"]), gate,
                                  f"cv2_resolution R3 child of {view.state_id}", settings))
    return kids


def _transitions(rec: Optional[Mapping[str, Any]], bounds: Tuple[float, float],
                 count: str = "replica-path") -> Dict[str, Any]:
    """Crossings R3 decides on, from a runs record {source, state_runs, replica_runs,
    replica_paths} (``cv2_resolution_io.parquet_runs_provider``). All three are
    ``count_transitions`` (core-to-core label changes, samples between the cores ignored):

    * "replica": within one replica's contiguous residence at the window (broken at every
      source, segment, step gap, NaN and change of replica), so an exchange swap is never a
      crossing. Under exchange (c9: ~2.5 samples per residence) it almost never reaches 10.
    * "replica-path" (default): each replica's own visits to the window joined in time order
      across its absences, one path per (source, segment, replica), NaN dropped, never broken
      at a step gap (every absence IS a step gap in the replica's own visits). A swap relabels
      the window but never moves coordinates, so every counted change is a crossing by that
      replica's own dynamics, possibly made while it sat in another window. The T2 harness
      estimator (t2c_eval.replica_path_count) and t3c_replica_path_c9.py; c9 epoch_002
      84 / 200 / 220: 22 / 12 / 21.
    * "state-series": every switch of the state-indexed series, swaps included.

    Ordering. replica <= replica-path and replica <= state-series always: a residence is a
    contiguous piece of both the replica's path and the state series (same source/segment
    breaks), and removing the boundaries between pieces can only add label changes. replica-path
    and state-series are NOT ordered: two walkers each fixed in one mode and alternating
    residences give state-series > 0 = replica-path, and a state series A A B B whose two
    replicas each go A -> B gives state-series 1 < replica-path 2 (pinned by tests). Under
    exchange state-series counts swap frequency (c9 370-469 vs replica-path 12-22).

    Every count needs a replica-resolved Parquet source; without one ``transitions`` is None
    and ``transitions_state_series_lower_bound`` (report v2+; v1 called it
    ``transitions_lower_bound``) holds the state-series count that exists -- from the full
    series when the Parquet has no replica column, 0 when no series was read -- a lower bound
    on the state-series count, an upper bound on the replica count only."""
    if count not in cr.TRANSITION_COUNTS:
        raise ValueError(f"refine_transition_count must be one of {cr.TRANSITION_COUNTS}, got {count!r}")
    rec = rec or {}
    src = str(rec.get("source", "none"))
    out = {"transitions_source": src, "core_bounds": list(bounds), "transitions_estimator": count,
           "transitions_state_series": cr.count_transitions(rec.get("state_runs", []), *bounds)}
    if src == "parquet":
        out["transitions_replica"] = cr.count_transitions(rec.get("replica_runs", []), *bounds)
        paths = rec.get("replica_paths")
        out["transitions_replica_path"] = None if paths is None else cr.count_transitions(paths, *bounds)
        out["transitions"] = {"replica": out["transitions_replica"], "replica-path": out["transitions_replica_path"],
                              "state-series": out["transitions_state_series"]}[count]
    else:
        out["transitions"] = None
    if out["transitions"] is None:
        out["transitions_state_series_lower_bound"] = out["transitions_state_series"]
    return out


_ESTIMATOR_WORDS = {"replica": "within-residence", "replica-path": "replica-path", "state-series": "state-series"}


def _r3_decide(view: cr.StateView, modes: Dict[str, Any], rec: Optional[Mapping[str, Any]],
               settings: cr.ResolutionSettings, gate: Any) -> Dict[str, Any]:
    metrics = {**_r3_metrics(view, settings), "mixture": modes["fit"], "depth": modes["depth"],
               "modes": modes["pair"], "trapped_or_orthogonal": False}
    lo, hi = modes["pair"]
    metrics.update(_transitions(rec, cr.core_bounds(lo, hi, float(modes["depth"]["barrier_z"])),
                                str(settings.refine_transition_count)))
    ids = [view.state_id]
    metrics["r3_gate_values"] = {**(modes.get("gate_values") or {}), "transitions": metrics["transitions"],
                                 "min_transitions": int(settings.refine_min_transitions),
                                 "transitions_estimator": metrics["transitions_estimator"]}
    if metrics["transitions"] is None:
        metrics["r3_gate"] = "no_replica_series"
        return cr.new_candidate("R3", "state", ids, "flagged", "transitions_unavailable: no replica-resolved "
                                "series (the recorded count is a state-series lower bound only)", metrics=metrics)
    if metrics["transitions"] < int(settings.refine_min_transitions):
        metrics["r3_gate"] = "transitions_below_min"
        metrics["trapped_or_orthogonal"] = True
        return cr.new_candidate("R3", "state", ids, "flagged",
                                f"trapped_or_orthogonal: bimodal (depth {modes['depth']['depth_kT']:.2f} kT) but "
                                f"{metrics['transitions']} < {settings.refine_min_transitions} "
                                f"{_ESTIMATOR_WORDS[str(settings.refine_transition_count)]} "
                                "transitions", metrics=metrics)
    metrics["r3_gate"] = "passed"
    kids = _r3_children(view, modes, settings, gate)
    proposal = {"parent_state_id": view.state_id, "children": kids}
    refusal = next((k["refusal"] for k in kids if k.get("refusal")), None)
    metrics["would_be"] = {"decision": "refused" if refusal else "proposed", "refusal": refusal,
                           "r3_mode": str(settings.refine_r3_mode)}
    if refusal:
        return cr.new_candidate("R3", "state", ids, "refused", f"{refusal}: a child spring cannot be placed",
                                refusal=refusal, metrics=metrics, proposal=proposal)
    what = (f"two modes {lo['mean']:.4g} / {hi['mean']:.4g} (depth {modes['depth']['depth_kT']:.2f} kT, "
            f"{metrics['transitions']} transitions)")
    if str(settings.refine_r3_mode) == "flag":
        return cr.new_candidate("R3", "state", ids, "flagged", f"{cr.R3_FLAG_ONLY}: {what} would get two windows "
                                "(refine_r3_mode flag: recorded, never inserted)", metrics=metrics, proposal=proposal)
    return cr.new_candidate("R3", "state", ids, "proposed", f"{what}; parent kept", metrics=metrics,
                            proposal=proposal)


def propose_r3(views: Mapping[int, cr.StateView], rep_ids: Sequence[int],
               subsamples: Mapping[int, Mapping[str, np.ndarray]], runs_for: Optional[RunsProvider],
               settings: cr.ResolutionSettings, epoch: int, gate: Any = None) -> List[Dict[str, Any]]:
    """R3 candidates for every eligible representative-rung state (one record per state analysed)."""
    out: List[Dict[str, Any]] = []
    bimodal: Dict[int, Dict[str, Any]] = {}
    for sid in rep_ids:
        view = views[sid]
        ok, why = cr.eligibility(view, settings)
        if not ok:
            continue
        if cr.is_protected(view, epoch, settings):
            out.append(cr.new_candidate("R3", "state", [sid], "skipped", "protected: created by a resolution "
                                        f"action within refine_protect_epochs={settings.refine_protect_epochs}"))
            continue
        sub = subsamples.get(sid) or {}
        z = np.asarray(sub.get("cv2", []), dtype=float)
        if np.isfinite(z).sum() < 3 * cr.MEMBER_BLOCKS:
            out.append(cr.new_candidate("R3", "state", [sid], "skipped", "too_few_samples",
                                        metrics={**_r3_metrics(view, settings), "r3_gate": "too_few_samples",
                                                 "r3_gate_values": {"n_finite": int(np.isfinite(z).sum()),
                                                                    "min_finite": 3 * cr.MEMBER_BLOCKS}}))
            continue
        keep = np.isfinite(z)
        src = sub.get("source_index")
        modes = cr.mode_analysis(z[keep], None if src is None else np.asarray(src)[keep], seed=sid)
        if not modes["bimodal"]:
            out.append(cr.new_candidate("R3", "state", [sid], "no_action", "no resolvable mode pair "
                                        f"(depth >= 1 kT, both >= 10 %): {modes['gate']}",
                                        metrics={**_r3_metrics(view, settings), "mixture": modes["fit"],
                                                 "r3_gate": modes["gate"], "r3_gate_values": modes["gate_values"]}))
            continue
        bimodal[sid] = modes
    runs = dict(runs_for(sorted(bimodal)) or {}) if (runs_for is not None and bimodal) else {}
    for sid, modes in bimodal.items():
        out.append(_r3_decide(views[sid], modes, runs.get(sid), settings, gate))
    return out


# ---- R1 ----------------------------------------------------------------------------------

def _component_map(edge_metric: Mapping[str, Any]) -> Dict[int, int]:
    comps = ((edge_metric.get("components") or {}).get("components")) or []
    return {int(s): i for i, comp in enumerate(comps) for s in comp}


def _edge_metrics(edge: Mapping[str, Any], d1, d2, comp_of) -> Dict[str, Any]:
    pm = edge.get("pairwise_mbar") or {}
    si, sj = int(edge["state_i"]), int(edge["state_j"])
    return {"edge_type": edge.get("edge_type"), "overlap": pm.get("overlap"),
            "overlap_lower": pm.get("overlap_lower"), "overlap_upper": pm.get("overlap_upper"),
            "union_overlap": edge.get("mbar_overlap"), "pairwise_status": pm.get("status"),
            "pairwise_reason": pm.get("reason"), "n_eff": pm.get("n_eff"), "d_cv1": d1, "d_cv2": d2,
            "components": [comp_of.get(si), comp_of.get(sj)], "cv1_marginal_overlap": edge.get("overlap")}


def _r1_edges(payload, views, rep, settings):
    """(edge, a, b, d1, d2) for every same-pattern geometry edge between eligible, CV2-mainly ends."""
    out, skipped = [], {"not_eligible": 0, "not_mainly_cv2": 0}
    for edge in payload.get("edges", []) or []:
        if str(edge.get("edge_type")) in cr.NON_BRIDGE_EDGE_TYPES:
            continue
        si, sj = int(edge["state_i"]), int(edge["state_j"])
        if si not in rep or sj not in rep:
            continue
        a, b = views[si], views[sj]
        if not (cr.eligibility(a, settings)[0] and cr.eligibility(b, settings)[0]):
            skipped["not_eligible"] += 1
            continue
        d1, d2 = cr.axis_distances(a, b, settings.rt)
        if not cr.mainly_cv2(d1, d2):
            skipped["not_mainly_cv2"] += 1
            continue
        out.append((edge, a, b, d1, d2))
    return out, skipped


BRIDGE_COLUMN_TOL = 1e-6


def bridge_columns_from_plan(plan: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per layout column, what ``bridge_set_children`` needs: the fit (rebuilt from the record)
    and the layout's own placement settings. [] when the plan has no shape record."""
    from gareus.adaptive.cv2_shape import fit_from_record  # noqa: PLC0415
    rec = (plan or {}).get("cv2_shape") or {}
    out = []
    for col in rec.get("columns") or []:
        try:
            out.append({"centre1": float(col["centre1"]), "fit": fit_from_record(col["fit"]),
                        "sigma_w_target": float(rec["sigma_w_target"]), "spacing_sigma": float(rec["spacing_sigma"]),
                        "k_min": float(rec["cv2_k_min"]), "k_max": float(rec["cv2_k_max"]),
                        "min_mean_compression": float((col.get("placement") or {}).get("min_mean_compression", 0.5))})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _column_for(columns: Optional[Sequence[Mapping[str, Any]]], a: cr.StateView, b: cr.StateView):
    if not columns or abs(float(a.c1) - float(b.c1)) > BRIDGE_COLUMN_TOL:
        return None
    near = [c for c in columns if abs(float(c["centre1"]) - float(a.c1)) <= BRIDGE_COLUMN_TOL]
    return near[0] if near else None


def bridge_set_children(a: cr.StateView, b: cr.StateView, column: Optional[Mapping[str, Any]],
                        settings: cr.ResolutionSettings, gate: Any = None) -> Dict[str, Any]:
    """R1's children for the gap a-b: every fill the layout column's placement model puts strictly
    between the two centres (same step rule and springs as the layout; fills within 0.5 sampled
    sigma of an endpoint dropped), each seeded from the NEARER endpoint and passed through the 3.4
    gate; or -- no column, no fill, or a placement failure -- today's single sampled-midpoint bridge
    (``midpoint_fallback``). ``fallback_children`` always carries that midpoint, for the budget."""
    from gareus.adaptive.cv2_shape import (bridge_fills, compression_floor_k2,  # noqa: PLC0415
                                           internal_barriers_kT, predicted_overlaps)
    midpoint = [cr.bridge_child(a, b, settings)]
    fallback = {"bridge_mode": "midpoint_fallback", "children": midpoint, "fallback_children": None,
                "seed_sources": None, "predicted_overlaps": None, "predicted_min_overlap": None,
                "internal_barrier_kT": None}
    if column is None or a.c2 is None or b.c2 is None:
        return fallback
    lo, hi = sorted((a, b), key=lambda v: float(v.c2))
    try:
        fills = bridge_fills(column["fit"], float(lo.c2), float(hi.c2), sigma_w_target=column["sigma_w_target"],
                             temperature_k=settings.temperature_k, k_min=column["k_min"], k_max=column["k_max"],
                             spacing_sigma=column["spacing_sigma"], min_mean_compression=column["min_mean_compression"])
    except (ValueError, FloatingPointError):
        return {**fallback, "bridge_mode": "midpoint_fallback_placement_failed"}
    keep = [i for i, (z, sig) in enumerate(zip(fills["centres"], fills["sampled_sigma"]))
            if abs(z - float(lo.c2)) >= 0.5 * sig and abs(z - float(hi.c2)) >= 0.5 * sig]
    if not keep:
        return fallback
    k_cap = float(settings.k2_max) if settings.k2_max is not None and float(settings.k2_max) > 0 else None
    c_min = float(column["min_mean_compression"])
    c1, k1 = 0.5 * (float(a.c1) + float(b.c1)), 0.5 * (float(a.k1) + float(b.k1))
    children, seeds = [], []
    for i in keep:
        z, k2, f2 = fills["centres"][i], float(fills["k2"][i]), float(fills["f2"][i])
        k2 = min(k2, k_cap) if k_cap is not None else k2
        floor_c = float(compression_floor_k2(f2, c_min))
        width = settings.rt / float(column["sigma_w_target"]) ** 2 - max(0.0, f2)
        near = lo if abs(z - float(lo.c2)) <= abs(z - float(hi.c2)) else hi
        child = {"parent_state_id": int(a.state_id), "seed_source_state_id": int(near.state_id),
                 "primary_center": c1, "primary_k": k1, "secondary_center": float(z), "centred_on": "bridge_set",
                 "k2": k2, "f2_est": f2, "predicted_sampled_sigma": float(fills["sampled_sigma"][i]),
                 "mean_compression": k2 / (k2 + max(0.0, f2)) if k2 + max(0.0, f2) > 0 else 1.0,
                 "min_mean_compression": c_min, "compression_floor_k2": floor_c,
                 "at_compression_floor": bool(floor_c > width and abs(k2 - floor_c) <= 1e-9 * max(1.0, floor_c)),
                 "refusal": None}
        if k2 <= max(0.0, float(settings.k2_min)) * (1.0 + 1e-12):
            child["refusal"] = "k2_at_floor"
        child = cr.gate_child(child, c1, k1, float(z), gate, f"cv2_resolution R1 bridge set {a.state_id}-{b.state_id}",
                              settings)
        children.append(child)
        seeds.append(int(near.state_id))
    cs = [float(lo.c2), *[c["secondary_center"] for c in children], float(hi.c2)]
    ks = [float(lo.k2), *[c["k2"] for c in children], float(hi.k2)]
    ov = predicted_overlaps(column["fit"], cs, ks, settings.temperature_k)
    barriers = internal_barriers_kT(column["fit"], cs[1:-1], ks[1:-1], settings.temperature_k)
    return {"bridge_mode": "set", "children": children, "fallback_children": midpoint, "seed_sources": seeds,
            "predicted_overlaps": ov, "predicted_min_overlap": float(min(ov)) if ov else None,
            "internal_barrier_kT": barriers}


def _bridge_set_candidate(cls, a, b, metrics, reason, settings, gate, columns):
    out = bridge_set_children(a, b, _column_for(columns, a, b), settings, gate)
    if out["bridge_mode"] != "set":
        metrics = {**metrics, "bridge_mode": out["bridge_mode"]}
        return _bridge_candidate(cls, a, b, metrics, reason, settings, gate)
    kids = out["children"]
    metrics = {**metrics, "bridge_mode": "set", "n_children": len(kids),
               "predicted_overlaps": out["predicted_overlaps"], "predicted_min_overlap": out["predicted_min_overlap"],
               "predicted_overlap_is_upper_bound": True, "internal_barrier_kT": out["internal_barrier_kT"]}
    proposal = {"parent_state_id": int(a.state_id), "children": kids, "seed_sources": out["seed_sources"]}
    ids = [a.state_id, b.state_id]
    refused = [c["refusal"] for c in kids if c.get("refusal")]
    if refused:                                       # atomic: one refused child refuses the set
        return cr.new_candidate("R1", "edge", ids, "refused", f"{refused[0]}: {reason}", cls=cls,
                                refusal=refused[0], metrics=metrics, proposal=proposal)
    fb = _bridge_candidate(cls, a, b, {}, reason, settings, gate)
    cand = cr.new_candidate("R1", "edge", ids, "proposed", reason, cls=cls, metrics=metrics, proposal=proposal)
    if fb["decision"] == "proposed":
        cand["fallback_proposal"] = fb["proposal"]    # funded instead when the whole set does not fit
    return cand


def _bridge_candidate(cls, a, b, metrics, reason, settings, gate):
    child = cr.bridge_child(a, b, settings)
    child = cr.gate_child(child, child["primary_center"], child["primary_k"], child["secondary_center"], gate,
                          f"cv2_resolution R1 bridge {a.state_id}-{b.state_id}", settings)
    proposal = {"parent_state_id": child["parent_state_id"], "children": [child]}
    ids = [a.state_id, b.state_id]
    if child.get("refusal"):
        return cr.new_candidate("R1", "edge", ids, "refused", f"{child['refusal']}: {reason}", cls=cls,
                                refusal=child["refusal"], metrics=metrics, proposal=proposal)
    return cr.new_candidate("R1", "edge", ids, "proposed", reason, cls=cls, metrics=metrics, proposal=proposal)


def _r1_decide(cls, edge, a, b, metrics, history, epoch, settings, gate, columns=None):
    key = cr.edge_key(a.state_id, b.state_id)
    thr = settings.threshold
    if settings.cv2_bridge_sets:
        def bridge(*xs):
            return _bridge_set_candidate(*xs, columns)
    else:
        bridge = _bridge_candidate
    if cls == "structural":
        return bridge(cls, a, b, metrics, f"component split at {thr:g}: more sampling cannot connect it",
                                 settings, gate)
    if cr.is_protected(a, epoch, settings) or cr.is_protected(b, epoch, settings):
        return cr.new_candidate("R1", "edge", [a.state_id, b.state_id], "skipped", "protected endpoint",
                                cls=cls, metrics=metrics)
    if cls == "weak":
        return bridge(cls, a, b, metrics, f"confidently below {thr:g} inside one component",
                                 settings, gate)
    n = cr.consecutive_bad(history, key, epoch)
    metrics["consecutive_bad_epochs"] = n
    need = 1 + cr.UNMEASURED_WAIT_EPOCHS
    if n >= need:
        return bridge(cls, a, b, metrics, f"still unmeasured/weak after {n - 1} epoch(s) of sampling",
                                 settings, gate)
    return cr.new_candidate("R1", "edge", [a.state_id, b.state_id], "extend",
                            f"unmeasured {n}/{need} epochs: extend sampling first (the extend is a lifecycle "
                            "record; the edge waits through ordinary epochs)", cls=cls, metrics=metrics)


def propose_r1(payload: Mapping[str, Any], views: Mapping[int, cr.StateView], rep_ids: Sequence[int],
               settings: cr.ResolutionSettings, history: Mapping[str, Any], epoch: int, gate: Any = None,
               bridge_columns: Optional[Sequence[Mapping[str, Any]]] = None
               ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """R1 candidates, the updated edge history and a status record."""
    em = payload.get("edge_metric") or {}
    if em.get("status") != "ok":
        why = "requires --ap-edge-metric pairwise-mbar" if not em else f"edge metric {em.get('status')}: {em.get('error')}"
        return [], dict(history), {"status": "unavailable", "reason": why}
    comp_of = _component_map(em)
    edges, skipped = _r1_edges(payload, views, set(rep_ids), settings)
    classes = [(e, a, b, d1, d2, cr.classify_edge(e, comp_of, settings.threshold)) for e, a, b, d1, d2 in edges]
    statuses = {cr.edge_key(a.state_id, b.state_id): c for _e, a, b, _d1, _d2, c in classes}
    new_history = cr.update_history(history, epoch, statuses, views.keys())
    cands, bridged_pairs = [], set()
    for edge, a, b, d1, d2, cls in sorted(classes, key=lambda t: float(t[0].get("normalized_distance") or 0.0)):
        if cls == "ok":
            continue
        metrics = _edge_metrics(edge, d1, d2, comp_of)
        pair = tuple(sorted((comp_of.get(a.state_id, -1), comp_of.get(b.state_id, -1))))
        if cls == "structural" and pair in bridged_pairs:
            cands.append(cr.new_candidate("R1", "edge", [a.state_id, b.state_id], "no_action",
                                          "component_pair_already_bridged", cls=cls, metrics=metrics))
            continue
        cand = _r1_decide(cls, edge, a, b, metrics, new_history, epoch, settings, gate, bridge_columns)
        if cls == "structural" and cand["decision"] == "proposed":
            bridged_pairs.add(pair)
        cands.append(cand)
    n_comp = len(((em.get("components") or {}).get("components")) or [])
    status = {"status": "ok", "reason": None, "n_candidate_edges": len(edges), "skipped_edges": skipped,
              "classes": {c: sum(1 for *_x, k in classes if k == c) for c in ("structural", "weak", "unmeasured", "ok")},
              "n_components": n_comp, "components_without_candidate_edge": max(0, n_comp - 1 - len(bridged_pairs))}
    return cands, new_history, status


__all__ = ["bridge_columns_from_plan", "bridge_set_children", "propose_r1", "propose_r3"]
