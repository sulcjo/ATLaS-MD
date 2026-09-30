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


def _r1_decide(cls, edge, a, b, metrics, history, epoch, settings, gate):
    key = cr.edge_key(a.state_id, b.state_id)
    thr = settings.threshold
    if cls == "structural":
        return _bridge_candidate(cls, a, b, metrics, f"component split at {thr:g}: more sampling cannot connect it",
                                 settings, gate)
    if cr.is_protected(a, epoch, settings) or cr.is_protected(b, epoch, settings):
        return cr.new_candidate("R1", "edge", [a.state_id, b.state_id], "skipped", "protected endpoint",
                                cls=cls, metrics=metrics)
    if cls == "weak":
        return _bridge_candidate(cls, a, b, metrics, f"confidently below {thr:g} inside one component",
                                 settings, gate)
    n = cr.consecutive_bad(history, key, epoch)
    metrics["consecutive_bad_epochs"] = n
    need = 1 + cr.UNMEASURED_WAIT_EPOCHS
    if n >= need:
        return _bridge_candidate(cls, a, b, metrics, f"still unmeasured/weak after {n - 1} epoch(s) of sampling",
                                 settings, gate)
    return cr.new_candidate("R1", "edge", [a.state_id, b.state_id], "extend",
                            f"unmeasured {n}/{need} epochs: extend sampling first (the extend is a lifecycle "
                            "record; the edge waits through ordinary epochs)", cls=cls, metrics=metrics)


def propose_r1(payload: Mapping[str, Any], views: Mapping[int, cr.StateView], rep_ids: Sequence[int],
               settings: cr.ResolutionSettings, history: Mapping[str, Any], epoch: int, gate: Any = None
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
        cand = _r1_decide(cls, edge, a, b, metrics, new_history, epoch, settings, gate)
        if cls == "structural" and cand["decision"] == "proposed":
            bridged_pairs.add(pair)
        cands.append(cand)
    n_comp = len(((em.get("components") or {}).get("components")) or [])
    status = {"status": "ok", "reason": None, "n_candidate_edges": len(edges), "skipped_edges": skipped,
              "classes": {c: sum(1 for *_x, k in classes if k == c) for c in ("structural", "weak", "unmeasured", "ok")},
              "n_components": n_comp, "components_without_candidate_edge": max(0, n_comp - 1 - len(bridged_pairs))}
    return cands, new_history, status


__all__ = ["propose_r1", "propose_r3"]
