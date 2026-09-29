"""T3 task 1: per-phase edge-class distributions and weak-edge lists from replayed payloads.

Usage: python t3_edge_tables.py <out.json> <label=payload.json> [...]

Edge classes (representative rung, lambda = 0):
  CV1 chain / CV2 chain / 2D chain : collector geometry edges (primary_chain, nearest_2d) between
                                     two states of that restraint pattern;
  cross-pattern geometry           : collector geometry edges across patterns (never weak under 3.1);
  spanning                         : 3.1 appended cross-pattern edges;
  radius-neighbour                 : 3.1 appended same-pattern edges within the P7a radius;
  rung                             : same centre, adjacent lambda (union ``mbar_overlap`` only).
CV1 marginal and joint-2D histogram overlaps for appended edges are recomputed on the P4
subsample (``_hist_overlap`` / ``joint_overlap_2d``, the collectors' own functions).
"""
import json
import sys
from pathlib import Path

import numpy as np

import gareus.adaptive_production as ap
from gareus.adaptive.paired_cv import joint_overlap_2d, load_paired_subsamples

NAME = {(True, False): "CV1", (False, True): "CV2", (True, True): "2D", (False, False): "anchor"}
POL_NEW = ap.AdaptiveDecisionPolicy(edge_metric="pairwise-mbar")
POL_OLD = ap.AdaptiveDecisionPolicy()


def q(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return None
    return dict(n=int(v.size), min=float(v.min()), q10=float(np.quantile(v, .1)), med=float(np.median(v)),
                q90=float(np.quantile(v, .9)), max=float(v.max()))


def classify(e, pat):
    et = str(e.get("edge_type"))
    if et == "rung":
        return "rung"
    pa, pb = pat.get(int(e["state_i"])), pat.get(int(e["state_j"]))
    if et == "neighbour":
        return "radius-neighbour"
    if et == "spanning":
        return "spanning"
    if pa == pb and pa is not None:
        return f"{NAME[pa]} chain"
    return "cross-pattern geometry"


def analyse(payload):
    pat = {int(s["state_id"]): (bool(s["paired_cv"]["restraint"]["cv1_restrained"]),
                                bool(s["paired_cv"]["restraint"]["cv2_restrained"]))
           for s in payload["states"] if s.get("paired_cv", {}).get("restraint")}
    npz = (payload.get("paired_cv") or {}).get("npz")
    sub = load_paired_subsamples(Path(npz)) if npz else {}
    rows = []
    for e in payload["edges"]:
        cls = classify(e, pat)
        pm = e.get("pairwise_mbar") or {}
        si, sj = int(e["state_i"]), int(e["state_j"])
        marg, joint = e.get("overlap"), e.get("overlap_joint_2d")
        a, b = sub.get(si), sub.get(sj)
        if cls != "rung" and a is not None and b is not None:
            if marg is None:
                marg = ap._hist_overlap(a["cv1"], b["cv1"])
            if joint is None and pat.get(si, (0, 0))[1] and pat.get(sj, (0, 0))[1]:
                joint = joint_overlap_2d(a["cv1"], a["cv2"], b["cv1"], b["cv2"])
        rows.append(dict(i=si, j=sj, cls=cls, etype=e.get("edge_type"), gkind=pm.get("graph_kind"),
                         status=pm.get("status"), reason=pm.get("reason"), pt=pm.get("overlap_point"),
                         lo=pm.get("overlap_lower"), hi=pm.get("overlap_upper"), df=pm.get("delta_f_kT"),
                         neff=pm.get("n_eff"), tau=pm.get("tau"), marginal=marg, joint=joint,
                         mbar=e.get("mbar_overlap"), acc=e.get("exchange_acceptance"),
                         attempts=e.get("exchange_attempts"),
                         weak_new=bool(ap._edge_is_measured_weak(e, POL_NEW)),
                         weak_old=bool(ap._edge_is_measured_weak(e, POL_OLD)) if e.get("overlap_joint_2d_reason")
                         != "edge_metric_graph_edge" else False))
    classes = {}
    for cls in sorted({r["cls"] for r in rows}):
        rr = [r for r in rows if r["cls"] == cls]
        st = {}
        for r in rr:
            st[str(r["status"])] = st.get(str(r["status"]), 0) + 1
        classes[cls] = dict(
            n=len(rr), status=st, point=q(r["pt"] for r in rr), q10=q(r["lo"] for r in rr),
            q90=q(r["hi"] for r in rr), marginal=q(r["marginal"] for r in rr), joint=q(r["joint"] for r in rr),
            mbar=q(r["mbar"] for r in rr),
            tau=q(t for r in rr if r["tau"] for t in r["tau"]),
            neff=q(t for r in rr if r["neff"] for t in r["neff"]),
            weak_new=sum(r["weak_new"] for r in rr), weak_old=sum(r["weak_old"] for r in rr),
            point_below_015=sum(1 for r in rr if r["pt"] is not None and r["pt"] < 0.15),
            q90_below_015=sum(1 for r in rr if r["status"] == "ok" and r["hi"] is not None and r["hi"] < 0.15))
    rec = payload.get("edge_metric", {})
    weak_list = [r for r in rows if r["weak_new"] or r["weak_old"]]
    return dict(classes=classes, weak=weak_list, rows=rows,
                n_weak_new=sum(r["weak_new"] for r in rows), n_weak_old=sum(r["weak_old"] for r in rows),
                components=rec.get("components"), n_unmeasured=rec.get("n_unmeasured"),
                stride=q(s["paired_cv"]["subsample"]["stride"] for s in payload["states"] if s.get("paired_cv")),
                n_kept=q(s["paired_cv"]["subsample"]["n_kept"] for s in payload["states"] if s.get("paired_cv")))


out = Path(sys.argv[1])
res = {}
for spec in sys.argv[2:]:
    label, path = spec.split("=", 1)
    res[label] = analyse(json.loads(Path(path).read_text()))
    r = res[label]
    print(f"== {label}: weak marginal {r['n_weak_old']} -> pairwise {r['n_weak_new']}; unmeasured {r['n_unmeasured']}; "
          f"components {(r['components'] or {}).get('n_components')} / measured {(r['components'] or {}).get('n_measured_components')} "
          f"sizes {(r['components'] or {}).get('measured_component_sizes')}")
    for cls, c in r["classes"].items():
        f = lambda d: "-" if d is None else f"{d['med']:.3f} [{d['q10']:.3f},{d['q90']:.3f}] min {d['min']:.3f}"
        print(f"  {cls:24s} n={c['n']:3d} pw {f(c['point'])} | marg {f(c['marginal'])} | joint {f(c['joint'])} "
              f"| tau {f(c['tau'])} | neff {f(c['neff'])} | weak old/new {c['weak_old']}/{c['weak_new']} "
              f"q90<.15 {c['q90_below_015']}")
    for w in r["weak"]:
        print("   weak", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in w.items()
                          if k in ("i", "j", "cls", "etype", "pt", "hi", "marginal", "joint", "acc", "attempts", "weak_old", "weak_new")})
out.write_text(json.dumps(res, default=float))
