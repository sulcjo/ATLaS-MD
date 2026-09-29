"""T3 task 4: cap-ignoring proposer dry-run, marginal vs pairwise-mbar, on replayed payloads.

Usage: python t3_proposer_dryrun.py <adaptive_dir> <out.json> <payload.json> [<payload.json> ...]

Nothing is applied and nothing is written except ``out.json``: the registry is loaded fresh
(and deep-copied) for every call; ``propose_actions_from_diagnostics`` only reads it.
Caps lifted: ``max_new_windows_per_epoch`` and ``max_new_rungs_per_epoch`` = 10000
(``max_replicas_budget`` = 0 = unlimited; it is enforced only at apply time anyway).
The marginal run gets the payload with the 3.1 graph edges and ``pairwise_mbar`` keys removed.
"""
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import gareus.adaptive_production as ap

ad = Path(sys.argv[1])
out = Path(sys.argv[2])
man = json.loads((ad / "final" / "baseline" / "run_manifest.json").read_text()) \
    if (ad / "final" / "baseline" / "run_manifest.json").exists() else {}
ra = man.get("resolved_args", {})
k2max = ra.get("cv2_k_max")
T = float(ra.get("temperature_k", 300.0))
CAPS = dict(max_new_windows_per_epoch=10000, max_new_rungs_per_epoch=10000, max_replicas_budget=0)


def strip(payload):
    p = copy.deepcopy(payload)
    p["edges"] = [e for e in p["edges"] if e.get("overlap_joint_2d_reason") != "edge_metric_graph_edge"]
    for e in p["edges"]:
        e.pop("pairwise_mbar", None)
        e["warnings"] = [w for w in e.get("warnings", [])
                         if w not in ("low_pairwise_mbar_overlap", "pairwise_mbar_unmeasured")]
    p.pop("edge_metric", None)
    return p


def summarise(actions, plan):
    kinds = Counter(a[0] for a in actions)
    adds = []
    for a in actions:
        if a[0] in ("add", "add_rung", "split"):
            params = a[2] if len(a) > 2 else None
            adds.append(dict(kind=a[0], parent=a[1], params=params if isinstance(params, dict) else str(params),
                             reason=a[3] if len(a) > 3 else None))
    return dict(kinds=dict(kinds), adds=adds, bridge_plan=plan)


res = {}
for path in sys.argv[3:]:
    payload = json.loads(Path(path).read_text())
    row = {}
    for name, pol, pay in (
            ("marginal", ap.AdaptiveDecisionPolicy(**CAPS), strip(payload)),
            ("pairwise-mbar", ap.AdaptiveDecisionPolicy(edge_metric="pairwise-mbar", **CAPS), payload)):
        reg = copy.deepcopy(ap.WindowStateRegistry.load(ad))
        plan = []
        acts = ap.propose_actions_from_diagnostics(reg, pay, pol, temperature_K=T, secondary_k_max=k2max,
                                                   bridge_plan_out=plan)
        weak = []
        for e in pay["edges"]:
            if e.get("edge_type") == "rung":
                if ap._edge_is_measured_weak(e, pol):
                    weak.append(dict(i=e["state_i"], j=e["state_j"], type="rung", mbar=e.get("mbar_overlap")))
                continue
            if ap._edge_is_measured_weak(e, pol):
                pm = e.get("pairwise_mbar") or {}
                ov = e.get("overlap")
                acc = e.get("exchange_acceptance")
                trig = []
                if name == "marginal":
                    if ov is not None and ov < pol.target_overlap:
                        trig.append("cv1_marginal<0.30")
                    if acc is not None and acc < pol.min_exchange_acceptance:
                        trig.append("acceptance<0.08")
                weak.append(dict(i=e["state_i"], j=e["state_j"], type=e.get("edge_type"), marginal=ov,
                                 acceptance=acc, attempts=e.get("exchange_attempts"),
                                 pw=pm.get("overlap"), q90=pm.get("overlap_upper"), triggers=trig))
        row[name] = dict(weak_edges=weak, **summarise(acts, plan))
    res[path] = row
    print(path)
    for name in ("marginal", "pairwise-mbar"):
        r = row[name]
        print(f"  {name}: weak {len(r['weak_edges'])} kinds {r['kinds']} adds "
              f"{[(a['kind'], a['parent']) for a in r['adds']]}")
out.write_text(json.dumps(res, default=str))
