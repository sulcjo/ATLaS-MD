"""T3 task 2 summary: bin every lambda = 0 spatial edge by overlap and compare the references.

Usage: python t3_calibration_summary.py <union_ref.json> <exchanges.json> <phase,phase,...> <out.json>

Per overlap bin (two-state overlap on the union's own samples, ``o2s``):
  * n edges, n(z_union) and n(z_split) with |z| > 2 / 3, where
      z_union = (df2s - dfu) / sd_df2s   (two-state vs union Delta f, in bootstrap sigma)
      z_split = (df2s_h1 - df2s_h2) / (sqrt(2) * sd_df2s)   (first vs second half)
  * sd_df2s * sqrt(n_harm) vs the iid theory sqrt(1/O - 2) (n_harm = 2 Na Nb / (Na + Nb));
  * union-f pairwise overlap vs re-solved two-state overlap;
  * realised replica transitions (accepted exchanges) per ns summed over the given phases.
"""
import json
import math
import sys

import numpy as np

ref = json.loads(open(sys.argv[1]).read())
ex = json.loads(open(sys.argv[2]).read())
phases = sys.argv[3].split(",")
out = sys.argv[4]

pairs = {}
span = 0.0
for p in phases:
    span += ex[p]["span_ns"]
    for k, (att, acc) in ex[p]["pairs"].items():
        t = pairs.setdefault(k, [0, 0])
        t[0] += att
        t[1] += acc

rows = []
for r in ref["spatial"]:
    na, nb = r["union_n"]
    nh = 2.0 * na * nb / (na + nb)
    k = f"{r['i']}-{r['j']}"
    att, acc = pairs.get(k, [0, 0])
    sd = r["sd_df2s"]
    theory = math.sqrt(max(1.0 / r["o2s"] - 2.0, 0.0)) if r["o2s"] > 0 else float("inf")
    rows.append(dict(r, n_harm=nh, sd_scaled=sd * math.sqrt(nh), sd_theory_scaled=theory,
                     z_union=(r["df2s"] - r["dfu"]) / sd if sd > 0 else None,
                     z_split=(r["df2s_h1"] - r["df2s_h2"]) / (math.sqrt(2.0) * sd) if sd > 0 else None,
                     attempts=att, accepted=acc, trans_per_ns=acc / span if span else None,
                     split_dO=abs(r["o2s_h1"] - r["o2s_h2"])))

bins = [0.0, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.51]
summary = []
for lo, hi in zip(bins[:-1], bins[1:]):
    rr = [r for r in rows if lo <= r["o2s"] < hi]
    if not rr:
        continue
    zu = np.array([abs(r["z_union"]) for r in rr if r["z_union"] is not None])
    zs = np.array([abs(r["z_split"]) for r in rr if r["z_split"] is not None])
    ratio = np.array([r["sd_scaled"] / r["sd_theory_scaled"] for r in rr])
    ex_rr = [r for r in rr if r["attempts"] > 0]
    summary.append(dict(
        bin=[lo, hi], n=len(rr), kinds=dict((k, sum(1 for r in rr if (r["graph_kind"] or r["edge_type"]) == k))
                                           for k in {(r["graph_kind"] or r["edge_type"]) for r in rr}),
        sd_df_kT_med=float(np.median([r["sd_df2s"] for r in rr])),
        sd_over_theory_med=float(np.median(ratio)),
        z_union_abs_med=float(np.median(zu)), n_z_union_gt2=int((zu > 2).sum()), n_z_union_gt3=int((zu > 3).sum()),
        z_split_abs_med=float(np.median(zs)), n_z_split_gt2=int((zs > 2).sum()), n_z_split_gt3=int((zs > 3).sum()),
        ouf_minus_o2s_med=float(np.median([r["ouf"] - r["o2s"] for r in rr])),
        p4_minus_o2s_med=float(np.median([r["p4_point"] - r["o2s"] for r in rr if r["p4_point"] is not None]))
        if any(r["p4_point"] is not None for r in rr) else None,
        n_with_exchange=len(ex_rr),
        acceptance_med=float(np.median([r["accepted"] / r["attempts"] for r in ex_rr])) if ex_rr else None,
        trans_per_ns_med=float(np.median([r["trans_per_ns"] for r in ex_rr])) if ex_rr else None,
        trans_per_ns_min=float(np.min([r["trans_per_ns"] for r in ex_rr])) if ex_rr else None,
        n_zero_transitions=sum(1 for r in ex_rr if r["accepted"] == 0)))
for s in summary:
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}))
o = np.array([r["o2s"] for r in rows])
p4 = np.array([np.nan if r["p4_point"] is None else r["p4_point"] for r in rows])
m = np.isfinite(p4)
print("corr(o2s, p4_point)", float(np.corrcoef(o[m], p4[m])[0, 1]), "median |diff|", float(np.median(np.abs(o[m] - p4[m]))))
ex_rows = [r for r in rows if r["attempts"] >= 20]
if ex_rows:
    a = np.array([r["accepted"] / r["attempts"] for r in ex_rows])
    oo = np.array([r["o2s"] for r in ex_rows])
    from scipy.stats import spearmanr
    print("edges with >= 20 attempts", len(ex_rows), "spearman(o2s, acceptance)", spearmanr(oo, a))
zu_all = np.array([r["z_union"] for r in rows if r["z_union"] is not None])
zs_all = np.array([r["z_split"] for r in rows if r["z_split"] is not None])
print("all edges: z_union sd", float(np.std(zu_all)), "median", float(np.median(zu_all)),
      "| z_split sd", float(np.std(zs_all)), "frac |z_split|>2", float(np.mean(np.abs(zs_all) > 2)))
json.dump(dict(rows=rows, summary=summary, span_ns=span, phases=phases), open(out, "w"), default=float)
