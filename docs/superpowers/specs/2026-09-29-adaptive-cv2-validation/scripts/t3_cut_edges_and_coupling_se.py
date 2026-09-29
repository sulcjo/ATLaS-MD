import json
import numpy as np

# Cut-edge check on the c9 union-reference edge set (all 295 graded lambda=0 edges).
rows = json.load(open("/tmp/t3/calib_c9.json"))["rows"]
nodes = sorted({r["i"] for r in rows} | {r["j"] for r in rows})


def n_comp(edges):
    parent = {n: n for n in nodes}

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[f(a)] = f(b)
    return len({f(n) for n in nodes})


all_e = [(r["i"], r["j"]) for r in rows]
base = n_comp(all_e)
low = [r for r in rows if r["o2s"] < 0.15]
cut = [(r["i"], r["j"], round(r["o2s"], 3)) for r in low
       if n_comp([e for e in all_e if e != (r["i"], r["j"])]) > base]
print("components", base, "low (<0.15) edges", len(low), "cut edges among them", cut)
# Removing ALL edges < 0.15 at once:
print("components after removing every edge < 0.15:", n_comp([(r["i"], r["j"]) for r in rows if r["o2s"] >= 0.15]),
      "| < 0.10:", n_comp([(r["i"], r["j"]) for r in rows if r["o2s"] >= 0.10]))

# Coupling k1 = 0: realised shift across the 6 disjoint phases, mean +/- SE.
ph = ["epoch_000", "epoch_001", "epoch_002", "final", "final_extension_001", "final_extension_002"]
vals = {s: [] for s in (64, 68, 72, 76)}
pred = {s: [] for s in (64, 68, 72, 76)}
for p in ph:
    d = json.load(open(f"/tmp/t3/coupling_c9_{p}.json"))
    w = {r["state_id"]: r for r in d["windows"]}
    for s in vals:
        vals[s].append(w[s]["realised_shift_in_anchor_sd"])
        pred[s].append(w[s]["pred_shift_lr_anchor_sd"] / d["anchor_cv1_sd"])
for s in vals:
    v = np.array(vals[s])
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(v.size)
    pm = float(np.mean(pred[s]))
    print(s, "realised mean %+.3f SE %.3f (phase sd %.3f) | gate LR pred %+.3f | (pred-real)/SE %.1f | bound 0.419 is %.1f SE above |mean|"
          % (m, se, v.std(ddof=1), pm, (pm - m) / se, (0.419 - abs(m)) / se))
