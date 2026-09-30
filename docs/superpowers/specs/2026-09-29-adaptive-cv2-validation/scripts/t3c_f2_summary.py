"""Reduce t3c_f2_crosscheck.py output to the compact table kept in t2_data/.

python t3c_f2_summary.py IN.json OUT.json
"""
import json
import math
import sys

import numpy as np


def q(a, ps=(0.0, 0.1, 0.5, 0.9, 1.0)):
    a = [x for x in a if x is not None and math.isfinite(x)]
    return [round(float(np.quantile(a, p)), 3) for p in ps] if a else None


def rel_ci(point, ci, boot_point=None):
    """The bootstrap is on the P4 subsample; report its relative 5-95 % spread."""
    if not ci:
        return None
    mid = boot_point if boot_point else 0.5 * (ci[0] + ci[1])
    return [round(ci[0] / mid, 3), round(ci[1] / mid, 3)]


def main(inp, out):
    d = json.load(open(inp))
    p1, p2, p3 = d["p1_window_local"], d["p2_cv1_only"], d["p3_union"]
    res = {"provenance": d["provenance"], "quantiles": [0.0, 0.1, 0.5, 0.9, 1.0]}
    # pooled per column
    pooled = []
    for a in sorted(p2, key=lambda x: x["c1"]):
        b = next(x for x in p3 if abs(x["c1"] - a["c1"]) < 1e-3)
        pooled.append({"c1": round(a["c1"], 4), "swarm_rt_over_pooled_var": round(a["f2_pool_swarm_raw"], 3),
                       "cv1_only_window": round(a["f2_pool_prod"], 3),
                       "cv1_only_boot_rel_ci90": rel_ci(a["f2_pool_prod"], a["f2_pool_prod_ci90"]),
                       "union": round(b["f2_pool_union"], 3), "union_ci90": [round(x, 3) for x in b["f2_pool_union_ci90"]],
                       "ratio_cv1_only": round(a["ratio_pool"], 3), "ratio_union": round(b["ratio_pool"], 3),
                       "union_kish_n_eff": round(b["kish_n_eff"])})
    res["pooled_per_column"] = pooled
    res["pooled_ratio_cv1_only_q"] = q([x["ratio_cv1_only"] for x in pooled])
    res["pooled_ratio_union_q"] = q([x["ratio_union"] for x in pooled])
    # window-local
    loc = [{"state": x["state"], "c1": round(x["c1"], 4), "c2": round(x["c2"], 3), "cv2_mean": round(x["cv2_mean"], 3),
            "cv2_sd": round(x["cv2_sd"], 3), "f2_prod_cond": round(x["f2_prod_cond"], 3),
            "boot_rel_ci90": rel_ci(x["f2_prod_cond"], x["f2_prod_cond_ci90"]),
            "f2_swarm_at_mean": round(x["f2_swarm_at_mean"], 3), "swarm_fallback_pooled": x["swarm_fallback_pooled"],
            "ratio": round(x["ratio_cond"], 3)} for x in sorted(p1, key=lambda x: (x["c1"], x["c2"]))]
    res["window_local"] = loc
    r = [x["ratio"] for x in loc]
    res["window_local_ratio_q"] = q(r)
    res["window_local_log_ratio_mean_sd"] = [round(float(np.mean(np.log(r))), 3), round(float(np.std(np.log(r))), 3)]
    res["window_local_n_nonpositive"] = sum(1 for x in p1 if x["f2_prod_cond"] <= 0)
    res["window_local_f2_q"] = q([x["f2_prod_cond"] for x in loc])
    res["window_local_max_abs_corr"] = round(max(abs(x["corr"]) for x in p1), 3)
    # per mode
    modes = []
    for x in p3:
        for m in x["swarm_matches"]:
            modes.append({"c1": round(x["c1"], 4), "swarm_mean": round(m["swarm_mean"], 3), "swarm_sd": round(m["swarm_sd"], 3),
                          "swarm_weight": round(m["swarm_weight"], 3), "swarm_members": m["swarm_members"],
                          "f2_swarm_raw": round(m["f2_swarm_raw"], 2), "f2_swarm_est": round(m["f2_swarm_est"], 2),
                          "union_mean": round(m["union_mean"], 3), "union_sd": round(m["union_sd"], 3),
                          "union_weight": round(m["union_weight"], 3), "f2_union": round(m["f2_union_raw"], 2),
                          "f2_union_ci90": m["f2_union_raw_ci90"] and [round(v, 2) for v in m["f2_union_raw_ci90"]],
                          "mean_distance": round(m["mean_distance"], 3), "ratio_vs_est": round(m["ratio_vs_est"], 3)})
    res["per_mode"] = modes
    # only pairs whose means agree within half (strict) or one (loose) of the narrower sd are
    # the same mode; the rest are modes absent from, or merged in, the production distribution
    for tag, f in (("strict_0p5sd", 0.5), ("loose_1sd", 1.0)):
        mm = [m for m in modes if m["mean_distance"] < f * min(m["swarm_sd"], m["union_sd"])]
        res[f"per_mode_matched_{tag}"] = {"n": len(mm), "pairs": mm}
    res["per_mode_ratio_vs_est_q"] = q([m["ratio_vs_est"] for m in modes])
    narrow = [m for m in modes if m["swarm_mean"] < -1.2 and m["swarm_sd"] < 0.36]
    res["swarm_narrow_mode_minus1p4"] = {"n": len(narrow), "ratio_vs_est_q": q([m["ratio_vs_est"] for m in narrow]),
                                         "union_mean_q": q([m["union_mean"] for m in narrow]),
                                         "union_sd_q": q([m["union_sd"] for m in narrow])}
    # union narrow components (sd < 0.13) the swarm does not have
    un = [{"c1": round(x["c1"], 4), "mean": round(c["mean"], 3), "sd": round(c["sd"], 3), "weight": round(c["weight"], 3),
           "accepted": c["accepted"], "n_members": c["n_members"], "f2": round(c["f2_raw"], 1)}
          for x in p3 for c in x["union_components"] if c["sd"] < 0.13]
    res["union_narrow_components"] = un
    # centres
    cen = []
    for x in p3:
        for c in x["centres"]:
            locw = [w["f2_prod_cond"] for w in p1 if abs(w["c1"] - x["c1"]) < 1e-3
                    and abs(w["cv2_mean"] - c["centre"]) <= c["sampled_sigma_swarm_pred"]]
            cen.append({**c, "c1": x["c1"], "f2_local": float(np.mean(locw)) if locw else None})
    cu = [c["compression_realised"] for c in cen]
    lo = [c for c in cen if c["f2_local"] is not None]
    cl = [c["k2_shape"] / (c["k2_shape"] + c["f2_local"]) for c in lo]
    res["centres"] = {"n": len(cen),
                      "compression_union_model_q": q(cu), "n_below_0p5_union": sum(1 for v in cu if v < 0.5),
                      "n_below_0p25_union": sum(1 for v in cu if v < 0.25),
                      "n_with_local_window": len(lo), "compression_local_q": q(cl),
                      "n_below_0p5_local": sum(1 for v in cl if v < 0.5),
                      "n_above_0p7_local": sum(1 for v in cl if v > 0.7),
                      "f2_local_over_swarm_q": q([c["f2_local"] / c["f2_swarm"] for c in lo]),
                      "f2_union_over_swarm_q": q([c["f2_union"] / c["f2_swarm"] for c in cen]),
                      "sigma_ratio_swarm_over_union_design_q": q([c["sigma_ratio_if_union_f2"] for c in cen]),
                      "n_extrapolated": sum(1 for c in cen if c["extrapolated"]),
                      "extrapolated": [[round(c["c1"], 4), round(c["centre"], 3)] for c in cen if c["extrapolated"]]}
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
