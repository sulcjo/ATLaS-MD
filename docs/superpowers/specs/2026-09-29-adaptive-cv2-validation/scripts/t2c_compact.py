"""Compact T2 calibration record for t2_data/ from the t2c summary and the e2e outputs.

python t2c_compact.py SUMMARY.json OUT.json E2E.json [E2E.json ...]
"""
import json
import sys
from collections import defaultdict

import numpy as np

HOLES = ("plateau-hardgap-k20", "plateau-hardgap-k30", "plateau-hardgap-k45", "harmonic-rowgap", "plateau-rowgap")
R2_COMBOS = ("1.0|0.25", "1.0|0.5", "2.0|0.25", "2.0|0.5", "2.0|1.0", "2.0|inf", "3.0|0.25", "3.0|0.5", "3.0|inf")
R3_DEC = ("replica|10|x4", "state_series|10|x4", "replica_path|10|x4", "replica_path|20|x4", "replica_path|50|x4",
          "replica|10|x32", "state_series|10|x32", "replica_path|10|x32", "replica_path|50|x32")


def e2e_r3(files):
    agg = defaultdict(list)
    seen = set()
    for f in files:
        for r in json.load(open(f)).get("r3", []):
            key = (r["scenario"], r["seed"], r.get("parent_min_transitions", 10))
            if key in seen or not r["arms"]:
                continue
            seen.add(key)
            base = r["arms"]["none"]["columns"]
            for arm, v in r["arms"].items():
                for c, b in zip(v["columns"], base):
                    agg[(r["scenario"], key[2], arm)].append(c | {"rmse_none": b.get("rmse_kT"),
                                                                   "pop_none": b.get("pop_lnodds_err_kT")})
    out = {}
    for (sc, mt, arm), cols in sorted(agg.items()):
        rm = np.array([c.get("rmse_kT", np.nan) for c in cols], float)
        rb = np.array([c["rmse_none"] if c["rmse_none"] is not None else np.nan for c in cols], float)
        pm = np.array([c.get("pop_lnodds_err_kT") if c.get("pop_lnodds_err_kT") is not None else np.nan for c in cols], float)
        pb = np.array([c["pop_none"] if c["pop_none"] is not None else np.nan for c in cols], float)
        ov = [o for c in cols for o in (c.get("child_parent_overlap") or [])]
        k2 = [k for c in cols for k in (c.get("child_k2") or [])]
        out[f"{sc}|min_t{mt}|{arm}"] = {
            "n_columns": len(cols), "rmse_mean_kT": round(float(np.nanmean(rm)), 4),
            "rmse_none_mean_kT": round(float(np.nanmean(rb)), 4), "n_worse_than_none": int(np.sum(rm > rb)),
            "pop_err_mean_kT": round(float(np.nanmean(pm)), 4), "pop_err_none_mean_kT": round(float(np.nanmean(pb)), 4),
            "n_pop_worse": int(np.sum(pm > pb)),
            "child_parent_overlap_min_med_max": [round(float(x), 3) for x in np.quantile(ov, [0, 0.5, 1])] if ov else None,
            "child_k2_min_med_max": [round(float(x), 2) for x in np.quantile(k2, [0, 0.5, 1])] if k2 else None}
    return out


def e2e_r3_per_seed(files):
    """Paired per-seed mean differences (arm - none) over that seed's parent columns: columns
    within a seed share one base run, so the seed (n = 3) is the unit."""
    out, seen = defaultdict(list), set()
    for f in files:
        for r in json.load(open(f)).get("r3", []):
            key = (r["scenario"], r["seed"])
            if key in seen or not r["arms"]:
                continue
            seen.add(key)
            base = r["arms"]["none"]["columns"]
            for arm in ("cap4_forced", "uncapped"):
                cols = r["arms"][arm]["columns"]
                nan = float("nan")
                dr = np.nanmean([(c.get("rmse_kT") or nan) - (b.get("rmse_kT") or nan) for c, b in zip(cols, base)])
                dp = np.nanmean([(c.get("pop_lnodds_err_kT") if c.get("pop_lnodds_err_kT") is not None else nan)
                                 - (b.get("pop_lnodds_err_kT") if b.get("pop_lnodds_err_kT") is not None else nan)
                                 for c, b in zip(cols, base)])
                out[f"{r['scenario']}|{arm}"].append({"seed": r["seed"], "n_parents": r["n_parents"],
                                                      "parent_min_transitions": r.get("parent_min_transitions", 10),
                                                      "d_rmse_kT": round(float(dr), 4), "d_pop_err_kT": round(float(dp), 4)})
    return dict(out)


def e2e_r2(files):
    agg = defaultdict(list)
    for f in files:
        for r in json.load(open(f)).get("r2", []):
            agg[(r["scenario"], r["combo"])].append(r)
    out = {}
    for (sc, combo), rs in sorted(agg.items()):
        out[f"{sc}|{combo}"] = {"seeds": len(rs), "n_proposed": [r["n_proposed"] for r in rs],
                                "n_in_hole": [sum(p["in_hole"] for p in r["proposals"]) for r in rs],
                                "hole_rmse_none_kT": [round(r["arms"]["none"]["rmse_mean_kT"], 3) for r in rs],
                                "hole_rmse_add_kT": [round(r["arms"]["add"]["rmse_mean_kT"], 3) if "add" in r["arms"] else None
                                                     for r in rs]}
    return out


def main(summary, out, *e2e):
    s = json.load(open(summary))
    res = {"n_jobs": s["n_jobs"], "scenarios": s["scenarios"], "seeds": s["seeds"], "regime_stats": s["regime_stats"],
           "pop_ok_kT": s["pop_ok_kT"], "controls": s["controls"], "r3": {}, "r2": {}}
    for key in ("indep|2000", "indep|8000", "re|2000", "re|8000"):
        r3 = s["r3"]
        res["r3"][key] = {"transitions": r3["transitions"][key],
                          "threshold_sweep": r3["threshold_sweep"][key],
                          "growth": r3["growth"][key], "min_sigma": r3["min_sigma"][key],
                          "controls_mixture_fp": r3["controls_mixture_fp"][key],
                          "resolvable_missed_by_gate": r3["resolvable_missed_by_gate"][key]}
        r2 = s["r2"][key]
        res["r2"][key] = {"sigma_calibration": r2["sigma_calibration"],
                          "combos": {c: {sc: {k: v for k, v in d.items() if k in ("proposed", "proposed_in_hole",
                                                                                     "hole_columns", "hole_columns_hit",
                                                                                     "by_contrib_only", "by_sigma_only",
                                                                                     "both", "proposed_err_q50")}
                                         for sc, d in r2["combos"][c].items() if d["proposed"] or sc in HOLES}
                                     for c in R2_COMBOS}}
    res["r3_decisions"] = {k: {**{x: v[x] for x in ("runs", "windows", "exact_resolvable", "mixture_passed",
                                                    "mixture_passed_exact_resolvable")},
                               "decisions": {d: v["decisions"].get(d) for d in R3_DEC}}
                           for k, v in s["r3_decisions"].items() if v["mixture_passed"] or v["exact_resolvable"]}
    res["e2e_r3"] = e2e_r3(e2e)
    res["e2e_r3_per_seed"] = e2e_r3_per_seed(e2e)
    res["e2e_r2"] = e2e_r2(e2e)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
