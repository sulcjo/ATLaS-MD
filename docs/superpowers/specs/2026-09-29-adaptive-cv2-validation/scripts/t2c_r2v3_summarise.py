"""Reduce t2c_r2v3 job JSONs (R2 same-column counting + bootstrap block rules) to one table.

python t2c_r2v3_summarise.py JOBS_DIR [RESOLVE_JOBS_DIR] OUT.json [--old-jobs /tmp/t2c/jobs]
"""
from __future__ import annotations

import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HOLES = ("plateau-hardgap-k20", "plateau-hardgap-k30", "plateau-hardgap-k45")
ROWGAPS = ("harmonic-rowgap", "plateau-rowgap")
CELLS = ("indep|2000", "indep|8000", "re|2000", "re|8000")


def load(d):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(f"{d}/*.json"))]


def _arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def combo_table(jobs):
    out = {}
    combos = sorted(jobs[0]["r2"]["2000"]["columns"][0]["combos"])
    for combo in combos:
        row = {}
        for cell in CELLS:
            regime, n = cell.split("|")
            hit = defaultdict(int)
            n_hole_cols = defaultdict(int)
            fp = {"n": 0, "where": defaultdict(int), "err_le_0.2": 0, "err_gt_0.5": 0}
            for j in jobs:
                if j["regime"] != regime:
                    continue
                sc = j["scenario"]
                for c in j["r2"][n]["columns"]:
                    props = [p for p in c["combos"][combo] if p["decision"] == "proposed"]
                    if any(c["in_hole"]):
                        n_hole_cols[sc] += 1
                        hit[sc] += int(any(p["in_hole"] for p in props))
                    if sc in HOLES:
                        continue
                    for p in props:
                        fp["n"] += 1
                        fp["where"][sc] += 1
                        e = p["max_err_kT"]
                        fp["err_le_0.2"] += int(e is not None and e <= 0.2)
                        fp["err_gt_0.5"] += int(e is not None and e > 0.5)
            row[cell] = {"holes_hit": {sc: f"{hit[sc]}/{n_hole_cols[sc]}" for sc in (*HOLES, *ROWGAPS)
                                       if n_hole_cols[sc]},
                         "non_hardgap_proposals": {**fp, "where": dict(fp["where"])}}
        out[combo] = row
    return out


def sigma_calibration(jobs, rules):
    out = {}
    for cell in CELLS:
        regime, n = cell.split("|")
        rec = {}
        for rule in rules:
            ratios, within1, within2, big_flag = [], [], [], {0.25: [], 0.5: []}
            per_sc = defaultdict(list)
            for j in jobs:
                if j["regime"] != regime or n not in j["r2"]:
                    continue
                for c in j["r2"][n]["columns"]:
                    if rule not in c["sigma"]:
                        continue
                    fr, er, sg = np.asarray(c["frac"]), _arr(c["err_kT"]), _arr(c["sigma"][rule])
                    ok = (fr >= 0.02) & np.isfinite(er) & np.isfinite(sg) & (sg > 0)
                    r = er[ok] / sg[ok]
                    ratios.extend(r.tolist())
                    per_sc[j["scenario"]].extend(r.tolist())
                    within1.extend((er[ok] <= sg[ok]).tolist())
                    within2.extend((er[ok] <= 2 * sg[ok]).tolist())
                    for thr in big_flag:
                        big = er[ok] > 0.5
                        big_flag[thr].extend((sg[ok][big] > thr).tolist())
            if not ratios:
                continue
            rec[rule] = {"n_intervals": len(ratios),
                         "err_over_sigma_q50": round(float(np.median(ratios)), 2),
                         "err_over_sigma_q90": round(float(np.quantile(ratios, 0.9)), 2),
                         "frac_err_le_sigma": round(float(np.mean(within1)), 3),
                         "frac_err_le_2sigma": round(float(np.mean(within2)), 3),
                         "recall_err_gt_0.5_at_sigma_gt": {str(k): round(float(np.mean(v)), 3) if v else None
                                                           for k, v in big_flag.items()},
                         "by_scenario_q50": {sc: round(float(np.median(v)), 2) for sc, v in sorted(per_sc.items())}}
        out[cell] = rec
    return out


def block_stats(jobs):
    out = defaultdict(list)
    for j in jobs:
        for n, r in j["r2"].items():
            for rule, info in (r.get("blocks") or {}).items():
                if info:
                    out[f"{j['regime']}|{n}|{rule}"].append((info["g_q10_50_90"][1], info["n_min_blocks_bound"],
                                                             info["n_states"]))
    return {k: {"g_median_q50_over_jobs": round(float(np.median([x[0] for x in v])), 1),
                "min_blocks_bound_states": int(sum(x[1] for x in v)), "states": int(sum(x[2] for x in v))}
            for k, v in sorted(out.items())}


def reproduction(jobs, old_dir):
    worst, n, finite_mismatch, contrib_mismatch = 0.0, 0, 0, 0
    for j in jobs:
        p = Path(old_dir) / f"{j['scenario']}__{j['regime']}__s{j['seed']}.json"
        if not p.exists():
            continue
        o = json.loads(p.read_text())
        for n_ in ("2000", "8000"):
            for cn, co in zip(j["r2"][n_]["columns"], o["r2"][n_]["columns"]):
                a, b = _arr(cn["sigma"]["fixed20"]), _arr(co["sigma"])
                ok = np.isfinite(a) & np.isfinite(b)
                finite_mismatch += int(not np.array_equal(np.isfinite(a), np.isfinite(b)))
                if ok.any():
                    worst = max(worst, float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(b[ok], 1e-12))))
                    n += int(ok.sum())
                contrib_mismatch += int(cn["n_contrib_any"] != co["n_contrib"])
    return {"n_intervals_compared": n, "max_rel_diff_fixed20_sigma_vs_t2c": worst,
            "columns_finite_pattern_mismatch": finite_mismatch, "columns_n_contrib_mismatch": contrib_mismatch}


def main(argv):
    old = argv[argv.index("--old-jobs") + 1] if "--old-jobs" in argv else "/tmp/t2c/jobs"
    pos = [a for a in argv if not a.startswith("--") and a != old]
    jobs = load(pos[0])
    resolve = load(pos[1]) if len(pos) == 3 else []
    out_path = pos[-1]
    out = {"n_jobs": len(jobs), "reproduction": reproduction(jobs, old), "blocks": block_stats(jobs),
           "sigma_calibration": sigma_calibration(jobs, ("fixed20", "g2", "g5")),
           "combos": combo_table(jobs)}
    if resolve:
        out["resolve_f"] = {"n_jobs": len(resolve),
                            "sigma_calibration": sigma_calibration(resolve, ("fixed20", "g5", "g5_resolve_f"))}
    Path(out_path).write_text(json.dumps(out, indent=1))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
