"""Reduce the t2c job JSONs to the compact calibration table (t2_data/t2c_calibration.json).

python t2c_summarise.py JOBS_DIR OUT.json
"""
from __future__ import annotations

import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2c_eval as E  # noqa: E402

CONTROLS = ("harmonic-bowl", "stiff-bowl", "plateau-walls", "harmonic-drop1", "plateau-drop1", "dbl-gap")
POP_OK = 0.5          # kT: a window's mode population (ln-odds) within 0.5 kT of exact = equilibrated
EST = ("t_replica", "t_state_series", "t_replica_path")


def load(jobs_dir):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(f"{jobs_dir}/*.json"))]


def _q(a, ps=(0.1, 0.5, 0.9)):
    a = [x for x in a if x is not None and math.isfinite(x)]
    return [round(float(np.quantile(a, p)), 3) for p in ps] if a else None


def r3_tables(jobs):
    out = {"gates": {}, "transitions": {}, "threshold_sweep": {}, "growth": {}, "controls_mixture_fp": {},
           "resolvable_missed_by_gate": {}, "min_sigma": {}}
    for regime in ("indep", "re"):
        for n in ("2000", "8000"):
            key = f"{regime}|{n}"
            gates = defaultdict(Counter)
            trans = defaultdict(lambda: defaultdict(list))
            passed = []            # (scenario, rec)
            ctrl = Counter()
            ctrl_runs = Counter()
            missed = defaultdict(Counter)
            for j in jobs:
                if j["regime"] != regime:
                    continue
                recs = j["r3"][n]
                sc = j["scenario"]
                if sc in CONTROLS:
                    ctrl_runs[sc] += 1
                for r in recs:
                    gates[sc][r["gate"]] += 1
                    if r["exact_resolvable"] and r["gate"] != "passed" and sc not in CONTROLS:
                        missed[sc][r["gate"]] += 1
                    if r["gate"] != "passed":
                        continue
                    if sc in CONTROLS:
                        ctrl[sc] += 1
                    # E / T only for exact-resolvable windows; a mixture pass on a window whose exact
                    # density has no resolvable pair is "spurious" (non-equilibrium / hidden-mode bimodality)
                    if sc in CONTROLS:
                        cls = "control"
                    elif not r["exact_resolvable"]:
                        cls = "spurious"
                    else:
                        cls = "E" if r["pop_lnodds_err"] <= POP_OK else "T"
                    r = {**r, "cls": cls, "scenario": sc}
                    passed.append(r)
                    for e in EST:
                        trans[cls][e].append(r.get(e))
            out["gates"][key] = {sc: dict(c) for sc, c in gates.items()}
            out["controls_mixture_fp"][key] = {sc: {"windows_passing_mixture_gate": ctrl[sc], "runs": ctrl_runs[sc]}
                                               for sc in ctrl_runs}
            out["resolvable_missed_by_gate"][key] = {sc: dict(c) for sc, c in missed.items()}
            out["transitions"][key] = {cls: {e: {"n": len(v), "q10_50_90": _q(v),
                                                 "per_1000_q50": (round(float(np.median(v)) * 1000 / int(n), 2)
                                                                  if v else None)}
                                             for e, v in d.items()} for cls, d in trans.items()}
            sweep = {}
            for e in EST:
                for thr in E.MIN_TRANSITIONS:
                    row = {}
                    for cls in ("E", "T", "spurious", "control"):
                        vals = [r.get(e) for r in passed if r["cls"] == cls]
                        vals = [v for v in vals if v is not None]
                        row[cls] = {"n": len(vals), "pass": sum(1 for v in vals if v >= thr)}
                    sweep[f"{e}|{thr}"] = row
            out["threshold_sweep"][key] = sweep
            grow = {}
            for g in E.GROWTH:
                need = [max(k["growth_needed"] for k in r["children"]) for r in passed if r.get("children")]
                grow[str(g)] = {"n_candidates": len(need), "refused_spring_cap": sum(1 for x in need if x > g * (1 + 1e-9))}
            needs = [max(k["growth_needed"] for k in r["children"]) for r in passed if r.get("children")]
            out["growth"][key] = {"sweep": grow, "growth_needed_q10_50_90": _q(needs),
                                  "by_scenario": {sc: _q([max(k["growth_needed"] for k in r["children"])
                                                          for r in passed if r["scenario"] == sc and r.get("children")])
                                                  for sc in sorted({r["scenario"] for r in passed})}}
            sig = [k["sigma_target"] for r in passed for k in (r.get("children") or [])]
            ratio = [k["sigma_target"] / (1.0 / math.sqrt(r["k2"])) for r in passed for k in (r.get("children") or [])]
            out["min_sigma"][key] = {"sigma_target_q10_50_90": _q(sig), "n": len(sig),
                                     "binding_at": {str(s): sum(1 for x in sig if x < s) for s in (0.05, 0.1, 0.2)},
                                     "sigma_target_over_parent_sigma_w_q10_50_90": _q(ratio)}
    return out


def r3_decisions(jobs):
    """Per scenario / regime / N: exact-resolvable windows, mixture passes, and the final R3
    decision under (estimator, min_transitions, growth cap) combinations."""
    combos = [(e, thr, g) for e in EST for thr in (10, 20, 50) for g in (4.0, 32.0)]
    out = {}
    for j in jobs:
        for n, recs in j["r3"].items():
            key = f"{j['scenario']}|{j['regime']}|{n}"
            row = out.setdefault(key, {"runs": 0, "windows": 0, "exact_resolvable": 0, "mixture_passed": 0,
                                       "mixture_passed_exact_resolvable": 0, "decisions": {}})
            row["runs"] += 1
            row["windows"] += len(recs)
            for r in recs:
                row["exact_resolvable"] += int(r["exact_resolvable"])
                if r["gate"] != "passed":
                    continue
                row["mixture_passed"] += 1
                row["mixture_passed_exact_resolvable"] += int(r["exact_resolvable"])
                need = max(k["growth_needed"] for k in r["children"]) if r.get("children") else None
                for e, thr, g in combos:
                    dkey = f"{e[2:]}|{thr}|x{g:g}"
                    d = row["decisions"].setdefault(dkey, Counter())
                    if (r.get(e) or 0) < thr:
                        d["flagged_trapped"] += 1
                    elif need is not None and need > g * (1 + 1e-9):
                        d["refused_spring_cap"] += 1
                    else:
                        d["proposed"] += 1
    for row in out.values():
        row["decisions"] = {k: dict(v) for k, v in row["decisions"].items()}
    return out


def r2_tables(jobs):
    out = {}
    for regime in ("indep", "re"):
        for n in ("2000", "8000"):
            key = f"{regime}|{n}"
            res = {}
            calib = []
            for j in jobs:
                if j["regime"] != regime:
                    continue
                for c in j["r2"][n]["columns"]:
                    for fr, sg, er, nc in zip(c["frac"], c["sigma"], c["err_kT"], c["n_contrib"]):
                        if fr >= 0.02 and sg is not None and er is not None:
                            calib.append((j["scenario"], sg, er, nc))
            for combo in [f"{a}|{b}" for a in E.COVERAGE_MIN_WINDOWS for b in E.PMF_SIGMA]:
                per = defaultdict(lambda: {"runs": set(), "proposed": 0, "proposed_in_hole": 0, "hole_columns": 0,
                                           "hole_columns_hit": 0, "flagged_no_action": 0, "by_contrib_only": 0,
                                           "by_sigma_only": 0, "both": 0, "proposed_err": []})
                for j in jobs:
                    if j["regime"] != regime:
                        continue
                    s = per[j["scenario"]]
                    s["runs"].add(j["seed"])
                    for c in j["r2"][n]["columns"]:
                        props = c["combos"][combo]
                        if any(c["in_hole"]):
                            s["hole_columns"] += 1
                            if any(p["decision"] == "proposed" and p["in_hole"] for p in props):
                                s["hole_columns_hit"] += 1
                        for p in props:
                            if p["decision"] == "proposed":
                                s["proposed"] += 1
                                s["proposed_in_hole"] += int(p["in_hole"])
                                s["proposed_err"].append(p["max_err_kT"])
                                if p["by_contrib"] and p["by_sigma"]:
                                    s["both"] += 1
                                elif p["by_contrib"]:
                                    s["by_contrib_only"] += 1
                                else:
                                    s["by_sigma_only"] += 1
                            else:
                                s["flagged_no_action"] += 1
                res[combo] = {sc: {**{k: v for k, v in d.items() if k not in ("runs", "proposed_err")},
                                   "runs": len(d["runs"]), "proposed_err_q50": (_q(d["proposed_err"], (0.5,)) or [None])[0]}
                              for sc, d in per.items()}
            sg = np.array([x[1] for x in calib]); er = np.array([x[2] for x in calib])
            nc = np.array([x[3] for x in calib])
            cal = {"n_intervals": int(sg.size), "err_over_sigma_q50": round(float(np.median(er / np.maximum(sg, 1e-9))), 2),
                   "err_over_sigma_q90": round(float(np.quantile(er / np.maximum(sg, 1e-9), 0.9)), 2),
                   "n_contrib_q10_50_90": _q(list(nc)), "frac_intervals_n_contrib_lt2": round(float(np.mean(nc < 2)), 3)}
            for thr in (0.25, 0.5, 1.0):
                big = er > 0.5
                cal[f"sigma_gt_{thr}"] = {"flag_rate": round(float(np.mean(sg > thr)), 3),
                                          "recall_err_gt_0.5kT": round(float(np.mean(sg[big] > thr)), 3) if big.any() else None,
                                          "precision_err_gt_0.5kT": (round(float(np.mean(er[sg > thr] > 0.5)), 3)
                                                                     if (sg > thr).any() else None)}
            cal["n_err_gt_0.5kT"] = int((er > 0.5).sum())
            out[key] = {"combos": res, "sigma_calibration": cal}
    return out


def main(jobs_dir, out_path):
    jobs = load(jobs_dir)
    sims = defaultdict(list)
    for j in jobs:
        sims[j["regime"]].append((j["sim"]["mean_residence"], j["sim"]["swap_acceptance"], j["wall_s"]))
    out = {"n_jobs": len(jobs), "scenarios": sorted({j["scenario"] for j in jobs}),
           "seeds": sorted({j["seed"] for j in jobs}),
           "regime_stats": {r: {"mean_residence_q50": _q([x[0] for x in v], (0.5,)),
                                "swap_acceptance_q50": _q([x[1] for x in v], (0.5,)),
                                "job_wall_s_total": round(sum(x[2] for x in v))} for r, v in sims.items()},
           "pop_ok_kT": POP_OK, "controls": CONTROLS,
           "r3": r3_tables(jobs), "r3_decisions": r3_decisions(jobs), "r2": r2_tables(jobs)}
    Path(out_path).write_text(json.dumps(out, indent=1, default=float))
    print("wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
