"""Read-only respring (--ap-cv2-respring) dry run on a replayed chignolin_9 payload.

Usage: python t3d_respring_dryrun.py <adaptive_dir> <payload.json> <phase|final_combined> <out_dir>
       [--no-x5] [--t3c t3c_f2_crosscheck.json]
Runs the shipped proposer (gareus.adaptive.cv2_respring_io.propose_respring) twice: with the
default per-epoch cap (0.25 of the centres) and with the cap off (max_fraction 1.0). g comes
from the X5 estimator over the phase's Parquet sources (read-only) unless --no-x5 (then the
P4 subsample's own g). Writes <out_dir>/respring_<phase>{,_nocap}.json and a compact
<out_dir>/respring_<phase>_summary.json. Nothing under RUNS is written.
"""
import json
import sys
import time
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np

import gareus.adaptive_production as ap
from gareus.adaptive import cv2_respring_io as rio

adaptive_dir, payload_path, phase, out_dir = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
out_dir.mkdir(parents=True, exist_ok=True)
t3c = Path(sys.argv[sys.argv.index("--t3c") + 1]) if "--t3c" in sys.argv else None
diag = json.loads(payload_path.read_text())
reg = ap.WindowStateRegistry.load(adaptive_dir)
policy = ap.AdaptiveDecisionPolicy(cv2_respring=True)
args = Namespace(temperature_k=300.0, cv2_k_min=0.0, cv2_k_max=1000.0, cv1_k_min=0.0)
plan, plan_src = rio.find_layout_plan(adaptive_dir.parent, reg)
settings = rio.settings_from(policy, args, plan, plan_src)
t0 = time.time()
views, n_centres = rio.state_rows(reg, diag)
ids = [v["state_id"] for v in views if v.get("c2") is not None and (v.get("k2") or 0) > 0]
effective = {}
if "--no-x5" not in sys.argv:
    sources = (ap._final_sample_dirs(adaptive_dir) if phase == "final_combined"
               else ap._sample_sources_from_run_root(phase, adaptive_dir / phase))
    effective = rio.effective_for(reg, sources, ids)
t_x5 = time.time() - t0
subs = rio._subsamples(diag)
res = {}
for label, s in (("cap", settings), ("nocap", replace(settings, respring_max_fraction=1.0))):
    _acts, rep = rio.propose_respring(reg, diag, [], s, epoch=3, effective=effective, subsamples=subs)
    rep["stage"] = "replay"
    (out_dir / f"respring_{phase}{'' if label == 'cap' else '_nocap'}.json").write_text(json.dumps(rep, indent=1))
    res[label] = rep


def q(xs, qs=(0.0, 0.1, 0.5, 0.9, 1.0)):
    xs = np.asarray([x for x in xs if x is not None], dtype=float)
    return None if not xs.size else [round(float(np.quantile(xs, p)), 3) for p in qs]


nocap = res["nocap"]["candidates"]
meas = [c for c in nocap if (c.get("metrics") or {}).get("c_interval") is not None]
trig = [c for c in nocap if c.get("triggered")]
width = [c["metrics"]["c_interval"][1] - c["metrics"]["c_interval"][0] for c in meas]
out = {"phase": phase, "payload": str(payload_path), "sigma_target_source": settings.sigma_target_source,
       "layout_plan": plan_src, "n_centres": n_centres, "n_candidates": len(nocap), "n_measured": len(meas),
       "wall_s": {"x5": round(t_x5, 1), "total": round(time.time() - t0, 1)},
       "g_cv2_source": sorted({str(c["metrics"].get("g_cv2_source")) for c in nocap if c.get("metrics")}),
       "g_variance_source": sorted({str(c["metrics"].get("g_variance_source")) for c in nocap if c.get("metrics")}),
       # report v2: g_variance = max(g_cv2, g_q) sets n_eff and the blocks (v1: the CV2 series' g)
       "g_cv2_q": q([c["metrics"].get("g_cv2") for c in nocap if c.get("metrics")]),
       "g_qobs_q": q([c["metrics"].get("g_q") for c in nocap if c.get("metrics")]),
       "g_variance_q": q([c["metrics"].get("g_variance") for c in meas]),
       "g_q_over_g_cv2_q": q([c["metrics"]["g_q"] / c["metrics"]["g_cv2"] for c in nocap
                              if (c.get("metrics") or {}).get("g_q") and (c.get("metrics") or {}).get("g_cv2")]),
       "n_eff_q": q([c["metrics"].get("n_eff") for c in meas]),
       "k2_q": q([c["k2"] for c in meas]),
       "c_real_q": q([c["metrics"]["c_real"] for c in meas]),
       "c_interval_width_q": q(width),
       "f2_prod_q": q([c["metrics"]["f2_prod"] for c in meas]),
       "n_c_point_below_0p5": sum(1 for c in meas if c["metrics"]["c_real"] < 0.5),
       "n_c_point_below_trigger": sum(1 for c in meas if c["metrics"]["c_real"] < settings.trigger_below),
       "n_triggered": len(trig),
       "triggered_c_real_q": q([c["metrics"]["c_real"] for c in trig]),
       "triggered_c_hi_q": q([c["metrics"]["c_interval"][1] for c in trig]),
       "proposed_k2_new_q": q([c["proposal"]["k2_new"] for c in trig if c.get("proposal")]),
       "proposed_k2_ratio_q": q([c["proposal"]["k2_ratio"] for c in trig if c.get("proposal")]),
       "proposed_predicted_c_q": q([c["proposal"]["predicted_c"] for c in trig if c.get("proposal")]),
       "n_over_compressed": sum(1 for c in meas if c["metrics"].get("over_compressed")),
       "cv1_free_triggered": [c["state_id"] for c in trig if not c.get("k1")],
       "summary_cap": res["cap"]["summary"], "cap": res["cap"]["cap"], "summary_nocap": res["nocap"]["summary"],
       "triggered": [{"state": c["state_id"], "c1": round(c["c1"], 4), "c2": round(c["c2"], 3), "k1": c["k1"],
                      "c_real": round(c["metrics"]["c_real"], 3),
                      "c_interval": [round(x, 3) for x in c["metrics"]["c_interval"]],
                      "f2_prod": round(c["metrics"]["f2_prod"], 3), "n_eff": round(c["metrics"]["n_eff"], 0),
                      "k2_new": None if not c.get("proposal") else round(c["proposal"]["k2_new"], 3),
                      "decision_cap": next(x["decision"] for x in res["cap"]["candidates"]
                                           if x["state_id"] == c["state_id"])} for c in trig]}
if t3c is not None:
    ref = {int(r["state"]): r for r in json.loads(t3c.read_text())["window_local"]}
    pairs = [(c, ref[c["state_id"]]) for c in meas if c["state_id"] in ref]
    rc = [c["metrics"]["f2_prod_conditional"] / r["f2_prod_cond"] for c, r in pairs
          if c["metrics"].get("f2_prod_conditional")]
    rm = [c["metrics"]["f2_prod"] / r["f2_prod_cond"] for c, r in pairs]
    out["t3c_compare"] = {"n": len(pairs), "cond_over_t3c_q": q(rc), "marginal_over_t3c_q": q(rm),
                          "n_t3c_below_0p5": sum(1 for _c, r in pairs if 1.18 / (1.18 + r["f2_prod_cond"]) < 0.5)}
(out_dir / f"respring_{phase}_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "triggered"}, indent=1))
