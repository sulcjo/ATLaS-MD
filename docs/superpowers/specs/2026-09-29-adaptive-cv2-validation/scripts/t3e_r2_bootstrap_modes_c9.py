"""chignolin_9 final-combined R2 (read-only): every heavy interval's bootstrap sigma under
--ap-coverage-bootstrap fixed-f and resolve-f, wall time and memory (T2 9.10 / T3 10.8).

python t3e_r2_bootstrap_modes_c9.py fixed-f resolve-f   (writes c9_sigmas.json in the cwd)
"""
import json, sys, time, resource
from pathlib import Path
import numpy as np
sys.path.insert(0, '/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler')
import gareus.adaptive_production as ap
from gareus.adaptive import cv2_coverage as cov, cv2_resolution_io as cio
from argparse import Namespace
A = Path('/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_9/adaptive_production')
diag = json.load(open('/home/sulcjo/.claude/jobs/45d68034/tmp/c9dry/replay/adaptive_final_combined_diagnostics.json'))
reg = ap.WindowStateRegistry.load(A)
rec = []
orig = cov._interval_stats
def spy(col, *a, **k):
    out = orig(col, *a, **k); frac, _ne, _nc, sig = out
    rec.append((col["c1"], frac, sig)); return out
cov._interval_stats = spy
res = {}
for mode in sys.argv[1:]:
    rec.clear()
    pol = ap.AdaptiveDecisionPolicy(cv2_resolution=True, edge_metric="pairwise-mbar", coverage_bootstrap=mode)
    st = cio.settings_from(pol, Namespace(temperature_k=300.0, cv2_k_min=0.0, cv2_k_max=1000.0, cv1_k_min=0.0))
    t = time.time(); cands, status = cio.union_coverage(A / 'adaptive_union_mbar.npz', cio._column_views(reg, diag), st, epoch=2, max_gb=16.0); wall = time.time() - t
    heavy = [(c1, float(s)) for c1, fr, sg in rec for f, s in zip(fr, sg) if f >= cov.MIN_INTERVAL_WEIGHT]
    sig = np.array([s for _, s in heavy])
    res[mode] = {"wall_s": wall, "mbar": status["mbar"], "boot": status["bootstrap"], "n_heavy": len(heavy),
                 "sigma_q50_max": [float(np.median(sig)), float(np.max(sig))], "n_gt_0.25": int((sig > 0.25).sum()), "n_gt_0.5": int((sig > 0.5).sum()),
                 "sigmas": [s for _, s in heavy],
                 "proposed": [(round(c["metrics"]["c1"], 4), [round(x, 3) for x in c["metrics"]["interval"]], c["metrics"]["pmf_sigma_kT"]) for c in cands if c["decision"] == "proposed"],
                 "maxrss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    print(mode, {k: v for k, v in res[mode].items() if k != "sigmas"}, flush=True)
if len(res) == 2:
    a, b = np.array(res["fixed-f"]["sigmas"]), np.array(res["resolve-f"]["sigmas"])
    print("resolve/fixed ratio q10/50/90", np.quantile(b / a, [0.1, 0.5, 0.9]))
json.dump(res, open("c9_sigmas.json", "w"), indent=1, default=float)
