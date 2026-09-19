import json
import numpy as np

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
nat = json.load(open(f"{OUT}/native_reference.json"))
N = z["q"].size
print(f"frames = {N:,}   failed files = {int(z['n_bad'][0])}   files = {len(z['files'])}\n")

nm, ns = nat["models_mean"], nat["models_std"]
nmin, nmax = nat["models_min"], nat["models_max"]
print("NATIVE (1UAO, 18 NMR models) vs ENSEMBLE, on identical definitions")
print(f"{'metric':10s} {'native mean+/-sd':>20s} {'native range':>16s} {'ens min':>9s} {'ens max':>9s} {'ens mean':>9s}")
for k in ["cv1", "rg", "d1", "d2", "yw", "q", "pca1", "pca2"]:
    a = z[k]
    print(f"{k:10s} {nm[k]:11.3f}+/-{ns[k]:<6.3f} {nmin[k]:7.2f}-{nmax[k]:<7.2f} "
          f"{a.min():9.3f} {a.max():9.3f} {a.mean():9.3f}")

print("\nBACKBONE RMSD (res 2-9, 32 atoms) TO NATIVE")
for k, lab in [("rmsd_bb_min18", "min over 18 NMR models"), ("rmsd_bb_m1", "vs model 1"),
               ("rmsd_ca_m1", "CA res1-10 vs model 1")]:
    a = z[k]
    print(f"  {lab:24s} min {a.min():6.3f}  p0.01 {np.percentile(a,0.01):6.3f}  "
          f"p1 {np.percentile(a,1):6.3f}  median {np.median(a):6.3f}  max {a.max():6.3f}")
print(f"  NMR-ensemble internal spread: mean {nat['model_pairwise_bb_rmsd_A']['mean']:.3f}, "
      f"max {nat['model_pairwise_bb_rmsd_A']['max']:.3f} A  <- the resolution floor for 'native'")

r = z["rmsd_bb_min18"]
q = z["q"]
yw = z["yw"]
print("\nHOW MANY FRAMES ARE NATIVE?  (thresholds fixed before counting)")
print(f"{'criterion':52s} {'frames':>9s} {'fraction':>11s}")
rows = [
    ("bb RMSD <= 1.0 A (inside NMR ensemble spread)", r <= 1.0),
    ("bb RMSD <= 1.5 A (strict native)", r <= 1.5),
    ("bb RMSD <= 2.0 A (native-like)", r <= 2.0),
    ("bb RMSD <= 2.5 A (loose)", r <= 2.5),
    ("bb RMSD <= 3.0 A (very loose)", r <= 3.0),
    ("Q >= 0.80 (native contact fraction)", q >= 0.80),
    ("Q >= 0.70", q >= 0.70),
    ("Q >= 0.60", q >= 0.60),
    ("Tyr2/Trp9 cluster <= 6.0 A (native 5.5+/-0.3)", yw <= 6.0),
    ("RMSD<=2.0 AND Q>=0.6 AND cluster<=7 A", (r <= 2.0) & (q >= 0.6) & (yw <= 7.0)),
    ("RMSD<=2.5 AND Q>=0.5", (r <= 2.5) & (q >= 0.5)),
]
for lab, m in rows:
    n = int(m.sum())
    print(f"{lab:52s} {n:9,d} {n/N:11.3e}")

print("\nPRIOR CANDIDATE FILTER (d1<4.0 & d2<6.0 A) vs the native structure itself")
old = (z["d1"] < 4.0) & (z["d2"] < 6.0)
print(f"  frames passing: {int(old.sum()):,} ({old.sum()/N:.3e})")
print(f"  native d2 over 18 models: {nmin['d2']:.2f}-{nmax['d2']:.2f} A "
      f"-> native itself {'PASSES' if nmax['d2'] < 6.0 else 'FAILS'} d2<6.0")
if old.sum():
    print(f"  their bb RMSD to native: min {r[old].min():.2f}, median {np.median(r[old]):.2f} A; "
          f"Q median {np.median(q[old]):.3f}")

print("\nNATIVE-WINDOW FILTER (each metric inside the NMR-model range, +/- 1 sd slack)")
win = np.ones(N, bool)
for k in ["cv1", "rg", "d1", "d2", "yw"]:
    lo, hi = nmin[k] - ns[k], nmax[k] + ns[k]
    m = (z[k] >= lo) & (z[k] <= hi)
    win &= m
    print(f"  {k:5s} in [{lo:6.2f},{hi:6.2f}]  {int(m.sum()):9,d} frames  cumulative {int(win.sum()):9,d}")
if win.sum():
    print(f"  surviving frames bb RMSD: min {r[win].min():.2f} median {np.median(r[win]):.2f} A, "
          f"Q median {np.median(q[win]):.3f}")

best = np.argsort(r)[:200]
np.save(f"{OUT}/best_idx.npy", best)
print("\nTOP 15 FRAMES BY BACKBONE RMSD TO NATIVE")
print(f"{'rank':>4s} {'rmsd':>6s} {'Q':>6s} {'cv1':>6s} {'rg':>6s} {'d1':>6s} {'d2':>6s} {'Y2W9':>6s} "
      f"{'pca1':>7s} {'pca2':>7s}  source")
files = z["files"]
for rank, i in enumerate(best[:15], 1):
    f = str(files[z["src_file"][i]]).split("/")[-1]
    print(f"{rank:4d} {r[i]:6.2f} {q[i]:6.3f} {z['cv1'][i]:6.3f} {z['rg'][i]:6.2f} {z['d1'][i]:6.2f} "
          f"{z['d2'][i]:6.2f} {yw[i]:6.2f} {z['pca1'][i]:7.2f} {z['pca2'][i]:7.2f}  {f}:{z['src_frame'][i]}")

print("\nWHERE THE PMF MINIMA SIT vs NATIVE (published production values)")
pub = {"cv1": 0.4873, "rg": 6.594, "d1": 7.206, "d2": 8.377, "pca1": 1.439, "pca2": -5.289}
for k, v in pub.items():
    lo, hi = nmin[k], nmax[k]
    inside = lo <= v <= hi
    sd = (v - nm[k]) / ns[k]
    print(f"  {k:5s} PMF min {v:8.3f}   native {lo:7.2f}-{hi:<7.2f}  "
          f"{'INSIDE' if inside else 'OUTSIDE'}  ({sd:+.1f} native sd)")
