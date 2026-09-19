"""Blind test: can the fold be IDENTIFIED with no reference at all?

Everything used to cluster and rank is reference-free: PC1/PC2 (fitted on the
ensemble), CV1 (the run's own contact fraction), Rg, and time/replica structure.
RMSD and Q to 1UAO are used ONLY at the end, to reveal where the native frames
landed -- never to build or rank the clusters.
"""
import numpy as np, json

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
p = np.load(f"{OUT}/pca_all_frames.npz"); z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
pc1, pc2 = p["pc1"].astype(float), p["pc2"].astype(float)
cv, rg = z["cv1"].astype(float), z["rg"].astype(float)
sf, fr = z["src_file"], z["src_frame"]
r, q = z["rmsd_bb_min18"], z["q"]                      # revealed only at the end
N = r.size

# ---- cluster on reference-free coordinates (coarse grid in PC1/PC2 x CV1) ----
def binid(x, lo, hi, n):
    return np.clip(((x - lo) / (hi - lo) * n).astype(int), 0, n - 1)
b1 = binid(pc1, pc1.min(), pc1.max(), 24)
b2 = binid(pc2, pc2.min(), pc2.max(), 24)
b3 = binid(cv, 0.0, 1.0, 10)
cid = (b1 * 24 + b2) * 10 + b3
uniq, inv, cnt = np.unique(cid, return_inverse=True, return_counts=True)
keep = cnt >= 500                                       # basins with enough frames to judge
print(f"{uniq.size} occupied cells; {keep.sum()} with >= 500 frames "
      f"covering {cnt[keep].sum()/N:.3f} of frames")

# ---- reference-free scores per basin ----
order = np.lexsort((fr, sf))                            # frames in trajectory order
cid_ord = cid[order]
same = cid_ord[1:] == cid_ord[:-1]
contig = np.r_[True, ~same]
ep_id = np.cumsum(contig) - 1
ep_cid = cid_ord[contig]
ep_len = np.bincount(ep_id)

rows = []
for k, c in enumerate(uniq):
    if not keep[k]:
        continue
    m = cid == c
    n = int(m.sum())
    eps = ep_len[ep_cid == c]
    # internal structural tightness: spread of the basin in its own PC plane
    tight = 1.0 / (1e-9 + np.sqrt(pc1[m].var() + pc2[m].var()))
    persistence = float(np.median(eps)) if eps.size else 0.0
    reentry = int(eps.size)                             # independent visits
    nrep = len(set(sf[m].tolist()))                     # spread across trajectories
    compact = 1.0 / (1e-9 + rg[m].mean())
    rows.append(dict(cell=int(c), n=n, tight=float(tight), persist=persistence,
                     reentry=reentry, nfiles=nrep, compact=float(compact),
                     pc1=float(pc1[m].mean()), pc2=float(pc2[m].mean()),
                     cv=float(cv[m].mean()), rg=float(rg[m].mean()),
                     # revealed afterwards, never used for ranking:
                     pct_native=float(100*(r[m] <= 1.5).mean()), maxQ=float(q[m].max())))

def zsc(v):
    v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-12)
T = zsc([x["tight"] for x in rows]); P = zsc([np.log1p(x["persist"]) for x in rows])
R = zsc([np.log1p(x["reentry"]) for x in rows]); C = zsc([x["compact"] for x in rows])
score = T + P + R + C
for i, x in enumerate(rows):
    x["score"] = float(score[i])
rows.sort(key=lambda x: -x["score"])

print(f"\nBASINS RANKED BY A REFERENCE-FREE SCORE (tightness + persistence + re-entry + compactness)")
print(f"{'rank':>4s} {'n':>8s} {'PC1':>7s} {'PC2':>7s} {'CV1':>6s} {'Rg':>6s} {'score':>7s} "
      f"| {'% native':>9s} {'maxQ':>6s}   <- revealed after ranking")
for i, x in enumerate(rows[:12], 1):
    print(f"{i:4d} {x['n']:8,d} {x['pc1']:7.2f} {x['pc2']:7.2f} {x['cv']:6.3f} {x['rg']:6.2f} "
          f"{x['score']:7.2f} | {x['pct_native']:8.2f}% {x['maxQ']:6.3f}")
best = max(range(len(rows)), key=lambda i: rows[i]["pct_native"])
print(f"\nThe most-native basin ({rows[best]['pct_native']:.1f}% native, maxQ {rows[best]['maxQ']:.3f}) "
      f"ranks #{best+1} of {len(rows)} by the reference-free score.")
top = [i for i, x in enumerate(rows) if x["pct_native"] > 20]
print(f"Basins that are >20% native: ranks {[i+1 for i in top]} of {len(rows)}")
json.dump({"n_basins": len(rows), "rank_of_most_native": int(best + 1),
           "most_native_pct": rows[best]["pct_native"],
           "top12": rows[:12]}, open(f"{OUT}/blind_identification.json", "w"), indent=2)
