"""Analysis of the recomputed all-frames PCA.

The question CV1 failed is whether a coordinate SEPARATES the native fold from
the compact non-native topologies that share its value.  This repeats the §4
degeneracy test on the new PCA basis, using per-frame RMSD and Q that already
exist for every frame.  It is geometric and needs no Boltzmann weights.
"""
import json
import numpy as np

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
p = np.load(f"{OUT}/pca_all_frames.npz", allow_pickle=False)
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
pc1, pc2 = p["pc1"].astype(np.float64), p["pc2"].astype(np.float64)
nat = p["native_scores"].astype(np.float64)
r, q, cv, yw = z["rmsd_bb_min18"], z["q"], z["cv1"], z["yw"]
N = r.size
n1, n2 = nat[:, 0], nat[:, 1]

print(f"frames {N:,} | PC1 {pc1.min():.2f}..{pc1.max():.2f} | PC2 {pc2.min():.2f}..{pc2.max():.2f}")
print(f"1UAO (18 models): PC1 {n1.mean():+.3f} +/- {n1.std():.3f} [{n1.min():+.2f},{n1.max():+.2f}]"
      f"   PC2 {n2.mean():+.3f} +/- {n2.std():.3f} [{n2.min():+.2f},{n2.max():+.2f}]")

# ---------- where is the sampling density, and where is native? ----------
H, xe, ye = np.histogram2d(pc1, pc2, bins=60)
ix, iy = np.unravel_index(np.argmax(H), H.shape)
print(f"\ndensest sampled cell (unweighted): PC1 {0.5*(xe[ix]+xe[ix+1]):+.2f}, "
      f"PC2 {0.5*(ye[iy]+ye[iy+1]):+.2f}  ({int(H[ix,iy]):,} frames)")
print("  NOTE: this is a SAMPLING density over a biased (umbrella + GaMD) ensemble.")
print("        It is NOT a free-energy surface and no minimum is claimed from it.")

# native region = the box spanned by the 18 models, padded by 1 sd
pad1, pad2 = n1.std(), n2.std()
box = ((pc1 >= n1.min()-pad1) & (pc1 <= n1.max()+pad1) &
       (pc2 >= n2.min()-pad2) & (pc2 <= n2.max()+pad2))
print(f"\nnative PCA box PC1 [{n1.min()-pad1:+.2f},{n1.max()+pad1:+.2f}] "
      f"PC2 [{n2.min()-pad2:+.2f},{n2.max()+pad2:+.2f}]: {int(box.sum()):,} frames "
      f"({box.sum()/N:.3e} of the archive)")

# ---------- does PC1 separate the fold where CV1 could not? ----------
print("\nDEGENERACY TEST ON PC1 (the analogue of the CV1 table)")
print(f"{'PC1 bin':>16s} {'frames':>9s} {'rmsd min':>9s} {'median':>8s} {'max':>7s} "
      f"{'%<1.5A':>8s} {'medianQ':>8s}")
edges = [-15, -10, -5, 0, 5, 8, 9.5, 11, 14]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (pc1 >= lo) & (pc1 < hi)
    if m.sum() < 50:
        continue
    print(f"  [{lo:+5.1f},{hi:+5.1f}) {int(m.sum()):9,d} {r[m].min():9.2f} {np.median(r[m]):8.2f} "
          f"{r[m].max():7.2f} {100*(r[m]<1.5).mean():8.2f} {np.median(q[m]):8.3f}")

print("\nHEAD-TO-HEAD: purity of the native window, PC1 vs CV1 vs the 2D PCA box")
def purity(mask, label):
    n = int(mask.sum())
    print(f"  {label:44s} {n:8,d} frames  {100*(r[mask]<1.5).mean():6.2f}% native  "
          f"RMSD {r[mask].min():.2f}-{r[mask].max():.2f} A  medianQ {np.median(q[mask]):.3f}")
purity((cv >= 0.82) & (cv < 0.93), "CV1 in native window [0.82,0.93)")
purity((pc1 >= n1.min()-pad1) & (pc1 <= n1.max()+pad1), "PC1 in native window (1D)")
purity(box, "PC1 AND PC2 in the native box (2D)")
purity(box & (cv >= 0.82) & (cv < 0.93), "native PCA box AND native CV1 window")

# ---------- where do the genuinely native frames live in PC space? ----------
for lab, m in [("RMSD <= 1.0 A", r <= 1.0), ("Q >= 0.80", q >= 0.80)]:
    print(f"\n{lab} ({int(m.sum())} frames): PC1 {pc1[m].mean():+.2f} +/- {pc1[m].std():.2f} "
          f"[{pc1[m].min():+.2f},{pc1[m].max():+.2f}]   "
          f"PC2 {pc2[m].mean():+.2f} +/- {pc2[m].std():.2f} [{pc2[m].min():+.2f},{pc2[m].max():+.2f}]")
    inbox = box[m].mean()
    print(f"   fraction inside the 1UAO PCA box: {100*inbox:.1f}%")

# ---------- how well does each coordinate rank native structures? ----------
def auroc(score, pos):
    """P(score of a native frame > score of a non-native frame), rank-based."""
    order = np.argsort(score)
    ranks = np.empty(score.size); ranks[order] = np.arange(1, score.size + 1)
    npos, nneg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)

pos = r <= 1.5
print(f"\nRANKING POWER for 'is this frame native' (RMSD <= 1.5 A, {int(pos.sum()):,} positives)")
print(f"{'coordinate':28s} {'AUROC':>8s}")
for lab, s in [("CV1 contact fraction", cv), ("PC1 (new, all frames)", pc1),
               ("-|PC2|", -np.abs(pc2)), ("-Rg", -z["rg"]),
               ("-dist to native in PC plane",
                -np.hypot(pc1 - n1.mean(), pc2 - n2.mean())),
               ("Q (native-referenced)", q)]:
    print(f"  {lab:28s} {auroc(s, pos):8.4f}")
print("  (0.5 = no discrimination; Q is reference-based and shown as the ceiling)")

json.dump({"native_box_frames": int(box.sum()),
           "native_box_pct_native": float(100*(r[box] < 1.5).mean()) if box.sum() else None,
           "cv1_window_pct_native": float(100*(r[(cv>=0.82)&(cv<0.93)] < 1.5).mean()),
           "auroc": {"cv1": float(auroc(cv, pos)), "pc1": float(auroc(pc1, pos)),
                     "pc_dist": float(auroc(-np.hypot(pc1-n1.mean(), pc2-n2.mean()), pos)),
                     "q": float(auroc(q, pos)), "rg": float(auroc(-z["rg"], pos))},
           "native_pc1": [float(n1.mean()), float(n1.std())],
           "native_pc2": [float(n2.mean()), float(n2.std())]},
          open(f"{OUT}/pca_analysis.json", "w"), indent=2)
