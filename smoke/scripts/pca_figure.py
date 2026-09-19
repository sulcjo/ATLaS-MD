"""Figures for the recomputed all-frames PCA."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
p = np.load(f"{OUT}/pca_all_frames.npz"); z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
pc1, pc2 = p["pc1"].astype(float), p["pc2"].astype(float)
nat = p["native_scores"].astype(float); evr = p["explained_variance_ratio"]
r, q = z["rmsd_bb_min18"], z["q"]

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
B = 90
rng = [[pc1.min(), pc1.max()], [pc2.min(), pc2.max()]]

# (a) sampling density -- explicitly NOT a free energy
H, xe, ye = np.histogram2d(pc1, pc2, bins=B, range=rng)
im = ax[0].imshow(np.ma.masked_where(H.T == 0, H.T), origin="lower", aspect="auto",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="viridis", norm=LogNorm())
plt.colorbar(im, ax=ax[0], label="frames per cell")
ax[0].set_title("(a) Sampling density, all 611,536 frames\n(biased ensemble — NOT a free-energy surface)",
                fontsize=10)

# (b) native fraction per cell
num, _, _ = np.histogram2d(pc1[r <= 1.5], pc2[r <= 1.5], bins=B, range=rng)
frac = np.where(H >= 20, num / np.maximum(H, 1), np.nan)
im = ax[1].imshow(np.ma.masked_invalid(frac.T * 100), origin="lower", aspect="auto",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="magma", vmin=0, vmax=60)
plt.colorbar(im, ax=ax[1], label="% of frames with backbone RMSD $\\leq$ 1.5 Å")
ax[1].set_title("(b) Where the native fold actually is\n(cells with $\\geq$20 frames)", fontsize=10)

# (c) the native-like frames themselves
ax[2].hexbin(pc1, pc2, gridsize=80, cmap="Greys", bins="log", mincnt=1)
ax[2].scatter(pc1[r <= 1.5], pc2[r <= 1.5], s=3, c="#2a78d6", alpha=0.35,
              label=f"RMSD $\\leq$ 1.5 Å (n={int((r<=1.5).sum()):,})")
ax[2].scatter(pc1[q >= 0.80], pc2[q >= 0.80], s=26, c="#e04b2a", edgecolor="k", linewidth=0.4,
              label=f"Q $\\geq$ 0.80 (n={int((q>=0.80).sum())})")
ax[2].set_title("(c) Native-like frames in the new PCA plane", fontsize=10)
ax[2].legend(loc="lower left", fontsize=8, framealpha=0.9)

for a in ax:
    a.scatter(nat[:, 0], nat[:, 1], marker="*", s=250, c="#ffd21e", edgecolor="k",
              linewidth=0.8, zorder=6, label="1UAO (18 NMR models)")
    a.set_xlabel(f"PC1 (Å)   [{evr[0]*100:.1f}% of variance]")
    a.set_ylabel(f"PC2 (Å)   [{evr[1]*100:.1f}%]")
ax[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
fig.suptitle("chignolin_7 PCA recomputed on ALL 611,536 frames "
             "(production surface used 103,664 = 2.02%)", fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(f"{OUT}/pca_all_frames.png", dpi=155, bbox_inches="tight")
print(f"wrote {OUT}/pca_all_frames.png")

# discrimination figure: native fraction vs PC1 and vs CV1
fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.2))
for a, (x, lab, rngx) in zip(ax2, [(pc1, "PC1 (Å), recomputed on all frames", (-15, 13.2)),
                                   (z["cv1"], "CV1 contact fraction (the biasing CV)", (0, 1))]):
    e = np.linspace(*rngx, 45)
    idx = np.digitize(x, e) - 1
    tot = np.bincount(idx.clip(0, len(e) - 2), minlength=len(e) - 1)
    hit = np.bincount(idx[r <= 1.5].clip(0, len(e) - 2), minlength=len(e) - 1)
    good = tot >= 50
    c = 0.5 * (e[:-1] + e[1:])
    a.bar(c[good], 100 * hit[good] / tot[good], width=(e[1] - e[0]) * 0.9, color="#2a78d6")
    a.set_xlabel(lab); a.set_ylabel("% of frames that are native (RMSD $\\leq$ 1.5 Å)")
    a.set_ylim(0, 60)
ax2[0].axvspan(nat[:, 0].min(), nat[:, 0].max(), color="#ffd21e", alpha=0.45, label="1UAO")
ax2[1].axvspan(0.82, 0.93, color="#ffd21e", alpha=0.45, label="1UAO")
for a in ax2: a.legend(fontsize=8)
fig2.suptitle("Which coordinate separates the fold?  PC1 AUROC 0.978 vs CV1 0.859", fontsize=11)
fig2.tight_layout()
fig2.savefig(f"{OUT}/pca_vs_cv1_discrimination.png", dpi=155, bbox_inches="tight")
print(f"wrote {OUT}/pca_vs_cv1_discrimination.png")
