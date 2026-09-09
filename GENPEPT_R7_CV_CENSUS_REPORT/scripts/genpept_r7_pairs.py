from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
ROOT = BASE / "RUNS" / "chignolin_genpept_r7"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "ORTHOGONAL_PAIRS.md"
OUT_JSON = REPORT / "genpept_r7_orthogonal_pairs.json"
OUT_TSVG = REPORT / "genpept_r7_orthogonal_pairs.csv"

CV1_CANDS = ["c_bb_s4_r010_b3", "c_ca_s4_r010_b3", "c_heavy_s4_r012_b3", "c_sch_s4_r012_b3"]
CV2_CANDS = ["resPC1_refit", "restorpca_pc1", "torpca_pc2", "hairpin_closure_ratio", "turn3537_mean_ca",
             "pack_y2", "burial_d3", "d_y2_w9_ca", "acylindricity", "shape_anisotropy", "chirality_ca",
             "e2e_nc_ca", "sasa_hydrophobic_frac", "contact_order_ca8", "strand_cross_far_r012_b3"]
NEGATIVE_CONTROLS = {"contact_order_ca8", "strand_cross_far_r012_b3", "e2e_nc_ca", "d_y2_w9_ca"}


def residual_pc1(Xc, cv1):
    A = np.column_stack([np.ones(len(Xc)), cv1])
    beta, *_ = np.linalg.lstsq(A, Xc, rcond=None)
    Xr = Xc - A @ beta
    _, S, Vt = np.linalg.svd(Xr, full_matrices=False)
    return Xr @ Vt[0]


def hist_occ(x, y, nx=16, ny=8):
    xb = np.quantile(x, np.linspace(0.001, 0.999, nx + 1))
    yb = np.quantile(y, np.linspace(0.001, 0.999, ny + 1))
    H, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    occupied = (H > 0)
    return occupied.mean(), int(H[occupied].min()) if occupied.any() else 0, H


def cond_iqr_frac(x, y, nb=16):
    bins = np.quantile(x, np.linspace(0, 1, nb + 1))
    iqr_global = np.percentile(y, 75) - np.percentile(y, 25)
    vals = []
    for i in range(nb):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() >= 20:
            vals.append(np.percentile(y[m], 75) - np.percentile(y[m], 25))
    vals = np.array(vals)
    if len(vals) == 0 or iqr_global <= 0:
        return 0.0, 0.0
    return float(np.median(vals / iqr_global)), float((vals >= 0.5 * iqr_global).mean())


def entropy_norm(v, n_bins=20):
    v = v[np.isfinite(v)]
    h, _ = np.histogram(v, bins=n_bins, range=(v.min(), v.max()))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(n_bins))


def main():
    census = pd.read_csv(REPORT / "genpept_r7_cv_census_values.csv").set_index("seed")
    files = sorted((ROOT / "final_implicit_survivor_seeds").glob("*.pdb"))
    census = census.loc[[f.stem for f in files]]
    n = len(census)
    traj = md.load(files)
    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    cols = []
    for arr in (phis, psis):
        for k in range(arr.shape[1]):
            cols.append(np.sin(arr[:, k]))
            cols.append(np.cos(arr[:, k]))
    X = np.stack(cols, axis=1)
    Xc = X - X.mean(axis=0)
    print(f"bank {n}, torsion features {X.shape}")

    stats = pd.read_csv(REPORT / "genpept_r7_cv_census_stats.csv").set_index("cv")

    rows = []
    for cv1n in CV1_CANDS:
        x = census[cv1n].to_numpy()
        res_pc1 = residual_pc1(Xc, x)
        for cv2n in CV2_CANDS:
            if cv2n == "resPC1_refit":
                y = res_pc1
                label2 = "resPC1(refit vs this CV1)"
            else:
                y = census[cv2n].to_numpy()
                label2 = cv2n
            pear = float(np.corrcoef(x, y)[0, 1])
            spear = float(spearmanr(x, y).correlation)
            occ, minbin, _H = hist_occ(x, y)
            medf, f50 = cond_iqr_frac(x, y)
            e1 = stats.loc[cv1n, "entropy"] if cv1n in stats.index else entropy_norm(x)
            e2 = stats.loc[cv2n, "entropy"] if cv2n in stats.index else entropy_norm(y)
            score = min(e1, e2) * (1 - abs(spear)) * occ * f50
            rows.append(dict(cv1=cv1n, cv2=cv2n, label2=label2, pear=pear, spear=spear,
                             occ=occ, min_bin=minbin, cond_iqr_med=medf, frac_bins_iqr50=f50,
                             ent1=e1, ent2=e2, score=score,
                             control="NEGATIVE" if cv2n in NEGATIVE_CONTROLS else ""))
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_TSVG, index=False)

    print(df.head(16)[["cv1", "cv2", "spear", "occ", "min_bin", "frac_bins_iqr50", "score"]].to_string(index=False))

    top = df[df.control == ""].head(6)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (_, r) in zip(axes.ravel(), top.iterrows()):
        x = census[r.cv1].to_numpy()
        if r.cv2 == "resPC1_refit":
            y = residual_pc1(Xc, x)
        else:
            y = census[r.cv2].to_numpy()
        ax.hexbin(x, y, gridsize=24, cmap="viridis", mincnt=1)
        ax.set_title(f"{r.cv1}\n x {r.cv2} (resPC1 re-fit)\n|r_s|={abs(r.spear):.2f}, occ={r.occ*100:.0f}%, min N={r.min_bin}"[:80],
                     fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("top alternative orthogonal pairs (bank r7, hex counts)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_orthogonal_pairs_top.png", dpi=150)
    plt.close(fig)

    lines = ["# Alternative orthogonal CV1×CV2 pairs — scored on the r7 bank\n"]
    lines.append(f"n={n} seeds. Per pair: Pearson & Spearman r, 16×8 quantile-grid occupancy, min bin count,")
    lines.append("median conditional CV2 IQR per CV1 bin relative to global, fraction of bins where conditional")
    lines.append("IQR ≥ 50% of global, per-axis 20-bin entropy. `resPC1_refit` = residual torsion PC1 re-fit by")
    lines.append("linear residualization of bank torsion features against that CV1. Negative controls expected to fail.\n")
    lines.append("Composite score = min(ent1, ent2) × (1−|r_s|) × occupancy × frac_bins_IQR50.\n")
    lines.append("| rank | CV1 | CV2 | r_p | r_s | occ% | min N | cond IQR med | f≥50% | ent1 | ent2 | score |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in df.iterrows():
        lines.append(f"| {i+1} | {r.cv1} | {r.cv2} | {r.pear:+.2f} | {r.spear:+.2f} | {r.occ*100:.0f} | {r.min_bin} | "
                     f"{r.cond_iqr_med:.2f} | {r.frac_bins_iqr50:.2f} | {r.ent1:.3f} | {r.ent2:.3f} | {r.score:.3f} |")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    json.dump({"score_formula": "min(ent1,ent2) * (1-|spearman|) * occupancy * frac_bins_iqr50",
               "rows": rows}, open(OUT_JSON, "w"), indent=1)
    print("wrote", OUT_MD, OUT_TSVG, OUT_JSON, FIG / "cv_orthogonal_pairs_top.png")


if __name__ == "__main__":
    main()
