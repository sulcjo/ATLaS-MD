import glob
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(os.environ.get("REPO_ROOT", "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler"))
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
SEED_DIR = BASE / "RUNS" / "chignolin_genpept_r7" / "final_implicit_survivor_seeds"
OBS_DIR = BASE / "RUNS" / "chignolin_6" / "adaptive_production" / "epoch_000" / "tica_obs"
VALUES = REPORT / "genpept_r7_cv_census_values.csv"
OUT_JSON = REPORT / "bank_torsion_pca_cv2.json"
OUT_MD = REPORT / "CV2_DEEPDIVE.md"
FIG_LOAD = REPORT / "figures" / "cv2_loadings.png"
FIG_DIFF = REPORT / "figures" / "cv2_diffusion_content.png"
FIG_2D = REPORT / "figures" / "cv2_joint_maps.png"

resnames_full = ["GLY", "TYR", "ASP", "PRO", "GLU", "THR", "GLY", "THR", "TRP", "GLY"]
RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def interleaved_feats(alltraj):
    phi_idx, phis = md.compute_phi(alltraj)
    psi_idx, psis = md.compute_psi(alltraj)
    cols = []
    labels = []
    for k in range(phis.shape[1]):
        cols.append(np.sin(phis[:, k]))
        cols.append(np.cos(phis[:, k]))
        r = int(alltraj.topology.atom(int(phi_idx[k][1])).residue.index)
        labels.extend([f"sin phi {RES3[r]}", f"cos phi {RES3[r]}"])
    for k in range(psis.shape[1]):
        cols.append(np.sin(psis[:, k]))
        cols.append(np.cos(psis[:, k]))
        r = int(alltraj.topology.atom(int(psi_idx[k][1])).residue.index)
        labels.extend([f"sin psi {RES3[r]}", f"cos psi {RES3[r]}"])
    return np.stack(cols, axis=1), labels


def pca_fit(X, cv1=None):
    Xw = X.copy()
    slope = None
    if cv1 is not None:
        A = np.column_stack([np.ones(len(X)), cv1])
        beta, *_ = np.linalg.lstsq(A, Xw, rcond=None)
        slope = beta[1]
        Xw = Xw - A @ beta
    mean = Xw.mean(axis=0)
    Xc = Xw - mean
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S * S) / max(1, len(X) - 1)
    evr = var / var.sum()
    return {"mean": mean, "slope": slope, "vt": Vt, "evr": evr, "raw_mean": X.mean(axis=0)}


def proj(fit, X, k):
    v = fit["vt"][k].copy()
    if fit["slope"] is not None:
        s = fit["slope"]
        sn = float(s @ s)
        if sn > 1e-12:
            v = v - s * (float(v @ s) / sn)
            v = v / np.linalg.norm(v)
    return (X - fit["mean"]) @ v, v / np.linalg.norm(v)


def cos_content(x):
    x = np.asarray(x, float)
    y = x - x.mean()
    n = len(y)
    if n < 4 or np.allclose(y, 0):
        return float("nan")
    t = (np.arange(n) + 0.5) / n * math.pi
    c = np.cos(t)
    num = (c * y).sum()
    den = (y * y).sum()
    return float(2.0 * num * num / (n * den))


def main():
    files = sorted(glob.glob(str(SEED_DIR / "*.pdb")))
    bank = md.load(files)
    Xb, labels = interleaved_feats(bank)
    n, d = Xb.shape
    print(f"bank features {Xb.shape}")

    df = pd.read_csv(VALUES)
    cv1_bb = df["c_bb_s4_r010_b3"].to_numpy()
    cv1_hv = df["c_heavy_s4_r012_b3"].to_numpy()

    fit_plain = pca_fit(Xb)
    fit_res_bb = pca_fit(Xb, cv1_bb)
    fit_res_hv = pca_fit(Xb, cv1_hv)

    _, v_plain1 = proj(fit_plain, Xb, 0)
    _, v_res1 = proj(fit_res_bb, Xb, 0)
    cos_axes = float(abs(np.dot(v_plain1, v_res1)))
    sub = np.linalg.svd(fit_plain["vt"][:5] @ fit_res_bb["vt"][:5].T, compute_uv=False)
    sub_overlap = float((sub ** 2).mean())

    print(f"plain evr[:6] {np.round(fit_plain['evr'][:6], 3).tolist()}")
    print(f"res-bb evr[:6] {np.round(fit_res_bb['evr'][:6], 3).tolist()}")
    print(f"|cos(PC1, resPC1_bb)| = {cos_axes:.3f}   top5 subspace overlap = {sub_overlap:.3f}")

    banks_only = {"weights": v_res1.tolist(), "mean": fit_res_bb["mean"].tolist(),
                  "feature_labels": labels, "feature_order": "interleaved sin_phi/cos_phi..., sin_psi/cos_psi...",
                  "axis": "residual v_res1 not orthogonalized-in-JSON; subtract slope then normalize before use"}
    OUT_JSON.write_text(json.dumps({
        "interleaved_order": labels,
        "plain_pc1": {"weights": v_plain1.tolist(), "mean": fit_plain["mean"].tolist()},
        "res_pc1_vs_c_bb_s4_r010_b3": {
            "weights_pc": fit_res_bb["vt"][0].tolist(),
            "slope_cv1": fit_res_bb["slope"].tolist(),
            "unit_axis": v_res1.tolist(),
            "mean": fit_res_bb["mean"].tolist(),
        },
        "plain_pc2": {"weights": fit_plain["vt"][1].tolist(), "mean": fit_plain["mean"].tolist()},
        "evr_plain": fit_plain["evr"][:10].tolist(),
        "evr_res_bb": fit_res_bb["evr"][:10].tolist(),
        "n_bank": n,
        "source": "RUNS/chignolin_genpept_r7/final_implicit_survivor_seeds",
    }, indent=1))

    top_load = np.argsort(-np.abs(v_res1))[:12]
    print("top res-PC1 loadings:")
    for i in top_load:
        print(f"   {labels[i]:<14} {v_res1[i]:+.3f}")

    obs_files = sorted(glob.glob(str(OBS_DIR / "dihedral_obs_*.npz")))
    traces_c1 = {"plain_pc1": [], "res_pc1": [], "own_pc1": []}
    halves_c1 = {"plain_pc1": [], "res_pc1": [], "own_pc1": []}
    corr_stored = []
    pooled = {"cv1": [], "cv2proj": [], "stored_cv2": []}
    for of in obs_files:
        z = np.load(of)
        X = z["features"]
        x_plain, _ = proj(fit_plain, X, 0)
        x_res, _ = proj(fit_res_bb, X, 0)
        Xc = X - X.mean(axis=0)
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        x_own = Xc @ Vt[0]
        for tag, arr in (("plain_pc1", x_plain), ("res_pc1", x_res), ("own_pc1", x_own)):
            traces_c1[tag].append(cos_content(arr))
            h = len(arr) // 2
            halves_c1[tag].append(cos_content(arr[:h]))
            halves_c1[tag].append(cos_content(arr[h:]))
        r_ = np.corrcoef(x_res, z["secondary_cv"])[0, 1]
        corr_stored.append(r_)
        pooled["cv1"].append(z["primary_cv"])
        pooled["cv2proj"].append(x_res)
        pooled["stored_cv2"].append(z["secondary_cv"])

    for tag in traces_c1:
        arr = np.array(traces_c1[tag])
        print(f"C1 {tag}: median {np.nanmedian(arr):.3f}  frac>=0.5: {(arr >= 0.5).mean():.3f}")
    arr_h = np.array(halves_c1["res_pc1"])
    print(f"C1 res_pc1 half-traces: median {np.nanmedian(arr_h):.3f}  frac>=0.5: {(arr_h >= 0.5).mean():.3f}")
    corr_stored = np.array(corr_stored)
    print(f"corr(bank resPC1 proj, stored secondary_cv) per trace: median {np.median(corr_stored):.3f} "
          f"[{corr_stored.min():.3f}, {corr_stored.max():.3f}]")

    cv1_p = np.concatenate(pooled["cv1"])
    cv2_p = np.concatenate(pooled["cv2proj"])
    s_cv2 = np.concatenate(pooled["stored_cv2"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = np.argsort(-np.abs(v_res1))[:18][::-1]
    axes[0].barh(range(len(order)), v_res1[order], color="steelblue")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([labels[i] for i in order], fontsize=8)
    axes[0].set_title("bank residual PC1 ax1 loadings (top 18)", fontsize=10)
    evrs = fit_res_bb["evr"][:12]
    axes[1].bar(range(len(evrs)), evrs, color="darkorange")
    axes[1].set_xticks(range(len(evrs)))
    axes[1].set_xticklabels([f"PC{i+1}" for i in range(len(evrs))], fontsize=8)
    axes[1].set_title("explained variance ratio (residualized on c_bb_s4_r010_b3)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_LOAD, dpi=150)
    plt.close(fig)

    tags = list(traces_c1)
    xpos = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [traces_c1[t] for t in tags]
    bp = ax.boxplot(data, positions=xpos, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor="lightsteelblue"))
    ax.axhline(0.5, color="red", ls="--", lw=1, label="C1=0.5 (Hess criterion)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["bank plain PC1", "bank res PC1", "trace-own PC1"], fontsize=9)
    ax.set_ylabel("cosine content C1")
    ax.set_title("diffusion content of projected axes, 32 chignolin_6 epoch-0 traces", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIFF, dpi=150)
    plt.close(fig)

    def joint(x, y, nx=16, ny=8):
        xb = np.linspace(np.quantile(x, 0.001), np.quantile(x, 0.999), nx + 1)
        yb = np.linspace(np.quantile(y, 0.001), np.quantile(y, 0.999), ny + 1)
        H, _, _ = np.histogram2d(x, y, bins=[xb, yb])
        occ = (H > 0).mean()
        iqr_per_bin = []
        for i in range(nx):
            m = (x >= xb[i]) & (x < xb[i + 1])
            if m.sum() >= 20:
                iqr_per_bin.append(np.percentile(y[m], 75) - np.percentile(y[m], 25))
        return H, xb, yb, occ, float(np.median(iqr_per_bin)), np.array(iqr_per_bin)

    bank_cv2_res, _ = proj(fit_res_bb, Xb, 0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    maps = [
        ("bank: c_bb_s4_r010_b3", cv1_bb, "bank resPC1", bank_cv2_res),
        ("bank: c_heavy_s4_r012_b3", cv1_hv, "bank resPC1", bank_cv2_res),
        ("chignolin_6 epoch_000: stored primary_cv", cv1_p, "bank resPC1 proj", cv2_p),
        ("chignolin_6 epoch_000: stored primary_cv", cv1_p, "stored secondary_cv", s_cv2),
    ]
    stats_2d = []
    for ax, (xt, xv, yt, yv) in zip(axes.ravel(), maps):
        H, xb, yb, occ, med_iqr, _ = joint(np.asarray(xv), np.asarray(yv))
        ax.pcolormesh(xb, yb, np.log10(H.T + 1), cmap="viridis")
        ax.set_xlabel(xt, fontsize=8)
        ax.set_ylabel(yt, fontsize=8)
        ax.set_title(f"16x8 occupancy {occ*100:.1f}%, per-CV1-bin CV2 IQR med {med_iqr:.3f}", fontsize=9)
        stats_2d.append((xt, yt, occ, med_iqr))
    fig.suptitle("CV1 x CV2 joint maps (log10 count)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(FIG_2D, dpi=150)
    plt.close(fig)

    def frac_ge05(a):
        a = np.asarray(a)
        return f"{(a >= 0.5).mean()*100:.1f}% ({int((a >= 0.5).sum())}/{len(a)})"

    md_lines = []
    md_lines.append("# CV2 deep dive — bank PCA axes and dynamics content\n")
    md_lines.append("Bank: 1,970 x 36 interleaved sin/cos features (canonical `gareus.tica` ordering).\n")
    md_lines.append(f"- plain PCA EVR[1-6]: {[round(float(x),3) for x in fit_plain['evr'][:6]]}")
    md_lines.append(f"- residual-on-c_bb_s4_r010_b3 EVR[1-6]: {[round(float(x),3) for x in fit_res_bb['evr'][:6]]}")
    md_lines.append(f"- |cos(plain PC1, res PC1)| = {cos_axes:.3f}; top-5 subspace overlap (mean cos^2 of principal angles) = {sub_overlap:.3f}")
    md_lines.append("- top residual-PC1 loadings: " + ", ".join(f"{labels[i]} {v_res1[i]:+.3f}" for i in top_load[:10]) + "\n")
    md_lines.append("## Diffusion content (Hess cosine C1) on 32 chignolin_6 epoch-0 traces (4,768 frames each)\n")
    for tag in tags:
        arr = np.array(traces_c1[tag])
        md_lines.append(f"- {tag}: median C1 {np.nanmedian(arr):.3f}, frac>=0.5 {frac_ge05(arr)}")
    md_lines.append(f"- res_pc1 half-traces: median {np.nanmedian(np.array(halves_c1['res_pc1'])):.3f}, "
                    f"frac>=0.5 {frac_ge05(np.array(halves_c1['res_pc1']))}\n")
    md_lines.append(f"Per-trace corr(bank resPC1 projection, stored secondary_cv): median {np.median(corr_stored):.3f} "
                    f"range [{corr_stored.min():.3f}, {corr_stored.max():.3f}]\n")
    md_lines.append("## CV1 x CV2 joint maps (16 x 8 over q0.1-q99.9)\n")
    for xt, yt, occ, miqr in stats_2d:
        md_lines.append(f"- {xt} x {yt}: occupancy {occ*100:.1f}%, per-CV1-bin CV2 IQR median {miqr:.3f}")
    md_lines.append("")
    OUT_MD.write_text("\n".join(md_lines))
    print("wrote", OUT_MD, FIG_LOAD, FIG_DIFF, FIG_2D, OUT_JSON, sep="\n  ")


if __name__ == "__main__":
    main()
