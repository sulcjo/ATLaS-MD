from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr
from scipy.signal import find_peaks

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
ROOT = BASE / "RUNS" / "chignolin_genpept_r7"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OBS = BASE / "RUNS" / "chignolin_6" / "adaptive_production" / "epoch_000" / "tica_obs"
OUT_MD = REPORT / "PLAIN_PC2_CV2.md"
OUT_JSON = REPORT / "genpept_r7_plain_pc2.json"

RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def circmean_deg(a):
    return float(np.degrees(np.angle(np.exp(1j * a).mean())))


def cos_content(x):
    x = np.asarray(x, float)
    y = x - x.mean()
    n = len(y)
    t = (np.arange(n) + 0.5) / n * math.pi
    c = np.cos(t)
    num = (c * y).sum()
    den = (y * y).sum()
    return float(2.0 * num * num / (n * den)) if den > 0 else float("nan")


def main():
    files = sorted((ROOT / "final_implicit_survivor_seeds").glob("*.pdb"))
    names = [f.stem for f in files]
    census = pd.read_csv(REPORT / "genpept_r7_cv_census_values.csv").set_index("seed").loc[names]
    n = len(names)
    traj = md.load(files)
    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    psi_res = [int(traj.topology.atom(int(psi_idx[k][1])).residue.index) for k in range(psis.shape[1])]
    phi_res = [int(traj.topology.atom(int(phi_idx[k][1])).residue.index) for k in range(phis.shape[1])]

    cols = []
    labels = []
    for k in range(phis.shape[1]):
        cols.append(np.sin(phis[:, k])); cols.append(np.cos(phis[:, k]))
        labels.extend([f"sin phi {RES3[phi_res[k]]}", f"cos phi {RES3[phi_res[k]]}"])
    for k in range(psis.shape[1]):
        cols.append(np.sin(psis[:, k])); cols.append(np.cos(psis[:, k]))
        labels.extend([f"sin psi {RES3[psi_res[k]]}", f"cos psi {RES3[psi_res[k]]}"])
    X = np.stack(cols, axis=1)
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc2_vec = Vt[1]
    pc2 = census["torpca_pc2"].to_numpy()

    top = np.argsort(-np.abs(pc2_vec))[:10]
    print("plain PC2 top loadings:")
    for i in top:
        print(f"  {labels[i]:<14} {pc2_vec[i]:+.3f}")

    q10, q90 = np.quantile(pc2, [0.1, 0.9])
    lo = np.where(pc2 <= q10)[0]
    hi = np.where(pc2 >= q90)[0]
    rot = []
    for kind, arr, rr in (("phi", phis, phi_res), ("psi", psis, psi_res)):
        for j, r in enumerate(rr):
            a_lo = circmean_deg(arr[lo, j])
            a_hi = circmean_deg(arr[hi, j])
            d = math.degrees(np.angle(np.exp(1j * np.radians(a_hi - a_lo))))
            rot.append((kind, RES3[r], a_lo, a_hi, d))
    big = sorted(rot, key=lambda t: -abs(t[4]))[:8]
    for k_, r_, l_, h_, d_ in big:
        print(f"  Δ{k_} {r_}: {l_:+.0f} -> {h_:+.0f} ({d_:+.0f})")

    cv1 = census["c_bb_s4_r010_b3"].to_numpy()
    print(f"mean CV1 in deciles: {cv1[lo].mean():.3f} (lo) vs {cv1[hi].mean():.3f} (hi)")

    obs_files = sorted(glob.glob(str(OBS / "dihedral_obs_*.npz")))
    c1s = []
    for of in obs_files:
        z = np.load(of)
        xx = z["features"] - z["features"].mean(axis=0)
        c1s.append(cos_content(xx @ pc2_vec))
    c1s = np.array(c1s)
    print(f"plain PC2 C1 on 32 traces: median {np.median(c1s):.3f}, frac>=0.5 {(c1s>=0.5).mean():.2f}")

    xs = np.linspace(pc2.min(), pc2.max(), 400)
    kde = gaussian_kde(pc2, bw_method=0.06)
    d = kde(xs)
    peaks, _ = find_peaks(d, prominence=0.04 * d.max())
    print(f"plain PC2 KDE modes (bw .06): {len(peaks)} at {np.round(xs[peaks], 2)}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    order = np.argsort(-np.abs(pc2_vec))[:16][::-1]
    ax.barh(range(len(order)), pc2_vec[order], color="#2c7fb8")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[i] for i in order], fontsize=7)
    ax.set_title("plain PC2 top-16 loadings", fontsize=9)
    ax = axes[1]
    ax.hexbin(cv1, pc2, gridsize=26, cmap="viridis", mincnt=1)
    ax.set_xlabel("c_bb_s4_r010_b3")
    ax.set_ylabel("plain torsion PC2")
    ax.set_title(f"joint map (spearman {spearmanr(cv1, pc2).correlation:+.2f})", fontsize=9)
    ax = axes[2]
    ax.hist(c1s, bins=12, color="#756bb1")
    ax.axvline(0.5, color="red", ls="--", lw=1)
    ax.set_xlabel("Hess C1 of plain PC2 projection")
    ax.set_title(f"diffusion content, 32 epoch-0 traces\nmedian {np.median(c1s):.3f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "cv2_plain_pc2.png", dpi=150)
    plt.close(fig)

    doc = {"top_loadings": [[labels[i], float(pc2_vec[i])] for i in top],
           "deciles": {"q10": float(q10), "q90": float(q90),
                       "cv1_lo": float(cv1[lo].mean()), "cv1_hi": float(cv1[hi].mean())},
           "rotations": [[k_, r_, float(l_), float(h_), float(d_)] for k_, r_, l_, h_, d_ in rot],
           "c1_traces": c1s.tolist(),
           "kde_modes": xs[peaks].tolist(),
           "spearman_with_cv1": float(spearmanr(cv1, pc2).correlation)}
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# Plain torsion PC2 as CV2 scalar — verification battery\n"]
    lines.append(f"top loadings: " + "; ".join(f"{labels[i]} {pc2_vec[i]:+.2f}" for i in top[:8]))
    lines.append(f"\ndeciles [{q10:.3f}, {q90:.3f}]; mean CV1 in deciles {cv1[lo].mean():.3f} vs {cv1[hi].mean():.3f} "
                 f"(compaction matched)")
    lines.append("\nlargest per-residue circular-mean rotations (low -> high decile):")
    for k_, r_, l_, h_, d_ in big:
        lines.append(f"- {k_} {r_}: {l_:+.0f} -> {h_:+.0f} ({d_:+.0f} deg)")
    lines.append(f"\ndiffusion content on 32 chignolin_6 epoch-0 traces: median C1 {np.median(c1s):.3f}, "
                 f"frac >= 0.5: {(c1s >= 0.5).mean():.2f}")
    lines.append(f"KDE modes (bw .06): {[round(float(m), 2) for m in xs[peaks]]}")
    lines.append(f"spearman with CV1 (bb r0 10 b3): {spearmanr(cv1, pc2).correlation:+.3f}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON, FIG / "cv2_plain_pc2.png")


if __name__ == "__main__":
    main()
