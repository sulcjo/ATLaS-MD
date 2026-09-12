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
from scipy.stats import ks_2samp, spearmanr

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R7 = BASE / "RUNS" / "chignolin_genpept_r7"
R2 = Path("/tmp/opencode/chignolin_genpept_rep2")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "REP2_COMPARISON.md"
OUT_JSON = REPORT / "genpept_r7_vs_rep2.json"

DEFS = {
    "bb_r010_b3": ("backbone-heavy", 4, 10.0, 3.0),
    "bb_r010_b6": ("backbone-heavy", 4, 10.0, 6.0),
    "ca_r010_b3": ("ca", 4, 10.0, 3.0),
    "heavy_r012_b3": ("heavy", 4, 12.0, 3.0),
    "heavy_r012_b1.5": ("heavy", 4, 12.0, 1.5),
    "legacy_heavy_r04.5_b6": ("heavy", 4, 4.5, 6.0),
}
RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def sel_pairs(top, selection, sep):
    BB = {"N", "CA", "C", "O"}
    idxs, res_of = [], []
    for a in top.atoms:
        keep = ((selection == "heavy" and (a.element is not None and a.element.symbol != "H"))
                or (selection == "backbone-heavy" and a.name in BB)
                or (selection == "ca" and a.name == "CA"))
        if keep:
            idxs.append(a.index)
        res_of.append(a.residue.index)
    idxs = np.array(idxs)
    res_of = np.array(res_of)
    ii, jj = np.triu_indices(len(idxs), k=1)
    keep = np.abs(res_of[idxs[ii]] - res_of[idxs[jj]]) >= sep
    return idxs[ii][keep], idxs[jj][keep]


def contact_cv(traj, ii, jj, r0, beta):
    d = np.linalg.norm(traj.xyz[:, ii] - traj.xyz[:, jj], axis=2) * 10.0
    return (0.5 * (1.0 - np.tanh(0.5 * beta * (d - r0)))).mean(axis=1)


def entropy_norm(v, n_bins=20):
    v = v[np.isfinite(v)]
    h, _ = np.histogram(v, bins=n_bins, range=(v.min(), v.max()))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(n_bins))


def nw(span, k=1200.0):
    return int(math.floor(span / (1.5 * math.sqrt(1.98720425864083e-3 * 300.0 / k))))


def iqr(v):
    q = np.quantile(v, [0.25, 0.75])
    return float(q[1] - q[0])


def torsion_pca(traj, cv1):
    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    cols = []
    for arr in (phis, psis):
        for k in range(arr.shape[1]):
            cols.append(np.sin(arr[:, k]))
            cols.append(np.cos(arr[:, k]))
    X = np.stack(cols, axis=1)
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    A = np.column_stack([np.ones(len(X)), cv1])
    beta, *_ = np.linalg.lstsq(A, Xc, rcond=None)
    Xr = Xc - A @ beta
    _, S2, Vt2 = np.linalg.svd(Xr, full_matrices=False)
    evr2 = (S2 ** 2) / (S2 ** 2).sum()
    return Vt, Vt2, evr, evr2, phis, psis, phi_idx, psi_idx, Xc


def psi_means_states(traj, psi_res, cv2, nstates=6):
    q = np.quantile(cv2, np.linspace(0, 1, nstates + 1))
    out = []
    for i in range(nstates):
        m = (cv2 >= q[i]) & (cv2 <= q[i + 1])
        d = {RES3[r]: float(np.degrees(np.angle(np.exp(1j * psi_col[m]).mean())))
             for r, psi_col in psi_res.items() if r in (1, 3, 8)}
        out.append(d)
    return out


def main():
    r7_pdbs = sorted((R7 / "final_implicit_survivor_seeds").glob("*.pdb"))
    r2_pdbs = sorted((R2 / "final_implicit_survivor_seeds").glob("*.pdb"))
    print(f"r7 bank {len(r7_pdbs)} seeds; rep2 bank {len(r2_pdbs)} seeds")
    t7 = md.load(r7_pdbs)
    t2 = md.load(r2_pdbs)

    rows = []
    vals = {}
    for name, (sel, sep, r0, beta) in DEFS.items():
        v7 = contact_cv(t7, *sel_pairs(t7.topology, sel, sep), r0, beta)
        v2 = contact_cv(t2, *sel_pairs(t2.topology, sel, sep), r0, beta)
        vals[name] = (v7, v2)
        ks = ks_2samp(v7, v2)
        rec = {
            "cv": name,
            "r7": {"median": float(np.median(v7)), "IQR": iqr(v7), "entropy": entropy_norm(v7),
                   "nw": nw(float(np.quantile(v7, .99) - np.quantile(v7, .01)))},
            "r2": {"median": float(np.median(v2)), "IQR": iqr(v2), "entropy": entropy_norm(v2),
                   "nw": nw(float(np.quantile(v2, .99) - np.quantile(v2, .01)))},
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        }
        rows.append(rec)
        print(f"{name}: r7 med {rec['r7']['median']:.3f} ent {rec['r7']['entropy']:.3f} nw {rec['r7']['nw']} | "
              f"r2 med {rec['r2']['median']:.3f} ent {rec['r2']['entropy']:.3f} nw {rec['r2']['nw']} | KS {ks.statistic:.3f} p={ks.pvalue:.1e}")

    cv1_7 = vals["bb_r010_b3"][0]
    cv1_2 = vals["bb_r010_b3"][1]
    Vt7, VtR7, evr7, evrR7, _, _, _, _, _ = torsion_pca(t7, cv1_7)
    Vt2, VtR2, evr2, evrR2, _, _, _, _, _ = torsion_pca(t2, cv1_2)
    axcos = {
        "plain_pc1": float(abs(np.dot(Vt7[0], Vt2[0]))),
        "plain_pc2": float(abs(np.dot(Vt7[1], Vt2[1]))),
        "res_pc1": float(abs(np.dot(VtR7[0], VtR2[0]))),
        "plain_pc1x2_top5_overlap": float((np.linalg.svd(Vt7[:5] @ Vt2[:5].T, compute_uv=False) ** 2).mean()),
    }
    print("axis cosines:", {k: round(v, 3) for k, v in axcos.items()})
    print(f"plain EVR[1-5] r7 {[round(float(x),3) for x in evr7[:5]]} vs r2 {[round(float(x),3) for x in evr2[:5]]}")

    _, _, _, _, _, psi7, _, psi7_idx, _ = torsion_pca(t7, cv1_7)
    psi7_res = {int(t7.topology.atom(int(psi7_idx[k][1])).residue.index): psi7[:, k] for k in range(psi7.shape[1])}
    _, _, _, _, _, psi2, _, psi2_idx, _ = torsion_pca(t2, cv1_2)
    psi2_res = {int(t2.topology.atom(int(psi2_idx[k][1])).residue.index): psi2[:, k] for k in range(psi2.shape[1])}
    _, _, _, _, _, _, _, _, Xc7 = torsion_pca(t7, cv1_7)
    _, _, _, _, _, _, _, _, Xc2 = torsion_pca(t2, cv1_2)
    pc2_7 = Xc7 @ Vt7[1]
    pc2_2 = Xc2 @ Vt2[1]
    states7 = psi_means_states(t7, psi7_res, pc2_7)
    states2 = psi_means_states(t2, psi2_res, pc2_2)
    print("curl ladder per bank (psi Y2/D3/T8 chords):")
    for i, (a, b) in enumerate(zip(states7, states2)):
        print(f"  state {i}: r7 {a} | r2 {b}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for ax, (name, (v7, v2)) in zip(axes.ravel(), vals.items()):
        edges = np.linspace(min(v7.min(), v2.min()), max(v7.max(), v2.max()), 55)
        ax.hist(v7, bins=edges, density=True, alpha=.55, label=f"r7 (n={len(v7)})", color="#2c7fb8")
        ax.hist(v2, bins=edges, density=True, alpha=.55, label=f"rep2 (n={len(v2)})", color="#e6550d")
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("bank-to-bank reproducibility: headline contact CVs, r7 vs rep2 (same params, different seed)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_rep2_comparison.png", dpi=150)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    labels = list(axcos)
    ax.bar(range(len(labels)), [axcos[k] for k in labels], color="#3182bd")
    ax.axhline(0.95, color="red", ls="--", lw=1, label="robustness bar 0.95")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("PCA axis agreement between independent banks (same params)", fontsize=10)
    fig2.savefig(FIG / "cv_rep2_axis_agreement.png", dpi=150)
    plt.close(fig2)

    doc = {"n_r7": len(t7), "n_r2": len(t2), "per_cv": rows, "axis_cos": axcos,
           "evr_plain": {"r7": evr7[:6].tolist(), "r2": evr2[:6].tolist()},
           "evr_res": {"r7": evrR7[:6].tolist(), "r2": evrR2[:6].tolist()},
           "ladder": {"r7": states7, "r2": states2}}
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# Bank-to-bank reproducibility: r7 vs rep2 (identical GENPEPT params, different seed)\n"]
    lines.append(f"r7 n={len(t7)}, rep2 n={len(t2)} (both 200k proposals, bias 0.15, chignolin preset, "
                 f"backend full, OpenCL; seed 12345 vs 54321)\n")
    lines.append("| CV | med r7 | med r2 | IQR r7 | IQR r2 | ent r7 | ent r2 | nw r7 | nw r2 | KS | p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['cv']} | {r['r7']['median']:.3f} | {r['r2']['median']:.3f} | "
                     f"{r['r7']['IQR']:.3f} | {r['r2']['IQR']:.3f} | {r['r7']['entropy']:.3f} | {r['r2']['entropy']:.3f} | "
                     f"{r['r7']['nw']} | {r['r2']['nw']} | {r['ks_stat']:.3f} | {r['ks_p']:.1e} |")
    lines.append("\nAxis agreement between banks:")
    for k, v in axcos.items():
        lines.append(f"- {k}: |cos| {v:.4f}")
    lines.append(f"- plain EVR[1-6] r7 {[round(float(x),3) for x in evr7[:6]]} vs r2 {[round(float(x),3) for x in evr2[:6]]}")
    lines.append("\nCurl ladder chords (psi circular means, Y2/D3/T8):")
    for i, (a, b) in enumerate(zip(states7, states2)):
        lines.append(f"- state {i}: r7 {a} vs r2 {b}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
