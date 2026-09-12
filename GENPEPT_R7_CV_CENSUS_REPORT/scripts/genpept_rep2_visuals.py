from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R7 = BASE / "RUNS" / "chignolin_genpept_r7"
R2 = BASE / "chignolin_genpept_rep2"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_JSON = REPORT / "genpept_r7_vs_rep2.json"

RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def bank_torsions(traj):
    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    cols = []
    for arr in (phis, psis):
        for k in range(arr.shape[1]):
            cols.append(np.sin(arr[:, k]))
            cols.append(np.cos(arr[:, k]))
    X = np.stack(cols, axis=1)
    psi_res = {int(traj.topology.atom(int(psi_idx[k][1])).residue.index): psis[:, k] for k in range(psis.shape[1])}
    phi_res = {int(traj.topology.atom(int(phi_idx[k][1])).residue.index): phis[:, k] for k in range(phis.shape[1])}
    return X, phi_res, psi_res


def main():
    t7 = md.load(sorted((R7 / "final_implicit_survivor_seeds").glob("*.pdb")))
    t2 = md.load(sorted((R2 / "final_implicit_survivor_seeds").glob("*.pdb")))
    X7, phi7, psi7 = bank_torsions(t7)
    X2, phi2, psi2 = bank_torsions(t2)
    Xc7 = X7 - X7.mean(0)
    _, _, Vt7 = np.linalg.svd(Xc7, full_matrices=False)
    a2 = Vt7[1]
    a1 = Vt7[0]
    c1_7, c1_2 = Xc7 @ a1, (X2 - X7.mean(0)) @ a1
    c2_7, c2_2 = Xc7 @ a2, (X2 - X7.mean(0)) @ a2

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    edges = np.linspace(min(c2_7.min(), c2_2.min()), max(c2_7.max(), c2_2.max()), 60)
    ax.hist(c2_7, bins=edges, density=True, alpha=.55, label="r7", color="#2c7fb8")
    ax.hist(c2_2, bins=edges, density=True, alpha=.55, label="rep2", color="#e6550d")
    ax.set_title("common-axis projection: r7 plain PC2\n(composition nearly identical)", fontsize=9)
    ax.legend(fontsize=8)
    q = np.quantile(c2_7, np.linspace(0, 1, 7))
    for i in range(1, 6):
        ax.axvline(q[i], color="#555555", lw=.5, ls=":")
    ax.set_xlabel("projection on r7 PC2")
    ax.tick_params(labelsize=8)

    ax = axes[0, 1]
    edges = np.linspace(min(c1_7.min(), c1_2.min()), max(c1_7.max(), c1_2.max()), 55)
    ax.hist(c1_7, bins=edges, density=True, alpha=.55, label="r7", color="#2c7fb8")
    ax.hist(c1_2, bins=edges, density=True, alpha=.55, label="rep2", color="#e6550d")
    ax.set_title("common-axis projection: r7 plain PC1 (|cos| 0.965 between banks)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlabel("projection on r7 PC1")
    ax.tick_params(labelsize=8)

    ladder = [(r,) for r in []]
    states_rows = []
    for i in range(6):
        m7 = (c2_7 >= q[i]) & (c2_7 <= q[i + 1])
        m2 = (c2_2 >= q[i]) & (c2_2 <= q[i + 1])
        row7 = [float(np.degrees(np.angle(np.exp(1j * psi7[res][m7]).mean()))) for res in (1, 2, 3, 4, 6, 7, 8)]
        row2 = [float(np.degrees(np.angle(np.exp(1j * psi2[res][m2]).mean()))) for res in (1, 2, 3, 4, 6, 7, 8)]
        states_rows.append((row7, row2, int(m7.sum()), int(m2.sum())))
    reslabs = ["psi Y2", "psi D3", "psi P4", "psi E5", "psi G7", "psi T8", "psi W9"]
    wm7 = np.array([r[0] for r in states_rows])
    wm2 = np.array([r[1] for r in states_rows])
    ax = axes[1, 0]
    im = ax.imshow(wm7, cmap="RdBu_r", vmin=-180, vmax=180, aspect="auto")
    ax.set_xticks(range(len(reslabs))); ax.set_xticklabels(reslabs, rotation=45, fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels([f"S{i} (n={r[2]})" for i, r in enumerate(states_rows)], fontsize=7)
    ax.set_title("r7: per-state circular-mean psi (deg)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.85)
    ax = axes[1, 1]
    im = ax.imshow(wm2, cmap="RdBu_r", vmin=-180, vmax=180, aspect="auto")
    ax.set_xticks(range(len(reslabs))); ax.set_xticklabels(reslabs, rotation=45, fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels([f"S{i} (n={r[3]})" for i, r in enumerate(states_rows)], fontsize=7)
    ax.set_title("rep2: same states on the common axis", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.85)
    fig.suptitle("bank-to-bank reproducibility: projections and 6-state ladder on the common r7 PC2 axis", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_rep2_common_axis.png", dpi=150)
    plt.close(fig)

    stg7 = json.load(open(R7 / "GENPEPT_turbo_summary.json"))
    stg2 = json.load(open(R2 / "GENPEPT_turbo_summary.json"))
    counts7 = stg7["counts"]
    counts2 = stg2["counts"]
    stages = ["candidate_seeds", "implicit_minima", "basin_hop_minima", "final_survivors"]
    labels = ["candidates", "implicit minima", "BH minima", "final survivors"]
    v7 = [counts7.get(s, 0) for s in stages]
    v2 = [counts2.get(s, 0) for s in stages]
    x = np.arange(len(stages))
    fig2, ax2 = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    ax2.bar(x - .2, v7, width=.4, label="r7", color="#2c7fb8")
    ax2.bar(x + .2, v2, width=.4, label="rep2", color="#e6550d")
    for xi, (a, b) in enumerate(zip(v7, v2)):
        ax2.text(xi - .2, a + max(v7 + v2) * .01, str(a), ha="center", fontsize=8)
        ax2.text(xi + .2, b + max(v7 + v2) * .01, str(b), ha="center", fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.set_yscale("log")
    ax2.legend(fontsize=9)
    ax2.set_title("pipeline stage counts (same params): r7 vs rep2", fontsize=10)
    fig2.savefig(FIG / "cv_rep2_stage_counts.png", dpi=150)
    plt.close(fig2)

    print("wrote", FIG / "cv_rep2_common_axis.png", FIG / "cv_rep2_stage_counts.png")


if __name__ == "__main__":
    main()
