from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import mdtraj as md

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R7 = BASE / "RUNS" / "chignolin_genpept_r7"
R2 = Path("/tmp/opencode/chignolin_genpept_rep2")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def bank_torsions(traj):
    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    cols, labels = [], []
    for k in range(phis.shape[1]):
        r = int(traj.topology.atom(int(phi_idx[k][1])).residue.index)
        cols += [np.sin(phis[:, k]), np.cos(phis[:, k])]
        labels += [f"sin phi {RES3[r]}", f"cos phi {RES3[r]}"]
    for k in range(psis.shape[1]):
        r = int(traj.topology.atom(int(psi_idx[k][1])).residue.index)
        cols += [np.sin(psis[:, k]), np.cos(psis[:, k])]
        labels += [f"sin psi {RES3[r]}", f"cos psi {RES3[r]}"]
    X = np.stack(cols, axis=1)
    psi_res = {int(traj.topology.atom(int(psi_idx[k][1])).residue.index): psis[:, k] for k in range(psis.shape[1])}
    return X, psi_res


def main():
    t7 = md.load(sorted((R7 / "final_implicit_survivor_seeds").glob("*.pdb")))
    t2 = md.load(sorted((R2 / "final_implicit_survivor_seeds").glob("*.pdb")))
    X7, psi7 = bank_torsions(t7)
    X2, psi2 = bank_torsions(t2)

    Xc7 = X7 - X7.mean(0)
    _, _, Vt7 = np.linalg.svd(Xc7, full_matrices=False)
    a_common = Vt7[1]
    c7 = Xc7 @ a_common
    c2 = (X2 - X7.mean(0)) @ a_common

    q = np.quantile(c7, np.linspace(0, 1, 7))
    table = []
    for i in range(6):
        m7 = (c7 >= q[i]) & (c7 <= q[i + 1])
        m2 = (c2 >= q[i]) & (c2 <= q[i + 1])
        row = {"state": i, "range_lo": float(q[i]), "range_hi": float(q[i + 1]),
               "n_r7": int(m7.sum()), "n_r2": int(m2.sum())}
        for res, lab in ((1, "Y2"), (2, "D3"), (3, "P4"), (8, "W9")):
            row[f"r7_psi_{lab}"] = float(np.degrees(np.angle(np.exp(1j * psi7[res][m7]).mean()))) if m7.any() else np.nan
            row[f"r2_psi_{lab}"] = float(np.degrees(np.angle(np.exp(1j * psi2[res][m2]).mean()))) if m2.any() else np.nan
        table.append(row)
        print(row)

    means = {}
    for lab, res in (("Y2", 1), ("D3", 2), ("P4", 3), ("W9", 8), ("E5", 4), ("G7", 6), ("T8", 7)):
        m7 = float(np.degrees(np.angle(np.exp(1j * psi7[res]).mean())))
        m2 = float(np.degrees(np.angle(np.exp(1j * psi2[res]).mean())))
        means[lab] = (m7, m2)
        print(f"psi mean {lab}: r7 {m7:+.0f} | r2 {m2:+.0f}")

    corr = float(np.corrcoef(np.concatenate([c7, c2]),
                             np.concatenate([np.zeros(len(c7)), np.ones(len(c2))]))[0, 1])
    print(f"corr(bank indicator, common-axis projection) = {corr:+.3f}")

    j = json.load(open(REPORT / "genpept_r7_vs_rep2.json"))
    j["common_axis_note"] = "rep2 bank projected onto the r7 plain-PC2 eigenvector; comparing own-axis states ACROSS banks is NOT meaningful because own axes are not the same (plain PC2 |cos| 0.523)"
    j["ladder_common_axis"] = table
    j["psi_mean_by_bank"] = means
    j["corr_bank_indicator_common_axis"] = corr
    json.dump(j, open(REPORT / "genpept_r7_vs_rep2.json", "w"), indent=1)

    lines = []
    lines.append("\n## Common-axis projection (rep2 onto r7's plain-PC2 eigenvector) — the meaningful way to compare states")
    lines.append(f"- corr(bank indicator, common-axis projection): {corr:+.3f}")
    for row in table:
        lines.append(f"- state {row['state']} [{row['range_lo']:.2f},{row['range_hi']:.2f}]: n r7/r2 {row['n_r7']}/{row['n_r2']}"
                     + "; " + "; ".join(f"{lab} r7 {row[f'r7_psi_{lab}']:+.0f} r2 {row[f'r2_psi_{lab}']:+.0f}"
                                        for lab in ("Y2", "D3", "P4", "W9")))
    lines.append("\n## Own-axis-per-bank state comparison is NOT meaningful (and therefore not used elsewhere)")
    lines.append("- plain PC2's direction differs across independent banks (|cos| 0.523), so state labels under own axes do not align.")
    lines.append("- plain PC1 DOES transfer across banks (|cos| 0.965): compaction-dominant direction is bank-invariant.")
    open(REPORT / "REP2_COMPARISON.md", "a").write("\n".join(lines))
    print("appended")


if __name__ == "__main__":
    main()
