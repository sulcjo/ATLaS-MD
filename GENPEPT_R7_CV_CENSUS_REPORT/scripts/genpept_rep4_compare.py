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
from scipy.stats import gaussian_kde, ks_2samp

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R3 = BASE / "chignolin_genpept_rep3_massive"
R4 = BASE / "chignolin_genpept_rep4_tuned"
if not R4.exists():
    R4 = Path("/tmp/opencode/chignolin_genpept_rep4_tuned")
NATIVE = Path("/tmp/opencode/1UAO.pdb")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "REP4_TUNED_COMPARISON.md"
OUT_JSON = REPORT / "genpept_rep4_vs_rep3.json"
RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def sel_pairs(top, selection, sep):
    BB = {"N", "CA", "C", "O"}
    idx, res_of = [], []
    for a in top.atoms:
        keep = ((selection == "heavy" and (a.element is not None and a.element.symbol != "H"))
                or (selection == "backbone-heavy" and a.name in BB)
                or (selection == "ca" and a.name == "CA"))
        if keep:
            idx.append(a.index)
        res_of.append(a.residue.index)
    idx = np.array(idx)
    ii, jj = np.triu_indices(len(idx), k=1)
    keep = np.abs(np.array(res_of)[idx[ii]] - np.array(res_of)[idx[jj]]) >= sep
    return idx[ii][keep], idx[jj][keep]


def contact_cv(traj, selection, sep, r0, beta):
    ii, jj = sel_pairs(traj.topology, selection, sep)
    d = np.linalg.norm(traj.xyz[:, ii] - traj.xyz[:, jj], axis=2) * 10.0
    return (0.5 * (1.0 - np.tanh(0.5 * beta * (d - r0)))).mean(axis=1)


def entropy_norm(v, nb=20):
    h, _ = np.histogram(v, bins=nb, range=(v.min(), v.max()))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(nb))


def nw(span):
    return int(math.floor(span / (1.5 * math.sqrt(1.98720425864083e-3 * 300.0 / 1200.0))))


def chir_per_frame(ca_xyz):
    v1 = ca_xyz[:, 4] - ca_xyz[:, 0]
    v2 = ca_xyz[:, 7] - ca_xyz[:, 0]
    v3 = ca_xyz[:, 9] - ca_xyz[:, 0]
    return np.linalg.det(np.stack([v1, v2, v3], axis=1)) * 1e3


def hbond_dists(traj):
    topo = traj.topology
    n = [a.index for a in topo.residue(2).atoms if a.name == "N"][0]
    o7 = [a.index for a in topo.residue(6).atoms if a.name == "O"][0]
    o8 = [a.index for a in topo.residue(7).atoms if a.name == "O"][0]
    d37 = np.linalg.norm(traj.xyz[:, n, :] - traj.xyz[:, o7, :], axis=1) * 10.0
    d38 = np.linalg.norm(traj.xyz[:, n, :] - traj.xyz[:, o8, :], axis=1) * 10.0
    return d37, d38


def main():
    native = md.load(NATIVE)
    env = json.load(open(REPORT / "genpept_rep3_fes_folded.json"))["native_envelope"]
    banks = {}
    for tag, root in (("rep3", R3), ("rep4", R4)):
        files = sorted((root / "final_implicit_survivor_seeds").glob("*.pdb"))
        traj = md.load(files)
        ca = traj.atom_slice([a.index for a in traj.topology.atoms if a.name == "CA"]).xyz
        ca_ref = [a.index for a in native.topology.atoms if a.name == "CA"]
        rmsd_stack = np.stack([md.rmsd(traj, native, frame=m, atom_indices=[a.index for a in traj.topology.atoms if a.name == "CA"], ref_atom_indices=ca_ref) for m in range(native.n_frames)], axis=1) * 10.0
        d37, d38 = hbond_dists(traj)
        in_env = (d37 >= env["d37"][0]) & (d37 <= env["d37"][1]) & (d38 >= env["d38"][0]) & (d38 <= env["d38"][1])
        cv_bb = contact_cv(traj, "backbone-heavy", 4, 10.0, 3.0)
        cv_ca = contact_cv(traj, "ca", 4, 10.0, 3.0)
        cv_hv = contact_cv(traj, "heavy", 4, 12.0, 3.0)
        sasa = md.shrake_rupley(traj, mode="residue").sum(axis=1)
        ch = chir_per_frame(ca)
        banks[tag] = {"files": files, "traj": traj, "cv": {"bb": cv_bb, "ca": cv_ca, "heavy": cv_hv},
                      "in_env": in_env, "rmsd_min": rmsd_stack.min(axis=1), "sasa": sasa, "chir": ch}
        print(f"{tag}: {len(files)} seeds | env {in_env.sum()} ({in_env.mean()*100:.2f}%) "
              f"| rmsd_min {banks[tag]['rmsd_min'].min():.2f} | q05 sasa {np.quantile(sasa, 0.05):.2f} "
              f"| chir frac+ {(ch>0).mean():.3f}")

    rows = []
    for cvn in ("bb", "ca", "heavy"):
        for tag in banks:
            rows.append({"cv": cvn, "bank": tag,
                         "median": float(np.median(banks[tag]["cv"][cvn])),
                         "iqr": float(np.percentile(banks[tag]["cv"][cvn], 75) - np.percentile(banks[tag]["cv"][cvn], 25)),
                         "entropy": entropy_norm(banks[tag]["cv"][cvn]),
                         "nw": nw(float(np.quantile(banks[tag]["cv"][cvn], .99) - np.quantile(banks[tag]["cv"][cvn], .01))),
                         "p05": float(np.quantile(banks[tag]["cv"][cvn], .05)),
                         "p95": float(np.quantile(banks[tag]["cv"][cvn], .95))})
    comp_df = pd.DataFrame(rows)

    funnel = {}
    for tag, root in (("rep3", R3), ("rep4", R4)):
        st = json.load(open(root / "GENPEPT_turbo_summary.json"))
        funnel[tag] = st["counts"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, cvn in enumerate(("bb", "ca", "heavy")):
        ax = axes[0, i]
        e = np.linspace(min(banks["rep3"]["cv"][cvn].min(), banks["rep4"]["cv"][cvn].min()),
                        max(banks["rep3"]["cv"][cvn].max(), banks["rep4"]["cv"][cvn].max()), 60)
        ax.hist(banks["rep3"]["cv"][cvn], bins=e, density=True, alpha=.55, label="rep3 (legacy bias)", color="#888888")
        ax.hist(banks["rep4"]["cv"][cvn], bins=e, density=True, alpha=.55, label="rep4 (CU+tuned bundle)", color="#e6550d")
        ax.legend(fontsize=8)
        ax.set_title(cvn, fontsize=9)
    for tag, col in (("rep3", "#888888"), ("rep4", "#e6550d")):
        ax = axes[1, 0]
        ax.hist(banks[tag]["sasa"], bins=50, density=True, alpha=.55, label=tag, color=col)
    ax = axes[1, 0]
    ax.legend(fontsize=8)
    ax.set_title("SASA total (nm^2)", fontsize=9)
    ax = axes[1, 1]
    for tag, col in (("rep3", "#888888"), ("rep4", "#e6550d")):
        ax.hist(banks[tag]["rmsd_min"], bins=50, density=True, alpha=.55, label=tag, color=col)
    ax.legend(fontsize=8)
    ax.set_title("min CA-RMSD to 1UAO", fontsize=9)
    ax = axes[1, 2]
    for tag, col in (("rep3", "#888888"), ("rep4", "#e6550d")):
        ax.hist(banks[tag]["chir"], bins=50, density=True, alpha=.55, label=tag, color=col)
    ax.legend(fontsize=8)
    ax.set_title("chirality (mA^3); native = negative", fontsize=9)
    fig.suptitle("rep4 (all tuning mechanisms) vs rep3 (legacy contact bias)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_rep4_vs_rep3.png", dpi=150)
    plt.close(fig)

    doc = {"funnel": funnel,
           "per_bank": {}}
    for tag in banks:
        b = banks[tag]
        doc["per_bank"][tag] = {
            "n": len(b["files"]),
            "env_count": int(b["in_env"].sum()),
            "env_frac": float(b["in_env"].mean()),
            "rmsd_min": float(b["rmsd_min"].min()),
            "q05_sasa": float(np.quantile(b["sasa"], 0.05)),
            "median_sasa": float(np.median(b["sasa"])),
            "chir_frac_pos": float((b["chir"] > 0).mean()),
            "cv_stats": comp_df[comp_df.bank == tag].to_dict("records"),
        }
    doc["comp_table"] = rows
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# rep4 (tuning bundle) vs rep3 (legacy contact bias), same bank scale\n"]
    lines.append("Funnel counts:\n")
    for tag in ("rep3", "rep4"):
        lines.append(f"- {tag}: {funnel[tag]}")
    lines.append("\nPer-cv stats:\n")
    lines.append("| cv | bank | median | IQR | entropy | nw@1200 | p05-p95 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['cv']} | {r['bank']} | {r['median']:.3f} | {r['iqr']:.3f} | {r['entropy']:.3f} | {r['nw']} | {r['p05']:.3f}-{r['p95']:.3f} |")
    lines.append("\nFolded-state / tail metrics:\n")
    for tag in ("rep3", "rep4"):
        b = banks[tag]
        lines.append(f"- {tag}: native-envelope {int(b['in_env'].sum())} seeds ({b['in_env'].mean()*100:.2f}%), "
                     f"best RMSD {b['rmsd_min'].min():.2f} A, q05 SASA {np.quantile(b['sasa'], 0.05):.2f} nm2, "
                     f"median SASA {np.median(b['sasa']):.2f}, chirality frac+ {(b['chir'] > 0).mean():.3f}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON, FIG / "cv_rep4_vs_rep3.png")


if __name__ == "__main__":
    main()
