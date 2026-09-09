from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, circmean, circstd

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
ROOT = BASE / "RUNS" / "chignolin_genpept_r7"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
VALUES = REPORT / "genpept_r7_cv_census_values.csv"
OUT_MD = REPORT / "TURN_AXIS.md"

RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def main():
    files = sorted((ROOT / "final_implicit_survivor_seeds").glob("*.pdb"))
    names = [f.stem for f in files]
    traj = md.load(files)
    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    phi_res = [int(traj.topology.atom(int(phi_idx[k][1])).residue.index) for k in range(phis.shape[1])]
    psi_res = [int(traj.topology.atom(int(psi_idx[k][1])).residue.index) for k in range(psis.shape[1])]

    census = pd.read_csv(VALUES).set_index("seed").loc[names]
    cv2 = census["restorpca_pc1"].to_numpy()
    cv1 = census["c_bb_s4_r010_b3"].to_numpy()

    q10, q90 = np.quantile(cv2, [0.1, 0.9])
    lo = np.where(cv2 <= q10)[0]
    hi = np.where(cv2 >= q90)[0]
    print(f"cv2 deciles: lo {len(lo)} (<= {q10:.3f}), hi {len(hi)} (>= {q90:.3f})")
    print(f"mean CV1 in deciles: {cv1[lo].mean():.3f} (lo) vs {cv1[hi].mean():.3f} (hi)")

    rows = []
    for k in range(phis.shape[1]):
        r = phi_res[k]
        a_lo = circmean(phis[lo, k], high=np.pi, low=-np.pi)
        a_hi = circmean(phis[hi, k], high=np.pi, low=-np.pi)
        dphi = math.degrees(np.angle(np.exp(1j * (a_hi - a_lo))))
        r_lo = float(np.abs(np.exp(1j * phis[lo, k]).mean()))
        r_hi = float(np.abs(np.exp(1j * phis[hi, k]).mean()))
        rows.append(("phi", r, math.degrees(a_lo), math.degrees(a_hi), dphi, r_lo, r_hi))
    for k in range(psis.shape[1]):
        r = psi_res[k]
        a_lo = circmean(psis[lo, k], high=np.pi, low=-np.pi)
        a_hi = circmean(psis[hi, k], high=np.pi, low=-np.pi)
        dpsi = math.degrees(np.angle(np.exp(1j * (a_hi - a_lo))))
        r_lo = float(np.abs(np.exp(1j * psis[lo, k]).mean()))
        r_hi = float(np.abs(np.exp(1j * psis[hi, k]).mean()))
        rows.append(("psi", r, math.degrees(a_lo), math.degrees(a_hi), dpsi, r_lo, r_hi))

    per_res = {}
    for kind, r, lo_a, hi_a, da, r_lo, r_hi in rows:
        per_res.setdefault(RES3[r], {})[kind] = (lo_a, hi_a, da, r_lo, r_hi)

    print(f"{'res':<5} {'phi lo->hi':<22} {'Δphi':>7}   {'psi lo->hi':<22} {'Δpsi':>7}")
    for res in RES3[1:9]:
        rec = per_res.get(res, {})
        p = rec.get("phi", (np.nan,) * 5)
        s = rec.get("psi", (np.nan,) * 5)
        print(f"{res:<5} {p[0]:+7.1f} -> {p[1]:+7.1f} {p[2]:+7.1f}   {s[0]:+7.1f} -> {s[1]:+7.1f} {s[2]:+7.1f}")

    corrs = {}
    for c in ["strand_cross_far_r012_b3", "c_bb_s4_r010_b3", "c_heavy_s4_r012_b3", "d_d3_t8_ca",
              "d_p4_g7_ca", "hairpin_closure_ratio", "turn3537_mean_ca", "rg_heavy"]:
        corrs[c] = spearmanr(cv2, census[c].to_numpy()).correlation
    print("corr(cv2, ...):", {k: round(v, 3) for k, v in corrs.items()})

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(3, 4, height_ratios=(1.4, 1.4, 1.0))
    rama_axes = []
    turn_res_idx = [2, 3, 4, 6, 7, 8]
    for i, r in enumerate(turn_res_idx):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        kphi = phi_res.index(r)
        kpsi = psi_res.index(r)
        ax.scatter(np.degrees(phis[:, kphi]), np.degrees(psis[:, kpsi]),
                   c=cv2, s=5, cmap="coolwarm", vmin=-2.5, vmax=2.5)
        ax.set_xlabel("phi"); ax.set_ylabel("psi")
        ax.set_title(f"Ramachandran {RES3[r]} (color = residual CV2)", fontsize=9)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        rama_axes.append(ax)
    axbar = fig.add_subplot(gs[0, 3])
    axcorr = fig.add_subplot(gs[1, 3])
    labs = []
    dp = []
    dps = []
    for res in RES3[1:9]:
        rec = per_res[res]
        labs.append(res)
        dp.append(rec["phi"][2] if "phi" in rec else np.nan)
        dps.append(rec["psi"][2] if "psi" in rec else np.nan)
    xpos = np.arange(len(labs))
    axbar.bar(xpos - .2, dp, width=.4, label="Δphi")
    axbar.bar(xpos + .2, dps, width=.4, label="Δpsi")
    axbar.set_xticks(xpos); axbar.set_xticklabels(labs, fontsize=8)
    axbar.set_ylabel("circular-mean rotation (deg), CV2 low-decile -> high-decile")
    axbar.axhline(0, color="k", lw=.8)
    axbar.legend(fontsize=8); axbar.grid(alpha=.3, axis="y")
    axbar.set_title("per-residue torsion rotation across the CV2 axis", fontsize=9)
    cks = list(corrs)
    cvs = [corrs[c] for c in cks]
    axcorr.barh(range(len(cks)), cvs, color="teal")
    axcorr.set_yticks(range(len(cks)))
    axcorr.set_yticklabels(cks, fontsize=7.5)
    axcorr.set_xlabel("Spearman rho with residual CV2")
    axcorr.axvline(0, color="k", lw=.8)
    axcorr.grid(alpha=.3, axis="x")
    axcorr.set_title("CV2 vs physical descriptors", fontsize=9)
    axlow = fig.add_subplot(gs[2, :2])
    axlow.hist(cv2[lo], bins=40, color="dodgerblue", alpha=.7, label=f"low decile (n={len(lo)})")
    axlow.hist(cv2[hi], bins=40, color="salmon", alpha=.7, label=f"high decile (n={len(hi)})")
    axlow.set_xlabel("residual torsion CV2")
    axlow.legend(fontsize=8)
    axlow.set_title("decile definitions", fontsize=9)
    axtxt = fig.add_subplot(gs[2, 2:])
    axtxt.axis("off")
    txt = "CV1 (bb r0=10 b3) is matched across deciles:\n"
    txt += f"   low {cv1[lo].mean():.3f} vs high {cv1[hi].mean():.3f}\n"
    txt += "=> rotation pattern above is at ~constant compaction.\n"
    txt += "Concentration changes (R_lo vs R_hi) per torsion in TURN_AXIS.md."
    axtxt.text(0.02, 0.7, txt, fontsize=9, va="top")
    fig.suptitle("Physical content of the residual torsion PCA CV2 axis (bank r7)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(FIG / "cv2_turn_axis.png", dpi=150)
    plt.close(fig)

    lines = ["# Physical turn content of the residual-torsion CV2 axis (bank r7)\n"]
    lines.append(f"deciles: low n={len(lo)} (CV2 <= {q10:.3f}), high n={len(hi)} (CV2 >= {q90:.3f}); "
                 f"groups matched on CV1: {cv1[lo].mean():.3f} vs {cv1[hi].mean():.3f} (bb r0=10 b3).")
    lines.append("\nPer-residue circular means (degrees) and rotation low->high:")
    lines.append("| residue | phi low -> high | Δphi | R_lo->R_hi | psi low -> high | Δpsi | R_lo->R_hi |")
    lines.append("|---|---|---|---|---|---|---|")
    for res in RES3[1:9]:
        rec = per_res.get(res, {})
        p = rec.get("phi")
        s = rec.get("psi")
        prow = f"{p[0]:+.0f} -> {p[1]:+.0f}" if p else "-"
        psrow = f"{s[0]:+.0f} -> {s[1]:+.0f}" if s else "-"
        lines.append(f"| {res} | {prow} | {p[2]:+.0f} | {p[3]:.2f}->{p[4]:.2f} | {psrow} | {s[2]:+.0f} | {s[3]:.2f}->{s[4]:.2f} |"
                     if p and s else f"| {res} | (partial) | | | | | |")
    lines.append("\nSpearman rho of CV2 with physical descriptors:")
    for c, v in corrs.items():
        lines.append(f"- {c}: {v:+.3f}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, FIG / "cv2_turn_axis.png")


if __name__ == "__main__":
    main()
