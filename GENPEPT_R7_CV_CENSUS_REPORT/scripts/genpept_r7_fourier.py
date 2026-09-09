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
from scipy.stats import spearmanr

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
ROOT = BASE / "RUNS" / "chignolin_genpept_r7"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
VALUES = REPORT / "genpept_r7_cv_census_values.csv"
PSI_CSV = REPORT / "genpept_r7_diffmap_psi.csv"
OUT_MD = REPORT / "FOURIER.md"
OUT_JSON = REPORT / "genpept_r7_fourier.json"

RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]
N_PERM = 300
RNG = np.random.default_rng(7)


def load_angles():
    files = sorted((ROOT / "final_implicit_survivor_seeds").glob("*.pdb"))
    traj = md.load(files)
    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    names = [f.stem for f in files]
    cols, labels = [], []
    for k in range(phis.shape[1]):
        r = int(traj.topology.atom(int(phi_idx[k][1])).residue.index)
        cols.append(phis[:, k]); labels.append(f"phi {RES3[r]}")
    for k in range(psis.shape[1]):
        r = int(traj.topology.atom(int(psi_idx[k][1])).residue.index)
        cols.append(psis[:, k]); labels.append(f"psi {RES3[r]}")
    return names, np.stack(cols, axis=1), labels


def circ_stats(theta, mmax=4):
    out = {"mean_dir": float(np.angle(np.exp(1j * theta).mean()))}
    for m in range(1, mmax + 1):
        out[f"R{m}"] = float(np.abs(np.exp(1j * m * theta).mean()))
    return out


def fish_lee(theta_i, theta_j):
    ti = theta_i - np.angle(np.exp(1j * theta_i).mean())
    tj = theta_j - np.angle(np.exp(1j * theta_j).mean())
    si, sj = np.sin(ti), np.sin(tj)
    den = math.sqrt((si * si).sum() * (sj * sj).sum())
    return float((si * sj).sum() / den) if den > 0 else float("nan")


def unwrap_ordered(theta, order):
    return np.unwrap(theta[order])


def winding(theta, order):
    u = unwrap_ordered(theta, order)
    n = len(u)
    x = np.arange(n)
    slope = np.polyfit(x, u, 1)[0]
    turns = abs(slope) * n / (2 * math.pi)
    y = u - u.mean()
    P = np.abs(np.fft.rfft(y)) ** 2
    conc = float(P[1:].max() / P[1:].sum()) if P[1:].sum() > 0 else float("nan")
    return float(turns), conc, float(slope)


def main():
    names, TH, labels = load_angles()
    n, d = TH.shape
    print(f"angles {TH.shape}; torsions: {labels}", flush=True)

    circ = {labels[j]: circ_stats(TH[:, j]) for j in range(d)}
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    wm = np.array([[circ[lab][f"R{m}"] for lab in labels] for m in (1, 2, 3, 4)])
    x = np.arange(d)
    for i, m in enumerate((1, 2, 3, 4)):
        axes[0].bar(x + (i - 1.5) * 0.2, wm[i], width=0.2, label=f"R{m}")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=60, fontsize=8)
    axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3, axis="y")
    axes[0].set_title("circular Fourier harmonics per torsion: R1 concentration, R2-R4 multimodality")

    M = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            M[i, j] = M[j, i] = fish_lee(TH[:, i], TH[:, j])
    im = axes[1].imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_xticks(range(d)); axes[1].set_yticks(range(d))
    axes[1].set_xticklabels(labels, rotation=60, fontsize=7)
    axes[1].set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=axes[1], shrink=.85, label="Fisher-Lee rho")
    axes[1].set_title("circular-circular torsion correlation")
    fig.tight_layout()
    fig.savefig(FIG / "cv_fourier_harmonics_corrmatrix.png", dpi=150)
    plt.close(fig)

    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            pairs.append((labels[i], labels[j], M[i, j]))
    pairs.sort(key=lambda t: -abs(t[2]))
    print("top circ-corr:", [(a, b, round(r, 3)) for a, b, r in pairs[:10]])

    census = pd.read_csv(VALUES).set_index("seed").loc[names]
    psidf = pd.read_csv(PSI_CSV).set_index("seed").loc[names]
    orderings = {
        "torsion-psi1": psidf["torsion_psi1"].to_numpy(),
        "ca_drmsd-psi1": psidf["ca_drmsd_psi1"].to_numpy(),
        "contact-CV1": census["c_bb_s4_r010_b3"].to_numpy(),
        "res-torsion-PC1": census["restorpca_pc1"].to_numpy(),
    }
    wind_out = {}
    for oname, xi in orderings.items():
        order = np.argsort(xi)
        per = []
        null_max = np.zeros(N_PERM)
        for j in range(d):
            t_, c_, s_ = winding(TH[:, j], order)
            per.append((labels[j], t_, c_, s_))
        for p in range(N_PERM):
            po = RNG.permutation(n)
            mx = 0.0
            for j in range(d):
                mx = max(mx, winding(TH[:, j], po)[0])
            null_max[p] = mx
        thr = float(np.quantile(null_max, 0.99))
        per.sort(key=lambda t: -t[1])
        flagged = [(lab, t_, c_, s_) for lab, t_, c_, s_ in per if t_ >= thr and t_ >= 0.5]
        wind_out[oname] = {"threshold_p99": thr, "ranked": per[:12], "flagged": flagged[:8]}
        print(f"{oname}: null p99 {thr:.2f} turns; flagged: {[(l, round(t,2)) for l,t,_,_ in flagged[:8]]}")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (oname, res) in zip(axes.ravel(), wind_out.items()):
        per = res["ranked"]
        labs = [p[0] for p in per]
        vals = [p[1] for p in per]
        ax.bar(range(len(labs)), vals, color="steelblue")
        ax.axhline(res["threshold_p99"], color="red", ls="--", lw=1, label="permutation p99")
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=60, fontsize=8)
        ax.set_ylabel("|winding| (turns of the torsion across the ordered bank)")
        ax.set_title(f"winding along {oname}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "cv_fourier_winding.png", dpi=150)
    plt.close(fig)

    best = max(((o, r) for o, r in wind_out.items()), key=lambda t: t[1]["ranked"][0][1])
    oname, res = best
    order = np.argsort(orderings[oname])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (lab, t_, c_, s_) in zip(axes, res["ranked"][:3]):
        j = labels.index(lab)
        u = unwrap_ordered(TH[:, j], order)
        ax.plot(u, lw=0.8)
        ax.set_xlabel(f"seeds sorted by {oname}")
        ax.set_ylabel("unwrapped angle (rad)")
        ax.set_title(f"{lab}: {t_:.2f} turns (FFT conc {c_:.2f})", fontsize=9)
        ax.grid(alpha=.3)
    fig.suptitle("strongest hidden winding signals", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "cv_fourier_winding_traces.png", dpi=150)
    plt.close(fig)

    doc = {"circ": circ, "circ_corr_top": pairs[:15], "winding": {
        o: {"threshold_p99": r["threshold_p99"], "flagged": r["flagged"]} for o, r in wind_out.items()}}
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# Fourier analysis of r7 seed-bank dihedrals\n"]
    lines.append(f"bank: {n} seeds; 18 backbone torsions (phi, psi per residue).")
    lines.append("\n## A. Circular Fourier harmonics R1..R4 per torsion")
    lines.append("R1 = concentration at one direction; R2..R4 high = multimodality/periodic components.")
    for lab in labels:
        c_ = circ[lab]
        flag = "  <- multimodal" if (c_["R1"] < 0.35 and c_["R2"] > 0.25) else ""
        lines.append(f"- {lab:<10} R1={c_['R1']:.3f} R2={c_['R2']:.3f} R3={c_['R3']:.3f} R4={c_['R4']:.3f} "
                     f"dir={math.degrees(c_['mean_dir']):+.0f}deg{flag}")
    lines.append("\n## B. Winding (hidden periodic signals) along ordered coordinates")
    lines.append(f"unwrap angle vs seeds sorted by coordinate; |turns| threshold = max-over-torsions permutation p99 (N={N_PERM}).")
    for oname, res in wind_out.items():
        if res["flagged"]:
            s = ", ".join(f"{l} {t:.2f} turns" for l, t, c_, s_ in res["flagged"][:8])
            lines.append(f"- {oname} (null p99 {res['threshold_p99']:.2f}): {s}")
        else:
            lines.append(f"- {oname} (null p99 {res['threshold_p99']:.2f}): none significant")
    lines.append("\n## C. Fisher-Lee circular-circular correlations (top pairs)")
    for a, b, r in pairs[:10]:
        lines.append(f"- {a} ~ {b}: rho={r:+.3f}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
