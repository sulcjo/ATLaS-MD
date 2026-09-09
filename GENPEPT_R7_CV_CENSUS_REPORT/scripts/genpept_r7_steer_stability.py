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

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
ROOT = BASE / "RUNS" / "chignolin_genpept_r7"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "STEERABILITY_STABILITY.md"
OUT_JSON = REPORT / "genpept_r7_steer_stability.json"

DEFS = {
    "bb_r010_b3": ("backbone-heavy", 4, 10.0, 3.0),
    "ca_r010_b3": ("ca", 4, 10.0, 3.0),
    "heavy_r012_b3": ("heavy", 4, 12.0, 3.0),
    "legacy_heavy_r04.5_b6": ("heavy", 4, 4.5, 6.0),
}
SR3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


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


def cv_and_grad(xyz, ii, jj, r0, beta):
    x = xyz[:, ii]
    y = xyz[:, jj]
    diff = x - y
    d = np.linalg.norm(diff, axis=2) * 10.0
    unit = diff / (d[..., None] + 1e-12)
    c = 0.5 * (1.0 - np.tanh(0.5 * beta * (d - r0)))
    gamp = -0.25 * beta * (1.0 - np.tanh(0.5 * beta * (d - r0)) ** 2)
    G = np.zeros_like(xyz, dtype=float)
    F, A, _ = xyz.shape
    fi = np.repeat(np.arange(F), len(ii))
    rows_i = fi * A + np.tile(ii, F)
    rows_j = fi * A + np.tile(jj, F)
    G2 = G.reshape(F * A, 3)
    vi = (gamp[..., None] * unit).reshape(F * len(ii), 3)
    vj = (-gamp[..., None] * unit).reshape(F * len(ii), 3)
    np.add.at(G2, rows_i, vi)
    np.add.at(G2, rows_j, vj)
    npairs = len(ii)
    return c.mean(axis=1), G / npairs


def main():
    files = sorted((ROOT / "final_implicit_survivor_seeds").glob("*.pdb"))
    names = [f.stem for f in files]
    traj = md.load(files)
    top = traj.topology
    xyz = traj.xyz
    n = len(names)

    steal = {}
    gradients = {}
    for k, (sel, sep, r0, beta) in DEFS.items():
        ii, jj = sel_pairs(top, sel, sep)
        c, G = cv_and_grad(xyz, ii, jj, r0, beta)
        gnorm = np.linalg.norm(G, axis=2)
        g_mean = gnorm.mean(axis=1)
        g_rms = np.sqrt((gnorm ** 2).mean(axis=1))
        bins = np.quantile(c, np.linspace(0, 1, 13))
        per_bin = []
        for b in range(12):
            m = (c >= bins[b]) & (c < bins[b + 1] if b < 11 else c <= bins[b + 1])
            per_bin.append(float(g_rms[m].mean()) if m.any() else float("nan"))
        q05, q95 = np.quantile(c, [0.05, 0.95])
        edge = (c >= q05) & (c <= q95)
        steep = float(np.mean(g_rms[edge]))
        frac_low = float((g_rms[edge] < 1e-3 * np.max(g_rms)).mean())
        steal[k] = {"rms_grad_mean_dynamics_band": steep, "frac_nearzero_grad_in_band": frac_low,
                    "per_bin_rms": per_bin, "p0505": (float(q05), float(q95))}
        gradients[k] = g_rms
        print(f"{k}: pairs {len(ii)}, rms-grad in p05-95 {steep:.4f}/nm/term, frac-nearzero {frac_low:.3f}")

    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    cols = []
    for arr in (phis, psis):
        for k in range(arr.shape[1]):
            cols.append(np.sin(arr[:, k]))
            cols.append(np.cos(arr[:, k]))
    X = np.stack(cols, axis=1)
    Xc = X - X.mean(axis=0)
    census = pd.read_csv(REPORT / "genpept_r7_cv_census_values.csv").set_index("seed").loc[names]
    cv1 = census["c_bb_s4_r010_b3"].to_numpy()

    def fit_axes(mask):
        Xs = Xc[mask]
        _, S, Vt = np.linalg.svd(Xs - Xs.mean(axis=0), full_matrices=False)
        pc1 = Vt[0]
        xc = Xs - Xs.mean(axis=0)
        A = np.column_stack([np.ones(mask.sum()), cv1[mask]])
        beta, *_ = np.linalg.lstsq(A, xc, rcond=None)
        Xr = xc - A @ beta
        _, S2, Vt2 = np.linalg.svd(Xr, full_matrices=False)
        res1 = Vt2[0]
        return pc1, res1

    pc_full, res_full = fit_axes(np.ones(n, bool))

    def gen_class(name):
        if "_nma" in name or "mode" in name:
            return "NMA"
        if "_hop_" in name:
            return "BH"
        if "explore_" in name:
            return "PCA-frontier"
        return "initial"

    gens = np.array([gen_class(s) for s in names])
    stab = {}
    for g in ("initial", "BH", "NMA", "PCA-frontier"):
        m = gens == g
        if m.sum() < 300:
            stab[g] = {"n": int(m.sum()), "note": "skipped (<300 seeds)"}
            continue
        pc_g, res_g = fit_axes(m)
        stab[g] = {
            "n": int(m.sum()),
            "cos_pc1": float(abs(np.dot(pc_g, pc_full))),
            "cos_res1": float(abs(np.dot(res_g, res_full))),
        }
        print(f"{g}: n={m.sum()}  |cos PC1|={stab[g]['cos_pc1']:.3f}  |cos resPC1|={stab[g]['cos_res1']:.3f}")

    rng = np.random.default_rng(3)
    half_res = []
    for rep in range(30):
        m = np.zeros(n, bool)
        m[rng.choice(n, n // 2, replace=False)] = True
        pc_h, res_h = fit_axes(m)
        half_res.append((float(abs(np.dot(pc_h, pc_full))), float(abs(np.dot(res_h, res_full)))))
    half_res = np.array(half_res)
    print(f"half-bank resampling: |cos PC1| {np.median(half_res[:,0]):.4f} [{half_res[:,0].min():.4f}], "
          f"|cos resPC1| {np.median(half_res[:,1]):.4f} [{half_res[:,1].min():.4f}]")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (k, grad) in zip(axes, [(k, gradients[k]) for k in ("bb_r010_b3", "ca_r010_b3", "legacy_heavy_r04.5_b6")]):
        c, G = cv_and_grad(xyz, *sel_pairs(top, *DEFS[k][:2]), *DEFS[k][2:])
        bins = np.quantile(c, np.linspace(0, 1, 13))
        centers = 0.5 * (bins[:-1] + bins[1:])
        ys = [grad[(c >= bins[b]) & (c < bins[b + 1] if b < 11 else c <= bins[b + 1])].mean() for b in range(12)]
        ax.plot(centers, ys, "o-", color="#2c7fb8")
        ax.set_title(f"{k}\nsteering force vs CV value", fontsize=9)
        ax.set_xlabel("CV value")
        ax.set_ylabel("RMS |dCV/dx| (1/nm/term)")
        ax.grid(alpha=.3)
    fig.suptitle("per-atom steering strength of candidate CV1s across their own range (r7 bank)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_steerability.png", dpi=150)
    plt.close(fig)

    doc = {"steerability": {k: {kk: vv for kk, vv in v.items() if kk != "per_bin_rms"} for k, v in steal.items()},
           "steerability_per_bin": {k: v["per_bin_rms"] for k, v in steal.items()},
           "generator_split": stab,
           "half_bank": {"pc1_median": float(np.median(half_res[:, 0])), "pc1_min": float(half_res[:, 0].min()),
                         "res1_median": float(np.median(half_res[:, 1])), "res1_min": float(half_res[:, 1].min())}}
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# CV1 steerability + bank axis stability\n"]
    for k, v in steal.items():
        lines.append(f"- {k}: RMS |dCV/dx| in p05–p95 band {v['rms_grad_mean_dynamics_band']:.4f} /nm/term; "
                     f"frac near-zero-gradient {v['frac_nearzero_grad_in_band']:.3f}")
    lines.append("")
    for g, v in stab.items():
        if "cos_pc1" in v:
            lines.append(f"- {g} (n={v['n']}): |cos(PC1)| {v['cos_pc1']:.3f}, |cos(resPC1)| {v['cos_res1']:.3f}")
        else:
            lines.append(f"- {g}: {v['note']}")
    lines.append(f"- half-bank 30x: |cos PC1| {np.median(half_res[:,0]):.4f} (min {half_res[:,0].min():.4f}); "
                 f"|cos resPC1| {np.median(half_res[:,1]):.4f} (min {half_res[:,1].min():.4f})")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
