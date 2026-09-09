from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
ROOT = BASE / "RUNS" / "chignolin_genpept_r7"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
VALUES = REPORT / "genpept_r7_cv_census_values.csv"
OUT_MD = REPORT / "DIFFMAP.md"
OUT_PSI = REPORT / "genpept_r7_diffmap_psi.csv"
OUT_JSON = REPORT / "genpept_r7_diffmap.json"

K_EIG = 10
RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def load_bank():
    import openmm.app as app_mod
    files = sorted((ROOT / "final_implicit_survivor_seeds").glob("*.pdb"))
    traj = md.load(files)
    return files, traj


def torsion_features(traj):
    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    cols = []
    labels = []
    for k in range(phis.shape[1]):
        cols += [np.sin(phis[:, k]), np.cos(phis[:, k])]
        r = int(traj.topology.atom(int(phi_idx[k][1])).residue.index)
        labels += [f"sin phi {RES3[r]}", f"cos phi {RES3[r]}"]
    for k in range(psis.shape[1]):
        cols += [np.sin(psis[:, k]), np.cos(psis[:, k])]
        r = int(traj.topology.atom(int(psi_idx[k][1])).residue.index)
        labels += [f"sin psi {RES3[r]}", f"cos psi {RES3[r]}"]
    return np.stack(cols, axis=1), labels


def ca_drmsd_features(traj):
    ca = np.array([a.index for a in traj.topology.atoms if a.name == "CA"])
    xyz = traj.atom_slice(ca).xyz
    iu = np.triu_indices(len(ca), k=1)
    return np.linalg.norm(xyz[:, iu[0]] - xyz[:, iu[1]], axis=2)


def zscore(A):
    return (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)


def dmap(D2, eps):
    n = D2.shape[0]
    K = np.exp(-D2 / (2.0 * eps * eps))
    np.fill_diagonal(K, 0.0)
    q = K.sum(axis=1)
    Dm12 = np.diag(1.0 / np.sqrt(q))
    S = Dm12 @ K @ Dm12
    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(-evals)
    evals, evecs = evals[order], evecs[:, order]
    psi = Dm12 @ evecs
    psi = psi / np.linalg.norm(psi, axis=0)
    lam = evals
    return lam, psi


def twonn(D, upper=0.9):
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    srt = np.partition(Dm, 2, axis=1)
    mu = srt[:, 1] / srt[:, 0]
    mu = np.sort(mu)
    n_len = len(mu)
    keep = int(upper * n_len)
    mu_k = mu[:keep]
    p_emp = (np.arange(1, n_len + 1) / n_len)[:keep]
    y = -np.log(1.0 - p_emp)
    x = np.log(mu_k)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    npx = len(x)
    s2 = (resid @ resid) / max(1, npx - 2)
    sxx = ((x - x.mean()) ** 2).sum()
    d_se = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    return float(slope), float(d_se), float(np.exp(intercept))


def spearman_tbl(psi, cand_df, k_top=6, k_max=12):
    out = {}
    for k in range(min(k_max, psi.shape[1] - 1)):
        col = psi[:, k + 1]
        rows = []
        for c in cand_df.columns:
            r_ = spearmanr(col, cand_df[c].to_numpy()).correlation
            rows.append((c, r_))
        rows.sort(key=lambda t: -abs(t[1]))
        out[f"psi{k+1}"] = rows[:k_top]
    return out


def main():
    files, traj = load_bank()
    n = len(files)
    names = [f.stem for f in files]
    Xtor, tlabels = torsion_features(traj)
    Fca = ca_drmsd_features(traj)
    print(f"bank {n}, torsion feats {Xtor.shape}, ca-dRMSD feats {Fca.shape}", flush=True)

    T = zscore(Xtor)
    D_tor = squareform(pdist(T))
    F = zscore(Fca)
    D_ca = squareform(pdist(F))
    print("pairwise distances done", flush=True)

    d_tw_tor, se_tw_tor, _ = twonn(D_tor)
    d_tw_ca, se_tw_ca, _ = twonn(D_ca)
    print(f"TwoNN d (torsion space): {d_tw_tor:.2f} ± {se_tw_tor:.2f}; (CA-dRMSD): {d_tw_ca:.2f} ± {se_tw_ca:.2f}")

    results = {}
    for tag, D in (("torsion", D_tor), ("ca_drmsd", D_ca)):
        off = D[np.triu_indices(D.shape[0], k=1)]
        eps_med = float(np.median(off))
        runs = {}
        for fac in (0.5, 1.0, 2.0):
            eps = eps_med * fac
            lam, psi = dmap(D ** 2 if False else D, eps)  # D are euclidean dists; K=exp(-d^2/2eps^2) via dmap(D,eps) using D2?
            runs[fac] = (lam, psi)
        results[tag] = dict(eps_med=eps_med, runs=runs)
    print("dmaps done", flush=True)

    for tag in results:
        r0 = results[tag]["runs"][1.0]
        for fac in (0.5, 2.0):
            rx = results[tag]["runs"][fac]
            r_ = abs(spearmanr(r0[1][:, 1], rx[1][:, 1]).correlation)
            results[tag].setdefault("psi1_stab", {})[f"{fac}x"] = float(r_)
            print(f"{tag}: |rho(psi1, psi1@{fac}x eps_med)| = {r_:.3f}")

    census = pd.read_csv(VALUES)
    census = census.set_index("seed").loc[names]
    cand = census.copy()

    for tag, res in results.items():
        res["spearman_top"] = spearman_tbl(res["runs"][1.0][1], cand)

    for tag, res in results.items():
        lam = res["runs"][1.0][0]
        gaps = [(i + 1, float(lam[i]), float(lam[i] - lam[i + 1])) for i in range(K_EIG)]
        res["spectrum"] = gaps
        print(f"{tag} spectrum (k, lambda, gap):", [(k, round(l, 4), round(g, 4)) for k, l, g in gaps[:8]])

    lasso_out = {}
    interp = pd.concat([census[["rg_heavy", "e2e_nc_ca", "shape_anisotropy", "acylindricity",
                                 "hairpin_closure_ratio", "contact_order_ca8", "d_y2_w9_ca",
                                 "d_d3_t8_ca", "d_p4_g7_ca", "d_e5_t8_ca", "turn3537_mean_ca",
                                 "c_bb_s4_r010_b3", "c_heavy_s4_r012_b3", "c_ca_s4_r010_b3",
                                 "sasa_total", "burial_y2", "burial_w9", "burial_d3"]],
                        census[[c for c in census.columns if c.startswith(("torpca", "restorpca", "rama"))]]], axis=1)
    Xint = zscore(interp.to_numpy())
    for tag, res in results.items():
        psi = res["runs"][1.0][1]
        for k in (1, 2, 3):
            y = psi[:, k]
            y = (y - y.mean()) / (y.std() + 1e-12)
            m = LassoCV(cv=5, n_alphas=40, max_iter=20000).fit(Xint, y)
            coefs = pd.Series(m.coef_, index=interp.columns)
            top = coefs.abs().sort_values(ascending=False).head(6)
            r2 = m.score(Xint, y)
            lasso_out[f"{tag}_psi{k}"] = {"r2": float(r2),
                                          "top": [(c, float(coefs[c])) for c in top.index if abs(coefs[c]) > 1e-9]}
            print(f"lasso {tag} psi{k}: R2={r2:.3f} top={[(c, round(coefs[c],3)) for c in top.index[:4]]}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, (tag, res) in enumerate(results.items()):
        lam, psi = res["runs"][1.0]
        ax = axes[row, 0]
        ax.plot(range(1, K_EIG + 1), lam[1:K_EIG + 1], "o-")
        ax.set_title(f"diffusion map spectrum ({tag})")
        ax.set_xlabel("k")
        ax.set_ylabel("lambda_k")
        ax.grid(alpha=.3)
        for kx, ccol, cname in ((1, census["c_bb_s4_r010_b3"], "c_bb_s4_r010_b3"),):
            ax2 = axes[row, 1]
            sc = ax2.scatter(psi[:, 1], psi[:, 2], c=ccol, s=7, cmap="viridis")
            ax2.set_xlabel("psi1")
            ax2.set_ylabel("psi2")
            ax2.set_title(f"{tag}: psi1 x psi2, color {cname}", fontsize=9)
            fig.colorbar(sc, ax=ax2, shrink=.8)
        ax3 = axes[row, 2]
        sc = ax3.scatter(psi[:, 1], psi[:, 2], c=census["restorpca_pc1"], s=7, cmap="coolwarm")
        ax3.set_xlabel("psi1")
        ax3.set_ylabel("psi2")
        ax3.set_title(f"{tag}: psi1 x psi2, color res-torsion PC1", fontsize=9)
        fig.colorbar(sc, ax=ax3, shrink=.8)
    fig.suptitle("Diffusion maps on 1,970 r7 seeds (epsilon = median heuristic)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(FIG / "cv_diffmap_spectra_maps.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for axx, D, d_est, lab in ((axes[0], D_tor, d_tw_tor, "torsion space"),
                               (axes[1], D_ca, d_tw_ca, "CA-dRMSD space")):
        Dm = D.copy()
        np.fill_diagonal(Dm, np.inf)
        srt = np.partition(Dm, 2, axis=1)
        mu = np.sort(srt[:, 1] / srt[:, 0])
        p_emp = np.arange(1, len(mu) + 1) / len(mu)
        axx.plot(np.log(mu), -np.log(1 - p_emp), ".", ms=2)
        xfit = np.log(mu[: int(0.9 * len(mu))])
        axx.plot(xfit, d_est * xfit, "-", label=f"TwoNN slope = d = {d_est:.2f}")
        axx.set_xlabel("log(mu)"); axx.set_ylabel("-log(1-P(mu))")
        axx.set_title(lab); axx.legend(fontsize=9); axx.grid(alpha=.3)
    fig.suptitle("Intrinsic dimensionality of the r7 seed bank (Facco et al. TwoNN)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "cv_twonn_id.png", dpi=150)
    plt.close(fig)

    psidf = {"seed": names}
    for tag, res in results.items():
        psi = res["runs"][1.0][1]
        for k in range(1, 7):
            psidf[f"{tag}_psi{k}"] = psi[:, k]
    pd.DataFrame(psidf).to_csv(OUT_PSI, index=False)

    doc = {
        "n_seeds": n,
        "twonn": {"torsion": {"d": d_tw_tor, "se": se_tw_tor}, "ca_drmsd": {"d": d_tw_ca, "se": se_tw_ca}},
        "spectra": {tag: results[tag]["spectrum"][:8] for tag in results},
        "psi1_stability": {tag: results[tag].get("psi1_stab", {}) for tag in results},
        "spearman_top": {tag: {k: v for k, v in results[tag]["spearman_top"].items()} for tag in results},
        "lasso": lasso_out,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# Diffusion-map / intrinsic-dimension analysis of the r7 seed bank\n"]
    lines.append(f"bank: {n} seeds; metrics: z-scored interleaved torsion sin/cos (36-D) and z-scored CA dRMSD (45-D).")
    lines.append(f"epsilon = median off-diagonal pairwise distance per metric; dmap via symmetric Markov normalization.")
    lines.append(f"\n## TwoNN intrinsic dimension")
    lines.append(f"- torsion space: d = {d_tw_tor:.2f} ± {2*se_tw_tor:.2f}")
    lines.append(f"- CA-dRMSD space: d = {d_tw_ca:.2f} ± {2*se_tw_ca:.2f}")
    lines.append("\n## Spectra (lambda_k, gap; trivial psi0 excluded)")
    for tag in results:
        sp = ", ".join(f"k={k}:{l:.3f}(-{g:.3f})" for k, l, g in results[tag]["spectrum"][:7])
        lines.append(f"- {tag}: {sp}")
    lines.append("\n## Epsilon robustness of psi1")
    for tag in results:
        stab = results[tag].get("psi1_stab", {})
        lines.append(f"- {tag}: |rho| vs 0.5x eps = {stab.get('0.5x', float('nan')):.3f}, vs 2.0x eps = {stab.get('2.0x', float('nan')):.3f}")
    lines.append("\n## Eigenvector identification (Spearman top |r| vs census CVs)")
    for tag in results:
        for kname, rows in results[tag]["spearman_top"].items():
            s = ", ".join(f"{c} {r:+.3f}" for c, r in rows[:4])
            lines.append(f"- {tag} {kname}: {s}")
    lines.append("\n## LASSO identification (interpretable features)")
    for kname, v in lasso_out.items():
        s = ", ".join(f"{c}({c_:+.2f})" for c, c_ in v["top"][:4])
        lines.append(f"- {kname}: R2={v['r2']:.3f}  {s}")
    lines.append("")
    lines.append("psi vectors per seed: genpept_r7_diffmap_psi.csv; machine-readable: genpept_r7_diffmap.json.")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_PSI, OUT_JSON)


if __name__ == "__main__":
    main()
