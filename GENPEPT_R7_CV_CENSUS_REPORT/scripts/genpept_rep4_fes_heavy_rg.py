from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R4 = BASE / "chignolin_genpept_rep4_tuned"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "FES_HEAVY_RG.md"
OUT_JSON = REPORT / "genpept_rep4_fes_heavy_rg.json"

K_B = 1.98720425864083e-3
T_K = 300.0
KT = K_B * T_K


def cu_weights(rg, e2e, cc, edges, w_tab):
    w = np.ones(len(rg))
    inside = np.ones(len(rg), bool)
    for arr, e in zip((rg, e2e, cc), edges):
        inside &= (arr >= e[0]) & (arr <= e[-1])
    bi = np.clip(np.digitize(rg, edges[0]) - 1, 0, w_tab.shape[0] - 1)
    bj = np.clip(np.digitize(e2e, edges[1]) - 1, 0, w_tab.shape[1] - 1)
    bk = np.clip(np.digitize(cc, edges[2]) - 1, 0, w_tab.shape[2] - 1)
    w[inside] = w_tab[bi[inside], bj[inside], bk[inside]]
    return w, inside


def main():
    files = sorted((R4 / "final_implicit_survivor_seeds").glob("*.pdb"))
    df = pd.read_csv(R4 / "final_survivor_seeds.csv")
    base2row = {Path(p).name: i for i, p in enumerate(df["survivor_pdb_path"])}
    row_of = np.array([base2row[f.name] for f in files])
    E = df["minimized_energy_kj_mol"].to_numpy()[row_of] / 4.184
    rg_feat = df["rg_nm"].to_numpy()[row_of] * 10.0
    e2e = df["end_to_end_nm"].to_numpy()[row_of] * 10.0
    cc = df["contact_count"].to_numpy()[row_of].astype(float)

    cu = json.loads((R4 / "coverage_uniformization.json").read_text())
    edges = [np.array(e) for e in cu["edges"]]
    w_cu, inside = cu_weights(rg_feat, e2e, cc, edges, np.array(cu["weights"]))
    w_prop = np.where(inside, 1.0 / w_cu, 1.0)
    w_thermo = w_prop * np.exp(-(E - E.min()) / KT)

    print(f"loading {len(files)} trajectories...", flush=True)
    traj = md.load(files)
    heavy = [a.index for a in traj.topology.atoms if a.element is not None and a.element.symbol != "H"]
    res_of = np.array([a.residue.index for a in traj.topology.atoms])
    harr = np.array(heavy)
    ii, jj = np.triu_indices(len(harr), k=1)
    keep = np.abs(res_of[harr[ii]] - res_of[harr[jj]]) >= 4
    ii, jj = harr[ii][keep], harr[jj][keep]
    d = np.linalg.norm(traj.xyz[:, ii] - traj.xyz[:, jj], axis=2) * 10.0
    cv_h = (0.5 * (1.0 - np.tanh(1.5 * (d - 12.0)))).mean(axis=1)
    rg = md.compute_rg(traj.atom_slice(heavy)) * 10.0

    env = json.loads((REPORT / "genpept_rep3_fes_folded.json").read_text())["native_envelope"]
    n_idx = [a.index for a in traj.topology.residue(2).atoms if a.name == "N"][0]
    o7 = [a.index for a in traj.topology.residue(6).atoms if a.name == "O"][0]
    o8 = [a.index for a in traj.topology.residue(7).atoms if a.name == "O"][0]
    d37 = np.linalg.norm(traj.xyz[:, n_idx] - traj.xyz[:, o7], axis=1) * 10.0
    d38 = np.linalg.norm(traj.xyz[:, n_idx] - traj.xyz[:, o8], axis=1) * 10.0
    in_env = ((d37 >= env["d37"][0]) & (d37 <= env["d37"][1])
              & (d38 >= env["d38"][0]) & (d38 <= env["d38"][1]))

    def fes2d(ax, x, y, w, title, n=45):
        xe = np.linspace(np.percentile(x, 0.2), np.percentile(x, 99.8), n + 1)
        ye = np.linspace(np.percentile(y, 0.2), np.percentile(y, 99.8), n + 1)
        H, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=w)
        F = np.full_like(H, np.nan)
        m = H > 0
        F[m] = -KT * np.log(H[m] / H.max())
        im = ax.pcolormesh(xe, ye, F.T, cmap="viridis_r", vmin=0, vmax=8)
        xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
        ax.contour(xc, yc, np.ma.masked_invalid(F).T, levels=(1, 2, 3, 5),
                   colors="white", linewidths=.6, alpha=.8)
        if in_env.any():
            ax.scatter(x[in_env], y[in_env], marker="*", s=70, color="red",
                       edgecolors="white", linewidths=.4, zorder=5,
                       label=f"native-envelope (n={int(in_env.sum())})")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("heavy contact fraction (r0=12 A, b3, |i-j|>=4)", fontsize=8)
        ax.set_ylabel("Rg heavy (A)", fontsize=8)
        ax.set_title(title, fontsize=9)
        return F

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.1))
    F_raw = fes2d(axes[0], cv_h, rg, np.ones_like(w_thermo), "raw pseudo-FES (search density, kcal/mol)")
    F_pr = fes2d(axes[1], cv_h, rg, w_prop, "de-biased proposal FES (1/w_CU, kcal/mol)")
    F_th = fes2d(axes[2], cv_h, rg, w_thermo, "thermodynamic starting FES (kcal/mol)")
    fig.colorbar(axes[2].collections[0], ax=axes, shrink=.85, label="F (kcal/mol)")
    fig.suptitle(f"rep4 starting FES — heavy contacts (r0=12, b3) x Rg, {len(files)} survivors, T=300 K", fontsize=11)
    fig.savefig(FIG / "cv_rep4_fes_heavy_rg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    def f1d(v, w, nbin=40):
        e = np.linspace(np.percentile(v, 0.2), np.percentile(v, 99.8), nbin + 1)
        H, _ = np.histogram(v, bins=e, weights=w)
        F = np.full_like(H, np.nan, dtype=float)
        m = H > 0
        F[m] = -KT * np.log(H[m] / H.max())
        return (e[:-1] + e[1:]) / 2, F

    xc, F1_raw = f1d(cv_h, np.ones_like(w_thermo))
    _, F1_th = f1d(cv_h, w_thermo)
    yr, F2_raw = f1d(rg, np.ones_like(w_thermo))
    _, F2_th = f1d(rg, w_thermo)

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    a1.plot(xc, F1_raw, "o-", ms=3, color="#888", label="raw")
    a1.plot(xc, F1_th, "s-", ms=3, color="#e6550d", label="thermo")
    a1.set_xlabel("heavy contact fraction (r0=12, b3)")
    a1.set_ylabel("F (kcal/mol)")
    a1.set_title("1-D F along heavy CV1", fontsize=9)
    a1.legend(fontsize=8)
    a2.plot(yr, F2_raw, "o-", ms=3, color="#888", label="raw")
    a2.plot(yr, F2_th, "s-", ms=3, color="#e6550d", label="thermo")
    a2.set_xlabel("Rg heavy (A)")
    a2.set_ylabel("F (kcal/mol)")
    a2.set_title("1-D F along Rg", fontsize=9)
    a2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(FIG / "cv_rep4_fes_heavy_rg_1d.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    m = ~np.isnan(F_raw) & ~np.isnan(F_th)
    rg_env = float(np.median(rg[in_env])) if in_env.any() else None
    cvh_env = float(np.median(cv_h[in_env])) if in_env.any() else None
    doc = {"n": int(len(files)), "T_K": T_K,
           "spearman_raw_vs_thermo": float(pd.Series(F_raw[m].ravel()).corr(pd.Series(F_th[m].ravel()), method="spearman")),
           "raw_range_kcal": float(np.nanmax(F1_raw)), "thermo_range_CVh_kcal": float(np.nanmax(F1_th)),
           "thermo_range_Rg_kcal": float(np.nanmax(F2_th)),
           "dF_CVh_min_max": [float(np.nanmin(F1_th - F1_raw)), float(np.nanmax(F1_th - F1_raw))],
           "dF_Rg_min_max": [float(np.nanmin(F2_th - F2_raw)), float(np.nanmax(F2_th - F2_raw))],
           "n_envelope": int(in_env.sum()),
           "envelope_median_rg_A": rg_env, "envelope_median_heavy_cv1": cvh_env,
           "Rg_thermo_min_at_A": float(yr[np.nanargmin(F2_th)]),
           "CVh_thermo_min_at": float(xc[np.nanargmin(F1_th)])}
    print(json.dumps(doc, indent=1))
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# rep4: starting FES — heavy contacts (r0=12, b3) x Rg\n",
             f"{len(files)} survivors, T = 300 K; weights = w_CU^-1 x exp(-(E-Emin)/kT)",
             "(recorded T1 acceptance table, GBn2 energies matched by basename).",
             "Figures: cv_rep4_fes_heavy_rg.png (3 maps), cv_rep4_fes_heavy_rg_1d.png.\n",
             f"- map agreement raw vs thermo: Spearman {doc['spearman_raw_vs_thermo']:.3f}",
             f"- thermo minimum at heavy CV1 = {doc['CVh_thermo_min_at']:.2f}, Rg = {doc['Rg_thermo_min_at_A']:.2f} A",
             f"- 1-D shifts: CV1 {doc['dF_CVh_min_max'][0]:+.1f}..{doc['dF_CVh_min_max'][1]:+.1f},",
             f"Rg {doc['dF_Rg_min_max'][0]:+.1f}..{doc['dF_Rg_min_max'][1]:+.1f} kcal/mol",
             f"- native-envelope seeds (n={doc['n_envelope']}): median Rg {rg_env:.2f} A,",
             f"median heavy CV1 {cvh_env:.2f} (red stars on the maps)",
             "\nCaveats: GBn2 single-solvent systematic ~1 kcal; not a converged PMF;\n",
             "the bank's dense coverage region defines the trustworthy area of the map.\n"]
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
