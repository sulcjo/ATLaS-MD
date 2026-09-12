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
OUT_MD = REPORT / "FES_STARTING.md"
OUT_JSON = REPORT / "genpept_rep4_fes_starting.json"

K_B = 1.98720425864083e-3
T_K = 300.0
KT = K_B * T_K
BB = {"N", "CA", "C", "O"}


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
    rg = df["rg_nm"].to_numpy()[row_of] * 10.0
    e2e = df["end_to_end_nm"].to_numpy()[row_of] * 10.0
    cc = df["contact_count"].to_numpy()[row_of].astype(float)
    print(f"{len(files)} survivors, energies matched by basename", flush=True)

    cu = json.loads((R4 / "coverage_uniformization.json").read_text())
    edges = [np.array(e) for e in cu["edges"]]
    w_cu, inside = cu_weights(rg, e2e, cc, edges, np.array(cu["weights"]))
    w_prop = np.where(inside, 1.0 / w_cu, 1.0)
    boltz = np.exp(-(E - E.min()) / KT)
    w_thermo = w_prop * boltz

    print("loading trajectories...", flush=True)
    traj = md.load(files)
    idx = np.array([a.index for a in traj.topology.atoms if a.name in BB])
    res_of = np.array([a.residue.index for a in traj.topology.atoms])
    ii, jj = np.triu_indices(len(idx), k=1)
    keep = np.abs(res_of[idx[ii]] - res_of[idx[jj]]) >= 4
    ii, jj = idx[ii][keep], idx[jj][keep]
    d = np.linalg.norm(traj.xyz[:, ii] - traj.xyz[:, jj], axis=2) * 10.0
    cv1 = (0.5 * (1.0 - np.tanh(1.5 * (d - 10.0)))).mean(axis=1)

    phis = md.compute_phi(traj)[1]
    psis = md.compute_psi(traj)[1]
    cols = []
    for arr in (phis, psis):
        for k in range(arr.shape[1]):
            cols += [np.sin(arr[:, k]), np.cos(arr[:, k])]
    X = np.stack(cols, axis=1)
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = Xc @ Vt[0]
    pc2 = Xc @ Vt[1]

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
                       label=f"native-envelope seeds (n={int(in_env.sum())})")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("CV1: bb contact fraction (r0=10 A, b3, |i-j|>=4)", fontsize=8)
        ax.set_ylabel("torsion PC1", fontsize=8)
        ax.set_title(title, fontsize=9)
        return F

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.1))
    F_raw = fes2d(axes[0], cv1, pc1, np.ones_like(w_thermo),
                  "raw pseudo-FES (search density, kcal/mol)")
    F_th = fes2d(axes[2], cv1, pc1, w_thermo,
                 "thermodynamic starting FES (de-biased + Boltzmann, kcal/mol)")
    F_pr = fes2d(axes[1], cv1, pc1, w_prop,
                 "de-biased proposal FES (1/w_CU, kcal/mol)")
    fig.colorbar(axes[2].collections[0], ax=axes, shrink=.85, label="F (kcal/mol)")
    fig.suptitle(f"rep4 starting FES on the working CV axes — {len(files)} survivors, T=300 K", fontsize=11)
    fig.savefig(FIG / "cv_rep4_fes_starting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    xe = np.linspace(cv1.min(), cv1.max(), 40)
    def f1d(w):
        H, _ = np.histogram(cv1, bins=xe, weights=w)
        F = np.full_like(H, np.nan, dtype=float)
        m = H > 0
        F[m] = -KT * np.log(H[m] / H.max())
        return F, H
    F1_raw, H1 = f1d(np.ones_like(w_thermo))
    F1_th, H2 = f1d(w_thermo)

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    xc = (xe[:-1] + xe[1:]) / 2
    a1.plot(xc, F1_raw, "o-", ms=3, color="#888", label="raw search density")
    a1.plot(xc, F1_th, "s-", ms=3, color="#e6550d", label="thermodynamic (de-bias + Boltzmann)")
    a1.set_xlabel("CV1 (bb contact fraction)")
    a1.set_ylabel("F (kcal/mol)")
    a1.set_title("1-D free-energy profile along CV1", fontsize=9)
    a1.legend(fontsize=8)
    fm = ~np.isnan(F1_raw) & ~np.isnan(F1_th)
    a2.plot(xc[fm], F1_th[fm] - F1_raw[fm], "d-", ms=4, color="#2c5f8a")
    a2.axhline(0, color="k", ls="--", lw=.8)
    a2.set_xlabel("CV1 (bb contact fraction)")
    a2.set_ylabel("F_thermo - F_raw (kcal/mol)")
    a2.set_title("where thermodynamics moves the landscape", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(FIG / "cv_rep4_fes_starting_1d.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    m = ~np.isnan(F_raw) & ~np.isnan(F_th)
    doc = {
        "n": int(len(files)), "T_K": T_K,
        "spearman_raw_vs_thermo_CV1xPC1": float(pd.Series(F_raw[m].ravel()).corr(pd.Series(F_th[m].ravel()), method="spearman")),
        "dF_max_CV1_1d_kcal": float(np.nanmax(F1_th - F1_raw)),
        "dF_min_CV1_1d_kcal": float(np.nanmin(F1_th - F1_raw)),
        "raw_range_kcal": float(np.nanmax(F1_raw)), "thermo_range_kcal": float(np.nanmax(F1_th)),
        "n_envelope_seeds": int(in_env.sum()),
        "envelope_F_thermo_kcal": float(np.nanpercentile(np.interp(cv1[in_env], xc, F1_th), [50])[0]) if in_env.any() else None,
        "frac_w_thermo_top10pct_from_topE_quintile": float(
            pd.Series(w_thermo).rank(pct=True).ge(0.9).astype(int)
            .corr(pd.Series(E).rank(pct=True).le(0.2).astype(int), method="spearman")),
    }
    print(json.dumps(doc, indent=1))
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# rep4: the starting FES on the working CV axes\n",
             f"{len(files)} survivors, T = 300 K. Weights: w = w_CU^-1 (recorded acceptance table) x",
             "exp(-(E-E_min)/kT) (GBn2 minimized energies, matched by basename).",
             "Maps on CV1 (bb r0=10 b3) x plain torsion PC1, 45x45 grid, red stars = native-envelope",
             f"seeds (n={int(in_env.sum())}). Figures: cv_rep4_fes_starting.png (3 maps),",
             "cv_rep4_fes_starting_1d.png (1-D profile + delta).\n",
             f"- map agreement raw vs thermo (CV1 x PC1): Spearman {doc['spearman_raw_vs_thermo_CV1xPC1']:.3f}",
             f"- 1-D along CV1: thermo shifts bins by {doc['dF_min_CV1_1d_kcal']:+.1f} to {doc['dF_max_CV1_1d_kcal']:+.1f} kcal/mol",
             f"- landscape depth: raw {doc['raw_range_kcal']:.1f} vs thermo {doc['thermo_range_kcal']:.1f} kcal/mol",
             "\nUsage: initialize GaREUS window centers/widths from the thermo map, use raw map",
             "only as a coverage diagnostic. Caveats: GBn2 single-solvent systematic ~1 kcal",
             "(consensus check), multiplicity not yet folded in per-seed (see THERMO_STARTING_FES.md",
             "for the basin-level version), not a converged PMF.\n"]
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
