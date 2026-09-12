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
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "FES_STARTING_REP5.md"
OUT_JSON = REPORT / "genpept_rep5_fes_starting.json"

K_B = 1.98720425864083e-3
T_K = 300.0
KT = K_B * T_K
ENV = ((5.666, 8.063), (2.336, 4.720))


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


def load_bank(bank):
    files = sorted((bank / "final_implicit_survivor_seeds").glob("*.pdb"))
    df = pd.read_csv(bank / "final_survivor_seeds.csv")
    base2row = {Path(p).name: i for i, p in enumerate(df["survivor_pdb_path"])}
    row_of = np.array([base2row[f.name] for f in files])
    E = df["minimized_energy_kj_mol"].to_numpy()[row_of] / 4.184
    rg_feat = df["rg_nm"].to_numpy()[row_of] * 10.0
    e2e = df["end_to_end_nm"].to_numpy()[row_of] * 10.0
    cc = df["contact_count"].to_numpy()[row_of].astype(float)
    cu = json.loads((bank / "coverage_uniformization.json").read_text())
    edges = [np.array(e) for e in cu["edges"]]
    w_cu, inside = cu_weights(rg_feat, e2e, cc, edges, np.array(cu["weights"]))
    w_prop = np.where(inside, 1.0 / w_cu, 1.0)
    w_thermo = w_prop * np.exp(-(E - E.min()) / KT)
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
    n3 = [a.index for a in traj.topology.residue(2).atoms if a.name == "N"][0]
    o7 = [a.index for a in traj.topology.residue(6).atoms if a.name == "O"][0]
    o8 = [a.index for a in traj.topology.residue(7).atoms if a.name == "O"][0]
    d37 = np.linalg.norm(traj.xyz[:, n3] - traj.xyz[:, o7], axis=1) * 10.0
    d38 = np.linalg.norm(traj.xyz[:, n3] - traj.xyz[:, o8], axis=1) * 10.0
    in_env = ((d37 >= ENV[0][0]) & (d37 <= ENV[0][1]) & (d38 >= ENV[1][0]) & (d38 <= ENV[1][1]))
    return {"files": files, "w": w_thermo, "w_prop": w_prop, "E": E,
            "cv_h": cv_h, "rg": rg, "in_env": in_env}


def fes_grid(x, y, w, xe, ye):
    H, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=w)
    F = np.full_like(H, np.nan, dtype=float)
    m = H > 0
    F[m] = -KT * np.log(H[m] / H.max())
    return F


def main():
    rep4 = load_bank(BASE / "chignolin_genpept_rep4_tuned")
    rep5 = load_bank(BASE / "chignolin_genpept_rep5_t7")
    print(f"rep4 {len(rep4['files'])} | rep5 {len(rep5['files'])}", flush=True)

    x4, y4, x5, y5 = rep4["cv_h"], rep4["rg"], rep5["cv_h"], rep5["rg"]
    xe = np.linspace(min(x4.min(), x5.min()), max(np.percentile(x4, 99.8), np.percentile(x5, 99.8)), 46)
    ye = np.linspace(min(y4.min(), y5.min()), max(np.percentile(y4, 99.8), np.percentile(y5, 99.8)), 46)

    F4 = fes_grid(x4, y4, rep4["w"], xe, ye)
    F5 = fes_grid(x5, y5, rep5["w"], xe, ye)
    F5_raw = fes_grid(x5, y5, np.ones(len(x5)), xe, ye)
    F5_prop = fes_grid(x5, y5, rep5["w_prop"], xe, ye)

    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.1))
    for ax, F, title in ((axes[0], F5_raw, "raw pseudo-FES (search density, kcal/mol)"),
                         (axes[1], F5_prop, "de-biased proposal FES (1/w_CU, kcal/mol)"),
                         (axes[2], F5, "thermodynamic starting FES (kcal/mol)")):
        ax.pcolormesh(xe, ye, F.T, cmap="viridis_r", vmin=0, vmax=8)
        ax.contour(xc, yc, np.ma.masked_invalid(F).T, levels=(1, 2, 3, 5),
                   colors="white", linewidths=.6, alpha=.8)
        if rep5["in_env"].any():
            ax.scatter(x5[rep5["in_env"]], y5[rep5["in_env"]], marker="*", s=70, color="red",
                       edgecolors="white", linewidths=.4, zorder=5,
                       label=f"native-envelope (n={int(rep5['in_env'].sum())})")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("heavy contact fraction (r0=12 A, b3, |i-j|>=4)", fontsize=8)
        ax.set_ylabel("Rg heavy (A)", fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.colorbar(axes[2].collections[0], ax=axes, shrink=.85, label="F (kcal/mol)")
    fig.suptitle(f"rep5 (T7 restraint-forming kicks) starting FES — heavy contacts x Rg, "
                 f"{len(rep5['files'])} survivors, T=300 K", fontsize=11)
    fig.savefig(FIG / "cv_rep5_fes_heavy_rg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    def f1d(v, w, nbin=40):
        e = np.linspace(np.percentile(v, 0.2), np.percentile(v, 99.8), nbin + 1)
        H, _ = np.histogram(v, bins=e, weights=w)
        F = np.full_like(H, np.nan, dtype=float)
        m = H > 0
        F[m] = -KT * np.log(H[m] / H.max())
        return (e[:-1] + e[1:]) / 2, F

    xv4, F14 = f1d(x4, rep4["w"])
    xv5, F15 = f1d(x5, rep5["w"])
    yr4, F24 = f1d(y4, rep4["w"])
    yr5, F25 = f1d(y5, rep5["w"])

    fig2, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.4))
    a1.plot(xv4, F14, "o-", ms=3, color="#888", label="rep4 thermo")
    a1.plot(xv5, F15, "s-", ms=3, color="#e6550d", label="rep5 (T7) thermo")
    a1.set_xlabel("heavy contact fraction (r0=12, b3)"); a1.set_ylabel("F (kcal/mol)")
    a1.set_title("1-D F along heavy CV1", fontsize=9); a1.legend(fontsize=8)
    a2.plot(yr4, F24, "o-", ms=3, color="#888", label="rep4 thermo")
    a2.plot(yr5, F25, "s-", ms=3, color="#e6550d", label="rep5 (T7) thermo")
    a2.set_xlabel("Rg heavy (A)"); a2.set_ylabel("F (kcal/mol)")
    a2.set_title("1-D F along Rg", fontsize=9); a2.legend(fontsize=8)
    dF = F5 - F4
    im = a3.pcolormesh(xe, ye, dF.T, cmap="coolwarm", vmin=-3, vmax=3)
    a3.set_xlabel("heavy contact fraction"); a3.set_ylabel("Rg heavy (A)")
    a3.set_title("thermo FES difference: rep5 (T7) - rep4", fontsize=9)
    fig2.colorbar(im, ax=a3, label="dF (kcal/mol)")
    if rep5["in_env"].any():
        a3.scatter(x5[rep5["in_env"]], y5[rep5["in_env"]], marker="*", s=60, color="red",
                   edgecolors="black", linewidths=.4, zorder=5)
    fig2.tight_layout()
    fig2.savefig(FIG / "cv_rep5_fes_vs_rep4.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    m = ~np.isnan(F5_raw) & ~np.isnan(F5)
    m4 = ~np.isnan(F4) & ~np.isnan(F5)
    med_env_rg = float(np.median(y5[rep5["in_env"]]))
    med_env_cv = float(np.median(x5[rep5["in_env"]]))
    bi = np.clip(np.digitize(x5, xe) - 1, 0, F5.shape[0] - 1)
    bj = np.clip(np.digitize(y5, ye) - 1, 0, F5.shape[1] - 1)
    env_F = float(np.nanmin(F5[bi, bj][rep5["in_env"]]))
    doc = {"n_rep5": int(len(rep5['files'])), "n_rep4": int(len(rep4['files'])), "T_K": T_K,
           "spearman_raw_vs_thermo_rep5": float(pd.Series(F5_raw[m].ravel()).corr(pd.Series(F5[m].ravel()), method="spearman")),
           "thermo_min_at_CVh": float(xv5[np.nanargmin(F15)]), "thermo_min_at_Rg_A": float(yr5[np.nanargmin(F25)]),
           "thermo_range_CVh_kcal": float(np.nanmax(F15)), "thermo_range_Rg_kcal": float(np.nanmax(F25)),
           "dF_map_min_max_kcal": [float(np.nanmin(dF)), float(np.nanmax(dF))],
           "n_envelope_rep5": int(rep5["in_env"].sum()),
           "envelope_median_rg_A": med_env_rg, "envelope_median_heavy_cv1": med_env_cv,
           "envelope_min_F_kcal": env_F}
    print(json.dumps(doc, indent=1))
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# rep5 (T7): starting FES — heavy contacts (r0=12, b3) x Rg\n",
             f"{len(rep5['files'])} rep5 survivors vs {len(rep4['files'])} rep4; T = 300 K;",
             "weights = w_CU^-1 x exp(-(E-E_min)/kT) (each bank's own CU table + GBn2 energies).\n",
             f"- rep5 raw-vs-thermo map agreement: Spearman {doc['spearman_raw_vs_thermo_rep5']:.3f}",
             f"- rep5 thermo minimum at heavy CV1 = {doc['thermo_min_at_CVh']:.2f}, Rg = {doc['thermo_min_at_Rg_A']:.2f} A",
             f"- map difference (rep5 - rep4): {doc['dF_map_min_max_kcal'][0]:+.1f} .. {doc['dF_map_min_max_kcal'][1]:+.1f} kcal/mol",
             f"- native-envelope seeds (n={doc['n_envelope_rep5']}): median Rg {med_env_rg:.2f} A,",
             f"  median heavy CV1 {med_env_cv:.2f}; lowest thermo F on their bins {env_F:.2f} kcal/mol",
             "\nFigures: cv_rep5_fes_heavy_rg.png (3 maps), cv_rep5_fes_vs_rep4.png (1-D + dF map).\n",
             "Caveats: GBn2 single-solvent systematic ~1 kcal; not a converged PMF; the T7 A/B",
             "carries a small GPU-noise floor at the BH stage (see REP5_T7.md).\n"]
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
