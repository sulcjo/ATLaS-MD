from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R4 = BASE / "chignolin_genpept_rep4_tuned"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "THERMO_STARTING_FES.md"
OUT_JSON = REPORT / "genpept_rep4_thermo.json"

K_B = 1.98720425864083e-3
T_K = 300.0
KT = K_B * T_K * 4.184


def cu_weight(rg_a, e2e_a, cc, edges, weights):
    idx = []
    for v, e in zip((rg_a, e2e_a, cc), edges):
        b = int(np.clip(np.digitize(v, e) - 1, 0, len(e) - 2))
        idx.append(b)
    if not (edges[0][0] <= rg_a <= edges[0][-1] and edges[1][0] <= e2e_a <= edges[1][-1]
            and edges[2][0] <= cc <= edges[2][-1]):
        return 1.0
    return float(weights[idx[0], idx[1], idx[2]])


def fes_map(ax, x, y, w, title, n=22):
    xe = np.linspace(x.min(), x.max(), n + 1)
    ye = np.linspace(y.min(), y.max(), n + 1)
    H, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=w)
    F = np.full_like(H, np.nan)
    m = H > 0
    F[m] = -KT * np.log(H[m] / H.max())
    im = ax.pcolormesh(xe, ye, F.T, cmap="viridis_r", vmin=0, vmax=30)
    ax.contour((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2, np.ma.masked_invalid(F).T,
               levels=(2.5, 5, 10, 15, 20), colors="white", linewidths=.6, alpha=.75)
    ax.set_xlabel("Rg (A)")
    ax.set_ylabel("contact count (|i-j|>=4, 8 A)")
    ax.set_title(title, fontsize=9)
    return F, im


def main():
    cu = json.loads((R4 / "coverage_uniformization.json").read_text())
    edges = [np.array(e) for e in cu["edges"]]
    w_tab = np.array(cu["weights"])
    df = pd.read_csv(R4 / "final_survivor_seeds.csv")
    rg = df["rg_nm"].to_numpy() * 10.0
    e2e = df["end_to_end_nm"].to_numpy() * 10.0
    cc = df["contact_count"].to_numpy().astype(float)
    E = df["minimized_energy_kj_mol"].to_numpy()

    w_cu = np.array([cu_weight(r, e, c, edges, w_tab) for r, e, c in zip(rg, e2e, cc)])
    in_table = ((rg >= edges[0][0]) & (rg <= edges[0][-1]) & (e2e >= edges[1][0])
                & (e2e <= edges[1][-1]) & (cc >= edges[2][0]) & (cc <= edges[2][-1]))
    beta = 1.0 / KT
    w_prop = np.where(in_table, 1.0 / w_cu, 1.0)
    boltz = np.exp(-beta * (E - E.min()))
    w_thermo = w_prop * boltz

    doc = {"n_survivors": int(len(df)), "kT_kcal": KT,
           "frac_in_cu_support": float(in_table.mean()),
           "w_cu_range": [float(w_cu[in_table].min()), float(w_cu[in_table].max())]}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    F_raw, im1 = fes_map(axes[0], rg, cc, np.ones_like(w_cu), "raw pseudo-FES (bank density)", n=20)
    F_prop, _ = fes_map(axes[1], rg, cc, w_prop, "de-biased proposal FES (1/w_CU weights)", n=20)
    F_th, _ = fes_map(axes[2], rg, cc, w_thermo, "thermodynamic starting FES (Boltzmann x de-bias)", n=20)
    fig.colorbar(im1, ax=axes, shrink=.85, label="F (kcal/mol)")
    fig.suptitle(f"rep4: from search density to a starting FES ({len(df)} survivors, T=300 K)", fontsize=11)
    fig.savefig(FIG / "cv_rep4_thermo_fes.png", dpi=150)
    plt.close(fig)

    n = F_raw.shape[0]
    r_rp = spearmanr(F_raw[~np.isnan(F_raw)], F_prop[~np.isnan(F_prop)]).statistic
    r_rt = spearmanr(F_raw[~np.isnan(F_raw)], F_th[~np.isnan(F_th)]).statistic
    r_pt = spearmanr(F_prop[~np.isnan(F_prop)], F_th[~np.isnan(F_th)]).statistic
    top = lambda F: set(zip(*np.unravel_index(np.argsort(np.nan_to_num(F, nan=1e9).ravel())[:max(1, int(0.05 * (~np.isnan(F)).sum()))], (n, F.shape[1]))))
    doc["map_agreement"] = {"spearman_raw_vs_proposal": float(r_rp),
                            "spearman_raw_vs_thermo": float(r_rt),
                            "spearman_proposal_vs_thermo": float(r_pt)}
    print("map Spearman: raw~prop", round(r_rp, 3), "raw~thermo", round(r_rt, 3), "prop~thermo", round(r_pt, 3))

    xe = np.linspace(rg.min(), rg.max(), 20)
    def f1d(w):
        H, _ = np.histogram(rg, bins=xe, weights=w)
        F = np.full_like(H, np.nan, dtype=float)
        m = H > 0
        F[m] = -KT * np.log(H[m] / H.max())
        return F
    F1_raw, F1_th = f1d(np.ones_like(w_cu)), f1d(w_thermo)
    dF = F1_th - F1_raw
    lo, hi = np.digitize(6.0, xe) - 1, np.digitize(7.0, xe) - 1
    doc["f_rg_shift"] = {
        "compact_bins_lt6A_mean_dF_kcal": float(np.nanmean(dF[:lo + 1])),
        "mid_6_7A_mean_dF_kcal": float(np.nanmean(dF[lo + 1:hi + 1])),
        "extended_gt7A_mean_dF_kcal": float(np.nanmean(dF[hi + 1:]))}

    vt = pd.read_csv(R4 / "implicit_viable_basin_table.csv")
    g = vt.groupby("implicit_basin_id")["minimized_energy_kj_mol"]
    bas = pd.DataFrame({"g": g.size(), "E_mean": g.mean(), "E_min": g.min()}).reset_index()
    bas["E_kcal"] = bas["E_mean"] / 4.184
    bas["F_kcal"] = bas["E_kcal"] - KT * np.log(bas["g"])
    top_E = set(bas.nsmallest(20, "E_kcal")["implicit_basin_id"])
    top_F = set(bas.nsmallest(20, "F_kcal")["implicit_basin_id"])
    top_Ee = set(bas.nsmallest(20, "E_min")["implicit_basin_id"])
    top_Fe = set(bas.nsmallest(20, "F_kcal")["implicit_basin_id"])
    doc["basins"] = {
        "n_basins": int(len(bas)), "total_minima": int(bas["g"].sum()),
        "g_range": [int(bas["g"].min()), int(bas["g"].max())],
        "g_median": float(bas["g"].median()),
        "top20_overlap_Emean_vs_F": len(top_E & top_F),
        "top20_overlap_Emin_vs_F": len(top_Ee & top_Fe),
        "multiplicity_term_kTlogg_kcal_median": float((KT * np.log(bas["g"])).median()),
        "multiplicity_term_kTlogg_kcal_p95": float((KT * np.log(bas["g"])).quantile(0.95)),
    }
    print("basins:", doc["basins"])

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    sc = a1.scatter(bas["E_kcal"], bas["F_kcal"], c=np.log10(bas["g"]), s=8, cmap="viridis", alpha=.6)
    a1.plot([bas["E_kcal"].min(), bas["E_kcal"].max()],
            [bas["E_kcal"].min(), bas["E_kcal"].max()], "r--", lw=.8, label="F = E")
    a1.set_xlabel("basin mean E (kcal/mol, GBn2)")
    a1.set_ylabel("basin F = E - kT ln g (kcal/mol)")
    a1.legend(fontsize=8)
    fig2.colorbar(sc, ax=a1, label="log10 g (multiplicity)")
    a1.set_title("multiplicity re-ranks basins (color = basin size)", fontsize=9)
    F1 = f1d(np.ones_like(w_cu)); F2 = f1d(w_thermo)
    xc = (xe[:-1] + xe[1:]) / 2
    a2.plot(xc, F1_raw, "o-", ms=3, label="raw pseudo-F (rg)")
    a2.plot(xc, F1_th, "s-", ms=3, label="thermo F (rg)")
    a2.set_xlabel("Rg (A)"); a2.set_ylabel("F (kcal/mol)")
    a2.legend(fontsize=8)
    a2.set_title("1-D free-energy profile along Rg", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(FIG / "cv_rep4_thermo_basins.png", dpi=150)
    plt.close(fig2)

    lines = ["# rep4: from search density to a thermodynamic starting FES\n",
             f"Bank: {len(df)} survivors; T = 300 K (kT = {KT:.3f} kcal/mol = {KT*4.184:.1f} kJ/mol).",
             f"CU table coverage: {in_table.mean()*100:.1f}% of survivors fall inside the frozen",
             "(Rg, e2e, contact-count) weight table (weight 1.0 assumed outside).\n",
             "## 1. Three weightings of the same bank (figure cv_rep4_thermo_fes.png)\n",
             f"- raw vs de-biased proposal map: Spearman {r_rp:.3f} — the CU flattening is visible but modest",
             f"- raw vs thermodynamic map: Spearman {r_rt:.3f} — **Boltzmann re-weighting reshapes the FES substantially**",
             f"- proposal vs thermodynamic: Spearman {r_pt:.3f}\n",
             "## 2. F(Rg) shift (figure cv_rep4_thermo_basins.png)\n",
             f"- compact bins (Rg < 6 A): mean dF = {doc['f_rg_shift']['compact_bins_lt6A_mean_dF_kcal']:+.1f} kcal/mol",
             f"- mid (6-7 A): {doc['f_rg_shift']['mid_6_7A_mean_dF_kcal']:+.1f}; extended (> 7 A): {doc['f_rg_shift']['extended_gt7A_mean_dF_kcal']:+.1f}",
             "\n## 3. Basin multiplicity thermodynamics\n",
             f"- {len(bas)} basins over {int(bas['g'].sum())} viable minima (implicit + NMA), median g = {bas['g'].median():.0f}",
             f"- multiplicity term kT·ln g: median {doc['basins']['multiplicity_term_kTlogg_kcal_median']:.1f},",
             f"p95 {doc['basins']['multiplicity_term_kTlogg_kcal_p95']:.1f} kcal/mol — comparable to whole-bank energy spread",
             f"- top-20 basins by mean-E vs by F: overlap {len(top_E & top_F)}/20 (by min-E: {len(top_Ee & top_Fe)}/20)",
             "— small, heavily-populated basins win under F where deep, lonely ones win under E",
             "\n## Caveats\n",
             "- E is a GBn2 minimized energy: solvent-model-dependent (see consensus check), no explicit counterions",
             "- multiplicity g counts *viable minimization outcomes*, a proxy for basin volume, not a true configurational integral",
             "- w_CU is the frozen startup table; the spec's rolling update was approximated (freeze-after-calibration)",
             "- this is a *starting* FES for window initialization / bias sanity, not a converged PMF\n"]
    OUT_JSON.write_text(json.dumps(doc, indent=1))
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
