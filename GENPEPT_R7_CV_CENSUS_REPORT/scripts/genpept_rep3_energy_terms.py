from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "REP3_ENERGY_TERMS.md"
OUT_JSON = REPORT / "genpept_rep3_energy_terms.json"
RNG = np.random.default_rng(41)

TERMS = ["bond", "angle", "torsion", "nonbonded", "gb_solvation", "total"]
NLABEL = {"bond": "bond (strain)", "angle": "angle (strain)", "torsion": "torsion (strain)",
          "nonbonded": "direct nonbonded", "gb_solvation": "GB solvation", "total": "total GBn2"}


def bin_stats(x, y, nx=14):
    edges = np.quantile(x, np.linspace(0, 1, nx + 1))
    centers, means, sems, cnt = [], [], [], []
    for i in range(nx):
        m = (x >= edges[i]) & (x < edges[i + 1] if i < nx - 1 else x <= edges[i + 1])
        if m.sum() >= 20:
            centers.append(edges[i] * 0.5 + edges[i + 1] * 0.5)
            means.append(y[m].mean())
            sems.append(y[m].std() / math.sqrt(m.sum()))
            cnt.append(int(m.sum()))
    return np.array(centers), np.array(means), np.array(sems)


def slope_ci(x, y, B=300):
    def s_():
        return float(np.polyfit(x, y, 1)[0])
    vals = []
    n = len(x)
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        vals.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
    return s_(), (float(np.quantile(vals, .025)), float(np.quantile(vals, .975)))


def mean_map(x, y, z, nx=60, nmin=20):
    xe = np.quantile(x, np.linspace(0.001, 0.999, nx + 1))
    ye = np.quantile(y, np.linspace(0.001, 0.999, nx + 1))
    Hc, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    Hz, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=z)
    M = np.full_like(Hc, np.nan, dtype=float)
    m = Hc >= nmin
    M[m] = Hz[m] / Hc[m]
    F = np.full_like(Hc, np.nan, dtype=float)
    mc = Hc > 0
    F[mc] = -np.log(Hc[mc] / Hc.max())
    return M, F, xe, ye, Hc


def main():
    census = pd.read_csv(REPORT / "genpept_rep3_cv_census_values.csv").set_index("seed")
    edf = pd.read_csv(REPORT / "genpept_rep3_energy_components.csv", index_col=0).loc[census.index]
    n = len(census)
    print(f"bank {n}")
    cv1 = census["c_bb_s4_r010_b3"].to_numpy()
    cv2 = census["torpca_pc2"].to_numpy()
    pc1 = census["torpca_pc1"].to_numpy()
    rg = census["rg_heavy"].to_numpy() * 10.0
    e2e = census["e2e_nc_ca"].to_numpy()

    adv = json.load(open(REPORT / "genpept_rep3_advanced.json"))
    states = adv["kde_states"]["states"]
    state_of = np.full(n, -1)
    for i, st_ in enumerate(states):
        m = (cv2 >= st_["lo"]) & (cv2 <= st_["hi"])
        state_of[m] = i
    print("state sizes:", [(i, int((state_of == i).sum())) for i in range(len(states))])

    folds = json.load(open(REPORT / "genpept_rep3_fes_folded.json"))
    near_native = set()
    j = folds["banks"]["rep3_massive"]
    # recompute envelope membership exactly
    import mdtraj as md
    env = folds["native_envelope"]
    files = sorted((BASE / "chignolin_genpept_rep3_massive" / "final_implicit_survivor_seeds").glob("*.pdb"))
    traj = md.load(files)
    def hb(top):
        n_ = [a.index for a in top.residue(2).atoms if a.name == "N"][0]
        o7 = [a.index for a in top.residue(6).atoms if a.name == "O"][0]
        o8 = [a.index for a in top.residue(7).atoms if a.name == "O"][0]
        return n_, o7, o8
    nn_ = hb(traj.topology)
    d37 = np.linalg.norm(traj.xyz[:, nn_[0], :] - traj.xyz[:, nn_[1], :], axis=1) * 10.0
    d38 = np.linalg.norm(traj.xyz[:, nn_[0], :] - traj.xyz[:, nn_[2], :], axis=1) * 10.0
    in_env = (d37 >= env["d37"][0]) & (d37 <= env["d37"][1]) & (d38 >= env["d38"][0]) & (d38 <= env["d38"][1])
    env_names = np.array(census.index)[in_env]
    print(f"native-envelope seeds: {in_env.sum()}")

    map_defs = [("CV1: bb r0=10 b3", cv1, "CV2: plain torsion PC2", cv2),
                ("Rg (A)", rg, "E2E N-C10 CA (A)", e2e),
                ("CV1: bb r0=10 b3", cv1, "plain torsion PC1", pc1)]
    fig, axes = plt.subplots(len(map_defs), len(TERMS), figsize=(20, 3.3 * len(map_defs)), constrained_layout=True)
    for row, (xl, x, yl, y) in enumerate(map_defs):
        for col, t in enumerate(TERMS):
            ax = axes[row, col]
            M, F, xe, ye, H = mean_map(x, y, edf[t].to_numpy())
            lo, hi = np.nanquantile(M, [0.02, 0.98])
            im = ax.pcolormesh(xe, ye, M.T, cmap="RdYlBu_r", vmin=lo, vmax=hi)
            xc = (xe[:-1] + xe[1:]) / 2
            yc = (ye[:-1] + ye[1:]) / 2
            ax.contour(xc, yc, np.ma.masked_invalid(F).T, levels=(1, 2, 3, 4), colors="black",
                       linewidths=.45, alpha=.35)
            med = np.nanmedian(M)
            ax.set_title(f"{NLABEL[t]} | med {med:.0f}", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=.75, pad=.01)
            if col == 0:
                ax.set_ylabel(yl, fontsize=8)
            if row == len(map_defs) - 1:
                ax.set_xlabel(xl, fontsize=8)
    fig.suptitle("mean GBn2 energy-term values on the FES maps (bins n>=20; black = pseudo-FES contours of the bank density)",
                 fontsize=11)
    fig.savefig(FIG / "cv_rep3_energy_term_maps.png", dpi=150)
    plt.close(fig)

    trends = {}
    axes_def = [("cv1", cv1), ("cv2", cv2), ("rg", rg), ("e2e", e2e)]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (aname, x) in zip(axes.ravel(), axes_def):
        for t, col in zip(TERMS, ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#333333")):
            c_, m_, s_ = bin_stats(x, edf[t].to_numpy())
            ax.plot(c_, m_, "o-", ms=3.5, color=col, label=NLABEL[t])
            ax.fill_between(c_, m_ - s_, m_ + s_, color=col, alpha=.12)
        ax.set_xlabel(aname)
        ax.set_ylabel("mean energy (kJ/mol)")
        ax.set_title(f"per-term trends along {aname} (binned mean +/- SEM)", fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=.3)
    fig.suptitle("GBn2 energy-term trends along the map coordinates (rep3, 10,019 seeds)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_rep3_energy_term_trends.png", dpi=150)
    plt.close(fig)
    for aname, x in axes_def:
        for t in TERMS:
            s_, (lo, hi) = slope_ci(x, edf[t].to_numpy())
            trends[f"{t}_vs_{aname}"] = {"slope_kj_per_unit": s_, "ci95": [lo, hi]}
        print(f"{aname}: " + "; ".join(f"{t} {trends[f'{t}_vs_{aname}']['slope_kj_per_unit']:+.1f}" for t in TERMS))

    X = edf[TERMS].to_numpy().T
    C = np.corrcoef(X)
    cov = np.cov(X)
    var_diag = np.diag(cov)
    var_total = var_diag[-1]
    part_sum = var_diag[:-1].sum()
    comp_coeff = 1.0 - var_total / part_sum

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    ax = axes[0, 0]
    im = ax.imshow(C[:5, :5], cmap="RdBu_r", vmin=-1, vmax=1)
    lab = TERMS[:5]
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(lab, rotation=45, fontsize=8)
    ax.set_yticklabels(lab, fontsize=8)
    for i in range(5):
        for jj in range(5):
            ax.text(jj, i, f"{C[i,jj]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(C[i, jj]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_title("partial-term correlation matrix (total excluded)", fontsize=10)

    ax = axes[0, 1]
    ax.scatter(edf["nonbonded"], edf["gb_solvation"], s=1.2, alpha=.15, color="#2c7fb8")
    r_ = np.corrcoef(edf["nonbonded"], edf["gb_solvation"])[0, 1]
    ax.set_xlabel("direct nonbonded (kJ/mol)")
    ax.set_ylabel("GB solvation (kJ/mol)")
    ax.set_title(f"compensation: rho = {r_:+.3f}", fontsize=9)

    ax = axes[1, 0]
    means = []
    for i in range(len(states)):
        m = state_of == i
        means.append([float(edf[t].to_numpy()[m].mean() if m.sum() else np.nan) for t in TERMS])
    means = np.array(means)
    med_bank = np.array([float(edf[t].median()) for t in TERMS])
    xpos = np.arange(len(TERMS))
    offsets = np.linspace(-0.3, 0.3, len(states))
    for i in range(len(states)):
        ax.bar(xpos + offsets[i], (means[i] - med_bank) [np.arange(len(TERMS))], width=0.6 / len(states),
               label=f"S{i} (n={(state_of == i).sum()})", alpha=.85)
    ax.set_xticks(xpos)
    ax.set_xticklabels(TERMS, fontsize=9)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("state mean - bank median (kJ/mol)")
    ax.set_title("per-term composition of the 3 KDE macro-states", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[1, 1]
    rest = edf[~in_env]
    envE = edf[in_env]
    deltas = []
    for t in TERMS:
        d = float(envE[t].mean() - rest[t].mean())
        perm = [float(rest[t].to_numpy()[RNG.permutation(len(rest))[:len(envE)]].mean() - envE[t].mean()) if False else 0.0][0]
        deltas.append(d)
    perm_p = {}
    for t in TERMS:
        obs = float(envE[t].mean() - rest[t].mean())
        cnt = 0
        B = 500
        for _ in range(B):
            fake = rest[t].to_numpy()[RNG.integers(0, len(rest), len(envE))].mean()
            if abs(fake - rest[t].mean()) >= abs(obs):
                cnt += 1
        perm_p[t] = (cnt + 1) / (B + 1)
    ax.bar(np.arange(len(TERMS)), deltas, color=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#333333"])
    for i, t in enumerate(TERMS):
        if perm_p[t] <= 0.002:
            ax.text(i, deltas[i] + (6 if deltas[i] >= 0 else -14), "**", ha="center", fontsize=14)
        elif perm_p[t] <= 0.05:
            ax.text(i, deltas[i] + (6 if deltas[i] >= 0 else -14), "*", ha="center", fontsize=14)
    ax.set_xticks(np.arange(len(TERMS)))
    ax.set_xticklabels(TERMS, fontsize=9)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("near-native mean - rest mean (kJ/mol)")
    ax.set_title(f"energy composition of the {in_env.sum()} native-envelope seeds (** = permutation p<0.002)", fontsize=9.5)
    ax.grid(alpha=.3, axis="y")
    fig.suptitle("energy compensation, state composition, and near-native energy signature", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_rep3_energy_compensation.png", dpi=150)
    plt.close(fig)

    print("deltas near-native:", {t: round(d, 1) for t, d in zip(TERMS, deltas)})
    print("perm p:", {t: round(perm_p[t], 4) for t in TERMS})

    result = {
        "trend_slopes": trends,
        "corr_partial_terms": {TERMS[i]: {TERMS[j]: float(C[i, j]) for j in range(5)} for i in range(5)},
        "compensation_coeff": float(comp_coeff),
        "state_means_minus_median": {f"S{i}": {t: float(means[i][j] - med_bank[j]) for j, t in enumerate(TERMS)}
                                     for i in range(len(states))},
        "near_native": {"n": int(in_env.sum()),
                        "delta_mean_minus_rest": {t: float(d) for t, d in zip(TERMS, deltas)},
                        "perm_p": perm_p},
    }
    OUT_JSON.write_text(json.dumps(result, indent=1))

    lines = ["# GBn2 energy-term behavior on the rep3 FES maps\n",
             f"bank n={n}; terms: bond/angle/torsion strain, direct nonbonded, GB solvation, total.\n",
             "## Trend slopes (kJ/mol per CV unit, bootstrap CI95)"]
    for aname, _ in axes_def:
        lines.append(f"\n[{aname}]")
        for t in TERMS:
            v = trends[f"{t}_vs_{aname}"]
            lines.append(f"- {t}: {v['slope_kj_per_unit']:+.1f} [{v['ci95'][0]:+.1f},{v['ci95'][1]:+.1f}]")
    lines.append(f"\n## Compensation\n- partial-term correlation (nonbonded vs gb_solvation): {C[3,4]:+.3f}")
    lines.append(f"- variance compensation coefficient 1 - Var(total)/sum Var(parts): {comp_coeff:.3f}")
    lines.append("\n## State composition (state mean - bank median, kJ/mol)")
    for i in range(len(states)):
        m = means[i] - med_bank
        lines.append(f"- S{i}: " + "; ".join(f"{t} {m[j]:+.1f}" for j, t in enumerate(TERMS)))
    lines.append("\n## Near-native (native envelope) energy signature vs rest")
    lines.append(f"- n={in_env.sum()}; " + "; ".join(f"{t} {d:+.1f} (p={perm_p[t]:.3f})" for t, d in zip(TERMS, deltas)))
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON, FIG / "cv_rep3_energy_term_maps.png", FIG / "cv_rep3_energy_term_trends.png", FIG / "cv_rep3_energy_compensation.png")


if __name__ == "__main__":
    main()
