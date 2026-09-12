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
from scipy.stats import spearmanr

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "SOLVATION_AXIS.md"
OUT_JSON = REPORT / "genpept_rep3_solvation_axis.json"
RNG = np.random.default_rng(43)

SOLV = ["gb_solvation", "sasa_total", "sasa_hydrophobic_frac", "burial_y2", "burial_w9", "burial_d3"]
OTHERS = ["c_bb_s4_r010_b3", "rg_heavy", "torpca_pc1", "torpca_pc2", "restorpca_pc1",
          "acylindricity", "chirality_ca", "hairpin_closure_ratio", "e2e_nc_ca", "shape_anisotropy"]


def kraskov_mi(x, y, k=3):
    from scipy.special import digamma
    from scipy.spatial import cKDTree
    n = len(x)
    zx = (x - x.mean()) / (x.std() + 1e-12)
    zy = (y - y.mean()) / (y.std() + 1e-12)
    data = np.stack([zx, zy], axis=1)
    tree = cKDTree(data, leafsize=16)
    dist, _ = tree.query(data, k + 1, p=np.inf)
    eps = dist[:, -1] + 1e-15
    tx = cKDTree(zx.reshape(-1, 1), leafsize=16)
    ty = cKDTree(zy.reshape(-1, 1), leafsize=16)
    nx = np.array([len(tx.query_ball_point(zx[i], eps[i], p=np.inf)) - 1 for i in range(n)])
    ny = np.array([len(ty.query_ball_point(zy[i], eps[i], p=np.inf)) - 1 for i in range(n)])
    return float(digamma(k) - (digamma(nx + 1) + digamma(ny + 1)).mean() + digamma(n))


def joint(x, y, nx=16, ny=8):
    xb = np.quantile(x, np.linspace(0.001, 0.999, nx + 1))
    yb = np.quantile(y, np.linspace(0.001, 0.999, ny + 1))
    H, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    occ = (H > 0).mean()
    iqr_g = np.percentile(y, 75) - np.percentile(y, 25)
    vals = []
    for i in range(nx):
        m = (x >= xb[i]) & (x < xb[i + 1] if i < nx - 1 else x <= xb[i + 1])
        if m.sum() >= 20:
            vals.append(np.percentile(y[m], 75) - np.percentile(y[m], 25))
    return float(occ), float(np.median(np.array(vals) / iqr_g)) if vals else 0.0


def main():
    census = pd.read_csv(REPORT / "genpept_rep3_cv_census_values.csv").set_index("seed")
    edf = pd.read_csv(REPORT / "genpept_rep3_energy_components.csv", index_col=0).loc[census.index]
    gb = edf["gb_solvation"].to_numpy()
    n = len(census)
    names = census.index.to_numpy()
    in_env = folds_in_env = None

    vals = {}
    vals["gb_solvation"] = gb
    for c in SOLV[1:]:
        vals[c] = census[c].to_numpy()
    for c in OTHERS:
        vals[c] = census[c].to_numpy()

    rows = []
    for s in SOLV:
        ent = np.load if False else None
        rho = {}
        for o in OTHERS:
            rho[o] = float(spearmanr(vals[s], vals[o]).correlation)
        mi_bb = kraskov_mi(vals[s], vals["c_bb_s4_r010_b3"])
        mi_pc2 = kraskov_mi(vals[s], vals["torpca_pc2"])
        occ, cond = joint(vals[s], vals["c_bb_s4_r010_b3"])
        rows.append({"axis": s, "spearman": rho, "mi_vs_cv1": mi_bb, "mi_vs_pc2": mi_pc2,
                     "occ_vs_cv1": occ, "cond_iqr_vs_cv1": cond})
        top = sorted(rho.items(), key=lambda t: -abs(t[1]))[:4]
        print(f"{s}: top |rho| {[(k, round(v,2)) for k,v in top]}; MI cv1 {mi_bb:.3f}, pc2 {mi_pc2:.3f}; occ {occ:.2f}, cond {cond:.2f}")

    native = md.load("/tmp/opencode/1UAO.pdb")
    from openmm import app, unit, openmm as omm
    tmp = Path("/tmp/opencode/_1uao_frames")
    tmp.mkdir(exist_ok=True)
    results_nat = []
    gmap = {"bond": 1, "angle": 2, "torsion": 3, "nonbonded": 4, "gb_solvation": 5}
    for m in range(native.n_frames):
        p = tmp / f"model_{m+1:02d}.pdb"
        native[m].save_pdb(str(p))
        pdb = app.PDBFile(str(p))
        ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        sys0 = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
        for force in sys0.getForces():
            g = {"HarmonicBondForce": "bond", "HarmonicAngleForce": "angle",
                 "PeriodicTorsionForce": "torsion", "NonbondedForce": "nonbonded",
                 "CustomGBForce": "gb_solvation"}.get(type(force).__name__)
            if g:
                force.setForceGroup(gmap[g])
        ctx = omm.Context(sys0, omm.VerletIntegrator(0.001))
        ctx.setPositions(pdb.positions)
        e = {t: ctx.getState(getEnergy=True, groups=1 << gi).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
             for t, gi in gmap.items()}
        e["total"] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        results_nat.append(e)
    gb_nat = np.array([e["gb_solvation"] for e in results_nat])
    print(f"1UAO GBn2 solvation: mean {gb_nat.mean():.0f} kJ/mol [{gb_nat.min():.0f}, {gb_nat.max():.0f}]")

    env = json.load(open(REPORT / "genpept_rep3_fes_folded.json"))["native_envelope"]
    files = sorted((BASE / "chignolin_genpept_rep3_massive" / "final_implicit_survivor_seeds").glob("*.pdb"))
    namesf = np.array([f.stem for f in files])
    traj = md.load(files)
    nn_ = [ [a.index for a in traj.topology.residue(2).atoms if a.name == "N"][0],
            [a.index for a in traj.topology.residue(6).atoms if a.name == "O"][0],
            [a.index for a in traj.topology.residue(7).atoms if a.name == "O"][0]]
    d37 = np.linalg.norm(traj.xyz[:, nn_[0], :] - traj.xyz[:, nn_[1], :], axis=1) * 10.0
    d38 = np.linalg.norm(traj.xyz[:, nn_[0], :] - traj.xyz[:, nn_[2], :], axis=1) * 10.0
    in_env = (d37 >= env["d37"][0]) & (d37 <= env["d37"][1]) & (d38 >= env["d38"][0]) & (d38 <= env["d38"][1])

    pctile = float((gb <= gb_nat.mean()).mean())
    print(f"bank fraction with gb_solvation <= native mean: {pctile:.4f}")
    gb_env = gb[in_env]
    gb_rest = gb[~in_env]
    print(f"gb near-native {gb_env.mean():.0f} vs rest {gb_rest.mean():.0f} (delta {gb_env.mean()-gb_rest.mean():+.1f})")

    sasa_total = vals["sasa_total"]
    native_sasa = md.shrake_rupley(native, mode="residue").sum(axis=1)
    print(f"1UAO SASA mean {native_sasa.mean():.1f} nm^2 vs bank median {np.median(sasa_total):.1f}: bank fraction <= native {float((sasa_total <= native_sasa.mean()).mean()):.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    ax = axes[0, 0]
    ax.hist(gb, bins=60, density=True, alpha=.6, color="#2c7fb8", label="bank gb_solvation")
    for v, lab in [(gb_nat.mean(), "1UAO mean"), (gb_env.mean(), "native-envelope mean")]:
        ax.axvline(v, color="red" if "1UAO" in lab else "#e6550d", ls="--", lw=1.2, label=lab)
    ax.set_xlabel("gb_solvation (kJ/mol)")
    ax.legend(fontsize=8)
    ax.set_title("solvation energy distribution vs native", fontsize=10)
    ax = axes[0, 1]
    hb_nn = ax.hexbin(vals["c_bb_s4_r010_b3"], gb, gridsize=35, cmap="viridis", mincnt=1)
    ax.set_xlabel("bb r0=10 b3 (CV1)")
    ax.set_ylabel("gb_solvation")
    ax.set_title("gb vs CV1 joint density", fontsize=9)
    fig.colorbar(hb_nn, ax=ax, shrink=.8)
    ax = axes[0, 2]
    hb2 = ax.hexbin(vals["torpca_pc2"], gb, gridsize=35, cmap="viridis", mincnt=1)
    ax.set_xlabel("plain torsion PC2")
    ax.set_ylabel("gb_solvation")
    ax.set_title("gb vs CV2 joint density", fontsize=9)
    fig.colorbar(hb2, ax=ax, shrink=.8)
    ax = axes[1, 0]
    ax.hist(sasa_total, bins=60, density=True, alpha=.6, color="#31a354", label="bank SASA")
    ax.axvline(native_sasa.mean(), color="red", ls="--", lw=1.2, label="1UAO mean")
    ax.set_xlabel("total SASA (nm^2)")
    ax.legend(fontsize=8)
    ax.set_title("surface exposure vs native", fontsize=10)
    ax = axes[1, 1]
    ax.hist(gb_rest, bins=50, density=True, alpha=.55, color="#999999", label=f"rest (n={(~in_env).sum()})")
    ax.hist(gb_env, bins=25, density=True, alpha=.7, color="#e6550d", label=f"native-envelope (n={in_env.sum()})")
    ax.axvline(gb_nat.mean(), color="red", ls="--", lw=1.2)
    ax.set_xlabel("gb_solvation (kJ/mol)")
    ax.legend(fontsize=8)
    ax.set_title("near-native gb vs rest", fontsize=10)
    ax = axes[1, 2]
    keys = ["c_bb_s4_r010_b3", "rg_heavy", "torpca_pc1", "torpca_pc2", "restorpca_pc1", "chirality_ca", "acylindricity", "e2e_nc_ca", "shape_anisotropy"]
    for s, col in zip(SOLV, ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")):
        pass
    cols_ = [(r["axis"], r["spearman"]) for r in rows]
    heat = np.array([[cols_[i][1][k] for k in keys] for i in range(len(cols_))])
    im = ax.imshow(heat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=50, fontsize=6.5)
    ax.set_yticks(range(len(cols_)))
    ax.set_yticklabels([c[0] for c in cols_], fontsize=7)
    for i in range(len(cols_)):
        for j in range(len(keys)):
            ax.text(j, i, f"{heat[i,j]:+.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if abs(heat[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_title("solvation axes vs everything else (Spearman)", fontsize=9)
    fig.suptitle("Solvation as a CV axis: orthogonality and folded-state relation (rep3, 10,019 seeds + 1UAO)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_rep3_solvation_axis.png", dpi=150)
    plt.close(fig)

    doc = {"rows": rows,
           "native": {"gb_mean": float(gb_nat.mean()), "gb_range": [float(gb_nat.min()), float(gb_nat.max())],
                      "sasa_mean": float(native_sasa.mean()),
                      "bank_frac_below_native_gb": pctile,
                      "bank_frac_below_native_sasa": float((sasa_total <= native_sasa.mean()).mean())},
           "near_native": {"n": int(in_env.sum()), "gb_mean": float(gb_env.mean()),
                           "gb_rest_mean": float(gb_rest.mean())},
           "d38_median": float(np.median(d38)), "d37_median": float(np.median(d37))}
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# Solvation as a CV axis (rep3 bank)\n",
             "## Orthogonality vs everything else (Spearman; top |rho| per solvation axis)"]
    for r in rows:
        top = sorted(r["spearman"].items(), key=lambda t: -abs(t[1]))[:4]
        lines.append(f"- {r['axis']}: " + "; ".join(f"{k} {v:+.2f}" for k, v in top)
                     + f"; MI vs CV1 {r['mi_vs_cv1']:.3f}, vs PC2 {r['mi_vs_pc2']:.3f}; "
                     + f"joint occ/cond-IQR vs CV1 {r['occ_vs_cv1']:.2f}/{r['cond_iqr_vs_cv1']:.2f}")
    lines.append("")
    lines.append("## Folded-state relation")
    lines.append(f"- 1UAO gb_solvation: {gb_nat.mean():.0f} kJ/mol [{gb_nat.min():.0f},{gb_nat.max():.0f}]; "
                 f"bank fraction with gb <= native: {pctile:.4f}")
    lines.append(f"- 1UAO SASA mean {native_sasa.mean():.1f} nm^2 vs bank median {np.median(sasa_total):.1f}; "
                 f"bank fraction <= native: {float((sasa_total <= native_sasa.mean()).mean()):.4f}")
    lines.append(f"- near-native envelope (n={in_env.sum()}): gb mean {gb_env.mean():.1f} vs rest {gb_rest.mean():.1f} "
                 f"(delta {gb_env.mean()-gb_rest.mean():+.1f} kJ/mol, permutation p~0.18 previously)")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON, FIG / "cv_rep3_solvation_axis.png")


if __name__ == "__main__":
    main()
