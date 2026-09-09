from __future__ import annotations

import csv
import json
import math
from collections import Counter
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
OUT_MD = REPORT / "RESIDUAL_THERM.md"
OUT_JSON = REPORT / "genpept_r7_residual_therm.json"

RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]
N_BAND_SEEDS = 24
ENERGY_TERMS = ["bond", "angle", "torsion", "nonbonded", "gb_solvation", "total"]


def binned_mean(x, y, nb=12):
    bins = np.quantile(x, np.linspace(0, 1, nb + 1))
    cm, sem, centers = [], [], []
    for i in range(nb):
        m = (x >= bins[i]) & (x < bins[i + 1] if i < nb - 1 else x <= bins[i + 1])
        if m.sum() >= 5:
            cm.append(y[m].mean())
            sem.append(y[m].std() / math.sqrt(m.sum()))
            centers.append(x[m].mean())
    return np.array(centers), np.array(cm), np.array(sem)


def main():
    files = sorted((ROOT / "final_implicit_survivor_seeds").glob("*.pdb"))
    names = [f.stem for f in files]
    census = pd.read_csv(REPORT / "genpept_r7_cv_census_values.csv").set_index("seed").loc[names]
    rows = {}
    for r in csv.DictReader((ROOT / "final_survivor_seeds.csv").open()):
        rows[Path(r["survivor_pdb_path"]).stem] = r
    seed_col = [r for r in names]
    e_min = np.array([float(rows[s]["minimized_energy_kj_mol"]) for s in names])
    e_drop = np.array([float(rows[s]["energy_drop_kj_mol"]) for s in names])

    cv2 = census["restorpca_pc1"].to_numpy()
    cv1 = census["c_bb_s4_r010_b3"].to_numpy()
    n = len(names)
    print(f"n={n}")

    traj = md.load(files)
    dssp = md.compute_dssp(traj)
    frac = {}
    for code, tag in (("H", "helix"), ("B", "bridge"), ("E", "strand"), ("C", "coil")):
        frac[tag] = (dssp == code).mean(axis=1)
    frac["other"] = 1.0 - frac["helix"] - frac["bridge"] - frac["strand"] - frac["coil"]
    print("dssp done")

    order = np.argsort(cv2)
    bins_dec = np.quantile(cv2, [0.1, 0.9])
    lo = order[: n // 10]
    hi = order[-(n // 10):]
    strings_lo = Counter("".join(r) for r in dssp[lo])
    strings_hi = Counter("".join(r) for r in dssp[hi])
    print("top DSSP low:", strings_lo.most_common(3))
    print("top DSSP high:", strings_hi.most_common(3))

    per_res_lo = (dssp[lo] == "E").mean(axis=0)
    per_res_hi = (dssp[hi] == "E").mean(axis=0)

    med_cv1 = np.median(cv1)
    band = np.where(np.abs(cv1 - med_cv1) <= 0.02)[0]
    print(f"fixed-CV1 band |CV1-{med_cv1:.3f}|<=0.02: {len(band)} seeds")
    band_sorted = band[np.argsort(cv2[band])]
    take = band_sorted[np.linspace(0, len(band_sorted) - 1, N_BAND_SEEDS).astype(int)]

    from openmm import app, unit, openmm as omm

    def energies_one(path):
        pdb = app.PDBFile(str(path))
        ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
        groups = {"bond": 1, "angle": 2, "torsion": 3, "nonbonded": 4, "gb_solvation": 5}
        for i, force in enumerate(system.getForces()):
            tname = type(force).__name__
            g = None
            if tname == "HarmonicBondForce":
                g = "bond"
            elif tname == "HarmonicAngleForce":
                g = "angle"
            elif tname == "PeriodicTorsionForce":
                g = "torsion"
            elif tname == "NonbondedForce":
                g = "nonbonded"
            elif tname == "CustomGBForce":
                g = "gb_solvation"
            if g:
                force.setForceGroup(groups[g])
        integ = omm.VerletIntegrator(0.001)
        ctx = omm.Context(system, integ)
        ctx.setPositions(pdb.positions)
        out = {}
        for g, gi in groups.items():
            st = ctx.getState(getEnergy=True, groups=1 << gi)
            out[g] = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        out["total"] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return out

    anchor = energies_one(ROOT / "final_implicit_survivor_seeds" / (names[0] + ".pdb"))
    print(f"anchor: total={anchor['total']:.2f} vs csv={float(rows[names[0]]['minimized_energy_kj_mol']):.2f}")

    bandE = []
    for idx in take:
        e = energies_one(files[idx])
        bandE.append({"seed": names[idx], "cv2": float(cv2[idx]), "cv1": float(cv1[idx]), **e})
        print(f"  {names[idx]}  cv2={cv2[idx]:+.2f}  total={e['total']:.0f}", flush=True)

    for term in ENERGY_TERMS:
        v = np.array([e[term] for e in bandE])
        r_ = spearmanr([e["cv2"] for e in bandE], v).correlation
        print(f"  rho(cv2, {term}) = {r_:+.3f}")

    sasa_t = census["sasa_total"].to_numpy()
    hyd = census["sasa_hydrophobic_frac"].to_numpy()
    bY = census["burial_y2"].to_numpy()
    bW = census["burial_w9"].to_numpy()
    bD = census["burial_d3"].to_numpy()
    corrs = {c: float(spearmanr(cv2, census[c].to_numpy()).correlation)
             for c in ["min_en:=placeholder".replace("min_en:=placeholder", "sasa_total"), "burial_y2", "burial_w9", "burial_d3", "sasa_hydrophobic_frac", "rg_heavy"]}
    corr_e = float(spearmanr(cv2, e_min).correlation)
    corr_d = float(spearmanr(cv2, e_drop).correlation)
    print(f"rho(cv2, E_min)={corr_e:+.3f}  rho(cv2, E_drop)={corr_d:+.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for tag, col in (("coil", "#888888"), ("strand", "#2c7fb8"), ("bridge", "#41ab5d"), ("helix", "#e6550d"), ("other", "#bdbdbd")):
        c_, m_, s_ = binned_mean(cv2, frac[tag])
        ax.plot(c_, m_, "o-", ms=4, label=tag, color=col)
    ax.set_xlabel("residual torsion CV2")
    ax.set_ylabel("mean DSSP fraction")
    ax.set_title("secondary structure along the residual axis (12 bins, with SEM)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3)

    ax = axes[0, 1]
    xx = np.arange(10)
    w = .4
    ax.bar(xx - w / 2, per_res_lo, width=w, color="dodgerblue", alpha=.85, label="CV2 low decile")
    ax.bar(xx + w / 2, per_res_hi, width=w, color="salmon", alpha=.85, label="CV2 high decile")
    ax.set_xticks(xx); ax.set_xticklabels(RES3, fontsize=8)
    ax.set_ylabel("fraction assigned E (strand, incl. bridge: DSSP 'E' only)")
    ax.set_title("per-residue strand content", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    ax = axes[1, 0]
    for term, col in zip(ENERGY_TERMS, ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#333333")):
        v = np.array([e[term] for e in bandE])
        ax.plot([e["cv2"] for e in bandE], v, "o-", ms=4, label=term, color=col)
    ax.set_xlabel("residual torsion CV2")
    ax.set_ylabel("energy (kJ/mol)")
    ax.set_title(f"GBn2 single-point decomposition at fixed CV1 (~{med_cv1:.2f}), {N_BAND_SEEDS} seeds", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.3)

    ax = axes[1, 1]
    for arr, lab, col in ((sasa_t, "SASA total (nm^2)", "#8856a7"), (bW, "burial W9", "#e6550d"),
                          (bY, "burial Y2", "#3182bd"), (bD, "burial D3", "#31a354")):
        c_, m_, s_ = binned_mean(cv2, arr)
        ax.plot(c_, m_, "o-", ms=4, label=lab)
    ax.set_xlabel("residual torsion CV2")
    ax.set_ylabel("value")
    ax.set_title("solvation descriptors along the axis", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.suptitle("Thermodynamic content of the residual (curl) coordinate", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(FIG / "cv2_thermo_dssp_energy.png", dpi=150)
    plt.close(fig)

    per_res_table = [{"res": RES3[i], "E_lo": float(per_res_lo[i]), "E_hi": float(per_res_hi[i])} for i in range(10)]
    doc = {
        "anchor": {"openmm_total": anchor["total"],
                   "csv_minimized": float(rows[names[0]]["minimized_energy_kj_mol"])},
        "dssp_lo_top3": strings_lo.most_common(3),
        "dssp_hi_top3": strings_hi.most_common(3),
        "per_res_strand": per_res_table,
        "band": bandE,
        "rho_cv2_energy": {term: float(spearmanr([e["cv2"] for e in bandE], [e[term] for e in bandE]).correlation) for term in ENERGY_TERMS},
        "rho_cv2_descriptors": corrs,
        "rho_cv2_e_min": corr_e,
        "rho_cv2_e_drop": corr_d,
    }
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# Thermodynamic + structural content of the residual (curl) coordinate\n"]
    lines.append(f"bank n={n}; CV2 band selection: |CV1 - {med_cv1:.3f}| <= 0.02 -> {len(band)} seeds, "
                 f"{N_BAND_SEEDS} evenly sampled across CV2.")
    lines.append(f"GBn2 total-energy anchor (seed {names[0]}): OpenMM {anchor['total']:.2f} vs generator CSV "
                 f"{float(rows[names[0]]['minimized_energy_kj_mol']):.2f} kJ/mol.")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON, FIG / "cv2_thermo_dssp_energy.png")


if __name__ == "__main__":
    main()
