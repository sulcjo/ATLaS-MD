from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from openmm import app, unit, openmm

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R4 = BASE / "chignolin_genpept_rep4_tuned"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "THERMO_CONSENSUS_MODEL.md"
OUT_JSON = REPORT / "genpept_rep4_consensus.json"


def main():
    files = sorted((R4 / "final_implicit_survivor_seeds").glob("*.pdb"))
    print(f"{len(files)} survivors", flush=True)
    traj = md.load(files)
    df = pd.read_csv(R4 / "final_survivor_seeds.csv")
    base2row = {Path(p).name: i for i, p in enumerate(df["survivor_pdb_path"])}
    row_of = np.array([base2row[f.name] for f in files])
    E_stored = df["minimized_energy_kj_mol"].to_numpy()[row_of] / 4.184

    top0 = app.PDBFile(str(files[0]))
    for forcefield_name, tag in (("amber14-all.xml", "implicit/gbn2"), ("amber14-all.xml", "implicit/obc2")):
        pass

    results = {}
    systems = {}
    platform = openmm.Platform.getPlatformByName("Reference")
    for model_xml in ("implicit/gbn2.xml", "implicit/obc2.xml"):
        ff = app.ForceField("amber14-all.xml", model_xml)
        system = ff.createSystem(top0.topology, nonbondedMethod=app.NoCutoff,
                                  constraints=None, rigidWater=False)
        integ = openmm.VerletIntegrator(0.001)
        sim = app.Simulation(top0.topology, system, integ, platform)
        Es = []
        for f in files:
            pdb = app.PDBFile(str(f))
            sim.context.setPositions(pdb.positions)
            st = sim.context.getState(getEnergy=True)
            Es.append(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / 4.184)
        systems[model_xml] = np.array(Es)
        print(model_xml, "done: range", round(Es[0], 1), flush=True)

    E_gbn2, E_obc2 = systems["implicit/gbn2.xml"], systems["implicit/obc2.xml"]

    phis = md.compute_phi(traj)[1]
    psis = md.compute_psi(traj)[1]
    flank_psi = psis[:, [0, 2, 5, 7]].mean(axis=1) * 180.0 / np.pi
    BB = {"N", "CA", "C", "O"}
    idx = np.array([a.index for a in traj.topology.atoms if a.name in BB])
    res_of = np.array([a.residue.index for a in traj.topology.atoms])
    ii, jj = np.triu_indices(len(idx), k=1)
    keep = np.abs(res_of[idx[ii]] - res_of[idx[jj]]) >= 4
    ii, jj = idx[ii][keep], idx[jj][keep]
    d = np.linalg.norm(traj.xyz[:, ii] - traj.xyz[:, jj], axis=2) * 10.0
    cv1 = (0.5 * (1.0 - np.tanh(1.5 * (d - 10.0)))).mean(axis=1)

    anchor = float(np.abs(E_gbn2 - E_stored).max())
    r_models = spearmanr(E_gbn2, E_obc2).statistic
    r_curl_gbn2 = spearmanr(flank_psi, E_gbn2).statistic
    r_curl_obc2 = spearmanr(flank_psi, E_obc2).statistic
    r_cv1_gbn2 = spearmanr(cv1, E_gbn2).statistic
    r_cv1_obc2 = spearmanr(cv1, E_obc2).statistic

    q = lambda E: pd.qcut(E, 5, labels=False, duplicates="drop")
    qg, qo = q(E_gbn2), q(E_obc2)
    stable_g = qg <= 0
    stable_o = qo <= 0
    consensus = stable_g & stable_o
    doc = {
        "n": len(files),
        "anchor_gbn2_vs_stored_max_abs_kcal": anchor,
        "E_gbn2_range_kcal": [float(E_gbn2.min()), float(E_gbn2.max())],
        "E_obc2_range_kcal": [float(E_obc2.min()), float(E_obc2.max())],
        "spearman_gbn2_vs_obc2": float(r_models),
        "dE_model_range_kcal": [float((E_obc2 - E_gbn2).min()), float((E_obc2 - E_gbn2).max())],
        "dE_model_std_kcal": float((E_obc2 - E_gbn2).std()),
        "rho_flankpsi_E_gbn2": float(r_curl_gbn2),
        "rho_flankpsi_E_obc2": float(r_curl_obc2),
        "rho_cv1_E_gbn2": float(r_cv1_gbn2),
        "rho_cv1_E_obc2": float(r_cv1_obc2),
        "frac_stable_both": float(consensus.mean()),
        "frac_gbn2_stable_but_not_obc2": float((stable_g & ~stable_o).mean()),
        "frac_obc2_stable_but_not_gbn2": float((stable_o & ~stable_g).mean()),
    }
    print(json.dumps(doc, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.scatter(E_gbn2, E_obc2, s=6, alpha=.35, c=cv1, cmap="coolwarm")
    lim = [min(E_gbn2.min(), E_obc2.min()), max(E_gbn2.max(), E_obc2.max())]
    ax.plot(lim, lim, "k--", lw=.8)
    ax.set_xlabel("E GBn2 (kcal/mol)")
    ax.set_ylabel("E OBC2 (kcal/mol)")
    ax.set_title(f"consensus implicit models: Spearman {r_models:.3f}", fontsize=9)
    ax = axes[1]
    ax.scatter(flank_psi, E_gbn2, s=6, alpha=.3, color="#3182bd", label=f"GBn2 rho {r_curl_gbn2:+.3f}")
    ax.scatter(flank_psi, E_obc2, s=6, alpha=.3, color="#e6550d", label=f"OBC2 rho {r_curl_obc2:+.3f}")
    ax.set_xlabel("mean flank psi (Y2,P4,T8,W9; deg; <0 = curled)")
    ax.set_ylabel("E (kcal/mol)")
    ax.legend(fontsize=8)
    ax.set_title("curl preference under two solvent models", fontsize=9)
    ax = axes[2]
    ax.scatter(cv1, E_gbn2 - E_obc2, s=6, alpha=.3, c=flank_psi, cmap="viridis")
    ax.axhline(0, color="k", ls="--", lw=.8)
    ax.set_xlabel("bb contact CV1 (r0=10, b3)")
    ax.set_ylabel("E_GBn2 - E_OBC2 (kcal/mol)")
    ax.set_title("model disagreement vs compaction (color = flank psi)", fontsize=9)
    fig.suptitle(f"rep4 consensus-model check: {len(files)} survivors", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG / "cv_rep4_consensus_model.png", dpi=150)
    plt.close(fig)

    lines = ["# rep4: consensus implicit-solvent model check\n",
             f"All {len(files)} survivors re-scored under GBn2 (anchor vs stored minimized energy:",
             f"max |diff| = {anchor:.1f} kcal/mol) and OBC2 single points.\n",
             f"- cross-model agreement: Spearman(E) = **{r_models:.3f}**;",
             f"model gap E_OBC2 - E_GBn2 std = {doc['dE_model_std_kcal']:.1f} kcal/mol",
             f"- curl preference (rho flank-psi vs E): GBn2 {r_curl_gbn2:+.3f} vs OBC2 {r_curl_obc2:+.3f}",
             f"- compaction preference (rho CV1 vs E): GBn2 {r_cv1_gbn2:+.3f} vs OBC2 {r_cv1_obc2:+.3f}",
             f"- most-stable-quintile overlap: {doc['frac_stable_both']*100:.1f}% of survivors are stable",
             f"in both models; {doc['frac_gbn2_stable_but_not_obc2']*100:.1f}% stable only under GBn2;",
             f"{doc['frac_obc2_stable_but_not_gbn2']*100:.1f}% only under OBC2",
             "\nReading: where the two solvent models disagree, single-model energy claims",
             "(survivor ranking, curl bias, basin depths in THERMO_STARTING_FES.md) carry a",
             "model systematic of the size of the disagreement. Treat consensus-unstable",
             "survivors as solvent-model-sensitive.\n"]
    OUT_JSON.write_text(json.dumps(doc, indent=1))
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
