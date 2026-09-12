from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openmm import app, unit, openmm

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
R4 = BASE / "chignolin_genpept_rep4_tuned"
NATIVE = Path("/tmp/opencode/1UAO.pdb")
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "GUIDED_EXPANSION.md"
OUT_JSON = REPORT / "genpept_rep4_guided_expansion.json"

N_SEEDS = 30
N_STEPS = 8
LAMBDA_KCAL = 1.5
T_BH = 675.0
KT_BH = 1.98720425864083e-3 * T_BH
MD_STEPS = 350
FRICTION = 5.0
DT_PS = 0.0035
MIN_ITERS = 300


def hbond_indices(top):
    n_d3 = [a.index for a in top.residue(2).atoms if a.name == "N"][0]
    o_g7 = [a.index for a in top.residue(6).atoms if a.name == "O"][0]
    o_t8 = [a.index for a in top.residue(7).atoms if a.name == "O"][0]
    return n_d3, o_g7, o_t8


def register_count(top, pos_nm):
    n_idx = [a.index for a in top.atoms if a.name == "N"]
    o_idx = [a.index for a in top.atoms if a.name == "O"]
    res_of = {a.index: a.residue.index for a in top.atoms}
    cnt = 0
    for ni in n_idx:
        for oi in o_idx:
            if abs(res_of[ni] - res_of[oi]) >= 3:
                d = float(np.linalg.norm(pos_nm[ni] - pos_nm[oi])) * 10.0
                if d < 3.8:
                    cnt += 1
    return cnt


def main():
    native = md.load(NATIVE)
    env = json.loads((REPORT / "genpept_rep3_fes_folded.json").read_text())["native_envelope"]
    files = sorted((R4 / "final_implicit_survivor_seeds").glob("*.pdb"))
    print(f"loading {len(files)} survivors...", flush=True)
    traj = md.load(files)
    ca = [a.index for a in traj.topology.atoms if a.name == "CA"]
    ca_ref = [a.index for a in native.topology.atoms if a.name == "CA"]
    rmsd_stack = np.stack([md.rmsd(traj, native, frame=m, atom_indices=ca, ref_atom_indices=ca_ref) for m in range(native.n_frames)], axis=1) * 10.0
    rmsd_min = rmsd_stack.min(axis=1)
    order = np.argsort(rmsd_min)
    sel = order[:N_SEEDS]
    print("selected RMSDs:", [round(rmsd_min[i], 2) for i in sel], flush=True)

    top0 = traj.topology
    n_idx, o7_idx, o8_idx = hbond_indices(top0)
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(top0.to_openmm(), nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False)
    platform = openmm.Platform.getPlatformByName("OpenCL")

    def score_state(sim, pos):
        sim.context.setPositions(pos)
        sim.context.setVelocitiesToTemperature(T_BH)
        sim.step(MD_STEPS)
        st = sim.context.getState(getPositions=True, getEnergy=True)
        pos_kick = st.getPositions(asNumpy=True)
        sim.minimizeEnergy(maxIterations=MIN_ITERS)
        st2 = sim.context.getState(getPositions=True, getEnergy=True)
        pos_min = st2.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        e = st2.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        nreg = register_count(top0, pos_min)
        d37 = float(np.linalg.norm(pos_min[n_idx] - pos_min[o7_idx])) * 10.0
        d38 = float(np.linalg.norm(pos_min[n_idx] - pos_min[o8_idx])) * 10.0
        return pos_min, e, nreg, d37, d38

    def rmsd_of(pos_nm):
        t = md.Trajectory(np.array(pos_nm)[None, ...], top0)
        return float(md.rmsd(t, native, 0, atom_indices=ca, ref_atom_indices=ca_ref)[0] * 10.0)

    results = {"control": [], "rewarded": []}
    rng_master = np.random.default_rng(20260910)
    for arm, lam in (("control", 0.0), ("rewarded", LAMBDA_KCAL)):
        print(f"=== arm {arm} (lambda={lam}) ===", flush=True)
        for k, si in enumerate(sel):
            sim = app.Simulation(top0.to_openmm(), system,
                                 openmm.LangevinMiddleIntegrator(T_BH * unit.kelvin, FRICTION / unit.picosecond,
                                                                 DT_PS * unit.picoseconds), platform)
            sim.integrator.setRandomNumberSeed(int(rng_master.integers(1, 2**30)))
            pos = np.array(traj.xyz[si])
            sim.context.setPositions(pos)
            sim.minimizeEnergy(maxIterations=MIN_ITERS)
            st = sim.context.getState(getPositions=True, getEnergy=True)
            pos_cur = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            e_cur = st.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
            nreg_cur = register_count(top0, pos_cur)
            f_cur = e_cur - lam * nreg_cur
            trace = [{"step": 0, "E": e_cur, "nreg": nreg_cur, "rmsd": rmsd_of(pos_cur)}]
            for step in range(1, N_STEPS + 1):
                pos_new, e_new, nreg_new, d37, d38 = score_state(sim, pos_cur)
                f_new = e_new - lam * nreg_new
                if f_new < f_cur or rng_master.random() < np.exp(-(f_new - f_cur) / KT_BH):
                    pos_cur, e_cur, nreg_cur, f_cur = pos_new, e_new, nreg_new, f_new
                trace.append({"step": step, "E": e_cur, "nreg": nreg_cur, "rmsd": rmsd_of(pos_cur)})
            d37 = float(np.linalg.norm(pos_cur[n_idx] - pos_cur[o7_idx])) * 10.0
            d38 = float(np.linalg.norm(pos_cur[n_idx] - pos_cur[o8_idx])) * 10.0
            in_env = (env["d37"][0] <= d37 <= env["d37"][1]) and (env["d38"][0] <= d38 <= env["d38"][1])
            results[arm].append({"start_rmsd": float(rmsd_min[si]), "end_rmsd": trace[-1]["rmsd"],
                                 "start_nreg": trace[0]["nreg"], "end_nreg": trace[-1]["nreg"],
                                 "end_E": trace[-1]["E"], "in_envelope": bool(in_env), "d37": d37, "d38": d38,
                                 "trace": trace})
            print(f"  seed {k}: rmsd {rmsd_min[si]:.2f} -> {trace[-1]['rmsd']:.2f}, "
                  f"nreg {trace[0]['nreg']} -> {trace[-1]['nreg']}, env={in_env}", flush=True)

    doc = {arm: {"n": len(v),
                 "rmsd_start_mean": float(np.mean([r["start_rmsd"] for r in v])),
                 "rmsd_end_mean": float(np.mean([r["end_rmsd"] for r in v])),
                 "rmsd_end_min": float(np.min([r["end_rmsd"] for r in v])),
                 "nreg_start_mean": float(np.mean([r["start_nreg"] for r in v])),
                 "nreg_end_mean": float(np.mean([r["end_nreg"] for r in v])),
                 "n_in_envelope": int(sum(r["in_envelope"] for r in v))}
            for arm, v in results.items()}
    print(json.dumps(doc, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, key in ((axes[0], "rmsd"), (axes[1], "nreg")):
        for arm, col in (("control", "#888888"), ("rewarded", "#e6550d")):
            for r in results[arm]:
                ax.plot([t["step"] for t in r["trace"]], [t[key] for t in r["trace"]],
                        color=col, alpha=.35, lw=.8)
        ax.set_xlabel("BH step")
        ax.set_ylabel(key)
        ax.set_title(f"{key}: grey = control (lambda 0), orange = register-rewarded (lambda {LAMBDA_KCAL})", fontsize=9)
    fig.suptitle(f"guided register-rewarded BH expansion, {N_SEEDS} near-native rep4 seeds, {N_STEPS} steps", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "cv_rep4_guided_expansion.png", dpi=150)
    plt.close(fig)

    lines = ["# Guided register-rewarded BH expansion (demo on rep4 near-native tail)\n",
             f"{N_SEEDS} best-RMSD rep4 survivors; {N_STEPS} BH steps each (675 K MD kick {MD_STEPS} x {DT_PS} ps",
             f"+ {MIN_ITERS}-iter GBn2 minimization); Metropolis on F = E - lambda * n_register",
             f"(backbone N-O < 3.8 A, |i-j| >= 3); lambda = 0 (control) vs {LAMBDA_KCAL} kcal per register.\n",
             f"- control: mean RMSD {doc['control']['rmsd_start_mean']:.2f} -> {doc['control']['rmsd_end_mean']:.2f} A,",
             f"  mean register count {doc['control']['nreg_start_mean']:.1f} -> {doc['control']['nreg_end_mean']:.1f},",
             f"  envelope seeds {doc['control']['n_in_envelope']}/{N_SEEDS}",
             f"- rewarded: mean RMSD {doc['rewarded']['rmsd_start_mean']:.2f} -> {doc['rewarded']['rmsd_end_mean']:.2f} A,",
             f"  mean register count {doc['rewarded']['nreg_start_mean']:.1f} -> {doc['rewarded']['nreg_end_mean']:.1f},",
             f"  envelope seeds {doc['rewarded']['n_in_envelope']}/{N_SEEDS}",
             "\nFigure: cv_rep4_guided_expansion.png. This is the expansion-side lever the rep4 verdict",
             "pointed at, tested without touching GENPEPT.py.\n"]
    OUT_JSON.write_text(json.dumps(doc, indent=1))
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
