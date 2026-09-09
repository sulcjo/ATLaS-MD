from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, gaussian_kde
from scipy.signal import find_peaks

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
ROOT = BASE / "RUNS" / "chignolin_genpept_r7"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "MI_STATES_ENERGY.md"
OUT_JSON = REPORT / "genpept_r7_mi_states_energy.json"
OUT_ENERGIES = REPORT / "genpept_r7_energy_components.csv"

RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]


def kraskov_mi(x, y, k=3):
    from scipy.spatial import cKDTree
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    zx = (x - x.mean()) / x.std()
    zy = (y - y.mean()) / y.std()
    data = np.stack([zx, zy], axis=1)
    tree = cKDTree(data, leafsize=16)
    dist, _ = tree.query(data, k + 1, p=np.inf)
    eps = dist[:, -1] + 1e-15
    tx = cKDTree(zx.reshape(-1, 1), leafsize=16)
    ty = cKDTree(zy.reshape(-1, 1), leafsize=16)
    nx = np.empty(n)
    ny = np.empty(n)
    for i in range(n):
        nx[i] = len(tx.query_ball_point(zx[i], eps[i], p=np.inf)) - 1
        ny[i] = len(ty.query_ball_point(zy[i], eps[i], p=np.inf)) - 1
    from scipy.special import digamma
    return float(digamma(k) - (digamma(nx + 1) + digamma(ny + 1)).mean() + digamma(n))


def entropy_norm(v, n_bins=20):
    v = v[np.isfinite(v)]
    h, _ = np.histogram(v, bins=n_bins, range=(v.min(), v.max()))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(n_bins))


def bc(v):
    from scipy import stats
    n = len(v)
    g1 = stats.skew(v)
    g2 = stats.kurtosis(v)
    den = g2 + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return float((g1 ** 2 + 1) / den) if den > 0 else float("nan")


def main():
    files = sorted((ROOT / "final_implicit_survivor_seeds").glob("*.pdb"))
    names = [f.stem for f in files]
    census = pd.read_csv(REPORT / "genpept_r7_cv_census_values.csv").set_index("seed").loc[names]
    n = len(names)
    traj = md.load(files)

    cv1n = "c_bb_s4_r010_b3"
    cv1 = census[cv1n].to_numpy()
    cv2_cands = ["restorpca_pc1", "torpca_pc2", "torpca_pc1", "acylindricity", "hairpin_closure_ratio",
                 "chirality_ca", "pack_y2", "burial_d3", "turn3537_mean_ca", "contact_order_ca8"]
    mi_rows = []
    rng = np.random.default_rng(19)
    for c in cv2_cands:
        y = census[c].to_numpy()
        mi = kraskov_mi(cv1, y)
        mi_null = np.mean([kraskov_mi(cv1, y[rng.permutation(n)]) for _ in range(5)])
        rs = float(spearmanr(cv1, y).correlation)
        mi_rows.append({"cv2": c, "mi_nats": mi, "mi_null_mean": mi_null, "mi_excess": mi - mi_null,
                        "spearman": rs})
        print(f"MI({cv1n},{c}) = {mi:.3f} (null {mi_null:.3f}), excess {mi - mi_null:+.3f}, rho {rs:+.3f}")

    cv2 = census["restorpca_pc1"].to_numpy()
    xs = np.linspace(cv2.min(), cv2.max(), 400)
    kde = gaussian_kde(cv2, bw_method=0.06)
    dens = kde(xs)
    peaks, props = find_peaks(dens, prominence=0.04 * dens.max())
    mins, _ = find_peaks(-dens, prominence=0.04 * dens.max())
    bounds = np.concatenate([[xs[0]], xs[mins], [xs[-1]]])
    states = []
    for i in range(len(bounds) - 1):
        m = (cv2 >= bounds[i]) & (cv2 <= bounds[i + 1])
        if m.sum() < 15:
            continue
        pk_in = peaks[(xs[peaks] > bounds[i]) & (xs[peaks] <= bounds[i + 1])]
        mode = float(xs[pk_in[np.argmax(dens[pk_in])]]) if len(pk_in) else float((bounds[i] + bounds[i + 1]) / 2)
        states.append({"lo": float(bounds[i]), "hi": float(bounds[i + 1]),
                       "mode_cv2": mode, "n": int(m.sum()), "frac": float(m.mean())})
    print(f"{len(states)} KDE states:", [(f"[{s['lo']:.2f},{s['hi']:.2f}]", s["frac"]) for s in states])

    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    psi_res = {int(traj.topology.atom(int(psi_idx[k][1])).residue.index): psis[:, k] for k in range(psis.shape[1])}
    for i, s in enumerate(states):
        m = (cv2 >= s["lo"]) & (cv2 <= s["hi"])
        s["psi_means"] = {f"psi {RES3[k[0]]}": float(np.degrees(np.angle(np.exp(1j * k[1][m]).mean())))
                          for k in ((1, psi_res[1]), (2, psi_res[2]), (3, psi_res[3]), (6, psi_res[6]), (7, psi_res[7]))}
        rep = names[np.where(m)[0][np.argmin(np.abs(cv2[m] - s["mode_cv2"]))]]
        s["representative"] = rep
        print(f"state {i}: rep {rep}, psi means {s['psi_means']}")

    from openmm import app, unit, openmm as omm
    def energies_one(path):
        pdb = app.PDBFile(str(path))
        ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
        system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
        groups = {"bond": 1, "angle": 2, "torsion": 3, "nonbonded": 4, "gb_solvation": 5}
        for force in system.getForces():
            tname = type(force).__name__
            g = {"HarmonicBondForce": "bond", "HarmonicAngleForce": "angle",
                 "PeriodicTorsionForce": "torsion", "NonbondedForce": "nonbonded",
                 "CustomGBForce": "gb_solvation"}.get(tname)
            if g:
                force.setForceGroup(groups[g])
        integ = omm.VerletIntegrator(0.001)
        ctx = omm.Context(system, integ)
        ctx.setPositions(pdb.positions)
        out = {}
        for g, gi in groups.items():
            out[g] = ctx.getState(getEnergy=True, groups=1 << gi).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        out["total"] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return out

    if OUT_ENERGIES.exists():
        edf = pd.read_csv(OUT_ENERGIES).set_index("seed").loc[names].reset_index()
        print("energies loaded from cache", flush=True)
    else:
        erows = []
        for i, f in enumerate(files):
            erows.append({"seed": names[i], **energies_one(f)})
            if i % 400 == 0:
                print(f"  energies {i}/{n}", flush=True)
        edf = pd.DataFrame(erows)
        edf.to_csv(OUT_ENERGIES, index=False)

    estats = {}
    for t in ("bond", "angle", "torsion", "nonbonded", "gb_solvation", "total"):
        v = edf[t].to_numpy()
        estats[t] = {"entropy": entropy_norm(v), "BC": bc(v),
                     "rho_cv1": float(spearmanr(v, cv1).correlation),
                     "rho_cv2": float(spearmanr(v, cv2).correlation),
                     "std": float(v.std()), "p5": float(np.quantile(v, .05)), "p95": float(np.quantile(v, .95))}
        print(f"{t}: ent {estats[t]['entropy']:.3f}, BC {estats[t]['BC']:.3f}, rho_cv1 {estats[t]['rho_cv1']:+.2f}, rho_cv2 {estats[t]['rho_cv2']:+.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax = axes[0, 0]
    ax.plot(xs, dens, color="#2c7fb8")
    ax.fill_between(xs, dens, color="#a6cee3", alpha=.5)
    for i, s in enumerate(states):
        ax.axvline((s["lo"] + s["hi"]) / 2 if i == 0 else s["lo"], color="#d95f02", ls="--", lw=.8)
        ax.text(s["mode_cv2"], dens.max() * (1.0 - 0.08 * i), f"S{i}\n{s['frac']*100:.0f}%", ha="center", fontsize=8)
    ax.set_xlabel("residual torsion CV2")
    ax.set_ylabel("KDE density")
    ax.set_title(f"discrete intermediates along the curl axis: {len(states)} states", fontsize=10)

    ax = axes[0, 1]
    st_labels = [f"S{i}" for i in range(len(states))]
    residues = ["psi Y2", "psi D3", "psi P4", "psi G7", "psi T8"]
    wm = np.array([[s["psi_means"][r] for r in residues] for s in states])
    im = ax.imshow(wm, cmap="RdBu_r", vmin=-180, vmax=180, aspect="auto")
    ax.set_xticks(range(len(residues))); ax.set_xticklabels(residues)
    ax.set_yticks(range(len(st_labels))); ax.set_yticklabels(st_labels)
    ax.set_title("circular-mean psi (deg) per state", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=.8)

    ax = axes[1, 0]
    for t, col in zip(("bond", "angle", "torsion", "nonbonded", "gb_solvation", "total"),
                      ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#333333")):
        ax.scatter(cv2, edf[t].to_numpy(), s=1.5, alpha=.25, label=t, color=col)
    ax.set_xlabel("residual torsion CV2")
    ax.set_ylabel("energy kJ/mol")
    ax.set_title("full-bank GBn2 components vs CV2 (1,970 seeds)", fontsize=10)
    ax.legend(fontsize=7, markerscale=6, ncol=2)

    ax = axes[1, 1]
    mi_df = pd.DataFrame(mi_rows)
    xpos = np.arange(len(mi_df))
    ax.bar(xpos - .2, mi_df.mi_nats, width=.4, label="MI (nats)", color="#3182bd")
    ax.bar(xpos + .2, mi_df.mi_null_mean, width=.4, label="permutation null", color="#cccccc")
    ax.set_xticks(xpos); ax.set_xticklabels(mi_df.cv2, rotation=60, fontsize=7)
    ax.set_title("mutual information CV1 x CV2 (Kraskov, k=3)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "cv_mi_states_energy.png", dpi=150)
    plt.close(fig)

    doc = {"mi": mi_rows, "kde_states": states, "energy_component_stats": estats}
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# Mutual information, discrete intermediate states, full-bank energy decomposition\n"]
    lines.append("## MI (Kraskov k=3, nats) of c_bb_s4_r010_b3 with CV2 candidates")
    for r in mi_rows:
        lines.append(f"- {r['cv2']}: MI {r['mi_nats']:.3f} (null {r['mi_null_mean']:.3f}), "
                     f"excess {r['mi_excess']:+.3f}, rho {r['spearman']:+.3f}")
    lines.append("\n## KDE states along residual CV2")
    for i, s in enumerate(states):
        lines.append(f"- S{i} [{s['lo']:.2f},{s['hi']:.2f}] n={s['n']} ({s['frac']*100:.1f}%), rep={s['representative']}, "
                     f"psi means {s['psi_means']}")
    lines.append("\n## GBn2 component stats (1,970 seeds)")
    for t, v in estats.items():
        lines.append(f"- {t}: entropy {v['entropy']:.3f}, BC {v['BC']:.3f}, rho_cv1 {v['rho_cv1']:+.2f}, rho_cv2 {v['rho_cv2']:+.2f}, "
                     f"std {v['std']:.1f}, p5–p95 [{v['p5']:.0f},{v['p95']:.0f}]")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON, OUT_ENERGIES)


if __name__ == "__main__":
    main()
