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
from scipy.stats import gaussian_kde, spearmanr, hmean
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
BANK = BASE / "chignolin_genpept_rep3_massive"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OUT_MD = REPORT / "REP3_ADVANCED.md"
OUT_JSON = REPORT / "genpept_rep3_advanced.json"
R7VALUES = REPORT / "genpept_r7_cv_census_values.csv"
RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]
RNG = np.random.default_rng(31)


def zscore(A):
    return (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)


def dmap(D, eps):
    K = np.exp(-(D ** 2) / (2.0 * eps * eps))
    np.fill_diagonal(K, 0.0)
    q = K.sum(axis=1)
    Dm12 = np.diag(1.0 / np.sqrt(q))
    S = Dm12 @ K @ Dm12
    evals, evecs = np.linalg.eigh(S)
    order = np.argsort(-evals)
    return evals[order], (Dm12 @ evecs[:, order])


def twonn_d(Dsub):
    Dm = Dsub.copy()
    np.fill_diagonal(Dm, np.inf)
    srt = np.partition(Dm, 2, axis=1)
    mu = np.sort(srt[:, 1] / srt[:, 0])
    p = np.arange(1, len(mu) + 1) / len(mu)
    keep = int(0.9 * len(mu))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - p[:keep])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope)


def fish_lee(a, b):
    ai = a - np.angle(np.exp(1j * a).mean())
    bi = b - np.angle(np.exp(1j * b).mean())
    si, sj = np.sin(ai), np.sin(bi)
    den = math.sqrt((si * si).sum() * (sj * sj).sum())
    return float((si * sj).sum() / den) if den > 0 else float("nan")


def circ_R(theta, m):
    return float(np.abs(np.exp(1j * m * theta).mean()))


def kraskov_mi(x, y, k=3):
    from scipy.special import digamma
    n = len(x)
    data = np.stack([x, y], axis=1)
    tree = cKDTree(data, leafsize=16)
    dist, _ = tree.query(data, k + 1, p=np.inf)
    eps = dist[:, -1] + 1e-15
    tx = cKDTree(x.reshape(-1, 1), leafsize=16)
    ty = cKDTree(y.reshape(-1, 1), leafsize=16)
    nx = np.array([len(tx.query_ball_point(x[i], eps[i], p=np.inf)) - 1 for i in range(n)])
    ny = np.array([len(ty.query_ball_point(y[i], eps[i], p=np.inf)) - 1 for i in range(n)])
    return float(digamma(k) - (digamma(nx + 1) + digamma(ny + 1)).mean() + digamma(n))


def main():
    files = sorted((BANK / "final_implicit_survivor_seeds").glob("*.pdb"))
    names = [f.stem for f in files]
    n = len(files)
    census = pd.read_csv(REPORT / "genpept_rep3_cv_census_values.csv").set_index("seed").loc[names]
    print(f"bank {n}", flush=True)
    traj = md.load(files)

    phis = md.compute_phi(traj)[1]
    psis = md.compute_psi(traj)[1]
    cols = []
    for arr in (phis, psis):
        for k in range(arr.shape[1]):
            cols += [np.sin(arr[:, k]), np.cos(arr[:, k])]
    X = np.stack(cols, axis=1)
    T = zscore(X)
    ca_idx = np.array([a.index for a in traj.topology.atoms if a.name == "CA"])
    xyz = traj.atom_slice(ca_idx).xyz
    iu = np.triu_indices(len(ca_idx), k=1)
    Fca = zscore(np.linalg.norm(xyz[:, iu[0]] - xyz[:, iu[1]], axis=2))
    print("features done", flush=True)

    sub = RNG.choice(n, min(3000, n), replace=False)
    D_tor_sub = squareform(pdist(T[sub]))
    D_ca_sub = squareform(pdist(Fca[sub]))
    tw = {"torsion_3k": twonn_d(D_tor_sub), "ca_drmsd_3k": twonn_d(D_ca_sub)}
    print(f"TwoNN on 3k sub: {tw}")

    dmap_out = {}
    for tag, Dsub in (("torsion", D_tor_sub), ("ca_drmsd", D_ca_sub)):
        off = Dsub[np.triu_indices(len(sub), k=1)]
        eps = float(np.median(off))
        evals, evecs = dmap(Dsub, eps)
        dmap_out[tag] = {"eps": eps, "lams": evals[:8].tolist()}
        print(f"dmap {tag}: lams {[round(float(l),4) for l in evals[:6]]}")

    psi_res = {}
    psi_idx = md.compute_psi(traj)[0]
    for k in range(psis.shape[1]):
        r = int(traj.topology.atom(int(psi_idx[k][1])).residue.index)
        psi_res[r] = psis[:, k]
    circ_rows = []
    for r in sorted(psi_res):
        a = psi_res[r]
        circ_rows.append({"torsion": f"psi {RES3[r]}", "R1": circ_R(a, 1), "R2": circ_R(a, 2),
                          "R3": circ_R(a, 3), "R4": circ_R(a, 4),
                          "mean_dir": float(np.degrees(np.angle(np.exp(1j * a).mean())))})
    pairs_fl = []
    arr_psi = np.stack([psi_res[r] for r in sorted(psi_res)], axis=1)
    for i in range(arr_psi.shape[1]):
        for j in range(i + 1, arr_psi.shape[1]):
            pairs_fl.append((f"psi {RES3[sorted(psi_res)[i]]}", f"psi {RES3[sorted(psi_res)[j]]}", fish_lee(arr_psi[:, i], arr_psi[:, j])))
    pairs_fl.sort(key=lambda t: -abs(t[2]))
    print("circ-corr top 5:", [(a, b, round(r, 3)) for a, b, r in pairs_fl[:5]])

    cv2 = census["torpca_pc2"].to_numpy()
    kded = gaussian_kde(cv2, bw_method=0.06)
    xs = np.linspace(cv2.min(), cv2.max(), 400)
    d = kded(xs)
    peaks, _ = find_peaks(d, prominence=0.04 * d.max())
    mins, _ = find_peaks(-d, prominence=0.04 * d.max())
    bounds = np.concatenate([[xs[0]], xs[mins], [xs[-1]]])
    kde_states = []
    for i in range(len(bounds) - 1):
        m = (cv2 >= bounds[i]) & (cv2 <= bounds[i + 1])
        if m.sum() >= 50:
            kde_states.append({"lo": float(bounds[i]), "hi": float(bounds[i + 1]), "n": int(m.sum())})
    print(f"KDE modes {len(peaks)} at {np.round(xs[peaks], 2)}; states {len(kde_states)}")

    cv1 = census["c_bb_s4_r010_b3"].to_numpy()
    mi_rows = []
    for c in ("torpca_pc2", "restorpca_pc1", "torpca_pc1", "acylindricity", "hairpin_closure_ratio",
              "chirality_ca", "turn3537_mean_ca", "contact_order_ca8"):
        y = census[c].to_numpy()
        mi = kraskov_mi(cv1, y)
        nulls = [kraskov_mi(cv1, y[RNG.permutation(n)]) for _ in range(3)]
        mi_rows.append({"cv2": c, "mi": mi, "null_max": float(np.max(nulls)), "excess": mi - float(np.mean(nulls))})
        print(f"MI cv1 x {c}: {mi:.3f} (null {np.mean(nulls):.3f})")

    def gen_class(name):
        if "_nma" in name or "mode" in name:
            return "NMA"
        if "_hop_" in name:
            return "BH"
        if "explore_" in name:
            return "PCA-frontier"
        return "initial"

    gens = np.array([gen_class(s) for s in names])
    Xc = X - X.mean(0)
    A5 = np.column_stack([np.ones(n), cv1])
    Xcc = Xc - (A5 @ np.linalg.lstsq(A5, Xc, rcond=None)[0])
    _, _, Vt_all = np.linalg.svd(Xc, full_matrices=False)
    _, _, Vt_res_all = np.linalg.svd(Xcc, full_matrices=False)
    stab = {}
    B = 25
    for k, Vt in (("plain_pc1", Vt_all), ("plain_pc2", Vt_all), ("residual_pc1", Vt_res_all)):
        kk = 0 if k == "plain_pc1" else 1
        kk2 = 0 if "residual" in k else kk
        vals = []
        for rep in range(B):
            m = np.zeros(n, bool)
            m[RNG.choice(n, n // 2, replace=False)] = True
            Xh = Xc[m] - Xc[m].mean(axis=0)
            if "residual" in k:
                A5m = np.column_stack([np.ones(m.sum()), cv1[m]])
                Xh = Xh - (A5m @ np.linalg.lstsq(A5m, Xh, rcond=None)[0])
                _, _, Vh = np.linalg.svd(Xh, full_matrices=False)
                vals.append(float(abs(np.dot(Vt_res_all[0], Vh[0]))))
            else:
                _, _, Vh = np.linalg.svd(Xh, full_matrices=False)
                vals.append(float(abs(np.dot(Vt[kk2], Vh[kk2]))))
        stab[k] = {"median": float(np.median(vals)), "min": float(np.min(vals))}
        print(f"half-bank |cos| {k}: median {stab[k]['median']:.3f} min {stab[k]['min']:.3f}")
    for g in ("initial", "NMA", "PCA-frontier", "BH"):
        m = gens == g
        if m.sum() < 500:
            stab[f"gen_{g}"] = {"n": int(m.sum()), "note": "skipped"}
            continue
        Xg = Xc[m] - Xc[m].mean(axis=0)
        _, _, Vg = np.linalg.svd(Xg, full_matrices=False)
        stab[f"gen_{g}"] = {"n": int(m.sum()), "pc1": float(abs(np.dot(Vg[0], Vt_all[0]))),
                            "pc2": float(abs(np.dot(Vg[1], Vt_all[1])))}
        print(f"gen {g}: pc1 {stab[f'gen_{g}']['pc1']:.3f} pc2 {stab[f'gen_{g}']['pc2']:.3f}")

    from openmm import app, unit, openmm as omm
    pdb0 = app.PDBFile(str(files[0]))
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    gmap = {"bond": 1, "angle": 2, "torsion": 3, "nonbonded": 4, "gb_solvation": 5}
    for force in sys0.getForces():
        g = {"HarmonicBondForce": "bond", "HarmonicAngleForce": "angle",
             "PeriodicTorsionForce": "torsion", "NonbondedForce": "nonbonded",
             "CustomGBForce": "gb_solvation"}.get(type(force).__name__)
        if g:
            force.setForceGroup(gmap[g])
    integ = omm.VerletIntegrator(0.001)
    ctx = omm.Context(sys0, integ)
    terms = list(gmap) + ["total"]
    energies = np.zeros((n, len(terms)), dtype=np.float64)
    for fi, f in enumerate(files):
        pdb = app.PDBFile(str(f))
        ctx.setPositions(pdb.positions)
        for gi, t in enumerate(terms[:-1]):
            energies[fi, gi] = ctx.getState(getEnergy=True, groups=1 << gmap[t]).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        energies[fi, -1] = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if fi % 2000 == 0:
            print(f"  energies {fi}/{n}", flush=True)
    edf = pd.DataFrame(energies, columns=terms, index=names)
    edf.to_csv(REPORT / "genpept_rep3_energy_components.csv")
    estats = {t: {"rho_cv1": float(spearmanr(edf[t].to_numpy(), cv1).correlation),
                  "rho_cv2": float(spearmanr(edf[t].to_numpy(), census["torpca_pc2"].to_numpy()).correlation),
                  "median": float(np.median(edf[t].to_numpy()))} for t in terms}

    cv1_heavy = census["c_heavy_s4_r012_b3"].to_numpy()
    cv1_legacy = census["c_heavy_s4_r04.5_b6"].to_numpy()
    q05, q95 = np.quantile(cv1_legacy, [0.05, 0.95])
    band = (cv1_legacy >= q05) & (cv1_legacy <= q95)
    print(f"legacy band seeds: {band.sum()}")

    doc = {"twonn_3k": tw, "dmap_3k": dmap_out, "circ_stats": circ_rows, "circ_corr_top": pairs_fl[:10],
           "kde_states": {"modes": xs[peaks].tolist(), "states": kde_states}, "mi": mi_rows,
           "stability": stab, "energies": estats}
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    ax = axes[0, 0]
    ax.plot(xs, d, color="#2c7fb8")
    ax.fill_between(xs, d, alpha=.3, color="#2c7fb8")
    for p in xs[peaks]:
        ax.axvline(p, color="#d95f02", ls="--", lw=.7)
    ax.set_title(f"torpca_pc2 KDE on rep3 (bw .06): {len(peaks)} modes")
    ax = axes[0, 1]
    wm = np.array([[c["R1"], c["R2"], c["R3"], c["R4"]] for c in circ_rows])
    im = ax.imshow(wm, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["R1", "R2", "R3", "R4"])
    ax.set_yticks(range(len(circ_rows)))
    ax.set_yticklabels([c["torsion"] for c in circ_rows], fontsize=6)
    fig.colorbar(im, ax=ax, shrink=.8)
    ax.set_title("circular harmonics per torsion")
    ax = axes[0, 2]
    for tag, col in (("torsion", "#2c7fb8"), ("ca_drmsd", "#e6550d")):
        ax.plot(range(1, 7), dmap_out[tag]["lams"][1:7], "o-", label=tag, color=col)
    ax.set_title("dmap spectra (3k subset)")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ci = [c["cv2"] for c in mi_rows]
    ax.bar(range(len(ci)), [c["excess"] for c in mi_rows], color="#3182bd")
    ax.set_xticks(range(len(ci)))
    ax.set_xticklabels(ci, rotation=55, fontsize=7)
    ax.set_title("MI excess (nats), cv1 x cv2")
    ax.grid(alpha=.3, axis="y")
    ax = axes[1, 1]
    labs = list(stab)
    vals = [stab[k].get("median") or np.nan for k in labs if "median" in stab[k] or "pc1" in stab[k]]
    labs2 = [k for k in labs if "median" in stab[k] or "pc1" in stab[k]]
    vals = [stab[k].get("median", stab[k].get("pc1")) for k in labs2]
    ax.bar(range(len(labs2)), vals, color="#7570b3")
    ax.axhline(0.95, color="red", ls="--", lw=1)
    ax.set_xticks(range(len(labs2)))
    ax.set_xticklabels(labs2, rotation=45, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("half-bank / generator |cos(axis)|")
    ax.grid(alpha=.3, axis="y")
    ax = axes[1, 2]
    for t, col in zip(terms, ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#333333")):
        ax.scatter(cv2, edf[t].to_numpy(), s=1, alpha=.2, color=col, label=t)
    ax.set_xlabel("torpca_pc2")
    ax.set_ylabel("GBn2 energy kJ/mol")
    ax.set_title("full-bank decomposition vs CV2 (10,019)", fontsize=9)
    ax.legend(fontsize=6, markerscale=6, ncol=2)
    fig.suptitle("rep3_massive advanced battery", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "cv_rep3_advanced.png", dpi=150)
    plt.close(fig)

    lines = ["# rep3_massive advanced battery (10,019 seeds)\n"]
    lines.append(f"- TwoNN (3k subset): torsion d={tw['torsion_3k']:.2f}, CA-dRMSD d={tw['ca_drmsd_3k']:.2f}")
    for tag in ("torsion", "ca_drmsd"):
        lines.append(f"- dmap {tag} lambdas[1:7]: {[round(float(l),4) for l in dmap_out[tag]['lams'][1:7]]}")
    lines.append(f"- KDE torpca_pc2: {len(peaks)} modes {[round(float(p),2) for p in xs[peaks]]}")
    lines.append(f"- MI excess top: " + "; ".join(f"{r['cv2']} {r['excess']:+.3f}" for r in sorted(mi_rows, key=lambda r: -r['excess'])[:4]))
    for k, v in stab.items():
        if "median" in v:
            lines.append(f"- stab {k}: median {v['median']:.3f} min {v['min']:.3f}")
        elif "pc1" in v:
            lines.append(f"- stab {k} (n={v['n']}): pc1 {v['pc1']:.3f} pc2 {v['pc2']:.3f}")
    for t, v in estats.items():
        lines.append(f"- energy {t}: rho_cv1 {v['rho_cv1']:+.2f} rho_cv2 {v['rho_cv2']:+.2f}")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON, REPORT / "genpept_rep3_energy_components.csv", FIG / "cv_rep3_advanced.png")


if __name__ == "__main__":
    main()
