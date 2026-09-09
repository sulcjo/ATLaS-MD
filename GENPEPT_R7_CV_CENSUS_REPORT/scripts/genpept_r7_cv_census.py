import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import mdtraj as md
from scipy.spatial.distance import cdist

BASE = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(__file__).resolve().parents[1]
SEED_DIR = str(BASE / "RUNS" / "chignolin_genpept_r7" / "final_implicit_survivor_seeds")
OUT_CSV = str(REPORT_DIR / "genpept_r7_cv_census_values.csv")

R0_BETA_CORE = [(4.5, 6.0), (6.0, 6.0), (8.0, 3.0), (8.0, 6.0), (10.0, 3.0), (12.0, 1.5), (12.0, 3.0)]
R0_BETA_EXT = [(6.0, 1.5), (6.0, 3.0), (8.0, 1.5), (10.0, 1.5), (10.0, 6.0), (12.0, 6.0), (14.0, 3.0)]
R0_BETA_BALANCED = [(12.0, 1.5), (12.0, 3.0), (4.5, 6.0)]
R0_BETA_RESBAL_FULL = [(8.0, 1.5), (8.0, 3.0), (8.0, 6.0), (10.0, 1.5), (10.0, 3.0), (10.0, 6.0),
                       (12.0, 1.5), (12.0, 3.0), (12.0, 6.0)]
RATIONAL_PARAMS = [(8.0, 6, 10), (8.0, 6, 12), (8.0, 8, 14), (10.0, 6, 10), (10.0, 6, 12), (10.0, 8, 14)]
RATIONAL_SELECTIONS = ["heavy", "bb", "ca"]
STRAND_R0_BETA = [(8.0, 3.0), (10.0, 3.0), (12.0, 3.0)]
SELECTIONS = ["heavy", "bb", "sch", "sca", "ca"]

BB_NAMES = {"N", "CA", "C", "O", "OXT"}
BB_H_NAMES = {"H", "H1", "H2", "H3", "HA", "HA2", "HA3"}


def switch_fraction(d, r0_a, beta_a):
    x = beta_a * (d - r0_a)
    return 0.5 * (1.0 - np.tanh(0.5 * x))


def atom_sets(top):
    sets = {"heavy": [], "bb": [], "sch": [], "sca": [], "ca": []}
    for a in top.atoms:
        idx = a.index
        is_h = a.element.symbol == "H" if a.element is not None else a.name.startswith("H")
        if not is_h:
            sets["heavy"].append(idx)
        if a.name in ("N", "CA", "C", "O"):
            sets["bb"].append(idx)
        if (not is_h) and a.name not in BB_NAMES:
            sets["sch"].append(idx)
        if (a.name not in BB_NAMES) and (a.name not in BB_H_NAMES):
            sets["sca"].append(idx)
        if a.name == "CA":
            sets["ca"].append(idx)
    return sets


def pair_lists(top, sets):
    res_of_all = np.array([a.residue.index for a in top.atoms])
    out = {}
    for sep in (2, 3, 4):
        for name, idxs in sets.items():
            idxs = np.array(idxs)
            ii, jj = np.triu_indices(len(idxs), k=1)
            dr = np.abs(res_of_all[idxs[ii]] - res_of_all[idxs[jj]])
            keep = dr >= sep
            out[(name, sep)] = (idxs[ii][keep], idxs[jj][keep])
    return out


def balanced_weights(top, sets):
    res_of_all = np.array([a.residue.index for a in top.atoms])
    out = {}
    for name, idxs in sets.items():
        idxs = np.array(idxs)
        ii, jj = np.triu_indices(len(idxs), k=1)
        dr = np.abs(res_of_all[idxs[ii]] - res_of_all[idxs[jj]])
        keep = dr >= 4
        ai, aj = idxs[ii][keep], idxs[jj][keep]
        rpairs = np.stack([np.minimum(res_of_all[ai], res_of_all[aj]),
                           np.maximum(res_of_all[ai], res_of_all[aj])], axis=1)
        keys, inv = np.unique(rpairs, axis=0, return_inverse=True)
        counts = np.bincount(inv)
        w = 1.0 / counts[inv]
        out[(name, 4)] = (ai, aj, w)
    return out


def strand_pairs(top, sets):
    heavy = np.array(sets["heavy"])
    res_of_all = np.array([a.residue.index for a in top.atoms])
    nres = top.n_residues
    half = nres // 2
    ii, jj = np.triu_indices(len(heavy), k=1)
    ri, rj = res_of_all[heavy[ii]], res_of_all[heavy[jj]]
    cross_all = (ri < half) & (rj >= half)
    cross_far = cross_all & (np.abs(ri - rj) >= 3)
    return {"cross_all": (heavy[ii][cross_all], heavy[jj][cross_all]),
            "cross_far": (heavy[ii][cross_far], heavy[jj][cross_far])}


def gyration_eigs(pos, idxs):
    p = pos[idxs] - pos[idxs].mean(axis=0)
    g = np.einsum("ij,ik->jk", p, p) / len(idxs)
    ev = np.linalg.eigvalsh(g)
    return np.sort(ev)[::-1]


def main():
    files = sorted(glob.glob(os.path.join(SEED_DIR, "*.pdb")))
    n = len(files)
    first = md.load(files[0])
    top = first.topology
    sets = atom_sets(top)
    plists = pair_lists(top, sets)
    blists = balanced_weights(top, sets)

    ca_idx = np.array(sets["ca"])
    heavy_idx = np.array(sets["heavy"])
    n_term = [a.index for a in top.residue(0).atoms if a.name == "N"][0]
    c_term = [a.index for a in top.residue(top.n_residues - 1).atoms if a.name == "C"][0]

    o_idx = {r.index: [a.index for a in r.atoms if a.name == "O"][0] for r in top.residues}
    n_idx = {r.index: [a.index for a in r.atoms if a.name == "N"][0] for r in top.residues}
    hb_res = sorted({(min(i, j), max(i, j)) for i in range(top.n_residues)
                     for j in range(top.n_residues) if abs(i - j) >= 3})
    hb_all_pairs = [(n_idx[i], o_idx[j], i, j) for i, j in hb_res]

    strand_pairs_res = [(1, 8), (2, 7), (3, 6), (0, 9)]
    strand_hb = []
    for i, j in strand_pairs_res:
        strand_hb.extend([(n_idx[i], o_idx[j]), (o_idx[i], n_idx[j])])

    y2_ring = [a.index for a in top.residue(1).atoms if a.name in ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")]
    w9_ring = [a.index for a in top.residue(8).atoms if a.name in ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2")]
    y2_side = [a.index for a in top.residue(1).atoms if a.name not in BB_NAMES]
    w9_side = [a.index for a in top.residue(8).atoms if a.name not in BB_NAMES]
    d3_side = [a.index for a in top.residue(2).atoms if a.name not in BB_NAMES]

    contact_jobs = []
    for (sel, sep), (ii, jj) in plists.items():
        for r0, beta in R0_BETA_CORE:
            contact_jobs.append({"base": f"c_{sel}_s{sep}", "r0": r0, "beta": beta, "ii": ii, "jj": jj, "w": None})
        if sep in (2, 4):
            for r0, beta in R0_BETA_EXT:
                contact_jobs.append({"base": f"c_{sel}_s{sep}", "r0": r0, "beta": beta, "ii": ii, "jj": jj, "w": None})
    for (sel, sep), (ii, jj, w) in blists.items():
        for r0, beta in R0_BETA_BALANCED + R0_BETA_RESBAL_FULL:
            contact_jobs.append({"base": f"cresb_{sel}_s{sep}", "r0": r0, "beta": beta, "ii": ii, "jj": jj, "w": w})
    for sel in RATIONAL_SELECTIONS:
        ii, jj = plists[(sel, 4)]
        for r0, nn, mm in RATIONAL_PARAMS:
            contact_jobs.append({"base": f"crat_{sel}_s4_n{nn}m{mm}", "r0": r0, "beta": None,
                                 "nm": (nn, mm), "ii": ii, "jj": jj, "w": None})

    cols = {}
    for job in contact_jobs:
        if job["beta"] is None:
            cols[f"{job['base']}_r0{job['r0']:g}"] = np.zeros(n)
        else:
            cols[f"{job['base']}_r0{job['r0']:g}_b{job['beta']:g}"] = np.zeros(n)

    spairs = strand_pairs(top, sets)
    for tag in spairs:
        for r0, beta in STRAND_R0_BETA:
            cols[f"strand_{tag}_r0{r0:g}_b{beta:g}"] = np.zeros(n)
    cols["strand_mindist"] = np.zeros(n)

    side_pack = {}
    res_of_all = np.array([a.residue.index for a in top.atoms])
    heavy_arr = np.array(sets["heavy"])
    for res_i, tag in ((1, "pack_y2"), (3, "pack_p4"), (8, "pack_w9")):
        sc = np.array([a.index for a in top.atoms
                       if a.residue.index == res_i and a.name not in BB_NAMES
                       and (a.element is not None and a.element.symbol != "H")])
        others = heavy_arr[res_of_all[heavy_arr] != res_i]
        side_pack[tag] = (sc, others)
        cols[tag] = np.zeros(n)

    base_names = [
        "rg_heavy", "rg_ca", "e2e_nc_ca", "e2e_n_c", "asphericity", "acylindricity",
        "shape_anisotropy", "max_ca_ca", "mean_ca_ca", "d_y2_w9_ca", "d_y2_w9_ring",
        "d_d3_t8_ca", "d_p4_g7_ca", "d_e5_t8_ca", "hairpin_closure_ratio",
        "turn3537_mean_ca", "hbond_bb_ge3", "hbond_i_i3", "hbond_i_i4",
        "hbond_crossstrand", "contact_order_ca8", "chirality_ca",
    ]
    base = {k: np.zeros(n) for k in base_names}

    dist_cache = {}
    for fi, f in enumerate(files):
        t = md.load(f)
        pos_nm = t.xyz[0]
        dist_cache.clear()
        for job in contact_jobs:
            key = (job["base"], job["w"] is not None)
            if key not in dist_cache:
                d = np.linalg.norm(pos_nm[job["ii"]] - pos_nm[job["jj"]], axis=1) * 10.0
                dist_cache[key] = d
            d = dist_cache[key]
            r0, beta = job["r0"], job["beta"]
            if beta is None:
                nn, mm = job["nm"]
                x = d / r0
                v = (1.0 - np.power(x, nn)) / (1.0 - np.power(x, mm))
                cname = f"{job['base']}_r0{r0:g}"
                cols[cname][fi] = v.mean()
                continue
            v = switch_fraction(d, r0, beta)
            cname = f"{job['base']}_r0{r0:g}_b{beta:g}"
            if job["w"] is None:
                cols[cname][fi] = v.mean()
            else:
                w = job["w"]
                cols[cname][fi] = (v * w).sum() / w.sum()

        for tag, (ii, jj) in spairs.items():
            d = np.linalg.norm(pos_nm[ii] - pos_nm[jj], axis=1) * 10.0
            for r0, beta in STRAND_R0_BETA:
                cols[f"strand_{tag}_r0{r0:g}_b{beta:g}"][fi] = switch_fraction(d, r0, beta).mean()
            if tag == "cross_all":
                cols["strand_mindist"][fi] = d.min() if len(d) else np.nan

        for tag, (sc, others) in side_pack.items():
            dp = cdist(pos_nm[sc], pos_nm[others]) * 10.0
            cols[tag][fi] = float((dp < 5.0).sum(axis=1).mean())

        ev_heavy = gyration_eigs(pos_nm, heavy_idx)
        ev_ca = gyration_eigs(pos_nm, ca_idx)
        base["rg_heavy"][fi] = math.sqrt(ev_heavy.sum())
        base["rg_ca"][fi] = math.sqrt(ev_ca.sum())
        l1, l2, l3 = ev_heavy
        base["asphericity"][fi] = l1 - 0.5 * (l2 + l3)
        base["acylindricity"][fi] = l2 - l3
        s = ev_heavy.sum()
        base["shape_anisotropy"][fi] = 1.0 - 3.0 * (l1 * l2 + l2 * l3 + l1 * l3) / (s * s) if s > 0 else 0.0

        base["e2e_n_c"][fi] = np.linalg.norm(pos_nm[n_term] - pos_nm[c_term]) * 10.0
        base["e2e_nc_ca"][fi] = np.linalg.norm(pos_nm[ca_idx[0]] - pos_nm[ca_idx[-1]]) * 10.0
        dca = cdist(pos_nm[ca_idx], pos_nm[ca_idx]) * 10.0
        base["max_ca_ca"][fi] = dca.max()
        base["mean_ca_ca"][fi] = dca[np.triu_indices(len(ca_idx), k=1)].mean()
        base["d_y2_w9_ca"][fi] = dca[1, 8]
        base["d_d3_t8_ca"][fi] = dca[2, 7]
        base["d_p4_g7_ca"][fi] = dca[3, 6]
        base["d_e5_t8_ca"][fi] = dca[4, 7]
        base["hairpin_closure_ratio"][fi] = (dca[1, 8] + 1e-9) / (dca[0, 9] + 1e-9)
        base["turn3537_mean_ca"][fi] = np.mean(
            [np.linalg.norm(pos_nm[ca_idx[i]] - pos_nm[ca_idx[i + 3]]) for i in range(2, 7)]) * 10.0
        base["d_y2_w9_ring"][fi] = np.linalg.norm(pos_nm[y2_ring].mean(axis=0) - pos_nm[w9_ring].mean(axis=0)) * 10.0

        nb_hb = nb_hb3 = nb_hb4 = 0.0
        for a, b, i, j in hb_all_pairs:
            d = np.linalg.norm(pos_nm[a] - pos_nm[b]) * 10.0
            v = switch_fraction(d, 3.5, 8.0)
            nb_hb += v
            if abs(i - j) == 3:
                nb_hb3 += v
            if abs(i - j) == 4:
                nb_hb4 += v
        base["hbond_bb_ge3"][fi] = nb_hb
        base["hbond_i_i3"][fi] = nb_hb3
        base["hbond_i_i4"][fi] = nb_hb4
        nb_cs = 0.0
        for a, b in strand_hb:
            d = np.linalg.norm(pos_nm[a] - pos_nm[b]) * 10.0
            nb_cs += switch_fraction(d, 3.5, 8.0)
        base["hbond_crossstrand"][fi] = nb_cs

        iu = np.triu_indices(top.n_residues, k=1)
        mask = dca[iu] < 8.0
        sepvals = np.abs(iu[0] - iu[1])[mask]
        base["contact_order_ca8"][fi] = sepvals.mean() / top.n_residues if mask.any() else 0.0

        m = np.stack([pos_nm[ca_idx[4]] - pos_nm[ca_idx[0]],
                      pos_nm[ca_idx[7]] - pos_nm[ca_idx[0]],
                      pos_nm[ca_idx[9]] - pos_nm[ca_idx[0]]])
        base["chirality_ca"][fi] = np.linalg.det(m) * 1e3

        if fi % 200 == 0:
            print(f"...frame {fi}", flush=True)

    for k, v in base.items():
        cols[k] = v

    alltraj = md.load(files)
    rmsd_ref = md.rmsd(alltraj, first, frame=0, atom_indices=ca_idx) * 10.0
    cols["rmsd_ca_to_survivor000"] = rmsd_ref

    phi_idx, phis = md.compute_phi(alltraj)
    psi_idx, psis = md.compute_psi(alltraj)
    omega_idx, omegas = md.compute_omega(alltraj)
    chi1_idx, chi1s = md.compute_chi1(alltraj)
    phi_deg, psi_deg = np.degrees(phis), np.degrees(psis)

    n_alpha = np.zeros(n)
    n_beta = np.zeros(n)
    n_ppii = np.zeros(n)
    n_left = np.zeros(n)
    for k in range(phis.shape[1]):
        p, q = phi_deg[:, k], psi_deg[:, k]
        n_alpha += (((-100 <= p) & (p <= -20)) & ((-80 <= q) & (q <= -10))).astype(float)
        n_beta += (((-180 <= p) & (p <= -100)) & ((80 <= q) & (q <= 180))).astype(float)
        n_ppii += (((-100 <= p) & (p <= -40)) & ((120 <= q) & (q <= 180))).astype(float)
        n_left += (((20 <= p) & (p <= 110)) & ((-10 <= q) & (q <= 100))).astype(float)
    cols["rama_alpha"] = n_alpha
    cols["rama_beta"] = n_beta
    cols["rama_ppii"] = n_ppii
    cols["rama_left"] = n_left
    cols["cis_omega_any"] = (np.abs(np.degrees(omegas)) < 45).any(axis=1).astype(float)

    feats = []
    for arr in (phis, psis):
        feats.append(np.sin(arr))
        feats.append(np.cos(arr))
    X = np.concatenate(feats, axis=1)
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    for k in range(5):
        cols[f"torpca_pc{k+1}"] = Xc @ Vt[k]
    print("  torsion PCA explained var:", np.round(evr[:8], 3).tolist())

    key_cv = cols["c_heavy_s4_r012_b3"]
    A = np.vstack([np.ones(n), key_cv]).T
    coef, *_ = np.linalg.lstsq(A, Xc, rcond=None)
    Xr = Xc - A @ coef
    Ur, Sr, Vtr = np.linalg.svd(Xr, full_matrices=False)
    evrr = (Sr ** 2) / (Sr ** 2).sum()
    for k in range(3):
        cols[f"restorpca_pc{k+1}"] = Xr @ Vtr[k]
    print("  residual torsion PCA explained var:", np.round(evrr[:8], 3).tolist())

    chi_res = [int(top.atom(int(idx[0])).residue.index) for idx in chi1_idx]
    for res_i, tag in ((1, "y2"), (8, "w9")):
        if res_i in chi_res:
            col = chi1s[:, chi_res.index(res_i)]
            cols[f"chi1_{tag}_sin"] = np.sin(col)
            cols[f"chi1_{tag}_cos"] = np.cos(col)

    sasa = md.shrake_rupley(alltraj, mode="atom")
    hydro = np.array([a.element is not None and a.element.symbol == "C" for a in top.atoms])
    tot = sasa.sum(axis=1)
    cols["sasa_total"] = tot
    cols["sasa_hydrophobic_frac"] = sasa[:, hydro].sum(axis=1) / tot
    for name, atoms in (("burial_y2", y2_side), ("burial_w9", w9_side), ("burial_d3", d3_side)):
        frac = sasa[:, np.array(atoms)].sum(axis=1)
        ref = frac.max() if frac.max() > 0 else 1.0
        cols[name] = 1.0 - frac / ref

    names = [os.path.basename(f)[:-4] for f in files]
    df = pd.DataFrame({"seed": names, **cols})
    df.to_csv(OUT_CSV, index=False)
    print("wrote", OUT_CSV, df.shape)


if __name__ == "__main__":
    main()
