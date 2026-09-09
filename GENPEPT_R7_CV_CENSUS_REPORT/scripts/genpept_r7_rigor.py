from __future__ import annotations

import csv
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
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

BASE = Path("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")
ROOT = BASE / "RUNS" / "chignolin_genpept_r7"
REPORT = BASE / "GENPEPT_R7_CV_CENSUS_REPORT"
FIG = REPORT / "figures"
OBS_DIR = BASE / "RUNS" / "chignolin_6" / "adaptive_production" / "epoch_000" / "tica_obs"
OUT_MD = REPORT / "RIGOR_HANDOFF.md"
OUT_JSON = REPORT / "genpept_r7_rigor.json"
OUT_GEOPDB = FIG / "r7v2_residual_cv2_geodesic_path.pdb"
RES3 = ["G1", "Y2", "D3", "P4", "E5", "T6", "G7", "T8", "W9", "G10"]
RNG = np.random.default_rng(23)


def entropy_norm(v, n_bins=20):
    v = v[np.isfinite(v)]
    h, _ = np.histogram(v, bins=n_bins, range=(v.min(), v.max()))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(n_bins))


def n_resolvable(span, k=1200.0, overlap=1.5, T=300.0):
    return math.floor(span / (overlap * math.sqrt(1.98720425864083e-3 * T / k)))


def bootstrap_entropy(v, B=400):
    v = np.asarray(v)
    n = len(v)
    vals = []
    for _ in range(B):
        vals.append(entropy_norm(v[RNG.integers(0, n, n)]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main():
    files = sorted((ROOT / "final_implicit_survivor_seeds").glob("*.pdb"))
    names = [f.stem for f in files]
    n = len(names)
    census = pd.read_csv(REPORT / "genpept_r7_cv_census_values.csv").set_index("seed").loc[names]
    stats = pd.read_csv(REPORT / "genpept_r7_cv_census_stats.csv").set_index("cv")
    traj = md.load(files)

    phi_idx, phis = md.compute_phi(traj)
    psi_idx, psis = md.compute_psi(traj)
    cols = []
    for arr in (phis, psis):
        for k in range(arr.shape[1]):
            cols.append(np.sin(arr[:, k]))
            cols.append(np.cos(arr[:, k]))
    X = np.stack(cols, axis=1)
    print("features", X.shape, flush=True)

    bt = {}
    for c in ["c_bb_s4_r010_b3", "c_ca_s4_r010_b3", "c_heavy_s4_r012_b3", "cresb_heavy_s4_r012_b3",
              "torpca_pc1", "restorpca_pc1"]:
        lo, hi = bootstrap_entropy(census[c].to_numpy())
        bt[c] = {"entropy": float(stats.loc[c, "entropy"]), "ci95": [lo, hi]}
        v = census[c].to_numpy()
        spans = []
        for _ in range(200):
            r = v[RNG.integers(0, n, n)]
            spans.append(n_resolvable(float(np.quantile(r, .99) - np.quantile(r, .01))))
        bt[c]["nw_k1200"] = int(stats.loc[c, "nw_k1200"])
        bt[c]["nw_ci95"] = [float(np.quantile(spans, .025)), float(np.quantile(spans, .975))]
        print(f"{c}: entropy {bt[c]['entropy']:.3f} [{lo:.3f},{hi:.3f}] nw {bt[c]['nw_k1200']} "
              f"[{bt[c]['nw_ci95'][0]:.0f},{bt[c]['nw_ci95'][1]:.0f}]")

    obs_files = sorted(glob.glob(str(OBS_DIR / "dihedral_obs_*.npz")))
    c1s = []
    for of in obs_files:
        z = np.load(of)
        Xo = z["features"] - z["features"].mean(axis=0)
        _, S, Vt = np.linalg.svd(Xo, full_matrices=False)
        x = Xo @ Vt[0]
        xx = x - x.mean()
        t = (np.arange(len(x)) + 0.5) / len(x) * math.pi
        c = np.cos(t)
        num = (c * xx).sum()
        c1s.append(float(2.0 * num * num / (len(x) * (xx * xx).sum())))
    c1s = np.array(c1s)
    boots = [np.median(c1s[RNG.integers(0, len(c1s), len(c1s))]) for _ in range(400)]
    bt["trace_own_pc1_C1"] = {"median": float(np.median(c1s)),
                              "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}
    print(f"trace-own PC1 C1 median {np.median(c1s):.3f} [{np.quantile(boots,.025):.3f},{np.quantile(boots,.975):.3f}]")

    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    k = 15
    knn = np.sort(D, axis=1)[:, k]
    med_knn = float(np.median(knn))
    rows, cols_, data_ = [], [], []
    for i in range(n):
        nbrs = np.argsort(D[i])[1:k + 1]
        for j in nbrs:
            rows.append(i); cols_.append(j); data_.append(D[i, j])
    G = csr_matrix((data_, (rows, cols_)), shape=(n, n))
    idx73 = names.index([s for s in names if s.startswith("survivor_073_")][0])
    idx1259 = names.index([s for s in names if s.startswith("survivor_1259_")][0])
    dist_geo, pred = dijkstra(G, indices=idx73, return_predecessors=True)
    path = [idx1259]
    while path[-1] != idx73:
        path.append(pred[path[-1]])
    path = path[::-1]
    edges = np.array([D[path[i], path[i + 1]] for i in range(len(path) - 1)])
    tree = cKDTree(X)
    a, b = X[idx73], X[idx1259]
    tt = np.linspace(0, 1, 21)
    line_d = np.array([tree.query((1 - t) * a + t * b)[0] for t in tt])
    frac_edge_ok = float((edges <= med_knn).mean())
    frac_line_ok = float((line_d <= med_knn).mean())
    print(f"geodesic: {len(path)} steps, frac edges within median k15 dist {frac_edge_ok:.2f}; "
          f"linear interp: frac interpolated points within {frac_line_ok:.2f}")
    print(f"line max dist {line_d.max():.3f} vs median edge {np.median(edges):.3f}")

    with OUT_GEOPDB.open("w") as h:
        traj[0].save_pdb("/tmp/opencode/_hdr.pdb")
        from openmm import app
        pdb_top = app.PDBFile("/tmp/opencode/_hdr.pdb")
        pos_all = traj.xyz
        app.PDBFile.writeHeader(traj.topology.to_openmm(), h)
        for mi, p in enumerate(path):
            app.PDBFile.writeModel(traj.topology.to_openmm(), pos_all[p], h, modelIndex=mi + 1)
        app.PDBFile.writeFooter(traj.topology.to_openmm(), h)

    cand = pd.read_csv(ROOT / "candidate_seeds.csv")
    cand_pdbs = sorted((ROOT / "candidate_seeds").glob("*.pdb"))
    print(f"candidate pool: {len(cand)} rows, {len(cand_pdbs)} pdb files")

    def gen_class(name):
        if "_nma" in name or "mode" in name:
            return "NMA"
        if "_hop_" in name:
            return "BH"
        if "explore_" in name:
            return "PCA-frontier"
        return "initial"

    gens = np.array([gen_class(s) for s in names])
    prov = {}
    cv2 = census["restorpca_pc1"].to_numpy()
    cv1 = census["c_bb_s4_r010_b3"].to_numpy()
    for g in ("initial", "BH", "NMA", "PCA-frontier"):
        m = gens == g
        prov[g] = {"n": int(m.sum()), "cv1_mean": float(cv1[m].mean()), "cv1_std": float(cv1[m].std()),
                   "cv2_mean": float(cv2[m].mean()), "cv2_std": float(cv2[m].std()),
                   "frac_cv2_gt1": float((cv2[m] > 1).mean()), "frac_cv2_lt-1": float((cv2[m] < -1).mean())}
        print(f"{g}: n={m.sum()} cv1 {prov[g]['cv1_mean']:.3f} cv2 {prov[g]['cv2_mean']:+.3f} "
              f"frac(cv2>1) {prov[g]['frac_cv2_gt1']:.2f}")

    om_idx, oms = md.compute_omega(traj)
    p4_cols = [k for k in range(oms.shape[1])
               if traj.topology.atom(int(om_idx[k][0])).residue.index == 2
               or traj.topology.atom(int(om_idx[k][0])).residue.index == 3]
    cis_frac = {}
    for k in p4_cols:
        r = int(traj.topology.atom(int(om_idx[k][0])).residue.index)
        r2 = int(traj.topology.atom(int(om_idx[k][2])).residue.index)
        frac = float((np.abs(np.degrees(oms[:, k])) < 45).mean())
        cis_frac[f"omega {RES3[r]}-{RES3[r2]}"] = frac
        print(f"omega {RES3[r]}-{RES3[r2]}: cis fraction {frac:.4f}")

    Z = census[["c_bb_s4_r010_b3", "restorpca_pc1", "rg_heavy", "e2e_nc_ca"]].to_numpy()
    Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    tiles = {}
    DZ = np.sqrt(((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(-1))
    for K in (8, 12, 16):
        sel = [int(np.argmax(((Zs - Zs.mean(0)) ** 2).sum(1)))]
        while len(sel) < K:
            dmin = DZ[:, sel].min(axis=1)
            dmin[sel] = -1.0
            sel.append(int(np.argmax(dmin)))
        tiles[K] = [names[i] for i in sel]
    print("tiling K=16 first five:", tiles[16][:5])

    pd.DataFrame({"seed": tiles[16]}).to_csv(REPORT / "seed_bank_tiling_k16.csv", index=False)

    doc = {"bootstrap": bt, "geodesic": {"len": len(path), "frac_edges_ok": frac_edge_ok,
           "frac_line_ok": frac_line_ok, "line_max_dist": float(line_d.max()),
           "median_edge": float(np.median(edges)), "median_knn": med_knn},
           "provenance": prov, "cis_omega": cis_frac, "tiling_k16": tiles[16]}
    OUT_JSON.write_text(json.dumps(doc, indent=1))

    lines = ["# Rigor & hand-off pack\n", "## Bootstrap 95% intervals (400 resamples)" ]
    for c, v in bt.items():
        if "entropy" in v:
            lines.append(f"- {c}: entropy {v['entropy']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]; "
                         f"nw {v['nw_k1200']} [{v['nw_ci95'][0]:.0f},{v['nw_ci95'][1]:.0f}]")
        else:
            lines.append(f"- {c}: median {v['median']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}]")
    lines.append("\n## Morph path realism")
    lines.append(f"- geodesic 73->1259: {len(path)} steps, {frac_edge_ok*100:.0f}% edges within median kNN dist "
                 f"({med_knn:.2f}); linear interpolation: {frac_line_ok*100:.0f}% of 21 points within, "
                 f"max gap {line_d.max():.2f}")
    lines.append(f"- geodesic PDB: figures/r7v2_residual_cv2_geodesic_path.pdb ({len(path)} models)")
    lines.append("\n## Provenance stratification (survivor bank)")
    for g, v in prov.items():
        lines.append(f"- {g} (n={v['n']}): CV1 {v['cv1_mean']:.3f}±{v['cv1_std']:.3f}, "
                     f"CV2 {v['cv2_mean']:+.3f}, frac(cv2>1) {v['frac_cv2_gt1']:.2f}, frac(cv2<-1) {v['frac_cv2_lt-1']:.2f}")
    lines.append("\n## X-Pro / X-Y omega cis fractions")
    for k, v in cis_frac.items():
        lines.append(f"- {k}: {v:.4f}")
    lines.append("\n## Ladder tiling")
    lines.append(f"- k-center (z-scored CV1,CV2,Rg,e2e) K=16 seeds written to seed_bank_tiling_k16.csv")
    lines.append("")
    OUT_MD.write_text("\n".join(lines))
    print("wrote", OUT_MD, OUT_JSON, OUT_GEOPDB, REPORT / "seed_bank_tiling_k16.csv")


if __name__ == "__main__":
    main()
