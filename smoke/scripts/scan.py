"""Structural search for the chignolin native fold in the ATLAS-MD v0.7 chignolin_7 ensemble.

Question: are natively folded conformations present in the trajectories at all,
irrespective of the free energy assigned to them?  A single metric is a weak
discriminator on a 10-mer, so this uses a set: backbone RMSD to all 18 1UAO NMR
models, native contact fraction Q, the two hairpin H-bonds, the Tyr2/Trp9
hydrophobic cluster, Rg, and the run's own biasing CV1 -- each with the native
structure's value on the same axis.
"""
import glob
import json
import os

import mdtraj as md
import numpy as np

RUN = "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production"
TOP_PDB = f"{RUN}/epoch_000/solute_only.pdb"
NATIVE_PDB = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/1uao.pdb"
PCA_NPZ = f"{RUN}/pmf_analysis/pca_scores.npz"
OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
os.makedirs(OUT, exist_ok=True)

# The run's own contact-CV parameters, from RUNS/chignolin_7/run_args.json:
# heavy atoms, |i-j| >= 4, r0 = 12 A, beta = 3 A^-1, normalized, atom-pairs.
CONTACT_R0_NM = 1.2
CONTACT_BETA_NM = 30.0
CONTACT_MIN_SEP = 4

top = md.load(TOP_PDB).topology
native = md.load(NATIVE_PDB)


def name_index(topology):
    return {(a.residue.resSeq, a.name): a.index for a in topology.atoms}


nat_idx = name_index(native.topology)
run_idx = name_index(top)
common = sorted(set(nat_idx) & set(run_idx))
heavy_common = [k for k in common if not k[1].startswith("H")]

# Backbone over residues 2-9.  The Gly termini fray in every chignolin ensemble
# and would inflate RMSD without reporting on the fold.
bb_pairs = [k for k in heavy_common if k[1] in {"N", "CA", "C", "O"} and 2 <= k[0] <= 9]
bb_run = np.array([run_idx[k] for k in bb_pairs])
bb_nat = np.array([nat_idx[k] for k in bb_pairs])

ca_pairs = [k for k in heavy_common if k[1] == "CA"]
ca_run = np.array([run_idx[k] for k in ca_pairs])
ca_nat = np.array([nat_idx[k] for k in ca_pairs])

heavy_run = np.array([run_idx[k] for k in heavy_common])
heavy_nat = np.array([nat_idx[k] for k in heavy_common])
heavy_res = np.array([k[0] for k in heavy_common])

# Native contacts: heavy-atom pairs |i-j| >= 3 closer than 4.5 A in at least
# half of the 18 NMR models, so Q does not inherit one model's idiosyncrasies.
pair_ij = np.array([(i, j) for i in range(len(heavy_nat)) for j in range(i + 1, len(heavy_nat))
                    if abs(heavy_res[i] - heavy_res[j]) >= 3])
pair_nat = np.array([(heavy_nat[i], heavy_nat[j]) for i, j in pair_ij])
d_models = md.compute_distances(native, pair_nat)
native_mask = (d_models < 0.45).mean(axis=0) >= 0.5
q_idx_run = np.array([(heavy_run[i], heavy_run[j]) for i, j in pair_ij[native_mask]])
q_idx_nat = pair_nat[native_mask]
q_r0 = d_models[:, native_mask].mean(axis=0)
n_native_contacts = int(native_mask.sum())

# The run's reference-free contact CV, recomputed so the native structure can be
# placed on the same axis as the production PMF.
cv_pairs_local = np.array([(i, j) for i in range(len(heavy_run)) for j in range(i + 1, len(heavy_run))
                           if abs(heavy_res[i] - heavy_res[j]) >= CONTACT_MIN_SEP])
cv_idx_run = np.array([(heavy_run[i], heavy_run[j]) for i, j in cv_pairs_local])
cv_idx_nat = np.array([(heavy_nat[i], heavy_nat[j]) for i, j in cv_pairs_local])

hb_run = np.array([[run_idx[(3, "N")], run_idx[(8, "O")]], [run_idx[(3, "N")], run_idx[(7, "O")]]])
hb_nat = np.array([[nat_idx[(3, "N")], nat_idx[(8, "O")]], [nat_idx[(3, "N")], nat_idx[(7, "O")]]])
tyr_ring = [k for k in heavy_common if k[0] == 2 and k[1] in
            {"CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"}]
trp_ring = [k for k in heavy_common if k[0] == 9 and k[1] in
            {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}]
tyr_run = np.array([run_idx[k] for k in tyr_ring])
trp_run = np.array([run_idx[k] for k in trp_ring])
tyr_nat = np.array([nat_idx[k] for k in tyr_ring])
trp_nat = np.array([nat_idx[k] for k in trp_ring])

# PCA basis produced by the production analysis (10 CA atoms, 2 components).
_pz = np.load(PCA_NPZ, allow_pickle=False)
pca_mean = np.asarray(_pz["mean_A"], dtype=np.float64)
pca_comp = np.asarray(_pz["components"], dtype=np.float64)
pca_ref = md.Trajectory((pca_mean.reshape(1, 10, 3) / 10.0).astype(np.float32),
                        md.load(TOP_PDB).atom_slice(ca_run).topology)


def contact_cv(traj, idx):
    d = md.compute_distances(traj, idx)
    return (0.5 * (1.0 - np.tanh(0.5 * CONTACT_BETA_NM * (d - CONTACT_R0_NM)))).mean(axis=1)


def q_fraction(traj, idx, r0):
    d = md.compute_distances(traj, idx)
    return (1.0 / (1.0 + np.exp(50.0 * (d - 1.2 * r0)))).mean(axis=1)


def ring_dist(traj, a, b):
    return np.linalg.norm(traj.xyz[:, a, :].mean(axis=1) - traj.xyz[:, b, :].mean(axis=1), axis=1)


def pca_scores(traj, ca):
    sub = traj.atom_slice(ca)
    sub.superpose(pca_ref, frame=0)
    return (sub.xyz.reshape(sub.n_frames, -1) * 10.0 - pca_mean) @ pca_comp.T


def metrics(traj, is_native):
    qi, cvi, hb, ty, tr, hv, ca = ((q_idx_nat, cv_idx_nat, hb_nat, tyr_nat, trp_nat, heavy_nat, ca_nat)
                                   if is_native else
                                   (q_idx_run, cv_idx_run, hb_run, tyr_run, trp_run, heavy_run, ca_run))
    hbd = md.compute_distances(traj, hb) * 10.0
    sc = pca_scores(traj, ca)
    return {"q": q_fraction(traj, qi, q_r0), "cv1": contact_cv(traj, cvi),
            "d1": hbd[:, 0], "d2": hbd[:, 1],
            "rg": md.compute_rg(traj.atom_slice(hv)) * 10.0,
            "yw": ring_dist(traj, ty, tr) * 10.0,
            "pca1": sc[:, 0], "pca2": sc[:, 1]}


nat_m = metrics(native, True)
model_rmsd = np.array([md.rmsd(native, native, frame=k, atom_indices=bb_nat) * 10.0
                       for k in range(native.n_frames)])
native_ref = {
    "n_native_contacts": n_native_contacts,
    "n_cv_pairs": int(len(cv_idx_run)),
    "n_bb_atoms": int(len(bb_run)),
    "per_model": {k: [float(x) for x in v] for k, v in nat_m.items()},
    "model1": {k: float(v[0]) for k, v in nat_m.items()},
    "models_mean": {k: float(np.mean(v)) for k, v in nat_m.items()},
    "models_std": {k: float(np.std(v)) for k, v in nat_m.items()},
    "models_min": {k: float(np.min(v)) for k, v in nat_m.items()},
    "models_max": {k: float(np.max(v)) for k, v in nat_m.items()},
    "model_pairwise_bb_rmsd_A": {"mean": float(model_rmsd[~np.eye(18, dtype=bool)].mean()),
                                 "max": float(model_rmsd.max()),
                                 "min_offdiag": float(model_rmsd[~np.eye(18, dtype=bool)].min())},
}
print(json.dumps({k: v for k, v in native_ref.items() if k != "per_model"}, indent=2), flush=True)
with open(f"{OUT}/native_reference.json", "w") as fh:
    json.dump(native_ref, fh, indent=2)

files = sorted(glob.glob(f"{RUN}/_merged_replica_trajectories/*.xtc"))
print(f"{len(files)} trajectory files", flush=True)

keys = ["rmsd_bb_min18", "rmsd_bb_m1", "rmsd_ca_m1", "q", "cv1", "d1", "d2", "rg", "yw",
        "pca1", "pca2"]
cols = {k: [] for k in keys}
src_file, src_frame = [], []
n_bad = 0
for n, f in enumerate(files):
    try:
        tr = md.load(f, top=top)
    except Exception as exc:  # a truncated xtc must be counted, never skipped silently
        n_bad += 1
        print(f"FAILED {f}: {exc}", flush=True)
        continue
    if tr.n_frames == 0:
        continue
    r18 = np.stack([md.rmsd(tr, native, frame=k, atom_indices=bb_run, ref_atom_indices=bb_nat)
                    for k in range(native.n_frames)]) * 10.0
    cols["rmsd_bb_min18"].append(r18.min(axis=0))
    cols["rmsd_bb_m1"].append(r18[0])
    cols["rmsd_ca_m1"].append(md.rmsd(tr, native, frame=0, atom_indices=ca_run,
                                      ref_atom_indices=ca_nat) * 10.0)
    m = metrics(tr, False)
    for k in ["q", "cv1", "d1", "d2", "rg", "yw", "pca1", "pca2"]:
        cols[k].append(m[k])
    src_file.append(np.full(tr.n_frames, n, dtype=np.int32))
    src_frame.append(np.arange(tr.n_frames, dtype=np.int32))
    if n % 100 == 0:
        print(f"  {n}/{len(files)} files, {sum(len(x) for x in src_frame)} frames", flush=True)

arr = {k: np.concatenate(v).astype(np.float32) for k, v in cols.items()}
arr["src_file"] = np.concatenate(src_file)
arr["src_frame"] = np.concatenate(src_frame)
np.savez_compressed(f"{OUT}/frame_metrics.npz", files=np.array(files), n_bad=np.array([n_bad]), **arr)
print(f"TOTAL FRAMES {len(arr['q'])}  failed_files {n_bad}", flush=True)
