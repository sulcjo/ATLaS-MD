"""Recompute the chignolin_7 PCA on ALL coordinate frames.

The production surface used a cache with 103,664 finite scores (2.02% of the
sample slots), fitted on 52,048 frames.  Here the basis is fitted on all 611,536
frames and every frame is projected.  Superposition is to the iteratively
refined ensemble mean rather than an arbitrary frame 0, so the basis does not
inherit the orientation of whichever trajectory happened to be read first.
"""
import glob, json, os
import mdtraj as md, numpy as np

RUN = "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production"
OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
TOP = f"{RUN}/epoch_000/solute_only.pdb"
top = md.load(TOP)
CA = top.topology.select("protein and name CA")
print(f"selection 'protein and name CA': {len(CA)} atoms")

files = sorted(glob.glob(f"{RUN}/_merged_replica_trajectories/*.xtc"))
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
order = [str(x) for x in z["files"]]
assert order == files, "trajectory file order changed since the metric scan"

# ---- pass 1: load every frame's CA coordinates (611,536 x 10 x 3 float32 = 73 MB) ----
chunks = []
for n, f in enumerate(files):
    t = md.load(f, top=top.topology, atom_indices=CA)
    if t.n_frames:
        chunks.append(t.xyz)
    if n % 300 == 0:
        print(f"  loaded {n}/{len(files)}", flush=True)
X = np.concatenate(chunks).astype(np.float64)
del chunks
N = X.shape[0]
print(f"loaded {N:,} frames x {X.shape[1]} CA atoms")
assert N == z["rmsd_bb_min18"].size, f"frame count mismatch {N} vs metrics"

# ---- iterative superposition onto the ensemble mean (Kabsch, no weights) ----
def kabsch_to(ref, P):
    """Superpose every frame in P onto ref; both centred in place."""
    P = P - P.mean(axis=1, keepdims=True)
    r = ref - ref.mean(axis=0, keepdims=True)
    # covariance per frame, then optimal rotation via SVD
    C = np.einsum("fij,ik->fjk", P, r)
    U, S, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", U, Vt)))
    D = np.zeros((P.shape[0], 3, 3))
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("fij,fjk,fkl->fil", U, D, Vt)
    return np.einsum("fij,fjk->fik", P, R)

ref = X[0] - X[0].mean(axis=0)
for it in range(6):
    Xa = kabsch_to(ref, X)
    new = Xa.mean(axis=0)
    shift = float(np.sqrt(((new - ref) ** 2).sum(axis=1).mean())) * 10.0
    print(f"  superposition iter {it}: mean structure moved {shift:.4f} A")
    ref = new
    if shift < 1e-4:
        break
X = Xa
print("superposed to the converged ensemble mean")

# ---- PCA on all frames ----
F = X.reshape(N, -1) * 10.0            # Angstrom, 30 dims
mean = F.mean(axis=0)
Fc = F - mean
cov = (Fc.T @ Fc) / (N - 1)
evals, evecs = np.linalg.eigh(cov)
idx = np.argsort(evals)[::-1]
evals, evecs = evals[idx], evecs[:, idx]
evr = evals / evals.sum()
comp = evecs[:, :2].T
scores = Fc @ comp.T
print(f"\nexplained variance: PC1 {evr[0]:.4f}  PC2 {evr[1]:.4f}  "
      f"PC3 {evr[2]:.4f}  (sum1-2 {evr[:2].sum():.4f})")

# ---- project 1UAO into the SAME basis ----
native = md.load("/home/sulcjo/.claude/jobs/b27cb9c3/tmp/1uao.pdb")
nat_i = {(a.residue.resSeq, a.name): a.index for a in native.topology.atoms}
run_i = {(a.residue.resSeq, a.name): a.index for a in top.topology.atoms}
ca_keys = [k for k in sorted(set(nat_i) & set(run_i)) if k[1] == "CA"]
nat_ca = native.xyz[:, np.array([nat_i[k] for k in ca_keys]), :].astype(np.float64)
nat_a = kabsch_to(ref, nat_ca)
nat_scores = (nat_a.reshape(native.n_frames, -1) * 10.0 - mean) @ comp.T
print(f"1UAO in the new basis: PC1 {nat_scores[:,0].mean():.3f} +/- {nat_scores[:,0].std():.3f} "
      f"[{nat_scores[:,0].min():.3f}, {nat_scores[:,0].max():.3f}]")
print(f"                       PC2 {nat_scores[:,1].mean():.3f} +/- {nat_scores[:,1].std():.3f} "
      f"[{nat_scores[:,1].min():.3f}, {nat_scores[:,1].max():.3f}]")

# ---- how does the new basis relate to the cached 2% one? ----
old = np.load(f"{RUN}/pmf_analysis/pca_scores.npz", allow_pickle=False)
oc = np.asarray(old["components"], dtype=np.float64)
cos = np.abs(comp @ oc.T)
print(f"\n|cos| between new and cached components:\n"
      f"  PC1.PC1' {cos[0,0]:.3f}   PC1.PC2' {cos[0,1]:.3f}\n"
      f"  PC2.PC1' {cos[1,0]:.3f}   PC2.PC2' {cos[1,1]:.3f}")
print(f"  cached explained variance: PC1 {float(old['explained_variance_ratio'][0]):.4f} "
      f"PC2 {float(old['explained_variance_ratio'][1]):.4f}")

np.savez_compressed(f"{OUT}/pca_all_frames.npz",
                    pc1=scores[:, 0].astype(np.float32), pc2=scores[:, 1].astype(np.float32),
                    components=comp.astype(np.float32), mean_A=mean.astype(np.float32),
                    explained_variance_ratio=evr[:5].astype(np.float32),
                    eigenvalues=evals.astype(np.float32),
                    native_scores=nat_scores.astype(np.float32),
                    ref_structure_nm=ref.astype(np.float32))
json.dump({"n_frames": int(N), "n_fit_frames": int(N), "selection": "protein and name CA",
           "evr": [float(x) for x in evr[:5]],
           "native_pc1_mean": float(nat_scores[:, 0].mean()), "native_pc1_std": float(nat_scores[:, 0].std()),
           "native_pc2_mean": float(nat_scores[:, 1].mean()), "native_pc2_std": float(nat_scores[:, 1].std()),
           "native_pc1_range": [float(nat_scores[:, 0].min()), float(nat_scores[:, 0].max())],
           "native_pc2_range": [float(nat_scores[:, 1].min()), float(nat_scores[:, 1].max())],
           "cos_new_vs_cached": cos.tolist(),
           "cached_evr": [float(x) for x in old["explained_variance_ratio"]],
           "superposition": "iterative ensemble mean"},
          open(f"{OUT}/pca_all_frames.json", "w"), indent=2)
print(f"\nwrote {OUT}/pca_all_frames.npz")
