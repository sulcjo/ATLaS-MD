"""Rank ab initio CV candidates by folded/unfolded discrimination.

Every candidate is constructible WITHOUT the native structure.  RMSD to 1UAO is
used only as the LABEL (folded <= 1.5 A, unfolded >= 4.0 A, Satoh-style), never
as an input.  Torsion features follow the two-CV design doc: sin/cos of every
internal backbone phi/psi (18 angles -> 36 features), PCA plain and residualized
against the scalar contact CV.
"""
import glob, json, re
import mdtraj as md, numpy as np

RUN = "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production"
OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
top = md.load(f"{RUN}/epoch_000/solute_only.pdb")
topo = top.topology
files = sorted(glob.glob(f"{RUN}/_merged_replica_trajectories/*.xtc"))
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
assert [str(x) for x in z["files"]] == files

# backbone H-bond candidate pairs: N(i)...O(j), |i-j| >= 3, both backbone
N = {r.resSeq: topo.select(f"resSeq {r.resSeq} and name N") for r in topo.residues}
O = {r.resSeq: topo.select(f"resSeq {r.resSeq} and name O") for r in topo.residues}
hb_pairs = np.array([[N[i][0], O[j][0]] for i in N for j in O
                     if len(N[i]) and len(O[j]) and abs(i - j) >= 3])
print(f"{len(hb_pairs)} nonlocal backbone N...O pairs")

phi_idx, _ = md.compute_phi(top)
psi_idx, _ = md.compute_psi(top)
print(f"{len(phi_idx)} phi + {len(psi_idx)} psi = {len(phi_idx)+len(psi_idx)} angles "
      f"-> {2*(len(phi_idx)+len(psi_idx))} sin/cos features")

feats, nhb, alpha, beta = [], [], [], []
for n, f in enumerate(files):
    t = md.load(f, top=topo)
    if t.n_frames == 0:
        continue
    phi = md.compute_phi(t)[1]; psi = md.compute_psi(t)[1]
    ang = np.concatenate([phi, psi], axis=1)
    feats.append(np.concatenate([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32))
    d = md.compute_distances(t, hb_pairs) * 10.0
    nhb.append((d < 3.5).sum(axis=1).astype(np.float32))
    # generic Ramachandran basin occupancy -- no native reference
    ph, ps = np.degrees(phi), np.degrees(psi)
    a = ((ph > -160) & (ph < -20) & (ps > -120) & (ps < 50)).mean(axis=1)
    b = ((ph > -180) & (ph < -20) & ((ps > 90) | (ps < -150))).mean(axis=1)
    alpha.append(a.astype(np.float32)); beta.append(b.astype(np.float32))
    if n % 300 == 0:
        print(f"  {n}/{len(files)}", flush=True)

X = np.concatenate(feats); del feats
nhb = np.concatenate(nhb); alpha = np.concatenate(alpha); beta = np.concatenate(beta)
print(f"features {X.shape}")
cv1 = z["cv1"].astype(np.float64)

def pca(M, k=6):
    mu = M.mean(axis=0); Mc = M - mu
    C = (Mc.T @ Mc) / (len(Mc) - 1)
    w, v = np.linalg.eigh(C)
    o = np.argsort(w)[::-1]
    return (Mc @ v[:, o[:k]]), w[o] / w.sum()

Xd = X.astype(np.float64)
plain, evr_p = pca(Xd)
# residualize each feature on scalar CV1 (quadratic), per the design doc
A = np.vstack([np.ones_like(cv1), cv1, cv1 ** 2]).T
coef, *_ = np.linalg.lstsq(A, Xd, rcond=None)
R = Xd - A @ coef
resid, evr_r = pca(R)
print(f"plain EVR[1-6]   {np.round(evr_p[:6],3)}")
print(f"residual EVR[1-6] {np.round(evr_r[:6],3)}")
np.savez_compressed(f"{OUT}/cv_candidates.npz", plain=plain.astype(np.float32),
                    resid=resid.astype(np.float32), nhb=nhb, alpha=alpha, beta=beta,
                    evr_plain=evr_p[:10], evr_resid=evr_r[:10])
print("wrote cv_candidates.npz")
