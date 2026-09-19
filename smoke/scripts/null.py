"""Null model: could 0.43 A backbone RMSD to 1UAO arise by chance from a compact ensemble?

The honest null is not a random coil -- umbrella sampling along a contact CV
deliberately produces compact structures.  It is: two UNRELATED compact
structures from this same run, compared to each other.  If the run's own compact
frames are typically several angstrom apart, then 0.43 A to an external
reference cannot be explained by compactness alone.
"""
import numpy as np, mdtraj as md, json

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
RUN = "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production"
TOP = f"{RUN}/epoch_000/solute_only.pdb"
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
files = [str(x) for x in z["files"]]
r, cv, sf, fr = z["rmsd_bb_min18"], z["cv1"], z["src_file"], z["src_frame"]

top = md.load(TOP)
native = md.load("/home/sulcjo/.claude/jobs/b27cb9c3/tmp/1uao.pdb")
nat_i = {(a.residue.resSeq, a.name): a.index for a in native.topology.atoms}
run_i = {(a.residue.resSeq, a.name): a.index for a in top.topology.atoms}
bb = [k for k in sorted(set(nat_i) & set(run_i)) if k[1] in {"N", "CA", "C", "O"} and 2 <= k[0] <= 9]
bb_run = np.array([run_i[k] for k in bb]); bb_nat = np.array([nat_i[k] for k in bb])

rng = np.random.default_rng(0)
pool = np.where((cv >= 0.82) & (cv <= 0.93))[0]
pick = rng.choice(pool, 400, replace=False)
frames = []
for i in pick:
    frames.append(md.load_frame(files[sf[i]], int(fr[i]), top=top.topology))
ens = frames[0]
for t in frames[1:]:
    ens = ens.join(t)
print(f"null ensemble: {ens.n_frames} compact frames, CV1 in the native window [0.82,0.93]")

pair = []
for k in range(ens.n_frames):
    d = md.rmsd(ens, ens, frame=k, atom_indices=bb_run) * 10.0
    pair.append(np.delete(d, k))
pair = np.concatenate(pair)
tonat = r[pick]
print("\nBACKBONE RMSD (res 2-9), compact frames only")
print(f"  compact-vs-compact (unrelated pairs, n={pair.size:,}):  "
      f"min {pair.min():.2f}  p0.1 {np.percentile(pair,0.1):.2f}  p1 {np.percentile(pair,1):.2f}  "
      f"median {np.median(pair):.2f}  max {pair.max():.2f}")
print(f"  same frames vs 1UAO (n={tonat.size}):                   "
      f"min {tonat.min():.2f}  p1 {np.percentile(tonat,1):.2f}  median {np.median(tonat):.2f}")
print(f"  P(unrelated compact pair within 1.0 A) = {(pair<=1.0).mean():.3e}")
print(f"  P(unrelated compact pair within 1.5 A) = {(pair<=1.5).mean():.3e}")
exp10 = (pair <= 1.0).mean() * 611536
print(f"\n  Expected frames within 1.0 A of an ARBITRARY compact reference: {exp10:,.0f}")
print(f"  Observed frames within 1.0 A of 1UAO:                          481")
print(f"  enrichment over the compact null: {481/max(exp10,1e-9):.1f}x")

print("\nMIN-OVER-18 INFLATION: counts against each single NMR model")
per = []
for k in range(native.n_frames):
    pass
single = {}
for thr in (1.0, 1.5, 2.0):
    n_min18 = int((r <= thr).sum())
    n_m1 = int((z["rmsd_bb_m1"] <= thr).sum())
    single[thr] = (n_min18, n_m1, n_min18 / max(n_m1, 1))
    print(f"  <= {thr} A : min-over-18 {n_min18:,}   vs model 1 only {n_m1:,}   inflation {single[thr][2]:.2f}x")

json.dump({"null_pair_min": float(pair.min()), "null_p_within_1.0": float((pair <= 1.0).mean()),
           "null_p_within_1.5": float((pair <= 1.5).mean()),
           "null_median": float(np.median(pair)),
           "expected_within_1.0_by_chance": float(exp10), "observed": 481,
           "enrichment": float(481 / max(exp10, 1e-9)),
           "min18_inflation": {str(k): v[2] for k, v in single.items()}},
          open(f"{OUT}/null_model.json", "w"), indent=2)
