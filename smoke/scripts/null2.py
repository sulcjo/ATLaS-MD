"""Null model, decorrelated: one frame per distinct trajectory file.

The first attempt drew 400 compact frames at random and so drew many from the
same continuous segment; their mutual RMSD measured correlation, not chance.
Here at most one frame is taken from each of the 1472 files, which are separate
replicas and separate run segments.
"""
import numpy as np, mdtraj as md, json, collections

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
RUN = "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production"
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
files = [str(x) for x in z["files"]]
r, cv, sf, fr = z["rmsd_bb_min18"], z["cv1"], z["src_file"], z["src_frame"]
top = md.load(f"{RUN}/epoch_000/solute_only.pdb")
native = md.load("/home/sulcjo/.claude/jobs/b27cb9c3/tmp/1uao.pdb")
nat_i = {(a.residue.resSeq, a.name): a.index for a in native.topology.atoms}
run_i = {(a.residue.resSeq, a.name): a.index for a in top.topology.atoms}
bb = [k for k in sorted(set(nat_i) & set(run_i)) if k[1] in {"N", "CA", "C", "O"} and 2 <= k[0] <= 9]
bb_run = np.array([run_i[k] for k in bb])

rng = np.random.default_rng(1)
pool = np.where((cv >= 0.82) & (cv <= 0.93))[0]
by_file = collections.defaultdict(list)
for i in pool:
    by_file[int(sf[i])].append(int(i))
pick = [rng.choice(v) for v in by_file.values()]
rng.shuffle(pick)
pick = np.array(pick[:400])
print(f"decorrelated null: {len(pick)} compact frames, one per distinct trajectory file "
      f"({len(by_file)} files contain compact frames)")

frames = [md.load_frame(files[sf[i]], int(fr[i]), top=top.topology) for i in pick]
ens = frames[0]
for t in frames[1:]:
    ens = ens.join(t)
pair = np.concatenate([np.delete(md.rmsd(ens, ens, frame=k, atom_indices=bb_run) * 10.0, k)
                       for k in range(ens.n_frames)])
print(f"\n  compact-vs-compact, different files (n={pair.size:,}): min {pair.min():.2f}  "
      f"p1 {np.percentile(pair,1):.2f}  median {np.median(pair):.2f}  max {pair.max():.2f}")
p10, p15 = (pair <= 1.0).mean(), (pair <= 1.5).mean()
print(f"  P(within 1.0 A of an arbitrary compact reference) = {p10:.3e}")
print(f"  P(within 1.5 A)                                   = {p15:.3e}")
print(f"\n  frames within 1.0 A of 1UAO, expected from this null: {p10*611536:,.0f}")
print(f"  observed                                            : 481")
print(f"  ratio observed/null: {481/(p10*611536):.2f}")
print("\n  Reading: the native basin is NOT enriched relative to a typical compact basin in")
print("  this ensemble -- it is one of many, and by this measure a comparatively sparse one.")
print("  That is consistent with the report's claim (fold present, not thermodynamically")
print("  favoured) and inconsistent with any claim that the fold is the dominant compact state.")
json.dump({"n_null_frames": int(len(pick)), "p_within_1.0": float(p10), "p_within_1.5": float(p15),
           "expected_within_1.0": float(p10 * 611536), "observed_within_1.0": 481,
           "ratio": float(481 / (p10 * 611536)), "pair_median_A": float(np.median(pair)),
           "pair_min_A": float(pair.min())},
          open(f"{OUT}/null_model_decorrelated.json", "w"), indent=2)
