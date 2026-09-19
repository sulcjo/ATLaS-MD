"""Matched null, corrected: BOTH rates on the SAME held-out test set.

The previous version compared a compact-conditioned null probability against an
observed rate computed over the whole archive (90% of which is not compact).
Different denominators, so the ratio was not matched -- the very error this null
exists to avoid.  Here one test set of compact frames (one per trajectory file)
is scored twice with the identical min-over-18 structure; only the reference set
differs: 18 NMR models vs 18 compact decoys drawn from the run itself.
"""
import collections, json
import mdtraj as md, numpy as np

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

rng = np.random.default_rng(7)
pool = np.where((cv >= 0.82) & (cv <= 0.93))[0]
by_file = collections.defaultdict(list)
for i in pool:
    by_file[int(sf[i])].append(int(i))
file_ids = list(by_file)
test_pick = np.array([rng.choice(by_file[f]) for f in file_ids])
test_frames = [md.load_frame(files[sf[i]], int(fr[i]), top=top.topology) for i in test_pick]
test = test_frames[0]
for t in test_frames[1:]:
    test = test.join(t)
print(f"test set: {test.n_frames} compact frames, one per trajectory file "
      f"(CV1 in the native window)")

hits10, hits15, kept = [], [], []
for trial in range(40):
    ref_files = rng.choice(file_ids, 18, replace=False)
    refs = []
    for f in ref_files:
        j = rng.choice(by_file[f])
        refs.append(md.load_frame(files[sf[j]], int(fr[j]), top=top.topology))
    ref = refs[0]
    for t in refs[1:]:
        ref = ref.join(t)
    keep = ~np.isin(sf[test_pick], ref_files)
    d = np.stack([md.rmsd(test, ref, frame=k, atom_indices=bb_run, ref_atom_indices=bb_run) * 10.0
                  for k in range(ref.n_frames)]).min(axis=0)[keep]
    hits10.append((d <= 1.0).mean()); hits15.append((d <= 1.5).mean()); kept.append(keep.sum())
h10, h15 = np.array(hits10), np.array(hits15)

# SAME test set, scored against the 18 NMR models (already stored per frame).
nat10 = (r[test_pick] <= 1.0).mean()
nat15 = (r[test_pick] <= 1.5).mean()

print("\nBOTH rates on the same test set, same min-over-18 structure:")
print(f"{'reference set':34s} {'P(<=1.0 A)':>12s} {'P(<=1.5 A)':>12s}")
print(f"{'18 compact decoys from this run':34s} {h10.mean():12.4f} {h15.mean():12.4f}"
      f"   (sd {h10.std():.3f}, 40 trials, ~{int(np.mean(kept))} frames each)")
print(f"{'18 native NMR models (1UAO)':34s} {nat10:12.4f} {nat15:12.4f}")
print(f"\nmatched ratio decoy/native: {h10.mean()/max(nat10,1e-9):.1f}x at 1.0 A, "
      f"{h15.mean()/max(nat15,1e-9):.1f}x at 1.5 A")
print("  -> among COMPACT structures, an arbitrary 18-decoy set is matched this many times")
print("     more often than the true native set. The native basin is that much sparser.")

# For reference: the whole-archive rate, which must NOT be mixed with the above.
print(f"\n  (whole-archive P(<=1.0 A of 1UAO) = {(r<=1.0).mean():.3e}; this is conditioned on a")
print("   different population and must not be divided by the compact-conditioned null.)")
comp = (cv >= 0.82) & (cv < 0.93)
print(f"  compact-conditioned observed rate = {(r[comp]<=1.0).mean():.3e} "
      f"({int((r[comp]<=1.0).sum())} of {int(comp.sum()):,})")

json.dump({"test_set_n": int(test.n_frames),
           "decoy_P_within_1.0": float(h10.mean()), "decoy_sd": float(h10.std()),
           "decoy_P_within_1.5": float(h15.mean()),
           "native_P_within_1.0_same_test_set": float(nat10),
           "native_P_within_1.5_same_test_set": float(nat15),
           "matched_ratio_1.0": float(h10.mean() / max(nat10, 1e-9)),
           "matched_ratio_1.5": float(h15.mean() / max(nat15, 1e-9)),
           "compact_conditioned_observed_rate": float((r[comp] <= 1.0).mean()),
           "whole_archive_observed_rate": float((r <= 1.0).mean()),
           "supersedes": "the 324x figure, which divided a compact-conditioned null by a whole-archive observed rate",
           "n_trials": 40},
          open(f"{OUT}/matched_null_corrected.json", "w"), indent=2)
