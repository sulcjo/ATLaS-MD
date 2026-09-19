"""Matched Q decoy null, corrected for reference-set tightness.

First attempt drew 18 unrelated compact frames as a decoy 'ensemble'.  Their
consensus contact map had ~15 pairs against 1UAO's 65, because a loose reference
set agrees on almost nothing -- so Q was trivially satisfiable and the null was
unmatched in the decoys' favour.  1UAO is a TIGHT ensemble (model-model backbone
RMSD 0.77 A mean, 1.37 A max).  Each decoy set here is therefore a cluster: a
seed frame plus its nearest neighbours drawn from other files, selected to
reproduce that internal spread, so the decoy map is comparable in size.
"""
import collections, json
import mdtraj as md, numpy as np

RUN = "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production"
OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
files = [str(x) for x in z["files"]]; sf, fr = z["src_file"], z["src_frame"]
r, q, cv = z["rmsd_bb_min18"], z["q"], z["cv1"]
top = md.load(f"{RUN}/epoch_000/solute_only.pdb")
heavy = top.topology.select("mass > 1.5")
res = np.array([top.topology.atom(int(a)).residue.resSeq for a in heavy])
pairs = np.array([(i, j) for i in range(len(heavy)) for j in range(i + 1, len(heavy))
                  if abs(res[i] - res[j]) >= 3])
pidx = np.array([(heavy[i], heavy[j]) for i, j in pairs])
bb = top.topology.select("name N CA C O and resSeq 2 to 9")

rng = np.random.default_rng(5)
pool = np.where((cv >= 0.82) & (cv < 0.93))[0]
by_file = collections.defaultdict(list)
for i in pool:
    by_file[int(sf[i])].append(int(i))
fids = list(by_file)

def load(idx):
    t = [md.load_frame(files[sf[i]], int(fr[i]), top=top.topology) for i in idx]
    out = t[0]
    for x in t[1:]: out = out.join(x)
    return out

# candidate pool for building clusters: one frame per file
cand_idx = np.array([rng.choice(by_file[f]) for f in fids])
cand = load(cand_idx)
print(f"cluster candidate pool: {cand.n_frames} frames (one per file)")

test_idx = np.concatenate([rng.choice(by_file[f], min(20, len(by_file[f])), replace=False)
                           for f in fids])
test = load(test_idx)
D = md.compute_distances(test, pidx)
qn = q[test_idx]
print(f"test population: {test.n_frames:,} compact frames")

hits80, hits70, nc, spread = [], [], [], []
trials = 0
for t in range(60):
    s = rng.integers(cand.n_frames)
    d = md.rmsd(cand, cand, frame=int(s), atom_indices=bb) * 10.0
    near = np.argsort(d)[:18]                       # the 18 mutually closest = a tight set
    if d[near].max() > 2.0:                         # require NMR-like tightness
        continue
    dec = cand[near]
    Dd = md.compute_distances(dec, pidx)
    mask = (Dd < 0.45).mean(axis=0) >= 0.5
    if mask.sum() < 30:
        continue
    r0 = Dd[:, mask].mean(axis=0)
    keep = ~np.isin(sf[test_idx], sf[cand_idx[near]])
    qd = (1.0 / (1.0 + np.exp(50.0 * (D[keep][:, mask] - 1.2 * r0)))).mean(axis=1)
    hits80.append((qd >= 0.80).mean()); hits70.append((qd >= 0.70).mean())
    nc.append(int(mask.sum())); spread.append(float(d[near].max())); trials += 1
    if trials >= 30:
        break
h80, h70 = np.array(hits80), np.array(hits70)
print(f"\n{trials} matched decoy sets")
print(f"  decoy contact-map size: median {int(np.median(nc))} pairs   (1UAO: 65)")
print(f"  decoy set internal spread: median max-RMSD {np.median(spread):.2f} A   "
      f"(1UAO: 1.37 A max)")
print(f"\n{'reference set':36s} {'P(Q>=0.80)':>12s} {'P(Q>=0.70)':>12s}")
print(f"{'18 tight compact decoys from the run':36s} {h80.mean():12.5f} {h70.mean():12.5f}"
      f"  (sd {h80.std():.5f})")
print(f"{'18 native NMR models (1UAO)':36s} {(qn>=0.80).mean():12.5f} {(qn>=0.70).mean():12.5f}")
print(f"\nOn this compact test population no frame reaches Q>=0.80 against 1UAO "
      f"({int((qn>=0.80).sum())}/{qn.size}),")
print(f"while a matched decoy set is reached {h80.mean()*100:.2f}% of the time.")
print("Reading: Q is NOT hard to satisfy for an arbitrary tight reference either, so Q alone")
print("does not make 1UAO special -- what is special is that the 33 archive-wide Q>=0.80 frames")
print("ALSO satisfy backbone RMSD <= 1.0 A and the Tyr2/Trp9 cluster against that same reference.")
json.dump({"n_trials": trials, "decoy_contacts_median": int(np.median(nc)),
           "decoy_spread_median_A": float(np.median(spread)),
           "decoy_P_q80": float(h80.mean()), "decoy_P_q80_sd": float(h80.std()),
           "decoy_P_q70": float(h70.mean()),
           "native_P_q80_on_test": float((qn >= 0.80).mean()),
           "native_P_q70_on_test": float((qn >= 0.70).mean()),
           "note": "supersedes q_decoy_null.json, whose decoy sets were untightened (15-pair maps)"},
          open(f"{OUT}/q_decoy_null_matched.json", "w"), indent=2)
