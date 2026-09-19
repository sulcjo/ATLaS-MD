"""Matched decoy null for Q -- the statistic that actually carries conclusion (b).

The earlier null tested RMSD only.  Q is reference-specific, which is the whole
argument for it, so the null must ask: if the reference were an arbitrary compact
structure set rather than 1UAO, how often would a frame reach Q >= 0.80?
Decoy contact maps are defined by exactly the same rule as 1UAO's: heavy-atom
pairs |i-j| >= 3 within 4.5 A in at least half of the 18 reference structures.
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
print(f"{len(heavy)} heavy atoms, {len(pidx)} candidate pairs (|i-j| >= 3)")

rng = np.random.default_rng(3)
pool = np.where((cv >= 0.82) & (cv < 0.93))[0]
by_file = collections.defaultdict(list)
for i in pool:
    by_file[int(sf[i])].append(int(i))
fids = list(by_file)

# test population: up to 20 compact frames per file
test_idx = np.concatenate([rng.choice(by_file[f], min(20, len(by_file[f])), replace=False)
                           for f in fids])
print(f"test population: {test_idx.size:,} compact frames from {len(fids)} files")
def load(idx):
    t = [md.load_frame(files[sf[i]], int(fr[i]), top=top.topology) for i in idx]
    out = t[0]
    for x in t[1:]: out = out.join(x)
    return out
test = load(test_idx)
D = md.compute_distances(test, pidx)          # nm
print(f"distance matrix {D.shape}")

def qfrac(D, mask, r0):
    return (1.0 / (1.0 + np.exp(50.0 * (D[:, mask] - 1.2 * r0)))).mean(axis=1)

# observed, against 1UAO -- recomputed here on the same test population
qn = q[test_idx]
obs80 = float((qn >= 0.80).mean()); obs70 = float((qn >= 0.70).mean())

hits80, hits70, ncontacts = [], [], []
for trial in range(30):
    dec_files = rng.choice(fids, 18, replace=False)
    dec = load([rng.choice(by_file[f]) for f in dec_files])
    Dd = md.compute_distances(dec, pidx)
    mask = (Dd < 0.45).mean(axis=0) >= 0.5
    if mask.sum() < 10:
        continue
    r0 = Dd[:, mask].mean(axis=0)
    keep = ~np.isin(sf[test_idx], dec_files)       # hold out the decoys' own files
    qd = qfrac(D[keep], mask, r0)
    hits80.append((qd >= 0.80).mean()); hits70.append((qd >= 0.70).mean())
    ncontacts.append(int(mask.sum()))
h80, h70 = np.array(hits80), np.array(hits70)
print(f"\ndecoy contact-map size: median {int(np.median(ncontacts))} pairs (1UAO: 65)")
print(f"{'reference set':34s} {'P(Q>=0.80)':>12s} {'P(Q>=0.70)':>12s}")
print(f"{'18 compact decoys from the run':34s} {h80.mean():12.4f} {h70.mean():12.4f}"
      f"   (sd {h80.std():.4f}, {len(h80)} trials)")
print(f"{'18 native NMR models (1UAO)':34s} {obs80:12.4f} {obs70:12.4f}")
print(f"\nA decoy reference set is matched at Q>=0.80 {h80.mean()/max(obs80,1e-12):.0f}x more often "
      f"than 1UAO is." if obs80 > 0 else
      f"\nNo test frame reaches Q>=0.80 against 1UAO, while decoys are matched at {h80.mean():.4f}.")

# Poisson interval on the 61x RMSD ratio, which rested on ~2 native events
from math import sqrt
n_ev = 0.0040 * 496
lo, hi = 0.0, 0.0
try:
    from scipy.stats import chi2
    lo = chi2.ppf(0.025, 2 * n_ev) / 2 if n_ev > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * (n_ev + 1)) / 2
except Exception:
    lo, hi = n_ev - 1.96 * sqrt(n_ev), n_ev + 1.96 * sqrt(n_ev)
print(f"\n61x RMSD ratio: it rests on ~{n_ev:.1f} native events in 496 test frames.")
print(f"  Poisson 95% CI on the event count: [{lo:.2f}, {hi:.2f}]")
print(f"  -> ratio 95% CI approx [{0.2469/ (hi/496):.0f}x, "
      f"{(0.2469/(lo/496)) if lo>0 else float('inf'):.0f}x]  (point estimate 61x)")
json.dump({"test_frames": int(test_idx.size), "decoy_P_q80": float(h80.mean()),
           "decoy_P_q80_sd": float(h80.std()), "decoy_P_q70": float(h70.mean()),
           "native_P_q80": obs80, "native_P_q70": obs70,
           "decoy_contacts_median": int(np.median(ncontacts)), "n_trials": int(len(h80)),
           "rmsd61x_events": float(n_ev), "rmsd61x_ci_events": [float(lo), float(hi)]},
          open(f"{OUT}/q_decoy_null.json", "w"), indent=2)
