import collections
import json
import re

import numpy as np

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
files = [str(x) for x in z["files"]]
r, q, sf, fr = z["rmsd_bb_min18"], z["q"], z["src_file"], z["src_frame"]


def replica(path):
    return int(re.search(r"replica_(\d+)", path.split("/")[-1]).group(1))


def segment(path):
    tgt = path  # symlink target holds the epoch/topup segment
    import os
    real = os.path.realpath(tgt)
    parts = real.split("/")
    return "/".join(parts[-4:-2])


for thr in (1.0, 1.5, 2.0):
    m = r <= thr
    reps = collections.Counter(replica(files[i]) for i in sf[m])
    segs = collections.Counter(segment(files[i]) for i in sf[m])
    fls = collections.Counter(int(i) for i in sf[m])
    print(f"\n=== bb RMSD <= {thr} A : {int(m.sum()):,} frames ===")
    print(f"  distinct replicas {len(reps)}/64, distinct trajectory files {len(fls)}/1472, "
          f"distinct run segments {len(segs)}")
    print("  replicas:", dict(sorted(reps.items(), key=lambda kv: -kv[1])[:8]))
    print("  segments:", dict(sorted(segs.items(), key=lambda kv: -kv[1])[:6]))

# Contiguity: how many independent visits, treating consecutive frames in one
# file as one visit.
m = r <= 1.5
idx = np.where(m)[0]
visits = []
cur = [idx[0]]
for a, b in zip(idx[:-1], idx[1:]):
    if sf[a] == sf[b] and fr[b] - fr[a] <= 2:
        cur.append(b)
    else:
        visits.append(cur)
        cur = [b]
visits.append(cur)
lens = np.array([len(v) for v in visits])
print(f"\n=== independent visits below 1.5 A (frames every 2500 steps = 5 ps) ===")
print(f"  {len(visits)} visits, length frames: median {np.median(lens):.0f}, max {lens.max()}, "
      f"total {lens.sum()}")
print(f"  visits in >1 file: {len(set(int(sf[v[0]]) for v in visits))} distinct files")
long = sorted(visits, key=len, reverse=True)[:5]
for v in long:
    f = files[sf[v[0]]].split("/")[-1]
    print(f"    {len(v):5d} frames ({len(v)*5:6d} ps)  {f}  frames {fr[v[0]]}-{fr[v[-1]]}  "
          f"best rmsd {r[v].min():.2f}  best Q {q[v].max():.3f}")

json.dump({"n_le_1.0": int((r <= 1.0).sum()), "n_le_1.5": int((r <= 1.5).sum()),
           "n_le_2.0": int((r <= 2.0).sum()), "n_visits_1.5": len(visits),
           "n_files_1.5": len(set(int(sf[v[0]]) for v in visits)),
           "min_rmsd": float(r.min()), "max_q": float(q.max())},
          open(f"{OUT}/locality.json", "w"), indent=2)
