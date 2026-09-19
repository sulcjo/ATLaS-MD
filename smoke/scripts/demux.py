"""Demultiplex chignolin_7: assign every coordinate frame its thermodynamic state.

The replica index is a continuous MD walker; what changes under exchange is the
STATE (umbrella centre x GaMD rung) it occupies.  The exchange records report
each replica's current window at every attempt step, so the assignment is read
directly rather than replayed from swap moves.

Frame <-> step is exact: every segment restarts its step counter at 1,010,400 and
writes a frame every 2500 steps, so frame i of a segment sits at
step 1,010,400 + (i+1)*2500.
"""
import collections, glob, json, os, re
import numpy as np, pandas as pd

RUN = "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/adaptive_production"
OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
STEP0, DSTEP = 1_010_400, 2500

# ---- state table: window -> (cv centre, gamd lambda) ----
w = pd.read_csv(f"{RUN}/final_active_windows.csv")
lam = dict(zip(w["state_id"].astype(int), w["gamd_lambda"].astype(float)))
cen = dict(zip(w["state_id"].astype(int), w["primary_cv_center"].astype(float)))
print(f"states: {len(lam)}; lambdas {sorted(set(lam.values()))}")

# ---- frames: (segment, replica, frame-in-segment) ----
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
files = [str(x) for x in z["files"]]
sf, fr = z["src_file"], z["src_frame"]
def seg_of(f):
    rp = os.path.realpath(f)
    return rp.split("adaptive_production/")[-1].rsplit("/replica_trajectories", 1)[0]
def rep_of(f):
    return int(re.search(r"replica_(\d+)", f.split("/")[-1]).group(1))
fseg = np.array([seg_of(f) for f in files])
frep = np.array([rep_of(f) for f in files])
# order each (segment, replica)'s files by the resume step encoded in the name
def resume_of(f):
    m = re.search(r"resume_from_(\d+)", f)
    return int(m.group(1)) if m else -1
grp = collections.defaultdict(list)
for i, f in enumerate(files):
    grp[(fseg[i], frep[i])].append(i)
frame_step = np.zeros(sf.size, np.int64)
for k, idxs in grp.items():
    idxs = sorted(idxs, key=lambda i: resume_of(files[i]))
    off = 0
    for i in idxs:
        m = sf == i
        n = int(m.sum())
        frame_step[m] = STEP0 + (off + fr[m] + 1) * DSTEP
        off += n
print(f"frames {sf.size:,} over {len(set(map(tuple, zip(fseg[sf], frep[sf]))))} (segment,replica) series")

# ---- exchange records -> per (segment, replica) step->window observations ----
exf = sorted(glob.glob(f"{RUN}/*/exchanges/*/data.parquet") +
             glob.glob(f"{RUN}/*/*/exchanges/*/data.parquet"))
obs = collections.defaultdict(lambda: ([], []))
acc_tot = acc_n = 0
for f in exf:
    seg = f.split("/exchanges/")[0].split("adaptive_production/")[-1]
    d = pd.read_parquet(f, columns=["step", "replica_i", "replica_j",
                                    "window_i", "window_j", "accepted"])
    acc_tot += int(d["accepted"].sum()); acc_n += len(d)
    for a, b in (("replica_i", "window_i"), ("replica_j", "window_j")):
        for rep, sub in d.groupby(a):
            s, wdw = obs[(seg, int(rep))]
            s.append(sub["step"].to_numpy()); wdw.append(sub[b].to_numpy())
print(f"exchange attempts {acc_n:,}, acceptance {acc_tot/acc_n:.4f}")

# ---- assign each frame its window (last observation at or before the frame step) ----
win = np.full(sf.size, -1, np.int32)
miss = 0
for (seg, rep), (slist, wlist) in obs.items():
    st = np.concatenate(slist); wd = np.concatenate(wlist).astype(np.int32)
    o = np.argsort(st, kind="stable"); st, wd = st[o], wd[o]
    uniq, first = np.unique(st, return_index=True)
    st, wd = uniq, wd[first]
    m = (fseg[sf] == seg) & (frep[sf] == rep)
    if not m.any():
        continue
    pos = np.searchsorted(st, frame_step[m], side="right") - 1
    ok = pos >= 0
    idx = np.where(m)[0]
    win[idx[ok]] = wd[pos[ok]]
    miss += int((~ok).sum())
got = win >= 0
print(f"frames assigned a state: {got.sum():,} / {win.size:,} ({got.mean():.4f}); "
      f"before first record: {miss:,}")
lam_of = np.array([lam.get(i, np.nan) for i in range(64)])
flam = np.where(got, lam_of[np.clip(win, 0, 63)], np.nan)
print("\ntime spent per GaMD rung (assigned frames):")
for L in sorted(set(lam.values())):
    print(f"  lambda={L:.3f}: {np.sum(np.isclose(flam, L)):8,d} frames "
          f"({np.mean(np.isclose(flam[got], L)):.3f})")
np.savez_compressed(f"{OUT}/demux.npz", window=win, lam=flam, frame_step=frame_step,
                    seg=fseg[sf], rep=frep[sf])
json.dump({"n_frames": int(win.size), "assigned": int(got.sum()),
           "acceptance": float(acc_tot/acc_n), "attempts": int(acc_n)},
          open(f"{OUT}/demux.json", "w"), indent=2)
print(f"\nwrote {OUT}/demux.npz")
