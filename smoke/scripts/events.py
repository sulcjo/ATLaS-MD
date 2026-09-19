"""Folding events along continuous walker trajectories, Satoh's sufficiency statistic.

Satoh 2006 counts transitions from unfolded (Calpha RMSD >= 4.0 A) into native
(< 1.0 A) and reports 159 as "statistically sufficient".  Here the same count is
made along each continuous (segment, replica) MD series -- the replica index is a
continuous walker, so these series are genuine trajectories, though under a
time-varying bias.
"""
import collections, json
import numpy as np

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
d = np.load(f"{OUT}/demux.npz", allow_pickle=True)
r = z["rmsd_bb_min18"].astype(float)          # backbone res2-9, min over 18 models
rca = z["rmsd_ca_m1"].astype(float)           # CA res1-10 vs model 1
q = z["q"]; step = d["frame_step"]; seg = d["seg"]; rep = d["rep"]; lam = d["lam"]; win = d["window"]

def count_events(metric, nat, unf, mask=None):
    """U->N transitions per continuous series, with hysteresis (must fully unfold first)."""
    ev = 0; series = 0; dwell = []
    for key in set(zip(seg, rep)):
        m = (seg == key[0]) & (rep == key[1])
        if mask is not None:
            m &= mask
        if m.sum() < 5:
            continue
        o = np.argsort(step[m]); v = metric[m][o]
        series += 1
        state = None; run = 0
        for x in v:
            if x < nat:
                if state == "U":
                    ev += 1
                    state = "N"; run = 1
                elif state == "N":
                    run += 1
                else:
                    state = "N"; run = 1
            elif x >= unf:
                if state == "N" and run:
                    dwell.append(run)
                state = "U"; run = 0
            else:
                if state == "N":
                    run += 1
    return ev, series, dwell

print("FOLDING EVENTS (unfolded -> native) along continuous walker series\n")
print(f"{'definition':52s} {'events':>8s} {'series':>7s} {'median dwell':>13s}")
for lab, met, nat, unf, msk in [
    ("Satoh thresholds on CA RMSD (<1.0 / >=4.0 A)", rca, 1.0, 4.0, None),
    ("backbone res2-9, min over 18 (<1.0 / >=4.0 A)", r, 1.0, 4.0, None),
    ("looser native (<1.5 / >=4.0 A)", r, 1.5, 4.0, None),
    ("at lambda=0 only (unboosted rung)", r, 1.5, 4.0, np.isclose(lam, 0.0)),
]:
    ev, ns, dw = count_events(met, nat, unf, msk)
    md = f"{np.median(dw)*8.75:.0f} ps" if dw else "-"
    print(f"{lab:52s} {ev:8,d} {ns:7d} {md:>13s}")
print("\n  (Satoh reports 159 folding events from 180 ns of multicanonical MD)")

# ---- ladder mixing: round trips in lambda per walker ----
rt = []
for key in set(zip(seg, rep)):
    m = (seg == key[0]) & (rep == key[1])
    if m.sum() < 10: continue
    o = np.argsort(step[m]); L = lam[m][o]
    top = L >= 0.99; bot = L <= 0.01
    seqs = np.where(top, 1, np.where(bot, -1, 0))
    seqs = seqs[seqs != 0]
    if seqs.size:
        rt.append(int(np.sum(np.diff(seqs) != 0) // 2))
rt = np.array(rt)
print(f"\nLAMBDA-LADDER ROUND TRIPS (lambda=0 <-> lambda=1) per walker series")
print(f"  series {rt.size}, total round trips {rt.sum():,}, "
      f"median {np.median(rt):.0f}, min {rt.min()}, max {rt.max()}")
print(f"  series with zero round trips: {(rt==0).sum()} ({(rt==0).mean():.3f})")

# ---- is the native basin reached while at lambda=0? ----
nat15 = r <= 1.5
for L in (0.0, 0.232349, 0.642932, 1.0):
    m = np.isclose(lam, L)
    print(f"  lambda={L:.3f}: {int((nat15&m).sum()):6,d} native frames of {int(m.sum()):7,d} "
          f"({100*(nat15&m).sum()/max(m.sum(),1):.3f}%)")
json.dump({"rt_total": int(rt.sum()), "rt_median": float(np.median(rt)),
           "rt_zero_fraction": float((rt==0).mean())},
          open(f"{OUT}/events.json", "w"), indent=2)
