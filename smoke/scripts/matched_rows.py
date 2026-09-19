"""Extract the coordinate-carrying ('matched') sample rows.

Trajectory-derived observables are written once per coordinate frame (every 2500
steps) and forward-filled across the ~10 sample rows that follow (every 250
steps).  Every row therefore carries a weight of its own, but only the first row
of each constant-observable block belongs to the configuration whose Rg it
reports.  This emits those block-start rows, per replica, resetting whenever the
step counter falls (a new run segment, since the file carries no segment column).
"""
import numpy as np

CSV = ("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/"
       "adaptive_production/pmf_analysis/rg_samples_with_weights.csv")
OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"

last_rg, last_step = {}, {}
m_step, m_rep, m_win, m_cv, m_rg, m_wu, m_wg = [], [], [], [], [], [], []
n_rows = 0
all_w = 0.0
with open(CSV) as fh:
    fh.readline()
    for line in fh:
        p = line.split(",")
        step = int(p[0]); rep = int(p[1])
        rg = float(p[4])
        n_rows += 1
        prev_step = last_step.get(rep)
        new_seg = prev_step is not None and step < prev_step
        if new_seg:
            last_rg.pop(rep, None)
        last_step[rep] = step
        if last_rg.get(rep) != rg:                 # first row of a new block
            last_rg[rep] = rg
            m_step.append(step); m_rep.append(rep); m_win.append(int(p[2]))
            m_cv.append(float(p[3])); m_rg.append(rg)
            m_wu.append(float(p[5])); m_wg.append(float(p[6]))
        if n_rows % 1_000_000 == 0:
            print(f"  {n_rows:,} rows -> {len(m_rg):,} matched", flush=True)

arr = {k: np.asarray(v) for k, v in
       dict(step=m_step, replica=m_rep, window=m_win, cv=m_cv, rg=m_rg,
            w_umbrella=m_wu, w_gamd=m_wg).items()}
print(f"\ntotal rows {n_rows:,}")
print(f"matched (block-start) rows {len(m_rg):,}  = {len(m_rg)/n_rows:.4f} of rows")
print(f"coordinate frames in the archive: 611,536")
print(f"ratio matched/frames: {len(m_rg)/611536:.4f}")
np.savez_compressed(f"{OUT}/matched_rows.npz", **arr)
print(f"wrote {OUT}/matched_rows.npz")
