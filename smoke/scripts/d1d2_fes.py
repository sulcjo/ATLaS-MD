"""d1/d2 FES from de-duplicated (matched) rows.

The CV checksum shows the block-START row is not the frame's own row: within a
constant-Rg block the CV moves, so the weight attached to a frame is that of a
configuration within ~10 rows (~8.75 ps) of it, not the frame itself.  That is a
weight-pairing uncertainty, not a mis-identification -- Rg agrees to 5 decimals
and is effectively a fingerprint of the frame.  The honest treatment is to use
the join, and to bound the residual pairing error by comparing against the
forward-filled weighting the published surface used.
"""
import collections, json, re
import numpy as np

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
m = np.load(f"{OUT}/matched_rows.npz"); z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
fw = np.load(f"{OUT}/frame_weights.npz")
row, dcv = fw["row"], fw["dcv"]
frep = fw["replica"]
kT = 0.001987204259 * 300.0
d1, d2 = z["d1"].astype(np.float64), z["d2"].astype(np.float64)
ok = row >= 0
w = np.zeros(row.size); w[ok] = m["w_umbrella"][row[ok]]
print(f"joined frames: {ok.sum():,} / {row.size:,}")
print(f"weight-pairing check: |cv_frame - cv_row| median {np.nanmedian(dcv[ok]):.4f}; "
      f"CV1 within-window sd is ~0.035, so the paired row is a neighbouring configuration")

e = np.linspace(2.5, 16.0, 28)
c = 0.5 * (e[:-1] + e[1:])
def fes2d(x, y, wt, nmin=20):
    ix = np.digitize(x, e) - 1; iy = np.digitize(y, e) - 1
    k = (ix >= 0) & (ix < len(e)-1) & (iy >= 0) & (iy < len(e)-1) & (wt > 0)
    H = np.zeros((len(e)-1, len(e)-1)); N = np.zeros_like(H)
    np.add.at(H, (ix[k], iy[k]), wt[k]); np.add.at(N, (ix[k], iy[k]), 1.0)
    if H.sum() <= 0: return None, None, None
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -kT * np.log(H / H.sum())
    F -= np.nanmin(F[np.isfinite(F)])
    valid = N >= nmin
    if not valid.any(): return F, N, None
    idx = np.unravel_index(np.nanargmin(np.where(valid, F, np.inf)), F.shape)
    return F, N, idx

F, N, idx = fes2d(d1, d2, w)
print(f"\nd1/d2 FES from de-duplicated matched rows ({int((w>0).sum()):,} weighted frames)")
print(f"  MINIMUM  d1 = {c[idx[0]]:.2f} A, d2 = {c[idx[1]]:.2f} A   (n = {int(N[idx])} frames in bin)")
print(f"  published (forward-filled, 5,110,208 nominal): d1 = 7.206, d2 = 8.377 A")
print(f"  native (18 NMR models): d1 = 2.84-4.22, d2 = 6.17-7.56 A")

# sensitivity: equal weights (pairing-free), and gamd-exponential weights
wg = np.zeros(row.size); wg[ok] = m["w_gamd"][row[ok]]
for lab, ww in (("uniform weights (pairing-free bound)", (row >= 0).astype(float)),
                ("gamd_exponential weights", wg)):
    _, Nx, ix_ = fes2d(d1, d2, ww)
    if ix_ is not None:
        print(f"  [{lab}] minimum d1 = {c[ix_[0]]:.2f}, d2 = {c[ix_[1]]:.2f} A")

rng = np.random.default_rng(2)
gi = np.where(ok & (w > 0))[0]
groups = {int(r): gi[frep[gi] == r] for r in np.unique(frep[gi])}
reps = np.array(list(groups)); b1, b2 = [], []
for _ in range(300):
    sel = np.concatenate([groups[int(r)] for r in rng.choice(reps, reps.size, replace=True)])
    _, _, ix_ = fes2d(d1[sel], d2[sel], w[sel])
    if ix_ is not None:
        b1.append(c[ix_[0]]); b2.append(c[ix_[1]])
b1, b2 = np.array(b1), np.array(b2)
print(f"\n  block bootstrap over {reps.size} replicas, {b1.size} resamples:")
print(f"    d1 minimum 95% CI [{np.percentile(b1,2.5):.2f}, {np.percentile(b1,97.5):.2f}] A"
      f"   native 2.84-4.22 -> {'OUTSIDE' if np.percentile(b1,2.5) > 4.22 else 'OVERLAPS'}")
print(f"    d2 minimum 95% CI [{np.percentile(b2,2.5):.2f}, {np.percentile(b2,97.5):.2f}] A"
      f"   native 6.17-7.56 -> {'OUTSIDE' if np.percentile(b2,2.5) > 7.56 else 'OVERLAPS'}")
json.dump({"n_joined": int(ok.sum()), "n_weighted": int((w>0).sum()),
           "d1_min_A": float(c[idx[0]]), "d2_min_A": float(c[idx[1]]),
           "n_in_min_bin": int(N[idx]),
           "d1_published": 7.206, "d2_published": 8.377,
           "d1_ci95": [float(np.percentile(b1,2.5)), float(np.percentile(b1,97.5))],
           "d2_ci95": [float(np.percentile(b2,2.5)), float(np.percentile(b2,97.5))],
           "dcv_median": float(np.nanmedian(dcv[ok])),
           "caveat": "weights are from the block-start row, a configuration within ~8.75 ps of the frame"},
          open(f"{OUT}/matched_pmf_d1d2.json", "w"), indent=2)
print("\nwrote matched_pmf_d1d2.json")
