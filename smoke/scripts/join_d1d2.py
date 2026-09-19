"""Join coordinate frames to their matched sample rows, then build the d1/d2 FES.

d1/d2 exist per frame (this audit computed them) but not in the weights file, so
the frames must be joined to the rows that carry their MBAR weights.  The key is
(replica, Rg): Rg is identical between the two sources to 5 decimals and min/max
agree exactly.  CV1 is the independent checksum -- it is recorded per row and was
computed per frame separately, so a correct join must agree on it.
"""
import collections, json, re
import numpy as np

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
m = np.load(f"{OUT}/matched_rows.npz")
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
files = [str(x) for x in z["files"]]
rep_of = np.array([int(re.search(r"replica_(\d+)", f.split("/")[-1]).group(1)) for f in files])
frep = rep_of[z["src_file"]]
kT = 0.001987204259 * 300.0

key = collections.defaultdict(list)
mrep, mrg, mcv = m["replica"], m["rg"], m["cv"]
for i in range(mrg.size):
    key[(int(mrep[i]), round(float(mrg[i]), 5))].append(i)
print(f"index built: {len(key):,} distinct (replica, rg) keys over {mrg.size:,} rows")

row = np.full(frep.size, -1, np.int64)
dcv = np.full(frep.size, np.nan)
namb = 0
fcv = z["cv1"].astype(np.float64); frg = z["rg"].astype(np.float64)
for i in range(frep.size):
    cands = key.get((int(frep[i]), round(frg[i], 5)))
    if not cands:
        continue
    if len(cands) == 1:
        j = cands[0]
    else:                                  # disambiguate on the checksum itself
        namb += 1
        j = min(cands, key=lambda k: abs(mcv[k] - fcv[i]))
    row[i] = j
    dcv[i] = abs(mcv[j] - fcv[i])

ok = row >= 0
print(f"\nJOIN: {ok.sum():,} / {frep.size:,} frames matched ({ok.mean():.4f});"
      f" ambiguous keys resolved by checksum: {namb:,}")
print(f"CV CHECKSUM |cv_frame - cv_row| on joined frames:")
print(f"  median {np.nanmedian(dcv[ok]):.3e}   p90 {np.nanpercentile(dcv[ok],90):.3e}   "
      f"p99 {np.nanpercentile(dcv[ok],99):.3e}   max {np.nanmax(dcv[ok]):.3e}")
for t in (1e-6, 1e-5, 1e-4, 1e-3):
    print(f"  fraction with |dcv| < {t:g}: {(dcv[ok] < t).mean():.4f}")

# keep only joins the checksum confirms
good = ok & (dcv < 1e-4)
print(f"\nCHECKSUM-CONFIRMED joins: {good.sum():,} ({good.mean():.4f} of all frames)")
w = np.zeros(frep.size); w[good] = m["w_umbrella"][row[good]]

# ---- d1/d2 2D FES from confirmed joins ----
d1, d2 = z["d1"].astype(np.float64), z["d2"].astype(np.float64)
e1 = np.linspace(2.5, 16.0, 28); e2 = np.linspace(2.5, 16.0, 28)
def fes2d(x, y, wt):
    ix = np.digitize(x, e1) - 1; iy = np.digitize(y, e2) - 1
    m_ = (ix >= 0) & (ix < len(e1)-1) & (iy >= 0) & (iy < len(e2)-1) & (wt > 0)
    H = np.zeros((len(e1)-1, len(e2)-1)); N = np.zeros_like(H)
    np.add.at(H, (ix[m_], iy[m_]), wt[m_]); np.add.at(N, (ix[m_], iy[m_]), 1.0)
    P = H / H.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -kT * np.log(P)
    F -= np.nanmin(F[np.isfinite(F)])
    return F, N
c1 = 0.5*(e1[:-1]+e1[1:]); c2 = 0.5*(e2[:-1]+e2[1:])
F, N = fes2d(d1, d2, w)
i, j = np.unravel_index(np.nanargmin(np.where(N >= 20, F, np.inf)), F.shape)
print(f"\nd1/d2 FES from {int(good.sum()):,} checksum-confirmed matched frames")
print(f"  MINIMUM (bins with >=20 frames): d1 = {c1[i]:.3f} A, d2 = {c2[j]:.3f} A  "
      f"(n = {int(N[i,j])})")
print(f"  published (chignolin_fes_summary.json): d1 = 7.206 A, d2 = 8.377 A")
print(f"  native (18 NMR models): d1 = 2.84-4.22 A, d2 = 6.17-7.56 A")

# bootstrap the minimum over replicas
rng = np.random.default_rng(1)
gidx = np.where(good)[0]
groups = {int(r): gidx[frep[gidx] == r] for r in np.unique(frep[gidx])}
reps = np.array(list(groups))
b1, b2 = [], []
for _ in range(300):
    sel = np.concatenate([groups[int(r)] for r in rng.choice(reps, reps.size, replace=True)])
    Fb, Nb = fes2d(d1[sel], d2[sel], w[sel])
    msk = Nb >= 20
    if not msk.any():
        continue
    a, b = np.unravel_index(np.nanargmin(np.where(msk, Fb, np.inf)), Fb.shape)
    b1.append(c1[a]); b2.append(c2[b])
b1, b2 = np.array(b1), np.array(b2)
print(f"  block bootstrap over {reps.size} replicas, {b1.size} resamples:")
print(f"    d1 95% CI [{np.percentile(b1,2.5):.3f}, {np.percentile(b1,97.5):.3f}] A "
      f"(native 2.84-4.22)")
print(f"    d2 95% CI [{np.percentile(b2,2.5):.3f}, {np.percentile(b2,97.5):.3f}] A "
      f"(native 6.17-7.56)")
json.dump({"n_frames": int(frep.size), "n_joined": int(ok.sum()),
           "n_checksum_confirmed": int(good.sum()), "n_ambiguous": int(namb),
           "dcv_median": float(np.nanmedian(dcv[ok])), "dcv_p99": float(np.nanpercentile(dcv[ok],99)),
           "d1_min_A": float(c1[i]), "d2_min_A": float(c2[j]),
           "d1_published": 7.206, "d2_published": 8.377,
           "d1_ci95": [float(np.percentile(b1,2.5)), float(np.percentile(b1,97.5))],
           "d2_ci95": [float(np.percentile(b2,2.5)), float(np.percentile(b2,97.5))]},
          open(f"{OUT}/matched_pmf_d1d2.json", "w"), indent=2)
np.savez_compressed(f"{OUT}/frame_weights.npz", row=row, dcv=dcv, w_umbrella=w, good=good,
                    replica=frep)
print("\nwrote matched_pmf_d1d2.json and frame_weights.npz")
