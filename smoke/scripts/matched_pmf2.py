"""Rg PMF from matched rows only, vs the forward-filled PMF the run published.

Subsetting to block-start rows is a valid importance-sampling subset: the
selection is by time index (every 10th row), independent of configuration.
"""
import json
import numpy as np, pandas as pd

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
CSV = ("/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS/chignolin_7/"
       "adaptive_production/pmf_analysis/rg_samples_with_weights.csv")
kT = 0.001987204259 * 300.0

df = pd.read_csv(CSV, usecols=["rg_A", "umbrella_mbar_weight"], engine="c")
rg_all = df["rg_A"].to_numpy(); w_all = df["umbrella_mbar_weight"].to_numpy()
del df
m = np.load(f"{OUT}/matched_rows.npz")
print(f"full (forward-filled) rows {rg_all.size:,}   matched rows {m['rg'].size:,}", flush=True)

edges = np.linspace(4.9, 10.8, 31)
nb = len(edges) - 1
def pmf(x, w):
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < nb)
    p = np.bincount(idx[ok], weights=w[ok], minlength=nb)
    n = np.bincount(idx[ok], minlength=nb)
    s = p.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        f = -kT * np.log(p / s)
    if np.isfinite(f).any():
        f -= np.nanmin(f[np.isfinite(f)])
    return f, n
c = 0.5 * (edges[:-1] + edges[1:])
f_all, n_all = pmf(rg_all, w_all)
f_mat, n_mat = pmf(m["rg"], m["w_umbrella"])
print(f"\n{'Rg (A)':>8s} {'F full':>9s} {'F matched':>10s} {'diff':>8s} {'n_matched':>10s}")
for i in range(nb):
    if np.isfinite(f_all[i]) or np.isfinite(f_mat[i]):
        print(f"{c[i]:8.3f} {f_all[i]:9.3f} {f_mat[i]:10.3f} {f_mat[i]-f_all[i]:8.3f} {n_mat[i]:10d}")
i_all, i_mat = int(np.nanargmin(f_all)), int(np.nanargmin(f_mat))
print(f"\nMINIMUM full/forward-filled : {c[i_all]:.3f} A")
print(f"MINIMUM matched rows only   : {c[i_mat]:.3f} A")
print(f"published rg_summary.json   : 6.594 A")
d = np.abs(f_mat - f_all); fin = np.isfinite(d)
print(f"max |dF| over populated bins: {d[fin].max():.3f} kcal/mol")

# block bootstrap over replicas (precomputed index groups)
rng = np.random.default_rng(0)
reps = np.unique(m["replica"])
groups = {int(r): np.where(m["replica"] == r)[0] for r in reps}
rgm, wm = m["rg"], m["w_umbrella"]
mins = []
for _ in range(400):
    sel = np.concatenate([groups[int(r)] for r in rng.choice(reps, reps.size, replace=True)])
    fb, _ = pmf(rgm[sel], wm[sel])
    if np.isfinite(fb).any():
        mins.append(c[int(np.nanargmin(fb))])
mins = np.array(mins)
lo, hi = np.percentile(mins, [2.5, 97.5])
print(f"\nblock bootstrap over {reps.size} replicas, {mins.size} resamples:")
print(f"  matched-row Rg minimum {c[i_mat]:.3f} A, 95% CI [{lo:.3f}, {hi:.3f}] A")
print(f"  native 5.14-5.66 A -> {'OUTSIDE native' if lo > 5.66 else 'CI OVERLAPS NATIVE'}")
json.dump({"rg_min_full_A": float(c[i_all]), "rg_min_matched_A": float(c[i_mat]),
           "rg_min_published_A": 6.594, "rg_min_matched_ci95": [float(lo), float(hi)],
           "max_abs_dF_kcal": float(d[fin].max()), "n_matched": int(rgm.size),
           "n_full": int(rg_all.size), "bootstrap_n": int(mins.size)},
          open(f"{OUT}/matched_pmf_rg.json", "w"), indent=2)
print("\nwrote matched_pmf_rg.json")
