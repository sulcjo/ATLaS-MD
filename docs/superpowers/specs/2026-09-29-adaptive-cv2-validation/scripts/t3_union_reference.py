"""T3 task 2 (a)+(c): union-MBAR reference for every lambda = 0 spatial edge, read-only.

Usage: python t3_union_reference.py <adaptive_dir> <final_combined payload.json> <out.json> [--row-stride S]
       [--resolve]   (re-solve the union with pymbar for the full dDelta_f matrix)

Per edge of the payload's 3.1 graph (representative rung), on the union snapshot's OWN samples
(``adaptive_union_mbar.npz``: already g-subsampled per state by the builder):

* ``o2s``      two-state BAR overlap, Delta f re-solved (``edge_metric.evaluate_edge``, so the
               same estimator as the collector, but on the union's sample set);
* ``df2s``     its Delta f = f_j - f_i (kT) and ``sd_df2s`` from an iid bootstrap (200);
* ``df2s_h1/h2`` the same on each state's first / second half (sample order within source);
* ``ouf``      ``pairwise_state_overlap`` with the UNION f_k held fixed (what post-union reads);
* ``dfu``      union f_j - f_i and, with ``--resolve``, its pymbar sigma.

Also the rung gate check: union pairwise overlap for every rung edge (must reproduce the
ladder_adapt replay's 0.243 / 0.272 / 0.402 medians on chignolin_9).
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from gareus.adaptive import edge_metric as em
from gareus.mbar_analysis.ladder import pairwise_state_overlap

ad = Path(sys.argv[1])
payload = json.loads(Path(sys.argv[2]).read_text())
out_path = Path(sys.argv[3])
stride = 1
if "--row-stride" in sys.argv:
    stride = int(sys.argv[sys.argv.index("--row-stride") + 1])
resolve = "--resolve" in sys.argv

z = np.load(ad / "adaptive_union_mbar.npz", allow_pickle=False)
state_ids = np.asarray(z["state_ids"], dtype=np.int64)
sampled = np.asarray(z["sampled_state_ids"], dtype=np.int64)[::stride]
cv1 = np.asarray(z["cv_A"], dtype=float)[::stride]
cv2 = np.asarray(z["secondary_cv"], dtype=float)[::stride]
lam = np.asarray(z["state_lambdas"], dtype=float)
u = np.asarray(z["umbrella_reduced_bias_nk"], dtype=float)[::stride]
idx_of = {int(s): i for i, s in enumerate(state_ids.tolist())}
window = np.asarray([idx_of.get(int(s), -1) for s in sampled], dtype=np.int64)
keep = (window >= 0) & np.isfinite(u).all(axis=1)
u, window, cv1, cv2, sampled = u[keep], window[keep], cv1[keep], cv2[keep], sampled[keep]
n_k = np.bincount(window, minlength=u.shape[1]).astype(np.int64)
print("union rows", u.shape, "dropped non-finite", int((~keep).sum()), "n_k min/med", n_k.min(), int(np.median(n_k)))

f_csv = ad / "adaptive_union_mbar_analysis.state_free_energies.csv"
f_k = sigma_rel0 = None
if f_csv.exists() and stride == 1:
    rows = list(csv.DictReader(f_csv.open()))
    f_k = np.array([float(r["f_relative_to_state0"]) for r in rows])
    sigma_rel0 = np.array([float(r["df_relative_to_state0"]) for r in rows])
    assert [int(r["state_id"]) for r in rows] == state_ids.tolist()
dd = None
if resolve or f_k is None:
    from pymbar import MBAR
    mb = MBAR(u.T, n_k, initial_f_k=None if f_k is None else f_k, solver_protocol="robust")
    fe = mb.compute_free_energy_differences()
    f_new = np.asarray(fe["Delta_f"][0], dtype=float)
    if f_k is not None:
        print("re-solve vs csv f: max |diff|", float(np.max(np.abs(f_new - f_k))))
    f_k = f_new
    dd = np.asarray(fe["dDelta_f"], dtype=float)
    del mb

beta = em.beta_mol_per_kcal(float(payload["edge_metric"]["temperature_k"]))
restr = {int(s["state_id"]): em.Restraint.from_record(s["paired_cv"]["restraint"])
         for s in payload["states"] if s.get("paired_cv", {}).get("restraint")}

# Validate the restraint-record u against the union's reduced bias on lambda = 0 columns.
chk = []
for sid, r in list(restr.items()):
    j = idx_of.get(sid)
    if j is None or abs(lam[j]) > 1e-9:
        continue
    rows_ = np.flatnonzero(window == j)[:500]
    mine = r.reduced_bias(cv1[rows_], cv2[rows_], beta)
    chk.append(float(np.nanmax(np.abs(mine - u[rows_, j]))))
print("restraint-record vs union reduced bias, lambda=0 cols: max |diff|", max(chk) if chk else None)


def samples(sid):
    j = idx_of[sid]
    rows_ = np.flatnonzero(window == j)
    return em.StateSamples(sid, restr[sid], cv1[rows_], cv2[rows_], None)


def half(s, which):
    n = s.cv1.size // 2
    sl = slice(0, n) if which == 0 else slice(n, None)
    return em.StateSamples(s.state_id, s.restraint, s.cv1[sl], s.cv2[sl], None)


def df_boot(a, b, n=200, seed=7):
    rng = np.random.default_rng([seed, a.state_id, b.state_id])
    ua = b.restraint.reduced_bias(a.cv1, a.cv2, beta) - a.restraint.reduced_bias(a.cv1, a.cv2, beta)
    ub = b.restraint.reduced_bias(b.cv1, b.cv2, beta) - a.restraint.reduced_bias(b.cv1, b.cv2, beta)
    ua, ub = ua[np.isfinite(ua)], ub[np.isfinite(ub)]
    point_o, point_df = em.two_state_overlap(ua, ub)
    vals = []
    for _ in range(n):
        ia = rng.integers(ua.size, size=ua.size)
        ib = rng.integers(ub.size, size=ub.size)
        vals.append(em.two_state_overlap(ua[ia], ub[ib], guess=point_df)[1])
    return point_o, point_df, float(np.std(vals, ddof=1)), int(ua.size), int(ub.size)


rows_out = []
for e in payload["edges"]:
    if e.get("edge_type") == "rung":
        continue
    si, sj = sorted((int(e["state_i"]), int(e["state_j"])))
    if si not in restr or sj not in restr or si not in idx_of or sj not in idx_of:
        continue
    pm = e.get("pairwise_mbar") or {}
    a, b = samples(si), samples(sj)
    if a.cv1.size < 20 or b.cv1.size < 20:
        continue
    o2s, df2s, sd_df2s, na, nb = df_boot(a, b)
    h = [em.two_state_overlap(
        b.restraint.reduced_bias(half(a, k).cv1, half(a, k).cv2, beta) - a.restraint.reduced_bias(half(a, k).cv1, half(a, k).cv2, beta),
        b.restraint.reduced_bias(half(b, k).cv1, half(b, k).cv2, beta) - a.restraint.reduced_bias(half(b, k).cv1, half(b, k).cv2, beta))
        for k in (0, 1)]
    ia, ib = idx_of[si], idx_of[sj]
    ouf = pairwise_state_overlap(u, window, f_k, n_k, ia, ib)
    rec = dict(i=si, j=sj, edge_type=e.get("edge_type"), graph_kind=pm.get("graph_kind"),
               pattern_pair=pm.get("pattern_pair"), p4_status=pm.get("status"), p4_point=pm.get("overlap_point"),
               p4_q10=pm.get("overlap_lower"), p4_q90=pm.get("overlap_upper"), p4_df=pm.get("delta_f_kT"),
               p4_neff=pm.get("n_eff"), marginal=e.get("overlap"), joint=e.get("overlap_joint_2d"),
               union_n=[na, nb], o2s=o2s, df2s=df2s, sd_df2s=sd_df2s,
               o2s_h1=h[0][0], o2s_h2=h[1][0], df2s_h1=h[0][1], df2s_h2=h[1][1],
               ouf=ouf, dfu=float(f_k[ib] - f_k[ia]),
               sd_dfu=None if dd is None else float(dd[ia, ib]),
               sd_rel0=None if sigma_rel0 is None else [float(sigma_rel0[ia]), float(sigma_rel0[ib])])
    rows_out.append(rec)

rung = []
for e in payload["edges"]:
    if e.get("edge_type") != "rung":
        continue
    si, sj = int(e["state_i"]), int(e["state_j"])
    if si not in idx_of or sj not in idx_of:
        continue
    ia, ib = idx_of[si], idx_of[sj]
    rung.append(dict(i=si, j=sj, lam=[float(lam[ia]), float(lam[ib])],
                     union_pairwise=pairwise_state_overlap(u, window, f_k, n_k, ia, ib),
                     payload_mbar_overlap=e.get("mbar_overlap")))
by = {}
for r in rung:
    by.setdefault(tuple(sorted(round(x, 4) for x in r["lam"])), []).append(r["union_pairwise"])
print("rung pairwise medians:", {k: round(float(np.median(v)), 3) for k, v in sorted(by.items())})
out_path.write_text(json.dumps(dict(spatial=rows_out, rung=rung, n_k=n_k.tolist(), stride=stride,
                                    state_ids=state_ids.tolist()), default=float))
print("wrote", out_path, len(rows_out), "spatial edges", len(rung), "rung edges")
