"""T2 calibration, end to end at matched budget (independent-window regime only).

R3: epoch 1 samples the layout for N0; R3 (shipped code, default knobs except the arm's cap)
picks parents; epoch 2 spends 2 N0 per parent either on two children at the mixture modes
(arm k2 rule) or, in the no-action arm, uniformly on the existing windows. Arms:
  none         -- matched budget on the existing windows;
  cap4_refused -- today: children refused k2_capped_below_compression spend nothing (= none);
  cap4_forced  -- children inserted at k2 = min(shape rule, 4 x parent k2) (drop the refusal);
  uncapped     -- children at the shape-rule k2 (compression floor k2 >= F''), no cap.
R2: epoch 1 N0; the R2 proposals of a knob setting get one window of N0 each; no-action arm
spends the same on the existing windows.
Metrics (per parent column / hole column): CV2 conditional PMF in the column slab vs the
exact surface (low-F-weighted RMSE over bins with exact F <= 6 kT), mode population ln-odds
error at the exact slab valley, and the exact child-parent pairwise overlap.

python t2c_e2e.py OUT.json [--seeds 0 1 2] [--workers 6]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")

import numpy as np  # noqa: E402

import t2c_sim as S  # noqa: E402
import t2c_eval as E  # noqa: E402
from gareus.adaptive import cv2_coverage as cov  # noqa: E402
from gareus.adaptive import cv2_resolution as cr  # noqa: E402
from gareus.adaptive.cv2_shape import shape_rule_k2  # noqa: E402
from gareus.synth.collector_adapter import exact_pair_overlap  # noqa: E402
from gareus.synth.sampler import Window  # noqa: E402


def pooled_weights(windows, blocks):
    """blocks: list of (windows_subset_indices, X (N, m, dim)). MBAR over all windows."""
    cv1, cv2, sidx = [], [], []
    for ids, X in blocks:
        for j, s in enumerate(ids):
            cv1.append(X[:, j, 0]); cv2.append(X[:, j, 1]); sidx.append(np.full(X.shape[0], s))
    cv1, cv2, sidx = np.concatenate(cv1), np.concatenate(cv2), np.concatenate(sidx)
    views = [cr.StateView(i, w.center1, w.k1, w.center2, w.k2, 0.0) for i, w in enumerate(windows)]
    u = cov.reduced_umbrella(cv1, cv2, views, 1.0)
    n_k = np.bincount(sidx, minlength=len(windows)).astype(float)
    info = {}
    f = cov.solve_mbar(u, n_k, info=info)
    lw = cov.log_weights(u, n_k, f)
    w = np.exp(lw - lw.max()); w /= w.sum()
    return cv1, cv2, w, info


def slab_metrics(sc, c1, half, cv1, cv2, w, modes=None, zrange=None):
    lo2, hi2 = sc.surface.cv2_bounds
    edges = np.linspace(lo2, hi2, int(round((hi2 - lo2) / 0.05)) + 1)
    exact = S.exact_slab_cv2(sc, c1, half, edges)
    exact = exact - np.nanmin(exact)
    sel = np.abs(cv1 - c1) <= half
    h, _ = np.histogram(cv2[sel], bins=edges, weights=w[sel])
    raw, _ = np.histogram(cv2[sel], bins=edges)
    mids = 0.5 * (edges[1:] + edges[:-1])
    ok = (exact <= 6.0) & (raw >= 5) & (h > 0)
    if zrange is not None:
        ok &= (mids >= zrange[0]) & (mids <= zrange[1])
    out = {"n_bins": int(ok.sum()), "coverage": float(np.mean((raw >= 5)[exact <= 6.0]))}
    if ok.sum() >= 3:
        est = -np.log(h[ok] / h.sum())
        wt = np.exp(-exact[ok]); wt /= wt.sum()
        d = est - exact[ok]
        d -= np.sum(wt * d)
        out["rmse_kT"] = float(np.sqrt(np.sum(wt * d * d)))
        out["max_err_kT"] = float(np.max(np.abs(d)))
    if modes is not None:
        a, b = sorted(modes)
        seg = (mids >= a) & (mids <= b)
        if seg.any():
            valley = float(mids[seg][np.argmax(exact[seg])])
            pe = np.exp(-exact); pe = pe[mids <= valley].sum() / pe.sum()
            ph = h[mids <= valley].sum() / h.sum() if h.sum() > 0 else float("nan")
            lo = lambda p: math.log(max(p, 1e-4) / max(1 - p, 1e-4))  # noqa: E731
            out["pop_lnodds_err_kT"] = abs(lo(ph) - lo(pe)) if math.isfinite(ph) else None
    return out


def column_half(sc, c1):
    c1s = sorted({round(w.center1, 6) for w in sc.windows})
    k1 = [w.k1 for w in sc.windows if abs(w.center1 - c1) < 1e-6][0]
    others = [abs(c1 - o) for o in c1s if abs(o - c1) > 1e-6]
    return min(0.5 * min(others), 2.0 / math.sqrt(k1)) if others else 2.0 / math.sqrt(k1)


def r3_job(args):
    name, seed, n0, min_t = args
    sc = S.scenarios()[name]
    truth = [S.exact_window_truth(sc, w) for w in sc.windows]
    n = len(sc.windows)
    probe = S.simulate(sc, "indep", seed, n0)
    recs = E.eval_r3(sc, probe, n0, truth)
    parents = [r for r in recs if r["gate"] == "passed" and (r.get("t_replica") or 0) >= min_t and r.get("children")]
    m = len(parents)
    extra = int(math.ceil(2 * m * n0 / n))
    base = S.simulate(sc, "indep", seed, n0 + extra)          # same seed: its first n0 samples = the probe's
    result = {"scenario": name, "seed": seed, "n0": n0, "parent_min_transitions": min_t, "n_parents": m, "extra_per_window": extra,
              "parents": [{"state": r["state"], "c1": r["c1"], "c2": r["c2"], "k2": r["k2"],
                           "children": r["children"], "t_replica": r["t_replica"]} for r in parents], "arms": {}}
    if not m:
        return result
    arms = {"none": None,
            "cap4_forced": lambda k, kp: min(shape_rule_k2(k["sigma_used"], k["f2_est"], E.T_UNIT, 0.0, 1e12), 4 * kp),
            "uncapped": lambda k, kp: shape_rule_k2(k["sigma_used"], k["f2_est"], E.T_UNIT, 0.0, 1e12)}
    for arm, rule in arms.items():
        if rule is None:
            wins = list(sc.windows)
            blocks = [(list(range(n)), base.x)]
        else:
            kids = [Window(p["c1"], [w.k1 for w in sc.windows if abs(w.center1 - p["c1"]) < 1e-9][0],
                           float(k["secondary_center"]), float(rule(k, p["k2"])))
                    for p in result["parents"] for k in p["children"]]
            sck = copy.copy(sc)
            sck.windows = kids
            ksim = S.simulate(sck, "indep", seed + 1000, n0)
            wins = list(sc.windows) + kids
            blocks = [(list(range(n)), base.x[:n0]), (list(range(n, n + len(kids))), ksim.x)]
        cv1, cv2, w, info = pooled_weights(wins, blocks)
        cols = []
        for p in result["parents"]:
            half = column_half(sc, p["c1"])
            modes = [k["secondary_center"] for k in p["children"]]
            met = slab_metrics(sc, p["c1"], half, cv1, cv2, w, modes=modes)
            if rule is not None:
                pw = sc.windows[p["state"]]
                met["child_k2"] = [float(rule(k, p["k2"])) for k in p["children"]]
                met["child_parent_overlap"] = [
                    exact_pair_overlap(sc.surface, pw, Window(p["c1"], pw.k1, float(k["secondary_center"]),
                                                              float(rule(k, p["k2"]))))["overlap"]
                    for k in p["children"]]
            cols.append({"state": p["state"], **met})
        result["arms"][arm] = {"mbar_converged": info.get("converged"), "columns": cols}
    return result


def r2_job(args):
    name, seed, n0, combo = args
    sc = S.scenarios()[name]
    n = len(sc.windows)
    probe = S.simulate(sc, "indep", seed, n0)
    r2 = E.eval_r2(sc, probe, n0, seed)
    props = [(c, p) for c in r2["columns"] for p in c["combos"][combo] if p["decision"] == "proposed"]
    m = len(props)
    extra = int(math.ceil(m * n0 / n))
    base = S.simulate(sc, "indep", seed, n0 + extra)
    out = {"scenario": name, "seed": seed, "n0": n0, "combo": combo, "n_proposed": m,
           "proposals": [{"c1": c["c1"], "interval": p["interval"], "in_hole": p["in_hole"],
                          "k2": (p["child"] or {}).get("k2"), "c2": (p["child"] or {}).get("secondary_center")}
                         for c, p in props], "arms": {}}
    kids = []
    for c, p in props:
        ch = p["child"]
        k1 = [w.k1 for w in sc.windows if abs(w.center1 - c["c1"]) < 1e-6][0]
        kids.append(Window(float(c["c1"]), float(k1), float(ch["secondary_center"]), float(ch["k2"])))
    for arm in ("none", "add") if m else ("none",):
        if arm == "none":
            wins, blocks = list(sc.windows), [(list(range(n)), base.x)]
        else:
            sck = copy.copy(sc); sck.windows = kids
            ksim = S.simulate(sck, "indep", seed + 1000, n0)
            wins = list(sc.windows) + kids
            blocks = [(list(range(n)), base.x[:n0]), (list(range(n, n + len(kids))), ksim.x)]
        cv1, cv2, w, info = pooled_weights(wins, blocks)
        cols = []
        for c1 in sorted({round(x.center1, 6) for x in sc.windows}):
            half = column_half(sc, c1)
            met = slab_metrics(sc, c1, half, cv1, cv2, w, zrange=sc.hole[1] if sc.hole else None)
            cols.append({"c1": c1, **met})
        out["arms"][arm] = {"columns": cols, "rmse_mean_kT": float(np.nanmean([x.get("rmse_kT", np.nan) for x in cols]))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--r3-scenarios", nargs="*", default=["c9-like-narrow", "dbl-saddle", "hidden-cv3"])
    ap.add_argument("--r2-scenarios", nargs="*", default=["plateau-rowgap", "harmonic-rowgap"])
    ap.add_argument("--r2-combos", nargs="+", default=["2.0|0.5", "2.0|0.25", "3.0|0.5", "2.0|1.0"])
    ap.add_argument("--n0-r3", type=int, default=4000)
    ap.add_argument("--r3-min-transitions", type=int, default=10)
    ap.add_argument("--n0-r2", type=int, default=2000)
    a = ap.parse_args(argv)
    r3_jobs = [(s, seed, a.n0_r3, a.r3_min_transitions) for s in a.r3_scenarios for seed in a.seeds]
    r2_jobs = [(s, seed, a.n0_r2, c) for s in a.r2_scenarios for seed in a.seeds for c in a.r2_combos]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        f3 = list(ex.map(r3_job, r3_jobs))
        f2 = list(ex.map(r2_job, r2_jobs))
    Path(a.out).write_text(json.dumps({"r3": f3, "r2": f2}, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
