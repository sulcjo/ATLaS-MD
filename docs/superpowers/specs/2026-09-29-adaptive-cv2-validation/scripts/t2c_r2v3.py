"""T2 re-run for report v3 (2026-09-30 follow-ups (h)): R2 same-column counting and the
autocorrelation-length bootstrap blocks, on the t2c scenarios (R2 only: no mixture fits).

One job = (scenario, regime, seed): simulate once (t2c_sim, same seeds as t2c_run), then at the
2,000- and 8,000-sample prefixes compute, per column interval: unbiased weight share, the
contributing-centre counts under ``any`` and ``same-column``, the exact error |F_est - F_exact|
(weight-shift removed, t2c_eval._slab_error) and the fixed-f block-bootstrap sigma under three
block rules:
  fixed20 -- the pre-v3 rule, 20 blocks per state (reproduces the t2c jobs' ``sigma`` exactly);
  g2      -- blocks of ceil(2 g) rows (cv2_coverage.autocorrelation_block_ids, multiple 2);
  g5      -- blocks of ceil(5 g) rows (the shipped v3 rule, BOOT_BLOCK_G_MULTIPLE).
Then every (count, block rule, coverage_min_windows, refine_pmf_sigma_kT) combination's R2
proposals (cv2_coverage._hole_runs / _hole_candidate, as coverage_holes does).

``--resolve-f``: also the bootstrap that RE-SOLVES the lambda = 0 MBAR per replicate
(``cv2_coverage.resolve_f_sigma``, the shipped ``--ap-coverage-bootstrap resolve-f``: g5 blocks,
cov.N_BOOT = 100 replicates, warm-started from the point f), at both prefixes, as block rule
``g5_resolve_f`` in the sigma records and the combos. (Before 9.10 this was a 40-replicate
diagnostic re-implementation at the 2,000 prefix only.)

MBAR: gareus-analyze's solver through ``cv2_coverage.solve_mbar`` (numba-anderson, tol 1e-12);
before 9.10 L-BFGS + polish (``fast_mbar``), which agreed to ~1e-8.

python t2c_r2v3.py OUT [--seeds 0 1 2 3] [--workers 8] [--scenarios ...] [--resolve-f]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")       # one thread per worker process
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")

import numpy as np  # noqa: E402

import t2c_eval as E  # noqa: E402
import t2c_sim as S  # noqa: E402
from gareus.adaptive import cv2_coverage as cov  # noqa: E402

PREFIXES = (2000, 8000)
BLOCKS = ("fixed20", "g2", "g5")
COUNTS = ("any", "same-column")
MIN_WINDOWS = (1.0, 2.0, 3.0)
SIGMAS = (0.25, 0.5, 1.0, float("inf"))
N_RESOLVE = 40


def fast_mbar(u, n_k, f0=None, polish=20):
    """MBAR f (f_0 = 0) on (K, N) ``u`` with rows ordered by state: gareus-analyze's solver
    (``cv2_coverage.solve_mbar``). ``polish`` is ignored (kept for the old call sites)."""
    return cov.solve_mbar(u, n_k, f_init=f0)


def _blocks(rule, sidx, cv1, cv2):
    if rule == "fixed20":
        return cov._block_ids(sidx, cov.BOOT_BLOCKS), None
    mult = {"g2": 2.0, "g5": 5.0}[rule]
    ids, info = cov.autocorrelation_block_ids(sidx, [cv1, cv2], multiple=mult)
    return ids, {k: v for k, v in info.items() if k != "states"}


def _resolve_sigma(u, sidx, blocks, cols_data, rng, f_full):
    """sigma of each column interval's F over the shipped re-solved-f block bootstrap."""
    return cov.resolve_f_sigma(np.ascontiguousarray(u.T), sidx, blocks, cols_data, f_full, rng, n_boot=cov.N_BOOT)


def eval_prefix(sc, sim, n, seed, resolve_f):
    X = sim.x[:n]
    views = E.views_for(sc, X)
    ids = sorted(views)
    K = len(ids)
    cv1 = X[:, :, 0].T.reshape(-1)
    cv2 = X[:, :, 1].T.reshape(-1)
    sidx = np.repeat(np.arange(K), n)
    vlist = [views[i] for i in ids]
    u = cov.reduced_umbrella(cv1, cv2, vlist, 1.0)
    n_k = np.bincount(sidx, minlength=K).astype(float)
    f = fast_mbar(u, n_k)
    lw = cov.log_weights(u, n_k, f)
    w = np.exp(lw - lw.max())
    w /= w.sum()
    base = E.settings()
    centre_of_state, centre_column = cov.centre_columns(vlist, base)
    blocks = {b: _blocks(b, sidx, cv1, cv2) for b in BLOCKS}
    # one bootstrap stream per block rule, consumed column after column as t2c_eval.eval_r2 does,
    # so fixed20 reproduces the t2c jobs' sigma where the columns' weights agree
    rngs = {b: np.random.default_rng([3303, int(seed)]) for b in BLOCKS}
    cols, cols_data = [], []
    ext = [cov._extend_to_weight(col, cv1, cv2, w) for col in cov.columns(views, ids, base)]
    rules = BLOCKS
    if resolve_f:
        t_rf = time.time()
        resolved = _resolve_sigma(u, sidx, blocks["g5"][0], [cov.column_bins(c, cv1, cv2) for c in ext],
                                  np.random.default_rng([cov._RESOLVE_SEED, int(seed)]), f)
        t_rf = time.time() - t_rf
        rules = BLOCKS + ("g5_resolve_f",)
    for ci, col in enumerate(ext):
        stats = {}
        if resolve_f:
            stats["g5_resolve_f"] = resolved[ci]
        for b in BLOCKS:
            extra = {}
            frac, n_eff, n_any, sigma = cov._interval_stats(col, cv1, cv2, sidx, w, centre_of_state, rngs[b],
                                                            block_ids=blocks[b][0], centre_column=centre_column,
                                                            info=extra)
            stats[b] = sigma
        n_same = extra["n_contrib_same_column"]
        edges = col["edges"]
        tot = frac.sum()
        with np.errstate(divide="ignore"):
            est_f = -np.log(frac / tot) if tot > 0 else np.full(frac.shape, np.nan)
        est_f = np.where(np.isfinite(est_f), est_f, np.nan)
        exact_f = S.exact_slab_cv2(sc, col["c1"], col["slab_half_width"], edges)
        err = E._slab_error(est_f, exact_f, frac)
        mids = 0.5 * (edges[1:] + edges[:-1])
        in_hole = np.zeros(frac.size, dtype=bool)
        if sc.hole is not None and (sc.hole[0] is None or abs(sc.hole[0] - col["c1"]) < 1e-6):
            in_hole = (mids >= sc.hole[1][0]) & (mids <= sc.hole[1][1])
        combos = {}
        for cnt in COUNTS:
            nc = n_same if cnt == "same-column" else n_any
            for b in rules:
                sigma = stats[b]
                for cmw in MIN_WINDOWS:
                    for ps in SIGMAS:
                        stg = E.settings(coverage_min_windows=cmw, refine_pmf_sigma_kT=ps, coverage_count=cnt)
                        heavy = frac >= cov.MIN_INTERVAL_WEIGHT
                        by_c = heavy & (nc < cmw)
                        by_s = heavy & (sigma > ps)
                        props = []
                        for i0, i1 in cov._hole_runs(by_c | by_s, edges, [float(v.c2) for v in col["members"]]):
                            cand = cov._hole_candidate(col, i0, i1, frac, n_eff, nc, sigma, stg)
                            seg = np.arange(i0, i1 + 1)
                            props.append({"decision": cand["decision"], "in_hole": bool(in_hole[seg].any()),
                                          "by_contrib": bool(by_c[seg].any()), "by_sigma": bool(by_s[seg].any()),
                                          "max_err_kT": (float(np.nanmax(err[seg])) if np.isfinite(err[seg]).any()
                                                         else None)})
                        combos[f"{cnt}|{b}|{cmw}|{ps}"] = props
        slab = np.abs(cv1 - col["c1"]) <= col["slab_half_width"]
        bins = np.clip(np.searchsorted(edges, cv2, side="right") - 1, 0, edges.size - 2)
        inside = slab & (cv2 >= edges[0]) & (cv2 <= edges[-1])
        cols_data.append((slab, bins, inside, edges.size - 1))
        cols.append({"c1": col["c1"], "frac": frac.tolist(), "n_contrib_any": n_any.tolist(),
                     "n_contrib_same": n_same.tolist(), "err_kT": [None if not math.isfinite(x) else float(x) for x in err],
                     "sigma": {b: [None if not math.isfinite(x) else float(x) for x in stats[b]] for b in rules},
                     "in_hole": in_hole.tolist(), "combos": combos})
    out = {"columns": cols, "blocks": {b: blocks[b][1] for b in BLOCKS}}
    if resolve_f:
        out["resolve_f_wall_s"] = t_rf
        out["resolve_f_shape"] = [int(u.shape[1]), int(u.shape[0])]
    return out


def job(a):
    name, regime, seed, out_dir, resolve_f = a
    path = Path(out_dir) / f"{name}__{regime}__s{seed}.json"
    if path.exists():
        return str(path), 0.0
    t0 = time.time()
    sc = S.scenarios()[name]
    sim = S.simulate(sc, regime, seed, max(PREFIXES), attempts_per_window=2.0)
    rec = {"scenario": name, "regime": regime, "seed": seed, "hole": sc.hole, "r2": {}}
    for p in PREFIXES:
        rec["r2"][str(p)] = eval_prefix(sc, sim, p, seed, resolve_f)
    rec["wall_s"] = time.time() - t0
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float))
    os.replace(tmp, path)
    return str(path), rec["wall_s"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenarios", nargs="+", default=list(S.scenarios()))
    ap.add_argument("--regimes", nargs="+", default=["indep", "re"])
    ap.add_argument("--resolve-f", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out) / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, r, seed, str(out), a.resolve_f) for s in a.scenarios for r in a.regimes for seed in a.seeds]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (p, w) in enumerate(ex.map(job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {Path(p).name} {w:.0f} s (elapsed {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
