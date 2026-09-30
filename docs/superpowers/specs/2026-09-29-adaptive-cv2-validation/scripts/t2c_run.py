"""T2 calibration driver: one job = (scenario, regime, seed) -> simulate once, evaluate R3 and
R2 at every knob value on prefixes of the same run. Per-job JSON in OUT/jobs/.

python t2c_run.py OUT [--n 8000] [--prefixes 2000 8000] [--seeds 0 1 2 3] [--workers 8]
                      [--scenarios ...] [--regimes indep re]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")

import t2c_sim as S  # noqa: E402
import t2c_eval as E  # noqa: E402


def job(args):
    name, regime, seed, n, prefixes, out_dir = args
    path = Path(out_dir) / f"{name}__{regime}__s{seed}.json"
    if path.exists():
        return str(path), 0.0
    t0 = time.time()
    sc = S.scenarios()[name]
    truth = [S.exact_window_truth(sc, w) for w in sc.windows]
    sim = S.simulate(sc, regime, seed, n, attempts_per_window=2.0)
    rec = {"scenario": name, "regime": regime, "seed": seed, "n": n, "expect_r3": sc.expect_r3,
           "expect_r2": sc.expect_r2, "hole": sc.hole, "n_windows": len(sc.windows),
           "sim": {"acceptance": sim.acceptance, "swap_acceptance": sim.swap_acceptance,
                   "mean_residence": sim.mean_residence, "wall_s": sim.wall_s, **sim.meta},
           "truth": truth, "r3": {}, "r2": {}}
    for p in prefixes:
        rec["r3"][str(p)] = E.eval_r3(sc, sim, p, truth)
        rec["r2"][str(p)] = E.eval_r2(sc, sim, p, seed)
    rec["wall_s"] = time.time() - t0
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float))
    os.replace(tmp, path)
    return str(path), rec["wall_s"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--prefixes", type=int, nargs="+", default=[2000, 8000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scenarios", nargs="+", default=list(S.scenarios()))
    ap.add_argument("--regimes", nargs="+", default=["indep", "re"])
    a = ap.parse_args(argv)
    out = Path(a.out) / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, r, seed, a.n, a.prefixes, str(out)) for s in a.scenarios for r in a.regimes for seed in a.seeds]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (p, w) in enumerate(ex.map(job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {Path(p).name} {w:.0f} s (elapsed {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
