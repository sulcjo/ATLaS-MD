"""Re-run only the R2 half of the t2c calibration jobs (t2c_run.py) after the R2 MBAR moved to
gareus-analyze's solver (T2 9.10). R2 is the only MBAR consumer in a t2c job; R3 (mixture fits,
the expensive part) never calls it, so each job is re-simulated with its own seed (t2c_sim is
deterministic; the simulation meta is checked against the old job), ``eval_r2`` is recomputed at
every prefix and the old job's ``r3`` / ``truth`` are kept. Then t2c_summarise.py and
t2c_compact.py rebuild t2_data/t2c_calibration.json.

python t2c_r2_rerun.py OLD_JOBS_DIR NEW_JOBS_DIR [--workers 8]
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
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")

import t2c_eval as E  # noqa: E402
import t2c_sim as S  # noqa: E402


def job(a):
    old_path, new_dir = a
    new = Path(new_dir) / Path(old_path).name
    if new.exists():
        return str(new), 0.0, True
    t0 = time.time()
    rec = json.loads(Path(old_path).read_text())
    sc = S.scenarios()[rec["scenario"]]
    sim = S.simulate(sc, rec["regime"], rec["seed"], rec["n"], attempts_per_window=2.0)
    same = (abs(sim.acceptance - rec["sim"]["acceptance"]) < 1e-12
            and abs(sim.mean_residence - rec["sim"]["mean_residence"]) < 1e-9)
    rec["r2"] = {p: E.eval_r2(sc, sim, int(p), rec["seed"]) for p in rec["r2"]}
    rec["r2_rerun"] = {"solver": "gareus.mbar_analysis numba-anderson via cv2_coverage.solve_mbar",
                       "sim_reproduced": bool(same), "wall_s": time.time() - t0}
    tmp = new.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float))
    os.replace(tmp, new)
    return str(new), time.time() - t0, same


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    Path(a.new).mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), a.new) for p in sorted(Path(a.old).glob("*.json"))]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (p, w, same) in enumerate(ex.map(job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {Path(p).name} {w:.0f} s sim_reproduced={same} (elapsed {time.time() - t0:.0f} s)",
                  flush=True)


if __name__ == "__main__":
    main()
