"""Re-run the MBAR-dependent parts of recorded t2c_e2e.py outputs after the R2 MBAR moved to
gareus-analyze's solver (T2 9.10).

* R2 records: ``t2c_e2e.r2_job`` in full (the R2 proposals themselves come from the MBAR).
* R3 records: the parents and their children are the recorded ones (they come from the mixture
  fits of ``eval_r3``, which never call the MBAR); the arms are re-simulated with the same seeds
  and their metrics recomputed with the new MBAR (``pooled_weights``). Same arms and rules as
  ``t2c_e2e.r3_job``.

python t2c_e2e_rerun.py OLD_E2E.json NEW_E2E.json [--workers 8]
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
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler")

import t2c_e2e as X  # noqa: E402
import t2c_eval as E  # noqa: E402
import t2c_sim as S  # noqa: E402
from gareus.adaptive.cv2_shape import shape_rule_k2  # noqa: E402
from gareus.synth.collector_adapter import exact_pair_overlap  # noqa: E402
from gareus.synth.sampler import Window  # noqa: E402


def r3_metrics(rec):
    """``t2c_e2e.r3_job`` after its parent selection, on the recorded parents."""
    sc = S.scenarios()[rec["scenario"]]
    n, n0, seed, m = len(sc.windows), rec["n0"], rec["seed"], rec["n_parents"]
    out = {k: v for k, v in rec.items() if k != "arms"}
    out["arms"] = {}
    if not m:
        return out
    base = S.simulate(sc, "indep", seed, n0 + rec["extra_per_window"])
    arms = {"none": None,
            "cap4_forced": lambda k, kp: min(shape_rule_k2(k["sigma_used"], k["f2_est"], E.T_UNIT, 0.0, 1e12), 4 * kp),
            "uncapped": lambda k, kp: shape_rule_k2(k["sigma_used"], k["f2_est"], E.T_UNIT, 0.0, 1e12)}
    for arm, rule in arms.items():
        if rule is None:
            wins, blocks = list(sc.windows), [(list(range(n)), base.x)]
        else:
            kids = [Window(p["c1"], [w.k1 for w in sc.windows if abs(w.center1 - p["c1"]) < 1e-9][0],
                           float(k["secondary_center"]), float(rule(k, p["k2"])))
                    for p in rec["parents"] for k in p["children"]]
            sck = copy.copy(sc)
            sck.windows = kids
            ksim = S.simulate(sck, "indep", seed + 1000, n0)
            wins = list(sc.windows) + kids
            blocks = [(list(range(n)), base.x[:n0]), (list(range(n, n + len(kids))), ksim.x)]
        cv1, cv2, w, info = X.pooled_weights(wins, blocks)
        cols = []
        for p in rec["parents"]:
            half = X.column_half(sc, p["c1"])
            met = X.slab_metrics(sc, p["c1"], half, cv1, cv2, w, modes=[k["secondary_center"] for k in p["children"]])
            if rule is not None:
                pw = sc.windows[p["state"]]
                met["child_k2"] = [float(rule(k, p["k2"])) for k in p["children"]]
                met["child_parent_overlap"] = [
                    exact_pair_overlap(sc.surface, pw, Window(p["c1"], pw.k1, float(k["secondary_center"]),
                                                              float(rule(k, p["k2"]))))["overlap"]
                    for k in p["children"]]
            cols.append({"state": p["state"], **met})
        out["arms"][arm] = {"mbar_converged": info.get("converged"), "columns": cols}
    return out


def r2_rec(rec):
    return X.r2_job((rec["scenario"], rec["seed"], rec["n0"], rec["combo"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    old = json.loads(Path(a.old).read_text())
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        r3 = list(ex.map(r3_metrics, old["r3"]))
        r2 = list(ex.map(r2_rec, old["r2"]))
    Path(a.new).write_text(json.dumps({"r3": r3, "r2": r2}, default=float))
    print("wrote", a.new)


if __name__ == "__main__":
    main()
