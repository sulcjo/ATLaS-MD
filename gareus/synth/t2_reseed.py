"""T2 / X3: slow-mode-aware reseeding on a landscape with a hidden slow CV3.

``hidden-slow-cv3`` and ``hidden-slow-cv3-weak`` (landscapes.py): cv3 is a 7 kBT double well that projects onto CV2
(cv2 ~ 0.45 cv3) and whose preferred side depends on CV1; windows restrain (cv1, cv2) only,
and D3 is small enough that no window crosses cv3's barrier within the campaign. Swarm
seeds come from short unbiased runs started stratified in (cv1, cv2) with cv3 on the -1
side for 80 % of members (a seed generator that does not see the hidden mode).

Two arms at the same MD budget (same windows, samples per window per epoch, burn-in):

* ``none``: every window continues its own chain across epochs;
* ``x3``: at each epoch boundary the SHIPPED planner ``slow_mode_reseed.plan_reseed``
  (fraction 0.25, one-sided = minority <= 20 %, admissible = within 2 sigma_w) picks windows
  and seeds from the pooled end states; the side split is the shipped ``split_point`` on
  the pooled cv3. The hidden coordinate is the oracle cv3 itself: production estimates it
  as the leading conditional tICA mode of the torsion residual, which this harness does
  not emulate. Only starting points change; both arms discard the same burn-in.

Reported per epoch: per-window pooled cv3 high-side fraction vs the exact equilibrium
value, one-sided windows (sampled vs exact), and MBAR PMF errors ((cv1, cv2) and CV2).
``python -m gareus.synth.t2_reseed --out DIR``.
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import collector_adapter as A
from .landscapes import LANDSCAPES_3D
from .sampler import Window
from .vec_langevin import run_chains

D_COEF = (1.0, 0.05, 0.05)
C1S = np.round(np.linspace(0.15, 0.85, 5), 6)
C2S = (-0.45, 0.0, 0.45)
N_PER_EPOCH = 1500
N_EPOCHS = 4
BURN_IN = 200
SWARM_MEMBERS = 60
SWARM_SAMPLES = 300
SWARM_MINORITY_SIDE = 0.2
RESEED_FRACTION = 0.25


def windows() -> List[Window]:
    k1 = 1.0 / ((C1S[1] - C1S[0]) / 1.5) ** 2
    k2 = 1.0 / ((C2S[1] - C2S[0]) / 1.5) ** 2
    return [Window(float(a), k1, float(b), k2) for a in C1S for b in C2S]


def exact_high_fraction(name: str, w: Window, res: int = 161) -> float:
    ls = LANDSCAPES_3D[name]
    s1, s2 = 1.0 / math.sqrt(w.k1), 1.0 / math.sqrt(w.k2)
    x = np.linspace(max(0.0, w.center1 - 7 * s1), min(1.0, w.center1 + 7 * s1), res)
    y = np.linspace(max(-1.0, w.center2 - 7 * s2), min(1.0, w.center2 + 7 * s2), res)
    z = np.linspace(-2.0, 2.0, 2 * res)
    g1, g2, g3 = np.meshgrid(x, y, z, indexing="ij")
    e = ls.energy(g1, g2, g3) + 0.5 * w.k1 * (g1 - w.center1) ** 2 + 0.5 * w.k2 * (g2 - w.center2) ** 2
    p = np.exp(-(e - e.min()))
    return float(p[:, :, z > ls.hidden_split].sum() / p.sum())


def _swarm(name: str, rng) -> np.ndarray:
    ls = LANDSCAPES_3D[name]
    n = SWARM_MEMBERS
    x0 = np.column_stack([(rng.permutation(n) + rng.uniform(size=n)) / n,
                          -1.0 + 2.0 * (rng.permutation(n) + rng.uniform(size=n)) / n,
                          np.where(rng.uniform(size=n) < SWARM_MINORITY_SIDE, 1.0, -1.0)])
    run = run_chains(ls, x0, SWARM_SAMPLES, D=D_COEF, dt=2.0e-3, steps_per_sample=20, rng=rng)
    return run.samples.reshape(-1, 3)


def _pmf(name: str, pooled, log_w) -> Dict[str, float]:
    ls = LANDSCAPES_3D[name]
    ax1, ax2, f2 = ls.marginal_2d(res=41)
    w = np.exp(log_w)
    b1 = np.linspace(0, 1, 42); b2 = np.linspace(-1, 1, 42)
    dens, _, _ = np.histogram2d(pooled[:, 0], pooled[:, 1], bins=[b1, b2], weights=w)
    raw, _, _ = np.histogram2d(pooled[:, 0], pooled[:, 1], bins=[b1, b2])
    # reference on bin centres
    from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415
    c1 = 0.5 * (b1[1:] + b1[:-1]); c2 = 0.5 * (b2[1:] + b2[:-1])
    ref = RegularGridInterpolator((ax1, ax2), f2)(np.stack(np.meshgrid(c1, c2, indexing="ij"), -1))
    ok = (raw >= 5) & (dens > 0) & (ref <= 5.0)
    d = -np.log(dens[ok]) - ref[ok]
    wt = np.exp(-ref[ok]); wt /= wt.sum()
    d -= np.sum(wt * d)
    out = {"pmf2d_rmse_lowf_kT": float(np.sqrt(np.sum(wt * d ** 2)))}
    m2 = dens.sum(0); r2 = raw.sum(0)
    from scipy.special import logsumexp  # noqa: PLC0415
    ref2 = -logsumexp(-f2, axis=0); ref2 = np.interp(c2, ax2, ref2); ref2 -= ref2.min()
    ok2 = (r2 >= 5) & (m2 > 0)
    d2 = -np.log(m2[ok2]) - ref2[ok2]
    wt2 = np.exp(-ref2[ok2]); wt2 /= wt2.sum()
    d2 -= np.sum(wt2 * d2)
    out["cv2_pmf_rmse_lowf_kT"] = float(np.sqrt(np.sum(wt2 * d2 ** 2)))
    return out


def run_arm(name: str, arm: str, seed: int) -> Dict[str, Any]:
    from gareus.adaptive import slow_mode_reseed as smr  # noqa: PLC0415
    from .t2_campaign import _log_weights  # noqa: PLC0415
    ls = LANDSCAPES_3D[name]
    ws = windows()
    truth = [exact_high_fraction(name, w) for w in ws]
    rng = np.random.default_rng([seed, 7])
    swarm = _swarm(name, rng)                      # identical in both arms (same seed)
    pos = np.array([swarm[int(np.argmin(w.k1 * (swarm[:, 0] - w.center1) ** 2
                                        + w.k2 * (swarm[:, 1] - w.center2) ** 2))] for w in ws])
    rt = 1.0 / A.beta_real()
    targets = [{"state_id": i, "primary_center": w.center1, "primary_k": A.k_to_kcal(w.k1),
                "secondary_center": w.center2, "secondary_k": A.k_to_kcal(w.k2), "gamd_lambda": 0.0}
               for i, w in enumerate(ws)]
    pooled_x: List[np.ndarray] = [np.empty((0, 3)) for _ in ws]
    records = []
    end_pool: List[np.ndarray] = []            # every epoch's end states (production: all final_pdbs)
    dt = min(2.0e-3, 0.3 / max(w.k1 for w in ws))
    for epoch in range(N_EPOCHS):
        erng = np.random.default_rng([seed, epoch])
        run = run_chains(ls, pos, N_PER_EPOCH, c1=[w.center1 for w in ws], k1=[w.k1 for w in ws],
                         c2=[w.center2 for w in ws], k2=[w.k2 for w in ws], D=D_COEF, dt=dt,
                         steps_per_sample=20, burn_in=BURN_IN, rng=erng)
        pooled_x = [np.concatenate([p, run.samples[i]]) for i, p in enumerate(pooled_x)]
        allz = np.concatenate([p[:, 2] for p in pooled_x])
        boundary, method = smr.split_point(allz)
        frac = [float(np.mean(p[:, 2] > boundary)) for p in pooled_x]
        dev = [abs(f - t) for f, t in zip(frac, truth)]
        one_sided = sum(1 for f in frac if min(f, 1 - f) <= smr.ONE_SIDED_MINORITY_MAX)
        one_sided_truth = sum(1 for t in truth if min(t, 1 - t) <= smr.ONE_SIDED_MINORITY_MAX)
        samples = {i: p[:, :2] for i, p in enumerate(pooled_x)}
        sids, pts, u, window, f, n_k = A.union_mbar(samples, dict(enumerate(ws)))
        pmf = _pmf(name, pts, _log_weights(u, window, f, n_k))
        rec = {"epoch": epoch, "split": [boundary, method], "mean_abs_side_dev": float(np.mean(dev)),
               "max_abs_side_dev": float(np.max(dev)), "n_one_sided": one_sided,
               "n_one_sided_truth": one_sided_truth, **pmf, "n_reseeded": 0}
        pos = run.final.copy()
        if arm == "x3" and epoch < N_EPOCHS - 1:
            sides = np.concatenate([(p[:, 2] > boundary).astype(int) for p in pooled_x])
            sids_rows = np.concatenate([np.full(len(p), i) for i, p in enumerate(pooled_x)])
            occ = smr.side_occupancy(sides, sids_rows)
            end_pool.append(run.final.copy())
            pool = np.concatenate(end_pool)
            n_w = len(ws)
            cands = [{"path": f"end_{j}", "cv1": float(x[0]), "cv2": float(x[1]), "side": int(x[2] > boundary),
                      "hidden_z": float(x[2]), "source_state_id": j % n_w} for j, x in enumerate(pool)]
            plan = smr.plan_reseed(targets, occ, cands, fraction=RESEED_FRACTION, rt_kcal=rt)
            harmful = 0
            for r in plan["reseeded"]:
                tid = int(r["target_state_id"])
                pos[tid] = pool[int(r["seed_path"].split("_")[1])]
                t = truth[tid]
                # the equilibrium itself is one-sided and the reseed goes to its minority side
                if min(t, 1 - t) <= smr.ONE_SIDED_MINORITY_MAX and (int(r["minority_side"]) == 1) == (t < 0.5):
                    harmful += 1
            rec.update(n_reseeded=int(plan["n_reseeded"]), n_eligible=int(plan["n_eligible"]),
                       n_reseeded_into_equilibrium_minority=harmful,
                       skipped_reasons=sorted({s["reason"] for s in plan["skipped"]}))
        records.append(rec)
    return {"landscape": name, "arm": arm, "seed": seed, "truth_high_fraction": truth, "epochs": records,
            "final_high_fraction": frac}


def _job(args):
    return run_arm(*args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, arm, s) for n in LANDSCAPES_3D for s in a.seeds for arm in ("none", "x3")]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_job, jobs))
    (out / "reseed.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
