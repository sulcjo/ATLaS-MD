"""T2 coupling-gate calibration (spec 3.4): how CV2-to-CV1 coupling distorts a CV1 ladder.

Surface ``coupled_tilt(a)``: F = 0.5 F''_1 (c - 0.5)^2 + 0.5 F''_y (z - a (c - 0.5))^2, so the
conditional mean of CV2 moves with CV1 at slope dz/dc = a and the CV1 PMF is the same for
every a (0.5 F''_1 (c - 0.5)^2). A CV1 ladder (8 windows, spacing 0.1, k1 designed by the
curvature rule k1 = RT / sigma_w1^2 - F''_1 with sigma_w1 = spacing / 1.5) carries one CV2
row with spring k2 at either

* ``flat``: z0 = 0 for every window (a CV2 row that does not follow the CV1 trend), or
* ``ridge``: z0 = a (c1 - 0.5) (a row on the conditional mean, as a residual CV2 is).

The spec's coupling fraction is the rigid bound cf = k2 a^2 / (k1 + F''_1) (the degree-1 case
of ``cv2_coupling_fraction``: its d2z/dc2 term vanishes); the curvature the umbrella really
adds is k2 a^2 F''_y / (k2 + F''_y) (the z-direction relaxes), cf_eff = that / (k1 + F''_1).

Measured, per (a, k2, row): exact (quadrature) CV1 sd / sigma_w1, CV1 mean shift in sigma_w1,
adjacent pairwise overlap along the ladder (min, median, and the a = 0 reference), and the
MBAR CV1-PMF error at a matched budget (MALA, several seeds). ``python -m
gareus.synth.t2_coupling --out DIR``.
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
from .landscapes import coupled_tilt
from .sampler import Window
from .vec_langevin import run_chains

F1 = 16.0
FY = 60.0
CENTRES = np.round(np.linspace(0.15, 0.85, 8), 6)
SPACING = float(CENTRES[1] - CENTRES[0])
SIGMA_W1 = SPACING / 1.5
K1 = 1.0 / SIGMA_W1 ** 2 - F1
SLOPES = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4)
K2S = (10.0, 30.0, 60.0, 120.0)
N_SAMPLES = 3000
N_SEEDS = 6


def layout(a: float, k2: float, row: str) -> List[Window]:
    return [Window(float(c), K1, float(a * (c - 0.5)) if row == "ridge" else 0.0, float(k2)) for c in CENTRES]


def cf_spec(a: float, k2: float) -> float:
    return k2 * a * a / (K1 + F1)


def cf_eff(a: float, k2: float) -> float:
    return (k2 * a * a * FY / (k2 + FY)) / (K1 + F1)


def exact_ladder(a: float, k2: float, row: str) -> Dict[str, Any]:
    ls = coupled_tilt(a, cv1_curvature=F1, cv2_curvature=FY)
    ws = layout(a, k2, row)
    mom = [A.exact_window_moments(ls, w) for w in ws]
    ov = [A.exact_pair_overlap(ls, ws[i], ws[i + 1])["overlap"] for i in range(len(ws) - 1)]
    return {"sd_ratio": [m["sd1"] / SIGMA_W1 for m in mom],
            "mean_shift_sigma": [(m["mean1"] - w.center1) / SIGMA_W1 for m, w in zip(mom, ws)],
            "adjacent_overlap": ov}


def _pmf_rmse(pooled, log_w) -> float:
    edges = np.linspace(0.0, 1.0, 61)
    x = pooled[:, 0]
    dens, _ = np.histogram(x, bins=edges, weights=np.exp(log_w))
    raw, _ = np.histogram(x, bins=edges)
    c = 0.5 * (edges[1:] + edges[:-1])
    ref = 0.5 * F1 * (c - 0.5) ** 2
    ok = (raw >= 5) & (dens > 0)
    est = -np.log(dens[ok])
    wt = np.exp(-ref[ok]); wt /= wt.sum()
    d = est - ref[ok]
    d -= np.sum(wt * d)
    return float(np.sqrt(np.sum(wt * d ** 2)))


def pmf_error(a: float, k2: float, row: str, seed: int) -> Dict[str, float]:
    from .t2_campaign import _log_weights  # noqa: PLC0415
    ls = coupled_tilt(a, cv1_curvature=F1, cv2_curvature=FY)
    ws = layout(a, k2, row)
    rng = np.random.default_rng([seed, int(1000 * a), int(k2), row == "ridge"])
    x0 = np.array([[w.center1, w.center2] for w in ws])
    dt = min(2.0e-3, 0.3 / max(K1 + F1, 0.05 * (k2 + FY)))
    run = run_chains(ls, x0, N_SAMPLES, c1=[w.center1 for w in ws], k1=K1, c2=[w.center2 for w in ws], k2=k2,
                     D=(1.0, 0.05), dt=dt, steps_per_sample=20, burn_in=500, rng=rng)
    samples = {i: run.samples[i] for i in range(len(ws))}
    sids, pooled, u, window, f, n_k = A.union_mbar(samples, dict(enumerate(ws)))
    lo_cov = float(np.mean([(np.abs(run.samples[:, :, 0] - c) < 0.5 * SPACING).mean() for c in CENTRES]))
    return {"cv1_pmf_rmse_kT": _pmf_rmse(pooled, _log_weights(u, window, f, n_k)), "coverage": lo_cov}


def _job(args):
    a, k2, row = args
    ex = exact_ladder(a, k2, row)
    errs = [pmf_error(a, k2, row, s) for s in range(N_SEEDS)]
    rm = np.array([e["cv1_pmf_rmse_kT"] for e in errs])
    return {"a": a, "k2": k2, "row": row, "cf_spec": cf_spec(a, k2), "cf_eff": cf_eff(a, k2),
            "sd_ratio_min": float(min(ex["sd_ratio"])), "sd_ratio_median": float(np.median(ex["sd_ratio"])),
            "predicted_sd_ratio_eff": 1.0 / math.sqrt(1.0 + cf_eff(a, k2)),
            "max_abs_mean_shift_sigma": float(np.max(np.abs(ex["mean_shift_sigma"]))),
            "min_adjacent_overlap": float(min(ex["adjacent_overlap"])),
            "median_adjacent_overlap": float(np.median(ex["adjacent_overlap"])),
            "pmf_rmse_mean_kT": float(rm.mean()), "pmf_rmse_sd_kT": float(rm.std(ddof=1)),
            "pmf_rmse_seeds": rm.tolist()}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, k2, row) for row in ("flat", "ridge") for k2 in K2S for s in SLOPES]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_job, jobs))
    meta = {"F1": F1, "FY": FY, "centres": CENTRES.tolist(), "sigma_w1": SIGMA_W1, "k1": K1,
            "n_samples": N_SAMPLES, "n_seeds": N_SEEDS}
    (out / "coupling.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
