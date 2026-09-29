"""T2 threshold calibration: the spec 3.1 edge metric against the analytic overlap.

For window pairs on known surfaces the exact pairwise-MBAR-scale overlap is computed
by quadrature (:func:`collector_adapter.exact_pair_overlap`). Each pair is sampled by
MALA chains (exact equilibrium, real autocorrelation; one chain per window, started at
the window centre), prefixes of increasing length give increasing N_eff, the P4 stride
subsample (<= 2000 pairs per state, ``paired_cv.stride_indices``) is taken exactly as
the collector does, and the SHIPPED ``edge_metric.evaluate_edge`` grades the edge
(point, bootstrap q10/q90, blocking N_eff, tau plateau flag). The adapter test checks
that the collector path gives the identical record for the same samples.

``python -m gareus.synth.t2_calibration --out DIR`` writes ``calibration_edges.json``
(one row per pair x length) and ``calibration_summary.json`` (rates, coverage, df error).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import collector_adapter as A
from .landscapes import LANDSCAPES
from .sampler import Window
from .vec_langevin import run_chains

# landscape -> (k1 range, k2 range) in kBT/CV^2 (log-uniform draws)
CAL_LANDSCAPES = {
    "harmonic-bowl": ((80.0, 600.0), (8.0, 120.0)),
    "slow-cv2-double-branch": ((80.0, 600.0), (8.0, 120.0)),
    "gated-barrier": ((80.0, 600.0), (8.0, 120.0)),
    "narrow-cv2-band-at-high-cv1": ((80.0, 600.0), (2.0, 30.0)),
    "plateau-walls": ((80.0, 600.0), (8.0, 120.0)),
}
LENGTHS = (250, 500, 1000, 2000, 4000, 8000, 16000)
THRESHOLDS = (0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30)
NEFF_FLOORS = (0, 50, 100, 200, 400, 800)
D_COEF = (1.0, 0.05)
STEPS_PER_SAMPLE = 20
BURN_IN_STEPS = 1000
MAX_PAIRS = 2000
TRAPPED_SD = 0.5          # sampled mean off the exact window mean by > 0.5 exact sd: not equilibrated


def _target_weight(o: float) -> float:
    """Acceptance weight for a candidate pair's truth overlap (dense near 0.15)."""
    if o < 0.01 or o > 0.45:
        return 0.0
    return 1.0 if 0.08 <= o <= 0.25 else 0.35


def draw_pairs(name: str, n_pairs: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Window pairs on ``name`` whose truth overlaps spread over 0.01-0.45, dense at 0.08-0.25."""
    ls = LANDSCAPES[name]
    (k1lo, k1hi), (k2lo, k2hi) = CAL_LANDSCAPES[name]
    c1g, c2g, f = ls.grid(res=80)
    low = np.argwhere(f <= 4.0)
    out: List[Dict[str, Any]] = []
    tries = 0
    while len(out) < n_pairs and tries < 40 * n_pairs:
        tries += 1
        i, j = low[rng.integers(len(low))]
        c1, c2 = float(c1g[i]), float(c2g[j])
        k1 = float(np.exp(rng.uniform(math.log(k1lo), math.log(k1hi))))
        k2 = float(np.exp(rng.uniform(math.log(k2lo), math.log(k2hi))))
        axis = ["cv1", "cv2", "diag"][int(rng.integers(3))]
        kb1 = k1 * float(np.exp(rng.uniform(-0.35, 0.35)))
        kb2 = k2 * float(np.exp(rng.uniform(-0.35, 0.35)))
        s1 = math.sqrt(1.0 / k1 + 1.0 / kb1)
        s2 = math.sqrt(1.0 / k2 + 1.0 / kb2)
        t = float(rng.uniform(0.6, 4.5))
        d1, d2 = {"cv1": (t * s1, 0.0), "cv2": (0.0, t * s2),
                  "diag": (t * s1 / math.sqrt(2), t * s2 / math.sqrt(2))}[axis]
        wa = Window(c1 - 0.5 * d1, k1, c2 - 0.5 * d2, k2)
        wb = Window(c1 + 0.5 * d1, kb1, c2 + 0.5 * d2, kb2)
        lo1, hi1 = ls.cv1_bounds
        lo2, hi2 = ls.cv2_bounds
        if not (lo1 < wa.center1 < hi1 and lo1 < wb.center1 < hi1 and lo2 < wa.center2 < hi2 and lo2 < wb.center2 < hi2):
            continue
        tr = A.exact_pair_overlap(ls, wa, wb)
        if rng.uniform() > _target_weight(tr["overlap"]):
            continue
        out.append({"landscape": name, "axis": axis, "wa": wa, "wb": wb, "truth": tr["overlap"],
                    "truth_delta_f": tr["delta_f"], "mom_a": A.exact_window_moments(ls, wa),
                    "mom_b": A.exact_window_moments(ls, wb)})
    return out


SAMPLERS = ("mala", "mala-eq", "exact")


def sample_pairs(name: str, pairs: Sequence[Dict[str, Any]], n_max: int, rng: np.random.Generator,
                 sampler: str = "mala") -> np.ndarray:
    """(n_pairs, 2, n_max, 2) samples; chain 2p is window a, 2p+1 window b of pair p.

    ``mala``: MALA chains started at the window centre with a BURN_IN_STEPS burn-in (the
    calibration's default); ``mala-eq``: MALA chains started from an exact equilibrium draw,
    no burn-in (removes the start transient, keeps the autocorrelation); ``exact``: iid
    draws from the grid sampler (res 600), the estimator's own iid reference.
    """
    ls = LANDSCAPES[name]
    ws = [w for p in pairs for w in (p["wa"], p["wb"])]
    if sampler == "exact":
        from .sampler import sample_window_exact  # noqa: PLC0415
        xs = np.stack([sample_window_exact(ls, w, n_max, rng=rng, res=600) for w in ws])
        return xs.reshape(len(pairs), 2, n_max, 2), float("nan")
    c1 = np.array([w.center1 for w in ws]); k1 = np.array([w.k1 for w in ws])
    c2 = np.array([w.center2 for w in ws]); k2 = np.array([w.k2 for w in ws])
    stiff = np.maximum(D_COEF[0] * k1, D_COEF[1] * (k2 + 2000.0 * (name == "plateau-walls")))
    dt = np.minimum(2.0e-3, 0.3 / stiff)
    if sampler == "mala-eq":
        from .sampler import sample_window_exact  # noqa: PLC0415
        x0 = np.concatenate([sample_window_exact(ls, w, 1, rng=rng, res=600) for w in ws])
        burn = 0
    else:
        x0 = np.column_stack([c1, c2])
        burn = BURN_IN_STEPS
    run = run_chains(ls, x0, n_max, c1=c1, k1=k1, c2=c2, k2=k2, D=D_COEF, dt=dt,
                     steps_per_sample=STEPS_PER_SAMPLE, burn_in=burn, rng=rng)
    return run.samples.reshape(len(pairs), 2, n_max, 2), run.acceptance


def _restraint(w: Window):
    from gareus.adaptive.edge_metric import Restraint  # noqa: PLC0415
    return Restraint(primary_center=w.center1, primary_k=A.k_to_kcal(w.k1),
                     secondary_center=w.center2, secondary_k=A.k_to_kcal(w.k2))


def _eval_job(job) -> Dict[str, Any]:
    from gareus.adaptive import edge_metric as em  # noqa: PLC0415
    from gareus.adaptive.paired_cv import stride_indices  # noqa: PLC0415
    wa, wb, xa, xb = job["wa"], job["wb"], job["xa"], job["xb"]
    ia, _ = stride_indices(len(xa), MAX_PAIRS)
    ib, _ = stride_indices(len(xb), MAX_PAIRS)
    a = em.StateSamples(0, _restraint(wa), xa[ia, 0], xa[ia, 1], np.zeros(ia.size, dtype=np.int32))
    b = em.StateSamples(1, _restraint(wb), xb[ib, 0], xb[ib, 1], np.zeros(ib.size, dtype=np.int32))
    res = em.evaluate_edge(a, b, A.beta_real(), min_neff=0.0)
    keep = ("status", "overlap", "overlap_point", "overlap_lower", "overlap_upper", "n", "n_eff", "tau",
            "tau_plateau", "g", "delta_f_kT")
    return {k: res.get(k) for k in keep}


def run_landscape(name: str, n_pairs: int, seed: int, workers: int, *, sampler: str = "mala",
                  lengths: Sequence[int] = LENGTHS) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    pairs = draw_pairs(name, n_pairs, rng)          # same pairs for every sampler (same seed)
    xs, acc = sample_pairs(name, pairs, max(lengths), rng, sampler=sampler)
    jobs, meta = [], []
    for p, pair in enumerate(pairs):
        for n in lengths:
            jobs.append({"wa": pair["wa"], "wb": pair["wb"], "xa": xs[p, 0, :n], "xb": xs[p, 1, :n]})
            ma, mb = pair["mom_a"], pair["mom_b"]
            dev = []
            for mom, x in ((ma, xs[p, 0, :n]), (mb, xs[p, 1, :n])):
                dev.append(max(abs(x[:, 0].mean() - mom["mean1"]) / mom["sd1"],
                               abs(x[:, 1].mean() - mom["mean2"]) / mom["sd2"]))
            meta.append({"landscape": name, "pair": p, "axis": pair["axis"], "n_raw": int(n),
                         "truth": pair["truth"], "truth_delta_f": pair["truth_delta_f"],
                         "max_mean_dev_sd": float(max(dev)),
                         "k1": [pair["wa"].k1, pair["wb"].k1], "k2": [pair["wa"].k2, pair["wb"].k2],
                         "mala_acceptance": acc, "sampler": sampler})
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_eval_job, jobs, chunksize=4))
    return [dict(m, **r) for m, r in zip(meta, results)]


# ---------------------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------------------

def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else round(num / den, 4)


def rates(rows: Sequence[Dict[str, Any]], threshold: float, floor: float) -> Dict[str, Any]:
    meas = [r for r in rows if r["overlap_upper"] is not None and min(r["n_eff"]) >= floor]
    unmeas = len(rows) - len(meas)
    strong = [r for r in meas if r["truth"] >= threshold]
    weak = [r for r in meas if r["truth"] < threshold]
    fw = sum(1 for r in strong if r["overlap_upper"] < threshold)
    fs = sum(1 for r in weak if r["overlap_upper"] >= threshold)
    fs_point = sum(1 for r in weak if r["overlap"] >= threshold)
    fw_point = sum(1 for r in strong if r["overlap"] < threshold)
    bands = {}
    for lo, hi in ((0.0, 0.02), (0.02, 0.05), (0.05, 1.0)):
        sub = [r for r in weak if lo <= threshold - r["truth"] < hi]
        bands[f"weak_margin_{lo:g}-{hi:g}"] = {"n": len(sub), "false_strong": _rate(
            sum(1 for r in sub if r["overlap_upper"] >= threshold), len(sub))}
        sub = [r for r in strong if lo <= r["truth"] - threshold < hi]
        bands[f"strong_margin_{lo:g}-{hi:g}"] = {"n": len(sub), "false_weak": _rate(
            sum(1 for r in sub if r["overlap_upper"] < threshold), len(sub))}
    return {"threshold": threshold, "neff_floor": floor, "n_rows": len(rows), "n_unmeasured": unmeas,
            "n_truth_strong": len(strong), "n_truth_weak": len(weak),
            "false_weak": _rate(fw, len(strong)), "false_strong": _rate(fs, len(weak)),
            "false_weak_point": _rate(fw_point, len(strong)), "false_strong_point": _rate(fs_point, len(weak)),
            "bands": bands}


NEFF_BINS = ((0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1e9))


def by_neff(rows, threshold: float) -> List[Dict[str, Any]]:
    out = []
    for lo, hi in NEFF_BINS:
        sub = [r for r in rows if r["overlap_upper"] is not None and lo <= min(r["n_eff"]) < hi]
        strong = [r for r in sub if r["truth"] >= threshold]
        weak = [r for r in sub if r["truth"] < threshold]
        cover = [r for r in sub if r["overlap_lower"] <= r["truth"] <= r["overlap_upper"]]
        width = [r["overlap_upper"] - r["overlap_lower"] for r in sub]
        # detectable margin: smallest (threshold - truth) band where >= 90 % of truth-weak edges are flagged
        det = None
        for m in (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15):
            deep = [r for r in weak if threshold - r["truth"] >= m]
            if len(deep) >= 5 and sum(1 for r in deep if r["overlap_upper"] < threshold) >= 0.9 * len(deep):
                det = m
                break
        out.append({"neff_bin": [lo, hi], "n": len(sub),
                    "false_weak": _rate(sum(1 for r in strong if r["overlap_upper"] < threshold), len(strong)),
                    "false_strong": _rate(sum(1 for r in weak if r["overlap_upper"] >= threshold), len(weak)),
                    "n_truth_weak": len(weak), "n_truth_strong": len(strong),
                    "q10_q90_coverage": _rate(len(cover), len(sub)),
                    "median_q10_q90_width": None if not width else round(float(np.median(width)), 4),
                    "tau_not_plateaued": _rate(sum(1 for r in sub if not all(r["tau_plateau"])), len(sub)),
                    "detectable_margin_90pct": det,
                    "median_abs_overlap_error": None if not sub else round(float(np.median(
                        [abs(r["overlap"] - r["truth"]) for r in sub])), 4)})
    return out


OV_BINS = ((0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 0.5))


def delta_f_error(rows) -> List[Dict[str, Any]]:
    """|df_est - df_exact| (kT) by truth overlap, at three N_eff levels (the threshold's grounding)."""
    out = []
    for nlo, nhi in ((100, 400), (400, 1000), (1000, 1e9)):
        for lo, hi in OV_BINS:
            sub = [r for r in rows if r["delta_f_kT"] is not None and lo <= r["truth"] < hi
                   and nlo <= min(r["n_eff"]) < nhi]
            err = np.array([abs(r["delta_f_kT"] - r["truth_delta_f"]) for r in sub])
            out.append({"neff_bin": [nlo, nhi], "truth_overlap_bin": [lo, hi], "n": int(err.size),
                        "median_abs_df_err_kT": None if not err.size else round(float(np.median(err)), 3),
                        "p90_abs_df_err_kT": None if not err.size else round(float(np.quantile(err, 0.9)), 3),
                        "frac_err_gt_0.5kT": None if not err.size else round(float(np.mean(err > 0.5)), 3)})
    return out


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    eq = [r for r in rows if r["max_mean_dev_sd"] <= TRAPPED_SD]
    return {
        "n_rows": len(rows), "n_pairs": len({(r["landscape"], r["pair"]) for r in rows}),
        "n_rows_equilibrated": len(eq),
        "rates_all": [rates(rows, t, f) for t in THRESHOLDS for f in NEFF_FLOORS],
        "rates_equilibrated": [rates(eq, t, f) for t in THRESHOLDS for f in NEFF_FLOORS],
        "by_neff_0.15_all": by_neff(rows, 0.15),
        "by_neff_0.15_equilibrated": by_neff(eq, 0.15),
        "by_neff_by_threshold_equilibrated": {str(t): by_neff(eq, t) for t in THRESHOLDS},
        "delta_f_error_equilibrated": delta_f_error(eq),
        "delta_f_error_all": delta_f_error(rows),
        "per_landscape_0.15_floor200": {name: rates([r for r in rows if r["landscape"] == name], 0.15, 200)
                                        for name in sorted({r["landscape"] for r in rows})},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--pairs-per-landscape", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--landscapes", nargs="*", default=list(CAL_LANDSCAPES))
    p.add_argument("--sampler", choices=SAMPLERS, default="mala")
    p.add_argument("--lengths", nargs="*", type=int, default=list(LENGTHS))
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for i, name in enumerate(a.landscapes):
        t0 = time.time()
        rows.extend(run_landscape(name, a.pairs_per_landscape, a.seed + 1000 * i, a.workers,
                                  sampler=a.sampler, lengths=a.lengths))
        print(f"[t2-cal] {name}: {time.time() - t0:.0f} s, {len(rows)} rows", flush=True)
    (out / "calibration_edges.json").write_text(json.dumps(rows, default=float))
    (out / "calibration_summary.json").write_text(json.dumps(summarise(rows), indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
