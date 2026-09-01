#!/usr/bin/env python3
"""Tier 0: which GaMD reweighting estimator, if any, can work on this run?

An analytic feasibility check on data you already have. It answers, per umbrella
state, whether the boost distribution is inside the domain where each estimator
is defensible -- before any compute is spent.

The two estimators fail for *different* reasons and must be judged separately.
Lumping them, as an earlier version of this script did, gets both wrong.

**Exponential reweighting** (weights exp(+beta*dV)).
    For a lower-bound GaMD boost, dV = 0.5*k*(E-V)^2, so beta*dV is a scaled
    noncentral chi-square with one degree of freedom: beta*dV ~ a * chi'^2_1(lam).
    Matching moments gives a = m - sqrt(m^2 - v/2) for mean m and variance v.
    The MGF E[exp(t*X)] exists only for t < 1/(2a), so

        a < 0.50  =>  E[w]  is finite   (the estimator is defined)
        a < 0.25  =>  E[w^2] is finite  (it has finite variance)

    Above a = 0.25 the importance weights have infinite variance: Kish ESS has
    no finite limit, does not grow with N, and any single number quoted for it
    is an artefact of which extreme frame happened to be drawn. That is a
    structural verdict, not a sampling-quality complaint.

**Cumulant expansion to second order (CE2).**
    CE2 is *exact* for a Gaussian dV at any width -- all cumulants above the
    second vanish -- so a threshold on beta*sigma alone is meaningless. What
    matters is non-Gaussianity, which this repo already measures:
    ``gareus.math_helpers.boost_anharmonicity`` (skew and excess kurtosis
    combined), labelled OK/WARN/BAD at 0.5/1.0.

    And because a PMF is defined only up to a constant, a truncation error that
    is uniform across the CV does not matter. The reportable quantity is the
    *spread across CV bins* of the neglected third-order term, in kcal/mol.

Usage:
    python audit_reweighting_feasibility.py RUNS/chignolin_6 [--bins 12]
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

from gareus.math_helpers import anharmonicity_label, boost_anharmonicity

_R_KJ = 0.008314462618
_KCAL_PER_KJ = 1.0 / 4.184

# Finite weight variance for the exponential estimator (see module docstring).
EXP_VAR_FINITE_A = 0.25
EXP_MEAN_FINITE_A = 0.50
# A PMF is defined up to a constant, so this is a spread across bins, not a level.
CE2_SPREAD_BAR_KCAL = 1.0


def _phase_dirs(run_dir: str) -> list[str]:
    out = []
    for pat in ("*/samples", "*/*/samples"):
        for p in sorted(glob.glob(os.path.join(run_dir, "adaptive_production", pat))):
            d = os.path.dirname(p)
            if os.path.isfile(os.path.join(d, "epoch_window_map.csv")):
                out.append(d)
    return out


def _load(run_dir: str):
    """(state_id, cv1, beta_dV) pooled over phases, mapped and de-duplicated.

    Two things an earlier version of this script got wrong, both of which this
    codebase has been bitten by before:

    * ``window_id`` in the Parquet is a **phase-local** index. Pooling on it
      merges different physical states under one label -- exactly the bug the
      2026-08-25 window-map work exists to fix. Every row is mapped through its
      own phase's ``epoch_window_map.csv``.
    * Four segment directories hold a consolidated ``data.parquet`` *and*
      un-deleted ``chunk_*.parquet`` covering the same steps (the known
      ``_consolidate`` leftover), inflating N by ~8%. Rows are de-duplicated on
      (state, step, replica).
    """
    import duckdb

    con = duckdb.connect()
    sids, cvs, dvs, keys = [], [], [], []
    for phase in _phase_dirs(run_dir):
        with open(os.path.join(phase, "epoch_window_map.csv")) as fh:
            amap = {int(r["epoch_window"]): int(r["state_id"]) for r in csv.DictReader(fh)}
        files = sorted(glob.glob(os.path.join(phase, "samples", "seg_*", "*.parquet")))
        if not files:
            continue
        t = con.execute(
            "select window_id, replica, step, cv1, gamd_boost_total from read_parquet(%r) "
            "where gamd_boost_total is not null" % files
        ).fetchnumpy()
        w = np.asarray(t["window_id"]).astype(np.int64)
        mapped = np.array([amap.get(int(x), -1) for x in w], dtype=np.int64)
        ok = mapped >= 0
        sids.append(mapped[ok])
        cvs.append(np.asarray(t["cv1"]).astype(float)[ok])
        dvs.append(np.asarray(t["gamd_boost_total"]).astype(float)[ok])
        # Packed into one int64 so the de-duplication is a single sort rather
        # than a lexsort over 12M rows. step < 2^31 and replica < 2^6 are
        # asserted, so the packing is injective.
        rep = np.asarray(t["replica"]).astype(np.int64)[ok]
        stp = np.asarray(t["step"]).astype(np.int64)[ok]
        if rep.size and (rep.max() >= 1 << 6 or stp.max() >= 1 << 31 or mapped[ok].max() >= 1 << 16):
            raise SystemExit("key packing would overflow; widen the shifts")
        keys.append((mapped[ok] << 37) | (rep << 31) | stp)

    if not sids:
        raise SystemExit(f"no mapped samples under {run_dir}")
    sid = np.concatenate(sids)
    cv1 = np.concatenate(cvs)
    dv = np.concatenate(dvs)
    key = np.concatenate(keys)
    _u, first = np.unique(key, return_index=True)
    first.sort()
    n_raw = sid.size
    return sid[first], cv1[first], dv[first], n_raw


def _chi2_scale(bdv: np.ndarray) -> float:
    """Fitted `a` in beta*dV ~ a * chi'^2_1(lam), by moment matching."""
    m = float(np.mean(bdv))
    v = float(np.var(bdv, ddof=1))
    disc = m * m - v / 2.0
    if disc < 0.0 or m <= 0.0:
        return float("nan")
    return m - math.sqrt(disc)


def _row(label, bdv):
    n = int(bdv.size)
    an = boost_anharmonicity(list(bdv)) if n >= 8 else {"score": float("nan")}
    return {
        "label": label, "n": n,
        "mean": float(np.mean(bdv)),
        "bsig": float(np.std(bdv, ddof=1)) if n > 1 else float("nan"),
        "anh": float(an.get("score", float("nan"))),
        "a": _chi2_scale(bdv) if n > 8 else float("nan"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--bins", type=int, default=12)
    args = ap.parse_args(argv)

    beta = 1.0 / (_R_KJ * float(args.temp))
    kT_kcal = _R_KJ * float(args.temp) * _KCAL_PER_KJ
    sid, cv1, dv_kj, n_raw = _load(args.run_dir)
    bdv = beta * dv_kj

    print(f"run          {args.run_dir}")
    print(f"samples      {bdv.size:,} after mapping to state_id and de-duplicating "
          f"({n_raw:,} raw, {100.0 * (n_raw - bdv.size) / max(1, n_raw):.1f}% dropped)")
    print(f"states       {len(set(sid.tolist()))} distinct state_ids")
    print()

    per_state = [_row(f"s{s:02d}", bdv[sid == s]) for s in sorted(set(sid.tolist()))]
    real = [r for r in per_state if r["n"] >= 1000]

    print("PER STATE (states with >=1000 samples)")
    print(f"  {'':<8} {'n':>10} {'<bdV>':>8} {'b*sig':>8} {'anharm':>8} {'label':>6} "
          f"{'a':>7}  exponential")
    for r in real:
        lab, _ = anharmonicity_label(r["anh"])
        exp_ok = ("finite var" if r["a"] < EXP_VAR_FINITE_A else
                  "INF VARIANCE" if r["a"] < EXP_MEAN_FINITE_A else "DIVERGENT")
        print(f"  {r['label']:<8} {r['n']:>10,} {r['mean']:>8.2f} {r['bsig']:>8.2f} "
              f"{r['anh']:>8.3f} {lab:>6} {r['a']:>7.3f}  {exp_ok}")
    if len(per_state) != len(real):
        skipped = [r["label"] for r in per_state if r["n"] < 1000]
        print(f"  ({len(skipped)} state(s) with <1000 samples omitted: {', '.join(skipped)})")
    print()

    pooled = _row("pooled", bdv)
    lab, _ = anharmonicity_label(pooled["anh"])

    # CE2: what matters is the SPREAD across CV bins of the neglected terms,
    # since a PMF is defined only up to a constant. Build the cumulant-expansion
    # PMF correction per bin at orders 2, 3 and 4 and report how much the CURVE
    # moves when the next term is added -- that increment, not the absolute size
    # of any single cumulant, is the error CE2 imposes on the PMF.
    edges = np.linspace(np.nanmin(cv1), np.nanmax(cv1), int(args.bins) + 1)
    which = np.clip(np.digitize(cv1, edges) - 1, 0, int(args.bins) - 1)
    curves = {2: [], 3: [], 4: []}
    for b in range(int(args.bins)):
        m = which == b
        if int(m.sum()) < 5000:
            continue
        y = bdv[m]
        x = y - y.mean()
        m2 = float(np.mean(x ** 2))
        m3 = float(np.mean(x ** 3))
        m4 = float(np.mean(x ** 4))
        k1, k2, k3 = float(y.mean()), m2, m3
        k4 = m4 - 3.0 * m2 * m2
        curves[2].append(kT_kcal * (k1 + k2 / 2.0))
        curves[3].append(kT_kcal * (k1 + k2 / 2.0 + k3 / 6.0))
        curves[4].append(kT_kcal * (k1 + k2 / 2.0 + k3 / 6.0 + k4 / 24.0))
    def _spread(v):
        return (max(v) - min(v)) if len(v) > 1 else float("nan")
    spread = _spread([c3 - c2 for c2, c3 in zip(curves[2], curves[3])])
    spread4 = _spread([c4 - c3 for c3, c4 in zip(curves[3], curves[4])])
    n_bins_used = len(curves[2])

    print("VERDICT, per estimator -- they fail for different reasons")
    print()
    print(f"  exponential  a = {pooled['a']:.3f}")
    if pooled["a"] >= EXP_VAR_FINITE_A:
        print(f"    a >= {EXP_VAR_FINITE_A}: the importance weights have INFINITE VARIANCE.")
        print("    Kish ESS has no finite limit and does not grow with N, so any single")
        print("    ESS figure is an artefact of which extreme frame was drawn.")
        print(f"    (E[w] itself is {'finite' if pooled['a'] < EXP_MEAN_FINITE_A else 'DIVERGENT'};"
              f" the estimator is {'defined but unusable' if pooled['a'] < EXP_MEAN_FINITE_A else 'undefined'}.)")
    else:
        print("    finite weight variance: usable.")
    print()
    print(f"  CE2          anharmonicity = {pooled['anh']:.3f} -> {lab}"
          f"   (b*sigma = {pooled['bsig']:.2f}, informational only)")
    print(f"    over {n_bins_used} CV1 bins, adding the next cumulant moves the PMF")
    print(f"    correction curve by  CE2->CE3 {spread:.2f}  CE3->CE4 {spread4:.2f} kcal/mol")
    print(f"    (peak-to-peak across bins, against a {CE2_SPREAD_BAR_KCAL:.1f} kcal/mol bar)")
    print("    CE2 is exact for a Gaussian boost at ANY width, so b*sigma alone")
    print("    condemns nothing; only non-Gaussianity does. And only the spread")
    print("    across the CV matters -- a uniform offset cancels from a PMF.")
    # For X ~ a*chi'^2_1(lam) the cumulants are exactly
    #     kappa_n = a^n * 2^(n-1) * (n-1)! * (1 + n*lam),
    # so successive terms kappa_n/n! shrink by a ratio tending to 2a. The series
    # therefore converges iff a < 0.5 -- the same bound as a finite E[w], which
    # is not a coincidence: the cumulant series IS the log-MGF at t = 1. Summing
    # the geometric tail turns the measured third-order term into an estimate of
    # everything CE2 throws away, not just the first thing it throws away.
    ratio = 2.0 * pooled["a"]
    if 0.0 < ratio < 1.0:
        tail = spread / (1.0 - ratio)
        print(f"    successive cumulant terms shrink by ~{ratio:.3f} per order "
              f"(exactly 2a for this family),")
        print(f"    so the FULL neglected tail is about {tail:.2f} kcal/mol of "
              f"CV-dependent distortion")
        print(f"    against the {CE2_SPREAD_BAR_KCAL:.1f} kcal/mol bar -- "
              f"{'inside it' if tail < CE2_SPREAD_BAR_KCAL else 'over it'}, "
              "but only just, either way.")
    else:
        print(f"    ratio 2a = {ratio:.3f} >= 1: the cumulant series does not "
              "converge; CE2 has no truncation guarantee.")
    print()
    _report_boost_setting(args.run_dir, pooled["a"])
    return 0


def _report_boost_setting(run_dir: str, a: float) -> None:
    """The boost width is a knob, so say which way to turn it.

    a ~ var(beta*dV) / (4*mean(beta*dV)) whenever the variance is small against
    the squared mean, and both moments scale with the boost strength, so `a` is
    roughly LINEAR in it. That makes the required change a simple ratio rather
    than a search.
    """
    import json

    cands = sorted(glob.glob(os.path.join(run_dir, "**", "shared_gamd_setup_globals.json"),
                             recursive=True))
    if not cands:
        print("  NOTE no shared_gamd_setup_globals.json found; cannot report k0.")
        return
    d = json.load(open(cands[0]))
    g = d.get("interesting_globals") or {}
    k0 = next((v for k, v in g.items() if k.startswith("k0_")), None)
    k0p = next((v for k, v in g.items() if k.startswith("k0prime_")), None)
    s0 = d.get("sigma0p_kcal_mol")
    print(f"  BOOST SETTING ({d.get('gamd_boost_type')}, from {os.path.relpath(cands[0], run_dir)})")
    print(f"    sigma0p = {s0} kcal/mol,  k0 = {k0},  k0' = {k0p}")
    if k0 is not None and k0p is not None and k0p > 1.0:
        print(f"    k0 is CLIPPED at 1.0 (k0' = {k0p:.3f}), so the run is at maximum")
        print(f"    boost and the requested sigma0 is not what is being applied.")
        if s0:
            print(f"    Lowering sigma0p to ~{s0 / k0p:.2f} kcal/mol would un-clip it and")
            print("    return control of the boost width to the setting.")
    if a > EXP_VAR_FINITE_A and s0:
        print(f"    For a < {EXP_VAR_FINITE_A} (finite weight variance), a is ~linear in")
        print(f"    boost strength, so roughly sigma0p <= {s0 * EXP_VAR_FINITE_A / a:.2f} kcal/mol.")


if __name__ == "__main__":
    sys.exit(main())
