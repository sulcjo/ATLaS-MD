#!/usr/bin/env python3
"""Tier 0 of the Boltzmann-validation programme: can GaMD reweighting work at all?

This is an *analytic* feasibility check on data you already have. It answers, per
window and per CV1 bin, whether the boost distribution is inside the domain where
each reweighting estimator is defensible -- before any compute is spent on a
validation campaign.

Three readouts, in increasing order of how badly they fail:

``beta*sigma_dV``
    The cumulant expansion truncated at second order (CE2) carries a leading
    error ~ beta^3*kappa_3/6, so it is defensible only for beta*sigma_dV <~ 1.
    Past that the expansion is not perturbative and CE2 is not "approximate",
    it is unbounded.

``ESS`` of the exp(+beta*dV) reweighting weights
    The exponential estimator is exact in principle and useless in practice once
    a handful of frames carry all the weight. Computed PER WINDOW: the pooled
    figure reads ~0 spuriously because it mixes states whose free energies differ.

``c1/c2/c3`` cumulant spread
    The established divergence sentinel. If c3 is not small against c2, the
    truncation is not converging and CE2's own error estimate is meaningless.

Usage:
    python audit_reweighting_feasibility.py RUNS/chignolin_6 [--bins 12] [--temp 300]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np

# kJ/mol/K -- matches gareus.units
_R_KJ = 0.008314462618
_KJ_PER_KCAL = 4.184

# CE2 is a perturbative expansion in beta*dV; past ~1 it is not small.
CE2_VALID_BETA_SIGMA = 1.0
# Below this fraction the exponential estimator is carried by too few frames.
ESS_FRACTION_FLOOR = 0.01


def _ess(logw: np.ndarray) -> float:
    """Kish effective sample size from log-weights, overflow-safe.

    Mirrors gareus.math_helpers.ess but takes logs, because exp(+beta*dV) with
    beta*dV ~ 5 overflows float64 long before the weights themselves matter.
    """
    finite = logw[np.isfinite(logw)]
    if finite.size == 0:
        return 0.0
    m = float(finite.max())
    w = np.exp(finite - m)
    s1 = float(w.sum())
    s2 = float((w * w).sum())
    return (s1 * s1 / s2) if s2 > 0.0 else 0.0


def _load(run_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(window_id, cv1, gamd_boost_total_kj) pooled over every phase."""
    import duckdb

    pats = [
        os.path.join(run_dir, "adaptive_production", "*", "samples", "seg_*", "*.parquet"),
        os.path.join(run_dir, "adaptive_production", "*", "*", "samples", "seg_*", "*.parquet"),
    ]
    files: list[str] = []
    for p in pats:
        files.extend(sorted(glob.glob(p)))
    if not files:
        raise SystemExit(f"no sample parquet found under {run_dir}")
    con = duckdb.connect()
    q = (
        "select window_id, cv1, gamd_boost_total from read_parquet(%r) "
        "where gamd_boost_total is not null" % files
    )
    tbl = con.execute(q).fetchnumpy()
    return (
        np.asarray(tbl["window_id"]).astype(np.int32),
        np.asarray(tbl["cv1"]).astype(np.float64),
        np.asarray(tbl["gamd_boost_total"]).astype(np.float64),
    )


def _row(label: str, bdv: np.ndarray) -> dict:
    n = int(bdv.size)
    mean = float(bdv.mean())
    sd = float(bdv.std(ddof=1)) if n > 1 else float("nan")
    # Central moments -> cumulants. c1=mean, c2=variance, c3=third central moment.
    d = bdv - mean
    c2 = float((d * d).mean())
    c3 = float((d * d * d).mean())
    ess = _ess(bdv)  # weights are exp(+beta*dV), so log-weights ARE beta*dV
    return {
        "label": label,
        "n": n,
        "mean": mean,
        "beta_sigma": sd,
        "c2": c2,
        "c3": c3,
        # |c3| / c2^{3/2} is the standardised skew; the CE2 truncation error is
        # governed by c3, so this is the ratio that says whether truncating helps.
        "skew": (c3 / (c2 ** 1.5)) if c2 > 0 else float("nan"),
        "ess": ess,
        "ess_frac": (ess / n) if n else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--temp", type=float, default=300.0, help="temperature in K")
    ap.add_argument("--bins", type=int, default=12, help="CV1 bins for the per-bin table")
    args = ap.parse_args(argv)

    beta = 1.0 / (_R_KJ * float(args.temp))
    win, cv1, dv_kj = _load(args.run_dir)
    bdv = beta * dv_kj

    print(f"run          {args.run_dir}")
    print(f"temperature  {args.temp} K   (beta = {beta:.5f} mol/kJ)")
    print(f"samples      {bdv.size:,} with a finite boost")
    print()
    print(f"CE2 is defensible for beta*sigma_dV <~ {CE2_VALID_BETA_SIGMA}; "
          f"exp-reweighting needs ESS/N >~ {ESS_FRACTION_FLOOR:.0%}")
    print()

    overall = _row("ALL POOLED", bdv)
    per_window = [_row(f"w{w:02d}", bdv[win == w]) for w in np.unique(win)]

    edges = np.linspace(np.nanmin(cv1), np.nanmax(cv1), int(args.bins) + 1)
    idx = np.clip(np.digitize(cv1, edges) - 1, 0, int(args.bins) - 1)
    per_bin = []
    for b in range(int(args.bins)):
        m = idx == b
        if int(m.sum()) >= 100:
            per_bin.append(_row(f"[{edges[b]:.3f},{edges[b+1]:.3f})", bdv[m]))

    def table(title: str, rows: list[dict]) -> None:
        print(title)
        print(f"  {'':<22} {'n':>10} {'<bdV>':>9} {'b*sigma':>9} {'skew':>7} "
              f"{'ESS':>10} {'ESS/N':>8}  verdict")
        for r in rows:
            bad_ce2 = r["beta_sigma"] > CE2_VALID_BETA_SIGMA
            bad_exp = r["ess_frac"] < ESS_FRACTION_FLOOR
            verdict = ("CE2 invalid" if bad_ce2 else "CE2 ok") + \
                      ("; exp dead" if bad_exp else "; exp ok")
            print(f"  {r['label']:<22} {r['n']:>10,} {r['mean']:>9.2f} "
                  f"{r['beta_sigma']:>9.2f} {r['skew']:>7.2f} "
                  f"{r['ess']:>10.1f} {r['ess_frac']:>8.2%}  {verdict}")
        print()

    table("PER WINDOW", per_window)
    table(f"PER CV1 BIN ({len(per_bin)} populated)", per_bin)
    table("POOLED (for reference only -- mixes states, reads low spuriously)",
          [overall])

    n_ce2_bad = sum(1 for r in per_window if r["beta_sigma"] > CE2_VALID_BETA_SIGMA)
    n_exp_bad = sum(1 for r in per_window if r["ess_frac"] < ESS_FRACTION_FLOOR)
    worst = max(per_window, key=lambda r: r["beta_sigma"])
    print("VERDICT")
    print(f"  windows where CE2 is outside its validity domain : "
          f"{n_ce2_bad}/{len(per_window)}")
    print(f"  windows where exp-reweighting has ESS/N < {ESS_FRACTION_FLOOR:.0%}      : "
          f"{n_exp_bad}/{len(per_window)}")
    print(f"  worst beta*sigma_dV                              : "
          f"{worst['beta_sigma']:.2f}  ({worst['label']}), "
          f"i.e. {worst['beta_sigma'] / CE2_VALID_BETA_SIGMA:.1f}x the CE2 domain")
    if n_ce2_bad or n_exp_bad:
        print()
        print("  => No GaMD reweighting estimator on this data is trustworthy at the")
        print("     ~1 kcal/mol level. This is a property of the boost width, not of")
        print("     sampling length: more frames do not shrink beta*sigma_dV.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
