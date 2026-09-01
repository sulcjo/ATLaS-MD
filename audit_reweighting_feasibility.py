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
    is uniform across the CV does not matter -- the reportable quantity is the
    *spread across CV bins*, which is what changes a PMF's shape.

    That spread is reported by three estimators which are allowed to disagree,
    because on a wide boost they do: the measured third-order term (a floor, not
    the tail), a closed-form value for the fitted noncentral chi-square (exact
    for that family only -- two moments do not constrain the higher cumulants),
    and the model-free empirical average. The empirical one needs the very
    exp(+beta*dV) average whose variance is infinite once a > 0.25, so when that
    holds the verdict is NOT RESOLVED rather than any single number.

    An earlier version summed a geometric tail at ratio 2a from the third-order
    term. That is wrong: 2a governs successive terms WITHIN a bin at fixed
    (a, lambda), not the spread ACROSS bins where both vary.

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
    """Every phase with both samples and its own window map, crashed ones excluded.

    A phase the driver abandoned is renamed with a ``_CRASHED_<date>`` suffix and
    a fresh one takes its place. Its samples are a partial, aborted segment and
    do not belong in a feasibility statistic -- on the run this was written
    against the crashed remnant held 72 rows against 1.7M in its replacement.
    """
    out, skipped = [], []
    for pat in ("*/samples", "*/*/samples"):
        for p in sorted(glob.glob(os.path.join(run_dir, "adaptive_production", pat))):
            d = os.path.dirname(p)
            if not os.path.isfile(os.path.join(d, "epoch_window_map.csv")):
                continue
            if "_CRASHED" in d:
                skipped.append(os.path.relpath(d, run_dir))
                continue
            out.append(d)
    if skipped:
        print(f"  note: skipping {len(skipped)} abandoned phase(s): {', '.join(skipped)}")
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
    sids, cvs, dvs, keys, regs = [], [], [], [], []
    for phase_idx, phase in enumerate(_phase_dirs(run_dir)):
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
        if int((~ok).sum()):
            print(f"  WARNING {int((~ok).sum()):,} rows in {os.path.relpath(phase, run_dir)} have a "
                  "window_id absent from that phase's own map; dropped. See "
                  "docs/chignolin_6_low_ess_root_cause.md")
        sids.append(mapped[ok])
        cvs.append(np.asarray(t["cv1"]).astype(float)[ok])
        dvs.append(np.asarray(t["gamd_boost_total"]).astype(float)[ok])
        # epoch_000 ran under the PRE-recalibration GaMD envelope; every later
        # phase ran under the recalibrated one. Same split the shipped pipeline
        # applies (`_epoch_zero_split_masks` in analyze_gareus_mbar.py).
        is_e0 = os.path.basename(phase) == "epoch_000" or os.sep + "epoch_000" + os.sep in phase
        regs.append(np.full(int(ok.sum()), 0 if is_e0 else 1, dtype=np.int8))
        # Packed into one int64 so the de-duplication is a single sort rather
        # than a lexsort over 13M rows.
        #
        # The PHASE INDEX is part of the key, and must be: `step` is
        # phase-local, not campaign-absolute. Every phase of this run starts at
        # step 460,100, so a key of (state, replica, step) alone silently merges
        # genuinely distinct samples from different phases. Measured before this
        # was fixed: 349,703 collisions, concentrated in the phase that was added
        # last. Same failure as treating a phase-local `window_id` as a state id.
        rep = np.asarray(t["replica"]).astype(np.int64)[ok]
        stp = np.asarray(t["step"]).astype(np.int64)[ok]
        if rep.size and (rep.max() >= 1 << 6 or stp.max() >= 1 << 31
                         or mapped[ok].max() >= 1 << 6 or phase_idx >= 1 << 6):
            raise SystemExit("key packing would overflow; widen the shifts")
        keys.append((phase_idx << 43) | (mapped[ok] << 37) | (rep << 31) | stp)

    if not sids:
        raise SystemExit(f"no mapped samples under {run_dir}")
    sid = np.concatenate(sids)
    cv1 = np.concatenate(cvs)
    dv = np.concatenate(dvs)
    reg = np.concatenate(regs)
    key = np.concatenate(keys)
    _u, first = np.unique(key, return_index=True)
    first.sort()
    n_raw = sid.size
    return sid[first], cv1[first], dv[first], reg[first], n_raw


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
    sid, cv1, dv_kj, regime, n_raw = _load(args.run_dir)
    bdv = beta * dv_kj

    print(f"run          {args.run_dir}")
    print(f"samples      {bdv.size:,} after mapping to state_id and de-duplicating "
          f"({n_raw:,} raw, {100.0 * (n_raw - bdv.size) / max(1, n_raw):.1f}% dropped)")
    print(f"states       {len(set(sid.tolist()))} distinct state_ids")
    print()

    # The two GaMD regimes are not comparable and must not be pooled into one
    # `a`. The shared-envelope recalibration fires at most once, so epoch_000 ran
    # under a different boost envelope from every later phase -- the pipeline
    # already splits its own PMF and boost report on exactly this
    # (`epoch_000_separate/`). Pooling also inflates `a` mechanically, since
    # a ~ var/(4*mean) and a between-regime mean offset adds to the variance.
    n_e0 = int((regime == 0).sum())
    if n_e0 and n_e0 < bdv.size:
        print("BOOST REGIMES (epoch_000 ran pre-recalibration; the rest, post-)")
        print(f"  {'':<22} {'n':>12} {'<bdV>':>8} {'b*sig':>8} {'anharm':>8} {'a':>8}")
        for lbl, msk in (("epoch_000 (pre-recal)", regime == 0),
                         ("epoch_001+final (post)", regime == 1),
                         ("pooled (NOT valid)", np.ones_like(regime, dtype=bool))):
            r = _row(lbl, bdv[msk])
            print(f"  {lbl:<22} {r['n']:>12,} {r['mean']:>8.2f} {r['bsig']:>8.2f} "
                  f"{r['anh']:>8.3f} {r['a']:>8.3f}")
        print("  The last row is shown only to expose the pooling artefact; the")
        print("  headline verdict below uses the POST-recalibration samples, which")
        print("  are the ones the main PMF report covers.")
        print()
        keep = regime == 1
        sid, cv1, bdv = sid[keep], cv1[keep], bdv[keep]

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

    # CE2 truncation error per CV bin, by three estimators that disagree in an
    # informative way. What matters is the SPREAD ACROSS BINS, since a PMF is
    # defined only up to a constant -- a uniform error cancels, a bin-dependent
    # one changes the PMF's shape.
    #
    #   empirical   ln(mean(e^X)) - CE2, model-free but needs the very exponential
    #               average whose variance is infinite when a > 0.25
    #   parametric  the closed form for the fitted noncentral chi-square,
    #               lam*a/(1-2a) - 0.5*ln(1-2a) - CE2. Exact FOR THAT FAMILY, but
    #               (a, lam) come from two moments only, and distributions sharing
    #               two moments can have arbitrarily different ln<e^X>. A
    #               sensitivity analysis, not a bound.
    #   third       the directly measured 3rd-order term. A floor, not the tail.
    #
    # An earlier version summed a geometric tail at ratio 2a and reported the
    # result as if it were the answer. That is wrong: 2a governs successive terms
    # WITHIN a bin at fixed (a, lam), not the spread ACROSS bins. When a varies
    # between bins the n*a^(n-1) sensitivity makes the spread ratio behave like
    # 2a(n+1)/n -- about 1.0 at n=3->4, not 0.754 -- and the measured spreads
    # (0.23 then 0.38) confirm the extrapolation fails.
    edges = np.linspace(np.nanmin(cv1), np.nanmax(cv1), int(args.bins) + 1)
    which = np.clip(np.digitize(cv1, edges) - 1, 0, int(args.bins) - 1)
    emp, par, third, ess_frac = [], [], [], []
    for b in range(int(args.bins)):
        msk = which == b
        if int(msk.sum()) < 5000:
            continue
        y = bdv[msk]
        x = y - y.mean()
        m2 = float(np.mean(x ** 2))
        ce2 = float(y.mean()) + m2 / 2.0
        third.append(kT_kcal * float(np.mean(x ** 3)) / 6.0)
        # empirical, in logs so the exponential does not overflow
        mx = float(y.max())
        lse = mx + math.log(float(np.exp(y - mx).sum()))
        ln_mean = lse - math.log(y.size)
        emp.append(kT_kcal * (ln_mean - ce2))
        # how many of this bin's samples the exponential average actually uses
        lse2 = 2.0 * mx + math.log(float(np.exp(2.0 * (y - mx)).sum()))
        ess_frac.append(math.exp(2.0 * lse - lse2) / y.size)
        # parametric
        a_b = _chi2_scale(y)
        if math.isfinite(a_b) and 0.0 < a_b < 0.5:
            lam_b = float(y.mean()) / a_b - 1.0
            exact = lam_b * a_b / (1.0 - 2.0 * a_b) - 0.5 * math.log(1.0 - 2.0 * a_b)
            par.append(kT_kcal * (exact - ce2))

    def _spread(v):
        return (max(v) - min(v)) if len(v) > 1 else float("nan")
    if not (len(third) == len(emp) == len(par)):
        # Never tabulate spreads computed over different bin sets. `par` skips a
        # bin whose fitted `a` leaves the valid range, and silently comparing a
        # 10-bin spread against an 11-bin one is the same "dropped with no
        # message" defect this audit exists to catch.
        print(f"  WARNING estimator bin counts differ (3rd={len(third)} "
              f"parametric={len(par)} empirical={len(emp)}); spreads below are NOT "
              "comparable. A bin's fitted `a` left the valid range 0 < a < 0.5.")
    spread = _spread(third)
    spread_emp = _spread(emp)
    spread_par = _spread(par)
    worst_ess = min(ess_frac) if ess_frac else float("nan")
    n_bins_used = len(third)
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
    print("    CE2 is exact for a Gaussian boost at ANY width, so b*sigma alone")
    print("    condemns nothing; only non-Gaussianity does. And only the spread")
    print("    across the CV matters -- a uniform offset cancels from a PMF.")
    print()
    print(f"    truncation error, spread across {n_bins_used} CV1 bins (kcal/mol):")
    print(f"      3rd-order term only   {spread:7.2f}   a floor, not the tail")
    print(f"      parametric (chi^2)    {spread_par:7.2f}   exact for the FITTED FAMILY only")
    print(f"      empirical             {spread_emp:7.2f}   model-free, but see below")
    print(f"    worst-bin exponential-average ESS: {100.0 * worst_ess:.4f}% of that bin's samples")
    if pooled["a"] >= EXP_VAR_FINITE_A:
        print()
        print("    VERDICT: NOT RESOLVED. The empirical column needs the same")
        print("    exp(+beta*dV) average whose variance is infinite at this a, so it is")
        print("    not a measurement -- it is one draw from a distribution with no")
        print("    finite spread. The parametric column is a sensitivity analysis: two")
        print("    moments fix CE2 but do not constrain the higher cumulants, and")
        print("    distributions sharing two moments can have arbitrarily different")
        print("    ln<e^X>. The honest statement is that CE2's truncation error on this")
        print(f"    run is unresolved, with a measured FLOOR of {spread:.2f} kcal/mol.")
        print(f"    (Against a {CE2_SPREAD_BAR_KCAL:.1f} kcal/mol accuracy target -- a chosen")
        print("    convention, not a derived threshold.)")
    else:
        agree = abs(spread_emp - spread_par)
        print(f"    empirical and parametric agree to {agree:.2f} kcal/mol; "
              f"exponential weights have finite variance, so the empirical column stands.")
    print()
    _report_boost_setting(args.run_dir, pooled["a"])
    return 0



def _production_gamd_setup(run_dir: str):
    """The GaMD setup that governed PRODUCTION, chosen explicitly.

    An earlier version took ``sorted(glob(...))[0]``, which is positional and
    therefore picks whatever sorts first. On a real run that is
    ``adaptive_feedback_round_01/`` -- the short diagnostic pilot, which
    calibrates against its own sampling and reports a DIFFERENT k0'
    (1.766 there against 1.694 in every production phase). Since the clip
    boundary is sigma0p/k0', reading the pilot moves the recommendation.

    Same failure shape as reading a phase-local ``window_id`` as if it were a
    state id: a plausible artifact that is not the right one. Selection is now
    by explicit preference, and the pilot is only used if nothing else exists.
    """
    root = os.path.join(run_dir, "adaptive_production")
    preferred = os.path.join(root, "global_shared_gamd_setup", "shared_gamd_setup_globals.json")
    if os.path.isfile(preferred):
        return preferred
    prod = [f for f in sorted(glob.glob(os.path.join(root, "**", "shared_gamd_setup_globals.json"),
                                        recursive=True))
            if "CRASHED" not in f]
    if prod:
        return prod[0]
    any_ = sorted(glob.glob(os.path.join(run_dir, "**", "shared_gamd_setup_globals.json"),
                            recursive=True))
    if any_:
        print(f"  WARNING falling back to {os.path.relpath(any_[0], run_dir)} -- this may be an "
              "adaptive-feedback pilot, whose calibration differs from production.")
        return any_[0]
    return None


def _report_boost_setting(run_dir: str, a: float) -> None:
    """The boost width is a knob, so say which way to turn it -- past the clip.

    dV = k0*(Vmax-V)^2 / (2*(Vmax-Vmin)) is pointwise proportional to k0, so
    mean(beta*dV) ~ k0 and var(beta*dV) ~ k0^2, and since
    a ~ var/(4*mean) whenever the variance is small against the squared mean,
    **a is linear in k0**.

    But k0 = min(1, k0') and k0' is proportional to sigma0. When k0' > 1 the
    boost is CLIPPED: lowering sigma0 does nothing at all until k0' drops below
    1. An earlier version of this function ignored that and recommended
    sigma0p * (target_a / a) directly -- which, on this run, lands at 1.66
    kcal/mol where k0' is still 1.17 and the boost is completely unchanged.
    The scaling only starts at the clip boundary sigma0 = sigma0p / k0'.
    """
    import json

    path = _production_gamd_setup(run_dir)
    if path is None:
        print("  NOTE no shared_gamd_setup_globals.json found; cannot report k0.")
        return
    cands = [path]
    d = json.load(open(path))
    g = d.get("interesting_globals") or {}
    k0 = next((v for k, v in g.items() if k.startswith("k0_")), None)
    k0p = next((v for k, v in g.items() if k.startswith("k0prime_")), None)
    s0 = d.get("sigma0p_kcal_mol")
    print(f"  BOOST SETTING ({d.get('gamd_boost_type')}, from {os.path.relpath(cands[0], run_dir)})")
    print(f"    sigma0p = {s0} kcal/mol,  k0 = {k0},  k0' = {k0p}")
    if not (k0 and k0p and s0):
        return
    if k0p > 1.0:
        clip = s0 / k0p
        print(f"    k0 is CLIPPED at 1.0 (k0' = {k0p:.3f}): the run is at maximum boost")
        print(f"    and the requested sigma0 is NOT what is being applied. Every value")
        print(f"    of sigma0p between {clip:.2f} and {s0} kcal/mol gives the IDENTICAL")
        print("    boost -- lowering it within that range changes nothing.")
    else:
        clip = float(s0)
    if a > EXP_VAR_FINITE_A:
        need = clip * (EXP_VAR_FINITE_A / a)
        print(f"    To reach a < {EXP_VAR_FINITE_A} (finite weight variance): a is linear in k0,")
        print(f"    and k0 only starts falling below sigma0p = {clip:.2f}, so")
        print(f"    sigma0p ~ {need:.2f} kcal/mol -- NOT {s0 * EXP_VAR_FINITE_A / a:.2f}, which")
        print("    is the answer you get by ignoring the clip.")
        print("    Caveat: this assumes Vmax/Vmin/sigmaV are unchanged by the new")
        print("    setting. They are re-measured at calibration, so treat it as a")
        print("    starting point and re-run this audit on the result.")


if __name__ == "__main__":
    sys.exit(main())
