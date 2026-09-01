#!/usr/bin/env python3
"""Tier 2: is each umbrella window's sample Boltzmann for its recorded bias?

Distribution-free, runs on production data you already have, needs no analytic
model and no reference simulation.

For two states k, l sampled from the same unbiased ensemble under harmonic
biases u_k, u_l (in kT), Boltzmann sampling implies exactly

    p_k(x) / p_l(x) = exp[(f_k - f_l) - du(x)],      du = u_k(x) - u_l(x)

Pool both states' samples and label each by its origin. The probability that a
pooled sample at x came from k is then a logistic function of du:

    P(k | x) = logistic( ln(n_k/n_l) + (f_k - f_l) - du(x) )

so a logistic regression of the origin label on du has **slope exactly -1**,
whatever the underlying free-energy surface is. The intercept absorbs the
unknown free-energy difference AND the unequal sample sizes, which is why no
reference is needed and why n_k != n_l is harmless (with a free intercept the
logistic slope is consistent under outcome-dependent sampling).

This is the Bennett/Crooks acceptance-ratio identity in regression form. It
holds with GaMD ON whenever the boost is a function of force groups that
exclude the umbrella, because dV is then k-independent and cancels from the
ratio.

WHICH AXIS A PAIR TESTS
-----------------------
This is the thing an earlier version of this script got silently wrong, and it
is the whole reason the pairing is explicit here.

A 2D window grid has windows differing in cv1, in cv2, or in both. If a pair
shares identical (c1, k1), the CV1 harmonic cancels from `du` to machine
precision and **du is a pure function of cv2** -- the fit says nothing at all
about the primary restraint. The earlier version paired windows by sorting on
c1, which (because `sorted` is stable) walked *within* each c1 tie-group and
produced 28 pairs that were every one of them CV2-only. It then reported that
as a test of "applied k or centre differing from the recorded one", i.e. of the
flagship contact CV, which it had never touched.

So both pairings are built explicitly and reported SEPARATELY, never pooled:

    cv2-axis pairs : identical (c1, k1), adjacent in c2  -> tests the secondary
    cv1-axis pairs : identical (c2, k2), adjacent in c1  -> tests the primary

Reporting them separately also buys a free, model-free consistency check: a
wrong temperature is ONE scalar and must move both axes the same way. Two axes
deviating in opposite directions rules a global beta error out, which no single
pooled "effective temperature" number can express.

POOLING
-------
Fixed-effect inverse-variance pooling assumes every pair estimates the same
slope. Cochran's Q tests that, and on real data it is rejected on both axes --
p ~ 2e-03, I^2 ~ 50% at the shipped 8000-frame detection head, and p ~ 1e-06,
I^2 ~ 67% at a 4000-frame head -- so the fixed-effect interval is invalid however
the per-pair SEs were computed. DerSimonian-Laird random effects is reported as
the headline and the fixed-effect number is shown only for comparison.

An earlier version instead widened the fixed-effect CI by the observed
instability in `g`. That reasoning was wrong twice over: the true full-trace g
ratio is ~11x, not the ~2.6x it measured off truncated heads, and a joint block
bootstrap shows the slope is far less sensitive to slow drift than the CV mean
is, so 11x would over-correct. It landed near a defensible interval by luck.
Heterogeneity, not autocorrelation, is what invalidates the pooling.

Detects: applied k or centre differing from the recorded one (slope scales as
k_applied/k_recorded); a wrong temperature (slope scales as beta_app/beta_rec);
windows that never equilibrated; and a boost type whose dV does not cancel.

Measured power on real data: with the random-effects SEs this script actually
reports (~0.012), 2-sigma detection needs |slope + 1| > ~0.024, i.e. it resolves
a mis-scaling of recorded k or beta of about 2.4% and no better. (A ~1.7% figure
derived from a tighter SE is quoted in some notes; it corresponds to a narrower
interval than this script defends.)

Blind to: GaMD *reweighting* error, MBAR solver error, Jacobian handling, and
any non-Boltzmann behaviour in coordinates orthogonal to (cv1, cv2). It is a
statement about the sampler, not about the free energies you extract from it.

Usage:
    python audit_crooks_slope.py RUNS/chignolin_6/adaptive_production/epoch_000
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from typing import NamedTuple

import numpy as np

from gareus.mbar_subsample import equilibrated_subsample

_R_KJ = 0.008314462618
NULL_SLOPE = -1.0
FWER = 0.05


class Win(NamedTuple):
    c1: float
    k1_kj: float
    c2: float
    k2_kj: float


class Fit(NamedTuple):
    label: str
    n_k: int
    n_l: int
    sd_du: float
    slope: float
    se: float
    status: str          # fitted | separated | too_few | singular


def _windows(phase: str) -> dict[int, Win]:
    """{window: Win} from the phase's own window table.

    Reads `primary_center` / `primary_openmm_k` and `secondary_cv_center` /
    `secondary_cv_k_kj_mol`: the columns that actually exist, already in kJ, so
    there is no unit conversion here to get wrong. An earlier version read
    `primary_cv_center` / `primary_cv_k_kcal`, which are not in this schema at
    all -- it fell through to the legacy `distance_center_A` /
    `distance_k_kcal_mol_A2` aliases. Numerically identical on this run, but
    those name an Angstrom distance while the CV is a dimensionless contact
    fraction, so the fallback was one schema change away from being wrong.
    """
    path = os.path.join(phase, "umbrella_explicit_windows.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"no umbrella_explicit_windows.csv in {phase}")
    out: dict[int, Win] = {}
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    need = {"window", "primary_center", "primary_openmm_k"}
    missing = need - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path} is missing required column(s) {sorted(missing)}")
    for row in rows:
        def _f(key):
            v = (row.get(key) or "").strip()
            return float(v) if v else float("nan")
        out[int(row["window"])] = Win(
            _f("primary_center"), _f("primary_openmm_k"),
            _f("secondary_cv_center"), _f("secondary_cv_k_kj_mol"),
        )
    return out


def _check_window_map(phase: str, wins: dict[int, Win]) -> str:
    """The window_id -> state join, on a campaign whose headline bug was a stale map.

    Not decorative. `window_id` in the Parquet is phase-local, and this run's
    known defect is exactly a map that stopped matching its window table after
    windows were auto-dropped. A Tier 2 verdict computed across a bad join would
    be meaningless in a way none of the statistics below could reveal.
    """
    path = os.path.join(phase, "epoch_window_map.csv")
    if not os.path.isfile(path):
        return "no epoch_window_map.csv (single-phase layout?)"
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != len(wins):
        return (f"MISMATCH: map has {len(rows)} rows, window table has {len(wins)} "
                "-- see docs/chignolin_6_low_ess_root_cause.md")
    bad = 0
    for r in rows:
        w = wins.get(int(r["epoch_window"]))
        if w is None:
            bad += 1
            continue
        if not math.isclose(float(r["primary_center"]), w.c1, abs_tol=1e-9):
            bad += 1
    ident = all(int(r["epoch_window"]) == int(r["state_id"]) for r in rows)
    return (f"{len(rows)} rows, {'identity' if ident else 'NON-identity'} map, "
            f"{bad} centre mismatch(es)")


def _samples(phase: str):
    import duckdb

    files = sorted(glob.glob(os.path.join(phase, "samples", "seg_*", "*.parquet")))
    if not files:
        raise SystemExit(f"no sample parquet under {phase}/samples")
    con = duckdb.connect()
    t = con.execute(
        "select window_id, step, cv1, cv2 from read_parquet(%r) order by window_id, step" % files
    ).fetchnumpy()
    return (np.asarray(t["window_id"]).astype(int), np.asarray(t["step"]).astype(np.int64),
            np.asarray(t["cv1"]).astype(float), np.asarray(t["cv2"]).astype(float))


def _bias_kt(cv1, cv2, w: Win, beta: float):
    """beta * u_w(x) for one window's recorded parameters."""
    u = 0.5 * w.k1_kj * (cv1 - w.c1) ** 2
    if math.isfinite(w.c2) and math.isfinite(w.k2_kj) and w.k2_kj > 0.0:
        with np.errstate(invalid="ignore"):
            d2 = cv2 - w.c2
        u = u + np.where(np.isfinite(d2), 0.5 * w.k2_kj * d2 * d2, np.nan)
    return beta * u


def _logistic_slope(du, label):
    """MLE slope of label ~ logistic(a + b*du), with the slope's Wald SE.

    Newton-Raphson on two parameters. Verified against sklearn (penalty=None)
    and scipy BFGS to 8 significant figures, and the returned SE against both
    the analytic Wald SE and an HC0 sandwich (<7% apart).

    Returns a status, because the failure mode here is silent: two states whose
    `du` supports are perfectly separated have no finite MLE, Newton diverges,
    every weight underflows, and the old version of this function returned NaN
    which the caller skipped WITHOUT PRINTING ANYTHING. Three real pairs were
    being dropped that way, and "28 pairs fitted" was the output of an
    unreported failure. Perfect separation is itself an MBAR-relevant finding
    (those two states share no support at all), so it is now reported.
    """
    k = np.asarray(label, dtype=float)
    a_du, b_du = du[k == 1.0], du[k == 0.0]
    if a_du.size == 0 or b_du.size == 0:
        return float("nan"), float("nan"), "too_few"
    if max(a_du.min(), b_du.min()) >= min(a_du.max(), b_du.max()):
        return float("nan"), float("nan"), "separated"

    X = np.column_stack([np.ones_like(du), du])
    beta = np.zeros(2)
    H = None
    converged = False
    for _ in range(200):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
        W = p * (1.0 - p)
        if not np.any(W > 1e-12):
            return float("nan"), float("nan"), "separated"
        H = (X * W[:, None]).T @ X
        try:
            step = np.linalg.solve(H, X.T @ (k - p))
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), "singular"
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            converged = True
            break
    if not converged or H is None:
        return float("nan"), float("nan"), "singular"
    try:
        se = float(math.sqrt(abs(np.linalg.inv(H)[1, 1])))
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), "singular"
    return float(beta[1]), se, "fitted"


def _pool(fits: list[Fit]) -> dict:
    """Fixed-effect and DerSimonian-Laird random-effects pooling, plus Cochran Q."""
    ok = [f for f in fits if f.status == "fitted" and math.isfinite(f.se) and f.se > 0]
    if len(ok) < 2:
        return {"n": len(ok)}
    y = np.array([f.slope for f in ok])
    v = np.array([f.se for f in ok]) ** 2
    w = 1.0 / v
    fe = float((w * y).sum() / w.sum())
    fe_se = float(math.sqrt(1.0 / w.sum()))
    Q = float((w * (y - fe) ** 2).sum())
    df = len(ok) - 1
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    ws = 1.0 / (v + tau2)
    re = float((ws * y).sum() / ws.sum())
    re_se = float(math.sqrt(1.0 / ws.sum()))
    i2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    # survival function of chi2_df without scipy
    try:
        from math import erfc
        # Wilson-Hilferty normal approximation, adequate at these df
        z = ((Q / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
        p = 0.5 * erfc(z / math.sqrt(2.0))
    except Exception:
        p = float("nan")
    return {"n": len(ok), "fe": fe, "fe_se": fe_se, "re": re, "re_se": re_se,
            "Q": Q, "df": df, "p": p, "i2": i2, "tau2": tau2}


def _pairings(wins: dict[int, Win]) -> dict[str, list[tuple[int, int]]]:
    """Adjacent pairs along each axis, holding the OTHER axis exactly fixed."""
    def _grp(keyfn, sortfn):
        buckets: dict[tuple, list[int]] = {}
        for w, p in wins.items():
            buckets.setdefault(keyfn(p), []).append(w)
        pairs = []
        for ws in buckets.values():
            ws = sorted(ws, key=lambda w: sortfn(wins[w]))
            pairs += list(zip(ws, ws[1:]))
        return pairs

    def _r(x):
        return round(float(x), 9) if math.isfinite(x) else "nan"

    return {
        "cv2": _grp(lambda p: (_r(p.c1), _r(p.k1_kj)), lambda p: p.c2),
        "cv1": _grp(lambda p: (_r(p.c2), _r(p.k2_kj)), lambda p: p.c1),
    }


def _fit_axis(pairs, wins, kept, cv1, cv2, beta, min_overlap) -> list[Fit]:
    fits = []
    for a, b in pairs:
        if a not in kept or b not in kept:
            continue
        ia, ib = kept[a], kept[b]
        lbl = f"w{a:02d}-w{b:02d}"
        if ia.size < min_overlap or ib.size < min_overlap:
            fits.append(Fit(lbl, ia.size, ib.size, float("nan"),
                            float("nan"), float("nan"), "too_few"))
            continue
        idx = np.concatenate([ia, ib])
        du = (_bias_kt(cv1[idx], cv2[idx], wins[a], beta)
              - _bias_kt(cv1[idx], cv2[idx], wins[b], beta))
        lab = np.concatenate([np.ones(ia.size), np.zeros(ib.size)])
        ok = np.isfinite(du)
        du, lab = du[ok], lab[ok]
        slope, se, status = _logistic_slope(du, lab)
        fits.append(Fit(lbl, int(ia.size), int(ib.size), float(np.std(du)) if du.size else float("nan"),
                        slope, se, status))
    return fits


def _report_axis(name, what, fits, temp):
    print(f"\n{'=' * 78}\n{name}   ({what})\n{'=' * 78}")
    print(f"  {'pair':<12} {'n_k':>7} {'n_l':>7} {'sd(du)':>8} {'slope':>9} "
          f"{'SE':>7} {'z vs -1':>8}  status")
    ok = [f for f in fits if f.status == "fitted"]
    bonf = _bonferroni_z(len(ok))
    for f in fits:
        if f.status != "fitted":
            print(f"  {f.label:<12} {f.n_k:>7} {f.n_l:>7} {'-':>8} {'-':>9} {'-':>7} "
                  f"{'-':>8}  {f.status.upper()}")
            continue
        z = (f.slope - NULL_SLOPE) / f.se
        flag = "REJECT" if abs(z) > bonf else "ok"
        print(f"  {f.label:<12} {f.n_k:>7} {f.n_l:>7} {f.sd_du:>8.2f} {f.slope:>9.4f} "
              f"{f.se:>7.4f} {z:>8.2f}  {flag}")

    p = _pool(fits)
    if p.get("n", 0) < 2:
        print(f"\n  only {p.get('n', 0)} pair(s) fitted -- nothing to pool")
        return None
    n_rej = sum(1 for f in ok if abs((f.slope - NULL_SLOPE) / f.se) > bonf)
    print(f"\n  heterogeneity  Cochran Q = {p['Q']:.1f} on {p['df']} df, "
          f"p = {p['p']:.2g}, I^2 = {p['i2']:.0f}%")
    if p["p"] < 0.05:
        print("    Q is significant: the pairs are NOT estimating one common slope,")
        print("    so the fixed-effect interval below is invalid. Use random effects.")
    print(f"  fixed effect   {p['fe']:+.5f} +/- {p['fe_se']:.5f}"
          f"   z vs -1 = {(p['fe'] - NULL_SLOPE) / p['fe_se']:+.2f}"
          f"   {'(INVALID, see above)' if p['p'] < 0.05 else ''}")
    zre = (p["re"] - NULL_SLOPE) / p["re_se"]
    print(f"  random effects {p['re']:+.5f} +/- {p['re_se']:.5f}"
          f"   z vs -1 = {zre:+.2f}   <-- headline")
    print(f"  per-pair gate  |z| > {bonf:.2f} (Bonferroni, {FWER:.0%} family-wise over "
          f"{len(ok)} tests)")
    print(f"  VERDICT        {len(ok) - n_rej}/{len(ok)} pairs consistent with Boltzmann "
          f"sampling of the recorded bias")
    print(f"  implied scale on this axis' recorded k (or on beta): {abs(p['re']):.4f}"
          f"  -> {temp / abs(p['re']):.1f} K if read as a temperature")
    return p


def _bonferroni_z(n_tests: int) -> float:
    """Two-sided |z| critical value at FWER over n simultaneous tests.

    An earlier version used a hardcoded |z| > 6, which at 28 tests is a
    ~5.5e-08 family-wise test -- so conservative that it reported 28/28 where
    the honest count at 5% FWER is 26/28. Bisection on erfc, no scipy.
    """
    if n_tests < 1:
        return float("inf")
    target = FWER / max(1, n_tests)
    lo, hi = 0.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if math.erfc(mid / math.sqrt(2.0)) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _time_stratified(pairs, wins, kept, step, cv1, cv2, beta, min_overlap, n_strata):
    """Refit with the samples split into time-ordered blocks, and pool.

    A systematic-uncertainty probe, not a better estimate. The trace was
    SIGTERM-interrupted and resumed, and per-segment refits do not bracket the
    whole-trace value -- so the whole-trace number carries a modelling
    uncertainty that appears in none of the intervals above. Reported so the
    size of that uncertainty is visible rather than assumed away.
    """
    out = []
    for a, b in pairs:
        if a not in kept or b not in kept:
            continue
        for s in range(n_strata):
            def _slice(arr):
                q0, q1 = np.quantile(step[arr], [s / n_strata, (s + 1) / n_strata])
                return arr[(step[arr] >= q0) & (step[arr] <= q1)]
            ia, ib = _slice(kept[a]), _slice(kept[b])
            if ia.size < min_overlap or ib.size < min_overlap:
                continue
            idx = np.concatenate([ia, ib])
            du = (_bias_kt(cv1[idx], cv2[idx], wins[a], beta)
                  - _bias_kt(cv1[idx], cv2[idx], wins[b], beta))
            lab = np.concatenate([np.ones(ia.size), np.zeros(ib.size)])
            m = np.isfinite(du)
            sl, se, st = _logistic_slope(du[m], lab[m])
            if st == "fitted":
                out.append(Fit(f"{a}-{b}/s{s}", int(ia.size), int(ib.size),
                               float(np.std(du[m])), sl, se, st))
    return _pool(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", help="a phase dir, e.g. .../adaptive_production/epoch_000")
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--min-overlap", type=int, default=200,
                    help="min subsampled frames per state to attempt a pair")
    ap.add_argument("--detect-head", type=int, default=8000,
                    help="contiguous frames used to estimate g; grows automatically "
                         "if the equilibration transient outlasts it")
    ap.add_argument("--strata", type=int, default=4,
                    help="time-ordered blocks for the systematic-uncertainty probe")
    args = ap.parse_args(argv)

    beta = 1.0 / (_R_KJ * float(args.temp))
    wins = _windows(args.phase)
    wid, step, cv1, cv2 = _samples(args.phase)

    kept, gs, exhausted = {}, {}, []
    for w in sorted(set(wid.tolist())):
        m = np.flatnonzero(wid == w)
        if m.size < 50:
            continue
        res = equilibrated_subsample(cv1[m], max_detect_points=int(args.detect_head))
        if res.status != "subsampled":
            print(f"  note: window {w} equilibration status={res.status}")
        if res.head_exhausted:
            exhausted.append(w)
        kept[w] = m[res.indices]
        gs[w] = res.g

    print(f"phase        {args.phase}")
    print(f"temperature  {args.temp} K")
    print(f"window map   {_check_window_map(args.phase, wins)}")
    print(f"windows      {len(kept)} with usable traces "
          f"(median g = {np.nanmedian(list(gs.values())):.1f})")
    if exhausted:
        print(f"  WARNING equilibration detection was exhausted on window(s) "
              f"{exhausted}: g there is unreliable and too small.")

    # g-convergence probe, on a few windows only (detect_equilibration is O(n^2)
    # and uncapped costs minutes PER WINDOW at these trace lengths -- measured).
    #
    # This is a WARNING, not a correction. An earlier version widened the pooled
    # CI by sqrt(this ratio), which was wrong on both counts: the ratio measured
    # off truncated heads understated the true one by ~4x, and a joint block
    # bootstrap shows the slope is far less sensitive to slow drift than the CV
    # mean, so the full ratio would over-correct. Dispersion between pairs is
    # already absorbed by the random-effects tau^2 below. What the probe is
    # legitimately good for is saying that g is not converged, and therefore
    # that every interval printed below is a LOWER BOUND.
    probe_wins = sorted(kept)[:: max(1, len(kept) // 4)][:4]
    ratios = []
    for w in probe_wins:
        m = np.flatnonzero(wid == w)
        wide = equilibrated_subsample(cv1[m], max_detect_points=2 * int(args.detect_head))
        if math.isfinite(wide.g) and wide.g > 0 and math.isfinite(gs[w]) and gs[w] > 0:
            ratios.append(max(wide.g, gs[w]) / min(wide.g, gs[w]))
    if ratios:
        med = float(np.median(ratios))
        print(f"  g-convergence  doubling the detection head changes g by a median "
              f"factor of {med:.1f}")
        if med > 1.2:
            print("    g is NOT converged, so n_eff is overstated and every interval")
            print("    below is a LOWER bound. Not applied as a correction -- see the")
            print("    module docstring for why that reasoning was wrong.")
    print()
    print("H0: slope = -1 exactly, on BOTH axes. Deviation => applied bias != recorded")
    print("    bias, wrong temperature, unequilibrated windows, or a boost that does")
    print("    not cancel from the ratio.")

    pairs = _pairings(wins)
    results = {}
    for axis, what in (("cv1", "primary / contact restraint -- pairs share (c2,k2) exactly"),
                       ("cv2", "secondary restraint -- pairs share (c1,k1) exactly")):
        fits = _fit_axis(pairs[axis], wins, kept, cv1, cv2, beta, int(args.min_overlap))
        if not fits:
            print(f"\n{axis.upper()} AXIS: no pairs (grid has no two windows differing "
                  "only along this axis)")
            continue
        results[axis] = _report_axis(f"{axis.upper()} AXIS", what, fits, float(args.temp))
        sp = _time_stratified(pairs[axis], wins, kept, step, cv1, cv2, beta,
                              int(args.min_overlap), int(args.strata))
        if results[axis] and sp.get("n", 0) >= 2:
            shift = abs(sp["re"] - results[axis]["re"])
            print(f"  systematic     splitting into {args.strata} time-ordered blocks moves the")
            print(f"                 pooled slope to {sp['re']:+.5f}, a shift of {shift:.4f} "
                  f"-- {shift / results[axis]['re_se']:.1f}x the RE standard error.")

    print(f"\n{'=' * 78}\nCROSS-AXIS CHECK\n{'=' * 78}")
    if len(results) == 2 and all(results.values()):
        s1, s2 = results["cv1"], results["cv2"]
        d = s1["re"] - s2["re"]
        dse = math.sqrt(s1["re_se"] ** 2 + s2["re_se"] ** 2)
        print(f"  cv1 axis {s1['re']:+.5f} -> {args.temp / abs(s1['re']):.1f} K")
        print(f"  cv2 axis {s2['re']:+.5f} -> {args.temp / abs(s2['re']):.1f} K")
        print(f"  difference {d:+.4f} +/- {dse:.4f}  (z = {d / dse:+.2f})")
        if abs(d / dse) > 2.0:
            print("  The two axes disagree. A wrong temperature is ONE scalar and must")
            print("  move both the same way, so a global beta error is disfavoured.")
        else:
            print("  The axes do not differ significantly, so a common scale factor")
            print("  (beta, or a systematic k error) remains a possible explanation.")
        print("  EITHER WAY, a single pooled 'effective temperature' is not a quantity")
        print("  this run has: the two axes give different ones, and only their")
        print("  agreement would license quoting one.")
        if ratios and float(np.median(ratios)) > 1.2:
            print(f"  SENSITIVITY: this z is not robust. g is unconverged (x{float(np.median(ratios)):.1f}")
            print("  per head doubling) and every SE above scales with it -- measured, this")
            print("  same comparison moves between |z| ~ 1.3 and ~2.2 as the detection head")
            print("  doubles. Treat the split as suggestive, not established.")
    else:
        print("  needs both axes; only one was fittable on this grid.")

    print(f"\n  CEILING: every interval above is a lower bound. With g ~ {np.nanmedian(list(gs.values())):.0f}")
    print("  frames per window this phase holds a few hundred independent samples per")
    print("  window, against a folding time orders of magnitude longer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
