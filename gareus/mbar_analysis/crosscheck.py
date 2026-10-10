"""lambda=0-only vs full-ladder PMF: the built-in check that the boost reweighting is right.

The lambda=0 rungs are plain umbrella sampling. Their PMF, computed with the global f_k
but only their own samples (subset N_k, never renormalised as if self-consistent -- see
_subset_logw_from_global_fk), must agree with the full-ladder PMF within error. Disagreement
means the ladder boost term folded into u_nk (gareus.mbar_analysis.ladder.apply_ladder_boost_to_u)
is not correcting the biased Hamiltonian the way MBAR needs -- i.e. the reweighting itself is
wrong, not just noisy.

Spec requirement is "agrees within (bootstrap) error". This module does NOT wire in the
analyzer's `--pmf-uncertainty` block-bootstrap sigma: at the one call site
(`analyze_gareus_mbar.py:_analyze_population`, right after `solve_mbar` and before
`run_pmf_and_gamd_boost_report`), no bootstrap sigma has been computed yet for anything --
the existing bootstrap machinery (`_bootstrap_pmf_uncertainty_1d`) runs LATER, inside
`run_pmf_and_gamd_boost_report`, and only for `d_main` (the post-epoch_000-split
population)'s SELECTED/main PMF -- not for the full `d` or a lambda=0-only subset this
check compares. Computing a bootstrap sigma for either of THOSE two populations would mean
running a brand new, unconditional (or newly-flagged), ~100-replicate block-bootstrap
resolve for each, which is a materially different and heavier scope than "read an
already-available value". So this always reports ``tolerance_source: "fixed_default"`` and
uses the fixed ``tol_kcal`` (default ``DEFAULT_TOL_KCAL``) -- a real gap, not a silent one.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .pmf import make_bins, pmf_from_weights
from .solvers import _subset_logw_from_global_fk, norm_logw

DEFAULT_TOL_KCAL = 0.5

# Poisson relative error in a bin's occupied sample count n is ~1/sqrt(n), so the
# corresponding fluctuation in F = -kT*ln(p) is ~kT/sqrt(n) (dF = -kT*dp/p). Requiring that
# single-bin noise stay at most 1/BIN_NOISE_SAFETY_FACTOR of the tolerance being tested
# against -- i.e. kT/sqrt(n) <= tol_kcal/BIN_NOISE_SAFETY_FACTOR -- gives a per-call minimum
# bin count that scales with BOTH the system's actual thermal energy (kbt_kcal) and the
# tolerance actually being enforced, rather than one constant tuned to look right on a
# particular seed. safety=3 leaves 2/3 of the tolerance budget for a genuine reweighting
# defect to still trip a bin's fluctuation past a "real" magnitude; MIN_BIN_COUNT_FLOOR=5
# is a hard lower bound regardless of tol_kcal (a bin with <5 samples is unreliable at any
# tolerance -- e.g. tol_kcal=0 would otherwise demand an infinite count).
BIN_NOISE_SAFETY_FACTOR = 3.0
MIN_BIN_COUNT_FLOOR = 5

# A comparison built from too few bins can be VACUOUS rather than merely noisy: with exactly
# one bin surviving the count gate, that bin IS the alignment reference (F_full[both].min()
# == F_full[that bin]), so diff at that bin is identically 0 and status would read "pass" at
# ANY tolerance, including tol_kcal=0 -- with nothing in the returned max_abs_diff_kcal to
# betray it (n_bins_compared==1 is the only tell). Two bins gives exactly one non-reference
# comparison point -- a single coincidental agreement (or disagreement) at that one bin
# would fully determine the verdict. Three is the minimum with at least two independent
# non-reference comparison points, so a single-bin fluke can no longer decide the outcome on
# its own. Below this floor the result is "skipped", not "pass" or "fail" -- there was no
# meaningful comparison to make a verdict from.
MIN_BINS_FOR_VERDICT = 3


def _min_bin_count(kbt_kcal: float, tol_kcal: float) -> int:
    tol = max(float(tol_kcal), 1e-9)
    return max(MIN_BIN_COUNT_FLOOR, int(np.ceil((BIN_NOISE_SAFETY_FACTOR * float(kbt_kcal) / tol) ** 2)))


def _subset(d: Any, mask: np.ndarray) -> Any:
    """Minimal Data-alike carrying only what
    ``_subset_logw_from_global_fk``/``pmf_from_weights`` read: ``cv``,
    ``window``, ``u_nk``, ``beta``, ``state_lambdas``, ``meta``."""
    s = d.__class__.__new__(d.__class__)
    s.cv = d.cv[mask]
    s.window = np.asarray(d.window)[mask]
    from .storage import select_matrix
    s.u_nk = select_matrix(d.u_nk, mask)
    s.beta = d.beta
    s.state_lambdas = d.state_lambdas
    s.meta = dict(d.meta)
    return s


def ladder_crosscheck(d: Any, f_k_global: np.ndarray, bins, kbt_kcal: float,
                       tol_kcal: float = DEFAULT_TOL_KCAL) -> dict:
    """Compare the full-population PMF against the lambda=0-only PMF, both
    built from the SAME global ``f_k_global`` (never re-solved), each with
    its own population's N_k in the MBAR denominator.

    ``bins`` may be an explicit edges array (what the analyzer already
    computes via ``make_bins`` and passes to every other PMF in the same
    report -- pass that through here unchanged) or a plain bin COUNT. A bin
    count is resolved into one fixed edges array from the FULL population's
    ``cv`` up front and reused for both curves: ``pmf_from_weights`` picks
    its own histogram range from whatever array it is given when ``bins``
    is a count, so calling it twice with an int would silently give the
    full-population and the lambda=0-subset curves DIFFERENT bin edges --
    it would compare same-index bins that do not correspond to the same CV
    range. A shared edges array (the analyzer's normal case) sidesteps that
    by construction; the count case is only a defensive equivalent, so a
    caller that hands this function a bare int does not still get the
    silent version of the bug the shared f_k plumbing exists to avoid.

    Returns
    -------
    dict with ``status`` in {"pass", "fail", "skipped"}. ``"fail"`` covers
    two distinct things: the ordinary out-of-tolerance disagreement, and the
    CONTRADICTION case -- ``meta['gamd_ladder']`` asserted while
    ``state_lambdas`` holds no λ > 0 (absent, or all-zero), which means a
    loader dropped the ladder and the comparison would be vacuous. The
    contradiction return carries a ``reason`` and NO ``max_abs_diff_kcal``/
    ``pmf_full``/``pmf_lambda0`` (no comparison was made), so any consumer
    formatting those must key on their presence, not on ``status``.
    ``"skipped"``
    covers three distinct reasons (see ``reason``): no state carries
    lambda == 0.0; one does but holds zero samples; or fewer than
    ``MIN_BINS_FOR_VERDICT`` bins survived the comparison gate (too sparse
    for a meaningful verdict either way -- NOT graded "pass"). Also
    carries ``max_abs_diff_kcal``, ``n_lambda0_samples``, ``tolerance_kcal``,
    ``tolerance_source`` (always ``"fixed_default"`` here -- see module
    docstring), ``n_bins_compared`` (how many bins actually entered the
    ``max_abs_diff_kcal`` comparison after the per-bin-count gate -- a
    "pass" resting on few bins is a much weaker statement than one resting
    on many), ``count_gate_fell_back`` (True when the count gate left fewer
    than ``MIN_BINS_FOR_VERDICT`` bins and the comparison fell back to the
    plain finite mask), and the two PMF dicts (``pmf_full``,
    ``pmf_lambda0``) as returned by ``pmf_from_weights``.
    """
    lambdas = getattr(d, "state_lambdas", None)
    if lambdas is None:
        lambdas = np.zeros(d.u_nk.shape[1])
    lambdas = np.asarray(lambdas, dtype=float)

    # CONTRADICTION GUARD (2026-09-07 final review, C1). `meta['gamd_ladder']`
    # is only ever set True by apply_ladder_boost_to_u, and only when some
    # state carries gamd_lambda > 0 -- so "ladder asserted" and "no λ > 0 in
    # state_lambdas" cannot both be true of the same correctly-loaded run.
    # When they are, a loader dropped state_lambdas on the way to Data (the
    # C2/C3 defects). Without this guard every column then reads λ = 0, the
    # "λ=0-only subset" IS the full population, the two PMFs are bit-identical
    # and the gate returns status 'pass' with max_abs_diff_kcal == 0.0 -- a
    # vacuous PASS that certifies exactly the runs whose ladder was lost.
    # Graded 'fail', not a new status string and not 'skipped': gareus_report's
    # _check_ladder_crosscheck dispatches on the three literals and grades
    # anything else -- 'skipped' included -- NA, which does not move `overall`.
    if bool((getattr(d, "meta", None) or {}).get("gamd_ladder")) and not np.any(lambdas > 0.0):
        # No comparison is made, so nothing that describes one is reported:
        # no max_abs_diff_kcal, no PMFs, and n_lambda0_samples 0 -- the λ=0
        # population is UNKNOWN here (which states are λ=0 is precisely what
        # was lost), not "all of them".
        return {
            "status": "fail",
            "reason": ("λ-ladder asserted (meta['gamd_ladder']) but the per-state λ is "
                       f"unavailable: state_lambdas is "
                       f"{'absent' if getattr(d, 'state_lambdas', None) is None else 'all zero'}. "
                       "The loader lost the per-state ladder rungs, so the λ=0 population cannot "
                       "be identified and the cross-check is impossible -- not passed, not skipped"),
            "n_lambda0_samples": 0,
            "tolerance_kcal": tol_kcal,
            "tolerance_source": "fixed_default",
            "n_bins_compared": 0,
        }

    lam0_states = np.where(lambdas == 0.0)[0]
    if lam0_states.size == 0:
        return {"status": "skipped", "reason": "no λ=0 states in state_lambdas",
                "n_lambda0_samples": 0, "tolerance_kcal": tol_kcal,
                "tolerance_source": "fixed_default"}

    window = np.asarray(d.window, dtype=np.int64)
    mask = np.isin(window, lam0_states)
    n_lambda0 = int(mask.sum())
    if n_lambda0 == 0:
        return {"status": "skipped",
                "reason": "λ=0 state(s) declared but hold zero samples",
                "n_lambda0_samples": 0, "tolerance_kcal": tol_kcal,
                "tolerance_source": "fixed_default"}

    f_k_global = np.asarray(f_k_global, dtype=np.float64)
    edges = bins if np.ndim(bins) > 0 else make_bins(d.cv, int(bins), None, None)

    full_logw = _subset_logw_from_global_fk(d, f_k_global)
    pmf_full = pmf_from_weights(d.cv, norm_logw(full_logw), edges, kbt_kcal)

    sub = _subset(d, mask)
    sub_logw = _subset_logw_from_global_fk(sub, f_k_global)
    pmf_lambda0 = pmf_from_weights(sub.cv, norm_logw(sub_logw), edges, kbt_kcal)

    F_full = np.asarray(pmf_full["pmf"], dtype=np.float64)
    F_lam0 = np.asarray(pmf_lambda0["pmf"], dtype=np.float64)
    counts_full = np.asarray(pmf_full["counts"])
    counts_lam0 = np.asarray(pmf_lambda0["counts"])
    finite = np.isfinite(F_full) & np.isfinite(F_lam0)

    min_bin_count = _min_bin_count(kbt_kcal, tol_kcal)
    count_gated = finite & (counts_full >= min_bin_count) & (counts_lam0 >= min_bin_count)
    if int(np.count_nonzero(count_gated)) >= MIN_BINS_FOR_VERDICT:
        both = count_gated
        count_gate_fell_back = False
    else:
        # The tolerance-derived count gate couldn't be satisfied (e.g. a very
        # tight tol_kcal drives min_bin_count very high) -- fall back to the
        # hard MIN_BIN_COUNT_FLOOR gate, NEVER to the plain finite mask. A
        # plain-finite fallback would readmit 1-2-count tail bins precisely
        # when the tolerance is tightest, i.e. the gate would filter LESS as
        # tol_kcal shrinks -- backwards. If even the floor gate leaves too
        # few bins, n_bins_compared below reports "skipped" rather than
        # comparing on noise. Recorded (count_gate_fell_back), not silent: a
        # fallback this permissive is itself a reason to weigh a "pass" more
        # lightly.
        both = finite & (counts_full >= MIN_BIN_COUNT_FLOOR) & (counts_lam0 >= MIN_BIN_COUNT_FLOOR)
        count_gate_fell_back = True
    n_bins_compared = int(np.count_nonzero(both))

    if n_bins_compared < MIN_BINS_FOR_VERDICT:
        return {
            "status": "skipped",
            "reason": (f"only {n_bins_compared} bin(s) had a comparable full/λ=0 pair "
                       f"(need >= {MIN_BINS_FOR_VERDICT}) -- too sparse for a verdict"),
            "n_lambda0_samples": n_lambda0,
            "tolerance_kcal": tol_kcal,
            "tolerance_source": "fixed_default",
            "n_bins_compared": n_bins_compared,
            "count_gate_fell_back": count_gate_fell_back,
            "pmf_full": pmf_full,
            "pmf_lambda0": pmf_lambda0,
        }

    # Re-align on the JOINTLY valid (gated) bins only -- not each curve's own
    # nanmin, which pmf_from_weights already applied over its own, possibly
    # wider, valid range -- the additive constant per curve is arbitrary and
    # must be pinned to a common reference before the two shapes can be
    # compared bin-for-bin. With MIN_BINS_FOR_VERDICT enforced above, `both`
    # always has at least 2 bins besides whichever bin sets the reference, so
    # this can no longer be a tautological zero.
    diff = (F_full - F_full[both].min()) - (F_lam0 - F_lam0[both].min())
    max_abs = float(np.max(np.abs(diff[both])))
    status = "pass" if max_abs <= tol_kcal else "fail"

    return {
        "status": status,
        "max_abs_diff_kcal": max_abs,
        "n_lambda0_samples": n_lambda0,
        "tolerance_kcal": tol_kcal,
        "tolerance_source": "fixed_default",
        "n_bins_compared": n_bins_compared,
        "count_gate_fell_back": count_gate_fell_back,
        "pmf_full": pmf_full,
        "pmf_lambda0": pmf_lambda0,
    }


# ---------------------------------------------------------------------------------------------
# Auxiliary states: ordinary-only vs all-states PMF crosscheck (CVaux adaptive, Task 15)
# ---------------------------------------------------------------------------------------------
DEFAULT_LOGW_BLOCK_ROWS = 65536
AUX_CROSSCHECK_METHOD = "raw_count_heuristic"
AUX_CROSSCHECK_STATUSES = ("heuristic_pass", "heuristic_fail", "skipped", "unavailable", "error")


def _target_logw(u: np.ndarray, window: np.ndarray, f: np.ndarray, *, block_rows: int = DEFAULT_LOGW_BLOCK_ROWS) -> np.ndarray:
    """Per-row log weight at the unbiased target (u = 0): -logsumexp_k(log N_k + f_k - u_nk). Computed per row
    block so no full N x K temporary (``a`` / ``where``) is ever built; ``u`` may be a memmap."""
    from scipy.special import logsumexp
    n, k = u.shape
    n_k = np.bincount(window, minlength=k).astype(float)
    with np.errstate(divide="ignore"):
        base = np.log(n_k) + np.asarray(f, float)
    block = max(1, int(block_rows))
    out = np.empty(n, dtype=np.float64)
    for lo in range(0, n, block):
        a = base[None, :] - np.asarray(u[lo:lo + block], dtype=np.float64)
        a[~np.isfinite(a)] = -np.inf
        out[lo:lo + block] = -logsumexp(a, axis=1)
    return out


def _axis_values(d: Any) -> dict:
    axes = {}
    cv = getattr(d, "cv", None)
    if cv is not None:
        axes["cv1"] = np.asarray(cv, float)
    cv2 = getattr(d, "cv2", None)
    if cv2 is not None and np.any(np.isfinite(np.asarray(cv2, float))):
        axes["cv2"] = np.asarray(cv2, float)
    z = getattr(d, "aux_z", None)
    if z is None:
        z = (getattr(d, "meta", None) or {}).get("aux_z")
    if z is not None:
        axes["z3"] = np.asarray(z, float)
    return axes


def aux_ordinary_crosscheck(d: Any, f_k_global: np.ndarray, bins_by_axis: dict, kbt_kcal: float, *,
                            ordinary_states, tol_kcal: float = DEFAULT_TOL_KCAL,
                            block_rows: int = DEFAULT_LOGW_BLOCK_ROWS) -> dict:
    """HEURISTIC (``method: raw_count_heuristic``, no statistical test): PMF along each axis from all states (global ``f_k_global``) vs from ordinary states only
    (``f`` re-solved on that subset with ``solve_rows``), both at the unbiased target. A
    disagreement means the auxiliary states' bias terms are wrong (or aux sampling is
    inconsistent), not noise. ``fail`` when any supported bin differs by more than
    ``max(tol, BIN_NOISE_SAFETY_FACTOR x bin noise)``; axes with fewer than MIN_BINS_FOR_VERDICT
    supported bins are skipped; all skipped -> overall skipped. Statuses: heuristic_pass / heuristic_fail /
    skipped; a pass is only "no bin differs by more than a raw-count noise bound", never evidence of agreement."""
    from gareus.adaptive.mbar_solve import solve_rows
    from .storage import select_matrix

    ordinary = np.asarray(sorted(int(x) for x in ordinary_states), dtype=np.int64)
    window = np.asarray(d.window, dtype=np.int64)
    rows = np.isin(window, ordinary)
    if ordinary.size == 0 or not rows.any():
        return {"status": "skipped", "method": AUX_CROSSCHECK_METHOD, "reason": "no ordinary-state samples",
                "axes": {}, "tolerance_kcal": tol_kcal}
    u_all = select_matrix(d.u_nk)      # never copied whole; _target_logw reads it per row block
    logw_all = _target_logw(u_all, window, np.asarray(f_k_global, float), block_rows=block_rows)
    u_ord = np.empty((int(rows.sum()), ordinary.size), dtype=np.float64)   # the one materialised subset
    ridx = np.flatnonzero(rows)
    for lo in range(0, ridx.size, max(1, int(block_rows))):
        sel = ridx[lo:lo + max(1, int(block_rows))]
        u_ord[lo:lo + sel.size] = np.asarray(u_all[sel], dtype=np.float64)[:, ordinary]
    del ridx
    remap = {int(s): i for i, s in enumerate(ordinary)}
    win_ord = np.fromiter((remap[int(w)] for w in window[rows]), dtype=np.int64, count=int(rows.sum()))
    f_ord, logw_ord = solve_rows(u_ord, win_ord)
    w_all = norm_logw(logw_all)
    w_ord = norm_logw(logw_ord)
    min_count = _min_bin_count(kbt_kcal, tol_kcal)
    values = _axis_values(d)
    out: dict = {}
    worst = 0.0
    any_verdict = False
    any_fail = False
    for name, edges in bins_by_axis.items():
        x = values.get(name)
        if x is None:
            out[name] = {"status": "skipped", "reason": "axis unavailable"}
            continue
        edges = np.asarray(edges, float)
        fin = np.isfinite(x)
        pa = pmf_from_weights(x[fin], w_all[fin], edges, kbt_kcal)
        xo = x[rows]
        fo = np.isfinite(xo)
        po = pmf_from_weights(xo[fo], w_ord[fo], edges, kbt_kcal)
        Fa, Fo = np.asarray(pa["pmf"]), np.asarray(po["pmf"])
        ca, co = np.asarray(pa["counts"]), np.asarray(po["counts"])
        ok = np.isfinite(Fa) & np.isfinite(Fo)
        gate = ok & (ca >= min_count) & (co >= min_count)
        if int(gate.sum()) < MIN_BINS_FOR_VERDICT:
            gate = ok & (ca >= MIN_BIN_COUNT_FLOOR) & (co >= MIN_BIN_COUNT_FLOOR)
        if int(gate.sum()) < MIN_BINS_FOR_VERDICT:
            out[name] = {"status": "skipped", "n_bins_compared": int(gate.sum()),
                         "reason": f"fewer than {MIN_BINS_FOR_VERDICT} supported bins"}
            continue
        diff = (Fa - Fa[gate].min()) - (Fo - Fo[gate].min())
        noise = kbt_kcal * np.sqrt(1.0 / np.maximum(ca, 1) + 1.0 / np.maximum(co, 1))
        limit = np.maximum(tol_kcal, BIN_NOISE_SAFETY_FACTOR * noise)
        bad = gate & (np.abs(diff) > limit)
        mx = float(np.max(np.abs(diff[gate])))
        any_verdict = True
        any_fail = any_fail or bool(bad.any())
        worst = max(worst, mx)
        out[name] = {"status": "heuristic_fail" if bad.any() else "heuristic_pass", "max_abs_diff_kcal": mx,
                     "n_bins_compared": int(gate.sum()), "n_bins_failing": int(bad.sum()),
                     "edges": edges.tolist(), "pmf_all": Fa.tolist(), "pmf_ordinary": Fo.tolist(),
                     "counts_all": ca.tolist(), "counts_ordinary": co.tolist()}
    status = "skipped" if not any_verdict else ("heuristic_fail" if any_fail else "heuristic_pass")
    res = {"status": status, "method": AUX_CROSSCHECK_METHOD, "axes": out, "tolerance_kcal": tol_kcal, "tolerance_source": "fixed_default",
           "n_ordinary_samples": int(rows.sum())}
    if any_verdict:
        res["max_abs_diff_kcal"] = worst
    else:
        res["reason"] = "no axis had enough supported bins"
    return res
