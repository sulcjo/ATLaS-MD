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
    s.u_nk = d.u_nk[mask]
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
    dict with ``status`` in {"pass", "fail", "skipped"}. ``"skipped"``
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
