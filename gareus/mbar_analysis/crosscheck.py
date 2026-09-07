"""lambda=0-only vs full-ladder PMF: the built-in check that the boost reweighting is right.

The lambda=0 rungs are plain umbrella sampling. Their PMF, computed with the global f_k
but only their own samples (subset N_k, never renormalised as if self-consistent -- see
_subset_logw_from_global_fk), must agree with the full-ladder PMF within error. Disagreement
means the ladder boost term folded into u_nk (gareus.mbar_analysis.ladder.apply_ladder_boost_to_u)
is not correcting the biased Hamiltonian the way MBAR needs -- i.e. the reweighting itself is
wrong, not just noisy.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .pmf import make_bins, pmf_from_weights
from .solvers import _subset_logw_from_global_fk, norm_logw

DEFAULT_TOL_KCAL = 0.5

# A bin with only a handful of samples has Poisson relative error large enough
# (~100% at count=1-2) that -kT*ln(prob) differences between two DIFFERENTLY
# SIZED populations (the full population vs. the smaller lambda=0-only subset)
# routinely exceed a kcal/mol from counting noise alone, with no ladder-boost
# defect involved -- two bins that happen to hold the same absolute count under
# different total N already differ by kT*ln(N_full/N_lambda0). Gating the
# comparison (and the common-reference bin picked for the alignment below) on a
# minimum per-bin count in BOTH curves keeps the check sensitive to a real
# reweighting defect without also tripping on tail-bin shot noise.
MIN_BIN_COUNT = 10


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
    dict with ``status`` in {"pass", "fail", "skipped"} (``"skipped"`` only
    when no state carries lambda == 0.0, or one does but holds zero
    samples), ``max_abs_diff_kcal``, ``n_lambda0_samples``,
    ``tolerance_kcal``, ``n_bins_compared`` (how many bins actually entered
    the ``max_abs_diff_kcal`` comparison after the ``MIN_BIN_COUNT`` gate --
    a "pass" resting on very few bins is a much weaker statement than one
    resting on many), ``count_gate_fell_back`` (True when the gate left
    nothing and the comparison fell back to the plain finite mask), and the
    two PMF dicts (``pmf_full``, ``pmf_lambda0``) as returned by
    ``pmf_from_weights``.
    """
    lambdas = getattr(d, "state_lambdas", None)
    if lambdas is None:
        lambdas = np.zeros(d.u_nk.shape[1])
    lambdas = np.asarray(lambdas, dtype=float)
    lam0_states = np.where(lambdas == 0.0)[0]
    if lam0_states.size == 0:
        return {"status": "skipped", "reason": "no λ=0 states in state_lambdas",
                "n_lambda0_samples": 0}

    window = np.asarray(d.window, dtype=np.int64)
    mask = np.isin(window, lam0_states)
    n_lambda0 = int(mask.sum())
    if n_lambda0 == 0:
        return {"status": "skipped",
                "reason": "λ=0 state(s) declared but hold zero samples",
                "n_lambda0_samples": 0}

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
    both = finite & (counts_full >= MIN_BIN_COUNT) & (counts_lam0 >= MIN_BIN_COUNT)
    count_gate_fell_back = False
    if not np.any(both):
        # Too sparse for the count gate to leave anything -- fall back to the
        # plain finite mask rather than refuse to report at all. Recorded
        # (not silent) because a fallback this thin is itself a reason to
        # distrust a "pass": see n_bins_compared below.
        both = finite
        count_gate_fell_back = True
    n_bins_compared = int(np.count_nonzero(both))
    if not np.any(both):
        max_abs = float("nan")
        status = "fail"
    else:
        # Re-align on the JOINTLY valid bins only (not each curve's own
        # nanmin, which pmf_from_weights already applied over its own,
        # possibly wider, valid range) -- the additive constant per curve
        # is arbitrary and must be pinned to a common reference before the
        # two shapes can be compared bin-for-bin.
        diff = (F_full - F_full[both].min()) - (F_lam0 - F_lam0[both].min())
        max_abs = float(np.max(np.abs(diff[both])))
        status = "pass" if max_abs <= tol_kcal else "fail"

    return {
        "status": status,
        "max_abs_diff_kcal": max_abs,
        "n_lambda0_samples": n_lambda0,
        "tolerance_kcal": tol_kcal,
        # How many bins actually entered the max_abs_diff_kcal comparison,
        # and whether the MIN_BIN_COUNT gate had to fall back to the plain
        # finite mask because it left nothing -- a "pass" resting on very
        # few bins (a minority-population lambda=0 rung set against a wide
        # `bins`) is a much weaker statement than one resting on many, and
        # neither status alone shows that.
        "n_bins_compared": n_bins_compared,
        "count_gate_fell_back": count_gate_fell_back,
        "pmf_full": pmf_full,
        "pmf_lambda0": pmf_lambda0,
    }
