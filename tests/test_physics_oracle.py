"""Physics-oracle tests for the MBAR umbrella-sampling estimator.

These tests feed the *real* GAREUS estimator (``analyze_gareus_mbar.solve_mbar``
and ``pmf_from_weights``) synthetic samples whose underlying free-energy
surface is known *analytically*, then assert the estimator recovers it.

Why this exists
---------------
The dominant GAREUS bug class is statistical/physical correctness in the
reweighting math (sign errors, dropped terms, factor mistakes, the
GaMD boost-exclusion that gave ESS~0).  Those bugs do not crash; they
produce plausible-but-wrong PMFs that only get caught in expensive
multi-microsecond GPU runs.  An analytic oracle catches the same bug class
at commit time, in ~1 second, with no OpenMM and no MD.

The harmonic oracle
-------------------
True (unbiased) 1-D potential:        U0(x) = 0.5 * k0 * x**2
Umbrella bias of window i:            w_i(x) = 0.5 * k * (x - x_i)**2

Because both are quadratic, the biased equilibrium distribution of window i,

    p_i(x) ∝ exp(-beta * [U0(x) + w_i(x)]),

is exactly Gaussian:

    variance  sigma**2 = 1 / (beta * (k0 + k))
    mean      mu_i      = k * x_i / (k0 + k)

so we can draw i.i.d. samples directly with NumPy -- no integrator, no
OpenMM.  We then build the reduced-potential matrix in GAREUS' convention
(BIAS ONLY; the unbiased state is implicit at u = 0):

    u_nk[n, k] = beta * 0.5 * k * (x_n - x_k)**2          # dimensionless

The common beta*U0(x_n) term is identical across all columns k for a given
sample n, so it cancels in MBAR's per-sample weights.  ``solve_mbar`` thus
reweights every sample to the zero-bias (unbiased) ensemble, and histogram
reweighting recovers F(x) = U0(x) = 0.5*k0*x**2 up to an additive constant.

Notes / traps (confirmed against an independent derivation):
  * Fitting F = a*x**2 + b*x + c gives a = 0.5*k0, so the *curvature* is
    F'' = 2*a.  We compare 2*a to k0 (a common factor-of-2 trap).
  * PMF == potential here ONLY because x is a flat-measure 1-D Cartesian
    coordinate (no Jacobian/entropy term).  Do NOT reuse this identity for
    an angle/radius/transformed CV without adding the Jacobian.
  * Units are kept self-consistently in kcal/mol throughout, so u_nk needs
    no kJ<->kcal (4.184) conversion.  The production code's 4.184 factor
    lives in ``reconstruct_bias_matrix`` (stored k is kcal but stored beta is
    kJ^-1); validating *that* conversion is a separate construction-guard
    test, deliberately decoupled from this estimator-math test.

Per CLAUDE.md, this test imports only NumPy and the pure-NumPy estimator
module -- no OpenMM, PeptideBuilder, or gamd-openmm.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# analyze_gareus_mbar.py lives at the repo root, not inside the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

agm = pytest.importorskip(
    "analyze_gareus_mbar",
    reason="estimator module (numpy-only) must be importable from repo root",
)

# --- Physical constants / oracle parameters --------------------------------
KB_KCAL = 0.0019872041  # Boltzmann constant, kcal/(mol*K)
TEMP_K = 300.0
KT = KB_KCAL * TEMP_K            # ~0.59616 kcal/mol
BETA = 1.0 / KT                  # (kcal/mol)^-1

K0_TRUE = 2.0                    # true well curvature, kcal/mol/unit^2
K_BIAS = 10.0                    # umbrella spring constant, kcal/mol/unit^2
N_WINDOWS = 19
N_PER_WINDOW = 4000
SEED = 12345


def _generate_umbrella_oracle(k0, k_bias, centers, n_per_window, beta, seed):
    """Draw exact i.i.d. samples for a harmonic-well umbrella set.

    Returns (cv, window, u_nk) ready to hand to ``solve_mbar``:
      cv      : (N,)   sampled CV values
      window  : (N,)   originating window index per sample
      u_nk    : (N, K) BIAS-ONLY reduced potentials (dimensionless)
    """
    rng = np.random.RandomState(seed)
    centers = np.asarray(centers, dtype=np.float64)
    K = centers.size

    sigma = np.sqrt(1.0 / (beta * (k0 + k_bias)))   # per-window stddev
    means = k_bias * centers / (k0 + k_bias)          # compressed means mu_i

    cv_parts, win_parts = [], []
    for i in range(K):
        cv_parts.append(rng.normal(means[i], sigma, size=n_per_window))
        win_parts.append(np.full(n_per_window, i, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(win_parts)

    # u_nk[n, k] = beta * 0.5 * k_bias * (x_n - center_k)^2   (bias only)
    diff = cv[:, None] - centers[None, :]
    u_nk = beta * 0.5 * k_bias * diff * diff
    return cv, window, u_nk


def _fit_curvature(cv_centers, pmf, counts, fit_halfwidth, min_counts):
    """Weighted parabola fit; returns (curvature=2a, linear=b, residual_rms)."""
    mask = (np.abs(cv_centers) <= fit_halfwidth) & (counts > min_counts) & np.isfinite(pmf)
    assert mask.sum() >= 5, f"too few usable bins for fit ({mask.sum()})"
    x = cv_centers[mask]
    f = pmf[mask]
    w = np.sqrt(counts[mask].astype(np.float64))  # PMF bin error ~ 1/sqrt(count)
    a, b, c = np.polyfit(x, f, 2, w=w)
    resid = f - (a * x * x + b * x + c)
    return 2.0 * a, b, float(np.sqrt(np.mean(resid * resid)))


def test_harmonic_pmf_recovery():
    """MBAR must recover the curvature of a known harmonic well within 10%."""
    centers = np.linspace(-2.0, 2.0, N_WINDOWS)
    cv, window, u_nk = _generate_umbrella_oracle(
        K0_TRUE, K_BIAS, centers, N_PER_WINDOW, BETA, SEED
    )

    res = agm.solve_mbar(u_nk, window, backend="anderson", tol=1e-10)
    assert res["converged"], f"MBAR did not converge: {res['max_delta']:.2e}"

    w = agm.norm_logw(res["logw"])               # unbiased-ensemble weights
    bins = np.linspace(-2.3, 2.3, 61)
    pmf = agm.pmf_from_weights(cv, w, bins, KT)   # F = -kT ln(prob), min shifted to 0

    curvature, linear, resid_rms = _fit_curvature(
        pmf["cv_A"], pmf["pmf"], pmf["counts"], fit_halfwidth=1.4, min_counts=50
    )

    rel_err = abs(curvature - K0_TRUE) / K0_TRUE
    assert rel_err < 0.10, (
        f"recovered curvature {curvature:.3f} vs true {K0_TRUE} "
        f"(rel err {rel_err:.1%}); bias-only u_nk or reweighting likely wrong"
    )
    # Symmetric well -> no linear term; large |b| signals a sign/centering bug.
    assert abs(linear) < 0.15, f"unexpected linear PMF term b={linear:.3f}"
    # Recovered shape must track the true parabola, not merely have right curvature.
    assert resid_rms < 0.10, f"PMF shape residual too large: {resid_rms:.3f} kcal/mol"


def test_flat_pmf_recovery():
    """With no true well (k0=0), the recovered PMF must be flat.

    This is the cheap normalization/double-counting check: every window
    samples a pure Gaussian around its (uncompressed) center, and correct
    reweighting must flatten them into a uniform unbiased density.
    """
    centers = np.linspace(-2.0, 2.0, N_WINDOWS)
    cv, window, u_nk = _generate_umbrella_oracle(
        0.0, K_BIAS, centers, N_PER_WINDOW, BETA, SEED
    )

    res = agm.solve_mbar(u_nk, window, backend="anderson", tol=1e-10)
    assert res["converged"], f"MBAR did not converge: {res['max_delta']:.2e}"

    w = agm.norm_logw(res["logw"])
    bins = np.linspace(-2.3, 2.3, 61)
    pmf = agm.pmf_from_weights(cv, w, bins, KT)

    # In the well-sampled core, a flat PMF should vary by < ~kT.
    centers_a = pmf["cv_A"]
    core = (np.abs(centers_a) <= 1.4) & (pmf["counts"] > 50) & np.isfinite(pmf["pmf"])
    span = float(np.ptp(pmf["pmf"][core]))
    assert span < KT, f"flat PMF should be flat; recovered span {span:.3f} kcal/mol"
