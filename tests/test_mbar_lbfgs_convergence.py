"""Regression test: solve_mbar_lbfgs must not report an iteration-limit hit
as convergence.

Bug: ``conv = result.success or (result.status in (0, 1))`` treated scipy's
L-BFGS-B status 1 ("TOTAL NO. OF ITERATIONS REACHED LIMIT") as converged.
Status 1 always has ``success=False`` and is explicitly NOT convergence --
only status 0 ("CONVERGENCE: ...") is.  Reporting a maxiter cutoff as
converged silently defeats ``analyze()``'s only correctness safeguard
(``if not m['converged']: warn.append(...)``) and the ``backend='auto'``
fallback-on-failure logic in ``solve_mbar``, since no failure is ever
signaled.

This uses the same harmonic-umbrella oracle construction as
test_physics_oracle.py (real, non-trivial MBAR problem, not a toy with a
degenerate optimum) so a tiny maxiter genuinely forces scipy to hit its
iteration limit rather than trivially "converging" in one step.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

agm = pytest.importorskip(
    "analyze_gareus_mbar",
    reason="estimator module (numpy-only) must be importable from repo root",
)

if not getattr(agm, "SCIPY_AVAILABLE", False):
    pytest.skip("solve_mbar_lbfgs requires scipy", allow_module_level=True)

KB_KCAL = 0.0019872041  # Boltzmann constant, kcal/(mol*K)
TEMP_K = 300.0
KT = KB_KCAL * TEMP_K
BETA = 1.0 / KT

K0_TRUE = 2.0
K_BIAS = 10.0
N_WINDOWS = 19
N_PER_WINDOW = 2000
SEED = 20260812


def _generate_umbrella_oracle(k0, k_bias, centers, n_per_window, beta, seed):
    """Draw i.i.d. samples for a harmonic-well umbrella set; returns
    (window, u_nk) ready to hand to solve_mbar_lbfgs -- see
    test_physics_oracle.py for the full derivation."""
    rng = np.random.RandomState(seed)
    centers = np.asarray(centers, dtype=np.float64)
    K = centers.size

    sigma = np.sqrt(1.0 / (beta * (k0 + k_bias)))
    means = k_bias * centers / (k0 + k_bias)

    win_parts = []
    cv_parts = []
    for i in range(K):
        cv_parts.append(rng.normal(means[i], sigma, size=n_per_window))
        win_parts.append(np.full(n_per_window, i, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(win_parts)

    diff = cv[:, None] - centers[None, :]
    u_nk = beta * 0.5 * k_bias * diff * diff
    return window, u_nk


def _build_problem():
    centers = np.linspace(-2.0, 2.0, N_WINDOWS)
    return _generate_umbrella_oracle(K0_TRUE, K_BIAS, centers, N_PER_WINDOW, BETA, SEED)


def test_solve_mbar_lbfgs_tiny_maxiter_is_not_converged():
    """Forcing scipy to hit the L-BFGS-B iteration cap must report
    converged=False (the regression case for the fixed bug)."""
    window, u_nk = _build_problem()

    res = agm.solve_mbar_lbfgs(u_nk, window, tol=1e-10, maxiter=2)

    # Downstream code only ever checks truthiness (`if not m['converged']`,
    # `bool(m['converged'])`), and `conv` is passed through from scipy's own
    # `result.success`/`result.status` (which may be numpy bool/int types on
    # some scipy builds) -- so assert truthiness, not `is False` identity.
    assert not res["converged"], (
        "solve_mbar_lbfgs reported converged=True after hitting the "
        "iteration limit (maxiter=2) -- status==1 must not count as "
        "convergence"
    )
    # Sanity check this genuinely hit the limit and is nowhere near gtol.
    assert res["max_delta"] > 1e-6


def test_solve_mbar_lbfgs_reasonable_maxiter_converges():
    """The same problem with a realistic maxiter budget must converge, so
    the fix doesn't just make converged always False."""
    window, u_nk = _build_problem()

    res = agm.solve_mbar_lbfgs(u_nk, window, tol=1e-10, maxiter=10000)

    assert res["converged"]
    # Sanity check only (not the regression assertion): a converged run's
    # gradient norm should be small, well below the "hit maxiter" case above.
    # (Not required to be <= tol itself: with ftol=0.0, scipy's L-BFGS-B can
    # also terminate via its "relative reduction of f" stopping criterion,
    # which is status=0/converged but not gradient-tolerance-tight.)
    assert res["max_delta"] < 1e-3
