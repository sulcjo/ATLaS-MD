"""The lambda-ladder centre model solves MBAR on gareus-analyze's solver (gareus.adaptive.mbar_solve).

The reference is the fixed point ``ladder_adapt._mbar_f`` carried before the swap, verbatim, run
to a tight tolerance: the rung free energies must agree to ~1e-10 in reduced units.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.adaptive import cv2_coverage as cov  # noqa: E402
from gareus.adaptive import ladder_adapt as la  # noqa: E402
from gareus.adaptive import mbar_solve as ms  # noqa: E402


def _logsumexp(a, axis):
    top = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(top, axis=axis) + np.log(np.sum(np.exp(a - top), axis=axis))


def _old_mbar_f(u_kn, n_k, tol=1e-13, max_iter=200000):
    f = np.zeros(u_kn.shape[0])
    log_n = np.log(n_k)
    for _ in range(max_iter):
        denom = _logsumexp(log_n[:, None] + f[:, None] - u_kn, axis=0)
        f_new = -_logsumexp(-u_kn - denom[None, :], axis=1)
        f_new = f_new - f_new[0]
        if np.max(np.abs(f_new - f)) < tol:
            return f_new
        f = f_new
    return f


def _rungs(seed=0):
    rng = np.random.default_rng(seed)
    n_k = np.array([800, 900, 700, 1000])
    x = rng.normal(size=int(n_k.sum())) * 1.5
    lam = np.array([0.0, 0.3, 0.6, 1.0])
    return np.array([0.5 * (1.0 - l) * x ** 2 + l * np.abs(x) for l in lam]), n_k


def test_centre_model_mbar_matches_the_pre_swap_fixed_point():
    u, n_k = _rungs()
    f = la._mbar_f(u, n_k)
    assert f[0] == 0.0
    np.testing.assert_allclose(f, _old_mbar_f(u, n_k.astype(float)), atol=1e-9)


def test_one_shared_wrapper_for_every_adaptive_mbar_solve():
    assert cov.solve_mbar is ms.solve_mbar and cov.solve_rows is ms.solve_rows
    assert ms.MBAR_BACKEND == "numba-anderson"
    src = (Path(__file__).resolve().parents[1] / "gareus" / "adaptive" / "ladder_adapt.py").read_text()
    assert "for _ in range(max_iter)" not in src          # the hand-rolled fixed point is gone
