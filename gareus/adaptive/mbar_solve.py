"""MBAR self-consistency for the adaptive modules, on gareus-analyze's solver.

One place for every adaptive-side MBAR solve (R2 coverage, the lambda-ladder centre model):
``gareus.mbar_analysis.solvers.solve_mbar`` with the ``numba-anderson`` backend (fixed-point
MBAR with Anderson mixing), falling back to its NumPy ``anderson`` backend without numba.
Deterministic: every per-sample and per-state reduction in the numba kernel is one thread's
serial loop, so f does not depend on the thread count; the gareus-analyze default ``sambar``
(stochastic warm start) is not used. Gauge f_0 = 0.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from gareus.mbar_analysis import solvers as _mbar

MBAR_BACKEND = "numba-anderson"  # gareus-analyze backend (deterministic, see the module docstring)
MBAR_FALLBACK_BACKEND = "anderson"
MBAR_TOL = 1e-12                 # max |f_new - f| of the un-mixed step (c9: 7e-12 from the 1e-13 answer, 45 iterations)
MBAR_MAX_ITER = 20000


def solve_rows(u_nk: np.ndarray, window: np.ndarray, *, tol: float = MBAR_TOL, max_iter: int = MBAR_MAX_ITER,
               f_init: Optional[np.ndarray] = None, info: Optional[Dict[str, Any]] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """MBAR on (N, K) reduced energies ``u_nk`` with ``window`` = each row's state index, by
    gareus-analyze's ``solve_mbar`` (MBAR_BACKEND, NumPy fallback). Returns (f, logw): f over
    all K states with f_0 = 0 (NaN for a state without rows, gauge then set on the first
    sampled one), logw the normalised per-row log weights. ``f_init`` warm-starts the solve.
    ``info`` receives ``converged``, ``iterations``, ``max_delta_f`` and ``backend``."""
    kw = dict(tol=float(tol), maxiter=int(max_iter), threads=0, f_init=f_init)
    try:
        res = _mbar.solve_mbar(u_nk, window, backend=MBAR_BACKEND, **kw)
    except RuntimeError as exc:
        if _mbar.NUMBA_AVAILABLE:
            raise
        res = _mbar.solve_mbar(u_nk, window, backend=MBAR_FALLBACK_BACKEND, **kw)
        res["fallback_reason"] = str(exc)
    f = np.asarray(res["f_k"], dtype=float).copy()
    fin = np.flatnonzero(np.isfinite(f))
    if fin.size:
        f -= f[fin[0]]
    if info is not None:
        info.update(converged=bool(res["converged"]), iterations=int(res["iterations"]),
                    max_delta_f=float(res["max_delta"]), backend=str(res["backend"]))
    return f, np.asarray(res["logw"], dtype=float)


def solve_mbar(u_kn: np.ndarray, n_k: np.ndarray, *, tol: float = MBAR_TOL, max_iter: int = MBAR_MAX_ITER,
               info: Optional[Dict[str, Any]] = None, f_init: Optional[np.ndarray] = None) -> np.ndarray:
    """Self-consistent MBAR free energies (f_0 = 0) of the sampled states, (K, N) layout: the
    pre-v3 signature, now a thin wrapper over ``solve_rows`` (gareus-analyze's solver). MBAR's f
    depends on the per-state counts only, not on which row came from which state, so the rows
    are given the window ``repeat(arange(K), n_k)``. ``n_k`` must be positive integers summing
    to N. ``info`` receives ``converged``, ``iterations``, ``max_delta_f`` and ``backend``."""
    u_kn = np.asarray(u_kn, dtype=float)
    n = np.asarray(n_k, dtype=float)
    counts = np.rint(n).astype(np.int64)
    if n.ndim != 1 or n.size != u_kn.shape[0] or np.any(np.abs(n - counts) > 1e-9) or np.any(counts <= 0) \
            or int(counts.sum()) != u_kn.shape[1]:
        raise ValueError("solve_mbar: n_k must be one positive integer count per row of u_kn, summing to its "
                         f"columns (got {n.size} counts summing to {n.sum():g} for u_kn {u_kn.shape})")
    window = np.repeat(np.arange(counts.size), counts)
    f, _logw = solve_rows(np.ascontiguousarray(u_kn.T), window, tol=tol, max_iter=max_iter, f_init=f_init,
                          info=info)
    return f


__all__ = ["MBAR_BACKEND", "MBAR_FALLBACK_BACKEND", "MBAR_MAX_ITER", "MBAR_TOL", "solve_mbar", "solve_rows"]
