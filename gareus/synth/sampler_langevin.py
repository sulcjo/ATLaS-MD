"""Adversarial overdamped-Langevin sampler for the adaptive-window harness.

Where the exact sampler draws independent points from the true biased
equilibrium, this propagates a single trajectory on ``F + U_w`` with anisotropic
diffusion (``D2 << D1`` makes CV2 slow).  Short trajectories then exhibit the
failure modes that actually break adaptive sampling: finite-sample noise,
autocorrelation, slow-CV2 hysteresis, and trapped/non-ergodic windows.

Overdamped Langevin (Euler-Maruyama):

    x_{t+dt} = x_t - D * beta * grad(F + U) * dt + sqrt(2 * D * dt) * xi

with reflecting boundaries at the CV domain edges.  ``grad F`` is taken by
central finite differences on the analytic surface; ``grad U`` is analytic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .landscapes import Landscape
from .sampler import Window


def _grad_f(landscape: Landscape, x1, x2, h=1e-3):
    f = landscape.energy
    g1 = (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h)
    g2 = (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h)
    return g1, g2


def _grad_u(window: Window, x1, x2):
    g1 = float(window.k1) * (x1 - float(window.center1))
    if window.center2 is not None and window.k2 is not None:
        g2 = float(window.k2) * (x2 - float(window.center2))
    else:
        g2 = 0.0
    return g1, g2


def _reflect(x, lo, hi):
    # reflect into [lo, hi] (handles a single overshoot per step)
    if x < lo:
        x = lo + (lo - x)
    elif x > hi:
        x = hi - (x - hi)
    return min(hi, max(lo, x))


def sample_window_langevin(landscape: Landscape, window: Window, n_samples: int,
                           *, beta: float = 1.0,
                           rng: Optional[np.random.Generator] = None,
                           D1: float = 1.0, D2: float = 0.05, dt: float = 2e-3,
                           steps_per_sample: int = 20, burn_in: int = 500,
                           reflect: bool = True, x0=None) -> np.ndarray:
    """Return ``(n_samples, 2)`` correlated samples from a short Langevin run."""
    rng = rng if rng is not None else np.random.default_rng()
    lo1, hi1 = landscape.cv1_bounds
    lo2, hi2 = landscape.cv2_bounds
    if x0 is not None:
        x1, x2 = float(x0[0]), float(x0[1])
    else:
        x1 = float(window.center1)
        x2 = float(window.center2) if window.center2 is not None else 0.5 * (lo2 + hi2)
    x1 = _reflect(x1, lo1, hi1)
    x2 = _reflect(x2, lo2, hi2)

    s1d = np.sqrt(2.0 * D1 * dt)
    s2d = np.sqrt(2.0 * D2 * dt)

    def step(x1, x2):
        gf1, gf2 = _grad_f(landscape, x1, x2)
        gu1, gu2 = _grad_u(window, x1, x2)
        x1n = x1 - D1 * beta * (gf1 + gu1) * dt + s1d * rng.standard_normal()
        x2n = x2 - D2 * beta * (gf2 + gu2) * dt + s2d * rng.standard_normal()
        if reflect:
            x1n = _reflect(x1n, lo1, hi1)
            x2n = _reflect(x2n, lo2, hi2)
        else:
            x1n = min(hi1, max(lo1, x1n))
            x2n = min(hi2, max(lo2, x2n))
        return x1n, x2n

    for _ in range(int(burn_in)):
        x1, x2 = step(x1, x2)

    out = np.empty((int(n_samples), 2), float)
    for i in range(int(n_samples)):
        for _ in range(int(steps_per_sample)):
            x1, x2 = step(x1, x2)
        out[i, 0] = x1
        out[i, 1] = x2
    return out
