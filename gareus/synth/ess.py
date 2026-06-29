"""Effective-sample-size cost model for the doubly-adaptive epoch study.

Exact i.i.d. sampling makes top-up a free, monotone variance reduction with no
equilibration or autocorrelation cost — which means epoch/top-up *policies* (where
to spend a finite budget) cannot be distinguished in pure exact mode. This module
adds the missing cost so allocation matters, without paying full Langevin dynamics:

    ESS = max(0, n_raw - burn_in) / (1 + 2 * tau_int)

where the integrated autocorrelation time ``tau_int`` grows for tight windows
(large k) and for windows sitting on a steep part of the surface (barriers / weak
regions), and a one-time ``burn_in`` is charged when a window is first created.

The synthetic sampler still draws i.i.d. points; we then *thin* each window's raw
samples down to its ESS before they enter diagnostics, overlap, and MBAR. So:
  - topping up a low-ESS window raises its ESS (more raw -> more effective);
  - a freshly added window wastes its first ``burn_in`` raws (eager-add is taxed);
  - tight / barrier windows convert raw budget to ESS less efficiently.
"""
from __future__ import annotations

import numpy as np

from .landscapes import Landscape
from .sampler import Window

TAU_BASE = 1.0          # baseline integrated autocorrelation time (in "frames")
K_REF = 50.0            # reference spring; tighter windows (k>K_REF) mix slower
TAU_K_FACTOR = 1.5      # how strongly tight k inflates tau
TAU_BARRIER_FACTOR = 4.0  # how strongly a steep surface inflates tau
BURN_IN = 120           # one-time raw samples discarded when a window is created


def _local_steepness(landscape: Landscape, c1: float, c2: float, h: float = 0.02) -> float:
    """Normalized |grad F| at a window center (high on barriers/ridges), in [0, ~1+]."""
    f = landscape.energy
    g1 = (float(f(np.array([c1 + h]), np.array([c2]))[0])
          - float(f(np.array([c1 - h]), np.array([c2]))[0])) / (2 * h)
    g2 = (float(f(np.array([c1]), np.array([c2 + h]))[0])
          - float(f(np.array([c1]), np.array([c2 - h]))[0])) / (2 * h)
    grad = float(np.hypot(g1, g2))
    return grad / (1.0 + grad)   # squashed to [0,1)


def tau_int(landscape: Landscape, window: Window,
            *, tau_base: float = TAU_BASE, k_factor: float = TAU_K_FACTOR,
            barrier_factor: float = TAU_BARRIER_FACTOR) -> float:
    """Integrated autocorrelation time for a window (>= tau_base)."""
    k = float(window.k1)
    tight = max(0.0, k / K_REF - 1.0)
    steep = _local_steepness(landscape, float(window.center1),
                             float(window.center2 if window.center2 is not None else 0.0))
    return float(tau_base * (1.0 + k_factor * tight + barrier_factor * steep))


def effective_count(n_raw: int, tau: float, *, burn_in: int = BURN_IN,
                    is_new: bool = False) -> float:
    """ESS from raw count: discard burn-in (only when newly created), divide by 1+2tau."""
    usable = max(0.0, float(n_raw) - (float(burn_in) if is_new else 0.0))
    return usable / (1.0 + 2.0 * float(tau))


def thin_to_ess(samples: np.ndarray, n_eff: int,
                rng: np.random.Generator | None = None) -> np.ndarray:
    """Return ``n_eff`` (<= len) decorrelated draws by even striding over the raws."""
    samples = np.asarray(samples, float)
    n = samples.shape[0]
    m = int(max(0, min(n, n_eff)))
    if m <= 0:
        return samples[:0]
    if m >= n:
        return samples
    idx = np.linspace(0, n - 1, m).round().astype(int)
    return samples[idx]
