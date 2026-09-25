"""Synthetic biased samplers that replace MD for the adaptive-window harness.

Each umbrella window samples from the biased Boltzmann distribution

    p_w(cv) ~ exp(-beta * (F(cv) + U_w(cv))),   U_w = 0.5*k1*(cv1-c1)^2 + 0.5*k2*(cv2-c2)^2

Two modes:

* ``exact``    -- grid inverse-CDF draw from the analytic surface.  This is the
  oracle/regression sampler: overlaps, exchange acceptance, and PMF baselines
  computed from it are exact up to grid resolution and sample count.
* ``langevin`` -- cheap overdamped Langevin on F+U (see ``sampler_langevin``).
  The adversarial sampler: finite counts, autocorrelation, slow-CV2 hysteresis,
  trapped windows.

Units note: ``F`` is already in kBT (``beta`` folded in), so the default
``beta=1.0`` keeps F in reduced units.  ``k`` is supplied in the same reduced
"per-CV^2" units used inside the harness; callers converting from the real
args' kcal/mol/CV^2 force constants must scale by ``beta_real`` first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .landscapes import Landscape


@dataclass(frozen=True)
class Window:
    """A 2D umbrella window. ``center2``/``k2`` may be ``None`` for 1D."""

    center1: float
    k1: float
    center2: Optional[float] = None
    k2: Optional[float] = None
    lam: float = 0.0          # lambda-ladder rung: boost strength in [0, 1] (see ``boost_dv``)


def BIAS(window: Window, cv1, cv2) -> np.ndarray:
    """Reduced umbrella bias energy ``U_w(cv)`` over arrays."""
    u = 0.5 * float(window.k1) * (np.asarray(cv1, float) - float(window.center1)) ** 2
    if window.center2 is not None and window.k2 is not None:
        u = u + 0.5 * float(window.k2) * (np.asarray(cv2, float) - float(window.center2)) ** 2
    return u


BOOST_A = 0.5               # flattening strength of the synthetic boost at lambda = 1
BOOST_REF_PERCENTILE = 60.0  # E_ref: this percentile of F over the landscape's grid
_E_REF_CACHE: dict = {}


def boost_reference_energy(landscape: Landscape, res: int = 120) -> float:
    """E_ref of the synthetic boost: the 60th percentile of raw F over a ``res x res`` grid."""
    key = (id(landscape), int(res))
    if key not in _E_REF_CACHE:
        g1, g2 = np.meshgrid(np.linspace(*landscape.cv1_bounds, res),
                             np.linspace(*landscape.cv2_bounds, res), indexing="ij")
        _E_REF_CACHE[key] = float(np.percentile(landscape.energy(g1, g2), BOOST_REF_PERCENTILE))
    return _E_REF_CACHE[key]


def boost_dv(landscape: Landscape, window: Window, cv1, cv2) -> np.ndarray:
    """Synthetic GaMD-like boost of a rung: ``dV = lam * a * max(0, E_ref - F(cv))`` in kBT.

    A lambda-scaled flattening of the analytic surface below E_ref, so states at one
    centre on different rungs are distinct Hamiltonians with a closed-form energy
    difference (energy-space rung overlap is well defined).
    """
    lam = float(window.lam)
    if lam == 0.0:
        return np.zeros(np.broadcast(np.asarray(cv1, float), np.asarray(cv2, float)).shape)
    f = landscape.energy(cv1, cv2)
    return lam * BOOST_A * np.maximum(0.0, boost_reference_energy(landscape) - f)


def sample_window_exact(landscape: Landscape, window: Window, n_samples: int,
                        *, beta: float = 1.0, res: int = 120,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draw ``n_samples`` ``(cv1, cv2)`` points from the biased (and, for lam > 0, boosted) equilibrium.

    Builds the biased probability on a ``res x res`` grid, normalizes, draws
    cell indices by inverse-CDF (multinomial), and adds intra-cell uniform
    jitter.  Deterministic given ``rng``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    # Cell-CENTERED grid: every cell has equal width d, so the multinomial
    # point-weight per cell equals its probability mass (no boundary
    # over-density that endpoint-inclusive linspace would introduce).
    lo1, hi1 = landscape.cv1_bounds
    lo2, hi2 = landscape.cv2_bounds
    d1 = (hi1 - lo1) / res
    d2 = (hi2 - lo2) / res
    c1 = lo1 + (np.arange(res) + 0.5) * d1
    c2 = lo2 + (np.arange(res) + 0.5) * d2
    g1, g2 = np.meshgrid(c1, c2, indexing="ij")
    f = landscape.energy(g1, g2)
    # The rung's boost is part of the sampled Hamiltonian (zero at lam = 0).
    logp = -beta * (f + BIAS(window, g1, g2) + boost_dv(landscape, window, g1, g2))
    logp = logp - logp.max()
    p = np.exp(logp).ravel()
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        # Degenerate surface: fall back to uniform over the grid.
        p = np.ones_like(p)
        total = p.sum()
    p = p / total
    idx = rng.choice(p.size, size=int(n_samples), p=p)
    i1, i2 = np.unravel_index(idx, f.shape)
    s1 = c1[i1] + rng.uniform(-0.5, 0.5, idx.size) * d1
    s2 = c2[i2] + rng.uniform(-0.5, 0.5, idx.size) * d2
    s1 = np.clip(s1, lo1, hi1)
    s2 = np.clip(s2, lo2, hi2)
    return np.column_stack([s1, s2])


def sample_window(landscape: Landscape, window: Window, n_samples: int,
                  *, mode: str = "exact", beta: float = 1.0,
                  rng: Optional[np.random.Generator] = None, **kw) -> np.ndarray:
    """Dispatch to the requested sampler mode."""
    if mode == "exact":
        return sample_window_exact(landscape, window, n_samples, beta=beta,
                                   res=int(kw.get("res", 120)), rng=rng)
    if mode == "langevin":
        from .sampler_langevin import sample_window_langevin
        lang_kw = {k: v for k, v in kw.items() if k != "res"}
        return sample_window_langevin(landscape, window, n_samples, beta=beta,
                                      rng=rng, **lang_kw)
    raise ValueError(f"unknown sampler mode: {mode!r}")
