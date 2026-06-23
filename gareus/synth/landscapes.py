"""Analytic 2D free-energy surfaces for the synthetic adaptive-window harness.

All surfaces return ``F(cv1, cv2)`` in units of ``kBT``.  ``Landscape.grid``
returns a surface shifted so ``min(F) == 0`` so thresholds like ``F <= min + 5
kBT`` are absolute.  CV1 is a contact fraction in ``[0, 1]``; CV2 is a
Ramachandran/dihedral map coordinate in ``[-1, 1]``.

These are toy potentials for *algorithm validation* (does the adaptive window
logic put windows where a known answer says they belong), not for production
science.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Cache of (cv1_axis, cv2_axis, F2d) keyed by (id(landscape), res). The grid is a
# pure function of the (immutable) landscape; oracle metrics rebuild it heavily.
_GRID_CACHE: dict = {}


@dataclass(frozen=True)
class Landscape:
    """An analytic free-energy surface with a known answer.

    Attributes
    ----------
    name : str
        Registry key / human label.
    f : callable
        ``f(cv1, cv2) -> F`` in kBT, broadcasting over numpy arrays.  Not
        required to be min-shifted; ``grid`` performs the shift.
    cv1_bounds, cv2_bounds : (float, float)
        Domain of the two collective variables.
    basins : tuple of (cv1, cv2)
        Approximate centers of the relevant low-F basins (used by tests and as
        sensible default window seeds).
    """

    name: str
    f: Callable[[np.ndarray, np.ndarray], np.ndarray]
    cv1_bounds: tuple[float, float] = (0.0, 1.0)
    cv2_bounds: tuple[float, float] = (-1.0, 1.0)
    basins: tuple[tuple[float, float], ...] = ()

    def energy(self, cv1, cv2) -> np.ndarray:
        """Raw F in kBT (not min-shifted), broadcasting over arrays."""
        return np.asarray(self.f(np.asarray(cv1, float), np.asarray(cv2, float)), float)

    def grid(self, res: int = 100):
        """Return ``(cv1_axis, cv2_axis, F2d)`` with ``F2d`` shifted to min 0.

        ``F2d`` has shape ``(res, res)`` indexed ``[i1, i2]`` (``indexing="ij"``).
        Memoized by (landscape identity, res): the grid is pure, and oracle
        metrics rebuild it thousands of times, so caching is a large speed-up.
        Returned arrays are read-only copies' views — treat as immutable.
        """
        key = (id(self), int(res))
        cached = _GRID_CACHE.get(key)
        if cached is not None:
            return cached
        c1 = np.linspace(self.cv1_bounds[0], self.cv1_bounds[1], res)
        c2 = np.linspace(self.cv2_bounds[0], self.cv2_bounds[1], res)
        g1, g2 = np.meshgrid(c1, c2, indexing="ij")
        f = self.energy(g1, g2)
        f = f - float(np.min(f))
        c1.flags.writeable = False
        c2.flags.writeable = False
        f.flags.writeable = False
        _GRID_CACHE[key] = (c1, c2, f)
        return _GRID_CACHE[key]


def _gauss(cv1, cv2, mu1, mu2, s1, s2):
    return np.exp(-0.5 * (((cv1 - mu1) / s1) ** 2 + ((cv2 - mu2) / s2) ** 2))


def mixture_wells() -> Landscape:
    """Mixture of 2D Gaussian wells of varied depth/width plus a weak decoy.

    ``F = -log(sum_i w_i N(mu_i, Sigma_i))``.  Exercises create / remove /
    economy: several distinct basins the windows should cover, and a shallow
    high-F decoy that should *not* attract a window.
    """
    # (weight, mu1, mu2, s1, s2)
    wells = [
        (1.0, 0.18, -0.65, 0.10, 0.22),   # broad, deep (global min basin)
        (0.55, 0.52, -0.05, 0.05, 0.10),  # narrow, shallow
        (0.85, 0.84, 0.55, 0.05, 0.08),   # narrow, deep
        (0.10, 0.40, 0.80, 0.06, 0.10),   # weak decoy (stays high F)
    ]

    def f(cv1, cv2):
        dens = np.zeros(np.broadcast(np.asarray(cv1, float), np.asarray(cv2, float)).shape)
        for (w, m1, m2, s1, s2) in wells:
            dens = dens + w * _gauss(cv1, cv2, m1, m2, s1, s2)
        dens = np.clip(dens, 1e-12, None)
        return -np.log(dens)

    basins = tuple((w[1], w[2]) for w in wells[:3])
    return Landscape("mixture-wells", f, (0.0, 1.0), (-1.0, 1.0), basins)


def gated_barrier() -> Landscape:
    """Mixture wells plus a CV1 ridge with a single low-F pass in CV2.

    Exercises 2D patch placement, bridge coverage, and exchange-graph
    connectivity: windows must find the one gap through the ridge.
    """
    base = mixture_wells()
    x0, sx = 0.50, 0.04          # ridge location/width in CV1
    ygap, sy = 0.20, 0.18        # the pass location/width in CV2
    amp = 6.0                    # ridge height in kBT

    def f(cv1, cv2):
        ridge = amp * np.exp(-((cv1 - x0) / sx) ** 2) * (1.0 - np.exp(-((cv2 - ygap) / sy) ** 2))
        return base.energy(cv1, cv2) + ridge

    return Landscape("gated-barrier", f, (0.0, 1.0), (-1.0, 1.0), base.basins)


def banana_valley() -> Landscape:
    """Curved low-F tube along ``cv2 = a*sin(2*pi*cv1)`` with variable width.

    Exercises center shifting and anisotropic k: the relevant region is a
    non-rectangular curved channel, so good windows must follow the curve.
    """
    a = 0.7

    def f(cv1, cv2):
        cv1 = np.asarray(cv1, float)
        cv2 = np.asarray(cv2, float)
        track = a * np.sin(2.0 * np.pi * cv1)
        width = 0.10 + 0.10 * (0.5 * (1.0 + np.cos(2.0 * np.pi * cv1)))  # narrows/widens
        perp = ((cv2 - track) / width) ** 2
        along = 0.8 * (cv1 - 0.5) ** 2 / (0.3 ** 2)  # gentle bias toward center
        return 0.5 * perp + along

    return Landscape("banana-valley", f, (0.0, 1.0), (-1.0, 1.0),
                     basins=((0.5, 0.0),))


def slow_cv2_double_branch() -> Landscape:
    """Double well in CV2 with weak CV1 coupling and a high CV2 saddle.

    Exact sampling covers both CV2 branches; a short Langevin run started in one
    branch stays trapped.  Exercises top-up vs. bad window moves and the
    non-ergodicity sensitivity of the adversarial sampler.
    """
    def f(cv1, cv2):
        cv1 = np.asarray(cv1, float)
        cv2 = np.asarray(cv2, float)
        # double well in cv2: minima at -0.6 and +0.6 (value 0), ~6 kBT saddle at 0
        dw = 6.0 * ((cv2 / 0.6) ** 2 - 1.0) ** 2
        # weak coupling + gentle cv1 confinement
        coupling = 0.6 * (cv1 - 0.5) * cv2
        conf = 1.2 * (cv1 - 0.5) ** 2 / (0.3 ** 2)
        return dw + coupling + conf

    return Landscape("slow-cv2-double-branch", f, (0.0, 1.0), (-1.0, 1.0),
                     basins=((0.5, -0.6), (0.5, 0.6)))


LANDSCAPES: dict[str, Landscape] = {
    "mixture-wells": mixture_wells(),
    "gated-barrier": gated_barrier(),
    "banana-valley": banana_valley(),
    "slow-cv2-double-branch": slow_cv2_double_branch(),
}
