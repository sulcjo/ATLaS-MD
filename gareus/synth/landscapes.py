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


def rugged_1d() -> Landscape:
    """Rugged 6-basin CV1 profile with a large barrier, plus a mild fast CV2 well.

    Built for the replica-count study: the PMF lives along CV1 (six basins of
    varied depth), with a deliberately LARGE (~10 kBT) barrier between basins 3
    and 4 that a window ladder must bridge. CV2 is a single shallow, fast well so
    the 2D machinery runs without CV2 confounding the few-vs-many-replica answer.
    """
    # (weight, mu1, sigma1) along CV1 — six basins of varied depth/width
    wells = [
        (1.00, 0.07, 0.035),   # deep edge basin
        (0.45, 0.20, 0.030),   # shallow
        (0.80, 0.36, 0.030),   # medium (left of the big barrier)
        (0.70, 0.56, 0.030),   # medium (right of the big barrier)
        (0.35, 0.73, 0.028),   # shallow
        (0.90, 0.92, 0.035),   # deep edge basin
    ]
    cv2_k = 3.0   # shallow, fast single CV2 well at 0 (mild nuisance coordinate)

    def f(cv1, cv2):
        cv1 = np.asarray(cv1, float)
        dens = np.zeros(np.broadcast(cv1, np.asarray(cv2, float)).shape)
        for (w, m, s) in wells:
            dens = dens + w * np.exp(-0.5 * ((cv1 - m) / s) ** 2)
        dens = np.clip(dens, 1e-12, None)
        g = -np.log(dens)
        # large explicit barrier between basin 3 (0.36) and basin 4 (0.56)
        g = g + 9.0 * np.exp(-0.5 * ((cv1 - 0.46) / 0.025) ** 2)
        # mild fast CV2 confinement (single shallow well at 0)
        g = g + 0.5 * cv2_k * np.asarray(cv2, float) ** 2
        return g

    basins = tuple((w[1], 0.0) for w in wells)
    return Landscape("rugged-1d", f, (0.0, 1.0), (-1.0, 1.0), basins)


def rugged_2d() -> Landscape:
    """Rugged in BOTH CVs: the rugged-1d CV1 profile (6 basins, ~14 kBT barrier)
    crossed with a structured CV2 — three CV2 wells (-0.6, 0, 0.6) separated by
    ~6 kBT barriers. Now a window ladder must bridge barriers along both axes, so
    a pure-CV1 ladder no longer suffices and the replica budget must cover 2D.
    """
    cv1_wells = [
        (1.00, 0.07, 0.035), (0.45, 0.20, 0.030), (0.80, 0.36, 0.030),
        (0.70, 0.56, 0.030), (0.35, 0.73, 0.028), (0.90, 0.92, 0.035),
    ]
    cv2_wells = [(1.0, -0.5, 0.13), (1.0, 0.5, 0.13)]   # double well

    def f(cv1, cv2):
        cv1 = np.asarray(cv1, float)
        cv2 = np.asarray(cv2, float)
        shape = np.broadcast(cv1, cv2).shape
        d1 = np.zeros(shape)
        for (w, m, s) in cv1_wells:
            d1 = d1 + w * np.exp(-0.5 * ((cv1 - m) / s) ** 2)
        d2 = np.zeros(shape)
        for (w, m, s) in cv2_wells:
            d2 = d2 + w * np.exp(-0.5 * ((cv2 - m) / s) ** 2)
        g = -np.log(np.clip(d1, 1e-12, None)) - np.log(np.clip(d2, 1e-12, None))
        g = g + 9.0 * np.exp(-0.5 * ((cv1 - 0.46) / 0.025) ** 2)        # big CV1 barrier
        g = g + 7.0 * np.exp(-0.5 * (cv2 / 0.10) ** 2)                  # CV2 barrier at 0
        return g

    basins = tuple((c1[1], c2[1]) for c1 in cv1_wells for c2 in cv2_wells)
    return Landscape("rugged-2d", f, (0.0, 1.0), (-1.0, 1.0), basins)


def chaos_2d() -> Landscape:
    """Highly rugged 2D: 8 CV1 basins × 3 CV2 wells (24 sub-basins) with two
    independent barriers per CV axis, scattered metastable traps, and anisotropic
    noise bumps. Much harder than rugged_2d — requires R≥16 for adequate coverage.

    Challenges added vs rugged_2d:
    - 8 (vs 6) CV1 basins + 3 (vs 2) CV2 wells = 24 sub-basins
    - Main CV1 barrier 18 kBT (vs implied ~14 kBT in rugged_2d)
    - Secondary CV1 barrier 10 kBT, CV2-dependent (asymmetric, diagonal bridging required)
    - Extra CV2 saddle at +0.33 (4 effective CV2 zones)
    - 10 noise Gaussian bumps (3–5 kBT) as reproducible metastable traps
    """
    cv1_wells = [
        (1.00, 0.05, 0.030),
        (0.50, 0.15, 0.025),
        (0.85, 0.28, 0.028),
        (0.75, 0.42, 0.025),
        (0.70, 0.58, 0.025),
        (0.60, 0.70, 0.025),
        (0.45, 0.82, 0.022),
        (0.95, 0.94, 0.032),
    ]
    cv2_wells = [
        (1.00, -0.65, 0.10),
        (0.70,  0.00, 0.08),
        (0.90,  0.65, 0.10),
    ]
    # Reproducible metastable traps: (amplitude, mu1, mu2, s1, s2)
    _traps = [
        (4.0, 0.22, -0.30, 0.040, 0.040),
        (5.0, 0.35,  0.40, 0.040, 0.050),
        (3.5, 0.64, -0.50, 0.050, 0.050),
        (4.5, 0.75,  0.25, 0.040, 0.040),
        (3.0, 0.12,  0.50, 0.050, 0.060),
        (4.0, 0.88, -0.30, 0.050, 0.040),
        (3.5, 0.50, -0.80, 0.060, 0.040),
        (4.0, 0.50,  0.80, 0.060, 0.040),
        (5.0, 0.35, -0.15, 0.030, 0.040),
        (4.5, 0.78,  0.55, 0.030, 0.030),
    ]

    def f(cv1, cv2):
        cv1 = np.asarray(cv1, float)
        cv2 = np.asarray(cv2, float)
        shape = np.broadcast(cv1, cv2).shape
        d1 = np.zeros(shape)
        for (w, m, s) in cv1_wells:
            d1 = d1 + w * np.exp(-0.5 * ((cv1 - m) / s) ** 2)
        d2 = np.zeros(shape)
        for (w, m, s) in cv2_wells:
            d2 = d2 + w * np.exp(-0.5 * ((cv2 - m) / s) ** 2)
        g = -np.log(np.clip(d1, 1e-12, None)) - np.log(np.clip(d2, 1e-12, None))
        # Main CV1 barrier 18 kBT
        g = g + 18.0 * np.exp(-0.5 * ((cv1 - 0.50) / 0.022) ** 2)
        # Secondary CV1 barrier 10 kBT, CV2-dependent (highest at CV2~0)
        g = g + 10.0 * np.exp(-0.5 * ((cv1 - 0.25) / 0.025) ** 2) \
              * (0.5 + 0.5 * np.exp(-0.5 * (cv2 / 0.30) ** 2))
        # CV2 main barrier 10 kBT at 0
        g = g + 10.0 * np.exp(-0.5 * (cv2 / 0.08) ** 2)
        # CV2 saddle 5 kBT at +0.33 (creates 4 effective CV2 zones)
        g = g + 5.0 * np.exp(-0.5 * ((cv2 - 0.33) / 0.07) ** 2)
        # Noise bumps / traps
        for (amp, m1, m2, s1, s2) in _traps:
            g = g + amp * np.exp(-0.5 * (((cv1 - m1) / s1) ** 2 + ((cv2 - m2) / s2) ** 2))
        return g

    basins = tuple((c1[1], c2[1]) for c1 in cv1_wells for c2 in cv2_wells)
    return Landscape("chaos-2d", f, (0.0, 1.0), (-1.0, 1.0), basins)


LANDSCAPES: dict[str, Landscape] = {
    "mixture-wells": mixture_wells(),
    "gated-barrier": gated_barrier(),
    "banana-valley": banana_valley(),
    "slow-cv2-double-branch": slow_cv2_double_branch(),
    "rugged-1d": rugged_1d(),
    "rugged-2d": rugged_2d(),
    "chaos-2d": chaos_2d(),
}
