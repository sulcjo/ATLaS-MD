"""Pure-Python GaMD joint-envelope calibration: Welford stats, pooling, and
the exact gamd-openmm threshold/k0 formulas.

No OpenMM or gamd-openmm import in this module -- it must stay safe for
tests/ (see CLAUDE.md's OpenMM-free unit-test constraint). It is exercised
end-to-end (with real OpenMM data) by gareus/production.py and
gareus/integration_test.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WindowEnergyStats:
    """Per-window, per-boost-group potential-energy statistics from a short recon."""
    group: str
    window: int
    vmax: float
    vmin: float
    mean: float
    var: float
    n: int


class WelfordAccumulator:
    """Online mean/variance/min/max accumulator for one (window, boost-group) recon."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.vmax = -math.inf
        self.vmin = math.inf

    def update(self, value: float) -> None:
        value = float(value)
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if value > self.vmax:
            self.vmax = value
        if value < self.vmin:
            self.vmin = value

    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.m2 / (self.n - 1)

    def to_stats(self, group: str, window: int) -> WindowEnergyStats:
        if self.n < 1:
            raise ValueError(f"WelfordAccumulator for group={group!r} window={window} has no samples")
        return WindowEnergyStats(
            group=group, window=window, vmax=self.vmax, vmin=self.vmin,
            mean=self.mean, var=self.variance, n=self.n,
        )


@dataclass
class PooledEnvelope:
    group: str
    vmax: float
    vmin: float
    vavg: float
    sigmav: float
    n_total: int
    n_windows: int


def pool_window_stats(stats: list[WindowEnergyStats]) -> PooledEnvelope:
    """Pool per-window statistics for ONE boost group into a joint envelope.

    Vmax/Vmin are the joint extrema across windows. Mean/variance are pooled
    with the exact combined-sample-variance formula:
        mu_pooled = sum(N_i * mu_i) / sum(N_i)
        M2 = sum((N_i - 1) * var_i + N_i * (mu_i - mu_pooled)^2)
        sigma_pooled = sqrt(M2 / (sum(N_i) - 1))
    """
    if not stats:
        raise ValueError("pool_window_stats requires at least one window's statistics")
    groups = {s.group for s in stats}
    if len(groups) != 1:
        raise ValueError(f"pool_window_stats requires a single boost group, got {sorted(groups)}")
    n_total = sum(s.n for s in stats)
    if n_total < 2:
        raise ValueError("pool_window_stats requires at least 2 total samples across windows")
    vmax = max(s.vmax for s in stats)
    vmin = min(s.vmin for s in stats)
    mu_pooled = sum(s.n * s.mean for s in stats) / n_total
    m2 = sum((s.n - 1) * s.var + s.n * (s.mean - mu_pooled) ** 2 for s in stats)
    sigma_pooled = math.sqrt(m2 / (n_total - 1))
    return PooledEnvelope(
        group=stats[0].group, vmax=vmax, vmin=vmin, vavg=mu_pooled, sigmav=sigma_pooled,
        n_total=n_total, n_windows=len(stats),
    )
