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


@dataclass
class ThresholdResult:
    group: str
    threshold_energy: float
    k0: float
    k: float
    boosted: bool  # False if a degenerate guard tripped (k0 forced to 0)


def _energy_scale(vmax: float, vmin: float, vavg: float) -> float:
    return max(max(abs(vmax), abs(vmin)), max(abs(vavg), 1.0))


def lower_bound_threshold_and_k0(envelope: PooledEnvelope, sigma0: float) -> ThresholdResult:
    """Reproduce gamd-openmm's lower-bound threshold/k0 formula exactly.

    Source: gamd/langevin/base_integrator.py:607-637
    (calculate_common_threshold_energy_and_effective_harmonic_constant +
    _lower_bound_calculate_threshold_energy_and_effective_harmonic_constant),
    verified 2026-07-03 against the installed gamd-openmm 0.9.2 package.
    """
    vmax, vmin, vavg, sigmav = envelope.vmax, envelope.vmin, envelope.vavg, envelope.sigmav
    scale = _energy_scale(vmax, vmin, vavg)
    guard = 0.001 * scale
    degenerate = sigmav <= guard or abs(vmax - vmin) <= guard or abs(vmax - vavg) <= guard
    if degenerate:
        return ThresholdResult(group=envelope.group, threshold_energy=vmax, k0=0.0, k=0.0, boosted=False)
    k0prime = (sigma0 / sigmav) * (vmax - vmin) / (vmax - vavg)
    k0 = min(1.0, k0prime)
    k = k0 / (vmax - vmin)
    return ThresholdResult(group=envelope.group, threshold_energy=vmax, k0=k0, k=k, boosted=True)


def upper_bound_threshold_and_k0(envelope: PooledEnvelope, sigma0: float) -> ThresholdResult:
    """Reproduce gamd-openmm's upper-bound threshold/k0 formula exactly.

    Source: gamd/langevin/base_integrator.py:639-696, verified 2026-07-03.
    Falls back to the lower-bound formula whenever k0doubleprime is outside
    the open interval (0, 1) -- this is what the package itself does
    (base_integrator.py:671-696, the k0doubleprime_window >= 0 gate).
    """
    vmax, vmin, vavg, sigmav = envelope.vmax, envelope.vmin, envelope.vavg, envelope.sigmav
    scale = _energy_scale(vmax, vmin, vavg)
    guard = 0.001 * scale
    degenerate = sigmav <= guard or abs(vmax - vmin) <= guard or abs(vavg - vmin) <= guard
    k0doubleprime = 0.0 if degenerate else (1.0 - sigma0 / sigmav) * (vmax - vmin) / (vavg - vmin)
    if k0doubleprime <= 0.0 or k0doubleprime >= 1.0:
        return lower_bound_threshold_and_k0(envelope, sigma0)
    k0 = k0doubleprime
    threshold_energy = vmin + (vmax - vmin) / k0
    k = k0 / (vmax - vmin)
    return ThresholdResult(group=envelope.group, threshold_energy=threshold_energy, k0=k0, k=k, boosted=True)


def threshold_and_k0(boost_type: str, envelope: PooledEnvelope, sigma0: float) -> ThresholdResult:
    """Dispatch to the lower- or upper-bound formula by --gamd-boost-type prefix."""
    normalized = str(boost_type or "").strip().lower()
    from .pep_gamd import PEP_GAMD_PREFIX
    if normalized.startswith(PEP_GAMD_PREFIX):
        normalized = normalized[len(PEP_GAMD_PREFIX):]
    if normalized.startswith("lower"):
        return lower_bound_threshold_and_k0(envelope, sigma0)
    if normalized.startswith("upper"):
        return upper_bound_threshold_and_k0(envelope, sigma0)
    raise ValueError(f"Unknown gamd_boost_type threshold mode: {boost_type!r} (expected a 'lower*' or 'upper*' prefix)")


@dataclass
class GroupCalibration:
    group: str
    vmax: float
    vmin: float
    vavg: float
    sigmav: float
    k0: float
    k: float
    threshold_energy: float
    boosted: bool
    n_total: int
    n_windows: int


def compute_group_calibration(boost_type: str, envelope: PooledEnvelope, sigma0: float) -> GroupCalibration:
    result = threshold_and_k0(boost_type, envelope, sigma0)
    return GroupCalibration(
        group=envelope.group, vmax=envelope.vmax, vmin=envelope.vmin, vavg=envelope.vavg,
        sigmav=envelope.sigmav, k0=result.k0, k=result.k, threshold_energy=result.threshold_energy,
        boosted=result.boosted, n_total=envelope.n_total, n_windows=envelope.n_windows,
    )


@dataclass
class ConvergenceResult:
    """Outcome of the self-consistent boosted-recon calibration loop for one group.

    ``sigmav_trace`` is ``[seed_sigmav, iter1_sigmav, ...]`` (the seed comes from
    the cMD recon; each subsequent entry is a boosted-recon measurement).
    ``iters`` is the number of boosted iterations actually run (0 = legacy
    cMD-only path). ``converged`` is True when the last two sigmaV values agree
    within tolerance (trivially True when ``iters == 0``).
    """
    calibration: GroupCalibration
    sigmav_trace: list
    converged: bool
    iters: int


def envelope_converged(prev: PooledEnvelope, curr: PooledEnvelope, tol: float) -> bool:
    """True when curr.sigmav is within relative ``tol`` of prev.sigmav.

    Returns False for a non-positive previous sigmaV (no meaningful baseline).
    """
    if prev.sigmav <= 0.0:
        return False
    return abs(curr.sigmav - prev.sigmav) / prev.sigmav < tol


def run_calibration_convergence(measure_fn, boost_type: str, sigma0_by_group: dict,
                                seed_envelopes: dict, max_iters: int,
                                tol: float) -> dict:
    """Jointly iterate a boosted recon to self-consistency on sigmaV, all groups.

    One boosted recon pass measures every boost group's potential energy at once,
    so this driver is joint: ``measure_fn(calibrations) -> dict[group ->
    PooledEnvelope]`` runs a single boosted recon seeded with the current
    per-group ``GroupCalibration`` mapping and returns each group's freshly
    measured envelope. It is injected so this module stays OpenMM-free (see the
    module docstring).

    The loop is a negative feedback (larger sigmaV -> smaller k0' -> weaker boost
    -> smaller sigmaV) and converges to a fixed point; it stops when *every*
    group's sigmaV is within relative ``tol`` of its previous iteration, or at
    ``max_iters``. ``max_iters <= 0`` skips the boosted loop and returns the
    cMD-seed calibrations unchanged (legacy behavior).

    Returns ``dict[group -> ConvergenceResult]``.
    """
    groups = list(seed_envelopes.keys())
    envelopes = dict(seed_envelopes)
    calibrations = {g: compute_group_calibration(boost_type, envelopes[g], sigma0_by_group[g]) for g in groups}
    traces = {g: [envelopes[g].sigmav] for g in groups}
    if max_iters <= 0:
        return {g: ConvergenceResult(calibrations[g], traces[g], True, 0) for g in groups}
    per_group_conv = {g: False for g in groups}
    iters = 0
    for _ in range(int(max_iters)):
        new_envelopes = measure_fn(calibrations)
        iters += 1
        for g in groups:
            traces[g].append(new_envelopes[g].sigmav)
            per_group_conv[g] = envelope_converged(envelopes[g], new_envelopes[g], tol)
            envelopes[g] = new_envelopes[g]
            calibrations[g] = compute_group_calibration(boost_type, new_envelopes[g], sigma0_by_group[g])
        if all(per_group_conv.values()):
            break
    return {g: ConvergenceResult(calibrations[g], traces[g], per_group_conv[g], iters) for g in groups}


_PHYSICS_GLOBAL_PREFIXES = {
    "Vmax": "vmax", "Vmin": "vmin", "Vavg": "vavg", "sigmaV": "sigmav",
    "k0": "k0", "k": "k", "threshold_energy": "threshold_energy",
}


def overwrite_physics_globals(globals_all: dict, calibrations: dict) -> dict:
    """Return a copy of `globals_all` with pooled/recon-derived physics values
    substituted in for each boost group present in `calibrations`.

    `globals_all` is a CustomIntegrator globals dict (name -> value), e.g. from
    gareus.production.all_integrator_globals(). `calibrations` maps boost-group
    name (e.g. "NonBonded", "Dihedral", "Total") to a GroupCalibration. Only
    the exact physics global names for each present group are overwritten;
    every other global (stepCount, stage, windowCount, ForceScalingFactor,
    sigma0_<group>, ...) is left untouched -- those describe integrator
    bookkeeping/config that stays self-consistent independent of which
    Vmax/Vmin/k0/threshold values are in effect.
    """
    out = dict(globals_all)
    applied = []
    for group, calib in calibrations.items():
        for global_prefix, attr in _PHYSICS_GLOBAL_PREFIXES.items():
            name = f"{global_prefix}_{group}"
            if name in out:
                out[name] = float(getattr(calib, attr))
                applied.append(name)
    if not applied:
        raise RuntimeError(
            f"overwrite_physics_globals matched no global names for groups {sorted(calibrations)}; "
            "check the group-name suffix convention against gamd-openmm's actual globals dict."
        )
    return out
