"""Acceptance-math gate for the biased-MC barostat (spec section 10).

A noninteracting molecular volume sampler must reproduce

    p(V) proportional to V^Nmol * exp(-beta P V)

which is Gamma(shape=Nmol+1, scale=1/(beta P)) in V. The reference distribution
comes from scipy, never from the implementation's own helpers. The molecules
are rigid multi-atom groups, so an implementation that accidentally counts
atoms (or constrained DOF) in the Jacobian lands on a different shape and must
fail the same test.

Also checks the pressure-unit anchor: <V> = (Nmol+1) kT / P, which is only
correct if bar*nm^3 -> kJ/mol uses BAR_NM3_TO_KJ_PER_MOL and beta = 1/(RT).
"""
import math

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
import openmm.unit as unit  # noqa: E402
import scipy.stats as sps  # noqa: E402

from gareus.npt import (  # noqa: E402
    BAR_NM3_TO_KJ_PER_MOL,
    BiasedMCBarostatController,
    EnergyBreakdown,
)

R_KJ_MOL_K = 8.31446261815324e-3

N_MOL = 5
ATOMS_PER_MOL = 3
TEMPERATURE_K = 300.0
BETA = 1.0 / (R_KJ_MOL_K * TEMPERATURE_K)
TARGET_MEAN_NM3 = 27.0  # the 3 nm starting cube
GAMMA_SHAPE = N_MOL + 1
BETA_P = GAMMA_SHAPE / TARGET_MEAN_NM3  # per nm^3, from <V> = (Nmol+1)/(beta P)
PRESSURE_BAR = BETA_P / (BETA * BAR_NM3_TO_KJ_PER_MOL)
GAMMA_SCALE = 1.0 / BETA_P
N_TRIALS = 24000
BURN_IN = 500


class ZeroAdapter:
    """Noninteracting: every energy component is identically zero."""

    adapter_id = "zero"

    def snapshot(self, context, integrator):
        return None

    def evaluate(self, context, snapshot):
        return EnergyBreakdown(0.0, 0.0, 0.0, 0.0, 0.0)


def _ideal_gas_system():
    system = openmm.System()
    bonds = openmm.HarmonicBondForce()
    for m in range(N_MOL):
        base = m * ATOMS_PER_MOL
        for k in range(ATOMS_PER_MOL):
            system.addParticle(1.0)
        for k in range(ATOMS_PER_MOL - 1):
            bonds.addBond(base + k, base + k + 1, 0.15, 500.0)
    system.addForce(bonds)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3))
    return system


def _volumes(n_trials=N_TRIALS, seed=20260912):
    system = _ideal_gas_system()
    integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    rng = np.random.default_rng(11)
    pos = rng.uniform(0.2, 2.8, size=(system.getNumParticles(), 3))
    ctx.setPositions(pos)
    # half-width ~ 8 nm^3 against the 27 nm^3 start
    ctrl = BiasedMCBarostatController.initialize(
        ctx, ZeroAdapter(), pressure_bar=PRESSURE_BAR, temperature_k=TEMPERATURE_K,
        frequency_steps=1, volume_step_fraction=8.0 / 27.0, seed=seed)
    assert ctrl.state_dict()["n_molecules"] == N_MOL
    vols = np.empty(n_trials)
    for step in range(1, n_trials + 1):
        res = ctrl.attempt_due(step)
        vols[step - 1] = res.old_volume_nm3 if not res.accepted else res.proposed_volume_nm3
    # no hidden MD: the integrator must not have advanced simulation time
    assert ctx.getState().getTime() == 0.0 * unit.picosecond
    acceptance = ctrl.state_dict()["counters"]["accepted"] / n_trials
    assert 0.05 < acceptance < 0.95, f"degenerate acceptance {acceptance}"
    return vols[BURN_IN:], acceptance


def _integrated_autocorr_time(x: np.ndarray) -> float:
    x = x - x.mean()
    n = len(x)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (np.arange(n, 0, -1) * x.var())
    tau = 0.5
    for k in range(1, min(2000, n)):
        if acf[k] < 0.0:
            break
        tau += acf[k]
    return float(tau)


def _thinned(vols: np.ndarray):
    """KS assumes iid samples; the volume chain is a random walk, so thin it by
    twice its measured integrated autocorrelation time."""
    thin = max(1, int(math.ceil(2.0 * _integrated_autocorr_time(vols))))
    return vols[::thin], thin


def test_volume_ensemble_is_gamma_over_molecules():
    vols, _ = _volumes()
    vt, thin = _thinned(vols)
    stat, pvalue = sps.kstest(vt, "gamma", args=(GAMMA_SHAPE, 0.0, GAMMA_SCALE))
    print(f"[thin={thin}, n={len(vt)}] KS vs Gamma(shape={GAMMA_SHAPE}, "
          f"scale={GAMMA_SCALE:.6f}): D={stat:.4f} p={pvalue:.4g}")
    assert pvalue > 5e-3, (
        f"volume distribution does not match Gamma(shape=Nmol+1): D={stat:.4f}, p={pvalue:.3g}"
    )


def test_volume_ensemble_is_not_gamma_over_atoms():
    # The same samples must FAIL against the atom-count Jacobian (15 atoms ->
    # shape 16): this is what makes an accidental atom count detectable.
    vols, _ = _volumes()
    vt, thin = _thinned(vols)
    wrong_shape = N_MOL * ATOMS_PER_MOL + 1
    stat, pvalue = sps.kstest(vt, "gamma", args=(wrong_shape, 0.0, GAMMA_SCALE))
    print(f"[thin={thin}] KS vs Gamma(shape={wrong_shape}): D={stat:.4f} p={pvalue:.4g}")
    assert pvalue < 1e-6, (
        "samples are compatible with an atom-count Jacobian; the molecule-count "
        f"test would not have teeth (D={stat:.4f})"
    )


def test_ideal_gas_mean_volume_checks_pressure_units():
    vols, _ = _volumes()
    expected_mean = GAMMA_SHAPE / BETA_P  # (Nmol+1) kT / P
    n_eff = len(vols) / (2.0 * _integrated_autocorr_time(vols))
    se = vols.std(ddof=1) / math.sqrt(n_eff)
    print(f"mean V = {vols.mean():.3f} nm^3 (expected {expected_mean:.3f}, se~{se:.3f})")
    assert abs(vols.mean() - expected_mean) < 5.0 * se
    # and a wrong pressure conversion is decisively excluded
    wrong = GAMMA_SHAPE / (BETA * PRESSURE_BAR)  # bar*nm^3 silently treated as kJ/mol
    assert abs(wrong - expected_mean) > 20.0 * se
