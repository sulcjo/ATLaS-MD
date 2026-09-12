"""Coupled volume/energy target gate for the biased-MC barostat (spec section 10).

A small analytic model: rigid multi-atom molecules whose U* components are
known functions of the box volume, including a Pep-GaMD-shaped dependent dual
boost (Dihedral channel first, then the Total channel on V_pep + b_d) and a
volume-dependent auxiliary energy. The controller under test is the real
BiasedMCBarostatController; the oracle is independent quadrature of

    p(V) proportional to V^Nmol * exp(-beta (U*(V) + P V))

never any implementation helper.

Mandatory negative variants, both of which must FAIL the ensemble test:
  * omitting the boost entirely;
  * adding the auxiliary energy -- which must fail even at lambda = 0, because
    the auxiliary term is volume-dependent.
Both are additionally checked to fail FOR THE RIGHT REASON: the same samples
must still match the quadrature of the (wrong) target they actually sampled.
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
TEMPERATURE_K = 300.0
BETA = 1.0 / (R_KJ_MOL_K * TEMPERATURE_K)
PRESSURE_BAR = 5.0
N_MOL = 4
N_TRIALS = 16000
BURN_IN = 800

# ---- the analytic model (kJ/mol, V in nm^3) -----------------------------------

C0, D0 = 0.8, 2.0      # E0: base physical, minimum at V=9
C2, D2 = 0.6, 1.0      # E2: dihedral-channel input, minimum at V=9.5
A1 = 3.0               # E1: auxiliary, deliberately volume-dependent
W_K, V_W = 0.5, 7.0    # W: umbrella-shaped bias, minimum at V=7

THETA_D, VMAX_D, VMIN_D, K0_D = 25.0, 40.0, -10.0, 0.6
THETA_T, VMAX_T, VMIN_T, K0_T = 50.0, 90.0, -30.0, 0.4


def e0(v): return C0 * (v - 9.0) ** 2 + D0
def e2(v): return C2 * (v - 9.5) ** 2 + D2
def e1(v): return A1 * v
def w(v): return W_K * (v - V_W) ** 2


def _channel(e, theta, vmax, vmin, k0):
    """Lower-bound GaMD channel on the model (same shape as the integrator's,
    written out here as part of the model definition)."""
    if abs(vmax - vmin) <= 0.001 * max(abs(theta), abs(e), 1.0):
        return 0.0
    b = 0.5 * k0 * (theta - e) ** 2 / (vmax - vmin)
    if not (e + b < theta):
        return 0.0
    return b


def model_boost(v, k0_d=K0_D, k0_t=K0_T):
    b_d = _channel(e2(v), THETA_D, VMAX_D, VMIN_D, k0_d)
    v_pep = e0(v) - e1(v) + e2(v)
    b_t = _channel(v_pep + b_d, THETA_T, VMAX_T, VMIN_T, k0_t)
    return b_d + b_t


def u_star_correct(v):
    return e0(v) + e2(v) + w(v) + model_boost(v)


def u_star_no_boost(v):
    return e0(v) + e2(v) + w(v)


def u_star_aux_added(v, k0_d=K0_D, k0_t=K0_T):
    return e0(v) + e2(v) + w(v) + e1(v) + model_boost(v, k0_d, k0_t)


# ---- the model adapter (test-side model, never the oracle) --------------------


def _read_volume(context):
    vectors = context.getState().getPeriodicBoxVectors()
    if hasattr(vectors, "value_in_unit"):
        vectors = vectors.value_in_unit(unit.nanometer)
    box = np.array([[float(v[i]) for i in range(3)] for v in vectors])
    return float(abs(np.linalg.det(box)))


class ModelAdapter:
    """Feeds the controller the model's U*; variant selects the target."""

    def __init__(self, target="correct", k0_scale=1.0):
        self.adapter_id = f"model-{target}"
        self._target = target
        self._k0_scale = k0_scale

    def snapshot(self, context, integrator):
        return None

    def evaluate(self, context, snapshot):
        v = _read_volume(context)
        k0_d, k0_t = K0_D * self._k0_scale, K0_T * self._k0_scale
        boost = model_boost(v, k0_d, k0_t)
        if self._target == "correct":
            effective = e0(v) + e2(v) + w(v) + boost
        elif self._target == "no_boost":
            effective = e0(v) + e2(v) + w(v)
        elif self._target == "aux_added":
            effective = e0(v) + e2(v) + w(v) + e1(v) + boost
        else:
            raise AssertionError(self._target)
        return EnergyBreakdown(e0(v) + e2(v), w(v), boost, e1(v), effective)


# ---- independent reference: quadrature ---------------------------------------


def _reference_cdf(u_star_fn, grid=None):
    if grid is None:
        grid = np.linspace(0.3, 60.0, 12000)
    logp = (N_MOL * np.log(grid)
            - BETA * (np.array([u_star_fn(v) for v in grid])
                      + PRESSURE_BAR * grid * BAR_NM3_TO_KJ_PER_MOL))
    logp -= logp.max()
    p = np.exp(logp)
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    return grid, cdf


def _ks_vs_reference(samples, u_star_fn):
    grid, cdf = _reference_cdf(u_star_fn)
    lookup = lambda x: np.interp(x, grid, cdf)
    return sps.kstest(samples, lookup)


def _integrated_autocorr_time(x):
    x = x - x.mean()
    n = len(x)
    acf = np.correlate(x, x, mode="full")[n - 1:] / (np.arange(n, 0, -1) * x.var())
    tau = 0.5
    for k in range(1, min(2000, n)):
        if acf[k] < 0.0:
            break
        tau += acf[k]
    return float(tau)


def _sample_volumes(adapter, seed=424242):
    system = openmm.System()
    bonds = openmm.HarmonicBondForce()
    for m in range(N_MOL):
        base = m * 3
        for k in range(3):
            system.addParticle(1.0)
        bonds.addBond(base, base + 1, 0.15, 500.0)
        bonds.addBond(base + 1, base + 2, 0.15, 500.0)
    system.addForce(bonds)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2.1, 0, 0), openmm.Vec3(0, 2.1, 0), openmm.Vec3(0, 0, 2.1))
    integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(np.random.default_rng(5).uniform(0.2, 1.9, size=(12, 3)))
    ctrl = BiasedMCBarostatController.initialize(
        ctx, adapter, pressure_bar=PRESSURE_BAR, temperature_k=TEMPERATURE_K,
        frequency_steps=1, volume_step_fraction=0.10, seed=seed)
    vols = np.empty(N_TRIALS)
    for step in range(1, N_TRIALS + 1):
        res = ctrl.attempt_due(step)
        vols[step - 1] = res.proposed_volume_nm3 if res.accepted else res.old_volume_nm3
    acc = ctrl.state_dict()["counters"]["accepted"] / N_TRIALS
    assert 0.05 < acc < 0.95, f"degenerate acceptance {acc}"
    vols = vols[BURN_IN:]
    thin = max(1, int(math.ceil(2.0 * _integrated_autocorr_time(vols))))
    return vols[::thin]


# ---- the gate -----------------------------------------------------------------

def test_coupled_target_is_sampled_correctly():
    vols = _sample_volumes(ModelAdapter("correct"))
    stat, pvalue = _ks_vs_reference(vols, u_star_correct)
    print(f"correct model: n={len(vols)} D={stat:.4f} p={pvalue:.4g}")
    assert pvalue > 5e-3, (
        f"controller does not sample the coupled target: D={stat:.4f}, p={pvalue:.3g}"
    )


def test_negative_variant_omitting_the_boost_fails():
    vols = _sample_volumes(ModelAdapter("no_boost"))
    stat_correct, p_correct = _ks_vs_reference(vols, u_star_correct)
    stat_wrong, p_wrong = _ks_vs_reference(vols, u_star_no_boost)
    print(f"no-boost variant: vs correct target D={stat_correct:.4f} p={p_correct:.3g}; "
          f"vs its own (wrong) target D={stat_wrong:.4f} p={p_wrong:.3g}")
    assert p_correct < 1e-6, (
        "omitting the boost still passes the ensemble test -- the gate has no teeth "
        f"(D={stat_correct:.4f})"
    )
    assert p_wrong > 5e-3, (
        "the no-boost samples do not match their own target either; the variant is "
        "failing for the wrong reason (broken sampling, not a mis-specified energy)"
    )


def test_negative_variant_adding_the_auxiliary_fails_even_at_lambda_zero():
    # lambda = 0: both channel k0 vanish, so the boost is identically zero and
    # the ONLY difference from the correct target is the auxiliary term.
    vols = _sample_volumes(ModelAdapter("aux_added", k0_scale=0.0))
    stat_correct, p_correct = _ks_vs_reference(vols, u_star_correct)
    stat_wrong, p_wrong = _ks_vs_reference(
        vols, lambda v: u_star_aux_added(v, k0_d=0.0, k0_t=0.0))
    print(f"aux variant (lambda=0): vs correct target D={stat_correct:.4f} "
          f"p={p_correct:.3g}; vs its own (wrong) target D={stat_wrong:.4f} p={p_wrong:.3g}")
    assert p_correct < 1e-6, (
        "adding the auxiliary energy still passes the ensemble test at lambda=0 -- "
        f"the gate has no teeth (D={stat_correct:.4f})"
    )
    assert p_wrong > 5e-3, (
        "the aux-added samples do not match their own target either; the variant is "
        "failing for the wrong reason"
    )


def test_negative_variant_adding_the_auxiliary_fails_at_full_lambda():
    vols = _sample_volumes(ModelAdapter("aux_added"))
    stat_correct, p_correct = _ks_vs_reference(vols, u_star_correct)
    stat_wrong, p_wrong = _ks_vs_reference(vols, u_star_aux_added)
    print(f"aux variant (lambda=1): vs correct target D={stat_correct:.4f} "
          f"p={p_correct:.3g}; vs its own (wrong) target D={stat_wrong:.4f} p={p_wrong:.3g}")
    assert p_correct < 1e-6
    assert p_wrong > 5e-3
