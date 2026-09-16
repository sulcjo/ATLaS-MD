"""Boost/force agreement gate for the Pep-GaMD NPT target adapter (spec section 10).

Oracles, all independent of the adapter implementation under test:
  1. the integrator's OWN BoostPotential_* globals after a step (gamd-openmm's
     kernels evaluate them from the pre-step configuration);
  2. gareus.pep_gamd._channel_boost / pep_gamd_boost_kj -- the pre-existing
     closed form the MBAR lambda-ladder already relies on;
  3. direct context group-energy reads for the physical/auxiliary/bias split;
  4. finite-difference gradients of the adapter's own boost vs the analytic
     force-scaling expression built from the integrator's ForceScalingFactor
     globals and per-group context forces.

Covers lambda 0 / intermediate / 1, active and inactive channels, dependent
ordering, stage transitions, and a detector for double application of lambda.
"""
import types

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
import openmm.unit as unit  # noqa: E402

from gareus.imports import import_openmm  # noqa: E402
from gareus import pep_gamd  # noqa: E402

from pep_gamd_fixture import solvated_dipeptide, _fresh_system  # noqa: E402


def _gamd_kwargs(unit):
    return dict(dt=0.002 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2,
                nteb=4, nstlim=100, ntave=2,
                sigma0p=6.0 * unit.kilocalories_per_mole,
                sigma0d=6.0 * unit.kilocalories_per_mole,
                collision_rate=1.0 / unit.picoseconds,
                temperature=300.0 * unit.kelvin)


def _args(**over):
    base = dict(run_mode="gamd", gamd_boost_type="pep-gamd-lower-dual")
    base.update(over)
    return types.SimpleNamespace(**base)


def _group_energy(context, groups):
    if not groups:
        return 0.0
    e = context.getState(getEnergy=True, groups=set(groups)).getPotentialEnergy()
    return float(e.value_in_unit(unit.kilojoule_per_mole))


def _group_forces(context, groups):
    f = context.getState(getForces=True, groups=set(groups)).getForces(asNumpy=True)
    return np.array(f.value_in_unit(unit.kilojoule_per_mole / unit.nanometer), dtype=float)


def _channel_globals(integ, names=("k0", "threshold_energy", "Vmax", "Vmin"),
                     channels=("Dihedral", "Total")):
    return {
        f"{name}_{ch}": integ.getGlobalVariableByName(f"{name}_{ch}")
        for ch in channels for name in names
    }


def _make_pep_context(umbrella=False):
    fx = solvated_dipeptide()
    system = _fresh_system()
    if umbrella:
        f = openmm.CustomBondForce("0.5*k*(r-r0)^2")
        f.addGlobalParameter("k", 100.0)
        f.addGlobalParameter("r0", 0.15)
        f.addBond(fx["peptide"][0], fx["peptide"][-1], [])
        f.setForceGroup(31)
        system.addForce(f)
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    integ = pep_gamd.PepGaMDLowerDualIntegrator(
        pep_gamd.DIHEDRAL_GROUP, bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(system), **_gamd_kwargs(unit))
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(fx["positions"])
    integ.setRandomNumberSeed(7)
    integ.step(1)  # fresh-Context warm-up (gamd-openmm's first step moves nothing)
    return system, integ, ctx, fx


# ------------------------------------------------- adapter dispatch per context
#
# The adapter must be chosen from the integrator actually driving THIS context,
# not from the run-wide args. The multi-window GaMD reconnaissance path
# (production.py: integrator_kind != "gamd" -> make_cmd_integrator) drives its
# windows with a plain LangevinMiddleIntegrator while the run-wide args still
# say run_mode=hmr-gamd / boost_type=pep-gamd-lower-dual. Dispatching on the
# args there handed those contexts the Pep-GaMD adapter, which demands a 'stage'
# global the plain integrator does not have, and killed chignolin_7 job 2390041
# with "integrator LangevinMiddleIntegrator exposes no 'stage' global".


def _plain_integrator():
    return openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)


def test_cmd_run_mode_reaches_the_conventional_adapter_with_zero_boost():
    """The documented route for an unboosted context: run_mode=cmd. U* must match
    the propagated dynamics -- no boost, auxiliary force excluded (that integrator
    excludes the aux group too, see production.make_cmd_integrator)."""
    system, _integ, ctx, _fx = _make_pep_context()
    integ = _plain_integrator()
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args(run_mode="cmd"))
    assert adapter.adapter_id == "conventional"
    breakdown = adapter.evaluate(ctx, adapter.snapshot(ctx, integ))
    assert breakdown.boost_kj_mol == 0.0
    assert breakdown.effective_kj_mol == pytest.approx(
        breakdown.physical_kj_mol + breakdown.bias_kj_mol)


def test_stage_integrator_context_still_gets_the_pep_gamd_adapter():
    """The fix must not downgrade a genuinely boosted context."""
    system, integ, _ctx, _fx = _make_pep_context()
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    assert adapter.adapter_id != "conventional"
    assert isinstance(adapter, pep_gamd.PepGamdLowerDualNptTargetAdapter)


def test_unsupported_boost_type_still_raises_even_with_a_plain_integrator():
    """The per-context fallback must not swallow an unvalidated boost type."""
    with pytest.raises(ValueError, match="no validated NPT target adapter"):
        pep_gamd.make_npt_target_adapter(
            _fresh_system(), _plain_integrator(), _args(gamd_boost_type="upper-dual"))


def _arm_channels(ctx, integ, *, k0_total, k0_dih, step_count=50,
                  theta_offset_dih=10.0, theta_offset_total=5.0,
                  vmax_pad=40.0, vmin_pad=20.0):
    """Put the integrator into stage 5 with active, self-consistent channel
    parameters around the CURRENT configuration's energies."""
    e_dih = _group_energy(ctx, {pep_gamd.DIHEDRAL_GROUP})
    e_aux = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    e0 = _group_energy(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    v_pep = e0 - e_aux + e_dih
    for ch, e, off in (("Dihedral", e_dih, theta_offset_dih),
                       ("Total", v_pep, theta_offset_total)):
        integ.setGlobalVariableByName(f"Vmax_{ch}", e + vmax_pad)
        integ.setGlobalVariableByName(f"Vmin_{ch}", e - vmin_pad)
        integ.setGlobalVariableByName(f"threshold_energy_{ch}", e + off)
    integ.setGlobalVariableByName("k0_Dihedral", k0_dih)
    integ.setGlobalVariableByName("k0_Total", k0_total)
    integ.setGlobalVariableByName("stepCount", step_count)
    integ.step(1)  # stage-5 step: BoostPotential_* now reflect fx["positions"]
    ctx.setPositions(fx_positions(integ))
    return e_dih, v_pep


_fx_cache = {}


def fx_positions(integ):
    """The configuration the armed integrator's BoostPotential globals describe:
    the positions this test armed the channels around (the pre-step ones)."""
    key = id(integ)
    if key not in _fx_cache:
        _fx_cache[key] = solvated_dipeptide()["positions"]
    return _fx_cache[key]


def _adapter_boost(ctx, adapter, integ):
    snap = adapter.snapshot(ctx, integ)
    return adapter.evaluate(ctx, snap)


# --------------------------------------------------------------------- agreement

@pytest.mark.parametrize("k0_total,k0_dih", [(1.0, 1.0), (0.5, 0.7), (0.28, 0.2)])
def test_adapter_boost_matches_the_integrators_own_boost_potentials(k0_total, k0_dih):
    system, integ, ctx, fx = _make_pep_context()
    _arm_channels(ctx, integ, k0_total=k0_total, k0_dih=k0_dih)
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    breakdown = _adapter_boost(ctx, adapter, integ)

    b_dih = integ.getGlobalVariableByName("BoostPotential_Dihedral")
    b_tot = integ.getGlobalVariableByName("BoostPotential_Total")
    assert b_dih > 0.0 and b_tot > 0.0, "channels must be active for this comparison"
    assert breakdown.boost_kj_mol == pytest.approx(b_dih + b_tot, rel=1e-9, abs=1e-9), (
        f"adapter boost {breakdown.boost_kj_mol} != integrator boost {b_dih + b_tot}"
    )


@pytest.mark.parametrize("k0_total,k0_dih", [(1.0, 1.0), (0.5, 0.7)])
def test_adapter_boost_matches_the_ladder_closed_form(k0_total, k0_dih):
    """Second, independent oracle: gareus.pep_gamd.pep_gamd_boost_kj, the closed
    form the MBAR lambda-ladder already uses (evaluated at lambda=1 because the
    integrator's k0 globals already carry lambda)."""
    system, integ, ctx, fx = _make_pep_context()
    _arm_channels(ctx, integ, k0_total=k0_total, k0_dih=k0_dih)
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    breakdown = _adapter_boost(ctx, adapter, integ)

    e_dih = _group_energy(ctx, {pep_gamd.DIHEDRAL_GROUP})
    e_aux = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    e0 = _group_energy(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    v_pep = e0 - e_aux + e_dih
    g = _channel_globals(integ)
    env = pep_gamd.PepGamdEnvelope(
        vmax_total=g["Vmax_Total"], vmin_total=g["Vmin_Total"],
        threshold_total=g["threshold_energy_Total"], k0max_total=g["k0_Total"],
        vmax_dih=g["Vmax_Dihedral"], vmin_dih=g["Vmin_Dihedral"],
        threshold_dih=g["threshold_energy_Dihedral"], k0max_dih=g["k0_Dihedral"])
    oracle = pep_gamd.pep_gamd_boost_kj(v_pep, e_dih, 1.0, env)
    assert breakdown.boost_kj_mol == pytest.approx(oracle, rel=1e-9, abs=1e-9)


def test_lambda_zero_gives_exactly_zero_boost():
    system, integ, ctx, fx = _make_pep_context()
    _arm_channels(ctx, integ, k0_total=0.0, k0_dih=0.0)
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    breakdown = _adapter_boost(ctx, adapter, integ)
    assert breakdown.boost_kj_mol == 0.0
    assert integ.getGlobalVariableByName("BoostPotential_Dihedral") == 0.0
    assert integ.getGlobalVariableByName("BoostPotential_Total") == 0.0


def test_double_application_of_lambda_is_detectable():
    """The integrator's k0 already carries lambda (k0 = lambda * k0max). An
    adapter that applied lambda a second time would evaluate the channel at
    k0^2/k0max. This must NOT match the integrator's own boost."""
    lam, k0max_d, k0max_t = 0.4, 1.0, 0.8
    system, integ, ctx, fx = _make_pep_context()
    _arm_channels(ctx, integ, k0_total=lam * k0max_t, k0_dih=lam * k0max_d)
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    breakdown = _adapter_boost(ctx, adapter, integ)

    b_dih = integ.getGlobalVariableByName("BoostPotential_Dihedral")
    b_tot = integ.getGlobalVariableByName("BoostPotential_Total")
    assert breakdown.boost_kj_mol == pytest.approx(b_dih + b_tot, rel=1e-9, abs=1e-9)

    # what a double-lambda implementation would produce: k0 -> lambda^2 * k0max
    e_dih = _group_energy(ctx, {pep_gamd.DIHEDRAL_GROUP})
    e_aux = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    e0 = _group_energy(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    v_pep = e0 - e_aux + e_dih
    g = _channel_globals(integ)
    env = pep_gamd.PepGamdEnvelope(
        vmax_total=g["Vmax_Total"], vmin_total=g["Vmin_Total"],
        threshold_total=g["threshold_energy_Total"], k0max_total=lam * lam * k0max_t,
        vmax_dih=g["Vmax_Dihedral"], vmin_dih=g["Vmin_Dihedral"],
        threshold_dih=g["threshold_energy_Dihedral"], k0max_dih=lam * lam * k0max_d)
    double = pep_gamd.pep_gamd_boost_kj(v_pep, e_dih, 1.0, env)
    assert double != pytest.approx(breakdown.boost_kj_mol, rel=1e-3), (
        "the double-lambda variant is indistinguishable; this test has no teeth"
    )
    assert abs(double - breakdown.boost_kj_mol) > 1e-6


def test_dependent_ordering_total_channel_sees_dihedral_boost():
    """The Total channel input is V_pep + b_d (stage_integrator
    _add_dihedral_boost_to_total_energy), not bare V_pep. Make b_d large enough
    that the two orderings differ and check only the correct one matches."""
    system, integ, ctx, fx = _make_pep_context()
    # large dihedral boost: k0=1, threshold far above the energy
    _arm_channels(ctx, integ, k0_total=1.0, k0_dih=1.0, theta_offset_dih=60.0,
                  theta_offset_total=50.0)
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    breakdown = _adapter_boost(ctx, adapter, integ)

    b_dih = integ.getGlobalVariableByName("BoostPotential_Dihedral")
    b_tot = integ.getGlobalVariableByName("BoostPotential_Total")
    assert b_dih > 5.0, "dihedral boost must be large enough to shift the Total input"
    e_dih = _group_energy(ctx, {pep_gamd.DIHEDRAL_GROUP})
    e_aux = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    e0 = _group_energy(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    v_pep = e0 - e_aux + e_dih
    g = _channel_globals(integ)
    # correct ordering: total input includes b_d
    env = pep_gamd.PepGamdEnvelope(
        vmax_total=g["Vmax_Total"], vmin_total=g["Vmin_Total"],
        threshold_total=g["threshold_energy_Total"], k0max_total=g["k0_Total"],
        vmax_dih=g["Vmax_Dihedral"], vmin_dih=g["Vmin_Dihedral"],
        threshold_dih=g["threshold_energy_Dihedral"], k0max_dih=g["k0_Dihedral"])
    correct = pep_gamd.pep_gamd_boost_kj(v_pep, e_dih, 1.0, env)
    # wrong ordering: total channel sees bare V_pep
    wrong = pep_gamd._channel_boost(
        v_pep, g["threshold_energy_Total"], g["Vmax_Total"], g["Vmin_Total"], g["k0_Total"]
    ) + pep_gamd._channel_boost(
        e_dih, g["threshold_energy_Dihedral"], g["Vmax_Dihedral"], g["Vmin_Dihedral"], g["k0_Dihedral"]
    )
    assert breakdown.boost_kj_mol == pytest.approx(correct, rel=1e-9, abs=1e-9)
    assert wrong != pytest.approx(correct, rel=1e-3), "orderings indistinguishable; no teeth"
    assert b_tot > 0.0


def test_inactive_channels_give_zero_boost():
    system, integ, ctx, fx = _make_pep_context()
    e_dih = _group_energy(ctx, {pep_gamd.DIHEDRAL_GROUP})
    e_aux = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    e0 = _group_energy(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    v_pep = e0 - e_aux + e_dih
    # threshold far BELOW the channel energies -> the (b + E < theta) guard zeroes it
    for ch, e in (("Dihedral", e_dih), ("Total", v_pep)):
        integ.setGlobalVariableByName(f"Vmax_{ch}", e + 40.0)
        integ.setGlobalVariableByName(f"Vmin_{ch}", e - 20.0)
        integ.setGlobalVariableByName(f"threshold_energy_{ch}", e - 50.0)
    integ.setGlobalVariableByName("k0_Dihedral", 1.0)
    integ.setGlobalVariableByName("k0_Total", 1.0)
    integ.setGlobalVariableByName("stepCount", 50)
    integ.step(1)
    ctx.setPositions(fx_positions(integ))
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    breakdown = _adapter_boost(ctx, adapter, integ)
    assert breakdown.boost_kj_mol == 0.0
    assert integ.getGlobalVariableByName("BoostPotential_Dihedral") == 0.0
    assert integ.getGlobalVariableByName("BoostPotential_Total") == 0.0


def test_small_range_channel_gives_zero_boost():
    system, integ, ctx, fx = _make_pep_context()
    e_dih = _group_energy(ctx, {pep_gamd.DIHEDRAL_GROUP})
    e_aux = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    e0 = _group_energy(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    v_pep = e0 - e_aux + e_dih
    # Vmax == Vmin -> |Vmax - Vmin| fails the boost_threshold guard
    for ch, e in (("Dihedral", e_dih), ("Total", v_pep)):
        integ.setGlobalVariableByName(f"Vmax_{ch}", e)
        integ.setGlobalVariableByName(f"Vmin_{ch}", e)
        integ.setGlobalVariableByName(f"threshold_energy_{ch}", e + 10.0)
    integ.setGlobalVariableByName("k0_Dihedral", 1.0)
    integ.setGlobalVariableByName("k0_Total", 1.0)
    integ.setGlobalVariableByName("stepCount", 50)
    integ.step(1)
    ctx.setPositions(fx_positions(integ))
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    assert _adapter_boost(ctx, adapter, integ).boost_kj_mol == 0.0


# --------------------------------------------------------------------- stages

@pytest.mark.parametrize("step_count,expected_stage", [
    (1, 1),    # conventional MD prep
    (3, 2),    # conventional MD statistics
    (5, 3),    # GaMD pre-equilibration (parameters frozen within the step)
    (50, 5),   # production
])
def test_stage_transitions_frozen_parameter_stages(step_count, expected_stage):
    """Stages 1-2 carry zero boost; in stages 3 and 5 the parameters are frozen
    during the step, so the integrator's own BoostPotential globals are the
    oracle for what the adapter must return."""
    system, integ, ctx, fx = _make_pep_context()
    _arm_channels(ctx, integ, k0_total=0.6, k0_dih=0.8, step_count=step_count)
    assert integ.getGlobalVariableByName("stage") == expected_stage
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    breakdown = _adapter_boost(ctx, adapter, integ)
    b_sum = (integ.getGlobalVariableByName("BoostPotential_Dihedral")
             + integ.getGlobalVariableByName("BoostPotential_Total"))
    if expected_stage >= 3:
        assert b_sum > 0.0
        assert breakdown.boost_kj_mol == pytest.approx(b_sum, rel=1e-9, abs=1e-9)
    else:
        assert breakdown.boost_kj_mol == 0.0, "cMD stages must carry zero boost"


def test_stage_four_adapter_uses_post_recalibration_parameters():
    """Stage 4 recalibrates k0/threshold DURING the step, after BoostPotential
    was computed. A volume move fires BETWEEN steps, so the adapter must use the
    post-recalibration globals -- exactly what the next integration step will
    apply -- not the step's own (stale) BoostPotential."""
    system, integ, ctx, fx = _make_pep_context()
    e_dih = _group_energy(ctx, {pep_gamd.DIHEDRAL_GROUP})
    e_aux = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    e0 = _group_energy(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    v_pep = e0 - e_aux + e_dih
    for ch, e in (("Dihedral", e_dih), ("Total", v_pep)):
        integ.setGlobalVariableByName(f"Vmax_{ch}", e + 40.0)
        integ.setGlobalVariableByName(f"Vmin_{ch}", e - 20.0)
        integ.setGlobalVariableByName(f"Vavg_{ch}", e)
        integ.setGlobalVariableByName(f"sigmaV_{ch}", 25.0)
        integ.setGlobalVariableByName(f"k0_{ch}", 0.0)  # must be recomputed by stage 4
    integ.setGlobalVariableByName("windowCount", 0)  # 0->1 stays below ntave: no boundary reset
    integ.setGlobalVariableByName("stepCount", 7)    # stage 4
    integ.step(1)
    assert integ.getGlobalVariableByName("stage") == 4
    ctx.setPositions(fx["positions"])

    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    breakdown = _adapter_boost(ctx, adapter, integ)

    # oracle: the closed form on the post-recalibration globals
    g = _channel_globals(integ)
    assert g["k0_Dihedral"] > 0.0 and g["k0_Total"] > 0.0, "stage 4 must have recalibrated k0"
    e_dih2 = _group_energy(ctx, {pep_gamd.DIHEDRAL_GROUP})
    e_aux2 = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    e02 = _group_energy(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    v_pep2 = e02 - e_aux2 + e_dih2
    env = pep_gamd.PepGamdEnvelope(
        vmax_total=g["Vmax_Total"], vmin_total=g["Vmin_Total"],
        threshold_total=g["threshold_energy_Total"], k0max_total=g["k0_Total"],
        vmax_dih=g["Vmax_Dihedral"], vmin_dih=g["Vmin_Dihedral"],
        threshold_dih=g["threshold_energy_Dihedral"], k0max_dih=g["k0_Dihedral"])
    oracle = pep_gamd.pep_gamd_boost_kj(v_pep2, e_dih2, 1.0, env)
    assert oracle > 0.0
    assert breakdown.boost_kj_mol == pytest.approx(oracle, rel=1e-9, abs=1e-9)


# --------------------------------------------------------------------- energy split

def test_breakdown_separates_physical_auxiliary_bias_and_boost():
    system, integ, ctx, fx = _make_pep_context(umbrella=True)
    _arm_channels(ctx, integ, k0_total=0.5, k0_dih=0.7)
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    breakdown = _adapter_boost(ctx, adapter, integ)

    e0 = _group_energy(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    e1 = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    e2 = _group_energy(ctx, {pep_gamd.DIHEDRAL_GROUP})
    e_bias = _group_energy(ctx, {31})
    full = _group_energy(ctx, set(range(32)))
    assert e_bias != 0.0, "umbrella must actually contribute energy"
    assert breakdown.physical_kj_mol == pytest.approx(e0 + e2, rel=1e-9, abs=1e-9)
    assert breakdown.auxiliary_kj_mol == pytest.approx(e1, rel=1e-9, abs=1e-9)
    assert breakdown.bias_kj_mol == pytest.approx(e_bias, rel=1e-9, abs=1e-9)
    assert breakdown.boost_kj_mol > 0.0
    assert breakdown.effective_kj_mol == pytest.approx(
        breakdown.physical_kj_mol + breakdown.bias_kj_mol + breakdown.boost_kj_mol,
        rel=1e-12, abs=1e-12)
    # the raw context potential must NOT equal physical or effective (aux inside)
    assert full != pytest.approx(breakdown.physical_kj_mol, rel=1e-6)
    assert full != pytest.approx(breakdown.effective_kj_mol, rel=1e-6)


def test_adapter_refuses_a_bias_force_inside_the_boosted_groups():
    system, integ, ctx, fx = _make_pep_context()
    stray = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    stray.addGlobalParameter("k", 1.0)
    stray.addGlobalParameter("r0", 0.3)
    stray.addBond(fx["peptide"][0], fx["peptide"][-1], [])
    stray.setForceGroup(0)
    system.addForce(stray)
    with pytest.raises(ValueError, match="group 0"):
        pep_gamd.make_npt_target_adapter(system, integ, _args())


# --------------------------------------------------------------------- forces

def test_finite_difference_boost_force_matches_scaling_factor_expression():
    """-grad(Delta) from finite differences of the adapter's boost must equal the
    force expression the integrator actually propagates:
    (FSF_T - 1)(f0 - f1) + (FSF_T*FSF_D - 1) f2, with FSF read from the
    integrator's own globals and f0/f1/f2 from context group forces."""
    system, integ, ctx, fx = _make_pep_context()
    _arm_channels(ctx, integ, k0_total=0.5, k0_dih=0.7)
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    snap = adapter.snapshot(ctx, integ)
    positions = np.array(
        ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer),
        dtype=float)

    fsf_t = integ.getGlobalVariableByName("ForceScalingFactor_Total")
    fsf_d = integ.getGlobalVariableByName("ForceScalingFactor_Dihedral")
    f0 = _group_forces(ctx, {pep_gamd.PHYSICAL_NONBONDED_GROUP})
    f1 = _group_forces(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    f2 = _group_forces(ctx, {pep_gamd.DIHEDRAL_GROUP})
    expected = (fsf_t - 1.0) * (f0 - f1) + (fsf_t * fsf_d - 1.0) * f2

    h = 2e-5
    probe_atoms = list(fx["peptide"])[:6]
    for i in probe_atoms:
        for axis in range(3):
            eps = np.zeros(3); eps[axis] = h
            ctx.setPositions(positions + np.where(np.arange(len(positions))[:, None] == i, eps, 0.0))
            bp = adapter.evaluate(ctx, snap).boost_kj_mol
            ctx.setPositions(positions + np.where(np.arange(len(positions))[:, None] == i, -eps, 0.0))
            bm = adapter.evaluate(ctx, snap).boost_kj_mol
            fd_force = -(bp - bm) / (2.0 * h)
            assert abs(fd_force - expected[i, axis]) < 5e-3 * max(1.0, abs(expected[i, axis])), (
                f"atom {i} axis {axis}: FD {fd_force} vs integrator expression "
                f"{expected[i, axis]}"
            )
    ctx.setPositions(positions)


# --------------------------------------------------------------------- dispatch

def test_dispatch_pep_gamd_lower_dual():
    system, integ, ctx, fx = _make_pep_context()
    adapter = pep_gamd.make_npt_target_adapter(system, integ, _args())
    assert adapter.adapter_id == "pep-gamd-lower-dual"


def test_dispatch_conventional_md_gets_zero_boost_adapter():
    system, integ, ctx, fx = _make_pep_context()
    adapter = pep_gamd.make_npt_target_adapter(
        system, integ, _args(run_mode="cmd", gamd_boost_type=""))
    assert adapter.adapter_id == "conventional"
    snap = adapter.snapshot(ctx, integ)
    breakdown = adapter.evaluate(ctx, snap)
    assert breakdown.boost_kj_mol == 0.0
    # the auxiliary force is excluded from the effective energy on the cMD path
    e_aux = _group_energy(ctx, {pep_gamd.AUX_NONBONDED_GROUP})
    full = _group_energy(ctx, set(range(32)))
    assert breakdown.effective_kj_mol == pytest.approx(full - e_aux, rel=1e-9, abs=1e-9)


def test_dispatch_unsupported_boost_type_fails():
    system, integ, ctx, fx = _make_pep_context()
    with pytest.raises(ValueError, match="upper-dual"):
        pep_gamd.make_npt_target_adapter(system, integ, _args(gamd_boost_type="upper-dual"))


def test_dispatch_pep_adapter_requires_the_pep_integrator():
    system, integ, ctx, fx = _make_pep_context()
    plain = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                             0.002 * unit.picoseconds)
    with pytest.raises(ValueError, match="integrator"):
        pep_gamd.make_npt_target_adapter(system, plain, _args())


# --------------------------------------------------------------------- stock lower-dihedral

def _make_stock_context(umbrella=False):
    from gareus.imports import import_gamd_factory
    fx = solvated_dipeptide()
    system = _fresh_system()
    factory = import_gamd_factory()()
    result = factory.get_integrator(
        "lower-dihedral", system, 300.0 * unit.kelvin, 0.002 * unit.picoseconds,
        2, 4, 2, 4, 100, 2)
    integ = result[2]
    integ.setRandomNumberSeed(7)
    if umbrella:
        f = openmm.CustomBondForce("0.5*k*(r-r0)^2")
        f.addGlobalParameter("k", 100.0)
        f.addGlobalParameter("r0", 0.15)
        f.addBond(fx["peptide"][0], fx["peptide"][-1], [])
        f.setForceGroup(31)
        system.addForce(f)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(fx["positions"])
    integ.step(1)
    return system, integ, ctx, fx


def test_stock_lower_dihedral_adapter_matches_integrator_boost():
    system, integ, ctx, fx = _make_stock_context()
    adapter = pep_gamd.make_npt_target_adapter(
        system, integ, _args(gamd_boost_type="lower-dihedral"))
    assert adapter.adapter_id == "lower-dihedral"

    e_dih = _group_energy(ctx, {2})
    integ.setGlobalVariableByName("Vmax_Dihedral", e_dih + 40.0)
    integ.setGlobalVariableByName("Vmin_Dihedral", e_dih - 20.0)
    integ.setGlobalVariableByName("threshold_energy_Dihedral", e_dih + 10.0)
    integ.setGlobalVariableByName("k0_Dihedral", 0.8)
    integ.setGlobalVariableByName("stepCount", 50)
    integ.step(1)
    ctx.setPositions(fx["positions"])

    breakdown = _adapter_boost(ctx, adapter, integ)
    b_dih = integ.getGlobalVariableByName("BoostPotential_Dihedral")
    assert b_dih > 0.0
    assert breakdown.boost_kj_mol == pytest.approx(b_dih, rel=1e-9, abs=1e-9)
    # no Total channel on this path
    names = {integ.getGlobalVariableName(i) for i in range(integ.getNumGlobalVariables())}
    assert "k0_Total" not in names
    # physical+bias covers every group (stock layout: everything in 0 and 2)
    full = _group_energy(ctx, set(range(32)))
    assert breakdown.effective_kj_mol == pytest.approx(full + b_dih, rel=1e-9, abs=1e-9)


def test_stock_lower_dihedral_adapter_reports_bias_groups():
    system, integ, ctx, fx = _make_stock_context(umbrella=True)
    adapter = pep_gamd.make_npt_target_adapter(
        system, integ, _args(gamd_boost_type="lower-dihedral"))
    snap = adapter.snapshot(ctx, integ)
    breakdown = adapter.evaluate(ctx, snap)
    e_bias = _group_energy(ctx, {31})
    assert e_bias != 0.0
    assert breakdown.bias_kj_mol == pytest.approx(e_bias, rel=1e-9, abs=1e-9)
    full = _group_energy(ctx, set(range(32)))
    assert breakdown.effective_kj_mol == pytest.approx(full, rel=1e-9, abs=1e-9)
