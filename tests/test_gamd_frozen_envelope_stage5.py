"""Frozen-envelope production must run in gamd-openmm stage 5, i.e. actually apply the boost.

chignolin_8 (2026-09-22): the swarm stage hands production a frozen envelope whose globals carry the
physics (Vmax/Vmin/threshold/k0/sigma0) but not the integrator's stepCount/stage. gamd-openmm's
StageIntegrator picks its stage from its own stepCount, so every replica started at step 0 and sat in
stage 2 (conventional MD collecting statistics) -- ForceScalingFactor exactly 1.0 on all 248 replicas
for the whole run, with Vmax drifting per replica. The pre-existing Pep-GaMD tests never saw it
because they set stepCount by hand.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from gareus.imports import import_openmm

from pep_gamd_fixture import solvated_dipeptide, _fresh_system  # noqa: E402

# A frozen envelope as the swarm delivers it: physics only, no stepCount/stage/windowCount.
# Thresholds sit far above any energy of the test system, so a live boost must give 0 < FSF < 1.
FROZEN = {
    "Vmax_Total": 1e6, "Vmin_Total": -1e6, "threshold_energy_Total": 1e6, "k0_Total": 1.0,
    "Vmax_Dihedral": 1e6, "Vmin_Dihedral": -1e6, "threshold_energy_Dihedral": 1e6, "k0_Dihedral": 1.0,
}
# Stage lengths small enough that an unseeded integrator is visibly still in stage 1/2 after 3 steps.
STAGES = dict(ntcmdprep=20, ntcmd=40, ntebprep=20, nteb=40, nstlim=40 + 40 + 20 + 20 + 1000, ntave=20)


def _integrator_and_context():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    integ = pep_gamd.PepGaMDLowerDualIntegrator(
        pep_gamd.DIHEDRAL_GROUP, bias_force_groups=pep_gamd.pep_gamd_bias_force_groups(system),
        dt=0.002 * unit.picoseconds, sigma0p=6.0 * unit.kilocalories_per_mole,
        sigma0d=6.0 * unit.kilocalories_per_mole, temperature=300.0 * unit.kelvin, **STAGES)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(fx["positions"])
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 11)
    return integ, ctx


def _copy(integ, values):
    from gareus.production import set_integrator_globals_from_dict
    copied, _skipped = set_integrator_globals_from_dict(integ, values)
    return copied


def test_unseeded_frozen_envelope_never_boosts__the_chignolin_8_defect():
    integ, _ctx = _integrator_and_context()
    _copy(integ, FROZEN)
    integ.step(3)
    assert integ.getGlobalVariableByName("stage") in (1.0, 2.0)
    assert integ.getGlobalVariableByName("ForceScalingFactor_Dihedral") == 1.0
    assert integ.getGlobalVariableByName("ForceScalingFactor_Total") == 1.0


def test_seeded_frozen_envelope_runs_in_stage_5_and_scales_forces():
    from gareus.production import seed_frozen_envelope_stage5
    integ, _ctx = _integrator_and_context()
    _copy(integ, FROZEN)
    seeded = seed_frozen_envelope_stage5(integ, FROZEN)
    assert seeded == integ.stage_5_start - 1
    integ.step(2)                                   # first step of a fresh gamd Context is inert
    assert integ.getGlobalVariableByName("stage") == 5.0
    for ch in ("Dihedral", "Total"):
        fsf = integ.getGlobalVariableByName(f"ForceScalingFactor_{ch}")
        assert 0.0 < fsf < 1.0, (ch, fsf)
    # stage 5 must not touch the frozen envelope
    for name in ("Vmax_Total", "Vmin_Total", "Vmax_Dihedral", "Vmin_Dihedral"):
        assert integ.getGlobalVariableByName(name) == FROZEN[name]


def test_seed_is_a_no_op_when_the_shared_globals_already_carry_the_stage_machine():
    # The old in-process shared setup exported stepCount (e.g. 460000 in chignolin_6); copying it is
    # what put those replicas in stage 5. Leave that path untouched.
    from gareus.production import seed_frozen_envelope_stage5
    integ, _ctx = _integrator_and_context()
    legacy = dict(FROZEN, stepCount=float(integ.stage_5_start + 5), stage=5.0)
    _copy(integ, legacy)
    assert seed_frozen_envelope_stage5(integ, legacy) is None
    assert integ.getGlobalVariableByName("stepCount") == float(integ.stage_5_start + 5)


def test_seed_ignores_integrators_without_a_stage_machine():
    from gareus.production import seed_frozen_envelope_stage5
    assert seed_frozen_envelope_stage5(types.SimpleNamespace(), FROZEN) is None


def test_stage5_verification_refuses_a_replica_outside_stage_5():
    from gareus.production import seed_frozen_envelope_stage5, verify_gamd_production_stage5
    integ, _ctx = _integrator_and_context()
    _copy(integ, FROZEN)
    with pytest.raises(RuntimeError, match="stage 5"):
        verify_gamd_production_stage5([integ], production_steps=100)
    seed_frozen_envelope_stage5(integ, FROZEN)
    verify_gamd_production_stage5([integ], production_steps=100)      # fits: passes


def test_stage5_verification_refuses_a_production_longer_than_the_stage_5_window():
    # stage 5 ends at nstlim; past it no stage block runs and the integrator stops updating atoms.
    from gareus.production import seed_frozen_envelope_stage5, verify_gamd_production_stage5
    integ, _ctx = _integrator_and_context()
    _copy(integ, FROZEN)
    seed_frozen_envelope_stage5(integ, FROZEN)
    budget = integ.stage_5_end - integ.stage_5_start + 1
    verify_gamd_production_stage5([integ], production_steps=budget)
    with pytest.raises(RuntimeError, match="stage-5 window"):
        verify_gamd_production_stage5([integ], production_steps=budget + 1)
