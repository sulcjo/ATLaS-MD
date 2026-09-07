"""Pep-GaMD: boost only the peptide essential potential.

V_pep = V_bonded(pep) + V_nb(pep-pep) + V_nb(pep-water); water-water excluded.
The exact partition is by subtraction: an auxiliary water-only NonbondedForce
(peptide charges/epsilon and peptide exceptions zeroed, identical PME parameters)
in its own force group, so V_pep = energy0 - energy1 + energy2.
"""
import numpy as np

from gareus.imports import import_openmm

from pep_gamd_fixture import (  # noqa: E402  (the tests directory is on sys.path when the suite runs)
    _CACHE, _fake_args, solvated_dipeptide, _fresh_system, _nonbonded, _energy,
)


# --------------------------------------------------------------------------- partition

def test_partition_adds_named_water_only_nonbonded_force_in_aux_group():
    from gareus import pep_gamd
    openmm, _app, _unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    n_before = system.getNumForces()

    idx = pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])

    assert system.getNumForces() == n_before + 1
    aux = system.getForce(idx)
    assert isinstance(aux, openmm.NonbondedForce)
    assert aux.getName() == pep_gamd.AUX_FORCE_NAME
    assert aux.getForceGroup() == pep_gamd.AUX_NONBONDED_GROUP
    physical = [f for f in _nonbonded(system, openmm) if f.getName() != pep_gamd.AUX_FORCE_NAME]
    assert len(physical) == 1
    physical = physical[0]
    assert physical.getForceGroup() == pep_gamd.PHYSICAL_NONBONDED_GROUP
    assert aux.getNumParticles() == physical.getNumParticles()

    pep = set(fx["peptide"])
    for i in range(aux.getNumParticles()):
        q, sig, eps = aux.getParticleParameters(i)
        q0, sig0, eps0 = physical.getParticleParameters(i)
        if i in pep:
            assert q._value == 0.0 and eps._value == 0.0, f"peptide atom {i} not zeroed in aux"
        else:
            assert q == q0 and sig == sig0 and eps == eps0, f"water atom {i} altered in aux"

    assert aux.getNumExceptions() == physical.getNumExceptions()
    for k in range(aux.getNumExceptions()):
        a, b, qq, sig, eps = aux.getExceptionParameters(k)
        a0, b0, qq0, sig0, eps0 = physical.getExceptionParameters(k)
        assert (a, b) == (a0, b0)
        if a in pep or b in pep:
            assert qq._value == 0.0 and eps._value == 0.0, f"peptide exception {k} not zeroed in aux"
        else:
            assert qq == qq0 and eps == eps0

    assert aux.getPMEParameters() == physical.getPMEParameters()
    assert aux.getPMEParameters()[0].value_in_unit(_unit.nanometer ** -1) > 0.0, "PME parameters must be pinned explicitly, not left to auto-choice"


def test_partition_assigns_physical_groups_and_leaves_custom_forces_out_of_them():
    from gareus import pep_gamd
    openmm, _app, _unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    umbrella = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    umbrella.addGlobalParameter("k", 0.0); umbrella.addGlobalParameter("r0", 0.5)
    umbrella.addBond(fx["peptide"][0], fx["peptide"][-1], [])
    umbrella.setForceGroup(31)
    system.addForce(umbrella)

    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])

    groups = {f.__class__.__name__: f.getForceGroup() for f in system.getForces() if f.getName() != pep_gamd.AUX_FORCE_NAME}
    assert groups["PeriodicTorsionForce"] == pep_gamd.DIHEDRAL_GROUP
    assert groups["HarmonicBondForce"] == pep_gamd.PHYSICAL_NONBONDED_GROUP
    assert groups["HarmonicAngleForce"] == pep_gamd.PHYSICAL_NONBONDED_GROUP
    assert groups["NonbondedForce"] == pep_gamd.PHYSICAL_NONBONDED_GROUP
    assert groups["CustomBondForce"] == 31, "umbrella must keep its own group"


def test_partition_is_idempotent():
    from gareus import pep_gamd
    fx = solvated_dipeptide()
    system = _fresh_system()
    i1 = pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    n = system.getNumForces()
    i2 = pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    assert i1 == i2 and system.getNumForces() == n


def test_partition_refuses_a_custom_force_sitting_in_a_physical_group():
    from gareus import pep_gamd
    openmm, _app, _unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    stray = openmm.CustomBondForce("0.5*k*(r-r0)^2")
    stray.addGlobalParameter("k", 0.0); stray.addGlobalParameter("r0", 0.5)
    stray.addBond(fx["peptide"][0], fx["peptide"][-1], [])
    stray.setForceGroup(0)  # would be silently boosted
    system.addForce(stray)
    try:
        pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    except ValueError as exc:
        assert "CustomBondForce" in str(exc) and "group 0" in str(exc)
    else:
        raise AssertionError("a Custom* force in a physical group must be rejected")


def test_aux_force_reproduces_an_independently_built_water_only_system():
    """The oracle: delete the peptide from the topology, build water-only with the
    same PME parameters, and the auxiliary force's energy must match it."""
    from gareus import pep_gamd
    from gareus.system_setup import create_system
    openmm, app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    e_aux = _energy(system, fx["positions"], {pep_gamd.AUX_NONBONDED_GROUP}, openmm, unit)

    physical = [f for f in _nonbonded(system, openmm) if f.getName() != pep_gamd.AUX_FORCE_NAME][0]
    alpha, nx, ny, nz = physical.getPMEParameters()

    m = app.Modeller(fx["topology"], fx["positions"])
    m.delete([r for r in fx["topology"].residues() if any(int(a.index) in set(fx["peptide"]) for a in r.atoms())])
    water_only = create_system(app, unit, fx["ff"], m.topology, fx["args"], include_barostat=False)
    nb_w = _nonbonded(water_only, openmm)[0]
    nb_w.setPMEParameters(alpha, nx, ny, nz)
    for f in water_only.getForces():
        f.setForceGroup(0)
    e_water = _energy(water_only, m.positions, {0}, openmm, unit)
    # water-only system also carries CMMotionRemover (0 energy); no bonded terms for rigid water
    assert abs(e_aux - e_water) < 0.05, f"aux {e_aux:.4f} vs independent water-only {e_water:.4f} kJ/mol"


def test_pinned_pme_parameters_match_openmm_auto_choice():
    """The formula must reproduce what OpenMM picks on its own for the same force."""
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    nb = _nonbonded(system, openmm)[0]
    assert nb.getPMEParameters()[0].value_in_unit(unit.nanometer ** -1) == 0.0, "fixture must start unpinned"
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001 * unit.picoseconds), openmm.Platform.getPlatformByName("Reference"))
    auto_alpha, ax, ay, az = nb.getPMEParametersInContext(ctx)
    del ctx
    box = [[float(v[i].value_in_unit(unit.nanometer)) for i in range(3)] for v in system.getDefaultPeriodicBoxVectors()]
    alpha, nx, ny, nz = pep_gamd.pme_parameters_from_tolerance(
        float(nb.getCutoffDistance().value_in_unit(unit.nanometer)), float(nb.getEwaldErrorTolerance()), box)
    auto_alpha = auto_alpha.value_in_unit(unit.nanometer ** -1) if hasattr(auto_alpha, "value_in_unit") else float(auto_alpha)
    assert abs(alpha - auto_alpha) < 1e-9
    assert (nx, ny, nz) == (ax, ay, az)


def test_peptide_essential_energy_is_e0_minus_e1_plus_e2():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(fx["positions"])
    e = {g: ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) for g in (0, 1, 2)}
    v_pep = pep_gamd.peptide_essential_energy_kj(ctx, unit)
    assert abs(v_pep - (e[0] - e[1] + e[2])) < 1e-2  # CPU platform evaluates groups in mixed precision
    # sanity: the peptide-only quantity is small next to the water-dominated total
    e_total = ctx.getState(getEnergy=True, groups={0, 2}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    assert abs(v_pep) < 0.2 * abs(e_total), (v_pep, e_total)


# --------------------------------------------------------------------------- integrator

def _gamd_kwargs(unit):
    return dict(dt=0.002 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4, nstlim=100, ntave=2,
                sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
                collision_rate=1.0 / unit.picoseconds, temperature=300.0 * unit.kelvin)


def _partitioned_system(zero_aux: bool = False):
    from gareus import pep_gamd
    openmm, _app, _unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    idx = pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    if zero_aux:
        aux = system.getForce(idx)
        for i in range(aux.getNumParticles()):
            _q, sig, _e = aux.getParticleParameters(i)
            aux.setParticleParameters(i, 0.0, sig, 0.0)
        for k in range(aux.getNumExceptions()):
            a, b, _qq, sig, _e = aux.getExceptionParameters(k)
            aux.setExceptionParameters(k, a, b, 0.0, sig, 0.0)
    return system, fx


def _one_step_positions(system, integ, positions, openmm, unit, seed=7, stage_globals=None):
    """Positions after ONE integrated step. gamd-openmm's very first step of a fresh
    Context moves nothing, so warm up one step, re-seat the start state, then step."""
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, seed)
    integ.step(1)
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, seed)
    if stage_globals:
        for k, v in stage_globals.items():
            integ.setGlobalVariableByName(k, v)
    integ.step(1)
    pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    return np.array(pos, dtype=float)


def _assert_moved(p_after, positions, unit):
    x0 = np.array(positions.value_in_unit(unit.nanometer), dtype=float)
    assert np.max(np.abs(p_after - x0)) > 1e-6, "the measured step did not move any atom; the comparison would be vacuous"


def test_integrator_total_channel_reads_peptide_essential_energy_not_bare_energy():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    system, fx = _partitioned_system()
    integ = pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **_gamd_kwargs(unit))
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(fx["positions"])
    v_pep = pep_gamd.peptide_essential_energy_kj(ctx, unit)
    e_dih = ctx.getState(getEnergy=True, groups={pep_gamd.DIHEDRAL_GROUP}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    integ.step(1)  # StartingPotentialEnergy_* are computed from the pre-step positions
    assert abs(integ.getGlobalVariableByName("StartingPotentialEnergy_Total") - v_pep) < 1e-3
    assert abs(integ.getGlobalVariableByName("StartingPotentialEnergy_Dihedral") - e_dih) < 1e-3


def test_integrator_cmd_stage_does_not_apply_the_auxiliary_force():
    """Same seed, same start: a system whose auxiliary force is all-zero must give
    the same trajectory as the real one. If the aux were applied, water-water
    would be counted twice and the positions would differ."""
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    sys_real, fx = _partitioned_system(zero_aux=False)
    sys_zero, _ = _partitioned_system(zero_aux=True)
    kw = _gamd_kwargs(unit)
    p_real = _one_step_positions(sys_real, pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **kw), fx["positions"], openmm, unit)
    p_zero = _one_step_positions(sys_zero, pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **kw), fx["positions"], openmm, unit)
    _assert_moved(p_real, fx["positions"], unit)
    assert np.max(np.abs(p_real - p_zero)) < 1e-9


def test_integrator_boost_stage_with_zero_k0_reduces_to_physical_forces():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    sys_real, fx = _partitioned_system(zero_aux=False)
    sys_zero, _ = _partitioned_system(zero_aux=True)
    kw = _gamd_kwargs(unit)
    # production stage, k0 = 0 -> BoostPotential = 0 -> ForceScalingFactor = 1 for both channels
    # stage is recomputed from stepCount every step; jumping past the stage-2/4 end
    # triggers keeps the k0 we set from being recalibrated.
    prod = {"stepCount": 50, "stage": 5, "k0_Total": 0.0, "k0_Dihedral": 0.0,
            "Vmax_Total": 1e6, "Vmin_Total": -1e6, "threshold_energy_Total": 1e6,
            "Vmax_Dihedral": 1e6, "Vmin_Dihedral": -1e6, "threshold_energy_Dihedral": 1e6}
    p_real = _one_step_positions(sys_real, pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **kw), fx["positions"], openmm, unit, stage_globals=prod)
    p_zero = _one_step_positions(sys_zero, pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **kw), fx["positions"], openmm, unit, stage_globals=prod)
    _assert_moved(p_real, fx["positions"], unit)
    assert np.max(np.abs(p_real - p_zero)) < 1e-9


def test_integrator_statistics_are_blind_to_the_umbrella():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    a, b = fx["peptide"][0], fx["peptide"][-1]
    pos = np.array(fx["positions"].value_in_unit(unit.nanometer), dtype=float)
    r_ab = float(np.linalg.norm(pos[a] - pos[b]))
    vals = []
    for k in (0.0, 5.0e4):
        system = _fresh_system()
        umb = openmm.CustomBondForce("0.5*umb_k*(r-umb_r0)^2")
        umb.addGlobalParameter("umb_k", k); umb.addGlobalParameter("umb_r0", r_ab + 0.1)
        umb.addBond(a, b, [])
        umb.setForceGroup(31)
        system.addForce(umb)
        pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
        integ = pep_gamd.PepGaMDLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **_gamd_kwargs(unit))
        ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions(fx["positions"])
        e_umb = ctx.getState(getEnergy=True, groups={31}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        integ.step(1)
        vals.append((e_umb, integ.getGlobalVariableByName("StartingPotentialEnergy_Total")))
    (e0, spe0), (e1, spe1) = vals
    assert e1 > 100.0 and abs(e0) < 1e-9, f"the stiff umbrella must actually carry energy: {e0}, {e1}"
    assert abs(spe0 - spe1) < 1e-3, f"Total channel moved with the umbrella: {spe0} vs {spe1}"
