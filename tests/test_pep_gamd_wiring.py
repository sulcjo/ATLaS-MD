"""Pep-GaMD wiring: energy helpers follow the integrator's definitions, plain
Langevin integrators never apply the auxiliary force, dispatch and CLI."""
import types

import numpy as np

from gareus.imports import import_openmm


def _three_group_system(openmm, unit):
    """Two particles, one harmonic bond per group 0/1/2/31 with distinct k so each
    group's energy is distinguishable: e_g = 0.5*k_g*(r-0.1)^2 at r=0.2."""
    s = openmm.System()
    s.addParticle(1.0); s.addParticle(1.0)
    ks = {0: 10.0, 1: 100.0, 2: 1000.0, 31: 10000.0}
    for g, k in ks.items():
        f = openmm.HarmonicBondForce(); f.addBond(0, 1, 0.1, k); f.setForceGroup(g); s.addForce(f)
    e = {g: 0.5 * k * 0.1 ** 2 for g, k in ks.items()}
    return s, e


def test_boost_target_energy_follows_stock_integrator_definition_all_groups():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    s, e = _three_group_system(openmm, unit)
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001 * unit.picoseconds), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0], [0.2, 0, 0]])
    stock = pep_gamd.total_energy_groups(types.SimpleNamespace())  # no TOTAL_ENERGY_* attributes -> bare energy
    assert abs(pep_gamd.boost_target_energy_kj(ctx, None, unit, total_groups=stock) - sum(e.values())) < 1e-9
    assert abs(pep_gamd.boost_target_energy_kj(ctx, 2, unit, total_groups=stock) - e[2]) < 1e-9


def test_boost_target_energy_follows_pep_gamd_definition_e0_minus_e1_plus_e2():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    s, e = _three_group_system(openmm, unit)
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001 * unit.picoseconds), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0], [0.2, 0, 0]])
    pep = pep_gamd.total_energy_groups(types.SimpleNamespace(TOTAL_ENERGY_PLUS_GROUPS={0, 2}, TOTAL_ENERGY_MINUS_GROUPS={1}))
    assert abs(pep_gamd.boost_target_energy_kj(ctx, None, unit, total_groups=pep) - (e[0] - e[1] + e[2])) < 1e-9
    assert abs(pep_gamd.boost_target_energy_kj(ctx, 2, unit, total_groups=pep) - e[2]) < 1e-9


def test_physical_potential_excludes_only_the_auxiliary_group_when_present():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    s, e = _three_group_system(openmm, unit)
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001 * unit.picoseconds), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0], [0.2, 0, 0]])
    # no auxiliary force in this system -> everything counts (umbrella included, as today)
    assert abs(pep_gamd.physical_potential_energy_kj(ctx, s, unit) - sum(e.values())) < 1e-9
    s.getForce(1).setName(pep_gamd.AUX_FORCE_NAME)
    ctx.reinitialize(preserveState=True)
    assert abs(pep_gamd.physical_potential_energy_kj(ctx, s, unit) - (sum(e.values()) - e[1])) < 1e-9


def test_cmd_integrator_excludes_aux_group_when_system_carries_it():
    from gareus import pep_gamd
    from gareus.production import make_cmd_integrator
    openmm, _app, unit = import_openmm()
    args = types.SimpleNamespace(temperature_k=300.0, friction_per_ps=1.0, timestep_fs=2.0, seed=1, run_mode="cmd")
    s, _e = _three_group_system(openmm, unit)
    integ, _info = make_cmd_integrator(openmm, args, unit, system=s)
    assert (integ.getIntegrationForceGroups() >> pep_gamd.AUX_NONBONDED_GROUP) & 1, "no aux present: nothing excluded"
    s.getForce(1).setName(pep_gamd.AUX_FORCE_NAME)
    integ2, _info = make_cmd_integrator(openmm, args, unit, system=s)
    mask = integ2.getIntegrationForceGroups() & 0xFFFFFFFF
    assert not (mask >> pep_gamd.AUX_NONBONDED_GROUP) & 1, "aux group must be excluded from a plain Langevin integrator"
    assert (mask >> 0) & 1 and (mask >> 2) & 1 and (mask >> 31) & 1


def test_make_gamd_integrator_dispatches_pep_gamd_and_partitions_the_system():
    from gareus import pep_gamd
    from gareus.production import make_gamd_integrator
    from pep_gamd_fixture import solvated_dipeptide, _fresh_system
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    args = types.SimpleNamespace(
        gamd_boost_type=pep_gamd.PEP_GAMD_BOOST_TYPE, gamd_cmd_prep_steps=2, gamd_cmd_steps=4,
        gamd_equil_prep_steps=2, gamd_equil_steps=4, gamd_production_steps=100, gamd_averaging_window=2,
        sigma0p_kcal_mol=6.0, sigma0d_kcal_mol=6.0, temperature_k=300.0, timestep_fs=2.0,
        friction_per_ps=1.0, seed=3, pep_gamd_peptide_atoms=list(fx["peptide"]),
    )
    n_before = system.getNumForces()
    integ, result = make_gamd_integrator(system, args, unit)
    assert type(integ).__name__ == "PepGaMDLowerDualIntegrator"
    assert result[2] is integ
    assert system.getNumForces() == n_before + 1
    assert pep_gamd.find_aux_force(system)[1] is not None
    integ2, _ = make_gamd_integrator(system, args, unit)  # idempotent on the system
    assert system.getNumForces() == n_before + 1
    ctx = openmm.Context(system, integ2, openmm.Platform.getPlatformByName("Reference"))  # the program compiles
    del ctx


def test_make_gamd_integrator_pep_gamd_requires_peptide_atoms():
    from gareus import pep_gamd
    from gareus.production import make_gamd_integrator
    from pep_gamd_fixture import _fresh_system
    _openmm, _app, unit = import_openmm()
    args = types.SimpleNamespace(
        gamd_boost_type=pep_gamd.PEP_GAMD_BOOST_TYPE, gamd_cmd_prep_steps=2, gamd_cmd_steps=4,
        gamd_equil_prep_steps=2, gamd_equil_steps=4, gamd_production_steps=100, gamd_averaging_window=2,
        sigma0p_kcal_mol=6.0, sigma0d_kcal_mol=6.0, temperature_k=300.0, timestep_fs=2.0, friction_per_ps=1.0, seed=3,
    )
    try:
        make_gamd_integrator(_fresh_system(), args, unit)
    except ValueError as exc:
        assert "pep_gamd_peptide_atoms" in str(exc)
    else:
        raise AssertionError("must refuse to build Pep-GaMD without the peptide atom set")


def test_cli_accepts_pep_gamd_boost_type_as_a_dual_boost():
    from gareus import pep_gamd
    from gareus.cli import build_gareus_parser, _SINGLE_BOOST_GAMD_TYPES
    args = build_gareus_parser().parse_args(["--seq", "GYDPETGTWG", "--gamd-boost-type", pep_gamd.PEP_GAMD_BOOST_TYPE])
    assert args.gamd_boost_type == pep_gamd.PEP_GAMD_BOOST_TYPE
    assert pep_gamd.PEP_GAMD_BOOST_TYPE not in _SINGLE_BOOST_GAMD_TYPES


def test_prepare_pep_gamd_args_records_peptide_atoms_from_topology():
    from gareus import pep_gamd
    from pep_gamd_fixture import solvated_dipeptide
    fx = solvated_dipeptide()
    args = types.SimpleNamespace(gamd_boost_type=pep_gamd.PEP_GAMD_BOOST_TYPE)
    pep_gamd.prepare_pep_gamd_args(args, fx["topology"])
    assert list(args.pep_gamd_peptide_atoms) == list(fx["peptide"])
    other = types.SimpleNamespace(gamd_boost_type="lower-dihedral")
    pep_gamd.prepare_pep_gamd_args(other, fx["topology"])
    assert not hasattr(other, "pep_gamd_peptide_atoms")


def test_total_energy_groups_derive_from_the_boost_type_not_the_stepping_integrator():
    """The cMD-kind recon steps a plain Langevin integrator on a partitioned system;
    the Total channel's definition is a property of the boost type, so it must come
    from args, never from whichever integrator happens to be stepping."""
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    plus, minus = pep_gamd.total_energy_groups_for_args(types.SimpleNamespace(gamd_boost_type=pep_gamd.PEP_GAMD_BOOST_TYPE))
    assert (set(plus), set(minus)) == ({0, 2}, {1})
    plus, minus = pep_gamd.total_energy_groups_for_args(types.SimpleNamespace(gamd_boost_type="lower-dual"))
    assert (set(plus), set(minus)) == (set(range(32)), set())
    s, e = _three_group_system(openmm, unit)
    plain = openmm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
    ctx = openmm.Context(s, plain, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0], [0.2, 0, 0]])
    groups = pep_gamd.total_energy_groups_for_args(types.SimpleNamespace(gamd_boost_type=pep_gamd.PEP_GAMD_BOOST_TYPE))
    assert abs(pep_gamd.boost_target_energy_kj(ctx, None, unit, total_groups=groups) - (e[0] - e[1] + e[2])) < 1e-9


def test_calibration_dispatch_accepts_the_pep_gamd_boost_type():
    """threshold_and_k0 dispatches lower/upper by string prefix; 'pep-gamd-lower-dual'
    must resolve to the lower-bound formula, identically to 'lower-dual'."""
    from gareus import pep_gamd
    from gareus.gamd_calibration import PooledEnvelope, compute_group_calibration
    env = PooledEnvelope(group="Total", vmax=120.0, vmin=-80.0, vavg=10.0, sigmav=12.0, n_total=1000, n_windows=4)
    ref = compute_group_calibration("lower-dual", env, 6.0 * 4.184)
    got = compute_group_calibration(pep_gamd.PEP_GAMD_BOOST_TYPE, env, 6.0 * 4.184)
    assert (got.k0, got.k, got.threshold_energy, got.boosted) == (ref.k0, ref.k, ref.threshold_energy, ref.boosted)
    assert got.boosted and 0.0 < got.k0 <= 1.0


def test_integrator_boosted_force_algebra_at_scaling_below_one():
    """0 K, no constraints, v0 = 0: x1 - x0 = dt*fscale*F_applied/m exactly, with
    F_applied = (F0 - F1)*FSF_T + F2*FSF_T*FSF_D + F1 + F31. Exercises FSF < 1,
    where the auxiliary/physical cancellation is nontrivial."""
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    s, _e = _three_group_system(openmm, unit)
    integ = pep_gamd.PepGaMDLowerDualIntegrator(
        2, dt=0.001 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4, nstlim=100, ntave=2,
        sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
        collision_rate=1.0 / unit.picoseconds, temperature=0.0 * unit.kelvin)
    ctx = openmm.Context(s, integ, openmm.Platform.getPlatformByName("Reference"))
    x0 = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    ctx.setPositions(x0)
    ctx.setVelocities(np.zeros((2, 3)))
    integ.step(1)  # gamd-openmm's first step of a fresh Context moves nothing; warm up, then re-seat
    ctx.setPositions(x0)
    ctx.setVelocities(np.zeros((2, 3)))
    F = {g: np.array(ctx.getState(getForces=True, groups={g}).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
         for g in (0, 1, 2, 31)}
    for k, v in {"stepCount": 50, "stage": 5, "k0_Total": 0.5, "k0_Dihedral": 0.5,
                 "Vmax_Total": 50.0, "Vmin_Total": -50.0, "threshold_energy_Total": 50.0,
                 "Vmax_Dihedral": 50.0, "Vmin_Dihedral": -50.0, "threshold_energy_Dihedral": 50.0}.items():
        integ.setGlobalVariableByName(k, v)
    integ.step(1)
    T = integ.getGlobalVariableByName("ForceScalingFactor_Total")
    D = integ.getGlobalVariableByName("ForceScalingFactor_Dihedral")
    assert 0.0 < T < 0.999 and 0.0 < D < 0.999, (T, D)
    fscale = integ.getGlobalVariableByName("fscale")
    dt = 0.001
    m = 1.0
    F_applied = (F[0] - F[1]) * T + F[2] * T * D + F[1] + F[31]
    x1 = np.array(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    assert np.max(np.abs(x1 - x0)) > 1e-6, "no motion: the step measured nothing"
    assert np.max(np.abs((x1 - x0) - dt * fscale * F_applied / m)) < 1e-9
