"""Pep-GaMD "internal" variant: boost only the peptide-internal energy.

    V_int = E3 + E2 = V_nb(pep-pep) + V_dihedral(pep)

There is no peptide-water term in the boosted channel, so no amount of boost can
scale down the forces that hold water off the peptide -- the solvent-collapse mode
that killed k0 = 1 at 4 fs with HMR for ``pep-gamd-lower-dual`` cannot exist here.
The partition is the mirror image of the water-only one: an auxiliary
peptide-only NonbondedForce (every non-peptide charge/epsilon zeroed, every
exception touching a non-peptide particle zeroed, PME pinned to the same
parameters) in force group 3.
"""
import math
import types

import numpy as np

from gareus.imports import import_openmm

from pep_gamd_fixture import (  # noqa: E402  (the tests directory is on sys.path when the suite runs)
    solvated_dipeptide, _fresh_system, _nonbonded, _energy,
)

INTERNAL = "pep-gamd-internal-lower-dual"


# --------------------------------------------------------------------------- partition

def test_internal_partition_adds_a_named_peptide_only_nonbonded_force_in_group_three():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    n_before = system.getNumForces()

    idx = pep_gamd.ensure_pep_gamd_internal_partition(system, fx["peptide"])

    assert system.getNumForces() == n_before + 1
    aux = system.getForce(idx)
    assert isinstance(aux, openmm.NonbondedForce)
    assert aux.getName() == pep_gamd.AUX_PEPTIDE_FORCE_NAME
    assert aux.getForceGroup() == pep_gamd.AUX_PEPTIDE_GROUP == 3
    # the internal variant never needs the water-only force
    assert all(f.getName() != pep_gamd.AUX_FORCE_NAME for f in system.getForces())

    physical = [f for f in _nonbonded(system, openmm) if f.getName() != pep_gamd.AUX_PEPTIDE_FORCE_NAME]
    assert len(physical) == 1
    physical = physical[0]
    assert physical.getForceGroup() == pep_gamd.PHYSICAL_NONBONDED_GROUP

    pep = set(fx["peptide"])
    for i in range(aux.getNumParticles()):
        q, sig, eps = aux.getParticleParameters(i)
        q0, sig0, eps0 = physical.getParticleParameters(i)
        if i in pep:
            assert q == q0 and sig == sig0 and eps == eps0, f"peptide atom {i} altered in the peptide-only aux"
        else:
            assert q._value == 0.0 and eps._value == 0.0, f"non-peptide atom {i} not zeroed in the peptide-only aux"

    assert aux.getNumExceptions() == physical.getNumExceptions()
    for k in range(aux.getNumExceptions()):
        a, b, qq, sig, eps = aux.getExceptionParameters(k)
        a0, b0, qq0, sig0, eps0 = physical.getExceptionParameters(k)
        assert (a, b) == (a0, b0)
        if a in pep and b in pep:
            assert qq == qq0 and eps == eps0
        else:
            assert qq._value == 0.0 and eps._value == 0.0, f"non-peptide exception {k} not zeroed"

    assert aux.getPMEParameters() == physical.getPMEParameters()
    assert aux.getPMEParameters()[0].value_in_unit(unit.nanometer ** -1) > 0.0, \
        "PME parameters must be pinned explicitly, not left to auto-choice"


def test_internal_partition_is_idempotent():
    from gareus import pep_gamd
    fx = solvated_dipeptide()
    system = _fresh_system()
    i1 = pep_gamd.ensure_pep_gamd_internal_partition(system, fx["peptide"])
    n = system.getNumForces()
    i2 = pep_gamd.ensure_pep_gamd_internal_partition(system, fx["peptide"])
    assert i1 == i2 and system.getNumForces() == n


def test_internal_partition_energy_is_invariant_when_the_water_is_displaced():
    """E3 carries no peptide-water term: move every water rigidly and it must not
    move at all, while the physical group 0 does.

    Reference platform, not CPU: the CPU platform evaluates PME in mixed precision and
    leaves ~2e-6 kJ/mol of summation noise on a ~400 kJ/mol E3, which is above the
    tolerance this invariance deserves. In double precision the invariance is exact.
    """
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_internal_partition(system, fx["peptide"])

    x0 = np.array(fx["positions"].value_in_unit(unit.nanometer), dtype=float)
    x1 = x0.copy()
    x1[np.array(sorted(fx["nonpeptide"]), dtype=int)] += 0.3

    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001 * unit.picoseconds),
                         openmm.Platform.getPlatformByName("Reference"))

    def _e(x, group):
        ctx.setPositions(x)
        return float(ctx.getState(getEnergy=True, groups={group}).getPotentialEnergy()
                     .value_in_unit(unit.kilojoule_per_mole))

    e0 = _e(x0, pep_gamd.AUX_PEPTIDE_GROUP)
    e1 = _e(x1, pep_gamd.AUX_PEPTIDE_GROUP)
    p0 = _e(x0, pep_gamd.PHYSICAL_NONBONDED_GROUP)
    p1 = _e(x1, pep_gamd.PHYSICAL_NONBONDED_GROUP)
    del ctx

    assert abs(e1 - e0) < 1e-6, f"peptide-internal energy moved with the water: {e0} -> {e1}"
    assert abs(p1 - p0) > 1.0, f"the displacement must actually change the physical energy: {p0} -> {p1}"


def test_peptide_only_aux_reproduces_an_independently_built_peptide_only_system():
    """The oracle: delete the water from the topology, build the peptide alone in the
    same box with the same PME parameters, and E3 must match it."""
    from gareus import pep_gamd
    from gareus.system_setup import create_system
    openmm, app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_internal_partition(system, fx["peptide"])
    e_aux = _energy(system, fx["positions"], {pep_gamd.AUX_PEPTIDE_GROUP}, openmm, unit)

    physical = [f for f in _nonbonded(system, openmm) if f.getName() != pep_gamd.AUX_PEPTIDE_FORCE_NAME][0]
    alpha, nx, ny, nz = physical.getPMEParameters()

    pep = set(fx["peptide"])
    m = app.Modeller(fx["topology"], fx["positions"])
    m.delete([r for r in fx["topology"].residues() if not any(int(a.index) in pep for a in r.atoms())])
    pep_only = create_system(app, unit, fx["ff"], m.topology, fx["args"], include_barostat=False)
    nb_p = [f for f in _nonbonded(pep_only, openmm)][0]
    nb_p.setPMEParameters(alpha, nx, ny, nz)
    for f in pep_only.getForces():
        f.setForceGroup(1)
    nb_p.setForceGroup(0)
    e_pep = _energy(pep_only, m.positions, {0}, openmm, unit)
    assert abs(e_aux - e_pep) < 0.05, f"aux {e_aux:.4f} vs independent peptide-only {e_pep:.4f} kJ/mol"


def test_nonbonded_energy_splits_into_water_plus_peptide_plus_cross():
    """E_nb(full) = E1 + E3 + E_cross, with the water-only and peptide-only halves
    taken from systems built independently of the auxiliary forces."""
    from gareus import pep_gamd
    from gareus.system_setup import create_system
    openmm, app, unit = import_openmm()
    fx = solvated_dipeptide()
    pep = set(fx["peptide"])

    system = _fresh_system()
    pep_gamd.ensure_pep_gamd_partition(system, fx["peptide"])
    pep_gamd.ensure_pep_gamd_internal_partition(system, fx["peptide"])
    physical = [f for f in _nonbonded(system, openmm)
                if f.getName() not in (pep_gamd.AUX_FORCE_NAME, pep_gamd.AUX_PEPTIDE_FORCE_NAME)][0]
    physical.setForceGroup(5)  # isolate the nonbonded term from bonds/angles in group 0
    alpha, nx, ny, nz = physical.getPMEParameters()
    e_full = _energy(system, fx["positions"], {5}, openmm, unit)
    e1 = _energy(system, fx["positions"], {pep_gamd.AUX_NONBONDED_GROUP}, openmm, unit)
    e3 = _energy(system, fx["positions"], {pep_gamd.AUX_PEPTIDE_GROUP}, openmm, unit)

    def _independent(keep_peptide: bool) -> float:
        m = app.Modeller(fx["topology"], fx["positions"])
        m.delete([r for r in fx["topology"].residues()
                  if any(int(a.index) in pep for a in r.atoms()) == (not keep_peptide)])
        sub = create_system(app, unit, fx["ff"], m.topology, fx["args"], include_barostat=False)
        nb = _nonbonded(sub, openmm)[0]
        nb.setPMEParameters(alpha, nx, ny, nz)
        for f in sub.getForces():
            f.setForceGroup(1)
        nb.setForceGroup(0)
        return _energy(sub, m.positions, {0}, openmm, unit)

    e_cross = e_full - _independent(False) - _independent(True)
    assert abs(e_full - (e1 + e3 + e_cross)) < 0.1, \
        f"E_nb {e_full:.4f} != E1 {e1:.4f} + E3 {e3:.4f} + E_cross {e_cross:.4f}"
    assert abs(e_cross) > 1.0, "the peptide-water cross term must be non-trivial for this check to bite"


# --------------------------------------------------------------------------- integrator

def _gamd_kwargs(unit, temperature_k=300.0):
    return dict(dt=0.002 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4, nstlim=100, ntave=2,
                sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
                collision_rate=1.0 / unit.picoseconds, temperature=temperature_k * unit.kelvin)


def _internally_partitioned_system(zero_aux: bool = False):
    from gareus import pep_gamd
    fx = solvated_dipeptide()
    system = _fresh_system()
    idx = pep_gamd.ensure_pep_gamd_internal_partition(system, fx["peptide"])
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


def test_internal_integrator_total_channel_reads_e3_plus_e2():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    system, fx = _internally_partitioned_system()
    integ = pep_gamd.PepGaMDInternalLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **_gamd_kwargs(unit))
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(fx["positions"])
    e3 = ctx.getState(getEnergy=True, groups={pep_gamd.AUX_PEPTIDE_GROUP}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    e2 = ctx.getState(getEnergy=True, groups={pep_gamd.DIHEDRAL_GROUP}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    v_int = pep_gamd.peptide_internal_energy_kj(ctx, unit)
    assert abs(v_int - (e3 + e2)) < 1e-3
    integ.step(1)  # StartingPotentialEnergy_* are computed from the pre-step positions
    assert abs(integ.getGlobalVariableByName("StartingPotentialEnergy_Total") - (e3 + e2)) < 1e-3
    assert abs(integ.getGlobalVariableByName("StartingPotentialEnergy_Dihedral") - e2) < 1e-3


def test_internal_integrator_cmd_stage_does_not_apply_the_peptide_only_force():
    """Same seed, same start: a system whose peptide-only auxiliary force is all-zero
    must give the same trajectory as the real one. If the aux were applied, the
    peptide-internal nonbonded term would be counted twice."""
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    sys_real, fx = _internally_partitioned_system(zero_aux=False)
    sys_zero, _ = _internally_partitioned_system(zero_aux=True)
    kw = _gamd_kwargs(unit)
    p_real = _one_step_positions(sys_real, pep_gamd.PepGaMDInternalLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **kw),
                                 fx["positions"], openmm, unit)
    p_zero = _one_step_positions(sys_zero, pep_gamd.PepGaMDInternalLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **kw),
                                 fx["positions"], openmm, unit)
    x0 = np.array(fx["positions"].value_in_unit(unit.nanometer), dtype=float)
    assert np.max(np.abs(p_real - x0)) > 1e-6, "the measured step did not move any atom; the comparison would be vacuous"
    assert np.max(np.abs(p_real - p_zero)) < 1e-9


def _four_group_system(openmm):
    """Two particles, one harmonic bond per group 0/2/3/31 with distinct k so each
    group's force is distinguishable. Group 3 stands in for the peptide-only aux."""
    s = openmm.System()
    s.addParticle(1.0); s.addParticle(1.0)
    ks = {0: 10.0, 2: 1000.0, 3: 300.0, 31: 10000.0}
    for g, k in ks.items():
        f = openmm.HarmonicBondForce(); f.addBond(0, 1, 0.1, k); f.setForceGroup(g); s.addForce(f)
    return s, {g: 0.5 * k * 0.1 ** 2 for g, k in ks.items()}


def test_internal_integrator_boosted_force_algebra_at_scaling_below_one():
    """0 K, no constraints, v0 = 0: x1 - x0 = dt*fscale*F_applied/m exactly, with
    F_applied = (F - F3) + F3*FSF_T + F2*(FSF_T*FSF_D - 1), F the force over every
    integrated group (0, 2, 3, 31) minus the auxiliary group 3."""
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    s, _e = _four_group_system(openmm)
    integ = pep_gamd.PepGaMDInternalLowerDualIntegrator(
        pep_gamd.DIHEDRAL_GROUP, dt=0.001 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4,
        nstlim=100, ntave=2, sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
        collision_rate=1.0 / unit.picoseconds, temperature=0.0 * unit.kelvin)
    ctx = openmm.Context(s, integ, openmm.Platform.getPlatformByName("Reference"))
    x0 = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    ctx.setPositions(x0)
    ctx.setVelocities(np.zeros((2, 3)))
    integ.step(1)  # warm up: gamd-openmm's first step of a fresh Context moves nothing
    ctx.setPositions(x0)
    ctx.setVelocities(np.zeros((2, 3)))
    F = {g: np.array(ctx.getState(getForces=True, groups={g}).getForces(asNumpy=True)
                     .value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
         for g in (0, 2, 3, 31)}
    for k, v in {"stepCount": 50, "stage": 5, "k0_Total": 0.5, "k0_Dihedral": 0.5,
                 "Vmax_Total": 50.0, "Vmin_Total": -50.0, "threshold_energy_Total": 50.0,
                 "Vmax_Dihedral": 50.0, "Vmin_Dihedral": -50.0, "threshold_energy_Dihedral": 50.0}.items():
        integ.setGlobalVariableByName(k, v)
    integ.step(1)
    T = integ.getGlobalVariableByName("ForceScalingFactor_Total")
    D = integ.getGlobalVariableByName("ForceScalingFactor_Dihedral")
    assert 0.0 < T < 0.999 and 0.0 < D < 0.999, (T, D)
    fscale = integ.getGlobalVariableByName("fscale")
    f_phys = F[0] + F[2] + F[31]
    F_applied = (f_phys - F[3]) + F[3] * T + F[2] * (T * D - 1.0)
    x1 = np.array(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    assert np.max(np.abs(x1 - x0)) > 1e-6, "no motion: the step measured nothing"
    assert np.max(np.abs((x1 - x0) - 0.001 * fscale * F_applied / 1.0)) < 1e-9


def test_internal_integrator_unboosted_step_applies_the_physical_force_only():
    """FSF_T = FSF_D = 1 must collapse the boosted update onto the cMD update."""
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    s, _e = _four_group_system(openmm)
    integ = pep_gamd.PepGaMDInternalLowerDualIntegrator(
        pep_gamd.DIHEDRAL_GROUP, dt=0.001 * unit.picoseconds, ntcmdprep=2, ntcmd=4, ntebprep=2, nteb=4,
        nstlim=100, ntave=2, sigma0p=6.0 * unit.kilocalories_per_mole, sigma0d=6.0 * unit.kilocalories_per_mole,
        collision_rate=1.0 / unit.picoseconds, temperature=0.0 * unit.kelvin)
    ctx = openmm.Context(s, integ, openmm.Platform.getPlatformByName("Reference"))
    x0 = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    ctx.setPositions(x0); ctx.setVelocities(np.zeros((2, 3)))
    integ.step(1)
    ctx.setPositions(x0); ctx.setVelocities(np.zeros((2, 3)))
    F = {g: np.array(ctx.getState(getForces=True, groups={g}).getForces(asNumpy=True)
                     .value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
         for g in (0, 2, 3, 31)}
    for k, v in {"stepCount": 50, "stage": 5, "k0_Total": 0.0, "k0_Dihedral": 0.0,
                 "Vmax_Total": 1e6, "Vmin_Total": -1e6, "threshold_energy_Total": 1e6,
                 "Vmax_Dihedral": 1e6, "Vmin_Dihedral": -1e6, "threshold_energy_Dihedral": 1e6}.items():
        integ.setGlobalVariableByName(k, v)
    integ.step(1)
    fscale = integ.getGlobalVariableByName("fscale")
    x1 = np.array(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    f_phys = F[0] + F[2] + F[31]
    assert np.max(np.abs(x1 - x0)) > 1e-6
    assert np.max(np.abs((x1 - x0) - 0.001 * fscale * f_phys / 1.0)) < 1e-9, \
        "at unit scaling factors the applied force must be the physical force, group 3 excluded"


# --------------------------------------------------------------------------- wiring

def test_pep_gamd_variant_names_both_boost_types():
    from gareus import pep_gamd
    assert pep_gamd.pep_gamd_variant(types.SimpleNamespace(gamd_boost_type=pep_gamd.PEP_GAMD_BOOST_TYPE)) == "essential"
    assert pep_gamd.pep_gamd_variant(types.SimpleNamespace(gamd_boost_type=INTERNAL)) == "internal"
    assert pep_gamd.pep_gamd_variant(types.SimpleNamespace(gamd_boost_type="lower-dual")) is None
    assert pep_gamd.pep_gamd_variant(types.SimpleNamespace()) is None
    assert pep_gamd.PEP_GAMD_INTERNAL_BOOST_TYPE == INTERNAL
    assert pep_gamd.is_pep_gamd(types.SimpleNamespace(gamd_boost_type=INTERNAL))
    assert pep_gamd.ladder_supports_boost_type(types.SimpleNamespace(gamd_boost_type=INTERNAL))
    assert INTERNAL in pep_gamd.LADDER_BOOST_TYPES


def test_energy_group_helpers_for_the_internal_variant():
    from gareus import pep_gamd
    args = types.SimpleNamespace(gamd_boost_type=INTERNAL)
    plus, minus = pep_gamd.total_energy_groups_for_args(args)
    assert (set(plus), set(minus)) == ({pep_gamd.AUX_PEPTIDE_GROUP, pep_gamd.DIHEDRAL_GROUP}, set())
    assert set(pep_gamd.physical_energy_groups_for_args(args)) == set(range(32)) - {pep_gamd.AUX_PEPTIDE_GROUP}
    # the essential variant is untouched
    ess = types.SimpleNamespace(gamd_boost_type=pep_gamd.PEP_GAMD_BOOST_TYPE)
    assert (set(pep_gamd.total_energy_groups_for_args(ess)[0]), set(pep_gamd.total_energy_groups_for_args(ess)[1])) == ({0, 2}, {1})
    assert set(pep_gamd.physical_energy_groups_for_args(ess)) == set(range(32)) - {pep_gamd.AUX_NONBONDED_GROUP}


def test_boost_target_energy_reads_e3_plus_e2_for_the_internal_variant():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    s, e = _four_group_system(openmm)
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001 * unit.picoseconds),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0], [0.2, 0, 0]])
    groups = pep_gamd.total_energy_groups_for_args(types.SimpleNamespace(gamd_boost_type=INTERNAL))
    assert abs(pep_gamd.boost_target_energy_kj(ctx, None, unit, total_groups=groups) - (e[3] + e[2])) < 1e-9
    assert abs(pep_gamd.boost_target_energy_kj(ctx, 2, unit, total_groups=groups) - e[2]) < 1e-9


class _FakeState:
    def __init__(self, energy):
        self._energy = energy

    def getPotentialEnergy(self):
        return self._energy


class _FakeContext:
    """Per-force-group energies, no OpenMM System behind them."""

    def __init__(self, per_group, unit):
        self._per_group = dict(per_group)
        self._unit = unit
        self.reads = []

    def getState(self, getEnergy=False, groups=None, **_kw):
        self.reads.append(frozenset(int(g) for g in groups))
        total = sum(self._per_group.get(int(g), 0.0) for g in groups)
        return _FakeState(total * self._unit.kilojoule_per_mole)


def test_fetch_v_pep_v_dih_returns_e3_plus_e2_for_the_internal_variant():
    from gareus.production import _fetch_v_pep_v_dih
    _openmm, _app, unit = import_openmm()
    per_group = {0: -1000.0, 1: -900.0, 2: 17.0, 3: -25.0}
    env = types.SimpleNamespace(has_total=True)

    ctx = _FakeContext(per_group, unit)
    v_tot, v_dih = _fetch_v_pep_v_dih(ctx, env, unit, types.SimpleNamespace(gamd_boost_type=INTERNAL))
    assert abs(v_dih - 17.0) < 1e-9
    assert abs(v_tot - (-25.0 + 17.0)) < 1e-9

    ctx2 = _FakeContext(per_group, unit)
    v_pep, v_dih2 = _fetch_v_pep_v_dih(ctx2, env, unit, types.SimpleNamespace(gamd_boost_type="pep-gamd-lower-dual"))
    assert abs(v_dih2 - 17.0) < 1e-9
    assert abs(v_pep - (-1000.0 + 900.0 + 17.0)) < 1e-9

    ctx3 = _FakeContext(per_group, unit)
    both_nan = _fetch_v_pep_v_dih(ctx3, None, unit, types.SimpleNamespace(gamd_boost_type=INTERNAL))
    assert all(math.isnan(x) for x in both_nan) and ctx3.reads == [], \
        "no ladder -> (nan, nan) without touching the Context"


def test_make_cmd_integrator_excludes_the_peptide_only_auxiliary_group():
    from gareus import pep_gamd
    from gareus.production import make_cmd_integrator
    openmm, _app, unit = import_openmm()
    args = types.SimpleNamespace(temperature_k=300.0, friction_per_ps=1.0, timestep_fs=2.0, seed=1, run_mode="cmd")
    s, _e = _four_group_system(openmm)
    integ, _info = make_cmd_integrator(openmm, args, unit, system=s)
    assert (integ.getIntegrationForceGroups() >> pep_gamd.AUX_PEPTIDE_GROUP) & 1, "no aux present: nothing excluded"
    # forces were added in dict order 0, 2, 3, 31 -> index 2 sits in group 3
    assert s.getForce(2).getForceGroup() == pep_gamd.AUX_PEPTIDE_GROUP
    s.getForce(2).setName(pep_gamd.AUX_PEPTIDE_FORCE_NAME)
    integ2, _info = make_cmd_integrator(openmm, args, unit, system=s)
    mask = integ2.getIntegrationForceGroups() & 0xFFFFFFFF
    assert not (mask >> pep_gamd.AUX_PEPTIDE_GROUP) & 1, \
        "the peptide-only aux group must be excluded from a plain Langevin integrator"
    assert (mask >> 0) & 1 and (mask >> 2) & 1 and (mask >> 31) & 1


def test_physical_potential_energy_excludes_every_auxiliary_group_present():
    from gareus import pep_gamd
    openmm, _app, unit = import_openmm()
    s, e = _four_group_system(openmm)
    ctx = openmm.Context(s, openmm.VerletIntegrator(0.001 * unit.picoseconds),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0, 0, 0], [0.2, 0, 0]])
    assert abs(pep_gamd.physical_potential_energy_kj(ctx, s, unit) - sum(e.values())) < 1e-9
    s.getForce(2).setName(pep_gamd.AUX_PEPTIDE_FORCE_NAME)
    ctx.reinitialize(preserveState=True)
    assert abs(pep_gamd.physical_potential_energy_kj(ctx, s, unit) - (sum(e.values()) - e[3])) < 1e-9


def test_make_gamd_integrator_dispatches_the_internal_variant_and_partitions_the_system():
    from gareus import pep_gamd
    from gareus.production import make_gamd_integrator
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    args = types.SimpleNamespace(
        gamd_boost_type=INTERNAL, gamd_cmd_prep_steps=2, gamd_cmd_steps=4,
        gamd_equil_prep_steps=2, gamd_equil_steps=4, gamd_production_steps=100, gamd_averaging_window=2,
        sigma0p_kcal_mol=6.0, sigma0d_kcal_mol=6.0, temperature_k=300.0, timestep_fs=2.0,
        friction_per_ps=1.0, seed=3, pep_gamd_peptide_atoms=list(fx["peptide"]),
    )
    n_before = system.getNumForces()
    integ, result = make_gamd_integrator(system, args, unit)
    assert type(integ).__name__ == "PepGaMDInternalLowerDualIntegrator"
    assert result[0] == pep_gamd.AUX_PEPTIDE_GROUP and result[2] is integ
    assert system.getNumForces() == n_before + 1
    assert pep_gamd.find_named_force(system, pep_gamd.AUX_PEPTIDE_FORCE_NAME)[1] is not None
    assert pep_gamd.find_named_force(system, pep_gamd.AUX_FORCE_NAME)[1] is None
    integ2, _ = make_gamd_integrator(system, args, unit)  # idempotent on the system
    assert system.getNumForces() == n_before + 1
    ctx = openmm.Context(system, integ2, openmm.Platform.getPlatformByName("Reference"))  # the program compiles
    del ctx


def test_calibration_dispatch_accepts_the_internal_boost_type():
    """threshold_and_k0 dispatches lower/upper by string prefix; the internal variant
    must resolve to the lower-bound formula, identically to 'lower-dual'."""
    from gareus.gamd_calibration import PooledEnvelope, compute_group_calibration
    env = PooledEnvelope(group="Total", vmax=120.0, vmin=-80.0, vavg=10.0, sigmav=12.0, n_total=1000, n_windows=4)
    ref = compute_group_calibration("lower-dual", env, 6.0 * 4.184)
    got = compute_group_calibration(INTERNAL, env, 6.0 * 4.184)
    assert (got.k0, got.k, got.threshold_energy, got.boosted) == (ref.k0, ref.k, ref.threshold_energy, ref.boosted)
    assert got.boosted and 0.0 < got.k0 <= 1.0


def test_cli_accepts_the_internal_boost_type_as_a_dual_boost():
    from gareus.cli import build_gareus_parser, _SINGLE_BOOST_GAMD_TYPES, _validate_gamd_args
    args = build_gareus_parser().parse_args(["--seq", "GYDPETGTWG", "--gamd-boost-type", INTERNAL])
    assert args.gamd_boost_type == INTERNAL
    assert INTERNAL not in _SINGLE_BOOST_GAMD_TYPES
    args.sigma0d_kcal_mol = 8.0
    _validate_gamd_args(args)  # dual boost: uses sigma0d, must not warn or raise


def test_provenance_records_the_pep_gamd_variant():
    from gareus.provenance import _method_settings
    args = types.SimpleNamespace(gamd_boost_type=INTERNAL)
    assert _method_settings(args)["pep_gamd_variant"] == "internal"
    assert _method_settings(types.SimpleNamespace(gamd_boost_type="pep-gamd-lower-dual"))["pep_gamd_variant"] == "essential"
    assert _method_settings(types.SimpleNamespace(gamd_boost_type="lower-dual"))["pep_gamd_variant"] is None


def test_helptext_documents_the_internal_variant():
    from gareus import helptext
    text = helptext._METHOD_ENCYCLOPEDIA
    assert INTERNAL in text, "the Pep-GaMD helptext section must name the internal variant"
    assert "desolvation" in text.lower(), "helptext must say which barriers are NOT accelerated"


def test_swarm_member_refuses_the_internal_variant():
    """The swarm stage fits the ESSENTIAL envelope (V_pep = E0 - E1 + E2). Handing it
    the internal boost type would silently calibrate the wrong channel, so it must
    refuse before it builds the water-only partition."""
    import pathlib
    import tempfile
    from gareus.swarm.members import run_member
    openmm, app, unit = import_openmm()
    fx = solvated_dipeptide()
    xml = openmm.XmlSerializer.serialize(_fresh_system())
    args = types.SimpleNamespace(gamd_boost_type=INTERNAL, temperature_k=300.0,
                                 friction_per_ps=1.0, timestep_fs=4.0)
    try:
        run_member(args, {"velocity_seed": 1}, pathlib.Path(tempfile.mkdtemp()),
                   openmm=openmm, app=app, unit=unit, topology=fx["topology"],
                   base_system_xml=xml, equil_state=None, conformer={"seed_id": "s"},
                   platform=openmm.Platform.getPlatformByName("Reference"),
                   props={}, contact_pairs=[])
    except ValueError as exc:
        assert "E3 + E2" in str(exc) and INTERNAL in str(exc)
    else:
        raise AssertionError("the swarm stage must refuse the peptide-internal variant")


def test_internal_integrator_carries_a_total_channel_the_lambda_ladder_can_scale():
    """The rung mechanism is set_replica_lambda(k0_c = lambda * k0max_c). If the internal
    integrator had no k0_Total global, every ladder run with this variant would silently
    boost the dihedral channel only, and PepGamdEnvelope would come back has_total=False."""
    from gareus import pep_gamd
    _openmm, _app, unit = import_openmm()
    integ = pep_gamd.PepGaMDInternalLowerDualIntegrator(pep_gamd.DIHEDRAL_GROUP, **_gamd_kwargs(unit))
    names = {str(integ.getGlobalVariableName(i)) for i in range(int(integ.getNumGlobalVariables()))}
    for required in ("k0_Total", "k0_Dihedral", "Vmax_Total", "Vmin_Total",
                     "threshold_energy_Total", "ForceScalingFactor_Total"):
        assert required in names, f"the internal integrator has no {required} global"

    pep_gamd.set_replica_lambda(integ, 0.25, {"Total": 0.8, "Dihedral": 0.6})
    assert abs(integ.getGlobalVariableByName("k0_Total") - 0.2) < 1e-12
    assert abs(integ.getGlobalVariableByName("k0_Dihedral") - 0.15) < 1e-12

    for k, v in {"Vmax_Total": 120.0, "Vmin_Total": -80.0, "threshold_energy_Total": 120.0,
                 "Vmax_Dihedral": 40.0, "Vmin_Dihedral": -10.0, "threshold_energy_Dihedral": 40.0,
                 "k0_Total": 0.8, "k0_Dihedral": 0.6}.items():
        integ.setGlobalVariableByName(k, v)
    g = {n: integ.getGlobalVariableByName(n) for n in names}
    env = pep_gamd.PepGamdEnvelope.from_integrator_globals(g)
    assert env.has_total and env.k0max_total == 0.8 and env.k0max_dih == 0.6
    assert pep_gamd.k0max_from_globals(g) == {"Total": 0.8, "Dihedral": 0.6}
    # a non-zero Total boost must actually come out of the closed form
    b = pep_gamd.pep_gamd_boost_kj(0.0, 0.0, 1.0, env)
    b_dih_only = pep_gamd.pep_gamd_boost_kj(0.0, 0.0, 1.0,
                                            pep_gamd.PepGamdEnvelope(0.0, 0.0, 0.0, 0.0, env.vmax_dih,
                                                                     env.vmin_dih, env.threshold_dih,
                                                                     env.k0max_dih, has_total=False))
    assert b > b_dih_only > 0.0, (b, b_dih_only)
