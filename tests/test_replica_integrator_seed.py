"""Every REUS replica (and every GaMD multi-window recon window) must get its own
Langevin random stream.

Before this fix ``make_production_integrator`` seeded every replica's integrator
with ``args.seed``, so all replicas drew the same random-force sequence. Replicas
that also share start positions, velocities and restraint centre then run as
near-copies, and in every other case the stochastic forces are correlated across
windows. The oracle here is OpenMM itself: the seed each integrator reports, and
the trajectories two contexts produce on the deterministic Reference platform.
"""
import ast
import types
from pathlib import Path

import numpy as np

from gareus.imports import import_openmm


def _cmd_args(seed=2026):
    return types.SimpleNamespace(temperature_k=300.0, friction_per_ps=1.0, timestep_fs=1.0,
                                 seed=seed, run_mode="cmd")


def _particle_system(openmm):
    """Two free argon-like particles joined by one stiff bond: cheap, and the
    Langevin noise dominates its motion over a few dozen steps."""
    s = openmm.System()
    s.addParticle(40.0); s.addParticle(40.0)
    f = openmm.HarmonicBondForce(); f.addBond(0, 1, 0.3, 1000.0); s.addForce(f)
    return s


def _run(openmm, unit, integrator, n_steps=50):
    system = _particle_system(openmm)
    ctx = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(0.3, 0, 0)] * unit.nanometer)
    ctx.setVelocities([openmm.Vec3(0, 0, 0)] * 2 * unit.nanometer / unit.picosecond)
    integrator.step(n_steps)
    pos = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    del ctx
    return np.asarray(pos)


def test_default_offset_keeps_the_historical_seed():
    from gareus.production import make_cmd_integrator, make_production_integrator
    openmm, _app, unit = import_openmm()
    integ, _ = make_cmd_integrator(openmm, _cmd_args(), unit)
    assert integ.getRandomNumberSeed() == 2026
    integ, _ = make_production_integrator(openmm, _particle_system(openmm), _cmd_args(), unit)
    assert integ.getRandomNumberSeed() == 2026


def test_replica_offsets_give_distinct_seeds():
    from gareus.production import make_production_integrator, replica_integrator_seed_offset
    openmm, _app, unit = import_openmm()
    seeds = [
        make_production_integrator(openmm, _particle_system(openmm), _cmd_args(), unit,
                                   seed_offset=replica_integrator_seed_offset(i))[0].getRandomNumberSeed()
        for i in range(8)
    ]
    assert len(set(seeds)) == 8
    assert 2026 not in seeds  # replica 0 does not reuse the single-context seed


def test_seed_zero_still_means_openmm_picks_a_unique_seed():
    """OpenMM treats seed 0 as "choose a unique seed per Context"; an offset must
    not turn that opt-out into a fixed, shared seed."""
    from gareus.production import make_cmd_integrator, replica_integrator_seed_offset
    openmm, _app, unit = import_openmm()
    integ, _ = make_cmd_integrator(openmm, _cmd_args(seed=0), unit, seed_offset=replica_integrator_seed_offset(3))
    assert integ.getRandomNumberSeed() == 0


def test_identical_replicas_diverge_only_with_distinct_offsets():
    from gareus.production import make_cmd_integrator, replica_integrator_seed_offset
    openmm, _app, unit = import_openmm()
    same_a = _run(openmm, unit, make_cmd_integrator(openmm, _cmd_args(), unit)[0])
    same_b = _run(openmm, unit, make_cmd_integrator(openmm, _cmd_args(), unit)[0])
    np.testing.assert_array_equal(same_a, same_b)  # the old behaviour: lockstep copies
    r0 = _run(openmm, unit, make_cmd_integrator(openmm, _cmd_args(), unit,
                                                seed_offset=replica_integrator_seed_offset(0))[0])
    r1 = _run(openmm, unit, make_cmd_integrator(openmm, _cmd_args(), unit,
                                                seed_offset=replica_integrator_seed_offset(1))[0])
    assert np.max(np.abs(r0 - r1)) > 1e-4


def test_gamd_integrator_honours_the_offset():
    from gareus import pep_gamd
    from gareus.production import make_gamd_integrator, replica_integrator_seed_offset
    from pep_gamd_fixture import solvated_dipeptide, _fresh_system
    openmm, _app, unit = import_openmm()
    fx = solvated_dipeptide()
    args = types.SimpleNamespace(
        gamd_boost_type=pep_gamd.PEP_GAMD_BOOST_TYPE, gamd_cmd_prep_steps=2, gamd_cmd_steps=4,
        gamd_equil_prep_steps=2, gamd_equil_steps=4, gamd_production_steps=100, gamd_averaging_window=2,
        sigma0p_kcal_mol=6.0, sigma0d_kcal_mol=6.0, temperature_k=300.0, timestep_fs=2.0,
        friction_per_ps=1.0, seed=3, pep_gamd_peptide_atoms=list(fx["peptide"]),
    )
    base, _ = make_gamd_integrator(_fresh_system(), args, unit)
    r0, _ = make_gamd_integrator(_fresh_system(), args, unit, seed_offset=replica_integrator_seed_offset(0))
    r1, _ = make_gamd_integrator(_fresh_system(), args, unit, seed_offset=replica_integrator_seed_offset(1))
    assert base.getRandomNumberSeed() == 3
    assert len({base.getRandomNumberSeed(), r0.getRandomNumberSeed(), r1.getRandomNumberSeed()}) == 3


def _integrator_calls_in_loops(source: str):
    """(function name, has seed_offset kw) for every integrator-factory call that
    sits inside a ``for`` loop in production.py - i.e. one built per replica/window."""
    names = {"make_production_integrator", "make_cmd_integrator", "make_gamd_integrator"}
    found = []
    for loop in (n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.For)):
        for call in (n for n in ast.walk(loop) if isinstance(n, ast.Call)):
            fn = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
            if fn in names:
                found.append((fn, call.lineno, any(k.arg == "seed_offset" for k in call.keywords)))
    return found


def test_every_per_replica_integrator_in_production_gets_a_seed_offset():
    import gareus.production as production
    calls = _integrator_calls_in_loops(Path(production.__file__).read_text(encoding="utf-8"))
    assert calls, "expected the replica and recon loops to build integrators"
    missing = [(fn, line) for fn, line, has in calls if not has]
    assert not missing, f"per-replica integrators built with the shared args.seed: {missing}"


def test_replica_and_recon_offsets_do_not_collide():
    from gareus.production import recon_integrator_seed_offset, replica_integrator_seed_offset
    replica = {replica_integrator_seed_offset(i) for i in range(1000)}
    recon = {recon_integrator_seed_offset(i) for i in range(1000)}
    assert len(replica) == 1000 and len(recon) == 1000
    assert not replica & recon
    assert 0 not in replica | recon
