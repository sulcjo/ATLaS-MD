"""Shared fixture for the Pep-GaMD tests: a GA dipeptide in a 2.4 nm TIP3P box built
with gareus's own builders, briefly minimised, plus small energy/copy helpers."""
import pathlib
import tempfile
import types

from gareus.imports import import_openmm

_CACHE: dict = {}


def _fake_args(**over):
    base = dict(nonbonded_cutoff_nm=0.9, ewald_error_tolerance=0.0005, hmr=False, hydrogen_mass_amu=0.0)
    base.update(over)
    return types.SimpleNamespace(**base)


def solvated_dipeptide():
    """GA dipeptide in a 2.4 nm TIP3P box, built with gareus's own builders. Cached."""
    if "sys" in _CACHE:
        return _CACHE["sys"]
    from gareus.system_setup import build_peptide_pdb, make_forcefield, create_system
    from gareus.energy_decomposition import peptide_atom_groups_from_topology
    openmm, app, unit = import_openmm()
    d = pathlib.Path(tempfile.mkdtemp())
    pdb = app.PDBFile(str(build_peptide_pdb("GA", d / "pep.pdb")))
    ff = make_forcefield(app, "tip3p")
    m = app.Modeller(pdb.topology, pdb.positions)
    m.addHydrogens(ff, pH=7.0)
    m.addSolvent(ff, boxSize=openmm.Vec3(2.4, 2.4, 2.4) * unit.nanometer, model="tip3p")
    args = _fake_args()
    system = create_system(app, unit, ff, m.topology, args, include_barostat=False)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001 * unit.picoseconds), openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(m.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    positions = ctx.getState(getPositions=True).getPositions()
    del ctx
    _groups, pep, nonpep = peptide_atom_groups_from_topology(m.topology, "all-peptide")
    out = dict(system=system, topology=m.topology, positions=positions, ff=ff, args=args,
               peptide=sorted(pep), nonpeptide=sorted(nonpep))
    _CACHE["sys"] = out
    return out


def _fresh_system():
    from gareus.seeding import deserialize_system
    openmm, _app, _unit = import_openmm()
    return deserialize_system(openmm, solvated_dipeptide()["system"])


def tiny_solvated_system():
    """A fresh (unpartitioned) copy of the cached GA-dipeptide-in-TIP3P system, plus its
    topology/positions, for tests that need real OpenMM energies on a tiny system."""
    openmm, app, unit = import_openmm()
    fx = solvated_dipeptide()
    system = _fresh_system()
    return openmm, app, unit, fx["topology"], system, fx["positions"]


def _nonbonded(system, openmm):
    return [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]


def _energy(system, positions, groups, openmm, unit):
    integ = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(positions)
    e = ctx.getState(getEnergy=True, groups=groups).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del ctx
    return float(e)


def build_small_simulation(platform="Reference"):
    """An app.Simulation on a fresh copy of the cached GA dipeptide in TIP3P, velocities set."""
    openmm, app, unit, topology, system, positions = tiny_solvated_system()
    integ = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    sim = app.Simulation(topology, system, integ, openmm.Platform.getPlatformByName(platform))
    sim.context.setPositions(positions)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    return sim


