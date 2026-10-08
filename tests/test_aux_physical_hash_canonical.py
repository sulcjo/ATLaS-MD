"""Task 14 fix round 1, F1: physical_system_sha256 is insensitive to bonded/constraint/exception term order and
pair direction (a fresh run builds from the in-memory topology, a resume from 01_solvated_start.pdb), never
reorders particles, and refuses force types it has no canonical form for."""
import pytest

from gareus.auxiliary_cv.runtime_definition import physical_system_sha256
from gareus.correctness._io import IntegrityError


def _mm():
    import openmm as mm
    return mm


def _fresh():
    from pep_gamd_fixture import _fresh_system
    return _fresh_system()


def _permuted(system):
    """The same physics with every term list reversed and every term written in reverse atom order."""
    mm = _mm()
    out = mm.System()
    for i in range(system.getNumParticles()):
        out.addParticle(system.getParticleMass(i))
    out.setDefaultPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    for i in reversed(range(system.getNumConstraints())):
        a, b, d = system.getConstraintParameters(i)
        out.addConstraint(b, a, d)
    for f in system.getForces():
        if isinstance(f, mm.HarmonicBondForce):
            g = mm.HarmonicBondForce()
            for i in reversed(range(f.getNumBonds())):
                a, b, r, k = f.getBondParameters(i)
                g.addBond(b, a, r, k)
        elif isinstance(f, mm.HarmonicAngleForce):
            g = mm.HarmonicAngleForce()
            for i in reversed(range(f.getNumAngles())):
                a, b, c, t, k = f.getAngleParameters(i)
                g.addAngle(c, b, a, t, k)
        elif isinstance(f, mm.PeriodicTorsionForce):
            g = mm.PeriodicTorsionForce()
            for i in reversed(range(f.getNumTorsions())):
                a, b, c, d, n, ph, k = f.getTorsionParameters(i)
                g.addTorsion(d, c, b, a, n, ph, k)
        elif isinstance(f, mm.NonbondedForce):
            g = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(f))
            ex = [g.getExceptionParameters(i) for i in range(g.getNumExceptions())]
            for i, (a, b, q, s, e) in enumerate(reversed(ex)):
                g.setExceptionParameters(i, b, a, q, s, e)
        else:
            g = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(f))
        g.setForceGroup(f.getForceGroup())
        out.addForce(g)
    return out


def test_term_order_and_direction_do_not_change_the_hash():
    mm = _mm()
    s = _fresh()
    p = _permuted(s)
    assert mm.XmlSerializer.serialize(p) != mm.XmlSerializer.serialize(s)
    assert physical_system_sha256(mm, p) == physical_system_sha256(mm, s)


def test_pdb_round_trip_system_hashes_equal_to_the_in_memory_one(tmp_path):
    """The real F1 case: the resume builds its System from the PDB round trip of the solvated topology."""
    from openmm import app, unit
    from pep_gamd_fixture import solvated_dipeptide
    from gareus.system_setup import create_system
    mm = _mm()
    fx = solvated_dipeptide()
    path = tmp_path / "01_solvated_start.pdb"
    with path.open("w") as fh:
        app.PDBFile.writeFile(fx["topology"], fx["positions"], fh, keepIds=True)
    pdb = app.PDBFile(str(path))
    resumed = create_system(app, unit, fx["ff"], pdb.topology, fx["args"], include_barostat=False)
    fresh = _fresh()
    assert physical_system_sha256(mm, resumed) == physical_system_sha256(mm, fresh)


def test_a_physical_change_or_a_particle_reorder_changes_the_hash():
    mm = _mm()
    s, base = _fresh(), None
    base = physical_system_sha256(mm, s)
    bonds = next(f for f in s.getForces() if isinstance(f, mm.HarmonicBondForce))
    a, b, r, k = bonds.getBondParameters(0)
    bonds.setBondParameters(0, a, b, r, k * 1.001)
    assert physical_system_sha256(mm, s) != base
    s2 = _fresh()
    m0, m1 = s2.getParticleMass(0), s2.getParticleMass(1)
    s2.setParticleMass(0, m1)
    s2.setParticleMass(1, m0)
    assert m0 != m1 and physical_system_sha256(mm, s2) != base
    s3 = _fresh()
    nb = next(f for f in s3.getForces() if isinstance(f, mm.NonbondedForce))
    q, sig, eps = nb.getParticleParameters(0)
    nb.setParticleParameters(0, q * 1.01, sig, eps)
    assert physical_system_sha256(mm, s3) != base


@pytest.mark.parametrize("make", ["custom", "cmap"])
def test_unknown_force_types_are_refused_not_hashed_raw(make):
    mm = _mm()
    s = _fresh()
    if make == "custom":
        s.addForce(mm.CustomExternalForce("0"))
    else:
        s.addForce(mm.CMAPTorsionForce())
    with pytest.raises(IntegrityError, match="no canonical form"):
        physical_system_sha256(mm, s)


def test_index_referenced_exception_offsets_are_refused():
    mm = _mm()
    s = _fresh()
    nb = next(f for f in s.getForces() if isinstance(f, mm.NonbondedForce))
    nb.addGlobalParameter("lam", 0.0)
    nb.addExceptionParameterOffset("lam", 0, 0.1, 0.0, 0.0)
    with pytest.raises(IntegrityError, match="ExceptionOffsets"):
        physical_system_sha256(mm, s)


def test_virtual_sites_are_refused_not_dropped():
    """Fix round 2: a virtual site serialises as a child of <Particle mass="0">; hashing only the particle's
    attributes would make Systems with different virtual sites hash equal."""
    mm = _mm()
    s = mm.System()
    for m in (12.0, 12.0, 0.0):
        s.addParticle(m)
    s.setVirtualSite(2, mm.TwoParticleAverageSite(0, 1, 0.5, 0.5))
    bonds = mm.HarmonicBondForce()
    bonds.addBond(0, 1, 0.15, 1000.0)
    s.addForce(bonds)
    with pytest.raises(IntegrityError, match="virtual site"):
        physical_system_sha256(mm, s)
