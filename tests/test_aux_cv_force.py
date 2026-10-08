# tests/test_aux_cv_force.py
import numpy as np
import pytest

from aux_cv_fixture import blocks_of, dipeptide, four_atoms_at, model_payload, perturbed_geometries
from gareus.auxiliary_cv.evaluate import aux_energy_kj, aux_forces_kj_nm, z_from_positions
from gareus.auxiliary_cv.force import (AUX_FORCE_NAME, build_aux_force, set_aux_parameters)
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError


def _context(model, n_atoms, force_group=7, platform="Reference"):
    import openmm as mm
    s = mm.System()
    for _ in range(n_atoms):
        s.addParticle(1.0)
    force, info = build_aux_force(mm, model, force_group=force_group)
    s.addForce(force)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform))
    return ctx, info, force


def _energy_forces(ctx):
    from openmm import unit
    st = ctx.getState(getEnergy=True, getForces=True)
    return (st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            np.asarray(st.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)))


def _model(d, conventions=None, coeffs=None, offset=0.4, scale=3.181):
    rng = np.random.default_rng(11)
    width = 2 * len(d["quads"])
    return AuxModel.from_mapping(model_payload(
        d["quads"], rng.normal(size=width) if coeffs is None else coeffs, offset=offset,
        conventions=conventions, blocks=blocks_of(d), scale=scale))


def _require_platform(name):
    import openmm as mm
    names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
    if name not in names:
        pytest.skip(f"OpenMM platform {name} unavailable")


@pytest.mark.parametrize("platform", ["Reference", "CPU"])
@pytest.mark.parametrize("conventions", [None, "mixed"])
@pytest.mark.parametrize("geometry", range(5))
def test_energy_and_forces_match_reference(conventions, geometry, platform):
    _require_platform(platform)
    d = dipeptide()
    x = perturbed_geometries(d)[geometry]
    conv = None if conventions is None else ["negated" if k % 2 == 0 else "direct" for k in range(len(d["quads"]))]
    m = _model(d, conv)
    ctx, info, _ = _context(m, len(x), platform=platform)
    ctx.setPositions(x)
    z = z_from_positions(x, m)[0]
    set_aux_parameters(ctx, info, center=z - 0.25, k_kcal=3.0)
    e, f = _energy_forces(ctx)
    etol = 1e-10 if platform == "Reference" else 1e-5     # CPU platform computes in single precision trig
    ftol = 1e-8 if platform == "Reference" else 1e-4
    assert e == pytest.approx(aux_energy_kj(np.array([z]), z - 0.25, 3.0)[0], rel=etol, abs=etol)
    np.testing.assert_allclose(f, aux_forces_kj_nm(x, m, z - 0.25, 3.0), rtol=ftol, atol=ftol)


def test_force_energy_continuous_across_the_pi_branch_cut():
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.3, -0.4], offset=0.1, scale=1.7))
    energies = []
    for rot in (np.pi - 1e-6, np.pi + 1e-6):
        ctx, info, _ = _context(m, 4)
        ctx.setPositions(four_atoms_at(rot))
        set_aux_parameters(ctx, info, center=-0.2, k_kcal=5.0)
        energies.append(_energy_forces(ctx)[0])
    assert abs(energies[0] - energies[1]) < 1e-4


def test_zero_strength_contributes_exactly_nothing_at_regular_geometry():
    d = dipeptide()
    m = _model(d)
    ctx, info, _ = _context(m, len(d["positions_nm"]))
    ctx.setPositions(d["positions_nm"])
    set_aux_parameters(ctx, info, center=5.0, k_kcal=0.0)
    e, f = _energy_forces(ctx)
    assert e == 0.0 and not np.any(f)
    assert ctx.getParameter(info.global_c) == 0.0


def test_select_makes_zero_strength_exact_even_when_the_square_overflows():
    # Without select, 0.5*0*(1e200 - 0)^2 = 0*inf = NaN. With select the energy is exactly 0.0.
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0], offset=1e200))
    ctx, info, force = _context(m, 4)
    assert force.getEnergyFunction().startswith("select(aux_k,")
    ctx.setPositions(four_atoms_at(1.0))
    set_aux_parameters(ctx, info, center=0.0, k_kcal=0.0)
    e, _f = _energy_forces(ctx)
    assert e == 0.0


def test_zero_strength_at_degenerate_geometry_has_zero_energy_but_nan_forces():
    """Documented limit (Global Constraints): at collinear torsion atoms OpenMM's torsion
    derivative is NaN, and 0 * NaN = NaN in the chain rule, so k = 0 gives E = 0 but NaN forces.
    Stock OpenMM torsion forces behave the same way; Stage B must never integrate such a state."""
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
    ctx, info, _ = _context(m, 4)
    ctx.setPositions(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0]], dtype=float) * 0.1)
    set_aux_parameters(ctx, info, center=0.3, k_kcal=0.0)
    e, f = _energy_forces(ctx)
    assert e == 0.0
    assert np.isnan(f[:4]).any()


def test_full_square_keeps_cross_terms():
    d = dipeptide()
    width = 2 * len(d["quads"])
    ca = np.zeros(width); ca[0] = 1.3
    cb = np.zeros(width); cb[-1] = -0.8
    m_ab = _model(d, coeffs=ca + cb, offset=0.2, scale=1.0)
    za = z_from_positions(d["positions_nm"], _model(d, coeffs=ca, offset=0.0, scale=1.0))[0]
    zb = z_from_positions(d["positions_nm"], _model(d, coeffs=cb, offset=0.0, scale=1.0))[0]
    ctx, info, _ = _context(m_ab, len(d["positions_nm"]))
    ctx.setPositions(d["positions_nm"])
    set_aux_parameters(ctx, info, center=0.0, k_kcal=1.0)
    e, _ = _energy_forces(ctx)
    full = 0.5 * 4.184 * (za + zb + 0.2) ** 2
    separate = 0.5 * 4.184 * ((za + 0.2) ** 2 + zb ** 2)
    assert e == pytest.approx(full, rel=1e-10)
    assert abs(full - separate) > 1e-3


def test_force_metadata_group_and_no_periodic_imaging():
    d = dipeptide()
    m = _model(d)
    _ctx, info, force = _context(m, len(d["positions_nm"]), force_group=12)
    assert force.getName() == AUX_FORCE_NAME and force.getForceGroup() == 12
    assert info.force_group == 12 and info.model_sha256 == m.model_sha256
    for i in range(force.getNumCollectiveVariables()):
        assert not force.getCollectiveVariable(i).usesPeriodicBoundaryConditions()


@pytest.mark.parametrize("group", [-1, 32])
def test_force_group_range(group):
    import openmm as mm
    d = dipeptide()
    with pytest.raises(IntegrityError, match="force group"):
        build_aux_force(mm, _model(d), force_group=group)


def test_set_parameters_rejects_invalid_strength():
    d = dipeptide()
    ctx, info, _ = _context(_model(d), len(d["positions_nm"]))
    with pytest.raises(IntegrityError):
        set_aux_parameters(ctx, info, center=0.0, k_kcal=-1.0)


@pytest.mark.parametrize("geometry", range(3))
def test_same_torsion_twice_in_one_group_matches_reference(geometry):
    """Review Focus 3: a torsion listed twice in one (trig, sign) group gets each feature's weight."""
    d = dipeptide()
    x = perturbed_geometries(d)[geometry]
    q0, q1 = d["quads"][0], d["quads"][1]
    m = AuxModel.from_mapping(model_payload([q0, q0, q1], [0.7, -0.2, 1.1, 0.4, -0.5, 0.3], offset=0.1,
                                            scale=1.3))
    ctx, info, force = _context(m, len(x))
    names = [force.getCollectiveVariableName(i) for i in range(force.getNumCollectiveVariables())]
    sin_neg = force.getCollectiveVariable(names.index("aux_sin_neg"))
    assert sin_neg.getNumTorsions() == 3          # q0 twice + q1, one weight per feature
    ctx.setPositions(x)
    z = z_from_positions(x, m)[0]
    set_aux_parameters(ctx, info, center=z + 0.3, k_kcal=2.5)
    e, f = _energy_forces(ctx)
    assert e == pytest.approx(aux_energy_kj(np.array([z]), z + 0.3, 2.5)[0], rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(f, aux_forces_kj_nm(x, m, z + 0.3, 2.5), rtol=1e-8, atol=1e-8)


def test_zero_coefficient_degenerate_torsion_agrees_with_evaluator():
    p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 1, 1]], dtype=float) * 0.1
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3), (1, 2, 3, 4)], [0.0, 0.0, 1.0, 0.5], offset=0.2))
    ctx, info, _ = _context(m, 5)
    ctx.setPositions(p)
    z = z_from_positions(p, m)[0]
    set_aux_parameters(ctx, info, center=z - 0.4, k_kcal=1.5)
    e, f = _energy_forces(ctx)
    assert np.all(np.isfinite(f))
    assert e == pytest.approx(aux_energy_kj(np.array([z]), z - 0.4, 1.5)[0], rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(f, aux_forces_kj_nm(p, m, z - 0.4, 1.5), rtol=1e-8, atol=1e-8)
