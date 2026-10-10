"""Independent numerical checks for the opt-in side-chain projection kernel."""
import numpy as np
import pytest

from gareus.auxiliary_cv.sidechain_core import Primitive, Projection, atom_map


def projection():
    return Projection((Primitive(((0, 1, 2, 3), (0, 1, 2, 4)), 'cos', 2),
                       Primitive(((0, 1, 2, 3),), 'sin', 1, -1)), (0.7, -0.3), 0.2, 1.7)


def test_gradient_against_cartesian_finite_difference():
    p = projection()
    x = np.array([[0.1, 0.2, 0.0], [0., 0., 0.], [.15, .0, .0], [.2, .2, .1], [.2, -.1, -.2]])
    z, grad = p.value_gradient(x)
    eps = 1e-7
    numerical = np.empty_like(x)
    for a, d in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[a, d] += eps
        minus[a, d] -= eps
        numerical[a, d] = (p.values(plus)[0] - p.values(minus)[0]) / (2 * eps)
    np.testing.assert_allclose(grad, numerical, atol=2e-8, rtol=2e-7)
    assert z == pytest.approx(p.values(x)[0])
    np.testing.assert_allclose(grad.sum(axis=0), 0, atol=1e-12)


def test_orbit_permutation_and_canonical_identity():
    a = Primitive(((0, 1, 2, 3), (0, 1, 2, 4)), 'cos', 2)
    b = Primitive(tuple(reversed(a.orbit)), 'cos', 2)
    p, q = Projection((a,), (1.,)), Projection((b,), (1.,))
    assert p.digest == q.digest
    x = np.random.default_rng(2).normal(size=(5, 3))
    perm = [0, 1, 2, 4, 3]
    z, g = p.value_gradient(x)
    z2, g2 = p.value_gradient(x[perm])
    assert z == pytest.approx(z2)
    np.testing.assert_allclose(g2, g[perm], atol=1e-12)


def test_mapping_nonprefix_and_duplicates():
    assert atom_map(('water', 'A:N', 'A:CA', 'A:C', 'A:CB'), ('A:CB', 'A:C', 'A:N', 'A:CA')) == (None, 2, 3, 1, 0)
    p = Projection((Primitive(((1, 2, 3, 4),), 'sin'),), (1.,))
    q = p.remap((None, 2, 3, 1, 0))
    xyz = np.random.default_rng(3).normal(size=(5, 3))
    np.testing.assert_allclose(p.values(xyz), q.values(xyz[[4, 3, 1, 2]]))
    with pytest.raises(ValueError, match='duplicate'):
        atom_map(('A', 'A'), ('A',))
    with pytest.raises(ValueError, match='missing'):
        p.remap((None, 0, 1, 2, None))


@pytest.mark.parametrize('kw', [{'harmonic': True}, {'harmonic': 0}, {'sign': 0}, {'trig': 'tan'},
                               {'orbit': ((0, 1, 2, 2),)}, {'orbit': ((0, 1, 2, 3.0),)},
                               {'orbit': ((0, 1, 2, 3), (0, 1, 2, 3))}])
def test_bad_primitives(kw):
    args = dict(orbit=((0, 1, 2, 3),), trig='sin')
    args.update(kw)
    with pytest.raises(ValueError):
        Primitive(**args)


def test_inactive_geometry_and_serialisation():
    p = Projection((Primitive(((0, 1, 2, 3),), 'sin'),), (0.,), offset=2.)
    np.testing.assert_array_equal(p.values(np.zeros((4, 3))), [2.])
    np.testing.assert_array_equal(p.value_gradient(np.zeros((4, 3)))[1], np.zeros((4, 3)))
    assert Projection.from_mapping(projection().to_mapping()).digest == projection().digest
    with pytest.raises(ValueError):
        Projection((Primitive(((0, 1, 2, 3),), 'sin'),), (1.,), scale=float('nan'))
    with pytest.raises(ValueError, match='degenerate'):
        Projection((Primitive(((0, 1, 2, 3),), 'sin'),), (1.,)).values(np.zeros((4, 3)))


def test_openmm_reference_matches_independent_forces():
    mm = pytest.importorskip('openmm')
    from openmm import unit
    p = projection()
    system = mm.System()
    for _ in range(5):
        system.addParticle(12.)
    system.addForce(p.build_force(mm, center=.3, k_kj=2.1))
    integrator = mm.VerletIntegrator(.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName('Reference'))
    x = np.random.default_rng(5).normal(size=(5, 3))
    context.setPositions(x)
    state = context.getState(getEnergy=True, getForces=True)
    z, grad = p.value_gradient(x)
    assert state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) == pytest.approx(.5 * 2.1 * (z-.3)**2)
    np.testing.assert_allclose(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer),
                               -2.1 * (z-.3) * grad, atol=1e-10, rtol=1e-9)
