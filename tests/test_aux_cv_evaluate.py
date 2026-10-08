# tests/test_aux_cv_evaluate.py
import numpy as np
import pytest

from aux_cv_fixture import blocks_of, dipeptide, four_atoms_at, model_payload, perturbed_geometries
from gareus.auxiliary_cv.evaluate import (aux_energy_kj, aux_forces_kj_nm, z_and_gradient,
                                          z_from_dihedrals, z_from_positions)
from gareus.auxiliary_cv.features import AuxGeometryError, openmm_dihedrals, unique_torsions
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError


def _mixed_model(quads, blocks=None, scale=3.181):
    rng = np.random.default_rng(7)
    coeffs = rng.normal(size=2 * len(quads))
    conv = ["negated" if k % 2 == 0 else "direct" for k in range(len(quads))]
    return AuxModel.from_mapping(model_payload(quads, coeffs, offset=0.4, conventions=conv, blocks=blocks,
                                               scale=scale))


def _geometries():
    """Minimised dipeptide, 4 perturbed copies, and two 4-atom geometries straddling +-pi."""
    d = dipeptide()
    cases = [(f"dipeptide-{i}", x, d["quads"], blocks_of(d)) for i, x in enumerate(perturbed_geometries(d))]
    for tag, rot in (("near+pi", np.pi - 1e-3), ("near-pi", np.pi + 1e-3)):
        cases.append((tag, four_atoms_at(rot), [(0, 1, 2, 3)], None))
    return cases


GEOMETRIES = _geometries()


def test_z_matches_hand_formula():
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [2.0, -1.0], offset=0.5, scale=2.0))
    theta = np.array([[0.7]])
    expected = (0.5 + 2.0 * np.sin(-0.7) - 1.0 * np.cos(-0.7)) / 2.0
    assert z_from_dihedrals(theta, m)[0] == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("tag, x, quads, blocks", GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_positions_and_dihedral_paths_agree(tag, x, quads, blocks):
    m = _mixed_model(quads, blocks=blocks)
    uq, _ = unique_torsions(m)
    a = z_from_positions(x, m)[0]
    b = z_from_dihedrals(openmm_dihedrals(x, uq), m)[0]
    c, _g = z_and_gradient(x, m)
    assert a == pytest.approx(b, abs=1e-12) and a == pytest.approx(c, abs=1e-12)


@pytest.mark.parametrize("tag, x, quads, blocks", GEOMETRIES, ids=[g[0] for g in GEOMETRIES])
def test_gradient_matches_finite_differences(tag, x, quads, blocks):
    m = _mixed_model(quads, blocks=blocks)
    x = x.copy()
    _z, g = z_and_gradient(x, m)
    atoms = sorted({a for q in quads for a in q})
    h = 1e-6
    for i in atoms:
        for k in range(3):
            xp, xm = x.copy(), x.copy()
            xp[i, k] += h
            xm[i, k] -= h
            fd = (z_from_positions(xp, m)[0] - z_from_positions(xm, m)[0]) / (2 * h)
            assert g[i, k] == pytest.approx(fd, rel=1e-5, abs=1e-6)
    untouched = np.setdiff1d(np.arange(len(x)), atoms)
    assert not np.any(g[untouched])


def test_z_continuous_across_the_pi_branch_cut():
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.3, -0.4], offset=0.1, scale=1.7))
    z = z_from_positions(np.stack([four_atoms_at(np.pi - 1e-6), four_atoms_at(np.pi + 1e-6)]), m)
    assert abs(z[0] - z[1]) < 1e-5


def test_forces_are_minus_energy_gradient_in_kj_nm():
    d = dipeptide()
    m = _mixed_model(d["quads"], blocks=blocks_of(d))
    x = d["positions_nm"].copy()
    z, g = z_and_gradient(x, m)
    f = aux_forces_kj_nm(x, m, center=z - 0.3, k_kcal=2.0)
    np.testing.assert_allclose(f, -2.0 * 4.184 * 0.3 * g, rtol=1e-12, atol=1e-12)


def test_zero_strength_energy_is_exact_zero_even_for_nan_z():
    e = aux_energy_kj(np.array([np.nan, 1.0, -3.0, 1e200]), center=0.0, k_kcal=0.0)
    assert e.dtype == np.float64 and np.all(e == 0.0)


def test_energy_units_and_value():
    e = aux_energy_kj(np.array([1.5]), center=0.5, k_kcal=2.0)
    assert e[0] == pytest.approx(0.5 * 2.0 * 4.184 * 1.0, abs=1e-12)


@pytest.mark.parametrize("bad_k", [-1.0, float("nan"), float("inf"), None, True])
def test_invalid_strength_is_an_error(bad_k):
    with pytest.raises(IntegrityError):
        aux_energy_kj(np.array([1.0]), center=0.0, k_kcal=bad_k)


def test_active_strength_needs_finite_center():
    with pytest.raises(IntegrityError, match="center"):
        aux_energy_kj(np.array([1.0]), center=float("nan"), k_kcal=1.0)


def test_degenerate_geometry_raises_in_gradient_and_is_nan_offline():
    p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 1, 1]], dtype=float)
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
    assert np.isnan(z_from_positions(p, m)[0])
    with pytest.raises(AuxGeometryError):
        z_and_gradient(p, m)


def test_degenerate_torsion_with_only_zero_coefficients_does_not_matter():
    """z depends only on nonzero-coefficient features, exactly like the force (which skips them):
    a degenerate torsion that carries no weight leaves z defined and finite."""
    p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0], [4, 1, 1]], dtype=float) * 0.1
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3), (1, 2, 3, 4)], [0.0, 0.0, 1.0, 0.5], offset=0.2))
    z = z_from_positions(p, m)[0]
    zg, g = z_and_gradient(p, m)
    assert np.isfinite(z) and zg == pytest.approx(z, abs=1e-12) and np.all(np.isfinite(g))
    m_live = AuxModel.from_mapping(model_payload([(0, 1, 2, 3), (1, 2, 3, 4)], [0.0, 1e-3, 1.0, 0.5]))
    with pytest.raises(AuxGeometryError):
        z_and_gradient(p, m_live)


def test_negative_centres_are_allowed():
    """z is a signed projection: finite_number(center, ...) is called without a minimum
    (its default is minimum=None), so negative centres are valid."""
    e = aux_energy_kj(np.array([-1.5]), center=-0.5, k_kcal=2.0)
    assert e[0] == pytest.approx(0.5 * 2.0 * 4.184 * 1.0, abs=1e-12)
