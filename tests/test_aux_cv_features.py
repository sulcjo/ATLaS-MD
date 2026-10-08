# tests/test_aux_cv_features.py
import numpy as np
import pytest

from aux_cv_fixture import blocks_of, dipeptide, four_atoms_at, model_payload
from gareus.auxiliary_cv.features import (AuxGeometryError, check_feature_atoms, feature_signs,
                                          feature_values, openmm_dihedrals, unique_torsions)
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError


def _openmm_theta(xyz_nm, quad):
    import openmm as mm
    from openmm import unit
    s = mm.System()
    for _ in range(len(xyz_nm)):
        s.addParticle(1.0)
    f = mm.CustomTorsionForce("theta")
    f.addTorsion(*[int(i) for i in quad], [])
    s.addForce(f)
    c = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    c.setPositions(xyz_nm)
    return c.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def test_dihedrals_equal_openmm_theta_on_real_backbone():
    d = dipeptide()
    theta = openmm_dihedrals(d["positions_nm"], d["quads"])[0]
    for t, quad in zip(theta, d["quads"]):
        assert t == pytest.approx(_openmm_theta(d["positions_nm"], quad), abs=1e-9)


def test_dihedrals_are_minus_the_tica_angle():
    from gareus.tica import _dihedral_rad
    rng = np.random.default_rng(3)
    p = rng.normal(size=(4, 3))
    assert openmm_dihedrals(p, [(0, 1, 2, 3)])[0, 0] == pytest.approx(-_dihedral_rad(*p), abs=1e-12)


def test_degenerate_torsion_gives_nan():
    p = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 1, 0]], dtype=float)  # p0,p1,p2 collinear
    assert np.isnan(openmm_dihedrals(p, [(0, 1, 2, 3)])[0, 0])


def test_positions_across_the_pi_branch_cut_give_continuous_features():
    # Rotating atom 3 through the branch cut flips theta from ~+pi to ~-pi (or back);
    # the sin/cos features computed from positions must not jump.
    eps = 1e-6
    xyz = np.stack([four_atoms_at(np.pi - eps), four_atoms_at(np.pi + eps)])
    theta = openmm_dihedrals(xyz, [(0, 1, 2, 3)])[:, 0]
    assert abs(abs(theta[0]) - np.pi) < 1e-5 and abs(abs(theta[1]) - np.pi) < 1e-5
    assert np.sign(theta[0]) != np.sign(theta[1])          # the raw angle really wraps
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 1.0]))
    f = feature_values(theta[:, None], m)
    assert np.abs(f[0] - f[1]).max() < 1e-5


def test_negated_and_direct_signs():
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3), (1, 2, 3, 4)], [1, 1, 1, 1],
                                            conventions=["negated", "direct"]))
    np.testing.assert_array_equal(feature_signs(m), [-1, -1, 1, 1])
    theta = np.array([[0.3, 0.3]])
    np.testing.assert_allclose(feature_values(theta, m)[0],
                               [np.sin(-0.3), np.cos(-0.3), np.sin(0.3), np.cos(0.3)], atol=1e-15)


def test_unique_torsions_shares_sin_and_cos():
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3), (1, 2, 3, 4)], [1, 0, 0, 1]))
    quads, idx = unique_torsions(m)
    assert quads == ((0, 1, 2, 3), (1, 2, 3, 4))
    np.testing.assert_array_equal(idx, [0, 0, 1, 1])


def test_topology_check_accepts_real_backbone_and_rejects_shifted_atoms():
    d = dipeptide()
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.0] * (2 * len(d["quads"]) - 1),
                                            blocks=blocks_of(d)))
    check_feature_atoms(m, d["topology"])
    check_feature_atoms(m, d["topology"], topology_sha256="0" * 64)   # the payload's own digest
    bad = [tuple(a + 1 for a in q) for q in d["quads"]]
    m_bad = AuxModel.from_mapping(model_payload(bad, [1.0] + [0.0] * (2 * len(bad) - 1),
                                                blocks=blocks_of(d)))
    with pytest.raises(IntegrityError, match="atom names"):
        check_feature_atoms(m_bad, d["topology"])


def test_topology_check_compares_topology_digest():
    d = dipeptide()
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.0] * (2 * len(d["quads"]) - 1),
                                            blocks=blocks_of(d)))
    with pytest.raises(IntegrityError, match="topology"):
        check_feature_atoms(m, d["topology"], topology_sha256="a" * 64)
