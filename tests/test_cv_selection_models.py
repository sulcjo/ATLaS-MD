"""Task 2: residual torsion components.

The load-bearing property is exact *weighted* orthogonality to the anchor.
"""
from __future__ import annotations

import numpy as np
import pytest

from gareus.cv_selection import contracts as C
from gareus.cv_selection.models import (coupling_curvature_kcal, evaluate_component,
                                        fit_residual_components, from_candidate_set,
                                        to_candidate_set)


def _synthetic(n=2000, d=8, degree=2, seed=0, quad_scale=3.0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 0.2, n)
    a_std = (a - a.mean()) / a.std()
    B0, B1 = rng.normal(size=d), rng.normal(size=d)
    B2 = rng.normal(size=d) * quad_scale * (degree == 2)
    hidden = rng.normal(size=(n, d)) @ np.diag(np.linspace(3.0, 0.2, d))
    return B0 + np.outer(a_std, B1) + np.outer(a_std ** 2, B2) + hidden, a


def _balanced(n, k=5, seed=0):
    cells = np.random.default_rng(seed).integers(0, k, n)
    counts = np.bincount(cells, minlength=k)
    return 1.0 / (k * counts[cells])


def _contact_definition():
    return {"kind": "nonlocal-contact-fraction", "units": "dimensionless",
            "definition": {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0,
                           "min_sequence_separation": 4, "atom_selection": "heavy",
                           "pair_rule": "all-pairs-min-sep", "normalize": True}}


@pytest.fixture
def feature_schema_8():
    rows = []
    for t in range(4):
        name = "phi" if t < 2 else "psi"
        for trig in ("sin", "cos"):
            rows.append({"index": len(rows), "name": f"{name}-{t + 1}-{trig}",
                         "torsion_name": f"{name}-{t + 1}", "residue_index": t + 1,
                         "atom_indices": [4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3],
                         "trig": trig, "dihedral_sign_convention": "negated"})
    return C.FeatureSchema.from_mapping({"schema": C.FEATURE_SCHEMA_VERSION,
                                         "topology_sha256": "a" * 64, "features": rows})


def test_residual_is_exactly_uncorrelated_with_the_anchor_unweighted():
    X, a = _synthetic(degree=1)
    fit = fit_residual_components(X, a, degree=1)
    for j in range(1, 7):
        z2 = evaluate_component(fit, j, X, a)
        assert abs(np.cov(z2, a)[0, 1]) < 1e-10
        assert abs(z2.mean()) < 1e-10 and abs(z2.std() - 1.0) < 1e-10


def test_residual_is_exactly_uncorrelated_under_the_balanced_weights_it_was_fit_with():
    """Production fits with balanced weights; orthogonality holds in THAT inner product."""
    X, a = _synthetic(degree=1)
    w = _balanced(len(a))
    fit = fit_residual_components(X, a, degree=1, weights=w)
    z2 = evaluate_component(fit, 1, X, a)
    z1 = (a - fit.anchor_mean) / fit.anchor_std
    assert abs(np.sum(w * z1 * z2)) < 1e-8            # weighted covariance: exact
    assert abs(np.cov(z1, z2)[0, 1]) > 1e-7             # unweighted: NOT zero -- never certify with it


def test_degree_two_removes_quadratic_dependence_that_degree_one_leaves_in_some_component():
    X, a = _synthetic(degree=2, seed=3, quad_scale=6.0)
    a_std = (a - a.mean()) / a.std()
    lin = fit_residual_components(X, a, degree=1)
    quad = fit_residual_components(X, a, degree=2)
    # The degree-2 regressor is T(a)^2 with T the declared clip (spec F02): exact orthogonality
    # is to THAT column; the 1 % of rows outside the clip keep their raw quadratic dependence
    # by construction, which is the deployed coordinate's honest definition.
    t = np.clip(a_std, *quad.anchor_clamp)
    q_fitted = t ** 2 - (t ** 2).mean()
    q_raw = a_std ** 2 - (a_std ** 2).mean()

    def corr(fit, j, q):
        return abs(np.corrcoef(evaluate_component(fit, j, X, a), q)[0, 1])

    assert all(corr(quad, j, q_fitted) < 1e-6 for j in range(1, 7))
    assert max(corr(lin, j, q_raw) for j in range(1, 7)) > 0.05     # somewhere, not necessarily PC1
    assert max(corr(quad, j, q_raw) for j in range(1, 7)) < 0.05    # and the clip leaves only a tail remnant


def test_components_are_individual_orthonormal_directions_with_positive_dominant_coefficient():
    X, a = _synthetic()
    V = fit_residual_components(X, a).right_vectors
    assert np.allclose(V @ V.T, np.eye(V.shape[0]), atol=1e-10)
    assert all(row[np.argmax(np.abs(row))] > 0 for row in V)


def test_candidate_set_round_trip_is_exact(feature_schema_8):
    X, a = _synthetic()
    fit = fit_residual_components(X, a)
    cs = to_candidate_set(fit, feature_schema_8, _contact_definition(), "b" * 64, "c" * 64,
                          {"numpy": np.__version__})
    assert cs.kind == C.CANDIDATE_KIND_QUADRATIC_RESIDUAL
    back = from_candidate_set(C.CandidateSet.from_mapping(cs.to_mapping()))
    assert np.array_equal(evaluate_component(fit, 2, X, a), evaluate_component(back, 2, X, a))
    assert back.degree == fit.degree and back.anchor_clamp == fit.anchor_clamp


def test_degree_survives_the_round_trip_for_a_linear_fit(feature_schema_8):
    X, a = _synthetic(degree=1)
    fit = fit_residual_components(X, a, degree=1)
    cs = to_candidate_set(fit, feature_schema_8, _contact_definition(), "b" * 64, "c" * 64,
                          {"numpy": np.__version__})
    assert from_candidate_set(cs).degree == 1


def test_degree_two_evaluation_freezes_the_anchor_outside_the_training_range_by_its_declared_transform():
    """The clip is a property of the fitted model (transform 'hard_clip'), applied at every
    stage -- not a call-site option (spec F02; the old `clamp=` override was finding I04)."""
    X, a = _synthetic(degree=2)
    fit = fit_residual_components(X, a, degree=2)
    assert fit.transform == "hard_clip"
    far = np.full(3, a.max() + 10 * a.std())
    edge = np.full(3, fit.anchor_mean + fit.anchor_clamp[1] * fit.anchor_std)
    assert np.allclose(evaluate_component(fit, 1, X[:3], far), evaluate_component(fit, 1, X[:3], edge))
    # a degree-1 fit has the identity transform: the same two anchors give different values
    X1, a1 = _synthetic(degree=1)
    fit1 = fit_residual_components(X1, a1, degree=1)
    assert fit1.transform == "identity"
    far1 = np.full(3, a1.max() + 10 * a1.std()); edge1 = np.full(3, a1.max())
    assert not np.allclose(evaluate_component(fit1, 1, X1[:3], far1), evaluate_component(fit1, 1, X1[:3], edge1))


def test_coupling_curvature_matches_its_definition():
    X, a = _synthetic(degree=1)
    fit = fit_residual_components(X, a, degree=1)
    v = fit.right_vectors[0]
    expected = 50.0 * (v @ fit.coefficients[1]) ** 2 / (fit.projection_std[0] ** 2 * fit.anchor_std ** 2)
    assert np.isclose(coupling_curvature_kcal(fit, 1, 50.0), expected)
    assert coupling_curvature_kcal(fit, 1, 0.0) == 0.0


def test_constant_anchor_and_rank_deficient_design_are_refused():
    X, a = _synthetic()
    with pytest.raises(ValueError, match="anchor"):
        fit_residual_components(X, np.full(len(X), 0.05))
    X2, a2 = _synthetic(n=2)
    with pytest.raises(ValueError, match="rank"):
        fit_residual_components(X2, a2, degree=2)


def test_width_mismatch_and_bad_component_index_are_refused():
    X, a = _synthetic()
    fit = fit_residual_components(X, a)
    with pytest.raises(ValueError, match="out of range"):
        evaluate_component(fit, 7, X, a)
    with pytest.raises(ValueError, match="must be"):
        evaluate_component(fit, 1, X[:, :4], a)


def test_candidate_set_width_must_match_the_feature_schema(feature_schema_8):
    X, a = _synthetic(d=6)
    fit = fit_residual_components(X, a)
    with pytest.raises(ValueError, match="width"):
        to_candidate_set(fit, feature_schema_8, _contact_definition(), "b" * 64, "c" * 64,
                         {"numpy": np.__version__})
