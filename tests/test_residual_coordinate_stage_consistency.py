"""F02/P04: fitting, scoring, design, force, fast path, positions evaluator and reprojection
evaluate ONE residual coordinate, tails included (spec F02; finding I04).

The oracle here is written independently of `residual_runtime`: plain NumPy arithmetic on the
fit's arrays with an explicit clip, so a shared mistake in the compiled object is visible.
"""
from __future__ import annotations

import numpy as np
import pytest

from gareus.cv_selection.models import ResidualFit, evaluate_component, fit_residual_components
from gareus.cv_selection.residual_runtime import (TRANSFORM_HARD_CLIP, TRANSFORM_IDENTITY,
                                                  CompiledResidualComponent, compile_component)


def _oracle(fit: ResidualFit, j: int, X, a):
    """Independent evaluation: explicit clip for degree 2, identity for degree 1."""
    a_std = (np.asarray(a) - fit.anchor_mean) / fit.anchor_std
    t = np.clip(a_std, *fit.anchor_clamp) if fit.degree == 2 else a_std
    design = np.column_stack([np.ones_like(t), t, t ** 2])
    residual = np.asarray(X) - design @ fit.coefficients
    score = (residual - fit.residual_mean) @ fit.right_vectors[j - 1]
    return (score - fit.projection_mean[j - 1]) / fit.projection_std[j - 1]


def _tail_dataset(seed=27):
    """Valid sin/cos features, a narrow main anchor population plus a compact tail (the review's I04 fixture)."""
    rng = np.random.default_rng(seed)
    a = np.r_[rng.uniform(0.2, 0.4, 990), rng.uniform(0.8, 0.9, 10)]
    angles = np.column_stack([4 * a * a + 0.02 * rng.normal(size=a.size), rng.normal(size=a.size),
                              rng.normal(size=a.size), rng.normal(size=a.size)])
    X = np.stack([np.sin(angles), np.cos(angles)], axis=-1).reshape(len(a), -1)
    return X, a


def test_degree_two_fit_regresses_on_the_clipped_anchor_and_is_orthogonal_to_the_fitted_regressors():
    X, a = _tail_dataset()
    fit = fit_residual_components(X, a, degree=2)
    assert fit.transform == TRANSFORM_HARD_CLIP
    lo, hi = fit.anchor_clamp
    a_std = (a - fit.anchor_mean) / fit.anchor_std
    t = np.clip(a_std, lo, hi)
    assert (a_std > hi).sum() >= 1                       # the tail really is outside the clip
    w = np.full(a.size, 1.0 / a.size)
    z = evaluate_component(fit, 1, X, a)
    # exact orthogonality to the fitted regressors [1, T(a), T(a)^2] under the training weights
    for reg in (np.ones_like(t), t, t ** 2):
        assert abs(np.sum(w * z * (reg - np.sum(w * reg)))) < 1e-8
    # raw-anchor covariance is a separately measured, generally nonzero quantity under clipping
    cov_raw = float(np.sum(w * z * (a_std - np.sum(w * a_std))))
    assert np.isfinite(cov_raw)


def test_every_stage_gives_identical_values_on_the_tail_rows():
    X, a = _tail_dataset()
    for degree in (1, 2):
        fit = fit_residual_components(X, a, degree=degree)
        for j in (1, 2):
            z_eval = evaluate_component(fit, j, X, a)
            z_oracle = _oracle(fit, j, X, a)
            np.testing.assert_allclose(z_eval, z_oracle, atol=1e-12, rtol=1e-12)
            compiled = compile_component(fit, j, norm=1.0)
            np.testing.assert_allclose(compiled.evaluate_features(X, a), z_oracle, atol=1e-12, rtol=1e-12)
            # sub-CV route: torsion sums grouped as the force groups them (one weighted sum per feature here)
            roles = [{"name": f"f{k}", "role": "torsion_sum"} for k in range(X.shape[1])] + [{"name": "c", "role": "contact_sum"}]
            for row in (0, len(a) - 1):                  # a main-population row and a tail row
                named = {f"f{k}": float(X[row, k] * compiled.feature_weights[k]) for k in range(X.shape[1])}
                named["c"] = float(a[row])               # norm = 1
                assert compiled.evaluate_subcvs(named, roles) == pytest.approx(z_oracle[row], abs=1e-12)
            # record round trip preserves float64 evaluation
            again = CompiledResidualComponent.from_record(compiled.as_record(), compiled.feature_weights)
            np.testing.assert_array_equal(again.evaluate_features(X, a), compiled.evaluate_features(X, a))


def test_the_old_call_site_override_no_longer_exists():
    X, a = _tail_dataset()
    fit = fit_residual_components(X, a, degree=2)
    with pytest.raises(TypeError):
        evaluate_component(fit, 1, X, a, clamp=False)    # noqa: the override was the bug


def test_legacy_degree_two_fit_object_evaluates_as_the_old_force_did():
    """A v1 degree-2 artifact was fitted unclipped but deployed clipped; reading it back must
    reproduce the DEPLOYED (clipped) coordinate, never a refit."""
    X, a = _tail_dataset()
    rng = np.random.default_rng(1)
    B = rng.normal(size=(3, X.shape[1])) * 0.05
    v = rng.normal(size=(1, X.shape[1])); v /= np.linalg.norm(v)
    legacy = ResidualFit(B, np.zeros(X.shape[1]), np.array([1.0]), v, float(a.mean()), float(a.std()),
                         (-1.5, 1.5), np.array([0.0]), np.array([0.3]), 2)
    assert legacy.transform == TRANSFORM_HARD_CLIP
    z = evaluate_component(legacy, 1, X, a)
    np.testing.assert_allclose(z, _oracle(legacy, 1, X, a), atol=1e-12)
    a_std = (a - legacy.anchor_mean) / legacy.anchor_std
    assert (np.abs(a_std) > 1.5).any()                    # rows outside the clip exist and were clipped


def test_openmm_expression_matches_the_compiled_evaluator():
    openmm = pytest.importorskip("openmm")
    X, a = _tail_dataset()
    fit = fit_residual_components(X, a, degree=2)
    compiled = compile_component(fit, 1, norm=1.0)
    names = [f"f{k}" for k in range(X.shape[1])]
    expr = compiled.openmm_expression(names, "c")
    # evaluate the expression symbolically through OpenMM's parser with a zero-particle CustomExternalForce trick:
    # a CustomCompoundBondForce with global parameters standing in for the sub-variables
    system = openmm.System(); system.addParticle(1.0)
    f = openmm.CustomExternalForce(expr.replace("^", "^"))
    for n in names + ["c"]:
        f.addGlobalParameter(n, 0.0)
    f.addParticle(0, [])
    system.addForce(f)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions([[0.0, 0.0, 0.0]])
    for row in (0, len(a) - 1):
        for k, n in enumerate(names):
            ctx.setParameter(n, float(X[row, k] * compiled.feature_weights[k]))
        ctx.setParameter("c", float(a[row]))
        e = ctx.getState(getEnergy=True).getPotentialEnergy()._value
        assert e == pytest.approx(compiled.evaluate_features(X[row:row + 1], a[row:row + 1])[0], abs=1e-9)


def test_anchor_partials_match_finite_differences_away_from_the_kink_and_flag_it():
    X, a = _tail_dataset()
    fit = fit_residual_components(X, a, degree=2)
    compiled = compile_component(fit, 1, norm=1.0)
    lo, hi = compiled.clamp_lo, compiled.clamp_hi
    x0 = X[:1]
    # inside the clip, holding the torsion projection fixed
    c_mid = compiled.anchor_mean + 0.5 * (lo + hi) * compiled.anchor_std
    z, dz, d2z, kink = compiled.anchor_partials(x0, [c_mid])
    h = 1e-5
    zp = compiled.evaluate_features(x0, [c_mid + h])[0]; zm = compiled.evaluate_features(x0, [c_mid - h])[0]
    assert dz[0] == pytest.approx((zp - zm) / (2 * h), rel=1e-6, abs=1e-8)
    assert d2z[0] == pytest.approx((zp - 2 * z[0] + zm) / h ** 2, rel=1e-4, abs=1e-5)
    assert not kink[0]
    # outside: flat; at the boundary: flagged, not differentiated across
    c_out = compiled.anchor_mean + (hi + 1.0) * compiled.anchor_std
    _, dz_out, d2z_out, _ = compiled.anchor_partials(x0, [c_out])
    assert dz_out[0] == 0.0 and d2z_out[0] == 0.0
    c_edge = compiled.anchor_mean + hi * compiled.anchor_std
    assert compiled.anchor_partials(x0, [c_edge])[3][0]


def test_quadratic_curvature_fixture_has_the_full_second_derivative_available():
    from fixtures.thermodynamic_repair import quadratic_curvature_case
    c = quadratic_curvature_case()
    compiled = compile_component(c.fit, 1, norm=1.0)
    z, dz, d2z, _ = compiled.anchor_partials(np.array([[0.0]]), [0.0])
    # U2 = k2 (z - center)^2 / 2 with z = -2c^2:  U2'' = k2[(z')^2 + (z - center) z''] = 1*[0 + (0-1)(-4)] = 4
    curvature = c.k2 * (dz[0] ** 2 + (z[0] - c.center) * d2z[0])
    assert curvature == pytest.approx(c.expected_curvature_at_zero, abs=1e-9)
