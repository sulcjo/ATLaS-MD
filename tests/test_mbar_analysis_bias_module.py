"""Confirms the bias-reconstruction helpers relocated from
analyze_gareus_mbar.py into gareus.mbar_analysis.bias are real delegations
(same object, not a coincidentally-equal reimplementation) and that the two
formula-bearing wrappers (_compute_u_nk_analytical,
_reconstruct_union_bias_block) still produce output identical -- within
floating-point tolerance, NOT bitwise -- to their pre-relocation selves on
the exact fixture data from tests/test_bias_reconstruction_nan_handling.py.
Not bitwise because the pre-relocation originals associate the
multiplication differently (_compute_u_nk_analytical precomputed
scale=beta*KJ_PER_KCAL once; _reconstruct_union_bias_block multiplied
beta*KJ_PER_KCAL into each term separately; the unified
reconstruct_bias_matrix accumulates in kcal and multiplies by beta once at
the end) -- floating-point arithmetic is not associative, so this differs
at the ~1e-16-relative level. Use assert_allclose/pytest.approx below, never
np.array_equal -- proving the delegation to
gareus.query.reconstruct_bias_matrix changed zero behavior for these two
(they were already correct before this relocation)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

import analyze_gareus_mbar as agm
import gareus.mbar_analysis.bias as bias_mod


def test_bias_module_is_importable():
    import gareus.mbar_analysis.bias  # noqa: F401


@pytest.mark.parametrize("name", [
    "_compute_u_nk_analytical",
    "_parse_epoch_window_map_native_params",
    "_epoch_bias_param_vectors",
    "_reconstruct_union_bias_block",
])
def test_analyze_gareus_mbar_reexports_the_same_object(name):
    assert getattr(agm, name) is getattr(bias_mod, name)


def test_compute_u_nk_analytical_matches_pre_relocation_formula():
    beta = 0.4
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([-2.0, -2.0])
    union_windows = [{
        'primary_center': 0.0, 'primary_k_kcal': 10.0,
        'secondary_cv_center': -2.1337, 'secondary_k_kcal': 11.87,
    }]

    u = agm._compute_u_nk_analytical(cv1, cv2, union_windows, beta)

    d1 = cv1 - 0.0
    d2 = cv2 - (-2.1337)
    expected = (beta * agm.KJ_PER_KCAL * 0.5 * 10.0 * d1 ** 2
                + beta * agm.KJ_PER_KCAL * 0.5 * 11.87 * d2 ** 2)
    np.testing.assert_allclose(u[:, 0], expected)


def test_compute_u_nk_analytical_no_restraint_window_still_finite():
    """Regression: the delegation must preserve the already-correct
    window-side guard (NaN secondary_k_kcal must not poison the row)."""
    beta = 0.4
    cv1 = np.array([0.0, 1.0, 2.0])
    cv2 = np.array([5.0, -3.0, np.nan])
    union_windows = [{
        'primary_center': 0.0, 'primary_k_kcal': 10.0,
        'secondary_cv_center': np.nan, 'secondary_k_kcal': np.nan,
    }]

    u = agm._compute_u_nk_analytical(cv1, cv2, union_windows, beta)

    assert np.all(np.isfinite(u))
    d1 = cv1 - 0.0
    expected = beta * agm.KJ_PER_KCAL * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(u[:, 0], expected)


def test_reconstruct_union_bias_block_matches_pre_relocation_formula():
    beta = 0.4
    cv = np.array([0.0, 0.0])
    cv2 = np.array([np.nan, -2.0])
    pc = np.array([0.0]); pk = np.array([10.0])
    sc = np.array([-2.1337]); sk = np.array([11.87])

    u = agm._reconstruct_union_bias_block(cv, cv2, beta, pc, pk, sc, sk)

    assert np.isnan(u[0, 0])
    assert np.isfinite(u[1, 0])
    d1 = cv[1] - pc[0]
    d2 = cv2[1] - sc[0]
    expected = beta * agm.KJ_PER_KCAL * 0.5 * pk[0] * d1 * d1 + beta * agm.KJ_PER_KCAL * 0.5 * sk[0] * d2 * d2
    assert u[1, 0] == pytest.approx(expected)
