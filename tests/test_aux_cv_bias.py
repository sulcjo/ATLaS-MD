# tests/test_aux_cv_bias.py
import numpy as np
import pytest

from gareus.correctness._io import IntegrityError
from gareus.correctness.bias import MissingCoordinateError, reconstruct_bias_matrix

SHA = "e" * 64
BETA = 1.0 / (0.0083144626 * 300.0)


def _rows():
    return [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0},
            {"window_id": 1, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
             "aux_model_sha256": SHA, "aux_center": 1.0, "aux_k": 2.0},
            {"window_id": 2, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
             "aux_k": 0.0}]


def test_aux_term_added_in_lambda_zero_path():
    cv1 = np.array([0.1, 0.3])
    z = np.array([0.5, 2.0])
    u = reconstruct_bias_matrix(cv1, None, _rows(), BETA, aux_z={SHA: z})
    extra = BETA * 4.184 * 0.5 * 2.0 * (z - 1.0) ** 2
    np.testing.assert_allclose(u[:, 1] - u[:, 0], extra, rtol=1e-12)
    np.testing.assert_array_equal(u[:, 2], u[:, 0])


def test_missing_model_column_raises():
    with pytest.raises(MissingCoordinateError, match="aux"):
        reconstruct_bias_matrix(np.array([0.1]), None, _rows(), BETA)


def test_nan_z_only_poisons_active_aux_column():
    u = reconstruct_bias_matrix(np.array([0.1, 0.3]), None, _rows(), BETA,
                                aux_z={SHA: np.array([np.nan, 1.0])})
    assert np.isnan(u[0, 1]) and np.isfinite(u[0, 0]) and np.isfinite(u[0, 2])
    assert np.isfinite(u[1]).all()


def test_zero_strength_rows_equal_legacy_matrix_bitwise():
    # Regression guard (passes before and after this task): inactive aux rows change nothing.
    legacy = [{k: v for k, v in r.items() if not k.startswith("aux_")} for r in _rows()]
    rows = _rows()
    rows[1]["aux_k"] = 0.0
    cv1 = np.array([0.1, 0.3, 0.7])
    a = reconstruct_bias_matrix(cv1, None, legacy, BETA)
    b = reconstruct_bias_matrix(cv1, None, rows, BETA)
    assert np.array_equal(a, b)


def test_aux_z_shape_is_checked():
    with pytest.raises(IntegrityError, match="shape"):
        reconstruct_bias_matrix(np.array([0.1, 0.3]), None, _rows(), BETA, aux_z={SHA: np.array([1.0])})


def test_aux_term_is_present_before_the_ladder_boost_when_lambda_is_positive():
    rows = _rows()
    for row in rows:
        row["gamd_lambda"] = 0.5
    seen = {}

    def fake_ladder(matrix, pep, dih, lambdas, envelope, beta, meta):
        seen["matrix"] = matrix.copy()          # what the boost helper receives
        return matrix + 1.0                      # stand-in boost, same for every column

    cv1 = np.array([0.1, 0.3])
    z = np.array([0.5, 2.0])
    u = reconstruct_bias_matrix(cv1, None, rows, BETA, v_pep=np.zeros(2), v_dih=np.zeros(2),
                                envelope=object(), aux_z={SHA: z}, _ladder_apply=fake_ladder)
    extra = BETA * 4.184 * 0.5 * 2.0 * (z - 1.0) ** 2
    np.testing.assert_allclose(seen["matrix"][:, 1] - seen["matrix"][:, 0], extra, rtol=1e-12)
    np.testing.assert_allclose(u[:, 1] - u[:, 0], extra, rtol=1e-12)
