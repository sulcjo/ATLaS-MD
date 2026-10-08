# tests/test_aux_cv_production_wiring.py
import ast
import inspect

import numpy as np
import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import AuxObservationError, aux_bias_matrix_kcal
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.production import assemble_bias_matrices

M = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
TABLE = AuxStateTable(M, (0.0, 1.0, 0.0), (0.0, 2.0, 0.0), (None, None, None))


def test_aux_matrix_values_and_exact_zero_rows():
    z = np.array([0.5, 1.5, -1.0, 0.25])
    out = aux_bias_matrix_kcal(z, TABLE)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_array_equal(out[2], 0.0)
    np.testing.assert_allclose(out[1], 0.5 * 2.0 * (z - 1.0) ** 2, rtol=1e-15)


def test_nonfinite_z_is_fatal_when_any_state_is_active():
    with pytest.raises(AuxObservationError):
        aux_bias_matrix_kcal(np.array([0.5, np.nan]), TABLE)


def test_all_zero_table_ignores_z():
    """Sham arm: observe_aux_z may record NaN; with no active state the matrix is exact zeros."""
    zero = AuxStateTable(M, (0.0, 0.0), (0.0, 0.0), (None, None))
    np.testing.assert_array_equal(aux_bias_matrix_kcal(np.array([np.nan, 1.0]), zero), 0.0)


def test_assemble_without_aux_is_bitwise_legacy():
    rng = np.random.default_rng(1)
    d, s, b = rng.normal(size=(3, 3)), rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    legacy_kcal = d + s + b / 4.184
    kcal, kj = assemble_bias_matrices(d, s, b)
    assert np.array_equal(kcal, legacy_kcal) and np.array_equal(kj, 4.184 * legacy_kcal)
    kcal0, _ = assemble_bias_matrices(d, s, b, aux_bias_kcal=np.zeros((3, 3)))
    assert np.array_equal(kcal0, kcal)


def test_assemble_adds_aux():
    rng = np.random.default_rng(2)
    d, s, b, a = (rng.normal(size=(2, 2)) for _ in range(4))
    kcal, kj = assemble_bias_matrices(d, s, b, aux_bias_kcal=a)
    np.testing.assert_allclose(kcal, d + s + a + b / 4.184, rtol=1e-15)
    np.testing.assert_allclose(kj, 4.184 * kcal, rtol=1e-15)


def _run_gareus_source():
    import gareus.production as production
    return ast.parse(inspect.getsource(production.run_gareus))


def _func(tree, name):
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(hits) == 1, f"{name} not found exactly once in run_gareus; re-anchor"
    return hits[0]


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id == name)
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == name))]


def test_both_fetch_closures_observe_z_through_one_helper():
    tree = _run_gareus_source()
    for name in ("_fetch_state", "_fetch_exchange_state"):
        assert _calls(_func(tree, name), "_aux_z_for_replica"), f"{name} does not observe aux z"


def test_sample_and_exchange_assembly_both_include_the_aux_matrix():
    tree = _run_gareus_source()
    exch = _func(tree, "_current_exchange_arrays")
    assert _calls(exch, "aux_bias_matrix_kcal")
    assert any(k.arg == "aux_bias_kcal" for c in _calls(exch, "assemble_bias_matrices") for k in c.keywords)
    all_assemble = _calls(tree, "assemble_bias_matrices")
    assert len(all_assemble) == 2 and all(any(k.arg == "aux_bias_kcal" for k in c.keywords) for c in all_assemble)
    assert len(_calls(tree, "aux_bias_matrix_kcal")) == 2


def test_sampled_umbrella_bias_stays_umbrella_only():
    src = inspect.getsource(__import__("gareus.production", fromlist=["run_gareus"]).run_gareus)
    line = [l for l in src.splitlines() if "sampled_umbrella_bias_kj = float(" in l]
    assert line and "aux" not in line[0], "sampled_umbrella_bias_kj must remain primary + secondary only"
