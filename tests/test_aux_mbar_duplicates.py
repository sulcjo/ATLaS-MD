import numpy as np
import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.adaptive.mbar_solve import solve_rows
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.duplicates import merge_identical_hamiltonians
from gareus.correctness._io import IntegrityError
from gareus.correctness.state_identity import hamiltonian_sha256

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))


def _defn():
    spec = [(-1.0, 4.0, 0.0, 0.0, "ordinary"), (0.0, 4.0, 0.0, 0.0, "ordinary"),
            (1.0, 4.0, 0.0, 0.0, "ordinary"), (0.0, 4.0, 0.0, 0.0, "sham")]
    return definition(rows(spec, MODEL.model_sha256), MODEL)


def _data(seed=4, counts=(1500, 2000, 1500, 900)):
    """Exact samples of harmonic umbrellas on a flat line; the sham (col 3) has its OWN, fewer samples."""
    rng = np.random.default_rng(seed)
    beta, k = 1.0, 4.0
    centers = np.array([-1.0, 0.0, 1.0, 0.0])
    x = np.concatenate([rng.normal(c, 1.0 / np.sqrt(beta * k), n) for c, n in zip(centers, counts)])
    origins = np.repeat(np.arange(4), counts)
    u = 0.5 * beta * k * (x[:, None] - centers[None, :]) ** 2
    return x, u, origins


def test_sham_and_parent_share_hamiltonian_id_and_merge():
    d = _defn()
    ids = [hamiltonian_sha256(d, w) for w in range(4)]
    assert ids[1] == ids[3] and len(set(ids)) == 3
    _x, u, origins = _data()
    merged_u, merged_origins, groups = merge_identical_hamiltonians(u, origins, ids)
    assert merged_u.shape == (u.shape[0], 3) and groups == [[0], [1, 3], [2]]
    f_sep, logw_sep = solve_rows(u, origins)
    f_mrg, logw_mrg = solve_rows(merged_u, merged_origins)
    assert f_sep[3] - f_sep[1] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(f_sep[[0, 1, 2]] - f_sep[0], f_mrg - f_mrg[0], atol=1e-8)
    w_sep = np.exp(logw_sep - logw_sep.max()); w_sep /= w_sep.sum()
    w_mrg = np.exp(logw_mrg - logw_mrg.max()); w_mrg /= w_mrg.sum()
    np.testing.assert_allclose(w_sep, w_mrg, rtol=1e-8, atol=1e-14)


def test_merge_refuses_non_identical_columns_with_same_id():
    _x, u, origins = _data()
    u = u.copy()
    u[0, 3] += 1e-9
    with pytest.raises(IntegrityError, match="bitwise"):
        merge_identical_hamiltonians(u, origins, ["a", "b", "c", "b"])
