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


# F10: observation identity validator (phase/source, replica, step) --------------------------------------------

from gareus.correctness.observation_keys import ObservationKeyRefusal, validate_observation_keys  # noqa: E402


def test_valid_keys_return_int64_and_phases_separate_equal_steps():
    ph, rep, st = validate_observation_keys(["a", "b", "a"], np.array([0, 0, 1], dtype=np.uint16), [5, 5, 5])
    assert rep.dtype == st.dtype == ph.dtype == np.int64 and ph.tolist() == [0, 1, 0]
    validate_observation_keys(None, np.array([0.0, 1.0]), np.array([3.0, 3.0]))   # integral floats ok


def test_identical_and_conflicting_duplicates_refuse():
    for segs in (["s1", "s1", "s2"], ["s1", "s2", "s2"]):
        with pytest.raises(ObservationKeyRefusal, match="duplicate"):
            validate_observation_keys(None, [0, 1, 0], [10, 10, 10], segment_ids=segs)
    with pytest.raises(ObservationKeyRefusal, match="duplicate"):
        validate_observation_keys(["a", "a"], [0, 0], [7, 7])


@pytest.mark.parametrize("replica,step,match", [
    (None, [1], "missing"), ([0], None, "missing"),
    (np.ma.masked_array([0, 1], mask=[False, True]), [1, 2], "masked"),
    ([0, 1], np.ma.masked_array([1, 2], mask=[True, False]), "masked"),
    ([0.0, 1.0], [1.5, 2.0], "fractional"),
    ([0, 1], [1, -2], "negative"),
    ([0, 1], [1.0, 2.0 ** 63], "int64"),
    ([0, 1], np.array([1, 2 ** 64 - 1], dtype=np.uint64), "int64"),
    ([0, np.nan], [1, 2], "non-finite"),
    ([0, 1], ["a", "b"], "integer-valued"),
])
def test_malformed_keys_refuse(replica, step, match):
    with pytest.raises(ObservationKeyRefusal, match=match):
        validate_observation_keys(None, replica, step)


def test_masked_phase_refuses():
    with pytest.raises(ObservationKeyRefusal, match="masked"):
        validate_observation_keys(np.ma.masked_array(["a", "b"], mask=[False, True]), [0, 1], [1, 2])


def test_pool_refuses_missing_key_and_added_duplicate_changes_nothing():
    from gareus.auxiliary_cv.offline import refuse_duplicate_sample_keys
    s = {"step": np.array([1, 1, 2, 2]), "replica": np.array([0, 1, 0, 1]),
         "window_id": np.array([0, 0, 1, 1]), "segment_id": np.array(["a"] * 4, dtype=object)}
    refuse_duplicate_sample_keys(s)
    assert np.bincount(s["window_id"]).tolist() == [2, 2]
    grown = {k: np.concatenate([v, v[:1]]) for k, v in s.items()}
    with pytest.raises(IntegrityError, match="duplicate"):
        refuse_duplicate_sample_keys(grown)
    with pytest.raises(ObservationKeyRefusal, match="missing"):
        refuse_duplicate_sample_keys({"step": s["step"]})
    with pytest.raises(ObservationKeyRefusal, match="missing"):
        refuse_duplicate_sample_keys({})
    refuse_duplicate_sample_keys({**s, "phase_id": np.array(["p", "p", "q", "q"])})


def test_flag_off_load_parquet_reaches_no_new_code(tmp_path, monkeypatch):
    """A non-aux run never calls the identity validator or pool_aux_segments."""
    import gareus.auxiliary_cv.offline as offline
    import gareus.correctness.observation_keys as ok
    import gareus.mbar_analysis.loaders as loaders
    calls = []
    monkeypatch.setattr(ok, "validate_observation_keys", lambda *a, **k: calls.append("v"))
    monkeypatch.setattr(offline, "pool_aux_segments", lambda *a, **k: calls.append("p"))
    import gareus.kernel_identity as ki
    monkeypatch.setattr(ki, "run_has_aux", lambda prod: False)
    import gareus.query as q
    sentinel = RuntimeError("stop after the aux gate")
    monkeypatch.setattr(q, "load_samples", lambda *a, **k: {"cv1": np.zeros(1)})
    monkeypatch.setattr(q, "load_windows", lambda *a, **k: (_ for _ in ()).throw(sentinel))
    with pytest.raises(RuntimeError, match="stop after the aux gate"):
        loaders.load_parquet(tmp_path)
    assert calls == []
