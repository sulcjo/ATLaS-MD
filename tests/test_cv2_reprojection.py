"""Tests for evaluating a run's other secondary-CV definition on its frames.

These are all synthetic: no test reads a real run. The point of the module is
that a CV2 definition is an affine map of backbone-torsion features, so a
synthetic model plus synthetic features pins every behaviour that matters --
the projection convention, the feature column order, the validation gate's
verdict, the exact-step join, and per-column regime selection in the bias
matrix.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gareus.mbar_analysis.bias import (  # noqa: E402
    _reconstruct_union_bias_block,
    _reconstruct_union_bias_block_per_regime,
)
from gareus.mbar_analysis.cv2_reprojection import (  # noqa: E402
    Cv2Model,
    join_on_step,
    load_cv2_model,
    load_stored_obs,
    project_cv2,
    reproject_stored_obs,
    validate_model_against_stored_obs,
)
from gareus.tica import backbone_dihedral_features  # noqa: E402

# Two phi and two psi torsions -> 2*(2+2) = 8 sin/cos features.
PHI = [[0, 1, 2, 3], [1, 2, 3, 4]]
PSI = [[2, 3, 4, 5], [3, 4, 5, 6]]
N_FEATURES = 2 * (len(PHI) + len(PSI))


def _model_dict(weights, mean, method="pca", **extra):
    weights = np.asarray(weights, dtype=float)
    mean = np.asarray(mean, dtype=float)
    d = {
        "weights": weights.tolist(),
        "mean": mean.tolist(),
        "offset": float(-mean @ weights),
        "eigenvalue": 0.5,
        "method": method,
        "phi_torsion_indices": PHI,
        "psi_torsion_indices": PSI,
        "n_samples": 100,
    }
    d.update(extra)
    return d


def _write_model(tmp_path, name, weights, mean, method="pca", **extra):
    p = tmp_path / name
    p.write_text(json.dumps(_model_dict(weights, mean, method, **extra)))
    return p


def _rand_features(rng, n):
    """Feature rows that are genuine sin/cos pairs, as the real ones are."""
    ang = rng.uniform(-np.pi, np.pi, size=(n, N_FEATURES // 2))
    out = np.empty((n, N_FEATURES))
    out[:, 0::2] = np.sin(ang)
    out[:, 1::2] = np.cos(ang)
    return out


# --- model loading ----------------------------------------------------------

def test_load_cv2_model_names_regime_from_method(tmp_path):
    """The two model files differ only by 'method'; the regime name follows the
    run's own naming so a reprojected column can be matched to a regime."""
    rng = np.random.default_rng(0)
    pca = load_cv2_model(_write_model(tmp_path, "pca.json", rng.normal(size=N_FEATURES),
                                       rng.normal(size=N_FEATURES), method="pca"))
    tica = load_cv2_model(_write_model(tmp_path, "tica.json", rng.normal(size=N_FEATURES),
                                        rng.normal(size=N_FEATURES), method="tica"))
    assert pca.regime == "torsion-pca"
    assert tica.regime == "tica-linear"
    assert pca.n_features == tica.n_features == N_FEATURES


def test_load_cv2_model_explicit_regime_overrides(tmp_path):
    rng = np.random.default_rng(1)
    m = load_cv2_model(_write_model(tmp_path, "m.json", rng.normal(size=N_FEATURES),
                                     rng.normal(size=N_FEATURES)), regime="custom")
    assert m.regime == "custom"


def test_load_cv2_model_rejects_weight_torsion_mismatch(tmp_path):
    """Weights must match 2*(n_phi + n_psi). A mismatch means the stored model
    and its torsion indices disagree, and any projection would silently use
    the wrong columns."""
    rng = np.random.default_rng(2)
    bad = _model_dict(rng.normal(size=N_FEATURES), rng.normal(size=N_FEATURES))
    bad["weights"] = bad["weights"][:-2]
    bad["mean"] = bad["mean"][:-2]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="features implied by"):
        load_cv2_model(p)


def test_load_cv2_model_rejects_missing_torsion_indices(tmp_path):
    """Without torsion indices the features cannot be recomputed at all."""
    rng = np.random.default_rng(3)
    d = _model_dict(rng.normal(size=N_FEATURES), rng.normal(size=N_FEATURES))
    d["phi_torsion_indices"] = []
    d["psi_torsion_indices"] = []
    d["weights"] = d["weights"]
    p = tmp_path / "noidx.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="no torsion indices"):
        load_cv2_model(p)


# --- projection -------------------------------------------------------------

def test_project_cv2_is_the_centered_affine_map(tmp_path):
    """The stored 'offset' already absorbs -mean@weights, so X@w + offset and
    (X - mean)@w must agree. Getting this convention wrong is a silent
    constant shift in the CV, which is exactly what the gate exists to catch."""
    rng = np.random.default_rng(4)
    w = rng.normal(size=N_FEATURES)
    mu = rng.normal(size=N_FEATURES)
    model = load_cv2_model(_write_model(tmp_path, "m.json", w, mu))
    X = _rand_features(rng, 50)
    np.testing.assert_allclose(project_cv2(model, X), (X - mu) @ w, rtol=0, atol=1e-12)


def test_project_cv2_rejects_wrong_feature_width(tmp_path):
    rng = np.random.default_rng(5)
    model = load_cv2_model(_write_model(tmp_path, "m.json", rng.normal(size=N_FEATURES),
                                         rng.normal(size=N_FEATURES)))
    with pytest.raises(ValueError, match="columns but the model"):
        project_cv2(model, _rand_features(rng, 10)[:, :-2])
    with pytest.raises(ValueError, match="must be 2-D"):
        project_cv2(model, np.zeros(N_FEATURES))


def test_project_cv2_matches_the_canonical_featurizer(tmp_path):
    """End-to-end through the SAME featurizer the run used, so the interleaved
    (sin, cos) per-torsion column order is pinned rather than assumed."""
    rng = np.random.default_rng(6)
    w = rng.normal(size=N_FEATURES)
    mu = np.zeros(N_FEATURES)
    model = load_cv2_model(_write_model(tmp_path, "m.json", w, mu))
    positions = rng.normal(size=(8, 3))
    feats = backbone_dihedral_features(positions, [tuple(q) for q in PHI],
                                        [tuple(q) for q in PSI])
    assert feats.size == N_FEATURES
    np.testing.assert_allclose(project_cv2(model, feats[None, :])[0],
                               float(feats @ w), rtol=0, atol=1e-12)


# --- the validation gate ----------------------------------------------------

def _obs_npz(tmp_path, name, features, secondary_cv, steps=None, window=None):
    n = features.shape[0]
    p = tmp_path / name
    np.savez(p, features=features, secondary_cv=secondary_cv,
             steps=(np.arange(n) * 100 if steps is None else steps),
             window=(np.zeros(n, dtype=np.int64) if window is None else window),
             primary_cv=np.zeros(n))
    return Path(str(p) + ".npz") if not str(p).endswith(".npz") else p


def test_gate_agrees_for_the_model_that_produced_the_stored_cv(tmp_path):
    rng = np.random.default_rng(7)
    w = rng.normal(size=N_FEATURES)
    mu = rng.normal(size=N_FEATURES)
    model = load_cv2_model(_write_model(tmp_path, "m.json", w, mu))
    X = _rand_features(rng, 200)
    stored = (X - mu) @ w
    obs = load_stored_obs([_obs_npz(tmp_path, "obs0.npz", X, stored)])
    res = validate_model_against_stored_obs(model, obs)
    assert res["agrees"] is True
    assert res["n"] == 200
    assert res["max_abs_deviation"] < 1e-12
    assert res["correlation"] == pytest.approx(1.0)


def test_gate_rejects_a_shifted_or_reordered_model(tmp_path):
    """Two realistic ways to get this wrong: forgetting that offset already
    centres the features, and permuting the sin/cos column order."""
    rng = np.random.default_rng(8)
    w = rng.normal(size=N_FEATURES)
    mu = rng.normal(size=N_FEATURES)
    model = load_cv2_model(_write_model(tmp_path, "m.json", w, mu))
    X = _rand_features(rng, 200)

    uncentered = load_stored_obs([_obs_npz(tmp_path, "o1.npz", X, X @ w)])
    assert validate_model_against_stored_obs(model, uncentered)["agrees"] is False

    perm = np.arange(N_FEATURES)[::-1]
    reordered = load_stored_obs([_obs_npz(tmp_path, "o2.npz", X, (X[:, perm] - mu) @ w)])
    assert validate_model_against_stored_obs(model, reordered)["agrees"] is False


def test_gate_reports_no_finite_pairs_rather_than_crashing(tmp_path):
    rng = np.random.default_rng(9)
    model = load_cv2_model(_write_model(tmp_path, "m.json", rng.normal(size=N_FEATURES),
                                         rng.normal(size=N_FEATURES)))
    X = _rand_features(rng, 10)
    obs = load_stored_obs([_obs_npz(tmp_path, "o.npz", X, np.full(10, np.nan))])
    res = validate_model_against_stored_obs(model, obs)
    assert res["agrees"] is False and res["n"] == 0


def test_reproject_stored_obs_emits_one_column_per_regime(tmp_path):
    rng = np.random.default_rng(10)
    a = load_cv2_model(_write_model(tmp_path, "a.json", rng.normal(size=N_FEATURES),
                                     np.zeros(N_FEATURES), method="pca"))
    b = load_cv2_model(_write_model(tmp_path, "b.json", rng.normal(size=N_FEATURES),
                                     np.zeros(N_FEATURES), method="tica"))
    X = _rand_features(rng, 40)
    obs = load_stored_obs([_obs_npz(tmp_path, "o.npz", X, np.zeros(40))])
    tbl = reproject_stored_obs([a, b], obs)
    assert set(tbl) == {"steps", "window", "cv2_torsion-pca", "cv2_tica-linear"}
    np.testing.assert_allclose(tbl["cv2_torsion-pca"], project_cv2(a, X))
    np.testing.assert_allclose(tbl["cv2_tica-linear"], project_cv2(b, X))


# --- the exact-step join ----------------------------------------------------

def test_join_on_step_is_exact_and_nans_the_rest():
    """Only ~10% of sample rows have a stored feature row, so most targets have
    no source. Those must be NaN, never a nearby frame's value: the two CV2
    definitions are near-orthogonal, so interpolation has no justification."""
    src_steps = np.array([100, 300, 500])
    src_vals = np.array([1.0, 3.0, 5.0])
    target = np.array([100, 200, 300, 400, 500, 600])
    vals, matched = join_on_step(target, src_steps, src_vals)
    np.testing.assert_array_equal(matched, [True, False, True, False, True, False])
    np.testing.assert_allclose(vals[matched], [1.0, 3.0, 5.0])
    assert np.isnan(vals[~matched]).all()


def test_join_on_step_handles_unsorted_and_duplicate_sources():
    vals, matched = join_on_step(np.array([5, 1, 9]),
                                  np.array([9, 1, 1, 5]),
                                  np.array([90.0, 10.0, 11.0, 50.0]))
    assert matched.all()
    np.testing.assert_allclose(vals, [50.0, 10.0, 90.0])


def test_join_on_step_empty_source_matches_nothing():
    vals, matched = join_on_step(np.array([1, 2]), np.array([], dtype=np.int64),
                                  np.array([]))
    assert not matched.any() and np.isnan(vals).all()


def test_join_on_step_rejects_length_mismatch():
    with pytest.raises(ValueError, match="differ in length"):
        join_on_step(np.array([1]), np.array([1, 2]), np.array([1.0]))


# --- per-column regime selection in the bias matrix -------------------------

def _bias_inputs(rng, n, K):
    return (rng.normal(size=n),                       # cv
            rng.uniform(-1, 1, size=K),               # primary centers
            rng.uniform(10, 50, size=K),              # primary k
            rng.uniform(-1, 1, size=K),               # secondary centers
            rng.uniform(10, 50, size=K))              # secondary k


def test_per_regime_block_reduces_to_the_single_cv2_block():
    """With one regime it must be byte-identical to the existing builder --
    single-regime runs (the common case) cannot change behaviour."""
    rng = np.random.default_rng(11)
    n, K, beta = 60, 4, 0.4
    cv, pc, pk, sc, sk = _bias_inputs(rng, n, K)
    cv2 = rng.normal(size=n)
    ref = _reconstruct_union_bias_block(cv, cv2, beta, pc, pk, sc, sk)
    got = _reconstruct_union_bias_block_per_regime(
        cv, {"only": cv2}, beta, pc, pk, sc, sk, ["only"] * K)
    np.testing.assert_array_equal(got, ref)


def test_per_regime_block_uses_each_columns_own_cv2():
    """Each column must be evaluated in the CV2 definition its own secondary
    params were written for -- that is what makes the pooled u_nk one
    Hamiltonian instead of a splice."""
    rng = np.random.default_rng(12)
    n, K, beta = 60, 4, 0.4
    cv, pc, pk, sc, sk = _bias_inputs(rng, n, K)
    cv2_a = rng.normal(size=n)
    cv2_b = rng.normal(size=n)
    regimes = ["a", "b", "a", "b"]

    got = _reconstruct_union_bias_block_per_regime(
        cv, {"a": cv2_a, "b": cv2_b}, beta, pc, pk, sc, sk, regimes)
    all_a = _reconstruct_union_bias_block(cv, cv2_a, beta, pc, pk, sc, sk)
    all_b = _reconstruct_union_bias_block(cv, cv2_b, beta, pc, pk, sc, sk)

    for k, r in enumerate(regimes):
        expected = all_a if r == "a" else all_b
        np.testing.assert_array_equal(got[:, k], expected[:, k])
    # And it is genuinely mixed, not silently one of them.
    assert not np.array_equal(got, all_a)
    assert not np.array_equal(got, all_b)


def test_per_regime_block_propagates_nan_for_unrecoverable_rows():
    """A row whose foreign CV2 was never recoverable (no frame at that step)
    must stay NaN so clean() drops it, rather than being biased to zero."""
    rng = np.random.default_rng(13)
    n, K, beta = 20, 2, 0.4
    cv, pc, pk, sc, sk = _bias_inputs(rng, n, K)
    cv2_a = rng.normal(size=n)
    cv2_b = rng.normal(size=n)
    cv2_b[3] = np.nan
    got = _reconstruct_union_bias_block_per_regime(
        cv, {"a": cv2_a, "b": cv2_b}, beta, pc, pk, sc, sk, ["a", "b"])
    assert np.isfinite(got[3, 0])
    assert not np.isfinite(got[3, 1])


def test_per_regime_block_rejects_unknown_regime_and_bad_length():
    rng = np.random.default_rng(14)
    n, K, beta = 10, 3, 0.4
    cv, pc, pk, sc, sk = _bias_inputs(rng, n, K)
    cv2 = rng.normal(size=n)
    with pytest.raises(ValueError, match="no cv2 supplied for regime"):
        _reconstruct_union_bias_block_per_regime(
            cv, {"a": cv2}, beta, pc, pk, sc, sk, ["a", "b", "a"])
    with pytest.raises(ValueError, match="state_regimes has"):
        _reconstruct_union_bias_block_per_regime(
            cv, {"a": cv2}, beta, pc, pk, sc, sk, ["a", "a"])


# --- loader wiring ----------------------------------------------------------

from gareus.mbar_analysis.cv2_reprojection import CV2_REPROJECTION_FILENAME  # noqa: E402
from gareus.mbar_analysis.loaders_union_parquet import (  # noqa: E402
    _load_cv2_reprojection,
    _per_regime_bias_block,
)


def _write_reproj_table(tmp_path, steps, columns):
    """Write a cv2_reprojected.parquet the way reproject_cv2.py does."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = {'steps': pa.array(np.asarray(steps, dtype=np.int64))}
    for name, vals in columns.items():
        data[f'cv2_{name}'] = pa.array(np.asarray(vals, dtype=np.float64))
    pq.write_table(pa.table(data), str(tmp_path / CV2_REPROJECTION_FILENAME))


def test_load_cv2_reprojection_absent_is_none(tmp_path):
    """The table is an optional side-car; a run without one must load normally."""
    assert _load_cv2_reprojection(tmp_path) is None


def test_load_cv2_reprojection_unreadable_is_none(tmp_path):
    """Unreadable is treated as missing rather than aborting the analysis."""
    (tmp_path / CV2_REPROJECTION_FILENAME).write_text('not a parquet file')
    assert _load_cv2_reprojection(tmp_path) is None


def test_load_cv2_reprojection_reads_columns_and_drops_nonfinite(tmp_path):
    """NaN rows are frames whose foreign cv2 was never recoverable; they must
    not enter the table's step index as if they had a value."""
    _write_reproj_table(tmp_path, [10, 20, 30],
                        {'torsion-pca': [1.0, np.nan, 3.0],
                         'tica-linear': [4.0, 5.0, 6.0]})
    got = _load_cv2_reprojection(tmp_path)
    assert set(got) == {'torsion-pca', 'tica-linear'}
    np.testing.assert_array_equal(got['torsion-pca']['steps'], [10, 30])
    np.testing.assert_allclose(got['torsion-pca']['cv2'], [1.0, 3.0])
    np.testing.assert_array_equal(got['tica-linear']['steps'], [10, 20, 30])


def test_per_regime_block_prefers_recorded_cv2_for_the_epochs_own_regime(tmp_path):
    """The epoch's own regime uses the cv2 recorded while those frames ran --
    exact for every row -- rather than a reprojection that only covers the
    rows whose step was recoverable."""
    rng = np.random.default_rng(20)
    n, K, beta = 12, 2, 0.4
    cv, pc, pk, sc, sk = _bias_inputs(rng, n, K)
    steps = np.arange(n, dtype=np.int64) * 100
    recorded = rng.normal(size=n)
    foreign = rng.normal(size=n)
    reproj = {'other': {'steps': steps, 'cv2': foreign},
              # A deliberately WRONG value for the own regime: if the block
              # used the table here instead of the recorded array, the
              # comparison below would fail.
              'own': {'steps': steps, 'cv2': recorded + 100.0}}

    block, cov = _per_regime_bias_block(cv, steps, beta, pc, pk, sc, sk,
                                         ['own', 'other'], reproj, recorded, 'own')
    assert block is not None
    assert cov == {'own': 1.0, 'other': 1.0}
    expected = _reconstruct_union_bias_block_per_regime(
        cv, {'own': recorded, 'other': foreign}, beta, pc, pk, sc, sk, ['own', 'other'])
    np.testing.assert_array_equal(block, expected)


def test_per_regime_block_reports_partial_foreign_coverage(tmp_path):
    """Only ~10% of sample rows have a recoverable foreign cv2 on the
    motivating run, so coverage must be reported, not assumed to be 1.0."""
    rng = np.random.default_rng(21)
    n, K, beta = 10, 2, 0.4
    cv, pc, pk, sc, sk = _bias_inputs(rng, n, K)
    steps = np.arange(n, dtype=np.int64) * 100
    recorded = rng.normal(size=n)
    reproj = {'other': {'steps': steps[:4], 'cv2': rng.normal(size=4)}}

    block, cov = _per_regime_bias_block(cv, steps, beta, pc, pk, sc, sk,
                                         ['own', 'other'], reproj, recorded, 'own')
    assert block is not None
    assert cov['other'] == pytest.approx(0.4)
    # Uncovered rows stay NaN in the foreign column so clean() drops them.
    assert np.isfinite(block[:4, 1]).all()
    assert not np.isfinite(block[4:, 1]).any()
    assert np.isfinite(block[:, 0]).all()


def test_per_regime_block_declines_when_a_regime_is_missing(tmp_path):
    """Rather than filling a needed regime with guesses, decline and let the
    caller keep the existing single-cv2 behaviour."""
    rng = np.random.default_rng(22)
    n, K, beta = 8, 2, 0.4
    cv, pc, pk, sc, sk = _bias_inputs(rng, n, K)
    steps = np.arange(n, dtype=np.int64) * 100
    block, cov = _per_regime_bias_block(cv, steps, beta, pc, pk, sc, sk,
                                         ['own', 'absent'], {}, rng.normal(size=n), 'own')
    assert block is None and cov == {}
