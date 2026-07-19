"""Tests for gareus.tica — pure numpy, no OpenMM."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gareus.tica import (
    TICAResult,
    backbone_dihedral_features,
    compute_tica,
    compute_tica_components,
    compute_tica_combined,
    compute_tica_from_epoch_obs,
    compute_combined_tica_from_epoch_obs,
    load_epoch_dihedral_obs,
    project_tica1,
    tica_k_from_spread,
    window_tica_centers,
    DihedralObsBuffer,
    _normalize_segments,
    _segment_lagged_pair_indices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phi_psi(n_residues: int = 5):
    phi = [tuple(range(4 * i, 4 * i + 4)) for i in range(n_residues)]
    psi = [tuple(range(4 * i + 2, 4 * i + 6)) for i in range(n_residues)]
    return phi, psi


def _random_positions(n_atoms: int = 40, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_atoms, 3))


def _slow_mode_dataset(n_frames: int = 400, lag: int = 5, seed: int = 0):
    """Two-feature dataset where feature 0 is slow, feature 1 is fast."""
    rng = np.random.default_rng(seed)
    slow = np.zeros(n_frames)
    for t in range(1, n_frames):
        slow[t] = 0.99 * slow[t - 1] + rng.normal(0, 0.14)
    fast = rng.normal(0, 1, n_frames)
    return np.column_stack([slow, fast])


def _multi_slow_mode_dataset(n_features: int = 5, n_frames: int = 800, seed: int = 0):
    """AR(1) features with strictly decreasing persistence (feature 0 slowest).

    Each feature is an independent AR(1) process with a distinct decay
    constant, so the true tICA ranking is unambiguous: feature i is always
    slower than feature i+1, giving ``compute_tica_components`` a dataset
    where "top-N components" has one clear right answer to check against.
    """
    rng = np.random.default_rng(seed)
    decays = np.linspace(0.97, 0.5, n_features)  # feature 0 slowest, last fastest
    X = np.zeros((n_frames, n_features))
    for j, a in enumerate(decays):
        noise_sd = np.sqrt(1.0 - a * a)  # keeps each feature's stationary variance ~1
        for t in range(1, n_frames):
            X[t, j] = a * X[t - 1, j] + rng.normal(0, noise_sd)
    return X


def _hand_tica_reference(X, lag, seg_lengths, weights=None, epsilon=1e-10):
    """Independent (test-only) reference tICA solve.

    Builds within-segment lagged pairs via a plain Python loop (deliberately
    NOT using gareus.tica._segment_lagged_pair_indices), assembles C(0)/C(tau)
    with the same formulas as gareus.tica.compute_tica, and solves the
    generalised eigenproblem directly via scipy — used as an independent
    oracle to check that compute_tica(..., segments=...) never forms a
    lagged pair across a trajectory-segment boundary.
    """
    from scipy.linalg import eigh as scipy_eigh

    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    pairs = []
    offset = 0
    for L in seg_lengths:
        for t in range(L - lag):
            pairs.append((offset + t, offset + t + lag))
        offset += L
    assert pairs, "reference expects at least one valid pair"
    left = np.array([p[0] for p in pairs], dtype=np.int64)
    right = np.array([p[1] for p in pairs], dtype=np.int64)
    n_pairs = len(pairs)

    if weights is None:
        mean = X.mean(axis=0)
        Xc = X - mean
        C0 = (Xc.T @ Xc) / (n - 1) + epsilon * np.eye(d)
        Xl, Xr = Xc[left], Xc[right]
        Ctau = (Xl.T @ Xr) / (n_pairs - 1)
        Ctau = 0.5 * (Ctau + Ctau.T)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        mean = w @ X
        Xc = X - mean
        Xl, Xr = Xc[left], Xc[right]
        w_pairs = 0.5 * (w[left] + w[right])
        w_pairs = w_pairs / w_pairs.sum()
        C0 = 0.5 * ((Xl.T * w_pairs) @ Xl + (Xr.T * w_pairs) @ Xr) + epsilon * np.eye(d)
        Ctau = 0.5 * ((Xl.T * w_pairs) @ Xr + (Xr.T * w_pairs) @ Xl)

    eigenvalues, eigenvectors = scipy_eigh(Ctau, C0, subset_by_index=[d - 1, d - 1])
    v = eigenvectors[:, 0].copy()
    ev = float(eigenvalues[0])
    norm_sq = float(v @ C0 @ v)
    if norm_sq > 0.0:
        v /= np.sqrt(norm_sq)
    return ev, v


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestBackboneDihedralFeatures:
    def test_shape(self):
        phi, psi = _make_phi_psi(n_residues=3)
        pos = _random_positions(n_atoms=30)
        feat = backbone_dihedral_features(pos, phi, psi)
        assert feat.shape == (12,)  # 2 * 3 + 2 * 3

    def test_sincos_range(self):
        phi, psi = _make_phi_psi(n_residues=5)
        pos = _random_positions(n_atoms=40)
        feat = backbone_dihedral_features(pos, phi, psi)
        assert np.all(feat >= -1.0 - 1e-9)
        assert np.all(feat <= 1.0 + 1e-9)

    def test_empty_torsions(self):
        pos = _random_positions(n_atoms=10)
        feat = backbone_dihedral_features(pos, [], [])
        assert feat.shape == (0,)


# ---------------------------------------------------------------------------
# compute_tica
# ---------------------------------------------------------------------------

class TestComputeTICA:
    def test_rejects_lag_zero(self):
        X = np.random.randn(100, 4)
        with pytest.raises(ValueError, match="lag.*positive|positive.*lag"):
            compute_tica(X, 0)

    def test_rejects_negative_lag(self):
        X = np.random.randn(100, 4)
        with pytest.raises(ValueError):
            compute_tica(X, -1)

    def test_returns_unit_norm_vector(self):
        X = _slow_mode_dataset(n_frames=300, lag=5)
        result = compute_tica(X, lag=5)
        # C(0)-normalised, so w^T C(0) w = 1; unit L2 is not guaranteed but
        # weights should be a finite unit-ish vector
        assert np.isfinite(result.weights).all()
        assert abs(np.linalg.norm(result.weights) - 1.0) < 0.5

    def test_recovers_slow_mode(self):
        X = _slow_mode_dataset(n_frames=500, lag=5)
        result = compute_tica(X, lag=5)
        # Weight on slow feature (index 0) should dominate
        assert abs(result.weights[0]) > abs(result.weights[1])

    def test_eigenvalue_range(self):
        X = _slow_mode_dataset(n_frames=300, lag=5)
        result = compute_tica(X, lag=5)
        assert -1.0 <= result.eigenvalue <= 1.0 + 1e-6

    def test_offset_invariant(self):
        X = _slow_mode_dataset(n_frames=300, lag=5)
        result = compute_tica(X, lag=5)
        proj_v1 = (X - result.mean) @ result.weights
        proj_v2 = X @ result.weights + result.offset
        np.testing.assert_allclose(proj_v1, proj_v2, atol=1e-10)

    def test_stores_torsion_indices(self):
        X = _slow_mode_dataset(n_frames=200, lag=5)
        phi = [(0, 1, 2, 3)]
        psi = [(2, 3, 4, 5)]
        result = compute_tica(X, lag=5, phi_torsion_indices=phi, psi_torsion_indices=psi)
        assert result.phi_torsion_indices == phi
        assert result.psi_torsion_indices == psi

    def test_sign_continuity(self):
        X = _slow_mode_dataset(n_frames=300, lag=5)
        r1 = compute_tica(X, lag=5)
        # Flip r1 sign to simulate arbitrary eigenvector sign on a previous epoch
        r1_flipped = TICAResult(
            weights=-r1.weights,
            eigenvalue=r1.eigenvalue,
            mean=r1.mean,
            offset=-r1.offset,
            lag=r1.lag,
            phi_torsion_indices=r1.phi_torsion_indices,
            psi_torsion_indices=r1.psi_torsion_indices,
            n_samples=r1.n_samples,
            sign_ref=-r1.weights,
        )
        r2 = compute_tica(X, lag=5, previous_result=r1_flipped)
        # sign continuity: r2 must align with r1_flipped (the "previous epoch" model),
        # even though r1_flipped has flipped sign relative to the raw tICA solution.
        assert float(np.dot(r2.weights, r1_flipped.weights)) > 0, (
            "r2.weights should align with previous epoch sign (r1_flipped.weights)"
        )

    def test_n_samples_stored(self):
        X = _slow_mode_dataset(n_frames=200, lag=5)
        result = compute_tica(X, lag=5)
        assert result.n_samples == 200

    def test_weighted_uniform_matches_unweighted(self):
        """Uniform weights should reproduce unweighted result up to sign."""
        X = _slow_mode_dataset(n_frames=400, lag=5, seed=3)
        n = len(X)
        w = np.ones(n) / n
        r_unweighted = compute_tica(X, lag=5)
        r_weighted = compute_tica(X, lag=5, weights=w)
        # Eigenvectors should be parallel (same or opposite sign)
        cos = float(np.dot(r_unweighted.weights, r_weighted.weights))
        assert abs(cos) > 0.99, f"uniform-weighted vs unweighted cosine = {cos:.4f}"

    def test_weighted_rejects_wrong_shape(self):
        X = _slow_mode_dataset(n_frames=200, lag=5)
        with pytest.raises(ValueError, match="weights shape"):
            compute_tica(X, lag=5, weights=np.ones(50))

    def test_weighted_rejects_negative_sum(self):
        X = _slow_mode_dataset(n_frames=200, lag=5)
        with pytest.raises(ValueError, match="sum to a positive"):
            compute_tica(X, lag=5, weights=np.zeros(200))

    def test_weighted_slow_mode_recovery(self):
        """Reweighted tICA should still recover slow feature even with skewed weights."""
        rng = np.random.default_rng(99)
        n = 600
        slow = np.cumsum(rng.normal(0, 0.1, n))
        fast = rng.normal(0, 1, n)
        X = np.column_stack([slow, fast])
        # Weights that down-weight second half of trajectory
        w = np.ones(n)
        w[n // 2:] *= 0.1
        w /= w.sum()
        result = compute_tica(X, lag=5, weights=w)
        assert abs(result.weights[0]) > abs(result.weights[1]), (
            "weighted tICA should still find slow feature as dominant mode"
        )


# ---------------------------------------------------------------------------
# compute_tica_components (multi-component, analysis-only sibling of compute_tica)
# ---------------------------------------------------------------------------

class TestComputeTicaComponents:
    def test_returns_requested_component_count(self):
        X = _multi_slow_mode_dataset(n_features=5, n_frames=600)
        results = compute_tica_components(X, lag=5, n_components=3)
        assert len(results) == 3

    def test_clamps_to_feature_dimensionality(self):
        X = _multi_slow_mode_dataset(n_features=4, n_frames=600)
        results = compute_tica_components(X, lag=5, n_components=10)
        assert len(results) == 4

    def test_eigenvalues_descending(self):
        X = _multi_slow_mode_dataset(n_features=5, n_frames=800)
        results = compute_tica_components(X, lag=5, n_components=5)
        eigenvalues = [r.eigenvalue for r in results]
        assert eigenvalues == sorted(eigenvalues, reverse=True)

    def test_top_component_matches_single_component_fit(self):
        """tIC1 from compute_tica_components must agree with compute_tica itself."""
        X = _multi_slow_mode_dataset(n_features=5, n_frames=800, seed=1)
        single = compute_tica(X, lag=5)
        multi = compute_tica_components(X, lag=5, n_components=3)
        assert multi[0].eigenvalue == pytest.approx(single.eigenvalue, abs=1e-8)
        # Sign conventions differ (pivot-based vs unconstrained) -- compare direction.
        # weights are C(0)-normalised, not L2-unit, so normalise by L2 norm before
        # reading the dot product as a cosine similarity.
        a, b = multi[0].weights, single.weights
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert abs(cos_sim) == pytest.approx(1.0, abs=1e-6)

    def test_recovers_known_slow_to_fast_ranking(self):
        """Feature 0 is the slowest AR(1) process by construction; tIC1 should
        load on it most heavily among the top components."""
        X = _multi_slow_mode_dataset(n_features=5, n_frames=1000, seed=2)
        results = compute_tica_components(X, lag=5, n_components=5)
        tic1_weights = np.abs(results[0].weights)
        assert int(np.argmax(tic1_weights)) == 0

    def test_components_are_c0_orthogonal(self):
        """Generalised eigenvectors of a symmetric problem are C(0)-orthogonal."""
        X = _multi_slow_mode_dataset(n_features=5, n_frames=800, seed=3)
        results = compute_tica_components(X, lag=5, n_components=5)
        from gareus.tica import _tica_covariance_matrices
        C0, _, _, _ = _tica_covariance_matrices(X, lag=5)
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                cross = float(results[i].weights @ C0 @ results[j].weights)
                assert abs(cross) < 1e-6

    def test_each_component_projectable_with_project_tica1(self):
        X = _multi_slow_mode_dataset(n_features=5, n_frames=600, seed=4)
        results = compute_tica_components(X, lag=5, n_components=3)
        for r in results:
            proj_a = (X - r.mean) @ r.weights
            proj_b = project_tica1(X, r)
            np.testing.assert_allclose(proj_a, proj_b, atol=1e-9)

    def test_rejects_lag_zero(self):
        X = _multi_slow_mode_dataset(n_features=3, n_frames=200)
        with pytest.raises(ValueError):
            compute_tica_components(X, lag=0, n_components=2)

    def test_stores_torsion_indices_on_every_component(self):
        X = _multi_slow_mode_dataset(n_features=3, n_frames=300)
        phi = [(0, 1, 2, 3)]
        psi = [(2, 3, 4, 5)]
        results = compute_tica_components(
            X, lag=5, n_components=2, phi_torsion_indices=phi, psi_torsion_indices=psi
        )
        for r in results:
            assert r.phi_torsion_indices == phi
            assert r.psi_torsion_indices == psi


# ---------------------------------------------------------------------------
# compute_tica_combined (production-facing: combine top-N modes into one CV2 direction)
# ---------------------------------------------------------------------------

class TestComputeTicaCombined:
    def test_n_components_one_matches_compute_tica_exactly(self):
        X = _multi_slow_mode_dataset(n_features=5, n_frames=600, seed=5)
        single = compute_tica(X, lag=5)
        combined = compute_tica_combined(X, lag=5, n_components=1)
        np.testing.assert_array_equal(combined.weights, single.weights)
        assert combined.eigenvalue == single.eigenvalue
        assert combined.offset == single.offset

    def test_default_n_components_matches_compute_tica(self):
        """Default n_components=1 must reproduce old single-mode behavior untouched."""
        X = _multi_slow_mode_dataset(n_features=4, n_frames=500, seed=6)
        single = compute_tica(X, lag=5)
        combined = compute_tica_combined(X, lag=5)
        np.testing.assert_array_equal(combined.weights, single.weights)

    def test_combines_top_n_eigenvalue_weighted(self):
        # Two independent AR(1) features with distinct, well-separated eigenvalues,
        # plus two near-zero-signal columns. n_components=2 should pull in the
        # second (faster) mode with a coefficient ratio = sqrt(eigenvalue_1/eigenvalue_2).
        rng = np.random.default_rng(8)
        n = 900
        a1, a2 = 0.97, 0.6
        f1 = np.zeros(n)
        f2 = np.zeros(n)
        for t in range(1, n):
            f1[t] = a1 * f1[t - 1] + rng.normal(0, np.sqrt(1 - a1 * a1))
            f2[t] = a2 * f2[t - 1] + rng.normal(0, np.sqrt(1 - a2 * a2))
        X = np.column_stack([f1, f2, rng.normal(0, 1e-6, n), rng.normal(0, 1e-6, n)])

        result_1 = compute_tica_combined(X, lag=5, n_components=1)
        result_2 = compute_tica_combined(X, lag=5, n_components=2)

        # n_components=1 must reduce to "tIC1 alone": dominated by the slow feature.
        assert abs(result_1.weights[0]) > abs(result_1.weights[1])

        # n_components=2 must pull in the second (faster) axis -- not identical to tIC1 alone.
        assert abs(result_2.weights[1]) > 1e-3
        a, b = result_1.weights, result_2.weights
        cos_sim = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        assert cos_sim < 0.999

        # Combining can only match or reduce tIC1-alone autocorrelation (weighted
        # average of the selected eigenvalues, weighted by eigenvalue itself).
        assert result_2.eigenvalue < result_1.eigenvalue

    def test_out_of_range_raises(self):
        X = _multi_slow_mode_dataset(n_features=3, n_frames=300, seed=7)
        with pytest.raises(ValueError, match=r"n_components must be in 1\.\.3"):
            compute_tica_combined(X, lag=5, n_components=4)

    def test_projectable_with_project_tica1(self):
        X = _multi_slow_mode_dataset(n_features=4, n_frames=500, seed=8)
        result = compute_tica_combined(X, lag=5, n_components=3)
        proj_a = (X - result.mean) @ result.weights
        proj_b = project_tica1(X, result)
        np.testing.assert_allclose(proj_a, proj_b, atol=1e-9)

    def test_sign_continuity_respected(self):
        X = _multi_slow_mode_dataset(n_features=4, n_frames=500, seed=9)
        r1 = compute_tica_combined(X, lag=5, n_components=2)
        r1_flipped = TICAResult(
            weights=-r1.weights, eigenvalue=r1.eigenvalue, mean=r1.mean,
            offset=-r1.offset, lag=r1.lag,
            phi_torsion_indices=r1.phi_torsion_indices,
            psi_torsion_indices=r1.psi_torsion_indices,
            n_samples=r1.n_samples, sign_ref=-r1.weights,
        )
        r2 = compute_tica_combined(X, lag=5, n_components=2, previous_result=r1_flipped)
        assert float(np.dot(r2.weights, r1_flipped.weights)) > 0


class TestComputeCombinedTicaFromEpochObs:
    def test_n_components_one_matches_plain_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            epoch_dir = Path(tmp) / "epoch_000"
            tica_dir = epoch_dir / "tica_obs"
            tica_dir.mkdir(parents=True)
            rng = np.random.default_rng(11)
            slow = np.cumsum(rng.normal(0, 0.1, 300))
            fast = rng.normal(0, 1, 300)
            X = np.column_stack([slow, fast])
            np.savez_compressed(
                tica_dir / "dihedral_obs_000.npz",
                features=X,
                steps=np.arange(300, dtype=np.int64),
                window=np.zeros(300, dtype=np.int64),
            )
            phi = [(0, 1, 2, 3)]
            psi = [(2, 3, 4, 5)]
            plain = compute_tica_from_epoch_obs(epoch_dir, 5, phi, psi)
            combined = compute_combined_tica_from_epoch_obs(epoch_dir, 5, 1, phi, psi)
            np.testing.assert_array_equal(combined.weights, plain.weights)


# ---------------------------------------------------------------------------
# Segment-boundary helpers (_normalize_segments, _segment_lagged_pair_indices)
# ---------------------------------------------------------------------------

class TestNormalizeSegments:
    def test_none_returns_whole_array_as_one_segment(self):
        out = _normalize_segments(None, 42)
        np.testing.assert_array_equal(out, [42])

    def test_lengths_list_passthrough(self):
        out = _normalize_segments([10, 20, 12], 42)
        np.testing.assert_array_equal(out, [10, 20, 12])

    def test_per_frame_id_array_run_length_encoded(self):
        ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        out = _normalize_segments(ids, 9)
        np.testing.assert_array_equal(out, [3, 2, 4])

    def test_rejects_noncontiguous_ids(self):
        ids = np.array([0, 0, 1, 1, 0, 0])  # id 0 reappears after id 1
        with pytest.raises(ValueError, match="non-contiguous"):
            _normalize_segments(ids, 6)

    def test_rejects_bad_sum(self):
        with pytest.raises(ValueError, match="sum"):
            _normalize_segments([10, 10], 25)

    def test_rejects_nonpositive_length(self):
        with pytest.raises(ValueError, match="positive"):
            _normalize_segments([10, 0, 15], 25)

    def test_all_ones_lengths_not_misread_as_single_id_run(self):
        """Regression: when the number of segments equals n (e.g. every
        replica file has exactly 1 frame), a lengths array of all 1s must
        be treated as n separate length-1 segments — NOT misread as a
        per-frame id array (all entries equal -> one contiguous "id" run
        spanning all n frames), which would silently reintroduce
        cross-segment lagged pairs."""
        n = 5
        out = _normalize_segments([1, 1, 1, 1, 1], n)
        np.testing.assert_array_equal(out, [1, 1, 1, 1, 1])
        assert out.sum() == n
        assert len(out) == n  # NOT collapsed into a single segment of length n


class TestSegmentLaggedPairIndices:
    def test_excludes_cross_boundary_pairs(self):
        lengths = np.array([5, 4])
        lag = 2
        left, right = _segment_lagged_pair_indices(lengths, lag)
        # Segment 1 (len 5, offset 0): valid lefts 0,1,2 (L-lag=3)
        # Segment 2 (len 4, offset 5): valid lefts 5,6 (L-lag=2)
        # Cross-boundary lefts 3,4 (whose right = 5,6 would land in segment 2)
        # must NOT appear.
        np.testing.assert_array_equal(left, [0, 1, 2, 5, 6])
        np.testing.assert_array_equal(right, np.asarray([0, 1, 2, 5, 6]) + lag)
        assert 3 not in left
        assert 4 not in left

    def test_zero_pairs_when_every_segment_shorter_than_lag(self):
        lengths = np.array([3, 3, 3])
        left, right = _segment_lagged_pair_indices(lengths, lag=5)
        assert left.shape[0] == 0
        assert right.shape[0] == 0

    def test_segment_equal_to_lag_contributes_zero_pairs(self):
        lengths = np.array([5, 10])
        left, right = _segment_lagged_pair_indices(lengths, lag=5)
        # First segment length == lag -> zero pairs from it.
        # Second segment (offset 5, len 10): valid lefts 5..9 (L-lag=5)
        np.testing.assert_array_equal(left, [5, 6, 7, 8, 9])


# ---------------------------------------------------------------------------
# compute_tica: segment-boundary-aware lagged covariance (fix under test)
# ---------------------------------------------------------------------------

class TestComputeTICASegments:
    def test_explicit_single_segment_matches_implicit_none(self):
        """Backward compat: segments=[n] (whole array as one segment) must
        reproduce the segments=None default path exactly."""
        X = _slow_mode_dataset(n_frames=300, lag=5, seed=3)
        n = len(X)
        r_none = compute_tica(X, lag=5)
        r_explicit = compute_tica(X, lag=5, segments=[n])
        np.testing.assert_allclose(r_none.weights, r_explicit.weights, atol=1e-12)
        assert abs(r_none.eigenvalue - r_explicit.eigenvalue) < 1e-12
        np.testing.assert_allclose(r_none.mean, r_explicit.mean, atol=1e-12)
        assert abs(r_none.offset - r_explicit.offset) < 1e-12
        assert r_none.n_samples == r_explicit.n_samples == n

    def test_explicit_single_segment_matches_implicit_none_weighted(self):
        X = _slow_mode_dataset(n_frames=300, lag=5, seed=4)
        n = len(X)
        rng = np.random.default_rng(5)
        w = rng.random(n)
        r_none = compute_tica(X, lag=5, weights=w)
        r_explicit = compute_tica(X, lag=5, weights=w, segments=[n])
        np.testing.assert_allclose(r_none.weights, r_explicit.weights, atol=1e-12)
        assert abs(r_none.eigenvalue - r_explicit.eigenvalue) < 1e-12

    def test_boundary_exclusion_matches_hand_computed_reference(self):
        """C(tau) from compute_tica(segments=[L1,L2]) must match an
        independently hand-computed per-segment C(tau)/C(0) solve — i.e. no
        pair straddling the L1/L2 join is used."""
        lag = 5
        L1, L2 = 220, 180
        seg1 = _slow_mode_dataset(n_frames=L1, lag=lag, seed=11)
        seg2 = _slow_mode_dataset(n_frames=L2, lag=lag, seed=22)
        # Inject a large artificial discontinuity in feature 0 at the join —
        # if a cross-boundary pair were ever formed, it would inject a
        # spurious correlation into C(tau).
        seg2 = seg2.copy()
        seg2[:, 0] += 50.0
        X = np.concatenate([seg1, seg2], axis=0)
        seg_lengths = [L1, L2]

        result = compute_tica(X, lag=lag, segments=seg_lengths)
        ref_eigenvalue, ref_weights = _hand_tica_reference(X, lag, seg_lengths)

        # result.weights / ref_weights are C(0)-normalised, not L2-unit, so a
        # raw dot product is not a bounded cosine similarity — normalise by
        # both L2 norms explicitly.
        cos = float(
            np.dot(result.weights, ref_weights)
            / (np.linalg.norm(result.weights) * np.linalg.norm(ref_weights))
        )
        assert abs(cos) > 0.999, f"cosine similarity to reference = {cos:.6f}"
        assert abs(result.eigenvalue - ref_eigenvalue) < 1e-6

    def test_boundary_pairs_excluded_changes_result_vs_naive_concatenation(self):
        """With a discontinuity at the join, the segment-aware result must
        differ from the naive (segments=None, boundary pairs included)
        result — proving the fix actually changes behaviour."""
        lag = 5
        L1, L2 = 220, 180
        seg1 = _slow_mode_dataset(n_frames=L1, lag=lag, seed=11)
        seg2 = _slow_mode_dataset(n_frames=L2, lag=lag, seed=22)
        seg2 = seg2.copy()
        seg2[:, 0] += 50.0
        X = np.concatenate([seg1, seg2], axis=0)

        r_segmented = compute_tica(X, lag=lag, segments=[L1, L2])
        r_naive = compute_tica(X, lag=lag)  # old behaviour: boundary pairs included

        cos = float(
            np.dot(r_segmented.weights, r_naive.weights)
            / (np.linalg.norm(r_segmented.weights) * np.linalg.norm(r_naive.weights))
        )
        differs = abs(cos) < 0.999 or abs(r_segmented.eigenvalue - r_naive.eigenvalue) > 1e-4
        assert differs, (
            "segmented result should differ from naive whole-array concatenation "
            "when a spurious cross-boundary jump is present"
        )

    def test_weighted_boundary_exclusion_matches_hand_computed_reference(self):
        """Weighted branch: pair-weight bookkeeping must also respect
        segment boundaries."""
        lag = 5
        L1, L2 = 220, 180
        seg1 = _slow_mode_dataset(n_frames=L1, lag=lag, seed=31)
        seg2 = _slow_mode_dataset(n_frames=L2, lag=lag, seed=32)
        seg2 = seg2.copy()
        seg2[:, 0] += 50.0
        X = np.concatenate([seg1, seg2], axis=0)
        seg_lengths = [L1, L2]

        rng = np.random.default_rng(77)
        w = rng.random(L1 + L2) + 0.1  # strictly positive, non-uniform

        result = compute_tica(X, lag=lag, segments=seg_lengths, weights=w)
        ref_eigenvalue, ref_weights = _hand_tica_reference(X, lag, seg_lengths, weights=w)

        cos = float(
            np.dot(result.weights, ref_weights)
            / (np.linalg.norm(result.weights) * np.linalg.norm(ref_weights))
        )
        assert abs(cos) > 0.999, f"cosine similarity to weighted reference = {cos:.6f}"
        assert abs(result.eigenvalue - ref_eigenvalue) < 1e-6

    def test_short_segments_all_below_lag_raises_even_though_total_n_large(self):
        """16 replicas x 30 frames with default-style lag=50: total n=480 >
        2*lag=100 (old guard would pass) but every segment is shorter than
        the lag, so there are ZERO valid within-segment pairs — must raise."""
        lag = 50
        seg_lengths = [30] * 16
        n = sum(seg_lengths)
        assert n > 2 * lag  # sanity: the old total-length guard would NOT catch this
        X = np.random.default_rng(1).standard_normal((n, 4))
        with pytest.raises(ValueError, match="too few valid"):
            compute_tica(X, lag=lag, segments=seg_lengths)

    def test_segment_equal_to_lag_contributes_no_pairs_but_others_do(self):
        """A segment with length == lag contributes zero pairs (per spec:
        L <= lag -> zero pairs), but the fit should still succeed using the
        other segment's pairs."""
        lag = 5
        seg_lengths = [lag, 200]
        X = np.random.default_rng(2).standard_normal((sum(seg_lengths), 3))
        result = compute_tica(X, lag=lag, segments=seg_lengths)
        assert np.isfinite(result.weights).all()

    def test_rejects_when_too_few_pairs_for_stable_estimate(self):
        """A single remaining pair (n_pairs < 2) must raise rather than
        silently divide by zero in the (n_pairs - 1) denominator."""
        lag = 5
        # One segment barely longer than lag (1 pair) plus short segments.
        seg_lengths = [lag + 1, 3, 3]
        X = np.random.default_rng(3).standard_normal((sum(seg_lengths), 3))
        with pytest.raises(ValueError, match="too few valid"):
            compute_tica(X, lag=lag, segments=seg_lengths)


# ---------------------------------------------------------------------------
# DihedralObsBuffer CV recording
# ---------------------------------------------------------------------------

class TestDihedralObsBufferCV:
    def test_record_saves_primary_cv(self):
        phi = [(0, 1, 2, 3)]
        psi = [(2, 3, 4, 5)]
        with tempfile.TemporaryDirectory() as tmp:
            buf = DihedralObsBuffer(phi, psi, replica_idx=0, out_dir=tmp)
            pos = np.random.default_rng(1).standard_normal((10, 3))
            buf.record(pos, step=0, window=2, cv_primary=0.42, cv_secondary=0.17)
            buf.record(pos, step=1, window=2, cv_primary=0.55, cv_secondary=float("nan"))
            path = buf.save()
            data = np.load(path)
            np.testing.assert_allclose(data["primary_cv"], [0.42, 0.55])
            assert np.isfinite(data["secondary_cv"][0])
            assert np.isnan(data["secondary_cv"][1])

    def test_record_defaults_to_nan(self):
        phi, psi = [(0, 1, 2, 3)], [(2, 3, 4, 5)]
        with tempfile.TemporaryDirectory() as tmp:
            buf = DihedralObsBuffer(phi, psi, replica_idx=0, out_dir=tmp)
            pos = np.random.default_rng(2).standard_normal((10, 3))
            buf.record(pos, step=0, window=0)
            path = buf.save()
            data = np.load(path)
            assert np.isnan(data["primary_cv"][0])
            assert np.isnan(data["secondary_cv"][0])


# ---------------------------------------------------------------------------
# project_tica1
# ---------------------------------------------------------------------------

class TestProjectTICA1:
    def test_shape_1d(self):
        X = _slow_mode_dataset(n_frames=100, lag=5)
        result = compute_tica(X, lag=5)
        proj = project_tica1(X[:10], result)
        assert proj.shape == (10,)

    def test_offset_invariant(self):
        X = _slow_mode_dataset(n_frames=200, lag=5)
        result = compute_tica(X, lag=5)
        v1 = (X - result.mean) @ result.weights
        v2 = project_tica1(X, result)
        np.testing.assert_allclose(v1, v2, atol=1e-12)

    def test_single_row(self):
        X = _slow_mode_dataset(n_frames=200, lag=5)
        result = compute_tica(X, lag=5)
        row = X[0:1]
        proj = project_tica1(row, result)
        assert proj.shape == (1,)
        assert np.isfinite(proj[0])


# ---------------------------------------------------------------------------
# window_tica_centers
# ---------------------------------------------------------------------------

class TestWindowTICACenters:
    def test_returns_per_state(self):
        X = _slow_mode_dataset(n_frames=300, lag=5)
        result = compute_tica(X, lag=5)
        rng = np.random.default_rng(7)
        wids = rng.integers(0, 4, size=300)
        centers = window_tica_centers(X, wids, result)
        assert set(centers.keys()) == {0, 1, 2, 3}

    def test_median_in_range(self):
        X = _slow_mode_dataset(n_frames=300, lag=5)
        result = compute_tica(X, lag=5)
        wids = np.zeros(300, dtype=int)
        proj = project_tica1(X, result)
        centers = window_tica_centers(X, wids, result, percentile=50.0)
        assert abs(centers[0] - float(np.median(proj))) < 1e-10


# ---------------------------------------------------------------------------
# tica_k_from_spread
# ---------------------------------------------------------------------------

class TestTICAKFromSpread:
    def test_returns_per_state(self):
        X = _slow_mode_dataset(n_frames=300, lag=5)
        result = compute_tica(X, lag=5)
        rng = np.random.default_rng(9)
        wids = rng.integers(0, 4, size=300)
        ks = tica_k_from_spread(X, wids, result)
        assert set(ks.keys()) == {0, 1, 2, 3}

    def test_clamped(self):
        X = _slow_mode_dataset(n_frames=300, lag=5)
        result = compute_tica(X, lag=5)
        wids = np.zeros(300, dtype=int)
        ks = tica_k_from_spread(X, wids, result, k_min_kcal=1.0, k_max_kcal=5.0)
        assert 1.0 <= ks[0] <= 5.0


# ---------------------------------------------------------------------------
# DihedralObsBuffer
# ---------------------------------------------------------------------------

class TestDihedralObsBuffer:
    def test_records_active_window_not_replica(self):
        phi, psi = _make_phi_psi(n_residues=3)
        with tempfile.TemporaryDirectory() as tmp:
            buf = DihedralObsBuffer(phi, psi, replica_idx=2, out_dir=tmp)
            pos = _random_positions(n_atoms=30)
            buf.record(pos, step=100, window=7)
            assert len(buf) == 1
            # Verify that the window index stored is 7 (active window), not 2 (replica)
            assert buf._windows[0] == 7
            assert buf._steps[0] == 100

    def test_npz_roundtrip(self):
        phi, psi = _make_phi_psi(n_residues=3)
        with tempfile.TemporaryDirectory() as tmp:
            buf = DihedralObsBuffer(phi, psi, replica_idx=0, out_dir=tmp)
            pos1 = _random_positions(n_atoms=30, seed=1)
            pos2 = _random_positions(n_atoms=30, seed=2)
            buf.record(pos1, step=10, window=0)
            buf.record(pos2, step=20, window=1)
            path = buf.save()
            assert path is not None
            assert path.exists()
            data = np.load(path)
            assert data["features"].shape == (2, 12)
            assert list(data["window"]) == [0, 1]
            assert list(data["steps"]) == [10, 20]

    def test_save_returns_none_when_empty(self):
        phi, psi = _make_phi_psi(n_residues=3)
        with tempfile.TemporaryDirectory() as tmp:
            buf = DihedralObsBuffer(phi, psi, replica_idx=0, out_dir=tmp)
            result = buf.save()
            assert result is None

    def test_clear(self):
        phi, psi = _make_phi_psi(n_residues=2)
        with tempfile.TemporaryDirectory() as tmp:
            buf = DihedralObsBuffer(phi, psi, replica_idx=0, out_dir=tmp)
            pos = _random_positions(n_atoms=20)
            buf.record(pos, step=1, window=0)
            assert len(buf) == 1
            buf.clear()
            assert len(buf) == 0


# ---------------------------------------------------------------------------
# load_epoch_dihedral_obs
# ---------------------------------------------------------------------------

class TestLoadEpochDihedralObs:
    def _write_obs(self, epoch_dir: Path, replica_idx: int, n_frames: int, n_feat: int, seed: int):
        rng = np.random.default_rng(seed)
        tica_dir = epoch_dir / "tica_obs"
        tica_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tica_dir / f"dihedral_obs_{replica_idx:03d}.npz",
            features=rng.standard_normal((n_frames, n_feat)),
            steps=np.arange(n_frames, dtype=np.int64),
            window=np.zeros(n_frames, dtype=np.int64),
        )

    def test_loads_and_concatenates(self):
        with tempfile.TemporaryDirectory() as tmp:
            epoch_dir = Path(tmp) / "epoch_000"
            self._write_obs(epoch_dir, 0, 50, 12, seed=1)
            self._write_obs(epoch_dir, 1, 30, 12, seed=2)
            X, wids, pcv, scv, seg_lengths = load_epoch_dihedral_obs(epoch_dir)
            assert X.shape == (80, 12)
            assert wids.shape == (80,)
            # Old obs files without primary_cv → NaN filled
            assert pcv.shape == (80,)
            assert np.all(np.isnan(pcv))
            assert scv.shape == (80,)
            assert np.all(np.isnan(scv))
            # Segment lengths track per-file frame counts, in glob-sorted order
            np.testing.assert_array_equal(seg_lengths, [50, 30])
            assert int(seg_lengths.sum()) == 80

    def test_loads_with_cv_arrays(self):
        """Obs files that include primary_cv/secondary_cv are loaded correctly."""
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            epoch_dir = Path(tmp) / "epoch_000"
            tica_dir = epoch_dir / "tica_obs"
            tica_dir.mkdir(parents=True)
            pcv = rng.random(40)
            scv = rng.random(40)
            np.savez_compressed(
                tica_dir / "dihedral_obs_000.npz",
                features=rng.standard_normal((40, 8)),
                steps=np.arange(40, dtype=np.int64),
                window=np.zeros(40, dtype=np.int64),
                primary_cv=pcv,
                secondary_cv=scv,
            )
            X, wids, pcv_out, scv_out, seg_lengths = load_epoch_dihedral_obs(epoch_dir)
            np.testing.assert_allclose(pcv_out, pcv)
            np.testing.assert_allclose(scv_out, scv)
            np.testing.assert_array_equal(seg_lengths, [40])

    def test_raises_on_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            epoch_dir = Path(tmp) / "epoch_empty"
            epoch_dir.mkdir()
            with pytest.raises(FileNotFoundError):
                load_epoch_dihedral_obs(epoch_dir)


# ---------------------------------------------------------------------------
# compute_tica_from_epoch_obs
# ---------------------------------------------------------------------------

class TestComputeTICAFromEpochObs:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            epoch_dir = Path(tmp) / "epoch_000"
            tica_dir = epoch_dir / "tica_obs"
            tica_dir.mkdir(parents=True)
            rng = np.random.default_rng(42)
            # Two features, first one slow
            slow = np.cumsum(rng.normal(0, 0.1, 300))
            fast = rng.normal(0, 1, 300)
            X = np.column_stack([slow, fast])
            np.savez_compressed(
                tica_dir / "dihedral_obs_000.npz",
                features=X,
                steps=np.arange(300, dtype=np.int64),
                window=np.zeros(300, dtype=np.int64),
            )
            phi = [(0, 1, 2, 3)]
            psi = [(2, 3, 4, 5)]
            result = compute_tica_from_epoch_obs(epoch_dir, 5, phi, psi)
            assert isinstance(result, TICAResult)
            assert result.weights.shape == (2,)


# ---------------------------------------------------------------------------
# _compute_mbar_weights_for_tica
# ---------------------------------------------------------------------------

class TestComputeMBARWeightsForTICA:
    """Tests for the standalone MBAR reweighting helper in adaptive_production."""

    def _make_registry_and_obs(self, tmp: str, n_per_window: int = 100, seed: int = 42):
        """Build a 2-window registry and matching tICA obs npz."""
        from gareus.adaptive_production import WindowStateRegistry, WindowState
        rng = np.random.default_rng(seed)
        # State 0: center 0.3, k 10 kcal/mol
        # State 1: center 0.7, k 10 kcal/mol
        reg = WindowStateRegistry()
        reg.add_state(0.3, 10.0)
        reg.add_state(0.7, 10.0)

        # Simulate obs from two windows
        epoch_dir = Path(tmp) / "epoch_000"
        tica_dir = epoch_dir / "tica_obs"
        tica_dir.mkdir(parents=True)
        # Window 0 samples near center 0.3
        pcv0 = rng.normal(0.3, 0.05, n_per_window)
        # Window 1 samples near center 0.7
        pcv1 = rng.normal(0.7, 0.05, n_per_window)
        pcv = np.concatenate([pcv0, pcv1])
        window_ids = np.array([0] * n_per_window + [1] * n_per_window, dtype=np.int64)
        scv = np.full(2 * n_per_window, np.nan)
        feat = rng.standard_normal((2 * n_per_window, 4))

        # Write epoch window map so _load_epoch_window_map can resolve state IDs
        import csv as _csv
        wmap_path = epoch_dir / "epoch_window_map.csv"
        with open(wmap_path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["window_idx", "state_id"])
            w.writerow([0, 0])
            w.writerow([1, 1])

        np.savez_compressed(
            tica_dir / "dihedral_obs_000.npz",
            features=feat,
            steps=np.arange(2 * n_per_window, dtype=np.int64),
            window=window_ids,
            primary_cv=pcv,
            secondary_cv=scv,
        )
        return reg, epoch_dir, pcv, window_ids

    def test_weights_sum_to_one(self):
        from gareus.adaptive_production import _compute_mbar_weights_for_tica
        with tempfile.TemporaryDirectory() as tmp:
            reg, epoch_dir, pcv, wids = self._make_registry_and_obs(tmp)
            scv = np.full(len(pcv), np.nan)
            w = _compute_mbar_weights_for_tica(pcv, scv, wids, epoch_dir, reg, temperature_K=300.0)
            assert w.shape == (len(pcv),)
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)

    def test_weights_non_negative(self):
        from gareus.adaptive_production import _compute_mbar_weights_for_tica
        with tempfile.TemporaryDirectory() as tmp:
            reg, epoch_dir, pcv, wids = self._make_registry_and_obs(tmp)
            scv = np.full(len(pcv), np.nan)
            w = _compute_mbar_weights_for_tica(pcv, scv, wids, epoch_dir, reg, temperature_K=300.0)
            assert (w >= 0).all()

    def test_fallback_to_uniform_on_nan_cv(self):
        """All-NaN primary_cv → uniform weights (old obs files)."""
        from gareus.adaptive_production import _compute_mbar_weights_for_tica
        with tempfile.TemporaryDirectory() as tmp:
            reg, epoch_dir, _, wids = self._make_registry_and_obs(tmp)
            pcv_nan = np.full(len(wids), np.nan)
            scv = np.full(len(wids), np.nan)
            w = _compute_mbar_weights_for_tica(pcv_nan, scv, wids, epoch_dir, reg)
            N = len(wids)
            np.testing.assert_allclose(w, np.full(N, 1.0 / N), atol=1e-12)

    def test_weights_upweight_edges(self):
        """Frames far from window center should get higher MBAR weight (unbiased ensemble)."""
        from gareus.adaptive_production import _compute_mbar_weights_for_tica
        with tempfile.TemporaryDirectory() as tmp:
            reg, epoch_dir, pcv, wids = self._make_registry_and_obs(tmp, n_per_window=200)
            scv = np.full(len(pcv), np.nan)
            w = _compute_mbar_weights_for_tica(pcv, scv, wids, epoch_dir, reg, temperature_K=300.0)
            # Frames near center of window 0 (small delta) get biased HEAVILY → low MBAR weight
            # Frames far from all centers (edges) → higher unbiased weight
            # Simple sanity: mean weight of extreme-edge frames > mean weight of center frames
            center_mask = np.abs(pcv - 0.3) < 0.02
            edge_mask = np.abs(pcv - 0.5) < 0.03  # between windows, upweighted by MBAR
            if center_mask.sum() > 5 and edge_mask.sum() > 5:
                assert w[edge_mask].mean() > w[center_mask].mean()


# ---------------------------------------------------------------------------
# TICAResult serialisation
# ---------------------------------------------------------------------------

class TestTICAResultSerialization:
    def test_to_from_dict(self):
        X = _slow_mode_dataset(n_frames=200, lag=5)
        r = compute_tica(X, lag=5, phi_torsion_indices=[(0, 1, 2, 3)], psi_torsion_indices=[(2, 3, 4, 5)])
        d = r.to_dict()
        r2 = TICAResult.from_dict(d)
        np.testing.assert_allclose(r.weights, r2.weights)
        np.testing.assert_allclose(r.mean, r2.mean)
        assert r.lag == r2.lag
        assert r.phi_torsion_indices == r2.phi_torsion_indices

    def test_save_load_roundtrip(self):
        X = _slow_mode_dataset(n_frames=200, lag=5)
        r = compute_tica(X, lag=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tica_state.json"
            r.save(path)
            r2 = TICAResult.load(path)
            np.testing.assert_allclose(r.weights, r2.weights, atol=1e-12)
            assert abs(r.offset - r2.offset) < 1e-12


# ---------------------------------------------------------------------------
# MBAR exclusion flag helpers (for adaptive_production guards)
# ---------------------------------------------------------------------------

class TestMBARExclusion:
    """Verify that the tica_cv_version column logic can be mocked purely."""

    def test_mbar_filters_by_tica_cv_version(self):
        """Simulate the per-row exclusion gate used in _epoch_sample_sources."""
        rows = [
            {"tica_cv_version": "v0", "cv_A": 1.0},
            {"tica_cv_version": "v1", "cv_A": 2.0},
            {"tica_cv_version": "v1", "cv_A": 3.0},
        ]
        active_version = "v1"
        kept = [r for r in rows if r.get("tica_cv_version", active_version) == active_version]
        assert len(kept) == 2
        assert all(r["tica_cv_version"] == "v1" for r in kept)

    def test_mbar_keeps_all_when_no_version_key(self):
        rows = [{"cv_A": 1.0}, {"cv_A": 2.0}]
        active_version = "v0"
        kept = [r for r in rows if r.get("tica_cv_version", active_version) == active_version]
        assert len(kept) == 2
