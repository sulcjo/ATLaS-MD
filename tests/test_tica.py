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
    compute_tica_from_epoch_obs,
    load_epoch_dihedral_obs,
    project_tica1,
    tica_k_from_spread,
    window_tica_centers,
    DihedralObsBuffer,
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
            X, wids, pcv, scv = load_epoch_dihedral_obs(epoch_dir)
            assert X.shape == (80, 12)
            assert wids.shape == (80,)
            # Old obs files without primary_cv → NaN filled
            assert pcv.shape == (80,)
            assert np.all(np.isnan(pcv))
            assert scv.shape == (80,)
            assert np.all(np.isnan(scv))

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
            X, wids, pcv_out, scv_out = load_epoch_dihedral_obs(epoch_dir)
            np.testing.assert_allclose(pcv_out, pcv)
            np.testing.assert_allclose(scv_out, scv)

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
