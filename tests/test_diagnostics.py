"""
Tests for F3 (per-window ESS) and F2 (calibration range warnings)
diagnostic fixes in gareus/diagnostics.py.

These tests are OpenMM-free and operate only on synthetic numpy arrays
written to temporary directories.
"""
from __future__ import annotations

import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: build a synthetic analysis_arrays.npz
# ---------------------------------------------------------------------------

def _make_npz(tmp_path, boost_kj, window_arr):
    """Write a minimal analysis_arrays.npz that compute_gamd_reweighting_diagnostics can read."""
    npz_path = tmp_path / "analysis_arrays.npz"
    np.savez(npz_path, gamd_boost_total_kj_mol=boost_kj, window=window_arr)
    return tmp_path


# ---------------------------------------------------------------------------
# F3: Arithmetic sanity -- pooled ESS is degraded vs per-window ESS
# ---------------------------------------------------------------------------

def test_per_window_ess_not_pooled_arithmetic():
    """
    Pooled ESS is significantly degraded when two windows have very different
    mean boosts, even though each window individually reweights cleanly.

    With N equal-size windows where one dominates all weight, the pooled ESS
    fraction approaches 1/N_windows (here 0.5 for 2 windows) while each
    per-window fraction should remain close to 1.  The key property is:
        per_window_ess_fraction >> pooled_ess_fraction

    This is a pure-arithmetic test that reproduces the problem and verifies
    the expected mathematical property without calling the real function.
    """
    rng = np.random.default_rng(42)
    b1 = rng.normal(5.0, 0.5, 500)
    b2 = rng.normal(20.0, 0.5, 500)
    boost = np.concatenate([b1, b2])
    beta = 1.0 / (0.00831446 * 300)

    # Pooled ESS -- the high-boost window dominates all weight, so pooled
    # ESS approaches n_high_window / n_total = 0.5 (not near 1.0).
    logw = beta * boost
    logw -= np.max(logw)
    w = np.exp(logw)
    ess_pool_frac = (w.sum() ** 2) / (w ** 2).sum() / boost.size

    # Per-window ESS should be healthy (near 1.0 for unimodal window)
    per_window_fracs = []
    for bid, bvals in [(0, b1), (1, b2)]:
        logw_w = beta * bvals
        logw_w -= np.max(logw_w)
        w_w = np.exp(logw_w)
        ess_w_frac = (w_w.sum() ** 2) / (w_w ** 2).sum() / bvals.size
        assert ess_w_frac > 0.5, (
            f"window {bid} per-window ESS should be healthy (unimodal data): {ess_w_frac:.4f}"
        )
        per_window_fracs.append(ess_w_frac)

    # Pooled ESS should be substantially lower than the per-window minimum
    min_pw = min(per_window_fracs)
    assert ess_pool_frac < 0.6 * min_pw, (
        f"pooled ESS ({ess_pool_frac:.4f}) should be substantially lower than "
        f"per-window min ({min_pw:.4f}) for bimodal data"
    )


# ---------------------------------------------------------------------------
# F3: Integration test -- per_window_ess keys present and correct in result
# ---------------------------------------------------------------------------

def test_per_window_ess_keys_in_result(tmp_path):
    """
    compute_gamd_reweighting_diagnostics must return per_window_ess,
    ess_fraction_per_window_min, and ess_fraction_per_window_median.

    With bimodal data (windows with very different mean boosts) the pooled
    fraction is substantially lower than per-window fractions, proving that
    the function correctly separates windows and each window is healthy.
    """
    from gareus.diagnostics import compute_gamd_reweighting_diagnostics

    rng = np.random.default_rng(0)
    b1 = rng.normal(5.0, 0.5, 500)
    b2 = rng.normal(20.0, 0.5, 500)
    boost = np.concatenate([b1, b2])
    windows = np.array([0] * 500 + [1] * 500, dtype=np.int32)
    _make_npz(tmp_path, boost, windows)

    result = compute_gamd_reweighting_diagnostics(tmp_path, temperature_k=300.0)

    assert result["status"] in ("ok", "warning"), f"unexpected status: {result['status']}"

    # Required new keys must be present
    assert "per_window_ess" in result, "per_window_ess key missing from result"
    assert "ess_fraction_per_window_min" in result
    assert "ess_fraction_per_window_median" in result

    pwe = result["per_window_ess"]
    assert set(pwe.keys()) == {0, 1}, f"expected windows 0 and 1, got {set(pwe.keys())}"

    # Each window entry has the three sub-keys
    for wid in (0, 1):
        entry = pwe[wid]
        assert "ess" in entry
        assert "ess_fraction" in entry
        assert "n_frames" in entry
        assert entry["n_frames"] == 500

    # Per-window ESS fractions should be high (unimodal per window)
    pw_min = result["ess_fraction_per_window_min"]
    assert math.isfinite(pw_min), "ess_fraction_per_window_min should be finite"
    assert pw_min > 0.5, (
        f"per-window ESS fraction should be > 0.5 for unimodal windows, got {pw_min:.4f}"
    )

    # Pooled ESS fraction should be substantially lower than per-window min
    # (bimodal pooling penalty: approaches 1/n_windows when one window dominates)
    pool_frac = result["reweighting_ess_fraction"]
    assert math.isfinite(pool_frac), "reweighting_ess_fraction should be finite"
    assert pool_frac < 0.6 * pw_min, (
        f"pooled ESS ({pool_frac:.4f}) should be substantially lower than "
        f"per-window min ({pw_min:.4f}) for bimodal data"
    )


# ---------------------------------------------------------------------------
# F2: Calibration range warnings -- outlier window triggers warning
# ---------------------------------------------------------------------------

def test_calibration_range_warning_fires(tmp_path):
    """
    When one window has mean boost ~4x the others, a calibration range
    warning must appear in calibration_range_warnings.
    """
    from gareus.diagnostics import compute_gamd_reweighting_diagnostics

    rng = np.random.default_rng(7)
    # 4 normal windows with mean boost ~10 kJ/mol, 1 outlier with ~40 kJ/mol
    normal_means = [10.0, 10.5, 9.5, 10.2]
    boost_parts = [rng.normal(mu, 0.5, 200) for mu in normal_means]
    boost_parts.append(rng.normal(40.0, 0.5, 200))  # window 4 is the outlier
    boost = np.concatenate(boost_parts)
    windows = np.concatenate([
        np.full(200, i, dtype=np.int32) for i in range(5)
    ])
    _make_npz(tmp_path, boost, windows)

    result = compute_gamd_reweighting_diagnostics(tmp_path, temperature_k=300.0)

    crw = result.get("calibration_range_warnings")
    assert crw is not None, "calibration_range_warnings key must always be present"
    assert len(crw) >= 1, (
        f"expected at least one calibration range warning for outlier window; got: {crw}"
    )
    # The warning should mention window 4
    assert any("window 4" in w for w in crw), (
        f"expected warning about window 4, got: {crw}"
    )


# ---------------------------------------------------------------------------
# F2: Calibration range warnings -- uniform windows produce no warnings
# ---------------------------------------------------------------------------

def test_calibration_range_warning_silent_when_uniform(tmp_path):
    """
    When all windows have similar mean boosts, calibration_range_warnings
    must be empty.
    """
    from gareus.diagnostics import compute_gamd_reweighting_diagnostics

    rng = np.random.default_rng(13)
    # 5 windows all with similar mean boost ~12 kJ/mol
    boost_parts = [rng.normal(12.0, 0.5, 300) for _ in range(5)]
    boost = np.concatenate(boost_parts)
    windows = np.concatenate([
        np.full(300, i, dtype=np.int32) for i in range(5)
    ])
    _make_npz(tmp_path, boost, windows)

    result = compute_gamd_reweighting_diagnostics(tmp_path, temperature_k=300.0)

    crw = result.get("calibration_range_warnings")
    assert crw is not None, "calibration_range_warnings key must always be present"
    assert crw == [], (
        f"expected empty calibration_range_warnings for uniform windows, got: {crw}"
    )
