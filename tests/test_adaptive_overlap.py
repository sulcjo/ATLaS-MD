"""
Tests for N2: per-cell 2D overlap check before primary window removal.
"""
import math
import pytest


def _mock_hist_overlap(samples_a, samples_b, lo, hi, bins=50):
    """Simple histogram overlap for testing without importing gareus."""
    import numpy as np
    if hi <= lo or len(samples_a) < 2 or len(samples_b) < 2:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(samples_a, bins=edges, density=True)
    hb, _ = np.histogram(samples_b, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.sum(np.minimum(ha, hb)) * width)


def _check_per_cell_overlap(primary_idx, n_secondary, samples_2d, min_overlap):
    """Mirror of the helper added to adaptive_feedback.py."""
    for j in range(n_secondary):
        left = samples_2d.get((primary_idx - 1, j), [])
        right = samples_2d.get((primary_idx + 1, j), [])
        if len(left) < 10 or len(right) < 10:
            return False
        all_vals = list(left) + list(right)
        lo, hi = min(all_vals), max(all_vals)
        if hi <= lo:
            return False
        ov = _mock_hist_overlap(left, right, lo, hi)
        if not math.isfinite(ov) or ov < min_overlap:
            return False
    return True


def test_n2_depopulated_cell_blocks_removal():
    """
    3×2 grid (3 primary, 2 secondary). Cell (1,0) depopulated.
    Pooled column-1 overlap might look fine, but per-cell check must block removal.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    # Column 1 removal check: need (0,j) and (2,j) to overlap for each j
    # Cell (1,0): left=(0,0), right=(2,0) — make them NOT overlap
    samples = {
        (0, 0): rng.normal(0.0, 0.05, 100).tolist(),
        (2, 0): rng.normal(1.0, 0.05, 100).tolist(),  # far from (0,0), no overlap
        (0, 1): rng.normal(0.5, 0.1, 100).tolist(),
        (2, 1): rng.normal(0.7, 0.1, 100).tolist(),   # good overlap with (0,1)
    }
    result = _check_per_cell_overlap(primary_idx=1, n_secondary=2,
                                     samples_2d=samples, min_overlap=0.15)
    assert result is False, "depopulated cell should block removal"


def test_n2_all_cells_overlap_allows_removal():
    """When all cells have good overlap, removal is permitted."""
    import numpy as np
    rng = np.random.default_rng(1)
    samples = {
        (0, 0): rng.normal(0.5, 0.15, 200).tolist(),
        (2, 0): rng.normal(0.7, 0.15, 200).tolist(),
        (0, 1): rng.normal(0.3, 0.15, 200).tolist(),
        (2, 1): rng.normal(0.5, 0.15, 200).tolist(),
    }
    result = _check_per_cell_overlap(primary_idx=1, n_secondary=2,
                                     samples_2d=samples, min_overlap=0.05)
    assert result is True, "good per-cell overlap should allow removal"


def test_n2_insufficient_data_blocks_removal():
    """Fewer than 10 samples in any neighboring cell blocks removal."""
    samples = {
        (0, 0): [0.1] * 5,   # only 5 samples — too few
        (2, 0): [0.3] * 100,
    }
    result = _check_per_cell_overlap(primary_idx=1, n_secondary=1,
                                     samples_2d=samples, min_overlap=0.1)
    assert result is False, "insufficient data should block removal"


def test_n2_missing_neighbor_blocks_removal():
    """A completely missing neighboring cell entry blocks removal."""
    samples = {
        # (0, 0) is absent entirely — treated as empty list
        (2, 0): [0.5] * 100,
    }
    result = _check_per_cell_overlap(primary_idx=1, n_secondary=1,
                                     samples_2d=samples, min_overlap=0.1)
    assert result is False, "missing neighbor cell should block removal"


def test_n2_real_helper_matches_local_mirror():
    """The real _check_per_cell_overlap_before_removal must agree with the local mirror."""
    import numpy as np
    from gareus.adaptive_feedback import _check_per_cell_overlap_before_removal

    rng = np.random.default_rng(42)
    samples = {
        (0, 0): rng.normal(0.5, 0.15, 200).tolist(),
        (2, 0): rng.normal(0.7, 0.15, 200).tolist(),
        (0, 1): rng.normal(0.3, 0.15, 200).tolist(),
        (2, 1): rng.normal(0.5, 0.15, 200).tolist(),
    }
    min_ov = 0.05
    real_result = _check_per_cell_overlap_before_removal(
        primary_idx=1, n_secondary=2,
        samples_2d_by_window=samples, minimum_valid_overlap=min_ov,
    )
    mirror_result = _check_per_cell_overlap(
        primary_idx=1, n_secondary=2,
        samples_2d=samples, min_overlap=min_ov,
    )
    assert real_result == mirror_result, (
        f"real helper ({real_result}) disagrees with local mirror ({mirror_result})"
    )
