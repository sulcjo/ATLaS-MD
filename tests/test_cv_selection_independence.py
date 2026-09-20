"""Task 4: held-out independence and information diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from gareus.cv_selection.independence import (frame_partition, grouped_folds,
                                              heldout_nonlinear_r2,
                                              incremental_cell_information)


def _groups(n, k=60, seed=0):
    return np.random.default_rng(seed).integers(0, k, n)


def test_independent_coordinates_have_near_zero_heldout_r2():
    rng = np.random.default_rng(0)
    z1, z2 = rng.normal(size=20000), rng.normal(size=20000)
    pooled, folds = heldout_nonlinear_r2(z2, z1, _groups(20000))
    assert pooled < 0.02 and folds.shape == (4,)


def test_a_quadratic_dependence_survives_linear_residualisation_and_is_detected():
    rng = np.random.default_rng(1)
    z1 = rng.normal(size=20000)
    z2 = z1 ** 2 - 1.0 + 0.3 * rng.normal(size=20000)
    assert abs(np.cov(z1, z2)[0, 1]) < 0.05
    assert heldout_nonlinear_r2(z2, z1, _groups(20000))[0] > 0.6


def test_folds_never_split_a_group_and_are_shuffled():
    groups = np.repeat(np.arange(40), 25)                 # sorted, like plan.csv order
    folds = grouped_folds(groups, 4)
    for hold in folds:
        rest = np.setdiff1d(np.arange(1000), hold)
        assert set(groups[hold]).isdisjoint(set(groups[rest]))
    assert sorted(np.unique(groups[folds[0]]).tolist()) != list(range(0, 40, 4))


def test_folds_are_reproducible_and_cover_every_row():
    groups = _groups(500, k=20)
    a, b = grouped_folds(groups, 4), grouped_folds(groups, 4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert sorted(np.concatenate(a).tolist()) == list(range(500))


def test_too_few_groups_for_the_folds_is_refused():
    with pytest.raises(ValueError, match="folds"):
        grouped_folds(np.array([0, 0, 1, 1]), 4)


def test_a_perfect_new_coordinate_scores_high_not_zero():
    """v0.1's min(L1, L2) - L12 returned -0.003 here. L1 - L12 must return the truth."""
    rng = np.random.default_rng(2)
    z1 = rng.normal(size=20000)
    z2 = rng.normal(size=20000)
    cells = (z2 > 0).astype(int)
    out = incremental_cell_information(cells, z1, z2, _groups(20000))
    assert out["gain"] > 0.5 and out["l2"] < out["l1"]


def test_a_copy_of_z1_adds_no_information():
    rng = np.random.default_rng(3)
    z1 = rng.normal(size=20000)
    cells = (z1 > 0).astype(int)
    assert abs(incremental_cell_information(cells, z1, 2.0 * z1, _groups(20000))["gain"]) < 0.05


def test_gain_of_pure_noise_sits_at_the_smoothing_floor():
    rng = np.random.default_rng(5)
    z1 = rng.normal(size=20000)
    noise = rng.normal(size=20000)
    cells = rng.integers(0, 4, 20000)
    gain = incremental_cell_information(cells, z1, noise, _groups(20000))["gain"]
    assert abs(gain) < 0.02


def test_frame_partition_is_deterministic_and_covers_every_frame():
    feats = np.random.default_rng(4).normal(size=(5000, 12))
    labels_a, centres = frame_partition(feats, n_cells=8)
    labels_b, _ = frame_partition(feats, n_cells=8)
    assert np.array_equal(labels_a, labels_b) and centres.shape == (8, 12)
    assert set(np.unique(labels_a).tolist()) == set(range(8))


def test_frame_partition_refuses_fewer_frames_than_cells():
    with pytest.raises(ValueError):
        frame_partition(np.zeros((3, 2)), n_cells=8)
