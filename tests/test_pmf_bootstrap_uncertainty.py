import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze_gareus_mbar as agm


def _make_data_stub(replica, epoch_source=None):
    """Minimal object exposing only the fields _sample_block_ids reads."""
    class _Stub:
        pass
    d = _Stub()
    d.replica = np.asarray(replica)
    d.meta = {}
    if epoch_source is not None:
        d.meta['_epoch_source'] = list(epoch_source)
    return d


def test_block_ids_group_by_replica_when_no_epoch_source():
    d = _make_data_stub(replica=[0, 0, 1, 1, 2])
    block_ids = agm._sample_block_ids(d)
    # Same replica -> same block id; different replica -> different block id.
    assert block_ids[0] == block_ids[1]
    assert block_ids[2] == block_ids[3]
    assert len({block_ids[0], block_ids[2], block_ids[4]}) == 3


def test_block_ids_distinguish_same_replica_across_epoch_sources():
    # Replica 0 in epoch source 0 and replica 0 in epoch source 1 must be
    # DIFFERENT blocks -- adaptive-production runs don't guarantee trajectory
    # continuity across epoch boundaries.
    d = _make_data_stub(replica=[0, 0, 0, 0], epoch_source=[0, 0, 1, 1])
    block_ids = agm._sample_block_ids(d)
    assert block_ids[0] == block_ids[1]
    assert block_ids[2] == block_ids[3]
    assert block_ids[0] != block_ids[2]


def test_block_ids_are_compact_zero_based():
    d = _make_data_stub(replica=[5, 5, 9, 9, 5])
    block_ids = agm._sample_block_ids(d)
    assert set(np.unique(block_ids)) == {0, 1}
    assert block_ids.dtype == np.int64


KBT_KCAL = 0.596  # ~300K, matches the file's own documented round-trip check


def _synthetic_blocked_cv(rng, n_blocks=5, samples_per_block=200,
                           between_block_std=0.3, within_block_std=0.5):
    """One window's worth of samples: `n_blocks` blocks, each block a tight
    cluster around its own randomly-offset center. between_block_std >>
    within_block_std means the block structure carries real information a
    naive per-sample bootstrap would miss."""
    block_offsets = rng.normal(0.0, between_block_std, size=n_blocks)
    cv_parts, block_id_parts = [], []
    for b in range(n_blocks):
        cv_parts.append(block_offsets[b] + rng.normal(0.0, within_block_std, size=samples_per_block))
        block_id_parts.append(np.full(samples_per_block, b, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    block_ids = np.concatenate(block_id_parts)
    window = np.zeros_like(cv, dtype=np.int64)
    logw = np.zeros_like(cv)  # uniform weight
    boost = np.full_like(cv, np.nan)  # no GaMD boost -> exercise umbrella_only
    return cv, block_ids, window, logw, boost


def _naive_per_sample_bootstrap_std(cv, logw, bins, n_boot, rng):
    """Reference implementation for the block-bootstrap validity test only
    -- NOT the production code path. Resamples individual samples, ignoring
    block structure entirely."""
    n = cv.size
    reps = np.full((n_boot, len(bins) - 1), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        w = agm.norm_logw(logw[idx])
        reps[b] = agm.pmf_from_weights(cv[idx], w, bins, KBT_KCAL)['pmf']
    with np.errstate(invalid='ignore'):
        return np.nanstd(reps, axis=0)


def test_block_bootstrap_reports_wider_uncertainty_than_naive_per_sample():
    rng = np.random.default_rng(1)
    cv, block_ids, window, logw, boost = _synthetic_blocked_cv(rng)
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    main_pmf = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)

    block_std = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=400, rng=np.random.default_rng(0),
    )['pmf_std']
    naive_std = _naive_per_sample_bootstrap_std(cv, logw, bins, n_boot=400, rng=np.random.default_rng(0))

    finite = np.isfinite(block_std) & np.isfinite(naive_std)
    assert finite.sum() >= 5
    assert np.mean(block_std[finite]) > np.mean(naive_std[finite])


def test_bootstrap_replicates_anchor_to_main_pmf_minimum_not_their_own():
    rng = np.random.default_rng(2)
    cv, block_ids, window, logw, boost = _synthetic_blocked_cv(rng)
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    main_pmf = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)
    minidx = int(np.nanargmin(main_pmf['pmf']))

    result = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=100, rng=np.random.default_rng(0),
    )
    # By construction every replicate is re-anchored to read exactly 0 at
    # the bin the MAIN pmf calls its minimum -- so the std AT THAT BIN must
    # be exactly 0. If replicates were instead anchored to their own
    # individual minima, this would generally be nonzero.
    assert result['pmf_std'][minidx] == pytest.approx(0.0, abs=1e-12)


def test_low_block_count_window_is_flagged():
    rng = np.random.default_rng(3)
    cv0, block_ids0, _, logw0, boost0 = _synthetic_blocked_cv(rng, n_blocks=5)
    cv1, block_ids1, _, logw1, boost1 = _synthetic_blocked_cv(rng, n_blocks=1, samples_per_block=50)
    cv = np.concatenate([cv0, cv1 + 10.0])  # push window-1 samples into their own CV range
    block_ids = np.concatenate([block_ids0, block_ids1 + 1000])  # keep block ids globally distinct
    window = np.concatenate([np.zeros_like(cv0, dtype=np.int64), np.ones_like(cv1, dtype=np.int64)])
    logw = np.concatenate([logw0, logw1])
    boost = np.concatenate([boost0, boost1])
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    main_pmf = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)

    result = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=50, rng=np.random.default_rng(0),
    )
    assert result['blocks_per_window'].tolist() == [5, 1]
    assert result['low_block_windows'] == [1]


def test_bootstrap_uncertainty_is_deterministic_given_same_seed():
    rng = np.random.default_rng(4)
    cv, block_ids, window, logw, boost = _synthetic_blocked_cv(rng)
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    main_pmf = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)

    result_a = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=50, rng=np.random.default_rng(7),
    )['pmf_std']
    result_b = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=50, rng=np.random.default_rng(7),
    )['pmf_std']
    result_c = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=50, rng=np.random.default_rng(8),
    )['pmf_std']
    assert np.array_equal(result_a, result_b, equal_nan=True)
    assert not np.array_equal(result_a, result_c, equal_nan=True)


def test_bootstrap_handles_gamd_cumulant2_selected_method():
    rng = np.random.default_rng(5)
    cv, block_ids, window, _, _ = _synthetic_blocked_cv(rng)
    logw = np.zeros_like(cv)
    boost = rng.normal(50.0, 8.0, size=cv.size)  # realistic GaMD boost scale, kJ/mol
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    base_w = agm.norm_logw(logw)
    main_pmf, _ = agm._cumulant_expansion(cv, base_w, boost, bins, beta, KBT_KCAL, order=2)

    result = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'gamd_cumulant2', main_pmf, n_boot=30, rng=np.random.default_rng(0),
    )
    assert result['pmf_std'].shape == main_pmf['pmf'].shape
    assert np.any(np.isfinite(result['pmf_std']))


def test_bootstrap_handles_gamd_exponential_selected_method():
    rng = np.random.default_rng(6)
    cv, block_ids, window, _, _ = _synthetic_blocked_cv(rng)
    logw = np.zeros_like(cv)
    boost = rng.normal(50.0, 8.0, size=cv.size)  # realistic GaMD boost scale, kJ/mol
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    # gamd_exponential computes PMF via: norm_logw(logw + beta*boost) -> pmf_from_weights
    w_exp = agm.norm_logw(logw + beta * boost)
    main_pmf = agm.pmf_from_weights(cv, w_exp, bins, KBT_KCAL)

    result = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'gamd_exponential', main_pmf, n_boot=30, rng=np.random.default_rng(0),
    )
    assert result['pmf_std'].shape == main_pmf['pmf'].shape
    assert np.any(np.isfinite(result['pmf_std']))


def test_bootstrap_handles_gamd_cumulant3_selected_method():
    rng = np.random.default_rng(7)
    cv, block_ids, window, _, _ = _synthetic_blocked_cv(rng)
    logw = np.zeros_like(cv)
    boost = rng.normal(50.0, 8.0, size=cv.size)  # realistic GaMD boost scale, kJ/mol
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    base_w = agm.norm_logw(logw)
    main_pmf, _ = agm._cumulant_expansion(cv, base_w, boost, bins, beta, KBT_KCAL, order=3)

    result = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'gamd_cumulant3', main_pmf, n_boot=30, rng=np.random.default_rng(0),
    )
    assert result['pmf_std'].shape == main_pmf['pmf'].shape
    assert np.any(np.isfinite(result['pmf_std']))
