import numpy as np
import pytest
import sys
import warnings
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


def test_block_ids_falls_back_to_replica_only_when_epoch_source_is_stale():
    """clean() row-filters every per-sample array by a finiteness mask but
    does NOT sync meta['_epoch_source'] -- every loader that populates
    _epoch_source calls clean(Data(...)), so a dropped sample leaves
    _epoch_source one (or more) entries too long relative to d.replica,
    permanently. _sample_block_ids must not crash on this (np.stack would
    otherwise raise 'all input arrays must have the same shape'); it should
    fall back to replica-only blocks, identical to calling it with no
    _epoch_source at all on the same replica array."""
    d_stale = _make_data_stub(replica=[0, 0, 1, 1, 2], epoch_source=[0, 0, 0, 1, 1, 2])
    d_no_epoch_source = _make_data_stub(replica=[0, 0, 1, 1, 2])

    block_ids_stale = agm._sample_block_ids(d_stale)  # must not raise
    block_ids_fallback = agm._sample_block_ids(d_no_epoch_source)

    assert np.array_equal(block_ids_stale, block_ids_fallback)


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


def test_bootstrap_smooth_logfac_sigma_reaches_cumulant_expansion():
    """smooth_logfac_sigma must be threaded through to the internal
    _cumulant_expansion call for the gamd_cumulant2/3 branches -- otherwise
    the bootstrap replicates describe a different (unsmoothed) curve than
    whatever smoothed main_pmf is actually being reported. Passing a
    nonzero value must change the result relative to smooth_logfac_sigma=0,
    and omitting the argument entirely must still default to 0.0 (matching
    every existing call site/test in this file, unmodified)."""
    rng = np.random.default_rng(8)
    cv, block_ids, window, _, _ = _synthetic_blocked_cv(rng)
    logw = np.zeros_like(cv)
    boost = rng.normal(50.0, 8.0, size=cv.size)
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    base_w = agm.norm_logw(logw)
    main_pmf, _ = agm._cumulant_expansion(cv, base_w, boost, bins, beta, KBT_KCAL, order=2, smooth_logfac_sigma=1.5)

    result_smoothed = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'gamd_cumulant2', main_pmf, n_boot=30, rng=np.random.default_rng(0),
        smooth_logfac_sigma=1.5,
    )['pmf_std']
    result_default = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'gamd_cumulant2', main_pmf, n_boot=30, rng=np.random.default_rng(0),
    )['pmf_std']
    result_explicit_zero = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'gamd_cumulant2', main_pmf, n_boot=30, rng=np.random.default_rng(0),
        smooth_logfac_sigma=0.0,
    )['pmf_std']

    # Omitting the argument is bit-for-bit identical to passing 0.0 explicitly.
    assert np.array_equal(result_default, result_explicit_zero, equal_nan=True)

    finite = np.isfinite(result_smoothed) & np.isfinite(result_default)
    assert finite.sum() >= 3
    assert not np.allclose(result_smoothed[finite], result_default[finite])


def test_2d_bootstrap_collapses_to_1d_case_with_degenerate_y():
    rng = np.random.default_rng(6)
    cv, block_ids, window, logw, boost = _synthetic_blocked_cv(rng)
    y = np.zeros_like(cv)
    bins = agm.make_bins(cv, 15, None, None)
    ybins = np.array([-0.5, 0.5])
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)

    main_pmf_1d = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)
    main_fes_2d = agm.pmf2d_from_weights(cv, y, agm.norm_logw(logw), bins, ybins, KBT_KCAL)

    result_1d = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf_1d, n_boot=100, rng=np.random.default_rng(42),
    )
    result_2d = agm._bootstrap_pmf_uncertainty_2d(
        cv, y, logw, boost, bins, ybins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_fes_2d, n_boot=100, rng=np.random.default_rng(42),
    )
    # Same seed, same window/block structure -> both helpers draw the exact
    # same sequence of resampled index sets, so the collapsed (single-y-bin)
    # 2D result must match the 1D result bit-for-bit.
    assert np.allclose(result_2d['pmf_std'][:, 0], result_1d['pmf_std'], atol=1e-10, equal_nan=True)


def test_pmf_uncertainty_flags_default_off_and_parse_correctly():
    args = agm.parse_args(['some_run_dir'])
    assert args.pmf_uncertainty is False
    assert args.pmf_uncertainty_n_boot == 100
    assert args.pmf_uncertainty_seed == 0

    args_on = agm.parse_args(['some_run_dir', '--pmf-uncertainty',
                               '--pmf-uncertainty-n-boot', '250',
                               '--pmf-uncertainty-seed', '7'])
    assert args_on.pmf_uncertainty is True
    assert args_on.pmf_uncertainty_n_boot == 250
    assert args_on.pmf_uncertainty_seed == 7


import csv as _csv
from pathlib import Path as _Path


def _make_real_data(tmp_path, rng, n_windows=3, samples_per_window=300, n_blocks_per_window=5):
    """A minimal but real analyze_gareus_mbar.Data with a genuine multi-window
    harmonic-umbrella u_nk, suitable for a real solve_mbar + run_pmf_and_gamd_boost_report call."""
    centers = np.linspace(-1.0, 1.0, n_windows)
    k_kcal = np.full(n_windows, 5.0)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)

    cv_parts, window_parts, replica_parts, epoch_src_parts = [], [], [], []
    for k in range(n_windows):
        cv_parts.append(centers[k] + rng.normal(0.0, 0.3, size=samples_per_window))
        window_parts.append(np.full(samples_per_window, k, dtype=np.int64))
        # Spread this window's samples across n_blocks_per_window (replica, epoch_source=0) blocks.
        replica_parts.append(np.repeat(np.arange(n_blocks_per_window), samples_per_window // n_blocks_per_window))
        epoch_src_parts.append(np.zeros(samples_per_window, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(window_parts)
    replica = np.concatenate([p[:len(w)] for p, w in zip(replica_parts, window_parts)])
    epoch_src = np.concatenate(epoch_src_parts)
    n = cv.size

    u_nk = np.zeros((n, n_windows), dtype=np.float64)
    for k in range(n_windows):
        u_nk[:, k] = beta * agm.KJ_PER_KCAL * 0.5 * k_kcal[k] * (cv - centers[k]) ** 2

    out_dir = _Path(tmp_path) / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)
    d = agm.Data(
        prod_dir=_Path(tmp_path), out_dir=out_dir, cv=cv, cv2=np.full(n, np.nan),
        rg_A=np.full(n, np.nan), window=window, replica=replica, step=np.arange(n),
        u_nk=u_nk, centers=centers, k_kcal=k_kcal, beta=beta, temp=300.0,
        boost_kj=np.full(n, np.nan), potential_kj=None, source='test',
        meta={'_epoch_source': epoch_src.tolist()},
    )
    return d


class _Args:
    def __init__(self, **kw):
        self.bins = 20
        self.min_neighbor_overlap = 0.0
        self.selected_method = 'auto'
        self.gamd_smooth_sigma = 0.0
        self.pmf_smooth_sigma = 0.0
        self.pmf_uncertainty = False
        self.pmf_uncertainty_n_boot = 30
        self.pmf_uncertainty_seed = 0
        self.__dict__.update(kw)


def test_pmf_uncertainty_off_by_default_leaves_csv_unchanged(tmp_path):
    rng = np.random.default_rng(10)
    d = _make_real_data(tmp_path, rng)
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    args_off = _Args(pmf_uncertainty=False)
    info_off = agm.run_pmf_and_gamd_boost_report(d, args_off, logw, bins, kbt_kcal, d.out_dir, [], None)
    assert info_off.get('pmf_uncertainty_std') is None

    with (d.out_dir / 'pmf_unbiased.csv').open() as f:
        header = next(_csv.reader(f))
    assert 'pmf_std_kcal_mol' not in header


def test_pmf_uncertainty_on_adds_std_column_and_return_value(tmp_path):
    rng = np.random.default_rng(11)
    d = _make_real_data(tmp_path, rng)
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    args_on = _Args(pmf_uncertainty=True)
    warnings = []
    info_on = agm.run_pmf_and_gamd_boost_report(d, args_on, logw, bins, kbt_kcal, d.out_dir, warnings, None)
    assert info_on.get('pmf_uncertainty_std') is not None
    assert info_on['pmf_uncertainty_std'].shape == info_on['pmfs'][info_on['selected']]['pmf'].shape

    with (d.out_dir / 'pmf_unbiased.csv').open() as f:
        header = next(_csv.reader(f))
    assert 'pmf_std_kcal_mol' in header


def test_pmf_uncertainty_warns_on_low_block_count_windows(tmp_path):
    rng = np.random.default_rng(12)
    # n_blocks_per_window=1 -> every window is a "low block count" window.
    d = _make_real_data(tmp_path, rng, n_blocks_per_window=1)
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    args_on = _Args(pmf_uncertainty=True)
    warnings = []
    agm.run_pmf_and_gamd_boost_report(d, args_on, logw, bins, kbt_kcal, d.out_dir, warnings, None)
    assert any('uncertainty' in w.lower() and 'block' in w.lower() for w in warnings)


def test_pmf_uncertainty_skips_gracefully_when_selected_pmf_is_all_nan(tmp_path):
    """sel['pmf'] can be entirely non-finite (see the finite/minidx guard
    right above the uncertainty insertion point in
    run_pmf_and_gamd_boost_report) -- an all-NaN case is reachable in
    practice by GaMD-cumulant NaN-bin-flagging combined with smoothing.
    _bootstrap_pmf_uncertainty_1d's own np.nanargmin on an all-NaN
    main_pmf['pmf'] raises ValueError (verified directly), so the
    integration point must not call it when minidx<0 -- exercised here via
    the function's own extra_pmfs/selected_method-forcing mechanism to
    construct a genuinely all-NaN selected PMF without fragile GaMD-data
    crafting."""
    rng = np.random.default_rng(13)
    d = _make_real_data(tmp_path, rng)
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL
    n_bins = len(bins) - 1

    all_nan_pmf = {
        'cv_A': 0.5 * (bins[:-1] + bins[1:]),
        'prob': np.zeros(n_bins),
        'pmf': np.full(n_bins, np.nan),
        'counts': np.zeros(n_bins, dtype=int),
    }
    args_on = _Args(pmf_uncertainty=True, selected_method='forced_all_nan')
    warnings = []
    info = agm.run_pmf_and_gamd_boost_report(
        d, args_on, logw, bins, kbt_kcal, d.out_dir, warnings, None,
        extra_pmfs={'forced_all_nan': all_nan_pmf},
    )
    assert info['selected'] == 'forced_all_nan'
    assert info.get('pmf_uncertainty_std') is None

    with (d.out_dir / 'pmf_unbiased.csv').open() as f:
        header = next(_csv.reader(f))
    assert 'pmf_std_kcal_mol' not in header


def test_pmf_uncertainty_dispatches_on_actual_pmf_not_forced_selected_label(tmp_path):
    """`selected` can be force-overridden to a gamd_* method name via
    --selected-method even when boost_ok is False -- `sel` is then really
    the umbrella-only PMF under the hood (d.boost_kj all-NaN, boost_ok
    False), but before the fix the bootstrap dispatched on the forced label
    and tried to rebuild gamd_cumulant2-style replicates from an all-NaN
    boost array, producing an all-NaN pmf_std plus a spurious numpy
    'Degrees of freedom <= 0 for slice' RuntimeWarning. The bootstrap must
    dispatch on what `sel` actually contains instead, so this is a real
    umbrella-only bootstrap that produces finite uncertainty and no
    warning."""
    rng = np.random.default_rng(14)
    d = _make_real_data(tmp_path, rng)  # boost_kj is all-NaN -> boost_ok False
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    args_on = _Args(pmf_uncertainty=True, selected_method='gamd_cumulant2')
    warn_list = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        info = agm.run_pmf_and_gamd_boost_report(
            d, args_on, logw, bins, kbt_kcal, d.out_dir, warn_list, None,
        )

    assert info['boost_ok'] is False
    assert info['selected'] == 'gamd_cumulant2'  # label is still forced, per existing behavior
    std = info.get('pmf_uncertainty_std')
    assert std is not None
    assert np.any(np.isfinite(std))  # a real umbrella-only bootstrap ran, not an all-NaN one

    dof_warnings = [w for w in caught if 'degrees of freedom' in str(w.message).lower()]
    assert not dof_warnings
