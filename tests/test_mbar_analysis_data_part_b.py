import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from gareus.mbar_analysis.data import (
    Data, K_B_KJ_PER_MOL_K, infer_temp_beta, clean, _masked_data,
    _apply_analysis_stride, _filter_epoch_source, _sample_block_ids,
    _skip_first_n_frames,
)


def _mk_data(n=6, k=2, meta=None):
    rng = np.arange(n, dtype=float)
    return Data(
        prod_dir=Path("/p"), out_dir=Path("/o"),
        cv=rng, cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=np.zeros(n, int), replica=np.array([i % 2 for i in range(n)]),
        step=np.arange(n), u_nk=np.zeros((n, k)),
        centers=np.zeros(k), k_kcal=np.zeros(k),
        beta=1.0, temp=300.0, boost_kj=np.full(n, np.nan), potential_kj=None,
        source="test", meta=meta if meta is not None else {},
    )


def test_infer_temp_beta_reads_meta_top_level(tmp_path):
    meta = {"temperature_K": 310.0}
    temp, beta = infer_temp_beta(tmp_path, meta)
    assert temp == 310.0
    assert beta == pytest.approx(1.0 / (K_B_KJ_PER_MOL_K * 310.0))


def test_infer_temp_beta_reads_nested_resolved_args(tmp_path):
    meta = {"resolved_args": {"temperature_k": 305.0}}
    temp, beta = infer_temp_beta(tmp_path, meta)
    assert temp == 305.0


def test_infer_temp_beta_falls_back_to_300k_with_warning(tmp_path):
    with pytest.warns(RuntimeWarning):
        temp, beta = infer_temp_beta(tmp_path, {})
    assert temp == 300.0


def test_clean_drops_nonfinite_cv_rows():
    d = _mk_data(n=4)
    d.cv[1] = float("nan")
    out = clean(d)
    assert out.cv.size == 3
    assert not np.any(np.isnan(out.cv))


def test_clean_keeps_epoch_source_in_sync_when_dropping_rows():
    """Regression: clean() row-filters every per-sample array on a genuine
    drop, but used to leave meta['_epoch_source'] untouched -- any consumer
    that zips it against a per-sample array (e.g. _sample_block_ids,
    _epoch_zero_split_masks, _secondary_cv_epoch_regime_masks) would then
    silently degrade (length-mismatch guard -> fall back to pooled/disabled)
    rather than crash. Found during Plan A3's final whole-plan review, once
    a real Parquet run could genuinely have clean() drop rows (a null cv2
    sample under a secondary-restrained window) for the first time."""
    d = _mk_data(n=4, meta={"_epoch_source": [0, 0, 1, 1]})
    d.cv[1] = float("nan")
    out = clean(d)
    assert out.cv.size == 3
    assert out.meta["_epoch_source"] == [0, 1, 1]


def test_clean_raises_on_shape_mismatch():
    d = _mk_data(n=4, k=2)
    d.u_nk = np.zeros((3, 2))
    with pytest.raises(ValueError):
        clean(d)


def test_masked_data_slices_per_sample_arrays_shares_window_arrays():
    d = _mk_data(n=6)
    mask = np.array([True, False, True, False, True, False])
    out = _masked_data(d, mask)
    assert out.cv.size == 3
    assert out.centers is d.centers  # window-space arrays shared, not sliced
    assert out.beta == d.beta


def test_apply_analysis_stride_keeps_every_nth_sample_per_replica():
    d = _mk_data(n=8)
    d.replica = np.zeros(8, int)
    d.step = np.arange(8)
    out = _apply_analysis_stride(d, stride=2, offset=0)
    assert out.cv.size == 4
    assert list(out.step) == [0, 2, 4, 6]


def test_apply_analysis_stride_noop_for_stride_one():
    d = _mk_data(n=5)
    out = _apply_analysis_stride(d, stride=1, offset=0)
    assert out is d


def test_filter_epoch_source_filters_meta_list_in_place():
    d = _mk_data(n=4, meta={"_epoch_source": [0, 0, 1, 1]})
    keep = np.array([True, False, True, True])
    _filter_epoch_source(d, keep)
    assert d.meta["_epoch_source"] == [0, 1, 1]


def test_sample_block_ids_groups_by_epoch_source_and_replica():
    d = _mk_data(n=4, meta={"_epoch_source": [0, 0, 1, 1]})
    d.replica = np.array([0, 0, 0, 0])
    block_ids = _sample_block_ids(d)
    # Same replica, different epoch source -> different blocks.
    assert block_ids[0] == block_ids[1]
    assert block_ids[0] != block_ids[2]
    assert block_ids[2] == block_ids[3]


def test_sample_block_ids_falls_back_to_replica_when_epoch_source_absent():
    d = _mk_data(n=4)
    d.replica = np.array([0, 1, 0, 1])
    block_ids = _sample_block_ids(d)
    assert block_ids[0] == block_ids[2]
    assert block_ids[1] == block_ids[3]
    assert block_ids[0] != block_ids[1]


def test_skip_first_n_frames_drops_earliest_steps_per_replica():
    d = _mk_data(n=6)
    d.replica = np.array([0, 0, 0, 1, 1, 1])
    d.step = np.array([0, 1, 2, 0, 1, 2])
    out = _skip_first_n_frames(d, 1)
    assert out.cv.size == 4
    assert sorted(out.step[out.replica == 0]) == [1, 2]
    assert sorted(out.step[out.replica == 1]) == [1, 2]
