"""Tests for gareus.query._concat_numpy_dicts across a mid-campaign schema change.

A production campaign can resume with new segments written after a sample-column
addition (e.g. the lambda-ladder's v_pep_kj_mol/v_dih_kj_mol/gamd_lambda). Older
segments' Parquet files simply lack the new column entirely -- not merely null
values, the key itself is absent from that segment's numpy column-dict. These
tests exercise _concat_numpy_dicts directly with hand-built column-dicts that
mimic DuckDB's fetchnumpy() output, so no Parquet/DuckDB round-trip is needed.
"""
from __future__ import annotations

import numpy as np
import pytest


def _old_segment():
    """A pre-schema-change segment: no gamd_lambda / v_pep_kj_mol column at all."""
    return {
        "step": np.array([0, 100, 200], dtype=np.int64),
        "replica": np.array([0, 1, 0], dtype=np.int16),
        "window_id": np.array([0, 1, 0], dtype=np.int16),
        "cv1": np.array([0.10, 0.20, 0.11], dtype=np.float64),
    }


def _new_segment():
    """A post-schema-change segment: carries the new columns."""
    return {
        "step": np.array([300, 400], dtype=np.int64),
        "replica": np.array([0, 1], dtype=np.int16),
        "window_id": np.array([0, 1], dtype=np.int16),
        "cv1": np.array([0.12, 0.21], dtype=np.float64),
        "gamd_lambda": np.array([0.5, 0.5], dtype=np.float32),
        "v_pep_kj_mol": np.array([12.5, 13.5], dtype=np.float32),
    }


def _as_nan_filled(col) -> np.ndarray:
    return np.ma.filled(col, np.nan).astype(np.float64) if np.ma.isMaskedArray(col) else np.asarray(col, dtype=np.float64)


def _assert_schema_evolution_result(combined: dict) -> None:
    assert {"step", "replica", "window_id", "cv1", "gamd_lambda", "v_pep_kj_mol"} <= set(combined.keys())
    # Sorted by step ascending regardless of input segment order.
    assert list(combined["step"]) == [0, 100, 200, 300, 400]
    lam = _as_nan_filled(combined["gamd_lambda"])
    vpep = _as_nan_filled(combined["v_pep_kj_mol"])
    # Old segment's 3 rows (step 0,100,200): the column was absent -> NaN, never a fabricated 0.0.
    assert np.all(np.isnan(lam[:3])) and np.all(np.isnan(vpep[:3]))
    # New segment's 2 rows (step 300,400): real values intact.
    assert np.allclose(lam[3:], [0.5, 0.5], atol=1e-6)
    assert np.allclose(vpep[3:], [12.5, 13.5], atol=1e-4)


def test_schema_evolution_old_segment_first_fills_nan_not_dropped():
    """oldest-first order (the typical segments.json order) must not silently
    drop the new column from the whole loaded dataset."""
    from gareus.query import _concat_numpy_dicts
    combined = _concat_numpy_dicts([_old_segment(), _new_segment()])
    _assert_schema_evolution_result(combined)


def test_schema_evolution_new_segment_first_does_not_crash():
    """reversed order must not raise a bare KeyError on the older segment."""
    from gareus.query import _concat_numpy_dicts
    combined = _concat_numpy_dicts([_new_segment(), _old_segment()])
    _assert_schema_evolution_result(combined)


def test_column_present_in_all_segments_concatenates_unchanged():
    """Regression: a column with no schema evolution concatenates exactly as
    it did before this fix (same values, same final sort order)."""
    from gareus.query import _concat_numpy_dicts
    seg_a = {
        "step": np.array([0, 10], dtype=np.int64),
        "replica": np.array([0, 0], dtype=np.int16),
        "cv1": np.array([1.0, 2.0], dtype=np.float64),
    }
    seg_b = {
        "step": np.array([20, 30], dtype=np.int64),
        "replica": np.array([0, 0], dtype=np.int16),
        "cv1": np.array([3.0, 4.0], dtype=np.float64),
    }
    combined = _concat_numpy_dicts([seg_a, seg_b])
    assert list(combined["step"]) == [0, 10, 20, 30]
    assert list(combined["cv1"]) == [1.0, 2.0, 3.0, 4.0]
    assert not np.ma.isMaskedArray(combined["cv1"])  # no schema evolution -> stays plain


def test_mismatched_column_length_within_one_segment_raises():
    """A ragged segment (a column shorter/longer than 'step') is a real bug
    and must fail loudly, not silently misalign or truncate."""
    from gareus.query import _concat_numpy_dicts
    bad_segment = {
        "step": np.array([0, 100, 200], dtype=np.int64),
        "cv1": np.array([0.1, 0.2], dtype=np.float64),  # length 2, not 3
    }
    with pytest.raises(ValueError):
        _concat_numpy_dicts([bad_segment])


def test_missing_int_column_is_masked_not_fabricated_zero():
    """A non-float column absent from a segment must be masked, never a bare 0."""
    from gareus.query import _concat_numpy_dicts
    seg_old = {
        "step": np.array([0, 100], dtype=np.int64),
        "cv1": np.array([0.1, 0.2], dtype=np.float64),
    }
    seg_new = {
        "step": np.array([200], dtype=np.int64),
        "cv1": np.array([0.3], dtype=np.float64),
        "window_id": np.array([2], dtype=np.int16),
    }
    combined = _concat_numpy_dicts([seg_old, seg_new])
    wid = combined["window_id"]
    assert np.ma.isMaskedArray(wid)
    assert bool(wid.mask[0]) and bool(wid.mask[1])   # the two old-segment rows are masked
    assert not bool(wid.mask[2]) and int(wid[2]) == 2
