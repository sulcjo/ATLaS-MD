"""Regression tests for the masked-array-to-NaN gap at the Parquet loader
boundary (Plan A3 Task 1b).

Task 1 (see tests/test_query_reconstruct_bias_matrix_nan_guard.py) fixed
`gareus.query.reconstruct_bias_matrix` so a sample whose own secondary CV
(cv2) is not finite gets NaN in that (sample, window) bias-matrix entry --
excluding it from MBAR -- instead of fabricating a "0 deviation" (as if the
sample were exactly on-target).

That fix had zero real-world effect for the two loaders that feed real
production analyses (`gareus/mbar_analysis/loaders.py`'s `load_parquet` and
`gareus/mbar_analysis/loaders_union_parquet.py`'s
`load_parquet_adaptive_union`), because DuckDB's `fetchnumpy()` returns a
`numpy.ma.MaskedArray` for any column with a real SQL NULL (e.g. an
unmeasured secondary CV on a CV1-only run), and both loaders cast that
masked array straight to float64 without unmasking first -- silently
exposing the masked array's arbitrary fill value (observed: 0.0 for a
float32 Parquet column) as if it were real, measured data.

Both loaders now route through the new `gareus.mbar_analysis.data`
helper `_fill_masked_nan`, mirroring the pattern already used by
`gareus.query.export_analysis_arrays_npz`.

While building a REAL null-containing Parquet fixture and reading it back
through the loaders' actual real code path (`gareus.query.load_samples`), a
SECOND, deeper bug was found upstream of both call sites named above, in
`gareus.query._concat_numpy_dicts` (the function `load_samples` always uses
to merge DuckDB's per-segment `fetchnumpy()` results): plain `np.concatenate`
on a list containing a `numpy.ma.MaskedArray` preserves the MaskedArray
*subclass* on its output but silently drops the *mask itself* -- reproduced
directly even for a single-element input list, i.e. every single-segment run
hits this, not just multi-segment concatenation. This meant `samples.get(
'cv2')` in both loaders never received a real mask to begin with, no matter
what the loader itself did with it -- and it equally defeats
`gareus.query.export_analysis_arrays_npz`'s own already-`np.ma`-aware cast
(the reference pattern this fix's loader changes were modeled on), since that
function also calls `load_samples` first. Fixed alongside the two named call
sites: `_concat_numpy_dicts` now uses `np.ma.concatenate` for any column
DuckDB actually returned as masked, keeping the cheaper plain
`np.concatenate` for ordinary (never-masked) columns like `step`/`window_id`.

Note on `clean()` (`gareus/mbar_analysis/data.py`): its sample mask is
`np.isfinite(d.cv) & np.all(np.isfinite(d.u_nk), axis=1)` -- ANY window
column being NaN for a row drops that row from the whole returned `Data`,
regardless of which window the sample was actually collected under (MBAR
needs every retained sample's energy in every window). So once a run has
at least one real secondary-restrained window anywhere in its state/window
set, a null-cv2 sample is *always* excluded by `clean()`, by design -- the
same "exclusion-by-propagation" Task 1 already established. To observe the
literal `Data.cv2 == NaN` (not 0.0) on a *surviving* row, the fixtures below
use a run with no secondary restraint at all (a genuine CV1-only run --
exactly the scenario this bug affects in production, per CLAUDE.md). To
independently confirm the u_nk-level fix (NaN, not a finite bias computed
against a fabricated cv2=0.0) for a window that DOES secondary-restrain,
each integration test also exercises the loader's own real
(`load_samples`/`reconstruct_bias_matrix` or
`load_samples`/`_reconstruct_union_bias_block`) code path directly on the
same real masked-array data, since that specific combination cannot survive
`clean()` to be inspected on the final returned `Data` object.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from gareus.units import K_B_KJ_PER_MOL_K
from gareus.mbar_analysis.data import _fill_masked_nan


# --- _fill_masked_nan, direct unit tests -------------------------------------

def test_fill_masked_nan_none_input_returns_none():
    assert _fill_masked_nan(None) is None


def test_fill_masked_nan_masked_array_maps_masked_entries_to_nan():
    arr = np.ma.array([1.0, 0.0, 3.0], mask=[False, True, False], dtype=np.float32)
    out = _fill_masked_nan(arr)
    assert not np.ma.isMaskedArray(out)
    assert out.dtype == np.float64
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], [1.0, 3.0])


def test_fill_masked_nan_plain_array_passes_through_unchanged():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _fill_masked_nan(arr)
    assert not np.ma.isMaskedArray(out)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_fill_masked_nan_plain_list_input_converts_to_float64():
    out = _fill_masked_nan([1, 2, 3])
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


# --- gareus.query._concat_numpy_dicts: the deeper, upstream root cause ------

def test_concat_numpy_dicts_preserves_mask_single_result():
    """Reproduces the exact `np.concatenate([single_masked_array])` mask-loss
    gotcha for a single-segment run (the common case -- every real run has
    at least one segment, so this path is always hit).
    """
    from gareus.query import _concat_numpy_dicts

    cv2 = np.ma.array([1.0, 0.0, 3.0], mask=[False, True, False])
    result = {"step": np.array([0, 1, 2]), "replica": np.array([0, 0, 0]), "cv2": cv2}
    out = _concat_numpy_dicts([result])
    assert np.ma.isMaskedArray(out["cv2"])
    assert list(np.ma.getmaskarray(out["cv2"])) == [False, True, False]


def test_concat_numpy_dicts_preserves_mask_multi_result():
    """Same gotcha, across two segments' worth of results being merged --
    also verifies the mask survives the function's own (step, replica)
    lexsort reindex, not just the initial concatenation.
    """
    from gareus.query import _concat_numpy_dicts

    r1 = {"step": np.array([0, 1]), "replica": np.array([0, 0]),
          "cv2": np.ma.array([1.0, 0.0], mask=[False, True])}
    r2 = {"step": np.array([3, 2]), "replica": np.array([0, 0]),
          "cv2": np.ma.array([0.0, 4.0], mask=[True, False])}
    out = _concat_numpy_dicts([r1, r2])
    # Reindexed to ascending step order: 0, 1, 2, 3
    np.testing.assert_array_equal(out["step"], [0, 1, 2, 3])
    assert list(np.ma.getmaskarray(out["cv2"])) == [False, True, False, True]
    np.testing.assert_allclose(np.ma.getdata(out["cv2"]), [1.0, 0.0, 4.0, 0.0])


def test_concat_numpy_dicts_plain_columns_stay_plain_ndarray():
    """Columns DuckDB never returns as masked (e.g. step) must not be
    needlessly wrapped in a MaskedArray -- this would allocate a mask array
    on every sample for no reason on the multi-million-row hot path.
    """
    from gareus.query import _concat_numpy_dicts

    result = {"step": np.array([0, 1]), "replica": np.array([0, 0]),
              "cv1": np.array([1.0, 2.0])}
    out = _concat_numpy_dicts([result])
    assert not np.ma.isMaskedArray(out["step"])
    assert not np.ma.isMaskedArray(out["cv1"])


# --- load_parquet: real null-containing Parquet through the real loader -----

def test_load_parquet_masks_null_cv2_as_nan_not_zero(tmp_path):
    from gareus.query import load_samples, load_windows, reconstruct_bias_matrix
    from gareus.mbar_analysis.loaders import load_parquet
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    prod = tmp_path

    # A genuine CV1-only window set (no secondary restraint anywhere) so a
    # null-cv2 sample survives clean() -- this is the realistic scenario this
    # bug hits in production (CV1-only runs always write a blank/null cv2).
    windows = [{"window_id": 0, "center1": 0.0, "k1": 10.0}]
    reg = SegmentRegistry(prod)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(prod).snapshot(seg_id, windows, cv1_type="contacts", cv2_type=None)

    writer = ParquetSampleWriter(prod / "samples" / seg_id, flush_rows=1000)
    writer.write_sample(0, 0, 0, 0.0, 1.0, -100.0, 1.0, 0.6, 0.4)     # finite cv2
    writer.write_sample(50, 0, 0, 0.0, None, -100.0, 1.0, 0.6, 0.4)   # real SQL NULL cv2
    writer.write_sample(100, 0, 0, 0.0, None, -100.0, 1.0, 0.6, 0.4)  # real SQL NULL cv2
    writer.write_sample(150, 0, 0, 0.0, 7.0, -100.0, 1.0, 0.6, 0.4)   # finite cv2, distinctly non-zero
    writer.close()
    reg.close_segment(seg_id, end_step=150)

    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))

    d = load_parquet(prod)

    # Confirm DuckDB really did hand back a masked array for this column
    # (sanity: if this ever stops being true, the rest of this test would
    # pass vacuously).
    raw_samples = load_samples(prod)
    assert np.ma.isMaskedArray(raw_samples["cv2"])

    # No window secondary-restrains, so clean() must keep every sample.
    assert d.cv.size == 4

    null_mask = np.isnan(d.cv2)
    assert int(null_mask.sum()) == 2
    # The old bug's exact symptom: the masked fill value (0.0) leaking through
    # as if it were real, measured data. Must never appear.
    assert not np.any(d.cv2 == 0.0)
    np.testing.assert_allclose(sorted(d.cv2[~null_mask]), [1.0, 7.0])

    # Independently confirm the u_nk-level fix for a window that DOES
    # secondary-restrain, via the loader's own real code path
    # (load_samples -> _fill_masked_nan -> reconstruct_bias_matrix). This
    # exact combination (null cv2 + a secondary-restrained window) cannot
    # survive clean()'s all-columns-finite mask on the final Data object --
    # see module docstring -- so it's checked directly here instead.
    restrained_windows = [{"center1": 0.0, "k1": 10.0, "center2": 0.0, "k2": 50.0}]
    beta = 1.0 / (K_B_KJ_PER_MOL_K * 300.0)
    cv1 = raw_samples["cv1"].astype(np.float64)

    cv2_fixed = _fill_masked_nan(raw_samples["cv2"])
    u_nk_fixed = reconstruct_bias_matrix(cv1, cv2_fixed, restrained_windows, beta)
    null_rows = np.where(np.isnan(cv2_fixed))[0]
    assert null_rows.size == 2
    assert np.all(np.isnan(u_nk_fixed[null_rows, 0]))

    # And reproduce the exact bug this replaces: `.astype()` on a MaskedArray
    # does preserve the mask itself (unlike np.concatenate -- see
    # gareus/query.py's _concat_numpy_dicts fix), but that mask is a landmine,
    # not a fix: the underlying (mask-blind) data buffer still holds the
    # arbitrary fill value, and the FIRST mask-unaware numpy call downstream
    # (np.asarray, further np.concatenate, plotting, ...) -- including
    # reconstruct_bias_matrix's own internal `np.asarray(cv2, ...)` cast --
    # silently exposes it as if it were real, measured cv2=0.0 data.
    cv2_buggy = raw_samples["cv2"].astype(np.float64)  # the old, buggy line
    assert np.all(np.ma.getdata(cv2_buggy)[null_rows] == 0.0)
    u_nk_buggy = reconstruct_bias_matrix(cv1, cv2_buggy, restrained_windows, beta)
    assert np.all(np.isfinite(u_nk_buggy[null_rows, 0]))


def test_load_parquet_excludes_null_cv2_sample_when_window_secondary_restrains(tmp_path):
    """End-to-end production-observable effect of the whole Task 1 + Task 1b
    fix chain: a run whose (single) window has a REAL secondary restraint
    (k2 > 0), fed a null-cv2 sample among otherwise-finite ones, must come
    back from `load_parquet` with exactly that one sample excluded -- not
    silently kept with a fabricated cv2=0.0 bias.
    """
    from gareus.mbar_analysis.loaders import load_parquet
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    prod = tmp_path
    windows = [{"window_id": 0, "center1": 0.0, "k1": 10.0, "center2": 0.0, "k2": 50.0}]
    reg = SegmentRegistry(prod)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(prod).snapshot(seg_id, windows, cv1_type="contacts", cv2_type="torsion-pca")

    writer = ParquetSampleWriter(prod / "samples" / seg_id, flush_rows=1000)
    writer.write_sample(0, 0, 0, 0.0, 1.0, -100.0, 1.0, 0.6, 0.4)     # finite cv2
    writer.write_sample(50, 0, 0, 0.0, None, -100.0, 1.0, 0.6, 0.4)   # real SQL NULL cv2
    writer.write_sample(100, 0, 0, 0.0, 3.0, -100.0, 1.0, 0.6, 0.4)   # finite cv2
    writer.close()
    reg.close_segment(seg_id, end_step=100)

    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))

    d = load_parquet(prod)

    assert d.cv.size == 2  # the null-cv2 sample was excluded, not kept as cv2=0.0
    np.testing.assert_allclose(sorted(d.cv2), [1.0, 3.0])
    assert np.all(np.isfinite(d.u_nk))


# --- load_parquet_adaptive_union: real null-containing multi-epoch layout --

def _write_epoch(epoch_dir: Path, samples: list, secondary_center: float, secondary_k: float,
                  primary_center: float = 0.0, primary_k: float = 44.3) -> None:
    """Write one epoch's Parquet samples + window snapshot + epoch_window_map.csv.

    `samples` is a list of (cv1, cv2) tuples; cv2 may be None for a real null.
    Reuses the same low-level (ParquetSampleWriter/SegmentRegistry/WindowSnapshot)
    building blocks as tests/test_union_mbar_per_epoch_bias.py's own `_write_epoch`,
    generalized to accept a per-sample cv2 (including None) instead of one
    uniform value for the whole epoch.
    """
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    epoch_dir.mkdir(parents=True, exist_ok=True)
    reg = SegmentRegistry(epoch_dir)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(epoch_dir).snapshot(
        seg_id, [{"window_id": 0, "center1": primary_center, "k1": primary_k,
                  "center2": secondary_center, "k2": secondary_k}],
        cv1_type="contacts", cv2_type="torsion-pca",
    )
    writer = ParquetSampleWriter(epoch_dir / "samples" / seg_id, flush_rows=1000)
    for i, (cv1_val, cv2_val) in enumerate(samples):
        writer.write_sample(i * 50, 0, 0, cv1_val, cv2_val, -100.0, 1.0, 0.6, 0.4)
    writer.close()
    reg.close_segment(seg_id, end_step=(len(samples) - 1) * 50)

    (epoch_dir / "epoch_window_map.csv").write_text(
        "epoch_window,state_id,primary_center,primary_k,secondary_center,secondary_k\n"
        f"0,0,{primary_center},{primary_k},{secondary_center},{secondary_k}\n",
        encoding="utf-8",
    )


def _write_registry(adaptive_dir: Path, rows: list) -> None:
    with (adaptive_dir / "state_registry.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "state_id", "active", "created_epoch", "retired_epoch", "parent_state_id",
            "primary_center", "primary_k", "secondary_center", "secondary_k",
            "usable_for_mbar", "burnin_steps",
        ])
        w.writeheader()
        for row in rows:
            w.writerow({
                "active": "True", "created_epoch": 0, "retired_epoch": "",
                "parent_state_id": "", "usable_for_mbar": "True", "burnin_steps": 0,
                **row,
            })


def test_load_parquet_adaptive_union_masks_null_cv2_as_nan_not_zero(tmp_path):
    pytest.importorskip("analyze_gareus_mbar")
    from analyze_gareus_mbar import _reconstruct_union_bias_block
    from gareus.query import load_samples
    from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union

    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir(parents=True)
    (tmp_path / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))

    # CV1-only state (secondary_k=0.0) so the null-cv2 samples survive
    # clean() -- see module docstring for why this is required to observe
    # Data.cv2 == NaN on the final returned object.
    epoch_dir = adaptive_dir / "epoch_000"
    _write_epoch(
        epoch_dir,
        samples=[(0.0, 1.0), (0.0, None), (0.0, None), (0.0, 7.0)],
        secondary_center=0.0, secondary_k=0.0,
    )
    _write_registry(adaptive_dir, [{
        "state_id": 0, "primary_center": 0.0, "primary_k": 44.3,
        "secondary_center": 0.0, "secondary_k": 0.0,
    }])

    data = load_parquet_adaptive_union(adaptive_dir)

    raw_samples = load_samples(epoch_dir)
    assert np.ma.isMaskedArray(raw_samples["cv2"])

    assert data.cv.size == 4
    null_mask = np.isnan(data.cv2)
    assert int(null_mask.sum()) == 2
    assert not np.any(data.cv2 == 0.0)
    np.testing.assert_allclose(sorted(data.cv2[~null_mask]), [1.0, 7.0])

    # Independently confirm the u_nk-level fix for a state that DOES
    # secondary-restrain, via the loader's own real code path
    # (load_samples -> _fill_masked_nan -> _reconstruct_union_bias_block).
    # As above, this combination cannot survive clean() on the final Data.
    beta = data.beta
    cv1 = raw_samples["cv1"].astype(np.float64)
    pc = np.array([0.0]); pk = np.array([44.3])
    sc = np.array([0.0]); sk = np.array([50.0])  # hypothetical real secondary restraint

    cv2_fixed = _fill_masked_nan(raw_samples["cv2"])
    null_rows = np.where(np.isnan(cv2_fixed))[0]
    assert null_rows.size == 2
    u_nk_fixed = _reconstruct_union_bias_block(cv1, cv2_fixed, beta, pc, pk, sc, sk)
    assert np.all(np.isnan(u_nk_fixed[null_rows, 0]))

    # As in the load_parquet test above: `.astype()` preserves the mask
    # itself, but the underlying (mask-blind) data buffer still holds the
    # arbitrary fill value, and _reconstruct_union_bias_block's own plain
    # (non-masked) `u` accumulator silently absorbs it as a real value the
    # first time a masked term is added to it.
    cv2_buggy = raw_samples["cv2"].astype(np.float64)  # the old, buggy line
    assert np.all(np.ma.getdata(cv2_buggy)[null_rows] == 0.0)
    u_nk_buggy = _reconstruct_union_bias_block(cv1, cv2_buggy, beta, pc, pk, sc, sk)
    assert np.all(np.isfinite(u_nk_buggy[null_rows, 0]))
