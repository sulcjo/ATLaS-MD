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

Fix round 1: the `_concat_numpy_dicts` fix above is column-agnostic -- it
also started correctly preserving masks for `gamd_boost_total` and
`gamd_boost_dihedral`, which are real SQL NULLs for every sample of a
non-GaMD/plain-umbrella run (`gareus/production.py`'s
`extract_gamd_boost_kj` returns `None` for a non-GaMD integrator, `4611-4614`).
`load_parquet`/`load_parquet_adaptive_union` cast those columns with a bare
`.astype()` (not `_fill_masked_nan`), so post-fix they started handing back
a genuinely-masked `numpy.ma.MaskedArray` for `Data.boost_kj` on any
non-GaMD run -- a regression this task's own upstream fix introduced.
`analyze_gareus_mbar.py`'s `run_pmf_and_gamd_boost_report` then crashed with
`ValueError: output array is read-only` from `np.nanstd(d.boost_kj)`
(reached because `boost_stats`'s own `boost[np.isfinite(boost)]` masked-
boolean-indexing bug incorrectly reported `available=True` for an
all-masked `boost`, defeating the `bool(bs.get('available')) and
np.nanstd(...)` short-circuit that would otherwise skip it). Fixed by
routing `boost`/`boost_dih`/`potential` (all three, for uniformity, even
though only the first two are proven reachable-null at the current writer)
through `_fill_masked_nan` in both loaders -- `loaders.py`'s `load_parquet`
(named in the review) and `loaders_union_parquet.py`'s
`load_parquet_adaptive_union` (not named in the review, but subject to the
exact same regression via its own local `np.concatenate(all_boost)`, which
denatures the mask the same way `_concat_numpy_dicts` used to -- see the
report's Fix round 1 section for the direct verification of this). With
this fix, `Data.boost_kj`/`boost_dih_kj`/`potential_kj` can never reach
`boost_stats` (or anything else) as a live `MaskedArray` via any current
call path -- see the report for the reachability audit -- so `boost_stats`'s
own masked-indexing bug, while real, is dead code for defensive purposes and
was intentionally left alone rather than patched reflexively.

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
import warnings
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
    assert np.all(np.isnan(u_nk_buggy[null_rows, 0]))


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
    assert np.all(np.isnan(u_nk_buggy[null_rows, 0]))


# --- Fix round 1: gamd_boost_total/gamd_boost_dihedral regression ----------
#
# _concat_numpy_dicts is shared by every column, not just cv2 -- once it
# started correctly preserving masks, a non-GaMD/plain-umbrella run's
# gamd_boost_total/gamd_boost_dihedral (real SQL NULLs for every sample of
# such a run, gareus/production.py:4611-4614) started reaching Data.boost_kj
# as a live MaskedArray via a bare `.astype()` cast, crashing
# analyze_gareus_mbar.py's run_pmf_and_gamd_boost_report with
# `ValueError: output array is read-only` from np.nanstd(d.boost_kj).
# Fixed by routing boost/boost_dih/potential through the same
# _fill_masked_nan helper as cv2, in both load_parquet and
# load_parquet_adaptive_union (the latter has its own local
# np.concatenate(all_boost) subject to the identical mask-drop hazard).

def test_load_parquet_non_gamd_run_boost_is_plain_nan_not_masked(tmp_path):
    """A plain-umbrella (non-GaMD) run: gamd_boost_total/gamd_boost_dihedral
    are real SQL NULLs for every sample. Data.boost_kj/boost_dih_kj must be
    plain NaN-filled arrays (boost_dih_kj None, since ALL its values are
    null) -- never a live MaskedArray reaching mask-unaware downstream
    consumers (boost_stats, np.nanstd).
    """
    pytest.importorskip("analyze_gareus_mbar")
    from analyze_gareus_mbar import boost_stats
    from gareus.mbar_analysis.loaders import load_parquet
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    prod = tmp_path
    windows = [{"window_id": 0, "center1": 0.0, "k1": 10.0}]
    reg = SegmentRegistry(prod)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(prod).snapshot(seg_id, windows, cv1_type="contacts", cv2_type=None)

    writer = ParquetSampleWriter(prod / "samples" / seg_id, flush_rows=1000)
    for i in range(5):
        # boost_total/boost_dihedral/boost_nonbonded all None: a non-GaMD run.
        writer.write_sample(i * 50, 0, 0, 0.0, 1.0 + 0.1 * i, -100.0, None, None, None)
    writer.close()
    reg.close_segment(seg_id, end_step=200)

    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))

    d = load_parquet(prod)

    assert not np.ma.isMaskedArray(d.boost_kj)
    assert np.all(np.isnan(d.boost_kj))
    assert d.boost_dih_kj is None  # _boost_dih_arg guard: no finite values at all

    # The exact crash repro: must not raise, and boost_stats must correctly
    # report unavailable (so run_pmf_and_gamd_boost_report's own
    # `bool(bs.get('available')) and np.nanstd(...)` short-circuits and never
    # even reaches np.nanstd in the real code path).
    bs = boost_stats(d.boost_kj, d.beta)
    assert bs == {"available": False}
    # "Degrees of freedom <= 0" is numpy's expected, harmless advisory for
    # nanstd on an all-NaN slice -- suppressed here since it's not the thing
    # under test; the thing under test is that no ValueError is raised.
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        val = np.nanstd(d.boost_kj)  # must not raise ValueError: output array is read-only
    assert np.isnan(val)


def test_load_parquet_adaptive_union_non_gamd_run_boost_is_plain_nan_not_masked(tmp_path):
    """Same regression, via load_parquet_adaptive_union -- this loader has
    its own local np.concatenate(all_boost)/(all_boost_dih)/(all_potential),
    independent of gareus.query._concat_numpy_dicts, subject to the same
    mask-drop-on-concatenate hazard once the per-epoch cast preserves a real
    mask.
    """
    pytest.importorskip("analyze_gareus_mbar")
    from analyze_gareus_mbar import boost_stats
    from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir(parents=True)
    (tmp_path / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))

    epoch_dir = adaptive_dir / "epoch_000"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    reg = SegmentRegistry(epoch_dir)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(epoch_dir).snapshot(
        seg_id, [{"window_id": 0, "center1": 0.0, "k1": 44.3, "center2": 0.0, "k2": 0.0}],
        cv1_type="contacts", cv2_type=None,
    )
    writer = ParquetSampleWriter(epoch_dir / "samples" / seg_id, flush_rows=1000)
    for i in range(4):
        writer.write_sample(i * 50, 0, 0, 0.0, None, -100.0, None, None, None)
    writer.close()
    reg.close_segment(seg_id, end_step=150)
    (epoch_dir / "epoch_window_map.csv").write_text(
        "epoch_window,state_id,primary_center,primary_k,secondary_center,secondary_k\n"
        "0,0,0.0,44.3,0.0,0.0\n",
        encoding="utf-8",
    )
    _write_registry(adaptive_dir, [{
        "state_id": 0, "primary_center": 0.0, "primary_k": 44.3,
        "secondary_center": 0.0, "secondary_k": 0.0,
    }])

    data = load_parquet_adaptive_union(adaptive_dir)

    assert not np.ma.isMaskedArray(data.boost_kj)
    assert np.all(np.isnan(data.boost_kj))
    assert data.boost_dih_kj is None
    # potential is real (never null) at the current writer, but confirm it
    # was correctly routed through _fill_masked_nan (plain array) too.
    assert not np.ma.isMaskedArray(data.potential_kj)
    np.testing.assert_allclose(data.potential_kj, -100.0)

    bs = boost_stats(data.boost_kj, data.beta)
    assert bs == {"available": False}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        val = np.nanstd(data.boost_kj)
    assert np.isnan(val)


def test_load_parquet_partially_null_boost_dihedral_is_nan_not_zero(tmp_path):
    """A GaMD run with a components gap on one sample only: boost_dihedral
    null for one sample, real for the rest. loaders.py:456's
    `_boost_dih_arg = boost_dih if np.any(np.isfinite(boost_dih)) else None`
    guard must keep the array (not collapse to None, since most values are
    real) with NaN -- not 0.0 -- at the null row.
    """
    from gareus.mbar_analysis.loaders import load_parquet
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    prod = tmp_path
    windows = [{"window_id": 0, "center1": 0.0, "k1": 10.0}]
    reg = SegmentRegistry(prod)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(prod).snapshot(seg_id, windows, cv1_type="contacts", cv2_type=None)

    writer = ParquetSampleWriter(prod / "samples" / seg_id, flush_rows=1000)
    writer.write_sample(0,   0, 0, 0.0, 1.0, -100.0, 1.0, 0.6, 0.4)
    writer.write_sample(50,  0, 0, 0.0, 1.1, -100.0, 1.0, None, 0.4)  # null boost_dihedral only
    writer.write_sample(100, 0, 0, 0.0, 1.2, -100.0, 1.0, 0.7, 0.4)
    writer.close()
    reg.close_segment(seg_id, end_step=100)

    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))

    d = load_parquet(prod)

    assert d.boost_dih_kj is not None
    assert not np.ma.isMaskedArray(d.boost_dih_kj)
    assert np.isnan(d.boost_dih_kj[1])
    assert not np.any(d.boost_dih_kj == 0.0)
    np.testing.assert_allclose(sorted(d.boost_dih_kj[np.isfinite(d.boost_dih_kj)]), [0.6, 0.7])
