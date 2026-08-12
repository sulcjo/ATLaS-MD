"""Regression tests for excluding epoch_000 from the main CV1 PMF and GaMD
boost diagnostics.

epoch_000 is systematically different from later epochs: the GaMD
shared-envelope recalibration (`_maybe_recalibrate_gamd_boost`,
`gareus/adaptive_production.py`) recalibrates the boost envelope from epoch
0's own real sampling and fires at most once, so epoch 0 runs under a
different GaMD envelope than every later epoch; the tICA CV2 auto-switch
also typically fires after epoch 0. Pooling epoch_000 into the main PMF/GaMD
report mixes samples generated under different conditions.

`_epoch_zero_split_masks` detects when a Data has real epoch_000 samples
alongside later-epoch samples. `run_pmf_and_gamd_boost_report` is the
extracted core (4 PMF methods + GaMD boost diagnostics + window/overlap
diagnostics) that `analyze()` now runs twice when a split exists: once for
epoch_000 alone (written to `epoch_000_separate/`), once for the rest (the
main/default report, at the normal paths) -- both reusing the same global
MBAR solve rather than re-solving.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import (
    Data,
    _epoch_number_for_run_dir,
    _epoch_zero_split_masks,
    _masked_data,
    parse_args,
    run_pmf_and_gamd_boost_report,
)


def _test_args():
    args = parse_args(["dummy_input", "--bins", "20", "--no-convergence"])
    args.no_poincare_map = True
    return args


def _data_with_epoch_source(epoch_src, n=None, cv=None, boost=None, run_dirs=None):
    n = n or len(epoch_src)
    cv = np.zeros(n) if cv is None else np.asarray(cv, dtype=np.float64)
    boost = np.zeros(n) if boost is None else np.asarray(boost, dtype=np.float64)
    meta = {"_epoch_source": list(epoch_src)}
    if run_dirs is not None:
        meta["adaptive_epoch_run_dirs"] = list(run_dirs)
    return Data(
        prod_dir=Path("."), out_dir=Path("."),
        cv=cv, cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=np.zeros(n, dtype=np.int32), replica=np.zeros(n, dtype=np.int32),
        step=np.arange(n, dtype=np.int64), u_nk=np.zeros((n, 1)),
        centers=np.zeros(1), k_kcal=np.zeros(1), beta=0.4, temp=300.0,
        boost_kj=boost, potential_kj=None, source="test",
        meta=meta,
    )


# --- _epoch_zero_split_masks -------------------------------------------------

def test_none_when_only_epoch_zero_present():
    d = _data_with_epoch_source([0, 0, 0])
    assert _epoch_zero_split_masks(d) is None


def test_none_when_no_epoch_source_metadata():
    d = _data_with_epoch_source([0, 1])
    d.meta = {}
    assert _epoch_zero_split_masks(d) is None


def test_splits_epoch_zero_from_everything_else():
    # Positions 0, 1, 2 map to literal epoch_000, epoch_001, epoch_002 --
    # positional order matches literal epoch number here, so the split
    # behaves the same as before the literal-epoch-number fix.
    d = _data_with_epoch_source(
        [0, 0, 1, 1, 2, 2],
        run_dirs=["/x/epoch_000", "/x/epoch_001", "/x/epoch_002"],
    )
    mask0, mask_rest = _epoch_zero_split_masks(d)
    np.testing.assert_array_equal(mask0, [True, True, False, False, False, False])
    np.testing.assert_array_equal(mask_rest, [False, False, True, True, True, True])


def test_none_when_metadata_size_mismatches_samples():
    d = _data_with_epoch_source([0, 1], run_dirs=["/x/epoch_000", "/x/epoch_001"])
    # deliberately wrong length vs len(d.cv) == 2
    d.meta = {
        "_epoch_source": [0, 1, 2],
        "adaptive_epoch_run_dirs": ["/x/epoch_000", "/x/epoch_001", "/x/epoch_002"],
    }
    assert _epoch_zero_split_masks(d) is None


def test_none_when_run_dirs_metadata_missing():
    """Even with valid _epoch_source, a missing adaptive_epoch_run_dirs means
    literal epoch numbers can't be resolved -- must not fall back to treating
    positions as literal epoch numbers.
    """
    d = _data_with_epoch_source([0, 0, 1, 1])
    assert _epoch_zero_split_masks(d) is None


# --- Fix A regression: literal epoch number, not load position ---------------

def test_none_when_position_zero_is_not_literal_epoch_zero():
    """--epoch 1 --epoch 2 loads epoch_001 as position 0 and epoch_002 as
    position 1. _epoch_source positions are [0, 0, 1, 1] (positional), but
    neither run_dir is literal epoch_000 -- there is no real epoch_000 data
    here at all, so the split must not fire.
    """
    d = _data_with_epoch_source(
        [0, 0, 1, 1],
        run_dirs=["/x/epoch_001/baseline", "/x/epoch_002/baseline"],
    )
    assert _epoch_zero_split_masks(d) is None


def test_splits_by_literal_epoch_number_when_position_zero_is_real_epoch_zero():
    """Original correct-detection case: position 0 maps to a literal
    epoch_000 run_dir, position 1+ to epoch_001/final -- split still fires
    and masks reflect literal epoch_000 membership.
    """
    d = _data_with_epoch_source(
        [0, 0, 1, 1, 1],
        run_dirs=["/x/epoch_000", "/x/epoch_001/baseline", "/x/final/baseline"],
    )
    mask0, mask_rest = _epoch_zero_split_masks(d)
    np.testing.assert_array_equal(mask0, [True, True, False, False, False])
    np.testing.assert_array_equal(mask_rest, [False, False, True, True, True])


def test_epoch_number_for_run_dir():
    assert _epoch_number_for_run_dir(Path("/x/epoch_003")) == 3
    assert _epoch_number_for_run_dir(Path("/x/epoch_003/baseline")) == 3
    assert _epoch_number_for_run_dir(Path("/x/epoch_003/topup_001_5000")) == 3
    assert _epoch_number_for_run_dir(Path("/x/final")) is None
    assert _epoch_number_for_run_dir(Path("/x/final/baseline")) is None


# --- _masked_data reuse (sanity: already covered by test_secondary_cv_regime_split.py) ---

def test_masked_data_slices_boost_and_cv_consistently():
    d = _data_with_epoch_source([0, 1, 0, 1], cv=[1.0, 2.0, 3.0, 4.0], boost=[10.0, 20.0, 30.0, 40.0])
    sub = _masked_data(d, np.array([True, False, True, False]))
    np.testing.assert_array_equal(sub.cv, [1.0, 3.0])
    np.testing.assert_array_equal(sub.boost_kj, [10.0, 30.0])


# --- run_pmf_and_gamd_boost_report (integration) ----------------------------

def _full_data(cv, boost, k=3, window=None):
    n = len(cv)
    window = np.zeros(n, dtype=np.int32) if window is None else np.asarray(window, dtype=np.int32)
    return Data(
        prod_dir=Path("."), out_dir=Path("."),
        cv=np.asarray(cv, dtype=np.float64), cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=window, replica=np.zeros(n, dtype=np.int32),
        step=np.arange(n, dtype=np.int64), u_nk=np.zeros((n, k)),
        centers=np.zeros(k), k_kcal=np.zeros(k), beta=0.4, temp=300.0,
        boost_kj=np.asarray(boost, dtype=np.float64), potential_kj=None, source="test",
        meta={},
    )


def test_report_writes_pmf_and_gamd_boost_files(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    cv = rng.normal(0, 1, n)
    boost = np.abs(rng.normal(2.0, 0.5, n)) * 4.184  # kJ/mol, variable -> boost_ok path
    d = _full_data(cv, boost)
    logw = np.zeros(n)
    bins = np.linspace(-3, 3, 21)

    info = run_pmf_and_gamd_boost_report(d, _test_args(), logw, bins, 0.6, tmp_path, [], None)

    assert info["boost_ok"]
    assert info["boost"]["available"]
    for fname in ("pmf_unbiased.csv", "pmf_all_methods.csv", "overlap_matrix.csv", "window_diagnostics.csv"):
        assert (tmp_path / fname).exists()
    assert (tmp_path / "pmf_unbiased.png").exists()


def test_window_diagnostics_samples_column_reflects_this_subset_not_global(tmp_path):
    """Regression: window_diagnostics.csv used to report the *global* MBAR
    n_k (all epochs pooled) even for a subset report. Its 'samples' column
    must describe the samples actually in this report.
    """
    rng = np.random.default_rng(1)
    n = 60
    cv = rng.normal(0, 1, n)
    boost = np.zeros(n)  # boost_ok False path -- simpler, still exercises the CSV writer
    window = np.array([0] * 40 + [1] * 20, dtype=np.int32)
    d = _full_data(cv, boost, k=2, window=window)
    logw = np.zeros(n)
    bins = np.linspace(-3, 3, 11)

    run_pmf_and_gamd_boost_report(d, _test_args(), logw, bins, 0.6, tmp_path, [], None)

    import csv
    with (tmp_path / "window_diagnostics.csv").open() as f:
        rows = {int(r["window"]): int(r["samples"]) for r in csv.DictReader(f)}
    assert rows == {0: 40, 1: 20}


def test_epoch0_and_main_reports_are_independent_and_not_blended(tmp_path):
    """The core scenario: epoch_000 (different boost regime, e.g. pre-GaMD-
    recalibration) must not be pooled into the main report, and vice versa.
    """
    rng = np.random.default_rng(2)
    n0, n_rest = 100, 100
    cv0 = rng.normal(-3.0, 0.3, n0)
    cv_rest = rng.normal(3.0, 0.3, n_rest)
    boost0 = np.abs(rng.normal(5.0, 0.2, n0)) * 4.184     # epoch_000: larger boost (pre-recalibration)
    boost_rest = np.abs(rng.normal(1.0, 0.2, n_rest)) * 4.184  # rest: recalibrated, smaller boost
    epoch_src = [0] * n0 + [1] * n_rest
    cv = np.concatenate([cv0, cv_rest])
    boost = np.concatenate([boost0, boost_rest])

    d = _full_data(cv, boost)
    d.meta = {
        "_epoch_source": epoch_src,
        "adaptive_epoch_run_dirs": ["/x/epoch_000", "/x/epoch_001/baseline"],
    }
    logw = np.zeros(n0 + n_rest)
    bins = np.linspace(-4, 4, 21)

    split = _epoch_zero_split_masks(d)
    assert split is not None
    mask0, mask_rest = split

    out_main = tmp_path / "main"; out_main.mkdir()
    out_epoch0 = tmp_path / "epoch_000_separate"; out_epoch0.mkdir()

    main_info = run_pmf_and_gamd_boost_report(_masked_data(d, mask_rest), _test_args(), logw[mask_rest], bins, 0.6, out_main, [], None)
    epoch0_info = run_pmf_and_gamd_boost_report(_masked_data(d, mask0), _test_args(), logw[mask0], bins, 0.6, out_epoch0, [], None)

    assert main_info["n_samples"] == n_rest
    assert epoch0_info["n_samples"] == n0
    # Boost means must reflect only their own regime, not a pooled average.
    assert epoch0_info["boost"]["mean_kcal_mol"] > main_info["boost"]["mean_kcal_mol"]
    # PMF minima land near each regime's own cv center, not blended.
    assert main_info["pmf_minimum_cv_A"] > 0
    assert epoch0_info["pmf_minimum_cv_A"] < 0
