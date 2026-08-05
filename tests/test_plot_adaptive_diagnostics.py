"""Tests for plot_adaptive_diagnostics.py's phase discovery and grid math.

Regression coverage for three real bugs found analyzing an actual completed
run (RUNS/chignolin/chignolin_quicktest):

1. discover_phases looked for baseline/topup_* segments inside a literal
   ap_dir/"epoch_001" directory. The adaptive-production driver always
   writes scheduled final-phase segments under ap_dir/"final" instead,
   independent of how many numbered epochs actually ran (the epoch loop can
   stop after 1 epoch via its convergence gate regardless of --ap-epochs).
   Real run: epoch_001/ existed but held only a stray CSV; final/baseline,
   final/topup_001_12000, final/topup_002_31000 held all the real samples.
   The old code produced a single-phase figure whenever the loop stopped
   early -- which is common.

2. All figure functions loaded only path/"samples"/"seg_001", but a
   resumed segment (interrupted then continued from checkpoint) writes a
   *second* segment directory (seg_002, seg_003, ...) rather than
   appending to seg_001. Real run: epoch_000/samples/ had seg_001 (200,556
   rows, the pre-interruption segment) AND seg_002 (2,924,448 rows, the
   post-resume segment) -- the old code silently dropped 93% of epoch_000's
   real samples.

3. _sample_count_grid (extracted from fig_window_layout) built its lookup
   dicts keyed by the raw unrounded float, then looked up with round(x, 6).
   secondary_center is a continuous CV projection with far more than 6
   significant digits, so every lookup missed and the heatmap panel stayed
   all-zero.

No OpenMM, PeptideBuilder, or gamd-openmm imports -- pure pandas/pathlib.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from plot_adaptive_diagnostics import _sample_count_grid, _secondary_cv_label, discover_phases


def _make_samples_dir(base, seg_names_and_rows):
    """Create base/samples/<seg>/... with a trivial parquet file of N rows each."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    samples = base / "samples"
    for seg_name, n_rows in seg_names_and_rows:
        seg_dir = samples / seg_name
        seg_dir.mkdir(parents=True, exist_ok=True)
        table = pa.table({
            "cv1": np.linspace(0.0, 1.0, n_rows),
            "cv2": np.linspace(-1.0, 1.0, n_rows),
            "window_id": np.zeros(n_rows, dtype=np.int64),
        })
        pq.write_table(table, seg_dir / "chunk_000.parquet")


def test_discover_phases_finds_final_baseline_and_topup_not_epoch1(tmp_path):
    ap_dir = tmp_path / "adaptive_production"
    ap_dir.mkdir()

    ep0 = ap_dir / "epoch_000"
    ep0.mkdir()
    _make_samples_dir(ep0, [("seg_001", 10)])
    (ep0 / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    # epoch_001 exists but never ran for real -- just a stray planning CSV,
    # no samples/ dir. Must NOT be treated as a phase, and must NOT be
    # searched for baseline/topup children.
    ep1 = ap_dir / "epoch_001"
    ep1.mkdir()
    (ep1 / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    final_dir = ap_dir / "final"
    baseline = final_dir / "baseline"
    baseline.mkdir(parents=True)
    _make_samples_dir(baseline, [("seg_001", 5)])
    (baseline / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    topup = final_dir / "topup_001_00500"
    topup.mkdir()
    _make_samples_dir(topup, [("seg_001", 7)])
    (topup / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    phases = discover_phases(ap_dir)
    names = [p["name"] for p in phases]

    assert names == ["epoch_000", "final/baseline", "final/topup_001_00500"]
    assert all(not n.startswith("epoch_001") for n in names)


def test_discover_phases_generalizes_to_multiple_real_epochs(tmp_path):
    """If epoch_001 (or epoch_002, ...) DOES run for real (has a samples/
    dir), it must be picked up as its own phase -- discover_phases should
    not be hardcoded to only ever look at epoch_000."""
    ap_dir = tmp_path / "adaptive_production"
    ap_dir.mkdir()

    for name in ("epoch_000", "epoch_001", "epoch_002"):
        d = ap_dir / name
        d.mkdir()
        _make_samples_dir(d, [("seg_001", 3)])
        (d / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    phases = discover_phases(ap_dir)
    assert [p["name"] for p in phases] == ["epoch_000", "epoch_001", "epoch_002"]
    assert phases[1]["label"] == "Epoch 1"
    assert phases[2]["label"] == "Epoch 2"


def test_discover_phases_uses_whole_samples_dir_not_just_seg_001(tmp_path):
    """A resumed segment writes seg_002 alongside seg_001; the returned
    phase path must be the samples/ parent so callers load every segment,
    not just the first."""
    ap_dir = tmp_path / "adaptive_production"
    ap_dir.mkdir()
    ep0 = ap_dir / "epoch_000"
    ep0.mkdir()
    _make_samples_dir(ep0, [("seg_001", 10), ("seg_002", 90)])
    (ep0 / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    phases = discover_phases(ap_dir)
    assert len(phases) == 1

    import pyarrow.dataset as ds
    table = ds.dataset(str(phases[0]["path"] / "samples"), format="parquet").to_table(columns=["window_id"])
    assert table.num_rows == 100  # both segments, not just seg_001's 10


def test_discover_phases_finds_scheduled_layout_for_numbered_epoch(tmp_path):
    """A numbered epoch can itself use the scheduled baseline/topup_* layout
    (run_scheduled_adaptive_epoch's allocation_scheduler isn't only used for
    "final") -- it must not be silently dropped just because it lacks a flat
    epoch_NNN/samples/ dir. Real run (chignolin_5): epoch_001 used this
    layout and held 61% of the run's real samples; the old code jumped
    straight from "Epoch 0 (initial)" to "Final Baseline" as if epoch_001
    never happened.
    """
    ap_dir = tmp_path / "adaptive_production"
    ap_dir.mkdir()

    ep0 = ap_dir / "epoch_000"
    ep0.mkdir()
    _make_samples_dir(ep0, [("seg_001", 10)])
    (ep0 / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    ep1 = ap_dir / "epoch_001"
    ep1_baseline = ep1 / "baseline"
    ep1_baseline.mkdir(parents=True)
    _make_samples_dir(ep1_baseline, [("seg_001", 20)])
    (ep1_baseline / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    ep1_topup = ep1 / "topup_001_5000"
    ep1_topup.mkdir()
    _make_samples_dir(ep1_topup, [("seg_001", 30)])
    (ep1_topup / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    phases = discover_phases(ap_dir)
    names = [p["name"] for p in phases]

    assert names == ["epoch_000", "epoch_001/baseline", "epoch_001/topup_001_5000"]
    assert phases[1]["label"] == "Epoch 1\nBaseline"
    assert "Epoch 1" in phases[2]["label"]


def test_discover_phases_orders_topups_by_creation_time_not_step_suffix(tmp_path):
    """topup_NNN_MMMMM's MMMMM is that segment's own extra-step duration, not
    a cumulative/creation-order marker -- a quality-gate extension loop can
    re-invoke the same epoch/final with a shrinking remaining-gap value each
    round, so ascending-suffix order is the *reverse* of creation order.
    Real run (chignolin_5/epoch_001): topup_001_34794000 created first, then
    topup_001_25733000, then topup_001_18937000 created last.
    """
    import os
    import time

    ap_dir = tmp_path / "adaptive_production"
    ep1 = ap_dir / "epoch_001"
    baseline = ep1 / "baseline"
    baseline.mkdir(parents=True)
    _make_samples_dir(baseline, [("seg_001", 1)])
    (baseline / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")

    # Created in this order: large suffix first, small suffix last --
    # mirrors the real run's shrinking-remaining-gap pattern.
    now = time.time()
    topup_big = ep1 / "topup_001_34794000"
    topup_big.mkdir()
    _make_samples_dir(topup_big, [("seg_001", 1)])
    (topup_big / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")
    os.utime(topup_big, (now, now))

    topup_small = ep1 / "topup_001_18937000"
    topup_small.mkdir()
    _make_samples_dir(topup_small, [("seg_001", 1)])
    (topup_small / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")
    os.utime(topup_small, (now + 10, now + 10))

    phases = discover_phases(ap_dir)
    names = [p["name"] for p in phases]

    assert names == [
        "epoch_001/baseline",
        "epoch_001/topup_001_34794000",  # created first -- must come before...
        "epoch_001/topup_001_18937000",  # ...the smaller-suffix one created later
    ]


def test_sample_count_grid_matches_high_precision_secondary_centers():
    """Regression for the rounding-mismatch bug: secondary_center values
    with more than 6 significant decimal digits must still populate the
    grid, not silently produce all-zeros."""
    state_reg = pd.DataFrame({
        "state_id": [0, 1, 2, 3],
        "primary_center": [0.0, 0.0, 0.8, 0.8],
        "secondary_center": [
            -0.1689698535455341,
            -0.1579850842029839,
            -0.1709088536625533,
            -0.3115591970517727,
        ],
    })
    state_counts = {0: 100, 1: 200, 2: 300, 3: 400}

    p_vals, s_vals, grid = _sample_count_grid(state_reg, state_counts)

    assert len(p_vals) == 2
    assert len(s_vals) == 4
    assert grid.sum() == 1000  # every state landed in a cell; none dropped
    assert set(grid.flatten()) == {0.0, 100.0, 200.0, 300.0, 400.0}


def test_sample_count_grid_empty_when_state_reg_and_counts_disagree():
    """States with no counts recorded should show as zero, not crash."""
    state_reg = pd.DataFrame({
        "state_id": [0, 1],
        "primary_center": [0.0, 0.8],
        "secondary_center": [0.1, 0.2],
    })
    p_vals, s_vals, grid = _sample_count_grid(state_reg, {})
    assert grid.sum() == 0.0


def _write_manifest(path, secondary_cv):
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_manifest.json").write_text(json.dumps({
        "method_settings": {"secondary_cv": secondary_cv},
    }))


def test_secondary_cv_label_prefers_last_phase(tmp_path):
    """CV2 can switch mid-run (torsion-pca -> tica-linear after a tICA
    refit); the label should reflect the last phase, where most of the
    plotted production weight actually sits."""
    ep0 = tmp_path / "epoch_000"
    _write_manifest(ep0, "torsion-pca")
    final = tmp_path / "final" / "baseline"
    _write_manifest(final, "tica-linear")

    phases = [
        {"name": "epoch_000", "path": ep0},
        {"name": "final/baseline", "path": final},
    ]
    assert _secondary_cv_label(phases) == "tICA"


def test_secondary_cv_label_known_types_and_fallback(tmp_path):
    for raw, expected in [("torsion-pca", "torsion-PCA"), ("rama-map", "rama"),
                          ("contacts", "contacts"), ("some-new-cv-type", "some-new-cv-type")]:
        d = tmp_path / raw
        _write_manifest(d, raw)
        assert _secondary_cv_label([{"name": raw, "path": d}]) == expected


def test_secondary_cv_label_falls_back_to_generic_when_no_manifest(tmp_path):
    phases = [{"name": "epoch_000", "path": tmp_path / "epoch_000"}]
    assert _secondary_cv_label(phases) == "secondary"
