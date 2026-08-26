"""Regression tests for per-epoch-native bias reconstruction in the adaptive-
production union MBAR loader.

`load_parquet_adaptive_union` used to reconstruct every sample's reduced bias
energy (u_nk) using a single static snapshot of window parameters (the live
or final `state_registry.csv`). That snapshot reflects whatever a state's
`secondary_center`/`secondary_k` (and `primary_center`/`primary_k`) look like
*today* -- but those fields get overwritten in place whenever a state is
recentered (e.g. the tICA CV2 auto-switch in `_apply_tica_centers_to_registry`,
`gareus/adaptive_production.py`). Any earlier epoch's samples that were
collected under the *old* center then got their bias energy reconstructed
against the *new* center -- injecting spurious bias energy on the order of
0.5 * k * (old_center - new_center)^2 into every one of that epoch's samples,
which feeds straight into MBAR's f_k solve.

Confirmed on a real run (chignolin_5): state 0's epoch_000 samples were
collected under secondary_center=-2.1337, but the live registry (used for
every epoch's bias reconstruction) held +1.1632 after the tICA CV2 switch --
a ~65 kcal/mol systematic bias-energy error for every epoch_000 sample of
that state.

Each epoch directory's own `epoch_window_map.csv` already records the window
parameters that were actually in effect for that specific epoch (it's written
once per epoch and never touched again), so the fix is to prefer that
per-epoch snapshot over the global one when reconstructing that epoch's own
samples' bias, falling back to the global row only for states absent from a
given epoch's snapshot (states created in a later epoch).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import (
    _epoch_bias_param_vectors,
    _parse_epoch_window_map_native_params,
    _reconstruct_union_bias_block,
    load_parquet_adaptive_union,
)


# --- _parse_epoch_window_map_native_params ---------------------------------

def test_parses_full_params_keyed_by_state_id():
    rows = [
        {"epoch_window": "0", "state_id": "0", "primary_center": "0.0",
         "primary_k": "44.3", "secondary_center": "-2.1337", "secondary_k": "11.87"},
        {"epoch_window": "1", "state_id": "5", "primary_center": "0.2",
         "primary_k": "44.3", "secondary_center": "1.5", "secondary_k": "12.0"},
    ]
    parsed = _parse_epoch_window_map_native_params(rows)
    assert set(parsed) == {0, 5}
    assert parsed[0] == {
        "primary_center": 0.0, "primary_k": 44.3,
        "secondary_center": -2.1337, "secondary_k": 11.87,
    }
    assert parsed[5]["secondary_center"] == 1.5


def test_blank_secondary_fields_become_nan():
    """Both secondary fields blank together: both must come back NaN so
    `_epoch_bias_param_vectors`'s `math.isfinite(...)` check falls back to the
    global registry value for both, not just `secondary_center`.
    """
    rows = [{"epoch_window": "0", "state_id": "0", "primary_center": "0.0",
             "primary_k": "44.3", "secondary_center": "", "secondary_k": ""}]
    parsed = _parse_epoch_window_map_native_params(rows)
    assert math.isnan(parsed[0]["secondary_center"])
    assert math.isnan(parsed[0]["secondary_k"])


def test_blank_secondary_k_alone_falls_back_to_global_k_not_zero():
    """Distinct from the blank-both case above: secondary_center is a real,
    present value but secondary_k is blank/missing on its own. Before the fix,
    secondary_k's missing-value default was the finite 0.0 (unlike its sibling
    fields, which default to NaN) -- so a row like this would force-override
    the epoch's secondary_k to 0.0 via `_epoch_bias_param_vectors`'s
    `math.isfinite` check, even though only secondary_k (not
    secondary_center) was actually missing from this row. That zero then
    silently disables the whole secondary-restraint bias term for this
    state's samples in `_reconstruct_union_bias_block`. Fixed: secondary_k's
    missing-value default is now NaN too, so a missing k correctly falls back
    to the global registry's real k instead of forcing zero.
    """
    rows = [{"epoch_window": "0", "state_id": "0", "primary_center": "0.0",
             "primary_k": "44.3", "secondary_center": "-2.1337", "secondary_k": ""}]
    native = _parse_epoch_window_map_native_params(rows)
    assert native[0]["secondary_center"] == -2.1337
    assert math.isnan(native[0]["secondary_k"])

    global_pc = np.array([0.0]); global_pk = np.array([44.3])
    global_sc = np.array([1.1632]); global_sk = np.array([11.87])  # real global k

    pc, pk, sc, sk = _epoch_bias_param_vectors(native, [0], global_pc, global_pk, global_sc, global_sk)

    assert sc[0] == pytest.approx(-2.1337)  # native center still used
    assert sk[0] == pytest.approx(11.87)    # falls back to GLOBAL k, not forced to 0.0


def test_rows_without_state_id_are_skipped():
    rows = [{"epoch_window": "0"}, {"epoch_window": "1", "state_id": "2",
             "primary_center": "0.0", "primary_k": "1.0"}]
    parsed = _parse_epoch_window_map_native_params(rows)
    assert set(parsed) == {2}


# --- _epoch_bias_param_vectors ----------------------------------------------

def test_native_params_override_global_fallback():
    global_pc = np.array([0.0, 0.0])
    global_pk = np.array([44.3, 44.3])
    global_sc = np.array([1.1632, 2.0])   # stale/current registry values
    global_sk = np.array([11.87, 12.0])
    native = {0: {"primary_center": 0.0, "primary_k": 44.3,
                  "secondary_center": -2.1337, "secondary_k": 11.87}}

    pc, pk, sc, sk = _epoch_bias_param_vectors(native, [0, 1], global_pc, global_pk, global_sc, global_sk)

    assert sc[0] == pytest.approx(-2.1337)   # overridden: this epoch's real center
    assert sc[1] == pytest.approx(2.0)       # untouched: no native row for state 1


def test_missing_native_row_keeps_global_values_unchanged():
    global_pc = np.array([0.5]); global_pk = np.array([10.0])
    global_sc = np.array([3.0]); global_sk = np.array([5.0])

    pc, pk, sc, sk = _epoch_bias_param_vectors({}, [0], global_pc, global_pk, global_sc, global_sk)

    assert (pc[0], pk[0], sc[0], sk[0]) == (0.5, 10.0, 3.0, 5.0)


def test_does_not_mutate_global_arrays():
    global_pc = np.array([0.0]); global_pk = np.array([44.3])
    global_sc = np.array([1.1632]); global_sk = np.array([11.87])
    native = {0: {"primary_center": 0.0, "primary_k": 44.3,
                  "secondary_center": -2.1337, "secondary_k": 11.87}}

    _epoch_bias_param_vectors(native, [0], global_pc, global_pk, global_sc, global_sk)

    assert global_sc[0] == 1.1632  # caller's array must be untouched


# --- _reconstruct_union_bias_block ------------------------------------------

def test_bias_block_matches_manual_harmonic_formula():
    beta = 0.4  # 1/(kJ/mol), arbitrary for this arithmetic check
    cv = np.array([0.0, 1.0])
    cv2 = np.array([-2.0, -2.0])
    pc = np.array([0.0]); pk = np.array([10.0])
    sc = np.array([-2.1337]); sk = np.array([11.87])

    u = _reconstruct_union_bias_block(cv, cv2, beta, pc, pk, sc, sk)

    d1 = cv - pc[0]
    d2 = cv2 - sc[0]
    expected = beta * 4.184 * 0.5 * pk[0] * d1 * d1 + beta * 4.184 * 0.5 * sk[0] * d2 * d2
    np.testing.assert_allclose(u[:, 0], expected)


def test_bias_block_skips_secondary_term_when_k_is_zero():
    beta = 0.4
    cv = np.array([0.0]); cv2 = np.array([100.0])  # would blow up if secondary term applied
    pc = np.array([0.0]); pk = np.array([10.0])
    sc = np.array([0.0]); sk = np.array([0.0])

    u = _reconstruct_union_bias_block(cv, cv2, beta, pc, pk, sc, sk)

    assert u[0, 0] == pytest.approx(0.0)


# --- Integration: load_parquet_adaptive_union picks per-epoch centers ------

def _write_epoch(epoch_dir: Path, cv1_val: float, cv2_val: float, n_samples: int,
                  secondary_center: float, secondary_k: float,
                  primary_center: float = 0.0, primary_k: float = 44.3) -> None:
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
    for i in range(n_samples):
        writer.write_sample(i * 50, 0, 0, cv1_val, cv2_val, -100.0, 1.0, 0.6, 0.4)
    writer.close()
    reg.close_segment(seg_id, end_step=(n_samples - 1) * 50)

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


def test_union_uses_epoch_native_secondary_center_not_stale_global_registry(tmp_path):
    """Reproduces the chignolin_5 bug: epoch_000 sampled under one secondary
    center; the (post-tICA-switch) global registry snapshot holds a different
    one. The union bias for epoch_000's own samples must use epoch_000's own
    center, not the registry's current value.
    """
    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir(parents=True)

    old_center, new_center = -2.1337, 1.1632
    sec_k = 11.87

    _write_epoch(adaptive_dir / "epoch_000", cv1_val=0.0, cv2_val=old_center,
                 n_samples=5, secondary_center=old_center, secondary_k=sec_k)
    # The live registry reflects the *post-switch* (current) center -- this is
    # the stale snapshot the old code used unconditionally for every epoch.
    _write_registry(adaptive_dir, [{
        "state_id": 0, "primary_center": 0.0, "primary_k": 44.3,
        "secondary_center": new_center, "secondary_k": sec_k,
    }])

    data = load_parquet_adaptive_union(adaptive_dir)

    assert data.cv.size == 5
    # Sample cv2 sits exactly on its own epoch's native center -> the harmonic
    # bias term must be ~0, not 0.5*k*(old-new)^2 (~65 kcal/mol-scale energy).
    beta = data.beta
    bogus_energy = beta * 4.184 * 0.5 * sec_k * (old_center - new_center) ** 2
    assert bogus_energy > 10.0  # sanity: the stale-center error would be large
    np.testing.assert_allclose(data.u_nk[:, 0], 0.0, atol=1e-6)


def test_union_falls_back_to_global_row_for_state_absent_from_epoch_snapshot(tmp_path):
    """State 1 is created in a later epoch and has no row in epoch_000's own
    snapshot (it didn't exist yet). MBAR still needs a u_nk column for it
    (cross term against epoch_000's samples) -- for that (state, epoch) pair
    there is no native value to prefer, so the global registry row is the
    only option and must still be used.
    """
    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir(parents=True)

    # epoch_000's own snapshot only ever mentions state 0.
    _write_epoch(adaptive_dir / "epoch_000", cv1_val=0.3, cv2_val=0.0,
                 n_samples=3, secondary_center=0.0, secondary_k=0.0)
    # The global registry additionally knows about state 1 (born later).
    _write_registry(adaptive_dir, [
        {"state_id": 0, "primary_center": 0.0, "primary_k": 44.3,
         "secondary_center": 0.0, "secondary_k": 0.0},
        {"state_id": 1, "primary_center": 0.5, "primary_k": 20.0,
         "secondary_center": 0.0, "secondary_k": 0.0},
    ])

    data = load_parquet_adaptive_union(adaptive_dir)

    assert data.u_nk.shape[1] == 2
    d1 = data.cv - 0.5
    expected_col1 = data.beta * 4.184 * 0.5 * 20.0 * d1 * d1
    np.testing.assert_allclose(data.u_nk[:, 1], expected_col1)


# --- A dropped state's row must NOT lose its per-epoch native params --------
#
# `--us-auto-drop-bad-windows` prunes umbrella windows post-pull and
# `gareus/production.py` renumbers the survivors 0..N-1, but the phase's own
# `epoch_window_map.csv` is never rewritten, so the loader has to compact it in
# memory (`_validate_and_repair_epoch_window_map`,
# `gareus/mbar_analysis/loaders_adaptive.py`,
# docs/chignolin_6_low_ess_root_cause.md).
#
# The trap that repair walks into: `_load_epoch_task` derives BOTH the local
# window -> state lookup AND `_parse_epoch_window_map_native_params` from the
# same row list. Removing a dropped state's row therefore removes that state's
# epoch-native window params too -- and MBAR still evaluates a bias column for
# it (cross-terms against every other epoch's samples), so the column silently
# falls back to the final-registry row. That is the same stale-snapshot bug the
# per-epoch fix at the top of this file exists to prevent, re-created for
# exactly the states that were dropped.
#
# It is a live hazard specifically for the flat `epoch_NNN/epoch_window_map.csv`
# maps written by `registry.write_epoch_window_map`: those carry primary_k and
# secondary_k as well as both centres, and they really do differ from the
# registry in every row (chignolin_6/epoch_000: secondary_center -2.1337 in the
# map vs -0.9058 in the registry, 20/20 rows; chignolin_5/epoch_000 27/27). The
# sub-run baseline/topup maps only have the two centre columns, which is why
# nothing currently on disk *happens* to show the damage.

def test_dropped_state_keeps_its_epoch_native_params_after_map_repair(tmp_path):
    """State 1's window was dropped post-pull, so it vanishes from the local
    window lookup -- but its own epoch_000 centre AND k must still drive its
    u_nk column, not the (later-recentered) registry row.
    """
    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir(parents=True)

    # State 0 is the one window that really ran; the samples sit on its centres.
    _write_epoch(adaptive_dir / "epoch_000", cv1_val=0.0, cv2_val=0.0,
                 n_samples=6, secondary_center=0.0, secondary_k=5.0)
    epoch = adaptive_dir / "epoch_000"

    # ... but the map the registry wrote still lists both states, identity-mapped,
    # with the params each really had while epoch_000 ran.
    native_c2, native_k2 = -2.0, 8.0
    native_c1, native_k1 = 0.5, 20.0
    (epoch / "epoch_window_map.csv").write_text(
        "epoch_window,state_id,primary_center,primary_k,secondary_center,secondary_k\n"
        "0,0,0.0,44.3,0.0,5.0\n"
        f"1,1,{native_c1},{native_k1},{native_c2},{native_k2}\n",
        encoding="utf-8",
    )
    (epoch / "gareus_metadata.json").write_text(
        json.dumps({"window_metadata": {"dropped_post_pull_bad_windows": [1]}}),
        encoding="utf-8")

    # The live registry has since recentered/re-stiffened state 1 (the tICA CV2
    # switch overwrites secondary_center in place, and the adaptive k update can
    # move secondary_k) -- these are the values that must NOT be used.
    stale_c2, stale_k2 = 2.0, 30.0
    _write_registry(adaptive_dir, [
        {"state_id": 0, "primary_center": 0.0, "primary_k": 44.3,
         "secondary_center": 0.0, "secondary_k": 5.0},
        {"state_id": 1, "primary_center": native_c1, "primary_k": native_k1,
         "secondary_center": stale_c2, "secondary_k": stale_k2},
    ])

    data = load_parquet_adaptive_union(adaptive_dir)

    assert data.cv.size == 6
    # Nothing may be attributed to state 1: its window never ran.
    assert set(np.asarray(data.window, dtype=np.int64).tolist()) == {0}
    assert data.u_nk.shape[1] == 2

    d1 = data.cv - native_c1
    d2 = data.cv2 - native_c2
    expected = data.beta * 4.184 * 0.5 * (native_k1 * d1 * d1 + native_k2 * d2 * d2)
    stale_d2 = data.cv2 - stale_c2
    stale_expected = data.beta * 4.184 * 0.5 * (native_k1 * d1 * d1
                                                + stale_k2 * stale_d2 * stale_d2)
    # Sanity: the two candidates are far apart, so this test has teeth.
    assert abs(float(expected[0] - stale_expected[0])) > 10.0
    np.testing.assert_allclose(data.u_nk[:, 1], expected, rtol=1e-9, atol=1e-9)


# --- Burnin filter must slice boost_dih alongside every other per-sample array --

def test_burnin_filter_keeps_boost_dih_kj_aligned_with_cv(tmp_path):
    """Every per-sample array except boost_dih used to be sliced by the
    burnin `keep` mask, leaving boost_dih at its original, unfiltered length
    whenever any registry state has burnin_steps > 0. `clean()` treats that
    length mismatch against the (already burnin-filtered) cv array as
    "corrupt" and silently sets boost_dih_kj to None -- so the whole
    dihedral-only GaMD boost component vanished from any burnin-filtered
    adaptive-production run, with no warning printed anywhere.

    Reproduce with a state whose burnin_steps discards some but not all of
    its samples, and assert boost_dih_kj survives at the same, filtered
    length as cv/window rather than being dropped to None.
    """
    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir(parents=True)

    # _write_epoch's writer.write_sample(...) call passes gamd_boost_dihedral=0.6
    # for every sample (non-trivial, non-NaN) and steps 0, 50, 100, 150, 200.
    _write_epoch(adaptive_dir / "epoch_000", cv1_val=0.3, cv2_val=0.0,
                 n_samples=5, secondary_center=0.0, secondary_k=0.0)
    _write_registry(adaptive_dir, [{
        "state_id": 0, "primary_center": 0.0, "primary_k": 44.3,
        "secondary_center": 0.0, "secondary_k": 0.0, "burnin_steps": 100,
    }])

    data = load_parquet_adaptive_union(adaptive_dir)

    # burnin_steps=100 keeps only step >= 100 -> 3 of the 5 samples survive.
    assert data.cv.size == 3
    assert data.boost_dih_kj is not None
    assert data.boost_dih_kj.size == data.cv.size
    np.testing.assert_allclose(data.boost_dih_kj, 0.6)
