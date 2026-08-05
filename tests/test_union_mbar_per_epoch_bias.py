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


def test_blank_secondary_fields_become_nan_center_zero_k():
    rows = [{"epoch_window": "0", "state_id": "0", "primary_center": "0.0",
             "primary_k": "44.3", "secondary_center": "", "secondary_k": ""}]
    parsed = _parse_epoch_window_map_native_params(rows)
    assert math.isnan(parsed[0]["secondary_center"])
    assert parsed[0]["secondary_k"] == 0.0


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
