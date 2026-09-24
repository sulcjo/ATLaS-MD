"""The driver's union MBAR inputs use each segment's OWN window parameters.

`build_union_state_mbar_inputs` (the driver-side union solve that feeds the
lambda-rung overlap and add_rung proposals) used the registry's CURRENT centres
for every sample. A state re-centred after an epoch ran (e.g. the tICA CV2
auto-switch rewrites `secondary_center` in place) then had that epoch's own
samples reconstructed against the new centre -- the stale-snapshot error class
the analysis loader fixed on 2026-08-04 (`load_parquet_adaptive_union`, ~124 kT
of spurious bias per sample on chignolin_5). Each segment's
`epoch_window_map.csv` records the centres actually in effect; they win over the
registry for the states that segment covers, and the registry stays the
fallback for states the segment never had.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_adaptive_segmented_diagnostics import _write_parquet_epoch_run  # noqa: E402

from gareus.adaptive_production import build_union_state_mbar_inputs, registry_from_window_csv  # noqa: E402

K_PRIMARY = 10.0
K_SECONDARY = 5.0
# Centres the sampling actually used (the fixture's samples sit on these).
NATIVE_PRIMARY = [0.2, 0.5, 0.8]
NATIVE_SECONDARY = [-0.5, 0.0, 0.5]


def _registry(tmp_path: Path, primary, secondary):
    path = tmp_path / "windows.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["primary_cv_center", "primary_cv_k_kcal",
                                           "secondary_cv_center", "secondary_cv_k_kcal_mol"])
        w.writeheader()
        for p, s in zip(primary, secondary):
            w.writerow({"primary_cv_center": p, "primary_cv_k_kcal": K_PRIMARY,
                        "secondary_cv_center": s, "secondary_cv_k_kcal_mol": K_SECONDARY})
    return registry_from_window_csv(path, epoch=0, source="test")


def _write_map(run_dir: Path, rows):
    with (run_dir / "epoch_window_map.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch_window", "state_id", "primary_center", "secondary_center"])
        w.writeheader()
        w.writerows(rows)


def _self_bias_kcal(meta, state_id):
    """Umbrella bias (kcal/mol) of `state_id`'s own samples in `state_id`'s own column."""
    with np.load(meta["arrays_npz"], allow_pickle=False) as d:
        col = list(d["state_ids"]).index(state_id)
        own = d["sampled_state_ids"] == state_id
        return d["umbrella_bias_kcal_mol_nk"][own, col]


def test_a_recentred_state_is_biased_against_the_centre_its_epoch_actually_used(tmp_path):
    # Registry today: state 0 was moved from 0.2 to 0.9 (primary) and -0.5 to 1.5
    # (secondary) AFTER epoch_000 ran with the native centres.
    registry = _registry(tmp_path, [0.9, 0.5, 0.8], [1.5, 0.0, 0.5])
    adaptive = tmp_path / "adaptive"
    epoch = adaptive / "epoch_000"
    _write_parquet_epoch_run(epoch, n_windows=3, rows_per_window=40)
    _write_map(epoch, [{"epoch_window": w, "state_id": w, "primary_center": NATIVE_PRIMARY[w],
                        "secondary_center": NATIVE_SECONDARY[w]} for w in range(3)])

    meta = build_union_state_mbar_inputs(adaptive, registry)

    stale = 0.5 * K_PRIMARY * (0.2 - 0.9) ** 2 + 0.5 * K_SECONDARY * (-0.5 - 1.5) ** 2   # 12.45 kcal
    own = _self_bias_kcal(meta, 0)
    assert own.size > 0
    assert float(np.median(own)) < 0.05 * stale        # native centres: ~0, not ~12 kcal/mol


def test_a_state_missing_from_the_segment_map_falls_back_to_the_registry(tmp_path):
    registry = _registry(tmp_path, NATIVE_PRIMARY, NATIVE_SECONDARY)
    adaptive = tmp_path / "adaptive"
    epoch = adaptive / "epoch_000"
    _write_parquet_epoch_run(epoch, n_windows=3, rows_per_window=40)
    # The map records only states 0 and 1; state 2's column must use the registry.
    _write_map(epoch, [{"epoch_window": w, "state_id": w, "primary_center": NATIVE_PRIMARY[w],
                        "secondary_center": NATIVE_SECONDARY[w]} for w in range(2)])

    meta = build_union_state_mbar_inputs(adaptive, registry)
    with np.load(meta["arrays_npz"], allow_pickle=False) as d:
        cv, sec = d["cv_A"], d["secondary_cv"]
        expected = 0.5 * K_PRIMARY * (cv - NATIVE_PRIMARY[2]) ** 2 + 0.5 * K_SECONDARY * (sec - NATIVE_SECONDARY[2]) ** 2
        np.testing.assert_allclose(d["umbrella_bias_kcal_mol_nk"][:, 2], expected, rtol=1e-12)


def test_each_segment_uses_its_own_map_not_another_segments(tmp_path):
    # Two segments of one epoch recorded different centres for state 0; each
    # segment's samples must be reconstructed with its own.
    registry = _registry(tmp_path, [0.9, 0.5, 0.8], NATIVE_SECONDARY)
    adaptive = tmp_path / "adaptive"
    base = adaptive / "epoch_000" / "baseline"
    top = adaptive / "epoch_000" / "topup_001_100"
    _write_parquet_epoch_run(base, n_windows=3, rows_per_window=40)
    _write_parquet_epoch_run(top, n_windows=1, rows_per_window=40)       # one-state topup
    _write_map(base, [{"epoch_window": w, "state_id": w, "primary_center": NATIVE_PRIMARY[w],
                       "secondary_center": NATIVE_SECONDARY[w]} for w in range(3)])
    _write_map(top, [{"epoch_window": 0, "state_id": 0, "primary_center": 0.2,
                      "secondary_center": -0.5}])

    meta = build_union_state_mbar_inputs(adaptive, registry)
    own = _self_bias_kcal(meta, 0)
    assert own.size > 0 and float(np.max(own)) < 1.0      # both segments on their native 0.2
