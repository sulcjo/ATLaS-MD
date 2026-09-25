import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_adaptive_segmented_diagnostics import N_WINDOWS, _write_parquet_epoch_run, _write_window_csv

from gareus.adaptive_production import build_union_state_mbar_inputs, registry_from_window_csv


def test_union_meta_records_equilibration_cut_and_inefficiency(tmp_path):
    registry = registry_from_window_csv(_write_window_csv(tmp_path / "w.csv"), epoch=0, source="t")
    adaptive = tmp_path / "adaptive"
    _write_parquet_epoch_run(adaptive / "final", n_windows=N_WINDOWS, rows_per_window=200)
    meta = build_union_state_mbar_inputs(adaptive, registry)
    counts = meta["subsample_counts_per_state"]
    assert set(counts) == {str(i) for i in range(N_WINDOWS)}
    for rec in counts.values():
        assert set(rec) >= {"raw", "t0", "kept", "g", "status"}
        assert rec["raw"] == 200 and 0 <= rec["t0"] < rec["raw"] and 0 < rec["kept"] <= rec["raw"] - rec["t0"]
        assert math.isfinite(rec["g"]) and rec["g"] >= 1.0
