import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gareus.mbar_analysis.loaders_union_parquet import (
    _is_usable_for_mbar, _merge_missing_usable_states, load_parquet_adaptive_union,
)


def test_is_usable_for_mbar_accepts_true_variants():
    assert _is_usable_for_mbar({"usable_for_mbar": "True"}) is True
    assert _is_usable_for_mbar({"usable_for_mbar": "1"}) is True
    assert _is_usable_for_mbar({"usable_for_mbar": "yes"}) is True
    assert _is_usable_for_mbar({"usable_for_mbar": "false"}) is False
    assert _is_usable_for_mbar({}) is False


def test_merge_missing_usable_states_adds_only_usable_missing_rows():
    primary = [{"state_id": "0"}]
    live = [
        {"state_id": "0", "usable_for_mbar": "True"},
        {"state_id": "1", "usable_for_mbar": "True"},
        {"state_id": "2", "usable_for_mbar": "False"},
    ]
    merged = _merge_missing_usable_states(primary, live)
    ids = {int(r["state_id"]) for r in merged}
    assert ids == {0, 1}


def _write_registry(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["state_id", "usable_for_mbar", "primary_center", "primary_k",
              "secondary_center", "secondary_k", "burnin_steps"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_epoch_window_map(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch_window", "state_id", "primary_center", "primary_k",
              "secondary_center", "secondary_k"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_load_parquet_adaptive_union_raises_without_registry(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_parquet_adaptive_union(tmp_path)


def test_load_parquet_adaptive_union_raises_without_epoch_dirs(tmp_path):
    import pytest
    _write_registry(tmp_path / "final_registry_used_for_mbar.csv", [
        {"state_id": "0", "usable_for_mbar": "True", "primary_center": "1.0",
         "primary_k": "10.0", "secondary_center": "", "secondary_k": "", "burnin_steps": "0"},
    ])
    with pytest.raises(FileNotFoundError):
        load_parquet_adaptive_union(tmp_path)


def _write_epoch_with_real_parquet(epoch_dir, samples, state_id=0, primary_center=0.0,
                                    primary_k=44.3, secondary_center=0.0, secondary_k=0.0):
    """Write one epoch's worth of real Parquet samples + window snapshot +
    epoch_window_map.csv, using the real gareus.store writers -- same fixture
    pattern already used by tests/test_perf_loading_and_solver_memory.py's
    _write_epoch and tests/test_union_mbar_per_epoch_bias.py's _write_epoch."""
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
    max_step = 0
    for step, replica, cv1, cv2 in samples:
        writer.write_sample(step, replica, 0, cv1, cv2, -100.0, 5.0, 2.0, 0.4)
        max_step = max(max_step, step)
    writer.close()
    reg.close_segment(seg_id, end_step=max_step)

    (epoch_dir / "epoch_window_map.csv").write_text(
        "epoch_window,state_id,primary_center,primary_k,secondary_center,secondary_k\n"
        f"0,{state_id},{primary_center},{primary_k},{secondary_center},{secondary_k}\n",
        encoding="utf-8",
    )


def test_load_parquet_adaptive_union_end_to_end_with_real_parquet_fixture(tmp_path):
    """Real invocation of the full success path -- exercises _load_epoch_task
    and both lazy reverse-imports of the four bias-reconstruction functions
    that stay behind in analyze_gareus_mbar.py for Plan A3. This is the test
    that would have caught a forgotten import (e.g. of _Arrays in loaders.py,
    or of the bias-math cluster here) at module-creation time rather than
    only surfacing later in Task 9's full-suite run."""
    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir(parents=True)
    samples = [(i * 50, i % 2, float(i) * 0.1, 0.0) for i in range(10)]
    _write_epoch_with_real_parquet(adaptive_dir / "epoch_000", samples, state_id=0, primary_center=0.0)
    _write_registry(adaptive_dir / "final_registry_used_for_mbar.csv", [
        {"state_id": "0", "usable_for_mbar": "True", "primary_center": "0.0",
         "primary_k": "44.3", "secondary_center": "", "secondary_k": "", "burnin_steps": "0"},
    ])
    (adaptive_dir.parent / "run_manifest.json").write_text(
        '{"resolved_args": {"temperature_k": 300.0}}', encoding="utf-8")

    d = load_parquet_adaptive_union(adaptive_dir)

    assert d.cv.size == 10
    assert d.u_nk.shape == (10, 1)
    assert np.all(np.isfinite(d.u_nk))
