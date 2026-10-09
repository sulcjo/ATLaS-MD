from pathlib import Path

import numpy as np
import pandas as pd
from aux_discovery_fixture import make_phase

from gareus.adaptive.aux_discovery.frames import build_frame_table, load_phase_samples, phase_epoch


def test_phase_epoch_labels():
    assert phase_epoch("epoch_002/baseline") == 2 and phase_epoch("epoch_000") == 0
    assert phase_epoch("final") is None and phase_epoch("final_extension_001") is None


def test_join_dedup_stride_and_lambda0(tmp_path: Path):
    make_phase(tmp_path, "epoch_000",
               {"replica_0.xtc": (0, 0, [300, 3300, 6300]),
                "replica_0_resume_from_3300.xtc": (0, 0, [3300, 6300, 9300]),
                "replica_1.xtc": (1, 1, [300, 3300, 6300])},
               {0: 0.0, 1: 0.2}, lambda s: s / 1e4)
    ft = build_frame_table(tmp_path, epochs=[0], registry_lambda={100: 0.0, 101: 0.2},
                           stride_steps=3000, max_frames=10 ** 6, seed=0)
    assert set(ft.state_id) == {100}
    assert sorted(ft.step.tolist()) == [300, 3300, 6300, 9300]
    assert np.allclose(ft.cv1, ft.step / 1e4)
    assert ft.tors.shape == (4, 36)
    assert set(ft.lineage) == {"epoch_000:0"} and ft.n == 4


def test_samples_dedup_keeps_last_segment(tmp_path: Path):
    ph = tmp_path / "p"
    (ph / "samples" / "seg_000").mkdir(parents=True)
    (ph / "samples" / "seg_001").mkdir()
    base = {"step": [300], "replica": [0], "window_id": [0], "cv2": [0.0], "gamd_lambda": [0.0]}
    pd.DataFrame({**base, "cv1": [1.0]}).to_parquet(ph / "samples" / "seg_000" / "data.parquet")
    pd.DataFrame({**base, "cv1": [2.0]}).to_parquet(ph / "samples" / "seg_001" / "data.parquet")
    df = load_phase_samples(ph)
    assert len(df) == 1 and float(df.cv1.iloc[0]) == 2.0


def test_budget_caps_total_frames(tmp_path: Path):
    make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, list(range(300, 300 + 3000 * 40, 3000)))},
               {0: 0.0}, lambda s: 0.0)
    ft = build_frame_table(tmp_path, epochs=[0], registry_lambda={100: 0.0}, stride_steps=3000,
                           max_frames=10, seed=0)
    assert ft.n == 10


def test_unwanted_epoch_and_run_dir_root(tmp_path: Path):
    make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 3300])}, {0: 0.0}, lambda s: 0.0)
    make_phase(tmp_path, "epoch_001", {"replica_0.xtc": (0, 0, [300, 3300])}, {0: 0.0}, lambda s: 0.0)
    ft = build_frame_table(tmp_path / "adaptive_production", epochs=[0], registry_lambda={100: 0.0},
                           stride_steps=3000, max_frames=100, seed=0)
    assert ft.n == 2 and set(ft.epoch) == {0}


def test_workers_pool_matches_serial(tmp_path: Path):
    for name in ("epoch_000", "epoch_001"):
        make_phase(tmp_path, name, {"replica_0.xtc": (0, 0, [300, 3300, 6300])}, {0: 0.0}, lambda s: s / 1e4)
    kw = dict(epochs=[0, 1], registry_lambda={100: 0.0}, stride_steps=3000, max_frames=100, seed=0)
    a = build_frame_table(tmp_path, **kw)
    b = build_frame_table(tmp_path, workers=2, **kw)
    assert a.n == b.n == 6 and np.array_equal(a.step, b.step) and np.allclose(a.tors, b.tors)


def test_stride_anchored_per_replica_across_resume_files(tmp_path: Path):
    make_phase(tmp_path, "epoch_000",
               {"replica_0.xtc": (0, 0, [300, 3300]),
                "replica_0_resume_from_4800.xtc": (0, 0, list(range(4800, 9301, 300)))},
               {0: 0.0}, lambda s: 0.0)
    ft = build_frame_table(tmp_path, epochs=[0], registry_lambda={100: 0.0}, stride_steps=3000,
                           max_frames=10 ** 6, seed=0)
    assert sorted(ft.step.tolist()) == [300, 3300, 6300, 9300]
    assert ft.sources[0]["n_dropped_unmapped"] == 0


def test_unmapped_state_counted(tmp_path: Path):
    make_phase(tmp_path, "epoch_000", {"replica_0.xtc": (0, 0, [300, 3300]),
                                       "replica_1.xtc": (1, 1, [300, 3300])},
               {0: 0.0, 1: 0.0}, lambda s: 0.0)
    ft = build_frame_table(tmp_path, epochs=[0], registry_lambda={100: 0.0}, stride_steps=3000,
                           max_frames=100, seed=0)
    assert ft.sources[0]["n_dropped_unmapped"] == 2 and set(ft.state_id) == {100}
