import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gareus.mbar_analysis.loaders_adaptive import (
    _find_adaptive_epoch_dirs, _find_selfcontained_epoch_dirs,
    _find_adaptive_epoch_csv_sources, _has_epoch_csv_layout,
    _find_adaptive_final_run_dirs, _find_gareus_round_dirs,
    _vectorized_map_lookup, _vectorized_map_lookup_or_self, _vectorized_map_index,
)


def test_find_adaptive_epoch_dirs_finds_flat_layout(tmp_path):
    ep = tmp_path / "epoch_000"
    (ep / "samples").mkdir(parents=True)
    (ep / "segments.json").write_text("[]")
    (ep / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")
    result = _find_adaptive_epoch_dirs(tmp_path)
    assert len(result) == 1
    assert result[0][0] == ep


def test_find_adaptive_epoch_dirs_finds_scheduled_baseline_topup_layout(tmp_path):
    ep = tmp_path / "epoch_001"
    ep.mkdir()
    (ep / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")
    base = ep / "baseline"
    (base / "samples").mkdir(parents=True)
    (base / "segments.json").write_text("[]")
    result = _find_adaptive_epoch_dirs(tmp_path)
    assert len(result) == 1
    assert result[0][0] == base


def test_find_selfcontained_epoch_dirs_requires_windows_dir(tmp_path):
    ep = tmp_path / "epoch_000"
    (ep / "samples").mkdir(parents=True)
    (ep / "segments.json").write_text("[]")
    (ep / "windows").mkdir()
    assert _find_selfcontained_epoch_dirs(tmp_path) == [ep]


def test_find_adaptive_epoch_csv_sources_and_has_epoch_csv_layout(tmp_path):
    ep = tmp_path / "epoch_000"
    ep.mkdir()
    (ep / "samples.csv").write_text("cv_A\n1.0\n")
    assert _has_epoch_csv_layout(tmp_path) is True
    assert _find_adaptive_epoch_csv_sources(tmp_path) == [ep]


def test_find_adaptive_final_run_dirs_finds_final_and_topup(tmp_path):
    final = tmp_path / "final"
    (final / "baseline").mkdir(parents=True)
    (final / "baseline" / "samples.csv").write_text("cv_A\n1.0\n")
    (final / "topup_001_1000").mkdir()
    (final / "topup_001_1000" / "samples.csv").write_text("cv_A\n1.0\n")
    result = _find_adaptive_final_run_dirs(tmp_path)
    assert final / "baseline" in result
    assert final / "topup_001_1000" in result


def test_find_gareus_round_dirs_requires_windows_csv_and_chunks(tmp_path):
    rd = tmp_path / "adaptive_feedback_round_01"
    (rd / "analysis_chunks").mkdir(parents=True)
    (rd / "analysis_chunks" / "chunk_000.npz").write_bytes(b"")
    (rd / "umbrella_windows.csv").write_text("center_A,k_kcal_mol_A2\n1.0,10.0\n")
    assert _find_gareus_round_dirs(tmp_path) == [rd]


def test_vectorized_map_lookup_matches_naive_dict_get():
    arr = np.array([0, 1, 2, 5])
    mapping = {0: 10, 1: 11, 2: 12}
    out = _vectorized_map_lookup(arr, mapping, default=-1, dtype=np.int64)
    assert list(out) == [10, 11, 12, -1]


def test_vectorized_map_lookup_or_self_falls_back_to_identity():
    arr = np.array([0, 1, 7])
    mapping = {0: 100}
    out = _vectorized_map_lookup_or_self(arr, mapping, dtype=np.int64)
    assert list(out) == [100, 1, 7]


def test_vectorized_map_index_raises_keyerror_on_missing_key():
    import pytest
    arr = np.array([0, 1, 9])
    mapping = {0: 10, 1: 11}
    with pytest.raises(KeyError):
        _vectorized_map_index(arr, mapping, dtype=np.int64)
