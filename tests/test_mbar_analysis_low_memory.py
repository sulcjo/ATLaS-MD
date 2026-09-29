from pathlib import Path
import json
import sys

import pytest

import analyze_gareus_mbar
from gareus.mbar_analysis import loaders
import gareus.mbar_analysis.loaders_union_parquet as parquet_loaders

sys.path.insert(0, str(Path(__file__).parent))


def test_analyze_parser_exposes_low_memory_flag():
    args = analyze_gareus_mbar.parse_args(["run", "--low-memory"])
    assert args.low_memory is True


def test_low_memory_adaptive_prefers_union_npz(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    adaptive = run_dir / "adaptive_production"
    adaptive.mkdir(parents=True)
    (adaptive / "adaptive_union_mbar.npz").write_bytes(b"placeholder")
    sentinel = object()
    calls = []

    monkeypatch.setattr(loaders, "check_union_npz_window_map_provenance", lambda p: [])
    monkeypatch.setattr(loaders, "load_union_npz", lambda p: calls.append(p) or sentinel)

    out = loaders.load_data(run_dir, None, low_memory=True)

    assert out is sentinel
    assert calls == [adaptive]
    assert "low-memory adaptive_union_mbar.npz" in capsys.readouterr().out


def test_low_memory_skips_a_union_npz_older_than_the_samples(tmp_path, monkeypatch, capsys):
    import os

    run_dir = tmp_path / "run"
    adaptive = run_dir / "adaptive_production"
    (adaptive / "final_extension_001" / "samples" / "seg_001").mkdir(parents=True)
    npz = adaptive / "adaptive_union_mbar.npz"
    npz.write_bytes(b"placeholder")
    newer = adaptive / "final_extension_001" / "samples" / "seg_001" / "data.parquet"
    newer.write_bytes(b"x")
    os.utime(npz, (1_000_000, 1_000_000))
    (adaptive / "state_registry.csv").write_text("state_id\n0\n")
    sentinel = object()
    calls = []
    monkeypatch.setattr(loaders, "load_union_npz", lambda p: calls.append("npz"))
    monkeypatch.setattr(loaders, "load_parquet_adaptive_union",
                        lambda p, **kw: calls.append(("parquet", kw["low_memory"])) or sentinel)

    out = loaders.load_data(run_dir, None, low_memory=True)

    assert out is sentinel and calls == [("parquet", True)]
    assert "older than final_extension_001/samples/seg_001/data.parquet" in capsys.readouterr().out


def test_low_memory_union_npz_refuses_stale_provenance(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    adaptive = run_dir / "adaptive_production"
    adaptive.mkdir(parents=True)
    (adaptive / "adaptive_union_mbar.npz").write_bytes(b"placeholder")

    def refuse(_path):
        raise ValueError("stale window map")

    monkeypatch.setattr(loaders, "check_union_npz_window_map_provenance", refuse)

    with pytest.raises(ValueError, match="stale window map"):
        loaders.load_data(run_dir, None, low_memory=True)


def test_low_memory_parquet_does_not_start_parallel_executor(tmp_path, monkeypatch):
    from test_loader_masked_cv2_nan import _write_epoch, _write_registry

    adaptive = tmp_path / "adaptive_production"
    adaptive.mkdir()
    (adaptive.parent / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    _write_registry(adaptive, [{"state_id": "0", "primary_center": "0.0", "primary_k": "44.3",
                                "secondary_center": "0.0", "secondary_k": "0.0"}])
    _write_epoch(adaptive / "epoch_000", [(1.0, 0.0)], 0.0, 0.0)
    _write_epoch(adaptive / "epoch_001", [(2.0, 0.0)], 0.0, 0.0)

    class UnexpectedParallelExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("low-memory loader started parallel executor")

    monkeypatch.setattr(parquet_loaders, "ThreadPoolExecutor", UnexpectedParallelExecutor)

    data = parquet_loaders.load_parquet_adaptive_union(adaptive, low_memory=True)

    assert data.cv.tolist() == [1.0, 2.0]


def test_low_memory_block_spool_avoids_concatenate_peak(tmp_path):
    """Low-memory merge writes blocks into one disk-backed matrix."""
    blocks = [
        __import__('numpy').arange(6, dtype=float).reshape(2, 3),
        __import__('numpy').arange(9, 15, dtype=float).reshape(2, 3),
    ]
    matrix, path = parquet_loaders._spool_u_nk_blocks(blocks, tmp_path)
    try:
        assert isinstance(matrix, __import__('numpy').memmap)
        assert matrix.shape == (4, 3)
        __import__('numpy').testing.assert_array_equal(matrix, __import__('numpy').vstack(blocks))
    finally:
        path.unlink(missing_ok=True)
