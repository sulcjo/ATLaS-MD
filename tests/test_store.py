"""Tests for gareus.store — Parquet-based timeseries storage layer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_parquet_writer_creates_chunk_on_flush(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter

    writer = ParquetSampleWriter(tmp_path / "seg_001", flush_rows=5)
    for i in range(5):
        writer.write_sample(i * 100, 0, 0, 0.1 + i * 0.01, -1.0 + i * 0.1,
                            -100.0 - i, 1.0, 0.5, 0.5)
    writer.flush()
    writer.close()

    # close() consolidates: no chunks remain, data.parquet holds all rows
    assert not list((tmp_path / "seg_001").glob("chunk_*.parquet"))
    assert (tmp_path / "seg_001" / "data.parquet").exists()
    assert pq.read_table(tmp_path / "seg_001" / "data.parquet").num_rows == 5


def test_parquet_writer_correct_columns(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter

    writer = ParquetSampleWriter(tmp_path / "seg", flush_rows=100)
    writer.write_sample(1000, 0, 2, 0.15, -0.5, -98.3, 1.2, 0.6, 0.6)
    writer.flush()
    writer.close()

    table = pq.read_table(tmp_path / "seg")
    assert set(table.column_names) == {
        "step", "replica", "window_id", "cv1", "cv2",
        "potential", "gamd_boost_total", "gamd_boost_dihedral", "gamd_boost_nonbonded",
    }


def test_parquet_writer_roundtrip_values(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter

    writer = ParquetSampleWriter(tmp_path / "seg", flush_rows=10)
    writer.write_sample(500, 1, 3, 0.25, -0.75, -105.0, 2.1, 1.0, 1.1)
    writer.flush()
    writer.close()

    tbl = pq.read_table(tmp_path / "seg")
    row = {col: tbl[col][0].as_py() for col in tbl.column_names}
    assert row["step"] == 500
    assert row["replica"] == 1
    assert row["window_id"] == 3
    assert abs(row["cv1"] - 0.25) < 1e-5
    assert abs(row["cv2"] - (-0.75)) < 1e-5
    assert abs(row["potential"] - (-105.0)) < 1e-3
    assert abs(row["gamd_boost_total"] - 2.1) < 1e-5
    assert abs(row["gamd_boost_dihedral"] - 1.0) < 1e-5
    assert abs(row["gamd_boost_nonbonded"] - 1.1) < 1e-5


def test_parquet_writer_multiple_chunks(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter

    writer = ParquetSampleWriter(tmp_path / "seg", flush_rows=3)
    for i in range(9):
        writer.write_sample(i, 0, 0, 0.1, -1.0, -100.0, 1.0, 0.5, 0.5)
    writer.flush()
    writer.close()

    # close() consolidates all chunks → data.parquet
    assert not list((tmp_path / "seg").glob("chunk_*.parquet"))
    tbl = pq.read_table(tmp_path / "seg" / "data.parquet")
    assert tbl.num_rows == 9


def test_parquet_writer_auto_flushes_at_threshold(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter

    writer = ParquetSampleWriter(tmp_path / "seg", flush_rows=4)
    for i in range(4):
        writer.write_sample(i, 0, 0, 0.1, -1.0, -100.0, 1.0, 0.5, 0.5)
    # auto-flush triggered by 4th write; close() consolidates to data.parquet
    writer.close()

    assert not list((tmp_path / "seg").glob("chunk_*.parquet"))
    tbl = pq.read_table(tmp_path / "seg" / "data.parquet")
    assert tbl.num_rows == 4


def test_parquet_writer_null_cv2(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter

    writer = ParquetSampleWriter(tmp_path / "seg", flush_rows=10)
    writer.write_sample(0, 0, 0, 0.1, None, -100.0, 1.0, 0.5, 0.5)
    writer.flush()
    writer.close()

    tbl = pq.read_table(tmp_path / "seg")
    assert tbl["cv2"][0].as_py() is None


def test_parquet_writer_no_tmp_files_after_close(tmp_path):
    from gareus.store import ParquetSampleWriter

    writer = ParquetSampleWriter(tmp_path / "seg", flush_rows=10)
    writer.write_sample(0, 0, 0, 0.1, -1.0, -100.0, 1.0, 0.5, 0.5)
    writer.close()

    assert not list((tmp_path / "seg").glob("*.tmp"))


# --- SegmentRegistry ---

def test_segment_registry_open_creates_entry(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment(run_id="run_001", parent_id=None, round_id=1)

    assert seg_id == "seg_001"
    data = json.loads((tmp_path / "segments.json").read_text())
    assert len(data) == 1
    assert data[0]["segment_id"] == "seg_001"
    assert data[0]["parent_segment_id"] is None
    assert data[0]["status"] == "running"
    assert data[0]["round_id"] == 1


def test_segment_registry_close_updates_status(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment("run_001", None, 1)
    reg.close_segment(seg_id, end_step=500000)

    data = json.loads((tmp_path / "segments.json").read_text())
    assert data[0]["status"] == "complete"
    assert data[0]["end_step"] == 500000


def test_segment_registry_sequential_ids(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    id1 = reg.open_segment("run_001", None, 1)
    reg.close_segment(id1, 100000)
    id2 = reg.open_segment("run_001", id1, 1)

    assert id1 == "seg_001"
    assert id2 == "seg_002"
    data = json.loads((tmp_path / "segments.json").read_text())
    assert data[1]["parent_segment_id"] == "seg_001"


def test_segment_registry_get_latest(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    id1 = reg.open_segment("run_001", None, 1)
    reg.close_segment(id1, 100000)
    id2 = reg.open_segment("run_001", id1, 2)

    latest = reg.get_latest_segment()
    assert latest["segment_id"] == id2
    assert latest["status"] == "running"


def test_segment_registry_persists_across_instances(tmp_path):
    from gareus.store import SegmentRegistry

    reg1 = SegmentRegistry(tmp_path)
    id1 = reg1.open_segment("run_001", None, 1)
    reg1.close_segment(id1, 50000)

    reg2 = SegmentRegistry(tmp_path)
    id2 = reg2.open_segment("run_001", id1, 1)
    assert id2 == "seg_002"


def test_segment_registry_empty_returns_none(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    assert reg.get_latest_segment() is None


def test_segment_registry_all_segments(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    id1 = reg.open_segment("run_001", None, 1)
    reg.close_segment(id1, 100000)
    reg.open_segment("run_001", id1, 2)

    segs = reg.all_segments()
    assert len(segs) == 2
    assert segs[0]["segment_id"] == "seg_001"
    assert segs[1]["segment_id"] == "seg_002"


def test_segment_registry_seal_interrupted(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment("run_001", None, 1)
    reg.seal_segment(seg_id, absolute_end_step=50000, status="interrupted")

    data = json.loads((tmp_path / "segments.json").read_text())
    assert data[0]["status"] == "interrupted"
    assert data[0]["end_step"] == 50000


def test_segment_registry_seal_abandoned(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment("run_001", None, 1)
    reg.seal_segment(seg_id, absolute_end_step=-1, status="abandoned")

    data = json.loads((tmp_path / "segments.json").read_text())
    assert data[0]["status"] == "abandoned"
    assert data[0]["end_step"] == -1


def test_segment_registry_seal_does_not_override_complete(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment("run_001", None, 1)
    reg.close_segment(seg_id, end_step=200000)
    reg.seal_segment(seg_id, absolute_end_step=100000, status="interrupted")

    data = json.loads((tmp_path / "segments.json").read_text())
    assert data[0]["status"] == "complete"
    assert data[0]["end_step"] == 200000


def test_segment_registry_set_start_step(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment("run_001", None, 1)
    assert reg.get_segment(seg_id)["start_step"] is None
    reg.set_segment_start_step(seg_id, 75000)

    data = json.loads((tmp_path / "segments.json").read_text())
    assert data[0]["start_step"] == 75000


def test_segment_registry_get_segment(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    id1 = reg.open_segment("run_001", None, 1)
    reg.close_segment(id1, 100000)
    id2 = reg.open_segment("run_001", id1, 2)

    seg = reg.get_segment(id1)
    assert seg is not None
    assert seg["segment_id"] == id1
    assert seg["status"] == "complete"

    seg2 = reg.get_segment(id2)
    assert seg2["status"] == "running"

    assert reg.get_segment("nonexistent") is None


def test_segment_registry_seal_persists_across_instances(tmp_path):
    from gareus.store import SegmentRegistry

    reg1 = SegmentRegistry(tmp_path)
    seg_id = reg1.open_segment("run_001", None, 1)
    reg1.seal_segment(seg_id, 60000, "interrupted")

    reg2 = SegmentRegistry(tmp_path)
    seg = reg2.get_segment(seg_id)
    assert seg["status"] == "interrupted"
    assert seg["end_step"] == 60000


def test_segment_registry_close_uses_absolute_step(tmp_path):
    from gareus.store import SegmentRegistry

    reg = SegmentRegistry(tmp_path)
    seg_id = reg.open_segment("run_001", None, 1)
    reg.close_segment(seg_id, end_step=125000)

    data = json.loads((tmp_path / "segments.json").read_text())
    assert data[0]["end_step"] == 125000
    assert data[0]["status"] == "complete"


# --- WindowSnapshot ---

def test_window_snapshot_writes_json(tmp_path):
    from gareus.store import WindowSnapshot

    windows = [
        {"window_id": 0, "center1": 0.1, "center2": -1.0, "k1": 200.0, "k2": 50.0},
        {"window_id": 1, "center1": 0.2, "center2": -0.33, "k1": 200.0, "k2": 50.0},
    ]
    snap = WindowSnapshot(tmp_path)
    snap.snapshot("seg_001", windows, cv1_type="contacts", cv2_type="rama-map")

    out = json.loads((tmp_path / "windows" / "seg_001.json").read_text())
    assert out["segment_id"] == "seg_001"
    assert out["cv1_type"] == "contacts"
    assert out["cv2_type"] == "rama-map"
    assert len(out["windows"]) == 2
    assert out["windows"][0]["center1"] == pytest.approx(0.1)


def test_window_snapshot_1d_no_cv2(tmp_path):
    from gareus.store import WindowSnapshot

    windows = [{"window_id": 0, "center1": 5.0, "k1": 10.0}]
    snap = WindowSnapshot(tmp_path)
    snap.snapshot("seg_001", windows, cv1_type="distance", cv2_type=None)

    out = json.loads((tmp_path / "windows" / "seg_001.json").read_text())
    assert out["cv2_type"] is None
    assert out["windows"][0]["center1"] == pytest.approx(5.0)


def test_window_snapshot_multiple_segments(tmp_path):
    from gareus.store import WindowSnapshot

    snap = WindowSnapshot(tmp_path)
    snap.snapshot("seg_001", [{"window_id": 0, "center1": 0.1, "k1": 200.0}],
                  cv1_type="contacts", cv2_type=None)
    snap.snapshot("seg_002", [{"window_id": 0, "center1": 0.15, "k1": 200.0}],
                  cv1_type="contacts", cv2_type=None)

    assert (tmp_path / "windows" / "seg_001.json").exists()
    assert (tmp_path / "windows" / "seg_002.json").exists()


# --- parse_gamd_boost_components ---

def test_parse_boost_components_named_dihedral_nonbonded():
    from gareus.store import parse_gamd_boost_components

    comps = {"dihedral_boost_kj_mol": 1.5, "nonbonded_boost_kj_mol": 2.3}
    total, dihe, nonb = parse_gamd_boost_components(3.8, comps)
    assert abs(total - 3.8) < 1e-5
    assert abs(dihe - 1.5) < 1e-5
    assert abs(nonb - 2.3) < 1e-5


def test_parse_boost_components_torsion_variant():
    from gareus.store import parse_gamd_boost_components

    comps = {"torsion_boost": 0.8, "lj_boost": 1.2}
    total, dihe, nonb = parse_gamd_boost_components(2.0, comps)
    assert abs(dihe - 0.8) < 1e-5
    assert abs(nonb - 1.2) < 1e-5


def test_parse_boost_components_two_unnamed():
    from gareus.store import parse_gamd_boost_components

    comps = {"boost_0": 0.5, "boost_1": 1.0}
    total, dihe, nonb = parse_gamd_boost_components(1.5, comps)
    assert abs(total - 1.5) < 1e-5
    # Two unnamed: assigned in sorted-key order
    assert dihe is not None
    assert nonb is not None


def test_parse_boost_components_empty():
    from gareus.store import parse_gamd_boost_components

    total, dihe, nonb = parse_gamd_boost_components(None, {})
    assert total is None
    assert dihe is None
    assert nonb is None


def test_parse_boost_components_single_component():
    from gareus.store import parse_gamd_boost_components

    total, dihe, nonb = parse_gamd_boost_components(2.0, {"total_boost": 2.0})
    assert abs(total - 2.0) < 1e-5
    assert dihe is None
    assert nonb is None


def test_parquet_writer_null_boost_fields(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter

    writer = ParquetSampleWriter(tmp_path / "seg", flush_rows=10)
    writer.write_sample(0, 0, 0, 0.1, None, -100.0, None, None, None)
    writer.flush()
    writer.close()

    tbl = pq.read_table(tmp_path / "seg")
    row = {col: tbl[col][0].as_py() for col in tbl.column_names}
    assert row["gamd_boost_total"] is None
    assert row["gamd_boost_dihedral"] is None
    assert row["gamd_boost_nonbonded"] is None


# --- DistanceLogger.no_file_persistence ---

def test_distance_logger_no_file_persistence(tmp_path):
    import argparse
    from gareus.logger import DistanceLogger

    args = argparse.Namespace()
    logger = DistanceLogger(tmp_path, args, no_file_persistence=True)

    assert logger.csv_writer is None
    assert logger.csv_handle is None
    assert logger.jsonl_handle is None
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.jsonl"))
    logger.close()


def test_distance_logger_default_creates_files(tmp_path):
    import argparse
    from gareus.logger import DistanceLogger

    args = argparse.Namespace()
    logger = DistanceLogger(tmp_path, args)

    # Default mode="both" creates both files
    assert logger.csv_writer is not None
    assert logger.jsonl_handle is not None
    logger.close()
