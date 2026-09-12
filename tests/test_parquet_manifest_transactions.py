"""Regression tests for manifest-backed Parquet publication and compaction."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest


def _write_samples(segment: Path, n: int, flush_rows: int = 2):
    from gareus.store import ParquetSampleWriter

    writer = ParquetSampleWriter(segment, flush_rows=flush_rows)
    for i in range(n):
        writer.write_sample(
            step=i * 10,
            replica=0,
            window_id=0,
            cv1=0.1 + i,
            cv2=None,
            potential=-10.0 - i,
            boost_total=0.0,
            boost_dihedral=0.0,
            boost_nonbonded=0.0,
        )
    return writer


def test_query_ignores_unreferenced_duplicate_parquet(tmp_path):
    """Physical leftovers must not duplicate the logical dataset."""
    from gareus.parquet_manifest import load_manifest
    from gareus.query import load_samples

    segment = tmp_path / "samples" / "seg_001"
    writer = _write_samples(segment, 6, flush_rows=2)
    writer.close()  # multiple chunks -> one manifest-authoritative compact file

    manifest = load_manifest(segment, expected_kind="samples", verify_hashes=True)
    assert len(manifest["files"]) == 1
    committed = segment / manifest["files"][0]["path"]

    # Reproduce the historical failure topology: a second physical Parquet
    # representation remains beside the authoritative one. It is intentionally
    # absent from the manifest and therefore must be invisible to load_samples.
    shutil.copy2(committed, segment / "orphan_duplicate.parquet")

    samples = load_samples(tmp_path)
    assert len(samples["step"]) == 6
    assert list(samples["step"]) == [0, 10, 20, 30, 40, 50]


def test_cleanup_failure_after_manifest_switch_does_not_duplicate(tmp_path, monkeypatch):
    """Old chunks may survive cleanup; the published manifest still selects one copy."""
    from gareus.parquet_manifest import load_manifest
    from gareus.query import load_samples

    segment = tmp_path / "samples" / "seg_001"
    writer = _write_samples(segment, 6, flush_rows=2)

    real_unlink = Path.unlink

    def fail_old_chunk_cleanup(self, *args, **kwargs):
        if self.name.startswith("chunk_") and self.suffix == ".parquet":
            raise OSError("injected cleanup failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_old_chunk_cleanup)
    writer.close()

    manifest = load_manifest(segment, expected_kind="samples", verify_hashes=True)
    assert len(manifest["files"]) == 1
    assert manifest["files"][0]["path"].startswith("compact_")
    assert list(segment.glob("chunk_*.parquet")), "fault injection should leave old chunks"

    samples = load_samples(tmp_path)
    assert len(samples["step"]) == 6


def test_present_but_corrupt_manifest_never_falls_back_to_glob(tmp_path):
    from gareus.parquet_manifest import ParquetManifestError, manifest_path
    from gareus.query import load_samples

    segment = tmp_path / "samples" / "seg_001"
    writer = _write_samples(segment, 2, flush_rows=2)
    writer.flush()

    path = manifest_path(segment)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "missing.parquet"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    # A real chunk is still physically present, but a corrupt manifest is an
    # integrity error, not permission to glob it and continue.
    assert list(segment.glob("*.parquet"))
    with pytest.raises(ParquetManifestError, match="missing file"):
        load_samples(tmp_path)


def test_legacy_chunks_only_remain_readable(tmp_path):
    """Unambiguous pre-manifest data stays backwards compatible."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gareus.query import load_samples

    segment = tmp_path / "samples" / "seg_001"
    segment.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "step": pa.array([1, 2], type=pa.uint64()),
            "replica": pa.array([0, 0], type=pa.uint16()),
            "window_id": pa.array([0, 0], type=pa.uint16()),
            "cv1": pa.array([0.1, 0.2], type=pa.float32()),
        }),
        segment / "chunk_000001.parquet",
    )

    samples = load_samples(tmp_path)
    assert list(samples["step"]) == [1, 2]


def test_legacy_data_plus_chunks_fails_instead_of_double_reading(tmp_path):
    """Known historical duplicate topology must become visible as an error."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gareus.parquet_manifest import ParquetManifestError
    from gareus.query import load_samples

    segment = tmp_path / "samples" / "seg_001"
    segment.mkdir(parents=True)
    table = pa.table({
        "step": pa.array([1], type=pa.uint64()),
        "replica": pa.array([0], type=pa.uint16()),
        "window_id": pa.array([0], type=pa.uint16()),
        "cv1": pa.array([0.1], type=pa.float32()),
    })
    pq.write_table(table, segment / "data.parquet")
    pq.write_table(table, segment / "chunk_000001.parquet")

    with pytest.raises(ParquetManifestError, match="Refusing to double-read"):
        load_samples(tmp_path)


def test_writer_refuses_to_append_to_legacy_segment(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gareus.parquet_manifest import ParquetManifestError
    from gareus.store import ParquetSampleWriter

    segment = tmp_path / "samples" / "seg_001"
    segment.mkdir(parents=True)
    pq.write_table(pa.table({"step": [1]}), segment / "data.parquet")

    with pytest.raises(ParquetManifestError, match="refusing to append to legacy"):
        ParquetSampleWriter(segment)


def test_exchange_writer_uses_independent_exchange_manifest(tmp_path):
    from gareus.parquet_manifest import load_manifest
    from gareus.query import load_exchanges
    from gareus.store import ParquetExchangeWriter

    segment = tmp_path / "exchanges" / "seg_001"
    writer = ParquetExchangeWriter(segment, flush_rows=1)
    writer.write_exchange(10, 0, 1, 0, 1, -0.5, True)
    writer.write_exchange(20, 1, 2, 1, 2, 0.2, False)
    writer.close()

    manifest = load_manifest(segment, expected_kind="exchanges", verify_hashes=True)
    assert manifest["n_rows"] == 2
    events = load_exchanges(tmp_path)
    assert list(events["step"]) == [10, 20]
