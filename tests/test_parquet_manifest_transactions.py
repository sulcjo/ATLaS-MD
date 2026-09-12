from __future__ import annotations

import json
import shutil

import pytest


def _sample(writer, step):
    writer.write_sample(step, 0, 0, 0.1, -1.0, -100.0, 1.0, 0.5, 0.5)


def test_sample_flush_publishes_manifest_and_reader_ignores_orphan(tmp_path):
    from gareus.query import load_samples
    from gareus.store import ParquetSampleWriter

    segment = tmp_path / "samples" / "seg_001"
    writer = ParquetSampleWriter(segment, flush_rows=10)
    _sample(writer, 100)
    writer.flush()

    manifest = json.loads((segment / "parquet_manifest.json").read_text())
    assert manifest["files"][0]["path"] == "chunk_000001.parquet"
    shutil.copy2(segment / "chunk_000001.parquet", segment / "orphan.parquet")

    assert len(load_samples(tmp_path)["step"]) == 1


def test_writer_resume_uses_manifest_next_chunk_index(tmp_path):
    from gareus.store import ParquetSampleWriter

    segment = tmp_path / "samples" / "seg_001"
    first = ParquetSampleWriter(segment, flush_rows=10)
    _sample(first, 100)
    first.flush()

    resumed = ParquetSampleWriter(segment, flush_rows=10)
    _sample(resumed, 200)
    resumed.flush()

    assert (segment / "chunk_000001.parquet").exists()
    assert (segment / "chunk_000002.parquet").exists()


def test_corrupt_manifest_does_not_fallback_to_directory_glob(tmp_path):
    from gareus.query import load_samples
    from gareus.store import ParquetSampleWriter
    from gareus.parquet_manifest import ParquetManifestError

    segment = tmp_path / "samples" / "seg_001"
    writer = ParquetSampleWriter(segment, flush_rows=10)
    _sample(writer, 100)
    writer.flush()
    manifest_path = segment / "parquet_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["path"] = "missing.parquet"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ParquetManifestError):
        load_samples(tmp_path)


def test_consolidation_switches_manifest_before_removing_sources(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter

    segment = tmp_path / "samples" / "seg_001"
    writer = ParquetSampleWriter(segment, flush_rows=1)
    _sample(writer, 100)
    _sample(writer, 200)
    writer.close()

    manifest = json.loads((segment / "parquet_manifest.json").read_text())
    assert len(manifest["files"]) == 1
    compacted = segment / manifest["files"][0]["path"]
    assert compacted.exists()
    assert pq.read_table(compacted).num_rows == 2
    assert not list(segment.glob("chunk_*.parquet"))


def test_exchange_reader_ignores_unreferenced_compaction_copy(tmp_path):
    import shutil

    from gareus.query import load_exchanges
    from gareus.store import ParquetExchangeWriter

    segment = tmp_path / "exchanges" / "seg_001"
    writer = ParquetExchangeWriter(segment, flush_rows=10)
    writer.write_exchange(100, 0, 1, 0, 1, 2.0, True)
    writer.flush()
    shutil.copy2(segment / "chunk_000001.parquet", segment / "data.parquet")

    assert len(load_exchanges(tmp_path)["step"]) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "../outside.parquet"),
        ("sha256", "not-the-file-hash"),
    ],
)
def test_manifest_integrity_errors_are_explicit(tmp_path, field, value):
    from gareus.parquet_manifest import ParquetManifestError, load_manifest
    from gareus.store import ParquetSampleWriter

    segment = tmp_path / "samples" / "seg_001"
    writer = ParquetSampleWriter(segment, flush_rows=10)
    _sample(writer, 100)
    writer.flush()
    manifest_path = segment / "parquet_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0][field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ParquetManifestError):
        load_manifest(segment, expected_kind="samples", verify_hashes=True)


def test_manifest_rejects_duplicate_file_records(tmp_path):
    from gareus.parquet_manifest import ParquetManifestError, load_manifest
    from gareus.store import ParquetSampleWriter

    segment = tmp_path / "samples" / "seg_001"
    writer = ParquetSampleWriter(segment, flush_rows=10)
    _sample(writer, 100)
    writer.flush()
    manifest_path = segment / "parquet_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"].append(dict(manifest["files"][0]))
    manifest["n_rows"] *= 2
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ParquetManifestError):
        load_manifest(segment, expected_kind="samples")
