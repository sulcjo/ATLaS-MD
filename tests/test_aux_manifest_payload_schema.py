# tests/test_aux_manifest_payload_schema.py
import json

import pytest

from gareus.parquet_manifest import (ParquetManifestError, append_file_to_manifest, load_manifest,
                                     manifest_path, replace_files_in_manifest)


def _chunk(tmp_path, name="chunk_000001.parquet"):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p = tmp_path / name
    pq.write_table(pa.table({"step": pa.array([1, 2], type=pa.uint64())}), p)
    from gareus.parquet_manifest import file_record
    return file_record(p, rows=2, first_step=1, last_step=2)


def test_legacy_manifest_key_set_is_pinned(tmp_path):
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2)
    raw = json.loads(manifest_path(tmp_path).read_text())
    assert "payload_schema" not in raw
    assert set(raw) == {"schema", "kind", "generation", "n_rows", "next_chunk_index", "files"}
    assert set(raw["files"][0]) == {"first_step", "last_step", "path", "rows", "sha256", "size_bytes"}


def test_payload_schema_survives_append_and_replace(tmp_path):
    ps = {"schema": "atlas-aux-samples-v1", "basis_sha256": "a" * 64}
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2,
                            payload_schema=ps)
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path, "chunk_000002.parquet"),
                            next_chunk_index=3)
    assert load_manifest(tmp_path)["payload_schema"] == ps
    rec = _chunk(tmp_path, "data.parquet")
    replace_files_in_manifest(tmp_path, kind="samples", records=[rec], next_chunk_index=3)
    assert load_manifest(tmp_path)["payload_schema"] == ps


def test_payload_schema_change_is_refused(tmp_path):
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2,
                            payload_schema={"schema": "atlas-aux-samples-v1", "basis_sha256": "a" * 64})
    with pytest.raises(ParquetManifestError, match="payload schema changed"):
        append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path, "chunk_000002.parquet"),
                                next_chunk_index=3,
                                payload_schema={"schema": "atlas-aux-samples-v1", "basis_sha256": "b" * 64})


def test_payload_schema_added_to_populated_legacy_segment_is_refused(tmp_path):
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2)
    with pytest.raises(ParquetManifestError, match="payload schema changed"):
        append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path, "chunk_000002.parquet"),
                                next_chunk_index=3, payload_schema={"schema": "atlas-aux-samples-v1"})


@pytest.mark.parametrize("bad", [[], {"schema": ""}, {"no_schema": 1}])
def test_malformed_payload_schema_rejected_on_load(tmp_path, bad):
    append_file_to_manifest(tmp_path, kind="samples", record=_chunk(tmp_path), next_chunk_index=2)
    raw = json.loads(manifest_path(tmp_path).read_text())
    raw["payload_schema"] = bad
    manifest_path(tmp_path).write_text(json.dumps(raw))
    with pytest.raises(ParquetManifestError, match="payload_schema"):
        load_manifest(tmp_path)
