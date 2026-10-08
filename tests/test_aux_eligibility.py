import json

import pytest

from gareus.kernel_identity import (AUX_SAMPLES_PAYLOAD_SCHEMA, ELIGIBLE_AUX_UNPERSISTED, ELIGIBLE_NOT_APPLICABLE,
                                    ELIGIBLE_VERIFIED, EXCHANGE_ENERGY_VERSION_AUX, classify_segment_kernel,
                                    raw_sample_payload_schema)

SHA = "e" * 64
PAYLOAD = {"schema": "atlas-aux-samples-v1", "model_shas": [SHA]}


def _snap(*, cv2="none", frozen=True, sha=SHA):
    snap = {"segment_id": "seg_001", "cv1_type": "contacts", "cv2_type": cv2,
            "kernel_identity": {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX, "aux_model_sha256": sha}}
    if frozen:
        snap["state_definition"] = {"aux_models": {SHA: {"schema": "atlas-aux-cv-model-v1"}}}
    return snap


def _segment_with_payload(run, seg_id, payload):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gareus.parquet_manifest import append_file_to_manifest, file_record
    seg = run / "samples" / seg_id
    seg.mkdir(parents=True)
    p = seg / "chunk_000001.parquet"
    pq.write_table(pa.table({"step": pa.array([1], type=pa.uint64())}), p)
    append_file_to_manifest(seg, kind="samples", record=file_record(p, rows=1, first_step=1, last_step=1),
                            next_chunk_index=2, payload_schema=payload)


def _registry(run, *ids):
    (run / "segments.json").write_text(json.dumps([{"segment_id": s, "status": "complete"} for s in ids]))


def test_payload_constant_matches_the_sample_schema():
    from gareus.auxiliary_cv.sample_schema import AUX_SAMPLES_SCHEMA
    assert AUX_SAMPLES_PAYLOAD_SCHEMA == AUX_SAMPLES_SCHEMA


def test_unpersisted_without_payload():
    assert classify_segment_kernel(_snap())[0] == ELIGIBLE_AUX_UNPERSISTED
    assert classify_segment_kernel(_snap(), sample_payload_schema=None)[0] == ELIGIBLE_AUX_UNPERSISTED


def test_verified_with_payload_and_frozen_snapshot():
    assert classify_segment_kernel(_snap(), sample_payload_schema=PAYLOAD)[0] == ELIGIBLE_VERIFIED


@pytest.mark.parametrize("snap, payload", [
    (_snap(frozen=False), PAYLOAD),
    (_snap(sha="f" * 64), PAYLOAD),
    (_snap(), {"schema": "atlas-aux-samples-v1", "model_shas": ["f" * 64]}),
    (_snap(), {"schema": "something-else", "model_shas": [SHA]}),
])
def test_any_missing_binding_stays_unpersisted(snap, payload):
    assert classify_segment_kernel(snap, sample_payload_schema=payload)[0] == ELIGIBLE_AUX_UNPERSISTED


def test_legacy_snapshots_unchanged():
    assert classify_segment_kernel({"cv2_type": "none"})[0] == ELIGIBLE_NOT_APPLICABLE
    assert classify_segment_kernel({"cv2_type": "none"}, sample_payload_schema=PAYLOAD)[0] == ELIGIBLE_NOT_APPLICABLE


def test_segment_eligibility_reads_payload_for_aux_snapshots(tmp_path):
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001")
    (tmp_path / "windows").mkdir()
    (tmp_path / "windows" / "seg_001.json").write_text(json.dumps(_snap()))
    _segment_with_payload(tmp_path, "seg_001", PAYLOAD)
    assert segment_eligibility(tmp_path)["seg_001"]["eligibility"] == ELIGIBLE_VERIFIED


def test_legacy_eligibility_never_reads_the_samples_manifest(tmp_path, monkeypatch):
    """B5: a legacy segment's broken or missing manifest is not a new failure mode."""
    import gareus.kernel_identity as ki
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001")
    (tmp_path / "windows").mkdir()
    (tmp_path / "windows" / "seg_001.json").write_text(json.dumps({"segment_id": "seg_001", "cv2_type": None,
                                                                  "windows": []}))
    (tmp_path / "samples" / "seg_001").mkdir(parents=True)
    (tmp_path / "samples" / "seg_001" / "parquet_manifest.json").write_text("{ not json")

    def _boom(*a, **k):
        raise AssertionError("legacy eligibility read a samples manifest")
    monkeypatch.setattr(ki, "raw_sample_payload_schema", _boom)
    assert segment_eligibility(tmp_path)["seg_001"]["eligibility"] == ELIGIBLE_NOT_APPLICABLE


def test_snapshotless_segment_of_an_aux_run_is_unpersisted(tmp_path):
    """H2: a missing snapshot in an auxiliary run is never 'not_applicable'."""
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001", "seg_002")
    (tmp_path / "windows").mkdir()
    (tmp_path / "windows" / "seg_002.json").write_text(json.dumps(dict(_snap(), segment_id="seg_002")))
    out = segment_eligibility(tmp_path)
    assert out["seg_001"]["eligibility"] == ELIGIBLE_AUX_UNPERSISTED
    assert "without a window snapshot" in out["seg_001"]["reason"]


def test_snapshotless_segment_with_aux_payload_is_unpersisted(tmp_path):
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001")
    _segment_with_payload(tmp_path, "seg_001", PAYLOAD)
    assert raw_sample_payload_schema(tmp_path, "seg_001") == PAYLOAD
    assert segment_eligibility(tmp_path)["seg_001"]["eligibility"] == ELIGIBLE_AUX_UNPERSISTED


def test_snapshotless_segment_of_a_legacy_run_is_unchanged(tmp_path):
    from gareus.query import segment_eligibility
    _registry(tmp_path, "seg_001")
    assert segment_eligibility(tmp_path)["seg_001"]["eligibility"] == ELIGIBLE_NOT_APPLICABLE
    assert raw_sample_payload_schema(tmp_path, "seg_001") is None
