# tests/test_aux_store_writers.py
import json

import numpy as np
import pytest

from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha, runtime_from_payload
from gareus.parquet_manifest import ParquetManifestError, load_manifest

#: Pinned on main dc30285 (tmp/verify_C/pins.py): legacy sample schema with dtypes.
LEGACY_SAMPLE_SCHEMA = ("step: uint64\nreplica: uint16\nwindow_id: uint16\ncv1: float\ncv2: float\n"
                        "potential: float\ngamd_boost_total: float\ngamd_boost_dihedral: float\n"
                        "gamd_boost_nonbonded: float\nv_pep_kj_mol: float\nv_dih_kj_mol: float\ngamd_lambda: float")
LEGACY_COLUMNS = [line.split(":")[0] for line in LEGACY_SAMPLE_SCHEMA.split("\n")]
QUADS = ((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5))
LABELS = ("phi-A1", "psi-A1", "phi-B2")
SCHEMA = AuxSampleSchema(QUADS, LABELS, ("e" * 64,), _basis_sha(QUADS, LABELS))
RUNTIME = {"platform": "Reference", "precision": "double"}


def _row(step, replica=0):
    return dict(step=step, replica=replica, window_id=replica, cv1=0.1, cv2=None, potential=-1.0,
                boost_total=None, boost_dihedral=None, boost_nonbonded=None)


def test_legacy_sample_schema_and_dtypes_pinned(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path)
    w.write_sample(**_row(10))
    w.close()
    tbl = pq.read_table(tmp_path / "data.parquet")
    assert tbl.schema.to_string(show_schema_metadata=False) == LEGACY_SAMPLE_SCHEMA
    assert tbl.schema.metadata is None or b"atlas_aux_samples" not in tbl.schema.metadata
    assert "payload_schema" not in load_manifest(tmp_path)


def test_legacy_writer_refuses_aux_values(tmp_path):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path)
    with pytest.raises(ValueError, match="aux"):
        w.write_sample(**_row(10), aux_z=[1.0], torsions=[0.0, 0.0, 0.0])


def test_aux_writer_requires_runtime(tmp_path):
    from gareus.store import ParquetSampleWriter
    with pytest.raises(ValueError, match="aux_runtime"):
        ParquetSampleWriter(tmp_path, aux_schema=SCHEMA)


def test_aux_columns_float64_metadata_runtime_survive_consolidation(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, flush_rows=1, aux_schema=SCHEMA, aux_runtime=RUNTIME)
    w.write_sample(**_row(10), aux_z=[0.123456789012345], torsions=[0.1, -3.1, np.nan])
    w.write_sample(**_row(20, 1), aux_z=[1.5], torsions=[0.2, 3.1, 1.0])
    w.close()
    tbl = pq.read_table(tmp_path / "data.parquet")
    assert tbl.column_names == LEGACY_COLUMNS + ["observation_phase", "tor_000", "tor_001", "tor_002", "aux_z_00"]
    assert str(tbl.schema.field("aux_z_00").type) == "double" and str(tbl.schema.field("tor_000").type) == "double"
    assert tbl.column("aux_z_00").to_pylist()[0] == 0.123456789012345
    assert np.isnan(tbl.column("tor_002").to_pylist()[0])
    assert set(tbl.column("observation_phase").to_pylist()) == {"pre_exchange"}
    meta = json.loads(tbl.schema.metadata[b"atlas_aux_samples"])
    assert AuxSampleSchema.from_payload(meta) == SCHEMA and runtime_from_payload(meta) == RUNTIME
    payload = load_manifest(tmp_path)["payload_schema"]
    assert AuxSampleSchema.from_payload(payload) == SCHEMA and runtime_from_payload(payload) == RUNTIME


@pytest.mark.parametrize("kw", [dict(aux_z=None, torsions=[0, 0, 0]), dict(aux_z=[1.0], torsions=None),
                                dict(aux_z=[1.0, 2.0], torsions=[0, 0, 0]), dict(aux_z=[1.0], torsions=[0, 0])])
def test_aux_writer_requires_complete_rows(tmp_path, kw):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, aux_schema=SCHEMA, aux_runtime=RUNTIME)
    with pytest.raises(ValueError, match="aux"):
        w.write_sample(**_row(10), **kw)


def test_aux_writer_refuses_populated_legacy_segment(tmp_path):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path)
    w.write_sample(**_row(10))
    w.flush()
    with pytest.raises(ParquetManifestError, match="payload schema"):
        ParquetSampleWriter(tmp_path, aux_schema=SCHEMA, aux_runtime=RUNTIME)


def test_legacy_writer_refuses_aux_segment(tmp_path):
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, aux_schema=SCHEMA, aux_runtime=RUNTIME)
    w.write_sample(**_row(10), aux_z=[1.0], torsions=[0.0, 0.0, 0.0])
    w.flush()
    with pytest.raises(ParquetManifestError, match="payload schema"):
        ParquetSampleWriter(tmp_path)


# ---------------------------------------------------------------- exchange event ledger
from gareus.auxiliary_cv.ledger import EVENT_KINDS, EXCHANGE_EVENT_SCHEMA, assignment_sha256

#: Pinned on main dc30285 (tmp/verify_C/pins.py).
LEGACY_EXCHANGE_SCHEMA = ("step: uint64\nreplica_i: uint16\nreplica_j: uint16\nwindow_i: uint16\n"
                          "window_j: uint16\ndelta_e: float\naccepted: bool")
LEGACY_EXCHANGE_COLUMNS = [line.split(":")[0] for line in LEGACY_EXCHANGE_SCHEMA.split("\n")]


def _event(step, seq, **kw):
    base = dict(step=step, attempt_seq=seq, selected_replica=0, replica_i=0, replica_j=1, window_i=0,
                window_j=1, kind="swap", delta_e_kj=-0.25, accepted=True, log_q_forward=np.log(0.4),
                log_q_reverse=np.log(0.3), p_accept=0.9, energy_version="state_bias_matrix_v3_aux",
                assignments_after=[1, 0, 2])
    base.update(kw)
    return base


def test_legacy_exchange_schema_pinned(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path)
    w.write_exchange(step=5, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=False)
    w.close()
    assert pq.read_table(tmp_path / "data.parquet").schema.to_string(show_schema_metadata=False) == LEGACY_EXCHANGE_SCHEMA
    assert "payload_schema" not in load_manifest(tmp_path)
    with pytest.raises(ValueError, match="event"):
        w.write_event(**_event(6, 0))


def test_event_rows_round_trip(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path, event_schema=EXCHANGE_EVENT_SCHEMA)
    w.write_event(**_event(100, 0))
    w.write_event(**_event(100, 1, kind="skip", selected_replica=2, replica_i=2, replica_j=2, window_i=2,
                           window_j=2, accepted=False, assignments_after=[1, 0, 2]))
    w.close()
    t = pq.read_table(tmp_path / "data.parquet")
    assert t.column_names == LEGACY_EXCHANGE_COLUMNS + [
        "attempt_seq", "selected_replica", "kind", "delta_e_kj", "log_q_forward", "log_q_reverse",
        "p_accept", "energy_version", "assignment_sha256_after"]
    assert t.column("attempt_seq").to_pylist() == [0, 1] and t.column("kind").to_pylist() == ["swap", "skip"]
    assert t.column("assignment_sha256_after").to_pylist()[0] == assignment_sha256([1, 0, 2])
    assert load_manifest(tmp_path)["payload_schema"] == {"schema": EXCHANGE_EVENT_SCHEMA}
    with pytest.raises(ValueError, match="write_event"):
        w.write_exchange(step=5, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=False)
    assert EVENT_KINDS == ("swap", "stay", "no_candidates", "skip")


@pytest.mark.parametrize("bad", [dict(kind="teleport"), dict(attempt_seq=-1),
                                 dict(kind="stay", accepted=True), dict(kind="skip", accepted=True),
                                 dict(energy_version=""), dict(delta_e_kj="x"), dict(assignments_after=["a"]),
                                 dict(replica_i=None)])
def test_event_validation(tmp_path, bad):
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path, event_schema=EXCHANGE_EVENT_SCHEMA)
    with pytest.raises((ValueError, TypeError)):
        w.write_event(**_event(1, 0, **bad))
    assert not any(w._buf.values())  # no ragged buffer after a refused row
    w.write_event(**_event(2, 1))
    w.close()


def test_event_consolidation_sorts_by_step_then_seq_and_schema_mismatch_refused(tmp_path):
    import pyarrow.parquet as pq
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path, flush_rows=1, event_schema=EXCHANGE_EVENT_SCHEMA)
    for step, seq in ((200, 1), (100, 1), (100, 0), (200, 0)):
        w.write_event(**_event(step, seq))
    w.close()
    t = pq.read_table(tmp_path / "data.parquet")
    assert list(zip(t.column("step").to_pylist(), t.column("attempt_seq").to_pylist())) == [
        (100, 0), (100, 1), (200, 0), (200, 1)]
    with pytest.raises(ParquetManifestError, match="payload schema"):
        ParquetExchangeWriter(tmp_path)
