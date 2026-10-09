# tests/test_aux_final_fix_wave.py
"""Stage C final-review fix wave (I1-I4, M1-M7): every change is auxiliary-gated."""
import types

import numpy as np
import pytest

from gareus.correctness._io import IntegrityError


# I3: compare_runs never passes vacuously -------------------------------------------------------

def _patched(monkeypatch, *, ledger_ctl, ledger_res, samples_ctl, samples_res):
    monkeypatch.setattr("gareus.query.load_exchanges", lambda p: ledger_ctl if "ctl" in str(p) else ledger_res)
    monkeypatch.setattr("gareus.query.load_samples", lambda p: samples_ctl if "ctl" in str(p) else samples_res)


EV = {"step": np.array([100]), "attempt_seq": np.array([0]), "assignment_sha256_after": np.array(["a"])}
S = {"step": np.array([100, 200]), "replica": np.array([0, 0]), "window_id": np.array([0, 1]),
     "tor_000": np.array([0.1, 0.2]), "aux_z_00": np.array([0.5, 0.6])}


def test_compare_runs_empty_ledger_has_no_key_error_and_is_not_ok(monkeypatch, tmp_path):
    import gareus.auxiliary_cv.resume_check as rc
    _patched(monkeypatch, ledger_ctl={}, ledger_res={}, samples_ctl=S, samples_res=dict(S))
    out = rc.compare_runs(tmp_path / "ctl", tmp_path / "res")
    assert out["ok"] is False and "ledger" in out["reason"]


def test_compare_runs_both_sides_empty_is_not_ok_with_a_reason(monkeypatch, tmp_path):
    import gareus.auxiliary_cv.resume_check as rc
    _patched(monkeypatch, ledger_ctl={}, ledger_res={}, samples_ctl={}, samples_res={})
    out = rc.compare_runs(tmp_path / "ctl", tmp_path / "res")
    assert out["ok"] is False and "empty" in out["reason"]


@pytest.mark.parametrize("side", ["ctl", "res"])
def test_compare_runs_one_side_without_samples_is_not_ok(monkeypatch, tmp_path, side):
    import gareus.auxiliary_cv.resume_check as rc
    _patched(monkeypatch, ledger_ctl=EV, ledger_res=EV, samples_ctl={} if side == "ctl" else S,
             samples_res={} if side == "res" else S)
    out = rc.compare_runs(tmp_path / "ctl", tmp_path / "res")
    assert out["ok"] is False and "samples" in out["reason"]


def test_compare_runs_both_sides_present_and_equal_is_ok(monkeypatch, tmp_path):
    import gareus.auxiliary_cv.resume_check as rc
    _patched(monkeypatch, ledger_ctl=EV, ledger_res=EV, samples_ctl=S, samples_res=dict(S))
    out = rc.compare_runs(tmp_path / "ctl", tmp_path / "res")
    assert out["ok"] is True and "reason" not in out


# I4b: no finite comparison with an active state refuses explicitly -------------------------------

def test_runtime_parity_all_nan_with_an_active_state_refuses():
    from gareus.auxiliary_cv.runtime import AuxObservationError
    from gareus.auxiliary_cv.runtime_io import check_runtime_parity
    with pytest.raises(AuxObservationError, match="no finite"):
        check_runtime_parity(np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), beta=0.4, k_max_kcal=2.0,
                             centers=[0.5], tolerance=1e-6)


def test_runtime_parity_partial_nan_compares_the_finite_rows():
    from gareus.auxiliary_cv.runtime import AuxObservationError
    from gareus.auxiliary_cv.runtime_io import check_runtime_parity
    check_runtime_parity(np.array([np.nan, 1.0]), np.array([np.nan, 1.0]), beta=0.4, k_max_kcal=2.0,
                         centers=[0.5], tolerance=1e-6)
    with pytest.raises(AuxObservationError, match="parity"):
        check_runtime_parity(np.array([np.nan, 1.0]), np.array([np.nan, 1.1]), beta=0.4, k_max_kcal=2.0,
                             centers=[0.5], tolerance=1e-6)


# I4a: strict k_max lookup ------------------------------------------------------------------------

def test_parity_bound_strict_lookup():
    from gareus.auxiliary_cv.offline import parity_bound, parity_context
    active = parity_context([{"aux_k": 2.0, "aux_center": 0.3, "aux_model_sha256": "a" * 64},
                             {"aux_k": 0.0, "aux_center": 0.0, "aux_model_sha256": "a" * 64}], 0.4)
    assert parity_bound(active, "a" * 64) == (2.0, [0.3])
    with pytest.raises(IntegrityError, match="k_max"):
        parity_bound(active, "b" * 64)
    sham = parity_context([{"aux_k": 0.0, "aux_center": 0.0, "aux_model_sha256": "a" * 64}], 0.4)
    assert parity_bound(sham, "a" * 64) == (0.0, [])


# I4c: precision fallback is recorded and warned -----------------------------------------------

class _Platform:
    def __init__(self, name, value=None, fail=False):
        self._name, self._value, self._fail = name, value, fail

    def getName(self):
        return self._name

    def getPropertyValue(self, context, key):
        if self._fail:
            raise RuntimeError("no such property")
        return self._value


def test_precision_info_records_a_failed_read():
    from gareus.auxiliary_cv.sample_schema import runtime_info, runtime_precision, runtime_precision_info
    assert runtime_precision_info("Reference") == ("double", None)
    assert runtime_precision_info("CUDA", _Platform("CUDA", "Mixed"), object()) == ("mixed", None)
    prec, reason = runtime_precision_info("CUDA", _Platform("CUDA", fail=True), object())
    assert prec == "single" and "no such property" in reason
    assert runtime_precision("CUDA", _Platform("CUDA", fail=True), object()) == "single"
    assert runtime_info("CUDA", "single") == {"platform": "CUDA", "precision": "single"}
    rec = runtime_info("CUDA", "single", fallback_reason=reason)
    assert rec["precision_fallback"] is True and rec["precision_fallback_reason"] == reason


def test_precision_fallback_survives_the_payload_round_trip():
    from gareus.auxiliary_cv.sample_schema import (AuxSampleSchema, _basis_sha, payload_with_runtime,
                                                   runtime_from_payload, runtime_info)
    quads, labels = ((0, 1, 2, 3),), ("phi-A1",)
    schema = AuxSampleSchema(quads, labels, ("e" * 64,), _basis_sha(quads, labels))
    rec = runtime_info("CUDA", "single", fallback_reason="read failed")
    assert runtime_from_payload(payload_with_runtime(schema, rec)) == rec
    plain = runtime_info("CUDA", "mixed")
    assert payload_with_runtime(schema, plain)["runtime"] == {"platform": "CUDA", "precision": "mixed"}


def test_aux_io_runtime_fields_and_precision_fallback(capsys):
    from test_aux_runtime_io import _record_setup
    from gareus.auxiliary_cv.runtime_definition import aux_io_runtime
    ctx, runtime, _force, d, _schema, _models = _record_setup(2.0)
    atoms = sorted({a for q in d["quads"] for a in q})
    args = types.SimpleNamespace(pep_gamd_peptide_atoms=atoms, aux_phase_kind="production",
                                 aux_equilibrium_eligible=True, aux_cv_model="m.json")
    io = aux_io_runtime(runtime, state_definition={"x": 1}, topology=d["topology"], args=args,
                        platform=_Platform("Reference"), context=ctx)
    assert io.runtime == {"platform": "Reference", "precision": "double"}
    assert io.phase_kind == "production" and io.equilibrium_eligible is True
    assert io.state_definition == {"x": 1} and list(io.models) == [runtime.table.model.model_sha256]
    assert "WARNING" not in capsys.readouterr().out
    io = aux_io_runtime(runtime, state_definition={}, topology=d["topology"], args=args,
                        platform=_Platform("CUDA"), context=ctx,
                        precision_info=("single", "Precision property read failed"))
    assert io.runtime["precision"] == "single" and io.runtime["precision_fallback"] is True
    out = capsys.readouterr().out
    assert out.count("WARNING") == 1 and "Precision property read failed" in out


# I4d: offline re-exports the merge utility ------------------------------------------------------

def test_offline_reexports_merge_identical_hamiltonians():
    from gareus.auxiliary_cv import duplicates, offline
    assert offline.merge_identical_hamiltonians is duplicates.merge_identical_hamiltonians


# I4f: a bad aux value never leaves a ragged buffer ----------------------------------------------

def _row(step, replica=0):
    return dict(step=step, replica=replica, window_id=replica, cv1=0.1, cv2=None, potential=-1.0,
                boost_total=None, boost_dihedral=None, boost_nonbonded=None)


@pytest.mark.parametrize("kw", [dict(aux_z=[1.0], torsions=[0.0, "x", 0.0]),
                                dict(aux_z=["x"], torsions=[0.0, 0.0, 0.0])])
def test_aux_writer_bad_value_leaves_no_ragged_buffer(tmp_path, kw):
    import pyarrow.parquet as pq
    from test_aux_store_writers import RUNTIME, SCHEMA
    from gareus.store import ParquetSampleWriter
    w = ParquetSampleWriter(tmp_path, aux_schema=SCHEMA, aux_runtime=RUNTIME)
    with pytest.raises((ValueError, TypeError)):
        w.write_sample(**_row(10), **kw)
    assert {len(v) for v in w._buf.values()} <= {0}
    w.write_sample(**_row(20), aux_z=[1.0], torsions=[0.1, 0.2, 0.3])
    w.close()
    tbl = pq.read_table(tmp_path / "data.parquet")
    assert tbl.num_rows == 1 and tbl.column("step").to_pylist() == [20]


# M5 cleanups ------------------------------------------------------------------------------------

def test_merge_refuses_out_of_range_origins():
    from gareus.auxiliary_cv.duplicates import merge_identical_hamiltonians
    u = np.zeros((3, 2))
    with pytest.raises(IntegrityError, match="origin"):
        merge_identical_hamiltonians(u, [0, 1, 2], ["a", "b"])
    with pytest.raises(IntegrityError, match="origin"):
        merge_identical_hamiltonians(u, [0, -1, 1], ["a", "b"])


def test_parity_report_missing_segment_rows_is_integrity_error(tmp_path, monkeypatch):
    from test_aux_loaders import _run
    import gareus.query as q
    from gareus.auxiliary_cv.parity_report import parity_report
    _run(tmp_path, historical=False)
    monkeypatch.setattr(q, "load_samples", lambda *a, **k: {})
    with pytest.raises(IntegrityError, match="no sample rows"):
        parity_report(tmp_path)


def _state(aux_k, with_center):
    row = {"window_id": 0, "aux_k": aux_k, "aux_model_sha256": "a" * 64}
    if with_center:
        row["aux_center"] = 0.3
    return {"windows": [row]}


def test_checkpoint_active_row_without_centre_is_integrity_error():
    from gareus.auxiliary_cv.checkpoint import expected_aux_parameters
    with pytest.raises(IntegrityError, match="aux_center"):
        expected_aux_parameters(_state(2.0, False), 0)
    assert expected_aux_parameters(_state(0.0, False), 0) == (0.0, 0.0)


def test_aux_table_from_checkpoint_refuses_an_active_row_without_centre(monkeypatch):
    import gareus.auxiliary_cv.checkpoint as ck
    model = types.SimpleNamespace(model_sha256="a" * 64)
    state = {"aux_models": {"a" * 64: {}}, "windows": [
        {"window_id": 0, "aux_k": 2.0, "instance": {}}, {"window_id": 1, "aux_k": 0.0, "instance": {}}]}
    monkeypatch.setattr(ck, "canonical_state_definition", lambda s: s)
    with pytest.raises(IntegrityError, match="aux_center"):
        ck.aux_table_from_checkpoint({"aux": {"state_definition": state}}, model=model)


# I4e: aux-only flags need --aux-cv-model ----------------------------------------------------------

BASE = ["--seq", "GA", "--cv1", "contacts"]
OK = ["--aux-cv-model", "m.json", "--windows-2d-csv", "w.csv", "--run-mode", "cmd", "--exchange-mode", "gibbs-walk"]


def _parse(extra, tmp_path):
    from gareus.cli import parse_args
    return parse_args(BASE + ["--out", str(tmp_path / "o")] + extra)


@pytest.mark.parametrize("extra", [["--aux-equilibrium-eligible"], ["--aux-phase-kind", "production"],
                                   ["--aux-phase-kind", "exploration"]])
def test_aux_only_flags_without_model_are_refused(tmp_path, capsys, extra):
    with pytest.raises(SystemExit):
        _parse(extra, tmp_path)
    assert "--aux-cv-model" in capsys.readouterr().err


def test_default_valued_aux_flags_without_model_are_accepted(tmp_path):
    a = _parse(["--aux-phase-kind", "pilot"], tmp_path)
    assert a.aux_cv_model is None and a.aux_phase_kind == "pilot" and a.aux_equilibrium_eligible is False


def test_production_and_eligible_with_model_are_accepted(tmp_path):
    a = _parse(OK + ["--aux-phase-kind", "production", "--aux-equilibrium-eligible"], tmp_path)
    assert a.aux_phase_kind == "production" and a.aux_equilibrium_eligible is True
