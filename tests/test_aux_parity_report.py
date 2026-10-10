# tests/test_aux_parity_report.py
import numpy as np
from test_aux_loaders import _run          # synthetic aux run with Reference/double runtime
from gareus.auxiliary_cv.parity_report import parity_report
from gareus.auxiliary_cv.sample_schema import PARITY_TOLERANCE


def test_parity_report_on_exact_data(tmp_path):
    _run(tmp_path, historical=False)
    rep = parity_report(tmp_path)
    (seg, row), = rep["segments"].items()
    assert row["precision"] == "double" and row["max_abs_dz"] <= PARITY_TOLERANCE["double"] and row["ok"] is True
    assert rep["ok"] is True


def test_double_precision_physical_disagreement_is_not_ok(tmp_path, monkeypatch):
    _run(tmp_path, historical=False)
    _patch_offline(monkeypatch, lambda z: z + 1e-3)
    (row,) = parity_report(tmp_path)["segments"].values()
    assert row["precision"] == "double" and row["max_abs_dz"] > PARITY_TOLERANCE["double"]
    assert row["ok"] is False


def _patch_offline(monkeypatch, fn):
    import gareus.auxiliary_cv.evaluate as ev
    real = ev.z_from_dihedrals
    monkeypatch.setattr(ev, "z_from_dihedrals", lambda th, m: fn(np.array(real(th, m), dtype=float)))


def test_all_nan_offline_z_is_not_ok(tmp_path, monkeypatch):
    _run(tmp_path, historical=False)
    _patch_offline(monkeypatch, lambda z: np.full_like(z, np.nan))
    rep = parity_report(tmp_path)
    (row,) = rep["segments"].values()
    assert row["n_compared"] == 0 and row["ok"] is False and rep["ok"] is False


def test_finite_vs_nan_mismatch_row_is_not_ok(tmp_path, monkeypatch):
    _run(tmp_path, historical=False)

    def one_nan(z):
        z[0] = np.nan
        return z
    _patch_offline(monkeypatch, one_nan)
    (row,) = parity_report(tmp_path)["segments"].values()
    assert row["n_finite_mismatch"] == 1 and row["n_compared"] > 0 and row["ok"] is False


def test_mixed_precision_offset_is_not_ok(tmp_path, monkeypatch):
    import test_aux_loaders as tl
    monkeypatch.setattr(tl, "RUNTIME", {"platform": "CUDA", "precision": "mixed"})
    _run(tmp_path, historical=False)
    _patch_offline(monkeypatch, lambda z: z + 1e-3)
    (row,) = parity_report(tmp_path)["segments"].values()
    assert row["precision"] == "mixed" and row["k_max_kcal"] > 0
    assert row["max_reduced"] > row["tolerance"] and row["ok"] is False


def test_no_aux_segments_is_not_ok(tmp_path):
    rep = parity_report(tmp_path)
    assert rep["ok"] is False and rep["reason"] == "no aux segments"


def test_missing_precision_raises(tmp_path, monkeypatch):
    import pytest
    from gareus.correctness._io import IntegrityError
    import gareus.auxiliary_cv.offline as off
    _run(tmp_path, historical=False)
    real = off.segment_aux_runtime
    monkeypatch.setattr(off, "segment_aux_runtime",
                        lambda d: {k: {"platform": "CUDA"} for k in real(d)})
    with pytest.raises(IntegrityError, match="precision"):
        parity_report(tmp_path)


def test_rows_record_the_runtime_z_source(tmp_path, monkeypatch):
    """F07: each row names the stored z's evaluator (from the samples' runtime block) and the reference."""
    import gareus.auxiliary_cv.offline as off
    _run(tmp_path, historical=False)
    rep = parity_report(tmp_path)
    (legacy,) = rep["segments"].values()
    assert legacy["runtime_z_source"] == "unrecorded" and legacy["reference_z_source"] == "positions"
    assert legacy["parity_kind"] == "stored_vs_offline_data_consistency" and legacy["force_parity"] is False
    assert rep["force_parity"] == "none" and rep["n_segments_without_force_parity"] == 1
    assert legacy["ok"] is True and rep["ok"] is True          # top-level ok semantics unchanged
    real = off.segment_aux_runtime
    monkeypatch.setattr(off, "segment_aux_runtime",
                        lambda d: {k: {**v, "aux_z_source": "force", "aux_z_reference": "positions"}
                                   for k, v in real(d).items()})
    rep = parity_report(tmp_path)
    (row,) = rep["segments"].values()
    assert row["runtime_z_source"] == "force" and row["reference_z_source"] == "positions" and row["ok"] is True
    assert row["parity_kind"] == "force_vs_positions" and "force_parity" not in row
    assert rep["force_parity"] == "all" and rep["n_segments_without_force_parity"] == 0


def test_force_parity_summary_is_partial_with_mixed_segments(tmp_path, monkeypatch):
    """Two aux segments, only the second records a force-side z: summary "partial", top-level ok unchanged."""
    import test_aux_loaders as tl
    import gareus.auxiliary_cv.offline as off
    from gareus.store import SegmentRegistry, WindowSnapshot
    (s1,) = _run(tmp_path, historical=False)
    reg = SegmentRegistry(tmp_path)
    s2 = reg.open_segment("run", s1, 1)
    tl._write_segment(tmp_path, s2, True, np.random.default_rng(2))
    reg.close_segment(s2, 1800)
    WindowSnapshot(tmp_path).snapshot(s2, [], "contacts", None, kernel_identity=tl.KI, state_definition=tl._defn(),
                                      equilibrium_analysis_eligible=True, phase_kind="production")
    real = off.segment_aux_runtime
    monkeypatch.setattr(off, "segment_aux_runtime",
                        lambda d: {k: ({**v, "aux_z_source": "force", "aux_z_reference": "positions"} if k == s2
                                       else v) for k, v in real(d).items()})
    rep = parity_report(tmp_path)
    assert set(rep["segments"]) == {s1, s2}
    assert rep["segments"][s1]["parity_kind"] == "stored_vs_offline_data_consistency"
    assert rep["segments"][s2]["parity_kind"] == "force_vs_positions"
    assert rep["force_parity"] == "partial" and rep["n_segments_without_force_parity"] == 1 and rep["ok"] is True


def test_sham_only_degenerate_frame_is_not_ok(tmp_path, monkeypatch):
    """Fail closed: stored force z finite, offline torsion z NaN, every state a sham (vacuous bound) -> not ok."""
    import test_aux_loaders as tl
    from aux_c_fixture import definition, rows
    monkeypatch.setattr(tl, "_defn", lambda: definition(rows([(0.2, 10.0, 0.0, 0.0, "ordinary"),
                                                               (0.2, 10.0, 0.0, 0.0, "ordinary")],
                                                              tl.MODEL.model_sha256), tl.MODEL))
    _run(tmp_path, historical=False)

    def one_nan(z):
        z[0] = np.nan
        return z
    _patch_offline(monkeypatch, one_nan)
    (row,) = parity_report(tmp_path)["segments"].values()
    assert row.get("bound_vacuous") is True and row["k_max_kcal"] == 0.0
    assert row["n_finite_mismatch"] == 1 and row["ok"] is False
