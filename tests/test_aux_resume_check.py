import numpy as np

from gareus.auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA
from gareus.auxiliary_cv.resume_check import compare_runs, ledger_completeness


def _ledger(run, decisions):
    from gareus.store import ParquetExchangeWriter, SegmentRegistry
    reg = SegmentRegistry(run)
    sid = reg.open_segment("run", None, 1)
    w = ParquetExchangeWriter(run / "exchanges" / sid, flush_rows=1, event_schema=EXCHANGE_EVENT_SCHEMA)
    for step, seq, kind, r in decisions:
        w.write_event(step=step, attempt_seq=seq, selected_replica=r, replica_i=r, replica_j=r, window_i=r,
                      window_j=r, kind=kind, delta_e_kj=0.0, accepted=False, log_q_forward=0.0,
                      log_q_reverse=0.0, p_accept=0.0, energy_version="v3", assignments_after=[0, 1])
    w.close(); reg.close_segment(sid, 1000)


def test_ledger_completeness_counts_one_event_per_selected_carrier(tmp_path):
    from gareus.query import load_exchanges
    _ledger(tmp_path, [(100, 0, "stay", 0), (100, 1, "skip", 1), (200, 0, "stay", 1)])
    rep = ledger_completeness(load_exchanges(tmp_path), selected_per_step=2)
    assert rep["ok"] is False and rep["bad_steps"] == {200: 1}


def test_compare_runs_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    for run in (a, b):
        _ledger(run, [(100, 0, "stay", 0), (100, 1, "stay", 1)])
    rep = compare_runs(a, b)
    assert rep["ok"] and rep["ledger_duplicates"] == 0


def _patched(monkeypatch, control, resumed):
    ev = {"step": np.array([100]), "attempt_seq": np.array([0]), "assignment_sha256_after": np.array(["a"])}
    monkeypatch.setattr("gareus.query.load_exchanges", lambda p: ev)
    monkeypatch.setattr("gareus.query.load_samples", lambda p: control if "ctl" in str(p) else resumed)


def test_compare_runs_detects_duplicate_sample_rows(monkeypatch, tmp_path):
    import gareus.auxiliary_cv.resume_check as rc
    base = {"step": np.array([100, 200]), "replica": np.array([0, 0]), "window_id": np.array([0, 1])}
    _patched(monkeypatch, base, {k: np.concatenate([v, v[-1:]]) for k, v in base.items()})
    out = rc.compare_runs(tmp_path / "ctl", tmp_path / "res")
    assert out["sample_duplicate_keys"]["resumed"] == 1 and out["sample_rows"] == {"control": 2, "resumed": 3}
    assert out["ok"] is False


def test_compare_runs_detects_torsion_drift_with_wrapping(monkeypatch, tmp_path):
    import gareus.auxiliary_cv.resume_check as rc
    base = {"step": np.array([100, 200]), "replica": np.array([0, 0]), "window_id": np.array([0, 1]),
            "tor_000": np.array([np.pi - 1e-12, 0.2]), "aux_z_00": np.array([0.5, 0.6])}
    same = dict(base, tor_000=np.array([-np.pi + 1e-12, 0.2]))           # equal angle across the branch cut
    _patched(monkeypatch, base, same)
    assert rc.compare_runs(tmp_path / "ctl", tmp_path / "res")["ok"] is True
    drift = dict(base, tor_000=np.array([np.pi - 1e-12, 0.3]))
    _patched(monkeypatch, base, drift)
    out = rc.compare_runs(tmp_path / "ctl", tmp_path / "res")
    assert out["ok"] is False and abs(out["max_abs_dtorsion"] - 0.1) < 1e-12
