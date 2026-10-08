import numpy as np
import pytest

from gareus.auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA, assignment_sha256, replay_assignments
from gareus.correctness._io import IntegrityError


def _writer(run_dir, seg_id):
    from gareus.store import ParquetExchangeWriter
    return ParquetExchangeWriter(run_dir / "exchanges" / seg_id, flush_rows=1, event_schema=EXCHANGE_EVENT_SCHEMA)


def _swap(w, step, seq, ri, rj, wi, wj, after, accepted=True):
    w.write_event(step=step, attempt_seq=seq, selected_replica=ri, replica_i=ri, replica_j=rj, window_i=wi,
                  window_j=wj, kind="swap", delta_e_kj=0.0, accepted=accepted, log_q_forward=0.0,
                  log_q_reverse=0.0, p_accept=1.0, energy_version="v3", assignments_after=after)


def _quiet(w, step, seq, r, win, after, kind="stay"):
    w.write_event(step=step, attempt_seq=seq, selected_replica=r, replica_i=r, replica_j=r, window_i=win,
                  window_j=win, kind=kind, delta_e_kj=0.0, accepted=False, log_q_forward=0.0,
                  log_q_reverse=float("nan"), p_accept=float("nan"), energy_version="v3", assignments_after=after)


def _crashed_parent(tmp_path):
    """Checkpoint at 200 ([1,2,0]); an accepted swap at 300; then an EXCEPTION exit (finalize_segment)."""
    from gareus.store import SegmentRegistry, finalize_segment
    reg = SegmentRegistry(tmp_path)
    seg = reg.open_segment("run", None, 1)
    w = _writer(tmp_path, seg)
    _swap(w, 100, 0, 0, 1, 0, 1, [1, 0, 2])
    _swap(w, 100, 1, 1, 2, 0, 2, [1, 2, 0])        # second accepted swap at the SAME step
    _quiet(w, 200, 0, 0, 1, [1, 2, 0])
    _quiet(w, 200, 1, 2, 0, [1, 2, 0], kind="skip")
    _swap(w, 200, 2, 0, 2, 1, 0, [1, 2, 0], accepted=False)
    _swap(w, 300, 0, 0, 1, 1, 2, [2, 1, 0])        # after the checkpoint: phantom once resumed
    w.flush()
    finalize_segment(reg, seg, completed_cleanly=False, writers_ok=True, end_step=300)
    return reg, seg


def test_exception_exit_seals_at_crash_step_then_resume_reseals_at_checkpoint(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import reseal_parent_for_resume
    reg, seg = _crashed_parent(tmp_path)
    assert reg.get_segment(seg)["end_step"] == 300 and int(np.max(load_exchanges(tmp_path)["step"])) == 300
    rec = reseal_parent_for_resume(reg, seg, 200)
    assert rec == {"segment_id": seg, "previous_status": "interrupted", "previous_end_step": 300, "end_step": 200}
    ev = load_exchanges(tmp_path)
    assert int(np.max(ev["step"])) == 200
    assert replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=10**9) == [1, 2, 0]


def test_reseal_running_parent_and_leave_complete_alone(tmp_path):
    from gareus.store import SegmentRegistry, reseal_parent_for_resume
    reg = SegmentRegistry(tmp_path)
    a = reg.open_segment("run", None, 1)
    assert reseal_parent_for_resume(reg, a, 500)["end_step"] == 500        # running, end None
    b = reg.open_segment("run", a, 1)
    reg.close_segment(b, 900)
    assert reseal_parent_for_resume(reg, b, 400) is None and reg.get_segment(b)["status"] == "complete"
    assert reseal_parent_for_resume(reg, "nope", 1) is None


def test_crash_during_flush_leaves_orphan_tmp_ignored(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import reseal_parent_for_resume
    reg, seg = _crashed_parent(tmp_path)
    reseal_parent_for_resume(reg, seg, 200)
    (tmp_path / "exchanges" / seg / "chunk_000099.parquet.tmp.4242").write_bytes(b"partial")
    assert replay_assignments(load_exchanges(tmp_path), [0, 1, 2], after_step=0, up_to_step=200) == [1, 2, 0]


def test_checksum_mismatch_raises(tmp_path):
    from gareus.query import load_exchanges
    _crashed_parent(tmp_path)
    ev = load_exchanges(tmp_path)
    ev["assignment_sha256_after"] = np.asarray(ev["assignment_sha256_after"], dtype=object).copy()
    ev["assignment_sha256_after"][1] = assignment_sha256([9, 9, 9])
    with pytest.raises(IntegrityError, match="step 100 seq 1"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)


def test_window_mismatch_raises(tmp_path):
    from gareus.query import load_exchanges
    _crashed_parent(tmp_path)
    with pytest.raises(IntegrityError, match="window"):
        replay_assignments(load_exchanges(tmp_path), [2, 1, 0], after_step=0, up_to_step=200)


def test_duplicate_step_seq_raises(tmp_path):
    from gareus.query import load_exchanges
    _crashed_parent(tmp_path)
    ev = load_exchanges(tmp_path)
    ev = {k: np.concatenate([np.asarray(v), np.asarray(v)[:1]]) for k, v in ev.items()}
    with pytest.raises(IntegrityError, match="duplicate"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)


def test_legacy_rows_mixed_into_event_ledger_raise(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import ParquetExchangeWriter, SegmentRegistry
    reg = SegmentRegistry(tmp_path)
    s0 = reg.open_segment("run", None, 1)
    w0 = ParquetExchangeWriter(tmp_path / "exchanges" / s0)
    w0.write_exchange(step=50, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=True)
    w0.close(); reg.close_segment(s0, 50)
    s1 = reg.open_segment("run", s0, 1)
    w1 = _writer(tmp_path, s1)
    _swap(w1, 100, 0, 0, 1, 1, 0, [0, 1, 2])
    w1.close(); reg.close_segment(s1, 100)
    with pytest.raises(IntegrityError, match="legacy exchange rows mixed"):
        replay_assignments(load_exchanges(tmp_path), [1, 0, 2], after_step=0, up_to_step=100)


def test_float_nan_event_columns_raise_before_any_int_cast(tmp_path):
    from gareus.query import load_exchanges
    reg_seg = "seg_001"
    w = _writer(tmp_path, reg_seg)
    _swap(w, 100, 0, 0, 1, 1, 0, [1, 0, 2])
    _swap(w, 200, 0, 0, 1, 1, 0, [0, 1, 2])
    w.close()
    ev = dict(load_exchanges(tmp_path))
    seq = np.asarray(np.ma.getdata(ev["attempt_seq"]), dtype=np.float64).copy()
    seq[1] = np.nan
    ev["attempt_seq"] = seq
    with pytest.raises(IntegrityError, match="legacy exchange rows mixed.*NaN"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)
    sha = np.asarray(ev["assignment_sha256_after"], dtype=object).copy()
    sha[0] = float("nan")
    ev["attempt_seq"] = np.asarray(np.ma.getdata(load_exchanges(tmp_path)["attempt_seq"]))
    ev["assignment_sha256_after"] = sha
    with pytest.raises(IntegrityError, match="legacy exchange rows mixed.*None/NaN"):
        replay_assignments(ev, [0, 1, 2], after_step=0, up_to_step=200)


def test_legacy_exchanges_cannot_be_replayed(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import ParquetExchangeWriter
    w = ParquetExchangeWriter(tmp_path / "exchanges" / "seg_001")
    w.write_exchange(step=5, replica_i=0, replica_j=1, window_i=0, window_j=1, delta_e=0.5, accepted=True)
    w.close()
    with pytest.raises(IntegrityError, match="ordered event"):
        replay_assignments(load_exchanges(tmp_path), [0, 1], after_step=0, up_to_step=10)


def test_production_resume_reseal_is_aux_gated_and_legacy_seal_is_unchanged():
    """B4 (user ruling): re-seal for auxiliary runs only; the legacy checkpoint-resume seal stays verbatim."""
    import ast
    import inspect
    import gareus.production as production
    tree = ast.parse(inspect.getsource(production.run_gareus))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def calls(name):
        return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", getattr(n.func, "id", None)) == name]

    def if_tests(node):
        out = []
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                out.append(ast.unparse(node.test))
        return out

    reseal = calls("reseal_parent_for_resume")
    assert len(reseal) == 1 and any("aux" in t for t in if_tests(reseal[0]))
    interrupted = [c for c in calls("seal_segment") if any(
        k.arg == "status" and isinstance(k.value, ast.Constant) and k.value.value == "interrupted"
        for k in c.keywords)]
    assert len(interrupted) == 1, "the legacy checkpoint-resume seal must stay"
    assert if_tests(interrupted[0])[0] == "_parent_was_running and _parent_seg_id is not None"
    assert ast.unparse(interrupted[0]) == (
        "_seg_registry.seal_segment(_parent_seg_id, absolute_end_step=int(manifest.get('absolute_step', 0)), "
        "status='interrupted')")
