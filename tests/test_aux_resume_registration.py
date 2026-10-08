"""Task 14 fix round 1, F4: an auxiliary resume refusal has no side effects, and the re-seal walks back over
orphan segments to the checkpoint's own segment (no step is pooled twice)."""
import ast
import inspect
import json

import numpy as np
import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.checkpoint import aux_checkpoint_block, expected_aux_parameters
from gareus.auxiliary_cv.force import AuxForceInfo
from gareus.auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.resume_check import ledger_completeness
from gareus.correctness._io import IntegrityError

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
INFO = AuxForceInfo("ATLaSAuxCVUmbrella", 7, MODEL.model_sha256, "aux_k", "aux_c", ("aux_cos_neg", "aux_sin_neg"))
TOPO, KID, A = "c" * 64, "d" * 64, [0, 1, 2]


def _defn(aux_k=1.2):
    spec = [(0.0, 10.0, 0.0, 0.0, "ordinary"), (0.2, 10.0, 0.0, 0.0, "ordinary"), (0.4, 10.0, aux_k, 1.5, "auxiliary")]
    return definition(rows(spec, MODEL.model_sha256), MODEL)


def _write(run, seg, steps):
    from gareus.store import ParquetExchangeWriter, ParquetSampleWriter
    sw = ParquetSampleWriter(run / "samples" / seg, flush_rows=1)
    ew = ParquetExchangeWriter(run / "exchanges" / seg, flush_rows=1, event_schema=EXCHANGE_EVENT_SCHEMA)
    for step in steps:
        for r in A:
            sw.write_sample(step, r, A[r], 0.1 * r, None, 0.0, 0.0, 0.0, 0.0)
            ew.write_event(step=step, attempt_seq=r, selected_replica=r, replica_i=r, replica_j=r, window_i=A[r],
                           window_j=A[r], kind="stay", delta_e_kj=0.0, accepted=False, log_q_forward=0.0,
                           log_q_reverse=float("nan"), p_accept=float("nan"), energy_version="v3",
                           assignments_after=A)
    sw.close()
    ew.close()


def _crashed_campaign(run, *, orphan):
    """seg_001: checkpoint at 200, exception exit at 300; optionally an empty orphan left by a refused resume."""
    from gareus.store import SegmentRegistry, finalize_segment
    reg = SegmentRegistry(run)
    seg = reg.open_segment("run", None, 1)
    _write(run, seg, [100, 200, 300])
    finalize_segment(reg, seg, completed_cleanly=False, writers_ok=True, end_step=300)
    if orphan:
        o = reg.open_segment("run", seg, 1)
        finalize_segment(reg, o, completed_cleanly=False, writers_ok=True, end_step=0)
    d = _defn()
    block = aux_checkpoint_block(
        state_definition=d, force_info=INFO, assignments=A,
        observed_params=[expected_aux_parameters(d, w) for w in A], topology_sha256=TOPO,
        kernel_identity_digest=KID, segment_id=seg,
        ledger_anchor={"segment_id": seg, "start_step": 0, "start_assignments": A},
        data_boundary={"samples": {"generation": 1, "n_rows": 6}, "exchanges": {"generation": 1, "n_rows": 6}})
    return seg, {"absolute_step": 200, "assignments": A, "aux": block}


def _prepare(run, manifest, defn):
    from gareus.auxiliary_cv.runtime_io import prepare_aux_resume
    from gareus.store import SegmentRegistry
    return prepare_aux_resume(run, SegmentRegistry(run), manifest, state_definition=defn, force_info=INFO,
                              topology_sha256=TOPO, kernel_identity_digest=KID)


def _child(run, steps):
    from gareus.store import SegmentRegistry
    reg = SegmentRegistry(run)
    child = reg.open_segment("run", reg.get_latest_segment()["segment_id"], 1)
    reg.set_segment_start_step(child, 200)
    _write(run, child, steps)
    reg.close_segment(child, max(steps))


@pytest.mark.parametrize("orphan", [False, True])
def test_refuse_then_fix_then_resume_pools_every_step_once(tmp_path, orphan):
    from gareus.query import load_exchanges, load_samples
    _seg, manifest = _crashed_campaign(tmp_path, orphan=orphan)
    before = (tmp_path / "segments.json").read_bytes()
    # 1. a refused resume (here: the state definition drifted) changes nothing on disk
    with pytest.raises(IntegrityError, match="state_definition_sha256"):
        _prepare(tmp_path, manifest, _defn(aux_k=2.0))
    assert (tmp_path / "segments.json").read_bytes() == before
    # 2. cause fixed: the resume re-seals (walking back over any orphan) and the child re-runs 300
    records = _prepare(tmp_path, manifest, _defn())
    segs = json.loads((tmp_path / "segments.json").read_text())
    assert segs[0]["end_step"] == 200 and segs[0]["status"] == "interrupted"
    if orphan:
        assert segs[1]["status"] == "abandoned" and len(records) == 2
    _child(tmp_path, [300, 400])
    s = load_samples(tmp_path)
    keys = list(zip(np.asarray(s["step"]).tolist(), np.asarray(s["replica"]).tolist()))
    assert len(keys) == len(set(keys)) == 4 * 3
    rep = ledger_completeness(load_exchanges(tmp_path), selected_per_step=3)
    assert rep["ok"] and rep["steps"] == 4


def test_reseal_chain_refuses_a_complete_segment_after_the_checkpoint(tmp_path):
    from gareus.store import SegmentRegistry, reseal_chain_for_resume
    reg = SegmentRegistry(tmp_path)
    a = reg.open_segment("run", None, 1)
    b = reg.open_segment("run", a, 1)
    reg.close_segment(b, 900)
    with pytest.raises(RuntimeError, match="complete"):
        reseal_chain_for_resume(reg, a, 200)
    with pytest.raises(RuntimeError, match="not in the segment registry"):
        reseal_chain_for_resume(reg, "seg_099", 200)


def test_aux_resume_without_checkpoint_refuses_when_a_committed_parent_exists(tmp_path):
    from gareus.auxiliary_cv.runtime_io import refuse_aux_resume_without_checkpoint
    from gareus.store import SegmentRegistry, finalize_segment
    refuse_aux_resume_without_checkpoint(tmp_path)                     # no registry: fresh start is fine
    reg = SegmentRegistry(tmp_path)
    e = reg.open_segment("run", None, 1)
    finalize_segment(reg, e, completed_cleanly=False, writers_ok=True, end_step=0)
    refuse_aux_resume_without_checkpoint(tmp_path)                     # only an empty segment: fine
    c = reg.open_segment("run", e, 1)
    _write(tmp_path, c, [100])
    finalize_segment(reg, c, completed_cleanly=False, writers_ok=True, end_step=100)
    with pytest.raises(IntegrityError, match="no production checkpoint"):
        refuse_aux_resume_without_checkpoint(tmp_path)


def test_discard_refused_segment_removes_only_an_empty_latest_segment(tmp_path):
    from gareus.auxiliary_cv.runtime_io import discard_refused_segment
    from gareus.store import ParquetExchangeWriter, ParquetSampleWriter, SegmentRegistry, WindowSnapshot
    reg = SegmentRegistry(tmp_path)
    a = reg.open_segment("run", None, 1)
    _write(tmp_path, a, [100])
    reg.close_segment(a, 100)
    assert discard_refused_segment(tmp_path, reg, a) is False          # holds data: kept
    b = reg.open_segment("run", a, 1)
    WindowSnapshot(tmp_path).snapshot(b, [], cv1_type="distance", cv2_type=None)
    ParquetSampleWriter(tmp_path / "samples" / b).close()
    ParquetExchangeWriter(tmp_path / "exchanges" / b, event_schema=EXCHANGE_EVENT_SCHEMA).close()
    assert discard_refused_segment(tmp_path, reg, b) is True
    assert [s["segment_id"] for s in json.loads((tmp_path / "segments.json").read_text())] == [a]
    assert not (tmp_path / "samples" / b).exists() and not (tmp_path / "exchanges" / b).exists()
    assert not (tmp_path / "windows" / f"{b}.json").exists()
    assert discard_refused_segment(tmp_path, SegmentRegistry(tmp_path), a) is False


def _run_gareus_tree():
    import gareus.production as production
    return ast.parse(inspect.getsource(production.run_gareus))


def test_aux_resume_checks_run_before_the_segment_is_registered():
    src = ast.unparse(_run_gareus_tree())
    prepare = src.index("prepare_aux_resume(")
    opens = [i for i in range(len(src)) if src.startswith("_seg_registry.open_segment(", i)]
    assert len(opens) == 2                     # legacy (unconditional for non-aux) + aux (after the checks)
    assert opens[0] < prepare < opens[1]
    assert "_seg_id = None if getattr(args, '_aux_runtime', None) is not None else " \
           "_seg_registry.open_segment(_run_id, _parent_seg_id, _round_id)" in src
    assert "refuse_aux_resume_without_checkpoint(out_dir)" in src
    assert "discard_refused_segment(out_dir, _seg_registry, _seg_id)" in src
