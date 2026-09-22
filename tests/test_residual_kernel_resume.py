"""F04/P07: kernel identity bound to segments, sample eligibility at the loader, marker digests,
resume refusal for a changed or unknown kernel (spec F04).
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from gareus.kernel_identity import (ELIGIBLE_AFFECTED, ELIGIBLE_NOT_APPLICABLE, ELIGIBLE_UNKNOWN, ELIGIBLE_VERIFIED,
                                    EXCHANGE_ENERGY_VERSION, RESIDUAL_EVALUATOR_VERSION, classify_segment_kernel,
                                    kernel_identity_for_run)


def _residual_meta(**over):
    meta = {"enabled": True, "mode": "residual-torsion-pc", "cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
            "pair_model_sha256": "a" * 64}
    meta.update(over)
    return meta


def test_kernel_identity_is_content_derived_and_names_the_affected_pre_f01_kernel():
    args = types.SimpleNamespace(secondary_cv="residual-torsion-pc", production_ensemble="npt", gamd_boost_type="pep-gamd-lower-dual")
    ident = kernel_identity_for_run(args, _residual_meta())
    assert ident["cv_evaluator_version"] == RESIDUAL_EVALUATOR_VERSION and ident["exchange_energy_version"] == EXCHANGE_ENERGY_VERSION
    assert ident["pair_model_sha256"] == "a" * 64 and len(ident["digest"]) == 64
    assert kernel_identity_for_run(args, _residual_meta())["digest"] == ident["digest"]
    old = kernel_identity_for_run(args, {"enabled": True, "mode": "residual-torsion-pc"})   # metadata written by the old builder
    assert old["cv_evaluator_version"] == "affected_pre_f01" and old["digest"] != ident["digest"]
    none = kernel_identity_for_run(types.SimpleNamespace(secondary_cv="none"), {"enabled": False})
    assert none["secondary_cv_mode"] == "none" and none["cv_evaluator_version"] is None


def test_segment_classification_covers_every_terminal_state():
    ok = {"cv2_type": "residual-torsion-pc", "kernel_identity": {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                                                  "exchange_energy_version": EXCHANGE_ENERGY_VERSION}}
    assert classify_segment_kernel(ok)[0] == ELIGIBLE_VERIFIED
    assert classify_segment_kernel({"cv2_type": "residual-torsion-pc"})[0] == ELIGIBLE_UNKNOWN
    assert classify_segment_kernel({"cv2_type": "residual-torsion-pc", "kernel_identity": {"cv_evaluator_version": "affected_pre_f01"}})[0] == ELIGIBLE_AFFECTED
    assert classify_segment_kernel({"cv2_type": "residual-torsion-pc", "kernel_identity": {"cv_evaluator_version": "residual_full_expression_v9", "exchange_energy_version": EXCHANGE_ENERGY_VERSION}})[0] == ELIGIBLE_UNKNOWN
    assert classify_segment_kernel({"cv2_type": "rama-map"})[0] == ELIGIBLE_NOT_APPLICABLE
    assert classify_segment_kernel({"cv2_type": None})[0] == ELIGIBLE_NOT_APPLICABLE


def test_window_snapshot_records_the_kernel(tmp_path):
    from gareus.store import WindowSnapshot
    snap = WindowSnapshot(tmp_path)
    ident = kernel_identity_for_run(types.SimpleNamespace(secondary_cv="residual-torsion-pc"), _residual_meta())
    snap.snapshot("seg_001", [{"window": 0}], cv1_type="contacts", cv2_type="residual-torsion-pc", kernel_identity=ident)
    assert snap.load("seg_001")["kernel_identity"]["digest"] == ident["digest"]
    snap.snapshot("seg_002", [{"window": 0}], cv1_type="contacts", cv2_type="none")
    assert "kernel_identity" not in snap.load("seg_002")


def _write_run(tmp_path, segments):
    """segments: list of (seg_id, cv2_type, kernel_identity or None, n_rows)."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    run = tmp_path / "run"
    (run / "windows").mkdir(parents=True)
    regs = []
    for seg_id, cv2, ident, n in segments:
        seg_dir = run / "samples" / seg_id
        seg_dir.mkdir(parents=True)
        table = pa.table({"step": np.arange(n, dtype=np.int64), "window_id": np.zeros(n, dtype=np.int64),
                          "replica": np.zeros(n, dtype=np.int64), "primary_cv": np.linspace(0, 1, n)})
        pq.write_table(table, seg_dir / "chunk_000.parquet")
        payload = {"segment_id": seg_id, "cv1_type": "contacts", "cv2_type": cv2, "windows": []}
        if ident is not None:
            payload["kernel_identity"] = ident
        (run / "windows" / f"{seg_id}.json").write_text(json.dumps(payload))
        regs.append({"segment_id": seg_id, "run_id": "r", "parent_segment_id": None, "round_id": 0,
                     "start_step": 0, "end_step": n - 1, "status": "complete"})
    (run / "segments.json").write_text(json.dumps(regs))
    return run


def test_loader_excludes_affected_and_unknown_residual_segments_but_keeps_conventional_ones(tmp_path):
    from gareus import query
    good = {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION, "exchange_energy_version": EXCHANGE_ENERGY_VERSION}
    run = _write_run(tmp_path, [("seg_001", "residual-torsion-pc", None, 10),                      # unknown (pre-F01)
                                ("seg_002", "residual-torsion-pc", {"cv_evaluator_version": "affected_pre_f01"}, 20),
                                ("seg_003", "residual-torsion-pc", good, 30),
                                ("seg_004", "none", None, 40)])                                     # conventional: untouched
    elig = query.segment_eligibility(run)
    assert {k: v["eligibility"] for k, v in elig.items()} == {
        "seg_001": ELIGIBLE_UNKNOWN, "seg_002": ELIGIBLE_AFFECTED, "seg_003": ELIGIBLE_VERIFIED, "seg_004": ELIGIBLE_NOT_APPLICABLE}
    data = query.load_samples(run)
    kept = set(np.unique(np.asarray(data["segment_id"]).astype(str)).tolist())
    assert kept == {"seg_003", "seg_004"} and len(data["step"]) == 70
    assert set(query.LAST_ELIGIBILITY_REPORT["excluded_segments"]) == {"seg_001", "seg_002"}
    everything = query.load_samples(run, include_ineligible=True)
    assert len(everything["step"]) == 100                       # nothing was deleted or relabelled
    # explicit segment_ids never smuggle an ineligible segment back in
    only = query.load_samples(run, segment_ids=["seg_001", "seg_003"])
    assert set(np.unique(np.asarray(only["segment_id"]).astype(str)).tolist()) == {"seg_003"}


def test_epoch0_marker_records_digests_and_a_changed_artefact_blocks_the_resume(tmp_path):
    from gareus.swarm import epoch0 as E
    an = tmp_path / "swarm" / "analysis"
    (an / "shared_gamd_setup").mkdir(parents=True)
    (an / "seed_bank").mkdir()
    (an / "windows_lambda_ladder.csv").write_text("window,primary_cv_center\n0,0.1\n")
    (an / "shared_gamd_setup" / "shared_gamd_setup_globals.json").write_text("{}")
    (an / "seed_bank" / "final_survivor_seeds.csv").write_text("seed_id\n")
    with pytest.raises(FileNotFoundError, match="ladder_run_args.yaml"):
        E.mark_epoch0_complete(tmp_path, n_members=1, ns_charged=1.0, auto_cv=True, selection_status="pair")
    (an / "ladder_run_args.yaml").write_text("cvs: {cv1: contacts, cv2: residual-torsion-pc}\n")
    with pytest.raises(FileNotFoundError, match="cv_pair_model.json"):
        E.mark_epoch0_complete(tmp_path, n_members=1, ns_charged=1.0, auto_cv=True, selection_status="pair")
    for name in E.PAIR_ARTEFACTS:
        (an / name).write_text("{}")
    (an / "swarm_gate.json").write_text(json.dumps({"status": "pass"}))
    (tmp_path / "swarm" / "round_000").mkdir()
    (tmp_path / "swarm" / "round_000" / "plan.csv").write_text("member_id\n")
    E.mark_epoch0_complete(tmp_path, n_members=1, ns_charged=1.0, auto_cv=True, selection_status="pair")
    marker = json.loads((an / "epoch0_complete.json").read_text())
    assert marker["selection_status"] == "pair" and "cv_pair_model.json" in marker["artefact_digests"]
    assert marker["artefact_digests"]["seed_bank/final_survivor_seeds.csv"]
    assert E.epoch0_status(tmp_path)["state"] == "complete"
    (an / "windows_lambda_ladder.csv").write_text("window,primary_cv_center\n0,0.9\n")      # someone edited the table
    status = E.epoch0_status(tmp_path)
    assert status["state"] == "blocked" and "windows_lambda_ladder.csv" in status["changed_artefacts"]


def test_resume_kernel_check_refuses_changed_or_unknown_kernels_and_passes_unrelated_runs(tmp_path):
    from gareus.production import verify_kernel_identity_on_resume
    (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {
        "exchange_energy_version": EXCHANGE_ENERGY_VERSION, "cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION}}))
    verify_kernel_identity_on_resume(types.SimpleNamespace(), tmp_path, _residual_meta())   # same kernel: fine
    (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {
        "exchange_energy_version": "state_bias_matrix_v1", "cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION}}))
    with pytest.raises(RuntimeError, match="exchanged with"):
        verify_kernel_identity_on_resume(types.SimpleNamespace(), tmp_path, _residual_meta())
    (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {"seq": "GYDPETGTWG"}}))   # pre-F01 manifest
    with pytest.raises(RuntimeError, match="records no kernel identity"):
        verify_kernel_identity_on_resume(types.SimpleNamespace(), tmp_path, _residual_meta())
    verify_kernel_identity_on_resume(types.SimpleNamespace(), tmp_path, {"enabled": False})            # conventional run: continues
    verify_kernel_identity_on_resume(types.SimpleNamespace(), tmp_path, {"enabled": True, "mode": "rama-map"})


def test_fresh_process_sees_the_same_epoch0_verdicts(tmp_path):
    """The status decisions are read off disk by whichever job starts next, so exercise them
    from a separate interpreter (a real fresh process, not this test's module state)."""
    from gareus.swarm import epoch0 as E
    an = tmp_path / "swarm" / "analysis"
    (an / "shared_gamd_setup").mkdir(parents=True); (an / "seed_bank").mkdir()
    (tmp_path / "swarm" / "round_000").mkdir(); (tmp_path / "swarm" / "round_000" / "plan.csv").write_text("member_id\n")
    (an / "windows_lambda_ladder.csv").write_text("w\n0\n")
    (an / "shared_gamd_setup" / "shared_gamd_setup_globals.json").write_text("{}")
    (an / "seed_bank" / "final_survivor_seeds.csv").write_text("seed_id\n")
    (an / "ladder_run_args.yaml").write_text("cvs: {cv1: contacts, cv2: none}\n")
    (an / "swarm_gate.json").write_text(json.dumps({"status": "pass"}))
    E.mark_epoch0_complete(tmp_path, n_members=1, ns_charged=1.0, auto_cv=True, selection_status="cv1_only")
    code = ("import json,sys; from gareus.swarm.epoch0 import epoch0_status, apply_epoch0_sidecar; import types\n"
            f"s = epoch0_status({str(tmp_path)!r}); a = types.SimpleNamespace(secondary_cv='auto')\n"
            f"apply_epoch0_sidecar(a, {str(tmp_path)!r})\n"
            "print(json.dumps({'state': s['state'], 'cv2': a.secondary_cv}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(Path(__file__).resolve().parents[1]))
    assert json.loads(out.stdout.strip().splitlines()[-1]) == {"state": "complete", "cv2": "none"}


def _snapshot(tmp_path, segment_id, identity):
    (tmp_path / "windows").mkdir(exist_ok=True)
    payload = {"segment_id": segment_id, "cv1_type": "contacts", "cv2_type": "residual-torsion-pc", "windows": []}
    if identity is not None:
        payload["kernel_identity"] = identity
    (tmp_path / "windows" / f"{segment_id}.json").write_text(json.dumps(payload))


def test_resume_kernel_check_falls_back_to_the_segment_snapshot_when_the_manifest_is_a_skeleton(tmp_path):
    """chignolin_8 job 2574830 (2026-09-22): the epoch manifest had been reduced to a skeleton
    (method_settings held only state_gamd_lambdas) while windows/seg_001.json still carried the full
    kernel_identity of the code that produced the samples. The snapshot is the record the loader
    classifies eligibility from, so the resume guard must read it before declaring 'pre-F01'."""
    from gareus.production import verify_kernel_identity_on_resume
    (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {"state_gamd_lambdas": [0.0, 1.0]}}))
    _snapshot(tmp_path, "seg_001", {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                    "exchange_energy_version": EXCHANGE_ENERGY_VERSION})
    verify_kernel_identity_on_resume(types.SimpleNamespace(), tmp_path, _residual_meta())   # recorded == current: continues


def test_resume_kernel_check_uses_the_latest_snapshot_and_still_refuses_a_changed_kernel(tmp_path):
    from gareus.production import verify_kernel_identity_on_resume
    (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {"state_gamd_lambdas": [0.0]}}))
    _snapshot(tmp_path, "seg_001", {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                    "exchange_energy_version": EXCHANGE_ENERGY_VERSION})
    _snapshot(tmp_path, "seg_002", {"cv_evaluator_version": "affected_pre_f01",
                                    "exchange_energy_version": EXCHANGE_ENERGY_VERSION})
    with pytest.raises(RuntimeError, match="cv_evaluator_version"):
        verify_kernel_identity_on_resume(types.SimpleNamespace(), tmp_path, _residual_meta())
    _snapshot(tmp_path, "seg_002", {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                    "exchange_energy_version": "state_bias_matrix_v1"})
    with pytest.raises(RuntimeError, match="exchanged with"):
        verify_kernel_identity_on_resume(types.SimpleNamespace(), tmp_path, _residual_meta())


def test_resume_kernel_check_still_refuses_when_neither_manifest_nor_snapshot_records_a_kernel(tmp_path):
    from gareus.production import verify_kernel_identity_on_resume
    (tmp_path / "run_manifest.json").write_text(json.dumps({"method_settings": {"seq": "GYDPETGTWG"}}))
    _snapshot(tmp_path, "seg_001", None)   # a genuinely pre-F01 segment: snapshot without kernel_identity
    with pytest.raises(RuntimeError, match="records no kernel identity"):
        verify_kernel_identity_on_resume(types.SimpleNamespace(), tmp_path, _residual_meta())
