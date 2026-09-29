"""Frozen-final extension round numbering must survive a walltime interrupt.

chignolin_9 (2026-09-29): job 2708570 was interrupted inside final_extension_002.
The interrupt summary it wrote carried no ``final_extension_summaries``, so the
next job read zero completed rounds, restarted numbering at round 1 and resumed
final_extension_001 with round 2's step target instead of continuing round 2.
"""
import ast
import inspect
import json
import textwrap
from pathlib import Path

import pytest

from gareus import adaptive_production as ap


def _make_ext_dir(adaptive_dir: Path, index: int, with_checkpoint: bool = True) -> Path:
    d = adaptive_dir / f"final_extension_{index:03d}"
    (d / "checkpoints").mkdir(parents=True)
    if with_checkpoint:
        (d / "checkpoints" / "production_checkpoint_manifest.json").write_text("{}")
    return d


def test_interrupt_summary_carries_extension_records(tmp_path):
    ext = [{"extension": 1, "dir": str(tmp_path / "final_extension_001"), "steps": 2592077}]
    payload = ap._interrupted_driver_summary(
        tmp_path / "final_extension_002", epoch_summaries=[{"epoch": 0}], extension_summaries=ext)

    assert payload["status"] == "interrupted_after_checkpoint"
    assert payload["interrupted_segment"] == str(tmp_path / "final_extension_002")
    assert payload["epochs_completed"] == 1
    assert payload["final_extension_summaries"] == ext


def test_interrupt_summary_round_trips_to_next_resume_round(tmp_path):
    # What the resume path reads back decides the next round's directory.
    ext = [{"extension": 1, "dir": "final_extension_001", "steps": 2592077}]
    summary = tmp_path / "adaptive_production_driver_summary.json"
    summary.write_text(json.dumps(ap._interrupted_driver_summary(
        tmp_path / "final_extension_002", epoch_summaries=[], extension_summaries=ext)))

    old = json.loads(summary.read_text())
    assert len(old.get("final_extension_summaries", [])) == 1  # -> resumes final_extension_002


def test_interrupt_summary_copies_extension_list(tmp_path):
    ext = [{"extension": 1}]
    payload = ap._interrupted_driver_summary(tmp_path, epoch_summaries=[], extension_summaries=ext)
    ext.append({"extension": 2})
    assert len(payload["final_extension_summaries"]) == 1


def test_interrupt_summary_extra_fields_are_kept(tmp_path):
    payload = ap._interrupted_driver_summary(
        tmp_path, epoch_summaries=[], extension_summaries=[], scheduled_final={"a": 1})
    assert payload["scheduled_final"] == {"a": 1}
    assert payload["final_extension_summaries"] == []


def test_guard_accepts_resuming_next_round(tmp_path):
    _make_ext_dir(tmp_path, 1)
    _make_ext_dir(tmp_path, 2)  # in-progress round 2 after one recorded round
    ap._check_extension_rounds_on_disk(tmp_path, start_ext_round=1)


def test_guard_accepts_fresh_campaign(tmp_path):
    ap._check_extension_rounds_on_disk(tmp_path, start_ext_round=0)


def test_guard_rejects_lost_extension_records(tmp_path):
    # The chignolin_9 state: round 2 holds a checkpoint but the summary records none.
    _make_ext_dir(tmp_path, 1)
    _make_ext_dir(tmp_path, 2)
    with pytest.raises(RuntimeError, match="final_extension_002"):
        ap._check_extension_rounds_on_disk(tmp_path, start_ext_round=0)


def test_guard_ignores_round_dir_without_checkpoint(tmp_path):
    # mkdir happens before run_gareus; a dir with no checkpoint holds no MD to lose.
    _make_ext_dir(tmp_path, 1)
    _make_ext_dir(tmp_path, 3, with_checkpoint=False)
    ap._check_extension_rounds_on_disk(tmp_path, start_ext_round=1)


def test_driver_writes_no_interrupt_summary_outside_helper():
    """Every driver-summary interrupt write must go through _interrupted_driver_summary."""
    src = textwrap.dedent(inspect.getsource(ap.run_adaptive_production_auto_loop))
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        pairs = {k.value: v for k, v in zip(node.keys, node.values)
                 if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        status = pairs.get("status")
        if isinstance(status, ast.Constant) and status.value == "interrupted_after_checkpoint":
            offenders.append(node.lineno)
    assert offenders == [], f"inline interrupt summaries at loop lines {offenders}"
    assert src.count("_interrupted_driver_summary(") >= 5
    assert "_check_extension_rounds_on_disk(" in src
