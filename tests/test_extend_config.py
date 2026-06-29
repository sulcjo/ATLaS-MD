"""Tests for --extend / extend_mode config surface and final-phase skip guard.

No OpenMM, PeptideBuilder, or gamd-openmm imports.
"""
from __future__ import annotations

import json
import types

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(argv: list[str]):
    """Run the real parse_args path (no OpenMM required)."""
    from gareus.cli import parse_args
    base = ["--seq", "AA", "--run-mode", "cmd"]
    return parse_args(base + argv)


# ---------------------------------------------------------------------------
# Test 1: Config parsing — YAML flat-key dispatch
# ---------------------------------------------------------------------------

class TestConfigParsing:
    def test_extend_default_is_false(self):
        """--extend defaults to False."""
        args = _parse([])
        assert args.extend is False

    def test_extend_mode_default_is_auto(self):
        """--extend-mode defaults to 'auto'."""
        args = _parse([])
        assert args.extend_mode == "auto"

    def test_extend_flag_sets_true(self):
        """--extend sets args.extend to True."""
        args = _parse(["--extend"])
        assert args.extend is True

    def test_extend_mode_frozen(self):
        """--extend-mode frozen sets args.extend_mode to 'frozen'."""
        args = _parse(["--extend-mode", "frozen"])
        assert args.extend_mode == "frozen"

    def test_extend_and_extend_mode_together(self):
        """--extend and --extend-mode frozen together work correctly."""
        args = _parse(["--extend", "--extend-mode", "frozen"])
        assert args.extend is True
        assert args.extend_mode == "frozen"

    def test_extend_from_yaml(self, tmp_path):
        """extend: true is accepted from YAML and propagates to args.extend."""
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            "sequence:\n  seq: AA\n"
            "output:\n"
            "  extend: true\n"
            "  extend_mode: frozen\n"
        )
        from gareus.cli import parse_args
        args = parse_args(["--config", str(cfg), "--run-mode", "cmd"])
        assert args.extend is True
        assert args.extend_mode == "frozen"

    def test_extend_mode_choices(self):
        """extend_mode accepts all documented choices."""
        for choice in ["auto", "regular", "adaptive", "frozen", "topup"]:
            args = _parse(["--extend-mode", choice])
            assert args.extend_mode == choice


# ---------------------------------------------------------------------------
# Test 2: Detection function
# ---------------------------------------------------------------------------

class TestIsAdaptiveProductionCompleted:
    def test_completed_status_returns_true(self, tmp_path):
        """status='completed' → True."""
        from gareus.adaptive_production import _is_adaptive_production_completed
        summary = tmp_path / "adaptive_production_driver_summary.json"
        summary.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        assert _is_adaptive_production_completed(tmp_path) is True

    def test_interrupted_status_returns_false(self, tmp_path):
        """status='interrupted_after_checkpoint' → False."""
        from gareus.adaptive_production import _is_adaptive_production_completed
        summary = tmp_path / "adaptive_production_driver_summary.json"
        summary.write_text(
            json.dumps({"status": "interrupted_after_checkpoint"}), encoding="utf-8"
        )
        assert _is_adaptive_production_completed(tmp_path) is False

    def test_missing_file_returns_false(self, tmp_path):
        """Missing summary file → False."""
        from gareus.adaptive_production import _is_adaptive_production_completed
        assert _is_adaptive_production_completed(tmp_path) is False

    def test_missing_status_key_returns_false(self, tmp_path):
        """Summary file without 'status' key → False."""
        from gareus.adaptive_production import _is_adaptive_production_completed
        summary = tmp_path / "adaptive_production_driver_summary.json"
        summary.write_text(json.dumps({"schema_version": "v1"}), encoding="utf-8")
        assert _is_adaptive_production_completed(tmp_path) is False

    def test_corrupt_json_returns_false(self, tmp_path):
        """Corrupt JSON → False (no exception raised)."""
        from gareus.adaptive_production import _is_adaptive_production_completed
        summary = tmp_path / "adaptive_production_driver_summary.json"
        summary.write_text("{not valid json}", encoding="utf-8")
        assert _is_adaptive_production_completed(tmp_path) is False


# ---------------------------------------------------------------------------
# Test 3: Final-phase skip condition inputs
# ---------------------------------------------------------------------------

class TestFinalPhaseSkipCondition:
    def test_both_conditions_true_when_completed_and_has_samples(self, tmp_path):
        """With status=completed and a non-empty final/samples.csv, both guards return True."""
        from gareus.adaptive_production import _is_adaptive_production_completed, _run_dir_has_samples

        # Create a completed summary
        summary = tmp_path / "adaptive_production_driver_summary.json"
        summary.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

        # Create a fake final/samples.csv with content
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        samples_csv = final_dir / "samples.csv"
        samples_csv.write_bytes(b"cv1,cv2\n0.1,0.2\n")

        assert _is_adaptive_production_completed(tmp_path) is True
        assert _run_dir_has_samples(final_dir) is True

    def test_skip_condition_false_when_no_samples(self, tmp_path):
        """Even with status=completed, empty final dir means skip condition is False."""
        from gareus.adaptive_production import _is_adaptive_production_completed, _run_dir_has_samples

        summary = tmp_path / "adaptive_production_driver_summary.json"
        summary.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

        # final dir exists but no samples
        final_dir = tmp_path / "final"
        final_dir.mkdir()

        assert _is_adaptive_production_completed(tmp_path) is True
        assert _run_dir_has_samples(final_dir) is False

    def test_skip_condition_false_when_not_completed(self, tmp_path):
        """With status=interrupted, skip condition is False even if samples exist."""
        from gareus.adaptive_production import _is_adaptive_production_completed, _run_dir_has_samples

        summary = tmp_path / "adaptive_production_driver_summary.json"
        summary.write_text(
            json.dumps({"status": "interrupted_after_checkpoint"}), encoding="utf-8"
        )

        final_dir = tmp_path / "final"
        final_dir.mkdir()
        samples_csv = final_dir / "samples.csv"
        samples_csv.write_bytes(b"cv1,cv2\n0.1,0.2\n")

        assert _is_adaptive_production_completed(tmp_path) is False
        assert _run_dir_has_samples(final_dir) is True


# ---------------------------------------------------------------------------
# Test: config.py template has extend and extend_mode keys
# ---------------------------------------------------------------------------

class TestBasicChignolinConfigHasExtendKeys:
    def test_extend_key_in_output(self):
        """_basic_chignolin_config has extend: False in output block."""
        from gareus.config import _basic_chignolin_config
        result = _basic_chignolin_config()
        output = result["output"]
        assert "extend" in output
        assert output["extend"] is False

    def test_extend_mode_key_in_output(self):
        """_basic_chignolin_config has extend_mode: 'auto' in output block."""
        from gareus.config import _basic_chignolin_config
        result = _basic_chignolin_config()
        output = result["output"]
        assert "extend_mode" in output
        assert output["extend_mode"] == "auto"

    def test_extend_follows_resume_in_key_order(self):
        """extend and extend_mode come immediately after resume in key order."""
        from gareus.config import _basic_chignolin_config
        result = _basic_chignolin_config()
        keys = list(result["output"].keys())
        resume_idx = keys.index("resume")
        extend_idx = keys.index("extend")
        extend_mode_idx = keys.index("extend_mode")
        assert extend_idx == resume_idx + 1
        assert extend_mode_idx == resume_idx + 2
