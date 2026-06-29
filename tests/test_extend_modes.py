"""Tests for --extend / extend_mode per-mode dispatch.

Tests _detect_extend_mode, _resolve_and_apply_extend_mode, and the per-mode
arg mutations (frozen, adaptive, topup, regular, auto).

No OpenMM, PeptideBuilder, or gamd-openmm imports.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> types.SimpleNamespace:
    """Create a minimal args namespace for testing."""
    defaults = {
        "extend": True,
        "extend_mode": "auto",
        "ap_extend_rounds": 1,
        "ap_extend_steps": 0,
        "adaptive_production_epochs": 5,
        "adaptive_production_final_quality_extension_rounds": 0,
        "adaptive_production_final_quality_extension_steps": 0,
        "adaptive_production_topup_only": False,
        "resume": False,
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Test group 1: _detect_extend_mode
# ---------------------------------------------------------------------------

class TestDetectExtendMode:
    def _fn(self):
        from gareus.adaptive_production import _detect_extend_mode
        return _detect_extend_mode

    def test_no_adaptive_dir_returns_regular(self, tmp_path):
        """No adaptive_production/ directory → 'regular'."""
        detect = self._fn()
        assert detect(tmp_path) == "regular"

    def test_has_dir_no_summary_returns_adaptive(self, tmp_path):
        """adaptive_production/ exists but no summary JSON → 'adaptive'."""
        (tmp_path / "adaptive_production").mkdir()
        detect = self._fn()
        assert detect(tmp_path) == "adaptive"

    def test_status_completed_returns_frozen(self, tmp_path):
        """Summary with status='completed' → 'frozen'."""
        ap_dir = tmp_path / "adaptive_production"
        ap_dir.mkdir()
        (ap_dir / "adaptive_production_driver_summary.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )
        detect = self._fn()
        assert detect(tmp_path) == "frozen"

    def test_status_interrupted_returns_adaptive(self, tmp_path):
        """Summary with status='interrupted_after_checkpoint' → 'adaptive'."""
        ap_dir = tmp_path / "adaptive_production"
        ap_dir.mkdir()
        (ap_dir / "adaptive_production_driver_summary.json").write_text(
            json.dumps({"status": "interrupted_after_checkpoint"}), encoding="utf-8"
        )
        detect = self._fn()
        assert detect(tmp_path) == "adaptive"

    def test_corrupt_json_returns_adaptive(self, tmp_path):
        """Corrupt JSON in summary → 'adaptive' (no exception raised)."""
        ap_dir = tmp_path / "adaptive_production"
        ap_dir.mkdir()
        (ap_dir / "adaptive_production_driver_summary.json").write_text(
            "{not valid json}", encoding="utf-8"
        )
        detect = self._fn()
        assert detect(tmp_path) == "adaptive"

    def test_missing_status_key_returns_adaptive(self, tmp_path):
        """Summary with no 'status' key → 'adaptive'."""
        ap_dir = tmp_path / "adaptive_production"
        ap_dir.mkdir()
        (ap_dir / "adaptive_production_driver_summary.json").write_text(
            json.dumps({"schema_version": "v1"}), encoding="utf-8"
        )
        detect = self._fn()
        assert detect(tmp_path) == "adaptive"


# ---------------------------------------------------------------------------
# Test group 2: frozen mode args
# ---------------------------------------------------------------------------

class TestFrozenModeArgs:
    def _resolve(self, args, out_dir):
        from gareus.adaptive_production import _resolve_and_apply_extend_mode
        return _resolve_and_apply_extend_mode(args, out_dir)

    def test_frozen_sets_extension_rounds(self, tmp_path):
        """extend=True, extend_mode=frozen, ap_extend_rounds=2 → extension_rounds == 2."""
        args = _make_args(extend=True, extend_mode="frozen", ap_extend_rounds=2)
        mode = self._resolve(args, tmp_path)
        assert mode == "frozen"
        assert args.adaptive_production_final_quality_extension_rounds == 2

    def test_frozen_defaults_to_1_when_ap_extend_rounds_is_1(self, tmp_path):
        """Default ap_extend_rounds=1 → extension_rounds == 1."""
        args = _make_args(extend=True, extend_mode="frozen", ap_extend_rounds=1)
        self._resolve(args, tmp_path)
        assert args.adaptive_production_final_quality_extension_rounds == 1

    def test_frozen_sets_extension_steps_when_nonzero(self, tmp_path):
        """ap_extend_steps > 0 → extension_steps updated."""
        args = _make_args(extend=True, extend_mode="frozen", ap_extend_rounds=1, ap_extend_steps=50000)
        self._resolve(args, tmp_path)
        assert args.adaptive_production_final_quality_extension_steps == 50000

    def test_frozen_does_not_set_steps_when_zero(self, tmp_path):
        """ap_extend_steps == 0 → extension_steps NOT updated."""
        args = _make_args(extend=True, extend_mode="frozen", ap_extend_rounds=1, ap_extend_steps=0)
        self._resolve(args, tmp_path)
        assert args.adaptive_production_final_quality_extension_steps == 0

    def test_frozen_does_not_change_epochs(self, tmp_path):
        """frozen mode must not alter adaptive_production_epochs."""
        args = _make_args(extend=True, extend_mode="frozen", ap_extend_rounds=2,
                          adaptive_production_epochs=5)
        self._resolve(args, tmp_path)
        assert args.adaptive_production_epochs == 5


# ---------------------------------------------------------------------------
# Test group 3: adaptive mode args
# ---------------------------------------------------------------------------

class TestAdaptiveModeArgs:
    def _resolve(self, args, out_dir):
        from gareus.adaptive_production import _resolve_and_apply_extend_mode
        return _resolve_and_apply_extend_mode(args, out_dir)

    def test_adaptive_adds_extra_epochs(self, tmp_path):
        """extend=True, extend_mode=adaptive, epochs=5, ap_extend_rounds=3 → epochs == 8."""
        args = _make_args(extend=True, extend_mode="adaptive",
                          adaptive_production_epochs=5, ap_extend_rounds=3)
        mode = self._resolve(args, tmp_path)
        assert mode == "adaptive"
        assert args.adaptive_production_epochs == 8

    def test_adaptive_adds_at_least_1_epoch_when_rounds_0(self, tmp_path):
        """ap_extend_rounds=0 → still adds at least 1 epoch."""
        args = _make_args(extend=True, extend_mode="adaptive",
                          adaptive_production_epochs=5, ap_extend_rounds=0)
        self._resolve(args, tmp_path)
        assert args.adaptive_production_epochs == 6

    def test_adaptive_does_not_change_extension_rounds(self, tmp_path):
        """adaptive mode must not set final_quality_extension_rounds."""
        args = _make_args(extend=True, extend_mode="adaptive",
                          adaptive_production_epochs=5, ap_extend_rounds=2)
        self._resolve(args, tmp_path)
        assert args.adaptive_production_final_quality_extension_rounds == 0

    def test_adaptive_uses_epochs_completed_when_less_than_configured(self, tmp_path):
        """configured=5, completed=3, extra=2 → max(3,5)+2 = 7, not 5+2=7 (same but via completed path)."""
        ap_dir = tmp_path / "adaptive_production"
        ap_dir.mkdir()
        (ap_dir / "adaptive_production_driver_summary.json").write_text(
            json.dumps({"epochs_completed": 3, "status": "interrupted_after_checkpoint"}),
            encoding="utf-8",
        )
        args = _make_args(extend=True, extend_mode="adaptive",
                          adaptive_production_epochs=5, ap_extend_rounds=2)
        mode = self._resolve(args, tmp_path)
        assert mode == "adaptive"
        # effective_base = max(3, 5) = 5; 5 + 2 = 7
        assert args.adaptive_production_epochs == 7

    def test_adaptive_uses_epochs_completed_when_greater_than_configured(self, tmp_path):
        """configured=5, completed=6 (already ran extra epoch), extra=1 → max(6,5)+1 = 7, not 5+1=6."""
        ap_dir = tmp_path / "adaptive_production"
        ap_dir.mkdir()
        (ap_dir / "adaptive_production_driver_summary.json").write_text(
            json.dumps({"epochs_completed": 6, "status": "interrupted_after_checkpoint"}),
            encoding="utf-8",
        )
        args = _make_args(extend=True, extend_mode="adaptive",
                          adaptive_production_epochs=5, ap_extend_rounds=1)
        mode = self._resolve(args, tmp_path)
        assert mode == "adaptive"
        # effective_base = max(6, 5) = 6; 6 + 1 = 7
        assert args.adaptive_production_epochs == 7


# ---------------------------------------------------------------------------
# Test group 4: topup mode args
# ---------------------------------------------------------------------------

class TestTopupModeArgs:
    def _resolve(self, args, out_dir):
        from gareus.adaptive_production import _resolve_and_apply_extend_mode
        return _resolve_and_apply_extend_mode(args, out_dir)

    def test_topup_sets_topup_only_flag(self, tmp_path):
        """extend=True, extend_mode=topup → adaptive_production_topup_only == True."""
        args = _make_args(extend=True, extend_mode="topup")
        mode = self._resolve(args, tmp_path)
        assert mode == "topup"
        assert args.adaptive_production_topup_only is True


# ---------------------------------------------------------------------------
# Test group 5: regular mode args
# ---------------------------------------------------------------------------

class TestRegularModeArgs:
    def _resolve(self, args, out_dir):
        from gareus.adaptive_production import _resolve_and_apply_extend_mode
        return _resolve_and_apply_extend_mode(args, out_dir)

    def test_regular_sets_resume_true(self, tmp_path):
        """extend=True, extend_mode=regular → args.resume == True."""
        args = _make_args(extend=True, extend_mode="regular", resume=False)
        mode = self._resolve(args, tmp_path)
        assert mode == "regular"
        assert args.resume is True


# ---------------------------------------------------------------------------
# Test group 6: auto mode detection and dispatch
# ---------------------------------------------------------------------------

class TestAutoModeDetection:
    def _resolve(self, args, out_dir):
        from gareus.adaptive_production import _resolve_and_apply_extend_mode
        return _resolve_and_apply_extend_mode(args, out_dir)

    def test_auto_no_adaptive_dir_resolves_regular(self, tmp_path):
        """auto → regular when no adaptive_production/ exists."""
        args = _make_args(extend=True, extend_mode="auto")
        mode = self._resolve(args, tmp_path)
        assert mode == "regular"
        assert args.extend_mode == "regular"
        assert args.resume is True

    def test_auto_interrupted_run_resolves_adaptive(self, tmp_path):
        """auto → adaptive when summary has interrupted status."""
        ap_dir = tmp_path / "adaptive_production"
        ap_dir.mkdir()
        (ap_dir / "adaptive_production_driver_summary.json").write_text(
            json.dumps({"status": "interrupted_after_checkpoint"}), encoding="utf-8"
        )
        args = _make_args(extend=True, extend_mode="auto", adaptive_production_epochs=5)
        mode = self._resolve(args, tmp_path)
        assert mode == "adaptive"
        assert args.extend_mode == "adaptive"
        assert args.adaptive_production_epochs == 6  # default +1

    def test_auto_completed_run_resolves_frozen(self, tmp_path):
        """auto → frozen when summary has completed status."""
        ap_dir = tmp_path / "adaptive_production"
        ap_dir.mkdir()
        (ap_dir / "adaptive_production_driver_summary.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )
        args = _make_args(extend=True, extend_mode="auto", ap_extend_rounds=2)
        mode = self._resolve(args, tmp_path)
        assert mode == "frozen"
        assert args.extend_mode == "frozen"
        assert args.adaptive_production_final_quality_extension_rounds == 2

    def test_auto_no_dir_does_not_touch_adaptive_epochs(self, tmp_path):
        """auto → regular should NOT modify adaptive_production_epochs."""
        args = _make_args(extend=True, extend_mode="auto", adaptive_production_epochs=5)
        self._resolve(args, tmp_path)
        assert args.adaptive_production_epochs == 5


# ---------------------------------------------------------------------------
# Test group 7: no-op when extend=False
# ---------------------------------------------------------------------------

class TestNoOpWhenExtendFalse:
    def _resolve(self, args, out_dir):
        from gareus.adaptive_production import _resolve_and_apply_extend_mode
        return _resolve_and_apply_extend_mode(args, out_dir)

    def test_extend_false_is_noop(self, tmp_path):
        """When extend=False, helper returns extend_mode unchanged and mutates nothing."""
        args = _make_args(extend=False, extend_mode="auto",
                          adaptive_production_epochs=5,
                          adaptive_production_final_quality_extension_rounds=0)
        self._resolve(args, tmp_path)
        assert args.adaptive_production_epochs == 5
        assert args.adaptive_production_final_quality_extension_rounds == 0
        assert args.resume is False


# ---------------------------------------------------------------------------
# Test group 8: CLI arg parsing for new --ap-extend-* args
# ---------------------------------------------------------------------------

class TestApExtendArgsParsing:
    def _parse(self, argv: list[str]):
        from gareus.cli import parse_args
        base = ["--seq", "AA", "--run-mode", "cmd"]
        return parse_args(base + argv)

    def test_ap_extend_rounds_default(self):
        """--ap-extend-rounds defaults to 1."""
        args = self._parse([])
        assert args.ap_extend_rounds == 1

    def test_ap_extend_steps_default(self):
        """--ap-extend-steps defaults to 0."""
        args = self._parse([])
        assert args.ap_extend_steps == 0

    def test_ap_extend_rounds_custom(self):
        """--ap-extend-rounds 3 → ap_extend_rounds == 3."""
        args = self._parse(["--ap-extend-rounds", "3"])
        assert args.ap_extend_rounds == 3

    def test_ap_extend_steps_custom(self):
        """--ap-extend-steps 50000 → ap_extend_steps == 50000."""
        args = self._parse(["--ap-extend-steps", "50000"])
        assert args.ap_extend_steps == 50000
