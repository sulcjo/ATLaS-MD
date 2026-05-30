"""Tests for find_resume_equil_state_path — pure path logic, no OpenMM needed."""

from pathlib import Path
import pytest
from gareus.checkpoints import find_resume_equil_state_path, equilibration_state_xml_path


def _make(tmp_path: Path, *parts: str) -> Path:
    p = tmp_path.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("dummy")
    return p


def test_returns_none_when_nothing_exists(tmp_path):
    assert find_resume_equil_state_path(tmp_path) is None


def test_primary_preferred(tmp_path):
    primary = _make(tmp_path, "03_npt_equilibrated_state.xml")
    _make(tmp_path, "adaptive_production", "global_shared_gamd_setup", "shared_gamd_setup_state.xml")
    assert find_resume_equil_state_path(tmp_path) == primary


def test_global_shared_gamd_over_epoch(tmp_path):
    global_state = _make(tmp_path, "adaptive_production", "global_shared_gamd_setup", "shared_gamd_setup_state.xml")
    _make(tmp_path, "adaptive_production", "epoch_000", "baseline", "shared_gamd_setup_state.xml")
    assert find_resume_equil_state_path(tmp_path) == global_state


def test_latest_epoch_baseline(tmp_path):
    _make(tmp_path, "adaptive_production", "epoch_000", "baseline", "shared_gamd_setup_state.xml")
    latest = _make(tmp_path, "adaptive_production", "epoch_002", "baseline", "shared_gamd_setup_state.xml")
    _make(tmp_path, "adaptive_production", "epoch_001", "baseline", "shared_gamd_setup_state.xml")
    assert find_resume_equil_state_path(tmp_path) == latest


def test_feedback_fallback(tmp_path):
    fb = _make(tmp_path, "adaptive_feedback_round_01", "shared_gamd_setup_state.xml")
    assert find_resume_equil_state_path(tmp_path) == fb


def test_equilibration_state_xml_path_name():
    p = equilibration_state_xml_path(Path("/some/run"))
    assert p.name == "03_npt_equilibrated_state.xml"
