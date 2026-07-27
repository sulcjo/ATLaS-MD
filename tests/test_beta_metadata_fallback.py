import json
import math
from pathlib import Path

from gareus.adaptive_production import _metadata_beta_1_over_kj_mol


def _expected_beta(temp_k: float) -> float:
    return 1.0 / (8.314462618e-3 * temp_k)


def test_falls_back_to_run_manifest_resolved_args(tmp_path):
    run_dir = tmp_path / "final" / "baseline"
    run_dir.mkdir(parents=True)
    (run_dir / "gareus_metadata.json").write_text(json.dumps({"gamd_enabled": True}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"temperature_k": 300.0}}), encoding="utf-8"
    )
    beta = _metadata_beta_1_over_kj_mol(run_dir)
    assert beta is not None
    assert math.isclose(beta, _expected_beta(300.0), rel_tol=1e-9)


def test_falls_back_to_run_manifest_method_settings(tmp_path):
    run_dir = tmp_path / "epoch_000"
    run_dir.mkdir(parents=True)
    (run_dir / "gareus_metadata.json").write_text(json.dumps({}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {}, "method_settings": {"temperature_k": 310.0}}), encoding="utf-8"
    )
    beta = _metadata_beta_1_over_kj_mol(run_dir)
    assert beta is not None
    assert math.isclose(beta, _expected_beta(310.0), rel_tol=1e-9)


def test_gareus_metadata_temperature_still_wins_if_present(tmp_path):
    run_dir = tmp_path / "final" / "baseline"
    run_dir.mkdir(parents=True)
    (run_dir / "gareus_metadata.json").write_text(json.dumps({"temperature_K": 320.0}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"temperature_k": 300.0}}), encoding="utf-8"
    )
    beta = _metadata_beta_1_over_kj_mol(run_dir)
    assert beta is not None
    assert math.isclose(beta, _expected_beta(320.0), rel_tol=1e-9)


def test_returns_none_when_no_temperature_anywhere(tmp_path):
    run_dir = tmp_path / "final" / "baseline"
    run_dir.mkdir(parents=True)
    (run_dir / "gareus_metadata.json").write_text(json.dumps({}), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"resolved_args": {}}), encoding="utf-8")
    assert _metadata_beta_1_over_kj_mol(run_dir) is None


def test_returns_none_when_files_missing(tmp_path):
    run_dir = tmp_path / "final" / "baseline"
    run_dir.mkdir(parents=True)
    assert _metadata_beta_1_over_kj_mol(run_dir) is None
