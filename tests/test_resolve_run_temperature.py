import json
import math

from gareus.io import resolve_run_temperature_k


def test_falls_back_to_run_manifest_resolved_args(tmp_path):
    (tmp_path / "gareus_metadata.json").write_text(json.dumps({"gamd_enabled": True}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"temperature_k": 300.0}}), encoding="utf-8"
    )
    assert math.isclose(resolve_run_temperature_k(tmp_path), 300.0)


def test_falls_back_to_run_manifest_method_settings(tmp_path):
    (tmp_path / "gareus_metadata.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {}, "method_settings": {"temperature_k": 310.0}}), encoding="utf-8"
    )
    assert math.isclose(resolve_run_temperature_k(tmp_path), 310.0)


def test_gareus_metadata_temperature_still_wins_if_present(tmp_path):
    (tmp_path / "gareus_metadata.json").write_text(json.dumps({"temperature_K": 320.0}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"temperature_k": 300.0}}), encoding="utf-8"
    )
    assert math.isclose(resolve_run_temperature_k(tmp_path), 320.0)


def test_returns_none_when_no_temperature_anywhere(tmp_path):
    (tmp_path / "gareus_metadata.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({"resolved_args": {}}), encoding="utf-8")
    assert resolve_run_temperature_k(tmp_path) is None


def test_returns_none_when_files_missing(tmp_path):
    assert resolve_run_temperature_k(tmp_path) is None


def test_ignores_non_positive_or_non_finite_values(tmp_path):
    (tmp_path / "gareus_metadata.json").write_text(json.dumps({"temperature_K": -1.0}), encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"temperature_k": "nan"}}), encoding="utf-8"
    )
    assert resolve_run_temperature_k(tmp_path) is None
