"""A run manifest that exists only as an update_run_manifest() skeleton must still get its
start-of-run provenance (method_settings with the kernel identity, start_time_utc, resolved_args)
when production starts in that directory. chignolin_8/epoch_000 (2026-09-22) ran three jobs on a
10-key skeleton because run_gareus tested file existence, not completeness."""
from __future__ import annotations

import json
import types

from gareus.kernel_identity import EXCHANGE_ENERGY_VERSION, RESIDUAL_EVALUATOR_VERSION
from gareus.provenance import ensure_run_manifest_initialized, run_manifest_is_skeleton, update_run_manifest


def _args(**over):
    a = types.SimpleNamespace(resume=True, secondary_cv="residual-torsion-pc", temperature_k=300.0, seq="GYDPETGTWG",
                              state_gamd_lambdas=[0.0, 1.0], config="", _config_values={})
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_skeleton_detection():
    assert run_manifest_is_skeleton({}) is True
    assert run_manifest_is_skeleton({"schema_version": "1.0", "run_id": "x", "status": "partial"}) is True
    assert run_manifest_is_skeleton({"start_time_utc": "2026-09-22T00:00:00+00:00", "method_settings": {"a": 1}}) is False


def test_skeleton_gets_initialized_and_keeps_its_patches(tmp_path):
    update_run_manifest(tmp_path, {"method_settings": {"state_gamd_lambdas": [0.0, 0.5, 1.0]}, "openmm_runtime": {"x": 1}})
    before = json.loads((tmp_path / "run_manifest.json").read_text())
    assert "start_time_utc" not in before and set(before["method_settings"]) == {"state_gamd_lambdas"}

    changed = ensure_run_manifest_initialized(_args(), tmp_path)

    after = json.loads((tmp_path / "run_manifest.json").read_text())
    assert changed is True
    assert after["run_id"] == before["run_id"]                       # identity of the directory is preserved
    assert after["start_time_utc"]
    assert after["method_settings"]["state_gamd_lambdas"] == [0.0, 0.5, 1.0]   # the patch wins over the recomputed value
    assert after["method_settings"]["exchange_energy_version"] == EXCHANGE_ENERGY_VERSION
    assert after["method_settings"]["cv_evaluator_version"] == RESIDUAL_EVALUATOR_VERSION
    assert after["openmm_runtime"] == {"x": 1}
    assert after["resolved_args"]["temperature_k"] == 300.0


def test_full_manifest_is_left_alone(tmp_path):
    from gareus.provenance import initialize_run_manifest
    initialize_run_manifest(_args(resume=False), tmp_path, argv=[])
    before = (tmp_path / "run_manifest.json").read_text()
    assert ensure_run_manifest_initialized(_args(), tmp_path) is False
    assert (tmp_path / "run_manifest.json").read_text() == before


def test_missing_manifest_is_created(tmp_path):
    assert ensure_run_manifest_initialized(_args(resume=False), tmp_path) is True
    m = json.loads((tmp_path / "run_manifest.json").read_text())
    assert m["start_time_utc"] and m["method_settings"]["exchange_energy_version"] == EXCHANGE_ENERGY_VERSION
