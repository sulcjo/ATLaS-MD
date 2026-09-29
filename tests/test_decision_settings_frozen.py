"""Spec P8: a campaign's adaptive decision rules are frozen at its first job."""
import json

import gareus.adaptive_production as ap
from gareus.adaptive_production import AdaptiveDecisionPolicy
from gareus.cli import parse_args


def test_first_job_records_every_decision_field(tmp_path):
    policy, record = ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy(target_overlap=0.3))
    saved = json.loads((tmp_path / ap.DECISION_SETTINGS_FILENAME).read_text())
    assert set(saved["settings"]) == set(ap.DECISION_SETTINGS_FIELDS)
    assert saved["settings"]["target_overlap"] == 0.3 and policy.target_overlap == 0.3
    assert record["ignored_job_values"] == {}


def test_a_resume_with_different_flags_keeps_the_recorded_rules(tmp_path, capsys):
    ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy(target_overlap=0.3, min_samples_for_add=50))
    policy, record = ap._resolve_decision_settings(
        tmp_path, AdaptiveDecisionPolicy(target_overlap=0.5, min_samples_for_add=50))
    assert policy.target_overlap == 0.3
    assert record["ignored_job_values"] == {"target_overlap": {"recorded": 0.3, "this_job": 0.5}}
    assert "--ap-decision-settings-override" in capsys.readouterr().out


def test_override_replaces_the_record(tmp_path):
    ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy(target_overlap=0.3))
    policy, _ = ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy(target_overlap=0.5), override=True)
    assert policy.target_overlap == 0.5
    policy, _ = ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy(target_overlap=0.9))
    assert policy.target_overlap == 0.5


def test_budgets_are_not_frozen(tmp_path):
    ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy(max_replicas_budget=100, epoch_step_budget=10))
    policy, _ = ap._resolve_decision_settings(
        tmp_path, AdaptiveDecisionPolicy(max_replicas_budget=200, epoch_step_budget=20))
    assert policy.max_replicas_budget == 200 and policy.epoch_step_budget == 20


def test_a_field_added_later_is_recorded_from_the_first_job_that_sees_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "DECISION_SETTINGS_FIELDS", ("target_overlap",))
    ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy(target_overlap=0.3))
    monkeypatch.setattr(ap, "DECISION_SETTINGS_FIELDS", ("target_overlap", "min_samples_for_add"))
    policy, _ = ap._resolve_decision_settings(
        tmp_path, AdaptiveDecisionPolicy(target_overlap=0.7, min_samples_for_add=77))
    saved = json.loads((tmp_path / ap.DECISION_SETTINGS_FILENAME).read_text())["settings"]
    assert saved == {"target_overlap": 0.3, "min_samples_for_add": 77}
    assert policy.target_overlap == 0.3 and policy.min_samples_for_add == 77


def test_the_manifest_mirror_holds_one_key(tmp_path):
    _, record = ap._resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy())
    ap._record_decision_settings_in_manifest(tmp_path, record)
    ap._record_decision_settings_in_manifest(tmp_path, record)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    entry = manifest["method_settings"]["adaptive_decision_settings"]
    assert entry["settings"] == record["settings"] and entry["file"] == ap.DECISION_SETTINGS_FILENAME


def test_cli_flag_reaches_the_driver_name(tmp_path):
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path), "--ap-decision-settings-override"])
    assert args.adaptive_production_decision_settings_override is True
    assert parse_args(["--seq", "GYDPETGTWG", "--out", str(tmp_path)]).adaptive_production_decision_settings_override is False


def test_every_frozen_field_is_a_policy_field():
    fields = set(AdaptiveDecisionPolicy.__dataclass_fields__)
    assert set(ap.DECISION_SETTINGS_FIELDS) <= fields
