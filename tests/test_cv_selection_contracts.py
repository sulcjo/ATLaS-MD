"""T00 acceptance: the selector's contracts refuse ambiguous scientific inputs.

Every case here is a rejection that later tasks depend on. A contract that
accepts a mislabelled artifact silently hands a wrong number to T05/T06.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from gareus.correctness._io import IntegrityError, json_bytes
from gareus.cv_selection import contracts as C


# --------------------------------------------------------------------------
# Fixtures: the smallest artifacts that are actually valid.
# --------------------------------------------------------------------------

def _feature_rows(n_torsions: int = 2) -> list[dict]:
    rows: list[dict] = []
    for t in range(n_torsions):
        name = "phi" if t % 2 == 0 else "psi"
        for trig in ("sin", "cos"):
            rows.append({
                "index": len(rows),
                "name": f"{name}-{t + 1}-{trig}",
                "torsion_name": f"{name}-{t + 1}",
                "residue_index": t + 1,
                "atom_indices": [4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3],
                "trig": trig,
                "dihedral_sign_convention": "negated",
            })
    return rows


@pytest.fixture
def feature_schema() -> C.FeatureSchema:
    return C.FeatureSchema.from_mapping({
        "schema": C.FEATURE_SCHEMA_VERSION,
        "topology_sha256": "a" * 64,
        "features": _feature_rows(),
    })


def _component(index: int) -> dict:
    return {
        "component_index": index,
        "singular_value": 2.0 / index,
        "eigenvalue_tie_flagged": False,
        "right_singular_vector": [0.5, 0.5, 0.5, -0.5],
        "residual_mean": [0.0, 0.0, 0.0, 0.0],
        "regression_coefficients": [[0.0] * 4, [0.1] * 4, [0.01] * 4],
        "primary_mean": 0.2,
        "primary_std": 0.05,
        "projection_mean": 0.0,
        "projection_std": 1.0,
    }


@pytest.fixture
def candidate_set(feature_schema) -> C.CandidateSet:
    return C.CandidateSet.from_mapping({
        "schema": C.CANDIDATE_SET_VERSION,
        "kind": C.CANDIDATE_KIND_QUADRATIC_RESIDUAL,
        "feature_schema_sha256": feature_schema.sha256,
        "physical_system_sha256": "b" * 64,
        "training_rows_sha256": "c" * 64,
        "primary_definition": {
            "kind": "nonlocal-contact-fraction",
            "units": "dimensionless",
            "definition": {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0,
                           "min_sequence_separation": 4},
        },
        "components": [_component(j) for j in (1, 2)],
    })


def _primary_observable(quantile: float, observable: str) -> dict:
    return {
        "observable_id": f"cdf::{observable}::q{quantile:g}",
        "kind": "cdf_probability",
        "source_observable": observable,
        "discovery_quantile": quantile,
        "halfwidth_tolerance": 0.02,
    }


@pytest.fixture
def observable_panel() -> C.ObservablePanel:
    return C.ObservablePanel.from_mapping({
        "schema": C.OBSERVABLE_PANEL_VERSION,
        "primary": [_primary_observable(q, obs)
                    for obs in ("rg_nm", "end_to_end_nm")
                    for q in (0.2, 0.4, 0.6, 0.8)],
        "cross_protocol_tolerance": 0.02,
        "diagnostic_partition_centers": 16,
        "diagnostic_histogram_bins": 20,
        "novelty_mass_tolerance": 0.02,
    })


def _protocol_mapping(**overrides) -> dict:
    base = {
        "schema": C.PROTOCOL_VERSION,
        "study_id": "chignolin-fixed-primary-001",
        "search_mode": C.SearchMode.FIXED_PRIMARY.value,
        "target": {"sequence": "GYDPETGTWG", "temperature_k": 300.0, "ensemble": "NPT",
                   "pressure_bar": 1.0, "physical_system_sha256": "b" * 64},
        "blindness": {"generator_preset": "broad", "require_native_blind_allowlist": True,
                      "historical_data_role": "exploratory_only"},
        "discovery": {"candidate_set_sha256": None, "feature_schema_sha256": None},
        "sampling": {"layout": None, "lambda_rungs": None, "burn_in_ticks": None,
                     "measurement_ticks": None, "campaigns_per_arm": None},
        "resources": {"gpus": 4, "cpu_thread_cap": 192, "max_replicas": 128,
                      "gpu_hours_per_campaign": None, "total_gpu_hour_ceiling": None},
        "evaluation": {"observable_panel_sha256": None},
        "uncertainty": {"policy": None, "calibration_certificate_sha256": None},
        "decision": {"provisional_choice_arm_id": None, "interval_family_size": None,
                     "advantage_claim_enabled": False},
    }
    base.update(overrides)
    return base


def _resolved_protocol_mapping(feature_schema, candidate_set, observable_panel) -> dict:
    return _protocol_mapping(
        discovery={"candidate_set_sha256": candidate_set.sha256,
                   "feature_schema_sha256": feature_schema.sha256},
        sampling={"layout": "grid24-v1", "lambda_rungs": 4, "burn_in_ticks": 1000,
                  "measurement_ticks": 20000, "campaigns_per_arm": 3},
        evaluation={"observable_panel_sha256": observable_panel.sha256},
        uncertainty={"policy": "joint-batch-v1", "calibration_certificate_sha256": "9" * 64},
        resources={"gpus": 4, "cpu_thread_cap": 192, "max_replicas": 128,
                   "gpu_hours_per_campaign": 8.0, "total_gpu_hour_ceiling": 288.0},
    )


@pytest.fixture
def protocol() -> C.ProtocolSpec:
    return C.ProtocolSpec.from_mapping(_protocol_mapping())


# --------------------------------------------------------------------------
# Feature identity
# --------------------------------------------------------------------------

def test_feature_schema_round_trips_and_digest_is_stable(feature_schema):
    reloaded = C.FeatureSchema.from_mapping(feature_schema.to_mapping())
    assert reloaded.sha256 == feature_schema.sha256
    assert reloaded.to_mapping() == feature_schema.to_mapping()


def test_feature_schema_digest_changes_when_atom_mapping_changes(feature_schema):
    payload = feature_schema.to_mapping()
    payload["features"][0]["atom_indices"] = [99, 1, 2, 3]
    assert C.FeatureSchema.from_mapping(payload).sha256 != feature_schema.sha256


def test_feature_schema_digest_changes_when_only_trig_order_changes(feature_schema):
    """Vector width is equal; the handoff's blocked sin/cos order is not this one."""
    payload = feature_schema.to_mapping()
    payload["features"][0]["trig"] = "cos"
    payload["features"][1]["trig"] = "sin"
    payload["features"][0]["name"] = "phi-1-cos"
    payload["features"][1]["name"] = "phi-1-sin"
    assert C.FeatureSchema.from_mapping(payload).sha256 != feature_schema.sha256


def test_feature_schema_rejects_non_contiguous_index(feature_schema):
    payload = feature_schema.to_mapping()
    payload["features"][1]["index"] = 7
    with pytest.raises(IntegrityError) as excinfo:
        C.FeatureSchema.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.FEATURE_INDEX_NOT_CONTIGUOUS


def test_feature_schema_rejects_unknown_field(feature_schema):
    payload = feature_schema.to_mapping()
    payload["features"][0]["cv_hint"] = "native"
    with pytest.raises(IntegrityError) as excinfo:
        C.FeatureSchema.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.UNKNOWN_FIELD


def test_feature_schema_rejects_unknown_trig(feature_schema):
    payload = feature_schema.to_mapping()
    payload["features"][0]["trig"] = "tan"
    with pytest.raises(IntegrityError) as excinfo:
        C.FeatureSchema.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.INVALID_ENUM


def test_feature_schema_rejects_short_atom_quadruplet(feature_schema):
    payload = feature_schema.to_mapping()
    payload["features"][0]["atom_indices"] = [0, 1, 2]
    with pytest.raises(IntegrityError) as excinfo:
        C.FeatureSchema.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.INVALID_ATOM_QUADRUPLET


def test_feature_schema_rejects_unknown_schema_version(feature_schema):
    payload = feature_schema.to_mapping()
    payload["schema"] = "atlas-cv-selection-feature-schema-v2"
    with pytest.raises(IntegrityError) as excinfo:
        C.FeatureSchema.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.UNKNOWN_SCHEMA_VERSION


# --------------------------------------------------------------------------
# Candidate set
# --------------------------------------------------------------------------

def test_candidate_set_binds_to_its_feature_schema(candidate_set, feature_schema):
    assert candidate_set.feature_schema_sha256 == feature_schema.sha256
    C.require_feature_binding(candidate_set, feature_schema)
    payload = candidate_set.to_mapping()
    payload["feature_schema_sha256"] = "d" * 64
    with pytest.raises(IntegrityError) as excinfo:
        C.require_feature_binding(C.CandidateSet.from_mapping(payload), feature_schema)
    assert excinfo.value.reason is C.ReasonCode.FEATURE_IDENTITY_MISMATCH


def test_candidate_set_rejects_width_mismatch_against_feature_schema(candidate_set, feature_schema):
    payload = candidate_set.to_mapping()
    payload["components"][0]["right_singular_vector"] = [1.0, 0.0, 0.0]
    payload["components"][0]["residual_mean"] = [0.0, 0.0, 0.0]
    payload["components"][0]["regression_coefficients"] = [[0.0] * 3, [0.1] * 3, [0.01] * 3]
    with pytest.raises(IntegrityError) as excinfo:
        C.require_feature_binding(C.CandidateSet.from_mapping(payload), feature_schema)
    assert excinfo.value.reason is C.ReasonCode.FEATURE_WIDTH_MISMATCH


def test_candidate_set_rejects_ragged_component_arrays(candidate_set):
    payload = candidate_set.to_mapping()
    payload["components"][0]["residual_mean"] = [0.0, 0.0, 0.0]
    with pytest.raises(IntegrityError) as excinfo:
        C.CandidateSet.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.FEATURE_WIDTH_MISMATCH


def test_candidate_set_rejects_duplicate_component_index(candidate_set):
    payload = candidate_set.to_mapping()
    payload["components"][1]["component_index"] = 1
    with pytest.raises(IntegrityError) as excinfo:
        C.CandidateSet.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.DUPLICATE_COMPONENT_INDEX


def test_candidate_set_rejects_out_of_range_component_index(candidate_set):
    payload = candidate_set.to_mapping()
    payload["components"][0]["component_index"] = 0
    with pytest.raises(IntegrityError) as excinfo:
        C.CandidateSet.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.COMPONENT_INDEX_OUT_OF_RANGE


def test_candidate_set_rejects_nonfinite_coefficient(candidate_set):
    payload = candidate_set.to_mapping()
    payload["components"][0]["right_singular_vector"][0] = float("nan")
    with pytest.raises(IntegrityError):
        C.CandidateSet.from_mapping(payload)


def test_candidate_set_rejects_zero_scaling(candidate_set):
    payload = candidate_set.to_mapping()
    payload["components"][0]["projection_std"] = 0.0
    with pytest.raises(IntegrityError) as excinfo:
        C.CandidateSet.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.NONPOSITIVE_SCALE


def test_candidate_set_rejects_path_identity_in_primary_definition(candidate_set):
    """Identity is contents; a filename is not a CV definition."""
    payload = candidate_set.to_mapping()
    payload["primary_definition"]["definition"]["model_path"] = "models/cv1.json"
    with pytest.raises(IntegrityError):
        C.CandidateSet.from_mapping(payload)


def test_candidate_set_keeps_components_individual_not_mixed(candidate_set):
    """A count-of-leading-PCs field would reintroduce the legacy ambiguity."""
    payload = candidate_set.to_mapping()
    payload["components"][0]["component_count"] = 2
    with pytest.raises(IntegrityError) as excinfo:
        C.CandidateSet.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.UNKNOWN_FIELD


def test_candidate_set_regression_keeps_all_three_polynomial_rows(candidate_set):
    payload = candidate_set.to_mapping()
    payload["components"][0]["regression_coefficients"] = [[0.0] * 4, [0.1] * 4]
    with pytest.raises(IntegrityError) as excinfo:
        C.CandidateSet.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.INCOMPLETE_REGRESSION


# --------------------------------------------------------------------------
# Observable panel
# --------------------------------------------------------------------------

def test_observable_panel_requires_the_frozen_primary_count(observable_panel):
    assert len(observable_panel.primary) == 8
    payload = observable_panel.to_mapping()
    payload["primary"] = payload["primary"][:7]
    with pytest.raises(IntegrityError) as excinfo:
        C.ObservablePanel.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.PRIMARY_PANEL_SIZE


def test_observable_panel_rejects_duplicate_observable_id(observable_panel):
    payload = observable_panel.to_mapping()
    payload["primary"][1]["observable_id"] = payload["primary"][0]["observable_id"]
    with pytest.raises(IntegrityError) as excinfo:
        C.ObservablePanel.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.DUPLICATE_OBSERVABLE_ID


def test_observable_panel_rejects_nonpositive_tolerance(observable_panel):
    payload = observable_panel.to_mapping()
    payload["primary"][0]["halfwidth_tolerance"] = 0.0
    with pytest.raises(IntegrityError) as excinfo:
        C.ObservablePanel.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.TOLERANCE_NOT_POSITIVE


def test_observable_panel_rejects_a_quantile_outside_the_unit_interval(observable_panel):
    payload = observable_panel.to_mapping()
    payload["primary"][0]["discovery_quantile"] = 1.0
    with pytest.raises(IntegrityError) as excinfo:
        C.ObservablePanel.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.INVALID_QUANTILE


def test_observable_panel_family_size_counts_primary_observables(observable_panel):
    assert observable_panel.interval_family_size == 8


# --------------------------------------------------------------------------
# Study role is not the sampling phase
# --------------------------------------------------------------------------

def test_study_role_and_phase_kind_are_separate_vocabularies():
    assert C.StudyRole.SCREEN.value not in {phase.value for phase in C.PhaseKind}
    assert C.measurement_phase_for(C.StudyRole.SCREEN) is C.PhaseKind.PRODUCTION
    assert C.measurement_phase_for(C.StudyRole.DISCOVERY) is C.PhaseKind.EXPLORATION


def test_measured_study_role_may_not_claim_a_non_production_phase():
    with pytest.raises(IntegrityError) as excinfo:
        C.validate_role_phase(C.StudyRole.SCREEN, C.PhaseKind.PILOT)
    assert excinfo.value.reason is C.ReasonCode.STUDY_ROLE_PHASE_CONFLICT
    C.validate_role_phase(C.StudyRole.SCREEN, C.PhaseKind.PRODUCTION)
    C.validate_role_phase(C.StudyRole.DISCOVERY, C.PhaseKind.EXPLORATION)


def test_phase_kind_vocabulary_matches_the_existing_correctness_api():
    from gareus.correctness.sampling_policy import SamplingPolicy

    policy = SamplingPolicy(C.PhaseKind.PRODUCTION.value, True, False)
    assert policy.phase_kind == "production"
    assert {phase.value for phase in C.PhaseKind} == {
        "production", "pilot", "exploration", "equilibration"}


# --------------------------------------------------------------------------
# Artifact digests
# --------------------------------------------------------------------------

def test_artifact_digest_excludes_its_own_digest_field(feature_schema):
    payload = feature_schema.to_mapping()
    assert payload["sha256"] == feature_schema.sha256
    without = {k: v for k, v in payload.items() if k != "sha256"}
    assert C.artifact_digest(without) == feature_schema.sha256


def test_verify_artifact_digest_detects_a_mutated_artifact(feature_schema):
    payload = feature_schema.to_mapping()
    C.verify_artifact_digest(payload)
    payload["topology_sha256"] = "e" * 64
    with pytest.raises(IntegrityError) as excinfo:
        C.verify_artifact_digest(payload)
    assert excinfo.value.reason is C.ReasonCode.ARTIFACT_DIGEST_MISMATCH


def test_loading_accepts_its_own_canonical_bytes(feature_schema):
    payload = json_bytes(feature_schema.to_mapping())
    assert C.FeatureSchema.from_json_bytes(payload).sha256 == feature_schema.sha256


def test_loading_rejects_duplicate_json_keys(feature_schema):
    text = json_bytes(feature_schema.to_mapping()).decode()
    marker = '"topology_sha256":'
    doubled = text.replace(marker, f'{marker}"{"f" * 64}",{marker}', 1)
    with pytest.raises(IntegrityError):
        C.FeatureSchema.from_json_bytes(doubled.encode())


def test_loading_rejects_nonfinite_numbers(feature_schema):
    text = json_bytes(feature_schema.to_mapping()).decode()
    assert '"index":0' in text
    with pytest.raises(IntegrityError):
        C.FeatureSchema.from_json_bytes(text.replace('"index":0', '"index":NaN', 1).encode())


# --------------------------------------------------------------------------
# Decision record
# --------------------------------------------------------------------------

def _decision_mapping(**overrides) -> dict:
    base = {
        "schema": C.DECISION_VERSION,
        "study_id": "chignolin-fixed-primary-001",
        "integrity": C.Integrity.PASS.value,
        "precision": C.Precision.UNRESOLVED.value,
        "dependence": C.Dependence.UNRESOLVED_CORRELATION.value,
        "reproducibility": C.Reproducibility.UNRESOLVED.value,
        "cross_protocol": C.CrossProtocol.NOT_COMPARED.value,
        "support": C.Support.UNRESOLVED.value,
        "advantage": C.Advantage.NOT_TESTED.value,
        "decision": C.DecisionOutcome.INSUFFICIENT_EVIDENCE.value,
        "reasons": [C.ReasonCode.UNRESOLVED_CORRELATION.value],
        "unresolved_regions": [],
        "tested_protocol_ids": ["contact-only", "contact+residual-pc01"],
        "input_sha256": {"candidate_set": "c" * 64, "observable_panel": "d" * 64},
        "selected_arm_id": None,
    }
    base.update(overrides)
    return base


def _confirmed_mapping(**overrides) -> dict:
    confirmed = {
        "decision": C.DecisionOutcome.CONFIRMED_FOR_DECLARED_PANEL.value,
        "precision": C.Precision.MET.value,
        "dependence": C.Dependence.SUPPORTED.value,
        "reproducibility": C.Reproducibility.NO_CONFLICT_DETECTED.value,
        "support": C.Support.SUPPORTED_ON_DECLARED_PANEL.value,
        "cross_protocol": C.CrossProtocol.AGREEMENT_SUPPORTED.value,
        "selected_arm_id": "contact+residual-pc02",
        "reasons": [],
    }
    confirmed.update(overrides)
    return _decision_mapping(**confirmed)


def test_decision_keeps_eight_separate_status_fields():
    decision = C.Decision.from_mapping(_decision_mapping())
    mapping = decision.to_mapping()
    for field in ("integrity", "precision", "dependence", "reproducibility",
                  "cross_protocol", "support", "advantage", "decision"):
        assert field in mapping
    assert C.Decision.from_mapping(mapping) == decision


def test_decision_rejects_an_invalid_status_value():
    with pytest.raises(IntegrityError) as excinfo:
        C.Decision.from_mapping(_decision_mapping(precision="GOOD"))
    assert excinfo.value.reason is C.ReasonCode.INVALID_ENUM


def test_decision_rejects_a_free_text_reason():
    with pytest.raises(IntegrityError) as excinfo:
        C.Decision.from_mapping(_decision_mapping(reasons=["it looked fine"]))
    assert excinfo.value.reason is C.ReasonCode.UNKNOWN_REASON_CODE


def test_a_valid_confirmation_round_trips():
    decision = C.Decision.from_mapping(_confirmed_mapping())
    assert decision.decision is C.DecisionOutcome.CONFIRMED_FOR_DECLARED_PANEL
    assert decision.exit_code is C.ExitCode.OK


def test_confirmation_requires_a_selected_arm():
    with pytest.raises(IntegrityError) as excinfo:
        C.Decision.from_mapping(_confirmed_mapping(selected_arm_id=None))
    assert excinfo.value.reason is C.ReasonCode.MISSING_SELECTED_ARM


def test_a_protocol_disagreement_cannot_be_reported_as_confirmed():
    with pytest.raises(IntegrityError) as excinfo:
        C.Decision.from_mapping(_confirmed_mapping(
            cross_protocol=C.CrossProtocol.PROTOCOL_DISAGREEMENT.value))
    assert excinfo.value.reason is C.ReasonCode.CONFIRMATION_BLOCKED


def test_unresolved_precision_cannot_be_reported_as_confirmed():
    with pytest.raises(IntegrityError) as excinfo:
        C.Decision.from_mapping(_confirmed_mapping(precision=C.Precision.UNRESOLVED.value))
    assert excinfo.value.reason is C.ReasonCode.CONFIRMATION_BLOCKED


def test_failed_integrity_forces_the_invalid_input_outcome():
    with pytest.raises(IntegrityError) as excinfo:
        C.Decision.from_mapping(_decision_mapping(integrity=C.Integrity.FAIL.value))
    assert excinfo.value.reason is C.ReasonCode.INTEGRITY_OUTCOME_CONFLICT


def test_an_inconclusive_decision_is_a_valid_recordable_result():
    decision = C.Decision.from_mapping(_decision_mapping())
    assert decision.decision is C.DecisionOutcome.INSUFFICIENT_EVIDENCE
    assert decision.exit_code is C.ExitCode.OK


def test_an_invalid_input_decision_reports_a_nonzero_exit_code():
    decision = C.Decision.from_mapping(_decision_mapping(
        integrity=C.Integrity.FAIL.value,
        decision=C.DecisionOutcome.INVALID_INPUT.value,
        reasons=[C.ReasonCode.ARTIFACT_DIGEST_MISMATCH.value]))
    assert decision.exit_code is C.ExitCode.INVALID_INPUT


# --------------------------------------------------------------------------
# Readiness
# --------------------------------------------------------------------------

def test_build_stage_is_ready_for_a_structurally_complete_protocol(protocol):
    report = C.validate_protocol(protocol, {}, stage=C.Stage.BUILD)
    assert report.ready is True
    assert report.missing == ()
    assert report.reason is None


def test_screen_stage_refuses_a_protocol_with_unresolved_policies(protocol):
    report = C.validate_protocol(protocol, {}, stage=C.Stage.SCREEN)
    assert report.ready is False
    missing = {item.requirement for item in report.missing}
    assert "sampling.burn_in_ticks" in missing
    assert "uncertainty.calibration_certificate_sha256" in missing
    assert all(item.producing_task for item in report.missing)
    assert report.reason is C.ReasonCode.PROTOCOL_NOT_READY
    assert report.exit_code is C.ExitCode.NOT_READY


def test_readiness_never_invents_a_default_for_a_missing_policy(protocol):
    report = C.validate_protocol(protocol, {}, stage=C.Stage.SCREEN)
    assert report.ready is False
    # The report enumerates the gap; it does not fill it in.
    assert protocol.sampling["burn_in_ticks"] is None
    assert C.ProtocolSpec.from_mapping(protocol.to_mapping()).sampling["burn_in_ticks"] is None


def test_engineering_stage_does_not_require_a_calibration_certificate(protocol):
    report = C.validate_protocol(protocol, {}, stage=C.Stage.ENGINEERING)
    missing = {item.requirement for item in report.missing}
    assert "uncertainty.calibration_certificate_sha256" not in missing


def test_stages_are_cumulative(feature_schema, candidate_set, observable_panel):
    spec = C.ProtocolSpec.from_mapping(
        _resolved_protocol_mapping(feature_schema, candidate_set, observable_panel))
    artifacts = {"candidate_set": candidate_set.sha256,
                 "feature_schema": feature_schema.sha256,
                 "observable_panel": observable_panel.sha256,
                 "calibration_certificate": "9" * 64}
    assert C.validate_protocol(spec, artifacts, stage=C.Stage.SCREEN).ready is True

    confirm = C.validate_protocol(spec, artifacts, stage=C.Stage.CONFIRM)
    assert confirm.ready is False
    assert {item.requirement for item in confirm.missing} >= {
        "decision.provisional_choice_arm_id", "decision.interval_family_size"}


def test_a_declared_artifact_digest_must_match_the_supplied_artifact(
        feature_schema, candidate_set, observable_panel):
    spec = C.ProtocolSpec.from_mapping(
        _resolved_protocol_mapping(feature_schema, candidate_set, observable_panel))
    artifacts = {"candidate_set": "0" * 64, "feature_schema": feature_schema.sha256,
                 "observable_panel": observable_panel.sha256,
                 "calibration_certificate": "9" * 64}
    report = C.validate_protocol(spec, artifacts, stage=C.Stage.SCREEN)
    assert report.ready is False
    assert any(item.reason is C.ReasonCode.ARTIFACT_DIGEST_MISMATCH for item in report.missing)


def test_a_missing_artifact_is_reported_rather_than_assumed(
        feature_schema, candidate_set, observable_panel):
    spec = C.ProtocolSpec.from_mapping(
        _resolved_protocol_mapping(feature_schema, candidate_set, observable_panel))
    report = C.validate_protocol(spec, {}, stage=C.Stage.SCREEN)
    assert report.ready is False
    assert any(item.reason is C.ReasonCode.MISSING_ARTIFACT for item in report.missing)


def test_protocol_rejects_a_fold_specific_generator_preset():
    with pytest.raises(IntegrityError) as excinfo:
        C.ProtocolSpec.from_mapping(_protocol_mapping(
            blindness={"generator_preset": "chignolin",
                       "require_native_blind_allowlist": True,
                       "historical_data_role": "exploratory_only"}))
    assert excinfo.value.reason is C.ReasonCode.NATIVE_DERIVED_INPUT


def test_protocol_rejects_historical_data_promoted_to_evidence():
    with pytest.raises(IntegrityError) as excinfo:
        C.ProtocolSpec.from_mapping(_protocol_mapping(
            blindness={"generator_preset": "broad",
                       "require_native_blind_allowlist": True,
                       "historical_data_role": "confirmatory"}))
    assert excinfo.value.reason is C.ReasonCode.INVALID_ENUM


def test_protocol_rejects_a_disabled_native_blind_allowlist():
    with pytest.raises(IntegrityError) as excinfo:
        C.ProtocolSpec.from_mapping(_protocol_mapping(
            blindness={"generator_preset": "broad",
                       "require_native_blind_allowlist": False,
                       "historical_data_role": "exploratory_only"}))
    assert excinfo.value.reason is C.ReasonCode.NATIVE_DERIVED_INPUT


def test_protocol_rejects_an_unknown_section(protocol):
    payload = protocol.to_mapping()
    payload["kinetics"] = {"target_rate": 1.0}
    with pytest.raises(IntegrityError) as excinfo:
        C.ProtocolSpec.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.UNKNOWN_FIELD


def test_protocol_rejects_an_npt_target_without_a_pressure():
    with pytest.raises(IntegrityError) as excinfo:
        C.ProtocolSpec.from_mapping(_protocol_mapping(
            target={"sequence": "GYDPETGTWG", "temperature_k": 300.0, "ensemble": "NPT",
                    "pressure_bar": None, "physical_system_sha256": "b" * 64}))
    assert excinfo.value.reason is C.ReasonCode.MISSING_FIELD


# --------------------------------------------------------------------------
# Trial plan
# --------------------------------------------------------------------------

def _trial_plan_mapping(**overrides) -> dict:
    arms = [{"arm_id": "contact-only", "secondary_component_index": None,
             "campaigns": 3, "gpu_hours_per_campaign": 8.0}]
    arms += [{"arm_id": f"contact+residual-pc{j:02d}", "secondary_component_index": j,
              "campaigns": 3, "gpu_hours_per_campaign": 8.0} for j in range(1, 7)]
    base = {
        "schema": C.TRIAL_PLAN_VERSION,
        "plan_id": "screen-001",
        "study_id": "chignolin-fixed-primary-001",
        "stage": C.Stage.SCREEN.value,
        "study_role": C.StudyRole.SCREEN.value,
        "measurement_phase_kind": C.PhaseKind.PRODUCTION.value,
        "protocol_sha256": "1" * 64,
        "layout": "grid24-v1",
        "lambda_rungs": 4,
        "states_per_arm": 96,
        "max_replicas": 128,
        "arms": arms,
    }
    base.update(overrides)
    return base


def test_trial_plan_gives_every_component_a_real_pilot_budget():
    plan = C.TrialPlan.from_mapping(_trial_plan_mapping())
    assert len(plan.arms) == 7
    assert {arm.secondary_component_index for arm in plan.arms} == {None, 1, 2, 3, 4, 5, 6}
    assert all(arm.campaigns > 0 and arm.gpu_hours_per_campaign > 0 for arm in plan.arms)
    assert plan.total_gpu_hours == pytest.approx(7 * 3 * 8.0)


def test_trial_plan_rejects_a_zero_budget_arm():
    payload = _trial_plan_mapping()
    payload["arms"][3]["gpu_hours_per_campaign"] = 0.0
    with pytest.raises(IntegrityError) as excinfo:
        C.TrialPlan.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.ARM_WITHOUT_BUDGET


def test_trial_plan_rejects_unequal_campaign_allocation_across_arms():
    payload = _trial_plan_mapping()
    payload["arms"][0]["campaigns"] = 5
    with pytest.raises(IntegrityError) as excinfo:
        C.TrialPlan.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.UNEQUAL_ARM_COST


def test_trial_plan_rejects_duplicate_arm_ids():
    payload = _trial_plan_mapping()
    payload["arms"][1]["arm_id"] = "contact-only"
    with pytest.raises(IntegrityError) as excinfo:
        C.TrialPlan.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.DUPLICATE_ARM_ID


def test_trial_plan_rejects_duplicate_component_arms():
    payload = _trial_plan_mapping()
    payload["arms"][2]["secondary_component_index"] = 1
    with pytest.raises(IntegrityError) as excinfo:
        C.TrialPlan.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.DUPLICATE_COMPONENT_INDEX


def test_trial_plan_rejects_more_states_than_the_replica_cap():
    with pytest.raises(IntegrityError) as excinfo:
        C.TrialPlan.from_mapping(_trial_plan_mapping(states_per_arm=512))
    assert excinfo.value.reason is C.ReasonCode.REPLICA_CAP_EXCEEDED


def test_trial_plan_rejects_a_measurement_phase_that_is_not_production():
    payload = _trial_plan_mapping(measurement_phase_kind=C.PhaseKind.PILOT.value)
    with pytest.raises(IntegrityError) as excinfo:
        C.TrialPlan.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.STUDY_ROLE_PHASE_CONFLICT


def test_trial_plan_rejects_a_state_count_not_divisible_by_the_rungs():
    payload = _trial_plan_mapping(states_per_arm=97)
    with pytest.raises(IntegrityError) as excinfo:
        C.TrialPlan.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.LAYOUT_INCONSISTENT


# --------------------------------------------------------------------------
# Exit codes and dependency footprint
# --------------------------------------------------------------------------

def test_exit_codes_are_distinct_and_reserve_zero_for_a_valid_inconclusive_run():
    values = [code.value for code in C.ExitCode]
    assert len(values) == len(set(values))
    assert C.ExitCode.OK.value == 0
    assert {C.ExitCode.INVALID_INPUT.value, C.ExitCode.UNAVAILABLE_DEPENDENCY.value,
            C.ExitCode.EXECUTION_FAILURE.value, C.ExitCode.NOT_READY.value} == {2, 3, 4, 5}


def test_contracts_import_with_a_numpy_only_core_installation():
    """Optional analysis/OpenMM extras must not be pulled in transitively."""
    blocked = ("pyarrow", "duckdb", "pymbar", "openmm", "yaml", "pandas", "mdtraj")
    script = (
        "import sys\n"
        f"blocked = {blocked!r}\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in blocked:\n"
        "            raise ImportError('blocked for this test: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "from gareus.cv_selection import contracts\n"
        "assert contracts.PROTOCOL_VERSION\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# --------------------------------------------------------------------------
# Stored fixtures: a schema edit must not silently invalidate saved artifacts
# --------------------------------------------------------------------------

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "cv_selection"

_FIXTURE_TYPES = {
    "feature_schema": C.FeatureSchema,
    "candidate_set": C.CandidateSet,
    "observable_panel": C.ObservablePanel,
    "protocol": C.ProtocolSpec,
    "trial_plan": C.TrialPlan,
    "decision": C.Decision,
}


@pytest.mark.parametrize("name", sorted(_FIXTURE_TYPES))
def test_stored_fixture_still_loads_and_matches_its_recorded_digest(name):
    payload = (FIXTURES / f"{name}.json").read_bytes()
    recorded = C.verify_artifact_digest(C.json_loads(payload))
    assert _FIXTURE_TYPES[name].from_json_bytes(payload).sha256 == recorded


def test_stored_candidate_set_binds_to_the_stored_feature_schema():
    schema = C.FeatureSchema.from_json_bytes((FIXTURES / "feature_schema.json").read_bytes())
    candidates = C.CandidateSet.from_json_bytes((FIXTURES / "candidate_set.json").read_bytes())
    C.require_feature_binding(candidates, schema)


def test_the_generated_example_protocol_does_not_look_launch_ready():
    spec = C.ProtocolSpec.from_json_bytes((FIXTURES / "protocol.json").read_bytes())
    assert C.validate_protocol(spec, {}, stage=C.Stage.SCREEN).ready is False
