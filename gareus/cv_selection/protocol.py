"""The declared study, and whether it may be used at a given stage.

Readiness is deliberately conservative: :func:`validate_protocol` enumerates
what is still unresolved and which task must produce it, and never supplies a
scientific default. A duration, burn-in length or boost envelope cannot be
invented from a schema, so a protocol that has not been through the
artifact-producing tasks is refused rather than quietly completed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._base import (ReasonCode, _Artifact, _enum, _exact_fields, _fail, _hex64,
                    _positive_float, _schema, artifact_digest)
from .vocabulary import ExitCode, SearchMode, Stage, _STAGE_ORDER

PROTOCOL_VERSION = "atlas-cv-selection-protocol-v1"

#: GENPEPT presets that encode a known fold bias are not native-blind sources.
NATIVE_BLIND_GENERATOR_PRESETS = frozenset({"broad"})



_PROTOCOL_SECTIONS = {
    "target": {"sequence", "temperature_k", "ensemble", "pressure_bar", "physical_system_sha256"},
    "blindness": {"generator_preset", "require_native_blind_allowlist", "historical_data_role"},
    "discovery": {"candidate_set_sha256", "feature_schema_sha256"},
    "sampling": {"layout", "lambda_rungs", "burn_in_ticks", "measurement_ticks",
                 "campaigns_per_arm"},
    "resources": {"gpus", "cpu_thread_cap", "max_replicas", "gpu_hours_per_campaign",
                  "total_gpu_hour_ceiling"},
    "evaluation": {"observable_panel_sha256"},
    "uncertainty": {"policy", "calibration_certificate_sha256"},
    "decision": {"provisional_choice_arm_id", "interval_family_size", "advantage_claim_enabled"},
}

_HISTORICAL_DATA_ROLES = frozenset({"exploratory_only", "excluded"})

#: Which task produces each resolvable policy, so a refusal is actionable.
_REQUIREMENT_TASKS = {
    "target": "T11", "discovery": "T03", "sampling": "T09", "resources": "T11",
    "evaluation": "T07", "uncertainty": "T06", "decision": "T07",
}

_STAGE_REQUIREMENTS: dict[Stage, tuple[str, ...]] = {
    Stage.BUILD: (),
    Stage.ENGINEERING: ("sampling.layout", "sampling.lambda_rungs"),
    Stage.SCREEN: ("discovery.candidate_set_sha256", "discovery.feature_schema_sha256",
                   "evaluation.observable_panel_sha256", "uncertainty.policy",
                   "uncertainty.calibration_certificate_sha256", "sampling.burn_in_ticks",
                   "sampling.measurement_ticks", "sampling.campaigns_per_arm",
                   "resources.gpu_hours_per_campaign", "resources.total_gpu_hour_ceiling"),
    Stage.CONFIRM: ("decision.provisional_choice_arm_id", "decision.interval_family_size"),
}

#: Protocol field holding each artifact's declared digest.
_ARTIFACT_FIELDS = {
    "candidate_set": "discovery.candidate_set_sha256",
    "feature_schema": "discovery.feature_schema_sha256",
    "observable_panel": "evaluation.observable_panel_sha256",
    "calibration_certificate": "uncertainty.calibration_certificate_sha256",
}


def _validate_target(section: Mapping[str, Any]) -> None:
    if section["ensemble"] not in {"NPT", "NVT"}:
        _fail(ReasonCode.INVALID_ENUM, "target.ensemble must be NPT or NVT")
    if section["ensemble"] == "NPT" and section["pressure_bar"] is None:
        _fail(ReasonCode.MISSING_FIELD, "an NPT target requires an explicit pressure_bar")
    if section["ensemble"] == "NVT" and section["pressure_bar"] is not None:
        _fail(ReasonCode.UNKNOWN_FIELD, "an NVT target must not carry a pressure_bar")
    _positive_float(section["temperature_k"], "target.temperature_k", ReasonCode.MISSING_FIELD)
    _hex64(section["physical_system_sha256"], "target.physical_system_sha256")
    if not isinstance(section["sequence"], str) or not section["sequence"]:
        _fail(ReasonCode.MISSING_FIELD, "target.sequence must be a nonempty string")


def _validate_blindness(section: Mapping[str, Any]) -> None:
    if section["require_native_blind_allowlist"] is not True:
        _fail(ReasonCode.NATIVE_DERIVED_INPUT,
              "prospective selection requires the native-blind allowlist; it cannot be disabled")
    if section["generator_preset"] not in NATIVE_BLIND_GENERATOR_PRESETS:
        _fail(ReasonCode.NATIVE_DERIVED_INPUT,
              f"generator preset {section['generator_preset']!r} encodes a fold bias; use one of "
              f"{sorted(NATIVE_BLIND_GENERATOR_PRESETS)}")
    if section["historical_data_role"] not in _HISTORICAL_DATA_ROLES:
        _fail(ReasonCode.INVALID_ENUM,
              f"historical_data_role must be one of {sorted(_HISTORICAL_DATA_ROLES)}; historical "
              "campaigns are never confirmatory evidence")


@dataclass(frozen=True)
class ProtocolSpec(_Artifact):
    """The whole declared study. Unresolved policies stay ``None`` on purpose."""

    study_id: str
    search_mode: SearchMode
    target: dict[str, Any]
    blindness: dict[str, Any]
    discovery: dict[str, Any]
    sampling: dict[str, Any]
    resources: dict[str, Any]
    evaluation: dict[str, Any]
    uncertainty: dict[str, Any]
    decision: dict[str, Any]
    sha256: str

    def value_at(self, dotted: str) -> Any:
        section, _, field = dotted.partition(".")
        return getattr(self, section)[field]

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "ProtocolSpec":
        _exact_fields(data, {"schema", "study_id", "search_mode", *_PROTOCOL_SECTIONS}, "protocol")
        _schema(data, PROTOCOL_VERSION, "protocol")
        if not isinstance(data["study_id"], str) or not data["study_id"]:
            _fail(ReasonCode.MISSING_FIELD, "protocol.study_id must be a nonempty string")
        mode = _enum(data["search_mode"], SearchMode, "protocol.search_mode")
        sections = {}
        for name, fields in _PROTOCOL_SECTIONS.items():
            section = data[name]
            if not isinstance(section, dict):
                _fail(ReasonCode.MISSING_FIELD, f"protocol.{name} must be a mapping")
            _exact_fields(section, fields, f"protocol.{name}")
            sections[name] = dict(section)
        _validate_target(sections["target"])
        _validate_blindness(sections["blindness"])
        for field in ("candidate_set_sha256", "feature_schema_sha256"):
            _hex64(sections["discovery"][field], f"protocol.discovery.{field}", allow_none=True)
        _hex64(sections["evaluation"]["observable_panel_sha256"],
               "protocol.evaluation.observable_panel_sha256", allow_none=True)
        _hex64(sections["uncertainty"]["calibration_certificate_sha256"],
               "protocol.uncertainty.calibration_certificate_sha256", allow_none=True)
        if sections["decision"]["advantage_claim_enabled"] not in (True, False):
            _fail(ReasonCode.INVALID_ENUM, "decision.advantage_claim_enabled must be a boolean")
        body = {"schema": PROTOCOL_VERSION, "study_id": data["study_id"],
                "search_mode": mode.value, **sections}
        return cls(data["study_id"], mode, sections["target"], sections["blindness"],
                   sections["discovery"], sections["sampling"], sections["resources"],
                   sections["evaluation"], sections["uncertainty"], sections["decision"],
                   artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        return {"schema": PROTOCOL_VERSION, "study_id": self.study_id,
                "search_mode": self.search_mode.value, "target": dict(self.target),
                "blindness": dict(self.blindness), "discovery": dict(self.discovery),
                "sampling": dict(self.sampling), "resources": dict(self.resources),
                "evaluation": dict(self.evaluation), "uncertainty": dict(self.uncertainty),
                "decision": dict(self.decision)}


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MissingRequirement:
    requirement: str
    stage: Stage
    producing_task: str
    reason: ReasonCode
    detail: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {"requirement": self.requirement, "stage": self.stage.value,
                "producing_task": self.producing_task, "reason": self.reason.value,
                "detail": self.detail}


@dataclass(frozen=True)
class ReadinessReport:
    stage: Stage
    ready: bool
    missing: tuple[MissingRequirement, ...]
    protocol_sha256: str
    reason: ReasonCode | None = None

    @property
    def exit_code(self) -> ExitCode:
        return ExitCode.OK if self.ready else ExitCode.NOT_READY

    def to_mapping(self) -> dict[str, Any]:
        return {"stage": self.stage.value, "ready": self.ready,
                "protocol_sha256": self.protocol_sha256,
                "reason": None if self.reason is None else self.reason.value,
                "missing": [item.to_mapping() for item in self.missing]}


def _requirements_through(stage: Stage) -> tuple[str, ...]:
    return tuple(requirement for level in _STAGE_ORDER[:stage.rank + 1]
                 for requirement in _STAGE_REQUIREMENTS[level])


def _artifact_findings(protocol: ProtocolSpec, artifacts: Mapping[str, str],
                       stage: Stage) -> list[MissingRequirement]:
    findings: list[MissingRequirement] = []
    required = set(_requirements_through(stage))
    for name, dotted in _ARTIFACT_FIELDS.items():
        if dotted not in required:
            continue
        declared = protocol.value_at(dotted)
        if declared is None:
            continue
        supplied = artifacts.get(name)
        section = dotted.split(".")[0]
        if supplied is None:
            findings.append(MissingRequirement(
                dotted, stage, _REQUIREMENT_TASKS[section], ReasonCode.MISSING_ARTIFACT,
                f"protocol declares {name} {declared} but no such artifact was supplied"))
        elif supplied != declared:
            findings.append(MissingRequirement(
                dotted, stage, _REQUIREMENT_TASKS[section], ReasonCode.ARTIFACT_DIGEST_MISMATCH,
                f"protocol declares {name} {declared}, supplied artifact hashes to {supplied}"))
    return findings


def validate_protocol(protocol: ProtocolSpec, artifacts: Mapping[str, str], *,
                      stage: Stage) -> ReadinessReport:
    """Report whether the protocol may be used at ``stage``; never fill a gap in.

    ``artifacts`` maps an artifact name to the digest of the file actually on
    hand. A declared-but-absent or declared-but-different artifact is reported,
    not assumed.
    """
    if not isinstance(stage, Stage):
        _fail(ReasonCode.INVALID_ENUM, f"stage must be a Stage, got {stage!r}")
    missing: list[MissingRequirement] = []
    for dotted in _requirements_through(stage):
        if protocol.value_at(dotted) is None:
            section = dotted.split(".")[0]
            missing.append(MissingRequirement(
                dotted, stage, _REQUIREMENT_TASKS[section], ReasonCode.MISSING_FIELD,
                "unresolved policy; the producing task must supply a concrete value"))
    missing.extend(_artifact_findings(protocol, artifacts, stage))
    ready = not missing
    return ReadinessReport(stage, ready, tuple(missing), protocol.sha256,
                           None if ready else ReasonCode.PROTOCOL_NOT_READY)


