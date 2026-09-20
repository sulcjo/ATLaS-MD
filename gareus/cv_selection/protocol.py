"""The declared study, and whether it may be used at a given stage.

Readiness is deliberately conservative: :func:`validate_protocol` enumerates
what is still unresolved and which task must produce it, and never supplies a
scientific default. A duration, burn-in length or boost envelope cannot be
invented from a schema, so a protocol that has not been through the
artifact-producing tasks is refused rather than quietly completed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._base import (ReasonCode, _Artifact, _enum, _exact_fields, _fail, _hex64,
                    _positive_float, _schema, artifact_digest, require_visible_text)
from .vocabulary import ExitCode, SearchMode, Stage, _STAGE_ORDER

PROTOCOL_VERSION = "atlas-cv-selection-protocol-v1"
READINESS_REPORT_VERSION = "atlas-cv-selection-readiness-report-v1"

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
    if section["ensemble"] == "NPT":
        if section["pressure_bar"] is None:
            _fail(ReasonCode.MISSING_FIELD, "an NPT target requires an explicit pressure_bar")
        _positive_float(section["pressure_bar"], "target.pressure_bar",
                        ReasonCode.NOT_POSITIVE)
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


#: Per-field value rules for the policies that may legitimately be unresolved.
#: ``None`` means "not decided yet" and is reported by the readiness gate; any
#: other value must already be a usable one. Without this, a protocol declaring
#: a negative burn-in or a string campaign count passes readiness and is
#: certified launch-ready, which is exactly the failure the machine-checked
#: readiness state exists to prevent.
_VALUE_RULES: dict[str, str] = {
    "sampling.layout": "nonempty_str",
    "sampling.lambda_rungs": "positive_int",
    "sampling.burn_in_ticks": "nonnegative_int",
    "sampling.measurement_ticks": "positive_int",
    "sampling.campaigns_per_arm": "positive_int",
    "resources.gpus": "positive_int",
    "resources.cpu_thread_cap": "positive_int",
    "resources.max_replicas": "positive_int",
    "resources.gpu_hours_per_campaign": "positive_number",
    "resources.total_gpu_hour_ceiling": "positive_number",
    "uncertainty.policy": "nonempty_str",
    "decision.provisional_choice_arm_id": "nonempty_str",
    "decision.interval_family_size": "positive_int",
}


def _check_value(value: Any, rule: str, label: str) -> None:
    if rule == "nonempty_str":
        require_visible_text(value, label)
        return
    if rule in {"positive_int", "nonnegative_int"}:
        # bool is a subclass of int; True would otherwise pass as 1.
        if isinstance(value, bool) or not isinstance(value, int):
            _fail(ReasonCode.WRONG_TYPE,
                  f"{label} must be an integer, got {type(value).__name__}")
        if rule == "positive_int" and value <= 0:
            _fail(ReasonCode.NOT_POSITIVE, f"{label} must be positive, got {value}")
        if rule == "nonnegative_int" and value < 0:
            _fail(ReasonCode.NEGATIVE_VALUE, f"{label} must not be negative, got {value}")
        return
    if rule == "positive_number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail(ReasonCode.WRONG_TYPE,
                  f"{label} must be a number, got {type(value).__name__}")
        if float(value) <= 0.0:
            _fail(ReasonCode.NOT_POSITIVE, f"{label} must be positive, got {value}")
        return
    raise AssertionError(f"unknown value rule {rule!r}")


#: Fields whose science is a real number. JSON cannot tell 300 from 300.0, so
#: without coercion the same target hashes two ways and a protocol digest
#: declared against one spelling can never match the other.
_REAL_VALUED_FIELDS = (
    ("target", "temperature_k"), ("target", "pressure_bar"),
    ("resources", "gpu_hours_per_campaign"), ("resources", "total_gpu_hour_ceiling"),
)


def _normalise_numeric_fields(sections) -> None:
    for section, field in _REAL_VALUED_FIELDS:
        value = sections[section][field]
        if value is not None and not isinstance(value, bool):
            sections[section][field] = float(value)


def _validate_resolvable_values(sections: Mapping[str, Mapping[str, Any]]) -> None:
    """A resolved policy must be usable; only ``None`` may mean 'not yet'."""
    for dotted, rule in _VALUE_RULES.items():
        section, _, field = dotted.partition(".")
        value = sections[section][field]
        if value is None:
            continue
        _check_value(value, rule, f"protocol.{dotted}")


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
        # `1 in (True, False)` is True: compare identity, not equality.
        if not isinstance(sections["decision"]["advantage_claim_enabled"], bool):
            _fail(ReasonCode.WRONG_TYPE,
                  "decision.advantage_claim_enabled must be a boolean, not a number")
        _validate_resolvable_values(sections)
        _normalise_numeric_fields(sections)
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
class ReadinessReport(_Artifact):
    """Persisted as ``readiness/<stage>.json``, so it is versioned and hashed.

    A readiness verdict is evidence about whether a campaign may launch. If it
    could not be round-tripped and checked like every other artifact, a stale or
    edited report could authorise a launch it never actually cleared.
    """

    stage: Stage
    ready: bool
    missing: tuple[MissingRequirement, ...]
    protocol_sha256: str
    reason: ReasonCode | None
    sha256: str

    @property
    def exit_code(self) -> ExitCode:
        return ExitCode.OK if self.ready else ExitCode.NOT_READY

    @classmethod
    def build(cls, stage: Stage, missing: Sequence[MissingRequirement],
              protocol_sha256: str) -> "ReadinessReport":
        ready = not missing
        reason = None if ready else ReasonCode.PROTOCOL_NOT_READY
        body = _readiness_body(stage, ready, tuple(missing), protocol_sha256, reason)
        return cls(stage, ready, tuple(missing), protocol_sha256, reason,
                   artifact_digest(body))

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "ReadinessReport":
        _exact_fields(data, {"schema", "stage", "ready", "protocol_sha256", "reason",
                             "missing"}, "readiness report")
        _schema(data, READINESS_REPORT_VERSION, "readiness report")
        stage = _enum(data["stage"], Stage, "readiness report.stage")
        if not isinstance(data["ready"], bool):
            _fail(ReasonCode.WRONG_TYPE, "readiness report.ready must be a boolean")
        rows = data["missing"]
        if not isinstance(rows, list):
            _fail(ReasonCode.WRONG_TYPE, "readiness report.missing must be a list")
        missing = tuple(_parse_missing_requirement(row, i) for i, row in enumerate(rows))
        reason = (None if data["reason"] is None
                  else _enum(data["reason"], ReasonCode, "readiness report.reason"))
        if data["ready"] != (not missing):
            _fail(ReasonCode.READINESS_INCONSISTENT,
                  "readiness report.ready must agree with its own missing list")
        if bool(missing) != (reason is ReasonCode.PROTOCOL_NOT_READY):
            _fail(ReasonCode.READINESS_INCONSISTENT,
                  "an unready report must carry PROTOCOL_NOT_READY, a ready one no reason")
        digest_of = _hex64(data["protocol_sha256"], "readiness report.protocol_sha256")
        body = _readiness_body(stage, data["ready"], missing, digest_of, reason)
        return cls(stage, data["ready"], missing, digest_of, reason, artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        return _readiness_body(self.stage, self.ready, self.missing,
                               self.protocol_sha256, self.reason)


def _readiness_body(stage: Stage, ready: bool, missing, protocol_sha256: str,
                    reason: ReasonCode | None) -> dict[str, Any]:
    return {"schema": READINESS_REPORT_VERSION, "stage": stage.value, "ready": ready,
            "protocol_sha256": protocol_sha256,
            "reason": None if reason is None else reason.value,
            "missing": [item.to_mapping() for item in missing]}


def _parse_missing_requirement(raw: Any, position: int) -> MissingRequirement:
    label = f"readiness report.missing[{position}]"
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.WRONG_TYPE, f"{label} must be a mapping")
    _exact_fields(raw, {"requirement", "stage", "producing_task", "reason", "detail"}, label)
    for key in ("requirement", "producing_task"):
        require_visible_text(raw[key], f"{label}.{key}")
    if not isinstance(raw["detail"], str):
        _fail(ReasonCode.WRONG_TYPE, f"{label}.detail must be a string")
    return MissingRequirement(raw["requirement"],
                              _enum(raw["stage"], Stage, f"{label}.stage"),
                              raw["producing_task"],
                              _enum(raw["reason"], ReasonCode, f"{label}.reason"),
                              raw["detail"])


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
    return ReadinessReport.build(stage, missing, protocol.sha256)


