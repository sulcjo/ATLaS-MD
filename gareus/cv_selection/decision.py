"""The verdict artifact: eight independent statuses plus its evidence.

Precision is not accuracy. A sampling run that is reproducibly trapped in the
wrong basin has tiny confidence intervals, perfect run-to-run agreement and a
badly biased mean, so a verdict that collapsed to one pass/fail number would
report it as confirmed. Every status here therefore stays separate, and a
confirmation must clear all five gating statuses, name the arm it selects,
record the digests it rests on, and cite no blocking reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ._base import (ReasonCode, _Artifact, _enum, _exact_fields, _fail, _hex64,
                    _schema, artifact_digest, has_visible_characters,
                    require_visible_text)
from .vocabulary import (Advantage, CrossProtocol, DecisionOutcome, Dependence,
                         ExitCode, Integrity, Precision, RegionStatus,
                         Reproducibility, Support)

DECISION_VERSION = "atlas-cv-selection-decision-v1"


_DECISION_STATUS_FIELDS: dict[str, type[Enum]] = {
    "integrity": Integrity, "precision": Precision, "dependence": Dependence,
    "reproducibility": Reproducibility, "cross_protocol": CrossProtocol,
    "support": Support, "advantage": Advantage, "decision": DecisionOutcome,
}

#: Reasons that describe a refusal. A verdict cannot both confirm and cite one.
_BLOCKING_REASONS = frozenset({
    ReasonCode.CONFIRMATION_BLOCKED.value,
    ReasonCode.INTEGRITY_OUTCOME_CONFLICT.value,
    ReasonCode.INSUFFICIENT_SUPPORT.value,
    ReasonCode.BUDGET_EXHAUSTED.value,
    ReasonCode.NATIVE_DERIVED_INPUT.value,
    ReasonCode.UNRESOLVED_CORRELATION.value,
    ReasonCode.PROTOCOL_NOT_READY.value,
})

#: Outcomes that select a winner and therefore must name it.
_OUTCOMES_NAMING_AN_ARM = frozenset({
    DecisionOutcome.CONFIRMED_FOR_DECLARED_PANEL,
    DecisionOutcome.PROVISIONAL_CHOICE,
})

#: Every status that must hold before a confirmation may be published.
_CONFIRMATION_REQUIRES = {
    "precision": Precision.MET,
    "dependence": Dependence.SUPPORTED,
    "reproducibility": Reproducibility.NO_CONFLICT_DETECTED,
    "support": Support.SUPPORTED_ON_DECLARED_PANEL,
    "cross_protocol": CrossProtocol.AGREEMENT_SUPPORTED,
}


@dataclass(frozen=True)
class Decision(_Artifact):
    """Eight independent statuses plus evidence. Precision is not accuracy."""

    study_id: str
    integrity: Integrity
    precision: Precision
    dependence: Dependence
    reproducibility: Reproducibility
    cross_protocol: CrossProtocol
    support: Support
    advantage: Advantage
    decision: DecisionOutcome
    reasons: tuple[ReasonCode, ...]
    unresolved_regions: tuple[RegionRecord, ...]
    tested_protocol_ids: tuple[str, ...]
    input_sha256: dict[str, str]
    selected_arm_id: str | None
    sha256: str

    @property
    def exit_code(self) -> ExitCode:
        """A valid inconclusive analysis still exits zero (plan section 11)."""
        return (ExitCode.INVALID_INPUT if self.decision is DecisionOutcome.INVALID_INPUT
                else ExitCode.OK)

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "Decision":
        _exact_fields(data, {"schema", "study_id", *_DECISION_STATUS_FIELDS, "reasons",
                             "unresolved_regions", "tested_protocol_ids", "input_sha256",
                             "selected_arm_id"}, "decision")
        _schema(data, DECISION_VERSION, "decision")
        statuses = {name: _enum(data[name], enum_cls, f"decision.{name}")
                    for name, enum_cls in _DECISION_STATUS_FIELDS.items()}
        reasons = _parse_reasons(data["reasons"])
        _check_decision_consistency(statuses, data["selected_arm_id"], reasons,
                                    data.get("input_sha256"), data.get("tested_protocol_ids"))
        regions = _parse_regions(data["unresolved_regions"])
        protocols = _string_tuple(data["tested_protocol_ids"], "decision.tested_protocol_ids")
        inputs = data["input_sha256"]
        if not isinstance(inputs, dict):
            _fail(ReasonCode.MISSING_FIELD, "decision.input_sha256 must be a mapping")
        for name, value in inputs.items():
            _hex64(value, f"decision.input_sha256[{name}]")
        selected = data["selected_arm_id"]
        if selected is not None:
            require_visible_text(selected, "decision.selected_arm_id")
        body = {"schema": DECISION_VERSION, "study_id": data["study_id"],
                **{name: status.value for name, status in statuses.items()},
                "reasons": [reason.value for reason in reasons],
                "unresolved_regions": [r.to_mapping() for r in regions],
                "tested_protocol_ids": list(protocols),
                "input_sha256": dict(inputs), "selected_arm_id": selected}
        return cls(data["study_id"], statuses["integrity"], statuses["precision"],
                   statuses["dependence"], statuses["reproducibility"],
                   statuses["cross_protocol"], statuses["support"], statuses["advantage"],
                   statuses["decision"], reasons, regions, protocols, dict(inputs), selected,
                   artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        return {"schema": DECISION_VERSION, "study_id": self.study_id,
                "integrity": self.integrity.value, "precision": self.precision.value,
                "dependence": self.dependence.value,
                "reproducibility": self.reproducibility.value,
                "cross_protocol": self.cross_protocol.value, "support": self.support.value,
                "advantage": self.advantage.value, "decision": self.decision.value,
                "reasons": [reason.value for reason in self.reasons],
                "unresolved_regions": [r.to_mapping() for r in self.unresolved_regions],
                "tested_protocol_ids": list(self.tested_protocol_ids),
                "input_sha256": dict(self.input_sha256),
                "selected_arm_id": self.selected_arm_id}


def _string_tuple(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        _fail(ReasonCode.MISSING_FIELD, f"{label} must be a list of strings")
    return tuple(values)


@dataclass(frozen=True)
class RegionRecord:
    """One evaluation region and what the evidence actually supports there."""

    region_id: str
    status: RegionStatus
    detail: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {"region_id": self.region_id, "status": self.status.value,
                "detail": self.detail}


def _parse_region(raw: Any, position: int) -> RegionRecord:
    label = f"decision.unresolved_regions[{position}]"
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.WRONG_TYPE,
              f"{label} must be a mapping with a region_id and a status; a bare "
              "region name cannot say whether its mass is bounded or unknown")
    _exact_fields(raw, {"region_id", "status", "detail"}, label)
    require_visible_text(raw["region_id"], f"{label}.region_id")
    if not isinstance(raw["detail"], str):
        _fail(ReasonCode.WRONG_TYPE, f"{label}.detail must be a string")
    try:
        status = RegionStatus(raw["status"])
    except ValueError:
        _fail(ReasonCode.INVALID_REGION_STATUS,
              f"{label}.status must be one of {[s.value for s in RegionStatus]}")
    if status is RegionStatus.RESOLVED_POPULATION:
        _fail(ReasonCode.INVALID_REGION_STATUS,
              f"{label}: a resolved region does not belong in unresolved_regions")
    return RegionRecord(raw["region_id"], status, raw["detail"])


def _parse_regions(values: Any) -> tuple[RegionRecord, ...]:
    if not isinstance(values, list):
        _fail(ReasonCode.WRONG_TYPE, "decision.unresolved_regions must be a list")
    regions = tuple(_parse_region(row, i) for i, row in enumerate(values))
    identifiers = [region.region_id for region in regions]
    if len(set(identifiers)) != len(identifiers):
        _fail(ReasonCode.DUPLICATE_REGION_ID, "each region may appear once")
    return regions


def _parse_reasons(values: Any) -> tuple[ReasonCode, ...]:
    if not isinstance(values, list):
        _fail(ReasonCode.UNKNOWN_REASON_CODE, "decision.reasons must be a list of reason codes")
    parsed = []
    for value in values:
        try:
            parsed.append(ReasonCode(value))
        except ValueError:
            _fail(ReasonCode.UNKNOWN_REASON_CODE,
                  f"{value!r} is not a declared reason code; free text is not evidence")
    return tuple(parsed)


def _check_decision_consistency(statuses: Mapping[str, Enum], selected_arm_id: Any,
                                reasons=(), inputs=None, protocols=()) -> None:
    if (statuses["integrity"] is Integrity.FAIL
            and statuses["decision"] is not DecisionOutcome.INVALID_INPUT):
        _fail(ReasonCode.INTEGRITY_OUTCOME_CONFLICT,
              "a failed integrity check can only produce INVALID_INPUT")
    # Both outcomes that name a winner must actually name one. A provisional
    # choice with no arm is not a weaker claim, it is an incoherent record.
    if statuses["decision"] in _OUTCOMES_NAMING_AN_ARM and not (
            isinstance(selected_arm_id, str)
            and has_visible_characters(selected_arm_id)):
        _fail(ReasonCode.MISSING_SELECTED_ARM,
              f"{statuses['decision'].value} must name the arm it selects, "
              f"visibly; got {selected_arm_id!r}")
    if statuses["decision"] is not DecisionOutcome.CONFIRMED_FOR_DECLARED_PANEL:
        return
    if not inputs:
        _fail(ReasonCode.MISSING_FIELD,
              "a confirmation must record the digests of the artifacts it confirms")
    if not protocols:
        _fail(ReasonCode.MISSING_FIELD,
              "a confirmation must name the protocols that were actually compared")
    blocking = sorted({reason.value for reason in reasons} & _BLOCKING_REASONS)
    if blocking:
        _fail(ReasonCode.CONFIRMATION_BLOCKED,
              f"a confirmation cannot also cite blocking reasons {blocking}")
    for name, required in _CONFIRMATION_REQUIRES.items():
        if statuses[name] is not required:
            _fail(ReasonCode.CONFIRMATION_BLOCKED,
                  f"confirmation requires {name}={required.value}, got {statuses[name].value}")
