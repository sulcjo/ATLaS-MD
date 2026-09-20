"""The verdict artifact: recorded numbers, and statuses re-derived from them.

Precision is not accuracy. A sampling run that is reproducibly trapped in the
wrong basin has tiny confidence intervals, perfect run-to-run agreement and a
badly biased mean, so a verdict collapsed to one pass/fail number would report
it as confirmed. Every status here therefore stays separate.

Separation alone is not enough. A status that the producer simply *types in*
is a claim, and a consistency rule over claims polices the grammar of a lie,
not its truth. So the two statuses that follow from numbers are **derived**,
and the parser recomputes them from the recorded evidence and refuses a
mismatch:

* ``precision`` follows from the selected arm's estimates:
  ``max_a (h_a / eps_a)^2 <= 1`` gives ``MET``, any unresolved half-width gives
  ``UNRESOLVED``, otherwise ``NOT_MET``.
* ``cross_protocol`` follows from the interval rule of plan section 6.4 applied
  to every pair of arms and every observable: a difference interval
  ``[d - (h_A + h_B), d + (h_A + h_B)]`` entirely inside ``[-delta, +delta]``
  supports agreement, entirely outside establishes disagreement, anything else
  is unresolved. One established disagreement anywhere dominates.

``dependence``, ``reproducibility`` and ``support`` rest on batch covariances
and per-campaign estimates that this artifact does not carry, so they remain
**asserted** and are named as such. A region marked ``MASS_BOUNDED_SMALL`` must
carry the bound and the method that produced it; without those it is only a
claim that the mass is small.

What this buys: fabricating a verdict now means fabricating *numbers*, tied to
artifact digests, which an auditor can re-derive and cross-check. It does not
make a fabricated number true. That is the most a contract layer can do.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any, Mapping

from ._base import (ReasonCode, _Artifact, _enum, _exact_fields, _fail, _hex64,
                    _positive_int, _schema, artifact_digest, has_visible_characters,
                    probability_tolerance, require_visible_text)
from .vocabulary import (Advantage, CrossProtocol, DecisionOutcome, Dependence,
                         ExitCode, Integrity, Precision, RegionStatus,
                         Reproducibility, Support)

DECISION_VERSION = "atlas-cv-selection-decision-v1"

#: A confirmation rests on these four artifacts. Naming them is the minimum
#: link between a verdict and the things it claims to have compared.
CONFIRMATION_REQUIRED_INPUTS = frozenset({
    "protocol", "trial_plan", "observable_panel", "candidate_set",
})

#: The only way a zero-visit region may currently be called small. The exact
#: one-sided bound ``1 - alpha**(1/n)`` holds for IID draws from the target;
#: it does not hold for weighted replica-exchange samples, and no Kish
#: effective sample size may be substituted for ``n``. A method outside this
#: set is refused rather than trusted, so widening it is a reviewable act.
SUPPORTED_BOUND_METHODS = frozenset({"exact_iid_binomial_zero_count"})

#: Statuses this artifact re-derives from its own numbers.
DERIVED_STATUSES = frozenset({"precision", "cross_protocol"})

#: Statuses this artifact can only carry as the producer's assertion.
ASSERTED_STATUSES = frozenset({"integrity", "dependence", "reproducibility",
                               "support", "advantage"})

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


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservableEstimate:
    """One primary observable for one arm: the number, its half-width, its bar."""

    observable_id: str
    estimate: float
    halfwidth: float | None
    tolerance: float
    n_campaigns: int

    def to_mapping(self) -> dict[str, Any]:
        return {"observable_id": self.observable_id, "estimate": self.estimate,
                "halfwidth": self.halfwidth, "tolerance": self.tolerance,
                "n_campaigns": self.n_campaigns}


@dataclass(frozen=True)
class ArmEvidence:
    """Every primary estimate one arm produced, at matched cost."""

    arm_id: str
    estimates: tuple[ObservableEstimate, ...]

    @property
    def observable_ids(self) -> frozenset[str]:
        return frozenset(item.observable_id for item in self.estimates)

    def by_observable(self) -> dict[str, ObservableEstimate]:
        return {item.observable_id: item for item in self.estimates}

    def to_mapping(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id,
                "estimates": [item.to_mapping() for item in self.estimates]}


def _unit_interval(value: Any, label: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(ReasonCode.INVALID_ESTIMATE, f"{label} must be a number, got {value!r}")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        _fail(ReasonCode.INVALID_ESTIMATE,
              f"{label}={number} is not a probability; the primary panel is CDF probabilities")
    return number


def _parse_estimate(raw: Any, label: str) -> ObservableEstimate:
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.WRONG_TYPE, f"{label} must be a mapping")
    _exact_fields(raw, {"observable_id", "estimate", "halfwidth", "tolerance", "n_campaigns"},
                  label)
    require_visible_text(raw["observable_id"], f"{label}.observable_id")
    estimate = _unit_interval(raw["estimate"], f"{label}.estimate")
    halfwidth = _unit_interval(raw["halfwidth"], f"{label}.halfwidth", allow_none=True)
    tolerance = probability_tolerance(raw["tolerance"], f"{label}.tolerance")
    campaigns = _positive_int(raw["n_campaigns"], f"{label}.n_campaigns",
                              ReasonCode.INVALID_ESTIMATE)
    return ObservableEstimate(raw["observable_id"], estimate, halfwidth, tolerance, campaigns)


def _parse_arm_evidence(raw: Any, position: int) -> ArmEvidence:
    label = f"decision.evidence[{position}]"
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.WRONG_TYPE, f"{label} must be a mapping")
    _exact_fields(raw, {"arm_id", "estimates"}, label)
    require_visible_text(raw["arm_id"], f"{label}.arm_id")
    rows = raw["estimates"]
    if not isinstance(rows, list) or not rows:
        _fail(ReasonCode.EVIDENCE_INCOMPLETE,
              f"{label}.estimates must hold at least one primary estimate")
    estimates = tuple(_parse_estimate(row, f"{label}.estimates[{i}]")
                      for i, row in enumerate(rows))
    identifiers = [item.observable_id for item in estimates]
    if len(set(identifiers)) != len(identifiers):
        _fail(ReasonCode.DUPLICATE_OBSERVABLE_ID,
              f"{label} reports the same observable twice")
    return ArmEvidence(raw["arm_id"], estimates)


def _parse_evidence(values: Any, protocols: tuple[str, ...]) -> tuple[ArmEvidence, ...]:
    if not isinstance(values, list):
        _fail(ReasonCode.WRONG_TYPE, "decision.evidence must be a list")
    arms = tuple(_parse_arm_evidence(row, i) for i, row in enumerate(values))
    identifiers = [arm.arm_id for arm in arms]
    if len(set(identifiers)) != len(identifiers):
        _fail(ReasonCode.DUPLICATE_ARM_ID, "decision.evidence names an arm twice")
    untested = sorted(set(identifiers) - set(protocols))
    if untested:
        _fail(ReasonCode.EVIDENCE_INCOMPLETE,
              f"evidence is recorded for arms {untested} that tested_protocol_ids does not "
              "list; an estimate cannot come from a protocol that was not run")
    panels = {arm.observable_ids for arm in arms}
    if len(panels) > 1:
        _fail(ReasonCode.EVIDENCE_INCOMPLETE,
              "every arm must report the same frozen primary panel; arms with different "
              "observable sets cannot be compared")
    return arms


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

def derive_precision(selected: ArmEvidence | None) -> Precision:
    """``max_a (h_a/eps_a)^2 <= 1`` over the selected arm's primary panel."""
    if selected is None:
        return Precision.UNRESOLVED
    if any(item.halfwidth is None for item in selected.estimates):
        return Precision.UNRESOLVED
    if all(item.halfwidth <= item.tolerance for item in selected.estimates):
        return Precision.MET
    return Precision.NOT_MET


def _pair_verdict(left: ObservableEstimate, right: ObservableEstimate,
                  delta: float) -> CrossProtocol:
    if left.halfwidth is None or right.halfwidth is None:
        return CrossProtocol.AGREEMENT_UNRESOLVED
    difference = left.estimate - right.estimate
    spread = left.halfwidth + right.halfwidth
    low, high = difference - spread, difference + spread
    if -delta <= low and high <= delta:
        return CrossProtocol.AGREEMENT_SUPPORTED
    if low > delta or high < -delta:
        return CrossProtocol.PROTOCOL_DISAGREEMENT
    return CrossProtocol.AGREEMENT_UNRESOLVED


def derive_cross_protocol(evidence: tuple[ArmEvidence, ...],
                          delta: float | None) -> CrossProtocol:
    """Plan section 6.4, over every pair of arms and every primary observable."""
    if len(evidence) < 2:
        return CrossProtocol.NOT_COMPARED
    if delta is None:
        _fail(ReasonCode.EVIDENCE_INCOMPLETE,
              "comparing two or more arms needs decision.cross_protocol_tolerance")
    verdicts: set[CrossProtocol] = set()
    for left, right in combinations(evidence, 2):
        left_by, right_by = left.by_observable(), right.by_observable()
        for observable_id in left_by:
            verdicts.add(_pair_verdict(left_by[observable_id], right_by[observable_id], delta))
    if CrossProtocol.PROTOCOL_DISAGREEMENT in verdicts:
        return CrossProtocol.PROTOCOL_DISAGREEMENT
    if verdicts == {CrossProtocol.AGREEMENT_SUPPORTED}:
        return CrossProtocol.AGREEMENT_SUPPORTED
    return CrossProtocol.AGREEMENT_UNRESOLVED


def _require_derived(name: str, declared: Enum, derived: Enum) -> None:
    if declared is not derived:
        _fail(ReasonCode.EVIDENCE_STATUS_MISMATCH,
              f"decision.{name} is declared {declared.value} but the recorded evidence "
              f"derives {derived.value}; a status is a summary of numbers, not an assertion")


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionRecord:
    """One evaluation region and what the evidence actually supports there."""

    region_id: str
    status: RegionStatus
    upper_bound: float | None
    bound_method: str | None
    detail: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {"region_id": self.region_id, "status": self.status.value,
                "upper_bound": self.upper_bound, "bound_method": self.bound_method,
                "detail": self.detail}


def _parse_region(raw: Any, position: int) -> RegionRecord:
    label = f"decision.unresolved_regions[{position}]"
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.WRONG_TYPE,
              f"{label} must be a mapping with a region_id and a status; a bare "
              "region name cannot say whether its mass is bounded or unknown")
    _exact_fields(raw, {"region_id", "status", "upper_bound", "bound_method", "detail"}, label)
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
    bound, method = raw["upper_bound"], raw["bound_method"]
    if status is RegionStatus.MASS_BOUNDED_SMALL:
        if bound is None or method is None:
            _fail(ReasonCode.UNSUPPORTED_BOUND_METHOD,
                  f"{label}: MASS_BOUNDED_SMALL is a claim about a bound; record the bound "
                  "and the method that produced it, or mark the region UNRESOLVED_SUPPORT")
        if method not in SUPPORTED_BOUND_METHODS:
            _fail(ReasonCode.UNSUPPORTED_BOUND_METHOD,
                  f"{label}.bound_method {method!r} is not a supported bound; "
                  f"known: {sorted(SUPPORTED_BOUND_METHODS)}. No Kish-ESS shortcut.")
        bound = _unit_interval(bound, f"{label}.upper_bound")
    elif bound is not None or method is not None:
        _fail(ReasonCode.INVALID_REGION_STATUS,
              f"{label}: an UNRESOLVED_SUPPORT region cannot also carry a bound")
    return RegionRecord(raw["region_id"], status, bound, method, raw["detail"])


def _parse_regions(values: Any) -> tuple[RegionRecord, ...]:
    if not isinstance(values, list):
        _fail(ReasonCode.WRONG_TYPE, "decision.unresolved_regions must be a list")
    regions = tuple(_parse_region(row, i) for i, row in enumerate(values))
    identifiers = [region.region_id for region in regions]
    if len(set(identifiers)) != len(identifiers):
        _fail(ReasonCode.DUPLICATE_REGION_ID, "each region may appear once")
    return regions


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision(_Artifact):
    """Recorded evidence plus eight statuses, two of them re-derived on load."""

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
    evidence: tuple[ArmEvidence, ...]
    cross_protocol_tolerance: float | None
    sha256: str

    @property
    def exit_code(self) -> ExitCode:
        """A valid inconclusive analysis still exits zero (plan section 11)."""
        return (ExitCode.INVALID_INPUT if self.decision is DecisionOutcome.INVALID_INPUT
                else ExitCode.OK)

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "Decision":
        require_visible_text(data.get("study_id"), "decision.study_id")
        _exact_fields(data, {"schema", "study_id", *_DECISION_STATUS_FIELDS, "reasons",
                             "unresolved_regions", "tested_protocol_ids", "input_sha256",
                             "selected_arm_id", "evidence", "cross_protocol_tolerance"},
                      "decision")
        _schema(data, DECISION_VERSION, "decision")
        statuses = {name: _enum(data[name], enum_cls, f"decision.{name}")
                    for name, enum_cls in _DECISION_STATUS_FIELDS.items()}
        reasons = _parse_reasons(data["reasons"])
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
        evidence = _parse_evidence(data["evidence"], protocols)
        delta = data["cross_protocol_tolerance"]
        if delta is not None:
            delta = probability_tolerance(delta, "decision.cross_protocol_tolerance")
        _check_decision_consistency(statuses, selected, reasons, inputs, protocols, regions,
                                    evidence, delta)
        body = _decision_body(data["study_id"], statuses, reasons, regions, protocols,
                              inputs, selected, evidence, delta)
        return cls(data["study_id"], statuses["integrity"], statuses["precision"],
                   statuses["dependence"], statuses["reproducibility"],
                   statuses["cross_protocol"], statuses["support"], statuses["advantage"],
                   statuses["decision"], reasons, regions, protocols, dict(inputs), selected,
                   evidence, delta, artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        statuses = {name: getattr(self, name) for name in _DECISION_STATUS_FIELDS}
        return _decision_body(self.study_id, statuses, self.reasons, self.unresolved_regions,
                              self.tested_protocol_ids, self.input_sha256,
                              self.selected_arm_id, self.evidence,
                              self.cross_protocol_tolerance)


def _decision_body(study_id, statuses, reasons, regions, protocols, inputs, selected,
                   evidence, delta) -> dict[str, Any]:
    return {"schema": DECISION_VERSION, "study_id": study_id,
            **{name: status.value for name, status in statuses.items()},
            "reasons": [reason.value for reason in reasons],
            "unresolved_regions": [region.to_mapping() for region in regions],
            "tested_protocol_ids": list(protocols),
            "input_sha256": dict(inputs), "selected_arm_id": selected,
            "evidence": [arm.to_mapping() for arm in evidence],
            "cross_protocol_tolerance": delta}


def _string_tuple(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        _fail(ReasonCode.MISSING_FIELD, f"{label} must be a list of strings")
    return tuple(values)


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
                                reasons, inputs, protocols, regions, evidence,
                                delta) -> None:
    if (statuses["integrity"] is Integrity.FAIL
            and statuses["decision"] is not DecisionOutcome.INVALID_INPUT):
        _fail(ReasonCode.INTEGRITY_OUTCOME_CONFLICT,
              "a failed integrity check can only produce INVALID_INPUT")
    # Both outcomes that name a winner must actually name one. A provisional
    # choice with no arm is not a weaker claim, it is an incoherent record.
    if statuses["decision"] in _OUTCOMES_NAMING_AN_ARM and not (
            isinstance(selected_arm_id, str) and has_visible_characters(selected_arm_id)):
        _fail(ReasonCode.MISSING_SELECTED_ARM,
              f"{statuses['decision'].value} must name the arm it selects, "
              f"visibly; got {selected_arm_id!r}")
    by_arm = {arm.arm_id: arm for arm in evidence}
    if selected_arm_id is not None and selected_arm_id not in by_arm:
        _fail(ReasonCode.EVIDENCE_INCOMPLETE,
              f"selected arm {selected_arm_id!r} has no recorded evidence; a choice with no "
              "numbers behind it cannot be audited")
    # The two statuses that follow from numbers are recomputed, never trusted.
    _require_derived("precision", statuses["precision"],
                     derive_precision(by_arm.get(selected_arm_id)))
    _require_derived("cross_protocol", statuses["cross_protocol"],
                     derive_cross_protocol(evidence, delta))
    if statuses["decision"] is not DecisionOutcome.CONFIRMED_FOR_DECLARED_PANEL:
        return
    if not inputs:
        _fail(ReasonCode.MISSING_FIELD,
              "a confirmation must record the digests of the artifacts it confirms")
    absent = sorted(CONFIRMATION_REQUIRED_INPUTS - set(inputs))
    if absent:
        _fail(ReasonCode.MISSING_FIELD,
              f"a confirmation must record the digests of {absent}; a verdict that cannot "
              "name what it rested on is not reproducible")
    if not protocols:
        _fail(ReasonCode.MISSING_FIELD,
              "a confirmation must name the protocols that were actually compared")
    unknown_mass = sorted(region.region_id for region in regions
                          if region.status is RegionStatus.UNRESOLVED_SUPPORT)
    if unknown_mass:
        _fail(ReasonCode.CONFIRMATION_BLOCKED,
              f"regions {unknown_mass} have unknown mass; an unqualified confirmation cannot "
              "stand while a relevant region is unresolved (a bounded-small region may)")
    blocking = sorted({reason.value for reason in reasons} & _BLOCKING_REASONS)
    if blocking:
        _fail(ReasonCode.CONFIRMATION_BLOCKED,
              f"a confirmation cannot also cite blocking reasons {blocking}")
    for name, required in _CONFIRMATION_REQUIRES.items():
        if statuses[name] is not required:
            _fail(ReasonCode.CONFIRMATION_BLOCKED,
                  f"confirmation requires {name}={required.value}, got {statuses[name].value}")
