"""Primitive validation, digests and the shared artifact base.

Everything here is mechanism: how a selector artifact is canonicalised, hashed
and refused. The scientific vocabulary lives in :mod:`vocabulary` and the
artifacts themselves in :mod:`contracts`, which is the public façade.

Canonical encoding, duplicate-key rejection and non-finite rejection are not
reimplemented: they already exist in ``gareus.correctness._io`` and are reused
so that one encoder serves the selector and the existing correctness APIs.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

from ..correctness._io import IntegrityError, digest, json_bytes, json_loads


class ReasonCode(str, Enum):
    """Stable machine-readable rejection reasons."""

    UNKNOWN_SCHEMA_VERSION = "UNKNOWN_SCHEMA_VERSION"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_ENUM = "INVALID_ENUM"
    INVALID_ATOM_QUADRUPLET = "INVALID_ATOM_QUADRUPLET"
    INVALID_QUANTILE = "INVALID_QUANTILE"
    NONPOSITIVE_SCALE = "NONPOSITIVE_SCALE"
    TOLERANCE_NOT_POSITIVE = "TOLERANCE_NOT_POSITIVE"
    FEATURE_INDEX_NOT_CONTIGUOUS = "FEATURE_INDEX_NOT_CONTIGUOUS"
    FEATURE_IDENTITY_MISMATCH = "FEATURE_IDENTITY_MISMATCH"
    FEATURE_WIDTH_MISMATCH = "FEATURE_WIDTH_MISMATCH"
    DUPLICATE_FEATURE_NAME = "DUPLICATE_FEATURE_NAME"
    DUPLICATE_COMPONENT_INDEX = "DUPLICATE_COMPONENT_INDEX"
    COMPONENT_INDEX_OUT_OF_RANGE = "COMPONENT_INDEX_OUT_OF_RANGE"
    INCOMPLETE_REGRESSION = "INCOMPLETE_REGRESSION"
    PRIMARY_PANEL_SIZE = "PRIMARY_PANEL_SIZE"
    DUPLICATE_OBSERVABLE_ID = "DUPLICATE_OBSERVABLE_ID"
    NATIVE_DERIVED_INPUT = "NATIVE_DERIVED_INPUT"
    STUDY_ROLE_PHASE_CONFLICT = "STUDY_ROLE_PHASE_CONFLICT"
    ARTIFACT_DIGEST_MISMATCH = "ARTIFACT_DIGEST_MISMATCH"
    MISSING_ARTIFACT = "MISSING_ARTIFACT"
    PROTOCOL_NOT_READY = "PROTOCOL_NOT_READY"
    UNKNOWN_REASON_CODE = "UNKNOWN_REASON_CODE"
    MISSING_SELECTED_ARM = "MISSING_SELECTED_ARM"
    CONFIRMATION_BLOCKED = "CONFIRMATION_BLOCKED"
    INTEGRITY_OUTCOME_CONFLICT = "INTEGRITY_OUTCOME_CONFLICT"
    ARM_WITHOUT_BUDGET = "ARM_WITHOUT_BUDGET"
    UNEQUAL_ARM_COST = "UNEQUAL_ARM_COST"
    DUPLICATE_ARM_ID = "DUPLICATE_ARM_ID"
    REPLICA_CAP_EXCEEDED = "REPLICA_CAP_EXCEEDED"
    LAYOUT_INCONSISTENT = "LAYOUT_INCONSISTENT"
    UNRESOLVED_CORRELATION = "UNRESOLVED_CORRELATION"
    INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ContractError(IntegrityError):
    """A persisted selector artifact cannot be read as what it claims to be."""

    def __init__(self, reason: ReasonCode, message: str) -> None:
        super().__init__(f"[{reason.value}] {message}")
        self.reason = reason


def _fail(reason: ReasonCode, message: str) -> None:
    raise ContractError(reason, message)

# ---------------------------------------------------------------------------
# Primitive validation
# ---------------------------------------------------------------------------

def _canonical(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip through the repository encoder: rejects NaN and odd types."""
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.UNKNOWN_FIELD, "artifact must be a mapping")
    return json_loads(json_bytes(dict(raw)))


def _exact_fields(data: Mapping[str, Any], required: set[str], label: str) -> None:
    unknown = sorted(set(data) - required)
    if unknown:
        _fail(ReasonCode.UNKNOWN_FIELD, f"{label} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(data))
    if missing:
        _fail(ReasonCode.MISSING_FIELD, f"{label} is missing field(s): {', '.join(missing)}")


def _schema(data: Mapping[str, Any], expected: str, label: str) -> None:
    if data.get("schema") != expected:
        _fail(ReasonCode.UNKNOWN_SCHEMA_VERSION,
              f"{label} requires schema {expected!r}, got {data.get('schema')!r}")


def _enum(value: Any, enum_cls: type[Enum], label: str) -> Any:
    try:
        return enum_cls(value)
    except ValueError:
        _fail(ReasonCode.INVALID_ENUM, f"{label}: {value!r} is not a valid {enum_cls.__name__}")


def _hex64(value: Any, label: str, *, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        _fail(ReasonCode.MISSING_FIELD, f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _positive_int(value: Any, label: str, reason: ReasonCode) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(reason, f"{label} must be a positive integer, got {value!r}")
    return value


def _positive_float(value: Any, label: str, reason: ReasonCode) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        _fail(reason, f"{label} must be a positive number, got {value!r}")
    return float(value)


def _finite_vector(values: Any, label: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        _fail(ReasonCode.MISSING_FIELD, f"{label} must be a nonempty numeric sequence")
    out = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            _fail(ReasonCode.MISSING_FIELD, f"{label} must contain numbers only")
        out.append(float(item))
    return tuple(out)


def artifact_digest(payload: Mapping[str, Any]) -> str:
    """Digest of the canonical encoding, excluding any embedded ``sha256``."""
    body = {key: value for key, value in _canonical(payload).items() if key != "sha256"}
    return digest(json_bytes(body))


def verify_artifact_digest(payload: Mapping[str, Any]) -> str:
    """Confirm a stored artifact still hashes to its recorded digest."""
    recorded = payload.get("sha256")
    actual = artifact_digest(payload)
    if recorded != actual:
        _fail(ReasonCode.ARTIFACT_DIGEST_MISMATCH,
              f"artifact digest {recorded!r} does not match its contents ({actual})")
    return actual

class _Artifact:
    """Shared load/serialize behaviour; subclasses supply ``_parse``/``_body``."""

    sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]):
        data = _canonical(raw)
        data.pop("sha256", None)
        return cls._parse(data)

    @classmethod
    def from_json_bytes(cls, data: bytes | str):
        return cls.from_mapping(json_loads(data))

    def to_mapping(self) -> dict[str, Any]:
        body = _canonical(self._body())
        return {**body, "sha256": artifact_digest(body)}

    def to_json_bytes(self) -> bytes:
        return json_bytes(self.to_mapping())

    @classmethod
    def _parse(cls, data: dict[str, Any]):  # pragma: no cover - abstract
        raise NotImplementedError

    def _body(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError
