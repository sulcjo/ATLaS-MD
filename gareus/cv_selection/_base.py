"""Primitive validation, digests and the shared artifact base.

Everything here is mechanism: how a selector artifact is canonicalised, hashed
and refused. The scientific vocabulary lives in :mod:`vocabulary` and the
artifacts themselves in :mod:`contracts`, which is the public façade.

Canonical encoding, duplicate-key rejection and non-finite rejection are not
reimplemented: they already exist in ``gareus.correctness._io`` and are reused
so that one encoder serves the selector and the existing correctness APIs.
"""
from __future__ import annotations

import unicodedata
from enum import Enum
from typing import Any, Mapping, Sequence

from ..correctness._io import IntegrityError, digest, json_bytes, json_loads

class ReasonCode(str, Enum):
    """Stable machine-readable rejection reasons."""

    UNKNOWN_SCHEMA_VERSION = "UNKNOWN_SCHEMA_VERSION"
    UNKNOWN_FIELD = "UNKNOWN_FIELD"
    MISSING_FIELD = "MISSING_FIELD"
    INVALID_ENUM = "INVALID_ENUM"
    DUPLICATE_JSON_KEY = "DUPLICATE_JSON_KEY"
    NONFINITE_VALUE = "NONFINITE_VALUE"
    MALFORMED_ARTIFACT_JSON = "MALFORMED_ARTIFACT_JSON"
    PATH_BASED_IDENTITY = "PATH_BASED_IDENTITY"
    INVALID_UNITS = "INVALID_UNITS"
    INVALID_CV_DEFINITION = "INVALID_CV_DEFINITION"
    NOT_POSITIVE = "NOT_POSITIVE"
    NEGATIVE_VALUE = "NEGATIVE_VALUE"
    WRONG_TYPE = "WRONG_TYPE"
    EMPTY_STRING = "EMPTY_STRING"
    READINESS_INCONSISTENT = "READINESS_INCONSISTENT"
    INVALID_REGION_STATUS = "INVALID_REGION_STATUS"
    DUPLICATE_REGION_ID = "DUPLICATE_REGION_ID"
    FREE_ENERGY_UNRESOLVED = "FREE_ENERGY_UNRESOLVED"
    NOVELTY_UNRESOLVED = "NOVELTY_UNRESOLVED"
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


#: Unicode categories that render as nothing: separators, control characters
#: and *format* characters. ``str.strip()`` removes only whitespace, so
#: U+200B ZERO WIDTH SPACE, U+FEFF, U+2060 WORD JOINER and U+00AD SOFT HYPHEN
#: all survive it. A name made only of those is invisible in every log, diff
#: and terminal while still comparing unequal to the empty string — worse than
#: a blank, because a reader sees a confirmed verdict naming nothing.
_INVISIBLE_CATEGORIES = frozenset({"Zs", "Zl", "Zp", "Cc", "Cf"})


def has_visible_characters(text: str) -> bool:
    """True when at least one character actually renders."""
    return any(unicodedata.category(char) not in _INVISIBLE_CATEGORIES for char in text)


def require_visible_text(value: Any, label: str) -> str:
    """A human-readable identifier must be readable by a human."""
    if not isinstance(value, str):
        _fail(ReasonCode.WRONG_TYPE, f"{label} must be a string, got {type(value).__name__}")
    if not has_visible_characters(value):
        _fail(ReasonCode.EMPTY_STRING,
              f"{label} must contain at least one visible character; got {value!r}")
    return value

# ---------------------------------------------------------------------------
# Primitive validation
# ---------------------------------------------------------------------------

#: Message prefixes raised by ``gareus.correctness._io``, mapped to selector
#: reason codes. This couples to that module's wording on purpose and is pinned
#: by tests: the alternative is either reimplementing its duplicate-key and
#: non-finite detection (two encoders that can drift) or leaving these paths
#: with no machine-readable code at all, which defeats "branch on the code,
#: never on message text".
_IO_REASONS = (
    ("Duplicate JSON key", ReasonCode.DUPLICATE_JSON_KEY),
    ("Nonfinite JSON number", ReasonCode.NONFINITE_VALUE),
    ("Nonfinite number", ReasonCode.NONFINITE_VALUE),
    ("must be finite", ReasonCode.NONFINITE_VALUE),
    ("Invalid UTF-8 JSON", ReasonCode.MALFORMED_ARTIFACT_JSON),
)

#: Same idea for ``state_identity._canonical_cv``, which owns CV-definition
#: identity (contents, never a filename) and the supported unit vocabulary.
_CV_REASONS = (
    ("embed model contents instead of", ReasonCode.PATH_BASED_IDENTITY),
    ("units not supported", ReasonCode.INVALID_UNITS),
)


def translate_integrity_error(exc: IntegrityError, table, default: ReasonCode,
                              label: str) -> "ContractError":
    """Give a borrowed validator's refusal a stable selector reason code."""
    if isinstance(exc, ContractError):
        return exc
    message = str(exc)
    for needle, reason in table:
        if needle in message:
            return ContractError(reason, f"{label}: {message}")
    return ContractError(default, f"{label}: {message}")


def _canonical(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Round-trip through the repository encoder: rejects NaN and odd types."""
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.WRONG_TYPE, "artifact must be a mapping")
    try:
        return json_loads(json_bytes(dict(raw)))
    except IntegrityError as exc:
        raise translate_integrity_error(
            exc, _IO_REASONS, ReasonCode.MALFORMED_ARTIFACT_JSON, "artifact") from exc


def canonical_cv_definition(raw: Any, label: str) -> dict[str, Any]:
    """Validate a CV definition through the existing state-identity rules.

    ``state_identity._canonical_cv`` already refuses filename-based identity and
    unknown units; this wrapper only attaches the reason code.
    """
    from ..correctness.state_identity import _canonical_cv
    try:
        return _canonical_cv(raw, label)
    except IntegrityError as exc:
        raise translate_integrity_error(
            exc, _CV_REASONS, ReasonCode.INVALID_CV_DEFINITION, label) from exc


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
        """Parse, then hold the payload to its own claimed identity.

        Identity is parsed *content*. A payload that carries a ``sha256`` must
        therefore agree with what parsing makes of it. Without this check a
        hand-written artifact whose JSON says ``1`` where the schema means
        ``1.0`` hashes one way on disk and another way once loaded, so a
        protocol that declares the on-disk digest can never match the loaded
        artifact -- and a tampered payload is silently re-hashed to whatever it
        now contains. Both failures are silent, which is the worst kind.
        """
        data = _canonical(raw)
        claimed = data.pop("sha256", None)
        parsed = cls._parse(data)
        if claimed is not None and claimed != parsed.sha256:
            _fail(ReasonCode.ARTIFACT_DIGEST_MISMATCH,
                  f"{cls.__name__} claims digest {claimed} but its contents parse to "
                  f"{parsed.sha256}; edit the artifact through its writer, not by hand")
        return parsed

    @classmethod
    def from_json_bytes(cls, data: bytes | str):
        try:
            loaded = json_loads(data)
        except IntegrityError as exc:
            raise translate_integrity_error(
                exc, _IO_REASONS, ReasonCode.MALFORMED_ARTIFACT_JSON,
                cls.__name__) from exc
        return cls.from_mapping(loaded)

    def to_mapping(self) -> dict[str, Any]:
        """Serialize, after proving the result would survive being read back.

        Dataclasses are publicly constructible, and the consistency rules live
        in ``_parse``. Without this round-trip a producer could hand-build a
        ``Decision`` that says ``integrity=FAIL`` and
        ``CONFIRMED_FOR_DECLARED_PANEL`` at once, write it with a perfectly
        valid digest, and the contradiction would only surface later, in
        whichever downstream task happened to load it. Refusing at the write
        end puts the error where it was made.
        """
        body = _canonical(self._body())
        self._parse(dict(body))
        return {**body, "sha256": artifact_digest(body)}

    def to_json_bytes(self) -> bytes:
        return json_bytes(self.to_mapping())

    @classmethod
    def _parse(cls, data: dict[str, Any]):  # pragma: no cover - abstract
        raise NotImplementedError

    def _body(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError
