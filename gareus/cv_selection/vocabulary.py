"""The selector's controlled vocabularies.

The important distinction encoded here is between a **study role** (what a body
of sampling is for) and a **sampling phase** (what the existing correctness API
already calls it). ``gareus.correctness.sampling_policy`` treats only
``phase_kind == "production"`` as equilibrium-analysis eligible, and a screening
campaign's measurement phase really is production. Keeping the two vocabularies
disjoint lets the selector describe itself without weakening that invariant.
"""
from __future__ import annotations

from enum import Enum

from ._base import ReasonCode, _fail


class ExitCode(Enum):
    """Process exit codes (plan section 11). Zero covers a valid abstention."""

    OK = 0
    INVALID_INPUT = 2
    UNAVAILABLE_DEPENDENCY = 3
    EXECUTION_FAILURE = 4
    NOT_READY = 5


class Stage(Enum):
    """Readiness levels, in increasing order of what they permit."""

    BUILD = "build"
    ENGINEERING = "engineering"
    SCREEN = "screen"
    CONFIRM = "confirm"

    @property
    def rank(self) -> int:
        return _STAGE_ORDER.index(self)


_STAGE_ORDER = (Stage.BUILD, Stage.ENGINEERING, Stage.SCREEN, Stage.CONFIRM)


class StudyRole(str, Enum):
    """What a body of sampling is *for*. Deliberately disjoint from PhaseKind."""

    DISCOVERY = "discovery"
    CALIBRATION = "calibration"
    ENGINEERING = "engineering"
    SCREEN = "screen"
    CONFIRM = "confirm"


class PhaseKind(str, Enum):
    """Mirrors ``gareus.correctness.sampling_policy`` exactly; do not extend."""

    PRODUCTION = "production"
    PILOT = "pilot"
    EXPLORATION = "exploration"
    EQUILIBRATION = "equilibration"


_ROLE_MEASUREMENT_PHASE = {
    StudyRole.DISCOVERY: PhaseKind.EXPLORATION,
    StudyRole.CALIBRATION: PhaseKind.EXPLORATION,
    StudyRole.ENGINEERING: PhaseKind.PILOT,
    StudyRole.SCREEN: PhaseKind.PRODUCTION,
    StudyRole.CONFIRM: PhaseKind.PRODUCTION,
}


class RegionStatus(str, Enum):
    """Per-region evidence status (adversarial-review finding R6).

    Zero observations in a region is not one situation but two, and collapsing
    them is a scientific error in both directions:

    * ``MASS_BOUNDED_SMALL`` — no visits, but a *valid* upper bound places the
      mass below the declared tolerance. For an IID target region of true
      probability 1e-6 with n=1000, zero visits happens ~99.9% of the time and
      the exact one-sided 95% bound is 1 - 0.05**(1/1000) ~= 0.003, already
      inside a 0.02 tolerance. Demanding a visit here would reject a perfectly
      adequate result.
    * ``UNRESOLVED_SUPPORT`` — no visits and no justified bound. The mass is
      unknown. It must never be reported as zero with zero uncertainty.

    The bound above is exact only for IID sampling from the target. Do not
    substitute a Kish effective sample size into it for weighted
    replica-exchange data; without a defensible bound the status is
    ``UNRESOLVED_SUPPORT``.
    """

    RESOLVED_POPULATION = "RESOLVED_POPULATION"
    MASS_BOUNDED_SMALL = "MASS_BOUNDED_SMALL"
    UNRESOLVED_SUPPORT = "UNRESOLVED_SUPPORT"


class SearchMode(str, Enum):
    JOINT_PAIR = "joint_pair"
    FIXED_PRIMARY = "fixed_primary"


class Integrity(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class Precision(str, Enum):
    MET = "MET"
    NOT_MET = "NOT_MET"
    UNRESOLVED = "UNRESOLVED"


class Dependence(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNRESOLVED_CORRELATION = "UNRESOLVED_CORRELATION"


class Reproducibility(str, Enum):
    NO_CONFLICT_DETECTED = "NO_CONFLICT_DETECTED"
    CONFLICT = "CONFLICT"
    UNRESOLVED = "UNRESOLVED"


class CrossProtocol(str, Enum):
    AGREEMENT_SUPPORTED = "AGREEMENT_SUPPORTED"
    PROTOCOL_DISAGREEMENT = "PROTOCOL_DISAGREEMENT"
    AGREEMENT_UNRESOLVED = "AGREEMENT_UNRESOLVED"
    NOT_COMPARED = "NOT_COMPARED"


class Support(str, Enum):
    SUPPORTED_ON_DECLARED_PANEL = "SUPPORTED_ON_DECLARED_PANEL"
    LIMITED = "LIMITED"
    UNRESOLVED = "UNRESOLVED"


class Advantage(str, Enum):
    NOT_TESTED = "NOT_TESTED"
    SUPPORTED = "SUPPORTED"
    UNRESOLVED = "UNRESOLVED"


class DecisionOutcome(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PROVISIONAL_CHOICE = "PROVISIONAL_CHOICE"
    CONFIRMED_FOR_DECLARED_PANEL = "CONFIRMED_FOR_DECLARED_PANEL"
    PANEL_REVISION_REQUIRED = "PANEL_REVISION_REQUIRED"


def measurement_phase_for(role: StudyRole) -> PhaseKind:
    """The only sampling phase whose records may be scored for that study role."""
    return _ROLE_MEASUREMENT_PHASE[role]


def validate_role_phase(role: StudyRole, phase: PhaseKind) -> None:
    expected = measurement_phase_for(role)
    if phase is not expected:
        _fail(ReasonCode.STUDY_ROLE_PHASE_CONFLICT,
              f"study role {role.value!r} measures in phase {expected.value!r}, not {phase.value!r}")
