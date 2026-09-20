"""Versioned artifacts for equilibrium CV selection, and the rules they enforce.

This module is the public façade: later tasks import everything they need from
``gareus.cv_selection.contracts``. Mechanism lives in :mod:`._base` and the
controlled vocabularies in :mod:`.vocabulary`; both are re-exported here.

Design rules this module exists to keep (plan sections 3, 5, 7):

* Identity is *contents*, never a filename. Digests are taken over the existing
  canonical JSON encoding from ``gareus.correctness._io``, excluding the
  artifact's own ``sha256`` field, so one encoder serves every artifact.
* A study role (screen/confirm/...) is not a sampling phase. The existing
  correctness API only treats ``phase_kind == "production"`` as equilibrium
  eligible, and a screening campaign's measurement phase really is production.
  Two separate vocabularies keep that invariant intact.
* Nothing here invents a scientific default. An unresolved policy produces a
  readiness failure that names the task which must produce the artifact.
* A decision carries eight independent statuses. Precision is not accuracy, so
  collapsing them into one pass/fail would let a reproducibly-trapped arm be
  reported as confirmed.

Every rejection raises :class:`ContractError` carrying a stable
:class:`ReasonCode`, so callers branch on the code rather than on message text.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ..correctness._io import IntegrityError, json_bytes, json_loads
from ._base import (ContractError, ReasonCode, _Artifact, _canonical, _enum,
                    _exact_fields, _fail, _finite_vector, _hex64, _positive_float,
                    _positive_int, _schema, artifact_digest, canonical_cv_definition,
                    has_visible_characters, require_visible_text, verify_artifact_digest)
from .decision import (DECISION_VERSION, Decision, RegionRecord)
from .protocol import (NATIVE_BLIND_GENERATOR_PRESETS, PROTOCOL_VERSION,
                       READINESS_REPORT_VERSION, MissingRequirement, ProtocolSpec,
                       ReadinessReport, validate_protocol)
from .vocabulary import (Advantage, CrossProtocol, DecisionOutcome, Dependence,
                         ExitCode, Integrity, PhaseKind, Precision, Reproducibility,
                         RegionStatus, SearchMode, Stage, StudyRole, Support,
                         measurement_phase_for, validate_role_phase)

__all__ = [
    "Advantage", "CandidateComponent", "CandidateSet", "ContractError", "CrossProtocol",
    "Decision", "DecisionOutcome", "Dependence", "ExitCode", "FeatureId", "FeatureSchema",
    "Integrity", "IntegrityError", "MissingRequirement", "ObservablePanel", "PhaseKind",
    "Precision", "PrimaryObservable", "ProtocolSpec", "ReadinessReport", "ReasonCode",
    "RegionStatus", "Reproducibility", "SearchMode", "Stage", "StudyRole", "Support",
    "TrialArm", "RegionRecord",
    "TrialPlan", "artifact_digest", "json_bytes", "json_loads", "measurement_phase_for",
    "require_feature_binding", "validate_protocol", "validate_role_phase",
    "verify_artifact_digest",
]

FEATURE_SCHEMA_VERSION = "atlas-cv-selection-feature-schema-v1"
CANDIDATE_SET_VERSION = "atlas-cv-selection-candidate-set-v1"
OBSERVABLE_PANEL_VERSION = "atlas-cv-selection-observable-panel-v1"
TRIAL_PLAN_VERSION = "atlas-cv-selection-trial-plan-v1"

CANDIDATE_KIND_QUADRATIC_RESIDUAL = "quadratic-residual-torsion-pc-v1"

#: The declared native-blind primitive dictionary (plan section 6.1). A CV kind
#: outside this set is refused rather than trusted, because the blindness
#: declaration is three strings in the protocol and cannot, by itself, see that
#: a primary coordinate is an RMSD to the experimentally known fold. This is a
#: fail-closed allowlist: widening it is a deliberate, reviewable act.
NATIVE_BLIND_CV_KINDS = frozenset({
    "nonlocal-contact-fraction",
    "radius-of-gyration",
    "end-to-end-distance",
    "torsion-pca-component",
    "quadratic-residual-torsion-pc",
})

#: Components are individual SVD directions numbered from one. The legacy
#: ``compute_bootstrap_torsion_pca(component=...)`` argument is a *count* of
#: leading PCs combined; these two meanings must never be interchanged.
MAX_COMPONENT_INDEX = 6

#: The frozen primary panel is eight CDF probabilities (plan section 7.1).
PRIMARY_PANEL_SIZE = 8

def _version_map(values: Any, label: str) -> dict[str, str]:
    """Library name -> version. An SVD basis is not reproducible without it."""
    if not isinstance(values, Mapping) or not values:
        _fail(ReasonCode.MISSING_FIELD, f"{label} must be a nonempty mapping")
    for name, version in values.items():
        require_visible_text(name, f"{label} key")
        require_visible_text(version, f"{label}[{name}]")
    return dict(values)


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

_FEATURE_FIELDS = {"index", "name", "torsion_name", "residue_index",
                   "atom_indices", "trig", "dihedral_sign_convention"}
_TRIG_VALUES = frozenset({"sin", "cos"})
_SIGN_CONVENTIONS = frozenset({"negated", "direct"})


@dataclass(frozen=True)
class FeatureId:
    """One scalar feature: which torsion, which trig function, which atoms."""

    index: int
    name: str
    torsion_name: str
    residue_index: int
    atom_indices: tuple[int, int, int, int]
    trig: str
    dihedral_sign_convention: str

    def to_mapping(self) -> dict[str, Any]:
        return {"index": self.index, "name": self.name, "torsion_name": self.torsion_name,
                "residue_index": self.residue_index, "atom_indices": list(self.atom_indices),
                "trig": self.trig, "dihedral_sign_convention": self.dihedral_sign_convention}


def _parse_feature(raw: Mapping[str, Any], position: int) -> FeatureId:
    label = f"feature[{position}]"
    _exact_fields(raw, _FEATURE_FIELDS, label)
    if raw["index"] != position or isinstance(raw["index"], bool):
        _fail(ReasonCode.FEATURE_INDEX_NOT_CONTIGUOUS,
              f"{label}.index must equal its position {position}, got {raw['index']!r}")
    if raw["trig"] not in _TRIG_VALUES:
        _fail(ReasonCode.INVALID_ENUM, f"{label}.trig must be sin or cos, got {raw['trig']!r}")
    if raw["dihedral_sign_convention"] not in _SIGN_CONVENTIONS:
        _fail(ReasonCode.INVALID_ENUM,
              f"{label}.dihedral_sign_convention must be one of {sorted(_SIGN_CONVENTIONS)}")
    atoms = raw["atom_indices"]
    if (not isinstance(atoms, list) or len(atoms) != 4
            or any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in atoms)
            or len(set(atoms)) != 4):
        _fail(ReasonCode.INVALID_ATOM_QUADRUPLET,
              f"{label}.atom_indices must be four distinct nonnegative atom indices")
    for key in ("name", "torsion_name"):
        if not isinstance(raw[key], str) or not raw[key]:
            _fail(ReasonCode.MISSING_FIELD, f"{label}.{key} must be a nonempty string")
    if isinstance(raw["residue_index"], bool) or not isinstance(raw["residue_index"], int):
        _fail(ReasonCode.MISSING_FIELD, f"{label}.residue_index must be an integer")
    return FeatureId(position, raw["name"], raw["torsion_name"], raw["residue_index"],
                     (atoms[0], atoms[1], atoms[2], atoms[3]), raw["trig"],
                     raw["dihedral_sign_convention"])


@dataclass(frozen=True)
class FeatureSchema(_Artifact):
    """The ordered canonical feature vector. Width alone is never identity."""

    topology_sha256: str
    features: tuple[FeatureId, ...]
    sha256: str

    @property
    def width(self) -> int:
        return len(self.features)

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "FeatureSchema":
        _exact_fields(data, {"schema", "topology_sha256", "features"}, "feature schema")
        _schema(data, FEATURE_SCHEMA_VERSION, "feature schema")
        rows = data["features"]
        if not isinstance(rows, list) or not rows:
            _fail(ReasonCode.MISSING_FIELD, "feature schema needs at least one feature")
        features = tuple(_parse_feature(row, position) for position, row in enumerate(rows))
        names = [feature.name for feature in features]
        if len(set(names)) != len(names):
            _fail(ReasonCode.DUPLICATE_FEATURE_NAME, "feature names must be unique")
        topology = _hex64(data["topology_sha256"], "feature schema.topology_sha256")
        body = {"schema": FEATURE_SCHEMA_VERSION, "topology_sha256": topology,
                "features": [feature.to_mapping() for feature in features]}
        return cls(topology, features, artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        return {"schema": FEATURE_SCHEMA_VERSION, "topology_sha256": self.topology_sha256,
                "features": [feature.to_mapping() for feature in self.features]}


# ---------------------------------------------------------------------------
# Candidate set
# ---------------------------------------------------------------------------

_COMPONENT_FIELDS = {"component_index", "singular_value", "eigenvalue_tie_flagged",
                     "right_singular_vector", "residual_mean", "regression_coefficients",
                     "primary_mean", "primary_std", "projection_mean", "projection_std"}


@dataclass(frozen=True)
class CandidateComponent:
    """One individual residual direction, never a mixture of leading PCs."""

    component_index: int
    singular_value: float
    eigenvalue_tie_flagged: bool
    right_singular_vector: tuple[float, ...]
    residual_mean: tuple[float, ...]
    regression_coefficients: tuple[tuple[float, ...], ...]
    primary_mean: float
    primary_std: float
    projection_mean: float
    projection_std: float

    @property
    def width(self) -> int:
        return len(self.right_singular_vector)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "component_index": self.component_index,
            "singular_value": self.singular_value,
            "eigenvalue_tie_flagged": self.eigenvalue_tie_flagged,
            "right_singular_vector": list(self.right_singular_vector),
            "residual_mean": list(self.residual_mean),
            "regression_coefficients": [list(row) for row in self.regression_coefficients],
            "primary_mean": self.primary_mean, "primary_std": self.primary_std,
            "projection_mean": self.projection_mean, "projection_std": self.projection_std,
        }


def _parse_component(raw: Mapping[str, Any], position: int) -> CandidateComponent:
    label = f"component[{position}]"
    _exact_fields(raw, _COMPONENT_FIELDS, label)
    index = raw["component_index"]
    if (isinstance(index, bool) or not isinstance(index, int)
            or not 1 <= index <= MAX_COMPONENT_INDEX):
        _fail(ReasonCode.COMPONENT_INDEX_OUT_OF_RANGE,
              f"{label}.component_index must be 1..{MAX_COMPONENT_INDEX}, got {index!r}")
    if not isinstance(raw["eigenvalue_tie_flagged"], bool):
        _fail(ReasonCode.MISSING_FIELD, f"{label}.eigenvalue_tie_flagged must be a boolean")
    vector = _finite_vector(raw["right_singular_vector"], f"{label}.right_singular_vector")
    mean = _finite_vector(raw["residual_mean"], f"{label}.residual_mean")
    if len(mean) != len(vector):
        _fail(ReasonCode.FEATURE_WIDTH_MISMATCH,
              f"{label}.residual_mean has width {len(mean)}, expected {len(vector)}")
    rows = raw["regression_coefficients"]
    if not isinstance(rows, list) or len(rows) != 3:
        _fail(ReasonCode.INCOMPLETE_REGRESSION,
              f"{label}.regression_coefficients needs exactly the constant, linear and "
              "quadratic rows of the primary-dependent subtraction")
    coefficients = []
    for order, row in enumerate(rows):
        values = _finite_vector(row, f"{label}.regression_coefficients[{order}]")
        if len(values) != len(vector):
            _fail(ReasonCode.FEATURE_WIDTH_MISMATCH,
                  f"{label}.regression_coefficients[{order}] has width {len(values)}, "
                  f"expected {len(vector)}")
        coefficients.append(values)
    if not any(abs(value) > 0.0 for value in vector):
        _fail(ReasonCode.NONPOSITIVE_SCALE,
              f"{label}.right_singular_vector is all zero; a zero direction defines no "
              "coordinate and its umbrella would apply no force")
    singular = raw["singular_value"]
    if isinstance(singular, bool) or not isinstance(singular, (int, float)) or singular < 0:
        _fail(ReasonCode.NONPOSITIVE_SCALE, f"{label}.singular_value must be nonnegative")
    for key in ("primary_mean", "projection_mean"):
        if isinstance(raw[key], bool) or not isinstance(raw[key], (int, float)):
            _fail(ReasonCode.MISSING_FIELD, f"{label}.{key} must be a number")
    primary_std = _positive_float(raw["primary_std"], f"{label}.primary_std",
                                  ReasonCode.NONPOSITIVE_SCALE)
    projection_std = _positive_float(raw["projection_std"], f"{label}.projection_std",
                                     ReasonCode.NONPOSITIVE_SCALE)
    return CandidateComponent(index, float(singular), raw["eigenvalue_tie_flagged"], vector,
                              mean, tuple(coefficients), float(raw["primary_mean"]), primary_std,
                              float(raw["projection_mean"]), projection_std)


@dataclass(frozen=True)
class CandidateSet(_Artifact):
    """Frozen residual components plus the primary CV they were fitted against."""

    kind: str
    feature_schema_sha256: str
    physical_system_sha256: str
    training_rows_sha256: str
    library_versions: dict[str, str]
    primary_definition: dict[str, Any]
    components: tuple[CandidateComponent, ...]
    sha256: str

    @property
    def width(self) -> int:
        return self.components[0].width

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "CandidateSet":
        _exact_fields(data, {"schema", "kind", "feature_schema_sha256", "physical_system_sha256",
                             "training_rows_sha256", "library_versions", "primary_definition",
                             "components"}, "candidate set")
        _schema(data, CANDIDATE_SET_VERSION, "candidate set")
        if data["kind"] != CANDIDATE_KIND_QUADRATIC_RESIDUAL:
            _fail(ReasonCode.INVALID_ENUM,
                  f"candidate set kind must be {CANDIDATE_KIND_QUADRATIC_RESIDUAL!r}")
        rows = data["components"]
        if not isinstance(rows, list) or not rows:
            _fail(ReasonCode.MISSING_FIELD, "candidate set needs at least one component")
        components = tuple(_parse_component(row, position) for position, row in enumerate(rows))
        indices = [component.component_index for component in components]
        if len(set(indices)) != len(indices):
            _fail(ReasonCode.DUPLICATE_COMPONENT_INDEX,
                  "each component index may appear once; these are individual directions")
        widths = {component.width for component in components}
        if len(widths) != 1:
            _fail(ReasonCode.FEATURE_WIDTH_MISMATCH,
                  f"components disagree on feature width: {sorted(widths)}")
        # An SVD basis is only reproducible against the library that produced it.
        libraries = _version_map(data["library_versions"], "candidate set.library_versions")
        # Reuse the existing CV canonicaliser: it rejects filename-based identity.
        primary = canonical_cv_definition(data["primary_definition"],
                                          "candidate set.primary_definition")
        if primary["kind"] not in NATIVE_BLIND_CV_KINDS:
            _fail(ReasonCode.NATIVE_DERIVED_INPUT,
                  f"primary CV kind {primary['kind']!r} is not in the declared native-blind "
                  f"dictionary {sorted(NATIVE_BLIND_CV_KINDS)}; a coordinate defined against a "
                  "known fold cannot enter a native-blind selection")
        body = {"schema": CANDIDATE_SET_VERSION, "kind": data["kind"],
                "feature_schema_sha256": _hex64(data["feature_schema_sha256"],
                                                "candidate set.feature_schema_sha256"),
                "physical_system_sha256": _hex64(data["physical_system_sha256"],
                                                 "candidate set.physical_system_sha256"),
                "training_rows_sha256": _hex64(data["training_rows_sha256"],
                                               "candidate set.training_rows_sha256"),
                "library_versions": dict(libraries),
                "primary_definition": primary,
                "components": [component.to_mapping() for component in components]}
        return cls(data["kind"], body["feature_schema_sha256"], body["physical_system_sha256"],
                   body["training_rows_sha256"], dict(libraries), primary, components,
                   artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        return {"schema": CANDIDATE_SET_VERSION, "kind": self.kind,
                "feature_schema_sha256": self.feature_schema_sha256,
                "physical_system_sha256": self.physical_system_sha256,
                "training_rows_sha256": self.training_rows_sha256,
                "library_versions": dict(self.library_versions),
                "primary_definition": dict(self.primary_definition),
                "components": [component.to_mapping() for component in self.components]}


def require_feature_binding(candidates: CandidateSet, schema: FeatureSchema) -> None:
    """A candidate set is only interpretable against the schema it was fitted on."""
    if candidates.feature_schema_sha256 != schema.sha256:
        _fail(ReasonCode.FEATURE_IDENTITY_MISMATCH,
              f"candidate set was fitted against feature schema "
              f"{candidates.feature_schema_sha256} but was given {schema.sha256}")
    if candidates.width != schema.width:
        _fail(ReasonCode.FEATURE_WIDTH_MISMATCH,
              f"candidate coefficients have width {candidates.width}, "
              f"feature schema declares {schema.width}")


# ---------------------------------------------------------------------------
# Observable panel
# ---------------------------------------------------------------------------

_OBSERVABLE_FIELDS = {"observable_id", "kind", "source_observable",
                      "discovery_quantile", "halfwidth_tolerance"}
_OBSERVABLE_KINDS = frozenset({"cdf_probability"})
_QUANTILE_ALGORITHMS = frozenset({"numpy.quantile.linear"})

#: A probability tolerance above one is not a tolerance, it is a way to make
#: every arm pass. Half-widths and practical-difference bands are bounded.
_MAX_PROBABILITY_TOLERANCE = 1.0


def _probability_tolerance(value: Any, label: str) -> float:
    tolerance = _positive_float(value, label, ReasonCode.TOLERANCE_NOT_POSITIVE)
    if tolerance > _MAX_PROBABILITY_TOLERANCE:
        _fail(ReasonCode.TOLERANCE_NOT_POSITIVE,
              f"{label} is a probability tolerance and must be <= "
              f"{_MAX_PROBABILITY_TOLERANCE}; {tolerance} would make the gate vacuous")
    return tolerance


def _require_panel_shape(primary: tuple["PrimaryObservable", ...]) -> None:
    """Eight copies of one measurement is not the declared panel.

    Plan section 7.1 pins the primary panel to both global-shape observables at
    four discovery quantiles each. Enforcing only the count would let a panel be
    eight repeats of a single easy probability, which passes precision trivially
    while measuring almost nothing.
    """
    pairs = [(item.source_observable, item.discovery_quantile) for item in primary]
    if len(set(pairs)) != len(pairs):
        _fail(ReasonCode.DUPLICATE_OBSERVABLE_ID,
              "each (source observable, quantile) pair may appear once in the primary panel")
    for source in _SOURCE_OBSERVABLES:
        quantiles = {q for src, q in pairs if src == source}
        if len(quantiles) != PRIMARY_PANEL_SIZE // len(_SOURCE_OBSERVABLES):
            _fail(ReasonCode.PRIMARY_PANEL_SIZE,
                  f"the primary panel needs "
                  f"{PRIMARY_PANEL_SIZE // len(_SOURCE_OBSERVABLES)} distinct quantiles for "
                  f"{source}, got {len(quantiles)}")
_SOURCE_OBSERVABLES = frozenset({"rg_nm", "end_to_end_nm"})


@dataclass(frozen=True)
class PrimaryObservable:
    observable_id: str
    kind: str
    source_observable: str
    discovery_quantile: float
    halfwidth_tolerance: float

    def to_mapping(self) -> dict[str, Any]:
        return {"observable_id": self.observable_id, "kind": self.kind,
                "source_observable": self.source_observable,
                "discovery_quantile": self.discovery_quantile,
                "halfwidth_tolerance": self.halfwidth_tolerance}


def _parse_observable(raw: Mapping[str, Any], position: int) -> PrimaryObservable:
    label = f"primary[{position}]"
    _exact_fields(raw, _OBSERVABLE_FIELDS, label)
    if raw["kind"] not in _OBSERVABLE_KINDS:
        _fail(ReasonCode.INVALID_ENUM, f"{label}.kind must be one of {sorted(_OBSERVABLE_KINDS)}")
    if raw["source_observable"] not in _SOURCE_OBSERVABLES:
        _fail(ReasonCode.INVALID_ENUM,
              f"{label}.source_observable must be one of {sorted(_SOURCE_OBSERVABLES)}")
    if not isinstance(raw["observable_id"], str) or not raw["observable_id"]:
        _fail(ReasonCode.MISSING_FIELD, f"{label}.observable_id must be a nonempty string")
    quantile = raw["discovery_quantile"]
    if (isinstance(quantile, bool) or not isinstance(quantile, (int, float))
            or not 0.0 < float(quantile) < 1.0):
        _fail(ReasonCode.INVALID_QUANTILE,
              f"{label}.discovery_quantile must lie strictly inside (0, 1), got {quantile!r}")
    tolerance = _probability_tolerance(raw["halfwidth_tolerance"],
                                       f"{label}.halfwidth_tolerance")
    return PrimaryObservable(raw["observable_id"], raw["kind"], raw["source_observable"],
                             float(quantile), tolerance)


@dataclass(frozen=True)
class ObservablePanel(_Artifact):
    """Frozen before any arm is scored; a candidate never gets its own panel."""

    primary: tuple[PrimaryObservable, ...]
    discovery_rows_sha256: str
    quantile_algorithm: str
    cross_protocol_tolerance: float
    diagnostic_partition_centers: int
    diagnostic_histogram_bins: int
    novelty_mass_tolerance: float
    sha256: str

    @property
    def interval_family_size(self) -> int:
        """Multiplicity family for the simultaneous primary intervals."""
        return len(self.primary)

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "ObservablePanel":
        _exact_fields(data, {"schema", "primary", "cross_protocol_tolerance",
                             "diagnostic_partition_centers", "diagnostic_histogram_bins",
                             "novelty_mass_tolerance", "discovery_rows_sha256",
                             "quantile_algorithm"}, "observable panel")
        _schema(data, OBSERVABLE_PANEL_VERSION, "observable panel")
        rows = data["primary"]
        if not isinstance(rows, list) or len(rows) != PRIMARY_PANEL_SIZE:
            _fail(ReasonCode.PRIMARY_PANEL_SIZE,
                  f"the frozen primary panel holds exactly {PRIMARY_PANEL_SIZE} observables, "
                  f"got {len(rows) if isinstance(rows, list) else type(rows).__name__}")
        primary = tuple(_parse_observable(row, position) for position, row in enumerate(rows))
        identifiers = [observable.observable_id for observable in primary]
        if len(set(identifiers)) != len(identifiers):
            _fail(ReasonCode.DUPLICATE_OBSERVABLE_ID, "primary observable ids must be unique")
        _require_panel_shape(primary)
        cross = _probability_tolerance(data["cross_protocol_tolerance"],
                                       "observable panel.cross_protocol_tolerance")
        novelty = _probability_tolerance(data["novelty_mass_tolerance"],
                                         "observable panel.novelty_mass_tolerance")
        centers = _positive_int(data["diagnostic_partition_centers"],
                                "observable panel.diagnostic_partition_centers",
                                ReasonCode.MISSING_FIELD)
        bins = _positive_int(data["diagnostic_histogram_bins"],
                             "observable panel.diagnostic_histogram_bins", ReasonCode.MISSING_FIELD)
        rows_digest = _hex64(data["discovery_rows_sha256"],
                             "observable panel.discovery_rows_sha256")
        algorithm = data["quantile_algorithm"]
        if algorithm not in _QUANTILE_ALGORITHMS:
            _fail(ReasonCode.INVALID_ENUM,
                  f"observable panel.quantile_algorithm must be one of "
                  f"{sorted(_QUANTILE_ALGORITHMS)}; a quantile is only reproducible "
                  "against a named interpolation rule")
        body = {"schema": OBSERVABLE_PANEL_VERSION,
                "primary": [observable.to_mapping() for observable in primary],
                "discovery_rows_sha256": rows_digest, "quantile_algorithm": algorithm,
                "cross_protocol_tolerance": cross, "diagnostic_partition_centers": centers,
                "diagnostic_histogram_bins": bins, "novelty_mass_tolerance": novelty}
        return cls(primary, rows_digest, algorithm, cross, centers, bins, novelty,
                   artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        return {"schema": OBSERVABLE_PANEL_VERSION,
                "primary": [observable.to_mapping() for observable in self.primary],
                "discovery_rows_sha256": self.discovery_rows_sha256,
                "quantile_algorithm": self.quantile_algorithm,
                "cross_protocol_tolerance": self.cross_protocol_tolerance,
                "diagnostic_partition_centers": self.diagnostic_partition_centers,
                "diagnostic_histogram_bins": self.diagnostic_histogram_bins,
                "novelty_mass_tolerance": self.novelty_mass_tolerance}


# ---------------------------------------------------------------------------
# Trial plan
# ---------------------------------------------------------------------------

_ARM_FIELDS = {"arm_id", "secondary_component_index", "campaigns", "gpu_hours_per_campaign"}


@dataclass(frozen=True)
class TrialArm:
    arm_id: str
    secondary_component_index: int | None
    campaigns: int
    gpu_hours_per_campaign: float

    def to_mapping(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id,
                "secondary_component_index": self.secondary_component_index,
                "campaigns": self.campaigns,
                "gpu_hours_per_campaign": self.gpu_hours_per_campaign}


def _parse_arm(raw: Mapping[str, Any], position: int) -> TrialArm:
    label = f"arm[{position}]"
    _exact_fields(raw, _ARM_FIELDS, label)
    require_visible_text(raw["arm_id"], f"{label}.arm_id")
    index = raw["secondary_component_index"]
    if index is not None and (isinstance(index, bool) or not isinstance(index, int)
                              or not 1 <= index <= MAX_COMPONENT_INDEX):
        _fail(ReasonCode.COMPONENT_INDEX_OUT_OF_RANGE,
              f"{label}.secondary_component_index must be null or 1..{MAX_COMPONENT_INDEX}")
    campaigns = _positive_int(raw["campaigns"], f"{label}.campaigns",
                              ReasonCode.ARM_WITHOUT_BUDGET)
    hours = _positive_float(raw["gpu_hours_per_campaign"], f"{label}.gpu_hours_per_campaign",
                            ReasonCode.ARM_WITHOUT_BUDGET)
    return TrialArm(raw["arm_id"], index, campaigns, hours)


@dataclass(frozen=True)
class TrialPlan(_Artifact):
    """A compiled, launchable set of arms. Compiling one never launches it."""

    plan_id: str
    study_id: str
    stage: Stage
    study_role: StudyRole
    measurement_phase_kind: PhaseKind
    protocol_sha256: str
    layout: str
    lambda_rungs: int
    states_per_arm: int
    max_replicas: int
    arms: tuple[TrialArm, ...]
    sha256: str

    @property
    def total_gpu_hours(self) -> float:
        return sum(arm.campaigns * arm.gpu_hours_per_campaign for arm in self.arms)

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "TrialPlan":
        _exact_fields(data, {"schema", "plan_id", "study_id", "stage", "study_role",
                             "measurement_phase_kind", "protocol_sha256", "layout",
                             "lambda_rungs", "states_per_arm", "max_replicas", "arms"},
                      "trial plan")
        _schema(data, TRIAL_PLAN_VERSION, "trial plan")
        stage = _enum(data["stage"], Stage, "trial plan.stage")
        role = _enum(data["study_role"], StudyRole, "trial plan.study_role")
        phase = _enum(data["measurement_phase_kind"], PhaseKind,
                      "trial plan.measurement_phase_kind")
        validate_role_phase(role, phase)
        rungs = _positive_int(data["lambda_rungs"], "trial plan.lambda_rungs",
                              ReasonCode.LAYOUT_INCONSISTENT)
        cap = _positive_int(data["max_replicas"], "trial plan.max_replicas",
                            ReasonCode.REPLICA_CAP_EXCEEDED)
        states = _positive_int(data["states_per_arm"], "trial plan.states_per_arm",
                               ReasonCode.LAYOUT_INCONSISTENT)
        if states > cap:
            _fail(ReasonCode.REPLICA_CAP_EXCEEDED,
                  f"{states} states exceed the {cap}-replica cap for one campaign")
        if states % rungs:
            _fail(ReasonCode.LAYOUT_INCONSISTENT,
                  f"{states} states do not divide evenly into {rungs} lambda rungs")
        arms = _parse_arms(data["arms"])
        body = {"schema": TRIAL_PLAN_VERSION, "plan_id": data["plan_id"],
                "study_id": data["study_id"], "stage": stage.value, "study_role": role.value,
                "measurement_phase_kind": phase.value,
                "protocol_sha256": _hex64(data["protocol_sha256"], "trial plan.protocol_sha256"),
                "layout": data["layout"], "lambda_rungs": rungs, "states_per_arm": states,
                "max_replicas": cap, "arms": [arm.to_mapping() for arm in arms]}
        for key in ("plan_id", "study_id", "layout"):
            require_visible_text(body[key], f"trial plan.{key}")
        return cls(body["plan_id"], body["study_id"], stage, role, phase,
                   body["protocol_sha256"], body["layout"], rungs, states, cap, arms,
                   artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        return {"schema": TRIAL_PLAN_VERSION, "plan_id": self.plan_id, "study_id": self.study_id,
                "stage": self.stage.value, "study_role": self.study_role.value,
                "measurement_phase_kind": self.measurement_phase_kind.value,
                "protocol_sha256": self.protocol_sha256, "layout": self.layout,
                "lambda_rungs": self.lambda_rungs, "states_per_arm": self.states_per_arm,
                "max_replicas": self.max_replicas,
                "arms": [arm.to_mapping() for arm in self.arms]}


def _parse_arms(rows: Any) -> tuple[TrialArm, ...]:
    if not isinstance(rows, list) or not rows:
        _fail(ReasonCode.ARM_WITHOUT_BUDGET, "a trial plan needs at least one arm")
    arms = tuple(_parse_arm(row, position) for position, row in enumerate(rows))
    identifiers = [arm.arm_id for arm in arms]
    if len(set(identifiers)) != len(identifiers):
        _fail(ReasonCode.DUPLICATE_ARM_ID, "arm ids must be unique")
    components = [arm.secondary_component_index for arm in arms]
    if len(set(components)) != len(components):
        _fail(ReasonCode.DUPLICATE_COMPONENT_INDEX,
              "each component (and the one-CV baseline) gets exactly one arm")
    if len({arm.campaigns for arm in arms}) != 1 or len(
            {arm.gpu_hours_per_campaign for arm in arms}) != 1:
        _fail(ReasonCode.UNEQUAL_ARM_COST,
              "arms are compared at matched cost; equalise campaigns and GPU-hours")
    return arms


