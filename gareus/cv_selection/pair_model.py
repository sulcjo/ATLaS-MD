"""The frozen (CV1, CV2) pair: which component, against which anchor, with what certificate.

A `PairModel` is what production loads. It binds to a `CandidateSet` and a
`FeatureSchema` by digest, embeds the anchor definition by contents, and
carries the independence certificate. Its parser refuses a stored pair whose
own certificate says it is not orthogonal under its training measure, a
coupling into the anchor larger than the anchor's own stiffness, or a
fold-specific GENPEPT preset -- each of those is a broken artifact, not a
weak one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ._base import (ReasonCode, _Artifact, _exact_fields, _fail, _hex64, _positive_int,
                    _schema, artifact_digest, canonical_cv_definition, require_visible_text)

PAIR_MODEL_VERSION_V1 = "atlas-cv-selection-pair-model-v1"
PAIR_MODEL_VERSION_V2 = "atlas-cv-selection-pair-model-v2"
PAIR_MODEL_VERSIONS = (PAIR_MODEL_VERSION_V1, PAIR_MODEL_VERSION_V2)
#: The legacy name kept for readers of v1 artifacts; the selector writes V2.
PAIR_MODEL_VERSION = PAIR_MODEL_VERSION_V1
CERTIFICATE_VERSION_V2 = "residual_certificate_v2"

#: The certificate is a fixed vocabulary so every downstream report reads the
#: same keys. Values are validated by the rules in `_parse_certificate`.
CERTIFICATE_FIELDS = frozenset({
    "cov_q_weighted", "r2_z2_given_z1_mean", "r2_z2_given_z1_se", "r2_z1_given_z2_mean",
    "coupling_curvature_kcal", "coupling_fraction_of_k1", "std_unweighted_z2",
    "design_measure", "n_frames", "n_seed_families", "half_split_agrees",
    "selected_gain_nats", "max_gain_nats",
})

#: v2 certificate: orthogonality is stated against the TRANSFORMED anchor T(a), which the
#: fitted regressors span; the raw-anchor covariance is reported, not certified (spec F02).
CERTIFICATE_FIELDS_V2 = (CERTIFICATE_FIELDS - {"cov_q_weighted"}) | {
    "certificate_version", "cov_transformed_anchor_weighted", "cov_raw_anchor_weighted",
}

#: Orthogonality under the training measure is exact by construction; anything
#: above this is a bug in the producer, not statistical noise.
MAX_CERTIFIED_COVARIANCE = 1e-8

DEPLOYMENT_FIELDS = frozenset({"topology_sha256", "physical_system_sha256", "contact_pair_list_sha256", "deployable"})


def _parse_deployment(raw: Any) -> dict[str, Any]:
    """Real bindings or an explicit ``deployable: false``; a placeholder digest with
    ``deployable: true`` is refused (spec F02: placeholders only in nondeployable artifacts)."""
    label = "pair model.deployment"
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.WRONG_TYPE, f"{label} must be a mapping")
    _exact_fields(raw, set(DEPLOYMENT_FIELDS), label)
    if not isinstance(raw["deployable"], bool):
        _fail(ReasonCode.WRONG_TYPE, f"{label}.deployable must be a boolean")
    out: dict[str, Any] = {"deployable": raw["deployable"]}
    for key in ("topology_sha256", "physical_system_sha256", "contact_pair_list_sha256"):
        out[key] = _hex64(raw[key], f"{label}.{key}", allow_none=True)
        if raw["deployable"] and out[key] is None:
            _fail(ReasonCode.MISSING_FIELD,
                  f"{label}.{key} is required for a deployable pair; mark deployable=false for a "
                  "discovery-only artifact instead")
    return out


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(ReasonCode.INVALID_ESTIMATE, f"{label} must be a number, got {value!r}")
    return float(value)


def _parse_certificate(raw: Any, version: str) -> dict[str, Any]:
    label = "pair model.certificate"
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.WRONG_TYPE, f"{label} must be a mapping")
    v2 = version == PAIR_MODEL_VERSION_V2
    _exact_fields(raw, set(CERTIFICATE_FIELDS_V2 if v2 else CERTIFICATE_FIELDS), label)
    out: dict[str, Any] = {}
    numeric = ["r2_z2_given_z1_mean", "r2_z2_given_z1_se", "r2_z1_given_z2_mean",
               "coupling_curvature_kcal", "coupling_fraction_of_k1", "std_unweighted_z2",
               "selected_gain_nats", "max_gain_nats"]
    numeric += ["cov_transformed_anchor_weighted", "cov_raw_anchor_weighted"] if v2 else ["cov_q_weighted"]
    for key in numeric:
        out[key] = _number(raw[key], f"{label}.{key}")
    certified = out["cov_transformed_anchor_weighted"] if v2 else out["cov_q_weighted"]
    if abs(certified) > MAX_CERTIFIED_COVARIANCE:
        _fail(ReasonCode.INVALID_ESTIMATE,
              f"{label}: certified covariance {certified:.3e} exceeds {MAX_CERTIFIED_COVARIANCE}: a residual "
              "component is orthogonal to its fitted regressors by construction; this artifact was not "
              "produced by the fit")
    if v2:
        if raw["certificate_version"] != CERTIFICATE_VERSION_V2:
            _fail(ReasonCode.INVALID_ENUM, f"{label}.certificate_version must be {CERTIFICATE_VERSION_V2!r}")
        out["certificate_version"] = CERTIFICATE_VERSION_V2
    if out["coupling_fraction_of_k1"] < 0.0 or out["coupling_fraction_of_k1"] > 1.0:
        _fail(ReasonCode.INVALID_ESTIMATE,
              f"{label}.coupling_fraction_of_k1 must lie in [0, 1]; above 1 the CV2 umbrella "
              "restrains the anchor harder than the CV1 umbrella does")
    if out["std_unweighted_z2"] <= 0.0:
        _fail(ReasonCode.NONPOSITIVE_SCALE, f"{label}.std_unweighted_z2 must be positive")
    require_visible_text(raw["design_measure"], f"{label}.design_measure")
    out["design_measure"] = raw["design_measure"]
    out["n_frames"] = _positive_int(raw["n_frames"], f"{label}.n_frames", ReasonCode.INVALID_ESTIMATE)
    out["n_seed_families"] = _positive_int(raw["n_seed_families"], f"{label}.n_seed_families",
                                           ReasonCode.INVALID_ESTIMATE)
    if not isinstance(raw["half_split_agrees"], bool):
        _fail(ReasonCode.WRONG_TYPE, f"{label}.half_split_agrees must be a boolean")
    out["half_split_agrees"] = raw["half_split_agrees"]
    return out


def _parse_runner_ups(raw: Any, selected: int, max_index: int) -> tuple[dict, ...]:
    label = "pair model.runner_ups"
    if not isinstance(raw, list):
        _fail(ReasonCode.WRONG_TYPE, f"{label} must be a list")
    seen = set()
    out = []
    for i, row in enumerate(raw):
        item = f"{label}[{i}]"
        if not isinstance(row, Mapping):
            _fail(ReasonCode.WRONG_TYPE, f"{item} must be a mapping")
        _exact_fields(row, {"component_index", "reason", "scores"}, item)
        index = row["component_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= max_index:
            _fail(ReasonCode.COMPONENT_INDEX_OUT_OF_RANGE, f"{item}.component_index out of range")
        if index == selected or index in seen:
            _fail(ReasonCode.DUPLICATE_COMPONENT_INDEX,
                  f"{item}: component {index} is the winner or listed twice")
        seen.add(index)
        require_visible_text(row["reason"], f"{item}.reason")
        if not isinstance(row["scores"], Mapping):
            _fail(ReasonCode.WRONG_TYPE, f"{item}.scores must be a mapping")
        out.append({"component_index": index, "reason": row["reason"], "scores": dict(row["scores"])})
    return tuple(out)


@dataclass(frozen=True)
class PairModel(_Artifact):
    candidate_set_sha256: str
    feature_schema_sha256: str
    anchor: dict[str, Any]
    selected_component_index: int
    degree: int
    certificate: dict[str, Any]
    runner_ups: tuple[dict, ...]
    genpept_preset: str
    schema_version: str
    basis_transform: Optional[dict[str, Any]]      # v2 only; a v1 pair defers to its candidate set
    deployment: dict[str, Any]                      # v1: {"deployable": False, digests None}
    sha256: str

    @property
    def native_blind_library(self) -> bool:
        """True when the seed library's GENPEPT preset carries no fold-specific bank."""
        from .protocol import NATIVE_BLIND_GENERATOR_PRESETS
        return self.genpept_preset in NATIVE_BLIND_GENERATOR_PRESETS

    @property
    def legacy(self) -> bool:
        return self.schema_version == PAIR_MODEL_VERSION_V1

    @property
    def legacy_training_certificate(self) -> bool:
        """A v1 certificate describes the historical (unclipped) training calculation, not
        necessarily the deployed coordinate; it is never upgraded to a v2 claim."""
        return self.legacy

    @property
    def deployable(self) -> bool:
        return bool(self.deployment.get("deployable", False))

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "PairModel":
        from .contracts import MAX_COMPONENT_INDEX, NATIVE_BLIND_CV_KINDS, _parse_basis_transform
        version = str(data.get("schema", ""))
        if version not in PAIR_MODEL_VERSIONS:
            _fail(ReasonCode.UNKNOWN_SCHEMA_VERSION,
                  f"pair model schema must be one of {list(PAIR_MODEL_VERSIONS)}, got {version!r}")
        required = {"schema", "candidate_set_sha256", "feature_schema_sha256", "anchor",
                    "selected_component_index", "degree", "certificate", "runner_ups", "genpept_preset"}
        if version == PAIR_MODEL_VERSION_V2:
            required |= {"basis_transform", "deployment"}
        _exact_fields(data, required, "pair model")
        cs_digest = _hex64(data["candidate_set_sha256"], "pair model.candidate_set_sha256")
        fs_digest = _hex64(data["feature_schema_sha256"], "pair model.feature_schema_sha256")
        anchor = canonical_cv_definition(data["anchor"], "pair model.anchor")
        if anchor["kind"] not in NATIVE_BLIND_CV_KINDS:
            _fail(ReasonCode.NATIVE_DERIVED_INPUT,
                  f"anchor kind {anchor['kind']!r} is outside the native-blind dictionary")
        selected = data["selected_component_index"]
        if (isinstance(selected, bool) or not isinstance(selected, int)
                or not 1 <= selected <= MAX_COMPONENT_INDEX):
            _fail(ReasonCode.COMPONENT_INDEX_OUT_OF_RANGE,
                  f"selected_component_index must be 1..{MAX_COMPONENT_INDEX}, got {selected!r}")
        degree = data["degree"]
        if degree not in (1, 2) or isinstance(degree, bool):
            _fail(ReasonCode.INVALID_ENUM, "pair model.degree must be 1 or 2")
        preset = data["genpept_preset"]
        require_visible_text(preset, "pair model.genpept_preset")
        # Recorded, not enforced: a fold-biased preset is a documented property of the pair
        # (see ``PairModel.native_blind_library``), not a parse error. 2026-09-21 user ruling.
        certificate = _parse_certificate(data["certificate"], version)
        runner_ups = _parse_runner_ups(data["runner_ups"], selected, MAX_COMPONENT_INDEX)
        body = {"schema": version, "candidate_set_sha256": cs_digest,
                "feature_schema_sha256": fs_digest, "anchor": anchor,
                "selected_component_index": selected, "degree": degree,
                "certificate": certificate, "runner_ups": list(runner_ups),
                "genpept_preset": preset}
        if version == PAIR_MODEL_VERSION_V2:
            raw_t = data["basis_transform"]
            if not isinstance(raw_t, Mapping) or set(raw_t) != {"kind", "lo", "hi"}:
                _fail(ReasonCode.MISSING_FIELD, "pair model.basis_transform needs exactly kind, lo, hi")
            clamp = (raw_t["lo"], raw_t["hi"])
            transform = _parse_basis_transform(raw_t, degree, clamp, "pair model.basis_transform")
            deployment = _parse_deployment(data["deployment"])
            body["basis_transform"] = dict(transform)
            body["deployment"] = dict(deployment)
        else:
            transform = None
            deployment = {"topology_sha256": None, "physical_system_sha256": None,
                          "contact_pair_list_sha256": None, "deployable": False}
        return cls(cs_digest, fs_digest, anchor, selected, degree, certificate, runner_ups,
                   preset, version, transform, deployment, artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        body = {"schema": self.schema_version, "candidate_set_sha256": self.candidate_set_sha256,
                "feature_schema_sha256": self.feature_schema_sha256, "anchor": dict(self.anchor),
                "selected_component_index": self.selected_component_index, "degree": self.degree,
                "certificate": dict(self.certificate), "runner_ups": [dict(r) for r in self.runner_ups],
                "genpept_preset": self.genpept_preset}
        if self.schema_version == PAIR_MODEL_VERSION_V2:
            body["basis_transform"] = dict(self.basis_transform or {})
            body["deployment"] = dict(self.deployment)
        return body
