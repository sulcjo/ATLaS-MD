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
from typing import Any, Mapping

from ._base import (ReasonCode, _Artifact, _exact_fields, _fail, _hex64, _positive_int,
                    _schema, artifact_digest, canonical_cv_definition, require_visible_text)

PAIR_MODEL_VERSION = "atlas-cv-selection-pair-model-v1"

#: The certificate is a fixed vocabulary so every downstream report reads the
#: same keys. Values are validated by the rules in `_parse_certificate`.
CERTIFICATE_FIELDS = frozenset({
    "cov_q_weighted", "r2_z2_given_z1_mean", "r2_z2_given_z1_se", "r2_z1_given_z2_mean",
    "coupling_curvature_kcal", "coupling_fraction_of_k1", "std_unweighted_z2",
    "design_measure", "n_frames", "n_seed_families", "half_split_agrees",
    "selected_gain_nats", "max_gain_nats",
})

#: Orthogonality under the training measure is exact by construction; anything
#: above this is a bug in the producer, not statistical noise.
MAX_CERTIFIED_COVARIANCE = 1e-8


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(ReasonCode.INVALID_ESTIMATE, f"{label} must be a number, got {value!r}")
    return float(value)


def _parse_certificate(raw: Any) -> dict[str, Any]:
    label = "pair model.certificate"
    if not isinstance(raw, Mapping):
        _fail(ReasonCode.WRONG_TYPE, f"{label} must be a mapping")
    _exact_fields(raw, set(CERTIFICATE_FIELDS), label)
    out: dict[str, Any] = {}
    for key in ("cov_q_weighted", "r2_z2_given_z1_mean", "r2_z2_given_z1_se",
                "r2_z1_given_z2_mean", "coupling_curvature_kcal", "coupling_fraction_of_k1",
                "std_unweighted_z2", "selected_gain_nats", "max_gain_nats"):
        out[key] = _number(raw[key], f"{label}.{key}")
    if abs(out["cov_q_weighted"]) > MAX_CERTIFIED_COVARIANCE:
        _fail(ReasonCode.INVALID_ESTIMATE,
              f"{label}.cov_q_weighted={out['cov_q_weighted']:.3e}: a residual component is "
              "orthogonal to its anchor by construction; this artifact was not produced by the fit")
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
    sha256: str

    @classmethod
    def _parse(cls, data: dict[str, Any]) -> "PairModel":
        from .contracts import MAX_COMPONENT_INDEX, NATIVE_BLIND_CV_KINDS
        from .protocol import NATIVE_BLIND_GENERATOR_PRESETS

        _exact_fields(data, {"schema", "candidate_set_sha256", "feature_schema_sha256", "anchor",
                             "selected_component_index", "degree", "certificate", "runner_ups",
                             "genpept_preset"}, "pair model")
        _schema(data, PAIR_MODEL_VERSION, "pair model")
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
        if preset not in NATIVE_BLIND_GENERATOR_PRESETS:
            _fail(ReasonCode.NATIVE_DERIVED_INPUT,
                  f"GENPEPT preset {preset!r} encodes a fold bias; a pair fitted on it is not native-blind")
        certificate = _parse_certificate(data["certificate"])
        runner_ups = _parse_runner_ups(data["runner_ups"], selected, MAX_COMPONENT_INDEX)
        body = {"schema": PAIR_MODEL_VERSION, "candidate_set_sha256": cs_digest,
                "feature_schema_sha256": fs_digest, "anchor": anchor,
                "selected_component_index": selected, "degree": degree,
                "certificate": certificate, "runner_ups": list(runner_ups),
                "genpept_preset": preset}
        return cls(cs_digest, fs_digest, anchor, selected, degree, certificate, runner_ups,
                   preset, artifact_digest(body))

    def _body(self) -> dict[str, Any]:
        return {"schema": PAIR_MODEL_VERSION, "candidate_set_sha256": self.candidate_set_sha256,
                "feature_schema_sha256": self.feature_schema_sha256, "anchor": dict(self.anchor),
                "selected_component_index": self.selected_component_index, "degree": self.degree,
                "certificate": dict(self.certificate), "runner_ups": [dict(r) for r in self.runner_ups],
                "genpept_preset": self.genpept_preset}
