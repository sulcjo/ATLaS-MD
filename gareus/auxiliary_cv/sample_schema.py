# gareus/auxiliary_cv/sample_schema.py
"""What every auxiliary-capable sample row stores: the full ordered backbone torsion basis
(OpenMM theta, float64) and z of every phase model (float64), schema ``atlas-aux-samples-v1``.

The basis is ALL backbone phi/psi, so a model fitted later can still be evaluated (spec Section 7).
Column names (tor_000, aux_z_00) are positions in the recorded lists; the lists and basis_sha256
are the identity. The payload's optional ``runtime`` block (platform, precision) is per segment and
selects the parity tolerance; it is not part of the schema identity.

Payloads are JSON-canonical (lists, plain str/int), so they survive a json round trip unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..correctness._io import IntegrityError, digest, json_bytes
from .evaluate import z_from_dihedrals
from .features import openmm_dihedrals, unique_torsions
from .model import AuxModel

AUX_SAMPLES_SCHEMA = "atlas-aux-samples-v1"
AUX_SAMPLES_SCHEMA_V2 = "atlas-aux-samples-v2"
SUPPORTED_AUX_SAMPLES_SCHEMAS = (AUX_SAMPLES_SCHEMA, AUX_SAMPLES_SCHEMA_V2)
TORSION_CONVENTION = "openmm_theta_radians"
PRECISIONS = ("double", "mixed", "single")
#: Reduced-energy parity tolerance between stored runtime z and offline z (spec Section 17).
PARITY_TOLERANCE = {"double": 1e-6, "mixed": 1e-4, "single": 1e-4}


def _basis_sha(quads, labels) -> str:
    return digest(json_bytes({"convention": TORSION_CONVENTION, "quads": [list(q) for q in quads],
                              "labels": list(labels)}))


def _basis_sha_v2(quads, labels, shas, model_maps) -> str:
    return digest(json_bytes({"convention": TORSION_CONVENTION, "quads": [list(q) for q in quads],
                              "labels": list(labels), "model_shas": list(shas),
                              "model_angle_maps": [[list(row) for row in mapping] for mapping in model_maps]}))


def _feature_angle_map(model, basis_quads):
    where = {q: i for i, q in enumerate(basis_quads)}
    mapped = []
    for feature in model.features:
        try:
            mapped.append(tuple(where[q] for q in feature.orbit))
        except KeyError as exc:
            raise IntegrityError(f"model feature {feature.name} is outside the recorded sample basis") from exc
    return tuple(mapped)


@dataclass(frozen=True)
class AuxSampleSchema:
    torsion_quads: tuple[tuple[int, int, int, int], ...]
    torsion_labels: tuple[str, ...]
    model_shas: tuple[str, ...]
    basis_sha256: str
    schema: str = AUX_SAMPLES_SCHEMA
    model_angle_maps: tuple[tuple[tuple[int, ...], ...], ...] = ()

    @property
    def torsion_columns(self) -> tuple[str, ...]:
        return tuple(f"tor_{i:03d}" for i in range(len(self.torsion_quads)))

    @property
    def z_columns(self) -> tuple[str, ...]:
        return tuple(f"aux_z_{i:02d}" for i in range(len(self.model_shas)))

    def to_payload(self) -> dict[str, Any]:
        payload = {"schema": self.schema, "torsion_convention": TORSION_CONVENTION,
                "torsion_quads": [[int(a) for a in q] for q in self.torsion_quads],
                "torsion_labels": [str(x) for x in self.torsion_labels],
                "model_shas": [str(x) for x in self.model_shas], "basis_sha256": str(self.basis_sha256)}
        if self.schema == AUX_SAMPLES_SCHEMA_V2:
            payload["model_angle_maps"] = [
                {"model_sha256": sha, "feature_orbit_columns": [list(x) for x in mapping]}
                for sha, mapping in zip(self.model_shas, self.model_angle_maps)]
        return payload

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> "AuxSampleSchema":
        version = raw.get("schema")
        if version not in SUPPORTED_AUX_SAMPLES_SCHEMAS or raw.get("torsion_convention") != TORSION_CONVENTION:
            raise IntegrityError(f"aux sample schema must be one of {SUPPORTED_AUX_SAMPLES_SCHEMAS} / "
                                 f"{TORSION_CONVENTION}")
        if version == AUX_SAMPLES_SCHEMA_V2 and set(raw) != {
                "schema", "torsion_convention", "torsion_quads", "torsion_labels", "model_shas",
                "basis_sha256", "model_angle_maps"}:
            raise IntegrityError("v2 sample schema has missing or unknown fields")
        if version == AUX_SAMPLES_SCHEMA_V2:
            raw_quads = raw.get("torsion_quads")
            if (not isinstance(raw_quads, list) or any(not isinstance(q, list) or len(q) != 4 or
                    any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in q) or
                    len(set(q)) != 4 for q in raw_quads) or len(set(tuple(q) for q in raw_quads)) != len(raw_quads)):
                raise IntegrityError("v2 sample torsion basis has invalid or duplicate atom quadruplets")
            raw_labels, raw_shas = raw.get("torsion_labels"), raw.get("model_shas")
            if (not isinstance(raw_labels, list) or any(not isinstance(x, str) or not x for x in raw_labels)
                    or len(set(raw_labels)) != len(raw_labels)):
                raise IntegrityError("v2 sample torsion labels must be nonempty unique strings")
            if (not isinstance(raw_shas, list) or len(raw_shas) != 1 or any(
                    not isinstance(x, str) or len(x) != 64 or
                    any(c not in "0123456789abcdef" for c in x) for x in raw_shas)):
                raise IntegrityError("v2 sample schema requires one full model SHA-256")
        quads = tuple(tuple(int(a) for a in q) for q in raw["torsion_quads"])
        labels = tuple(str(x) for x in raw["torsion_labels"])
        if len(quads) != len(labels) or any(len(q) != 4 for q in quads):
            raise IntegrityError("aux sample schema torsion_quads/labels are inconsistent")
        shas = tuple(str(x) for x in raw["model_shas"])
        if any(len(s) != 64 for s in shas):
            raise IntegrityError("aux sample schema model_shas must be full sha256 digests")
        model_maps = ()
        if version == AUX_SAMPLES_SCHEMA_V2:
            if len(shas) != 1:
                raise IntegrityError("v2 sample schema requires exactly one admitted model")
            maps_raw = raw.get("model_angle_maps")
            if not isinstance(maps_raw, list) or len(maps_raw) != len(shas):
                raise IntegrityError("v2 sample schema needs one explicit angle map per model")
            parsed = []
            for item, sha in zip(maps_raw, shas):
                if not isinstance(item, dict) or set(item) != {"model_sha256", "feature_orbit_columns"} or item[
                        "model_sha256"] != sha:
                    raise IntegrityError("v2 sample model angle map identity mismatch")
                rows = item["feature_orbit_columns"]
                if not isinstance(rows, list):
                    raise IntegrityError("v2 sample feature angle map must be a list")
                feature_map = []
                for row in rows:
                    if (not isinstance(row, list) or not row or any(isinstance(i, bool) or
                            not isinstance(i, int) or not 0 <= i < len(quads) for i in row)
                            or len(set(row)) != len(row)):
                        raise IntegrityError("v2 sample feature angle map contains invalid columns")
                    feature_map.append(tuple(row))
                parsed.append(tuple(feature_map))
            model_maps = tuple(parsed)
            expected = _basis_sha_v2(quads, labels, shas, model_maps)
        else:
            expected = _basis_sha(quads, labels)
        if raw.get("basis_sha256") != expected:
            raise IntegrityError(f"aux sample schema basis_sha256 {raw.get('basis_sha256')} != content {expected}")
        return cls(quads, labels, shas, expected, version, model_maps)

    def model_basis_index(self, model: AuxModel) -> np.ndarray:
        """Positions in the stored basis of ``unique_torsions(model)`` (first-appearance order), the
        column order Stage A ``z_from_dihedrals`` requires."""
        where = {q: i for i, q in enumerate(self.torsion_quads)}
        if getattr(model, "schema_version", 1) == 2:
            self._check_v2_model_map(model)
            quads = model.projection.quads
            missing = [q for q in quads if q not in where]
            if missing:
                raise IntegrityError(f"aux model {model.model_sha256} uses torsions outside the recorded basis: "
                                     f"{missing}")
            return np.asarray([where[q] for q in quads], dtype=np.int64)
        quads, _idx = unique_torsions(model)
        missing = [q for q in quads if q not in where]
        if missing:
            raise IntegrityError(f"aux model {model.model_sha256} uses torsions outside the recorded basis: {missing}")
        return np.asarray([where[q] for q in quads], dtype=np.int64)

    def _check_v2_model_map(self, model) -> None:
        if self.schema != AUX_SAMPLES_SCHEMA_V2 or model.model_sha256 not in self.model_shas:
            raise IntegrityError("v2 model requires matching v2 sample basis identity")
        model_position = self.model_shas.index(model.model_sha256)
        if self.model_angle_maps[model_position] != _feature_angle_map(model, self.torsion_quads):
            raise IntegrityError("sample feature-to-angle column map differs from frozen model basis")


def build_sample_schema(topology, peptide_atoms, models: Sequence[AuxModel]) -> AuxSampleSchema:
    from ..mbar_analysis.thermo_frames import backbone_torsion_quads
    shas: list[str] = []
    for m in models:
        if m.model_sha256 not in shas:
            shas.append(m.model_sha256)
    if len(shas) > 1:
        raise IntegrityError("More than one auxiliary model per phase is Stage F (spec Section 16); MVP supports one")
    v2_models = [m for m in models if getattr(m, "schema_version", 1) == 2]
    if v2_models:
        if len(models) != 1:
            raise IntegrityError("v2 sample schema requires the single admitted model for this phase")
        model = v2_models[0]
        model.validate_topology(topology)
        quads, labels, where = [], [], {}
        model_map = []
        for feature in model.features:
            columns = []
            for orbit_index, quad in enumerate(feature.orbit):
                if quad not in where:
                    where[quad] = len(quads)
                    quads.append(quad)
                    labels.append(f"{feature.name}.orbit_{orbit_index}")
                columns.append(where[quad])
            model_map.append(tuple(columns))
        maps = (tuple(model_map),)
        quads, labels = tuple(quads), tuple(labels)
        schema = AuxSampleSchema(quads, labels, tuple(shas), _basis_sha_v2(quads, labels, tuple(shas), maps),
                                 AUX_SAMPLES_SCHEMA_V2, maps)
        schema.model_basis_index(model)
        return schema
    quads, labels = backbone_torsion_quads(topology, peptide_atoms)
    quads = tuple(tuple(int(a) for a in q) for q in quads)
    schema = AuxSampleSchema(quads, tuple(labels), tuple(shas), _basis_sha(quads, labels))
    for m in models:
        schema.model_basis_index(m)
    return schema


@dataclass(frozen=True)
class AuxObservation:
    torsions: np.ndarray
    z: np.ndarray


def observe_carrier(xyz_nm, schema: AuxSampleSchema, models: Mapping[str, AuxModel]) -> AuxObservation:
    theta = openmm_dihedrals(xyz_nm, schema.torsion_quads)[0].astype(np.float64)
    z = np.empty(len(schema.model_shas), dtype=np.float64)
    for i, sha in enumerate(schema.model_shas):
        if sha not in models:
            raise IntegrityError(f"no loaded aux model for schema model {sha}")
        model = models[sha]
        angle_index = schema.model_basis_index(model)
        if getattr(model, "schema_version", 1) == 2:
            z[i] = model.projection.from_angles(theta[angle_index][None, :])[0]
        else:
            z[i] = z_from_dihedrals(theta[angle_index][None, :], model)[0]
    return AuxObservation(theta, z)


def runtime_precision_info(platform_name: str, platform=None, context=None) -> tuple[str, Optional[str]]:
    """(precision, fallback reason or None). Reference/CPU are fixed; a GPU platform's ``Precision``
    property is read from the Context. A failed or unrecognised read falls back to ``single`` (the
    loosest parity tolerance) and says why, so it can be warned about and recorded (final fix wave I4c).
    """
    name = str(platform_name)
    if name == "Reference":
        return "double", None
    if name == "CPU":
        return "mixed", None
    if platform is None or context is None:
        return "single", f"no platform/context to read the {name} Precision property from"
    try:
        value = str(platform.getPropertyValue(context, "Precision")).strip().lower()
    except Exception as exc:  # noqa: BLE001 -- any failed read is recorded, never silent
        return "single", f"{name} Precision property read failed: {type(exc).__name__}: {exc}"
    if value in PRECISIONS:
        return value, None
    return "single", f"{name} Precision property value {value!r} is not one of {PRECISIONS}"


def runtime_precision(platform_name: str, platform=None, context=None) -> str:
    return runtime_precision_info(platform_name, platform, context)[0]


#: Where the stored runtime z (``aux_z_00``) comes from (F07): the aux force on each replica's Context.
AUX_Z_SOURCE = "force"
#: The independent value runtime parity compares the stored z with.
AUX_Z_REFERENCE = "positions"
#: Values a samples payload may record for them.
AUX_Z_SOURCES = (AUX_Z_SOURCE,)
AUX_Z_REFERENCES = (AUX_Z_REFERENCE,)


def runtime_info(platform_name: str, precision: str, fallback_reason: Optional[str] = None, *,
                 z_source: Optional[str] = None, z_reference: Optional[str] = None) -> dict[str, Any]:
    if precision not in PRECISIONS:
        raise IntegrityError(f"runtime precision must be one of {PRECISIONS}, got {precision!r}")
    info: dict[str, Any] = {"platform": str(platform_name), "precision": str(precision)}
    if fallback_reason is not None:
        # Only on a fallback: the normal block (and its payload bytes) is unchanged.
        info["precision_fallback"] = True
        info["precision_fallback_reason"] = str(fallback_reason)
    if (z_source is None) != (z_reference is None):
        raise IntegrityError("runtime aux z source and reference are recorded together or not at all")
    if z_source is not None:
        # F07: written by every run since the force-side observer; absent in older payloads ("unrecorded").
        if z_source not in AUX_Z_SOURCES or z_reference not in AUX_Z_REFERENCES:
            raise IntegrityError(f"runtime aux z source/reference must be one of {AUX_Z_SOURCES}/"
                                 f"{AUX_Z_REFERENCES}, got {z_source!r}/{z_reference!r}")
        info["aux_z_source"] = str(z_source)
        info["aux_z_reference"] = str(z_reference)
    return info


def _z_sources(block: Mapping[str, Any]) -> dict[str, Optional[str]]:
    return {"z_source": block.get("aux_z_source"), "z_reference": block.get("aux_z_reference")}


def _fallback_reason(block: Mapping[str, Any]) -> Optional[str]:
    return str(block.get("precision_fallback_reason", "")) if block.get("precision_fallback") else None


def payload_with_runtime(schema: AuxSampleSchema, info: Mapping[str, str]) -> dict[str, Any]:
    payload = schema.to_payload()
    payload["runtime"] = runtime_info(info["platform"], info["precision"], _fallback_reason(info), **_z_sources(info))
    return payload


def runtime_from_payload(payload: Mapping[str, Any]) -> Optional[dict[str, str]]:
    block = payload.get("runtime")
    if block is None:
        return None
    return runtime_info(block.get("platform", ""), block.get("precision", ""), _fallback_reason(block),
                        **_z_sources(block))
