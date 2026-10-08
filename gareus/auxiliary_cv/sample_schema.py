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
TORSION_CONVENTION = "openmm_theta_radians"
PRECISIONS = ("double", "mixed", "single")
#: Reduced-energy parity tolerance between stored runtime z and offline z (spec Section 17).
PARITY_TOLERANCE = {"double": 1e-6, "mixed": 1e-4, "single": 1e-4}


def _basis_sha(quads, labels) -> str:
    return digest(json_bytes({"convention": TORSION_CONVENTION, "quads": [list(q) for q in quads],
                              "labels": list(labels)}))


@dataclass(frozen=True)
class AuxSampleSchema:
    torsion_quads: tuple[tuple[int, int, int, int], ...]
    torsion_labels: tuple[str, ...]
    model_shas: tuple[str, ...]
    basis_sha256: str

    @property
    def torsion_columns(self) -> tuple[str, ...]:
        return tuple(f"tor_{i:03d}" for i in range(len(self.torsion_quads)))

    @property
    def z_columns(self) -> tuple[str, ...]:
        return tuple(f"aux_z_{i:02d}" for i in range(len(self.model_shas)))

    def to_payload(self) -> dict[str, Any]:
        return {"schema": AUX_SAMPLES_SCHEMA, "torsion_convention": TORSION_CONVENTION,
                "torsion_quads": [[int(a) for a in q] for q in self.torsion_quads],
                "torsion_labels": [str(x) for x in self.torsion_labels],
                "model_shas": [str(x) for x in self.model_shas], "basis_sha256": str(self.basis_sha256)}

    @classmethod
    def from_payload(cls, raw: Mapping[str, Any]) -> "AuxSampleSchema":
        if raw.get("schema") != AUX_SAMPLES_SCHEMA or raw.get("torsion_convention") != TORSION_CONVENTION:
            raise IntegrityError(f"aux sample schema must be {AUX_SAMPLES_SCHEMA} / {TORSION_CONVENTION}")
        quads = tuple(tuple(int(a) for a in q) for q in raw["torsion_quads"])
        labels = tuple(str(x) for x in raw["torsion_labels"])
        if len(quads) != len(labels) or any(len(q) != 4 for q in quads):
            raise IntegrityError("aux sample schema torsion_quads/labels are inconsistent")
        shas = tuple(str(x) for x in raw["model_shas"])
        if any(len(s) != 64 for s in shas):
            raise IntegrityError("aux sample schema model_shas must be full sha256 digests")
        expected = _basis_sha(quads, labels)
        if raw.get("basis_sha256") != expected:
            raise IntegrityError(f"aux sample schema basis_sha256 {raw.get('basis_sha256')} != content {expected}")
        return cls(quads, labels, shas, expected)

    def model_basis_index(self, model: AuxModel) -> np.ndarray:
        """Positions in the stored basis of ``unique_torsions(model)`` (first-appearance order), the
        column order Stage A ``z_from_dihedrals`` requires."""
        where = {q: i for i, q in enumerate(self.torsion_quads)}
        quads, _idx = unique_torsions(model)
        missing = [q for q in quads if q not in where]
        if missing:
            raise IntegrityError(f"aux model {model.model_sha256} uses torsions outside the recorded basis: {missing}")
        return np.asarray([where[q] for q in quads], dtype=np.int64)


def build_sample_schema(topology, peptide_atoms, models: Sequence[AuxModel]) -> AuxSampleSchema:
    from ..mbar_analysis.thermo_frames import backbone_torsion_quads
    shas: list[str] = []
    for m in models:
        if m.model_sha256 not in shas:
            shas.append(m.model_sha256)
    if len(shas) > 1:
        raise IntegrityError("More than one auxiliary model per phase is Stage F (spec Section 16); MVP supports one")
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
        z[i] = z_from_dihedrals(theta[schema.model_basis_index(model)][None, :], model)[0]
    return AuxObservation(theta, z)


def runtime_precision(platform_name: str, platform=None, context=None) -> str:
    name = str(platform_name)
    if name == "Reference":
        return "double"
    if name == "CPU":
        return "mixed"
    if platform is not None and context is not None:
        try:
            value = str(platform.getPropertyValue(context, "Precision")).strip().lower()
        except Exception:
            value = ""
        if value in PRECISIONS:
            return value
    return "single"


def runtime_info(platform_name: str, precision: str) -> dict[str, str]:
    if precision not in PRECISIONS:
        raise IntegrityError(f"runtime precision must be one of {PRECISIONS}, got {precision!r}")
    return {"platform": str(platform_name), "precision": str(precision)}


def payload_with_runtime(schema: AuxSampleSchema, info: Mapping[str, str]) -> dict[str, Any]:
    payload = schema.to_payload()
    payload["runtime"] = runtime_info(info["platform"], info["precision"])
    return payload


def runtime_from_payload(payload: Mapping[str, Any]) -> Optional[dict[str, str]]:
    block = payload.get("runtime")
    if block is None:
        return None
    return runtime_info(block.get("platform", ""), block.get("precision", ""))
