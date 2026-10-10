"""Frozen auxiliary-CV model, schema ``atlas-aux-cv-model-v1``.

z(x) = (offset + sum_j coefficients[j] * f_j(x)) / scale, with f_j = trig_j(s_j * theta_j), where
theta_j is OpenMM's CustomTorsionForce ``theta`` for feature j's atoms and s_j = -1 for the "negated"
sign convention (the tica / swarm convention) or +1 for "direct". ``scale`` is the frozen
normalisation divisor (positive); ``periodic_imaging`` records how torsion atoms are imaged ("none":
raw Context coordinates, no minimum-image; the only supported value). Identity is content:
``label`` and ``provenance`` never enter ``model_sha256``; -0.0 is canonicalised to 0.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..correctness._io import IntegrityError, atomic_bytes, digest, json_bytes, json_loads
from ..cv_selection.contracts import FeatureSchema

AUX_MODEL_SCHEMA = "atlas-aux-cv-model-v1"
SUPPORTED_UNITS = frozenset({"dimensionless"})
SUPPORTED_PERIODIC_IMAGING = frozenset({"none"})
_REQUIRED = {"schema", "feature_schema", "coefficients", "offset", "scale", "periodic_imaging", "units"}
_OPTIONAL = {"label", "provenance", "model_sha256"}


class AuxModelError(IntegrityError):
    """An auxiliary-CV model payload cannot be used as the claimed frozen function."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuxModelError(f"{label} must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise AuxModelError(f"{label} must be finite, got {value!r}")
    return out + 0.0          # canonicalise -0.0 -> 0.0 (identity must not depend on the sign of zero)


@dataclass(frozen=True)
class AuxModel:
    feature_schema: FeatureSchema
    coefficients: tuple[float, ...]
    offset: float
    scale: float
    periodic_imaging: str
    units: str
    label: str
    provenance_json: str
    model_sha256: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AuxModel":
        if isinstance(raw, Mapping) and raw.get("schema") == "atlas-aux-cv-model-v2":
            from .sidechain_model import SidechainModel
            return SidechainModel.from_mapping(raw)
        data = json_loads(json_bytes(dict(raw)))
        unknown = sorted(set(data) - _REQUIRED - _OPTIONAL)
        if unknown:
            raise AuxModelError(f"aux model has unknown field(s): {', '.join(unknown)}")
        missing = sorted(_REQUIRED - set(data))
        if missing:
            raise AuxModelError(f"aux model is missing field(s): {', '.join(missing)}")
        if data["schema"] != AUX_MODEL_SCHEMA:
            raise AuxModelError(f"aux model requires schema {AUX_MODEL_SCHEMA!r}, got {data['schema']!r}")
        schema = FeatureSchema.from_mapping(data["feature_schema"])
        coeffs = data["coefficients"]
        if not isinstance(coeffs, list) or len(coeffs) != schema.width:
            raise AuxModelError(f"aux model coefficients must list {schema.width} numbers (one per feature)")
        coefficients = tuple(_finite(c, f"coefficients[{i}]") for i, c in enumerate(coeffs))
        if not any(c != 0.0 for c in coefficients):
            raise AuxModelError("aux model needs at least one nonzero coefficient (else z is constant)")
        offset = _finite(data["offset"], "offset")
        scale = _finite(data["scale"], "scale")
        if scale <= 0.0:
            raise AuxModelError(f"aux model scale must be positive, got {scale!r}")
        if data["periodic_imaging"] not in SUPPORTED_PERIODIC_IMAGING:
            raise AuxModelError(f"aux model periodic_imaging {data['periodic_imaging']!r} not supported "
                                f"({sorted(SUPPORTED_PERIODIC_IMAGING)})")
        if data["units"] not in SUPPORTED_UNITS:
            raise AuxModelError(f"aux model units {data['units']!r} not supported ({sorted(SUPPORTED_UNITS)})")
        label = data.get("label", "")
        if not isinstance(label, str):
            raise AuxModelError("aux model label must be a string")
        provenance = data.get("provenance", {})
        if not isinstance(provenance, dict):
            raise AuxModelError("aux model provenance must be an object")
        sha = digest(json_bytes(cls._identity(schema, coefficients, offset, scale,
                                              data["periodic_imaging"], data["units"])))
        claimed = data.get("model_sha256")
        if claimed is not None and claimed != sha:
            raise AuxModelError(f"aux model claims digest {claimed} but its contents hash to {sha}")
        return cls(schema, coefficients, offset, scale, data["periodic_imaging"], data["units"], label,
                   json_bytes(provenance).decode("utf-8"), sha)

    @staticmethod
    def _identity(schema, coefficients, offset, scale, periodic_imaging, units) -> dict[str, Any]:
        return {"schema": AUX_MODEL_SCHEMA, "feature_schema": schema.to_mapping(),
                "coefficients": list(coefficients), "offset": offset, "scale": scale,
                "periodic_imaging": periodic_imaging, "units": units}

    def identity_mapping(self) -> dict[str, Any]:
        """Identity body + model_sha256 only (what state-definition registries embed)."""
        body = self._identity(self.feature_schema, self.coefficients, self.offset, self.scale,
                              self.periodic_imaging, self.units)
        return {**body, "model_sha256": self.model_sha256}

    @classmethod
    def load(cls, path: Path | str) -> "AuxModel":
        return cls.from_mapping(json_loads(Path(path).read_bytes()))

    def to_mapping(self) -> dict[str, Any]:
        return {**self.identity_mapping(), "label": self.label,
                "provenance": json_loads(self.provenance_json)}

    def write(self, path: Path | str) -> None:
        atomic_bytes(Path(path), json_bytes(self.to_mapping()))
