"""Immutable scientific state definitions, separate from engine checkpoints.

This is an explicit metadata API: it never invents historical CV coefficients,
boost envelopes, system identity, or intervention history from today's config.
Writers must call it after final window filtering/resume reconciliation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

from ._io import IntegrityError, atomic_bytes, digest, json_bytes, json_loads
from .bias import finite_number, normalize_windows

STATE_SCHEMA = "atlas-fixed-state-v1"


@dataclass(frozen=True)
class CanonicalStateTable:
    definition: dict[str, Any]
    windows: tuple[dict[str, Any], ...]
    column_window_ids: tuple[int, ...]
    definition_sha256: str


def _hash_string(value: Any, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise IntegrityError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_cv(raw: Mapping[str, Any] | None, label: str) -> dict | None:
    if raw is None:
        return None
    cv = json_loads(json_bytes(dict(raw)))
    if set(cv) != {"kind", "units", "definition"}:
        raise IntegrityError(f"{label} requires kind, units, and embedded definition (no filename identity)")
    if not isinstance(cv["kind"], str) or not cv["kind"]:
        raise IntegrityError(f"{label}.kind must be explicit")
    if cv["units"] not in {"dimensionless", "angstrom", "nanometer", "radian"}:
        raise IntegrityError(f"{label}.units not supported: {cv['units']!r}")
    if not isinstance(cv["definition"], dict) or not cv["definition"]:
        raise IntegrityError(f"{label}.definition must embed the actual CV parameters")
    def reject_path_only(value):
        if isinstance(value, dict):
            for key, val in value.items():
                if key.endswith(("_path", "_file", "_filename")) or key in {"path", "file", "filename"}:
                    raise IntegrityError(f"{label}: embed model contents instead of {key}")
                reject_path_only(val)
        elif isinstance(value, list):
            for val in value:
                reject_path_only(val)
    reject_path_only(cv["definition"])
    return cv


def canonical_state_definition(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = json_loads(json_bytes(dict(raw)))
    required = {"schema", "physical_system_sha256", "ensemble", "temperature_k",
                "pressure_bar", "fixed_box_vectors_nm", "cv1", "cv2",
                "boost", "energy_unit", "windows"}
    if set(data) != required or data.get("schema") != STATE_SCHEMA:
        raise IntegrityError(f"State definition requires exact schema {STATE_SCHEMA}; missing/extra fields")
    _hash_string(data["physical_system_sha256"], "physical_system_sha256")
    data["temperature_k"] = finite_number(data["temperature_k"], "temperature_k", positive=True)
    ensemble = data["ensemble"]
    if ensemble == "NVT":
        if data["pressure_bar"] is not None:
            raise IntegrityError("An NVT target must not carry a pressure parameter")
        box = np.asarray(data["fixed_box_vectors_nm"], dtype=np.float64)
        if box.shape != (3, 3) or not np.isfinite(box).all() or np.linalg.det(box) <= 0:
            raise IntegrityError("NVT requires its fixed positive-volume cell")
        data["fixed_box_vectors_nm"] = box.tolist()
    elif ensemble == "NPT":
        data["pressure_bar"] = finite_number(data["pressure_bar"], "pressure_bar")
        if data["fixed_box_vectors_nm"] is not None:
            raise IntegrityError("Instantaneous NPT volume is a sample variable, not a state identity")
    else:
        raise IntegrityError("This fixed-state exporter supports explicit NVT or NPT targets only")
    data["cv1"] = _canonical_cv(data["cv1"], "cv1")
    data["cv2"] = _canonical_cv(data["cv2"], "cv2")
    energy_unit = data["energy_unit"]
    if energy_unit not in {"kcal/mol", "kJ/mol"}:
        raise IntegrityError("State energy_unit must be kcal/mol or kJ/mol")
    windows = normalize_windows(data["windows"])
    if not windows:
        raise IntegrityError("A state table needs at least one window")
    seen = set()
    for row in windows:
        state_id = row.get("window_id")
        if isinstance(state_id, bool) or not isinstance(state_id, int) or state_id < 0 or state_id in seen:
            raise IntegrityError(f"Missing/invalid/duplicate window_id: {state_id!r}")
        seen.add(state_id)
        if set(row) - {"window_id", "center1", "k1", "center2", "k2", "gamd_lambda"}:
            raise IntegrityError(f"Window {state_id} has unknown physics fields; normalize explicitly")
        if energy_unit == "kJ/mol":
            row["k1"] /= 4.184
            row["k2"] /= 4.184
        if row["k1"] > 0 and data["cv1"] is None:
            raise IntegrityError(f"Window {state_id} needs a frozen CV1 definition")
        if row["k2"] > 0 and data["cv2"] is None:
            raise IntegrityError(f"Window {state_id} needs a frozen CV2 definition")
    data["energy_unit"] = "kcal/mol"
    data["windows"] = sorted(windows, key=lambda row: row["window_id"])
    if any(row["gamd_lambda"] > 0 for row in windows) and not isinstance(data["boost"], dict):
        raise IntegrityError("Active lambda states require an embedded boost definition")
    if data["boost"] is not None:
        if not isinstance(data["boost"], dict) or not data["boost"]:
            raise IntegrityError("boost must be null or a nonempty frozen definition")
        # Both static bias and lambda-dependent boosts belong here. Even a
        # shared bias matters when later recovering the physical target.
        if "kind" not in data["boost"] or "envelope" not in data["boost"]:
            raise IntegrityError("boost requires kind and embedded envelope")
        if not isinstance(data["boost"]["envelope"], dict):
            raise IntegrityError("boost.envelope must embed parameters, not a filename")
    return data


def make_state_definition(
    windows: Sequence[Mapping[str, Any]], *, physical_system_sha256: str,
    ensemble: str, temperature_k: float, cv1: Mapping | None, cv2: Mapping | None,
    pressure_bar: float | None = None, fixed_box_vectors_nm=None,
    boost: Mapping | None = None, energy_unit: str = "kcal/mol",
) -> dict[str, Any]:
    return canonical_state_definition({
        "schema": STATE_SCHEMA, "physical_system_sha256": physical_system_sha256,
        "ensemble": ensemble, "temperature_k": temperature_k,
        "pressure_bar": pressure_bar, "fixed_box_vectors_nm": fixed_box_vectors_nm,
        "cv1": cv1, "cv2": cv2, "boost": boost,
        "energy_unit": energy_unit, "windows": list(windows),
    })


def state_definition_hash(definition: Mapping[str, Any]) -> str:
    return digest(json_bytes(canonical_state_definition(definition)))


def compare_state_definitions(reference: Any, candidate: Any, path: str = "state") -> list[dict]:
    """Exact, field-level comparison; do not merge states using loose isclose."""
    if isinstance(reference, dict) and isinstance(candidate, dict):
        differences = []
        for key in sorted(set(reference) | set(candidate)):
            child = f"{path}.{key}"
            if key not in reference or key not in candidate:
                differences.append({"field": child, "reference": reference.get(key, "<missing>"),
                                    "candidate": candidate.get(key, "<missing>")})
            else:
                differences.extend(compare_state_definitions(reference[key], candidate[key], child))
        return differences
    if isinstance(reference, list) and isinstance(candidate, list):
        if len(reference) != len(candidate):
            return [{"field": path + ".length", "reference": len(reference), "candidate": len(candidate)}]
        return [difference for i, (left, right) in enumerate(zip(reference, candidate))
                for difference in compare_state_definitions(left, right, f"{path}[{i}]")]
    if reference != candidate:
        return [{"field": path, "reference": reference, "candidate": candidate}]
    return []


def freeze_snapshot(segment_id: str, definition: Mapping[str, Any], *,
                    equilibrium_analysis_eligible: bool,
                    phase_kind: str) -> dict[str, Any]:
    if not isinstance(segment_id, str) or not segment_id:
        raise IntegrityError("A snapshot needs an explicit segment_id")
    if not isinstance(equilibrium_analysis_eligible, bool):
        raise IntegrityError("Analysis eligibility must be explicit, not truthy metadata")
    if phase_kind not in {"production", "pilot", "exploration", "equilibration"}:
        raise IntegrityError("Unknown explicit phase_kind")
    if equilibrium_analysis_eligible and phase_kind != "production":
        raise IntegrityError("Exploration/equilibration cannot be marked production eligible")
    state = canonical_state_definition(definition)
    return {
        "segment_id": segment_id, "snapshot_schema": "atlas-window-snapshot-v2",
        "windows": state["windows"],
        "cv1_type": state["cv1"]["kind"] if state["cv1"] is not None else None,
        "cv2_type": state["cv2"]["kind"] if state["cv2"] is not None else None,
        "state_definition": state, "state_definition_sha256": state_definition_hash(state),
        "sampling_policy": {"phase_kind": phase_kind,
                            "equilibrium_analysis_eligible": equilibrium_analysis_eligible},
    }


def write_frozen_snapshot(path: Path | str, snapshot: Mapping[str, Any]) -> None:
    """Refuse to rewrite an existing segment's definition; identical is a no-op.

    Caller must hold the run lock. This is a single-writer metadata API.
    """
    validate_fixed_state_segments({str(snapshot["segment_id"]): snapshot}, require_eligible=False)
    path = Path(path)
    payload = json_bytes(snapshot)
    if path.exists():
        if path.read_bytes() != payload:
            raise IntegrityError(f"Refusing to overwrite immutable state snapshot {path}")
        return
    atomic_bytes(path, payload)


def validate_fixed_state_segments(
    snapshots: Mapping[str, Mapping[str, Any]], *, require_eligible: bool = True,
) -> CanonicalStateTable:
    if not snapshots:
        raise IntegrityError("No state snapshots were selected")
    first = None
    first_id = None
    for segment_id, snapshot in snapshots.items():
        if snapshot.get("segment_id") != segment_id:
            raise IntegrityError(f"Snapshot segment_id mismatch for {segment_id}")
        if snapshot.get("snapshot_schema") != "atlas-window-snapshot-v2":
            raise IntegrityError(
                f"Segment {segment_id} lacks a frozen state definition. Do not infer historical "
                "CVs/envelopes from current files; export through a reviewed legacy migration."
            )
        policy = snapshot.get("sampling_policy")
        if (not isinstance(policy, dict)
                or policy.get("phase_kind") not in {"production", "pilot", "exploration", "equilibration"}
                or not isinstance(policy.get("equilibrium_analysis_eligible"), bool)):
            raise IntegrityError(f"Segment {segment_id} has invalid sampling-policy metadata")
        if policy["equilibrium_analysis_eligible"] and policy["phase_kind"] != "production":
            raise IntegrityError(f"Segment {segment_id} inconsistently marks exploration as production eligible")
        if require_eligible and policy["equilibrium_analysis_eligible"] is not True:
            raise IntegrityError(f"Segment {segment_id} is exploratory or has unknown sampling eligibility")
        state = canonical_state_definition(snapshot["state_definition"])
        if state_definition_hash(state) != snapshot.get("state_definition_sha256"):
            raise IntegrityError(f"State-definition checksum mismatch for {segment_id}")
        if normalize_windows(snapshot.get("windows", [])) != state["windows"]:
            # Top-level windows may be ordered differently; compare by IDs below.
            visible = sorted(normalize_windows(snapshot.get("windows", [])),
                             key=lambda row: row.get("window_id", -1))
            if visible != state["windows"]:
                raise IntegrityError(f"Visible windows disagree with immutable definition in {segment_id}")
        if first is None:
            first, first_id = state, segment_id
        else:
            differences = compare_state_definitions(first, state)
            if differences:
                detail = differences[0]
                raise IntegrityError(
                    f"Segments {first_id!r} and {segment_id!r} are not fixed-state compatible: "
                    f"{detail['field']}: {detail['reference']!r} != {detail['candidate']!r}. "
                    "Use a validated union-state workflow instead of changing columns by segment."
                )
    return CanonicalStateTable(first, tuple(first["windows"]),
                               tuple(row["window_id"] for row in first["windows"]),
                               state_definition_hash(first))
