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
STATE_SCHEMA_V2 = "atlas-fixed-state-v2"
STATE_ROLES = frozenset({"ordinary", "auxiliary", "sham"})
_V1_WINDOW_FIELDS = {"window_id", "center1", "k1", "center2", "k2", "gamd_lambda"}
_V2_WINDOW_FIELDS = _V1_WINDOW_FIELDS | {"aux_model_sha256", "aux_center", "aux_k", "instance"}
_INSTANCE_FIELDS = {"state_instance_id", "state_role", "spawn_parent_state_id",
                    "spawn_source_observation", "matched_additional_slot_id"}
_OBSERVATION_FIELDS = {"run", "segment", "carrier", "state", "checkpoint", "step"}
_SHARED_FIELDS = ("physical_system_sha256", "ensemble", "temperature_k", "pressure_bar",
                  "fixed_box_vectors_nm", "cv1", "cv2", "boost", "energy_unit")


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


def _canonical_aux_models(raw: Any) -> dict[str, dict]:
    from ..auxiliary_cv.model import AuxModel   # lazy: avoids a cv_selection import cycle
    if not isinstance(raw, dict):
        raise IntegrityError("aux_models must map model_sha256 -> embedded model payload")
    if len(raw) > 1:
        raise IntegrityError("More than one auxiliary model per state definition is Stage F "
                             "(spec Section 16); MVP supports one")
    out = {}
    for key, payload in raw.items():
        model = AuxModel.from_mapping(payload)
        if key != model.model_sha256:
            raise IntegrityError(f"aux_models key {key} does not match the model's digest {model.model_sha256}")
        out[key] = model.identity_mapping()     # identity body + sha only: no label/provenance
    return out


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _canonical_instance(raw: Any, state_id: int) -> dict:
    if not isinstance(raw, dict) or set(raw) != _INSTANCE_FIELDS:
        raise IntegrityError(f"Window {state_id} instance needs exactly {sorted(_INSTANCE_FIELDS)}")
    if raw["state_role"] not in STATE_ROLES:
        raise IntegrityError(f"Window {state_id} instance.state_role must be one of {sorted(STATE_ROLES)}")
    if not _nonempty_str(raw["state_instance_id"]):
        raise IntegrityError(f"Window {state_id} instance.state_instance_id must be a nonempty string")
    parent = raw["spawn_parent_state_id"]
    if parent is not None and not _nonempty_str(parent):
        raise IntegrityError(f"Window {state_id} instance.spawn_parent_state_id must be null or the "
                             "parent's state_instance_id (string)")
    obs = raw["spawn_source_observation"]
    if obs is not None:
        if not isinstance(obs, dict) or set(obs) != _OBSERVATION_FIELDS:
            raise IntegrityError(f"Window {state_id} instance.spawn_source_observation must be null or "
                                 f"exactly {sorted(_OBSERVATION_FIELDS)}")
        bad = [k for k in ("run", "segment", "checkpoint") if not _nonempty_str(obs[k])]
        bad += [k for k in ("carrier", "state", "step") if not _nonnegative_int(obs[k])]
        if bad:
            raise IntegrityError(f"Window {state_id} instance.spawn_source_observation has invalid {bad} "
                                 "(run/segment/checkpoint: nonempty string; carrier/state/step: integer >= 0)")
    slot = raw["matched_additional_slot_id"]
    if slot is not None and not _nonempty_str(slot):
        raise IntegrityError(f"Window {state_id} instance.matched_additional_slot_id must be null or a nonempty string")
    return json_loads(json_bytes(raw))


def _check_instances(windows: list[dict]) -> None:
    """v2 slot table: every row has instance metadata, ids are unique, roles match the energy,
    parents exist, W/B slots pair consistently."""
    missing = [row["window_id"] for row in windows if "instance" not in row]
    if missing:
        raise IntegrityError(f"v2 state tables need instance metadata on every row; missing on windows {missing}")
    ids = [row["instance"]["state_instance_id"] for row in windows]
    if len(set(ids)) != len(ids):
        raise IntegrityError(f"duplicate state_instance_id in state table: {sorted(ids)}")
    known = set(ids)
    for row in windows:
        inst, k = row["instance"], row["aux_k"]
        if inst["state_role"] == "auxiliary" and k <= 0:
            raise IntegrityError(f"Window {row['window_id']}: state_role auxiliary requires aux_k > 0")
        if inst["state_role"] in ("ordinary", "sham") and k != 0:
            raise IntegrityError(f"Window {row['window_id']}: state_role {inst['state_role']} requires aux_k == 0")
        parent = inst["spawn_parent_state_id"]
        if parent is not None and parent == inst["state_instance_id"]:
            raise IntegrityError(f"Window {row['window_id']}: state is its own parent "
                                 f"(spawn_parent_state_id == state_instance_id {parent!r})")
        if parent is not None and parent not in known:
            raise IntegrityError(f"Window {row['window_id']}: spawn_parent_state_id {parent!r} is not a "
                                 "state_instance_id in this table")
    _check_slot_pairing(windows)


_BASELINE_FIELDS = ("center1", "k1", "center2", "k2", "gamd_lambda")


def _check_slot_pairing(windows: list[dict]) -> None:
    """W/B matched-slot rules that one table can check (spec 5, 11.1).

    Ordinary rows carry no slot. Within one table a slot id names at most one auxiliary and at
    most one sham; when both are present (e.g. a combined analysis table) the sham's baseline
    restraints and lambda must equal its auxiliary partner's. Pairing ACROSS arm tables (W has
    the auxiliary, B the sham) is checked by the Stage D preregistration validator.
    """
    by_slot: dict[str, dict[str, dict]] = {}
    for row in windows:
        inst = row["instance"]
        slot = inst["matched_additional_slot_id"]
        if slot is None:
            continue
        role = inst["state_role"]
        if role == "ordinary":
            raise IntegrityError(f"Window {row['window_id']}: an ordinary state cannot carry "
                                 f"matched_additional_slot_id {slot!r}")
        roles = by_slot.setdefault(slot, {})
        if role in roles:
            raise IntegrityError(f"matched_additional_slot_id {slot!r} names more than one {role} state "
                                 f"(windows {roles[role]['window_id']} and {row['window_id']})")
        roles[role] = row
    for slot, roles in by_slot.items():
        if "auxiliary" in roles and "sham" in roles:
            aux, sham = roles["auxiliary"], roles["sham"]
            differ = [f for f in _BASELINE_FIELDS if aux[f] != sham[f]]
            if differ:
                raise IntegrityError(f"matched_additional_slot_id {slot!r}: sham window {sham['window_id']} "
                                     f"baseline {differ} differs from auxiliary window {aux['window_id']}")


def canonical_state_definition(raw: Mapping[str, Any]) -> dict[str, Any]:
    data = json_loads(json_bytes(dict(raw)))
    schema = data.get("schema")
    base_required = {"schema", "physical_system_sha256", "ensemble", "temperature_k",
                     "pressure_bar", "fixed_box_vectors_nm", "cv1", "cv2",
                     "boost", "energy_unit", "windows"}
    if schema == STATE_SCHEMA:
        required, allowed_window = base_required, _V1_WINDOW_FIELDS
    elif schema == STATE_SCHEMA_V2:
        required, allowed_window = base_required | {"aux_models"}, _V2_WINDOW_FIELDS
    else:
        raise IntegrityError(f"State definition requires schema {STATE_SCHEMA} or {STATE_SCHEMA_V2}")
    if set(data) != required:
        raise IntegrityError(f"State definition requires exact schema {schema}; missing/extra fields")
    aux_models = _canonical_aux_models(data["aux_models"]) if schema == STATE_SCHEMA_V2 else None
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
        if set(row) - allowed_window:
            raise IntegrityError(f"Window {state_id} has unknown physics fields; normalize explicitly")
        if energy_unit == "kJ/mol":
            row["k1"] /= 4.184
            row["k2"] /= 4.184
        if schema == STATE_SCHEMA_V2:
            if "aux_k" not in row:
                raise IntegrityError(f"Window {state_id}: v2 requires explicit aux_k (inactive is 0)")
            if energy_unit == "kJ/mol":
                row["aux_k"] /= 4.184
            if row["aux_k"] > 0 and row["aux_model_sha256"] not in aux_models:
                raise IntegrityError(f"Window {state_id} names an unknown auxiliary model "
                                     f"{row['aux_model_sha256']}")
            if "instance" in row:
                row["instance"] = _canonical_instance(row["instance"], state_id)
        if row["k1"] > 0 and data["cv1"] is None:
            raise IntegrityError(f"Window {state_id} needs a frozen CV1 definition")
        if row["k2"] > 0 and data["cv2"] is None:
            raise IntegrityError(f"Window {state_id} needs a frozen CV2 definition")
    if schema == STATE_SCHEMA_V2:
        _check_instances(windows)
    if aux_models is not None:
        data["aux_models"] = aux_models
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
    aux_models: Mapping | None = None,
) -> dict[str, Any]:
    d = {
        "schema": STATE_SCHEMA, "physical_system_sha256": physical_system_sha256,
        "ensemble": ensemble, "temperature_k": temperature_k,
        "pressure_bar": pressure_bar, "fixed_box_vectors_nm": fixed_box_vectors_nm,
        "cv1": cv1, "cv2": cv2, "boost": boost,
        "energy_unit": energy_unit, "windows": list(windows),
    }
    if aux_models is not None:
        d["schema"] = STATE_SCHEMA_V2
        d["aux_models"] = dict(aux_models)
    return canonical_state_definition(d)


def hamiltonian_sha256(definition: Mapping[str, Any], window_id: int) -> str:
    """Identity of one state's potential and ensemble, excluding slot id and spawn provenance."""
    if isinstance(window_id, bool) or not isinstance(window_id, int):
        raise IntegrityError(f"window_id must be an integer, got {window_id!r}")
    state = canonical_state_definition(definition)
    rows = [row for row in state["windows"] if row["window_id"] == window_id]
    if len(rows) != 1:
        raise IntegrityError(f"No window {window_id} in this state definition")
    physics = {key: value for key, value in rows[0].items() if key not in ("window_id", "instance")}
    if physics.get("aux_k", 0.0) == 0:
        # An inactive auxiliary term is no term: hash like the same v1 physics.
        for key in ("aux_model_sha256", "aux_center", "aux_k"):
            physics.pop(key, None)
    # An active term is identified by aux_model_sha256 (the model's content hash) inside `physics`;
    # labels/provenance never enter.
    shared = {key: state[key] for key in _SHARED_FIELDS}
    return digest(json_bytes({"shared": shared, "window": physics}))


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
