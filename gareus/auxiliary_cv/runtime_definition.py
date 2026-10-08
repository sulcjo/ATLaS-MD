"""Build the frozen v2 state definition and the storage-facing runtime of an auxiliary run (D10).

Only imported on the auxiliary path (``--aux-cv-model``); every OpenMM / production import is lazy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..correctness._io import IntegrityError, digest, file_digest, json_bytes, json_loads
from ..correctness.bias import normalize_windows
from ..correctness.state_identity import _canonical_cv, make_state_definition

_PATH_SUFFIXES = ("_path", "_file", "_filename")
_PATH_KEYS = {"path", "file", "filename"}
_RESTRAINT_FIELDS = ("center1", "k1", "center2", "k2")
SOLVATED_START_PDB = "01_solvated_start.pdb"


def physical_system_sha256(openmm, system) -> str:
    """Identity of the physical System (call BEFORE any umbrella/auxiliary bias force is added).

    The default periodic box is replaced by the unit cell before hashing: a resume rebuilds the System from
    01_solvated_start.pdb (rounded CRYST1), a fresh run from the in-memory topology. Box identity lives in
    the state definition (fixed_box_vectors_nm for NVT; a sample variable for NPT).
    """
    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    copy.setDefaultPeriodicBoxVectors(openmm.Vec3(1, 0, 0), openmm.Vec3(0, 1, 0), openmm.Vec3(0, 0, 1))
    return digest(openmm.XmlSerializer.serialize(copy).encode("utf-8"))


def solvated_start_topology_identities(out_dir, model) -> tuple[str, str]:
    """(Stage B ``canonical_topology_sha256``, Stage C ``topology_identity_sha256``) from ONE source.

    Both identities are read from ``<out_dir>/01_solvated_start.pdb`` -- the file setup writes and a resume
    rebuilds its topology from (``checkpoints.py``) -- so a fresh run and its resume hash the same topology.
    """
    from openmm import app
    from .checkpoint import topology_identity_sha256
    from .runtime import canonical_topology_sha256
    path = Path(out_dir) / SOLVATED_START_PDB
    if not path.is_file():
        raise IntegrityError(f"auxiliary run needs {path} for its topology identity (missing)")
    topology = app.PDBFile(str(path)).topology
    return canonical_topology_sha256(topology, model), topology_identity_sha256(topology)


def _path_stem(key: str) -> str:
    if key in _PATH_KEYS:
        return key
    return key[: -len(next(s for s in _PATH_SUFFIXES if key.endswith(s)))]


def _embed(value):
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if key.endswith(_PATH_SUFFIXES) or key in _PATH_KEYS:
                if val in (None, ""):
                    continue
                path = Path(str(val))
                if not path.is_file():
                    raise IntegrityError(f"cannot embed {key}: missing file {path}")
                out[f"{_path_stem(key)}_content_sha256"] = file_digest(path)
            elif val is None or val == "":
                continue
            else:
                out[key] = _embed(val)
        return out
    if isinstance(value, list):
        return [_embed(v) for v in value]
    return value


def embed_cv_definition(kind: str, units: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical ``{kind, units, definition}``: file references become content digests, empty values drop."""
    definition = _embed(json_loads(json_bytes(dict(raw))))
    return _canonical_cv({"kind": str(kind), "units": str(units), "definition": definition}, f"cv[{kind}]")


def _applied_rows(applied_window_key) -> list[dict[str, Any]]:
    rows = []
    for w, entry in enumerate(applied_window_key):
        row = {"window_id": w, "center1": entry[0], "k1": entry[1]}
        if len(entry) == 4:
            row.update(center2=entry[2], k2=entry[3])
        elif len(entry) != 2:
            raise IntegrityError(f"applied window key {w} must be (center1, k1[, center2, k2]), got {entry!r}")
        rows.append(row)
    return normalize_windows(rows)


def _merge_rows(snapshot_rows, aux_table) -> list[dict[str, Any]]:
    merged = []
    for i, (snap, aux) in enumerate(zip(snapshot_rows, aux_table.window_rows())):
        if "instance" not in aux:
            raise IntegrityError(f"window {i}: the auxiliary state table carries no instance provenance; a v2 "
                                 "state definition needs it on every row (ruling M2)")
        row = dict(snap)
        row.update({k: v for k, v in aux.items() if k != "instance"})
        row["instance"] = dict(aux["instance"])
        merged.append(row)
    return merged


def _check_applied(definition, applied_window_key) -> None:
    """L3: the canonical definition's restraints equal the arrays ``set_window`` applies."""
    applied = _applied_rows(applied_window_key)
    have_rows = sorted(definition["windows"], key=lambda r: r["window_id"])
    for want, have in zip(applied, have_rows):
        diff = [k for k in _RESTRAINT_FIELDS if float(want[k]) != float(have[k])]
        if diff or want["window_id"] != have["window_id"]:
            raise IntegrityError(f"state definition window {have['window_id']} {diff} disagrees with the restraints "
                                 f"set_window applies: applied {[want[k] for k in diff]}, "
                                 f"definition {[have[k] for k in diff]}")


def build_runtime_state_definition(*, physical_system_sha256: str, ensemble: str, temperature_k: float,
                                   pressure_bar, box_vectors_nm, cv1, cv2, boost, snapshot_rows: Sequence[Mapping],
                                   aux_table, applied_window_key: Sequence[Sequence]) -> dict[str, Any]:
    if not (len(snapshot_rows) == aux_table.n == len(applied_window_key)):
        raise IntegrityError(f"snapshot has {len(snapshot_rows)} windows, auxiliary table {aux_table.n}, "
                             f"applied restraints {len(applied_window_key)}")
    model = aux_table.model
    definition = make_state_definition(
        _merge_rows(snapshot_rows, aux_table), physical_system_sha256=physical_system_sha256, ensemble=ensemble,
        temperature_k=float(temperature_k), pressure_bar=pressure_bar, fixed_box_vectors_nm=box_vectors_nm,
        cv1=cv1, cv2=cv2, boost=boost, aux_models={model.model_sha256: model.to_mapping()})
    _check_applied(definition, applied_window_key)
    return definition


@dataclass(frozen=True)
class AuxIORuntime:
    state_definition: dict
    models: Mapping[str, Any]
    force_info: Any
    sample_schema: Any
    runtime: Mapping[str, str]
    energy_version: str
    phase_kind: str
    equilibrium_eligible: bool
    topology_sha256: str


def aux_io_runtime(runtime, *, state_definition, topology, args, platform, context) -> AuxIORuntime:
    """Storage-facing view of a resolved Stage B ``AuxRuntime``.

    ``topology`` must be the one read from ``01_solvated_start.pdb`` (see
    ``solvated_start_topology_identities``) so fresh runs and resumes carry the same identity.
    """
    from ..energy_decomposition import peptide_atom_groups_from_topology
    from ..kernel_identity import exchange_energy_version_for_args
    from .checkpoint import topology_identity_sha256
    from .sample_schema import build_sample_schema, runtime_info, runtime_precision
    atoms = getattr(args, "pep_gamd_peptide_atoms", None)
    if not atoms:
        atoms = peptide_atom_groups_from_topology(topology, "all-peptide")[1]
    model = runtime.table.model
    schema = build_sample_schema(topology, list(atoms), [model])
    info = runtime_info(platform.getName(), runtime_precision(platform.getName(), platform, context))
    return AuxIORuntime(state_definition, {model.model_sha256: model}, runtime.info, schema, info,
                        exchange_energy_version_for_args(args), str(args.aux_phase_kind),
                        bool(args.aux_equilibrium_eligible), topology_identity_sha256(topology))
