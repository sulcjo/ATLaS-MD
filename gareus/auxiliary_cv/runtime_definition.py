"""Build the frozen v2 state definition and the storage-facing runtime of an auxiliary run (D10).

Only imported on the auxiliary path (``--aux-cv-model``); every OpenMM / production import is lazy.
"""
from __future__ import annotations

import re
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
_OPENMM_VERSION_ATTR = re.compile(r'\sopenmmVersion="[^"]*"')


# Canonical forms of the forces a base production System (ForceField.createSystem) carries. Per force:
# child list tag -> ("terms", item tag, symmetry) | ("ordered", None, None) | ("empty", None, None).
# Symmetry: "pair" = (i, j) ~ (j, i); "angle" = (i, j, k) ~ (k, j, i); "torsion" = (i, j, k, l) ~ (l, k, j, i)
# (a dihedral angle is invariant under full reversal, propers and impropers alike).
_TERMS, _ORDERED, _EMPTY = "terms", "ordered", "empty"
_FORCE_CANONICAL = {
    "HarmonicBondForce": {"Bonds": (_TERMS, "Bond", "pair")},
    "HarmonicAngleForce": {"Angles": (_TERMS, "Angle", "angle")},
    "PeriodicTorsionForce": {"Torsions": (_TERMS, "Torsion", "torsion")},
    "NonbondedForce": {"GlobalParameters": (_ORDERED, None, None), "Particles": (_ORDERED, None, None),
                       "Exceptions": (_TERMS, "Exception", "pair"),
                       # offsets reference particle/exception INDICES: sorting exceptions would break them
                       "ParticleOffsets": (_EMPTY, None, None), "ExceptionOffsets": (_EMPTY, None, None)},
    "CMMotionRemover": {},
}
_SYSTEM_CHILDREN = {"PeriodicBoxVectors", "Particles", "Constraints", "Forces"}


def _canonical_term(element, symmetry) -> list:
    atom_keys = sorted((k for k in element.attrib if k[:1] == "p" and k[1:].isdigit()), key=lambda k: int(k[1:]))
    atoms = [int(element.attrib[k]) for k in atom_keys]
    if symmetry == "pair":
        atoms = sorted(atoms)
    elif symmetry == "angle" and atoms[0] > atoms[-1]:
        atoms = atoms[::-1]
    elif symmetry == "torsion" and atoms[::-1] < atoms:
        atoms = atoms[::-1]
    params = sorted((k, v) for k, v in element.attrib.items() if k not in atom_keys)
    return [atoms, params]


def _canonical_list(parent, child, rule, where):
    kind, item, symmetry = rule
    items = list(child)
    if kind == _EMPTY:
        if items:
            raise IntegrityError(f"physical_system_sha256: {where} has {child.tag} ({len(items)}); they reference "
                                 "term indices, so this System has no order-insensitive canonical form")
        return []
    if kind == _ORDERED:
        if any(len(e) for e in items):
            raise IntegrityError(f"physical_system_sha256: {where}/{child.tag} item with child elements has no "
                                 "canonical form")
        return [[e.tag, sorted(e.attrib.items())] for e in items]
    if any(e.tag != item or len(e) for e in items):
        raise IntegrityError(f"physical_system_sha256: unexpected element in {where}/{child.tag}")
    return sorted(_canonical_term(e, symmetry) for e in items)


def physical_system_sha256(openmm, system) -> str:
    """Identity of the physical System (call BEFORE any umbrella/auxiliary bias force is added).

    Order-insensitive (Task 14 F1): a fresh run builds its System from the in-memory topology, a resume from
    01_solvated_start.pdb, and the PDB round trip lists bonds, constraints and bonded terms in another order
    and direction (same physics, different serialization). Bond, angle, torsion, constraint and exception
    terms are therefore direction-normalised and sorted; particles are never reordered (their index is
    their identity), forces keep their order. A force type without a canonical form here is refused, never
    hashed raw, and so is a System with virtual sites (children of <Particle>), extra System elements, or
    index-referenced parameter offsets. The default periodic box is excluded (box identity lives in the state definition:
    fixed_box_vectors_nm for NVT, a sample variable for NPT), as are barostats (ensemble machinery; the
    ensemble and pressure live in the state definition; Task 13 carry-over 7) and the OpenMM release stamp
    (an OpenMM update alone must not make an auxiliary run unresumable; kernel identity and parity checks
    cover physics changes).
    """
    import xml.etree.ElementTree as ET
    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    for i in reversed(range(copy.getNumForces())):
        if "Barostat" in copy.getForce(i).__class__.__name__:
            copy.removeForce(i)
    root = ET.fromstring(_OPENMM_VERSION_ATTR.sub("", openmm.XmlSerializer.serialize(copy), count=1))
    extra = {c.tag for c in root} - _SYSTEM_CHILDREN
    if extra:
        raise IntegrityError(f"physical_system_sha256: System element(s) {sorted(extra)} have no canonical form")
    particles = list(root.find("Particles"))
    if any(len(e) for e in particles):
        # A virtual site serialises as a child of its <Particle mass="0">: hashing the attributes alone would
        # let Systems with different virtual sites hash equal (fix round 2).
        raise IntegrityError("physical_system_sha256: a particle carries a virtual site (child element); "
                             "virtual sites have no canonical form here")
    out: dict[str, Any] = {
        "system": sorted((k, v) for k, v in root.attrib.items() if k != "openmmVersion"),
        "particles": [sorted(e.attrib.items()) for e in particles],
        "constraints": sorted(_canonical_term(e, "pair") for e in root.find("Constraints")),
        "forces": [],
    }
    for force in root.find("Forces"):
        ftype = force.attrib.get("type")
        rules = _FORCE_CANONICAL.get(ftype)
        if rules is None:
            raise IntegrityError(f"physical_system_sha256: force type {ftype!r} has no canonical form; refusing "
                                 "to hash it raw (extend _FORCE_CANONICAL with its term symmetries)")
        unknown = {c.tag for c in force} - set(rules)
        if unknown:
            raise IntegrityError(f"physical_system_sha256: {ftype} child element(s) {sorted(unknown)} have no "
                                 "canonical form")
        out["forces"].append({"attrs": sorted(force.attrib.items()),
                              "lists": {c.tag: _canonical_list(force, c, rules[c.tag], ftype) for c in force}})
    return digest(json_bytes(out))


def solvated_start_topology_identities(out_dir, model) -> tuple[str, str]:
    """(Stage B ``canonical_topology_sha256``, Stage C ``topology_identity_sha256``) from ONE source.

    Both identities are read from ``<out_dir>/01_solvated_start.pdb`` -- the file setup writes and a resume
    rebuilds its topology from (``checkpoints.py``) -- so a fresh run and its resume hash the same topology.
    """
    from .checkpoint import topology_identity_sha256
    from .runtime import canonical_topology_sha256
    topology = solvated_start_topology(out_dir)
    return canonical_topology_sha256(topology, model), topology_identity_sha256(topology)


def solvated_start_topology(out_dir):
    """The topology of ``<out_dir>/01_solvated_start.pdb`` (missing file -> IntegrityError)."""
    from openmm import app
    path = Path(out_dir) / SOLVATED_START_PDB
    if not path.is_file():
        raise IntegrityError(f"auxiliary run needs {path} for its topology identity (missing)")
    return app.PDBFile(str(path)).topology


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
