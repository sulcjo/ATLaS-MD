"""Production runtime of auxiliary-CV states: force placement, observation and bias matrices.

The auxiliary restraint must enter the integrator once, at full strength, outside every boost
channel. Its force group is therefore allocated from an audit of the System, never assumed:
groups 0/1/2 belong to Pep-GaMD's physical and auxiliary-nonbonded channels, and the two
umbrella groups are reserved even when the shared layout leaves one empty.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Optional

from ..correctness._io import IntegrityError
from .force import AuxForceInfo, build_aux_force, set_aux_parameters
from .state_table import AuxStateTable

RESERVED_PHYSICAL_GROUPS = frozenset({0, 1, 2})


class AuxObservationError(IntegrityError):
    """A live auxiliary z needed by an active state is missing or non-finite."""


def force_group_audit(system) -> list[dict]:
    out = []
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        out.append({"index": i, "class": f.__class__.__name__, "name": f.getName(),
                    "group": int(f.getForceGroup())})
    return out


def allocate_free_force_group(system, reserved: Iterable[int] = ()) -> int:
    blocked = ({a["group"] for a in force_group_audit(system)} | set(RESERVED_PHYSICAL_GROUPS)
               | {int(g) for g in reserved})
    for group in range(3, 32):
        if group not in blocked:
            return group
    raise IntegrityError(f"no free force group for the auxiliary restraint (blocked: {sorted(blocked)})")


@dataclass(frozen=True)
class AuxRuntime:
    table: AuxStateTable
    info: AuxForceInfo
    force_index: int
    topology_sha256: Optional[str] = None


def _reserved_groups(args) -> set[int]:
    return {int(getattr(args, "umbrella_force_group", 31)), int(getattr(args, "secondary_cv_force_group", 29))}


def add_aux_cv_force(openmm, system, table: AuxStateTable, args, *,
                     force_group: Optional[int] = None,
                     topology_sha256: Optional[str] = None) -> AuxRuntime:
    """Add the auxiliary restraint capability to ``system``; call before any integrator is built.

    The force is created with aux_k = aux_c = 0 (inactive) as its global defaults, so every Context
    built from this system starts with the restraint off until ``set_window(..., aux_state=)``.
    """
    reserved = _reserved_groups(args)
    if force_group is None:
        group = allocate_free_force_group(system, reserved)
    else:
        group = int(force_group)
        used = {a["group"] for a in force_group_audit(system)}
        if group in used or group in RESERVED_PHYSICAL_GROUPS or group in reserved:
            raise IntegrityError(f"auxiliary force group {group} is not free in this system "
                                 f"(used {sorted(used)}, reserved {sorted(RESERVED_PHYSICAL_GROUPS | reserved)})")
    force, info = build_aux_force(openmm, table.model, force_group=group)
    index = int(system.addForce(force))
    return AuxRuntime(table, info, index, topology_sha256)


def deactivate_aux_parameters(context, runtime: Optional[AuxRuntime]) -> None:
    """Restraint off (k = 0, centre 0) on a recon/calibration Context; no-op without a runtime."""
    if runtime is not None:
        set_aux_parameters(context, runtime.info, center=0.0, k_kcal=0.0)


def refuse_aux_population_change(n_expected: int, n_now: int, *, cause: str) -> None:
    if int(n_now) != int(n_expected):
        raise RuntimeError(
            f"auxiliary-CV state population changed by {cause}: {n_expected} -> {n_now} windows. "
            "An auxiliary state table is frozen and cannot be renumbered (spec Section 6); "
            "fix the window table instead.")


def aux_snapshot_rows(rows: list[dict], table: AuxStateTable) -> list[dict]:
    """Legacy window-snapshot rows plus the auxiliary fields (D5): strict readers then fail closed."""
    if len(rows) != table.n:
        raise RuntimeError(f"window snapshot has {len(rows)} rows for {table.n} auxiliary states")
    out = []
    for row, aux in zip(rows, table.window_rows()):
        new = dict(row)
        new.update(aux_model_sha256=aux["aux_model_sha256"], aux_center=float(aux["aux_center"]),
                   aux_k=float(aux["aux_k"]))
        out.append(new)
    return out


def _feature_atoms(model) -> set[int]:
    return {int(a) for f in model.feature_schema.features for a in f.atom_indices}


def canonical_topology_sha256(topology, model) -> str:
    """Content identity of the atom map the model's torsions index: every atom of the chains holding a
    feature atom, as (index, name, element, residue name, residue index, chain index)."""
    wanted = _feature_atoms(model)
    atoms = list(topology.atoms())
    if wanted and max(wanted) >= len(atoms):
        raise IntegrityError(f"aux model indexes atom {max(wanted)} beyond the topology ({len(atoms)} atoms)")
    chains = {atoms[i].residue.chain.index for i in wanted}
    body = [[a.index, a.name, (a.element.symbol if a.element is not None else ""),
             a.residue.name, a.residue.index, a.residue.chain.index]
            for a in atoms if a.residue.chain.index in chains]
    payload = json.dumps({"schema": "atlas-aux-topology-v1", "atoms": body},
                         sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def canonical_topology_sha256_from_pdb(path, model) -> str:
    from openmm import app
    return canonical_topology_sha256(app.PDBFile(str(path)).topology, model)


# Entry point. ASSUMPTION (board caveat 2): the digest includes each atom's residue and chain index, so it
# must be computed on the PRODUCTION topology (same chain order, same solvent/ion chains before the
# peptide). Solvent atoms never enter the body, but inserting or reordering chains ahead of the peptide's
# chain shifts the chain/residue indices and gives a different digest. Stage D computes it from the
# production topology.pdb of the run that will deploy the model.
if __name__ == "__main__":  # python -m gareus.auxiliary_cv.runtime topology-sha PDB MODEL
    import sys
    from .model import AuxModel
    if len(sys.argv) != 4 or sys.argv[1] != "topology-sha":
        raise SystemExit("usage: python -m gareus.auxiliary_cv.runtime topology-sha PDB MODEL")
    print(canonical_topology_sha256_from_pdb(sys.argv[2], AuxModel.load(sys.argv[3])))
