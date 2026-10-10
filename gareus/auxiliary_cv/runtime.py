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

import numpy as np

from ..correctness._io import IntegrityError
from .evaluate import z_from_positions
from .features import active_feature_mask, openmm_dihedrals, unique_torsions
from .force import (AUX_FORCE_NAME, AuxForceInfo, aux_energy_function, aux_sub_cv_spec, build_aux_force,
                    set_aux_parameters)
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


def _where(replica) -> str:
    return "" if replica is None else f" on replica {replica}"


def check_aux_force(force, runtime: AuxRuntime, *, replica=None) -> None:
    """The force-side z evaluator of one replica must be this runtime's auxiliary force (F07).

    Its name, sub-CV names and energy expression (whose literals carry the model's offset and scale) must
    equal what ``build_aux_force`` writes for the runtime's model, so ``aux_z_from_force``'s projection is
    exactly the force's own z. A missing force is refused: the position evaluator is never substituted.
    """
    where = _where(replica)
    if force is None:
        raise AuxObservationError(f"force-side aux z evaluator unavailable{where}: no auxiliary force to read "
                                  "(the position evaluator is never substituted for it, F07)")
    try:
        name = force.getName()
        names = tuple(force.getCollectiveVariableName(i) for i in range(force.getNumCollectiveVariables()))
        expr = force.getEnergyFunction()
    except Exception as exc:  # noqa: BLE001 -- anything that is not a CustomCVForce is unavailable
        raise AuxObservationError(f"force-side aux z evaluator unavailable{where}: {type(exc).__name__}: {exc}") \
            from exc
    if name != AUX_FORCE_NAME:
        raise AuxObservationError(f"force-side aux z evaluator{where} is {name!r}, not {AUX_FORCE_NAME!r}")
    if names != tuple(runtime.info.sub_cv_names):
        raise AuxObservationError(f"auxiliary force{where} has sub-CVs {list(names)}, the runtime expects "
                                  f"{list(runtime.info.sub_cv_names)}")
    want = aux_energy_function(runtime.table.model, runtime.info.sub_cv_names)
    if expr != want:
        raise AuxObservationError(f"auxiliary force{where} energy expression {expr!r} is not the runtime model's "
                                  f"{want!r} (offset/scale differ)")
    _check_aux_sub_cvs(force, runtime, where)


def _check_aux_sub_cvs(force, runtime: AuxRuntime, where: str) -> None:
    """Every sub-CV's torsions (atom quads and weights, in order) and expression equal the model's, so a
    force with a wrong weight or atom is refused even when every state is inactive (no parity runs)."""
    spec = aux_sub_cv_spec(runtime.table.model)
    if len(spec) != force.getNumCollectiveVariables():
        raise AuxObservationError(f"auxiliary force{where} has {force.getNumCollectiveVariables()} sub-CVs, the "
                                  f"model {len(spec)}")
    for i, (name, expr, torsions) in enumerate(spec):
        try:
            tf = force.getCollectiveVariable(i)
            got_expr = tf.getEnergyFunction()
            got = [tf.getTorsionParameters(j) for j in range(tf.getNumTorsions())]
            periodic = bool(tf.usesPeriodicBoundaryConditions())
        except Exception as exc:  # noqa: BLE001 -- an unreadable sub-CV is unavailable, never trusted
            raise AuxObservationError(f"auxiliary force{where} sub-CV {name!r} unreadable: "
                                      f"{type(exc).__name__}: {exc}") from exc
        if got_expr != expr or periodic:
            raise AuxObservationError(f"auxiliary force{where} sub-CV {name!r} expression {got_expr!r} "
                                      f"(periodic {periodic}) is not the model's {expr!r} (non-periodic)")
        got_t = [(tuple(int(a) for a in t[:4]), [float(w) for w in t[4]]) for t in got]
        want_t = [(quad, [weight]) for quad, weight in torsions]
        if got_t != want_t:
            bad = next((j for j, (g, w) in enumerate(zip(got_t, want_t)) if g != w), min(len(got_t), len(want_t)))
            raise AuxObservationError(f"auxiliary force{where} sub-CV {name!r} torsion {bad} (atoms/weight) differs "
                                      f"from the model's ({len(got_t)} vs {len(want_t)} torsions)")


def resolve_aux_force(system, runtime: AuxRuntime, *, replica=None):
    """The auxiliary force of one replica's own System (never the setup force object), checked."""
    n = int(system.getNumForces())
    if not 0 <= runtime.force_index < n:
        raise AuxObservationError(f"replica System{_where(replica)} has {n} forces; no auxiliary force at index "
                                  f"{runtime.force_index}")
    force = system.getForce(runtime.force_index)
    check_aux_force(force, runtime, replica=replica)
    return force


def aux_z_from_force(context, force, runtime: AuxRuntime, *, replica=None) -> float:
    """z as the aux force on ``context`` computes it: (offset + sum of its sub-CV values) / scale.

    The caller has checked ``force`` (``check_aux_force``). A failed read raises; it never becomes NaN.
    """
    where = _where(replica)
    if force is None:
        raise AuxObservationError(f"force-side aux z evaluator unavailable{where}: no auxiliary force to read")
    try:
        values = np.asarray(force.getCollectiveVariableValues(context), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 -- recorded as a failure, never substituted
        raise AuxObservationError(f"force-side aux z read failed{where}: {type(exc).__name__}: {exc}") from exc
    if values.shape != (len(runtime.info.sub_cv_names),):
        raise AuxObservationError(f"auxiliary force{where} returned {values.size} sub-CV values for "
                                  f"{len(runtime.info.sub_cv_names)} sub-CVs")
    model = runtime.table.model
    return float((model.offset + values.sum()) / model.scale)


def observe_aux_z(context, runtime: AuxRuntime, *, force=None, positions_nm=None) -> float:
    """z of one carrier under the phase's auxiliary model, whatever state it occupies.

    ``force=``: the aux force's own value on ``context`` (checked first). ``positions_nm=``: the independent
    NumPy evaluator (used for parity and for the pull end-state check, never as the runtime z).
    Never raises on a non-finite value: the observation is recorded as is, and
    ``aux_bias_matrix_kcal`` refuses it when an active state needs it (sham arms may record NaN).
    The force cannot see a degenerate torsion (OpenMM returns a finite theta there).
    """
    if (force is None) == (positions_nm is None):
        raise ValueError("observe_aux_z needs exactly one of force= (force side) or positions_nm= (positions)")
    if force is not None:
        check_aux_force(force, runtime)
        return aux_z_from_force(context, force, runtime)
    return float(z_from_positions(positions_nm, runtime.table.model)[0])


def check_aux_geometry(positions_nm, runtime: AuxRuntime, *, replica=None) -> None:
    """Fail closed when a model torsion is degenerate (spec 3.3).

    Same rule as Stage A ``openmm_dihedrals`` (NaN when |b1 x b2|^2 or |b2 x b3|^2 < DEGENERATE_CROSS2_NM4),
    over the torsions that carry weight: those with at least one nonzero-coefficient feature
    (Stage A ``active_feature_mask``). Stage A's evaluator and the force both ignore zero-weight
    torsions, so z stays defined when only a zero-weight torsion is degenerate.
    """
    model = runtime.table.model
    all_quads, idx = unique_torsions(model)
    active_t = sorted(set(idx[active_feature_mask(model)].tolist()))
    quads = [all_quads[t] for t in active_t]
    theta = openmm_dihedrals(positions_nm, quads)[0]
    if np.isnan(theta).any():
        bad = [quads[t] for t in np.flatnonzero(np.isnan(theta))]
        where = "" if replica is None else f" on replica {replica}"
        raise AuxObservationError(f"degenerate auxiliary torsion(s) {bad}{where} while auxiliary states are "
                                  "active; failing the segment (spec 3.3)")


def checked_aux_forces(runtime: AuxRuntime, aux_forces) -> list:
    """Every replica's aux force, checked at observer construction (F07)."""
    forces = list(aux_forces)
    for r, force in enumerate(forces):
        check_aux_force(force, runtime, replica=r)
    return forces


def make_aux_z_observer(runtime: AuxRuntime, *, aux_forces, unit):
    """The single per-carrier z observer for both the sample and the exchange path.

    z is always the aux force's own value on the replica's Context (F07), whichever CV1/CV2 path the run
    observes its umbrella CVs on. With any active state one positions read feeds the geometry check
    (the force sees a finite theta at degenerate geometry); with none, no positions are read.
    """
    any_active = any(float(k) > 0.0 for k in runtime.table.k_kcal)
    forces = checked_aux_forces(runtime, aux_forces)

    def observe(r, sim) -> float:
        if any_active:
            pos = sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            check_aux_geometry(pos, runtime, replica=r)
        return aux_z_from_force(sim.context, forces[r], runtime, replica=r)

    return observe


def aux_bias_matrix_kcal(z_values, table: AuxStateTable) -> "np.ndarray":
    """[state, replica] auxiliary bias in kcal/mol; inactive states contribute exact zeros.

    Every replica's z enters every active row, including replicas whose own state is ordinary
    (spec 4.2). A non-finite z with any active state is fatal: the matrix never drops a state.
    """
    z = np.asarray(z_values, dtype=np.float64)
    k = np.asarray(table.k_kcal, dtype=np.float64)
    c = np.asarray(table.centers, dtype=np.float64)
    out = np.zeros((k.size, z.size), dtype=np.float64)
    active = k > 0.0
    if not np.any(active):
        return out
    if not np.all(np.isfinite(z)):
        bad = np.flatnonzero(~np.isfinite(z)).tolist()
        raise AuxObservationError(f"non-finite auxiliary z for replica(s) {bad} with active auxiliary states")
    d = z[np.newaxis, :] - c[active][:, np.newaxis]
    out[active] = 0.5 * k[active][:, np.newaxis] * d * d
    return out
