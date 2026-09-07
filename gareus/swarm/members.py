"""One swarm member's MD (spec Sec.2 S1 product (1)): graft a stratified seed into the
equilibrated box, equilibrate, run one unbiased ``--swarm-seed-ns`` production trajectory
with the Pep-GaMD auxiliary force present but excluded from integration (boost off, no
umbrella), writing a per-frame trace CSV plus periodic seed-frame PDBs and a last-frame PDB.

The bookkeeping (``run_member_loop``) takes injected ``step_fn``/``measure_fn``/
``write_frame_fn`` callables so it is fully unit-tested without any OpenMM/MD. A failed
graft is a failed member (global constraint: every member starts from a grafted seed, never
from the extended chain) -- ``run_member`` records ``status: "graft_failed"`` and returns
without ever stepping the extended-chain coordinates.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from gareus.pep_gamd import (
    DIHEDRAL_GROUP,
    ensure_pep_gamd_partition,
    peptide_essential_energy_kj,
    physical_potential_energy_kj,
)
from gareus.production import make_cmd_integrator
from gareus.seeding import graft_conformer_into_context
from gareus.cv import (
    find_atom_in_residue,
    nonlocal_contact_cv_from_positions_nm,
    peptide_residues,
    solute_atom_indices,
)
from gareus.system_setup import run_steps_safely, write_state_pdb

TRACE_COLUMNS = ["frame", "t_ps", "cv1", "rg_nm", "e2e_nm", "v_pep_kj", "v_dih_kj", "potential_kj"]

_DEFAULT_GRAFT_MINIMIZE_ITERS = 100


def measure_frame(context, system, unit, positions_nm, contact_pairs, ca_indices, args) -> Dict[str, float]:
    """One trace row's measured quantities (without ``frame``/``t_ps``, added by the caller)."""
    from gareus.swarm.stratify import rg_and_e2e_nm

    v_pep = peptide_essential_energy_kj(context, unit)
    v_dih = float(
        context.getState(getEnergy=True, groups={DIHEDRAL_GROUP}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    )
    potential = physical_potential_energy_kj(context, system, unit)
    cv1 = nonlocal_contact_cv_from_positions_nm(positions_nm, contact_pairs, args)
    ca_pos_nm = np.asarray(positions_nm, dtype=float)[np.asarray(ca_indices, dtype=int)]
    rg_nm, e2e_nm = rg_and_e2e_nm(ca_pos_nm)
    return {
        "cv1": float(cv1),
        "rg_nm": float(rg_nm),
        "e2e_nm": float(e2e_nm),
        "v_pep_kj": float(v_pep),
        "v_dih_kj": float(v_dih),
        "potential_kj": float(potential),
    }


def run_member_loop(
    *,
    n_equil_steps: int,
    n_prod_steps: int,
    steps_per_frame: int,
    seed_frame_every: int,
    step_fn: Callable[[int], Any],
    measure_fn: Callable[[], dict],
    write_frame_fn: Callable[[int], Any],
    trace_path: Path,
    timestep_ps: float = 0.004,
) -> dict:
    """Pure bookkeeping: equilibrate (discarded), then step/measure/write in chunks.

    ``step_fn(n)`` advances the simulation by ``n`` steps; ``measure_fn()`` returns one
    trace row's fields (without ``frame``/``t_ps``); ``write_frame_fn(frame_index)`` writes
    a seed-frame PDB for the current state. No OpenMM object is referenced here -- callers
    close over whatever simulation state they need in these three callables.
    """
    if n_prod_steps % steps_per_frame:
        raise ValueError(f"n_prod_steps={n_prod_steps} not divisible by steps_per_frame={steps_per_frame}")
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if n_equil_steps > 0:
        step_fn(int(n_equil_steps))  # discarded by construction; never enters the trace
    n_frames = n_prod_steps // steps_per_frame
    with trace_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
        w.writeheader()
        for i in range(n_frames):
            step_fn(int(steps_per_frame))
            row = dict(measure_fn())
            row["frame"] = i
            row["t_ps"] = (i + 1) * steps_per_frame * timestep_ps
            w.writerow({k: row.get(k, "") for k in TRACE_COLUMNS})
            f.flush()
            if seed_frame_every > 0 and (i + 1) % seed_frame_every == 0:
                write_frame_fn(i)
    return {"n_frames": n_frames, "n_equil_steps": int(n_equil_steps), "n_prod_steps": int(n_prod_steps)}


def member_done(member_dir: Path) -> bool:
    """True iff ``<member_dir>/done.json`` exists and parses as JSON."""
    path = Path(member_dir) / "done.json"
    if not path.exists():
        return False
    try:
        with path.open() as f:
            json.load(f)
        return True
    except (OSError, ValueError):
        return False


def _terminal_ca_atoms(topology) -> tuple[int, int]:
    """First and last peptide residue's CA atom index (used only to report the graft's
    before/after distance CV in provenance; not the stratification CV1)."""
    residues = peptide_residues(topology)
    return find_atom_in_residue(residues[0], "CA"), find_atom_in_residue(residues[-1], "CA")


def _write_peptide_only_pdb(path: Path, app, topology, positions, peptide_indices) -> None:
    """Write a peptide-atom-only PDB (GENPEPT-seed-compatible: no water/ions)."""
    keep = {int(i) for i in peptide_indices}
    modeller = app.Modeller(topology, positions)
    to_delete = [a for a in topology.atoms() if int(a.index) not in keep]
    modeller.delete(to_delete)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)


def run_member(
    args,
    member_row: dict,
    member_dir: Path,
    *,
    openmm,
    app,
    unit,
    topology,
    base_system_xml: str,
    equil_state,
    conformer: Optional[dict],
    platform,
    props,
    contact_pairs,
    progress=None,
) -> dict:
    """Run one swarm member: graft, equilibrate, 1 ns unbiased, write trace + seed frames.

    Every member starts from a grafted seed (global constraint) -- ``conformer`` is
    required. A failed graft is a failed member: ``done.json`` records
    ``status: "graft_failed"`` and no trace/frames are written; the extended-chain
    coordinates from ``equil_state`` are never stepped.
    """
    if conformer is None:
        raise ValueError("swarm member needs a seed conformer")

    member_dir = Path(member_dir)
    member_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    system = openmm.XmlSerializer.deserialize(base_system_xml)  # fresh System per member
    peptide_atoms = solute_atom_indices(topology)
    ensure_pep_gamd_partition(system, peptide_atoms)

    velocity_seed = int(member_row["velocity_seed"])
    had_seed = hasattr(args, "seed")
    original_seed = getattr(args, "seed", None)
    try:
        args.seed = velocity_seed
        integrator, _ = make_cmd_integrator(openmm, args, unit, system=system)
    finally:
        if had_seed:
            args.seed = original_seed
        else:
            delattr(args, "seed")

    sim = app.Simulation(topology, system, integrator, platform, props)
    sim.context.setPeriodicBoxVectors(*equil_state.getPeriodicBoxVectors())
    sim.context.setPositions(equil_state.getPositions())

    cv_atom1, cv_atom2 = _terminal_ca_atoms(topology)
    minimize_iters = int(getattr(args, "swarm_graft_minimize_iters", _DEFAULT_GRAFT_MINIMIZE_ITERS))
    graft_status = graft_conformer_into_context(
        sim, topology, conformer, cv_atom1, cv_atom2,
        float(args.temperature_k), unit,
        minimize_iters=minimize_iters, seed=velocity_seed,
    )
    if graft_status.get("fallback"):
        done = {
            "member_id": member_row.get("member_id"),
            "n_frames": 0,
            "graft_fallback": True,
            "status": "graft_failed",
            "graft_fallback_reason": graft_status.get("fallback_reason"),
            "wall_s": time.time() - t0,
            "ns_per_day": 0.0,
        }
        with (member_dir / "done.json").open("w") as f:
            json.dump(done, f, indent=2)
        return done

    timestep_fs = float(args.timestep_fs)
    timestep_ps = timestep_fs / 1000.0
    output_interval_ps = float(getattr(args, "swarm_output_interval_ps", 2.0))
    steps_per_frame = max(1, round(output_interval_ps * 1000.0 / timestep_fs))
    seed_ns = float(getattr(args, "swarm_seed_ns", 1.0))
    n_prod_steps_requested = max(steps_per_frame, round(seed_ns * 1e6 / timestep_fs))
    n_frames_wanted = -(-n_prod_steps_requested // steps_per_frame)  # ceil: round up to a whole multiple
    n_prod_steps = int(n_frames_wanted * steps_per_frame)
    n_equil_steps = round(float(getattr(args, "swarm_equil_ps", 100.0)) * 1000.0 / timestep_fs)
    seed_frame_interval_ps = float(getattr(args, "swarm_seed_frame_interval_ps", 20.0))
    seed_frame_every = max(1, round(seed_frame_interval_ps / output_interval_ps))

    ca_indices = [find_atom_in_residue(res, "CA") for res in peptide_residues(topology)]

    def step_fn(n):
        run_steps_safely(sim, n, "swarm", member_dir, app, topology, unit, progress=progress)

    def measure_fn():
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        positions_nm = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        return measure_frame(sim.context, system, unit, positions_nm, contact_pairs, ca_indices, args)

    def write_frame_fn(i):
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        frame_path = member_dir / "frames" / f"frame_{i:05d}.pdb"
        _write_peptide_only_pdb(frame_path, app, topology, state.getPositions(), peptide_atoms)

    loop_summary = run_member_loop(
        n_equil_steps=n_equil_steps, n_prod_steps=n_prod_steps, steps_per_frame=steps_per_frame,
        seed_frame_every=seed_frame_every, step_fn=step_fn, measure_fn=measure_fn,
        write_frame_fn=write_frame_fn, trace_path=member_dir / "trace.csv", timestep_ps=timestep_ps,
    )

    final_state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    write_state_pdb(member_dir / "last_frame.pdb", app, topology, final_state.getPositions())

    wall_s = time.time() - t0
    actual_prod_ns = n_prod_steps * timestep_ps / 1000.0
    ns_per_day = (actual_prod_ns / wall_s * 86400.0) if wall_s > 0 else float("nan")
    done = {
        "member_id": member_row.get("member_id"),
        "n_frames": loop_summary["n_frames"],
        "graft_fallback": False,
        "status": "ok",
        "n_equil_steps": n_equil_steps,
        "n_prod_steps": n_prod_steps,
        "actual_prod_ns": actual_prod_ns,
        "wall_s": wall_s,
        "ns_per_day": ns_per_day,
        "graft": {k: v for k, v in graft_status.items() if k != "fallback"},
    }
    with (member_dir / "done.json").open("w") as f:
        json.dump(done, f, indent=2)
    return done
