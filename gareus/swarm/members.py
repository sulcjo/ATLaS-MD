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
import re
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
    secondary_structure_torsions,
    solute_atom_indices,
)
from gareus.io import write_json
from gareus.system_setup import run_steps_safely, write_solute_only_pdb, write_state_pdb
from gareus.tica import backbone_dihedral_features

TRACE_COLUMNS = ["frame", "t_ps", "cv1", "rg_nm", "e2e_nm", "v_pep_kj", "v_dih_kj", "potential_kj"]

_DEFAULT_GRAFT_MINIMIZE_ITERS = 500  # measured 2026-09-08: 100 left 18/99 real swarm members
# NaN 0-3s into equilibration; the umbrella-seeding graft caller already uses 1000
# (--us-pull-minimize-iterations) for the same graft_conformer_into_context call.


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
    feature_fn: Optional[Callable[[], Any]] = None,
    features_path: Optional[Path] = None,
) -> dict:
    """Pure bookkeeping: equilibrate (discarded), then step/measure/write in chunks.

    ``step_fn(n)`` advances the simulation by ``n`` steps; ``measure_fn()`` returns one
    trace row's fields (without ``frame``/``t_ps``); ``write_frame_fn(frame_index)`` writes
    a seed-frame PDB for the current state. No OpenMM object is referenced here -- callers
    close over whatever simulation state they need in these three callables.

    ``feature_fn()``, when given, returns one row of canonical torsion features measured on
    the SAME frame as the trace row; the rows are stacked and written atomically to
    ``features_path`` after the loop, so row ``i`` of the features is trace row ``i``.
    """
    if n_prod_steps % steps_per_frame:
        raise ValueError(f"n_prod_steps={n_prod_steps} not divisible by steps_per_frame={steps_per_frame}")
    if (feature_fn is None) != (features_path is None):
        raise ValueError("feature_fn and features_path must be given together")
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if n_equil_steps > 0:
        step_fn(int(n_equil_steps))  # discarded by construction; never enters the trace
    n_frames = n_prod_steps // steps_per_frame
    feature_rows: list = []
    with trace_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
        w.writeheader()
        for i in range(n_frames):
            step_fn(int(steps_per_frame))
            row = dict(measure_fn())
            if feature_fn is not None:
                feature_rows.append(np.asarray(feature_fn(), dtype=np.float64))
            row["frame"] = i
            row["t_ps"] = (i + 1) * steps_per_frame * timestep_ps
            w.writerow({k: row.get(k, "") for k in TRACE_COLUMNS})
            f.flush()
            if seed_frame_every > 0 and (i + 1) % seed_frame_every == 0:
                write_frame_fn(i)
    if feature_fn is not None:
        _write_features_atomic(Path(features_path), feature_rows)
    return {"n_frames": n_frames, "n_equil_steps": int(n_equil_steps), "n_prod_steps": int(n_prod_steps)}


def _write_features_atomic(features_path: Path, feature_rows: list) -> None:
    """Stack and publish the feature rows so a killed member never leaves a short file.

    ``np.save`` appends ``.npy`` to any name that does not already end in it, so a
    ``.npy.tmp`` temp name would silently become ``.npy.tmp.npy`` and the rename would
    fail. Keep the temp name ending in ``.npy`` and write through an open handle.
    """
    stacked = np.vstack(feature_rows) if feature_rows else np.zeros((0, 0), dtype=np.float64)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = features_path.with_name(features_path.stem + ".tmp.npy")
    with tmp.open("wb") as fh:
        np.save(fh, stacked)
    tmp.replace(features_path)


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


def _count_trace_rows(trace_path: Path) -> int:
    """Number of data rows already flushed to ``trace_path`` (0 if it doesn't exist)."""
    trace_path = Path(trace_path)
    if not trace_path.exists():
        return 0
    with trace_path.open() as f:
        return sum(1 for _ in csv.DictReader(f))


_CRASH_STEP_RE_TEMPLATE = r"^CRASH_{label}_before_nan_step_(\d+)\.pdb$"


def _latest_crash_pdb(member_dir: Path, label: str) -> Optional[Path]:
    """The highest-step ``CRASH_<label>_before_nan_step_<n>.pdb`` under ``member_dir``, if any.

    ``run_steps_safely`` (gareus.system_setup) writes this file with the last finite
    coordinates before a crash; picking the highest step number is robust even if more
    than one such file were ever present (filenames are not zero-padded, so a lexical
    sort is not numerically correct).
    """
    pattern = re.compile(_CRASH_STEP_RE_TEMPLATE.format(label=re.escape(label)))
    best, best_step = None, -1
    for p in Path(member_dir).glob(f"CRASH_{label}_before_nan_step_*.pdb"):
        m = pattern.match(p.name)
        if m and int(m.group(1)) > best_step:
            best, best_step = p, int(m.group(1))
    return best


def _run_loop_recording_failure(
    *,
    n_equil_steps: int,
    n_prod_steps: int,
    steps_per_frame: int,
    seed_frame_every: int,
    step_fn: Callable[[int], Any],
    measure_fn: Callable[[], dict],
    write_frame_fn: Callable[[int], Any],
    trace_path: Path,
    timestep_ps: float,
    member_dir: Path,
    member_id,
    crash_label: str,
    t0: float,
    feature_fn: Optional[Callable[[], Any]] = None,
    features_path: Optional[Path] = None,
) -> dict:
    """Run ``run_member_loop``; a mid-loop exception is a failed member, never re-raised.

    Returns ``{"failed": False, "loop_summary": ...}`` on success or
    ``{"failed": True, "done": ...}`` (a ready-to-write ``done.json`` payload with
    ``status: "md_failed"``) on any ``Exception`` from ``step_fn``/``measure_fn``/
    ``write_frame_fn`` (e.g. ``run_steps_safely`` raising on a NaN coordinate).
    """
    try:
        loop_summary = run_member_loop(
            n_equil_steps=n_equil_steps, n_prod_steps=n_prod_steps, steps_per_frame=steps_per_frame,
            seed_frame_every=seed_frame_every, step_fn=step_fn, measure_fn=measure_fn,
            write_frame_fn=write_frame_fn, trace_path=trace_path, timestep_ps=timestep_ps,
            feature_fn=feature_fn, features_path=features_path,
        )
    except Exception as exc:
        frames_written = _count_trace_rows(trace_path)
        crash_pdb = _latest_crash_pdb(member_dir, crash_label)
        done = {
            "member_id": member_id,
            "n_frames": frames_written,
            "graft_fallback": False,
            "status": "md_failed",
            "error": repr(exc),
            "frames_written": frames_written,
            "crash_pdb": str(crash_pdb) if crash_pdb else None,
            "wall_s": time.time() - t0,
            "ns_per_day": 0.0,
        }
        print(f"WARNING: swarm member {member_id} failed mid-MD ({frames_written} frames written): {exc!r}")
        return {"failed": True, "done": done}
    return {"failed": False, "loop_summary": loop_summary}


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
    # Canonical torsion features per trace row: the discovery data for CV2 selection.
    # The atom quadruplets travel with the features so a later stage can bind them.
    phi_torsions, psi_torsions = secondary_structure_torsions(topology)
    write_json(member_dir / "torsion_index.json", {
        "phi_torsions": [list(map(int, t)) for t in phi_torsions],
        "psi_torsions": [list(map(int, t)) for t in psi_torsions],
    })
    last_positions: dict = {}

    def step_fn(n):
        run_steps_safely(sim, n, "swarm", member_dir, app, topology, unit, progress=progress)

    def measure_fn():
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        positions_nm = np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        last_positions["nm"] = positions_nm  # one State fetch per frame, shared with feature_fn
        return measure_frame(sim.context, system, unit, positions_nm, contact_pairs, ca_indices, args)

    def feature_fn():
        return backbone_dihedral_features(last_positions["nm"], phi_torsions, psi_torsions)

    def write_frame_fn(i):
        state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
        frame_path = member_dir / "frames" / f"frame_{i:05d}.pdb"
        write_solute_only_pdb(frame_path, app, topology, state.getPositions(), peptide_atoms)

    result = _run_loop_recording_failure(
        n_equil_steps=n_equil_steps, n_prod_steps=n_prod_steps, steps_per_frame=steps_per_frame,
        seed_frame_every=seed_frame_every, step_fn=step_fn, measure_fn=measure_fn,
        write_frame_fn=write_frame_fn, trace_path=member_dir / "trace.csv", timestep_ps=timestep_ps,
        member_dir=member_dir, member_id=member_row.get("member_id"), crash_label="swarm", t0=t0,
        feature_fn=feature_fn, features_path=member_dir / "torsion_features.npy",
    )
    if result["failed"]:
        done = result["done"]
        with (member_dir / "done.json").open("w") as f:
            json.dump(done, f, indent=2)
        return done
    loop_summary = result["loop_summary"]

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
