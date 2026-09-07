"""Unbiased swarm stage orchestrator (spec Sec.2 S1).

Builds or reloads the solvated/equilibrated system once, applies the Pep-GaMD
partition per member (never at the shared-system level -- ``base_system.xml``
is serialised *without* the auxiliary force), loads the GENPEPT seed library
(round 0) or a production-frames CSV (round >= 1, frozen coordinate), describes
and stratifies the seeds deterministically, and runs members with
``--swarm-member-range`` sharding and ``done.json``-based resume.

Ab initio (global constraint): every member starts from a grafted seed. The
stage refuses to run at all without ``--seed-conformers-dir`` pointing at a
readable ``final_survivor_seeds.csv`` -- a contact-CV start from the extended
chain cannot be pulled into coverage (S3 pilot finding).

Seed-library loading, per-row CSV validation and per-member conformer
identity resolution live in ``gareus.swarm.seed_library`` (split out to keep
this module under the line-count cap).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gareus.cv import prepare_primary_cv_definition
from gareus.imports import import_openmm
from gareus.system_setup import (
    create_system,
    make_forcefield,
    minimize_and_npt_equilibrate,
    setup_platform_and_properties,
    write_state_pdb,
)
from gareus.swarm.members import member_done, run_member
from gareus.swarm.seed_library import (
    _ca_indices_in_seed,
    _load_seed_library_for_round,
    _resolve_conformer,
)
from gareus.swarm.stratify import (
    SeedDescriptor,
    _bin_index,
    describe_seeds,
    plan_members,
    quantile_edges,
    stratify_cells,
)

PLAN_COLUMNS = ["member_id", "cell_id", "cell_cv1", "cell_rg", "cell_e2e",
                "seed_id", "seed_pdb", "replicate", "velocity_seed"]

_ROUND_DIR_RE = re.compile(r"^round_(\d+)$")


def swarm_root(out_dir) -> Path:
    return Path(out_dir) / "swarm"


def round_dir(out_dir, round_index) -> Path:
    return swarm_root(out_dir) / f"round_{int(round_index):03d}"


def latest_round_index(out_dir) -> Optional[int]:
    root = swarm_root(out_dir)
    if not root.exists():
        return None
    indices = [int(m.group(1)) for p in root.iterdir() if p.is_dir()
               for m in [_ROUND_DIR_RE.match(p.name)] if m]
    return max(indices) if indices else None


def parse_member_range(spec: Optional[str], n_members: int) -> range:
    """``"a:b"`` -> ``range(a, b)`` clipped to ``[0, n_members]`` (half-open); ``None`` -> all."""
    if not spec:
        return range(n_members)
    a_str, b_str = str(spec).split(":", 1)
    a = max(0, min(int(a_str), n_members))
    b = max(0, min(int(b_str), n_members))
    if b < a:
        b = a
    return range(a, b)


def _parse_bins(spec) -> Tuple[int, int, int]:
    if isinstance(spec, (tuple, list)):
        values = [int(x) for x in spec]
    else:
        values = [int(x) for x in str(spec).split(",")]
    if len(values) != 3:
        raise ValueError(f"--swarm-bins needs exactly 3 comma-separated ints, got {spec!r}")
    return tuple(values)  # type: ignore[return-value]


def stratify_with_frozen_edges(seeds: List[SeedDescriptor], edges: Dict[str, list]) -> Dict[Tuple[int, int, int], List[SeedDescriptor]]:
    """Bin ``seeds`` against previously persisted quantile edges (round >= 1: the
    coordinate is frozen after round 0, spec Sec.3.6 -- never re-fit here)."""
    e_cv1 = np.asarray(edges["cv1"], dtype=float)
    e_rg = np.asarray(edges["rg"], dtype=float)
    e_e2e = np.asarray(edges["e2e"], dtype=float)
    cells: Dict[Tuple[int, int, int], List[SeedDescriptor]] = {}
    for s in seeds:
        key = (_bin_index(s.cv1, e_cv1), _bin_index(s.rg_nm, e_rg), _bin_index(s.e2e_nm, e_e2e))
        cells.setdefault(key, []).append(s)
    return dict(sorted(cells.items()))


def _load_frozen_edges(out_dir) -> Dict[str, list]:
    meta_path = round_dir(out_dir, 0) / "plan_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"--swarm-round >= 1 needs {meta_path} (round 0's frozen bin edges); run round 0 first")
    edges = json.loads(meta_path.read_text()).get("edges")
    if not edges:
        raise SystemExit(f"{meta_path} has no 'edges' -- round 0 must persist quantile_edges before round >= 1 reuses them")
    return edges


def _write_plan(rd: Path, rows: List[dict], meta: dict) -> None:
    rd.mkdir(parents=True, exist_ok=True)
    with (rd / "plan.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in PLAN_COLUMNS})
    (rd / "plan_meta.json").write_text(json.dumps(meta, indent=2))


_INT_PLAN_FIELDS = ("member_id", "cell_cv1", "cell_rg", "cell_e2e", "replicate", "velocity_seed")


def _load_plan(rd: Path) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    with (rd / "plan.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            out_row = dict(row)
            for key in _INT_PLAN_FIELDS:
                out_row[key] = int(row[key])
            rows.append(out_row)
    meta = json.loads((rd / "plan_meta.json").read_text())
    return rows, meta


def build_or_load_plan(args, out_dir, round_index: int, *, topology, contact_pairs) -> Tuple[List[dict], dict]:
    """``plan.csv``/``plan_meta.json`` for one round: build deterministically once,
    then read back verbatim on every later call (resume must never re-derive a
    different plan)."""
    rd = round_dir(out_dir, round_index)
    if (rd / "plan.csv").exists() and (rd / "plan_meta.json").exists():
        return _load_plan(rd)

    bins = _parse_bins(getattr(args, "swarm_bins", "4,3,3"))
    seed_ns = float(getattr(args, "swarm_seed_ns", 1.0))
    replicates_per_cell = int(getattr(args, "swarm_replicates_per_cell", 3))
    budget_ns_raw = getattr(args, "swarm_budget_ns", None)
    budget_ns = float(budget_ns_raw) if budget_ns_raw is not None else None
    base_seed = int(getattr(args, "seed", 0) or 0)

    primary_cv_def = prepare_primary_cv_definition(topology, args)
    library = _load_seed_library_for_round(args, round_index, topology=topology, primary_cv_def=primary_cv_def)

    if round_index == 0:
        if not contact_pairs:
            raise SystemExit(
                "--swarm-stage needs a nonlocal-contacts primary CV (contact_pairs is empty); "
                "heavy-CV1 is the swarm's stratification coordinate and a distance-mode CV "
                "cannot stratify it"
            )
        ca_indices_in_seed = _ca_indices_in_seed(library, topology)
        dropped: list = []
        seeds = describe_seeds(library, ca_indices_in_seed, contact_pairs, args, dropped_out=dropped)
        cells = stratify_cells(seeds, bins)
        edges = {
            "cv1": quantile_edges(np.array([s.cv1 for s in seeds]), bins[0]).tolist(),
            "rg": quantile_edges(np.array([s.rg_nm for s in seeds]), bins[1]).tolist(),
            "e2e": quantile_edges(np.array([s.e2e_nm for s in seeds]), bins[2]).tolist(),
        }
    else:
        # cv1/rg_nm/e2e_nm were already validated (finite, non-empty) by
        # _validate_production_frame_row when the library was loaded -- no
        # re-parsing of raw CSV strings here.
        seeds = [
            SeedDescriptor(
                seed_id=f"seed_{i:05d}", pdb_path=str(entry["pdb_path"]),
                cv1=float(entry["primary_cv_value"]),
                rg_nm=float(entry["validated_rg_nm"]),
                e2e_nm=float(entry["validated_e2e_nm"]),
            )
            for i, entry in enumerate(library)
        ]
        edges = _load_frozen_edges(out_dir)
        cells = stratify_with_frozen_edges(seeds, edges)

    rows, meta = plan_members(cells, replicates_per_cell, seed_ns=seed_ns, budget_ns=budget_ns, base_seed=base_seed)
    meta["edges"] = edges
    meta["round_index"] = int(round_index)
    meta["bins"] = list(bins)
    _write_plan(rd, rows, meta)
    return rows, meta


def ensure_system(args, out_dir, progress) -> dict:
    """Build the solvated/equilibrated system once, or reload it on resume/later rounds.

    Writes/reads ``swarm/system/{topology.pdb,equil_state.xml,base_system.xml}``.
    ``base_system.xml`` never carries the Pep-GaMD auxiliary force -- each member
    adds its own partition (``ensure_pep_gamd_partition``) to a fresh deserialised
    copy, per the frozen-envelope/no-shared-mutable-System discipline.
    """
    out_dir = Path(out_dir)
    sys_dir = swarm_root(out_dir) / "system"
    topology_pdb = sys_dir / "topology.pdb"
    equil_state_xml = sys_dir / "equil_state.xml"
    base_system_xml_path = sys_dir / "base_system.xml"

    if topology_pdb.exists() and equil_state_xml.exists() and base_system_xml_path.exists():
        openmm, app, unit = import_openmm()
        forcefield = make_forcefield(app, args.water_model)
        pdb = app.PDBFile(str(topology_pdb))
        equil_state = openmm.XmlSerializer.deserialize(equil_state_xml.read_text(encoding="utf-8"))
        base_system_xml = base_system_xml_path.read_text(encoding="utf-8")
        platform, props = setup_platform_and_properties(openmm, args)
        return {
            "openmm": openmm, "app": app, "unit": unit, "forcefield": forcefield,
            "topology": pdb.topology, "equil_state": equil_state, "base_system_xml": base_system_xml,
            "platform": platform, "props": props, "reloaded": True,
        }

    sys_dir.mkdir(parents=True, exist_ok=True)
    openmm, app, unit, forcefield, topology, final_state = minimize_and_npt_equilibrate(args, out_dir, progress)
    write_state_pdb(topology_pdb, app, topology, final_state.getPositions())
    equil_state_xml.write_text(openmm.XmlSerializer.serialize(final_state), encoding="utf-8")
    base_system = create_system(app, unit, forcefield, topology, args, include_barostat=True)
    base_system_xml = openmm.XmlSerializer.serialize(base_system)
    base_system_xml_path.write_text(base_system_xml, encoding="utf-8")
    platform, props = setup_platform_and_properties(openmm, args)
    return {
        "openmm": openmm, "app": app, "unit": unit, "forcefield": forcefield,
        "topology": topology, "equil_state": final_state, "base_system_xml": base_system_xml,
        "platform": platform, "props": props, "reloaded": False,
    }


def run_swarm_stage(args, out_dir, progress=None) -> dict:
    """Build/reload the system, build/load one round's plan, and run its members.

    ``--swarm-member-range`` shards the round: this call only iterates its own
    shard, so the returned ``failed_in_range``/``missing_in_range`` (and
    ``status_counts``) describe only the members attempted *by this call*, not
    the whole round -- pooling across shards to answer "is the round done" is
    Task 8's job (``analyze_swarm_stage`` reads every member directory under
    the round, regardless of which shard produced it).
    """
    out_dir = Path(out_dir)
    seed_conformers_dir = getattr(args, "seed_conformers_dir", None)
    if not seed_conformers_dir or not (Path(seed_conformers_dir) / "final_survivor_seeds.csv").exists():
        raise SystemExit(
            "--swarm-stage run needs --seed-conformers-dir <GENPEPT library> (final_survivor_seeds.csv); "
            "a contact-CV start from the extended chain is not possible"
        )

    sysinfo = ensure_system(args, out_dir, progress)
    topology = sysinfo["topology"]

    round_index = int(getattr(args, "swarm_round", 0) or 0)
    rd = round_dir(out_dir, round_index)

    primary_cv_def = prepare_primary_cv_definition(topology, args)
    contact_pairs = primary_cv_def["contact_pairs"]

    rows, meta = build_or_load_plan(args, out_dir, round_index, topology=topology, contact_pairs=contact_pairs)
    library = _load_seed_library_for_round(args, round_index, topology=topology, primary_cv_def=primary_cv_def)
    library_by_path = {str(entry.get("pdb_path", "")): entry for entry in library}

    member_range = parse_member_range(getattr(args, "swarm_member_range", None), len(rows))

    status_counts: Dict[str, int] = {}
    n_run = n_skipped_resume = 0
    failed_in_range: List[int] = []     # done.json exists, status != "ok" (graft/MD failure) -- this shard only
    missing_in_range: List[int] = []    # attempted this call but done.json still absent afterwards -- this shard only
    for i in member_range:
        row = rows[i]
        member_id = int(row["member_id"])
        member_dir = rd / f"member_{member_id:04d}"

        if member_done(member_dir):
            n_skipped_resume += 1
            done = json.loads((member_dir / "done.json").read_text())
            status = str(done.get("status", "ok"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != "ok":
                failed_in_range.append(member_id)
            continue

        conformer = _resolve_conformer(row, library, library_by_path)

        done = run_member(
            args, row, member_dir,
            openmm=sysinfo["openmm"], app=sysinfo["app"], unit=sysinfo["unit"], topology=topology,
            base_system_xml=sysinfo["base_system_xml"], equil_state=sysinfo["equil_state"], conformer=conformer,
            platform=sysinfo["platform"], props=sysinfo["props"], contact_pairs=contact_pairs, progress=progress,
        )
        n_run += 1
        status = str(done.get("status", "ok"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if not member_done(member_dir):
            missing_in_range.append(member_id)
        elif status != "ok":
            failed_in_range.append(member_id)

        print(
            f"member {member_id:04d}/{len(rows):04d} cell {row['cell_id']} seed {row['seed_id']} "
            f"rep {row['replicate']} status {status} {float(done.get('ns_per_day', 0.0)):.0f} ns/day"
        )
        if progress is not None:
            progress.emit({
                "event": "swarm_member_done", "round": round_index, "member_id": member_id,
                "n_members": len(rows), "cell_id": row["cell_id"], "seed_id": row["seed_id"],
                "status": status, "ns_per_day": float(done.get("ns_per_day", 0.0)),
            })

    return {
        "status": "ok",
        "round": round_index,
        "n_members": len(rows),
        "n_run": n_run,
        "failed_in_range": failed_in_range,
        "n_skipped_resume": n_skipped_resume,
        "status_counts": status_counts,
        "missing_in_range": missing_in_range,
        "plan_meta": meta,
    }
