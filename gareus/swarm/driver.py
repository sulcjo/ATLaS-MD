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
import concurrent.futures
import copy
import threading
from pathlib import Path

from gareus.io import read_json_file, write_csv_atomic, write_json
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
    """Persist one round's member plan, all-or-nothing.

    ``plan.csv`` is the round's definition of how many members exist; every
    later job in the chain reads it back verbatim rather than re-deriving it.
    A truncated plan is therefore not a crash but a silently smaller round, so
    the CSV is staged and renamed. ``plan_meta.json`` is written last and acts
    as the completion marker that ``build_or_load_plan`` gates on.
    """
    rd.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(
        rd / "plan.csv",
        PLAN_COLUMNS,
        ([row.get(k, "") for k in PLAN_COLUMNS] for row in rows),
    )
    write_json(rd / "plan_meta.json", meta)


SEED_DESCRIPTOR_COLUMNS = ["seed_id", "pdb_path", "cv1", "rg_nm", "e2e_nm"]


def _write_seed_descriptors(rd: Path, seeds: List[SeedDescriptor]) -> None:
    """Round 0's per-seed heavy-CV1/Rg/E2E descriptors, one row per library seed in
    ``describe_seeds``'s own (deterministic) order. Analysis reads this back as the
    *real* library CV1 sample for ``_library_cv1_for_round0``'s reported diagnostic
    (``report["ladder_design"]["library_q99"]``, no longer a ladder-design veto as of
    the 2026-09-08 fix) -- ``plan_meta["edges"]["cv1"]`` is only bin *boundaries* and
    understates its q99 (controller review, round 1 fix)."""
    rd.mkdir(parents=True, exist_ok=True)
    with (rd / "seed_descriptors.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEED_DESCRIPTOR_COLUMNS)
        w.writeheader()
        for s in seeds:
            w.writerow({"seed_id": s.seed_id, "pdb_path": s.pdb_path, "cv1": s.cv1, "rg_nm": s.rg_nm, "e2e_nm": s.e2e_nm})


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
        _write_seed_descriptors(rd, seeds)
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

    n_workers, device_tokens = resolve_member_workers(args)
    missing_in_range: List[int] = []
    _missing_lock = threading.Lock()

    def _run_one(worker_args, row, member_dir, device):
        member_id = int(row["member_id"])
        conformer = _resolve_conformer(row, library, library_by_path)
        # Each worker pins its own accelerator; the property key differs by
        # platform, so set whichever one this platform actually published.
        props = dict(sysinfo["props"] or {})
        for key in ("CudaDeviceIndex", "DeviceIndex", "OpenCLDeviceIndex"):
            if key in props:
                props[key] = str(device)
                break
        done = run_member(
            worker_args, row, member_dir,
            openmm=sysinfo["openmm"], app=sysinfo["app"], unit=sysinfo["unit"], topology=topology,
            base_system_xml=sysinfo["base_system_xml"], equil_state=sysinfo["equil_state"], conformer=conformer,
            platform=sysinfo["platform"], props=props, contact_pairs=contact_pairs, progress=progress,
        )
        status = str(done.get("status", "ok"))
        if not member_done(member_dir):
            with _missing_lock:
                missing_in_range.append(member_id)
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
        return done

    if n_workers > 1:
        print(f"    swarm: {n_workers} concurrent members over devices {', '.join(device_tokens)}")
    tally = execute_members(args, rows, member_range, rd, _run_one,
                            n_workers=n_workers, device_tokens=device_tokens)
    n_run = tally["n_run"]
    n_skipped_resume = tally["n_skipped_resume"]
    failed_in_range = tally["failed_in_range"]
    status_counts = tally["status_counts"]
    missing_in_range = sorted(set(missing_in_range))

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


#: Platforms where separate workers land on separate accelerators. Everything
#: else shares one device and gains nothing from concurrency here.
_GPU_PLATFORM_NAMES = frozenset({"CUDA", "HIP", "OpenCL"})


def resolve_member_workers(args) -> Tuple[int, List[str]]:
    """How many members to run at once, and which devices to spread them over.

    "auto" means one worker per device token, matching what --us-pull-workers
    already does for umbrella seeding. An unparseable value falls back to serial
    rather than raising: a bad worker count should slow a campaign down, never
    stop one that is otherwise ready to run.
    """
    dev_str = str(getattr(args, "device_index", "") or "")
    device_tokens = [x.strip() for x in dev_str.split(",") if x.strip()] or ["0"]
    platform_name = str(getattr(args, "setup_platform", "") or getattr(args, "platform", "") or "")
    raw = str(getattr(args, "swarm_member_workers", "auto") or "auto").lower().strip()
    if raw in {"auto", "0", ""}:
        n_workers = len(device_tokens) if platform_name in _GPU_PLATFORM_NAMES else 1
    else:
        try:
            n_workers = max(1, int(raw))
        except (TypeError, ValueError):
            n_workers = 1
    return n_workers, device_tokens


def execute_members(args, rows, member_range, rd, run_one, *, n_workers: int = 1,
                    device_tokens: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run one round's outstanding members, concurrently when asked to.

    Members are independent by construction -- each grafts its own seed and
    writes its own directory -- so the only shared state that matters is ``args``.
    run_member sets ``args.seed`` to the member's velocity seed and restores it
    afterwards, which is safe serially and a race as soon as two members overlap:
    they would sample each other's seeds and the round would stop being
    reproducible. Each worker therefore gets its own shallow copy.

    A member that raises is recorded and the round continues. Members legitimately
    blow up (a graft that will not relax, a NaN), and losing the other 173 to one
    bad conformer would be a far worse outcome than an incomplete round the gate
    can judge on its merits.
    """
    rd = Path(rd)
    device_tokens = list(device_tokens or ["0"])
    n_workers = max(1, int(n_workers))

    status_counts: Dict[str, int] = {}
    n_run = n_skipped_resume = 0
    failed_in_range: List[int] = []
    missing_in_range: List[int] = []

    pending: List[Tuple[int, dict, Path]] = []
    for i in member_range:
        row = rows[i]
        member_id = int(row["member_id"])
        member_dir = rd / f"member_{member_id:04d}"
        if member_done(member_dir):
            n_skipped_resume += 1
            done = read_json_file(member_dir / "done.json", {}) or {}
            status = str(done.get("status", "ok"))
            status_counts[status] = status_counts.get(status, 0) + 1
            if status != "ok":
                failed_in_range.append(member_id)
            continue
        pending.append((member_id, row, member_dir))

    results: Dict[int, Any] = {}
    errors: Dict[int, BaseException] = {}

    def _one(slot: int, member_id: int, row: dict, member_dir: Path):
        worker_args = copy.copy(args)
        device = device_tokens[slot % len(device_tokens)]
        return run_one(worker_args, row, member_dir, device)

    if n_workers == 1 or len(pending) <= 1:
        for slot, (member_id, row, member_dir) in enumerate(pending):
            try:
                results[member_id] = _one(slot, member_id, row, member_dir)
            except BaseException as exc:      # noqa: BLE001 - recorded, not swallowed
                errors[member_id] = exc
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_one, slot, member_id, row, member_dir): member_id
                for slot, (member_id, row, member_dir) in enumerate(pending)
            }
            for future in concurrent.futures.as_completed(futures):
                member_id = futures[future]
                try:
                    results[member_id] = future.result()
                except BaseException as exc:  # noqa: BLE001 - recorded, not swallowed
                    errors[member_id] = exc

    for member_id, _row, member_dir in pending:
        if member_id in errors:
            failed_in_range.append(member_id)
            status_counts["error"] = status_counts.get("error", 0) + 1
            print(f"WARNING: swarm member {member_id} raised: {errors[member_id]}")
            continue
        n_run += 1
        done = results.get(member_id) or {}
        status = str(done.get("status", "ok"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "ok":
            failed_in_range.append(member_id)

    return {
        "n_run": n_run,
        "n_skipped_resume": n_skipped_resume,
        "failed_in_range": sorted(set(failed_in_range)),
        "missing_in_range": sorted(set(missing_in_range)),
        "status_counts": status_counts,
        "n_workers": n_workers,
    }
