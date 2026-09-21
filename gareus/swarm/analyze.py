"""S1 analysis entry (spec Sec.2 S1 -> S2 handoff): pool member traces -> envelope ->
gates -> ladder design -> seed frames -> windows CSV -> sidecar run-args YAML -> report.

Round 0 fits and freezes the envelope + ladder (spec Sec.3.6/Sec.5, global constraint
"Frozen envelope"); round >= 1 only adds seeds and an out-of-envelope diagnostic against
the frozen round-0 envelope/ladder, never recalibrating. A failed gate withholds
``windows_lambda_ladder.csv``/``ladder_run_args.yaml`` (S2 must never see a ladder built
on an unstable/uncovered swarm) but every other artefact is still written so the extension
plan and the diagnostics are visible.

Ab initio (global constraint): pooling, gating and seed selection use only heavy-CV1,
Rg, end-to-end distance and the Pep-GaMD energies -- no reference structure anywhere.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gareus.config import _write_yaml_or_json
from gareus.correctness._io import digest, file_digest
from gareus.cv import (
    contact_normalization_denominator,
    contact_scheme,
    prepare_primary_cv_definition,
    secondary_cv_mode,
)
from gareus.cv_selection import contracts as C
from gareus.cv_selection.anchor import AnchorCandidate
from gareus.cv_selection.models import (
    contact_pair_list_digest,
    coupling_curvature_kcal,
    evaluate_component,
    from_candidate_set,
)
from gareus.cv_selection.select_pair import SelectionConfig, SwarmDataset, select_cv_pair
from gareus.cv_selection.coverage import build_region_inventory, region_centres
from gareus.io import write_json
from gareus.pep_gamd import PepGamdEnvelope
from gareus.swarm.driver import _load_plan, round_dir, swarm_root
from gareus.swarm.envelope import (
    discard_frames_from_trace,
    pool_member_envelopes,
    pooled_discard,
    write_envelope_setup_dir,
)
from gareus.swarm.gates import evaluate_gates, extension_plan, graft_gate
from gareus.swarm.ladder_design import (
    cv1_centers_from_samples,
    cv1_curvature_kcal,
    cv1_force_constants_from_curvature,
    LAYOUT_STATUS_PROPOSED,
    autotune_cv1_upper_bound,
    cv2_force_constants_per_gap,
    deltav_max_kj,
    design_exploration_layout,
    design_lambda_ladder,
    layout_plan_record,
    layout_rows,
    fsf_floor_per_rung,
    n_resolvable_windows,
    reweighted_cv2_centers,
    window_sigma_cv,
    write_ladder_windows_2d_csv,
    write_ladder_windows_csv,
)
from gareus.swarm.members import TRACE_COLUMNS, member_done
from gareus.swarm.seeds import export_seed_bank, select_window_seed_frames


def _read_trace(path: Path) -> Dict[str, np.ndarray]:
    cols: Dict[str, list] = {c: [] for c in TRACE_COLUMNS}
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            for c in TRACE_COLUMNS:
                cols[c].append(float(row[c]))
    return {c: np.asarray(v, dtype=float) for c, v in cols.items()}


def _load_members(rd: Path, rows: List[dict]) -> Tuple[Dict[int, dict], Dict[int, Dict[str, np.ndarray]], List[dict], List[int], Dict[int, Optional[tuple]]]:
    """Read every planned member's ``done.json`` (+ ``trace.csv`` when ``status == "ok"``).

    Returns ``(done_summaries, ok_traces, frame_candidates, missing_members, ok_features)``.
    ``ok_features[m]`` is ``(features_array, torsion_index_dict)`` for an ok member that
    recorded ``torsion_features.npy`` + ``torsion_index.json`` (Task 1), else ``None``;
    only automatic CV2 selection reads it, and only for ok members.
    ``ok_traces`` only holds members whose ``done.json`` status is ``"ok"`` (missing key
    defaults to ``"ok"``, matching ``driver.run_swarm_stage``'s own convention) -- a
    ``graft_failed``/``md_failed`` member contributes nothing to the envelope, gates,
    ladder or seed bank. ``frame_candidates`` are ok-member trace rows that have a
    matching ``frames/frame_XXXXX.pdb`` snapshot on disk (Task 5's sparse dump).
    """
    done_summaries: Dict[int, dict] = {}
    ok_traces: Dict[int, Dict[str, np.ndarray]] = {}
    frame_candidates: List[dict] = []
    missing_members: List[int] = []
    ok_features: Dict[int, Optional[tuple]] = {}
    for row in rows:
        member_id = int(row["member_id"])
        member_dir = rd / f"member_{member_id:04d}"
        if not member_done(member_dir):
            missing_members.append(member_id)
            continue
        done = json.loads((member_dir / "done.json").read_text())
        done_summaries[member_id] = done
        status = str(done.get("status", "ok"))
        trace_path = member_dir / "trace.csv"
        if status != "ok" or not trace_path.exists():
            continue
        tr = _read_trace(trace_path)
        ok_traces[member_id] = tr
        features_path = member_dir / "torsion_features.npy"
        index_path = member_dir / "torsion_index.json"
        ok_features[member_id] = (
            (np.load(features_path), json.loads(index_path.read_text()))
            if features_path.exists() and index_path.exists() else None
        )
        frames_dir = member_dir / "frames"
        for i, frame_val in enumerate(tr["frame"]):
            frame_idx = int(round(frame_val))
            pdb_path = frames_dir / f"frame_{frame_idx:05d}.pdb"
            if pdb_path.exists():
                frame_candidates.append({
                    "member_id": member_id, "frame": frame_idx, "cv1": float(tr["cv1"][i]),
                    "pdb_path": str(pdb_path), "member_dir": str(member_dir),
                })
    return done_summaries, ok_traces, frame_candidates, missing_members, ok_features


def _pool(traces: Dict[int, Dict[str, np.ndarray]], key: str, discard: int) -> np.ndarray:
    if not traces:
        return np.asarray([], dtype=float)
    parts = [tr[key][int(discard):] for tr in traces.values()]
    v = np.concatenate(parts) if parts else np.asarray([], dtype=float)
    return v[np.isfinite(v)]


def _write_sidecar(an: Path, seed_bank_dir: Path, windows_csv: Path, *,
                   cvs: Optional[dict] = None, pair_paths: Optional[dict] = None) -> None:
    """The ``starting_structures``/``windows``/``gamd`` fragment a ``windows_2d_csv``
    production run must consume, so the seed-selection path travels with the CSV
    (global constraint 5c; dests verified against ``gareus/cli.py``).

    With automatic CV2 selection the fragment also carries the resolved ``cvs`` and,
    for a selected pair, the three frozen artifact paths plus ``tica_switch_cv2: false``
    -- a frozen pair is never redefined mid-campaign."""
    payload: Dict[str, Any] = {}
    if cvs is not None:
        payload["cvs"] = dict(cvs)
    if pair_paths is not None:
        payload.update({
            "secondary_cv_model": str(Path(pair_paths["pair_model"]).resolve()),
            "secondary_cv_candidate_set": str(Path(pair_paths["candidate_set"]).resolve()),
            "secondary_cv_feature_schema": str(Path(pair_paths["feature_schema"]).resolve()),
            "tica_switch_cv2": False,
        })
    payload |= {
        "windows": {"window_mode": "manual", "windows_2d_csv": str(windows_csv.resolve())},
        "starting_structures": {
            "seed_conformers_dir": str(seed_bank_dir.resolve()),
            "seed_selection_mode": "active-cv",
            "seed_max_reuse_per_conformer": 0,
            "us_seed_preflight_max_score": 1.2,
            "us_pull_steps_per_window": 150000,
            "us_pull_timestep_fs": 3.0,
            "us_pull_k": 300.0,
            "us_pull_ramp_stages": 10,
        },
        "gamd": {
            "gamd_boost_type": "pep-gamd-lower-dual",
            "shared_gamd_setup_dir": str((an / "shared_gamd_setup").resolve()),
        },
    }
    _write_yaml_or_json(an / "ladder_run_args.yaml", payload)


def _envelope_summary(env: PepGamdEnvelope, setup_dir: Path) -> dict:
    return {
        "setup_dir": str(setup_dir.resolve()),
        "Total": {"vmax_kj_mol": env.vmax_total, "vmin_kj_mol": env.vmin_total,
                   "threshold_kj_mol": env.threshold_total, "k0max": env.k0max_total},
        "Dihedral": {"vmax_kj_mol": env.vmax_dih, "vmin_kj_mol": env.vmin_dih,
                     "threshold_kj_mol": env.threshold_dih, "k0max": env.k0max_dih},
    }


def _withhold_ladder_artifacts(an: Path) -> None:
    """A gate that used to pass (a prior analysis run) but now fails must not leave a stale
    ``windows_lambda_ladder.csv``/``ladder_run_args.yaml`` for S2 to pick up -- these are
    removed, not merely skipped, on every fail path."""
    # Both files go, and only on a fail path. This is deliberate deletion, not a
    # write-ordering concern: production preflights on the ladder's presence, so a
    # ladder left behind by an earlier passing analysis would let a run start on an
    # envelope the current gate rejects. (The happy path never deletes --
    # write_ladder_windows_csv stages and renames, so a passing re-analysis replaces
    # the ladder atomically and it is never momentarily absent.)
    (an / "windows_lambda_ladder.csv").unlink(missing_ok=True)
    (an / "ladder_run_args.yaml").unlink(missing_ok=True)


def _library_cv1_for_round0(rd0: Path, plan_meta: dict, warnings: List[str]) -> np.ndarray:
    """The real per-seed heavy-CV1 sample for the "centre must lie inside library
    coverage" cap, read from round 0's ``seed_descriptors.csv`` (written by
    ``driver._write_seed_descriptors``). Falls back to ``plan_meta["edges"]["cv1"]``
    (bin *boundaries*, not the raw sample -- ``quantile_edges`` pads the top edge to
    ``max(v) + 1e-12``, so a quantile of the edges alone resolves close to the
    library's max rather than its q99, understating the cap's strictness) only when
    the descriptors file is missing, and records why in ``warnings`` when it does.
    """
    path = rd0 / "seed_descriptors.csv"
    if path.exists():
        with path.open(newline="") as f:
            vals = [float(row["cv1"]) for row in csv.DictReader(f)]
        if vals:
            return np.asarray(vals, dtype=float)
    warnings.append(
        f"{path} not found (or empty); the library-CV1 coverage cap fell back to "
        "plan_meta['edges']['cv1'] bin edges, which resolves close to the library's "
        "max rather than its true q99 -- looser than the intended cap"
    )
    return np.asarray(plan_meta.get("edges", {}).get("cv1", [0.0, 1.0]), dtype=float)


def _measured_budget_ns(done_summaries: Dict[int, dict], ok_member_ids: List[int], output_interval_ps: float) -> float:
    """Sum of each ok member's actually-written production time (``n_frames`` steps of
    ``output_interval_ps`` each), not the planned ``seed_ns`` per member -- a member that
    completes fewer frames than planned (e.g. resumed mid-way) must not be counted as a
    full ``seed_ns`` of budget consumed."""
    total_ps = 0.0
    for m in ok_member_ids:
        n_frames = float(done_summaries.get(m, {}).get("n_frames", 0) or 0)
        total_ps += n_frames * output_interval_ps
    return total_ps / 1000.0


def _load_frozen_envelope_and_ladder(an: Path) -> Tuple[PepGamdEnvelope, dict]:
    setup_path = an / "shared_gamd_setup" / "shared_gamd_setup_globals.json"
    ladder_path = an / "ladder_design.json"
    if not setup_path.exists() or not ladder_path.exists():
        raise SystemExit(
            f"--swarm-round >= 1 needs {setup_path} and {ladder_path} (round 0's frozen envelope "
            "and ladder); run round 0's analysis first"
        )
    return PepGamdEnvelope.from_json(setup_path), json.loads(ladder_path.read_text())


def _extension_meta(out_dir: Path, round_index: int, plan_meta: dict) -> dict:
    """``plan_meta`` plus ``base_replicates_per_cell`` = round 0's R (Task 7's ``extension_plan``
    caps growth against the *campaign's* starting replicate count, not a later round's)."""
    meta = dict(plan_meta)
    if round_index == 0:
        meta.setdefault("base_replicates_per_cell", plan_meta.get("replicates_per_cell", 1))
    else:
        base_meta = json.loads((round_dir(out_dir, 0) / "plan_meta.json").read_text())
        meta.setdefault("base_replicates_per_cell", base_meta.get("replicates_per_cell", plan_meta.get("replicates_per_cell", 1)))
    return meta


# ---------------------------------------------------------------------------
# Automatic CV2 selection (plan Task 8): dataset assembly, selection, 2-D layout
# ---------------------------------------------------------------------------

def _library_versions() -> Dict[str, str]:
    return {"numpy": str(np.__version__), "python": sys.version.split()[0]}


def _seed_library_preset(args, warnings: List[str]) -> str:
    """The GENPEPT preset the swarm's seed library was generated with.

    The library's own ``generation_config.json`` is authoritative -- a hairpin-biased
    library must not pass the native-blind check because a config forgot to say so.
    ``args.diversity_bank_preset`` is used only when the library carries no record;
    with neither, ``"unknown"`` is recorded. A fold-biased preset is accepted and
    recorded in the pair model, never silently upgraded to native-blind."""
    configured = getattr(args, "diversity_bank_preset", None)
    recorded = None
    lib = getattr(args, "seed_conformers_dir", None)
    if lib:
        cfg = Path(lib) / "generation_config.json"
        if cfg.exists():
            try:
                recorded = json.loads(cfg.read_text()).get("diversity_bank_preset")
            except (OSError, ValueError) as exc:
                warnings.append(f"cv selection: could not read {cfg} ({exc!r}); falling back to the "
                                "configured diversity_bank_preset")
    if recorded:
        if configured and str(configured) != str(recorded):
            warnings.append(f"cv selection: config says diversity_bank_preset={configured!r} but the seed "
                            f"library records {recorded!r}; the library's record is authoritative")
        return str(recorded)
    if configured:
        return str(configured)
    warnings.append("cv selection: neither <seed_conformers_dir>/generation_config.json nor "
                    "diversity_bank_preset records a GENPEPT preset; recording 'unknown' -- the pair "
                    "model cannot claim a native-blind library")
    return "unknown"


def _swarm_contact_pairs(out_dir: Path, args, warnings: List[str]) -> Optional[list]:
    """The contact pairs the members measured CV1 with, rebuilt exactly as the driver did.

    ``swarm/system/topology.pdb`` is what ``driver`` fed to
    ``prepare_primary_cv_definition``; without it (or without OpenMM to read it) the
    anchor's pair list cannot be bound to the model and the report says so."""
    topology_pdb = swarm_root(out_dir) / "system" / "topology.pdb"
    if not topology_pdb.exists():
        warnings.append("cv selection: swarm/system/topology.pdb absent; the anchor's contact pair list "
                        "is NOT bound into the pair model (deployment can only check its parameters)")
        return None
    try:
        from openmm import app  # noqa: WPS433 -- runtime dependency of the swarm stage itself
        topology = app.PDBFile(str(topology_pdb)).topology
        return list(prepare_primary_cv_definition(topology, args).get("contact_pairs", []))
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.append(f"cv selection: could not rebuild contact pairs from topology.pdb ({exc!r}); "
                        "the anchor's pair list is NOT bound into the pair model")
        return None


def _anchor_definition(args, contact_pairs: Optional[list]) -> Dict[str, Any]:
    """The contact CV by contents. Parameters always; pair list and norm when available."""
    min_sep = int(getattr(args, "contact_min_sequence_separation", 4) or 4)
    selection = str(getattr(args, "contact_atom_selection", "heavy") or "heavy")
    definition: Dict[str, Any] = {
        "r0_angstrom": float(getattr(args, "contact_r0_a", 4.5)),
        "beta_per_angstrom": float(getattr(args, "contact_beta_a_inv", 6.0)),
        "min_sequence_separation": min_sep,
        "atom_selection": selection,
        "normalize": bool(getattr(args, "contact_normalize", True)),
        "pair_rule": f"{contact_scheme(args)}:{selection}:min-sep-{min_sep}",
    }
    if contact_pairs:
        definition["pair_list_sha256"] = contact_pair_list_digest(contact_pairs)
        definition["norm"] = float(contact_normalization_denominator(contact_pairs, args))
    return definition


def _feature_schema_from_index(index: dict, topology_sha256: str) -> C.FeatureSchema:
    """Canonical feature order: phi block then psi block, sin then cos per torsion."""
    rows = []
    for block, quads in (("phi", index["phi_torsions"]), ("psi", index["psi_torsions"])):
        for k, quad in enumerate(quads):
            for trig in ("sin", "cos"):
                rows.append({"index": len(rows), "name": f"{block}-{k}-{trig}", "torsion_name": f"{block}-{k}",
                             "residue_index": int(k), "atom_indices": [int(x) for x in quad],
                             "trig": trig, "dihedral_sign_convention": "negated"})
    return C.FeatureSchema.from_mapping({"schema": C.FEATURE_SCHEMA_VERSION,
                                         "topology_sha256": topology_sha256, "features": rows})


def _build_swarm_dataset(rows: List[dict], ok_traces: Dict[int, Dict[str, np.ndarray]],
                         ok_features: Dict[int, Optional[tuple]], discard: int, args,
                         contact_pairs: Optional[list], topology_sha256: str) -> Tuple[SwarmDataset, dict]:
    """Aligned per-frame rows past the discard: features, anchor, shape, energies, seed family.

    Alignment is by row index (Task 1 guarantees feature row i is trace row i); rows with
    a non-finite value anywhere are dropped jointly, never per column."""
    seed_by_member = {int(r["member_id"]): str(r.get("seed_id", r["member_id"])) for r in rows}
    index_ref: Optional[dict] = None
    first_member: Optional[int] = None
    parts: Dict[str, list] = {k: [] for k in ("features", "cv1", "rg", "e2e", "v_pep", "v_dih", "groups")}
    for m in sorted(ok_traces):
        record = ok_features.get(m)
        if record is None:
            raise RuntimeError(f"swarm member {m} completed (status ok) without torsion_features.npy; "
                               "cv2=auto needs canonical torsion features from every ok member")
        features, index = record
        if index_ref is None:
            index_ref, first_member = index, m
        elif index != index_ref:
            raise RuntimeError(f"member {m} torsion_index.json differs from member {first_member}; "
                               "one feature schema per swarm")
        tr = ok_traces[m]
        n_rows = len(tr["cv1"])
        if features.shape[0] != n_rows:
            raise RuntimeError(f"member {m}: {features.shape[0]} feature rows vs {n_rows} trace rows; "
                               "features are not aligned with the trace")
        sl = slice(int(discard), n_rows)
        parts["features"].append(np.asarray(features[sl], dtype=np.float64))
        parts["cv1"].append(tr["cv1"][sl]); parts["rg"].append(tr["rg_nm"][sl]); parts["e2e"].append(tr["e2e_nm"][sl])
        parts["v_pep"].append(tr["v_pep_kj"][sl]); parts["v_dih"].append(tr["v_dih_kj"][sl])
        parts["groups"].append(np.full(n_rows - int(discard), seed_by_member[m], dtype=object))
    if index_ref is None:
        raise RuntimeError("no ok member with torsion features; cv2=auto has nothing to fit on")
    features = np.vstack(parts["features"])
    cv1, rg, e2e = (np.concatenate(parts[k]) for k in ("cv1", "rg", "e2e"))
    v_pep, v_dih = (np.concatenate(parts[k]) for k in ("v_pep", "v_dih"))
    groups = np.concatenate(parts["groups"])
    finite = (np.isfinite(features).all(axis=1) & np.isfinite(cv1) & np.isfinite(rg) & np.isfinite(e2e)
              & np.isfinite(v_pep) & np.isfinite(v_dih))
    schema = _feature_schema_from_index(index_ref, topology_sha256)
    anchor = AnchorCandidate("nonlocal-contact-fraction", _anchor_definition(args, contact_pairs), cv1[finite])
    dataset = SwarmDataset(features[finite], schema, anchor, np.column_stack([rg[finite], e2e[finite]]),
                           groups[finite])
    aux = {"v_pep": v_pep[finite], "v_dih": v_dih[finite],
           "rows_sha256": digest(b"".join(np.ascontiguousarray(f).tobytes() for f in parts["features"])),
           "n_dropped_nonfinite": int((~finite).sum())}
    return dataset, aux


def _selection_config(args, k1_max_kcal: float, temperature_k: float) -> SelectionConfig:
    return SelectionConfig(
        residual_degree=int(getattr(args, "cv_selection_residual_degree", 1)),
        max_nonlinear_r2=float(getattr(args, "cv_selection_max_nonlinear_r2", 0.20)),
        max_coupling_fraction=float(getattr(args, "cv_selection_max_coupling_fraction", 0.25)),
        k1_kcal_reference=float(k1_max_kcal),
        k2_kcal_reference=float(getattr(args, "cv_selection_k2_reference_kcal", 1.0)),
        min_gain_nats=float(getattr(args, "cv_selection_min_gain_nats", 0.02)),
        min_windows_cv1=int(getattr(args, "cv_selection_min_windows_cv1", 4)),
        temperature_k=float(temperature_k),
    )


def _pool_member_ids(ok_traces: Dict[int, Dict[str, np.ndarray]], discard: int) -> np.ndarray:
    """Member id per pooled ``cv1`` row, aligned with ``_pool(ok_traces, "cv1", discard)``: same
    member order, same discard, same finiteness filter (``_pool`` drops non-finite rows)."""
    parts = []
    for m, tr in ok_traces.items():
        v = np.asarray(tr["cv1"][int(discard):], dtype=float)
        parts.append(np.full(int(np.isfinite(v).sum()), int(m), dtype=int))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=int)
def analyze_swarm_stage(out_dir, args) -> dict:
    out_dir = Path(out_dir)
    round_index = int(getattr(args, "swarm_round", 0) or 0)
    rd = round_dir(out_dir, round_index)
    an = swarm_root(out_dir) / "analysis"
    an.mkdir(parents=True, exist_ok=True)

    rows, plan_meta = _load_plan(rd)
    done_summaries, ok_traces, frame_candidates, missing_members, ok_features = _load_members(rd, rows)

    graft_failed_members = sorted(m for m, d in done_summaries.items() if str(d.get("status", "ok")) == "graft_failed")
    md_failed_members = sorted(m for m, d in done_summaries.items() if str(d.get("status", "ok")) == "md_failed")
    failed_members = sorted(graft_failed_members + md_failed_members)
    ok_member_ids = sorted(ok_traces.keys())

    output_interval_ps = float(getattr(args, "swarm_output_interval_ps", 2.0))
    block = int(getattr(args, "swarm_discard_block_frames", 25))
    min_discard_ps = float(getattr(args, "swarm_min_discard_ps", 0.0))
    per_member_discard = {m: discard_frames_from_trace(tr["v_pep_kj"], block=block) for m, tr in ok_traces.items()}
    floor_frames = int(math.ceil(min_discard_ps / output_interval_ps)) if min_discard_ps > 0 else 0
    discard = pooled_discard(list(per_member_discard.values()), floor_frames=floor_frames)

    write_json(an / "envelope_discard.json", {
        "per_member_discard_frames": {str(m): v for m, v in per_member_discard.items()},
        "pooled_discard_frames": discard,
        "pooled_discard_ps": discard * output_interval_ps,
        "block_frames": block,
        "min_discard_ps": min_discard_ps,
        "output_interval_ps": output_interval_ps,
        "round_index": round_index,
    })

    ns_values = [float(d.get("ns_per_day", 0.0)) for m, d in done_summaries.items() if m in ok_member_ids]
    ns_per_day_median = float(np.median(ns_values)) if ns_values else 0.0
    seed_ns = float(plan_meta.get("seed_ns", 0.0))
    budget_ns_planned = seed_ns * len(ok_member_ids)
    budget_ns_done = _measured_budget_ns(done_summaries, ok_member_ids, output_interval_ps)

    seeds_per_window = int(getattr(args, "swarm_seeds_per_window", 3))
    temperature_k = float(args.temperature_k)
    warnings: List[str] = []

    report: Dict[str, Any] = {
        "round": round_index, "n_members": len(rows), "n_ok_members": len(ok_member_ids),
        "missing_members": missing_members, "graft_failed_members": graft_failed_members,
        "failed_members": failed_members, "ns_per_day_median": ns_per_day_median,
        "budget_ns_done": budget_ns_done, "budget_ns_planned": budget_ns_planned,
        "discard_frames": discard, "warnings": warnings,
    }

    if round_index == 0:
        envelopes = pool_member_envelopes(ok_traces, discard)
        sigma0_kj = {"Total": 4.184 * float(args.sigma0p_kcal_mol), "Dihedral": 4.184 * float(args.sigma0d_kcal_mol)}
        setup_meta = {
            "discard_frames": discard, "discard_ps": discard * output_interval_ps,
            "output_interval_ps": output_interval_ps, "round_index": round_index,
            "n_members_pooled": len(ok_traces), "seed_ns": seed_ns,
        }
        setup_path = write_envelope_setup_dir(an / "shared_gamd_setup", envelopes, sigma0_kj=sigma0_kj,
                                               temperature_k=temperature_k, meta=setup_meta)
        env = PepGamdEnvelope.from_json(setup_path)

        v_pep = _pool(ok_traces, "v_pep_kj", discard)
        v_dih = _pool(ok_traces, "v_dih_kj", discard)
        cv1_all = _pool(ok_traces, "cv1", discard)
        coverage_range = float(cv1_all.max() - cv1_all.min())
        # Reported diagnostic only (2026-09-08 fix) -- round-0 windows seed from the swarm's
        # OWN frames (seed_cv1=cv1_all below), never from the GENPEPT library, so this no
        # longer gates the round; its q99 still travels with the report for provenance.
        library_cv1 = _library_cv1_for_round0(rd, plan_meta, warnings)
        library_q99 = float(np.quantile(library_cv1, 0.99)) if library_cv1.size else None

        k_max = float(getattr(args, "contact_adaptive_max_k_kcal", 1200.0))
        k_min = float(getattr(args, "contact_adaptive_min_k_kcal", 5.0))
        overlap_sigma = float(getattr(args, "swarm_overlap_sigma", 1.5))
        max_seed_gap_sigma = float(getattr(args, "swarm_max_seed_gap_sigma", 0.5))
        n_win = min(int(args.swarm_n_windows), n_resolvable_windows(coverage_range, temperature_k, k_max_kcal=k_max, overlap_sigma=overlap_sigma))
        probe_out: Dict[str, Any] = {}
        # The probe pool is the EXPORTED frames (those with a written PDB) past the discard --
        # exactly what select_window_seed_frames draws from below. The trace is ~10x denser
        # (every trace row vs one PDB per swarm_seed_frame_interval_ps), so probing the trace
        # would "support" a centre with a frame no window could ever start from.
        seed_pool_cv1 = np.asarray(
            [float(fr["cv1"]) for fr in frame_candidates if int(fr["frame"]) >= discard], dtype=float
        )
        # Region inventory (spec F05; review I02): the supported CV1 intervals of the discovery
        # sample, each with a representative that has an exported seed structure. Centres are
        # placed inside supported intervals only; a gap between regions is bridged by the
        # unrestrained anchor stack, never by lowering the upper bound until a seedless centre
        # disappears. The old tuner is kept as a diagnostic that can no longer place centres.
        member_of_row = _pool_member_ids(ok_traces, discard)
        n_failed_members = sum(1 for d in done_summaries.values() if str(d.get("status", "ok")) != "ok")
        inventory = build_region_inventory(cv1_all, groups=member_of_row, seed_values=seed_pool_cv1,
                                           temperature_k=temperature_k, k_max_kcal=k_max,
                                           n_failed_seed_preparations=n_failed_members)
        region_of_centre: List[str] = []
        region_rep_indices: List[int] = []
        try:
            design_c = region_centres(inventory, n_win, temperature_k=temperature_k, k_max_kcal=k_max,
                                      overlap_sigma=overlap_sigma)
            if not design_c["centres"]:
                raise ValueError("no supported CV1 region found in the swarm frames")
            centers = np.asarray(design_c["centres"], dtype=float)
            region_of_centre = list(design_c["region_of_centre"])
            representatives = {r.region_id: r.representative for r in inventory.cv1_regions}
            region_rep_indices = [i for i, c in enumerate(centers)
                                  if abs(float(c) - representatives[region_of_centre[i]]) < 1e-12]
            n_win = int(centers.size)
            tol = float(max_seed_gap_sigma) * float(window_sigma_cv(k_max, temperature_k))
            nearest_gap = [float(np.min(np.abs(seed_pool_cv1 - c))) if seed_pool_cv1.size else float("nan") for c in centers]
            probe_out.update({"autotuned": False, "hi": float(centers.max()), "hi_initial": float(cv1_all.max()),
                              "lo": float(centers.min()), "tol": tol, "nearest_seed_gap": nearest_gap,
                              "n_probes": int(centers.size), "design": "region_inventory_v1",
                              "n_regions": len(inventory.cv1_regions), "n_requested_windows": int(design_c["n_requested"])})
            try:
                legacy = autotune_cv1_upper_bound(cv1_all, seed_pool_cv1, n_windows=int(design_c["n_requested"]),
                                                  temperature_k=temperature_k, k_max_kcal=k_max,
                                                  max_seed_gap_sigma=max_seed_gap_sigma)
                probe_out["legacy_autotune"] = {"hi": float(legacy["hi"]), "hi_initial": float(legacy["hi_initial"]),
                                                "autotuned": bool(legacy["autotuned"])}
                if bool(legacy["autotuned"]) and float(legacy["hi"]) < float(centers.max()) - 1e-12:
                    warnings.append(f"cv1 design: the legacy endpoint tuner would have lowered the CV1 upper bound to "
                                    f"{float(legacy['hi']):.4f} (retained support reaches {float(centers.max()):.4f}); "
                                    "the region inventory keeps the upper region (review I02)")
            except Exception as exc:  # diagnostic only
                probe_out["legacy_autotune"] = {"error": str(exc)}
        except ValueError as exc:
            # The autotuned probe found no CV1 upper bound anywhere with seed support --
            # this is a hard ladder-design failure, not a soft gate the swarm could pass by
            # extending replicates. No ladder/gate/seed bank can be built without centres;
            # the failure is still recorded (not raised) so the report and the already-
            # written envelope/discard artefacts survive for the controller to inspect.
            report.update({
                "status": "fail", "reasons": [f"ladder design: {exc}"],
                "envelope": _envelope_summary(env, an / "shared_gamd_setup"),
                "ladder_design": {"library_q99": library_q99, "n_seed_pool": int(seed_pool_cv1.size)},
            })
            _withhold_ladder_artifacts(an)
            write_json(an / "swarm_report.json", report)
            return report
        curvature = cv1_curvature_kcal(cv1_all, centers, temperature_k)
        k_warnings: List[dict] = []
        ks = cv1_force_constants_from_curvature(centers, curvature, temperature_k, overlap_sigma=overlap_sigma,
                                                 k_min_kcal=k_min, k_max_kcal=k_max, coverage_range=coverage_range,
                                                 warnings_out=k_warnings)

        ladder = design_lambda_ladder(
            deltav_max_kj(v_pep, v_dih, env), temperature_k,
            target_beta_sigma=float(getattr(args, "swarm_target_beta_sigma", 1.0)),
            min_rungs=int(getattr(args, "swarm_min_rungs", 3)), max_rungs=int(getattr(args, "swarm_max_rungs", 12)),
            ess_floor=int(getattr(args, "swarm_ess_floor", 50)),
        )
        fsf = fsf_floor_per_rung(ladder["lambdas"], env, warn_threshold=float(getattr(args, "swarm_fsf_floor_warn", 0.5)))
        max_nearest_seed_gap = float(np.max(probe_out["nearest_seed_gap"])) if len(probe_out.get("nearest_seed_gap", [])) else None
        cv1_upper_bound_probe = {
            "autotuned": bool(probe_out.get("autotuned")),
            "hi": float(probe_out["hi"]), "hi_initial": float(probe_out["hi_initial"]),
            "tol": float(probe_out["tol"]), "max_nearest_seed_gap": max_nearest_seed_gap,
            "n_probes": int(probe_out.get("n_probes", 0)), "n_seed_pool": int(seed_pool_cv1.size),
        }
        ladder_design = dict(ladder)
        ladder_design.update({
            "centers": [float(c) for c in centers], "curvature_kcal": [float(c) for c in curvature],
            "k_kcal": [float(k) for k in ks], "window_sigma_cv": [window_sigma_cv(k, temperature_k) for k in ks],
            "k_warnings": k_warnings, "fsf_floor_per_rung": fsf, "n_windows": int(n_win),
            "n_states": int(n_win) * len(ladder["lambdas"]), "coverage_range": coverage_range,
            "library_q99": library_q99, "cv1_upper_bound_probe": cv1_upper_bound_probe,
        })
        write_json(an / "ladder_design.json", ladder_design)
        report["region_inventory"] = inventory.as_record()
        report["region_inventory"]["region_of_centre"] = list(region_of_centre)
        write_json(an / "region_inventory.json", report["region_inventory"])

        # --- automatic CV2 selection against the configured contact anchor ------------
        selection: Optional[Dict[str, Any]] = None
        pair_layout: Optional[tuple] = None
        pair_paths: Optional[dict] = None
        if secondary_cv_mode(args) == "auto":
            contact_pairs = _swarm_contact_pairs(out_dir, args, warnings)
            system_xml = swarm_root(out_dir) / "system" / "base_system.xml"
            topology_pdb = swarm_root(out_dir) / "system" / "topology.pdb"
            # Real bindings or none: a placeholder digest never makes an artifact deployable
            # (spec F02). Missing swarm system/topology files mean a discovery-only pair.
            physical_sha = file_digest(system_xml) if system_xml.exists() else None
            topology_sha = file_digest(topology_pdb) if topology_pdb.exists() else None
            deployment = {
                "topology_sha256": topology_sha, "physical_system_sha256": physical_sha,
                "contact_pair_list_sha256": contact_pair_list_digest(contact_pairs) if contact_pairs else None,
            }
            deployment["deployable"] = all(deployment[k] for k in ("topology_sha256", "physical_system_sha256",
                                                                   "contact_pair_list_sha256"))
            if not deployment["deployable"]:
                warnings.append("cv selection: the pair model will NOT be deployable (missing swarm system, "
                                "topology or contact pair list); production refuses a discovery-only artifact")
            dataset, aux = _build_swarm_dataset(rows, ok_traces, ok_features, discard, args, contact_pairs,
                                                topology_sha or digest(b"swarm-topology-unavailable"))
            sel = select_cv_pair(
                dataset, _selection_config(args, k_max, temperature_k),
                physical_system_sha256=physical_sha or digest(b"swarm-system-unavailable"),
                training_rows_sha256=aux["rows_sha256"],
                library_versions=_library_versions(),
                genpept_preset=_seed_library_preset(args, warnings),
                deployment=deployment,
            )
            if not sel.report.get("native_blind_library", False):
                warnings.append(f"cv selection: seed library preset {sel.report.get('genpept_preset')!r} is not "
                                "native-blind; the selected CV2 rests on a fold-aware library and this run "
                                "cannot be quoted as ab initio")
            selection = {
                "status": sel.status, "anchor": sel.anchor.kind, "anchor_reasons": list(sel.anchor.reasons),
                "anchor_n_resolvable": int(sel.anchor.n_resolvable),
                "selection_reason": sel.report.get("selection_reason"),
                "n_frames": int(dataset.features.shape[0]), "n_dropped_nonfinite": aux["n_dropped_nonfinite"],
                "physical_system_bound": bool(system_xml.exists()),
                "anchor_pair_list_bound": contact_pairs is not None,
                "deployable": bool(deployment["deployable"]),
            }
            (an / "cv_feature_schema.json").write_bytes(dataset.feature_schema.to_json_bytes())
            if sel.candidate_set is not None:
                (an / "cv_candidate_set.json").write_bytes(sel.candidate_set.to_json_bytes())
            if sel.pair_model is not None:
                (an / "cv_pair_model.json").write_bytes(sel.pair_model.to_json_bytes())
                selection["pair_model_sha256"] = sel.pair_model.sha256
                selection["selected_component_index"] = int(sel.pair_model.selected_component_index)
                selection["certificate"] = dict(sel.pair_model.certificate)
            if sel.status == "pair":
                fit = from_candidate_set(sel.candidate_set)
                j = int(sel.pair_model.selected_component_index)
                z2_all = evaluate_component(fit, j, dataset.features, dataset.anchor.values)
                dv_rows = deltav_max_kj(aux["v_pep"], aux["v_dih"], env)
                n2 = max(2, int(getattr(args, "swarm_n_windows_cv2", 4)))
                cv2_design = reweighted_cv2_centers(z2_all, dv_rows, temperature_k, ladder["lambdas"], n2)
                k2_min = float(getattr(args, "cv2_k_min", 0.0) or 0.0) or 1e-3
                ks2 = cv2_force_constants_per_gap(cv2_design["centers"], temperature_k, overlap_sigma=overlap_sigma,
                                                  k_min_kcal=k2_min, k_max_kcal=float(getattr(args, "cv2_k_max", 1000.0)))
                layout = design_exploration_layout(int(n_win), n2, n_rungs=len(ladder["lambdas"]),
                                                   max_replicas=int(getattr(args, "max_replicas", 0) or 0),
                                                   region_centre_indices=region_rep_indices)
                if layout["status"] != LAYOUT_STATUS_PROPOSED:
                    warnings.append(f"layout: {layout['status']}: {layout.get('reason')}")
                # The CV2 umbrella also restrains the anchor through the chain-rule term;
                # the CV1 windows are narrower than their own k says by this much.
                coupling = coupling_curvature_kcal(fit, j, max(ks2))
                shrink = [1.0 - math.sqrt(float(k) / (float(k) + coupling)) for k in ks]
                if max(shrink) > 0.10:
                    warnings.append(f"cv selection: the CV2 umbrella narrows CV1 windows by up to "
                                    f"{100 * max(shrink):.0f}% (coupling curvature {coupling:.1f} kcal/mol/CV^2 "
                                    f"at k2={max(ks2):.2f}); the CV1 overlap design assumed k1 alone")
                selection.update({
                    "layout": {k: v for k, v in layout.items() if k not in ("cells", "state_roles")},
                    "cv2_centers": [float(c) for c in cv2_design["centers"]],
                    "cv2_k_kcal": [float(k) for k in ks2],
                    "cv2_per_rung_quantiles": {str(l): list(q) for l, q in cv2_design["per_rung_quantiles"].items()},
                    "cv2_per_rung_ess": {str(l): e for l, e in cv2_design["per_rung_ess"].items()},
                    "coupling_curvature_kcal_at_max_k2": float(coupling),
                    "cv1_width_shrink_max": float(max(shrink)),
                })
                rows_2d = layout_rows(layout, centers, ks, cv2_design["centers"], ks2) if layout["cells"] else []
                plan_record = layout_plan_record(layout, rows_2d, ladder["lambdas"],
                                                 region_inventory=report["region_inventory"],
                                                 region_of_centre=region_of_centre)
                pair_layout = (layout, cv2_design["centers"], ks2, rows_2d, plan_record)
                pair_paths = {"pair_model": an / "cv_pair_model.json",
                              "candidate_set": an / "cv_candidate_set.json",
                              "feature_schema": an / "cv_feature_schema.json"}
            # Written after the layout block so the file carries the 2-D design too.
            write_json(an / "cv_selection_report.json", {**sel.report, **selection})
            report["cv_selection"] = selection

        gate = evaluate_gates(
            rows, set(ok_member_ids), ok_traces, discard, ladder, list(done_summaries.values()),
            min_done_fraction=float(getattr(args, "swarm_min_done_fraction", 0.9)),
            sigma_rel_tol=float(getattr(args, "swarm_stability_sigma_rel_tol", 0.10)),
            extrema_sigma_tol=float(getattr(args, "swarm_stability_extrema_sigma_tol", 1.0)),
            ladder_ess_floor=int(getattr(args, "swarm_ess_floor", 50)),
            max_graft_fallback_fraction=float(getattr(args, "swarm_max_graft_fallback_fraction", 0.10)),
            selection=selection,
            pair_fallback=str(getattr(args, "cv_selection_fallback", "cv1_only") or "cv1_only"),
            layout_plan=(pair_layout[4] if pair_layout is not None else None),
        )
        write_json(an / "swarm_gate.json", gate)
        # Also fold gate warnings (e.g. ladder_ess_gate's advisory ESS/extrapolation
        # findings) into the flat report["warnings"] list: cli.py's --swarm-stage
        # analyze console printout surfaces result["warnings"] directly, never
        # result["gate"]["warnings"], so a warning left only nested under "gate"
        # would never reach the operator's terminal.
        warnings.extend(gate.get("warnings", []))

        selection = select_window_seed_frames(frame_candidates, centers, per_window=seeds_per_window, discard_frames=discard)
        seed_bank_dir = an / "seed_bank"
        export_seed_bank(seed_bank_dir, selection, centers, round_index=round_index)

        report.update({
            "status": gate["status"], "gate": gate, "n_windows": int(n_win), "n_states": ladder_design["n_states"],
            "envelope": _envelope_summary(env, an / "shared_gamd_setup"),
            "fsf_warn": fsf["warn"], "n_k_warnings": len(k_warnings),
            "ladder_design": {"library_q99": library_q99, **cv1_upper_bound_probe},
        })
        if gate["status"] == "pass":
            if pair_layout is not None:
                layout, centers2, ks2, rows_2d, plan_record = pair_layout
                windows_csv = write_ladder_windows_2d_csv(an / "windows_lambda_ladder.csv", rows_2d, ladder["lambdas"])
                # Companion artifact: state roles, region coverage, mandatory ids -- separately
                # digested, never inside the physics rows (spec F05).
                write_json(an / "layout_plan.json", plan_record)
                report["n_states"] = int(plan_record["n_states"])
                report["layout_plan"] = {k: plan_record[k] for k in ("status", "kind", "spatial_states", "n_states",
                                                                     "mandatory_state_ids", "unresolved")}
                _write_sidecar(an, seed_bank_dir, windows_csv,
                               cvs={"cv1": "contacts", "cv2": "residual-torsion-pc"}, pair_paths=pair_paths)
            else:
                windows_csv = write_ladder_windows_csv(an / "windows_lambda_ladder.csv", centers, ks, ladder["lambdas"])
                _write_sidecar(an, seed_bank_dir, windows_csv,
                               cvs={"cv1": "contacts", "cv2": "none"} if selection is not None else None)
        else:
            _withhold_ladder_artifacts(an)
            report["withheld_ladder_artifacts"] = True
            report["extension"] = extension_plan(_extension_meta(out_dir, round_index, plan_meta), gate)
    else:
        env, frozen_ladder = _load_frozen_envelope_and_ladder(an)
        centers = frozen_ladder["centers"]

        v_pep = _pool(ok_traces, "v_pep_kj", discard)
        v_dih = _pool(ok_traces, "v_dih_kj", discard)
        out_of_envelope_fraction = {
            "Total": float(np.mean((v_pep > env.vmax_total) | (v_pep < env.vmin_total))) if v_pep.size else None,
            "Dihedral": float(np.mean((v_dih > env.vmax_dih) | (v_dih < env.vmin_dih))) if v_dih.size else None,
        }

        selection = select_window_seed_frames(frame_candidates, centers, per_window=seeds_per_window, discard_frames=discard)
        seed_bank_dir = an / f"seed_bank_round_{round_index:03d}"
        export_seed_bank(seed_bank_dir, selection, centers, round_index=round_index)

        graft = graft_gate(list(done_summaries.values()), max_fallback_fraction=float(getattr(args, "swarm_max_graft_fallback_fraction", 0.10)))
        gate = {"status": "pass" if graft["ok"] else "fail", "gates": {"graft": graft}, "reasons": graft["reasons"]}
        write_json(an / "swarm_gate.json", gate)

        report.update({
            "status": gate["status"], "gate": gate, "out_of_envelope_fraction": out_of_envelope_fraction,
            "envelope": _envelope_summary(env, an / "shared_gamd_setup"),
            "fsf_warn": frozen_ladder.get("fsf_floor_per_rung", {}).get("warn"),
            "n_k_warnings": len(frozen_ladder.get("k_warnings", [])),
        })
        if gate["status"] == "fail":
            report["extension"] = extension_plan(_extension_meta(out_dir, round_index, plan_meta), gate)

    write_json(an / "swarm_report.json", report)
    return report
