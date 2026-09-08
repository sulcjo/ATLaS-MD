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
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from gareus.config import _write_yaml_or_json
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
    deltav_max_kj,
    design_lambda_ladder,
    fsf_floor_per_rung,
    n_resolvable_windows,
    window_sigma_cv,
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


def _load_members(rd: Path, rows: List[dict]) -> Tuple[Dict[int, dict], Dict[int, Dict[str, np.ndarray]], List[dict], List[int]]:
    """Read every planned member's ``done.json`` (+ ``trace.csv`` when ``status == "ok"``).

    Returns ``(done_summaries, ok_traces, frame_candidates, missing_members)``.
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
        frames_dir = member_dir / "frames"
        for i, frame_val in enumerate(tr["frame"]):
            frame_idx = int(round(frame_val))
            pdb_path = frames_dir / f"frame_{frame_idx:05d}.pdb"
            if pdb_path.exists():
                frame_candidates.append({
                    "member_id": member_id, "frame": frame_idx, "cv1": float(tr["cv1"][i]),
                    "pdb_path": str(pdb_path), "member_dir": str(member_dir),
                })
    return done_summaries, ok_traces, frame_candidates, missing_members


def _pool(traces: Dict[int, Dict[str, np.ndarray]], key: str, discard: int) -> np.ndarray:
    if not traces:
        return np.asarray([], dtype=float)
    parts = [tr[key][int(discard):] for tr in traces.values()]
    v = np.concatenate(parts) if parts else np.asarray([], dtype=float)
    return v[np.isfinite(v)]


def _write_sidecar(an: Path, seed_bank_dir: Path, windows_csv: Path) -> None:
    """The ``starting_structures``/``windows``/``gamd`` fragment a ``windows_2d_csv``
    production run must consume, so the seed-selection path travels with the CSV
    (global constraint 5c; dests verified against ``gareus/cli.py``)."""
    payload = {
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


def analyze_swarm_stage(out_dir, args) -> dict:
    out_dir = Path(out_dir)
    round_index = int(getattr(args, "swarm_round", 0) or 0)
    rd = round_dir(out_dir, round_index)
    an = swarm_root(out_dir) / "analysis"
    an.mkdir(parents=True, exist_ok=True)

    rows, plan_meta = _load_plan(rd)
    done_summaries, ok_traces, frame_candidates, missing_members = _load_members(rd, rows)

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
        try:
            centers = cv1_centers_from_samples(
                cv1_all, n_windows=n_win, seed_cv1=seed_pool_cv1, temperature_k=temperature_k,
                k_max_kcal=k_max, max_seed_gap_sigma=max_seed_gap_sigma, probe_out=probe_out,
            )
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

        gate = evaluate_gates(
            rows, set(ok_member_ids), ok_traces, discard, ladder, list(done_summaries.values()),
            min_done_fraction=float(getattr(args, "swarm_min_done_fraction", 0.9)),
            sigma_rel_tol=float(getattr(args, "swarm_stability_sigma_rel_tol", 0.10)),
            extrema_sigma_tol=float(getattr(args, "swarm_stability_extrema_sigma_tol", 1.0)),
            ladder_ess_floor=int(getattr(args, "swarm_ess_floor", 50)),
            max_graft_fallback_fraction=float(getattr(args, "swarm_max_graft_fallback_fraction", 0.10)),
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
            windows_csv = write_ladder_windows_csv(an / "windows_lambda_ladder.csv", centers, ks, ladder["lambdas"])
            _write_sidecar(an, seed_bank_dir, windows_csv)
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
