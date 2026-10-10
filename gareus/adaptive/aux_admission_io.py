"""Boundary hook: adaptive auxiliary-CV discovery and admission (spec 2026-10-09, Section 4). Never raises."""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings, resolve_aux_settings
from gareus.adaptive.aux_discovery.validation import check_validation_record

REPORT_SCHEMA = "aux_discovery_report_v1"
REPORT_NAME = "aux_discovery_report.json"
ADMISSION_SCHEMA = "atlas-aux-admission-v1"
ADMISSION_FILENAME = "aux_admission.json"
MODEL_FILENAME = "aux_model.json"
PARTITION_FILENAME = "aux_eval_partition.pkl"
VALIDATION_FILENAME = "aux_validation.json"
HISTORY_FILENAME = "aux_discovery_history.json"
HISTORY_SCHEMA = "aux_discovery_history_v1"
HISTORY_NOTE = ("One entry per discovery attempt. The same epochs' frames are reused as holdout across attempts, so "
                "per-attempt gates give NO campaign-wide false-discovery control.")
ACTION = "admit_aux"
_FROZEN_NAMES = (MODEL_FILENAME, PARTITION_FILENAME, PARTITION_FILENAME + ".sha256", ADMISSION_FILENAME)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def append_discovery_history(adaptive_dir: Path, report: dict, epoch: int) -> None:
    """Append this attempt (boundary epoch, outcome, holdout boundary) to adaptive_production's history file."""
    path = Path(adaptive_dir) / HISTORY_FILENAME
    try:
        rec = json.loads(path.read_text())
        entries = list(rec.get("attempts") or [])
    except Exception:
        entries = []
    n_hold = (report.get("discovery") or {}).get("n_holdout") if isinstance(report.get("discovery"), dict) else None
    entries.append({"boundary_epoch": int(epoch), "status": report.get("status"),
                    "holdout_epochs": [int(epoch)], "train_epochs": list(range(int(epoch))), "n_holdout_frames": report.get("n_holdout", n_hold),
                    "attempt_index": len(entries) + 1,
                    "z3_chosen": (report.get("z3_search") or {}).get("chosen") is not None,
                    "null_gate_passed": ((report.get("z3_search") or {}).get("null_gate") or {}).get("passed")})
    _write_json(path, {"schema": HISTORY_SCHEMA, "note": HISTORY_NOTE, "attempts": entries})


def _resolved_distance_interval(args) -> int:
    """The distance_interval production resolves (production.py, aux boundary-alignment check)."""
    d = int(getattr(args, "distance_output_interval", 0) or 0)
    if d <= 0:
        d = max(1, min(int(getattr(args, "report_interval", 0) or 0) or int(args.exchange_interval),
                       int(args.exchange_interval)))
    return max(1, d)


def phase_alignment_ok(args, *, calib_steps: int) -> Tuple[bool, str]:
    from gareus.auxiliary_cv.runtime_io import check_exchange_boundary_alignment
    try:
        check_exchange_boundary_alignment(exchange_interval=int(args.exchange_interval),
                                          distance_interval=_resolved_distance_interval(args),
                                          traj_interval=int(getattr(args, "traj_interval", 0) or 0),
                                          calib_steps=int(calib_steps))
        return True, "ok"
    except Exception as exc:          # the runtime check raises; the hook records it
        return False, str(exc)


def _build_frames(*, adaptive_dir, registry, epoch, settings, args):
    from gareus.adaptive.aux_discovery.frames import build_frame_table
    lam = {int(s.state_id): float(s.gamd_lambda) for s in registry.all_states()}
    stride = int(settings.frame_stride_steps) or int(args.exchange_interval)
    return build_frame_table(adaptive_dir, epochs=range(0, int(epoch) + 1), registry_lambda=lam,
                             stride_steps=stride, max_frames=settings.max_frames, seed=settings.partition_seed)


def _full_topology(out_dir: Path):
    from openmm import app
    return app.PDBFile(str(Path(out_dir) / "01_solvated_start.pdb")).topology


def _discover(*, ft, epoch, settings, out_dir, k3_max, eligible_parents=None):
    from gareus.adaptive.aux_discovery.pipeline import run_discovery
    train = ft.epoch < int(epoch)
    holdout = ft.epoch == int(epoch)
    return run_discovery(ft, train=train, holdout=holdout, settings=settings, full_topology=_full_topology(out_dir),
                         k3_max=k3_max, epoch=epoch, eligible_parents=eligible_parents)


def _ordinary_lambda0_active(registry) -> List[int]:
    from gareus.adaptive_production import is_auxiliary_state
    return sorted(int(s.state_id) for s in registry.active_states()
                  if not is_auxiliary_state(s) and abs(float(s.gamd_lambda or 0.0)) <= 1.0e-12)


def admission_limits(registry, actions: Sequence[Tuple], epoch, policy, args) -> dict:
    """How many workers this boundary may admit and from which parents (final fix wave C1).

    The epoch's other actions are applied to a COPY of the registry first (the order the applier will use:
    admit_aux is appended last), so a parent retired by them (retire, respring, split, respace) is never
    chosen and the replica budget sees their adds. Coupling gate None on the copy (the real gate records its
    decisions). Slots = min(aux_reserve_slots - live workers, max_replicas_budget - active states) (budget 0 =
    unlimited). Best effort: on any failure the live registry is used and the failure recorded; the applier
    and ``reconcile_admission_with_registry`` stay authoritative."""
    import contextlib
    import copy
    import io
    from gareus.adaptive_production import _apply_registry_actions, _resolve_secondary_k_max, is_auxiliary_state
    out = {"simulated": True}
    sim = registry
    try:
        try:
            k2max = _resolve_secondary_k_max(args)
        except Exception:
            k2max = None
        sim = copy.deepcopy(registry)
        with contextlib.redirect_stdout(io.StringIO()):
            refused = _apply_registry_actions(sim, list(actions), int(epoch), policy=policy, secondary_k_max=k2max,
                                              coupling_gate=None)
        out["n_other_actions"] = len(actions)
        out["n_other_refused"] = len(refused)
    except Exception as exc:
        sim = registry
        out.update(simulated=False, simulation_error=f"{type(exc).__name__}: {exc}")
    eligible = _ordinary_lambda0_active(sim)
    if not out["simulated"]:                      # at least never pick a parent the epoch retires outright
        retiring = {int(a[1]) for a in actions if str(a[0]) in ("retire", "respring", "split") and len(a) > 1}
        eligible = [sid for sid in eligible if sid not in retiring]
    n_workers = sum(1 for s in sim.active_states() if is_auxiliary_state(s))
    slots = int(getattr(policy, "aux_reserve_slots", 0) or 0) - n_workers
    budget = int(getattr(policy, "max_replicas_budget", 0) or 0)
    n_active = len(sim.active_states())
    free = (budget - n_active) if budget > 0 else None
    n_slots = max(0, min(slots, free) if free is not None else slots)
    out.update(eligible_parents=eligible, n_live_workers=int(n_workers), aux_reserve_slots=int(
        getattr(policy, "aux_reserve_slots", 0) or 0), max_replicas_budget=budget, n_active_after_actions=n_active,
        budget_free=free, n_slots=int(n_slots))
    return out


def _calib_steps_for_phase(args, adaptive_dir: Path) -> int:
    """The calibration steps a numbered-epoch phase will hand to the runtime alignment check.

    Trace (production.py): an adaptive phase imports the campaign's shared GaMD setup
    (``load_reusable_shared_gamd_setup``: ``--shared-gamd-setup-dir``, else
    ``<adaptive_dir>/global_shared_gamd_setup``) and takes ``calib_steps = payload["calibration_steps"]``
    (:4488). Without a recorded setup the phase calibrates itself with
    ``gamd_cmd_prep + gamd_cmd + gamd_equil_prep + gamd_equil`` steps (:6283); a run without GaMD uses 0."""
    from gareus.production import gamd_enabled
    if not gamd_enabled(args):        # --run-mode cmd/hmr-cmd: production sets calib_steps = 0 (production.py:7535)
        return 0
    setup =str(getattr(args, "shared_gamd_setup_dir", "") or "").strip()
    path = (Path(setup) if setup else Path(adaptive_dir) / "global_shared_gamd_setup") / "shared_gamd_setup_globals.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                return int(payload.get("calibration_steps", 0) or 0)
        except (OSError, ValueError):
            pass
    return int(sum(int(getattr(args, n, 0) or 0) for n in
                   ("gamd_cmd_prep_steps", "gamd_cmd_steps", "gamd_equil_prep_steps", "gamd_equil_steps")))


def _unfreeze(adaptive_dir: Path) -> None:
    from gareus.adaptive.aux_backfill import remove_backfills
    remove_backfills(adaptive_dir)
    for name in _FROZEN_NAMES:
        try:
            (Path(adaptive_dir) / name).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"WARNING: could not remove partially frozen {name}: {exc}")


def physical_system_check(args, out_dir) -> dict:
    """Production computes ``physical_system_sha256`` of the bare System before adding the aux force and refuses
    CMAP / virtual sites / non-canonical forces there (IntegrityError). Run the same check on the campaign's
    system before admitting, so a doomed admission never freezes. Status ``ok``, ``refused`` (admission blocked)
    or ``not_checked`` (the System could not be built here; production still checks)."""
    try:
        from openmm import app, unit
        import openmm
        from gareus.system_setup import create_system, make_forcefield_from_args
        pdb = app.PDBFile(str(Path(out_dir) / "01_solvated_start.pdb"))
        system = create_system(app, unit, make_forcefield_from_args(app, args), pdb.topology, args,
                               include_barostat=False)
    except Exception as exc:
        return {"status": "not_checked", "detail": f"{type(exc).__name__}: {exc}"}
    from gareus.auxiliary_cv.runtime_definition import physical_system_sha256
    try:
        return {"status": "ok", "physical_system_sha256": physical_system_sha256(openmm, system)}
    except Exception as exc:                     # IntegrityError: CMAP, virtual sites, unknown force
        return {"status": "refused", "detail": f"{type(exc).__name__}: {exc}"}


def run_epoch_aux_discovery(*, adaptive_dir, epoch_dir, epoch, registry, diagnostics, actions, policy, args,
                            out_dir, phase_dirs, gate=None, max_epochs: Optional[int] = None) -> List[tuple]:
    adaptive_dir = Path(adaptive_dir); epoch_dir = Path(epoch_dir)
    actions = list(actions)
    if not getattr(policy, "aux_discovery", False) or int(epoch) < 1:
        return actions
    last_epoch = max_epochs is not None and int(epoch) >= int(max_epochs) - 1
    if (adaptive_dir / ADMISSION_FILENAME).exists():
        _post_admission_spot_check(adaptive_dir, epoch_dir, epoch, phase_dirs)
        if last_epoch:
            return actions + _drop_missing_workers_at_last_epoch(adaptive_dir, epoch_dir, epoch, registry)
        return actions + _readmit_missing_workers(adaptive_dir, epoch_dir, epoch, registry)
    report = {"schema": REPORT_SCHEMA, "epoch": int(epoch), "started_unix": time.time()}
    freeze_started = False
    try:
        if last_epoch:
            # A worker admitted here would run only in the final phase, whose label carries no epoch: its burn-in
            # phase (epoch + 1) would never match and the pull transient would pool unfiltered (fix wave I2).
            report["status"] = "last_epoch"
            return actions
        settings, srec = resolve_aux_settings(adaptive_dir, AuxDiscoverySettings(),
                                              override=bool(getattr(args, "adaptive_production_aux_settings_override", False)))
        report["settings"] = srec.get("settings")
        val = check_validation_record(adaptive_dir / VALIDATION_FILENAME, timestep_fs=float(args.timestep_fs))
        report["validation"] = {"ok": val.ok, "reason": val.reason, "k3_max": val.k3_max}
        calib = _calib_steps_for_phase(args, adaptive_dir)
        ok, why = phase_alignment_ok(args, calib_steps=calib)
        report["alignment"] = {"ok": ok, "detail": why, "calib_steps": int(calib)}
        limits = admission_limits(registry, actions, epoch, policy, args)
        report["limits"] = limits
        if limits["n_slots"] <= 0:
            report["status"] = "no_slots"
            return actions
        physical = physical_system_check(args, out_dir)
        report["physical_system"] = physical
        if physical["status"] == "refused":
            report["status"] = "physical_system_unsupported"
            return actions
        # effective max = min(frozen settings.max_workers, free aux slots); the frozen settings record is untouched
        run_settings = dataclasses.replace(settings, max_workers=min(int(settings.max_workers), int(limits["n_slots"])))
        ft = _build_frames(adaptive_dir=adaptive_dir, registry=registry, epoch=epoch, settings=settings, args=args)
        res = _discover(ft=ft, epoch=epoch, settings=run_settings, out_dir=out_dir, k3_max=val.k3_max,
                        eligible_parents=limits["eligible_parents"])
        report.update(res.report); report["status"] = res.status
        if res.status != "ok":
            return actions
        if not val.ok:
            report["status"] = val.reason
            return actions
        if not ok:
            report["status"] = "alignment"
            return actions
        allowed = set(limits["eligible_parents"])
        chosen = [c for c in res.placement["chosen"] if int(c["state_id"]) in allowed][: int(limits["n_slots"])]
        dropped = [int(c["state_id"]) for c in res.placement["chosen"] if c not in chosen]
        if dropped:
            report["placement_filtered"] = dropped
        if not chosen:
            report["status"] = "no_eligible_worker"
            return actions
        new = []
        for rank, c in enumerate(chosen):
            new.append((ACTION, int(c["state_id"]),
                        {"aux_center": float(c["c3"]), "aux_k_kcal_mol": float(c["k3"]),
                         "aux_model_sha256": res.model.model_sha256, "burnin_steps": 0},
                        f"aux_discovery epoch {int(epoch)}",
                        {"aux": {"discovery_epoch": int(epoch), "placement_rank": rank,
                                 "burnin_phase_epoch": int(epoch) + 1,
                                 "forecast": {k: c[k] for k in ("O", "TV", "null", "net", "utility_q10")}}}))
        freeze_started = True
        report["backfill"] = _freeze(adaptive_dir, epoch, res, new, settings, srec, val)
        report["admitted"] = [[a[1], a[2]] for a in new]
        return actions + new
    except Exception as exc:
        if freeze_started:
            _unfreeze(adaptive_dir)          # a half-frozen admission would block every later boundary
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return actions
    finally:
        report["finished_unix"] = time.time()
        report["false_discovery_control"] = "per_attempt_only"   # never campaign-wide: see aux_discovery_history.json
        try:
            if report.get("status") not in (None, "last_epoch", "no_slots"):
                append_discovery_history(adaptive_dir, report, int(epoch))
        except Exception as exc:
            print(f"WARNING: could not append {HISTORY_FILENAME}: {exc}")
        try:
            _write_json(epoch_dir / REPORT_NAME, report)
        except Exception as exc:             # the hook never raises
            print(f"WARNING: could not write {epoch_dir / REPORT_NAME}: {exc}")


def _missing_admitted_workers(admission: dict, registry) -> List[dict]:
    """The admission record's workers with no registry worker of the same (parent, centre, k)."""
    from gareus.adaptive.aux_pooling import worker_table
    have = worker_table((s.state_id, s.metadata) for s in registry.all_states())
    out = []
    for w in admission.get("workers") or []:
        if not any(rec.get("spawn_parent_state_id") is not None
                   and int(rec["spawn_parent_state_id"]) == int(w["parent_state_id"])
                   and abs(float(rec["aux_center"]) - float(w["aux_center"])) < 1e-9
                   and abs(float(rec["aux_k_kcal_mol"]) - float(w["aux_k_kcal_mol"])) < 1e-9
                   for rec in have.values()):
            out.append(w)
    return out


def _readmit_missing_workers(adaptive_dir: Path, epoch_dir: Path, epoch, registry) -> List[tuple]:
    """A job killed after the freeze but before ``registry.save`` leaves aux_admission.json whose workers never
    reached the saved registry. Re-emit them from the record (never re-discover): without this every later
    boundary returns early and pooling refuses the campaign (``require_admitted_workers``). The epoch's
    discovery report keeps every original field (the only record of the search and placement); the event is
    appended under ``readmitted``. Never raises."""
    event = {"epoch": int(epoch), "started_unix": time.time()}
    try:
        admission = json.loads((adaptive_dir / ADMISSION_FILENAME).read_text())
        missing = _missing_admitted_workers(admission, registry)
        if not missing:
            return []
        adm_epoch = int(admission["epoch"])
        if int(epoch) > adm_epoch:     # phases since the admission ran without the model: z by backfill
            from gareus.adaptive.aux_backfill import merge_backfill_entries
            from gareus.auxiliary_cv.model import AuxModel
            event["backfill"] = _backfill(adaptive_dir, epoch, AuxModel.load(adaptive_dir / MODEL_FILENAME))
            # the rewritten files' hashes replace the record's (pooling verifies every backfill read against it)
            _write_json(adaptive_dir / ADMISSION_FILENAME,
                        {**admission, "backfill": merge_backfill_entries(admission.get("backfill"), event["backfill"])})
        new = [(ACTION, int(w["parent_state_id"]),
                {"aux_center": float(w["aux_center"]), "aux_k_kcal_mol": float(w["aux_k_kcal_mol"]),
                 "aux_model_sha256": str(admission["model_sha256"]), "burnin_steps": 0},
                f"aux_discovery epoch {adm_epoch} (re-admitted from {ADMISSION_FILENAME} at epoch {int(epoch)})",
                {"aux": {"discovery_epoch": adm_epoch, "placement_rank": int(w.get("placement_rank", 0)),
                         "burnin_phase_epoch": int(epoch) + 1, "readmitted_epoch": int(epoch)}})
               for w in missing]
        event.update(status="readmitted_from_record", admitted=[[a[1], a[2]] for a in new])
        return new
    except Exception as exc:
        event.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return []
    finally:
        if "status" in event:
            event["finished_unix"] = time.time()
            path = Path(epoch_dir) / REPORT_NAME
            try:
                report = json.loads(path.read_text()) if path.exists() else {"schema": REPORT_SCHEMA,
                                                                             "epoch": int(epoch)}
                report.setdefault("readmitted", []).append(event)
                _write_json(path, report)
            except Exception as exc:
                print(f"WARNING: could not record the re-admission in {path}: {exc}")


def _backfill(adaptive_dir: Path, epoch, model) -> List[dict]:
    from gareus.adaptive.aux_backfill import backfill_all
    return backfill_all(adaptive_dir, model, up_to_epoch=int(epoch))


def _freeze(adaptive_dir: Path, epoch, res, new_actions, settings, srec, val) -> List[dict]:
    res.model.write(adaptive_dir / MODEL_FILENAME)
    backfill = _backfill(adaptive_dir, epoch, res.model)     # BackfillIncomplete propagates: caller unfreezes
    part_sha = res.eval_partition.to_file(adaptive_dir / PARTITION_FILENAME)
    settings_sha = hashlib.sha256(json.dumps(srec.get("settings"), sort_keys=True).encode()).hexdigest()
    _write_json(adaptive_dir / ADMISSION_FILENAME, {
        "schema": ADMISSION_SCHEMA, "epoch": int(epoch), "model_sha256": res.model.model_sha256,
        "eval_partition_sha256": part_sha, "settings_sha256": settings_sha,
        "validation": {"k3_max": val.k3_max, "reason": val.reason}, "backfill": backfill,
        "workers": [{"parent_state_id": a[1], "aux_center": a[2]["aux_center"],
                     "aux_k_kcal_mol": a[2]["aux_k_kcal_mol"], "placement_rank": a[4]["aux"]["placement_rank"],
                     "burnin_phase_epoch": a[4]["aux"]["burnin_phase_epoch"]} for a in new_actions]})
    return backfill


def _append_report_event(epoch_dir: Path, epoch, key: str, event: dict) -> None:
    path = Path(epoch_dir) / REPORT_NAME
    try:
        report = json.loads(path.read_text()) if path.exists() else {"schema": REPORT_SCHEMA, "epoch": int(epoch)}
        report.setdefault(key, []).append(event)
        _write_json(path, report)
    except Exception as exc:
        print(f"WARNING: could not record {key} in {path}: {exc}")


def _registry_worker_records(registry) -> List[dict]:
    from gareus.adaptive_production import aux_params, is_auxiliary_state
    out = []
    for s in sorted(registry.all_states(), key=lambda st: int(st.state_id)):
        if not is_auxiliary_state(s):
            continue
        a = aux_params(s)
        out.append({"parent_state_id": int(a["spawn_parent_state_id"]), "aux_center": float(a["aux_center"]),
                    "aux_k_kcal_mol": float(a["aux_k_kcal_mol"]), "placement_rank": int(a.get("placement_rank", 0)),
                    "burnin_phase_epoch": a.get("burnin_phase_epoch"), "state_id": int(s.state_id)})
    return out


def reconcile_admission_with_registry(adaptive_dir: Path, registry) -> dict:
    """Make ``aux_admission.json``'s worker list exactly the registry's workers (final fix wave C1).

    Called after the applier and BEFORE ``registry.save``: an admission the applier refused in part keeps only
    the admitted workers (written atomically, the dropped ones under ``refused_workers``), one refused entirely
    is un-frozen (model, partition, admission, backfills removed) so a later boundary discovers again. Raises
    RuntimeError when the record cannot be made consistent: the job stops before the save, and the resumed
    epoch re-admits from the unchanged record (an inconsistent record would make the campaign unpoolable)."""
    adaptive_dir = Path(adaptive_dir)
    path = adaptive_dir / ADMISSION_FILENAME
    if not path.exists():
        return {"status": "no_admission"}
    try:
        admission = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{path}: unreadable admission record ({exc}); cannot reconcile it with the registry") from exc
    have = _registry_worker_records(registry)
    if not have:
        _unfreeze(adaptive_dir)
        if path.exists():
            raise RuntimeError(f"{path}: every admitted worker was refused but the record could not be removed")
        return {"status": "unfrozen", "dropped": list(admission.get("workers") or [])}
    from gareus.adaptive.aux_pooling import _worker_key_matches
    recorded = list(admission.get("workers") or [])
    kept, pool = [], list(have)
    for w in recorded:
        hit = next((h for h in pool if _worker_key_matches({"spawn_parent_state_id": h["parent_state_id"],
                                                            "aux_center": h["aux_center"],
                                                            "aux_k_kcal_mol": h["aux_k_kcal_mol"]}, w)), None)
        if hit is not None:
            pool.remove(hit)
            kept.append(w)
    dropped = [w for w in recorded if w not in kept]
    if not dropped and not pool:
        return {"status": "unchanged"}
    rewritten = dict(admission)
    rewritten["workers"] = kept + [{k: v for k, v in h.items() if k != "state_id"} for h in pool]
    if dropped:
        rewritten["refused_workers"] = list(admission.get("refused_workers") or []) + dropped
    try:
        _write_json(path, rewritten)
    except Exception as exc:
        raise RuntimeError(f"{path}: could not rewrite the admission record to the admitted workers ({exc})") from exc
    return {"status": "rewritten", "dropped": dropped, "added_from_registry": len(pool)}


def annotate_report_with_refusals(adaptive_dir: Path, epoch_dir: Path, actions: Sequence[Tuple],
                                  refused: Sequence[Mapping], registry=None) -> None:
    """Record the applier's refusals of ``admit_aux`` (``apply.refused``) and, with ``registry`` (the driver
    always passes it), reconcile ``aux_admission.json`` with the registry's actual workers
    (``reconcile_admission_with_registry``; may raise RuntimeError, see there). Without a registry: when every
    admission of the epoch was refused the frozen files are removed."""
    n_admit = sum(1 for a in actions if str(a[0]) == ACTION)
    if not n_admit:
        return
    path = Path(epoch_dir) / REPORT_NAME
    mine = []
    for r in refused:
        i = r.get("index")
        a = actions[int(i)] if i is not None and int(i) < len(actions) else None
        if a is not None and str(a[0]) == ACTION:
            mine.append({**dict(r), "proposal": [a[0], int(a[1])]})
    rec = None
    if registry is not None:
        rec = reconcile_admission_with_registry(adaptive_dir, registry)
    elif len(mine) == n_admit:
        _unfreeze(adaptive_dir)
    if not path.exists():
        return
    try:
        report = json.loads(path.read_text())
        report["apply"] = {"refused": mine, **({"reconcile": rec} if rec is not None else {})}
        if len(mine) == n_admit:
            report["status"] = "refused_by_applier"
        _write_json(path, report)
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not annotate {path} with the applier's refusals ({exc})")


def _drop_missing_workers_at_last_epoch(adaptive_dir: Path, epoch_dir: Path, epoch, registry) -> List[tuple]:
    """At the last numbered epoch a re-admitted worker would run only in the final phase, whose burn-in can never
    be filtered (I2): never re-admit there; reconcile the record with the registry instead. Never raises."""
    try:
        admission = json.loads((Path(adaptive_dir) / ADMISSION_FILENAME).read_text())
        if not _missing_admitted_workers(admission, registry):
            return []
        rec = reconcile_admission_with_registry(adaptive_dir, registry)
        _append_report_event(epoch_dir, epoch, "readmitted", {"epoch": int(epoch), "status": "last_epoch_not_readmitted",
                                                             "reconcile": rec})
    except Exception as exc:
        print(f"WARNING: could not reconcile {ADMISSION_FILENAME} at the last epoch: {exc}")
        _append_report_event(epoch_dir, epoch, "readmitted", {"epoch": int(epoch), "status": "error",
                                                             "error": f"{type(exc).__name__}: {exc}"})
    return []


def _post_admission_spot_check(adaptive_dir: Path, epoch_dir: Path, epoch, phase_dirs) -> None:
    """Once, on the first post-admission phase that ran with the admitted model: recorded aux_z_00 vs the
    frame-derived z the backfill uses, judged per worker in kT (``check_backfill_against_recorded``). Recorded
    in aux_admission.json[``spot_check``] and the epoch report; pooling requires ``ok`` true once a phase ran
    with the model (``aux_pooling.require_spot_check``). A check that cannot run there is recorded as a failure
    (``ok`` false, ``status`` error + the error), never left out, and is re-run at the next boundary (earlier
    errors kept under ``previous_errors``); a measured verdict is final. Phases run without the model (a
    re-admission's boundary) are not checked and nothing is recorded. Never raises."""
    adaptive_dir = Path(adaptive_dir)
    path = adaptive_dir / ADMISSION_FILENAME
    try:
        admission = json.loads(path.read_text())
        from gareus.adaptive.aux_pooling import phase_runtime_model_shas, spot_check_is_final
        previous = admission.get("spot_check")
        if (previous is not None and spot_check_is_final(previous)) or int(epoch) <= int(admission.get("epoch", epoch)):
            return
        workers = [(float(w["aux_center"]), float(w["aux_k_kcal_mol"])) for w in admission.get("workers") or []]
        if not workers:
            return
        sha = str(admission.get("model_sha256"))
        candidates = [Path(ph) for ph in (phase_dirs or [epoch_dir]) if sha in phase_runtime_model_shas(Path(ph))]
        if not candidates:
            return
    except Exception as exc:
        print(f"WARNING: aux z spot check failed before it could run: {type(exc).__name__}: {exc}")
        return
    try:
        event = {"epoch": int(epoch), **_run_spot_check(adaptive_dir, candidates, workers)}
    except Exception as exc:              # setup failed (model, settings): a failed check, not a missing one
        event = {"epoch": int(epoch), "ok": False, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if previous is not None:            # an errored check being re-run: keep what failed before
        event["previous_errors"] = list(previous.get("previous_errors") or []) + [
            {k: v for k, v in previous.items() if k != "previous_errors"}]
    _append_report_event(epoch_dir, epoch, "spot_check", event)
    try:
        _write_json(path, {**admission, "spot_check": event})
    except Exception as exc:              # pooling then refuses as due-but-missing
        print(f"WARNING: could not record the aux z spot check in {path}: {type(exc).__name__}: {exc}")
    if not event.get("ok"):
        what = (f"frame-derived z implies a worker energy error of {event['max_energy_err_kt']:.3g} kT "
                f"(> {event['tol_kt']} kT) on {event['phase']}" if event.get("max_energy_err_kt") is not None
                else f"{event.get('error') or event.get('errors')}")
        print(f"WARNING: aux z spot check failed ({what}): pooling of this campaign is refused")


def _run_spot_check(adaptive_dir: Path, candidates, workers) -> dict:
    """The verdict on the first candidate phase the check runs on; ``ok`` false + every error when none."""
    from gareus.adaptive.aux_backfill import check_backfill_against_recorded
    from gareus.auxiliary_cv.model import AuxModel
    try:
        temperature = float(json.loads((adaptive_dir / "aux_settings.json").read_text())["settings"]["temperature_k"])
    except Exception:
        temperature = float(AuxDiscoverySettings().temperature_k)
    model = AuxModel.load(adaptive_dir / MODEL_FILENAME)
    errors = []
    for ph in candidates:
        try:
            return check_backfill_against_recorded(ph, model, adaptive_dir=adaptive_dir, workers=workers,
                                                   temperature_k=temperature)
        except Exception as exc:
            errors.append(f"{ph}: {type(exc).__name__}: {exc}")
    return {"ok": False, "status": "error", "errors": errors}


def inject_aux_phase_args(phase_args, adaptive_dir: Path) -> None:
    # Deliberately ungated by the flag: a phase whose window CSV holds a worker must get the model, else the CSV loader refuses aux columns.
    adaptive_dir = Path(adaptive_dir)
    if not (adaptive_dir / ADMISSION_FILENAME).exists():
        return
    csv_path = getattr(phase_args, "windows_2d_csv", None)
    if not csv_path or not Path(csv_path).exists():
        return
    with Path(csv_path).open() as fh:
        reader = csv.DictReader(fh)
        has_aux_columns = "aux_k_kcal_mol" in (reader.fieldnames or [])
        roles = {row.get("state_role") for row in reader}
    # A worker-less segment of a post-admission epoch still carries the aux columns (the registry holds a worker):
    # it needs the model too, and records z for every sample (sham-only population, ruling M5) for pooling.
    if "auxiliary" not in roles and not has_aux_columns:
        return
    phase_args.aux_cv_model = str(adaptive_dir / MODEL_FILENAME)
    phase_args.aux_phase_kind = "production"
    phase_args.aux_equilibrium_eligible = True


@dataclass
class DiscoveryResultStub:  # test helper shared with the replay CLI's --dry-admit
    @classmethod
    def ok_with_workers(cls, workers):
        from gareus.adaptive.aux_discovery.pipeline import DiscoveryResult
        class _M:
            model_sha256 = "e" * 64

            def write(self, path):
                Path(path).write_text("{}")

        class _P:
            def to_file(self, path):
                Path(path).write_bytes(b"")
                return "f" * 64
        chosen = [{"state_id": p, "c3": c3, "k3": k3, "O": 0.3, "TV": 0.2, "null": 0.05, "net": 0.15,
                   "utility_q10": 0.04} for p, c3, k3 in workers]
        return DiscoveryResult("ok", {}, _M(), _P(), {"chosen": chosen})
