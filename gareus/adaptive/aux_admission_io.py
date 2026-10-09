"""Boundary hook: adaptive auxiliary-CV discovery and admission (spec 2026-10-09, Section 4). Never raises."""
from __future__ import annotations

import csv
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
ACTION = "admit_aux"
_FROZEN_NAMES = (MODEL_FILENAME, PARTITION_FILENAME, PARTITION_FILENAME + ".sha256", ADMISSION_FILENAME)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


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


def _discover(*, ft, epoch, settings, out_dir, k3_max):
    from gareus.adaptive.aux_discovery.pipeline import run_discovery
    train = ft.epoch < int(epoch)
    holdout = ft.epoch == int(epoch)
    return run_discovery(ft, train=train, holdout=holdout, settings=settings, full_topology=_full_topology(out_dir),
                         k3_max=k3_max, epoch=epoch)


def _calib_steps_for_phase(args, adaptive_dir: Path) -> int:
    """The calibration steps a numbered-epoch phase will hand to the runtime alignment check.

    Trace (production.py): an adaptive phase imports the campaign's shared GaMD setup
    (``load_reusable_shared_gamd_setup``: ``--shared-gamd-setup-dir``, else
    ``<adaptive_dir>/global_shared_gamd_setup``) and takes ``calib_steps = payload["calibration_steps"]``
    (:4488). Without a recorded setup the phase calibrates itself with
    ``gamd_cmd_prep + gamd_cmd + gamd_equil_prep + gamd_equil`` steps (:6283); a run without GaMD uses 0."""
    setup = str(getattr(args, "shared_gamd_setup_dir", "") or "").strip()
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
    for name in _FROZEN_NAMES:
        try:
            (Path(adaptive_dir) / name).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"WARNING: could not remove partially frozen {name}: {exc}")


def run_epoch_aux_discovery(*, adaptive_dir, epoch_dir, epoch, registry, diagnostics, actions, policy, args,
                            out_dir, phase_dirs, gate=None) -> List[tuple]:
    adaptive_dir = Path(adaptive_dir); epoch_dir = Path(epoch_dir)
    actions = list(actions)
    if not getattr(policy, "aux_discovery", False) or int(epoch) < 1:
        return actions
    if (adaptive_dir / ADMISSION_FILENAME).exists():
        return actions
    report = {"schema": REPORT_SCHEMA, "epoch": int(epoch), "started_unix": time.time()}
    freeze_started = False
    try:
        settings, srec = resolve_aux_settings(adaptive_dir, AuxDiscoverySettings(),
                                              override=bool(getattr(args, "adaptive_production_aux_settings_override", False)))
        report["settings"] = srec.get("settings")
        val = check_validation_record(adaptive_dir / VALIDATION_FILENAME, timestep_fs=float(args.timestep_fs))
        report["validation"] = {"ok": val.ok, "reason": val.reason, "k3_max": val.k3_max}
        calib = _calib_steps_for_phase(args, adaptive_dir)
        ok, why = phase_alignment_ok(args, calib_steps=calib)
        report["alignment"] = {"ok": ok, "detail": why, "calib_steps": int(calib)}
        ft = _build_frames(adaptive_dir=adaptive_dir, registry=registry, epoch=epoch, settings=settings, args=args)
        res = _discover(ft=ft, epoch=epoch, settings=settings, out_dir=out_dir, k3_max=val.k3_max)
        report.update(res.report); report["status"] = res.status
        if res.status != "ok":
            return actions
        if not val.ok:
            report["status"] = val.reason
            return actions
        if not ok:
            report["status"] = "alignment"
            return actions
        new = []
        for rank, c in enumerate(res.placement["chosen"]):
            new.append((ACTION, int(c["state_id"]),
                        {"aux_center": float(c["c3"]), "aux_k_kcal_mol": float(c["k3"]),
                         "aux_model_sha256": res.model.model_sha256, "burnin_steps": 0},
                        f"aux_discovery epoch {int(epoch)}",
                        {"aux": {"discovery_epoch": int(epoch), "placement_rank": rank,
                                 "burnin_phase_epoch": int(epoch) + 1,
                                 "forecast": {k: c[k] for k in ("O", "TV", "null", "net", "utility_q10")}}}))
        freeze_started = True
        _freeze(adaptive_dir, epoch, res, new, settings, srec, val)
        report["admitted"] = [[a[1], a[2]] for a in new]
        return actions + new
    except Exception as exc:
        if freeze_started:
            _unfreeze(adaptive_dir)          # a half-frozen admission would block every later boundary
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        return actions
    finally:
        report["finished_unix"] = time.time()
        try:
            _write_json(epoch_dir / REPORT_NAME, report)
        except Exception as exc:             # the hook never raises
            print(f"WARNING: could not write {epoch_dir / REPORT_NAME}: {exc}")


def _freeze(adaptive_dir: Path, epoch, res, new_actions, settings, srec, val) -> None:
    res.model.write(adaptive_dir / MODEL_FILENAME)
    part_sha = res.eval_partition.to_file(adaptive_dir / PARTITION_FILENAME)
    settings_sha = hashlib.sha256(json.dumps(srec.get("settings"), sort_keys=True).encode()).hexdigest()
    _write_json(adaptive_dir / ADMISSION_FILENAME, {
        "schema": ADMISSION_SCHEMA, "epoch": int(epoch), "model_sha256": res.model.model_sha256,
        "eval_partition_sha256": part_sha, "settings_sha256": settings_sha,
        "validation": {"k3_max": val.k3_max, "reason": val.reason},
        "workers": [{"parent_state_id": a[1], "aux_center": a[2]["aux_center"],
                     "aux_k_kcal_mol": a[2]["aux_k_kcal_mol"], "placement_rank": a[4]["aux"]["placement_rank"],
                     "burnin_phase_epoch": a[4]["aux"]["burnin_phase_epoch"]} for a in new_actions]})


def annotate_report_with_refusals(adaptive_dir: Path, epoch_dir: Path, actions: Sequence[Tuple],
                                  refused: Sequence[Mapping]) -> None:
    """Record the applier's refusals of ``admit_aux`` (``apply.refused``). When every admission of the epoch was
    refused the frozen files are removed, so the next boundary discovers again instead of being blocked."""
    path = Path(epoch_dir) / REPORT_NAME
    if not path.exists():
        return
    try:
        mine = []
        n_admit = sum(1 for a in actions if str(a[0]) == ACTION)
        for r in refused:
            i = r.get("index")
            a = actions[int(i)] if i is not None and int(i) < len(actions) else None
            if a is not None and str(a[0]) == ACTION:
                mine.append({**dict(r), "proposal": [a[0], int(a[1])]})
        if not n_admit:
            return
        report = json.loads(path.read_text())
        report["apply"] = {"refused": mine}
        if len(mine) == n_admit:
            report["status"] = "refused_by_applier"
            _unfreeze(adaptive_dir)
        _write_json(path, report)
    except (OSError, ValueError) as exc:
        print(f"WARNING: could not annotate {path} with the applier's refusals ({exc})")


def inject_aux_phase_args(phase_args, adaptive_dir: Path) -> None:
    # Deliberately ungated by the flag: a phase whose window CSV holds a worker must get the model, else the CSV loader refuses aux columns.
    adaptive_dir = Path(adaptive_dir)
    if not (adaptive_dir / ADMISSION_FILENAME).exists():
        return
    csv_path = getattr(phase_args, "windows_2d_csv", None)
    if not csv_path or not Path(csv_path).exists():
        return
    with Path(csv_path).open() as fh:
        roles = {row.get("state_role") for row in csv.DictReader(fh)}
    if "auxiliary" not in roles:
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
