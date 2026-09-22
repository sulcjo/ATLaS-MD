"""Run provenance and reproducibility manifest helpers for GAREUS.

The regular effective-config files preserve the resolved command-line/config
state.  This module writes a richer ``run_manifest`` describing the execution
environment, dependency versions, package/source hashes, requested platform
settings, input-file hashes, selected output-artifact hashes, and start/end
status.  The manifest is intentionally best-effort: missing optional packages or
large output files should be recorded, not allowed to crash a simulation.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata as _metadata
import json
import os
import platform as _platform
import socket
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .io import _json_ready, read_json_file

try:  # Avoid importing config at module import time unless YAML output is used.
    from .config import _write_yaml_or_json, _argv_as_list, _shell_join
except Exception:  # pragma: no cover - defensive fallback for unusual partial imports.
    _write_yaml_or_json = None  # type: ignore[assignment]
    _argv_as_list = None  # type: ignore[assignment]
    _shell_join = None  # type: ignore[assignment]

__all__ = [
    "RUN_MANIFEST_SCHEMA_VERSION",
    "sha256_file",
    "initialize_run_manifest",
    "ensure_run_manifest_initialized",
    "run_manifest_is_skeleton",
    "update_run_manifest",
    "finalize_run_manifest",
    "collect_artifact_hashes",
    "collect_environment_snapshot",
]

RUN_MANIFEST_SCHEMA_VERSION = "1.0"
_HASH_CHUNK_BYTES = 1024 * 1024
_DEFAULT_HASH_LIMIT_MB = 256.0


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _safe_rel(path: Path, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(path)


def _manifest_paths(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    return [out_dir / "run_manifest.json", out_dir / "config" / "run_manifest.json"]


def _manifest_yaml_paths(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    return [out_dir / "run_manifest.yaml", out_dir / "config" / "run_manifest.yaml"]


def _write_manifest(out_dir: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON and YAML manifest mirrors using atomic-ish replaces where easy."""
    out_dir = Path(out_dir)
    clean = _json_ready(dict(payload))
    for path in _manifest_paths(out_dir):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            tmp.replace(path)
        except Exception:
            # Provenance must never make a long MD run fail.
            pass
    if _write_yaml_or_json is not None:
        for path in _manifest_yaml_paths(out_dir):
            try:
                _write_yaml_or_json(path, clean)  # type: ignore[misc]
            except Exception:
                pass


def _read_manifest(out_dir: Path) -> dict[str, Any]:
    for path in _manifest_paths(Path(out_dir)):
        data = read_json_file(path, None)
        if isinstance(data, dict):
            return data
    return {}


def sha256_file(path: Path, *, max_size_mb: Optional[float] = _DEFAULT_HASH_LIMIT_MB) -> dict[str, Any]:
    """Return a reproducibility record for a file.

    Files larger than ``max_size_mb`` are recorded by size and mtime but not
    hashed, to avoid turning final trajectory bookkeeping into a second
    expensive I/O workload.  Pass ``max_size_mb=None`` for an unconditional hash.
    """
    path = Path(path)
    record: dict[str, Any] = {"path": str(path)}
    try:
        st = path.stat()
    except Exception as exc:
        record.update({"exists": False, "error": str(exc)})
        return record
    record.update({
        "exists": True,
        "size_bytes": int(st.st_size),
        "mtime_utc": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc).isoformat(),
    })
    if max_size_mb is not None and st.st_size > float(max_size_mb) * 1024.0 * 1024.0:
        record.update({
            "sha256": None,
            "hash_skipped": True,
            "hash_limit_mb": float(max_size_mb),
            "reason": "file_exceeds_hash_limit",
        })
        return record
    h = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                h.update(chunk)
        record.update({"sha256": h.hexdigest(), "hash_skipped": False})
    except Exception as exc:
        record.update({"sha256": None, "hash_skipped": True, "error": str(exc)})
    return record


def _package_source_hash() -> dict[str, Any]:
    """Hash the installed source files that define this package instance."""
    root = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    for rel in ["pyproject.toml", "gareus_peptide.py", "GENPEPT.py"]:
        p = root / rel
        if p.exists():
            candidates.append(p)
    package_dir = root / "gareus"
    if package_dir.exists():
        candidates.extend(sorted(package_dir.glob("*.py")))
    entries: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in sorted(set(candidates), key=lambda p: str(p.relative_to(root))):
        rec = sha256_file(path, max_size_mb=None)
        rel = _safe_rel(path, root)
        rec["relative_path"] = rel
        entries.append(rec)
        combined.update(rel.encode("utf-8"))
        combined.update(b"\0")
        combined.update(str(rec.get("sha256") or "").encode("ascii", errors="ignore"))
        combined.update(b"\0")
    return {
        "root": str(root),
        "file_count": len(entries),
        "combined_sha256": combined.hexdigest(),
        "files": entries,
    }


def _dist_version(distribution: str) -> Optional[str]:
    try:
        return _metadata.version(distribution)
    except Exception:
        return None


def _optional_module_version(module_name: str, attrs: Iterable[str] = ("__version__", "version")) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    for attr in attrs:
        val = getattr(module, attr, None)
        if val is not None and not callable(val):
            return str(val)
    return None


def _openmm_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "importable": False,
        "version": None,
        "available_platforms": [],
        "default_platform": None,
    }
    try:
        import openmm  # type: ignore
        snap["importable"] = True
        snap["version"] = getattr(openmm, "version", None).full_version if getattr(openmm, "version", None) is not None else _dist_version("openmm")
        platforms = []
        for i in range(openmm.Platform.getNumPlatforms()):
            try:
                p = openmm.Platform.getPlatform(i)
                platforms.append({"index": int(i), "name": str(p.getName()), "speed": float(p.getSpeed())})
            except Exception:
                pass
        snap["available_platforms"] = platforms
        try:
            snap["default_platform"] = str(openmm.Platform.getPlatform(0).getName()) if platforms else None
        except Exception:
            snap["default_platform"] = None
    except Exception as exc:
        snap["error"] = str(exc)
    return snap


def collect_environment_snapshot() -> dict[str, Any]:
    """Return Python/OS/dependency environment metadata for reproducibility."""
    distributions = {
        "gareus-peptide": _dist_version("gareus-peptide"),
        "numpy": _dist_version("numpy"),
        "openmm": _dist_version("openmm"),
        "gamd-openmm": _dist_version("gamd-openmm"),
        "PeptideBuilder": _dist_version("PeptideBuilder"),
        "pdbfixer": _dist_version("pdbfixer"),
        "pymbar": _dist_version("pymbar"),
        "PyYAML": _dist_version("PyYAML"),
    }
    module_versions = {
        "numpy": _optional_module_version("numpy"),
        "pymbar": _optional_module_version("pymbar"),
        "yaml": _optional_module_version("yaml"),
        "PeptideBuilder": _optional_module_version("PeptideBuilder"),
        "gamd": _optional_module_version("gamd"),
    }
    return {
        "python": {
            "version": sys.version.split()[0],
            "full_version": sys.version,
            "executable": sys.executable,
            "implementation": _platform.python_implementation(),
        },
        "os": {
            "platform": sys.platform,
            "system": _platform.system(),
            "release": _platform.release(),
            "version": _platform.version(),
            "machine": _platform.machine(),
            "processor": _platform.processor(),
            "hostname": socket.gethostname(),
        },
        "process": {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
        },
        "dependencies": distributions,
        "module_versions": module_versions,
        "openmm": _openmm_snapshot(),
    }


def _platform_request(args: Any) -> dict[str, Any]:
    keys = [
        "platform", "precision", "device_index", "replica_device_mode", "replica_device_map",
        "cuda_use_cpu_pme", "cuda_use_blocking_sync", "cuda_deterministic_forces",
        "platform_temp_directory", "cpu_threads", "setup_platform", "setup_precision",
        "setup_device_index", "setup_cpu_threads", "us_pull_workers",
    ]
    return {k: getattr(args, k, None) for k in keys if hasattr(args, k)}


def pair_model_sha256(path) -> Optional[str]:
    """Content digest recorded inside a cv_pair_model.json, or None when there is no model."""
    if not path:
        return None
    model_path = Path(path)
    if not model_path.exists():
        # A configured model that is not on disk must not degrade a resume refusal
        # into a "no model" warning.
        raise FileNotFoundError(f"--secondary-cv-model points at a missing file: {model_path}")
    from .correctness._io import json_loads
    doc = json_loads(model_path.read_text(encoding="utf-8"))
    try:
        return str(doc["sha256"])
    except KeyError:
        raise ValueError(f"pair model without sha256 field: {model_path}")


def _method_settings(args: Any) -> dict[str, Any]:
    keys = [
        "seq", "seed", "water_model", "box_shape", "padding_nm", "ionic_strength_molar",
        "temperature_k", "pressure_bar", "barostat_frequency", "production_ensemble",
        "production_barostat_frequency", "timestep_fs", "friction_per_ps", "nonbonded_cutoff_nm",
        "ewald_error_tolerance", "run_mode", "hmr", "hydrogen_mass_amu", "cv1", "cv2",
        "primary_cv", "cv_mode", "contact_scheme", "contact_atom_selection",
        "window_mode", "secondary_cv", "secondary_cv_centers",
        "gamd_boost_type", "sigma0p_kcal_mol", "sigma0d_kcal_mol", "gamd_production_steps",
        "exchange_mode", "exchange_interval", "traj_format", "sample_potential_energy",
        "flush_every_log", "analysis_array_dtype", "state_gamd_lambdas",
        # NPT correction (2026-09-12 design): the resolved backend and the
        # fixed-width fraction are load-bearing for checkpoint compatibility --
        # a resume must not silently change who owns volume moves.
        "npt_barostat_backend", "barostat_volume_step_fraction",
        # Unbiased swarm stage (gareus/swarm/): every --swarm-* argparse dest plus the
        # public --shared-gamd-setup-dir flag it shares with production's envelope reuse
        # path (global-constraints.md's binding anchor -- these are ab initio, no native
        # reference of any kind is recorded here or anywhere else in this stage).
        "swarm_stage", "swarm_seed_ns", "swarm_replicates_per_cell", "swarm_budget_ns",
        "swarm_bins", "swarm_equil_ps", "swarm_output_interval_ps", "swarm_seed_frame_interval_ps",
        "swarm_graft_minimize_iters", "swarm_member_workers", "swarm_member_range", "swarm_round", "swarm_seed_source",
        "swarm_production_seed_csv", "swarm_n_windows", "swarm_overlap_sigma", "swarm_target_beta_sigma",
        "swarm_min_rungs", "swarm_max_rungs", "swarm_ess_floor", "swarm_seeds_per_window",
        "swarm_discard_block_frames", "swarm_min_discard_ps", "swarm_fsf_floor_warn",
        "swarm_max_seed_gap_sigma", "shared_gamd_setup_dir",
        "swarm_stability_sigma_rel_tol", "swarm_stability_extrema_sigma_tol",
        "swarm_min_done_fraction", "swarm_max_graft_fallback_fraction",
        # Frozen CV2 pair model, plan 2026-09-20-auto-cv-pair.
        "secondary_cv_model", "secondary_cv_candidate_set", "secondary_cv_feature_schema",
        "cv_selection_residual_degree", "cv_selection_max_nonlinear_r2",
        "cv_selection_max_coupling_fraction", "cv_selection_k2_reference_kcal",
        "cv_selection_min_gain_nats", "cv_selection_min_windows_cv1",
        "cv_selection_fallback", "swarm_n_windows_cv2",
    ]
    settings = {k: getattr(args, k, None) for k in keys if hasattr(args, k)}
    settings["cv_pair_model_sha256"] = pair_model_sha256(getattr(args, "secondary_cv_model", None))
    # Numerical-kernel identity of this segment (spec F01/F04): which CV evaluator and
    # which exchange-energy assembly produced its samples. A resume compares these.
    from .kernel_identity import EXCHANGE_ENERGY_VERSION, RESIDUAL_EVALUATOR_VERSION
    settings["exchange_energy_version"] = EXCHANGE_ENERGY_VERSION
    settings["cv_evaluator_version"] = (RESIDUAL_EVALUATOR_VERSION
                                        if str(getattr(args, "secondary_cv", "") or "") in ("residual-torsion-pc", "residual-pc")
                                        else None)
    # The λ-ladder is frozen for the whole campaign (spec §3.1), so its
    # per-state lambdas and the envelope it was calibrated against both
    # belong in the immutable run manifest, next to gamd_boost_type above.
    # This is also what Part B's --resume reload
    # (production._reload_state_gamd_lambdas_on_resume) reads
    # state_gamd_lambdas back from when args.state_gamd_lambdas does not
    # survive the restart.
    try:
        from .pep_gamd import is_pep_gamd
        _pep_gamd = bool(is_pep_gamd(args))
    except Exception:
        _pep_gamd = False
    # Records the adaptive-production campaign-export convention only;
    # gareus.mbar_analysis.ladder.load_pep_gamd_envelope(run_dir) also accepts
    # the bare single-run filename ("shared_gamd_setup_globals.json" directly
    # under run_dir, no "global_shared_gamd_setup/" prefix) as a fallback, so
    # this field is a hint for where to look first, not the only valid path.
    settings["pep_gamd_envelope_path"] = (
        "global_shared_gamd_setup/shared_gamd_setup_globals.json" if _pep_gamd else None
    )
    return settings


def _forcefield_settings(args: Any) -> dict[str, Any]:
    # Derived, never duplicated: this dict used to carry its own copy of the
    # water map, so updating system_setup alone made the provenance record
    # disagree with what actually ran.
    from .system_setup import forcefield_selection_from_args, forcefield_xml_paths
    _ff, _water, _allow = forcefield_selection_from_args(args)
    try:
        _xml = forcefield_xml_paths(_ff, _water, allow_mismatch=True)
    except ValueError:
        _xml = []
    return {
        "forcefield_xml": _xml,
        "forcefield": _ff,
        "water_model": getattr(args, "water_model", None) or "tip3p",
        "nonbonded_method": "PME",
        "constraints": "HBonds",
        "rigid_water": True,
        "ewald_error_tolerance": getattr(args, "ewald_error_tolerance", None),
        "nonbonded_cutoff_nm": getattr(args, "nonbonded_cutoff_nm", None),
        "hydrogen_mass_amu": getattr(args, "hydrogen_mass_amu", None),
        "hmr": getattr(args, "hmr", None),
    }


def _public_args(args: Any) -> dict[str, Any]:
    try:
        return {k: v for k, v in vars(args).items() if not str(k).startswith("_")}
    except Exception:
        return {}


def _input_file_hashes(args: Any) -> dict[str, Any]:
    candidates: dict[str, Optional[Path]] = {}
    for key in ["config", "windows_2d_csv"]:
        val = getattr(args, key, None)
        if val:
            candidates[key] = Path(val)
    seed_dir = getattr(args, "seed_conformers_dir", None)
    if seed_dir:
        seed_path = Path(seed_dir)
        candidates["seed_conformers_dir"] = seed_path
        for name in ["final_survivor_seeds.csv", "seeds.csv"]:
            p = seed_path / name
            if p.exists():
                candidates[f"seed_conformers_{name}"] = p
        # r7 (and any other GENPEPT seed library) is described from its own summary
        # file, never from the campaign YAML (global-constraints.md's binding anchor):
        # record its hash here so provenance can attribute the library that was
        # actually read, not a stale spec-file description of it.
        summary_p = seed_path / "GENPEPT_turbo_summary.json"
        if summary_p.exists():
            candidates["seed_conformers_GENPEPT_turbo_summary.json"] = summary_p
    return {name: sha256_file(path) if path is not None and path.is_file() else {"path": str(path), "exists": bool(path and path.exists()), "is_directory": bool(path and path.is_dir())} for name, path in candidates.items()}


def _key_artifact_paths(out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    return {
        "built_peptide_pdb": out_dir / "00_built_peptide.pdb",
        "solvated_start_pdb": out_dir / "01_solvated_start.pdb",
        "minimized_pdb": out_dir / "02_minimized.pdb",
        "npt_equilibrated_pdb": out_dir / "02_npt_equilibrated.pdb",
        "umbrella_windows_csv": out_dir / "umbrella_windows.csv",
        "umbrella_explicit_windows_csv": out_dir / "umbrella_explicit_windows.csv",
        "umbrella_pymbar_metadata_json": out_dir / "umbrella_pymbar_metadata.json",
        "gareus_metadata_json": out_dir / "gareus_metadata.json",
        "samples_csv": out_dir / "samples.csv",
        "exchanges_csv": out_dir / "exchanges.csv",
        "analysis_chunks_manifest_json": out_dir / "analysis_chunks_manifest.json",
        "analysis_arrays_metadata_json": out_dir / "analysis_arrays_metadata.json",
        "exchange_tuning_report_json": out_dir / "exchange_tuning_report.json",
        "final_report_json": out_dir / "final_report.json",
        "output_layout_json": out_dir / "output_layout.json",
        "checkpoint_manifest_json": out_dir / "checkpoints" / "production_checkpoint_manifest.json",
        # Unbiased swarm stage (gareus/swarm/analyze.py's real output paths):
        # the frozen envelope, the CV1 lambda ladder and its run-args sidecar, the
        # exported seed bank, and the stage's own report -- gated, so a failed gate
        # withholds windows_lambda_ladder_csv (analyze.py deletes stale copies too).
        "swarm_shared_gamd_setup_globals_json": out_dir / "swarm" / "analysis" / "shared_gamd_setup" / "shared_gamd_setup_globals.json",
        "swarm_windows_lambda_ladder_csv": out_dir / "swarm" / "analysis" / "windows_lambda_ladder.csv",
        "swarm_ladder_run_args_yaml": out_dir / "swarm" / "analysis" / "ladder_run_args.yaml",
        "swarm_seed_bank_final_survivor_seeds_csv": out_dir / "swarm" / "analysis" / "seed_bank" / "final_survivor_seeds.csv",
        "swarm_report_json": out_dir / "swarm" / "analysis" / "swarm_report.json",
        "swarm_ladder_design_json": out_dir / "swarm" / "analysis" / "ladder_design.json",
        "swarm_gate_json": out_dir / "swarm" / "analysis" / "swarm_gate.json",
    }


def _directory_summary(path: Path, *, patterns: Iterable[str]) -> dict[str, Any]:
    path = Path(path)
    rows = []
    if path.exists():
        for pattern in patterns:
            for p in sorted(path.glob(pattern)):
                if p.is_file():
                    try:
                        st = p.stat()
                        rows.append({"relative_path": _safe_rel(p, path), "size_bytes": int(st.st_size)})
                    except Exception:
                        pass
    return {"path": str(path), "exists": path.exists(), "file_count": len(rows), "files": rows}


def collect_artifact_hashes(out_dir: Path, *, hash_limit_mb: float = _DEFAULT_HASH_LIMIT_MB) -> dict[str, Any]:
    """Collect hashes and summaries for key reproducibility artifacts."""
    out_dir = Path(out_dir)
    artifact_records = {
        name: sha256_file(path, max_size_mb=hash_limit_mb)
        for name, path in _key_artifact_paths(out_dir).items()
        if path.exists()
    }
    trajectory_summary = _directory_summary(out_dir / "replica_trajectories", patterns=["*.dcd", "*.xtc"])
    final_pdb_summary = _directory_summary(out_dir / "final_pdbs", patterns=["*.pdb"])
    topology_hashes = {
        name: artifact_records[name]
        for name in ["built_peptide_pdb", "solvated_start_pdb", "npt_equilibrated_pdb"]
        if name in artifact_records
    }
    window_hashes = {
        name: artifact_records[name]
        for name in ["umbrella_windows_csv", "umbrella_explicit_windows_csv", "umbrella_pymbar_metadata_json"]
        if name in artifact_records
    }
    return {
        "hash_limit_mb": float(hash_limit_mb),
        "topology_hashes": topology_hashes,
        "window_table_hashes": window_hashes,
        "artifacts": artifact_records,
        "trajectory_files": trajectory_summary,
        "final_pdb_files": final_pdb_summary,
    }


def _warn_on_provenance_drift(existing: dict[str, Any], out_dir: Path, args: Any) -> None:
    """Compare this resume's freshly-computed provenance hashes against the
    ones the manifest already recorded, and warn (never raise or block) on a
    mismatch.

    The hashes have been captured for reproducibility since this manifest's
    inception, but nothing ever compared them again after the first write -
    so code/dependency/input drift between two invocations of the SAME
    campaign (a fix deployed mid-campaign, an environment update between
    SLURM restarts) was invisible even though the data to detect it already
    existed. This is deliberately warn-only: a version bump between restarts
    is not automatically wrong, and this check must never turn a legitimate
    resume into a hard failure.
    """
    try:
        old_source = existing.get("package_source")
        new_source = _package_source_hash()
        if isinstance(old_source, dict) and isinstance(new_source, dict) and old_source != new_source:
            print(
                f"WARNING: provenance drift detected resuming {out_dir}: package_source hash "
                f"differs from the run manifest recorded at start (code changed between restarts "
                f"of this campaign - old={old_source.get('hash', old_source)!r} "
                f"new={new_source.get('hash', new_source)!r})."
            )
        old_inputs = existing.get("input_files")
        new_inputs = _input_file_hashes(args)
        if isinstance(old_inputs, dict) and isinstance(new_inputs, dict) and old_inputs != new_inputs:
            changed = sorted(set(old_inputs) | set(new_inputs))
            changed = [k for k in changed if old_inputs.get(k) != new_inputs.get(k)]
            print(
                f"WARNING: provenance drift detected resuming {out_dir}: input_files hash(es) "
                f"differ from the run manifest recorded at start for: {changed}."
            )
    except Exception as exc:
        print(f"WARNING: provenance drift check failed (non-fatal, resume continues): {exc}")


def initialize_run_manifest(args: Any, out_dir: Path, argv: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """Create/update the start-of-run provenance manifest."""
    out_dir = Path(out_dir)
    arg_list = list(_argv_as_list(argv) if _argv_as_list is not None else (sys.argv[1:] if argv is None else list(argv)))  # type: ignore[misc]
    existing = _read_manifest(out_dir)
    resume = bool(getattr(args, "resume", False))
    if resume and existing:
        _warn_on_provenance_drift(existing, out_dir, args)
    run_id = existing.get("run_id") if resume and existing.get("run_id") else str(uuid.uuid4())
    start_time = existing.get("start_time_utc") if resume and existing.get("start_time_utc") else _utc_now()
    command_line = _shell_join([sys.executable, "-m", "gareus"] + [str(x) for x in arg_list]) if _shell_join is not None else " ".join([sys.executable, "-m", "gareus"] + [str(x) for x in arg_list])  # type: ignore[misc]
    payload: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "started",
        "start_time_utc": start_time,
        "last_updated_utc": _utc_now(),
        "end_time_utc": None,
        "out_dir": str(out_dir),
        "main_dir": getattr(args, "_main_dir", None),
        "scratchdir_active": bool(getattr(args, "_main_dir", None)),
        "command": {
            "argv": [sys.executable, "-m", "gareus"] + [str(x) for x in arg_list],
            "command_line": command_line,
            "working_directory": str(Path.cwd()),
        },
        "environment": collect_environment_snapshot(),
        "package_source": _package_source_hash(),
        "platform_request": _platform_request(args),
        "forcefield": _forcefield_settings(args),
        "method_settings": _method_settings(args),
        "config_source": {
            "config": str(getattr(args, "config", "") or "") or None,
            "config_values": getattr(args, "_config_values", {}),
        },
        "input_files": _input_file_hashes(args),
        "resolved_args": _public_args(args),
        "artifacts": {},
        "notes": [
            "effective_config.yaml/json preserve resolved CLI/config values; run_manifest.* adds environment, dependency, source, and artifact hashes.",
            "Large trajectory files are summarized by size/count and are not fully hashed by default.",
        ],
    }
    _write_manifest(out_dir, payload)
    return payload


def _deep_update(target: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = value
    return target


def run_manifest_is_skeleton(payload: Mapping[str, Any]) -> bool:
    """True when a manifest never went through initialize_run_manifest().

    update_run_manifest() creates a three-key skeleton (schema_version, run_id,
    status) when it patches a directory that has no manifest yet, so a manifest
    can exist on disk without any start-of-run provenance. ``start_time_utc`` is
    written only by initialize_run_manifest(), which makes it the marker.
    """
    return not payload or not payload.get("start_time_utc")


# Skeleton keys that the fresh initialisation must own; everything else recorded
# before the initialisation (method_settings patches, openmm_runtime, ...) is kept.
_SKELETON_KEYS_OWNED_BY_INIT = frozenset({
    "schema_version", "status", "last_updated_utc", "start_time_utc", "run_id", "end_time_utc", "error",
})


def ensure_run_manifest_initialized(args: Any, out_dir: Path, argv: Optional[Iterable[str]] = None) -> bool:
    """Initialise the run manifest unless a complete one already exists.

    Returns True when a manifest was written (missing or skeleton), False when
    the existing manifest was already complete and was left untouched. A
    skeleton keeps its ``run_id`` and every patch recorded into it before the
    initialisation -- those patches were written by code that ran closest to
    the final value (e.g. ``method_settings.state_gamd_lambdas``), so they win
    over the recomputed defaults.

    chignolin_8/epoch_000 (2026-09-22): production ran three jobs against a
    manifest holding only ``method_settings.state_gamd_lambdas``, because the
    caller tested file existence. Without the kernel identity that
    initialize_run_manifest() records, the F04 resume guard read the skeleton as
    a pre-F01 campaign and refused to continue an eligible segment.
    """
    out_dir = Path(out_dir)
    existing = _read_manifest(out_dir)
    if existing and not run_manifest_is_skeleton(existing):
        return False
    payload = initialize_run_manifest(args, out_dir, argv=argv)
    if existing:
        if existing.get("run_id"):
            payload["run_id"] = existing["run_id"]
        overlay = {k: v for k, v in existing.items() if k not in _SKELETON_KEYS_OWNED_BY_INIT}
        _deep_update(payload, overlay)
        _write_manifest(out_dir, payload)
    return True


def update_run_manifest(out_dir: Path, patch: Mapping[str, Any]) -> dict[str, Any]:
    """Best-effort merge a patch into the run manifest."""
    payload = _read_manifest(out_dir)
    if not payload:
        payload = {"schema_version": RUN_MANIFEST_SCHEMA_VERSION, "run_id": str(uuid.uuid4()), "status": "partial"}
    payload["last_updated_utc"] = _utc_now()
    _deep_update(payload, dict(patch))
    _write_manifest(out_dir, payload)
    return payload


def finalize_run_manifest(
    args: Any,
    out_dir: Path,
    *,
    status: str = "completed",
    error: Optional[BaseException | Mapping[str, Any]] = None,
    hash_limit_mb: float = _DEFAULT_HASH_LIMIT_MB,
) -> dict[str, Any]:
    """Finalize the manifest with end time, status, artifact hashes and errors."""
    out_dir = Path(out_dir)
    payload = _read_manifest(out_dir)
    if not payload:
        payload = initialize_run_manifest(args, out_dir, argv=[])
    payload["status"] = str(status)
    payload["last_updated_utc"] = _utc_now()
    if status in {"completed", "failed", "interrupted"}:
        payload["end_time_utc"] = _utc_now()
    if error is not None:
        if isinstance(error, Mapping):
            err = dict(error)
        else:
            err = {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            }
        payload["error"] = err
    payload["environment_final"] = collect_environment_snapshot()
    payload["artifact_hashes"] = collect_artifact_hashes(out_dir, hash_limit_mb=hash_limit_mb)
    # If this is a parent adaptive-feedback directory, also summarize final_production.
    final_dir = out_dir / "final_production"
    if final_dir.exists() and final_dir.is_dir():
        payload["final_production_artifact_hashes"] = collect_artifact_hashes(final_dir, hash_limit_mb=hash_limit_mb)
    _write_manifest(out_dir, payload)
    return payload
