"""Tiny end-to-end workflow test command for the GAREUS package.

The normal pytest smoke tests intentionally avoid requiring OpenMM, PeptideBuilder,
and, for GaMD modes, gamd-openmm.  This module provides an optional *real* integration test that
runs a deliberately tiny peptide workflow when the MD stack is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "build_tiny_run_argv",
    "check_optional_md_dependencies",
    "validate_tiny_run_outputs",
    "main",
]


DEFAULT_EXPECTED_ARTIFACTS = [
    "effective_config.json",
    "effective_config.yaml",
    "run_manifest.json",
    "run_manifest.yaml",
    "00_built_peptide.pdb",
    "01_solvated_start.pdb",
    "02_minimized.pdb",
    "02_npt_equilibrated.pdb",
    "umbrella_windows.csv",
    "umbrella_explicit_windows.csv",
    "umbrella_pymbar_metadata.json",
    "samples.csv",
    "exchanges.csv",
    "final_run_report.json",
    "final_run_report.md",
]

OPTIONAL_EXPECTED_ARTIFACTS = [
    "analysis_arrays.npz",
    "analysis_chunks",
    "distances.csv",
    "progress.jsonl",
    "shared_gamd_setup_globals.json",
    "replica_shared_gamd_copy_report.json",
]


def _shell_join(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in argv)


def build_tiny_run_argv(
    *,
    out: str | os.PathLike[str],
    seq: str = "AA",
    platform: str = "Reference",
    setup_platform: str | None = None,
    box_shape: str = "dodecahedron",
    production_ensemble: str = "npt",
    traj_format: str = "none",
    run_mode: str = "gamd",
    extra_args: Sequence[str] | None = None,
) -> list[str]:
    """Build the tiny real-workflow command line passed to :mod:`gareus`.

    The defaults are intentionally tiny but still exercise the expensive path:
    peptide build, solvation, staged equilibration, umbrella-window creation,
    umbrella-start preparation, optional shared GaMD setup, production sampling,
    and at least one exchange attempt.
    """
    setup_platform = setup_platform or platform
    argv = [
        "--seq", str(seq),
        "--out", str(out),
        "--seed", "2026",
        "--platform", str(platform),
        "--setup-platform", str(setup_platform),
        "--box-shape", str(box_shape),
        "--padding-nm", "0.55",
        "--ionic-strength-molar", "0.0",
        "--temperature-k", "300.0",
        "--production-ensemble", str(production_ensemble),
        "--run-mode", str(run_mode),
        "--timestep-fs", "0.5",
        "--friction-per-ps", "5.0",
        "--minimize-iterations", "5",
        "--nvt-warmup-steps", "2",
        "--nvt-warmup-timestep-fs", "0.25",
        "--npt-ramp-steps", "2",
        "--npt-ramp-timestep-fs", "0.25",
        "--npt-steps", "4",
        "--window-mode", "manual",
        "--windows-a", "3.5", "4.5",
        "--window-k-kcal-a2", "1.0", "1.0",
        "--us-starting-structure-mode", "pull",
        "--us-pull-steps-per-window", "2",
        "--us-pull-timestep-fs", "0.25",
        "--us-pull-minimize-iterations", "1",
        "--gamd-cmd-steps", "100",
        "--equil-steps", "100",
        "--production-steps", "20",
        "--gamd-averaging-window", "50",
        "--gamd-multiwindow-recon-prep-steps", "1",
        "--gamd-multiwindow-recon-steps", "2",
        "--gamd-multiwindow-recon-report-interval", "1",
        "--exchange-mode", "neighbor",
        "--exchange-interval", "2",
        "--report-interval", "2",
        "--distance-output-interval", "2",
        "--distance-output-mode", "csv",
        "--traj-format", str(traj_format),
        "--checkpoint-interval", "0",
        "--progress-mode", "none",
        "--tui-mode", "none",
    ]
    if extra_args:
        argv.extend(str(x) for x in extra_args)
    return argv


def check_optional_md_dependencies(run_mode: str = "gamd") -> dict:
    """Return import/platform availability for the optional MD stack."""
    mode = str(run_mode or "gamd").strip().lower().replace("_", "-")
    require_gamd = mode in {"gamd", "hmr-gamd"}
    result: dict[str, object] = {"ok": True, "missing": [], "imports": {}, "platforms": [], "run_mode": mode, "requires_gamd_openmm": require_gamd}
    checks = [
        ("openmm", "openmm", True),
        ("openmm.app", "openmm.app", True),
        ("openmm.unit", "openmm.unit", True),
        ("gamd.integrator_factory", "gamd.integrator_factory", require_gamd),
        ("PeptideBuilder", "PeptideBuilder", True),
        ("Bio.PDB", "Bio.PDB", True),
    ]
    missing: list[str] = []
    imports: dict[str, dict[str, str]] = {}
    for label, module_name, required in checks:
        try:
            module = __import__(module_name, fromlist=["*"])
            imports[label] = {"status": "ok", "version": str(getattr(module, "__version__", "")), "required": bool(required)}
        except Exception as exc:  # pragma: no cover - depends on optional runtime stack
            imports[label] = {"status": "missing", "required": bool(required), "error": f"{type(exc).__name__}: {exc}"}
            if required:
                missing.append(label)
    result["imports"] = imports
    result["missing"] = missing
    result["ok"] = not missing
    if not missing:
        try:  # pragma: no cover - optional runtime stack
            import openmm  # type: ignore[import-not-found]
            platforms = []
            for i in range(openmm.Platform.getNumPlatforms()):
                platforms.append(openmm.Platform.getPlatform(i).getName())
            result["platforms"] = platforms
        except Exception as exc:
            result["platforms_error"] = f"{type(exc).__name__}: {exc}"
    return result


def validate_tiny_run_outputs(out_dir: str | os.PathLike[str]) -> dict:
    """Validate that a completed tiny run produced core workflow artifacts."""
    root = Path(out_dir)
    missing_required = [name for name in DEFAULT_EXPECTED_ARTIFACTS if not (root / name).exists()]
    present_optional = [name for name in OPTIONAL_EXPECTED_ARTIFACTS if (root / name).exists()]
    artifact_sizes = {}
    for name in DEFAULT_EXPECTED_ARTIFACTS + OPTIONAL_EXPECTED_ARTIFACTS:
        path = root / name
        if path.exists() and path.is_file():
            artifact_sizes[name] = path.stat().st_size
        elif path.exists() and path.is_dir():
            artifact_sizes[name] = sum(1 for _ in path.rglob("*"))
    ok = not missing_required
    gamd_globals_path = root / "shared_gamd_setup_globals.json"
    gamd_calibration_ok = True
    gamd_calibration_note = "shared_gamd_setup_globals.json not present (conventional MD run_mode)"
    if gamd_globals_path.exists():
        gamd_globals = json.loads(gamd_globals_path.read_text())
        mode = str(gamd_globals.get("mode", ""))
        if mode.startswith("disabled_"):
            gamd_calibration_note = f"GaMD disabled for this run_mode ({mode})"
        elif mode == "joint_envelope_gamd_calibration" and gamd_globals.get("joint_envelope"):
            gamd_calibration_note = f"joint_envelope_gamd_calibration OK, groups={sorted(gamd_globals['joint_envelope'])}"
        else:
            gamd_calibration_ok = False
            gamd_calibration_note = f"expected mode='joint_envelope_gamd_calibration' with non-empty 'joint_envelope', got mode={mode!r}"
    ok = ok and gamd_calibration_ok
    return {
        "ok": ok,
        "out_dir": str(root),
        "missing_required": missing_required,
        "present_optional": present_optional,
        "artifact_sizes": artifact_sizes,
        "gamd_calibration_ok": gamd_calibration_ok,
        "gamd_calibration_note": gamd_calibration_note,
    }


def _write_report(out_dir: Path, report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tiny_integration_test_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# GAREUS tiny integration test report",
        "",
        f"Status: **{report.get('status', 'unknown')}**",
        f"Started UTC: `{report.get('started_utc', '')}`",
        f"Finished UTC: `{report.get('finished_utc', '')}`",
        "",
        "## Command",
        "",
        "```bash",
        report.get("command", ""),
        "```",
        "",
    ]
    deps = report.get("dependencies") or {}
    if deps:
        lines.extend(["## Dependency check", "", f"OK: `{deps.get('ok')}`", ""])
        missing = deps.get("missing") or []
        if missing:
            lines.append("Missing: " + ", ".join(str(x) for x in missing))
            lines.append("")
    validation = report.get("validation") or {}
    if validation:
        lines.extend(["## Output validation", "", f"OK: `{validation.get('ok')}`", ""])
        missing = validation.get("missing_required") or []
        if missing:
            lines.append("Missing required artifacts:")
            for item in missing:
                lines.append(f"- `{item}`")
            lines.append("")
    if report.get("error"):
        lines.extend(["## Error", "", "```text", str(report["error"]), "```", ""])
    (out_dir / "tiny_integration_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gareus-test-run",
        description="Run or print a tiny real GAREUS OpenMM/GaMD integration workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="gareus_tiny_integration_test", help="Output directory for the tiny run.")
    parser.add_argument("--seq", default="AA", help="Tiny test peptide sequence.")
    parser.add_argument("--platform", default="Reference", help="OpenMM production platform for the tiny test.")
    parser.add_argument("--setup-platform", default="", help="OpenMM setup platform; empty reuses --platform.")
    parser.add_argument("--box-shape", choices=["dodecahedron", "cube", "octahedron"], default="dodecahedron")
    parser.add_argument("--production-ensemble", choices=["npt", "nvt"], default="npt")
    parser.add_argument("--run-mode", choices=["cmd", "hmr-cmd", "gamd", "hmr-gamd"], default="gamd", help="Tiny-test dynamics mode. cmd/hmr-cmd do not require gamd-openmm; gamd/hmr-gamd exercise the GaMD path. The tiny test intentionally uses a conservative explicit timestep for speed/stability.")
    parser.add_argument("--traj-format", choices=["none", "dcd", "xtc"], default="none")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved gareus command without executing it.")
    parser.add_argument("--check-deps", action="store_true", help="Only check optional MD dependencies/platforms and exit.")
    parser.add_argument("--skip-if-missing", action="store_true", help="Return success instead of failure when optional MD dependencies are missing.")
    parser.add_argument("--force", action="store_true", help="Remove an existing --out directory before running.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Append one additional raw argument token to the underlying gareus command. Repeat for multiple tokens; use --extra-arg=--flag-name when the token begins with a dash.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out)
    run_argv = build_tiny_run_argv(
        out=out_dir,
        seq=args.seq,
        platform=args.platform,
        setup_platform=args.setup_platform or None,
        box_shape=args.box_shape,
        production_ensemble=args.production_ensemble,
        traj_format=args.traj_format,
        run_mode=args.run_mode,
        extra_args=args.extra_arg,
    )
    command_display = "gareus " + _shell_join(run_argv)

    if args.dry_run:
        print(command_display)
        return 0

    deps = check_optional_md_dependencies(run_mode=args.run_mode)
    if args.check_deps:
        print(json.dumps(deps, indent=2, sort_keys=True))
        return 0 if deps.get("ok") or args.skip_if_missing else 2

    started = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": "gareus_tiny_integration_test_report_v1",
        "status": "started",
        "started_utc": started,
        "command": command_display,
        "dependencies": deps,
    }

    if not deps.get("ok"):
        report["status"] = "skipped" if args.skip_if_missing else "missing_dependencies"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_report(out_dir, report)
        print(json.dumps(deps, indent=2, sort_keys=True))
        print(f"Optional MD dependencies missing; report written to {out_dir / 'tiny_integration_test_report.json'}")
        return 0 if args.skip_if_missing else 2

    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.force:
            report["status"] = "refused_existing_out"
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            report["error"] = f"Output directory is not empty: {out_dir}. Pass --force to remove it."
            _write_report(out_dir, report)
            print(report["error"], file=sys.stderr)
            return 2
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.time()
        from .core import main as gareus_main

        gareus_main(run_argv)
        elapsed = time.time() - t0
        validation = validate_tiny_run_outputs(out_dir)
        report["status"] = "completed" if validation.get("ok") else "failed_validation"
        report["elapsed_seconds"] = elapsed
        report["validation"] = validation
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_report(out_dir, report)
        print(f"Tiny integration test status: {report['status']}")
        print(f"Report: {out_dir / 'tiny_integration_test_report.md'}")
        return 0 if validation.get("ok") else 1
    except BaseException as exc:  # pragma: no cover - optional runtime stack
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_report(out_dir, report)
        raise


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
