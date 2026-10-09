"""atlas-aux-validation-v1: evidence that the aux restraint is safe at the campaign's timestep (finite-timestep
check up to k3_max), under NPT (controlled distribution), and affordable (236-context cost). Written by the
validation runs, read by the admission hook."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VALIDATION_SCHEMA = "atlas-aux-validation-v1"
REQUIRED_CHECKS = ("finite_timestep", "npt", "cost")


@dataclass(frozen=True)
class ValidationStatus:
    ok: bool
    reason: str
    k3_max: Optional[float] = None


def check_validation_record(path: Path, *, timestep_fs: float) -> ValidationStatus:
    path = Path(path)
    if not path.exists():
        return ValidationStatus(False, "validation_missing")
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return ValidationStatus(False, "validation_unreadable")
    if not isinstance(rec, dict) or rec.get("schema") != VALIDATION_SCHEMA:
        return ValidationStatus(False, "validation_schema")
    checks = rec.get("checks") or {}
    for name in REQUIRED_CHECKS:
        if checks.get(name) != "pass":
            return ValidationStatus(False, f"validation_failed:{name}")
    try:
        ts = float(rec.get("timestep_fs", -1.0))
        k3 = float(rec["k3_max_validated"])
    except (KeyError, TypeError, ValueError):
        return ValidationStatus(False, "validation_unreadable")
    if abs(ts - float(timestep_fs)) > 1e-9:
        return ValidationStatus(False, "validation_timestep_mismatch")
    return ValidationStatus(True, "ok", k3)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m gareus.adaptive.aux_discovery.validation")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", required=True)
    w.add_argument("--commit", required=True)
    w.add_argument("--timestep-fs", type=float, required=True)
    w.add_argument("--k3-max", type=float, required=True)
    for name in REQUIRED_CHECKS:
        w.add_argument(f"--{name.replace('_', '-')}", choices=("pass", "fail"), required=True)
    w.add_argument("--evidence", nargs="*", default=[])
    a = p.parse_args(argv)
    rec = {"schema": VALIDATION_SCHEMA, "code_commit": a.commit, "timestep_fs": a.timestep_fs,
           "k3_max_validated": a.k3_max, "checks": {n: getattr(a, n) for n in REQUIRED_CHECKS},
           "evidence": list(a.evidence), "written_unix": time.time()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
