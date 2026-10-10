"""atlas-aux-validation-v1: evidence that the aux restraint is safe at the campaign's timestep (finite-timestep
check up to k3_max), under NPT (controlled distribution), and affordable (236-context cost). Written by the
validation runs, read by the admission hook."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

VALIDATION_SCHEMA = "atlas-aux-validation-v1"
KNOWN_SCHEMAS = (VALIDATION_SCHEMA,)
REQUIRED_CHECKS = ("finite_timestep", "npt", "cost")


@dataclass(frozen=True)
class ValidationStatus:
    ok: bool
    reason: str
    k3_max: Optional[float] = None


def _is_real(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _evidence_ok(ev) -> bool:
    if isinstance(ev, str):
        return bool(ev.strip())
    return (isinstance(ev, list) and len(ev) > 0
            and all(isinstance(e, str) and e.strip() for e in ev))


def check_validation_record(path: Path, *, timestep_fs: float) -> ValidationStatus:
    """Typed, evidence-bearing check of ``aux_validation.json``. One validator for the driver start and the
    admission hook (F03): a malformed record fails before MD, not at the first boundary."""
    path = Path(path)
    if not path.exists():
        return ValidationStatus(False, "validation_missing")
    try:
        rec = json.loads(path.read_text())
    except (OSError, ValueError):
        return ValidationStatus(False, "validation_unreadable")
    if not isinstance(rec, dict) or rec.get("schema") not in KNOWN_SCHEMAS:
        return ValidationStatus(False, "validation_schema")
    ts, k3 = rec.get("timestep_fs"), rec.get("k3_max_validated")
    if not (_is_real(ts) and ts > 0 and _is_real(k3) and k3 > 0):
        return ValidationStatus(False, "validation_unreadable")
    if "temperature_k" in rec:
        t = rec["temperature_k"]
        if not (_is_real(t) and t > 0):
            return ValidationStatus(False, "validation_unreadable")
    checks = rec.get("checks")
    if not isinstance(checks, dict):
        return ValidationStatus(False, "validation_unreadable")
    for name in REQUIRED_CHECKS:
        chk = checks.get(name)
        if not isinstance(chk, dict):
            if chk == "fail":
                return ValidationStatus(False, f"validation_failed:{name}")
            return ValidationStatus(False, f"validation_evidence_missing:{name}")
        if chk.get("status") != "pass":
            return ValidationStatus(False, f"validation_failed:{name}")
        if not _evidence_ok(chk.get("evidence")):
            return ValidationStatus(False, f"validation_evidence_missing:{name}")
    if abs(float(ts) - float(timestep_fs)) > 1e-9:
        return ValidationStatus(False, "validation_timestep_mismatch")
    return ValidationStatus(True, "ok", float(k3))


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
        w.add_argument(f"--{name.replace('_', '-')}-evidence", nargs="*", default=[],
                       help="non-empty evidence (log paths, result summaries) for this check; required for pass")
    w.add_argument("--evidence", nargs="*", default=[])
    a = p.parse_args(argv)
    for name in REQUIRED_CHECKS:
        if getattr(a, name) == "pass" and not any(str(e).strip() for e in getattr(a, f"{name}_evidence")):
            p.error(f"--{name.replace('_', '-')} pass needs non-empty --{name.replace('_', '-')}-evidence")
    rec = {"schema": VALIDATION_SCHEMA, "code_commit": a.commit, "timestep_fs": a.timestep_fs,
           "k3_max_validated": a.k3_max, "checks": {n: {"status": getattr(a, n), "evidence": list(getattr(a, f"{n}_evidence"))}
                      for n in REQUIRED_CHECKS},
           "evidence": list(a.evidence), "written_unix": time.time()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
