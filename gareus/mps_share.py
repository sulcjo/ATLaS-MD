"""Process-wide CUDA MPS active-thread percentage (spec 2026-09-26 §5).

Applied once, right after parse_args, before any OpenMM platform exists: the
MPS client reads CUDA_MPS_ACTIVE_THREAD_PERCENTAGE when CUDA first
initialises, and setup, US-pull and production all run in this one process.
Importing gareus.core/gareus.cli and running parse_args loads no openmm
(pinned by tests/test_replica_admission_config.py), so the variable is set in
time. The openmm guard cannot see CUDA initialised by a non-OpenMM import
(none exists today). MPS reports no effective share back, so the manifest
records the value as "requested".
"""

from __future__ import annotations

import os
import sys
from typing import Any, MutableMapping, Optional

ENV_VAR = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


def apply_mps_thread_percentage(
    args: Any,
    environ: Optional[MutableMapping[str, str]] = None,
    modules: Optional[dict] = None,
) -> None:
    """Apply ``args.cuda_mps_active_thread_percentage``; record the inherited value on ``args``."""
    env = os.environ if environ is None else environ
    mods = sys.modules if modules is None else modules
    requested = getattr(args, "cuda_mps_active_thread_percentage", "inherit")
    inherited = env.get(ENV_VAR)
    args._cuda_mps_inherited_env = inherited
    if requested == "inherit":
        return
    if "openmm" in mods:
        raise SystemExit(
            f"[mps] --cuda-mps-active-thread-percentage {requested} requested, but openmm is already "
            "imported in this process; CUDA may already have read the MPS environment. Refusing to set "
            f"{ENV_VAR} too late."
        )
    if inherited is not None and inherited.strip() != str(requested):
        raise SystemExit(
            f"[mps] {ENV_VAR}={inherited!r} is already set in the environment, but "
            f"--cuda-mps-active-thread-percentage {requested} was requested. Remove one of them; "
            "the setting is never overridden silently."
        )
    env[ENV_VAR] = str(requested)
    if not env.get("CUDA_MPS_PIPE_DIRECTORY"):
        print(
            f"[mps] WARNING: --cuda-mps-active-thread-percentage {requested} set, but CUDA_MPS_PIPE_DIRECTORY "
            "is not set; MPS does not appear to be running and the setting has no effect.",
            file=sys.stderr,
            flush=True,
        )
