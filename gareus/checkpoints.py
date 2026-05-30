"""Checkpoint and resume helpers for GAREUS."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .io import read_json_file
from .imports import import_openmm
from .system_setup import make_forcefield
from .production import checkpoint_manifest_path

__all__ = [
    "equilibration_state_xml_path",
    "find_resume_equil_state_path",
    "production_checkpoint_available",
    "load_existing_openmm_setup_for_resume",
]


def equilibration_state_xml_path(out_dir: Path) -> Path:
    return Path(out_dir) / "03_npt_equilibrated_state.xml"


def find_resume_equil_state_path(out_dir: Path) -> Optional[Path]:
    """Return best available state XML path for resume without deserializing.

    Pure path logic — no OpenMM import — so it can be tested without GPU/OpenMM.

    Search order (first match wins):
    1. Primary: 03_npt_equilibrated_state.xml written by system_setup
    2. Campaign-wide shared GaMD setup (adaptive-production global)
    3. Latest epoch baseline shared GaMD state
    4. Latest feedback pilot shared GaMD state
    """
    out_dir = Path(out_dir)
    primary = equilibration_state_xml_path(out_dir)
    if primary.exists():
        return primary
    candidate = out_dir / "adaptive_production" / "global_shared_gamd_setup" / "shared_gamd_setup_state.xml"
    if candidate.exists():
        return candidate
    epoch_states = sorted(out_dir.glob("adaptive_production/epoch_*/baseline/shared_gamd_setup_state.xml"))
    if epoch_states:
        return epoch_states[-1]
    feedback_states = sorted(out_dir.glob("adaptive_feedback_round_*/shared_gamd_setup_state.xml"))
    if feedback_states:
        return feedback_states[-1]
    return None


def production_checkpoint_available(out_dir: Path) -> bool:
    manifest = read_json_file(checkpoint_manifest_path(out_dir), None)
    if not isinstance(manifest, dict):
        return False
    files = manifest.get("replica_checkpoint_files", [])
    if not files:
        return False
    chk_dir = checkpoint_manifest_path(out_dir).parent
    return all((chk_dir / str(f)).exists() for f in files)


def load_existing_openmm_setup_for_resume(args, out_dir: Path, require_equil_state: bool = False):
    """Load topology/force field, and optionally the saved NPT state, without rerunning setup.

    Production checkpoint resume only needs a matching topology/system before
    Context.loadCheckpoint().  If the production checkpoint exists, the saved
    NPT State XML is optional.  If we only want to skip minimization/NVT/NPT and
    start a new production segment, the State XML is required because it carries
    velocities and box vectors.

    When require_equil_state is False and the primary 03_npt_equilibrated_state.xml
    is absent (e.g. runs pre-dating the save), fallback state paths are tried so
    that adaptive-production epoch segments get a valid starting configuration.
    """
    out_dir = Path(out_dir)
    solvated_pdb = out_dir / "01_solvated_start.pdb"
    if not solvated_pdb.exists():
        return None
    openmm, app, unit = import_openmm()
    forcefield = make_forcefield(app, args.water_model)
    pdb = app.PDBFile(str(solvated_pdb))
    state = None
    state_path = equilibration_state_xml_path(out_dir)
    if state_path.exists():
        try:
            state = openmm.XmlSerializer.deserialize(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            if require_equil_state:
                raise RuntimeError(f"Could not load saved equilibrated state {state_path}: {exc}") from exc
            print(f"WARNING: could not load {state_path}; will try fallback state paths: {exc}")
    elif require_equil_state:
        return None
    if state is None and not require_equil_state:
        fallback = find_resume_equil_state_path(out_dir)
        if fallback is not None and fallback != state_path:
            try:
                state = openmm.XmlSerializer.deserialize(fallback.read_text(encoding="utf-8"))
                print(f"[resume] Using fallback equilibration state: {fallback.relative_to(out_dir)}")
            except Exception as exc:
                print(f"WARNING: could not load fallback state {fallback}: {exc}")
    return openmm, app, unit, forcefield, pdb.topology, state
