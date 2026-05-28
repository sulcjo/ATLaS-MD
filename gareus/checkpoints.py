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
    "production_checkpoint_available",
    "load_existing_openmm_setup_for_resume",
]


def equilibration_state_xml_path(out_dir: Path) -> Path:
    return Path(out_dir) / "03_npt_equilibrated_state.xml"

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
            print(f"WARNING: could not load {state_path}; production checkpoint resume can still continue: {exc}")
    elif require_equil_state:
        return None
    return openmm, app, unit, forcefield, pdb.topology, state
