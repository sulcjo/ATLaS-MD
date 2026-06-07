#!/usr/bin/env python3
"""
rama_to_bh_twostage_seeds.py

Fast rough peptide energy-surface mapper:

sequence
  -> Ramachandran-state conformer generation
  -> broad contact/shape candidate selection
  -> AMBER14 + GBn2 implicit minimization of candidates
  -> optional fast implicit basin hopping from candidate minima
       (reuse one OpenMM implicit system/context per seed)
       short Langevin burst -> minimize -> store endpoint minima
  -> optional ANM/NMA local expansion
  -> combine all implicit-relaxed minima
  -> reject bad structures
  -> cluster relaxed minima by contact/shape to cover basins
  -> choose low-badness representative from each basin
  -> optional explicit-solvent minimization only on final survivors

This is not a Boltzmann sampler. It is a basin-discovery / seed-generation workflow.

Example GPU-ish if CUDA exists:
    python rama_to_bh_twostage_seeds.py \
      --seq ACDEFGHIKL \
      --n 10000 \
      --out ACDEFGHIKL_bh \
      --clean \
      --n-candidate-seeds 60 \
      --n-final-seeds 20 \
      --jobs 8 \
      --min-jobs 1 \
      --platform CUDA \
      --precision mixed \
      --two-stage \
      --basin-hop \
      --bh-steps 12 \
      --bh-md-steps 250 \
      --bh-temperature 600

CPU:
    python rama_to_bh_twostage_seeds.py \
      --seq ACDEFGHIKL \
      --n 10000 \
      --out ACDEFGHIKL_bh_cpu \
      --clean \
      --n-candidate-seeds 60 \
      --n-final-seeds 20 \
      --jobs 8 \
      --min-jobs 4 \
      --platform CPU \
      --cpu-threads 1 \
      --two-stage \
      --basin-hop \
      --bh-steps 8 \
      --bh-md-steps 150
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import signal
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Avoid accidental thread oversubscription when many Python/OpenMM worker
# processes are launched. Users can still override these in the shell.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# -----------------------------
# Lightweight YAML/JSON config support for combined GENPEPT + GAREUS configs
# -----------------------------

def _load_yaml_or_json_config(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                f"Config file {path} looks like YAML, but PyYAML is not installed. "
                "Install pyyaml or use a JSON config."
            ) from exc
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping/object at top level: {path}")
    return data

GENPEPT_CONFIG_BLOCKS = ("conformer_generation", "genpept", "seed_generation")

# Top-level sections that belong to the downstream GaREUS / GROMACS workflow.
# GENPEPT can live in the same YAML file as these blocks, but it should not try
# to flatten them into peptide-seed-generation arguments.
GAREUS_FRIENDLY_TOP_LEVEL_BLOCKS = {
    "schema_version", "description", "comments", "notes",
    "sequence", "cvs", "primary_cv", "contact_cv", "secondary_cv",
    "starting_structures", "output", "platform", "simulation", "windows",
    "gamd", "exchange", "tui", "analysis", "restraints", "pulling",
    "umbrella", "reus", "hmr", "solvent", "system", "files",
}

_CONFIG_COMPAT_MESSAGES: list[str] = []
_CONFIG_COMPAT_REPORT: dict = {}

def _normalize_config_key(key) -> str:
    key = str(key).strip().replace("-", "_")
    aliases = {
        "sequence": "seq",
        "peptide_sequence": "seq",
        "output": "out",
        "output_dir": "out",
        "candidates": "n_candidate_seeds",
        "candidate_seeds": "n_candidate_seeds",
        "final_seeds": "n_final_seeds",
        "clean_output": "clean",
        "platform_precision": "precision",
        "device": "device_index",
        # Common GaREUS-ish names when people reuse one combined YAML.
        "setup_platform": "platform",
        "setup_precision": "precision",
        "setup_device_index": "device_index",
        "setup_cpu_threads": "cpu_threads",
        "water_model": "water",
        "padding_nm": "padding",
        "ionic_strength_molar": "ionic_strength",
        "contact_min_sequence_separation": "contact_min_sep",
    }
    return aliases.get(key, key)

def _flatten_genpept_config(config: dict, known_dests: set[str], path: str = "") -> tuple[dict, list[str]]:
    flat = {}
    unknown = []
    for raw_key, value in (config or {}).items():
        key = _normalize_config_key(raw_key)
        here = f"{path}.{raw_key}" if path else str(raw_key)
        if key in {"schema_version", "comments", "notes", "description"}:
            continue
        if key in known_dests and not isinstance(value, dict):
            flat[key] = value
        elif isinstance(value, dict):
            sub_flat, sub_unknown = _flatten_genpept_config(value, known_dests, here)
            flat.update(sub_flat)
            unknown.extend(sub_unknown)
        elif key in known_dests:
            flat[key] = value
        else:
            unknown.append(here)
    return flat, unknown

def _path_arg_dests(parser: argparse.ArgumentParser) -> set[str]:
    return {str(a.dest) for a in parser._actions if getattr(a, "type", None) is Path and getattr(a, "dest", None)}

def _set_if_missing(flat: dict, dest: str, value, source: str, inherited: list[dict]):
    if value in (None, "") or dest in flat:
        return
    flat[dest] = value
    inherited.append({"dest": dest, "source": source, "value": str(value)})

def _nested_get(mapping: dict, *keys):
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

def _apply_gareus_friendly_hints(flat: dict, raw: dict) -> list[dict]:
    """Borrow safe GENPEPT defaults from a combined GaREUS YAML.

    These are only used when the explicit genpept/conformer_generation block did
    not already provide the corresponding option. This keeps GENPEPT in charge
    while making one-file GAREUS configs pleasant to use.
    """
    inherited: list[dict] = []

    seq_block = raw.get("sequence")
    if isinstance(seq_block, dict):
        _set_if_missing(flat, "seq", seq_block.get("seq") or seq_block.get("sequence") or seq_block.get("peptide_sequence"), "sequence.seq", inherited)
    elif isinstance(seq_block, str):
        _set_if_missing(flat, "seq", seq_block, "sequence", inherited)

    start_block = raw.get("starting_structures")
    if isinstance(start_block, dict):
        _set_if_missing(flat, "out", start_block.get("seed_conformers_dir"), "starting_structures.seed_conformers_dir", inherited)

    platform_block = raw.get("platform")
    if isinstance(platform_block, dict):
        _set_if_missing(flat, "platform", platform_block.get("setup_platform") or platform_block.get("platform"), "platform.setup_platform/platform", inherited)
        _set_if_missing(flat, "precision", platform_block.get("setup_precision") or platform_block.get("precision"), "platform.setup_precision/precision", inherited)
        _set_if_missing(flat, "device_index", platform_block.get("setup_device_index") or platform_block.get("device_index"), "platform.setup_device_index/device_index", inherited)
        # Prefer the generic CPU thread hint over setup_cpu_threads; setup_cpu_threads
        # often belongs to a different preparation stage and can oversubscribe GENPEPT.
        _set_if_missing(flat, "cpu_threads", platform_block.get("cpu_threads"), "platform.cpu_threads", inherited)

    sim_block = raw.get("simulation")
    if isinstance(sim_block, dict):
        _set_if_missing(flat, "water", sim_block.get("water_model"), "simulation.water_model", inherited)
        _set_if_missing(flat, "padding", sim_block.get("padding_nm"), "simulation.padding_nm", inherited)
        _set_if_missing(flat, "ionic_strength", sim_block.get("ionic_strength_molar"), "simulation.ionic_strength_molar", inherited)

    contact_block = raw.get("contact_cv")
    if isinstance(contact_block, dict):
        _set_if_missing(flat, "contact_min_sep", contact_block.get("contact_min_sequence_separation"), "contact_cv.contact_min_sequence_separation", inherited)

    return inherited

def _genpept_config_defaults(config_path, parser: argparse.ArgumentParser, strict: bool = False) -> tuple[dict, list[str]]:
    global _CONFIG_COMPAT_MESSAGES, _CONFIG_COMPAT_REPORT
    _CONFIG_COMPAT_MESSAGES = []
    _CONFIG_COMPAT_REPORT = {}
    if not config_path:
        return {}, []

    raw = _load_yaml_or_json_config(Path(config_path))
    known_dests = {str(a.dest) for a in parser._actions if getattr(a, "dest", None) and a.dest != argparse.SUPPRESS}

    selected_block_name = None
    body = None
    for key in GENPEPT_CONFIG_BLOCKS:
        value = raw.get(key)
        if isinstance(value, dict):
            selected_block_name = key
            body = value
            break

    unknown_top_level: list[str] = []
    if body is None:
        # Legacy mode: no explicit GENPEPT section. Flatten only the parts that
        # are not clearly downstream GaREUS blocks, so a full combined YAML no
        # longer explodes into dozens of irrelevant unknown CV/window keys.
        selected_block_name = "legacy_top_level_filtered"
        body = {}
        for raw_key, value in raw.items():
            norm = _normalize_config_key(raw_key)
            if raw_key in GAREUS_FRIENDLY_TOP_LEVEL_BLOCKS or norm in GAREUS_FRIENDLY_TOP_LEVEL_BLOCKS:
                continue
            body[raw_key] = value
        unknown_top_level = [str(k) for k in raw.keys() if str(k) not in GAREUS_FRIENDLY_TOP_LEVEL_BLOCKS and str(k) not in GENPEPT_CONFIG_BLOCKS and str(k) not in body]
        _CONFIG_COMPAT_MESSAGES.append(
            "No top-level genpept/conformer_generation/seed_generation block found; "
            "using legacy filtered top-level mode and ignoring recognized GaREUS sections."
        )
    else:
        ignored = [str(k) for k in raw.keys() if k not in GENPEPT_CONFIG_BLOCKS and k not in {selected_block_name}]
        ignored_gareus = [k for k in ignored if k in GAREUS_FRIENDLY_TOP_LEVEL_BLOCKS]
        if ignored_gareus:
            _CONFIG_COMPAT_MESSAGES.append(
                f"Using top-level {selected_block_name!r} block for GENPEPT; ignored GaREUS sections: "
                + ", ".join(ignored_gareus[:12])
                + (" ..." if len(ignored_gareus) > 12 else "")
            )

    flat, unknown = _flatten_genpept_config(body, known_dests)
    inherited = _apply_gareus_friendly_hints(flat, raw)
    if inherited:
        _CONFIG_COMPAT_MESSAGES.append(
            "Inherited missing GENPEPT defaults from combined YAML: "
            + ", ".join(f"{x['dest']}<-{x['source']}" for x in inherited)
        )

    _config_parent = Path(config_path).parent.resolve() if config_path else None
    for dest in list(flat):
        if dest in _path_arg_dests(parser) and flat[dest] is not None:
            p_val = Path(str(flat[dest]))
            if not p_val.is_absolute() and _config_parent is not None:
                p_val = (_config_parent / p_val).resolve()
            flat[dest] = p_val

    ignored_unknown = [] if strict else list(unknown)
    if ignored_unknown:
        _CONFIG_COMPAT_MESSAGES.append(
            "Ignored unknown keys inside the GENPEPT config block: "
            + ", ".join(ignored_unknown[:20])
            + (f" ... and {len(ignored_unknown) - 20} more" if len(ignored_unknown) > 20 else "")
        )
        unknown = []

    _CONFIG_COMPAT_REPORT = {
        "config_path": str(config_path),
        "selected_block": selected_block_name,
        "strict_config": bool(strict),
        "inherited_defaults": inherited,
        "unknown_keys": list(unknown),
        "ignored_unknown_keys": ignored_unknown,
        "messages": list(_CONFIG_COMPAT_MESSAGES),
    }
    return flat, unknown
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

# Keep worker output readable. Biopython can emit noisy environment warnings,
# especially under bleeding-edge Python installations.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="Bio.PDB.vectors")
warnings.filterwarnings("ignore", message=".*You may be importing Biopython from inside the source tree.*")
warnings.filterwarnings("ignore", module="Bio")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Lightweight caches for repeated geometry work.
_PDB_GEOM_CACHE: dict[tuple[str, int, int, float, int], tuple[np.ndarray, int, float, float]] = {}
_PAIR_INDEX_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
_CLASH_PAIR_INDEX_CACHE: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}
_FEATURE_CACHE_MAP: dict[tuple[str, int, int, float, int], tuple[np.ndarray, int, float, float]] = {}
_FEATURE_CACHE_LOADED_FOR: Optional[str] = None
_FEATURE_CACHE_DIRTY = False
_ATOM_LAYOUT_CACHE = {}

AA3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

STATE_CENTERS = {
    "A": (-60.0, -45.0),
    "B": (-120.0, 130.0),
    "P": (-75.0, 145.0),
    "L": (60.0, 40.0),
    "T": (-80.0, 0.0),
}

BACKBONE_BOND_N_CA = 1.458
BACKBONE_BOND_CA_C = 1.525
BACKBONE_BOND_C_N = 1.329
BACKBONE_ANGLE_C_N_CA = 121.7
BACKBONE_ANGLE_N_CA_C = 111.2
BACKBONE_ANGLE_CA_C_N = 116.2
DEFAULT_TERMINAL_PHI = -60.0
DEFAULT_TERMINAL_PSI = -45.0
DEFAULT_OMEGA = 180.0

RAMA_STATES = ("A", "B", "P", "L", "T")
GENERAL_WEIGHTS = {"B": 0.35, "P": 0.25, "A": 0.25, "T": 0.10, "L": 0.05}
GLY_WEIGHTS     = {"B": 0.30, "P": 0.20, "A": 0.20, "L": 0.20, "T": 0.10}
PRO_WEIGHTS     = {"P": 0.45, "B": 0.30, "A": 0.20, "T": 0.05, "L": 0.00}

FF_CHOICES = {
    "amber14": {
        "protein": "amber14-all.xml",
        "implicit": "implicit/gbn2.xml",
        "waters": {
            "tip3p": "amber14/tip3p.xml",
            "tip3pfb": "amber14/tip3pfb.xml",
            "spce": "amber14/spce.xml",
            "tip4pew": "amber14/tip4pew.xml",
        },
    },
    "amber99sb": {
        "protein": "amber99sbildn.xml",
        "implicit": "implicit/obc2.xml",
        "waters": {"tip3p": "tip3p.xml"},
    },
}


@dataclass
class GenConfig:
    seq: str
    n_requested: int
    angle_sd_deg: float
    clash_cutoff_A: float
    contact_cutoff_A: float
    contact_min_sep: int
    max_clashes: int
    seed: int
    write_pdbs: bool = True
    rama_sampling: str = "stratified"
    generation_backend: str = "fast"
    diversity_bank_preset: str = "off"
    diversity_bank_wide_angle_sd: float = 45.0


@dataclass
class ConformerRecord:
    conformer_id: int
    pdb_path: str
    state_string: str
    phi_degrees: str
    psi_degrees: str
    rg_A: float
    end_to_end_A: float
    contact_count: int
    clash_count: int
    bank_name: str = "default"


@dataclass
class AtomLayout:
    n_atoms: int
    atom_fields: tuple[str, ...]
    resnames: tuple[str, ...]
    chains: tuple[str, ...]
    resseq: np.ndarray
    elements: tuple[str, ...]
    ca_indices: np.ndarray
    clash_i: np.ndarray
    clash_j: np.ndarray


@dataclass
class MinConfig:
    mode: str
    ff: str
    water: str
    padding_nm: float
    ionic_strength_molar: float
    neutralize: bool
    ph: float
    temperature_k: float
    friction_per_ps: float
    nonbonded_cutoff_nm: float
    max_iterations: int
    restraint_k_kj_mol_nm2: float
    platform: str
    precision: str
    device_index: str
    cpu_threads: int
    save_systems: bool
    constraints: str
    contact_cutoff_A: float
    contact_min_sep: int
    adaptive_min: bool = True
    adaptive_min_chunk_iterations: int = 25
    adaptive_min_energy_tol_kj_mol: float = 0.05
    adaptive_min_force_tol_kj_mol_nm: float = 500.0
    adaptive_min_stall_rounds: int = 2
    # 1 preserves the original behavior: fetch forces after every minimization
    # chunk. 0 enables energy-only early stopping and retrieves forces once at
    # the end; values >1 check forces periodically. This is useful for many
    # tiny implicit minimizations where force readback overhead is visible.
    adaptive_min_force_check_every: int = 1
    # Candidate/minimized seed batches generated by this script normally share
    # identical topology. Enabling this skips repeated topology-signature walks
    # after the first context build.
    assume_same_implicit_topology: bool = False


@dataclass
class MinResult:
    seed_name: str
    input_pdb: str
    output_pdb: str
    success: bool
    error: str
    mode: str
    n_atoms: int
    n_residues: int
    n_waters: int
    n_ions: int
    initial_energy_kj_mol: float
    minimized_energy_kj_mol: float
    energy_drop_kj_mol: float
    max_force_kj_mol_nm: float
    rg_nm: float
    end_to_end_nm: float
    contact_count: int
    minimization_iterations_used: int = 0
    minimization_rounds: int = 0
    minimization_stop_reason: str = ""


# -----------------------------
# Imports / helpers
# -----------------------------

def validate_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    bad = sorted(set(seq) - set(AA3))
    if not seq:
        raise ValueError("Empty sequence.")
    if bad:
        raise ValueError(f"Unsupported residues: {bad}. Only canonical one-letter amino acids are supported.")
    if len(seq) < 2:
        raise ValueError("Need at least 2 residues.")
    return seq


def require_generation_imports():
    try:
        import PeptideBuilder
        from Bio.PDB import Polypeptide
        return PeptideBuilder, Polypeptide
    except Exception as exc:
        raise RuntimeError("Generation requires PeptideBuilder and Biopython.") from exc


def import_openmm():
    try:
        import openmm
        import openmm.app as app
        import openmm.unit as unit
        from openmm import XmlSerializer
        return openmm, app, unit, XmlSerializer
    except Exception as exc:
        raise RuntimeError("Minimization/basin hopping requires OpenMM.") from exc


def get_min_jobs(args):
    requested = int(getattr(args, "min_jobs", 0) or 0)
    if requested > 0:
        return requested
    # GPU/OpenCL contexts are expensive and usually contend with each other.
    # If the user leaves --min-jobs at auto/0, keep one OpenMM worker on
    # accelerator platforms and still allow explicit override for multi-GPU runs.
    platform = str(getattr(args, "platform", "") or "").strip().lower()
    if platform in {"cuda", "opencl"}:
        return 1
    return int(getattr(args, "jobs", 1) or 1)


def available_openmm_platforms(openmm) -> list[str]:
    return [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]


def resolve_openmm_platform_name(openmm, requested: str | None):
    requested = str(requested or "").strip()
    if not requested or requested.lower() == "auto":
        return None
    available = available_openmm_platforms(openmm)
    for name in available:
        if name.lower() == requested.lower():
            return name
    raise ValueError(
        f"Requested OpenMM platform {requested!r} is not available. "
        f"Available platforms: {', '.join(available)}. Use --platform auto or one of the available names."
    )


def write_dict_csv(path: Path, fieldnames: list[str], rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


UI_CLEAR_SCREEN = True
UI_COLOR = True
UI_STAGE = ""
UI_STAGE_ICON = "🧬"
UI_PHASE_KEY = "start"
UI_CYCLE_STEP = None
UI_CYCLE_ROUND = None
UI_CYCLE_TOTAL = None
UI_PROGRESS = ""
UI_MESSAGES = []
UI_PANELS = {}
UI_START_TIME = time.time()
UI_MAX_MESSAGES = 8

PHASE_ICONS = {
    "Peptide basin-seed workflow": "🧬",
    "1) Ramachandran conformer generation": "🎲",
    "2) Broad candidate selection": "🧭",
    "OpenMM implicit minimization": "⚡",
    "OpenMM explicit minimization": "💧",
    "Implicit basin hopping": "🐇",
    "ANM/NMA local expansion": "〰️",
    "Adaptive PCA frontier map": "🗺️",
    "Final basin selection": "🏁",
    "done": "✅",
}

WORKFLOW_PATH = [
    ("start", "Start"),
    ("generation", "Conformers"),
    ("selection", "Select"),
    ("implicit_min", "Implicit min"),
    ("exploration", "Explore loop"),
    ("final_selection", "Final select"),
    ("explicit_min", "Explicit min"),
    ("done", "Done"),
]
WORKFLOW_INDEX = {key: i for i, (key, _label) in enumerate(WORKFLOW_PATH)}

EXPLORE_CYCLE = [
    ("archive", "archive/score"),
    ("bh", "BH/MC"),
    ("nma", "NMA"),
    ("map", "PCA map"),
    ("propose", "proposals"),
    ("minimize", "minimize hits"),
]
EXPLORE_CYCLE_INDEX = {key: i for i, (key, _label) in enumerate(EXPLORE_CYCLE)}

COLOR = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "pink": "\033[95m",
    "orange": "\033[38;5;208m",
}


def c(text, color=None, bold=False, dim=False):
    if not UI_COLOR:
        return str(text)
    s = ""
    if bold:
        s += COLOR["bold"]
    if dim:
        s += COLOR["dim"]
    if color:
        s += COLOR.get(color, "")
    return s + str(text) + COLOR["reset"]


def infer_ui_phase(title: str) -> tuple[str, Optional[str]]:
    """Best-effort stage-title -> linear/cycle position for the dashboard."""
    text = str(title)
    low = text.lower()
    if text.startswith("1) ") or "ramachandran" in low or "conformer generation" in low:
        return "generation", None
    if text.startswith("2) ") or "candidate selection" in low:
        return "selection", None
    if "explicit minimization" in low:
        return "explicit_min", None
    if "implicit minimization" in low:
        return "implicit_min", None
    if "basin hopping" in low or "basin-hop" in low or "bh/mc" in low or "monte carlo" in low:
        return "exploration", "bh"
    if "nma" in low or "anm" in low:
        return "exploration", "nma"
    if "adaptive pca frontier map" in low or "frontier map" in low:
        return "exploration", "map"
    if "adaptive pca exploration" in low or "pca explore" in low or "proposal" in low:
        return "exploration", "propose"
    if "archive" in low:
        return "exploration", "archive"
    if "final basin selection" in low:
        return "final_selection", None
    if low.strip() in {"done", "complete", "completed"}:
        return "done", None
    if "peptide basin-seed workflow" in low:
        return "start", None
    return UI_PHASE_KEY or "start", UI_CYCLE_STEP


def phase_icon_for(title: str, phase_key: Optional[str] = None, cycle_step: Optional[str] = None) -> str:
    if str(title) in PHASE_ICONS:
        return PHASE_ICONS[str(title)]
    if phase_key == "generation":
        return "🎲"
    if phase_key == "selection":
        return "🧭"
    if phase_key == "implicit_min":
        return "⚡"
    if phase_key == "exploration":
        return {
            "archive": "📦",
            "bh": "🐇",
            "nma": "〰️",
            "map": "🗺️",
            "propose": "🎯",
            "minimize": "⚡",
        }.get(str(cycle_step), "🔁")
    if phase_key == "final_selection":
        return "🏁"
    if phase_key == "explicit_min":
        return "💧"
    if phase_key == "done":
        return "✅"
    return "🧪"


def _workflow_token(key: str, label: str) -> str:
    current_idx = WORKFLOW_INDEX.get(UI_PHASE_KEY, 0)
    idx = WORKFLOW_INDEX.get(key, 0)
    if idx < current_idx:
        return c(f"● {label}", "green", bold=True)
    if idx == current_idx:
        marker = "↻" if key == "exploration" else "⬤"
        return c(f"{marker} {label}", "red", bold=True)
    return c(f"○ {label}", "dim")


def _cycle_token(key: str, label: str) -> str:
    if key == UI_CYCLE_STEP:
        return c(f"⬤ {label}", "red", bold=True)
    return c(f"○ {label}", "dim")


def _render_workflow_path() -> str:
    path = " ─▶ ".join(_workflow_token(key, label) for key, label in WORKFLOW_PATH)
    lines = [c("workflow", "yellow", bold=True), "  " + path]
    if UI_PHASE_KEY == "exploration" or UI_CYCLE_STEP is not None:
        round_text = ""
        if UI_CYCLE_ROUND is not None:
            if UI_CYCLE_TOTAL is not None and int(UI_CYCLE_TOTAL) > 0:
                round_text = f"  {c(f'round {UI_CYCLE_ROUND}/{UI_CYCLE_TOTAL}', 'cyan', bold=True)}"
            else:
                round_text = f"  {c(f'round {UI_CYCLE_ROUND}', 'cyan', bold=True)}"
        cycle = " → ".join(_cycle_token(key, label) for key, label in EXPLORE_CYCLE) + " ↺"
        lines.extend([c("exploration cycle", "magenta", bold=True) + round_text, "  " + cycle])
    return "\n".join(lines) + "\n"


def fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:4.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:02d}:{sec:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{sec:02d}"


def progress_bar(done: int, total: int, width: int = 46) -> str:
    if total <= 0:
        frac = 1.0
    else:
        frac = min(1.0, max(0.0, done / total))
    fill = int(round(width * frac))
    bar = "█" * fill + "░" * (width - fill)
    color = "red" if frac < 0.33 else "orange" if frac < 0.70 else "green"
    return c("[" + bar + "]", color, bold=True)


def _clear_screen():
    if UI_CLEAR_SCREEN:
        sys.stdout.write("\033[2J\033[H")


def ui_message(msg: str):
    if msg is None:
        return
    UI_MESSAGES.append(str(msg))
    del UI_MESSAGES[:-UI_MAX_MESSAGES]


def _box_line(width=104):
    return c("═" * width, "cyan", bold=True)


def render_dashboard():
    _clear_screen()
    width = 104
    elapsed = fmt_seconds(time.time() - UI_START_TIME)
    title = f"{UI_STAGE_ICON}  {UI_STAGE or 'Peptide basin-seed workflow'}"
    sys.stdout.write(_box_line(width) + "\n")
    sys.stdout.write(c(title, "cyan", bold=True) + c(f"   elapsed {elapsed}", "dim") + "\n")
    sys.stdout.write(_box_line(width) + "\n")
    sys.stdout.write(_render_workflow_path())
    if UI_PROGRESS:
        sys.stdout.write(UI_PROGRESS + "\n")
    if UI_MESSAGES:
        sys.stdout.write("\n" + c("status", "yellow", bold=True) + "\n")
        for msg in UI_MESSAGES[-UI_MAX_MESSAGES:]:
            sys.stdout.write(f"  {c('›', 'green', bold=True)} {msg}\n")
    if UI_PANELS:
        sys.stdout.write("\n" + c("live distributions", "magenta", bold=True) + "\n")
        for _name, lines in UI_PANELS.items():
            for line in lines:
                sys.stdout.write(line + "\n")
    sys.stdout.write("\n" + c("Press Ctrl+C to stop workers and exit.", "dim") + "\n")
    sys.stdout.flush()


class CliProgress:
    def __init__(self, title: str, total: int, update_every: float = 0.35):
        self.title = title
        self.total = int(total)
        self.start = time.time()
        self.last = 0.0
        self.update_every = update_every

    def update(self, done: int, **stats):
        global UI_PROGRESS
        now = time.time()
        if done < self.total and (now - self.last) < self.update_every:
            return
        self.last = now
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        pct = 100.0 * (done / self.total) if self.total else 100.0
        stat_text = "  ".join(c(f"{k}=", "dim") + c(v, "white", bold=True) for k, v in stats.items())
        line = (
            f"{c(self.title, 'white', bold=True):<35} "
            f"{progress_bar(done, self.total)} "
            f"{c(f'{done:>6}/{self.total:<6}', 'cyan', bold=True)} "
            f"{c(f'{pct:5.1f}%', 'green', bold=True)} "
            f"{c('rate=', 'dim')}{rate:7.2f}/s "
            f"{c('eta=', 'dim')}{fmt_seconds(eta)}"
        )
        if stat_text:
            line += "  " + stat_text
        UI_PROGRESS = line
        render_dashboard()

    def done(self, done: Optional[int] = None, **stats):
        if done is None:
            done = self.total
        self.update(done, **stats)


def _spark_hist(values, bins=42):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return c("[no data]", "dim")
    lo = float(np.nanpercentile(arr, 2))
    hi = float(np.nanpercentile(arr, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if lo == hi:
            return c(f"[all {lo:.3g}]", "green")
    hist, _edges = np.histogram(arr, bins=bins, range=(lo, hi))
    maxh = max(1, int(hist.max()))
    chars = " ▁▂▃▄▅▆▇█"
    bars = "".join(chars[int(round((len(chars) - 1) * h / maxh))] for h in hist)
    return f"{c(f'{lo:.3g}', 'dim')} {c(bars, 'green', bold=True)} {c(f'{hi:.3g}', 'dim')}"


def _ascii_pca_density(scores, width: int = 42, height: int = 14, overlay_scores=None):
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return [c("  [no PCA points yet]", "dim")]
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.shape[0] == 0:
        return [c("  [no PCA points yet]", "dim")]

    width = max(12, int(width))
    height = max(6, int(height))
    x = arr[:, 0]
    y = arr[:, 1]
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin, xmax = xmin - 1.0, xmax + 1.0
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        ymin, ymax = ymin - 1.0, ymax + 1.0
    pad_x = max(1e-6, 0.05 * (xmax - xmin))
    pad_y = max(1e-6, 0.05 * (ymax - ymin))
    xr = (xmin - pad_x, xmax + pad_x)
    yr = (ymin - pad_y, ymax + pad_y)
    hist, _xedges, _yedges = np.histogram2d(x, y, bins=[width, height], range=[xr, yr])

    overlay_hist = None
    if overlay_scores is not None:
        over = np.asarray(overlay_scores, dtype=float)
        if over.ndim == 2 and over.shape[0] > 0 and over.shape[1] >= 2:
            over = over[np.isfinite(over).all(axis=1)]
            if over.shape[0] > 0:
                overlay_hist, _, _ = np.histogram2d(over[:, 0], over[:, 1], bins=[width, height], range=[xr, yr])

    vmax = float(np.nanmax(hist)) if hist.size else 0.0
    chars = " .:-=+*#%@"
    lines = []
    for j in range(height - 1, -1, -1):
        row = []
        for i in range(width):
            hv = float(hist[i, j])
            ov = float(overlay_hist[i, j]) if overlay_hist is not None else 0.0
            if ov > 0:
                row.append(c("@", "red", bold=True))
            elif hv <= 0 or vmax <= 0:
                row.append(c("·", "dim"))
            else:
                idx = min(len(chars) - 1, max(1, int(round((len(chars) - 1) * hv / vmax))))
                row.append(c(chars[idx], "green", bold=True))
        lines.append("  " + "".join(row))
    occupied = int(np.count_nonzero(hist))
    total_bins = int(hist.size) if hist.size else 0
    lines.append(
        f"  {c('PC1', 'cyan', bold=True)} {xmin:6.2f}..{xmax:6.2f}   "
        f"{c('PC2', 'cyan', bold=True)} {ymin:6.2f}..{ymax:6.2f}   "
        f"{c('bins', 'dim')}={occupied}/{total_bins}"
    )
    legend = "  " + c("legend:", "dim") + " " + c("density", "green", bold=True)
    if overlay_hist is not None:
        legend += "  " + c("@ new/round hits", "red", bold=True)
    lines.append(legend)
    return lines


def set_cli_pca_panel(title: str, scores, overlay_scores=None, subtitle: Optional[str] = None, width: int = 42, height: int = 14):
    try:
        arr = np.asarray(scores, dtype=float)
    except Exception:
        return
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return
    lines = [f"  {c(title, 'pink', bold=True)}"]
    if subtitle:
        lines.append(f"    {c(subtitle, 'dim')}")
    lines.extend(_ascii_pca_density(arr[:, :2], width=width, height=height, overlay_scores=overlay_scores))
    UI_PANELS[title] = lines


class LiveStats:
    def __init__(self, title: str, enabled: bool = True, every: int = 1000):
        self.title = title
        self.enabled = bool(enabled)
        self.every = max(1, int(every))
        self.values = {}
        self.last_n = 0

    def add(self, **kwargs):
        if not self.enabled:
            return
        for k, v in kwargs.items():
            try:
                x = float(v)
            except Exception:
                continue
            if np.isfinite(x):
                self.values.setdefault(k, []).append(x)

    def maybe_print(self, n: int, force: bool = False):
        if not self.enabled:
            return
        if not force and (n - self.last_n) < self.every:
            return
        self.last_n = n
        if not self.values:
            return
        lines = [f"  {c(self.title, 'pink', bold=True)}"]
        for key, vals in self.values.items():
            arr = np.asarray(vals, dtype=float)
            if len(arr) == 0:
                continue
            lines.append(
                f"    {c(key, 'cyan'):<26} "
                f"{c('n=', 'dim')}{len(arr):<6d} "
                f"{c('med=', 'dim')}{np.nanmedian(arr):8.3g} "
                f"{c('mean=', 'dim')}{np.nanmean(arr):8.3g}  "
                f"{_spark_hist(arr, bins=42)}"
            )
        UI_PANELS[self.title] = lines
        render_dashboard()


def print_stage(
    title: str,
    phase_key: Optional[str] = None,
    cycle_step: Optional[str] = None,
    cycle_round: Optional[int] = None,
    cycle_total: Optional[int] = None,
):
    global UI_STAGE, UI_STAGE_ICON, UI_PHASE_KEY, UI_CYCLE_STEP, UI_CYCLE_ROUND, UI_CYCLE_TOTAL
    global UI_PROGRESS, UI_MESSAGES, UI_PANELS

    sticky_panels = {}
    for key in ("CLI PCA space",):
        if key in UI_PANELS:
            sticky_panels[key] = UI_PANELS[key]

    inferred_phase, inferred_cycle = infer_ui_phase(str(title))
    next_phase = phase_key or inferred_phase
    UI_PHASE_KEY = next_phase
    UI_CYCLE_STEP = cycle_step if cycle_step is not None else inferred_cycle
    UI_CYCLE_ROUND = cycle_round
    UI_CYCLE_TOTAL = cycle_total
    UI_STAGE = str(title)
    UI_STAGE_ICON = phase_icon_for(str(title), UI_PHASE_KEY, UI_CYCLE_STEP)
    UI_PROGRESS = ""
    UI_MESSAGES = []
    UI_PANELS = {}
    if next_phase == "exploration":
        UI_PANELS.update(sticky_panels)
    render_dashboard()


def shutdown_process_pool_now(ex):
    """Best-effort fast ProcessPoolExecutor shutdown after Ctrl+C.

    The default context-manager shutdown waits for workers, which can turn one
    Ctrl+C into a small traceback festival. Python 3.14 adds terminate_workers
    and kill_workers; older versions at least support cancel_futures.
    """
    if ex is None:
        return
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        try:
            ex.shutdown(wait=False)
        except Exception:
            pass
    except Exception:
        pass
    for method in ("terminate_workers", "kill_workers"):
        fn = getattr(ex, method, None)
        if fn is None:
            continue
        try:
            fn()
            break
        except Exception:
            pass


def cancel_futures_now(futures):
    for fut in futures or []:
        try:
            fut.cancel()
        except Exception:
            pass


def _worker_ignore_sigint():
    """Let the parent process own Ctrl+C/SIGINT handling for worker pools."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def make_process_pool(max_workers: int):
    return ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_ignore_sigint)


def worker_failure_traceback(exc: BaseException) -> str:
    """Format worker failures, including BaseException subclasses."""
    try:
        return traceback.format_exc()
    except Exception:
        return f"{type(exc).__name__}: {exc}"


# -----------------------------
# Ramachandran generation
# -----------------------------

def residue_state_weights(residue: str, bank: Optional[dict] = None) -> dict[str, float]:
    """Return Ramachandran-state weights, optionally overridden by a diversity bank."""
    residue = str(residue).upper()
    if bank:
        if residue == "G" and isinstance(bank.get("gly_weights"), dict):
            return bank["gly_weights"]
        if residue == "P" and isinstance(bank.get("pro_weights"), dict):
            return bank["pro_weights"]
        if isinstance(bank.get("weights"), dict):
            return bank["weights"]
    return GLY_WEIGHTS if residue == "G" else PRO_WEIGHTS if residue == "P" else GENERAL_WEIGHTS


def state_distribution(residue: str, bank: Optional[dict] = None):
    weights = residue_state_weights(residue, bank=bank)
    states = [s for s in RAMA_STATES if float(weights.get(s, 0.0)) > 0.0]
    if not states:
        weights = residue_state_weights(residue, bank=None)
        states = [s for s in RAMA_STATES if float(weights.get(s, 0.0)) > 0.0]
    probs = np.asarray([float(weights.get(s, 0.0)) for s in states], dtype=float)
    total = float(probs.sum())
    if total <= 0.0 or not np.isfinite(total):
        return state_distribution(residue, bank=None)
    probs /= total
    return states, probs


def state_from_unit(residue: str, u: float, bank: Optional[dict] = None) -> str:
    states, probs = state_distribution(residue, bank=bank)
    cdf = np.cumsum(probs)
    idx = int(np.searchsorted(cdf, float(u) % 1.0, side="right"))
    idx = max(0, min(idx, len(states) - 1))
    return str(states[idx])


def choose_state(residue: str, rng: np.random.Generator, bank: Optional[dict] = None) -> str:
    states, probs = state_distribution(residue, bank=bank)
    return str(rng.choice(states, p=probs))


def diversity_bank_presets(wide_angle_sd: float = 45.0) -> dict[str, list[dict]]:
    """Cheap banked torsion priors for faster coarse coverage of backbone classes."""
    wide_angle_sd = float(wide_angle_sd)
    broad = [
        {"name": "beta_extended", "fraction": 0.30, "weights": {"B": 0.55, "P": 0.25, "A": 0.10, "T": 0.10, "L": 0.00}},
        {"name": "alpha_compact", "fraction": 0.25, "weights": {"A": 0.65, "T": 0.15, "B": 0.10, "P": 0.05, "L": 0.05}},
        {"name": "mixed_turns", "fraction": 0.25, "weights": {"T": 0.35, "P": 0.25, "B": 0.20, "A": 0.10, "L": 0.10}},
        {"name": "gly_left", "fraction": 0.10, "gly_weights": {"L": 0.45, "B": 0.25, "A": 0.15, "P": 0.10, "T": 0.05}},
        {"name": "wide_random", "fraction": 0.10, "angle_sd_deg": wide_angle_sd, "weights": {"B": 0.25, "P": 0.25, "A": 0.25, "T": 0.15, "L": 0.10}},
    ]
    chignolin = [
        {"name": "hairpin_beta", "fraction": 0.35, "weights": {"B": 0.60, "P": 0.25, "T": 0.10, "A": 0.05, "L": 0.00}},
        {"name": "turn_rich", "fraction": 0.30, "weights": {"T": 0.40, "P": 0.25, "B": 0.20, "A": 0.10, "L": 0.05}},
        {"name": "compact_alpha_turn", "fraction": 0.15, "weights": {"A": 0.50, "T": 0.25, "B": 0.10, "P": 0.10, "L": 0.05}},
        {"name": "gly_left_turn", "fraction": 0.10, "gly_weights": {"L": 0.55, "T": 0.15, "B": 0.15, "A": 0.10, "P": 0.05}},
        {"name": "wide_random", "fraction": 0.10, "angle_sd_deg": wide_angle_sd, "weights": {"B": 0.25, "P": 0.25, "A": 0.20, "T": 0.20, "L": 0.10}},
    ]
    return {"broad": broad, "turbo": broad, "chignolin": chignolin}


def normalize_diversity_banks(banks: list[dict]) -> list[dict]:
    cleaned = []
    for i, bank in enumerate(banks or []):
        if not isinstance(bank, dict):
            continue
        b = dict(bank)
        b["name"] = str(b.get("name") or f"bank_{i + 1}")
        b["fraction"] = max(0.0, float(b.get("fraction", 0.0)))
        cleaned.append(b)
    total = sum(float(b.get("fraction", 0.0)) for b in cleaned)
    if total <= 0.0:
        return []
    for b in cleaned:
        b["fraction"] = float(b.get("fraction", 0.0)) / total
    return cleaned


def select_diversity_bank(cfg: GenConfig, conformer_idx: int) -> Optional[dict]:
    preset = str(getattr(cfg, "diversity_bank_preset", "off") or "off").strip().lower()
    if preset in {"", "off", "none", "false", "no"}:
        return None
    banks = normalize_diversity_banks(diversity_bank_presets(getattr(cfg, "diversity_bank_wide_angle_sd", 45.0)).get(preset, []))
    if not banks:
        return None
    u = stratified_unit(int(conformer_idx), 9973, int(cfg.seed), int(cfg.n_requested))
    cumulative = 0.0
    for bank in banks:
        cumulative += float(bank.get("fraction", 0.0))
        if u <= cumulative:
            return bank
    return banks[-1]


def _splitmix64(value: int) -> int:
    x = (int(value) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (x ^ (x >> 31)) & 0xFFFFFFFFFFFFFFFF


def _splitmix64_unit(value: int) -> float:
    return ((_splitmix64(value) >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def stratified_unit(conformer_idx: int, dimension: int, base_seed: int, n_requested: int) -> float:
    """Latin-hypercube-like unit value for one torsion/state dimension.

    Each dimension gets a deterministic, seed-scrambled permutation of the
    conformer indices, so the full requested batch covers the unit interval
    much more evenly than independent random draws while remaining reproducible
    under chunked multiprocessing.
    """
    n = max(1, int(n_requested))
    if n == 1:
        return _splitmix64_unit(int(base_seed) + 104729 * (int(dimension) + 1))

    dim_key = _splitmix64(int(base_seed) + 104729 * (int(dimension) + 1))
    stride = int(dim_key % n)
    if stride == 0:
        stride = 1
    while math.gcd(stride, n) != 1:
        stride += 1
        if stride >= n:
            stride = 1
    offset = int(_splitmix64(dim_key + 0xD1B54A32D192ED03) % n)
    bin_idx = (int(conformer_idx) * stride + offset) % n
    jitter = _splitmix64_unit(dim_key + 0x94D049BB133111EB * (int(conformer_idx) + 1))
    return (float(bin_idx) + jitter) / float(n)


def normal_quantile(p: float) -> float:
    """Fast inverse standard-normal CDF approximation for stratified offsets."""
    p = float(p)
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    # Acklam-style rational approximation. More than accurate enough for
    # generating angle perturbations, and much faster than NormalDist.inv_cdf.
    a = (
        -3.969683028665376e01,
         2.209460984245205e02,
        -2.759285104469687e02,
         1.383577518672690e02,
        -3.066479806614716e01,
         2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
         1.615858368580409e02,
        -1.556989798598866e02,
         6.680131188771972e01,
        -1.328068155288572e01,
    )
    ccoef = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
         4.374664141464968e00,
         2.938163982698783e00,
    )
    d = (
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e00,
         3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((ccoef[0] * q + ccoef[1]) * q + ccoef[2]) * q + ccoef[3]) * q + ccoef[4]) * q + ccoef[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((ccoef[0] * q + ccoef[1]) * q + ccoef[2]) * q + ccoef[3]) * q + ccoef[4]) * q + ccoef[5]) / (
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def normal_offset_from_unit(u: float, sd: float) -> float:
    sd = float(sd)
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0
    # Keep quasi-random tails finite. The old random-normal path remains
    # available with --rama-sampling random for fully stochastic tails.
    u = min(0.999, max(0.001, float(u)))
    return float(normal_quantile(u) * sd)


def wrap_degrees(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


def sample_states_and_angles(
    seq: str,
    angle_sd_deg: float,
    rng: np.random.Generator,
    rama_sampling: str = "stratified",
    conformer_idx: int = 0,
    n_requested: int = 1,
    base_seed: int = 0,
    bank: Optional[dict] = None,
):
    states, phis, psis = [], [], []
    strategy = str(rama_sampling or "random").strip().lower()
    local_angle_sd = float(bank.get("angle_sd_deg", angle_sd_deg)) if bank else float(angle_sd_deg)
    for pos, residue in enumerate(seq[1:-1], start=1):
        if strategy in {"stratified", "qmc", "balanced"}:
            state = state_from_unit(
                residue,
                stratified_unit(
                    conformer_idx=conformer_idx,
                    dimension=3 * pos,
                    base_seed=base_seed,
                    n_requested=n_requested,
                ),
                bank=bank,
            )
            phi_offset = normal_offset_from_unit(
                stratified_unit(
                    conformer_idx=conformer_idx,
                    dimension=3 * pos + 1,
                    base_seed=base_seed,
                    n_requested=n_requested,
                ),
                local_angle_sd,
            )
            psi_offset = normal_offset_from_unit(
                stratified_unit(
                    conformer_idx=conformer_idx,
                    dimension=3 * pos + 2,
                    base_seed=base_seed,
                    n_requested=n_requested,
                ),
                local_angle_sd,
            )
        else:
            state = choose_state(residue, rng, bank=bank)
            phi_offset = float(rng.normal(0.0, local_angle_sd))
            psi_offset = float(rng.normal(0.0, local_angle_sd))
        phi0, psi0 = STATE_CENTERS[state]
        phi = wrap_degrees(phi0 + phi_offset)
        psi = wrap_degrees(psi0 + psi_offset)
        states.append(state)
        phis.append(phi)
        psis.append(psi)
    return "".join(states), phis, psis


def encode_angle_list(values) -> str:
    return ";".join(f"{float(v):.10g}" for v in values)


def decode_angle_list(value) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [float(x) for x in text.split(";") if x]


def build_structure(seq: str, phis_deg: list[float], psis_deg: list[float]):
    PeptideBuilder, _ = require_generation_imports()
    from PeptideBuilder import Geometry

    n = len(seq)
    phi_by_idx = {i: phis_deg[i - 1] for i in range(1, n - 1)}
    psi_by_idx = {i: psis_deg[i - 1] for i in range(1, n - 1)}

    structure = PeptideBuilder.initialize_res(Geometry.geometry(AA3[seq[0]]))
    for i, aa in enumerate(seq[1:], start=1):
        geo = Geometry.geometry(AA3[aa])
        if i in phi_by_idx and hasattr(geo, "phi"):
            geo.phi = float(phi_by_idx[i])
        prev_i = i - 1
        if prev_i in psi_by_idx and hasattr(geo, "psi_im1"):
            geo.psi_im1 = float(psi_by_idx[prev_i])
        if hasattr(geo, "omega"):
            geo.omega = 180.0
        PeptideBuilder.add_residue(structure, geo)

    try:
        PeptideBuilder.add_terminal_OXT(structure)
    except Exception:
        pass

    return structure


def place_internal_atom(a, b, c_atom, length_A: float, angle_deg: float, dihedral_deg: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c_atom = np.asarray(c_atom, dtype=np.float64)
    bc = c_atom - b
    bc_norm = np.linalg.norm(bc)
    if bc_norm == 0.0 or not np.isfinite(bc_norm):
        raise ValueError("Cannot place atom from degenerate internal coordinates.")
    bc /= bc_norm

    normal = np.cross(b - a, bc)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < 1.0e-12 or not np.isfinite(normal_norm):
        # Deterministic fallback for the first near-planar placement.
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(axis, bc))) > 0.95:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        normal = np.cross(axis, bc)
        normal_norm = np.linalg.norm(normal)
    normal /= normal_norm
    side = np.cross(normal, bc)

    theta = math.radians(float(angle_deg))
    phi = math.radians(float(dihedral_deg))
    direction = (
        -math.cos(theta) * bc
        + math.sin(theta) * (math.cos(phi) * side + math.sin(phi) * normal)
    )
    return (c_atom + float(length_A) * direction).astype(np.float32)


def fast_backbone_coords_from_angles(seq: str, phis_deg: list[float], psis_deg: list[float]) -> np.ndarray:
    """Approximate N/CA/C backbone coordinates for fast proposal screening."""
    n = len(seq)
    coords = np.zeros((max(0, n) * 3, 3), dtype=np.float32)
    if n <= 0:
        return coords

    coords[0] = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # N0
    coords[1] = np.array([BACKBONE_BOND_N_CA, 0.0, 0.0], dtype=np.float32)  # CA0
    if n == 1:
        return coords

    theta = math.radians(BACKBONE_ANGLE_N_CA_C)
    coords[2] = coords[1] + np.array(
        [
            BACKBONE_BOND_CA_C * math.cos(math.pi - theta),
            BACKBONE_BOND_CA_C * math.sin(math.pi - theta),
            0.0,
        ],
        dtype=np.float32,
    )

    n_internal = max(0, n - 2)
    for i in range(0, n - 1):
        base = 3 * i
        next_base = 3 * (i + 1)
        psi_i = psis_deg[i - 1] if 1 <= i <= n_internal else DEFAULT_TERMINAL_PSI
        coords[next_base] = place_internal_atom(
            coords[base],
            coords[base + 1],
            coords[base + 2],
            BACKBONE_BOND_C_N,
            BACKBONE_ANGLE_CA_C_N,
            psi_i,
        )
        coords[next_base + 1] = place_internal_atom(
            coords[base + 1],
            coords[base + 2],
            coords[next_base],
            BACKBONE_BOND_N_CA,
            BACKBONE_ANGLE_C_N_CA,
            DEFAULT_OMEGA,
        )
        if i + 1 < n:
            phi_next = phis_deg[i] if 1 <= i + 1 <= n_internal else DEFAULT_TERMINAL_PHI
            coords[next_base + 2] = place_internal_atom(
                coords[base + 2],
                coords[next_base],
                coords[next_base + 1],
                BACKBONE_BOND_CA_C,
                BACKBONE_ANGLE_N_CA_C,
                phi_next,
            )
    return coords


def fast_ca_coords_from_angles(seq: str, phis_deg: list[float], psis_deg: list[float]) -> np.ndarray:
    return fast_backbone_coords_from_angles(seq, phis_deg, psis_deg)[1::3]


def iter_atoms(structure):
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    yield atom


def atom_records_from_structure(structure):
    records = []
    for atom in iter_atoms(structure):
        residue = atom.get_parent()
        chain = residue.get_parent()
        records.append({
            "name": atom.get_name(),
            "element": (atom.element or atom.get_name()[0]).strip().upper(),
            "resname": residue.get_resname(),
            "chain": chain.id if chain.id and chain.id != " " else "A",
            "resseq": int(residue.id[1]),
            "coord": np.array(atom.get_coord(), dtype=float),
        })
    return records


def atom_layout_and_coords(seq: str, structure) -> tuple[AtomLayout, np.ndarray]:
    """Return cached atom metadata plus current coordinates for a PeptideBuilder structure."""
    seq = str(seq)
    layout = _ATOM_LAYOUT_CACHE.get(seq)
    if layout is not None:
        coords = np.empty((layout.n_atoms, 3), dtype=np.float32)
        n_seen = 0
        mismatch = False
        for atom in iter_atoms(structure):
            if n_seen >= layout.n_atoms:
                mismatch = True
                break
            coords[n_seen] = atom.get_coord()
            n_seen += 1
        if not mismatch and n_seen == layout.n_atoms:
            return layout, coords

    atoms = list(iter_atoms(structure))
    coords = np.empty((len(atoms), 3), dtype=np.float32)
    atom_fields, resnames, chains, elements = [], [], [], []
    resseq = np.empty(len(atoms), dtype=np.int32)
    ca_indices = []

    for i, atom in enumerate(atoms):
        residue = atom.get_parent()
        chain = residue.get_parent()
        name = atom.get_name()
        coords[i] = atom.get_coord()
        atom_fields.append(name.rjust(4) if len(name) < 4 else name[:4])
        resnames.append(residue.get_resname())
        chains.append((chain.id if chain.id and chain.id != " " else "A")[:1])
        resseq[i] = int(residue.id[1])
        elements.append((atom.element or name[0]).strip().upper()[:2])
        if name == "CA":
            ca_indices.append(i)

    clash_i, clash_j = clash_pair_indices_from_resseq(resseq)
    layout = AtomLayout(
        n_atoms=len(atoms),
        atom_fields=tuple(atom_fields),
        resnames=tuple(resnames),
        chains=tuple(chains),
        resseq=resseq,
        elements=tuple(elements),
        ca_indices=np.asarray(ca_indices, dtype=np.int32),
        clash_i=clash_i,
        clash_j=clash_j,
    )
    _ATOM_LAYOUT_CACHE[seq] = layout
    return layout, coords


def write_pdb_records(records, path: Path):
    with path.open("w") as f:
        serial = 1
        for r in records:
            x, y, z = r["coord"]
            atom_field = r["name"].rjust(4) if len(r["name"]) < 4 else r["name"][:4]
            element = r["element"][:2].strip()
            f.write(
                f"ATOM  {serial:5d} {atom_field} {r['resname']:>3s} {r['chain'][:1]:1s}"
                f"{r['resseq']:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )
            serial += 1
        f.write("TER\nEND\n")


def write_pdb_arrays(layout: AtomLayout, coords: np.ndarray, path: Path):
    coords = np.asarray(coords, dtype=np.float32)
    with path.open("w") as f:
        for serial in range(1, layout.n_atoms + 1):
            i = serial - 1
            x, y, z = coords[i]
            f.write(
                f"ATOM  {serial:5d} {layout.atom_fields[i]} {layout.resnames[i]:>3s} {layout.chains[i]:1s}"
                f"{int(layout.resseq[i]):4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {layout.elements[i]:>2s}\n"
            )
        f.write("TER\nEND\n")


def write_pdb_direct(structure, path: Path):
    write_pdb_records(atom_records_from_structure(structure), path)


def write_conformer_from_record(seq: str, rec: ConformerRecord, path: Path):
    structure = build_structure(seq, decode_angle_list(rec.phi_degrees), decode_angle_list(rec.psi_degrees))
    layout, coords = atom_layout_and_coords(seq, structure)
    write_pdb_arrays(layout, coords, path)


def ca_coords_from_records(records):
    return np.array([r["coord"] for r in records if r["name"] == "CA"], dtype=float)


def radius_of_gyration(coords: np.ndarray) -> float:
    center = coords.mean(axis=0)
    return float(np.sqrt(((coords - center) ** 2).sum(axis=1).mean()))


def contact_pair_indices(n: int, min_sep: int):
    """Cached upper-triangle CA pair indices for contact vectors."""
    n = int(n)
    min_sep = max(0, int(min_sep))
    if n <= 1 or min_sep >= n:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    key = (n, min_sep)
    cached = _PAIR_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    i, j = np.triu_indices(n, k=min_sep)
    i = i.astype(np.int32, copy=False)
    j = j.astype(np.int32, copy=False)
    _PAIR_INDEX_CACHE[key] = (i, j)
    return i, j


def contact_vector_from_coords(ca: np.ndarray, cutoff_A: float, min_sep: int):
    ca = np.asarray(ca, dtype=np.float32)
    n = len(ca)
    i, j = contact_pair_indices(n, min_sep)
    if i.size == 0:
        return np.zeros(0, dtype=np.int8), 0
    diff = ca[i] - ca[j]
    d2 = np.einsum("ij,ij->i", diff, diff, optimize=True)
    contacts = (d2 < np.float32(float(cutoff_A) ** 2)).astype(np.int8, copy=False)
    return contacts, int(contacts.sum(dtype=np.int64))


def clash_pair_indices_from_resseq(resseq: np.ndarray):
    """Cached atom-pair indices eligible for non-neighbor clash checks."""
    resseq = np.asarray(resseq, dtype=np.int32)
    n = int(resseq.size)
    if n <= 1:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    # PeptideBuilder gives the same atom/residue ordering for a given sequence,
    # so this tuple cache is effective across generated conformers.
    key = tuple(int(x) for x in resseq.tolist())
    cached = _CLASH_PAIR_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    i, j = np.triu_indices(n, k=1)
    keep = np.abs(resseq[i] - resseq[j]) > 1
    i = i[keep].astype(np.int32, copy=False)
    j = j[keep].astype(np.int32, copy=False)
    _CLASH_PAIR_INDEX_CACHE[key] = (i, j)
    return i, j


def clash_count_from_coords(
    coords: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    cutoff_A: float,
    max_clashes: Optional[int] = None,
) -> int:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[0] <= 1 or pair_i.size == 0:
        return 0
    cutoff2 = np.float32(float(cutoff_A) ** 2)
    max_clashes = None if max_clashes is None else int(max_clashes)
    clashes = 0
    chunk = 32768
    for start in range(0, pair_i.size, chunk):
        stop = min(pair_i.size, start + chunk)
        diff = coords[pair_i[start:stop]] - coords[pair_j[start:stop]]
        d2 = np.einsum("ij,ij->i", diff, diff, optimize=True)
        clashes += int(np.count_nonzero(d2 < cutoff2))
        if max_clashes is not None and clashes > max_clashes:
            return clashes
    return clashes


def clash_count(records, cutoff_A: float, max_clashes: Optional[int] = None) -> int:
    coords = np.asarray([r["coord"] for r in records], dtype=np.float32)
    resseq = np.asarray([r["resseq"] for r in records], dtype=np.int32)
    i, j = clash_pair_indices_from_resseq(resseq)
    return clash_count_from_coords(coords, i, j, cutoff_A, max_clashes)


def materialize_conformer_from_record(
    seq: str,
    rec: ConformerRecord,
    cfg: GenConfig,
    path: Optional[Path] = None,
    write_pdb: bool = False,
):
    phis = decode_angle_list(rec.phi_degrees)
    psis = decode_angle_list(rec.psi_degrees)
    structure = build_structure(seq, phis, psis)
    layout, coords = atom_layout_and_coords(seq, structure)
    clashes = clash_count_from_coords(coords, layout.clash_i, layout.clash_j, cfg.clash_cutoff_A, cfg.max_clashes)
    if clashes > cfg.max_clashes:
        return None, None, None

    ca = coords[layout.ca_indices]
    if len(ca) < 2 or not np.isfinite(ca).all():
        return None, None, None

    rg = radius_of_gyration(ca)
    e2e = float(np.linalg.norm(ca[-1] - ca[0]))
    cvec, ccount = contact_vector_from_coords(ca, cfg.contact_cutoff_A, cfg.contact_min_sep)

    pdb_path = str(path) if path is not None and write_pdb else rec.pdb_path
    if path is not None and write_pdb:
        write_pdb_arrays(layout, coords, Path(path))

    full_rec = ConformerRecord(
        int(rec.conformer_id),
        pdb_path,
        str(rec.state_string),
        str(rec.phi_degrees),
        str(rec.psi_degrees),
        rg,
        e2e,
        ccount,
        clashes,
        str(getattr(rec, "bank_name", "default") or "default"),
    )
    desc = np.array([rg, e2e, float(ccount)], dtype=np.float32)
    return full_rec, cvec, desc


def generate_one(task):
    seq, idx, out_conf_dir, cfg_dict, worker_seed = task
    cfg = GenConfig(**cfg_dict)
    rng = np.random.default_rng(worker_seed)

    try:
        bank = select_diversity_bank(cfg, int(idx))
        bank_name = str(bank.get("name", "default")) if bank else "default"
        state_string, phis, psis = sample_states_and_angles(
            seq,
            cfg.angle_sd_deg,
            rng,
            rama_sampling=cfg.rama_sampling,
            conformer_idx=int(idx),
            n_requested=int(cfg.n_requested),
            base_seed=int(cfg.seed),
            bank=bank,
        )

        backend = str(getattr(cfg, "generation_backend", "full") or "full").strip().lower()
        if backend == "fast" and not cfg.write_pdbs:
            ca = fast_ca_coords_from_angles(seq, phis, psis)
            if len(ca) < 2 or not np.isfinite(ca).all():
                return None, None, None

            rg = radius_of_gyration(ca)
            e2e = float(np.linalg.norm(ca[-1] - ca[0]))
            cvec, ccount = contact_vector_from_coords(ca, cfg.contact_cutoff_A, cfg.contact_min_sep)
            rec = ConformerRecord(
                idx,
                "",
                state_string,
                encode_angle_list(phis),
                encode_angle_list(psis),
                rg,
                e2e,
                ccount,
                -1,
                bank_name,
            )
            desc = np.array([rg, e2e, float(ccount)], dtype=np.float32)
            return rec, cvec, desc

        structure = build_structure(seq, phis, psis)
        layout, coords = atom_layout_and_coords(seq, structure)
        clashes = clash_count_from_coords(coords, layout.clash_i, layout.clash_j, cfg.clash_cutoff_A, cfg.max_clashes)
        if clashes > cfg.max_clashes:
            return None, None, None

        ca = coords[layout.ca_indices]
        if len(ca) < 2 or not np.isfinite(ca).all():
            return None, None, None

        rg = radius_of_gyration(ca)
        e2e = float(np.linalg.norm(ca[-1] - ca[0]))
        cvec, ccount = contact_vector_from_coords(ca, cfg.contact_cutoff_A, cfg.contact_min_sep)

        pdb_path = Path(out_conf_dir) / f"conf_{idx:06d}.pdb"
        if cfg.write_pdbs:
            write_pdb_arrays(layout, coords, pdb_path)

        rec = ConformerRecord(
            idx,
            str(pdb_path) if cfg.write_pdbs else "",
            state_string,
            encode_angle_list(phis),
            encode_angle_list(psis),
            rg,
            e2e,
            ccount,
            clashes,
            bank_name,
        )
        desc = np.array([rg, e2e, float(ccount)], dtype=np.float32)
        return rec, cvec, desc

    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"



def generate_chunk(task):
    seq, index_seed_pairs, out_conf_dir, cfg_dict = task
    chunk_records = []
    chunk_contacts = []
    chunk_descs = []
    failures = {}
    for idx, worker_seed in index_seed_pairs:
        rec, cvec, desc = generate_one((seq, int(idx), out_conf_dir, cfg_dict, int(worker_seed)))
        if rec is not None:
            chunk_records.append(rec)
            chunk_contacts.append(cvec)
            chunk_descs.append(desc)
        elif isinstance(desc, str):
            key = desc.split(":", 1)[0]
            failures[key] = failures.get(key, 0) + 1
    return chunk_records, chunk_contacts, chunk_descs, failures, len(index_seed_pairs)


def make_generation_chunks(seq, out_conf_dir, cfg_dict, worker_seeds, n_chunks):
    n = len(worker_seeds)
    n_chunks = max(1, min(int(n_chunks), n))
    chunks = [[] for _ in range(n_chunks)]
    for idx, seed in enumerate(worker_seeds):
        chunks[idx % n_chunks].append((idx, int(seed)))
    return [(seq, chunk, str(out_conf_dir), cfg_dict) for chunk in chunks if chunk]


def _angle_matrix_from_records(records: list[ConformerRecord], attr: str, width: int):
    mat = np.full((len(records), max(0, int(width))), np.nan, dtype=np.float32)
    if width <= 0:
        return mat
    for i, rec in enumerate(records):
        vals = decode_angle_list(getattr(rec, attr))
        if vals:
            n = min(width, len(vals))
            mat[i, :n] = np.asarray(vals[:n], dtype=np.float32)
    return mat


class CompactGenerationStore:
    """Array-backed generation metadata for scalable selection and archival.

    The code still keeps ConformerRecord objects long enough to write optional
    CSVs and reconstruct selected PDBs, but all heavy numerical data used for
    selection is packed into contiguous arrays and written to disk. That is the
    practical halfway point before turning the entire generator into a database.
    """
    def __init__(self, seq: str, records: list[ConformerRecord], contacts, descriptors):
        self.seq = str(seq)
        self.records = records
        self.n_torsions = max(0, len(self.seq) - 2)
        self.conformer_id = np.asarray([r.conformer_id for r in records], dtype=np.int64)
        state_width = max(1, self.n_torsions)
        self.state_string = np.asarray([str(r.state_string).encode("ascii", "ignore") for r in records], dtype=f"S{state_width}")
        bank_width = max(8, max((len(str(getattr(r, "bank_name", "default") or "default")) for r in records), default=8))
        self.bank_name = np.asarray([str(getattr(r, "bank_name", "default") or "default").encode("ascii", "ignore") for r in records], dtype=f"S{bank_width}")
        self.phi_degrees = _angle_matrix_from_records(records, "phi_degrees", self.n_torsions)
        self.psi_degrees = _angle_matrix_from_records(records, "psi_degrees", self.n_torsions)
        self.rg_A = np.asarray([r.rg_A for r in records], dtype=np.float32)
        self.end_to_end_A = np.asarray([r.end_to_end_A for r in records], dtype=np.float32)
        self.contact_count = np.asarray([r.contact_count for r in records], dtype=np.int32)
        self.clash_count = np.asarray([r.clash_count for r in records], dtype=np.int16)
        self.contacts = np.vstack(contacts).astype(np.int8, copy=False) if contacts else np.zeros((0, 0), dtype=np.int8)
        self.descriptors = np.vstack(descriptors).astype(np.float32, copy=False) if descriptors else np.zeros((0, 3), dtype=np.float32)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            seq=np.asarray([self.seq]),
            conformer_id=self.conformer_id,
            state_string=self.state_string,
            bank_name=self.bank_name,
            phi_degrees=self.phi_degrees,
            psi_degrees=self.psi_degrees,
            rg_A=self.rg_A,
            end_to_end_A=self.end_to_end_A,
            contact_count=self.contact_count,
            clash_count=self.clash_count,
            contacts=self.contacts,
            descriptors=self.descriptors,
        )


# -----------------------------
# Feature selection
# -----------------------------

def standardize_features(X: np.ndarray, dtype=np.float32) -> np.ndarray:
    X = np.asarray(X, dtype=dtype)
    # Accumulate in float64 for stable centering, return compact float32.
    mu = np.nanmean(X, axis=0, dtype=np.float64).astype(dtype, copy=False)
    sd = np.nanstd(X, axis=0, dtype=np.float64).astype(dtype, copy=False)
    sd[sd == 0] = dtype(1.0)
    return ((X - mu) / sd).astype(dtype, copy=False)


def robust_z(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        sd = np.nanstd(x)
        return np.zeros_like(x) if sd == 0 or not np.isfinite(sd) else (x - np.nanmean(x)) / sd
    return 0.6745 * (x - med) / mad


def build_feature_matrix(contacts: np.ndarray, descriptors: np.ndarray, dtype=np.float32):
    contacts = np.asarray(contacts, dtype=dtype)
    desc_z = standardize_features(descriptors, dtype=dtype)
    out = np.empty((contacts.shape[0], contacts.shape[1] + desc_z.shape[1]), dtype=dtype)
    out[:, :contacts.shape[1]] = contacts
    out[:, contacts.shape[1]:] = desc_z
    return out


def _state_string_to_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", "ignore").strip("\x00")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("ascii", "ignore").strip("\x00")
    return str(value)


def state_onehot_feature_matrix(state_strings: np.ndarray, seq: str, dtype=np.float32) -> np.ndarray:
    width = max(0, len(str(seq)) - 2)
    n = len(state_strings)
    if n == 0 or width <= 0:
        return np.zeros((n, 0), dtype=dtype)
    out = np.zeros((n, width * len(RAMA_STATES)), dtype=dtype)
    state_index = {s: i for i, s in enumerate(RAMA_STATES)}
    for row_i, raw_state in enumerate(state_strings):
        state_string = _state_string_to_text(raw_state)
        for pos, state in enumerate(state_string[:width]):
            j = state_index.get(state)
            if j is not None:
                out[row_i, pos * len(RAMA_STATES) + j] = dtype(1.0)
    return out


def torsion_feature_matrix(phi_degrees: np.ndarray, psi_degrees: np.ndarray, dtype=np.float32) -> np.ndarray:
    phi = np.asarray(phi_degrees, dtype=dtype)
    psi = np.asarray(psi_degrees, dtype=dtype)
    if phi.ndim != 2 or psi.ndim != 2 or phi.shape[0] == 0 or phi.shape[1] == 0:
        return np.zeros((phi.shape[0] if phi.ndim else 0, 0), dtype=dtype)
    phi_rad = np.deg2rad(np.nan_to_num(phi, nan=0.0)).astype(dtype, copy=False)
    psi_rad = np.deg2rad(np.nan_to_num(psi, nan=0.0)).astype(dtype, copy=False)
    return np.hstack([
        np.sin(phi_rad), np.cos(phi_rad),
        np.sin(psi_rad), np.cos(psi_rad),
    ]).astype(dtype, copy=False)


def build_initial_selection_feature_matrix(args, gen_store: CompactGenerationStore, dtype=np.float32):
    """Feature matrix for choosing raw candidate seeds from generated conformers."""
    mode = str(getattr(args, "initial_diversity_features", "mixed") or "mixed").strip().lower()
    shape = build_feature_matrix(gen_store.contacts, gen_store.descriptors, dtype=dtype)
    if mode == "shape":
        return shape

    parts = []
    shape_w = float(getattr(args, "shape_feature_weight", 1.0))
    state_w = float(getattr(args, "state_feature_weight", 0.75))
    torsion_w = float(getattr(args, "torsion_feature_weight", 0.35))

    if mode in {"mixed", "shape"} and shape.shape[1] > 0 and shape_w != 0.0:
        parts.append(shape * dtype(shape_w))

    if mode in {"mixed", "torsion"}:
        state = state_onehot_feature_matrix(gen_store.state_string, gen_store.seq, dtype=dtype)
        torsion = torsion_feature_matrix(gen_store.phi_degrees, gen_store.psi_degrees, dtype=dtype)
        if state.shape[1] > 0 and state_w != 0.0:
            parts.append(state * dtype(state_w))
        if torsion.shape[1] > 0 and torsion_w != 0.0:
            parts.append(standardize_features(torsion, dtype=dtype) * dtype(torsion_w))

    if not parts:
        return shape
    return np.hstack(parts).astype(dtype, copy=False)


def nearest_center_labels_chunked(
    X: np.ndarray,
    centers: np.ndarray,
    row_chunk_size: int = 8192,
    center_chunk_size: int = 256,
) -> np.ndarray:
    """Assign each row to the nearest center without materializing n_rows*n_centers distances."""
    X = np.asarray(X, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    n = int(X.shape[0])
    if n == 0 or centers.shape[0] == 0:
        return np.zeros(n, dtype=np.int32)

    labels = np.zeros(n, dtype=np.int32)
    best = np.full(n, np.inf, dtype=np.float32)
    center_norm = np.einsum("ij,ij->i", centers, centers, optimize=True).astype(np.float32, copy=False)

    row_chunk_size = max(1, int(row_chunk_size))
    center_chunk_size = max(1, int(center_chunk_size))
    for row_start in range(0, n, row_chunk_size):
        row_stop = min(n, row_start + row_chunk_size)
        Xr = X[row_start:row_stop]
        xr_norm = np.einsum("ij,ij->i", Xr, Xr, optimize=True).astype(np.float32, copy=False)
        row_best = np.full(row_stop - row_start, np.inf, dtype=np.float32)
        row_labels = np.zeros(row_stop - row_start, dtype=np.int32)
        for c_start in range(0, centers.shape[0], center_chunk_size):
            c_stop = min(centers.shape[0], c_start + center_chunk_size)
            C = centers[c_start:c_stop]
            # Squared Euclidean distance via norms and matrix multiply keeps
            # temporary memory bounded to row_chunk_size*center_chunk_size.
            dist2 = xr_norm[:, None] + center_norm[c_start:c_stop][None, :] - np.float32(2.0) * (Xr @ C.T)
            local = np.argmin(dist2, axis=1)
            local_best = dist2[np.arange(dist2.shape[0]), local]
            improve = local_best < row_best
            if np.any(improve):
                row_best[improve] = local_best[improve]
                row_labels[improve] = (c_start + local[improve]).astype(np.int32, copy=False)
        best[row_start:row_stop] = row_best
        labels[row_start:row_stop] = row_labels
    return labels


def _entropy_and_neff_from_counter(counter: Counter):
    counts = np.fromiter(counter.values(), dtype=float)
    return _safe_entropy_from_counts(counts)


def _unique_contact_patterns(contacts: np.ndarray, max_rows: int = 250000) -> tuple[int, bool]:
    contacts = np.asarray(contacts, dtype=np.uint8)
    if contacts.ndim != 2 or contacts.shape[0] == 0:
        return 0, False
    sampled = False
    if contacts.shape[0] > max_rows:
        idx = np.linspace(0, contacts.shape[0] - 1, int(max_rows), dtype=np.int64)
        contacts = contacts[idx]
        sampled = True
    if contacts.shape[1] == 0:
        return 1, sampled
    packed = np.packbits(contacts, axis=1)
    return int(np.unique(packed, axis=0).shape[0]), sampled


def write_generation_diversity_reports(out_dir: Path, gen_store: CompactGenerationStore, selected_idx: np.ndarray, args):
    out_dir = Path(out_dir)
    records = gen_store.records
    if not records:
        return

    selected_idx = np.asarray(selected_idx, dtype=int)
    selected_idx = selected_idx[(selected_idx >= 0) & (selected_idx < len(records))]
    selected_set = {int(i) for i in selected_idx.tolist()}

    state_counts = Counter(str(r.state_string) for r in records)
    selected_counts = Counter(str(records[i].state_string) for i in selected_idx)
    h_all, neff_all = _entropy_and_neff_from_counter(state_counts)
    h_sel, neff_sel = _entropy_and_neff_from_counter(selected_counts)

    bank_counts = Counter(str(getattr(r, "bank_name", "default") or "default") for r in records)
    selected_bank_counts = Counter(str(getattr(records[i], "bank_name", "default") or "default") for i in selected_idx)
    bank_rows = []
    for bank, count in sorted(bank_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        candidate_count = int(selected_bank_counts.get(bank, 0))
        bank_rows.append({
            "bank_name": bank,
            "accepted_count": int(count),
            "candidate_count": candidate_count,
            "accepted_fraction": float(count) / max(1, len(records)),
            "candidate_fraction": float(candidate_count) / max(1, len(selected_idx)),
        })
    pd.DataFrame(bank_rows).to_csv(out_dir / "generation_bank_counts.csv", index=False)

    rows = []
    for state, count in sorted(state_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        candidate_count = int(selected_counts.get(state, 0))
        rows.append({
            "state_string": state,
            "accepted_count": int(count),
            "candidate_count": candidate_count,
            "accepted_fraction": float(count) / max(1, len(records)),
            "candidate_fraction": float(candidate_count) / max(1, len(selected_idx)),
        })
    pd.DataFrame(rows).to_csv(out_dir / "generation_state_string_counts.csv", index=False)

    pos_rows = []
    width = max(0, len(gen_store.seq) - 2)
    for pos in range(width):
        residue = str(gen_store.seq)[pos + 1]
        allowed_states, _probs = state_distribution(residue)
        for state in RAMA_STATES:
            acc = sum(1 for r in records if len(str(r.state_string)) > pos and str(r.state_string)[pos] == state)
            cand = sum(
                1 for i in selected_set
                if len(str(records[i].state_string)) > pos and str(records[i].state_string)[pos] == state
            )
            pos_rows.append({
                "internal_position": pos + 1,
                "sequence_index_0based": pos + 1,
                "residue": residue,
                "state": state,
                "allowed_for_residue": state in allowed_states,
                "accepted_count": int(acc),
                "candidate_count": int(cand),
                "accepted_fraction": float(acc) / max(1, len(records)),
                "candidate_fraction": float(cand) / max(1, len(selected_idx)),
            })
    pd.DataFrame(pos_rows).to_csv(out_dir / "generation_position_state_counts.csv", index=False)

    contact_count_counter = Counter(int(x) for x in gen_store.contact_count.tolist())
    selected_contact_counts = gen_store.contact_count[selected_idx] if selected_idx.size else np.asarray([], dtype=np.int32)
    selected_contact_counter = Counter(int(x) for x in selected_contact_counts.tolist())
    contact_rows = []
    for count, n_accepted in sorted(contact_count_counter.items()):
        n_candidate = int(selected_contact_counter.get(count, 0))
        contact_rows.append({
            "contact_count": int(count),
            "accepted_count": int(n_accepted),
            "candidate_count": n_candidate,
            "accepted_fraction": float(n_accepted) / max(1, len(records)),
            "candidate_fraction": float(n_candidate) / max(1, len(selected_idx)),
        })
    pd.DataFrame(contact_rows).to_csv(out_dir / "generation_contact_count_counts.csv", index=False)

    h_contact_all, neff_contact_all = _entropy_and_neff_from_counter(contact_count_counter)
    h_contact_sel, neff_contact_sel = _entropy_and_neff_from_counter(selected_contact_counter)
    unique_contacts_all, unique_contacts_all_sampled = _unique_contact_patterns(gen_store.contacts)
    unique_contacts_sel, unique_contacts_sel_sampled = _unique_contact_patterns(gen_store.contacts[selected_idx] if selected_idx.size else gen_store.contacts[:0])

    rg = np.asarray(gen_store.rg_A, dtype=float)
    e2e = np.asarray(gen_store.end_to_end_A, dtype=float)
    contacts = np.asarray(gen_store.contact_count, dtype=float)
    sel_rg = rg[selected_idx] if selected_idx.size else np.asarray([], dtype=float)
    sel_e2e = e2e[selected_idx] if selected_idx.size else np.asarray([], dtype=float)
    sel_contacts = contacts[selected_idx] if selected_idx.size else np.asarray([], dtype=float)

    def stat(prefix: str, values: np.ndarray) -> dict:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {f"{prefix}_min": float("nan"), f"{prefix}_median": float("nan"), f"{prefix}_max": float("nan")}
        return {
            f"{prefix}_min": float(np.nanmin(values)),
            f"{prefix}_median": float(np.nanmedian(values)),
            f"{prefix}_max": float(np.nanmax(values)),
        }

    actual_backend = str(getattr(args, "actual_generation_backend", getattr(args, "generation_backend", "unknown")))
    summary = {
        "generation_backend": actual_backend,
        "proposal_clash_counts_are_all_atom": actual_backend == "full",
        "rama_sampling": str(getattr(args, "rama_sampling", "unknown")),
        "initial_diversity_features": str(getattr(args, "initial_diversity_features", "unknown")),
        "diversity_bank_preset": str(getattr(args, "diversity_bank_preset", "off")),
        "unique_generation_banks": int(len(bank_counts)),
        "generation_bank_counts": {str(k): int(v) for k, v in sorted(bank_counts.items())},
        "candidate_bank_counts": {str(k): int(v) for k, v in sorted(selected_bank_counts.items())},
        "preselection_bin_quota": int(getattr(args, "preselection_bin_quota", 0) or 0),
        "shape_feature_weight": float(getattr(args, "shape_feature_weight", 1.0)),
        "state_feature_weight": float(getattr(args, "state_feature_weight", 0.75)),
        "torsion_feature_weight": float(getattr(args, "torsion_feature_weight", 0.35)),
        "n_requested": int(getattr(args, "n", 0)),
        "n_accepted": int(len(records)),
        "n_candidates": int(len(selected_idx)),
        "unique_state_strings_accepted": int(len(state_counts)),
        "unique_state_strings_candidates": int(len(selected_counts)),
        "state_string_entropy_accepted": h_all,
        "state_string_neff_accepted": neff_all,
        "state_string_entropy_candidates": h_sel,
        "state_string_neff_candidates": neff_sel,
        "contact_count_entropy_accepted": h_contact_all,
        "contact_count_neff_accepted": neff_contact_all,
        "contact_count_entropy_candidates": h_contact_sel,
        "contact_count_neff_candidates": neff_contact_sel,
        "unique_contact_patterns_accepted": unique_contacts_all,
        "unique_contact_patterns_accepted_sampled": unique_contacts_all_sampled,
        "unique_contact_patterns_candidates": unique_contacts_sel,
        "unique_contact_patterns_candidates_sampled": unique_contacts_sel_sampled,
        **stat("rg_A", rg),
        **stat("candidate_rg_A", sel_rg),
        **stat("end_to_end_A", e2e),
        **stat("candidate_end_to_end_A", sel_e2e),
        **stat("contact_count", contacts),
        **stat("candidate_contact_count", sel_contacts),
        "note": "Initial diversity is encouraged by stratified/banked Ramachandran state/angle sampling, optional Rg/E2E/contact quota preselection, and candidate selection in mixed contact/shape/torsion/state space. With generation_backend=fast, full all-atom clash checking is applied during candidate materialization.",
    }
    (out_dir / "generation_diversity_summary.json").write_text(json.dumps(summary, indent=2))


def select_by_minibatch_or_farthest(X: np.ndarray, n_select: int, method: str, badness: Optional[np.ndarray] = None):
    n = len(X)
    n_select = max(1, min(int(n_select), n))

    if method == "minibatch":
        try:
            from sklearn.cluster import MiniBatchKMeans
        except Exception as exc:
            raise RuntimeError("MiniBatchKMeans requires scikit-learn. Use --cluster-method farthest.") from exc

        model = MiniBatchKMeans(n_clusters=n_select, random_state=0, batch_size=min(2048, n), n_init="auto")
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
        selected = []
        for c in range(n_select):
            inds = np.where(labels == c)[0]
            if len(inds) == 0:
                continue
            chosen = int(inds[np.argmin(badness[inds])]) if badness is not None else int(inds[np.argmin(np.linalg.norm(X[inds] - centers[c], axis=1))])
            selected.append(chosen)

    elif method == "farthest":
        if badness is not None:
            first = int(np.argmin(badness))
        else:
            center = np.nanmean(X, axis=0, dtype=np.float64)
            first = int(np.argmin(np.linalg.norm(X - center.astype(X.dtype, copy=False), axis=1)))
        selected = [first]
        dmin = np.linalg.norm(X - X[first], axis=1)
        for _ in range(1, n_select):
            if badness is None:
                score = dmin.copy()
            else:
                score = dmin - 0.25 * standardize_features(badness.reshape(-1, 1)).ravel()
            score[selected] = -np.inf
            j = int(np.argmax(score))
            selected.append(j)
            dmin = np.minimum(dmin, np.linalg.norm(X - X[j], axis=1))
        centers = X[selected]
        labels = nearest_center_labels_chunked(X, centers)
    else:
        raise ValueError(f"Unknown cluster method: {method}")

    return np.array(selected, dtype=int), np.array(labels, dtype=int)


def squared_distance_to_point_chunked(X: np.ndarray, point: np.ndarray, row_chunk_size: int = 262144) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    point = np.asarray(point, dtype=np.float32)
    out = np.empty(X.shape[0], dtype=np.float32)
    row_chunk_size = max(1, int(row_chunk_size))
    for start in range(0, X.shape[0], row_chunk_size):
        stop = min(X.shape[0], start + row_chunk_size)
        diff = X[start:stop] - point
        out[start:stop] = np.einsum("ij,ij->i", diff, diff, optimize=True)
    return out


def iter_cluster_fallback_indices(X: np.ndarray, labels: np.ndarray, selected: list[int], yielded: np.ndarray):
    labels = np.asarray(labels)
    if labels.shape[0] != len(X):
        return

    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    unique_labels, starts = np.unique(sorted_labels, return_index=True)
    label_bounds = {}
    for pos, label in enumerate(unique_labels):
        start = int(starts[pos])
        end = int(starts[pos + 1]) if pos + 1 < len(starts) else int(len(order))
        label_bounds[int(label)] = (start, end)

    queues = []
    seen_clusters = set()
    for rep in selected:
        cluster = int(labels[int(rep)])
        if cluster in seen_clusters:
            continue
        seen_clusters.add(cluster)
        bounds = label_bounds.get(cluster)
        if bounds is None:
            continue
        start, end = bounds
        inds = order[start:end]
        inds = inds[~yielded[inds]]
        if inds.size == 0:
            continue
        dist2 = squared_distance_to_point_chunked(X[inds], X[int(rep)], row_chunk_size=max(1, len(inds)))
        queues.append(inds[np.argsort(dist2, kind="mergesort")])

    pointers = [0] * len(queues)
    active = True
    while active:
        active = False
        for q_i, queue in enumerate(queues):
            p = pointers[q_i]
            while p < len(queue) and yielded[int(queue[p])]:
                p += 1
            pointers[q_i] = p
            if p >= len(queue):
                continue
            idx = int(queue[p])
            yielded[idx] = True
            pointers[q_i] = p + 1
            active = True
            yield idx


def iter_candidate_indices_by_diversity(X: np.ndarray, selected_idx: np.ndarray, labels: Optional[np.ndarray] = None):
    """Yield selected indices first, then farthest unused fallbacks if validation rejects any."""
    n = len(X)
    if n == 0:
        return
    yielded = np.zeros(n, dtype=bool)
    selected = [int(i) for i in np.asarray(selected_idx, dtype=int).tolist() if 0 <= int(i) < n]
    selected = list(dict.fromkeys(selected))
    for i in selected:
        yielded[i] = True
        yield i

    if labels is not None:
        yield from iter_cluster_fallback_indices(X, labels, selected, yielded)
        if bool(np.all(yielded)):
            return

    if selected:
        # Fallback for callers without labels. Avoid the old dense
        # n_conformers*n_selected distance tensor; a small deterministic subset
        # is enough to seed diverse emergency fallbacks.
        stride = max(1, int(math.ceil(len(selected) / 64)))
        seed_centers = selected[::stride][:64]
        dmin = np.full(n, np.inf, dtype=np.float32)
        for center_idx in seed_centers:
            dmin = np.minimum(dmin, squared_distance_to_point_chunked(X, X[int(center_idx)]))
    else:
        center = np.nanmean(X, axis=0, dtype=np.float64).astype(X.dtype, copy=False)
        dmin = squared_distance_to_point_chunked(X, center)

    while not bool(np.all(yielded)):
        score = dmin.copy()
        score[yielded] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            return
        yielded[j] = True
        yield j
        dmin = np.minimum(dmin, squared_distance_to_point_chunked(X, X[j]))



def preselection_quota_indices(args, gen_store: CompactGenerationStore) -> tuple[np.ndarray, dict]:
    """Cheap diversity-preserving quota filter before expensive clustering.

    Keeps at most N conformers per coarse (Rg, end-to-end, contact-count, bank)
    bin. This avoids spending clustering/minimization budget on thousands of
    near-duplicate noodles while retaining broad shape/contact coverage.
    """
    quota = int(getattr(args, "preselection_bin_quota", 0) or 0)
    n = int(len(gen_store.records))
    if quota <= 0 or n <= 0:
        return np.arange(n, dtype=int), {
            "enabled": False,
            "input_count": n,
            "kept_count": n,
            "unique_bins": 0,
        }

    rg_bin = max(1.0e-6, float(getattr(args, "preselection_rg_bin_A", 1.0) or 1.0))
    e2e_bin = max(1.0e-6, float(getattr(args, "preselection_e2e_bin_A", 1.0) or 1.0))
    contact_bin = max(1, int(getattr(args, "preselection_contact_bin", 2) or 2))
    use_bank = bool(getattr(args, "preselection_include_bank", True))

    kept = []
    counts = {}
    bank_values = []
    if hasattr(gen_store, "bank_name"):
        for raw in gen_store.bank_name:
            bank_values.append(_state_string_to_text(raw) or "default")
    else:
        bank_values = [str(getattr(r, "bank_name", "default") or "default") for r in gen_store.records]

    for i in range(n):
        bank = bank_values[i] if use_bank and i < len(bank_values) else "all"
        key = (
            int(math.floor(float(gen_store.rg_A[i]) / rg_bin)),
            int(math.floor(float(gen_store.end_to_end_A[i]) / e2e_bin)),
            int(math.floor(int(gen_store.contact_count[i]) / contact_bin)),
            bank,
        )
        cbin = counts.get(key, 0)
        if cbin < quota:
            kept.append(i)
            counts[key] = cbin + 1

    kept_idx = np.asarray(kept, dtype=int)
    if kept_idx.size == 0:
        kept_idx = np.arange(n, dtype=int)
    summary = {
        "enabled": True,
        "input_count": n,
        "kept_count": int(kept_idx.size),
        "unique_bins": int(len(counts)),
        "quota_per_bin": int(quota),
        "rg_bin_A": float(rg_bin),
        "end_to_end_bin_A": float(e2e_bin),
        "contact_count_bin": int(contact_bin),
        "include_bank": bool(use_bank),
    }
    return kept_idx, summary


def write_quota_preselection_report(out_dir: Path, summary: dict):
    out_dir = Path(out_dir)
    try:
        (out_dir / "generation_quota_preselection_summary.json").write_text(json.dumps(summary, indent=2))
    except Exception:
        pass

def generate_candidates(args):
    seq = validate_sequence(args.seq)
    out_dir = Path(args.out)
    conf_dir = out_dir / "conformers"
    cand_dir = out_dir / "candidate_seeds"

    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Avoid stale PDBs turning a small candidate set into thousands of minimizations.
    shutil.rmtree(conf_dir, ignore_errors=True)
    shutil.rmtree(cand_dir, ignore_errors=True)
    conf_dir.mkdir(exist_ok=True)
    cand_dir.mkdir(exist_ok=True)

    generation_backend = "full" if bool(args.keep_all_pdbs) else str(getattr(args, "generation_backend", "fast"))
    args.actual_generation_backend = generation_backend
    cfg = GenConfig(
        seq,
        args.n,
        args.angle_sd,
        args.clash_cutoff,
        args.contact_cutoff,
        args.contact_min_sep,
        args.max_clashes,
        args.seed,
        write_pdbs=bool(args.keep_all_pdbs),
        rama_sampling=str(getattr(args, "rama_sampling", "stratified")),
        generation_backend=generation_backend,
        diversity_bank_preset=str(getattr(args, "diversity_bank_preset", "off")),
        diversity_bank_wide_angle_sd=float(getattr(args, "diversity_bank_wide_angle_sd", 45.0)),
    )
    (out_dir / "generation_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    require_generation_imports()

    rng = np.random.default_rng(args.seed)
    worker_seeds = rng.integers(0, 2**32 - 1, size=args.n, dtype=np.uint32)

    records, contact_rows, desc_rows = [], [], []
    print_stage("1) Ramachandran conformer generation")
    ui_message(f"Sequence: {seq} | trials: {args.n} | workers: {args.jobs}")
    progress = CliProgress("generate", args.n)
    live = LiveStats("live generated conformers", enabled=args.live_hist, every=max(1, getattr(args, "hist_every", 1000)))
    failures = {}
    n_fail_reported = 0

    if args.jobs <= 1:
        cfg_payload = asdict(cfg)
        for i in range(1, args.n + 1):
            rec, cvec, desc = generate_one((seq, i - 1, str(conf_dir), cfg_payload, int(worker_seeds[i - 1])))
            if rec is not None:
                records.append(rec); contact_rows.append(cvec); desc_rows.append(desc)
                live.add(rg_A=rec.rg_A, end_to_end_A=rec.end_to_end_A, contacts=rec.contact_count)
            elif isinstance(desc, str):
                key = desc.split(":", 1)[0]
                failures[key] = failures.get(key, 0) + 1
                n_fail_reported += 1
            if i % max(1, args.progress_every) == 0 or i == args.n:
                acc = len(records)
                progress.update(i, accepted=acc, acc_rate=f"{100*acc/i:4.1f}%", fails=n_fail_reported)
                live.maybe_print(i)
    else:
        n_by_size = math.ceil(args.n / max(1, args.gen_chunk_size))
        n_by_workers = max(1, args.jobs * max(1, args.gen_chunks_per_worker))
        n_chunks = max(1, min(args.n, max(n_by_size, n_by_workers)))
        chunk_tasks = make_generation_chunks(seq, conf_dir, asdict(cfg), worker_seeds, n_chunks)
        ui_message(f"Chunked generation: {len(chunk_tasks)} worker tasks (~{math.ceil(args.n / max(1, len(chunk_tasks)))} conformers/chunk)")
        render_dashboard()

        ex = None
        futures = []
        try:
            ex = make_process_pool(args.jobs)
            futures = [ex.submit(generate_chunk, t) for t in chunk_tasks]
            completed = 0
            for fut in as_completed(futures):
                chunk_records, chunk_contacts, chunk_descs, chunk_failures, n_done = fut.result()
                completed += int(n_done)
                records.extend(chunk_records)
                contact_rows.extend(chunk_contacts)
                desc_rows.extend(chunk_descs)

                for rec in chunk_records:
                    live.add(rg_A=rec.rg_A, end_to_end_A=rec.end_to_end_A, contacts=rec.contact_count)

                for key, val in chunk_failures.items():
                    failures[key] = failures.get(key, 0) + int(val)
                    n_fail_reported += int(val)

                acc = len(records)
                progress.update(completed, accepted=acc, acc_rate=f"{100*acc/max(1,completed):4.1f}%", fails=n_fail_reported)
                live.maybe_print(completed)
        except KeyboardInterrupt:
            ui_message("Interrupted: terminating generation workers.")
            render_dashboard()
            cancel_futures_now(futures)
            shutdown_process_pool_now(ex)
            raise SystemExit(130)
        finally:
            if ex is not None:
                shutdown_process_pool_now(ex)
    progress.done(args.n, accepted=len(records), acc_rate=f"{100*len(records)/max(1,args.n):4.1f}%", fails=n_fail_reported)
    live.maybe_print(args.n, force=True)
    if failures:
        ui_message("Generation failure summary: " + ", ".join(f"{k}={v}" for k, v in sorted(failures.items())[:5]))
        render_dashboard()

    if not records:
        raise RuntimeError("No conformers accepted.")

    if len(records) > 1:
        order = np.argsort(np.asarray([r.conformer_id for r in records], dtype=np.int64))
        if not np.all(order == np.arange(len(order))):
            records = [records[int(i)] for i in order]
            contact_rows = [contact_rows[int(i)] for i in order]
            desc_rows = [desc_rows[int(i)] for i in order]

    gen_store = CompactGenerationStore(seq, records, contact_rows, desc_rows)
    gen_store.save(out_dir / "generation_compact_records.npz")
    contacts = gen_store.contacts
    descriptors = gen_store.descriptors
    # The numerical selection path now uses compact arrays. The records list is
    # retained only for optional CSV output and selected-PDB reconstruction.
    contact_rows = desc_rows = None
    X_shape = build_feature_matrix(contacts, descriptors)
    X = build_initial_selection_feature_matrix(args, gen_store)

    bank_counter = Counter(str(getattr(r, "bank_name", "default") or "default") for r in records)
    bank_preview = ", ".join(f"{k}={v}" for k, v in bank_counter.most_common(6))
    if bank_preview:
        ui_message(f"Generation banks: {bank_preview}")
        render_dashboard()

    n_candidate = min(args.n_candidate_seeds, len(records))
    pre_idx, pre_summary = preselection_quota_indices(args, gen_store)
    write_quota_preselection_report(out_dir, pre_summary)
    ui_message(f"Accepted conformers: {len(records)} / {args.n}")
    print_stage("2) Broad candidate selection")
    if pre_summary.get("enabled"):
        ui_message(
            f"Quota preselection: kept {pre_summary['kept_count']}/{pre_summary['input_count']} "
            f"from {pre_summary['unique_bins']} Rg/E2E/contact bins before clustering."
        )
    ui_message(f"Selecting {n_candidate} candidates by {args.cluster_method} clustering.")
    render_dashboard()

    X_select = X[pre_idx] if pre_idx.size and pre_idx.size < len(X) else X
    selected_local_idx, _labels_local = select_by_minibatch_or_farthest(X_select, n_candidate, args.cluster_method, None)
    selected_idx = pre_idx[selected_local_idx] if X_select is not X else selected_local_idx
    labels = nearest_center_labels_chunked(X, X[selected_idx]) if len(selected_idx) else np.zeros(len(X), dtype=int)

    if args.write_conformer_table:
        ui_message(f"Writing full conformer table: {len(records)} rows.")
        render_dashboard()
        conformer_fields = list(ConformerRecord.__dataclass_fields__.keys()) + ["generation_cluster_id"]
        write_dict_csv(
            out_dir / "conformers.csv",
            conformer_fields,
            ({**asdict(rec), "generation_cluster_id": int(labels[i])} for i, rec in enumerate(records)),
        )
    else:
        ui_message("Skipping full conformer table; use --write-conformer-table to save conformers.csv.")
        render_dashboard()

    selected_rows = []
    candidate_feature_indices = []
    rejected_candidates = 0
    candidate_progress = CliProgress("candidate pdbs", n_candidate)
    for idx in iter_candidate_indices_by_diversity(X, selected_idx, labels):
        if len(selected_rows) >= n_candidate:
            break
        rec = records[int(idx)]
        rank = len(selected_rows)
        dst = cand_dir / f"candidate_{rank:03d}_gencluster_{int(labels[idx]):03d}_conf_{rec.conformer_id:06d}.pdb"
        src = Path(rec.pdb_path) if rec.pdb_path else None
        if src is not None and src.exists() and int(rec.clash_count) >= 0 and int(rec.clash_count) <= int(cfg.max_clashes):
            shutil.copy2(src, dst)
        else:
            full_rec, _cvec, _desc = materialize_conformer_from_record(seq, rec, cfg, dst, write_pdb=True)
            if full_rec is None:
                rejected_candidates += 1
                continue
            rec = full_rec
        selected_rows.append({**asdict(rec), "candidate_rank": rank, "generation_cluster_id": int(labels[idx]), "candidate_pdb_path": str(dst)})
        candidate_feature_indices.append(int(idx))
        candidate_progress.update(rank + 1)
    if len(selected_rows) < n_candidate:
        raise RuntimeError(
            f"Only {len(selected_rows)} all-atom-valid candidate seeds found after checking {len(records)} proposals. "
            "Increase --n, relax --max-clashes, or use --generation-backend full for stricter proposal filtering."
        )
    candidate_progress.done(n_candidate)
    if rejected_candidates:
        ui_message(f"Rejected {rejected_candidates} selected/fallback candidates during all-atom materialization.")
        render_dashboard()
    write_generation_diversity_reports(out_dir, gen_store, np.asarray(candidate_feature_indices, dtype=int), args)
    ui_message(
        f"Initial diversity: backend={cfg.generation_backend}, sampling={args.rama_sampling}, "
        f"features={args.initial_diversity_features}; reports written to generation_diversity_summary.json."
    )
    render_dashboard()

    ui_message(f"Writing candidate manifest: {len(selected_rows)} rows.")
    render_dashboard()
    candidate_fields = list(ConformerRecord.__dataclass_fields__.keys()) + [
        "candidate_rank",
        "generation_cluster_id",
        "candidate_pdb_path",
    ]
    write_dict_csv(out_dir / "candidate_seeds.csv", candidate_fields, selected_rows)
    np.save(out_dir / "generation_contacts.npy", contacts)
    np.save(out_dir / "generation_shape_features.npy", X_shape)
    np.save(out_dir / "generation_cluster_features.npy", X)
    np.save(out_dir / "generation_selection_features.npy", X)

    if not args.keep_all_pdbs:
        shutil.rmtree(conf_dir, ignore_errors=True)

    ui_message(f"Candidate seeds written: {cand_dir}")
    render_dashboard()
    return cand_dir


# -----------------------------
# OpenMM minimization and basin hopping
# -----------------------------

def make_forcefield(app, cfg: MinConfig):
    info = FF_CHOICES[cfg.ff]
    if cfg.mode == "implicit":
        return app.ForceField(info["protein"], info["implicit"])
    if cfg.mode == "explicit":
        return app.ForceField(info["protein"], info["waters"][cfg.water])
    raise ValueError(f"Unknown mode {cfg.mode}")


def make_platform_simulation(openmm, app, topology, system, integrator, cfg: MinConfig):
    platform_name = resolve_openmm_platform_name(openmm, cfg.platform)
    if platform_name is not None:
        platform = openmm.Platform.getPlatformByName(platform_name)
        props = {}
        if platform_name in {"CUDA", "OpenCL"}:
            if cfg.precision:
                props["Precision"] = cfg.precision
            if cfg.device_index:
                props["DeviceIndex"] = cfg.device_index
        if platform_name == "CPU" and cfg.cpu_threads > 0:
            props["Threads"] = str(cfg.cpu_threads)
        return app.Simulation(topology, system, integrator, platform, props) if props else app.Simulation(topology, system, integrator, platform)
    return app.Simulation(topology, system, integrator)



def openmm_constraints(app, cfg: MinConfig):
    """Map CLI constraint choice to OpenMM's createSystem constraints argument."""
    choice = str(getattr(cfg, "constraints", "none")).strip().lower()
    if choice in {"", "none", "no", "false", "off"}:
        return None
    if choice in {"hbonds", "hbond", "h-bonds", "h_bonds"}:
        return app.HBonds
    if choice in {"allbonds", "all-bonds", "all_bonds"}:
        return app.AllBonds
    if choice in {"hangles", "h-angles", "h_angles"}:
        return app.HAngles
    raise ValueError(f"Unknown constraint mode {choice!r}. Use: none, hbonds, allbonds, hangles.")


def hydrogen_mass_quantity(unit, mass_amu: float):
    mass_amu = float(mass_amu or 0.0)
    if mass_amu <= 0.0:
        return None
    mass_unit = getattr(unit, "amu", None)
    if mass_unit is None:
        mass_unit = getattr(unit, "dalton")
    return mass_amu * mass_unit


def create_implicit_system(ff, topology, app, unit, cfg: MinConfig, hydrogen_mass_amu: float = 0.0, force_hbonds: bool = False):
    constraints = openmm_constraints(app, cfg)
    if constraints is None and bool(force_hbonds):
        constraints = app.HBonds
    kwargs = {
        "nonbondedMethod": app.NoCutoff,
        "constraints": constraints,
    }
    hmass = hydrogen_mass_quantity(unit, hydrogen_mass_amu)
    if hmass is not None:
        kwargs["hydrogenMass"] = hmass
    return ff.createSystem(topology, **kwargs)


def topology_signature(topology):
    """A compact safety signature for reusing an OpenMM Context across same-sequence seeds."""
    sig = []
    for atom in topology.atoms():
        elem = atom.element.symbol if atom.element is not None else ""
        res = atom.residue
        chain = res.chain.id if res.chain is not None else ""
        sig.append((atom.index, atom.name, elem, res.name, res.index, chain))
    return tuple(sig)


def state_energy(unit, simulation):
    state = simulation.context.getState(getEnergy=True)
    return float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))


def state_energy_forces_positions(unit, simulation):
    state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    max_force = float(np.sqrt((forces ** 2).sum(axis=1)).max())
    positions = state.getPositions()
    return energy, max_force, positions

def count_waters_ions(topology):
    water_names = {"HOH", "WAT", "SOL"}
    ion_names = {"NA", "CL", "K", "MG", "CA", "ZN"}
    nw = ni = 0
    for residue in topology.residues():
        if residue.name in water_names:
            nw += 1
        elif residue.name in ion_names:
            ni += 1
    return nw, ni


def peptide_ca_indices(topology):
    return [atom.index for atom in topology.atoms() if atom.name == "CA" and atom.residue.name not in {"HOH", "WAT", "SOL"}]


def ca_positions_angstrom(unit, positions, indices):
    return np.array([positions[i].value_in_unit(unit.angstrom) for i in indices], dtype=float)


def rg_e2e_contact(unit, positions, indices, cutoff_A: float, min_sep: int):
    if len(indices) < 2:
        return float("nan"), float("nan"), 0, np.zeros(0, dtype=np.int8)
    ca = ca_positions_angstrom(unit, positions, indices)
    rg_A = radius_of_gyration(ca)
    e2e_A = float(np.linalg.norm(ca[-1] - ca[0]))
    cvec, ccount = contact_vector_from_coords(ca, cutoff_A, min_sep)
    return rg_A / 10.0, e2e_A / 10.0, ccount, cvec


def energy_forces(unit, simulation):
    state = simulation.context.getState(getEnergy=True, getForces=True)
    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    max_force = float(np.sqrt((forces ** 2).sum(axis=1)).max())
    return energy, max_force


def minimize_energy_with_optional_adaptive(unit, simulation, cfg: MinConfig, max_iterations: Optional[int] = None):
    """Run OpenMM minimization, optionally in chunks with cheaper convergence checks.

    Original behavior is preserved when adaptive_min_force_check_every == 1:
    every chunk fetches energy, forces, and positions. For many small implicit
    minimizations, OpenMM force/position readback can become a meaningful fixed
    overhead. Setting adaptive_min_force_check_every=0 uses energy-only early
    stopping and fetches forces/positions once at the end; values >1 fetch
    forces periodically and always validate force before a force-based stop.
    """
    max_iter = int(cfg.max_iterations if max_iterations is None else max_iterations)
    max_iter = max(0, max_iter)
    adaptive = bool(getattr(cfg, "adaptive_min", False)) and max_iter > 0
    if not adaptive:
        simulation.minimizeEnergy(maxIterations=max_iter)
        energy, max_force, positions = state_energy_forces_positions(unit, simulation)
        return energy, max_force, positions, max_iter, 1, "max_iterations"

    chunk = max(1, int(getattr(cfg, "adaptive_min_chunk_iterations", 25) or 25))
    chunk = min(chunk, max_iter)
    energy_tol = max(0.0, float(getattr(cfg, "adaptive_min_energy_tol_kj_mol", 0.05)))
    force_tol = max(0.0, float(getattr(cfg, "adaptive_min_force_tol_kj_mol_nm", 500.0)))
    needed_stall = max(1, int(getattr(cfg, "adaptive_min_stall_rounds", 2) or 2))
    force_check_every = int(getattr(cfg, "adaptive_min_force_check_every", 1) or 0)

    used = 0
    rounds = 0
    stall = 0
    prev_e = state_energy(unit, simulation)
    last_e = prev_e
    last_force = float("nan")
    last_positions = None
    last_force_checked_round = -1

    while used < max_iter:
        n_iter = min(chunk, max_iter - used)
        simulation.minimizeEnergy(maxIterations=n_iter)
        used += n_iter
        rounds += 1

        check_force_now = force_check_every > 0 and (rounds % force_check_every == 0 or used >= max_iter)
        if check_force_now:
            last_e, last_force, last_positions = state_energy_forces_positions(unit, simulation)
            last_force_checked_round = rounds
        else:
            last_e = state_energy(unit, simulation)

        dE = abs(prev_e - last_e) if np.isfinite(prev_e) and np.isfinite(last_e) else float("inf")
        if dE <= energy_tol:
            stall += 1
        else:
            stall = 0

        if stall >= needed_stall:
            if force_check_every <= 0:
                last_e, last_force, last_positions = state_energy_forces_positions(unit, simulation)
                return last_e, last_force, last_positions, used, rounds, "adaptive_energy_converged"
            if last_force_checked_round != rounds:
                last_e, last_force, last_positions = state_energy_forces_positions(unit, simulation)
                last_force_checked_round = rounds
            if last_force <= force_tol:
                return last_e, last_force, last_positions, used, rounds, "adaptive_converged"
            # Energy has flattened, but force criterion is not satisfied.
            # Reset the stall counter so we do not force-check every subsequent
            # chunk unless the energy remains flat again.
            stall = 0

        prev_e = last_e

    if last_positions is None:
        last_e, last_force, last_positions = state_energy_forces_positions(unit, simulation)
    return last_e, last_force, last_positions, used, rounds, "max_iterations"

def make_min_config(args, mode: str) -> MinConfig:
    return MinConfig(
        mode=mode,
        ff=args.ff,
        water=args.water,
        padding_nm=args.padding,
        ionic_strength_molar=args.ionic_strength,
        neutralize=not args.no_neutralize,
        ph=args.ph,
        temperature_k=args.temperature,
        friction_per_ps=args.friction,
        nonbonded_cutoff_nm=args.cutoff,
        max_iterations=args.implicit_max_iterations if mode == "implicit" else args.explicit_max_iterations,
        restraint_k_kj_mol_nm2=args.restraint_k,
        platform=args.platform,
        precision=args.precision,
        device_index=args.device_index,
        cpu_threads=args.cpu_threads,
        save_systems=args.save_systems,
        constraints=getattr(args, "constraints", "none"),
        contact_cutoff_A=args.contact_cutoff,
        contact_min_sep=args.contact_min_sep,
        adaptive_min=bool((mode == "implicit" and getattr(args, "adaptive_min", True)) or (mode == "explicit" and getattr(args, "adaptive_explicit_min", False))),
        adaptive_min_chunk_iterations=getattr(args, "adaptive_min_chunk_iterations", 25),
        adaptive_min_energy_tol_kj_mol=getattr(args, "adaptive_min_energy_tol", 0.05),
        adaptive_min_force_tol_kj_mol_nm=getattr(args, "adaptive_min_force_tol", 500.0),
        adaptive_min_stall_rounds=getattr(args, "adaptive_min_stall_rounds", 2),
        adaptive_min_force_check_every=getattr(args, "adaptive_min_force_check_every", 1),
        assume_same_implicit_topology=bool(getattr(args, "assume_same_implicit_topology", False)),
    )


def minimize_one(task):
    pdb_path_str, out_min_dir_str, system_dir_str, cfg_dict = task
    openmm, app, unit, XmlSerializer = import_openmm()

    pdb_path = Path(pdb_path_str)
    out_min_dir = Path(out_min_dir_str)
    system_dir = Path(system_dir_str)
    cfg = MinConfig(**cfg_dict)

    seed_name = pdb_path.stem
    out_pdb = out_min_dir / f"{seed_name}_{cfg.mode}_min.pdb"

    try:
        pdb = app.PDBFile(str(pdb_path))
        ff = make_forcefield(app, cfg)
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff, pH=cfg.ph)

        if cfg.mode == "explicit":
            modeller.addSolvent(
                ff,
                padding=cfg.padding_nm * unit.nanometer,
                ionicStrength=cfg.ionic_strength_molar * unit.molar,
                neutralize=cfg.neutralize,
            )
            system = ff.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=cfg.nonbonded_cutoff_nm * unit.nanometer,
                constraints=openmm_constraints(app, cfg),
                rigidWater=True,
                ewaldErrorTolerance=1e-4,
            )
        else:
            system = create_implicit_system(ff, modeller.topology, app, unit, cfg)

        integrator = openmm.LangevinMiddleIntegrator(
            cfg.temperature_k * unit.kelvin,
            cfg.friction_per_ps / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        simulation = make_platform_simulation(openmm, app, modeller.topology, system, integrator, cfg)
        simulation.context.setPositions(modeller.positions)

        initial_energy = state_energy(unit, simulation)
        minimized_energy, max_force, positions, iters_used, min_rounds, stop_reason = minimize_energy_with_optional_adaptive(unit, simulation, cfg)

        with out_pdb.open("w") as handle:
            app.PDBFile.writeFile(simulation.topology, positions, handle)

        if cfg.save_systems:
            with (system_dir / f"{seed_name}_{cfg.mode}.xml").open("w") as handle:
                handle.write(XmlSerializer.serialize(system))

        nw, ni = count_waters_ions(simulation.topology)
        rg, e2e, ccount, cvec = rg_e2e_contact(unit, positions, peptide_ca_indices(simulation.topology), cfg.contact_cutoff_A, cfg.contact_min_sep)
        remember_pdb_geometry(out_pdb, cfg.contact_cutoff_A, cfg.contact_min_sep, cvec, ccount, rg * 10.0, e2e * 10.0)

        return MinResult(
            seed_name=seed_name,
            input_pdb=str(pdb_path),
            output_pdb=str(out_pdb),
            success=True,
            error="",
            mode=cfg.mode,
            n_atoms=sum(1 for _ in simulation.topology.atoms()),
            n_residues=sum(1 for _ in simulation.topology.residues()),
            n_waters=nw,
            n_ions=ni,
            initial_energy_kj_mol=initial_energy,
            minimized_energy_kj_mol=minimized_energy,
            energy_drop_kj_mol=initial_energy - minimized_energy,
            max_force_kj_mol_nm=max_force,
            rg_nm=rg,
            end_to_end_nm=e2e,
            contact_count=ccount,
            minimization_iterations_used=int(iters_used),
            minimization_rounds=int(min_rounds),
            minimization_stop_reason=str(stop_reason),
        )

    except BaseException as exc:
        return MinResult(
            seed_name=seed_name,
            input_pdb=str(pdb_path),
            output_pdb=str(out_pdb),
            success=False,
            error=worker_failure_traceback(exc),
            mode=cfg.mode,
            n_atoms=0,
            n_residues=0,
            n_waters=0,
            n_ions=0,
            initial_energy_kj_mol=float("nan"),
            minimized_energy_kj_mol=float("nan"),
            energy_drop_kj_mol=float("nan"),
            max_force_kj_mol_nm=float("nan"),
            rg_nm=float("nan"),
            end_to_end_nm=float("nan"),
            contact_count=0,
        )


def write_min_csv(path: Path, rows: list[MinResult]):
    fields = list(MinResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def _resume_csv_done(csv_path, min_rows: int = 1) -> bool:
    """True if csv_path exists and has at least min_rows data rows (excluding header)."""
    try:
        p = Path(csv_path)
        if not p.exists():
            return False
        with p.open() as f:
            f.readline()  # skip header
            for i, _ in enumerate(f, 1):
                if i >= min_rows:
                    return True
        return False
    except Exception:
        return False


def minresults_from_score_csv(score_csv, mode_override: str = "implicit", out_dir=None) -> list:
    """Reconstruct MinResult list from a completed minimization score CSV (used for --resume).

    out_dir: when stored paths are relative to a prior run's CWD, pass args.out so that
    paths are also tried relative to out_dir.parent (the original run's working directory).
    """
    p = Path(score_csv)
    if not p.exists():
        return []
    try:
        import pandas as _pd
        df = _pd.read_csv(p)
    except Exception:
        return []
    _fallback_root = Path(out_dir).parent if out_dir is not None else None
    out = []
    for _, row in df.iterrows():
        if not bool(row.get("success", False)):
            continue
        pdb = Path(str(row.get("output_pdb", "")))
        if not pdb.exists():
            if _fallback_root is not None and not pdb.is_absolute():
                pdb = _fallback_root / pdb
            if not pdb.exists():
                continue
        try:
            out.append(MinResult(
                seed_name=str(row.get("seed_name", pdb.stem)),
                input_pdb=str(row.get("input_pdb", "")),
                output_pdb=str(pdb),
                success=True,
                error="",
                mode=str(row.get("mode", mode_override)),
                n_atoms=int(row.get("n_atoms", 0)),
                n_residues=int(row.get("n_residues", 0)),
                n_waters=int(row.get("n_waters", 0)),
                n_ions=int(row.get("n_ions", 0)),
                initial_energy_kj_mol=float(row.get("initial_energy_kj_mol", float("nan"))),
                minimized_energy_kj_mol=float(row.get("minimized_energy_kj_mol", float("nan"))),
                energy_drop_kj_mol=float(row.get("energy_drop_kj_mol", float("nan"))),
                max_force_kj_mol_nm=float(row.get("max_force_kj_mol_nm", float("nan"))),
                rg_nm=float(row.get("rg_nm", float("nan"))),
                end_to_end_nm=float(row.get("end_to_end_nm", float("nan"))),
                contact_count=int(row.get("contact_count", 0)),
                minimization_iterations_used=int(row.get("minimization_iterations_used", 0)),
                minimization_rounds=int(row.get("minimization_rounds", 0)),
                minimization_stop_reason=str(row.get("minimization_stop_reason", "")),
            ))
        except Exception:
            continue
    return out


class ReusableImplicitMinimizer:
    """Reuse one implicit OpenMM System/Simulation across same-topology seed PDBs."""
    def __init__(self, cfg: MinConfig, out_min_dir: Path, system_dir: Path):
        self.openmm, self.app, self.unit, self.XmlSerializer = import_openmm()
        self.cfg = cfg
        self.out_min_dir = Path(out_min_dir)
        self.system_dir = Path(system_dir)
        self.ff = make_forcefield(self.app, cfg)
        self.simulation = None
        self.signature = None
        self.ca_idx = []
        self.n_atoms = 0
        self.n_residues = 0
        self.n_waters = 0
        self.n_ions = 0
        self.context_builds = 0
        self.context_reuses = 0

    def _prepare_modeller(self, pdb_path: Path):
        pdb = self.app.PDBFile(str(pdb_path))
        modeller = self.app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(self.ff, pH=self.cfg.ph)
        return modeller

    def _build_context(self, modeller, seed_name: str):
        system = create_implicit_system(self.ff, modeller.topology, self.app, self.unit, self.cfg)
        integrator = self.openmm.LangevinMiddleIntegrator(
            self.cfg.temperature_k * self.unit.kelvin,
            self.cfg.friction_per_ps / self.unit.picosecond,
            0.002 * self.unit.picoseconds,
        )
        self.simulation = make_platform_simulation(self.openmm, self.app, modeller.topology, system, integrator, self.cfg)
        self.signature = topology_signature(modeller.topology)
        self.ca_idx = peptide_ca_indices(modeller.topology)
        self.n_atoms = sum(1 for _ in modeller.topology.atoms())
        self.n_residues = sum(1 for _ in modeller.topology.residues())
        self.n_waters, self.n_ions = count_waters_ions(modeller.topology)
        self.context_builds += 1
        if self.cfg.save_systems:
            with (self.system_dir / f"{seed_name}_{self.cfg.mode}.xml").open("w") as handle:
                handle.write(self.XmlSerializer.serialize(system))

    def minimize(self, pdb_path: Path) -> MinResult:
        pdb_path = Path(pdb_path)
        seed_name = pdb_path.stem
        out_pdb = self.out_min_dir / f"{seed_name}_{self.cfg.mode}_min.pdb"
        try:
            modeller = self._prepare_modeller(pdb_path)
            if self.simulation is None:
                self._build_context(modeller, seed_name)
            elif bool(getattr(self.cfg, "assume_same_implicit_topology", False)):
                self.context_reuses += 1
            else:
                sig = topology_signature(modeller.topology)
                if sig != self.signature:
                    self._build_context(modeller, seed_name)
                else:
                    self.context_reuses += 1
            self.simulation.context.setPositions(modeller.positions)

            initial_energy = state_energy(self.unit, self.simulation)
            minimized_energy, max_force, positions, iters_used, min_rounds, stop_reason = minimize_energy_with_optional_adaptive(self.unit, self.simulation, self.cfg)

            with out_pdb.open("w") as handle:
                self.app.PDBFile.writeFile(self.simulation.topology, positions, handle)

            rg, e2e, ccount, cvec = rg_e2e_contact(
                self.unit, positions, self.ca_idx, self.cfg.contact_cutoff_A, self.cfg.contact_min_sep
            )
            remember_pdb_geometry(out_pdb, self.cfg.contact_cutoff_A, self.cfg.contact_min_sep, cvec, ccount, rg * 10.0, e2e * 10.0)
            return MinResult(
                seed_name=seed_name,
                input_pdb=str(pdb_path),
                output_pdb=str(out_pdb),
                success=True,
                error="",
                mode=self.cfg.mode,
                n_atoms=self.n_atoms,
                n_residues=self.n_residues,
                n_waters=self.n_waters,
                n_ions=self.n_ions,
                initial_energy_kj_mol=initial_energy,
                minimized_energy_kj_mol=minimized_energy,
                energy_drop_kj_mol=initial_energy - minimized_energy,
                max_force_kj_mol_nm=max_force,
                rg_nm=rg,
                end_to_end_nm=e2e,
                contact_count=ccount,
                minimization_iterations_used=int(iters_used),
                minimization_rounds=int(min_rounds),
                minimization_stop_reason=str(stop_reason),
            )
        except BaseException as exc:
            return MinResult(
                seed_name=seed_name,
                input_pdb=str(pdb_path),
                output_pdb=str(out_pdb),
                success=False,
                error=worker_failure_traceback(exc),
                mode=self.cfg.mode,
                n_atoms=0,
                n_residues=0,
                n_waters=0,
                n_ions=0,
                initial_energy_kj_mol=float("nan"),
                minimized_energy_kj_mol=float("nan"),
                energy_drop_kj_mol=float("nan"),
                max_force_kj_mol_nm=float("nan"),
                rg_nm=float("nan"),
                end_to_end_nm=float("nan"),
                contact_count=0,
            )


def minimize_chunk(task):
    pdb_path_strs, out_min_dir_str, system_dir_str, cfg_dict = task
    cfg = MinConfig(**cfg_dict)
    out_min_dir = Path(out_min_dir_str)
    system_dir = Path(system_dir_str)
    if cfg.mode == "implicit":
        session = ReusableImplicitMinimizer(cfg, out_min_dir, system_dir)
        results = [session.minimize(Path(p)) for p in pdb_path_strs]
        stats = {"context_builds": session.context_builds, "context_reuses": session.context_reuses}
        return results, stats
    results = [minimize_one((str(p), str(out_min_dir), str(system_dir), asdict(cfg))) for p in pdb_path_strs]
    return results, {"context_builds": 0, "context_reuses": 0}


def make_minimization_chunks(pdbs: list[Path], n_chunks: int):
    n = len(pdbs)
    n_chunks = max(1, min(int(n_chunks), n))
    chunks = [[] for _ in range(n_chunks)]
    for i, pth in enumerate(pdbs):
        chunks[i % n_chunks].append(Path(pth))
    return [chunk for chunk in chunks if chunk]


def make_minimization_chunks_by_size(pdbs: list[Path], chunk_size: int):
    """Contiguous chunks for OpenMM minimization.

    The old parallel path used one large chunk per worker, which preserved
    context reuse but made the progress bar silent until a whole worker chunk
    finished. These smaller chunks give the dashboard a heartbeat while still
    allowing context reuse within each chunk.
    """
    pdbs = [Path(p) for p in pdbs]
    chunk_size = max(1, int(chunk_size))
    return [pdbs[i:i + chunk_size] for i in range(0, len(pdbs), chunk_size)]


def minimization_chunk_size(args, mode: str, min_jobs: int) -> int:
    requested = int(getattr(args, "min_chunk_size", 0) or 0)
    if requested > 0:
        return requested
    # Single-worker implicit minimization is handled with one reusable session
    # and per-PDB progress updates, so chunk size is only relevant for the
    # multiprocessing path. Keep small chunks for visible progress, but not
    # so small that context construction dominates on CPU runs.
    if str(mode).lower() == "implicit":
        # Favor context reuse over ultra-fine progress updates. The old value
        # was 4, which was responsive but rebuilt many OpenMM contexts on CPU.
        return 16 if int(min_jobs) > 1 else 1
    return 1


def run_minimization_batch(args, pdb_dir: Path, out_pdb_dir: Path, mode: str, score_csv: Path, ui_phase_key: Optional[str] = None, ui_cycle_step: Optional[str] = None, ui_cycle_round: Optional[int] = None, ui_cycle_total: Optional[int] = None):
    openmm, _app, _unit, _XmlSerializer = import_openmm()
    out_pdb_dir.mkdir(exist_ok=True)
    system_dir = Path(args.out) / f"{mode}_systems"
    system_dir.mkdir(exist_ok=True)

    manifest = Path(args.out) / "candidate_seeds.csv"
    if Path(pdb_dir).name == "candidate_seeds" and manifest.exists():
        try:
            mdf = pd.read_csv(manifest)
            col = "candidate_pdb_path" if "candidate_pdb_path" in mdf.columns else None
            pdbs = [Path(x) for x in mdf[col].dropna().astype(str).tolist()] if col else sorted(pdb_dir.glob("*.pdb"))
            pdbs = [p for p in pdbs if p.exists()]
        except Exception:
            pdbs = sorted(pdb_dir.glob("*.pdb"))
    else:
        pdbs = sorted(pdb_dir.glob("*.pdb"))
    if not pdbs:
        raise RuntimeError(f"No PDBs found in {pdb_dir}")

    cfg = make_min_config(args, mode)
    resolve_openmm_platform_name(openmm, cfg.platform)
    (Path(args.out) / f"{mode}_minimization_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    results = []
    n_fail_reported = 0
    min_jobs = get_min_jobs(args)
    context_builds = 0
    context_reuses = 0

    print_stage(
        f"OpenMM {mode} minimization",
        phase_key=ui_phase_key or ("explicit_min" if mode == "explicit" else "implicit_min"),
        cycle_step=ui_cycle_step,
        cycle_round=ui_cycle_round,
        cycle_total=ui_cycle_total,
    )
    ui_message(f"Seeds: {len(pdbs)} | OpenMM workers: {min_jobs} | platform: {args.platform} | constraints: {cfg.constraints}")
    progress = CliProgress(f"{mode} min", len(pdbs))
    live = LiveStats(f"live {mode}-minimized structures", enabled=args.live_hist, every=max(1, min(len(pdbs), getattr(args, "hist_every", 1000))))
    fail_reasons = {}
    ok_count = 0

    def _record_result(r):
        nonlocal n_fail_reported, ok_count
        results.append(r)
        if r.success:
            ok_count += 1
            live.add(
                rg_nm=r.rg_nm,
                end_to_end_nm=r.end_to_end_nm,
                contacts=r.contact_count,
                energy_kj_mol=r.minimized_energy_kj_mol,
                max_force=r.max_force_kj_mol_nm,
            )
        else:
            last = r.error.strip().splitlines()[-1] if r.error.strip() else "unknown"
            key = last.split(":", 1)[0]
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
            n_fail_reported += 1

    chunk_size = minimization_chunk_size(args, mode, min_jobs)
    progress.update(0, ok=0, fail=0, ctx="0/0")

    if min_jobs <= 1:
        # Important: do not send all PDBs through one huge minimize_chunk call.
        # That made implicit minimization look frozen until the complete batch
        # returned. This keeps one reusable implicit Context but updates the
        # dashboard after every minimized seed.
        if cfg.mode == "implicit":
            session = ReusableImplicitMinimizer(cfg, out_pdb_dir, system_dir)
            for pth in pdbs:
                r = session.minimize(Path(pth))
                context_builds = int(session.context_builds)
                context_reuses = int(session.context_reuses)
                _record_result(r)
                progress.update(len(results), ok=ok_count, fail=n_fail_reported, ctx=f"{context_builds}/{context_reuses}")
                live.maybe_print(len(results))
        else:
            for pth in pdbs:
                r = minimize_one((str(pth), str(out_pdb_dir), str(system_dir), asdict(cfg)))
                _record_result(r)
                progress.update(len(results), ok=ok_count, fail=n_fail_reported, ctx=f"{context_builds}/{context_reuses}")
                live.maybe_print(len(results))
    else:
        chunks = make_minimization_chunks_by_size(pdbs, chunk_size)
        chunk_tasks = [([str(p) for p in chunk], str(out_pdb_dir), str(system_dir), asdict(cfg)) for chunk in chunks]
        ui_message(f"Minimization progress granularity: {len(chunk_tasks)} chunks of up to {chunk_size} seed(s).")
        render_dashboard()

        ex = None
        futures = []
        try:
            ex = make_process_pool(min_jobs)
            futures = [ex.submit(minimize_chunk, t) for t in chunk_tasks]
            for fut in as_completed(futures):
                try:
                    chunk_results, stats = fut.result()
                except KeyboardInterrupt:
                    raise
                except BaseException as exc:
                    # A worker-side BaseException should count as a failed chunk,
                    # not masquerade as the user stopping the whole run.
                    n_fail_reported += 1
                    fail_reasons[type(exc).__name__] = fail_reasons.get(type(exc).__name__, 0) + 1
                    progress.update(len(results), ok=ok_count, fail=n_fail_reported, ctx=f"{context_builds}/{context_reuses}")
                    continue
                context_builds += int(stats.get("context_builds", 0))
                context_reuses += int(stats.get("context_reuses", 0))
                for r in chunk_results:
                    _record_result(r)
                    progress.update(len(results), ok=ok_count, fail=n_fail_reported, ctx=f"{context_builds}/{context_reuses}")
                    live.maybe_print(len(results))
        except KeyboardInterrupt:
            ui_message(f"Interrupted: terminating {mode} minimization workers.")
            render_dashboard()
            cancel_futures_now(futures)
            shutdown_process_pool_now(ex)
            raise SystemExit(130)
        finally:
            if ex is not None:
                shutdown_process_pool_now(ex)

    ok = ok_count
    progress.done(len(pdbs), ok=ok, fail=n_fail_reported)
    live.maybe_print(len(pdbs), force=True)
    if fail_reasons:
        ui_message("Failure summary: " + ", ".join(f"{k}={v}" for k, v in sorted(fail_reasons.items())[:6]))
        render_dashboard()
    if mode == "implicit":
        ui_message(f"OpenMM context reuse: built={context_builds}, reused={context_reuses}")
        render_dashboard()

    results = sorted(results, key=lambda r: r.seed_name)
    write_min_csv(score_csv, results)
    fail = [r for r in results if not r.success]
    if fail:
        write_min_csv(Path(args.out) / f"failed_{mode}_minimizations.csv", fail)
    ui_message(f"{mode} successful: {len(results) - len(fail)}/{len(results)}")
    render_dashboard()
    return results


def clone_args_with_overrides(args, **overrides):
    data = dict(vars(args))
    data.update(overrides)
    return argparse.Namespace(**data)


def select_tier_refinement_inputs(args, scout_results: list[MinResult], out_dir: Path, n_select: int):
    """Pick a smaller diverse subset of scout-minimized structures for refinement."""
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in scout_results if r.success and Path(r.output_pdb).exists()]
    n_select = max(1, min(int(n_select), len(ok))) if ok else 0
    if n_select <= 0:
        raise RuntimeError("Tiered implicit minimization: no successful scout minima to refine.")

    rows, contacts, descs = [], [], []
    for r in ok:
        geom = cached_pdb_geometry_features(args, Path(r.output_pdb))
        if geom is None:
            continue
        cvec, ccount, rg_A, e2e_A = geom
        rows.append(r)
        contacts.append(cvec)
        descs.append(np.array([rg_A, e2e_A, float(ccount), r.minimized_energy_kj_mol, r.max_force_kj_mol_nm], dtype=float))
    if not rows:
        raise RuntimeError("Tiered implicit minimization: scout minima had no readable CA geometry.")

    contacts_arr = np.vstack(contacts)
    desc_arr = np.vstack(descs)
    X = build_feature_matrix(contacts_arr, desc_arr[:, :3])
    energies = desc_arr[:, 3]
    forces = desc_arr[:, 4]
    badness = robust_z(energies) + 0.50 * robust_z(forces)
    selected_idx, labels = select_by_minibatch_or_farthest(X, n_select, args.cluster_method, badness)

    selected_rows = []
    for rank, idx in enumerate(selected_idx):
        idx = int(idx)
        src = Path(rows[idx].output_pdb)
        dst = out_dir / f"tier_refine_{rank:03d}_scoutcluster_{int(labels[idx]):03d}_{src.name}"
        shutil.copy2(src, dst)
        selected_rows.append({
            **asdict(rows[idx]),
            "tier_refine_rank": rank,
            "tier_scout_cluster_id": int(labels[idx]),
            "tier_scout_badness": float(badness[idx]),
            "tier_refine_input_pdb_path": str(dst),
        })
    pd.DataFrame(selected_rows).to_csv(Path(args.out) / "tiered_implicit_refine_inputs.csv", index=False)
    ui_message(f"Tiered implicit: selected {len(selected_rows)}/{len(rows)} scout minima for refinement.")
    render_dashboard()
    return out_dir, selected_rows


def run_tiered_implicit_minimization(args, candidate_dir: Path):
    """Scout-minimize many candidates, then refine only a diverse subset."""
    scout_dir = Path(args.out) / "aa_implicit_scout_minimized_pdbs"
    scout_scores = Path(args.out) / "implicit_scout_minimization_scores.csv"
    scout_args = clone_args_with_overrides(
        args,
        implicit_max_iterations=int(getattr(args, "tier_scout_iterations", 25)),
        adaptive_min=bool(getattr(args, "tier_scout_adaptive_min", False)),
    )
    print_stage("OpenMM implicit scout minimization", phase_key="implicit_min")
    ui_message(
        f"Tiered implicit scout: iterations={scout_args.implicit_max_iterations}, "
        f"adaptive={scout_args.adaptive_min}; candidates={len(list(Path(candidate_dir).glob('*.pdb')))}"
    )
    render_dashboard()
    scout_results = run_minimization_batch(scout_args, candidate_dir, scout_dir, "implicit", scout_scores)

    n_refine = int(getattr(args, "tier_refine_seeds", 0) or 0)
    if n_refine <= 0:
        # If the user explicitly enables tiering but does not choose a size,
        # make it actually tiered: refine roughly half the candidates, while
        # preserving at least ~2x the requested final survivor count.
        n_refine = min(
            int(getattr(args, "n_candidate_seeds", 0) or 0),
            max(int(getattr(args, "n_final_seeds", 20) or 20) * 2, int(getattr(args, "n_candidate_seeds", 0) or 0) // 2),
        )
    n_refine = max(1, n_refine)
    refine_input_dir = Path(args.out) / "tiered_implicit_refine_input_seeds"
    refine_input_dir, selected_rows = select_tier_refinement_inputs(args, scout_results, refine_input_dir, n_refine)

    refine_dir = Path(args.out) / "aa_implicit_minimized_pdbs"
    refine_scores = Path(args.out) / "implicit_minimization_scores.csv"
    refine_iterations = int(getattr(args, "tier_refine_iterations", 0) or 0)
    refine_args = args
    if refine_iterations > 0:
        refine_args = clone_args_with_overrides(args, implicit_max_iterations=refine_iterations)
    print_stage("OpenMM implicit refine minimization", phase_key="implicit_min")
    ui_message(
        f"Tiered implicit refine: inputs={len(selected_rows)}, iterations={refine_args.implicit_max_iterations}, "
        f"adaptive={refine_args.adaptive_min}."
    )
    render_dashboard()
    refined_results = run_minimization_batch(refine_args, refine_input_dir, refine_dir, "implicit", refine_scores)

    summary = {
        "enabled": True,
        "scout_iterations": int(scout_args.implicit_max_iterations),
        "scout_adaptive_min": bool(scout_args.adaptive_min),
        "scout_total": int(len(scout_results)),
        "scout_success": int(sum(1 for r in scout_results if r.success)),
        "refine_inputs": int(len(selected_rows)),
        "refine_iterations": int(refine_args.implicit_max_iterations),
        "refine_total": int(len(refined_results)),
        "refine_success": int(sum(1 for r in refined_results if r.success)),
        "scout_scores_csv": str(scout_scores),
        "refine_scores_csv": str(refine_scores),
        "refine_input_manifest": str(Path(args.out) / "tiered_implicit_refine_inputs.csv"),
    }
    (Path(args.out) / "tiered_implicit_summary.json").write_text(json.dumps(summary, indent=2))
    return refined_results


# -----------------------------
# Fast implicit basin hopping
# -----------------------------

@dataclass
class HopRecord:
    parent_seed: str
    hop_index: int
    pdb_path: str
    success: bool
    error: str
    initial_energy_kj_mol: float
    minimized_energy_kj_mol: float
    max_force_kj_mol_nm: float
    rg_nm: float
    end_to_end_nm: float
    contact_count: int
    minimization_iterations_used: int = 0
    minimization_rounds: int = 0
    minimization_stop_reason: str = ""


def stable_parent_seed(base_seed: int, parent: str) -> int:
    # Python's hash() is intentionally randomized between processes. CRC32 is
    # stable and sufficient for deterministic velocity seeds here.
    import zlib
    return (int(base_seed) + int(zlib.crc32(str(parent).encode("utf-8")))) % (2**31 - 1)


class ReusableBasinHopper:
    """Reuse one implicit OpenMM System/Simulation across same-topology BH parents."""
    def __init__(self, cfg: MinConfig, out_dir: Path):
        self.openmm, self.app, self.unit, _XmlSerializer = import_openmm()
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.ff = make_forcefield(self.app, cfg)
        self.simulation = None
        self.signature = None
        self.integrator_signature = None
        self.ca_idx = []
        self.context_builds = 0
        self.context_reuses = 0

    def _prepare_modeller(self, pdb_path: Path):
        pdb = self.app.PDBFile(str(pdb_path))
        modeller = self.app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(self.ff, pH=self.cfg.ph)
        return modeller

    def _bh_integrator_signature(self, bh: dict):
        return (
            float(bh["temperature"]),
            float(bh["friction"]),
            float(bh["timestep_ps"]),
            float(bh.get("hmr_mass_amu", 0.0) or 0.0),
            bool(bh.get("hmr_auto_constraints", True)),
            float(bh.get("constraint_tolerance", 1.0e-5) or 0.0),
        )

    def _build_context(self, modeller, bh: dict):
        hmr_mass = float(bh.get("hmr_mass_amu", 0.0) or 0.0)
        force_hbonds = bool(bh.get("hmr_auto_constraints", True)) and hmr_mass > 0.0
        system = create_implicit_system(
            self.ff,
            modeller.topology,
            self.app,
            self.unit,
            self.cfg,
            hydrogen_mass_amu=hmr_mass,
            force_hbonds=force_hbonds,
        )
        integrator = self.openmm.LangevinMiddleIntegrator(
            float(bh["temperature"]) * self.unit.kelvin,
            float(bh["friction"]) / self.unit.picosecond,
            float(bh["timestep_ps"]) * self.unit.picoseconds,
        )
        constraint_tol = float(bh.get("constraint_tolerance", 1.0e-5) or 0.0)
        if constraint_tol > 0.0 and hasattr(integrator, "setConstraintTolerance"):
            integrator.setConstraintTolerance(constraint_tol)
        self.simulation = make_platform_simulation(self.openmm, self.app, modeller.topology, system, integrator, self.cfg)
        self.signature = topology_signature(modeller.topology)
        self.integrator_signature = self._bh_integrator_signature(bh)
        self.ca_idx = peptide_ca_indices(modeller.topology)
        self.context_builds += 1

    def _ensure_context(self, modeller, bh: dict):
        sig = topology_signature(modeller.topology)
        integ_sig = self._bh_integrator_signature(bh)
        if self.simulation is None or sig != self.signature or integ_sig != self.integrator_signature:
            self._build_context(modeller, bh)
        else:
            self.context_reuses += 1

    def hop_parent(self, pdb_path: Path, bh: dict) -> list[HopRecord]:
        pdb_path = Path(pdb_path)
        parent = pdb_path.stem
        records = []
        try:
            modeller = self._prepare_modeller(pdb_path)
            self._ensure_context(modeller, bh)
            simulation = self.simulation
            unit = self.unit
            app = self.app
            simulation.context.setPositions(modeller.positions)

            # Parent relaxation can also use adaptive minimization, but with the
            # BH-specific iteration budget.
            parent_e, parent_force, min0_positions, parent_iters, parent_rounds, parent_stop = minimize_energy_with_optional_adaptive(
                unit, simulation, self.cfg, max_iterations=int(bh["initial_min_iterations"])
            )
            rng = np.random.default_rng(stable_parent_seed(int(bh["seed"]), parent))

            include_parent = bool(bh.get("include_parent", True))
            hop_offset = int(bh.get("hop_offset", 1))
            n_proposals = max(0, int(bh["steps"]))
            hop_plan = []
            if include_parent:
                hop_plan.append((0, True))
            hop_plan.extend((hop_offset + i, False) for i in range(n_proposals))

            for hop_i, is_parent_endpoint in hop_plan:
                try:
                    if is_parent_endpoint:
                        simulation.context.setPositions(min0_positions)
                        e_before = parent_e
                        e_after = parent_e
                        max_force = parent_force
                        positions = min0_positions
                        iters_used = parent_iters
                        min_rounds = parent_rounds
                        stop_reason = "initial_" + str(parent_stop)
                    else:
                        if bool(bh["restart_from_parent"]):
                            simulation.context.setPositions(min0_positions)
                        simulation.context.setVelocitiesToTemperature(
                            float(bh["temperature"]) * unit.kelvin,
                            int(rng.integers(0, 2**31 - 1)),
                        )
                        simulation.step(int(bh["md_steps"]))

                        e_before = state_energy(unit, simulation)
                        e_after, max_force, positions, iters_used, min_rounds, stop_reason = minimize_energy_with_optional_adaptive(
                            unit, simulation, self.cfg, max_iterations=int(bh["min_iterations"])
                        )

                    rg, e2e, ccount, cvec = rg_e2e_contact(
                        unit, positions, self.ca_idx, self.cfg.contact_cutoff_A, self.cfg.contact_min_sep
                    )

                    out_pdb = self.out_dir / f"bh_{parent}_hop_{hop_i:03d}.pdb"
                    with out_pdb.open("w") as handle:
                        app.PDBFile.writeFile(simulation.topology, positions, handle)

                    records.append(HopRecord(
                        parent_seed=parent,
                        hop_index=int(hop_i),
                        pdb_path=str(out_pdb),
                        success=True,
                        error="",
                        initial_energy_kj_mol=e_before,
                        minimized_energy_kj_mol=e_after,
                        max_force_kj_mol_nm=max_force,
                        rg_nm=rg,
                        end_to_end_nm=e2e,
                        contact_count=ccount,
                        minimization_iterations_used=int(iters_used),
                        minimization_rounds=int(min_rounds),
                        minimization_stop_reason=str(stop_reason),
                    ))
                    remember_pdb_geometry(
                        out_pdb,
                        self.cfg.contact_cutoff_A,
                        self.cfg.contact_min_sep,
                        cvec,
                        int(ccount),
                        float(rg) * 10.0,
                        float(e2e) * 10.0,
                    )

                except BaseException as exc:
                    records.append(HopRecord(
                        parent_seed=parent,
                        hop_index=int(hop_i),
                        pdb_path="",
                        success=False,
                        error=worker_failure_traceback(exc),
                        initial_energy_kj_mol=float("nan"),
                        minimized_energy_kj_mol=float("nan"),
                        max_force_kj_mol_nm=float("nan"),
                        rg_nm=float("nan"),
                        end_to_end_nm=float("nan"),
                        contact_count=0,
                    ))

        except BaseException as exc:
            records.append(HopRecord(
                parent_seed=parent,
                hop_index=-1,
                pdb_path="",
                success=False,
                error=worker_failure_traceback(exc),
                initial_energy_kj_mol=float("nan"),
                minimized_energy_kj_mol=float("nan"),
                max_force_kj_mol_nm=float("nan"),
                rg_nm=float("nan"),
                end_to_end_nm=float("nan"),
                contact_count=0,
            ))
        return records


def basin_hop_one(task):
    pdb_path_str, out_dir_str, cfg_dict, bh_dict = task
    cfg = MinConfig(**cfg_dict)
    session = ReusableBasinHopper(cfg, Path(out_dir_str))
    return session.hop_parent(Path(pdb_path_str), bh_dict)


def basin_hop_chunk(task):
    """Process several BH parents in one worker while reusing one Context."""
    items, out_dir_str, cfg_dict = task
    cfg = MinConfig(**cfg_dict)
    session = ReusableBasinHopper(cfg, Path(out_dir_str))
    all_records = []
    for pdb_path_str, bh in items:
        all_records.extend(session.hop_parent(Path(pdb_path_str), bh))
    stats = {
        "parents": len(items),
        "context_builds": session.context_builds,
        "context_reuses": session.context_reuses,
    }
    return all_records, stats


def make_basin_hop_batches(items, chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def select_basin_hop_parent_pdbs(args, pdbs: list[Path]):
    n_parent = int(getattr(args, "bh_parent_seeds", 0) or 0)
    if n_parent <= 0 or n_parent >= len(pdbs):
        return list(pdbs), []
    rows = []
    contacts = []
    descs = []
    valid = []
    for p in pdbs:
        geom = cached_pdb_geometry_features(args, p)
        if geom is None:
            continue
        cvec, ccount, rg_A, e2e_A = geom
        valid.append(p)
        contacts.append(cvec)
        descs.append(np.array([rg_A, e2e_A, float(ccount)], dtype=float))
        rows.append({"pdb_path": str(p), "rg_A": rg_A, "end_to_end_A": e2e_A, "contact_count": ccount})
    if len(valid) <= n_parent:
        return valid, rows
    X = build_feature_matrix(np.vstack(contacts), np.vstack(descs))
    selected_idx, labels = select_by_minibatch_or_farthest(X, n_parent, args.cluster_method, None)
    selected = []
    selected_rows = []
    for rank, idx in enumerate(selected_idx):
        idx = int(idx)
        p = valid[idx]
        selected.append(p)
        row = dict(rows[idx])
        row["bh_parent_rank"] = rank
        row["bh_parent_cluster"] = int(labels[idx]) if len(labels) > idx else rank
        selected_rows.append(row)
    return selected, selected_rows



def _hop_coarse_key(record: HopRecord, rg_bin_nm: float = 0.05, e2e_bin_nm: float = 0.05):
    if not record.success or not np.isfinite([record.rg_nm, record.end_to_end_nm, record.minimized_energy_kj_mol]).all():
        return None
    return (
        int(round(float(record.rg_nm) / rg_bin_nm)),
        int(round(float(record.end_to_end_nm) / e2e_bin_nm)),
        int(record.contact_count),
    )


def _transition_entropy(keys: list):
    keys = [k for k in keys if k is not None]
    if not keys:
        return 0.0
    counts = np.fromiter(Counter(keys).values(), dtype=float)
    p = counts / max(1.0, float(counts.sum()))
    return max(0.0, float(-np.sum(p * np.log(p + 1e-12))))


def _allocate_adaptive_hop_steps(pdbs: list[Path], records: list[HopRecord], remaining_budget: int):
    """Allocate remaining BH proposals toward parents that generated diverse/low-energy endpoints."""
    remaining_budget = max(0, int(remaining_budget))
    if remaining_budget <= 0 or not pdbs:
        return {p.stem: 0 for p in pdbs}

    parent_to_records = {p.stem: [] for p in pdbs}
    for r in records:
        if r.success and int(r.hop_index) > 0 and r.parent_seed in parent_to_records:
            parent_to_records[r.parent_seed].append(r)

    all_energies = [r.minimized_energy_kj_mol for rs in parent_to_records.values() for r in rs if np.isfinite(r.minimized_energy_kj_mol)]
    med_e = float(np.nanmedian(all_energies)) if all_energies else 0.0
    mad_e = float(np.nanmedian(np.abs(np.asarray(all_energies) - med_e))) if all_energies else 1.0
    if not np.isfinite(mad_e) or mad_e <= 1e-12:
        mad_e = 1.0

    rows = []
    weights = []
    for pth in pdbs:
        parent = pth.stem
        rs = parent_to_records.get(parent, [])
        keys = [_hop_coarse_key(r) for r in rs]
        unique_keys = {k for k in keys if k is not None}
        entropy = _transition_entropy(keys)
        best_e = min([r.minimized_energy_kj_mol for r in rs if np.isfinite(r.minimized_energy_kj_mol)], default=med_e)
        low_e_bonus = max(0.0, (med_e - best_e) / (3.0 * mad_e))
        # Keep a nonzero exploration floor, but favor parents that actually go somewhere.
        weight = 0.25 + float(len(unique_keys)) + 0.50 * entropy + 0.50 * low_e_bonus
        weights.append(max(0.05, weight))
        rows.append({
            "parent_seed": parent,
            "probe_successes": len(rs),
            "probe_unique_endpoint_keys": len(unique_keys),
            "probe_transition_entropy": entropy,
            "probe_best_energy_kj_mol": best_e,
            "adaptive_weight": weight,
        })

    weights_arr = np.asarray(weights, dtype=float)
    weights_arr /= weights_arr.sum() if weights_arr.sum() > 0 else len(weights_arr)
    raw = weights_arr * remaining_budget
    alloc = np.floor(raw).astype(int)
    leftover = remaining_budget - int(alloc.sum())
    if leftover > 0:
        order = np.argsort(-(raw - alloc))
        for idx in order[:leftover]:
            alloc[int(idx)] += 1

    allocation = {p.stem: int(alloc[i]) for i, p in enumerate(pdbs)}
    for i, row in enumerate(rows):
        row["allocated_extra_steps"] = int(alloc[i])
    return allocation, rows


def _run_basin_hop_task_batch(tasks, min_jobs: int, add_records, progress: CliProgress, live: LiveStats, start_done: int = 0):
    """Run reusable basin-hop batches and update progress by parent count."""
    done = int(start_done)
    context_builds = 0
    context_reuses = 0

    if min_jobs <= 1:
        # Local path keeps one reusable context per batch and still updates the
        # dashboard after every parent. For the usual single-GPU case this is the
        # best compromise: reuse the expensive Context, keep the UI alive.
        for items, out_dir_str, cfg_dict in tasks:
            cfg = MinConfig(**cfg_dict)
            session = ReusableBasinHopper(cfg, Path(out_dir_str))
            for pdb_path_str, bh in items:
                recs = session.hop_parent(Path(pdb_path_str), bh)
                add_records(recs)
                done += 1
                context_builds = int(session.context_builds)
                context_reuses = int(session.context_reuses)
                progress.update(done, endpoints=add_records.success_count, fail=add_records.fail_count, ctx=f"{context_builds}/{context_reuses}")
                live.maybe_print(done)
        return done

    ex = None
    futures = []
    try:
        ex = make_process_pool(min_jobs)
        futures = [ex.submit(basin_hop_chunk, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                recs, stats = fut.result()
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                add_records([HopRecord(
                    parent_seed="worker_chunk",
                    hop_index=-1,
                    pdb_path="",
                    success=False,
                    error="Escaped worker exception: " + repr(exc),
                    initial_energy_kj_mol=float("nan"),
                    minimized_energy_kj_mol=float("nan"),
                    max_force_kj_mol_nm=float("nan"),
                    rg_nm=float("nan"),
                    end_to_end_nm=float("nan"),
                    contact_count=0,
                )])
                progress.update(done, endpoints=add_records.success_count, fail=add_records.fail_count, ctx=f"{context_builds}/{context_reuses}")
                continue
            add_records(recs)
            done += int(stats.get("parents", 1))
            context_builds += int(stats.get("context_builds", 0))
            context_reuses += int(stats.get("context_reuses", 0))
            progress.update(done, endpoints=add_records.success_count, fail=add_records.fail_count, ctx=f"{context_builds}/{context_reuses}")
            live.maybe_print(done)
        return done
    except KeyboardInterrupt:
        ui_message("Interrupted: terminating basin-hop workers.")
        render_dashboard()
        cancel_futures_now(futures)
        shutdown_process_pool_now(ex)
        raise SystemExit(130)
    finally:
        if ex is not None:
            shutdown_process_pool_now(ex)


def basin_hop_chunk_size(args, min_jobs: int, n_items: int) -> int:
    requested = int(getattr(args, "bh_chunk_size", 0) or 0)
    if requested > 0:
        return requested
    # Single-worker/GPU path uses one batch and updates per parent inside the
    # batch, preserving context reuse without freezing progress. Parallel CPU
    # path uses small batches to balance context reuse with dashboard heartbeat.
    if int(min_jobs) <= 1:
        return max(1, int(n_items))
    return 4


def run_basin_hopping(
    args,
    seed_dir: Path,
    out_dir: Optional[Path] = None,
    hop_csv: Optional[Path] = None,
    label: str = "Implicit basin hopping / Monte Carlo basin search",
    cycle_round: Optional[int] = None,
    cycle_total: Optional[int] = None,
):
    import_openmm()
    out_dir = Path(out_dir) if out_dir is not None else Path(args.out) / "basin_hop_minima"
    hop_csv = Path(hop_csv) if hop_csv is not None else Path(args.out) / "basin_hop_minima.csv"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    hop_csv.parent.mkdir(parents=True, exist_ok=True)

    # Prefer the explicit manifest when the source is candidate_seeds to avoid stale PDBs.
    manifest = Path(args.out) / "candidate_seeds.csv"
    if Path(seed_dir).name == "candidate_seeds" and manifest.exists():
        try:
            mdf = pd.read_csv(manifest)
            if "candidate_pdb_path" in mdf.columns:
                pdbs_all = [Path(x) for x in mdf["candidate_pdb_path"].dropna().astype(str).tolist()]
                pdbs_all = [p for p in pdbs_all if p.exists()]
            else:
                pdbs_all = sorted(seed_dir.glob("*.pdb"))
        except Exception:
            pdbs_all = sorted(seed_dir.glob("*.pdb"))
    else:
        pdbs_all = sorted(seed_dir.glob("*.pdb"))
    if not pdbs_all:
        raise RuntimeError(f"No PDBs for basin hopping in {seed_dir}")

    pdbs, bh_parent_rows = select_basin_hop_parent_pdbs(args, pdbs_all)
    parent_csv = (Path(args.out) / "basin_hop_parent_seeds.csv") if hop_csv.name == "basin_hop_minima.csv" else hop_csv.with_name(hop_csv.stem + "_parent_seeds.csv")
    if bh_parent_rows:
        pd.DataFrame(bh_parent_rows).to_csv(parent_csv, index=False)

    cfg = make_min_config(args, "implicit")
    base_bh = {
        "steps": args.bh_steps,
        "md_steps": args.bh_md_steps,
        "temperature": args.bh_temperature,
        "friction": args.bh_friction,
        "timestep_ps": args.bh_timestep,
        "hmr_mass_amu": args.bh_hmr_mass,
        "hmr_auto_constraints": args.bh_hmr_auto_constraints,
        "constraint_tolerance": args.bh_constraint_tolerance,
        "initial_min_iterations": args.bh_initial_min_iterations,
        "min_iterations": args.bh_min_iterations,
        "restart_from_parent": args.bh_restart_from_parent,
        "seed": args.seed,
        "include_parent": True,
        "hop_offset": 1,
    }
    config_path = (Path(args.out) / "basin_hop_config.json") if hop_csv.name == "basin_hop_minima.csv" else hop_csv.with_name(hop_csv.stem + "_config.json")
    config_path.write_text(json.dumps({**asdict(cfg), **base_bh, "smart_search": bool(getattr(args, "smart_search", True)), "seed_dir": str(seed_dir), "out_dir": str(out_dir), "hop_csv": str(hop_csv)}, indent=2))

    min_jobs = get_min_jobs(args)
    smart = bool(getattr(args, "smart_search", True)) and int(args.bh_steps) >= 4 and len(pdbs) > 1
    print_stage(label, phase_key="exploration", cycle_step="bh", cycle_round=cycle_round, cycle_total=cycle_total)
    parent_note = f"selected {len(pdbs)} / {len(pdbs_all)} parents" if len(pdbs) != len(pdbs_all) else f"{len(pdbs)} parents"
    mode_note = "adaptive" if smart else "uniform"
    hmr_note = f" | HMR={args.bh_hmr_mass:g} amu" if float(args.bh_hmr_mass or 0.0) > 0.0 else ""
    ui_message(
        f"Parent seeds: {parent_note} | OpenMM workers: {min_jobs} | mode: {mode_note} | "
        f"requested hops/seed: {args.bh_steps} | dt={args.bh_timestep:g} ps{hmr_note}"
    )

    all_records = []
    fail_reasons = {}

    class AddRecords:
        success_count = 0
        fail_count = 0
        def __call__(self, recs):
            all_records.extend(recs)
            for r in recs:
                if r.success:
                    self.success_count += 1
                    live.add(
                        rg_nm=r.rg_nm,
                        end_to_end_nm=r.end_to_end_nm,
                        contacts=r.contact_count,
                        energy_kj_mol=r.minimized_energy_kj_mol,
                        max_force=r.max_force_kj_mol_nm,
                    )
                else:
                    last = r.error.strip().splitlines()[-1] if r.error.strip() else "unknown"
                    key = last.split(":", 1)[0]
                    fail_reasons[key] = fail_reasons.get(key, 0) + 1
                    self.fail_count += 1

    add_records = AddRecords()
    live = LiveStats("live basin-hop endpoint structures", enabled=args.live_hist, every=max(1, min(len(pdbs), getattr(args, "hist_every", 1000))))

    if smart:
        probe_steps = min(2, int(args.bh_steps))
        remaining_budget = len(pdbs) * max(0, int(args.bh_steps) - probe_steps)
        probe_items = []
        for p in pdbs:
            bh = dict(base_bh)
            bh.update({"steps": probe_steps, "include_parent": True, "hop_offset": 1})
            probe_items.append((str(p), bh))
        bh_chunk = basin_hop_chunk_size(args, min_jobs, len(probe_items))
        probe_tasks = [(batch, str(out_dir), asdict(cfg)) for batch in make_basin_hop_batches(probe_items, bh_chunk)]
        progress = CliProgress("basin hop", len(probe_items) + len(pdbs))
        ui_message(f"Adaptive probe: {probe_steps} proposals/parent, then reallocating {remaining_budget} extra proposals.")
        render_dashboard()
        done = _run_basin_hop_task_batch(probe_tasks, min_jobs, add_records, progress, live, 0)

        allocation, alloc_rows = _allocate_adaptive_hop_steps(pdbs, all_records, remaining_budget)
        alloc_csv = (Path(args.out) / "basin_hop_adaptive_allocation.csv") if hop_csv.name == "basin_hop_minima.csv" else hop_csv.with_name(hop_csv.stem + "_adaptive_allocation.csv")
        pd.DataFrame(alloc_rows).to_csv(alloc_csv, index=False)
        extra_items = []
        for p in pdbs:
            n_extra = int(allocation.get(p.stem, 0))
            if n_extra <= 0:
                continue
            bh = dict(base_bh)
            bh.update({"steps": n_extra, "include_parent": False, "hop_offset": probe_steps + 1})
            extra_items.append((str(p), bh))
        extra_tasks = [(batch, str(out_dir), asdict(cfg)) for batch in make_basin_hop_batches(extra_items, bh_chunk)]
        progress.total = done + len(extra_items)
        ui_message(f"Adaptive extra phase: {len(extra_items)} productive/frontier parents receive extra proposals.")
        render_dashboard()
        if extra_tasks:
            done = _run_basin_hop_task_batch(extra_tasks, min_jobs, add_records, progress, live, done)
        progress.done(done, endpoints=add_records.success_count, fail=add_records.fail_count)
    else:
        items = []
        for p in pdbs:
            bh = dict(base_bh)
            bh.update({"steps": int(args.bh_steps), "include_parent": True, "hop_offset": 1})
            items.append((str(p), bh))
        bh_chunk = basin_hop_chunk_size(args, min_jobs, len(items))
        tasks = [(batch, str(out_dir), asdict(cfg)) for batch in make_basin_hop_batches(items, bh_chunk)]
        progress = CliProgress("basin hop", len(items))
        done = _run_basin_hop_task_batch(tasks, min_jobs, add_records, progress, live, 0)
        progress.done(done, endpoints=add_records.success_count, fail=add_records.fail_count)

    live.maybe_print(max(1, len(pdbs)), force=True)
    if fail_reasons:
        ui_message("Basin-hop failure summary: " + ", ".join(f"{k}={v}" for k, v in sorted(fail_reasons.items())[:6]))
        render_dashboard()

    df = pd.DataFrame([asdict(r) for r in all_records])
    df.to_csv(hop_csv, index=False)

    ok = int(df["success"].sum()) if len(df) and "success" in df.columns else 0
    ui_message(f"Basin-hop successful endpoints: {ok}/{len(df)}")
    render_dashboard()
    return out_dir


# -----------------------------
# ANM/NMA expansion
# -----------------------------

def read_pdb_atoms_for_displacement(path: Path):
    atoms = []
    ca_by_reskey = {}
    lines = path.read_text().splitlines()
    for line_no, line in enumerate(lines):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21:22].strip() or "A"
        try:
            resseq = int(line[22:26])
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
        except ValueError:
            continue
        reskey = (chain, resseq, resname)
        atoms.append({"line_no": line_no, "line": line, "atom_name": atom_name, "resname": resname, "reskey": reskey, "coord": coord})
        if atom_name == "CA" and resname not in {"HOH", "WAT", "SOL"}:
            ca_by_reskey[reskey] = coord
    reskeys = list(ca_by_reskey.keys())
    ca = np.array([ca_by_reskey[k] for k in reskeys], dtype=float)
    return lines, atoms, reskeys, ca


def write_displaced_pdb(parent_pdb: Path, out_pdb: Path, displacement_by_reskey: dict):
    lines, atoms, _, _ = read_pdb_atoms_for_displacement(parent_pdb)
    new_lines = list(lines)
    for atom in atoms:
        disp = displacement_by_reskey.get(atom["reskey"])
        if disp is None:
            continue
        xyz = atom["coord"] + disp
        old = atom["line"]
        new_lines[atom["line_no"]] = f"{old[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{old[54:]}"
    out_pdb.write_text("\n".join(new_lines) + "\n")


def compute_anm_modes(ca: np.ndarray, cutoff_A: float, n_modes: int):
    ca = np.asarray(ca, dtype=float)
    n = len(ca)
    if n < 3:
        return []
    h = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            rij = ca[j] - ca[i]
            dist = float(np.linalg.norm(rij))
            if dist <= 1e-8 or dist > cutoff_A:
                continue
            u = rij / dist
            block = -np.outer(u, u)
            si = slice(3 * i, 3 * i + 3)
            sj = slice(3 * j, 3 * j + 3)
            h[si, sj] += block
            h[sj, si] += block
            h[si, si] -= block
            h[sj, sj] -= block
    try:
        vals, vecs = np.linalg.eigh(h)
    except np.linalg.LinAlgError:
        return []
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    nonzero = np.where(vals > 1e-6)[0]
    if len(nonzero) == 0:
        nonzero = np.arange(min(6, len(vals)), len(vals))
    modes = []
    for k in nonzero[:n_modes]:
        v = vecs[:, k].reshape(n, 3)
        rms = float(np.sqrt((v * v).sum(axis=1).mean()))
        if np.isfinite(rms) and rms > 1e-12:
            modes.append((float(vals[k]), v / rms))
    return modes


def parse_amplitudes(text: str):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def nma_expand_from_dir(args, seed_dir: Path):
    out_dir = Path(args.out) / "nma_probe_seeds"
    out_dir.mkdir(exist_ok=True)
    pdbs = sorted(seed_dir.glob("*.pdb"))
    if not pdbs:
        return out_dir

    amps = parse_amplitudes(args.nma_amplitudes)
    written = 0
    print_stage("ANM/NMA local expansion", phase_key="exploration", cycle_step="nma")
    print(f"Source PDBs: {len(pdbs)} | modes: {args.nma_modes} | amplitudes: {args.nma_amplitudes}")
    progress = CliProgress("nma expand", len(pdbs))
    for pi, parent in enumerate(pdbs, start=1):
        try:
            _, _, reskeys, ca = read_pdb_atoms_for_displacement(parent)
            modes = compute_anm_modes(ca, args.nma_cutoff, args.nma_modes)
            for mode_i, (_, mode_vec) in enumerate(modes, start=1):
                for amp in amps:
                    for sign in (-1.0, 1.0):
                        disp = {reskey: sign * amp * mode_vec[i] for i, reskey in enumerate(reskeys)}
                        tag = "p" if sign > 0 else "m"
                        out_pdb = out_dir / f"nma_{parent.stem}_mode{mode_i:02d}_{tag}{amp:.2f}A.pdb"
                        write_displaced_pdb(parent, out_pdb, disp)
                        written += 1
        except Exception:
            pass
        progress.update(pi, probes=written)
    progress.done(len(pdbs), probes=written)
    print(f"NMA probes written: {written}")
    return out_dir


# -----------------------------
# Survivor selection
# -----------------------------

def read_pdb_ca_coords_simple(path: Path):
    coords = []
    with path.open() as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() == "CA" and line[17:20].strip() not in {"HOH", "WAT", "SOL"}:
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    pass
    return np.array(coords, dtype=float)



def _geometry_cache_key(path: Path, cutoff_A: float, min_sep: int):
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size), float(cutoff_A), int(min_sep))


def remember_pdb_geometry(path: Path, cutoff_A: float, min_sep: int, cvec: np.ndarray, ccount: int, rg_A: float, e2e_A: float):
    """Register freshly computed/written geometry in memory and persistent cache."""
    global _FEATURE_CACHE_DIRTY
    key = _geometry_cache_key(path, cutoff_A, min_sep)
    if key is None:
        return
    geom = (np.asarray(cvec, dtype=np.int8), int(ccount), float(rg_A), float(e2e_A))
    _PDB_GEOM_CACHE[key] = geom
    _FEATURE_CACHE_MAP[key] = geom
    _FEATURE_CACHE_DIRTY = True


def _feature_cache_path(args) -> Optional[Path]:
    out = getattr(args, "out", None)
    if out is None:
        return None
    return Path(out) / "feature_cache" / "structure_geometry_features.npz"


def load_feature_cache(args):
    global _FEATURE_CACHE_LOADED_FOR, _FEATURE_CACHE_MAP
    path = _feature_cache_path(args)
    if path is None:
        return
    cache_id = str(path.resolve())
    if _FEATURE_CACHE_LOADED_FOR == cache_id:
        return
    _FEATURE_CACHE_LOADED_FOR = cache_id
    _FEATURE_CACHE_MAP = {}
    if not path.exists():
        return
    try:
        data = np.load(path, allow_pickle=False)
        paths = data["paths"].astype(str)
        mtimes = data["mtimes_ns"].astype(np.int64)
        sizes = data["sizes"].astype(np.int64)
        cutoff = float(data["cutoff_A"][0]) if "cutoff_A" in data.files and len(data["cutoff_A"]) else float(getattr(args, "contact_cutoff", 8.0))
        min_sep = int(data["min_sep"][0]) if "min_sep" in data.files and len(data["min_sep"]) else int(getattr(args, "contact_min_sep", 3))
        contacts = data["contacts"].astype(np.int8, copy=False)
        ccounts = data["contact_count"].astype(np.int32)
        rg = data["rg_A"].astype(np.float32)
        e2e = data["end_to_end_A"].astype(np.float32)
        for i, path_str in enumerate(paths):
            key = (str(path_str), int(mtimes[i]), int(sizes[i]), float(cutoff), int(min_sep))
            _FEATURE_CACHE_MAP[key] = (contacts[i].astype(np.int8, copy=False), int(ccounts[i]), float(rg[i]), float(e2e[i]))
    except Exception:
        # Corrupt/stale cache should never kill a simulation run.
        _FEATURE_CACHE_MAP = {}


def save_feature_cache(args):
    global _FEATURE_CACHE_DIRTY
    path = _feature_cache_path(args)
    if path is None or not _FEATURE_CACHE_DIRTY or not _FEATURE_CACHE_MAP:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        items = list(_FEATURE_CACHE_MAP.items())
        # Only save entries matching current contact parameters and same vector length.
        cutoff = float(getattr(args, "contact_cutoff", 8.0))
        min_sep = int(getattr(args, "contact_min_sep", 3))
        filtered = [(k, v) for k, v in items if float(k[3]) == cutoff and int(k[4]) == min_sep]
        if not filtered:
            return
        lengths = Counter(len(v[0]) for _k, v in filtered)
        target_len = lengths.most_common(1)[0][0]
        filtered = [(k, v) for k, v in filtered if len(v[0]) == target_len]
        paths = np.asarray([k[0] for k, _v in filtered], dtype=f"U{max(1, max(len(k[0]) for k, _v in filtered))}")
        mtimes = np.asarray([k[1] for k, _v in filtered], dtype=np.int64)
        sizes = np.asarray([k[2] for k, _v in filtered], dtype=np.int64)
        contacts = np.vstack([v[0] for _k, v in filtered]).astype(np.int8, copy=False)
        ccount = np.asarray([v[1] for _k, v in filtered], dtype=np.int32)
        rg = np.asarray([v[2] for _k, v in filtered], dtype=np.float32)
        e2e = np.asarray([v[3] for _k, v in filtered], dtype=np.float32)
        np.savez_compressed(
            path,
            paths=paths,
            mtimes_ns=mtimes,
            sizes=sizes,
            cutoff_A=np.asarray([cutoff], dtype=np.float32),
            min_sep=np.asarray([min_sep], dtype=np.int32),
            contacts=contacts,
            contact_count=ccount,
            rg_A=rg,
            end_to_end_A=e2e,
        )
        _FEATURE_CACHE_DIRTY = False
    except Exception:
        pass


def cached_pdb_geometry_features(args, path: Path):
    load_feature_cache(args)
    key = _geometry_cache_key(path, getattr(args, "contact_cutoff", 8.0), getattr(args, "contact_min_sep", 3))
    if key is None:
        return None
    if key in _PDB_GEOM_CACHE:
        return _PDB_GEOM_CACHE[key]
    if key in _FEATURE_CACHE_MAP:
        geom = _FEATURE_CACHE_MAP[key]
        _PDB_GEOM_CACHE[key] = geom
        return geom
    geom = pdb_geometry_features(path, getattr(args, "contact_cutoff", 8.0), getattr(args, "contact_min_sep", 3))
    if geom is not None:
        remember_pdb_geometry(path, getattr(args, "contact_cutoff", 8.0), getattr(args, "contact_min_sep", 3), *geom)
    return geom


def pdb_geometry_features(path: Path, cutoff_A: float, min_sep: int):
    """Cached PDB -> (contact_vector, contact_count, rg_A, end_to_end_A)."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size), float(cutoff_A), int(min_sep))
    cached = _PDB_GEOM_CACHE.get(key) if '_PDB_GEOM_CACHE' in globals() else None
    if cached is not None:
        return cached
    ca = read_pdb_ca_coords_simple(path)
    if len(ca) < 2 or not np.isfinite(ca).all():
        return None
    rg_A = radius_of_gyration(ca)
    e2e_A = float(np.linalg.norm(ca[-1] - ca[0]))
    cvec, ccount = contact_vector_from_coords(ca, cutoff_A, min_sep)
    geom = (cvec, int(ccount), float(rg_A), float(e2e_A))
    if '_PDB_GEOM_CACHE' in globals():
        _PDB_GEOM_CACHE[key] = geom
    return geom


def feature_from_geometry(cvec: np.ndarray, rg_A: float, e2e_A: float, contact_count: int):
    cvec = np.asarray(cvec, dtype=np.float32).reshape(-1)
    feat = np.empty(cvec.size + 3, dtype=np.float32)
    feat[:cvec.size] = cvec
    feat[cvec.size:] = (np.float32(rg_A), np.float32(e2e_A), np.float32(contact_count))
    return feat

def minresults_from_hop_csv(args, hop_csv: Path):
    if not hop_csv.exists():
        return []
    df = pd.read_csv(hop_csv)
    out = []
    for _, row in df.iterrows():
        if not bool(row.get("success", False)) or not isinstance(row.get("pdb_path", ""), str) or not row.get("pdb_path", ""):
            continue
        p = Path(row["pdb_path"])
        if not p.exists():
            continue
        out.append(MinResult(
            seed_name=p.stem,
            input_pdb=str(p),
            output_pdb=str(p),
            success=True,
            error="",
            mode="implicit_bh",
            n_atoms=0,
            n_residues=0,
            n_waters=0,
            n_ions=0,
            initial_energy_kj_mol=float(row.get("initial_energy_kj_mol", np.nan)),
            minimized_energy_kj_mol=float(row.get("minimized_energy_kj_mol", np.nan)),
            energy_drop_kj_mol=float(row.get("initial_energy_kj_mol", np.nan)) - float(row.get("minimized_energy_kj_mol", np.nan)),
            max_force_kj_mol_nm=float(row.get("max_force_kj_mol_nm", np.nan)),
            rg_nm=float(row.get("rg_nm", np.nan)),
            end_to_end_nm=float(row.get("end_to_end_nm", np.nan)),
            contact_count=int(row.get("contact_count", 0)),
        ))
    return out


# -----------------------------
# Basin archive / MSM-lite diagnostics
# -----------------------------

def _safe_entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[np.isfinite(counts) & (counts > 0)]
    if counts.size == 0:
        return 0.0, 1.0
    p = counts / counts.sum()
    h = max(0.0, float(-np.sum(p * np.log(p + 1e-12))))
    return h, float(np.exp(h))


def _safe_float(value, default=float("nan")):
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def compute_space_explored_metrics(X: np.ndarray, basin_labels: Optional[np.ndarray] = None) -> dict:
    """Return a compact scalar summary of explored conformational/search space.

    The main score is intentionally heuristic. It combines:
      * geometric spread in the contact/shape feature space, and
      * entropy/effective count of occupied archive basins.

    Larger values mean the current search pool is broader and/or distributed
    over more distinct basins. It is a search-coverage diagnostic, not a
    thermodynamic entropy. Tiny printout, large caveat; science remains rude.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return {
            "space_explored_score": 0.0,
            "space_rms_radius": 0.0,
            "space_pc1_std": 0.0,
            "space_pc2_std": 0.0,
            "space_pc12_area": 0.0,
            "space_basin_entropy": 0.0,
            "space_basin_neff": 0.0,
        }
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xc = X - X.mean(axis=0, keepdims=True)
    sq_radius = np.sum(Xc * Xc, axis=1)
    rms_radius = float(np.sqrt(np.mean(sq_radius))) if X.shape[0] else 0.0

    pc1_std = 0.0
    pc2_std = 0.0
    pc12_area = 0.0
    if X.shape[0] > 1 and X.shape[1] > 0:
        try:
            svals = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
            denom = math.sqrt(max(1, X.shape[0] - 1))
            pc_stds = svals / denom
            if pc_stds.size >= 1:
                pc1_std = float(pc_stds[0])
            if pc_stds.size >= 2:
                pc2_std = float(pc_stds[1])
            pc12_area = float(pc1_std * pc2_std)
        except Exception:
            pass

    if basin_labels is None:
        basin_entropy = 0.0
        basin_neff = 1.0 if X.shape[0] else 0.0
    else:
        labels = np.asarray(basin_labels)
        labels = labels[np.isfinite(labels)] if np.issubdtype(labels.dtype, np.number) else labels
        if labels.size:
            _, counts = np.unique(labels, return_counts=True)
            basin_entropy, basin_neff = _safe_entropy_from_counts(counts)
        else:
            basin_entropy, basin_neff = 0.0, 0.0

    # Main scalar: feature-space radius times sqrt(effective occupied basins).
    # sqrt keeps the score readable instead of letting basin count dominate.
    score = float(rms_radius * math.sqrt(max(0.0, basin_neff)))
    return {
        "space_explored_score": score,
        "space_rms_radius": rms_radius,
        "space_pc1_std": pc1_std,
        "space_pc2_std": pc2_std,
        "space_pc12_area": pc12_area,
        "space_basin_entropy": float(basin_entropy),
        "space_basin_neff": float(basin_neff),
    }


def record_space_explored(args, step_label: str, prefix: str, metrics: dict) -> dict:
    """Append/update the cumulative exploration-space metric log."""
    out_path = Path(args.out) / "exploration_space_metrics.csv"
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step_label": str(step_label),
        "archive_prefix": str(prefix),
    }
    row.update({k: _safe_float(v, 0.0) for k, v in metrics.items()})

    try:
        if out_path.exists():
            old = pd.read_csv(out_path)
            # Replace same prefix so reruns/resumes do not create misleading duplicates.
            if "archive_prefix" in old.columns:
                old = old[old["archive_prefix"].astype(str) != str(prefix)]
        else:
            old = pd.DataFrame()
        prev_score = float(old["space_explored_score"].iloc[-1]) if (not old.empty and "space_explored_score" in old.columns) else np.nan
        score = float(row.get("space_explored_score", 0.0))
        row["space_delta_from_previous"] = float(score - prev_score) if np.isfinite(prev_score) else float("nan")
        new = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(out_path, index=False)
    except Exception:
        row["space_delta_from_previous"] = float("nan")
    return row


def format_space_explored_message(row: dict) -> str:
    score = _safe_float(row.get("space_explored_score"), 0.0)
    delta = _safe_float(row.get("space_delta_from_previous"), float("nan"))
    neff = _safe_float(row.get("space_basin_neff"), 0.0)
    radius = _safe_float(row.get("space_rms_radius"), 0.0)
    pc_area = _safe_float(row.get("space_pc12_area"), 0.0)
    delta_txt = "n/a" if not np.isfinite(delta) else f"{delta:+.2f}"
    return (
        f"Space explored: score={score:.2f} (Δ {delta_txt}); "
        f"N_eff basins={neff:.1f}; radius={radius:.2f}; PC1×PC2 area={pc_area:.2f}"
    )



def pca_scores_from_feature_matrix(X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] == 0:
        return None
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    try:
        U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    except Exception:
        return None
    k = min(2, S.size)
    if k == 0:
        return None
    scores = U[:, :k] * S[:k]
    if scores.shape[1] == 1:
        scores = np.hstack([scores, np.zeros((scores.shape[0], 1), dtype=scores.dtype)])
    return np.asarray(scores[:, :2], dtype=float)


def _choose_archive_basin_count(args, n_points: int):
    if n_points <= 1:
        return max(1, n_points)
    target = max(int(getattr(args, "n_final_seeds", 20)) * 2, int(round(math.sqrt(n_points))))
    return max(1, min(int(target), int(n_points)))


def write_basin_archive(args, results: list[MinResult], out_dir: Path, prefix: str = "basin_archive", hop_csv: Optional[Path] = None):
    """Cluster minimized structures into a live basin archive and write MSM-lite diagnostics.

    This is a search diagnostic/controller artifact, not a physical kinetic MSM.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, contacts, descs = [], [], []
    for r in results:
        if not r.success:
            continue
        p = Path(r.output_pdb)
        if not p.exists():
            continue
        geom = cached_pdb_geometry_features(args, p)
        if geom is None:
            continue
        cvec, ccount, rg_A, e2e_A = geom
        rows.append({
            "seed_name": r.seed_name,
            "mode": r.mode,
            "pdb_path": str(p),
            "energy_kj_mol": float(r.minimized_energy_kj_mol),
            "max_force_kj_mol_nm": float(r.max_force_kj_mol_nm),
            "rg_nm": float(r.rg_nm),
            "end_to_end_nm": float(r.end_to_end_nm),
            "contact_count": int(r.contact_count),
        })
        contacts.append(cvec)
        descs.append(np.array([rg_A, e2e_A, float(ccount)], dtype=float))

    if not rows:
        return None

    contacts_arr = np.vstack(contacts)
    desc_arr = np.vstack(descs)
    X = build_feature_matrix(contacts_arr, desc_arr)
    n_basins = _choose_archive_basin_count(args, len(rows))
    _selected, labels = select_by_minibatch_or_farthest(X, n_basins, args.cluster_method, None)

    points = pd.DataFrame(rows)
    points["basin_id"] = labels.astype(int)
    points.to_csv(out_dir / f"{prefix}_points.csv", index=False)

    basin_rows = []
    for basin_id, grp in points.groupby("basin_id"):
        energies = grp["energy_kj_mol"].to_numpy(dtype=float)
        best_idx = int(np.nanargmin(energies)) if len(energies) else 0
        best_row = grp.iloc[best_idx]
        basin_rows.append({
            "basin_id": int(basin_id),
            "occupancy": int(len(grp)),
            "best_energy_kj_mol": float(best_row["energy_kj_mol"]),
            "median_energy_kj_mol": float(np.nanmedian(energies)),
            "best_seed_name": str(best_row["seed_name"]),
            "representative_pdb_path": str(best_row["pdb_path"]),
            "mean_rg_nm": float(np.nanmean(grp["rg_nm"])),
            "mean_end_to_end_nm": float(np.nanmean(grp["end_to_end_nm"])),
            "mean_contact_count": float(np.nanmean(grp["contact_count"])),
        })
    basins = pd.DataFrame(basin_rows).sort_values(["best_energy_kj_mol", "occupancy"], ascending=[True, False])
    basins.to_csv(out_dir / f"{prefix}_basins.csv", index=False)

    h_count, neff_count = _safe_entropy_from_counts(basins["occupancy"].to_numpy(dtype=float))
    best_e = basins["best_energy_kj_mol"].to_numpy(dtype=float)
    rt = 0.00831446261815324 * float(getattr(args, "temperature", 300.0))
    if len(best_e) and np.isfinite(best_e).any() and rt > 0:
        e0 = float(np.nanmin(best_e))
        w = np.exp(-(best_e - e0) / rt)
        h_boltz, neff_boltz = _safe_entropy_from_counts(w)
    else:
        h_boltz, neff_boltz = 0.0, 1.0
    space_metrics = compute_space_explored_metrics(X, labels)
    space_row = record_space_explored(args, step_label=prefix, prefix=prefix, metrics={
        **space_metrics,
        "n_structures": int(len(points)),
        "n_basins": int(len(basins)),
        "H_count_dimensionless": h_count,
        "N_eff_count": neff_count,
    })
    summary = pd.DataFrame([{
        "n_structures": int(len(points)),
        "n_basins": int(len(basins)),
        "H_count_dimensionless": h_count,
        "N_eff_count": neff_count,
        "H_energy_weighted_dimensionless": h_boltz,
        "N_eff_energy_weighted": neff_boltz,
        "space_explored_score": space_metrics["space_explored_score"],
        "space_delta_from_previous": space_row.get("space_delta_from_previous", float("nan")),
        "space_rms_radius": space_metrics["space_rms_radius"],
        "space_pc1_std": space_metrics["space_pc1_std"],
        "space_pc2_std": space_metrics["space_pc2_std"],
        "space_pc12_area": space_metrics["space_pc12_area"],
        "space_basin_entropy": space_metrics["space_basin_entropy"],
        "space_basin_neff": space_metrics["space_basin_neff"],
        "temperature_K": float(getattr(args, "temperature", 300.0)),
        "note": "Search/archive entropy and space_explored_score are exploration diagnostics, not thermodynamic conformational entropy.",
    }])
    summary.to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    scores2 = pca_scores_from_feature_matrix(X)
    if scores2 is not None:
        delta_val = _safe_float(space_row.get("space_delta_from_previous"), float("nan"))
        delta_txt = "n/a" if not np.isfinite(delta_val) else f"{delta_val:+.2f}"
        subtitle = (
            f"step={prefix} | score={space_metrics['space_explored_score']:.2f} | "
            f"Δ {delta_txt} | n={len(points)}"
        )
        set_cli_pca_panel("CLI PCA space", scores2, subtitle=subtitle)
    ui_message(f"{prefix}: " + format_space_explored_message(space_row))
    render_dashboard()

    if hop_csv is not None and Path(hop_csv).exists():
        try:
            hdf = pd.read_csv(hop_csv)
            name_to_basin = dict(zip(points["seed_name"].astype(str), points["basin_id"].astype(int)))
            trans_rows = []
            for _, hr in hdf.iterrows():
                if not bool(hr.get("success", False)):
                    continue
                p = Path(str(hr.get("pdb_path", "")))
                endpoint = p.stem
                parent = str(hr.get("parent_seed", ""))
                trans_rows.append({
                    "parent_seed": parent,
                    "endpoint_seed": endpoint,
                    "parent_basin_id": name_to_basin.get(parent, np.nan),
                    "endpoint_basin_id": name_to_basin.get(endpoint, np.nan),
                    "hop_index": int(hr.get("hop_index", -1)),
                    "energy_kj_mol": float(hr.get("minimized_energy_kj_mol", np.nan)),
                })
            if trans_rows:
                trans = pd.DataFrame(trans_rows)
                trans.to_csv(out_dir / f"{prefix}_transitions.csv", index=False)
                graph = (trans.dropna(subset=["parent_basin_id", "endpoint_basin_id"])
                              .groupby(["parent_basin_id", "endpoint_basin_id"])
                              .size().reset_index(name="count"))
                graph.to_csv(out_dir / f"{prefix}_transition_graph.csv", index=False)
        except Exception as exc:
            (out_dir / f"{prefix}_transition_graph_error.txt").write_text(str(exc))

    np.save(out_dir / f"{prefix}_features.npy", X)
    save_feature_cache(args)
    return {"points": points, "basins": basins, "summary": summary}


# -----------------------------
# Adaptive PCA frontier exploration
# -----------------------------

def pca_frontier_feature_from_pdb(args, pdb_path: Path):
    geom = cached_pdb_geometry_features(args, Path(pdb_path))
    if geom is None:
        return None
    cvec, ccount, rg_A, e2e_A = geom
    return feature_from_geometry(cvec, rg_A, e2e_A, ccount)


def fit_adaptive_pca_model(args, results: list[MinResult], out_prefix: Path):
    rows = []
    feats = []

    for r in results:
        if not r.success:
            continue
        p = Path(r.output_pdb)
        if not p.exists():
            continue
        feat = pca_frontier_feature_from_pdb(args, p)
        if feat is None or not np.isfinite(feat).all():
            continue
        rows.append({
            "seed_name": r.seed_name,
            "mode": r.mode,
            "pdb_path": str(p),
            "energy_kj_mol": r.minimized_energy_kj_mol,
            "max_force_kj_mol_nm": r.max_force_kj_mol_nm,
            "rg_nm": r.rg_nm,
            "end_to_end_nm": r.end_to_end_nm,
            "contact_count": r.contact_count,
        })
        feats.append(feat)

    if len(feats) < 3:
        raise RuntimeError(f"Need at least 3 valid structures for adaptive PCA; got {len(feats)}.")

    X = np.vstack(feats)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    Xz = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)

    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    components = Vt[:2]
    scores = U[:, :2] * S[:2]
    eigvals = (S ** 2) / max(1, Xz.shape[0] - 1)
    explained = eigvals / eigvals.sum() if eigvals.sum() > 0 else eigvals

    df = pd.DataFrame(rows)
    df["pc1"] = scores[:, 0]
    df["pc2"] = scores[:, 1]
    df.to_csv(out_prefix.with_suffix(".pca_points.csv"), index=False)
    np.save(out_prefix.with_suffix(".feature_mean.npy"), mu)
    np.save(out_prefix.with_suffix(".feature_std.npy"), sd)
    np.save(out_prefix.with_suffix(".pca_components.npy"), components)
    np.save(out_prefix.with_suffix(".pca_scores.npy"), scores)
    np.save(out_prefix.with_suffix(".pca_explained_variance_ratio.npy"), explained[:2])

    return {"mu": mu, "sd": sd, "components": components, "scores": scores, "points": df, "explained": explained[:2]}


def project_adaptive_feature(feat: np.ndarray, model: dict):
    X = np.asarray(feat, dtype=float).reshape(1, -1)
    Xz = np.nan_to_num((X - model["mu"]) / model["sd"], nan=0.0, posinf=0.0, neginf=0.0)
    return (Xz @ model["components"].T)[0]


def neighbor_mask_8(mask: np.ndarray):
    out = np.zeros_like(mask, dtype=bool)
    nx, ny = mask.shape
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            src_x0 = max(0, -dx)
            src_x1 = min(nx, nx - dx)
            src_y0 = max(0, -dy)
            src_y1 = min(ny, ny - dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(nx, nx + dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(ny, ny + dy)
            out[dst_x0:dst_x1, dst_y0:dst_y1] |= mask[src_x0:src_x1, src_y0:src_y1]
    return out


def build_adaptive_target_grid(args, scores: np.ndarray, out_prefix: Path):
    scores = np.asarray(scores, dtype=float)
    x = scores[:, 0]
    y = scores[:, 1]

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    dx = xmax - xmin if xmax > xmin else 1.0
    dy = ymax - ymin if ymax > ymin else 1.0
    pad = float(args.explore_pca_padding)
    xr = (xmin - pad * dx, xmax + pad * dx)
    yr = (ymin - pad * dy, ymax + pad * dy)

    H, xedges, yedges = np.histogram2d(x, y, bins=args.explore_bins, range=[xr, yr])
    occupied = H > 0
    low = H <= args.explore_target_max_count
    adjacent = neighbor_mask_8(occupied)

    if args.explore_frontier_only:
        target = low & adjacent & (~occupied)
    else:
        target = low

    rows = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            rows.append({
                "bin_i": i,
                "bin_j": j,
                "pc1_center": 0.5 * (xedges[i] + xedges[i + 1]),
                "pc2_center": 0.5 * (yedges[j] + yedges[j + 1]),
                "count": H[i, j],
                "occupied": bool(occupied[i, j]),
                "target": bool(target[i, j]),
            })

    pd.DataFrame(rows).to_csv(out_prefix.with_suffix(".pca_target_bins.csv"), index=False)
    np.save(out_prefix.with_suffix(".pca_hist_counts.npy"), H)
    np.save(out_prefix.with_suffix(".pca_target_mask.npy"), target.astype(np.int8))
    np.save(out_prefix.with_suffix(".pca_xedges.npy"), xedges)
    np.save(out_prefix.with_suffix(".pca_yedges.npy"), yedges)
    return {"H": H, "target": target, "xedges": xedges, "yedges": yedges}


def adaptive_score_is_target(score, grid, allow_outside=True):
    """Return (accept, reason, bin_key) for an adaptive PCA proposal.

    The bin key is used to cap redundant kept proposals per PCA bin before
    expensive minimization. Outside-grid hits get overflow bin keys rather than
    all collapsing into one undifferentiated outside bucket.
    """
    x, y = float(score[0]), float(score[1])
    xedges = grid["xedges"]
    yedges = grid["yedges"]
    target = grid["target"]
    nx, ny = target.shape

    raw_ix = int(np.searchsorted(xedges, x, side="right") - 1)
    raw_iy = int(np.searchsorted(yedges, y, side="right") - 1)
    outside = raw_ix < 0 or raw_ix >= nx or raw_iy < 0 or raw_iy >= ny

    if outside:
        ix_key = max(-1, min(raw_ix, nx))
        iy_key = max(-1, min(raw_iy, ny))
        bin_key = f"outside_{ix_key}_{iy_key}"
        if not bool(allow_outside):
            return False, "outside_grid", bin_key
        return True, f"outside_grid_bin_{ix_key}_{iy_key}", bin_key

    ix = max(0, min(raw_ix, nx - 1))
    iy = max(0, min(raw_iy, ny - 1))
    bin_key = f"bin_{ix}_{iy}"

    if bool(target[ix, iy]):
        return True, f"target_bin_{ix}_{iy}", bin_key
    return False, f"non_target_bin_{ix}_{iy}", bin_key


def generate_adaptive_frontier_proposals(args, round_dir: Path, round_i: int, model: dict, grid: dict):
    seq = validate_sequence(args.seq)
    proposal_dir = round_dir / "pca_frontier_proposals_raw"
    kept_dir = round_dir / "pca_frontier_kept_proposals"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    kept_dir.mkdir(parents=True, exist_ok=True)

    cfg = GenConfig(
        seq,
        args.explore_proposals,
        args.angle_sd,
        args.clash_cutoff,
        args.contact_cutoff,
        args.contact_min_sep,
        args.max_clashes,
        args.seed + 100000 * (round_i + 1),
        write_pdbs=False,
        rama_sampling=str(getattr(args, "rama_sampling", "stratified")),
        generation_backend=str(getattr(args, "generation_backend", "fast")),
        diversity_bank_preset=str(getattr(args, "diversity_bank_preset", "off")),
        diversity_bank_wide_angle_sd=float(getattr(args, "diversity_bank_wide_angle_sd", 45.0)),
    )

    rng = np.random.default_rng(args.seed + 424242 + round_i)
    worker_seeds = rng.integers(0, 2**32 - 1, size=args.explore_proposals, dtype=np.uint32)

    kept_rows = []
    tried = 0
    generated = 0
    kept = 0
    n_fail = 0
    bin_cap_rejects = 0
    per_bin_cap = max(0, int(getattr(args, "explore_max_kept_per_bin", 5) or 0))
    kept_by_bin = {}

    print_stage(f"Adaptive PCA exploration round {round_i + 1}", phase_key="exploration", cycle_step="propose", cycle_round=round_i + 1, cycle_total=max(0, int(args.explore_rounds)))
    cap_note = "disabled" if per_bin_cap <= 0 else str(per_bin_cap)
    ui_message(
        f"Generating {args.explore_proposals} proposals; keeping up to {args.explore_keep} PCA-frontier hits "
        f"(max kept/bin: {cap_note})."
    )
    progress = CliProgress("pca explore", args.explore_proposals)
    live = LiveStats("live adaptive proposal hits", enabled=args.live_hist, every=max(1, min(args.explore_proposals, getattr(args, "hist_every", 1000))))

    def handle_result(rec, cvec, desc):
        nonlocal tried, generated, kept, n_fail, bin_cap_rejects
        tried += 1
        if rec is None:
            if isinstance(desc, str):
                n_fail += 1
            return

        generated += 1
        feat = np.hstack([np.asarray(cvec, dtype=float), np.asarray(desc, dtype=float)])

        if feat.shape[0] != model["mu"].shape[0]:
            return

        score = project_adaptive_feature(feat, model)
        accept, reason, bin_key = adaptive_score_is_target(score, grid, allow_outside=args.explore_allow_outside)

        if accept and kept < args.explore_keep:
            bin_count = int(kept_by_bin.get(bin_key, 0))
            if per_bin_cap > 0 and bin_count >= per_bin_cap:
                bin_cap_rejects += 1
                return

            next_rank = kept + 1
            dst = kept_dir / f"explore_r{round_i + 1:02d}_hit_{next_rank:04d}_conf_{rec.conformer_id:06d}.pdb"
            full_rec, _full_cvec, _full_desc = materialize_conformer_from_record(seq, rec, cfg, dst, write_pdb=True)
            if full_rec is None:
                return

            kept = next_rank
            kept_by_bin[bin_key] = bin_count + 1
            rec = full_rec
            row = asdict(rec)
            row["pdb_path"] = str(dst)
            row.update({
                "adaptive_round": round_i + 1,
                "kept_rank": kept,
                "proposal_pdb_path": str(dst),
                "pc1": float(score[0]),
                "pc2": float(score[1]),
                "target_reason": reason,
                "target_bin_key": bin_key,
                "target_bin_kept_count": int(kept_by_bin.get(bin_key, 0)),
            })
            kept_rows.append(row)
            live.add(pc1=score[0], pc2=score[1], rg_A=rec.rg_A, end_to_end_A=rec.end_to_end_A, contacts=rec.contact_count)

    if args.jobs <= 1:
        cfg_payload = asdict(cfg)
        for i in range(1, args.explore_proposals + 1):
            task = (seq, i - 1, str(proposal_dir), cfg_payload, int(worker_seeds[i - 1]))
            rec, cvec, desc = generate_one(task)
            handle_result(rec, cvec, desc)
            if i % max(1, args.progress_every) == 0 or i == args.explore_proposals or kept >= args.explore_keep:
                progress.update(i, generated=generated, kept=kept, bins=len(kept_by_bin), bin_skip=bin_cap_rejects, fail=n_fail)
                live.maybe_print(i)
            if kept >= args.explore_keep:
                break
    else:
        n_by_size = math.ceil(args.explore_proposals / max(1, args.gen_chunk_size))
        n_by_workers = max(1, args.jobs * max(1, args.gen_chunks_per_worker))
        n_chunks = max(1, min(args.explore_proposals, max(n_by_size, n_by_workers)))
        chunk_tasks = make_generation_chunks(seq, proposal_dir, asdict(cfg), worker_seeds, n_chunks)

        ex = None
        futures = []
        try:
            ex = make_process_pool(args.jobs)
            futures = [ex.submit(generate_chunk, t) for t in chunk_tasks]
            for fut in as_completed(futures):
                chunk_records, chunk_contacts, chunk_descs, chunk_failures, n_done = fut.result()
                accepted_in_chunk = len(chunk_records)
                error_failures = sum(int(v) for v in chunk_failures.values())
                silent_rejects = max(0, int(n_done) - accepted_in_chunk - error_failures)
                for rec, cvec, desc in zip(chunk_records, chunk_contacts, chunk_descs):
                    handle_result(rec, cvec, desc)
                    if kept >= args.explore_keep:
                        break
                n_fail += error_failures
                tried += error_failures + silent_rejects
                progress.update(min(tried, args.explore_proposals), generated=generated, kept=kept, bins=len(kept_by_bin), bin_skip=bin_cap_rejects, fail=n_fail)
                live.maybe_print(min(tried, args.explore_proposals))
                if kept >= args.explore_keep:
                    cancel_futures_now(futures)
                    break
        except KeyboardInterrupt:
            ui_message("Interrupted: terminating adaptive PCA proposal workers.")
            render_dashboard()
            cancel_futures_now(futures)
            shutdown_process_pool_now(ex)
            raise SystemExit(130)
        finally:
            if ex is not None:
                shutdown_process_pool_now(ex)

    progress.done(min(tried, args.explore_proposals), generated=generated, kept=kept, bins=len(kept_by_bin), bin_skip=bin_cap_rejects, fail=n_fail)
    live.maybe_print(min(tried, args.explore_proposals), force=True)
    kept_df = pd.DataFrame(kept_rows)
    kept_df.to_csv(round_dir / "pca_frontier_kept_proposals.csv", index=False)
    if kept_by_bin:
        pd.DataFrame([
            {"target_bin_key": key, "kept_count": int(val)}
            for key, val in sorted(kept_by_bin.items())
        ]).to_csv(round_dir / "pca_frontier_kept_bin_counts.csv", index=False)
    shutil.rmtree(proposal_dir, ignore_errors=True)
    ui_message(
        f"Adaptive round {round_i + 1}: kept {len(kept_df)} proposals across {len(kept_by_bin)} bins; "
        f"skipped {bin_cap_rejects} saturated-bin hits."
    )
    render_dashboard()
    return kept_dir, kept_df


def run_adaptive_exploration_loop(args, combined_results: list[MinResult]):
    """
    Adaptive seed expansion:
      current minimized/hop structures -> PCA frontier map
      -> generate Ramachandran proposals -> keep frontier hits
      -> implicit minimize kept hits -> add to final selection pool.
    """
    added_all = []
    base_dir = Path(args.out) / "adaptive_pca_exploration"
    base_dir.mkdir(parents=True, exist_ok=True)

    for round_i in range(max(0, int(args.explore_rounds))):
        round_dir = base_dir / f"round_{round_i + 1:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Resume: skip rounds whose minimization CSV already exists.
        if getattr(args, "resume", False):
            score_csv_r = round_dir / "pca_frontier_implicit_minimization_scores.csv"
            bh_csv_r = round_dir / "pca_frontier_basin_hop_minima.csv"
            bh_expected = bool(getattr(args, "basin_hop", False)) and bool(getattr(args, "explore_bh", True))
            if _resume_csv_done(score_csv_r):
                prior = minresults_from_score_csv(score_csv_r, out_dir=Path(args.out))
                bh_prior = minresults_from_hop_csv(args, bh_csv_r) if bh_expected and _resume_csv_done(bh_csv_r) else []
                ui_message(f"[resume] PCA round {round_i + 1}: skipping — {len(prior)} min + {len(bh_prior)} BH results loaded from CSV.")
                render_dashboard()
                combined_results.extend(prior)
                combined_results.extend(bh_prior)
                added_all.extend(prior)
                added_all.extend(bh_prior)
                continue

        print_stage(f"Adaptive PCA frontier map round {round_i + 1}", phase_key="exploration", cycle_step="map", cycle_round=round_i + 1, cycle_total=max(0, int(args.explore_rounds)))
        model = fit_adaptive_pca_model(args, combined_results, round_dir / "frontier")
        grid = build_adaptive_target_grid(args, model["scores"], round_dir / "frontier")
        n_target = int(np.asarray(grid["target"], dtype=bool).sum())
        ev = model["explained"]
        ui_message(f"PCA source structures: {len(model['points'])} | target bins: {n_target} | EV: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}")
        set_cli_pca_panel(
            "CLI PCA space",
            model["scores"],
            subtitle=(
                f"adaptive round {round_i + 1} source pool | points={len(model['points'])} | "
                f"target bins={n_target} | EV=({ev[0]:.2f},{ev[1]:.2f})"
            ),
        )
        render_dashboard()

        proposal_dir, kept_df = generate_adaptive_frontier_proposals(args, round_dir, round_i, model, grid)
        if kept_df.empty:
            print("No adaptive PCA-target proposals kept; stopping adaptive loop.")
            break

        min_dir = round_dir / "aa_implicit_minimized_pca_frontier"
        score_csv = round_dir / "pca_frontier_implicit_minimization_scores.csv"
        new_results = run_minimization_batch(args, proposal_dir, min_dir, "implicit", score_csv, ui_phase_key="exploration", ui_cycle_step="minimize", ui_cycle_round=round_i + 1, ui_cycle_total=max(0, int(args.explore_rounds)))
        ok_new = [r for r in new_results if r.success]
        added_all.extend(ok_new)
        combined_results.extend(ok_new)

        bh_round_results = []
        if ok_new and bool(getattr(args, "basin_hop", False)) and bool(getattr(args, "explore_bh", True)):
            pca_bh_dir = round_dir / "basin_hop_from_pca_frontier"
            pca_bh_csv = round_dir / "pca_frontier_basin_hop_minima.csv"
            ui_message(f"Adaptive round {round_i + 1}: restarting BH/MC from {len(ok_new)} newly minimized PCA hits.")
            render_dashboard()
            run_basin_hopping(
                args,
                min_dir,
                out_dir=pca_bh_dir,
                hop_csv=pca_bh_csv,
                label=f"BH/MC restart from PCA hits round {round_i + 1}",
                cycle_round=round_i + 1,
                cycle_total=max(0, int(args.explore_rounds)),
            )
            bh_round_results = minresults_from_hop_csv(args, pca_bh_csv)
            added_all.extend(bh_round_results)
            combined_results.extend(bh_round_results)

        print_stage(
            f"Adaptive PCA archive update round {round_i + 1}",
            phase_key="exploration",
            cycle_step="archive",
            cycle_round=round_i + 1,
            cycle_total=max(0, int(args.explore_rounds)),
        )
        if bh_round_results:
            ui_message(
                f"Adaptive round {round_i + 1}: added {len(ok_new)} minimized frontier structures "
                f"+ {len(bh_round_results)} BH/MC endpoints."
            )
        else:
            ui_message(f"Adaptive round {round_i + 1}: added {len(ok_new)} successful minimized frontier structures.")
        try:
            if ok_new or bh_round_results:
                overlay_scores = kept_df[["pc1", "pc2"]].to_numpy(dtype=float) if {"pc1", "pc2"}.issubset(kept_df.columns) else None
                all_feats = []
                for r in combined_results:
                    if not getattr(r, "success", False):
                        continue
                    pth = Path(r.output_pdb)
                    if not pth.exists():
                        continue
                    feat = pca_frontier_feature_from_pdb(args, pth)
                    if feat is not None and np.isfinite(feat).all():
                        all_feats.append(feat)
                if all_feats:
                    all_scores = np.vstack([project_adaptive_feature(np.asarray(f, dtype=float), model) for f in all_feats])
                    subtitle = (
                        f"adaptive round {round_i + 1} updated pool | total={len(all_scores)} | "
                        f"new minimized={len(ok_new)} | BH endpoints={len(bh_round_results)}"
                    )
                    set_cli_pca_panel("CLI PCA space", all_scores, overlay_scores=overlay_scores, subtitle=subtitle)
        except Exception:
            pass
        render_dashboard()

        if not ok_new:
            print("No successful adaptive minimized structures; stopping adaptive loop.")
            break

    if added_all:
        write_min_csv(base_dir / "adaptive_pca_added_minimization_scores.csv", added_all)
    print(f"Adaptive PCA exploration added {len(added_all)} minimized structures in total.")
    return added_all


def select_implicit_survivors(args, implicit_results: list[MinResult]):
    print_stage("Final basin selection")
    out_dir = Path(args.out)
    survivor_dir = out_dir / "final_implicit_survivor_seeds"
    survivor_dir.mkdir(exist_ok=True)

    ok = [r for r in implicit_results if r.success and Path(r.output_pdb).exists()]
    if not ok:
        raise RuntimeError("No successful implicit/basin-hop minima.")

    rows, contacts, descriptors, viable_rows = [], [], [], []
    for r in ok:
        reject = []
        finite = np.isfinite([r.minimized_energy_kj_mol, r.max_force_kj_mol_nm, r.rg_nm, r.end_to_end_nm]).all()
        if not finite:
            reject.append("nonfinite")
        if r.max_force_kj_mol_nm > args.reject_max_force:
            reject.append("max_force")
        if r.rg_nm < args.reject_min_rg:
            reject.append("too_compact_rg")
        if args.reject_max_rg > 0 and r.rg_nm > args.reject_max_rg:
            reject.append("too_extended_rg")
        if r.end_to_end_nm < args.reject_min_e2e:
            reject.append("too_collapsed_e2e")
        if args.reject_max_e2e > 0 and r.end_to_end_nm > args.reject_max_e2e:
            reject.append("too_extended_e2e")

        geom = cached_pdb_geometry_features(args, Path(r.output_pdb))
        if geom is None:
            reject.append("bad_ca_coords")
            cvec = np.zeros(0, dtype=np.int8)
        else:
            cvec, _ccount, _rg_A, _e2e_A = geom

        row = asdict(r)
        row["implicit_reject"] = bool(reject)
        row["implicit_reject_reason"] = ";".join(reject)
        rows.append(row)

        if not reject:
            viable_rows.append(row)
            contacts.append(cvec)
            descriptors.append(np.array([r.rg_nm, r.end_to_end_nm, float(r.contact_count), r.minimized_energy_kj_mol, r.max_force_kj_mol_nm], dtype=float))

    pd.DataFrame(rows).to_csv(out_dir / "implicit_filter_table.csv", index=False)

    if not viable_rows:
        raise RuntimeError("All implicit/hop minima were rejected; relax thresholds.")

    contacts_arr = np.vstack(contacts)
    desc_arr = np.vstack(descriptors)

    energies, forces, rg, e2e = desc_arr[:, 3], desc_arr[:, 4], desc_arr[:, 0], desc_arr[:, 1]
    badness = robust_z(energies) + 0.75 * robust_z(forces) + 0.15 * np.abs(robust_z(rg)) + 0.15 * np.abs(robust_z(e2e))

    X_basin = build_feature_matrix(contacts_arr, desc_arr[:, :3])
    n_final = min(args.n_final_seeds, len(viable_rows))
    selected_idx, basin_labels = select_by_minibatch_or_farthest(X_basin, n_final, args.cluster_method, badness)

    final_rows = []
    for rank, idx in enumerate(selected_idx):
        src = Path(viable_rows[int(idx)]["output_pdb"])
        dst = survivor_dir / f"survivor_{rank:03d}_basin_{int(basin_labels[idx]):03d}_{src.name}"
        shutil.copy2(src, dst)
        row = dict(viable_rows[int(idx)])
        row.update({"survivor_rank": rank, "implicit_basin_id": int(basin_labels[idx]), "implicit_badness": float(badness[idx]), "survivor_pdb_path": str(dst)})
        final_rows.append(row)

    viable_labeled = []
    for i, row in enumerate(viable_rows):
        row2 = dict(row)
        row2["implicit_basin_id"] = int(basin_labels[i])
        row2["implicit_badness"] = float(badness[i])
        viable_labeled.append(row2)

    pd.DataFrame(viable_labeled).to_csv(out_dir / "implicit_viable_basin_table.csv", index=False)
    pd.DataFrame(final_rows).to_csv(out_dir / "final_survivor_seeds.csv", index=False)
    np.save(out_dir / "implicit_basin_contacts.npy", contacts_arr)
    np.save(out_dir / "implicit_basin_features.npy", X_basin)

    save_feature_cache(args)
    print(f"Viable implicit/hop minima: {len(viable_rows)}/{len(ok)}")
    print(f"Final survivor seeds: {len(final_rows)}")
    return survivor_dir




# -----------------------------
# Integrated PCA / pseudo-FES post-processing
# (adapted from pca_pseudo_fes_from_seeds.py)
# -----------------------------

import tempfile
import shlex
import subprocess

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "quickgen_matplotlib"
    try:
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)
    except Exception:
        pass

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

try:
    from scipy.interpolate import RegularGridInterpolator
except Exception:
    RegularGridInterpolator = None

DEFAULT_SCORE_TABLE_NAMES = [
    "implicit_minimization_scores.csv",
    "nma_implicit_minimization_scores.csv",
    "basin_hop_minima.csv",
    "explicit_minimization_scores.csv",
    "aa_minimization_scores.csv",
]

DEFAULT_HIGHLIGHT_DIR_NAMES = [
    "final_implicit_survivor_seeds",
    "explicit_survivor_seeds",
    "aa_explicit_minimized_pdbs",
    "survivor_seeds",
    "selected_seeds",
    "final_seeds",
]

RUN_ROOT_PDB_DIR_NAMES = [
    "aa_implicit_minimized_pdbs",
    "basin_hop_minima",
    "aa_nma_implicit_minimized_pdbs",
    "aa_explicit_minimized_pdbs",
    "final_implicit_survivor_seeds",
    "explicit_survivor_seeds",
]

FALLBACK_RUN_ROOT_PDB_DIR_NAMES = [
    "candidate_seeds",
]


# -----------------------------
# Structure parsing / features
# -----------------------------

def read_ca_coords_pdb(path: Path) -> np.ndarray:
    coords = []
    with path.open() as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[17:20].strip() in {"HOH", "WAT", "SOL"}:
                continue
            try:
                coords.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
            except ValueError:
                continue
    return np.asarray(coords, dtype=float)


def pairwise_ca_distances(ca: np.ndarray, min_sep: int = 1) -> np.ndarray:
    vals = []
    n = len(ca)
    for i in range(n):
        for j in range(i + min_sep, n):
            vals.append(float(np.linalg.norm(ca[i] - ca[j])))
    return np.asarray(vals, dtype=float)


def contact_vector(ca: np.ndarray, cutoff: float = 8.0, min_sep: int = 3, soft: bool = True) -> np.ndarray:
    vals = []
    n = len(ca)
    for i in range(n):
        for j in range(i + min_sep, n):
            d = float(np.linalg.norm(ca[i] - ca[j]))
            if soft:
                vals.append(1.0 / (1.0 + (d / cutoff) ** 6))
            else:
                vals.append(1.0 if d < cutoff else 0.0)
    return np.asarray(vals, dtype=float)


def rg_e2e_contact_count(ca: np.ndarray, cutoff: float = 8.0, min_sep: int = 3):
    center = ca.mean(axis=0)
    rg = float(np.sqrt(((ca - center) ** 2).sum(axis=1).mean()))
    e2e = float(np.linalg.norm(ca[-1] - ca[0]))
    contacts = contact_vector(ca, cutoff=cutoff, min_sep=min_sep, soft=False)
    return rg, e2e, int(contacts.sum())


def make_features(ca: np.ndarray, mode: str, cutoff: float, min_sep: int) -> np.ndarray:
    if mode == "distances":
        return pairwise_ca_distances(ca, min_sep=1)
    if mode == "contacts":
        return contact_vector(ca, cutoff=cutoff, min_sep=min_sep, soft=True)
    if mode == "mixed":
        d = pairwise_ca_distances(ca, min_sep=1)
        c = contact_vector(ca, cutoff=cutoff, min_sep=min_sep, soft=True)
        rg, e2e, cc = rg_e2e_contact_count(ca, cutoff=cutoff, min_sep=min_sep)
        return np.concatenate([d, c, np.array([rg, e2e, float(cc)], dtype=float)])
    raise ValueError(f"Unknown feature mode: {mode}")


def standardize(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


def pca_svd(X: np.ndarray, n_components: int = 2):
    Xz, mu, sd = standardize(X)
    Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    scores = U[:, :n_components] * S[:n_components]
    eigvals = (S ** 2) / max(1, Xz.shape[0] - 1)
    explained = eigvals / eigvals.sum() if eigvals.sum() > 0 else eigvals
    return scores, Vt[:n_components], explained[:n_components], mu, sd


# -----------------------------
# File collection / metadata
# -----------------------------

def _has_matching_pdbs(path: Path, pattern: str, recursive: bool = False) -> bool:
    try:
        iterator = path.rglob(pattern) if recursive else path.glob(pattern)
        return next(iterator, None) is not None
    except Exception:
        return False


def expand_pdb_dirs(pdb_dirs, pattern="*.pdb", recursive=False):
    """Expand run-root directories to the standard PDB output subdirectories."""
    expanded = []
    notes = []

    for raw in pdb_dirs:
        d = Path(raw)
        if recursive or not d.is_dir() or _has_matching_pdbs(d, pattern, recursive=False):
            expanded.append(d)
            continue

        found = []
        for name in RUN_ROOT_PDB_DIR_NAMES:
            child = d / name
            if child.is_dir() and _has_matching_pdbs(child, pattern, recursive=False):
                found.append(child)

        adaptive_root = d / "adaptive_pca_exploration"
        if adaptive_root.is_dir():
            for child in sorted(adaptive_root.glob("round_*/aa_implicit_minimized_pca_frontier")):
                if child.is_dir() and _has_matching_pdbs(child, pattern, recursive=False):
                    found.append(child)

        if not found:
            for name in FALLBACK_RUN_ROOT_PDB_DIR_NAMES:
                child = d / name
                if child.is_dir() and _has_matching_pdbs(child, pattern, recursive=False):
                    found.append(child)

        if found:
            expanded.extend(found)
            notes.append((d, found))
        else:
            expanded.append(d)

    clean, seen = [], set()
    for d in expanded:
        key = str(d.resolve()) if d.exists() else str(d)
        if key not in seen:
            seen.add(key)
            clean.append(d)
    return clean, notes


def collect_pdbs(pdb_dirs, pattern="*.pdb", recursive=False):
    pdbs = []
    for d in pdb_dirs:
        d = Path(d)
        pdbs.extend(sorted(d.rglob(pattern) if recursive else d.glob(pattern)))
    seen, out = set(), []
    for p in pdbs:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def candidate_score_tables(pdb_dirs):
    tables, seen = [], set()
    for d in pdb_dirs:
        d = Path(d)
        candidates = []
        candidates.extend(d.glob("*.csv"))
        candidates.extend(d.parent.glob("*.csv"))
        if d.parent.parent.exists():
            candidates.extend(d.parent.parent.glob("*.csv"))
        for c in candidates:
            if c.name in DEFAULT_SCORE_TABLE_NAMES or any(k in c.name for k in ["minimization", "minima", "scores"]):
                key = str(c.resolve())
                if key not in seen:
                    seen.add(key)
                    tables.append(c)
    return tables


def path_keys(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value)
    if not s or s.lower() == "nan":
        return []
    p = Path(s)
    keys = {s, p.name, p.stem}
    try:
        keys.add(str(p.resolve()))
    except Exception:
        pass
    return list(keys)


def file_match_keys(path: Path):
    keys = set(path_keys(str(path)))
    keys.add(path.name)
    keys.add(path.stem)
    stem = path.stem
    suffix = path.suffix

    for token in ("candidate_", "explore_", "bh_"):
        pos = stem.find(token)
        if pos > 0:
            tail = stem[pos:]
            keys.add(tail)
            keys.add(tail + suffix)

    for suf in ("_implicit_min", "_explicit_min", "_min"):
        if stem.endswith(suf):
            keys.add(stem[: -len(suf)])

    return keys


def load_weight_records(csv_paths, energy_column="minimized_energy_kj_mol", weight_column="weight"):
    path_cols = ["output_pdb", "pdb_path", "survivor_pdb_path", "candidate_pdb_path", "input_pdb"]
    lookup = {}
    used_tables = []

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue

        available_path_cols = [c for c in path_cols if c in df.columns]
        if not available_path_cols:
            continue

        has_energy = energy_column in df.columns
        has_weight = weight_column in df.columns
        if not has_energy and not has_weight:
            continue

        used_tables.append(str(csv_path))

        for _, row in df.iterrows():
            info = {}
            if has_energy:
                try:
                    info["energy"] = float(row[energy_column])
                except Exception:
                    pass
            if has_weight:
                try:
                    info["raw_weight"] = float(row[weight_column])
                except Exception:
                    pass
            for extra in ["seed_name", "parent_seed", "hop_index", "mode", "rg_nm", "end_to_end_nm", "contact_count"]:
                if extra in df.columns:
                    info[extra] = row[extra]
            if not info:
                continue
            for col in available_path_cols:
                for key in path_keys(row[col]):
                    lookup[key] = dict(info)
    return lookup, used_tables


def match_weight_info(pdb_path: Path, lookup):
    keys = list(file_match_keys(pdb_path))
    for key in keys:
        if key in lookup:
            return lookup[key]
    return {}


def auto_highlight_dirs(input_dirs):
    candidates = []
    seen = set()
    for d in input_dirs:
        d = Path(d)
        search_roots = [d, d.parent, d.parent.parent] if d.parent.exists() else [d]
        for root in search_roots:
            if not root.exists():
                continue
            for name in DEFAULT_HIGHLIGHT_DIR_NAMES:
                p = root / name
                if p.exists() and p.is_dir():
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        candidates.append(p)
    return candidates


def highlight_key_set(highlight_dirs, pattern="*.pdb", recursive=False):
    keys = set()
    files = []
    for d in highlight_dirs:
        d = Path(d)
        pdbs = sorted(d.rglob(pattern) if recursive else d.glob(pattern))
        for p in pdbs:
            files.append(p)
            keys.update(file_match_keys(p))
    return keys, files


# -----------------------------
# Weighting / dE grid
# -----------------------------

def compute_sample_weights(df, mode, temperature, energy_cap_kj=80.0, normalize=True):
    n = len(df)
    if mode == "none":
        return np.ones(n, dtype=float)

    if mode == "csv-weight":
        if "raw_weight" not in df.columns:
            raise ValueError("--weight-mode csv-weight requires a matched raw weight column.")
        w = df["raw_weight"].to_numpy(dtype=float)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w[w < 0] = 0.0
    elif mode == "boltzmann":
        if "energy_kj_mol" not in df.columns:
            raise ValueError("--weight-mode boltzmann requires matched energies.")
        E = df["energy_kj_mol"].to_numpy(dtype=float)
        finite = np.isfinite(E)
        if not finite.any():
            raise ValueError("No finite energies available for Boltzmann weights.")
        Emin = np.nanmin(E[finite])
        dE = E - Emin
        dE[~finite] = np.inf
        if energy_cap_kj > 0:
            dE = np.minimum(dE, energy_cap_kj)
        RT = 0.00831446261815324 * temperature
        w = np.exp(-dE / RT)
        w[~finite] = 0.0
    elif mode == "inverse-energy-rank":
        if "energy_kj_mol" not in df.columns:
            raise ValueError("--weight-mode inverse-energy-rank requires matched energies.")
        E = df["energy_kj_mol"].to_numpy(dtype=float)
        finite = np.isfinite(E)
        ranks = np.full(n, np.inf)
        finite_idx = np.where(finite)[0]
        order = np.argsort(E[finite])
        ranks[finite_idx[order]] = np.arange(1, len(finite_idx) + 1)
        w = 1.0 / ranks
        w[~finite] = 0.0
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    if normalize and w.sum() > 0:
        w = w * (len(w) / w.sum())
    return w


def effective_sample_size(w):
    w = np.asarray(w, dtype=float)
    s = w.sum()
    s2 = np.square(w).sum()
    return 0.0 if s2 <= 0 else float(s * s / s2)


def pmf_from_probability(P, RT: float) -> np.ndarray:
    """Convert probability/occupancy to PMF and set the finite minimum exactly to zero.

    PMF convention used here:
        F = -RT ln(P) + C
    with C chosen so min(F) = 0. This is equivalent to:
        F = -RT ln(P / Pmax)
    for the finite, nonzero plotted grid.
    """
    P = np.asarray(P, dtype=float)
    P = np.where(np.isfinite(P) & (P > 0.0), P, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -RT * np.log(P)
    finite = np.isfinite(F)
    if finite.any():
        F = F - np.nanmin(F)
        F[np.isfinite(F) & (np.abs(F) < 1.0e-12)] = 0.0
    return F


def relative_probability(P) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.where(np.isfinite(P) & (P > 0.0), P, np.nan)
    if np.isfinite(P).any() and np.nanmax(P) > 0:
        return P / np.nanmax(P)
    return P


def deltaE_2d(x, y, bins=50, RT=2.494, min_count=1, weights=None, padding=0.05):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones_like(x)
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights[weights < 0] = 0.0

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    dx = xmax - xmin if xmax > xmin else 1.0
    dy = ymax - ymin if ymax > ymin else 1.0
    xr = (xmin - padding * dx, xmax + padding * dx)
    yr = (ymin - padding * dy, ymax + padding * dy)

    H_count, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[xr, yr])
    H_weight, _, _ = np.histogram2d(x, y, bins=bins, range=[xr, yr], weights=weights)

    P = H_weight / H_weight.sum() if H_weight.sum() > 0 else H_weight
    P_for_pmf = P.copy()
    P_for_pmf[H_count < min_count] = np.nan
    F = pmf_from_probability(P_for_pmf, RT)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    return H_count, H_weight, P, F, xcenters, ycenters


def smooth_nan_grid(F: np.ndarray, sigma: float, shift_min: bool = False) -> np.ndarray:
    if sigma <= 0 or gaussian_filter is None:
        out = F.copy()
        if shift_min and np.isfinite(out).any():
            out = out - np.nanmin(out)
        return out
    valid = np.isfinite(F)
    if not valid.any():
        return F.copy()
    values = np.where(valid, F, 0.0)
    weights = valid.astype(float)
    smooth_values = gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.divide(smooth_values, smooth_weights, out=np.full_like(smooth_values, np.nan), where=smooth_weights > 1e-8)
    out[smooth_weights <= 1e-8] = np.nan
    if shift_min and np.isfinite(out).any():
        out = out - np.nanmin(out)
        out[np.isfinite(out) & (np.abs(out) < 1.0e-12)] = 0.0
    return out


# -----------------------------
# Plotting
# -----------------------------

def make_zero_anchored_cmap(base_cmap: str = "turbo"):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap(base_cmap)
    # Force an obvious visual anchor at dE = 0 while keeping the rest of the map vivid.
    colors = [(1.0, 1.0, 1.0, 1.0)]
    colors.extend(base(v) for v in np.linspace(0.08, 1.0, 255))
    return LinearSegmentedColormap.from_list(f"{base_cmap}_zero_anchor", colors, N=256)


def interpolate_surface(xcenters, ycenters, Z, factor: int = 4):
    xcenters = np.asarray(xcenters, dtype=float)
    ycenters = np.asarray(ycenters, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if factor <= 1:
        return xcenters, ycenters, Z.copy()

    nx = max(int(len(xcenters) * factor), len(xcenters))
    ny = max(int(len(ycenters) * factor), len(ycenters))
    xnew = np.linspace(xcenters[0], xcenters[-1], nx)
    ynew = np.linspace(ycenters[0], ycenters[-1], ny)

    valid = np.isfinite(Z)
    if not valid.any():
        return xnew, ynew, np.full((nx, ny), np.nan, dtype=float)

    values = np.where(valid, Z, 0.0)
    weights = valid.astype(float)

    if RegularGridInterpolator is not None:
        pts = np.stack(np.meshgrid(xnew, ynew, indexing="ij"), axis=-1).reshape(-1, 2)
        interp_values = RegularGridInterpolator(
            (xcenters, ycenters), values, method="linear", bounds_error=False, fill_value=np.nan
        )
        interp_weights = RegularGridInterpolator(
            (xcenters, ycenters), weights, method="linear", bounds_error=False, fill_value=0.0
        )
        v = interp_values(pts).reshape(nx, ny)
        w = interp_weights(pts).reshape(nx, ny)
        out = np.divide(v, w, out=np.full((nx, ny), np.nan, dtype=float), where=w > 1.0e-8)
        return xnew, ynew, out

    # Fallback: 1D linear interpolation along both axes.
    tmp = np.empty((len(xcenters), ny), dtype=float)
    for i in range(len(xcenters)):
        row_valid = np.isfinite(Z[i])
        if row_valid.any():
            tmp[i] = np.interp(ynew, ycenters[row_valid], Z[i, row_valid])
        else:
            tmp[i] = np.nan

    out = np.empty((nx, ny), dtype=float)
    for j in range(ny):
        col = tmp[:, j]
        col_valid = np.isfinite(col)
        if col_valid.any():
            out[:, j] = np.interp(xnew, xcenters[col_valid], col[col_valid])
        else:
            out[:, j] = np.nan
    return xnew, ynew, out


def prepare_surface(Z, xcenters, ycenters, smooth_sigma: float = 0.8, interp_factor: int = 4, rebase_zero: bool = False):
    Z_plot = smooth_nan_grid(Z, smooth_sigma)
    xi, yi, Zi = interpolate_surface(xcenters, ycenters, Z_plot, factor=interp_factor)
    if rebase_zero and np.isfinite(Zi).any():
        Zi = Zi - np.nanmin(Zi)
        Zi[np.abs(Zi) < 1.0e-12] = 0.0
    return xi, yi, Zi


def probability_surface_from_deltaE(F, RT: float):
    F = np.asarray(F, dtype=float)
    with np.errstate(over='ignore', invalid='ignore'):
        P_rel = np.exp(-F / RT)
    P_rel[~np.isfinite(F)] = np.nan
    return relative_probability(P_rel)


def prepare_pmf_surface_from_probability(P, xcenters, ycenters, RT: float, smooth_sigma: float = 0.8, interp_factor: int = 4):
    P_rel = relative_probability(P)
    xi, yi, P_plot = prepare_surface(P_rel, xcenters, ycenters, smooth_sigma=smooth_sigma, interp_factor=interp_factor, rebase_zero=False)
    P_plot = relative_probability(P_plot)
    F_plot = pmf_from_probability(P_plot, RT)
    return xi, yi, P_plot, F_plot


def overlay_structures(ax, df, highlight_df=None, highlight_label='selected survivor minima'):
    if 'sample_weight' in df.columns and np.nanmax(df['sample_weight'].to_numpy(dtype=float)) > 0:
        sizes = 8.0 + 22.0 * np.sqrt(df['sample_weight'] / np.nanmax(df['sample_weight']))
    else:
        sizes = 10.0
    ax.scatter(
        df['pc1'], df['pc2'], s=sizes, c='white', edgecolors='black', linewidths=0.30,
        alpha=0.35, label='all structures'
    )

    if highlight_df is not None and len(highlight_df) > 0:
        ax.scatter(
            highlight_df['pc1'], highlight_df['pc2'],
            s=120, marker='*', c='#ff2e88', edgecolors='black', linewidths=0.8,
            alpha=0.95, label=highlight_label, zorder=6,
        )


def save_probability_heatmap(F, xcenters, ycenters, df, out_png: Path, RT: float, interpolation: str,
                             smooth_sigma: float, interp_factor: int, highlight_df: pd.DataFrame | None,
                             highlight_label: str):
    import matplotlib.pyplot as plt

    P_rel = probability_surface_from_deltaE(F, RT)
    xi, yi, P_plot = prepare_surface(P_rel, xcenters, ycenters, smooth_sigma=smooth_sigma, interp_factor=interp_factor)
    finite = np.isfinite(P_plot)
    if not finite.any():
        raise ValueError('No finite probability values available for plotting.')

    extent = [xi[0], xi[-1], yi[0], yi[-1]]
    P_masked = np.ma.masked_invalid(P_plot.T)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        P_masked,
        origin='lower',
        extent=extent,
        aspect='auto',
        interpolation=interpolation,
        cmap='magma',
        vmin=0.0,
        vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('relative probability  P / Pmax')

    overlay_structures(ax, df, highlight_df=highlight_df, highlight_label=highlight_label)
    ax.set_xlabel('PCA1 / imaginary CV1')
    ax.set_ylabel('PCA2 / imaginary CV2')
    ax.set_title('Weighted PCA occupancy map (interpolated histogram)')
    ax.legend(loc='best', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def save_heatmap(F, xcenters, ycenters, df, out_png: Path, cmap: str, interpolation: str, smooth_sigma: float,
                 interp_factor: int, vmax_percentile: float, vmax_kj: float | None, show_contours: bool,
                 highlight_df: pd.DataFrame | None, highlight_label: str, P_grid=None, RT: float | None = None):
    import matplotlib.pyplot as plt

    if P_grid is not None and RT is not None:
        xi, yi, _P_plot, F_plot = prepare_pmf_surface_from_probability(
            P_grid, xcenters, ycenters, RT=RT, smooth_sigma=smooth_sigma, interp_factor=interp_factor
        )
    else:
        xi, yi, F_plot = prepare_surface(F, xcenters, ycenters, smooth_sigma=smooth_sigma,
                                         interp_factor=interp_factor, rebase_zero=True)
    finite = np.isfinite(F_plot)
    if not finite.any():
        raise ValueError('No finite dE values available for plotting.')

    if vmax_kj is not None and vmax_kj > 0:
        vmax = float(vmax_kj)
    else:
        vmax = float(np.nanpercentile(F_plot[finite], vmax_percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(F_plot[finite]))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    extent = [xi[0], xi[-1], yi[0], yi[-1]]
    F_masked = np.ma.masked_invalid(F_plot.T)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    im = ax.imshow(
        F_masked,
        origin='lower',
        extent=extent,
        aspect='auto',
        interpolation=interpolation,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
    )

    ticks = np.linspace(0.0, vmax, 6)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, extend='max', ticks=ticks)
    cbar.set_label('ΔE = -RT ln(P)  [kJ mol$^{-1}$]\nminimum shifted to 0')

    if show_contours:
        contour_levels = np.linspace(0.0, vmax, 10)
        XX, YY = np.meshgrid(xi, yi, indexing='ij')
        try:
            cs = ax.contour(XX, YY, F_plot, levels=contour_levels, linewidths=0.75, colors='k', alpha=0.42)
            ax.clabel(cs, cs.levels[::2], inline=True, fontsize=7, fmt='%.1f')
            try:
                ax.contour(XX, YY, F_plot, levels=[0.0], linewidths=1.2, colors='white', alpha=0.95)
            except Exception:
                pass
        except Exception:
            pass

    overlay_structures(ax, df, highlight_df=highlight_df, highlight_label=highlight_label)
    ax.set_xlabel('PCA1 / imaginary CV1')
    ax.set_ylabel('PCA2 / imaginary CV2')
    ax.set_title('Weighted PCA pseudo-FES  (ΔE = -RT ln P, zero-anchored at minima)')
    ax.legend(loc='best', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def save_fes_panel(F, xcenters, ycenters, df, out_png: Path, RT: float, cmap: str, interpolation: str,
                   smooth_sigma: float, interp_factor: int, vmax_percentile: float, vmax_kj: float | None,
                   highlight_df: pd.DataFrame | None, highlight_label: str, P_grid=None):
    import matplotlib.pyplot as plt

    if P_grid is not None:
        xi_f, yi_f, P_plot, F_plot = prepare_pmf_surface_from_probability(
            P_grid, xcenters, ycenters, RT=RT, smooth_sigma=smooth_sigma, interp_factor=interp_factor
        )
        xi_p, yi_p = xi_f, yi_f
    else:
        xi_p, yi_p, P_plot = prepare_surface(
            probability_surface_from_deltaE(F, RT), xcenters, ycenters,
            smooth_sigma=smooth_sigma, interp_factor=interp_factor, rebase_zero=False
        )
        P_plot = relative_probability(P_plot)
        xi_f, yi_f, F_plot = prepare_surface(
            F, xcenters, ycenters,
            smooth_sigma=smooth_sigma, interp_factor=interp_factor, rebase_zero=True
        )

    finite = np.isfinite(F_plot)
    if not finite.any():
        raise ValueError('No finite dE values available for panel plotting.')

    if vmax_kj is not None and vmax_kj > 0:
        vmax = float(vmax_kj)
    else:
        vmax = float(np.nanpercentile(F_plot[finite], vmax_percentile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(F_plot[finite]))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, axes = plt.subplots(figsize=(15, 6.5), ncols=2)
    panels = [
        (
            axes[0], np.ma.masked_invalid(P_plot.T), [xi_p[0], xi_p[-1], yi_p[0], yi_p[-1]],
            'magma', 0.0, 1.0, 'relative probability  P / Pmax',
            'Weighted occupancy'
        ),
        (
            axes[1], np.ma.masked_invalid(F_plot.T), [xi_f[0], xi_f[-1], yi_f[0], yi_f[-1]],
            cmap, 0.0, vmax, 'ΔE = -RT ln(P)  [kJ mol$^{-1}$]',
            'Pseudo-FES (zero at global minimum)'
        ),
    ]

    for ax, Zmask, extent, cm, vmin, vmax_local, cbar_label, title in panels:
        im = ax.imshow(
            Zmask, origin='lower', extent=extent, aspect='auto', interpolation=interpolation,
            cmap=cm, vmin=vmin, vmax=vmax_local,
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)
        overlay_structures(ax, df, highlight_df=highlight_df, highlight_label=highlight_label)
        ax.set_xlabel('PCA1 / imaginary CV1')
        ax.set_ylabel('PCA2 / imaginary CV2')
        ax.set_title(title)

    axes[0].legend(loc='best', framealpha=0.9)
    fig.suptitle('Interpolated PCA histogram views', y=0.98)
    fig.tight_layout()
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def save_scatter(df, out_png: Path, color_by: str, highlight_df: pd.DataFrame | None = None, highlight_label: str = 'selected survivor minima'):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    if 'sample_weight' in df.columns and df['sample_weight'].max() > 0:
        sizes = 12.0 + 40.0 * np.sqrt(df['sample_weight'] / df['sample_weight'].max())
    else:
        sizes = 18.0

    if color_by in df.columns:
        sc = ax.scatter(df['pc1'], df['pc2'], c=df[color_by], s=sizes, alpha=0.85, edgecolors='black', linewidths=0.25)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_by)
    else:
        ax.scatter(df['pc1'], df['pc2'], s=sizes, alpha=0.85, edgecolors='black', linewidths=0.25)

    if highlight_df is not None and len(highlight_df) > 0:
        ax.scatter(highlight_df['pc1'], highlight_df['pc2'], s=110, marker='*', c='#ff2e88', edgecolors='black', linewidths=0.8, alpha=0.95, label=highlight_label)
        ax.legend(loc='best', framealpha=0.9)

    ax.set_xlabel('PCA1 / imaginary CV1')
    ax.set_ylabel('PCA2 / imaginary CV2')
    ax.set_title(f'PCA projection colored by {color_by}')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_trajectory_plot(df, out_png: Path, connect_sort: str):
    import matplotlib.pyplot as plt

    d = df.copy()
    if connect_sort in d.columns:
        d = d.sort_values(connect_sort)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(d['pc1'], d['pc2'], linewidth=0.8, alpha=0.45)
    ax.scatter(d['pc1'], d['pc2'], s=14, alpha=0.8)
    ax.set_xlabel('PCA1 / imaginary CV1')
    ax.set_ylabel('PCA2 / imaginary CV2')
    ax.set_title('Pseudo-trajectory through PCA seed space')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def save_basin_hop_parent_traces(df, out_png: Path):
    if 'parent_seed' not in df.columns or 'hop_index' not in df.columns:
        return False
    import matplotlib.pyplot as plt

    d = df.dropna(subset=['parent_seed', 'hop_index']).copy()
    if d.empty:
        return False

    fig, ax = plt.subplots(figsize=(8, 6))
    for parent, g in d.groupby('parent_seed'):
        g = g.sort_values('hop_index')
        ax.plot(g['pc1'], g['pc2'], linewidth=0.8, alpha=0.35)
        ax.scatter(g['pc1'], g['pc2'], s=10, alpha=0.55)

    ax.set_xlabel('PCA1 / imaginary CV1')
    ax.set_ylabel('PCA2 / imaginary CV2')
    ax.set_title('Basin-hop endpoint traces in PCA space')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


# -----------------------------
# Adaptive PCA round plotting
# -----------------------------

def frontier_feature_from_ca(ca: np.ndarray, cutoff: float = 8.0, min_sep: int = 3) -> np.ndarray:
    """
    Feature layout used by the adaptive frontier loop in rama_to_aa_minimized_seeds.py:
      binary contact vector + [Rg_A, end_to_end_A, contact_count]
    """
    vals = []
    n = len(ca)
    for i in range(n):
        for j in range(i + min_sep, n):
            d = float(np.linalg.norm(ca[i] - ca[j]))
            vals.append(1.0 if d < cutoff else 0.0)
    cvec = np.asarray(vals, dtype=float)
    rg, e2e, cc = rg_e2e_contact_count(ca, cutoff=cutoff, min_sep=min_sep)
    return np.concatenate([cvec, np.array([rg, e2e, float(cc)], dtype=float)])


def discover_adaptive_rounds(roots):
    rounds = []
    seen = set()
    for root in roots:
        root = Path(root)
        candidates = []
        if root.name.startswith('round_'):
            candidates.append(root)
        if root.name == 'adaptive_pca_exploration':
            candidates.extend(sorted(root.glob('round_*')))
        if (root / 'adaptive_pca_exploration').exists():
            candidates.extend(sorted((root / 'adaptive_pca_exploration').glob('round_*')))
        candidates.extend(sorted(root.glob('adaptive_pca_exploration/round_*')))
        for rd in candidates:
            if not rd.is_dir():
                continue
            if not (rd / 'frontier.pca_points.csv').exists():
                continue
            key = str(rd.resolve())
            if key not in seen:
                seen.add(key)
                rounds.append(rd)
    return sorted(rounds)


def load_first_round_reference_model(round_dirs):
    if not round_dirs:
        raise ValueError('No adaptive round dirs provided.')
    rd = Path(round_dirs[0])
    mean_path = rd / 'frontier.feature_mean.npy'
    std_path = rd / 'frontier.feature_std.npy'
    comp_path = rd / 'frontier.pca_components.npy'
    if not (mean_path.exists() and std_path.exists() and comp_path.exists()):
        raise FileNotFoundError(
            'First adaptive round is missing frontier.feature_mean.npy / frontier.feature_std.npy / '
            'frontier.pca_components.npy required for reprojecting all rounds onto round-1 PCA space.'
        )
    return {
        'round_dir': str(rd),
        'mu': np.load(mean_path),
        'sd': np.load(std_path),
        'components': np.load(comp_path),
    }


def project_features_to_reference_pca(X: np.ndarray, ref_model: dict) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = np.asarray(ref_model['mu'], dtype=float)
    sd = np.asarray(ref_model['sd'], dtype=float).copy()
    sd[sd == 0] = 1.0
    Xz = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return Xz @ np.asarray(ref_model['components'], dtype=float).T


def load_adaptive_rounds_projected(round_dirs, cutoff: float = 8.0, min_sep: int = 3):
    ref_model = load_first_round_reference_model(round_dirs)
    all_points = []
    summary_rows = []

    ref_dim = int(np.asarray(ref_model['mu']).shape[0])

    for round_idx, rd in enumerate(round_dirs, start=1):
        rd = Path(rd)
        pca_csv = rd / 'frontier.pca_points.csv'
        target_csv = rd / 'frontier.pca_target_bins.csv'
        kept_csv = rd / 'pca_frontier_kept_proposals.csv'
        score_csv = rd / 'pca_frontier_implicit_minimization_scores.csv'

        points = pd.read_csv(pca_csv)
        rows = []
        feats = []
        skipped = 0
        for row in points.to_dict(orient='records'):
            pdb_path = Path(str(row.get('pdb_path', '')))
            if not pdb_path.exists():
                skipped += 1
                continue
            ca = read_ca_coords_pdb(pdb_path)
            if len(ca) < 2 or not np.isfinite(ca).all():
                skipped += 1
                continue
            feat = frontier_feature_from_ca(ca, cutoff=cutoff, min_sep=min_sep)
            if feat.shape[0] != ref_dim or not np.isfinite(feat).all():
                skipped += 1
                continue
            rows.append(row)
            feats.append(feat)

        if rows:
            feats = np.vstack(feats)
            ref_scores = project_features_to_reference_pca(feats, ref_model)
            df = pd.DataFrame(rows)
            if 'pc1' in df.columns:
                df = df.rename(columns={'pc1': 'pc1_local'})
            if 'pc2' in df.columns:
                df = df.rename(columns={'pc2': 'pc2_local'})
            df['pc1'] = ref_scores[:, 0]
            df['pc2'] = ref_scores[:, 1]
            df['adaptive_round'] = round_idx
            df['round_dir'] = str(rd)
            all_points.append(df)
            n_valid = len(df)
        else:
            df = pd.DataFrame()
            n_valid = 0

        n_target_bins = 0
        if target_csv.exists():
            try:
                td = pd.read_csv(target_csv)
                if 'target' in td.columns:
                    n_target_bins = int(td['target'].astype(bool).sum())
            except Exception:
                pass

        n_kept = 0
        if kept_csv.exists():
            try:
                kd = pd.read_csv(kept_csv)
                n_kept = len(kd)
            except Exception:
                pass

        n_min_ok = 0
        if score_csv.exists():
            try:
                sd = pd.read_csv(score_csv)
                if 'success' in sd.columns:
                    vals = sd['success']
                    if vals.dtype == bool:
                        n_min_ok = int(vals.sum())
                    else:
                        n_min_ok = int(pd.to_numeric(vals, errors='coerce').fillna(0).astype(int).sum())
            except Exception:
                pass

        summary_rows.append({
            'adaptive_round': round_idx,
            'round_dir': str(rd),
            'n_points_csv': len(points),
            'n_valid_reprojected': n_valid,
            'n_skipped_reproject': skipped,
            'n_target_bins_local': n_target_bins,
            'n_kept_proposals': n_kept,
            'n_minimized_success': n_min_ok,
        })

    points_df = pd.concat(all_points, ignore_index=True) if all_points else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return points_df, summary_df, ref_model


def convex_hull_2d(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return np.empty((0, 2), dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return np.empty((0, 2), dtype=float)
    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=float)
    return hull


def polygon_area_2d(poly_xy: np.ndarray) -> float:
    poly = np.asarray(poly_xy, dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_adaptive_bin_progress(points_df: pd.DataFrame, bins: int = 60) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if points_df.empty:
        return points_df.copy(), pd.DataFrame(), {}

    df = points_df.copy()
    x = df['pc1'].to_numpy(dtype=float)
    y = df['pc2'].to_numpy(dtype=float)
    xr = float(np.ptp(x)) if len(x) else 0.0
    yr = float(np.ptp(y)) if len(y) else 0.0
    padx = max(0.05 * xr, 1e-3)
    pady = max(0.05 * yr, 1e-3)
    xedges = np.linspace(float(np.nanmin(x)) - padx, float(np.nanmax(x)) + padx, int(max(8, bins)) + 1)
    yedges = np.linspace(float(np.nanmin(y)) - pady, float(np.nanmax(y)) + pady, int(max(8, bins)) + 1)

    ix = np.clip(np.searchsorted(xedges, x, side='right') - 1, 0, len(xedges) - 2)
    iy = np.clip(np.searchsorted(yedges, y, side='right') - 1, 0, len(yedges) - 2)
    df['bin_i'] = ix.astype(int)
    df['bin_j'] = iy.astype(int)

    bin_first = (
        df.groupby(['bin_i', 'bin_j'], as_index=False)['adaptive_round']
        .min()
        .rename(columns={'adaptive_round': 'first_round_in_bin'})
    )
    df = df.merge(bin_first, on=['bin_i', 'bin_j'], how='left')
    df['is_new_bin_round'] = df['adaptive_round'].astype(int) == df['first_round_in_bin'].astype(int)

    rounds = sorted(int(r) for r in df['adaptive_round'].unique())
    first_round_grid = np.full((len(xedges) - 1, len(yedges) - 1), np.nan, dtype=float)
    for row in bin_first.itertuples(index=False):
        first_round_grid[int(row.bin_i), int(row.bin_j)] = float(row.first_round_in_bin)

    cumulative_hulls = {}
    round_hulls = {}
    centroids = {}
    summary_rows = []

    for r in rounds:
        g = df[df['adaptive_round'] == r]
        gc = df[df['adaptive_round'] <= r]
        new_bins = int((bin_first['first_round_in_bin'] == r).sum())
        cumulative_bins = int((bin_first['first_round_in_bin'] <= r).sum())
        hull_round = convex_hull_2d(g[['pc1', 'pc2']].to_numpy(dtype=float))
        hull_cum = convex_hull_2d(gc[['pc1', 'pc2']].to_numpy(dtype=float))
        round_hulls[r] = hull_round
        cumulative_hulls[r] = hull_cum
        centroids[r] = (
            float(np.nanmean(g['pc1'].to_numpy(dtype=float))),
            float(np.nanmean(g['pc2'].to_numpy(dtype=float))),
        )
        summary_rows.append({
            'adaptive_round': r,
            'n_new_bins_round1_space': new_bins,
            'n_cumulative_bins_round1_space': cumulative_bins,
            'round_hull_area_round1_space': polygon_area_2d(hull_round),
            'cumulative_hull_area_round1_space': polygon_area_2d(hull_cum),
        })

    meta = {
        'xedges': xedges,
        'yedges': yedges,
        'first_round_grid': first_round_grid,
        'rounds': rounds,
        'cumulative_hulls': cumulative_hulls,
        'round_hulls': round_hulls,
        'centroids': centroids,
    }
    return df, pd.DataFrame(summary_rows), meta


def save_adaptive_round_plot(points_df: pd.DataFrame, round_idx: int, out_png: Path, title: str, progress_meta: dict, summary_row: pd.Series | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.25, 7.5))
    prev = points_df[points_df['adaptive_round'] < round_idx]
    curr = points_df[points_df['adaptive_round'] == round_idx]

    if not prev.empty:
        ax.scatter(
            prev['pc1'], prev['pc2'],
            s=15,
            alpha=0.22,
            color='lightgray',
            edgecolors='none',
            label='earlier rounds',
        )

    color_col = 'energy_kj_mol' if 'energy_kj_mol' in curr.columns else None
    if color_col:
        sc = ax.scatter(
            curr['pc1'], curr['pc2'],
            c=curr[color_col],
            s=32,
            alpha=0.88,
            cmap='viridis',
            edgecolors='black',
            linewidths=0.25,
            label=f'round {int(round_idx)} structures',
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_col)
    else:
        ax.scatter(
            curr['pc1'], curr['pc2'],
            s=32,
            alpha=0.88,
            color='tab:blue',
            edgecolors='black',
            linewidths=0.25,
            label=f'round {int(round_idx)} structures',
        )

    new_hits = curr[curr.get('is_new_bin_round', False)] if 'is_new_bin_round' in curr.columns else curr.iloc[0:0]
    if not new_hits.empty:
        ax.scatter(
            new_hits['pc1'], new_hits['pc2'],
            s=78,
            marker='s',
            facecolors='none',
            edgecolors='red',
            linewidths=1.0,
            label='new PCA bins reached this round',
        )

    prev_hull = progress_meta.get('cumulative_hulls', {}).get(round_idx - 1)
    curr_hull = progress_meta.get('cumulative_hulls', {}).get(round_idx)
    if prev_hull is not None and len(prev_hull) >= 3:
        closed = np.vstack([prev_hull, prev_hull[0]])
        ax.plot(closed[:, 0], closed[:, 1], linestyle='--', linewidth=1.2, color='0.35', alpha=0.7, label='previous cumulative hull')
    if curr_hull is not None and len(curr_hull) >= 3:
        closed = np.vstack([curr_hull, curr_hull[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=1.8, color='red', alpha=0.88, label='current cumulative hull')

    if summary_row is not None and len(summary_row):
        box = (
            f"new bins: {int(summary_row.get('n_new_bins_round1_space', 0))}\n"
            f"cumulative bins: {int(summary_row.get('n_cumulative_bins_round1_space', 0))}\n"
            f"cum. hull area: {summary_row.get('cumulative_hull_area_round1_space', 0.0):.2f}"
        )
        ax.text(
            0.02, 0.98, box,
            transform=ax.transAxes,
            va='top', ha='left',
            fontsize=9.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.86, edgecolor='0.55'),
        )

    ax.set_xlabel('PCA1 / round-1 reference PCA space')
    ax.set_ylabel('PCA2 / round-1 reference PCA space')
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.92)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_adaptive_all_rounds_overlay(points_df: pd.DataFrame, out_png: Path, progress_meta: dict):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 7.8))
    rounds = progress_meta.get('rounds', sorted(points_df['adaptive_round'].unique()))
    cmap = plt.get_cmap('turbo', max(1, len(rounds)))

    for i, r in enumerate(rounds):
        g = points_df[points_df['adaptive_round'] == r]
        color = cmap(i)
        ax.scatter(
            g['pc1'], g['pc2'],
            s=22,
            alpha=0.62,
            color=color,
            edgecolors='black',
            linewidths=0.18,
            label=f'round {int(r)}',
        )
        hull = progress_meta.get('cumulative_hulls', {}).get(r)
        if hull is not None and len(hull) >= 3:
            closed = np.vstack([hull, hull[0]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.55, alpha=0.92)
        centroid = progress_meta.get('centroids', {}).get(r)
        if centroid is not None and np.isfinite(centroid).all():
            ax.scatter([centroid[0]], [centroid[1]], s=90, marker='o', color=color, edgecolors='white', linewidths=1.1, zorder=5)
            ax.text(centroid[0], centroid[1], str(int(r)), color='white', ha='center', va='center', fontsize=8.5, weight='bold', zorder=6)

    centroid_path = np.array([progress_meta.get('centroids', {}).get(r, (np.nan, np.nan)) for r in rounds], dtype=float)
    finite = np.isfinite(centroid_path).all(axis=1)
    centroid_path = centroid_path[finite]
    if len(centroid_path) >= 2:
        ax.plot(centroid_path[:, 0], centroid_path[:, 1], color='black', linewidth=1.0, alpha=0.5, linestyle='--', label='round centroid path')

    ax.set_xlabel('PCA1 / round-1 reference PCA space')
    ax.set_ylabel('PCA2 / round-1 reference PCA space')
    ax.set_title('Adaptive exploration growth in round-1 PCA space')
    ax.legend(loc='best', framealpha=0.92, ncol=2)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_adaptive_first_seen_map(points_df: pd.DataFrame, out_png: Path, progress_meta: dict):
    import matplotlib.pyplot as plt

    xedges = progress_meta['xedges']
    yedges = progress_meta['yedges']
    grid = progress_meta['first_round_grid']
    rounds = progress_meta['rounds']

    fig, ax = plt.subplots(figsize=(9.5, 7.8))
    cmap = plt.get_cmap('turbo', max(1, len(rounds)))
    masked = np.ma.masked_invalid(grid.T)
    mesh = ax.pcolormesh(
        xedges,
        yedges,
        masked,
        shading='auto',
        cmap=cmap,
        vmin=min(rounds) - 0.5,
        vmax=max(rounds) + 0.5,
        alpha=0.95,
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02, ticks=rounds)
    cbar.set_label('first round reaching each PCA bin')

    ax.scatter(points_df['pc1'], points_df['pc2'], s=6, color='black', alpha=0.12, edgecolors='none')
    for i, r in enumerate(rounds):
        color = cmap(i)
        hull = progress_meta.get('cumulative_hulls', {}).get(r)
        if hull is not None and len(hull) >= 3:
            closed = np.vstack([hull, hull[0]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, linewidth=1.35, alpha=0.9)
        centroid = progress_meta.get('centroids', {}).get(r)
        if centroid is not None and np.isfinite(centroid).all():
            ax.text(centroid[0], centroid[1], str(int(r)), color='black', fontsize=9, weight='bold', ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.18', facecolor='white', alpha=0.75, edgecolor=color, linewidth=1.0))

    ax.set_xlabel('PCA1 / round-1 reference PCA space')
    ax.set_ylabel('PCA2 / round-1 reference PCA space')
    ax.set_title('Exploration chronology map: bins colored by the first round that reached them')
    ax.grid(alpha=0.12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_adaptive_round_summary_plot(summary_df: pd.DataFrame, out_png: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    x = summary_df['adaptive_round'].to_numpy(dtype=float)
    for col, label in [
        ('n_points_csv', 'round source points'),
        ('n_valid_reprojected', 'reprojected points'),
        ('n_target_bins_local', 'local target bins'),
        ('n_kept_proposals', 'kept proposals'),
        ('n_minimized_success', 'successful minimized'),
        ('n_new_bins_round1_space', 'new bins in round-1 PCA space'),
        ('n_cumulative_bins_round1_space', 'cumulative bins in round-1 PCA space'),
    ]:
        if col in summary_df.columns:
            ax.plot(x, summary_df[col], marker='o', linewidth=1.8, label=label)
    ax.set_xlabel('adaptive round')
    ax.set_ylabel('count')
    ax.set_title('Adaptive exploration round summary')
    ax.legend(loc='best', framealpha=0.9, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def save_adaptive_space_growth_plot(summary_df: pd.DataFrame, out_png: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    x = summary_df['adaptive_round'].to_numpy(dtype=float)
    if 'n_new_bins_round1_space' in summary_df.columns:
        ax.bar(x - 0.15, summary_df['n_new_bins_round1_space'], width=0.30, alpha=0.75, label='new bins this round')
    if 'n_cumulative_bins_round1_space' in summary_df.columns:
        ax.plot(x, summary_df['n_cumulative_bins_round1_space'], marker='o', linewidth=2.0, label='cumulative occupied bins')
    ax.set_xlabel('adaptive round')
    ax.set_ylabel('occupied PCA bins')
    ax.grid(alpha=0.22)
    ax2 = ax.twinx()
    if 'cumulative_hull_area_round1_space' in summary_df.columns:
        ax2.plot(x, summary_df['cumulative_hull_area_round1_space'], marker='s', linewidth=1.8, linestyle='--', color='tab:red', label='cumulative hull area')
    ax2.set_ylabel('cumulative hull area')

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc='best', framealpha=0.9)
    ax.set_title('How each exploration round extends the PCA map')
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def plot_adaptive_rounds(args):
    roots = args.adaptive_root if args.adaptive_root else args.pdb_dir
    round_dirs = discover_adaptive_rounds(roots)
    if not round_dirs:
        raise SystemExit(
            'No adaptive PCA rounds found. Expected files like '
            'adaptive_pca_exploration/round_01/frontier.pca_points.csv'
        )

    out_dir = args.out / 'adaptive_pca_rounds'
    out_dir.mkdir(parents=True, exist_ok=True)

    points_df, summary_df, ref_model = load_adaptive_rounds_projected(
        round_dirs,
        cutoff=args.contact_cutoff,
        min_sep=args.contact_min_sep,
    )
    if points_df.empty:
        raise SystemExit('Adaptive rounds were found, but no structures could be reprojected into round-1 PCA space.')

    points_df, progress_summary, progress_meta = compute_adaptive_bin_progress(points_df, bins=max(50, int(args.bins)))
    if not progress_summary.empty:
        summary_df = summary_df.merge(progress_summary, on='adaptive_round', how='left')

    points_df.to_csv(out_dir / 'all_round_frontier_pca_points_round1_space.csv', index=False)
    summary_df.to_csv(out_dir / 'adaptive_round_summary.csv', index=False)
    np.save(out_dir / 'reference_round1_feature_mean.npy', ref_model['mu'])
    np.save(out_dir / 'reference_round1_feature_std.npy', ref_model['sd'])
    np.save(out_dir / 'reference_round1_pca_components.npy', ref_model['components'])
    np.save(out_dir / 'first_round_reached_bin_grid.npy', progress_meta['first_round_grid'])
    np.save(out_dir / 'first_round_reached_bin_xedges.npy', progress_meta['xedges'])
    np.save(out_dir / 'first_round_reached_bin_yedges.npy', progress_meta['yedges'])

    for rd_idx in sorted(points_df['adaptive_round'].unique()):
        row = summary_df[summary_df['adaptive_round'] == rd_idx]
        row = row.iloc[0] if not row.empty else None
        save_adaptive_round_plot(
            points_df,
            int(rd_idx),
            out_dir / f'round_{int(rd_idx):02d}_frontier_pca_round1_space.png',
            title=f'Adaptive round {int(rd_idx)} in round-1 PCA space (with map growth)',
            progress_meta=progress_meta,
            summary_row=row,
        )

    save_adaptive_all_rounds_overlay(points_df, out_dir / 'all_rounds_frontier_pca_round1_space_overlay.png', progress_meta)
    save_adaptive_first_seen_map(points_df, out_dir / 'adaptive_rounds_first_seen_bin_map.png', progress_meta)
    save_adaptive_round_summary_plot(summary_df, out_dir / 'adaptive_round_summary.png')
    save_adaptive_space_growth_plot(summary_df, out_dir / 'adaptive_round_space_growth.png')

    summary = {
        'n_rounds': int(len(round_dirs)),
        'round_dirs': [str(x) for x in round_dirs],
        'reference_round': 1,
        'reference_round_dir': ref_model['round_dir'],
        'note': 'All structures from all adaptive rounds were reprojected into the PCA basis of round 1.',
        'extra_outputs': [
            'adaptive_rounds_first_seen_bin_map.png',
            'adaptive_round_space_growth.png',
            'first_round_reached_bin_grid.npy',
        ],
    }
    (out_dir / 'adaptive_pca_rounds_summary.json').write_text(json.dumps(summary, indent=2))

    print()
    print('Adaptive PCA round plots done.')
    print(f'Rounds found: {len(round_dirs)}')
    print(f'Reference PCA basis: round 1 ({ref_model["round_dir"]})')
    print(f'Output: {out_dir}')
    print('New visuals: per-round growth plots, first-seen bin chronology map, and PCA-space growth summary.')
    print('All adaptive-round points are now plotted in the same round-1 PCA space.')
    return out_dir


# -----------------------------
# Main
# -----------------------------

def pca_pseudo_fes_main(argv=None):
    p = argparse.ArgumentParser(description="Build weighted PCA dE=-RTln(P) landscape from PDB seed/minimized structures.")
    p.add_argument("--pdb-dir", required=True, nargs="+", type=Path, help="One or more directories containing PDB files.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--pattern", default="*.pdb")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--features", choices=["distances", "contacts", "mixed"], default="mixed")
    p.add_argument("--contact-cutoff", type=float, default=8.0)
    p.add_argument("--contact-min-sep", type=int, default=3)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--min-count", type=int, default=1)
    p.add_argument("--color-by", default="energy_kj_mol")
    p.add_argument("--connect-sort", default="filename")

    p.add_argument("--weight-mode", choices=["none", "boltzmann", "csv-weight", "inverse-energy-rank"], default="none")
    p.add_argument("--weights-csv", nargs="*", default=[])
    p.add_argument("--auto-weights", action="store_true")
    p.add_argument("--energy-column", default="minimized_energy_kj_mol")
    p.add_argument("--weight-column", default="weight")
    p.add_argument("--energy-cap-kj", type=float, default=80.0)
    p.add_argument("--no-normalize-weights", action="store_true")

    p.add_argument("--fes-cmap", default="turbo")
    p.add_argument("--fes-interpolation", choices=["nearest", "bilinear", "bicubic", "lanczos"], default="bicubic")
    p.add_argument("--fes-smooth-sigma", type=float, default=0.8)
    p.add_argument("--fes-interp-factor", type=int, default=4, help="Densify the PCA histogram for smoother plotted surfaces.")
    p.add_argument("--fes-vmax-percentile", type=float, default=90.0, help="Use a lower percentile to improve contrast and lower the colorbar max.")
    p.add_argument("--fes-vmax-kj", type=float, default=None, help="Absolute maximum for the color scale in kJ/mol. Overrides percentile if set.")
    p.add_argument("--no-fes-contours", action="store_true")

    p.add_argument("--highlight-dir", nargs="*", default=None, help="Directories containing selected survivor minima / final survivors to highlight as stars.")
    p.add_argument("--no-auto-highlight", action="store_true", help="Disable automatic search for common survivor directories.")
    p.add_argument("--highlight-label", default="selected survivor minima")

    p.add_argument("--plot-adaptive-rounds", action="store_true", help="Load all adaptive exploration rounds and plot them all in the PCA space of round 1.")
    p.add_argument("--adaptive-root", nargs="*", type=Path, default=None, help="Main run dir, adaptive_pca_exploration dir, or explicit round_* dirs. Defaults to --pdb-dir roots.")

    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.plot_adaptive_rounds:
        plot_adaptive_rounds(args)
        return

    input_pdb_dirs, expansion_notes = expand_pdb_dirs(args.pdb_dir, pattern=args.pattern, recursive=args.recursive)
    pdbs = collect_pdbs(input_pdb_dirs, pattern=args.pattern, recursive=args.recursive)
    if not pdbs:
        raise SystemExit(
            f"No PDBs found in: {args.pdb_dir}. "
            "Point --pdb-dir at a PDB directory, use --recursive, or pass a run root with standard output folders."
        )

    weight_csvs = [Path(x) for x in args.weights_csv]
    if args.auto_weights:
        weight_csvs.extend(candidate_score_tables(input_pdb_dirs))
    weight_lookup, used_tables = load_weight_records(weight_csvs, args.energy_column, args.weight_column)

    highlight_dirs = []
    if args.highlight_dir is not None:
        highlight_dirs.extend(Path(x) for x in args.highlight_dir)
    if not args.no_auto_highlight:
        highlight_dirs.extend(auto_highlight_dirs(input_pdb_dirs))
    # dedupe existing dirs only
    seen = set()
    clean_highlight_dirs = []
    for d in highlight_dirs:
        if d.exists() and d.is_dir():
            key = str(d.resolve())
            if key not in seen:
                seen.add(key)
                clean_highlight_dirs.append(d)
    highlight_keys, highlight_files = highlight_key_set(clean_highlight_dirs, pattern=args.pattern, recursive=args.recursive)

    if expansion_notes:
        print("Expanded run-root inputs:")
        for root, dirs in expansion_notes:
            print(f"  - {root} -> {len(dirs)} PDB directories")
            for d in dirs[:10]:
                print(f"      {d}")
            if len(dirs) > 10:
                print(f"      ... {len(dirs) - 10} more")
    print(f"Reading PDBs: {len(pdbs)}")
    if weight_csvs:
        print(f"Weight tables requested/found: {len(weight_csvs)}")
        print(f"Weight tables used: {len(used_tables)}")
        for t in used_tables[:10]:
            print(f"  - {t}")
    if clean_highlight_dirs:
        print(f"Highlight dirs: {len(clean_highlight_dirs)}")
        for d in clean_highlight_dirs[:10]:
            print(f"  - {d}")

    rows, feats = [], []
    n_ca_expected = None
    skipped = 0

    for idx, pdb in enumerate(pdbs):
        ca = read_ca_coords_pdb(pdb)
        if len(ca) < 2:
            skipped += 1
            continue
        if n_ca_expected is None:
            n_ca_expected = len(ca)
        if len(ca) != n_ca_expected:
            skipped += 1
            continue

        feat = make_features(ca, args.features, args.contact_cutoff, args.contact_min_sep)
        if not np.isfinite(feat).all():
            skipped += 1
            continue

        rg, e2e, cc = rg_e2e_contact_count(ca, args.contact_cutoff, args.contact_min_sep)
        info = match_weight_info(pdb, weight_lookup)

        row = {
            "index": idx,
            "filename": pdb.name,
            "stem": pdb.stem,
            "pdb_path": str(pdb),
            "source_dir": str(pdb.parent),
            "rg_A": rg,
            "end_to_end_A": e2e,
            "contact_count": cc,
            "is_highlight": bool(file_match_keys(pdb) & highlight_keys),
        }
        if "energy" in info:
            row["energy_kj_mol"] = info["energy"]
        if "raw_weight" in info:
            row["raw_weight"] = info["raw_weight"]
        for extra in ["seed_name", "parent_seed", "hop_index", "mode"]:
            if extra in info:
                row[extra] = info[extra]
        rows.append(row)
        feats.append(feat)

    if len(feats) < 3:
        raise SystemExit(f"Need at least 3 valid structures. Valid={len(feats)}, skipped={skipped}")

    X = np.vstack(feats)
    scores, components, explained, _, _ = pca_svd(X, 2)

    df = pd.DataFrame(rows)
    df["pc1"] = scores[:, 0]
    df["pc2"] = scores[:, 1]

    if args.weight_mode != "none" and "energy_kj_mol" not in df.columns and "raw_weight" not in df.columns:
        raise SystemExit(
            f"Weight mode {args.weight_mode!r} requested, but no matched energies/weights found. "
            "Use --weights-csv or --auto-weights and check path matching."
        )

    df["sample_weight"] = compute_sample_weights(
        df,
        args.weight_mode,
        args.temperature,
        args.energy_cap_kj,
        normalize=not args.no_normalize_weights,
    )

    RT = 0.00831446261815324 * args.temperature
    H_count, H_weight, P, F, xcenters, ycenters = deltaE_2d(
        df["pc1"].values,
        df["pc2"].values,
        bins=args.bins,
        RT=RT,
        min_count=args.min_count,
        weights=df["sample_weight"].values,
    )

    highlight_df = df[df["is_highlight"]].copy()

    df.to_csv(args.out / "pca_points.csv", index=False)
    highlight_df.to_csv(args.out / "highlighted_survivor_points.csv", index=False)
    np.save(args.out / "features.npy", X)
    np.save(args.out / "pca_components.npy", components)
    np.save(args.out / "pca_explained_variance_ratio.npy", explained)
    np.save(args.out / "hist_counts_unweighted.npy", H_count)
    np.save(args.out / "hist_weights.npy", H_weight)
    P_for_pmf = P.copy()
    P_for_pmf[H_count < args.min_count] = np.nan
    P_relative = relative_probability(P_for_pmf)
    P_smoothed = relative_probability(smooth_nan_grid(P_relative, args.fes_smooth_sigma))
    F_smoothed = pmf_from_probability(P_smoothed, RT)
    xi_f, yi_f, P_interpolated, F_interpolated = prepare_pmf_surface_from_probability(
        P_for_pmf, xcenters, ycenters,
        RT=RT,
        smooth_sigma=args.fes_smooth_sigma,
        interp_factor=args.fes_interp_factor,
    )

    np.save(args.out / "probability_weighted.npy", P)
    np.save(args.out / "probability_relative_to_max.npy", P_relative)
    np.save(args.out / "probability_relative_to_max_interpolated.npy", P_interpolated)
    np.save(args.out / "deltaE_pca_kjmol.npy", F)
    np.save(args.out / "deltaE_pca_kjmol_smoothed.npy", F_smoothed)
    np.save(args.out / "deltaE_pca_kjmol_interpolated.npy", F_interpolated)
    np.save(args.out / "pca1_centers.npy", xcenters)
    np.save(args.out / "pca2_centers.npy", ycenters)
    np.save(args.out / "pca1_centers_interpolated.npy", xi_f)
    np.save(args.out / "pca2_centers_interpolated.npy", yi_f)

    grid_rows = []
    for i, xc in enumerate(xcenters):
        for j, yc in enumerate(ycenters):
            grid_rows.append({
                "pc1": xc,
                "pc2": yc,
                "count_unweighted": H_count[i, j],
                "weight_sum": H_weight[i, j],
                "probability_weighted": P[i, j],
                "dE_kj_mol": F[i, j],
            })
    pd.DataFrame(grid_rows).to_csv(args.out / "deltaE_grid.csv", index=False)

    interp_rows = []
    for i, xc in enumerate(xi_f):
        for j, yc in enumerate(yi_f):
            interp_rows.append({
                "pc1": xc,
                "pc2": yc,
                "probability_relative": P_interpolated[i, j] if np.isfinite(P_interpolated[i, j]) else np.nan,
                "dE_kj_mol": F_interpolated[i, j] if np.isfinite(F_interpolated[i, j]) else np.nan,
            })
    pd.DataFrame(interp_rows).to_csv(args.out / "deltaE_grid_interpolated.csv", index=False)

    summary = {
        "input_dirs": [str(x) for x in args.pdb_dir],
        "expanded_pdb_dirs": [str(x) for x in input_pdb_dirs],
        "n_input_pdbs": len(pdbs),
        "n_valid_structures": len(df),
        "n_skipped": skipped,
        "n_ca": n_ca_expected,
        "features": args.features,
        "feature_dim": int(X.shape[1]),
        "temperature_K": args.temperature,
        "RT_kj_mol": RT,
        "weight_mode": args.weight_mode,
        "weights_csv_used": used_tables,
        "n_structures_with_energy": int(df["energy_kj_mol"].notna().sum()) if "energy_kj_mol" in df.columns else 0,
        "sample_weight_sum": float(df["sample_weight"].sum()),
        "sample_weight_min": float(df["sample_weight"].min()),
        "sample_weight_max": float(df["sample_weight"].max()),
        "effective_sample_size": effective_sample_size(df["sample_weight"].values),
        "pca1_explained_variance_ratio": float(explained[0]),
        "pca2_explained_variance_ratio": float(explained[1]),
        "n_highlight_structures": int(len(highlight_df)),
        "highlight_dirs": [str(x) for x in clean_highlight_dirs],
        "fes_cmap": args.fes_cmap,
        "fes_interpolation": args.fes_interpolation,
        "fes_smooth_sigma": args.fes_smooth_sigma,
        "fes_interp_factor": args.fes_interp_factor,
        "fes_vmax_percentile": args.fes_vmax_percentile,
        "fes_vmax_kj": args.fes_vmax_kj,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    try:
        save_heatmap(
            F,
            xcenters,
            ycenters,
            df,
            args.out / "weighted_deltaE_pca1_pca2.png",
            cmap=args.fes_cmap,
            interpolation=args.fes_interpolation,
            smooth_sigma=args.fes_smooth_sigma,
            interp_factor=args.fes_interp_factor,
            vmax_percentile=args.fes_vmax_percentile,
            vmax_kj=args.fes_vmax_kj,
            show_contours=not args.no_fes_contours,
            highlight_df=highlight_df,
            highlight_label=args.highlight_label,
            P_grid=P_for_pmf,
            RT=RT,
        )
        save_probability_heatmap(
            F,
            xcenters,
            ycenters,
            df,
            args.out / "weighted_probability_pca1_pca2.png",
            RT=RT,
            interpolation=args.fes_interpolation,
            smooth_sigma=args.fes_smooth_sigma,
            interp_factor=args.fes_interp_factor,
            highlight_df=highlight_df,
            highlight_label=args.highlight_label,
        )
        save_fes_panel(
            F,
            xcenters,
            ycenters,
            df,
            args.out / "weighted_probability_and_deltaE_panel.png",
            RT=RT,
            cmap=args.fes_cmap,
            interpolation=args.fes_interpolation,
            smooth_sigma=args.fes_smooth_sigma,
            interp_factor=args.fes_interp_factor,
            vmax_percentile=args.fes_vmax_percentile,
            vmax_kj=args.fes_vmax_kj,
            highlight_df=highlight_df,
            highlight_label=args.highlight_label,
            P_grid=P_for_pmf,
        )
        save_scatter(df, args.out / f"pca_scatter_by_{args.color_by}.png", args.color_by, highlight_df=highlight_df, highlight_label=args.highlight_label)
        save_scatter(df, args.out / "pca_scatter_by_sample_weight.png", "sample_weight", highlight_df=highlight_df, highlight_label=args.highlight_label)
        save_trajectory_plot(df, args.out / "pca_pseudo_trajectory.png", args.connect_sort)
        save_basin_hop_parent_traces(df, args.out / "pca_basin_hop_parent_traces.png")
    except Exception as exc:
        print(f"Plotting failed: {exc}")

    print()
    print("Done.")
    print(f"Valid structures: {len(df)} / {len(pdbs)}")
    print(f"Skipped: {skipped}")
    print(f"Highlighted survivor minima matched: {len(highlight_df)} / {len(highlight_files)} files")
    print(f"Feature dim: {X.shape[1]}")
    print(f"PCA explained variance: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Effective sample size: {effective_sample_size(df['sample_weight'].values):.2f} / {len(df)}")
    if "energy_kj_mol" in df.columns:
        print(f"Structures with matched energy: {int(df['energy_kj_mol'].notna().sum())} / {len(df)}")
    print(f"Output: {args.out}")
    print()
    print("Reminder: dE=-RT ln(P) is shifted to min=0. Minimized-energy weighting is heuristic and ignores basin entropy.")




# -----------------------------
# CLI
# -----------------------------

def parse_args(argv=None):
    argv_list = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre.add_argument("--strict-config", action="store_true")
    pre_args, _ = pre.parse_known_args(argv_list)

    p = argparse.ArgumentParser(description="Fast Ramachandran + implicit basin-hopping peptide seed workflow.")

    p.add_argument("--config", default=None, help="YAML/JSON config file. In combined workflow YAMLs, GENPEPT reads the top-level conformer_generation/genpept/seed_generation block and ignores GAREUS blocks.")
    p.add_argument("--strict-config", action="store_true", help="Treat unknown keys inside the GENPEPT config block as fatal. Default is friendly/warn-only for combined GaREUS YAMLs.")
    p.add_argument("--seq", default=None)
    p.add_argument("--out", default=None, type=Path)
    p.add_argument("--clean", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip pipeline stages whose output CSVs already exist. "
                        "Safe restart after any between-stage interruption.")

    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--angle-sd", type=float, default=20.0)
    p.add_argument("--generation-backend", choices=["fast", "full"], default="fast",
                   help="Initial conformer backend. fast screens proposals with an approximate backbone CA trace and all-atom-checks selected candidates; full builds and clash-checks every proposal.")
    p.add_argument("--rama-sampling", choices=["random", "stratified"], default="stratified",
                   help="Ramachandran sampling strategy. stratified uses deterministic low-discrepancy state and phi/psi angle draws to improve batch coverage.")
    p.add_argument("--initial-diversity-features", choices=["shape", "torsion", "mixed"], default="mixed",
                   help="Feature space used to select initial candidate seeds. mixed covers both contact/shape and Ramachandran torsion/state diversity.")
    p.add_argument("--shape-feature-weight", type=float, default=1.0,
                   help="Weight for contact/Rg/E2E shape features during initial candidate selection.")
    p.add_argument("--state-feature-weight", type=float, default=0.75,
                   help="Weight for one-hot Ramachandran state-string features during initial candidate selection.")
    p.add_argument("--torsion-feature-weight", type=float, default=0.35,
                   help="Weight for circular sin/cos(phi/psi) torsion features during initial candidate selection.")
    p.add_argument("--diversity-bank-preset", choices=["off", "broad", "chignolin", "turbo"], default="off",
                   help="Generate conformers from multiple Ramachandran-prior banks before global diversity selection. 'broad/turbo' covers generic alpha/beta/turn/left/wide banks; 'chignolin' biases toward hairpin/turn-rich banks.")
    p.add_argument("--diversity-bank-wide-angle-sd", type=float, default=45.0,
                   help="Angle SD used by the wide_random diversity bank.")
    p.add_argument("--preselection-bin-quota", type=int, default=0,
                   help="If >0, keep at most this many generated conformers per coarse Rg/E2E/contact-count/bin before clustering. Cheaply removes duplicate shapes.")
    p.add_argument("--preselection-rg-bin-A", type=float, default=1.0)
    p.add_argument("--preselection-e2e-bin-A", type=float, default=1.0)
    p.add_argument("--preselection-contact-bin", type=int, default=2)
    p.add_argument("--preselection-include-bank", action=argparse.BooleanOptionalAction, default=True,
                   help="Include diversity-bank name in quota bins so each bank keeps its own coarse coverage. Default: enabled.")
    p.add_argument("--clash-cutoff", type=float, default=1.6)
    p.add_argument("--max-clashes", type=int, default=0)
    p.add_argument("--contact-cutoff", type=float, default=8.0)
    p.add_argument("--contact-min-sep", type=int, default=3)
    p.add_argument("--n-candidate-seeds", type=int, default=60)
    p.add_argument("--n-final-seeds", type=int, default=20)
    p.add_argument("--cluster-method", choices=["minibatch", "farthest"], default="minibatch")
    p.add_argument("--keep-all-pdbs", action="store_true")
    p.add_argument("--write-conformer-table", action="store_true",
                   help="Write full conformers.csv for all accepted conformers. Slower on large runs.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--progress-every", type=int, default=1000)

    p.add_argument("--jobs", type=int, default=1, help="Conformer generation workers.")
    p.add_argument("--gen-chunk-size", type=int, default=250,
                   help="Approximate number of conformers generated per worker task.")
    p.add_argument("--gen-chunks-per-worker", type=int, default=4,
                   help="Minimum number of generation chunks per worker.")
    p.add_argument("--min-jobs", type=int, default=0, help="OpenMM workers. 0 means use --jobs. For one GPU, use 1.")
    p.add_argument("--min-chunk-size", type=int, default=0,
                   help="PDBs per OpenMM worker task in parallel minimization. 0 chooses a responsive default: implicit=4, explicit=1. Single-worker implicit still updates after every seed while reusing one Context.")
    p.add_argument("--adaptive-min", action=argparse.BooleanOptionalAction, default=True,
                   help="Use chunked adaptive minimization for implicit minimization and basin hopping. Default: enabled.")
    p.add_argument("--adaptive-explicit-min", action=argparse.BooleanOptionalAction, default=False,
                   help="Also use adaptive chunked minimization for explicit solvent. Default: disabled.")
    p.add_argument("--adaptive-min-chunk-iterations", type=int, default=25,
                   help="Iterations per adaptive minimization chunk.")
    p.add_argument("--adaptive-min-energy-tol", type=float, default=0.05,
                   help="Stop after repeated chunks when absolute energy drop is below this kJ/mol and force criterion is also met.")
    p.add_argument("--adaptive-min-force-tol", type=float, default=500.0,
                   help="Stop after repeated chunks when max force is below this kJ/mol/nm and energy criterion is also met.")
    p.add_argument("--adaptive-min-stall-rounds", type=int, default=2,
                   help="Number of consecutive low-change chunks required for adaptive minimization stop.")
    p.add_argument("--adaptive-min-force-check-every", type=int, default=1,
                   help="Force/position readback cadence during adaptive minimization chunks. 1 preserves original behavior; 0 uses energy-only early stop and checks forces once at the end; N>1 checks every N chunks.")
    p.add_argument("--assume-same-implicit-topology", action=argparse.BooleanOptionalAction, default=False,
                   help="Skip repeated topology-signature checks after the first implicit Context build. Safe for GENPEPT candidate batches from one sequence; disable for mixed-topology inputs.")

    p.add_argument("--two-stage", action="store_true")
    p.add_argument("--implicit-only", action="store_true")
    p.add_argument("--no-explicit", action="store_true")

    p.add_argument("--basin-hop", action="store_true")
    p.add_argument("--explore-bh", action=argparse.BooleanOptionalAction, default=True,
                   help="When --basin-hop and --explore-loop are active, restart BH/MC from newly minimized PCA-frontier hits after each adaptive round. Default: enabled.")
    p.add_argument("--bh-parent-seeds", type=int, default=0,
                   help="If >0, basin-hop only this many diverse parent seeds selected from candidate_seeds.")
    p.add_argument("--bh-steps", type=int, default=8, help="Number of MD-hop endpoints per candidate seed; plus hop 0 parent minimum.")
    p.add_argument("--bh-md-steps", type=int, default=150, help="Langevin steps per hop.")
    p.add_argument("--bh-temperature", type=float, default=600.0)
    p.add_argument("--bh-friction", type=float, default=5.0)
    p.add_argument("--bh-timestep", type=float, default=0.001, help="ps")
    p.add_argument("--bh-hmr-mass", type=float, default=0.0,
                   help="Hydrogen mass in amu for basin-hop MD only. Use 3.0-4.0 with a larger --bh-timestep, e.g. 0.003-0.004 ps. 0 disables HMR.")
    p.add_argument("--bh-hmr-auto-constraints", action=argparse.BooleanOptionalAction, default=True,
                   help="When --bh-hmr-mass > 0 and global --constraints is none, use HBonds constraints for the BH system. Default: enabled.")
    p.add_argument("--bh-constraint-tolerance", type=float, default=1.0e-5,
                   help="Constraint tolerance for the basin-hop Langevin integrator.")
    p.add_argument("--bh-initial-min-iterations", type=int, default=200)
    p.add_argument("--bh-min-iterations", type=int, default=150)
    p.add_argument("--bh-restart-from-parent", action="store_true", default=True)
    p.add_argument("--bh-chunk-size", type=int, default=0,
                   help="Parent PDBs per reusable basin-hop worker batch. 0 chooses one batch for single-worker runs and 4 for parallel runs.")

    p.add_argument("--nma-expand", action="store_true")
    p.add_argument("--nma-modes", type=int, default=1)
    p.add_argument("--nma-amplitudes", default="1.5")
    p.add_argument("--nma-cutoff", type=float, default=12.0)

    p.add_argument("--ff", choices=sorted(FF_CHOICES), default="amber14")
    p.add_argument("--water", default="tip3pfb")
    p.add_argument("--padding", type=float, default=0.7)
    p.add_argument("--ionic-strength", type=float, default=0.15)
    p.add_argument("--no-neutralize", action="store_true")
    p.add_argument("--ph", type=float, default=7.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--friction", type=float, default=1.0)
    p.add_argument("--cutoff", type=float, default=1.0)
    p.add_argument("--implicit-max-iterations", type=int, default=300)
    p.add_argument("--explicit-max-iterations", type=int, default=500)
    p.add_argument("--tiered-implicit-min", action=argparse.BooleanOptionalAction, default=False,
                   help="Run a fast scout implicit minimization on many candidates, diversity-select a smaller subset, then refine only those. Speeds up diverse basin discovery.")
    p.add_argument("--tier-scout-iterations", type=int, default=25,
                   help="Max iterations for the scout minimization tier.")
    p.add_argument("--tier-scout-adaptive-min", action=argparse.BooleanOptionalAction, default=False,
                   help="Use adaptive minimization during the scout tier. Default off for speed.")
    p.add_argument("--tier-refine-seeds", type=int, default=0,
                   help="Number of scout-minimized structures passed to the refine tier. 0 uses --n-candidate-seeds.")
    p.add_argument("--tier-refine-iterations", type=int, default=0,
                   help="Max iterations for the refine tier. 0 uses --implicit-max-iterations.")
    p.add_argument("--tier-refine-from-scout-pdbs", action=argparse.BooleanOptionalAction, default=True,
                   help="Start refinement from scout-minimized PDBs rather than raw candidates. Default: enabled.")
    p.add_argument("--turbo-summary", action=argparse.BooleanOptionalAction, default=True,
                   help="Write an end-of-run GENPEPT_turbo_summary.json with counts, knobs, and key output files. Default: enabled.")
    p.add_argument("--restraint-k", type=float, default=0.0)
    p.add_argument("--platform", default="auto", help="auto, CPU, CUDA, OpenCL, Reference")
    p.add_argument("--precision", default="mixed", choices=["single", "mixed", "double"])
    p.add_argument("--device-index", default="")
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument("--save-systems", action="store_true")
    p.add_argument("--constraints", choices=["none", "hbonds", "allbonds", "hangles"], default="none",
                   help="OpenMM constraints for minimization/BH systems. Default none; use hbonds to restore old behavior.")
    p.set_defaults(smart_search=True, bh_from_candidates=False)
    p.add_argument("--no-smart-search", dest="smart_search", action="store_false", help=argparse.SUPPRESS)
    p.add_argument("--bh-from-candidates", dest="bh_from_candidates", action="store_true", help=argparse.SUPPRESS)

    # Adaptive PCA frontier exploration. This is a cheap proposal/minimize loop:
    # current relaxed archive -> 2D PCA/contact-shape frontier map -> cheap
    # Ramachandran proposals -> keep sparse/frontier hits -> implicit minimize.
    p.add_argument("--explore-loop", action="store_true",
                   help="Enable adaptive PCA frontier exploration after BH/NMA, before final survivor selection.")
    p.add_argument("--explore-rounds", type=int, default=0,
                   help="Number of adaptive PCA frontier rounds. If --explore-loop is set and this is 0, a safe default of 1 is used.")
    p.add_argument("--explore-proposals", type=int, default=0,
                   help="Cheap Ramachandran proposals generated per PCA frontier round. If 0 with --explore-loop, defaults to max(10000, 20*explore_keep).")
    p.add_argument("--explore-keep", type=int, default=0,
                   help="Maximum frontier proposal PDBs kept/minimized per round. If 0 with --explore-loop, defaults to max(100, n_final_seeds//5).")
    p.add_argument("--explore-max-kept-per-bin", type=int, default=5,
                   help="Maximum accepted adaptive-PCA proposals kept per PCA bin before minimization. Use 0 to disable the per-bin cap.")
    p.add_argument("--explore-bins", type=int, default=40)
    p.add_argument("--explore-target-max-count", type=int, default=0)
    p.add_argument("--explore-pca-padding", type=float, default=0.10)
    p.add_argument("--explore-frontier-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--explore-allow-outside", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--live-hist", action="store_true")
    p.add_argument("--hist-every", type=int, default=1000,
                   help="Update live ASCII histograms every N completed items/chunks.")
    p.add_argument("--no-clear-screen", action="store_true")
    p.add_argument("--no-color", action="store_true")

    p.add_argument("--reject-max-force", type=float, default=1.0e5)
    p.add_argument("--reject-min-rg", type=float, default=0.05)
    p.add_argument("--reject-max-rg", type=float, default=0.0)
    p.add_argument("--reject-min-e2e", type=float, default=0.02)
    p.add_argument("--reject-max-e2e", type=float, default=0.0)


    # Integrated PCA / pseudo-FES post-processing. These options run the former
    # pca_pseudo_fes_from_seeds.py stage automatically at the end of this workflow.
    p.add_argument("--pca-fes", action="store_true",
                   help="After seed generation/minimization, build PCA pseudo-FES plots from this run's PDB outputs.")
    p.add_argument("--pca-out", type=Path, default=None,
                   help="Output directory for PCA pseudo-FES files. Default: <out>/pca_pseudo_fes")
    p.add_argument("--pca-pdb-dir", nargs="*", type=Path, default=None,
                   help="PDB directories or run roots for PCA. Default: the main --out run root.")
    p.add_argument("--pca-features", choices=["distances", "contacts", "mixed"], default="mixed")
    p.add_argument("--pca-bins", type=int, default=50)
    p.add_argument("--pca-min-count", type=int, default=1)
    p.add_argument("--pca-weight-mode", choices=["none", "boltzmann", "csv-weight", "inverse-energy-rank"], default="none")
    p.add_argument("--pca-auto-weights", action=argparse.BooleanOptionalAction, default=True,
                   help="Automatically search standard workflow CSV score tables for PCA weights/energies. Default: enabled.")
    p.add_argument("--pca-energy-column", default="minimized_energy_kj_mol")
    p.add_argument("--pca-weight-column", default="weight")
    p.add_argument("--pca-energy-cap-kj", type=float, default=80.0)
    p.add_argument("--pca-color-by", default="energy_kj_mol")
    p.add_argument("--pca-fes-cmap", default="turbo")
    p.add_argument("--pca-fes-vmax-kj", type=float, default=None)
    p.add_argument("--pca-fes-vmax-percentile", type=float, default=90.0)
    p.add_argument("--pca-plot-adaptive-rounds", action="store_true",
                   help="Also run the adaptive-round PCA growth plots if adaptive_pca_exploration outputs exist.")
    p.add_argument("--pca-args", default="",
                   help="Advanced: extra quoted arguments passed directly to the integrated PCA parser, e.g. '--recursive --fes-smooth-sigma 1.2'.")


    # SIRAH CG continuation stage from pre-explicit survivor PDBs.
    p.add_argument("--sirah-cg", action="store_true",
                   help="After final implicit survivor selection, convert survivor PDBs to SIRAH CG, solvate, ionize, minimize, equilibrate, and write PSF. This SIRAH variant disables atomistic explicit minimization.")
    p.add_argument("--sirah-input-dir", type=Path, default=None,
                   help="Input PDB directory or single PDB for SIRAH conversion. Default: <out>/final_implicit_survivor_seeds")
    p.add_argument("--sirah-out", type=Path, default=None,
                   help="Output root for per-seed SIRAH/GROMACS runs. Default: <out>/sirah_cg_equilibrated_seeds")
    p.add_argument("--sirah-pattern", default="*.pdb")
    p.add_argument("--sirah-max-seeds", type=int, default=0,
                   help="Only process the first N input PDBs. 0 means all.")
    p.add_argument("--sirah-clean", action=argparse.BooleanOptionalAction, default=True,
                   help="Remove existing per-seed SIRAH work directories before rerun. Default: enabled.")
    p.add_argument("--sirah-ff-dir", type=Path, default=Path("sirah_x2.2_20-07.ff"),
                   help="Path to SIRAH force-field directory containing wt416.gro and tools/CGCONV/cgconv.pl.")
    p.add_argument("--sirah-mdp-dir", type=Path, default=None,
                   help="Optional override directory containing em1.mdp, em2.mdp, eq1.mdp, eq2.mdp. If omitted, embedded SIRAH MDP templates are written automatically.")
    p.add_argument("--sirah-pdb2pqr", default="pdb2pqr")
    p.add_argument("--sirah-gmx", default="gmx_mpi",
                   help="GROMACS executable, e.g. gmx_mpi or gmx.")
    p.add_argument("--sirah-ph", type=float, default=7.0)
    p.add_argument("--sirah-salt", type=float, default=0.15,
                   help="Salt concentration passed to genion -conc.")
    p.add_argument("--sirah-ntomp", type=int, default=1)
    p.add_argument("--sirah-device", default="auto",
                   help="For eq mdrun: pass to -pme and -nb. Use auto to omit both flags; otherwise cpu/gpu.")
    p.add_argument("--sirah-box-distance", type=float, default=1.5)
    p.add_argument("--sirah-maxwarn", type=int, default=5)
    p.add_argument("--sirah-use-gmxapi", action=argparse.BooleanOptionalAction, default=True,
                   help="Use gmxapi.commandline_operation for GROMACS command wrappers when available. Falls back to subprocess. Default: enabled.")
    p.add_argument("--sirah-pdb2gmx-style", choices=["menu", "flags"], default="menu",
                   help="How to run pdb2gmx. 'menu' exactly reproduces the classic interactive SIRAH call: gmx_mpi pdb2gmx -f seed_cg.pdb -o cg.gro with stdin menu choices. 'flags' uses -ff/-water. Default: menu.")
    p.add_argument("--sirah-pdb2gmx-ff", default=None,
                   help="Short force-field name passed to gmx pdb2gmx -ff when --sirah-pdb2gmx-style flags is used. Default: basename of --sirah-ff-dir without .ff, e.g. sirah_x2.2_20-07.")
    p.add_argument("--sirah-pdb2gmx-water", default="none",
                   help="Water model passed to gmx pdb2gmx -water when --sirah-pdb2gmx-style flags is used. Default: none, because WT4 solvation is performed explicitly later.")
    p.add_argument("--sirah-pdb2gmx-stdin", default="1\n1\n",
                   help="Raw stdin menu choices for pdb2gmx when --sirah-pdb2gmx-style menu is used. Default exactly matches the locally tested heredoc: 1 then 1.")
    p.add_argument("--sirah-pdb2gmx-runner", choices=["subprocess", "gmxapi"], default="subprocess",
                   help="Runner for pdb2gmx. Default subprocess because menu-driven pdb2gmx is interactive and must write topol.top in the seed directory.")
    p.add_argument("--sirah-reuse-topology", action=argparse.BooleanOptionalAction, default=True,
                   help="Run pdb2gmx/genrestr once for a reference seed, cache the peptide topology/restraint files, and reuse them for later seeds. Later seeds still get their own box, solvation, close-water removal, genion, grompp, minimization, and equilibration. Default: enabled.")
    p.add_argument("--sirah-reuse-strict-signature", action=argparse.BooleanOptionalAction, default=True,
                   help="Require the CG protein bead signature from later seeds to match the cached reference topology before reuse. Default: enabled.")
    p.add_argument("--sirah-copy-ff-into-seed", action=argparse.BooleanOptionalAction, default=True,
                   help="Copy the SIRAH .ff directory into each per-seed working directory before any GROMACS command, then use that local copy for cgconv, wt416.gro, pdb2gmx discovery, and g_top2psf. Default: enabled.")
    p.add_argument("--sirah-seed-name-format", default="seed{index:03d}",
                   help="Compact per-seed directory/name format. Available fields: index, input_stem. Default: seed001, seed002, ...")
    p.add_argument("--sirah-prepare-md-tpr", action=argparse.BooleanOptionalAction, default=True,
                   help="After CG equilibration, prepare a production-style TPR from the equilibrated structure. Default: enabled.")
    p.add_argument("--sirah-sample-mdp", type=Path, default=None,
                   help="Optional production/sample MDP override. If omitted, an embedded sample.mdp template is written into each seed directory.")
    p.add_argument("--sirah-clusters-sel-name", default="SIRAH_SEEDS_SEL",
                   help="Name of the root-level file listing successfully prepared compact seed names. Default: SIRAH_SEEDS_SEL.")

    config_defaults, config_unknown = _genpept_config_defaults(pre_args.config, p, strict=bool(getattr(pre_args, "strict_config", False)))
    if config_unknown:
        preview = ", ".join(config_unknown[:20])
        more = "" if len(config_unknown) <= 20 else f" ... and {len(config_unknown) - 20} more"
        raise SystemExit(f"Unknown GENPEPT config key(s): {preview}{more}")
    if config_defaults:
        p.set_defaults(**config_defaults)

    args = p.parse_args(argv_list)
    args.config_compat_messages = list(_CONFIG_COMPAT_MESSAGES)
    args.config_compat_report = dict(_CONFIG_COMPAT_REPORT)
    if args.seq in (None, ""):
        p.error("--seq is required unless provided by --config")
    if args.out in (None, ""):
        p.error("--out is required unless provided by --config conformer_generation.out")
    if not isinstance(args.out, Path):
        args.out = Path(str(args.out))
    return args



def run_integrated_pca_fes(args):
    """Run the integrated PCA/pseudo-FES post-processing stage."""
    if not getattr(args, "pca_fes", False) and not getattr(args, "pca_plot_adaptive_rounds", False):
        return None

    pca_out = Path(args.pca_out) if args.pca_out is not None else Path(args.out) / "pca_pseudo_fes"
    pca_dirs = list(args.pca_pdb_dir) if args.pca_pdb_dir else [Path(args.out)]

    pca_argv = [
        "--pdb-dir", *[str(x) for x in pca_dirs],
        "--out", str(pca_out),
        "--features", str(args.pca_features),
        "--contact-cutoff", str(args.contact_cutoff),
        "--contact-min-sep", str(args.contact_min_sep),
        "--bins", str(args.pca_bins),
        "--temperature", str(args.temperature),
        "--min-count", str(args.pca_min_count),
        "--weight-mode", str(args.pca_weight_mode),
        "--energy-column", str(args.pca_energy_column),
        "--weight-column", str(args.pca_weight_column),
        "--energy-cap-kj", str(args.pca_energy_cap_kj),
        "--color-by", str(args.pca_color_by),
        "--fes-cmap", str(args.pca_fes_cmap),
        "--fes-vmax-percentile", str(args.pca_fes_vmax_percentile),
    ]
    if args.pca_fes_vmax_kj is not None:
        pca_argv.extend(["--fes-vmax-kj", str(args.pca_fes_vmax_kj)])
    if bool(args.pca_auto_weights):
        pca_argv.append("--auto-weights")
    if bool(args.pca_plot_adaptive_rounds):
        pca_argv.append("--plot-adaptive-rounds")
    extra = str(getattr(args, "pca_args", "") or "").strip()
    if extra:
        pca_argv.extend(shlex.split(extra))

    print_stage("PCA pseudo-FES post-processing", phase_key="exploration", cycle_step="map")
    ui_message(f"PCA/FES input: {', '.join(str(x) for x in pca_dirs)}")
    ui_message(f"PCA/FES output: {pca_out}")
    render_dashboard()
    pca_pseudo_fes_main(pca_argv)
    return pca_out



# -----------------------------
# SIRAH coarse-grain continuation stage
# -----------------------------


# Embedded SIRAH/GROMACS MDP templates. These are written into each per-seed
# working directory unless --sirah-mdp-dir is provided as an explicit override.
# This keeps the SIRAH continuation stage self-contained: the user only needs
# the SIRAH force-field directory beside the script plus normal executables.
EMBEDDED_SIRAH_MDP_FILES = {
    'em1.mdp': '; VARIOUS PREPROCESSING OPTIONS\n  title                  = EMIN\n  cpp                    = /lib/cpp\n  include                =\n  define                 = -DGN_GO\n\n; RUN CONTROL PARAMETERS\n  integrator             = steep\n  nsteps                 = 50000\n\n; LANGEVIN DYNAMICS OPTIONS\n; None (see manual)\n\n; ENERGY MINIMIZATION OPTIONS\n  emtol                  = 1.0\n  emstep                 = 0.04\n; nstcgsteep             = 500\n  nbfgscorr              = 10\n\n; OUTPUT CONTROL OPTIONS\n  nstxout                = 0\n  nstvout                = 0\n  nstfout                = 0\n  nstlog                 = 0\n  nstenergy              = 50  \n  nstxtcout              = 0\n  xtc-precision          = 1000   \n  xtc-grps               =    \n  energygrps             = \n\n; NEIGHBORSEARCHING PARAMETERS\n  nstlist                = 10\n  ns_type                = grid\n  pbc                    = xyz\n  rlist                  = 1.4\n\n; OPTIONS FOR ELECTROSTATICS AND VDW\n  coulombtype            = PME\n  rcoulomb               = 1.2\n  vdwtype                = Cut-off\n  rvdw                   = 1.2\n\n  optimize_fft           = yes\n  fourierspacing         = 0.2\n\n; GPU\n  cutoff-scheme          = Verlet\n  verlet-buffer-drift    =-1\n\n; GENERALIZED BORN ELECTROSTATICS\n; None (see manual)\n\n; OPTIONS FOR WEAK COUPLING ALGORITHMS\n; None (see manual)\n\n; SIMULATED ANNEALING\n; None (see manual)\n\n; GENERATE VELOCITIES FOR STARTUP RUN\n; None (see manual)\n\n;---- OTHER OPTIONS ------------------\n\n; OPTIONS FOR BONDS\n; None (see manual)\n\n; ENERGY GROUP EXCLUSIONS\n; None (see manual)\n\n; NMR refinement stuff \n; None (see manual)\n\n;-------------------------------------\n',
    'em2.mdp': '; VARIOUS PREPROCESSING OPTIONS\n  title                  = EMIN\n  cpp                    = /lib/cpp\n  include                =\n  define                 = \n\n; RUN CONTROL PARAMETERS\n  integrator             = steep\n  nsteps                 = 10000\n\n; LANGEVIN DYNAMICS OPTIONS\n; None (see manual)\n\n; ENERGY MINIMIZATION OPTIONS\n  emtol                  = 1.0\n  emstep                 = 0.05\n; nstcgsteep             = 500\n  nbfgscorr              = 10\n\n; OUTPUT CONTROL OPTIONS\n  nstxout                = 0\n  nstvout                = 0\n  nstfout                = 0\n  nstlog                 = 0\n  nstenergy              = 50  \n  nstxtcout              = 0\n  xtc-precision          = 1000   \n  xtc-grps               =    \n  energygrps             = \n\n; NEIGHBORSEARCHING PARAMETERS\n  nstlist                = 10\n  ns_type                = grid\n  pbc                    = xyz\n  rlist                  = 1.4\n\n; OPTIONS FOR ELECTROSTATICS AND VDW\n  coulombtype            = PME\n  rcoulomb               = 1.2\n  vdwtype                = Cut-off\n  rvdw                   = 1.2\n\n  optimize_fft           = yes\n  fourierspacing         = 0.2\n\n; GPU\n  cutoff-scheme          = Verlet\n  verlet-buffer-drift    =-1\n\n; GENERALIZED BORN ELECTROSTATICS\n; None (see manual)\n\n; OPTIONS FOR WEAK COUPLING ALGORITHMS\n; None (see manual)\n\n; SIMULATED ANNEALING\n; None (see manual)\n\n; GENERATE VELOCITIES FOR STARTUP RUN\n; None (see manual)\n\n;---- OTHER OPTIONS ------------------\n\n; OPTIONS FOR BONDS\n; None (see manual)\n\n; ENERGY GROUP EXCLUSIONS\n; None (see manual)\n\n; NMR refinement stuff \n; None (see manual)\n\n;-------------------------------------\n',
    'eq1.mdp': '; INPUT FILE \n\n; GENERAL INPUT FLAGS\n  title                  = 5ns equilibration NVT\n  cpp                    = /usr/bin/cpp\n  define                 = -DPOSRES\n\n; RUN FLAGS\n  integrator             = md\n  dt                     = 0.005\n  nsteps                 = 100000\n  pbc                    = xyz\n\n; NON-BONDED INTERACTIONS FLAGS\n  ns_type                = grid\n  nstlist                = 10\n  rlist                  = 1.4\n  coulombtype            = PME\n  rcoulomb               = 1.2\n  vdwtype                = Cut-off\n  rvdw                   = 1.2\n\n  optimize_fft           = yes\n  fourierspacing         = 0.2\n\n; GPU\n  cutoff-scheme          = Verlet\n  verlet-buffer-drift    =-1\n\n; TEMPERATURE COUPLING FLAGS\n  tcoupl                 = V-rescale\n  tc-grps                = System\n  tau_t                  = 1.0\n  ref_t                  = 300 \n\n; INITIAL STEP VELOCITIES FLAGS\n  gen_vel                = no\n  gen_temp               = 300\n  gen_seed               = -1\n\n; PRESSURE COUPLING FLAGS\n  Pcoupl                 = no\n\n; CONSTRAINTS\n; None (see manual)\n\n; DISTANCE RESTRAINTS\n; None (see manual)\n\n; OUTPUT FLAGS\n  nstxout                = 0\n  nstvout                = 0\n  nstlog                 = 5000\n  nstenergy              = 0\n  nstxtcout              = 0\n',
    'eq2.mdp': '; INPUT FILE \n\n; GENERAL INPUT FLAGS\n  title                  = equilibration NVT\n  cpp                    = /usr/bin/cpp\n  define                 = -DGN_GO_SOFT\n\n; RUN FLAGS\n  integrator             = md\n  dt                     = 0.020\n  nsteps                 = 200000\n  pbc                    = xyz\n\n; NON-BONDED INTERACTIONS FLAGS\n  ns_type                = grid\n  nstlist                = 10\n  rlist                  = 1.4\n  coulombtype            = PME\n  rcoulomb               = 1.2\n  vdwtype                = Cut-off\n  rvdw                   = 1.2\n\n  optimize_fft           = yes\n  fourierspacing         = 0.2\n\n; GPU\n  cutoff-scheme          = Verlet\n  verlet-buffer-drift    =-1\n\n; TEMPERATURE COUPLING FLAGS\n  tcoupl                 = V-rescale\n  tc-grps                = System\n  tau_t                  = 1.0\n  ref_t                  = 300 \n\n; INITIAL STEP VELOCITIES FLAGS\n  gen_vel                = no\n  gen_temp               = 300\n  gen_seed               = -1\n\n; PRESSURE COUPLING FLAGS\n; PRESSURE COUPLING FLAGS\n  Pcoupl                 = Berendsen\n  pcoupltype             = isotropic\n  tau_p                  = 8.0\n  compressibility        = 4.5e-5\n  ref_p                  = 1.0\n  refcoord-scaling       = com\n\n; CONSTRAINTS\n; None (see manual)\n\n; DISTANCE RESTRAINTS\n; None (see manual)\n\n; OUTPUT FLAGS\n  nstxout                = 0\n  nstvout                = 0\n  nstlog                 = 5000\n  nstenergy              = 0\n  nstxtcout              = 0\n',
}


EMBEDDED_SIRAH_SAMPLE_MDP = """; INPUT FILE 

; GENERAL INPUT FLAGS
  cpp                    = /usr/bin/cpp
  define                 = 
  compressed-x-grps    = System_&_!WT4_&_!NAW_&_!CLW
; RUN FLAGS
  integrator             = md
  dt                     = 0.020
  nsteps                 = 25000000
  pbc                    = xyz

; NON-BONDED INTERACTIONS FLAGS
  ns_type                = grid
  nstlist                = 50
  rlist                  = 1.4
  coulombtype            = PME
  rcoulomb               = 1.2
  vdwtype                = Cut-off
  rvdw                   = 1.2

  optimize_fft           = yes
  fourierspacing         = 0.2

; GPU
  cutoff-scheme          = Verlet
  verlet-buffer-drift    =-1

; TEMPERATURE COUPLING FLAGS
  tcoupl                 = V-rescale
  tc-grps                = System
  tau_t                  = 2.0
  ref_t                  = 300

; INITIAL STEP VELOCITIES FLAGS
  gen_vel                = no
  gen_temp               = 300
  gen_seed               = -1

; PRESSURE COUPLING FLAGS
  Pcoupl                 = Parrinello-Rahman
  pcoupltype             = isotropic
  tau_p                  = 8.0
  compressibility        = 4.5e-5
  ref_p                  = 1.0
  refcoord-scaling       = com

; CONSTRAINTS
; None (see manual)

; DISTANCE RESTRAINTS
; None (see manual)

; OUTPUT FLAGS
  nstxout                = 0
  nstvout                = 0
  nstlog                 = 10000
  nstenergy              = 5000
  nstxtcout              = 5000
"""

@dataclass
class SirahResult:
    seed_index: int
    cluster_id: int
    seed_name: str
    source_seed_name: str
    work_dir: str
    input_pdb: str
    pqr_path: str
    cg_pdb_path: str
    final_gro_path: str
    final_psf_path: str
    topology_path: str
    production_gro_path: str
    production_ndx_path: str
    production_tpr_path: str
    success: bool
    error: str


def _shell_quote_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def write_csv_dataclass(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def count_wt4_from_gro(gro_path: Path) -> int:
    """Count WT4 waters from the WP1 sites in a SIRAH WT4 .gro file."""
    with Path(gro_path).open() as handle:
        return sum(1 for line in handle if "WP1" in line)


def update_topol_wt4_count(topol_path: Path, wt4_count: int):
    path = Path(topol_path)
    lines = path.read_text().splitlines()
    out, replaced = [], False
    for line in lines:
        if line.strip().startswith("WT4"):
            out.append(f"WT4 {int(wt4_count)}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"WT4 {int(wt4_count)}")
    path.write_text("\n".join(out) + "\n")


def inject_sirah_backbone_restraints(topol_path: Path):
    """Replace pdb2gmx POSRES include with SIRAH GN/GO restraint includes.

    This intentionally mirrors the classic SIRAH shell workflow::

        sed 's/#include "posre.itp"\n#endif/#endif\n#ifdef GN_GO .../' topol.top

    Do *not* keep ``#include "posre.itp"``. Some SIRAH pdb2gmx/menu
    setups leave a POSRES block in topol.top without writing a usable
    posre.itp. Keeping that include makes later grompp fail when an MDP defines
    -DPOSRES. The SIRAH workflow uses bkbres.itp / bkbres_soft.itp instead.
    """
    path = Path(topol_path)
    text = path.read_text()
    block = (
        '#ifdef GN_GO\n#include "bkbres.itp"\n#endif\n'
        '#ifdef GN_GO_SOFT\n#include "bkbres_soft.itp"\n#endif\n'
    )

    # Idempotent cleanup: if a previous version appended the GN/GO block but
    # left posre.itp in place, remove only the posre include line.
    lines = text.splitlines()
    lines = [line for line in lines if line.strip() != '#include "posre.itp"']
    text = "\n".join(lines) + "\n"

    if '#include "bkbres.itp"' in text and '#include "bkbres_soft.itp"' in text:
        path.write_text(text)
        return

    # Prefer to insert immediately after the closing #endif of the POSRES block,
    # which is what the original sed effectively does after removing posre.itp.
    marker = '#ifdef POSRES'
    pos = text.find(marker)
    if pos >= 0:
        endif_pos = text.find('#endif', pos)
        if endif_pos >= 0:
            insert_at = endif_pos + len('#endif')
            text = text[:insert_at] + "\n" + block.rstrip("\n") + text[insert_at:]
        else:
            text = text.rstrip() + "\n" + block
    else:
        text = text.rstrip() + "\n" + block

    path.write_text(text)


def sirah_seed_name(pdb_path: Path) -> str:
    stem = Path(pdb_path).stem
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return safe or "seed"


def copy_mdp_files_for_sirah(args, work_dir: Path):
    """Write embedded MDP files, unless an override directory is supplied."""
    work_dir = Path(work_dir)
    required = ("em1.mdp", "em2.mdp", "eq1.mdp", "eq2.mdp")
    if args.sirah_mdp_dir is not None:
        mdp_dir = Path(args.sirah_mdp_dir)
        missing = []
        for name in required:
            src = mdp_dir / name
            if src.exists():
                shutil.copy2(src, work_dir / name)
            else:
                missing.append(str(src))
        if missing:
            raise FileNotFoundError(
                "Missing SIRAH MDP override files: " + ", ".join(missing) +
                ". Omit --sirah-mdp-dir to use the embedded templates."
            )
        return
    for name in required:
        (work_dir / name).write_text(EMBEDDED_SIRAH_MDP_FILES[name].rstrip() + "\n")


def copy_sample_mdp_for_sirah(args, work_dir: Path):
    """Write sample.mdp for final production-TPR preparation."""
    work_dir = Path(work_dir)
    if args.sirah_sample_mdp is not None:
        shutil.copy2(Path(args.sirah_sample_mdp), work_dir / "sample.mdp")
    else:
        (work_dir / "sample.mdp").write_text(EMBEDDED_SIRAH_SAMPLE_MDP.rstrip() + "\n")



def format_sirah_seed_name(args, pdb_path: Path, index: int) -> str:
    fmt = str(getattr(args, "sirah_seed_name_format", "seed{index:03d}") or "seed{index:03d}")
    input_stem = sirah_seed_name(pdb_path)
    try:
        name = fmt.format(index=int(index), input_stem=input_stem)
    except Exception:
        name = f"seed{int(index):03d}"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))
    return safe or f"seed{int(index):03d}"


SIRAH_SOLVENT_RESNAMES = {"WT4", "WAT", "SOL", "HOH"}
SIRAH_ION_RESNAMES = {"NaW", "ClW", "NAW", "CLW", "NA", "CL", "K", "KW"}
SIRAH_NONPROTEIN_RESNAMES = SIRAH_SOLVENT_RESNAMES | SIRAH_ION_RESNAMES


def _gro_atom_fields(line: str):
    """Parse one fixed-width .gro atom line. Coordinates are in nm."""
    if len(line) < 44:
        raise ValueError(f"Malformed .gro atom line: {line!r}")
    return {
        "resnr": line[0:5],
        "resname": line[5:10].strip(),
        "atomname": line[10:15].strip(),
        "atomnr": line[15:20],
        "x": float(line[20:28]),
        "y": float(line[28:36]),
        "z": float(line[36:44]),
        "tail": line[44:],
        "raw": line,
    }


def _read_gro(path: Path):
    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Not a valid .gro file: {path}")
    try:
        n_atoms = int(lines[1].strip())
    except Exception as exc:
        raise ValueError(f"Could not parse atom count in .gro file: {path}") from exc
    atom_lines = lines[2:2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"Atom count mismatch in .gro file: {path}")
    box_line = lines[2 + n_atoms] if len(lines) > 2 + n_atoms else ""
    return lines[0], n_atoms, atom_lines, box_line


def _is_sirah_nonprotein_gro_line(line: str) -> bool:
    return _gro_atom_fields(line)["resname"] in SIRAH_NONPROTEIN_RESNAMES


def protein_atom_signature_from_gro(gro_path: Path):
    """Return a topology-order signature for non-solvent/non-ion beads."""
    _title, _n, atom_lines, _box = _read_gro(gro_path)
    sig = []
    for line in atom_lines:
        if _is_sirah_nonprotein_gro_line(line):
            continue
        f = _gro_atom_fields(line)
        sig.append((f["resname"], f["atomname"]))
    return sig


class SirahTopologyCache:
    """Reusable SIRAH peptide topology generated from the first compatible seed.

    Only the peptide topology/restraint part is cached. Each seed still performs
    its own editconf box creation, WT4 solvation, close-water removal, genion,
    and all grompp/mdrun stages, so solvent and ion counts remain seed-specific.
    """
    BASE_TOP = "base_pdb2gmx_topol.top"
    SIGNATURE = "protein_signature.json"
    RESTRAINTS = ("bkbres.itp", "bkbres_soft.itp")

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)

    @property
    def has_base_topology(self) -> bool:
        return (self.cache_dir / self.BASE_TOP).exists() and (self.cache_dir / self.SIGNATURE).exists()

    @property
    def ready(self) -> bool:
        return self.has_base_topology and all((self.cache_dir / name).exists() for name in self.RESTRAINTS)

    def capture_base_topology(self, topol_top: Path, cg_gro: Path):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(topol_top, self.cache_dir / self.BASE_TOP)
        sig = protein_atom_signature_from_gro(cg_gro)
        (self.cache_dir / self.SIGNATURE).write_text(json.dumps(sig, indent=2))

    def capture_restraints(self, work_dir: Path):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for name in self.RESTRAINTS:
            src = Path(work_dir) / name
            if not src.exists():
                raise FileNotFoundError(f"Cannot cache SIRAH restraint file; missing {src}")
            shutil.copy2(src, self.cache_dir / name)

    def materialize_base_topology(self, work_dir: Path):
        if not self.has_base_topology:
            raise RuntimeError(f"SIRAH topology cache has no base topology: {self.cache_dir}")
        shutil.copy2(self.cache_dir / self.BASE_TOP, Path(work_dir) / "topol.top")

    def materialize_restraints(self, work_dir: Path):
        if not self.ready:
            raise RuntimeError(f"SIRAH topology cache is not ready: {self.cache_dir}")
        for name in self.RESTRAINTS:
            shutil.copy2(self.cache_dir / name, Path(work_dir) / name)

    def validate_signature(self, cg_gro: Path, strict: bool = True):
        if not strict:
            return
        sig_path = self.cache_dir / self.SIGNATURE
        if not sig_path.exists():
            raise RuntimeError(f"SIRAH topology cache has no signature file: {sig_path}")
        ref = [tuple(x) for x in json.loads(sig_path.read_text())]
        cur = protein_atom_signature_from_gro(cg_gro)
        if ref != cur:
            first = next((i for i, (a, b) in enumerate(zip(ref, cur)) if a != b), None)
            if first is None and len(ref) != len(cur):
                first = min(len(ref), len(cur))
            ref_item = ref[first] if first is not None and first < len(ref) else None
            cur_item = cur[first] if first is not None and first < len(cur) else None
            raise ValueError(
                "Cannot reuse SIRAH topology: CG bead signature differs from the reference. "
                f"reference_beads={len(ref)}, current_beads={len(cur)}, first_mismatch={first}, "
                f"reference={ref_item}, current={cur_item}."
            )

class SirahCGRunner:
    """One-seed SIRAH conversion/equilibration runner.

    gmxapi is used for the fragile GROMACS command-line calls that benefit
    from explicit file maps. In particular, pdb2gmx is run through
    gmxapi.commandline_operation with absolute input/output paths and GMXLIB
    pointed at the per-seed work directory, so it sees the copied local SIRAH
    force-field directory without depending on shell cwd or fragile stdin.
    """
    GMXAPI_STEPS = {"grompp", "mdrun"}

    def __init__(self, args, pdb_path: Path, out_root: Path, topology_cache: Optional[SirahTopologyCache] = None, use_cached_topology: bool = False, seed_index: int = 0, seed_name: Optional[str] = None):
        self.args = args
        self.topology_cache = topology_cache
        self.use_cached_topology = bool(use_cached_topology)
        self.input_pdb = Path(pdb_path).resolve()
        self.seed_index = int(seed_index or 0)
        self.cluster_id = self.seed_index if self.seed_index > 0 else 1
        self.source_seed_name = sirah_seed_name(self.input_pdb)
        self.seed = str(seed_name or self.source_seed_name)
        self.work = Path(out_root) / self.seed
        self.source_sirah_ff = Path(args.sirah_ff_dir).resolve()
        self.sirah_ff = self.source_sirah_ff
        self.sirah_ff_actual = self.source_sirah_ff
        self.local_sirah_ff = self.work / self.source_sirah_ff.name
        self.cgconv = self.sirah_ff / "tools" / "CGCONV" / "cgconv.pl"
        self.wt416 = self.sirah_ff / "wt416.gro"
        self.g_top2psf = self.sirah_ff / "tools" / "g_top2psf.pl"
        self.gmx = str(args.sirah_gmx)
        self.pqr = self.work / f"{self.seed}.pqr"
        self.cg_pdb = self.work / f"{self.seed}_cg.pdb"
        self.log_path = self.work / "sirah_cg_workflow.log"
        self._gmxapi = None
        self._gmxapi_failed = False

    def result(self, success: bool, error: str = "") -> SirahResult:
        cid = int(self.cluster_id)
        return SirahResult(
            seed_index=int(self.seed_index),
            cluster_id=cid,
            seed_name=self.seed,
            source_seed_name=self.source_seed_name,
            work_dir=str(self.work),
            input_pdb=str(self.input_pdb),
            pqr_path=str(self.pqr),
            cg_pdb_path=str(self.cg_pdb),
            final_gro_path=str(self.work / "cg_eq2.gro"),
            final_psf_path=str(self.work / "cg_eq2.psf"),
            topology_path=str(self.work / "topol.top"),
            production_gro_path=str(self.work / "sample.gro"),
            production_ndx_path=str(self.work / "prot.ndx"),
            production_tpr_path=str(self.work / "sample.tpr"),
            success=success,
            error=error,
        )

    def _set_sirah_paths(self, ff_dir: Path, prefer_local_command_paths: bool = False):
        """Set SIRAH paths used in commands.

        When the force field has been copied into the per-seed working
        directory, command arguments must be relative to that directory, e.g.
        ./sirah_x2.2_20-07.ff/tools/CGCONV/cgconv.pl. Passing a path like
        VADIV.../seed/sirah_x2.2_20-07.ff while also running with cwd=seed
        makes subprocess look below seed/VADIV..., which is wrong.
        """
        actual = Path(ff_dir)
        self.sirah_ff_actual = actual
        cmd_base = actual
        if prefer_local_command_paths:
            try:
                work_resolved = self.work.resolve()
                actual_resolved = actual.resolve()
                actual_resolved.relative_to(work_resolved)
                cmd_base = Path(".") / actual.name
            except Exception:
                cmd_base = actual
        if prefer_local_command_paths and cmd_base != actual:
            # Keep the command log and subprocess args explicitly local.
            # pathlib would render Path("./sirah.ff") as "sirah.ff", so use
            # strings here to preserve the leading ./ requested for clarity.
            ff_cmd = f"./{actual.name}"
            self.sirah_ff = ff_cmd
            self.cgconv = f"{ff_cmd}/tools/CGCONV/cgconv.pl"
            self.wt416 = f"{ff_cmd}/wt416.gro"
            self.g_top2psf = f"{ff_cmd}/tools/g_top2psf.pl"
        else:
            self.sirah_ff = cmd_base
            self.cgconv = self.sirah_ff / "tools" / "CGCONV" / "cgconv.pl"
            self.wt416 = self.sirah_ff / "wt416.gro"
            self.g_top2psf = self.sirah_ff / "tools" / "g_top2psf.pl"

    def _validate_sirah_ff(self, ff_dir: Path):
        ff_dir = Path(ff_dir)
        required = (
            ff_dir,
            ff_dir / "tools" / "CGCONV" / "cgconv.pl",
            ff_dir / "wt416.gro",
            ff_dir / "tools" / "g_top2psf.pl",
        )
        for path in required:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required SIRAH path not found: {path}")

    def copy_sirah_ff_into_seed_workdir(self):
        self._validate_sirah_ff(self.source_sirah_ff)
        if not bool(getattr(self.args, "sirah_copy_ff_into_seed", True)):
            self._set_sirah_paths(self.source_sirah_ff)
            return

        target = self.local_sirah_ff
        try:
            source_resolved = self.source_sirah_ff.resolve()
            target_resolved = target.resolve() if target.exists() else target.absolute()
            if source_resolved == target_resolved:
                self._set_sirah_paths(target, prefer_local_command_paths=True)
                return
        except Exception:
            pass

        if target.exists():
            if not target.is_dir():
                raise FileExistsError(f"SIRAH local force-field target exists but is not a directory: {target}")
        else:
            shutil.copytree(self.source_sirah_ff, target)
        self._validate_sirah_ff(target)
        self._set_sirah_paths(target, prefer_local_command_paths=True)

    def prepare(self):
        if self.args.sirah_clean and self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.input_pdb, self.work / f"{self.seed}.pdb")
        copy_mdp_files_for_sirah(self.args, self.work)
        copy_sample_mdp_for_sirah(self.args, self.work)
        (self.work / "seed_info.json").write_text(json.dumps({
            "seed_index": int(self.seed_index),
            "cluster_id": int(self.cluster_id),
            "seed_name": self.seed,
            "source_seed_name": self.source_seed_name,
            "input_pdb": str(self.input_pdb),
        }, indent=2) + "\n")
        self.copy_sirah_ff_into_seed_workdir()

    def log(self, msg: str):
        self._log.write(msg)
        if not msg.endswith("\n"):
            self._log.write("\n")
        self._log.flush()

    def _load_gmxapi(self):
        if not getattr(self.args, "sirah_use_gmxapi", True) or self._gmxapi_failed:
            return None
        if self._gmxapi is None:
            try:
                import gmxapi as gmxapi_module
                self._gmxapi = gmxapi_module
                self.log("[gmxapi] available; using it for grompp/mdrun where possible")
            except Exception as exc:
                self._gmxapi_failed = True
                self.log(f"[gmxapi] unavailable; falling back to subprocess: {exc}")
                return None
        return self._gmxapi

    def run(self, cmd, input_text: Optional[str] = None, allow_gmxapi: bool = True):
        cmd = [str(x) for x in cmd]
        self.log("\n$ " + _shell_quote_join(cmd))
        if input_text:
            self.log("[stdin]\n" + input_text.rstrip("\n"))
        if allow_gmxapi and not input_text and cmd[0] == self.gmx and len(cmd) > 1 and cmd[1] in self.GMXAPI_STEPS:
            try:
                return self.run_gmxapi(cmd[1:])
            except Exception as exc:
                self._gmxapi_failed = True
                self.log(f"[gmxapi] step failed; retrying with subprocess: {type(exc).__name__}: {exc}")
        return self.run_subprocess(cmd, input_text=input_text)

    def _subprocess_env(self, cmd=None):
        env = os.environ.copy()
        if cmd and len(cmd) > 0 and str(cmd[0]) == str(self.gmx):
            # The script globally sets OMP_NUM_THREADS=1 to protect Python/OpenMM
            # worker pools from oversubscription. GROMACS, however, errors if
            # OMP_NUM_THREADS disagrees with -ntomp. For every GROMACS call made
            # by the SIRAH stage, keep the environment and command line aligned.
            ntomp = max(1, int(getattr(self.args, "sirah_ntomp", 1) or 1))
            env["OMP_NUM_THREADS"] = str(ntomp)
        return env

    def run_subprocess(self, cmd, input_text: Optional[str] = None):
        proc = subprocess.run(
            cmd,
            cwd=str(self.work),
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self._subprocess_env(cmd),
        )
        self.log(proc.stdout or "")
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {proc.returncode}: {_shell_quote_join(cmd)}")
        return proc.stdout or ""

    def _gmxapi_env(self, gmxapi_module):
        """Environment for gmxapi commandline_operation.

        gmxapi runs the command in a managed work directory, not in self.work.
        Therefore pdb2gmx will not automatically see the copied local SIRAH
        force-field directory. Setting GMXLIB to the seed work directory lets
        pdb2gmx find ./sirah_x2.2_20-07.ff by its short -ff name.
        """
        try:
            env = dict(gmxapi_module.runtime.filtered_mpi_environ())
        except Exception:
            mpi_prefixes = ("DCMF_", "MPICH_", "MPIEXEC_", "MPIO_", "MV2_", "MVAPICH_", "HYDRA_", "OMPI_", "PMI_", "PMIX_", "I_MPI_")
            env = {k: v for k, v in os.environ.items() if not any(k.startswith(prefix) for prefix in mpi_prefixes)}
        # Keep the practical loader/search path variables if filtered_mpi_environ
        # is minimal in this gmxapi build.
        for key in ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PYTHONPATH"):
            if key in os.environ and key not in env:
                env[key] = os.environ[key]
        env["GMXLIB"] = str(self.work.resolve())
        ntomp = max(1, int(getattr(self.args, "sirah_ntomp", 1) or 1))
        env["OMP_NUM_THREADS"] = str(ntomp)
        return env

    def _abs_work_path(self, value):
        path = Path(str(value))
        if path.is_absolute():
            return str(path)
        return str((self.work / path).resolve())

    def _log_mapped_gmxapi_command(self, subcommand, arguments, input_files, output_files, stdin):
        display = [self.gmx, subcommand] + [str(x) for x in arguments]
        for flag, path in (input_files or {}).items():
            display.extend([str(flag), str(path)])
        for flag, path in (output_files or {}).items():
            display.extend([str(flag), str(path)])
        self.log("\n$ " + _shell_quote_join(display) + "  # via gmxapi.commandline_operation")
        if stdin:
            self.log("[stdin:gmxapi]\n" + str(stdin).rstrip("\n"))

    def run_gmxapi_mapped(self, subcommand, arguments=(), input_files=None, output_files=None, stdin: Optional[str] = None):
        """Run a GROMACS subcommand through gmxapi with explicit file maps.

        Unlike the older wrapper, this does not rely on cwd. gmxapi explicitly
        documents that commandline_operation runs in a managed directory, so all
        seed files are passed as absolute input/output mappings.
        """
        gmxapi = self._load_gmxapi()
        if gmxapi is None:
            raise RuntimeError("gmxapi not active")
        input_files = {str(k): self._abs_work_path(v) for k, v in (input_files or {}).items()}
        output_files = {str(k): self._abs_work_path(v) for k, v in (output_files or {}).items()}
        args = [str(x) for x in arguments]
        self._log_mapped_gmxapi_command(subcommand, args, input_files, output_files, stdin)
        op = gmxapi.commandline_operation(
            executable=self.gmx,
            arguments=[str(subcommand)] + args,
            input_files=input_files or None,
            output_files=output_files or None,
            stdin=stdin,
            env=self._gmxapi_env(gmxapi),
        )
        op.run()
        out = _future_text(getattr(op.output, "stdout", None))
        err = _future_text(getattr(op.output, "stderr", None)) or _future_text(getattr(op.output, "erroroutput", None))
        rc = _future_value(getattr(op.output, "returncode", None))
        if out:
            self.log(out)
        if err:
            self.log(err)
        if rc not in (None, 0):
            raise RuntimeError(f"gmxapi command returned {rc}: {self.gmx} {subcommand} {' '.join(args)}")
        missing = [path for path in output_files.values() if not Path(path).exists()]
        if missing:
            raise RuntimeError("gmxapi command finished but expected output file(s) are missing: " + ", ".join(missing))
        return out or ""

    def pdb2gmx_forcefield_name(self) -> str:
        name = str(getattr(self.args, "sirah_pdb2gmx_ff", "") or "").strip()
        if name:
            return name
        ff_name = Path(self.source_sirah_ff).name
        return ff_name[:-3] if ff_name.endswith(".ff") else ff_name

    def run_pdb2gmx(self):
        """Generate cg.gro/topol.top from the SIRAH CG PDB.

        Default behavior intentionally reproduces the user's locally working
        command exactly::

            gmx_mpi pdb2gmx -f <seed>_cg.pdb -o cg.gro

        with stdin menu choices ``1\n1\n``. This is more reliable for SIRAH
        installations whose force-field/water menus do not behave correctly
        with ``-ff``/``-water``. A flags-based mode remains available for
        installations where it has been validated.
        """
        style = str(getattr(self.args, "sirah_pdb2gmx_style", "menu") or "menu").lower()
        stdin_text = str(getattr(self.args, "sirah_pdb2gmx_stdin", "1\n1\n") or "")
        if style == "menu" and not stdin_text:
            stdin_text = "1\n1\n"
        if stdin_text and not stdin_text.endswith("\n"):
            stdin_text += "\n"

        if style == "flags":
            ff_name = self.pdb2gmx_forcefield_name()
            water = str(getattr(self.args, "sirah_pdb2gmx_water", "none") or "none")
            cmd = [
                self.gmx, "pdb2gmx",
                "-ff", ff_name,
                "-water", water,
                "-f", self.cg_pdb.name,
                "-o", "cg.gro",
                "-p", "topol.top",
                "-i", "posre.itp",
            ]
        else:
            # Exact locally validated SIRAH path. pdb2gmx writes topol.top by
            # default, and avoiding -ff/-water lets the SIRAH menu logic choose
            # exactly as in the manual/heredoc workflow.
            cmd = [
                self.gmx, "pdb2gmx",
                "-f", self.cg_pdb.name,
                "-o", "cg.gro",
            ]

        def _verify_outputs(label: str):
            missing = [name for name in ("cg.gro", "topol.top") if not (self.work / name).exists()]
            if missing:
                log_tail = ""
                try:
                    text = self.log_path.read_text(errors="replace")
                    log_tail = "\nLast pdb2gmx log lines:\n" + "\n".join(text.splitlines()[-80:])
                except Exception:
                    pass
                raise RuntimeError(
                    f"{label} finished but did not create expected pdb2gmx output(s): "
                    f"{', '.join(missing)}.{log_tail}"
                )

        runner = str(getattr(self.args, "sirah_pdb2gmx_runner", "subprocess") or "subprocess").lower()

        # gmxapi is only attempted for the flags-based route. Menu-driven
        # pdb2gmx is intentionally run via subprocess so stdin is consumed just
        # like the working shell heredoc.
        if style == "flags" and runner == "gmxapi" and getattr(self.args, "sirah_use_gmxapi", True):
            ff_name = self.pdb2gmx_forcefield_name()
            water = str(getattr(self.args, "sirah_pdb2gmx_water", "none") or "none")
            args = ["-ff", ff_name, "-water", water]
            input_files = {"-f": self.cg_pdb.name}
            output_files = {"-o": "cg.gro", "-p": "topol.top", "-i": "posre.itp"}
            try:
                out = self.run_gmxapi_mapped("pdb2gmx", args, input_files=input_files, output_files=output_files, stdin=stdin_text or None)
                _verify_outputs("gmxapi pdb2gmx")
                return out
            except Exception as exc:
                self._gmxapi_failed = True
                self.log(f"[gmxapi:pdb2gmx] failed or produced no topol.top; retrying with direct subprocess: {type(exc).__name__}: {exc}")

        # Reliable default: direct subprocess in the seed working directory with
        # the copied local ./sirah_x2.2_20-07.ff visible to pdb2gmx.
        out = self.run_subprocess(cmd, input_text=stdin_text or None)
        _verify_outputs("subprocess pdb2gmx")
        return out

    def run_gmxapi(self, gmx_args):
        gmxapi = self._load_gmxapi()
        if gmxapi is None:
            raise RuntimeError("gmxapi not active")
        # commandline_operation runs in the current process working directory, so
        # temporarily chdir to the per-seed work directory. This keeps relative
        # paths identical to the subprocess path and avoids fragile input/output
        # file maps across gmxapi versions.
        old_cwd = Path.cwd()
        try:
            os.chdir(self.work)
            op = gmxapi.commandline_operation(executable=self.gmx, arguments=list(gmx_args))
            op.run()
            out = _future_text(getattr(op.output, "stdout", None))
            err = _future_text(getattr(op.output, "stderr", None)) or _future_text(getattr(op.output, "erroroutput", None))
            rc = _future_value(getattr(op.output, "returncode", None))
        finally:
            os.chdir(old_cwd)
        if out:
            self.log(out)
        if err:
            self.log(err)
        if rc not in (None, 0):
            raise RuntimeError(f"gmxapi command returned {rc}: {self.gmx} {' '.join(gmx_args)}")
        return out or ""

    def convert_to_cg(self):
        local_pdb = f"{self.seed}.pdb"
        self.run([
            self.args.sirah_pdb2pqr, "--ff", "PARSE", "--ffout", "AMBER",
            "--with-ph", str(self.args.sirah_ph), "--neutraln", "--neutralc",
            local_pdb, self.pqr.name,
        ], allow_gmxapi=False)
        self.run([str(self.cgconv), "-i", self.pqr.name, "-o", self.cg_pdb.name], allow_gmxapi=False)

    def build_protein_gro_and_topology(self):
        """Create cg.gro and topol.top before seed-specific solvation.

        Reference/full mode runs pdb2gmx once and caches the resulting peptide
        topology. Reuse mode skips pdb2gmx and converts the SIRAH CG PDB to GRO
        with editconf, then copies the cached base topol.top into this work dir.
        """
        if self.use_cached_topology:
            if self.topology_cache is None:
                raise RuntimeError("Topology reuse requested without a SirahTopologyCache.")
            self.run([self.gmx, "editconf", "-f", self.cg_pdb.name, "-o", "cg.gro"], allow_gmxapi=False)
            self.topology_cache.validate_signature(self.work / "cg.gro", strict=bool(self.args.sirah_reuse_strict_signature))
            self.topology_cache.materialize_base_topology(self.work)
            self.log("Reused cached SIRAH peptide topol.top; seed-specific solvation and ionization will still be regenerated.")
        else:
            self.run_pdb2gmx()
            if self.topology_cache is not None and not self.topology_cache.has_base_topology:
                self.topology_cache.capture_base_topology(self.work / "topol.top", self.work / "cg.gro")
                self.log("Captured reference SIRAH peptide topology for reuse by later seeds.")

    def build_topology_and_solvate(self):
        a = self.args
        self.build_protein_gro_and_topology()
        self.run([self.gmx, "editconf", "-f", "cg.gro", "-o", "cg_box.gro", "-bt", "dodecahedron", "-d", str(a.sirah_box_distance), "-c"], allow_gmxapi=False)
        self.run([self.gmx, "solvate", "-cp", "cg_box.gro", "-cs", str(self.wt416), "-o", "cg_sol1.gro", "-p", "topol.top"], allow_gmxapi=False)
        self.log(f"WT4 after first solvation: {count_wt4_from_gro(self.work / 'cg_sol1.gro')}")

    def remove_close_waters_and_ionize(self):
        a = self.args
        self.run([self.gmx, "make_ndx", "-f", "cg_sol1.gro", "-o", "cg_sol1.ndx"], input_text='"System" & ! "WT4"\nq\n', allow_gmxapi=False)
        self.grompp("em1.mdp", "topol.top", "cg_sol1.gro", "cg_sol1.tpr")
        self.run([
            self.gmx, "select", "-f", "cg_sol1.gro", "-s", "cg_sol1.tpr", "-n", "cg_sol1.ndx",
            "-on", "rm_close_wt4.ndx",
            "-select", 'not (same residue as (resname WT4 and within 0.3 of group "System_&_!WT4"))',
        ], allow_gmxapi=False)
        self.run([self.gmx, "editconf", "-f", "cg_sol1.gro", "-o", "cg_sol2.gro", "-n", "rm_close_wt4.ndx"], allow_gmxapi=False)
        wt4 = count_wt4_from_gro(self.work / "cg_sol2.gro")
        update_topol_wt4_count(self.work / "topol.top", wt4)
        self.log(f"WT4 after close-water removal: {wt4}")
        self.grompp("em1.mdp", "topol.top", "cg_sol2.gro", "cg_sol2.tpr", ref="cg_sol2.gro")
        self.run([self.gmx, "genion", "-s", "cg_sol2.tpr", "-o", "cg_ion.gro", "-pname", "NaW", "-nname", "ClW", "-p", "topol.top", "-neutral", "-conc", str(a.sirah_salt)], input_text="WT4\n", allow_gmxapi=False)

    def make_restraints(self):
        self.run([self.gmx, "make_ndx", "-f", "cg_ion.gro", "-o", "cg_ion.ndx"], input_text="a GN GO\nq\n", allow_gmxapi=False)
        if self.use_cached_topology:
            if self.topology_cache is None:
                raise RuntimeError("Topology reuse requested without a SirahTopologyCache.")
            self.topology_cache.materialize_restraints(self.work)
            self.log("Reused cached SIRAH GN/GO restraint ITP files; cg_ion.ndx was still regenerated for this seed.")
        else:
            self.run([self.gmx, "genrestr", "-f", "cg_ion.gro", "-n", "cg_ion.ndx", "-o", "bkbres.itp"], input_text="GN_GO\n", allow_gmxapi=False)
            self.run([self.gmx, "genrestr", "-f", "cg_ion.gro", "-n", "cg_ion.ndx", "-o", "bkbres_soft.itp", "-fc", "100", "100", "100"], input_text="GN_GO\n", allow_gmxapi=False)
            if self.topology_cache is not None and not self.topology_cache.ready:
                self.topology_cache.capture_restraints(self.work)
                self.log("Captured reference SIRAH GN/GO restraint files for reuse by later seeds.")
        inject_sirah_backbone_restraints(self.work / "topol.top")

    def grompp(self, mdp, top, conf, tpr, ndx=None, ref=None):
        cmd = [self.gmx, "grompp", "-f", mdp, "-p", top, "-c", conf, "-o", tpr, "-maxwarn", str(self.args.sirah_maxwarn)]
        if ndx:
            cmd[cmd.index("-c"):cmd.index("-c")] = ["-n", ndx]
        if ref:
            cmd.extend(["-r", ref])
        self.run(cmd)

    def mdrun(self, deffnm, use_device=False):
        cmd = [self.gmx, "mdrun", "-deffnm", deffnm, "-v", "-ntomp", str(self.args.sirah_ntomp), "-pin", "on"]
        if use_device and str(self.args.sirah_device).lower() != "auto":
            cmd.extend(["-pme", str(self.args.sirah_device), "-nb", str(self.args.sirah_device)])
        self.run(cmd)

    def prepare_production_tpr(self):
        """Prepare the equilibrated seed as a production-ready cluster TPR.

        This mirrors the downstream shell pattern, but with compact per-seed names:
            gmx_mpi grompp -f sample.mdp -n prot.ndx -p topol.top \
                -c sample.gro -r sample.gro -o sample.tpr -maxwarn 5
        """
        if not bool(getattr(self.args, "sirah_prepare_md_tpr", True)):
            self.log("Production TPR preparation disabled (--no-sirah-prepare-md-tpr).")
            return
        cid = int(self.cluster_id)
        start_gro = "sample.gro"
        tpr = "sample.tpr"
        if not (self.work / "cg_eq2.gro").exists():
            raise FileNotFoundError("Cannot prepare production TPR; missing cg_eq2.gro")
        shutil.copy2(self.work / "cg_eq2.gro", self.work / start_gro)
        # Build the production index explicitly because the embedded/sample MDP
        # uses compressed-x-grps = System_&_!WT4_&_!NAW_&_!CLW. Do not merely
        # copy cg_ion.ndx: earlier setup indices may not contain this exact
        # group name.
        ndx_cmd = [self.gmx, "make_ndx", "-f", start_gro, "-o", "prot.ndx"]
        if (self.work / "cg_ion.ndx").exists():
            ndx_cmd.extend(["-n", "cg_ion.ndx"])
        self.run(
            ndx_cmd,
            input_text='"System" & ! "WT4" & ! "NAW" & ! "CLW"\nq\n',
            allow_gmxapi=False,
        )
        self.grompp("sample.mdp", "topol.top", start_gro, tpr, ndx="prot.ndx", ref=start_gro)
        self.log(f"Seed {self.seed} workflow complete: {tpr} generated from {start_gro}")

    def minimize_and_equilibrate(self):
        self.grompp("em1.mdp", "topol.top", "cg_ion.gro", "cg_em1.tpr", ndx="cg_ion.ndx", ref="cg_ion.gro")
        self.mdrun("cg_em1")
        self.grompp("em2.mdp", "topol.top", "cg_em1.gro", "cg_em2.tpr", ndx="cg_ion.ndx")
        self.mdrun("cg_em2")
        self.run([self.gmx, "make_ndx", "-f", "cg_em1.gro", "-o", "cg_ion.ndx", "-n", "cg_ion.ndx"], input_text='"System" & ! "WT4"\nq\n', allow_gmxapi=False)
        self.grompp("eq1.mdp", "topol.top", "cg_em2.gro", "cg_eq1.tpr", ndx="cg_ion.ndx", ref="cg_em2.gro")
        self.mdrun("cg_eq1", use_device=True)
        self.grompp("eq2.mdp", "topol.top", "cg_eq1.gro", "cg_eq2.tpr", ndx="cg_ion.ndx", ref="cg_eq1.gro")
        self.mdrun("cg_eq2", use_device=True)
        self.run([str(self.g_top2psf), "-i", "topol.top", "-o", "cg_eq2.psf"], allow_gmxapi=False)
        self.prepare_production_tpr()

    def execute(self) -> SirahResult:
        try:
            self.prepare()
            with self.log_path.open("w") as handle:
                self._log = handle
                self.log(f"SIRAH CG workflow for {self.input_pdb}")
                self.log(f"Compact seed: {self.seed} | cluster_id: {self.cluster_id} | source seed: {self.source_seed_name}")
                self.log(f"SIRAH FF: {self.sirah_ff}")
                self.log(f"GROMACS executable: {self.gmx}")
                self.convert_to_cg()
                self.build_topology_and_solvate()
                self.remove_close_waters_and_ionize()
                self.make_restraints()
                self.minimize_and_equilibrate()
            return self.result(True)
        except BaseException:
            return self.result(False, traceback.format_exc())


def _future_value(obj):
    if obj is None:
        return None
    try:
        return obj.result()
    except Exception:
        try:
            return obj
        except Exception:
            return None


def _future_text(obj) -> str:
    val = _future_value(obj)
    if val is None:
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    return str(val)


def run_sirah_for_one_seed(args, pdb_path: Path, out_root: Path, topology_cache: Optional[SirahTopologyCache] = None, use_cached_topology: bool = False, seed_index: int = 0, seed_name: Optional[str] = None) -> SirahResult:
    return SirahCGRunner(
        args, pdb_path, out_root, topology_cache=topology_cache,
        use_cached_topology=use_cached_topology, seed_index=seed_index, seed_name=seed_name,
    ).execute()

def discover_sirah_input_pdbs(args) -> list[Path]:
    if args.sirah_input_dir is not None:
        input_dir = Path(args.sirah_input_dir)
    else:
        primary = Path(args.out) / "final_implicit_survivor_seeds"
        legacy = Path(args.out) / "explicit_survivor_seeds"
        input_dir = primary if primary.exists() else legacy
    if input_dir.is_file() and input_dir.suffix.lower() == ".pdb":
        pdbs = [input_dir]
    else:
        pdbs = sorted(Path(input_dir).glob(str(args.sirah_pattern)))
    if int(args.sirah_max_seeds or 0) > 0:
        pdbs = pdbs[: int(args.sirah_max_seeds)]
    return [Path(p) for p in pdbs]


def run_sirah_cg_stage(args):
    if not getattr(args, "sirah_cg", False):
        return None
    out_root = Path(args.sirah_out) if args.sirah_out is not None else Path(args.out) / "sirah_seeds"
    out_root.mkdir(parents=True, exist_ok=True)
    pdbs = discover_sirah_input_pdbs(args)
    if not pdbs:
        raise RuntimeError(
            "No final implicit survivor PDBs found for SIRAH conversion. "
            "Use --sirah-input-dir or run with --sirah-cg/--implicit-only so final_implicit_survivor_seeds exists."
        )

    print_stage("SIRAH coarse-grain conversion + GROMACS equilibration", phase_key="explicit_min")
    ui_message(f"Input PDBs: {len(pdbs)} | output: {out_root}")
    ui_message(f"SIRAH FF: {args.sirah_ff_dir} | salt: {args.sirah_salt} M | ntomp: {args.sirah_ntomp}")
    render_dashboard()

    seed_plan = []
    for i, pdb_path in enumerate(pdbs, start=1):
        seed_name = format_sirah_seed_name(args, pdb_path, i)
        seed_plan.append({
            "seed_index": i,
            "cluster_id": i,
            "seed_name": seed_name,
            "source_seed_name": sirah_seed_name(pdb_path),
            "input_pdb": str(Path(pdb_path).resolve()),
            "work_dir": str((out_root / seed_name).resolve()),
        })
    write_dict_csv(out_root / "sirah_seed_manifest.csv", list(seed_plan[0].keys()), seed_plan)

    results = []
    progress = CliProgress("SIRAH CG seeds", len(pdbs))
    topology_cache = SirahTopologyCache(out_root / "_topology_cache") if bool(getattr(args, "sirah_reuse_topology", True)) else None
    if topology_cache is not None:
        state = "ready" if topology_cache.ready else "will be built from the first successful seed"
        ui_message(f"SIRAH topology reuse enabled: cache {state}. Solvation, close-water removal, and genion remain per-seed.")
        render_dashboard()
    for i, pdb_path in enumerate(pdbs, start=1):
        seed_name = seed_plan[i - 1]["seed_name"]
        use_cached = bool(topology_cache is not None and topology_cache.ready)
        mode = "reuse-topol" if use_cached else "reference/full-topol"
        ui_message(f"SIRAH {seed_name} ({i}/{len(pdbs)}) [{mode}]: {pdb_path.name}")
        render_dashboard()
        res = run_sirah_for_one_seed(
            args, pdb_path, out_root, topology_cache=topology_cache,
            use_cached_topology=use_cached, seed_index=i, seed_name=seed_name,
        )
        results.append(res)
        ok = sum(1 for r in results if r.success)
        cache_state = "yes" if bool(topology_cache is not None and topology_cache.ready) else "no"
        progress.update(i, ok=ok, fail=len(results)-ok, cache=cache_state)
    progress.done(len(pdbs), ok=sum(1 for r in results if r.success), fail=sum(1 for r in results if not r.success))

    summary_csv = out_root / "sirah_cg_equilibration_summary.csv"
    write_csv_dataclass(summary_csv, results)
    selected = [r for r in results if r.success and (not bool(getattr(args, "sirah_prepare_md_tpr", True)) or Path(r.production_tpr_path).exists())]
    clusters_sel = out_root / str(getattr(args, "sirah_clusters_sel_name", "SIRAH_SEEDS_SEL") or "SIRAH_SEEDS_SEL")
    clusters_sel.write_text("".join(f"{r.seed_name}\n" for r in selected))
    write_csv_dataclass(out_root / "sirah_generated_seed_summary.csv", results)
    fail = [r for r in results if not r.success]
    if fail:
        write_csv_dataclass(out_root / "failed_sirah_cg_equilibrations.csv", fail)
        ui_message(f"SIRAH failures: {len(fail)}; see failed_sirah_cg_equilibrations.csv and per-seed logs.")
    else:
        ui_message(f"SIRAH CG stage completed for all seeds. Prepared seed names written to {clusters_sel.name}.")
    render_dashboard()
    return out_root


def _csv_count_summary(path: Path, success_col: str = "success") -> dict:
    path = Path(path)
    if not path.exists():
        return {"exists": False, "rows": 0, "success": 0, "fail": 0}
    try:
        df = pd.read_csv(path)
        rows = int(len(df))
        if success_col in df.columns:
            success = int(df[success_col].astype(bool).sum())
        else:
            success = rows
        return {"exists": True, "rows": rows, "success": success, "fail": max(0, rows - success)}
    except Exception as exc:
        return {"exists": True, "rows": 0, "success": 0, "fail": 0, "error": str(exc)}


def write_turbo_summary(args, sirah_out=None, pca_out=None):
    """Write a compact end-of-run summary with the most useful run diagnostics."""
    if not bool(getattr(args, "turbo_summary", True)):
        return None
    out = Path(args.out)
    summary = {
        "sequence": str(args.seq),
        "output_dir": str(out),
        "config_compatibility": getattr(args, "config_compat_report", {}),
        "knobs": {
            "n": int(args.n),
            "n_candidate_seeds": int(args.n_candidate_seeds),
            "n_final_seeds": int(args.n_final_seeds),
            "generation_backend": str(args.generation_backend),
            "rama_sampling": str(args.rama_sampling),
            "diversity_bank_preset": str(getattr(args, "diversity_bank_preset", "off")),
            "preselection_bin_quota": int(getattr(args, "preselection_bin_quota", 0) or 0),
            "tiered_implicit_min": bool(getattr(args, "tiered_implicit_min", False)),
            "basin_hop": bool(args.basin_hop),
            "nma_expand": bool(args.nma_expand),
            "explore_loop": bool(args.explore_loop),
            "platform": str(args.platform),
            "jobs": int(args.jobs),
            "min_jobs": int(get_min_jobs(args)),
            "implicit_max_iterations": int(args.implicit_max_iterations),
            "adaptive_min": bool(args.adaptive_min),
            "adaptive_min_force_check_every": int(getattr(args, "adaptive_min_force_check_every", 1)),
        },
        "counts": {
            "candidate_seeds": len(list((out / "candidate_seeds").glob("*.pdb"))) if (out / "candidate_seeds").exists() else 0,
            "implicit_minima": len(list((out / "aa_implicit_minimized_pdbs").glob("*.pdb"))) if (out / "aa_implicit_minimized_pdbs").exists() else 0,
            "basin_hop_minima": len(list((out / "basin_hop_minima").glob("*.pdb"))) if (out / "basin_hop_minima").exists() else 0,
            "final_survivors": len(list((out / "final_implicit_survivor_seeds").glob("*.pdb"))) if (out / "final_implicit_survivor_seeds").exists() else 0,
        },
        "score_tables": {
            "implicit": _csv_count_summary(out / "implicit_minimization_scores.csv"),
            "implicit_scout": _csv_count_summary(out / "implicit_scout_minimization_scores.csv"),
            "basin_hop": _csv_count_summary(out / "basin_hop_minima.csv"),
            "nma_implicit": _csv_count_summary(out / "nma_implicit_minimization_scores.csv"),
        },
        "reports": {
            "config_compatibility": str(out / "GENPEPT_config_compatibility.json"),
            "generation_diversity_summary": str(out / "generation_diversity_summary.json"),
            "generation_bank_counts": str(out / "generation_bank_counts.csv"),
            "generation_quota_preselection_summary": str(out / "generation_quota_preselection_summary.json"),
            "tiered_implicit_summary": str(out / "tiered_implicit_summary.json"),
            "exploration_space_metrics": str(out / "exploration_space_metrics.csv"),
            "final_survivor_seeds": str(out / "final_survivor_seeds.csv"),
        },
        "optional_outputs": {
            "sirah_cg_seeds": str(sirah_out) if sirah_out is not None else None,
            "pca_pseudo_fes": str(pca_out) if pca_out is not None else None,
        },
    }
    try:
        gen_summary_path = out / "generation_diversity_summary.json"
        if gen_summary_path.exists():
            summary["generation_diversity"] = json.loads(gen_summary_path.read_text())
    except Exception:
        pass
    try:
        tier_path = out / "tiered_implicit_summary.json"
        if tier_path.exists():
            summary["tiered_implicit"] = json.loads(tier_path.read_text())
    except Exception:
        pass
    compat_report = getattr(args, "config_compat_report", {})
    if compat_report:
        try:
            (out / "GENPEPT_config_compatibility.json").write_text(json.dumps(compat_report, indent=2))
        except Exception:
            pass
    path = out / "GENPEPT_turbo_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return path

def main(argv=None):
    global UI_CLEAR_SCREEN, UI_COLOR
    args = parse_args(argv)
    UI_CLEAR_SCREEN = not getattr(args, "no_clear_screen", False)
    UI_COLOR = not getattr(args, "no_color", False)
    args.seq = validate_sequence(args.seq)
    if args.no_explicit:
        args.implicit_only = True
    if getattr(args, "sirah_cg", False):
        # SIRAH consumes the final selected implicit minima directly. Do not run
        # the atomistic explicit-solvent minimization stage in this workflow.
        args.implicit_only = True

    if getattr(args, "explore_loop", False):
        # Make --explore-loop useful even when a config only toggles it on.
        # These defaults are deliberately modest; production configs should set
        # explicit values. They also prevent the confusing "enabled but did
        # exactly zero rounds" outcome.
        if int(getattr(args, "explore_rounds", 0) or 0) <= 0:
            args.explore_rounds = 1
        if int(getattr(args, "explore_keep", 0) or 0) <= 0:
            args.explore_keep = max(100, int(getattr(args, "n_final_seeds", 500) or 500) // 5)
        if int(getattr(args, "explore_proposals", 0) or 0) <= 0:
            args.explore_proposals = max(10000, int(args.explore_keep) * 20)

    if args.water not in FF_CHOICES[args.ff]["waters"]:
        valid = ", ".join(FF_CHOICES[args.ff]["waters"])
        raise ValueError(f"Invalid --water {args.water!r} for --ff {args.ff!r}. Valid: {valid}")

    print_stage("Peptide basin-seed workflow")
    ui_message(f"Sequence: {args.seq} | output: {args.out}")
    for _msg in getattr(args, "config_compat_messages", [])[:3]:
        ui_message("Config: " + str(_msg))
    ui_message(f"Trials: {args.n} | candidate seeds: {args.n_candidate_seeds} | final seeds: {args.n_final_seeds}")
    ui_message(f"Generation jobs: {args.jobs} | gen chunk size: {args.gen_chunk_size} | OpenMM jobs: {get_min_jobs(args)}")
    ui_message(f"Platform: {args.platform} | basin hop: {args.basin_hop} | NMA expand: {args.nma_expand}")
    diversity_bits = []
    if str(getattr(args, "diversity_bank_preset", "off")) != "off":
        diversity_bits.append(f"banks={args.diversity_bank_preset}")
    if int(getattr(args, "preselection_bin_quota", 0) or 0) > 0:
        diversity_bits.append(f"quota={args.preselection_bin_quota}/bin")
    if diversity_bits:
        ui_message("Diversity turbo: " + " | ".join(diversity_bits))
    if getattr(args, "tiered_implicit_min", False):
        refine_n = int(getattr(args, "tier_refine_seeds", 0) or args.n_candidate_seeds)
        ui_message(f"Tiered implicit: scout={args.tier_scout_iterations} iter -> refine {refine_n} seeds at {args.tier_refine_iterations or args.implicit_max_iterations} iter")
    ui_message("Atomistic explicit minimization: disabled" if args.sirah_cg else f"Explicit solvent: {not args.implicit_only and (args.two_stage or args.implicit_only)}")
    if args.explore_loop:
        explore_bh_note = " | BH restart after PCA hits" if args.basin_hop and getattr(args, "explore_bh", True) else ""
        ui_message(f"Adaptive explore: enabled | rounds={args.explore_rounds} proposals/round={args.explore_proposals} keep/round={args.explore_keep}{explore_bh_note}")
    render_dashboard()

    _resume = getattr(args, "resume", False)
    _out = Path(args.out)

    # ── Stage 1: Generation ────────────────────────────────────────────────────
    _cand_csv = _out / "candidate_seeds.csv"
    _cand_dir = _out / "candidate_seeds"
    if _resume and _resume_csv_done(_cand_csv) and _cand_dir.is_dir() and any(_cand_dir.glob("*.pdb")):
        ui_message(f"[resume] Generation: skipping — existing {_cand_dir.name}/ ({_cand_csv.stat().st_size // 1024} KB CSV)")
        render_dashboard()
        candidate_dir = _cand_dir
    else:
        candidate_dir = generate_candidates(args)

    if args.two_stage or args.implicit_only:
        implicit_dir = _out / "aa_implicit_minimized_pdbs"
        implicit_scores = _out / "implicit_minimization_scores.csv"

        # ── Stage 2: Implicit minimization ────────────────────────────────────
        if _resume and _resume_csv_done(implicit_scores):
            ui_message(f"[resume] Implicit min: skipping — loading {implicit_scores.name}")
            render_dashboard()
            implicit_results = minresults_from_score_csv(implicit_scores, out_dir=_out)
        elif getattr(args, "tiered_implicit_min", False):
            implicit_results = run_tiered_implicit_minimization(args, candidate_dir)
        else:
            implicit_results = run_minimization_batch(args, candidate_dir, implicit_dir, "implicit", implicit_scores)

        combined_results = list(implicit_results)
        archive_dir = _out / "basin_archive"

        _bh_csv = _out / "basin_hop_minima.csv"
        _bh_done = _resume and _resume_csv_done(_bh_csv)
        if not _bh_done:
            print_stage("Initial basin archive / scoring", phase_key="exploration", cycle_step="archive")
            ui_message("Archiving/scoring initial implicit minima before optional exploration stages.")
            render_dashboard()
            write_basin_archive(args, combined_results, archive_dir, prefix="initial_implicit")

        # ── Stage 3: Basin hopping ─────────────────────────────────────────────
        if args.basin_hop:
            if _bh_done:
                ui_message(f"[resume] Basin hopping: skipping — loading {_bh_csv.name}")
                render_dashboard()
                combined_results.extend(minresults_from_hop_csv(args, _bh_csv))
            else:
                # Smart default: hop from minimized basins, not raw candidates. This avoids
                # spending the midphase on structures that immediately collapse during minimization.
                bh_source_dir = candidate_dir if getattr(args, "bh_from_candidates", False) else implicit_dir
                bh_dir = run_basin_hopping(args, bh_source_dir)
                combined_results.extend(minresults_from_hop_csv(args, _bh_csv))
                print_stage("Post-BH/MC basin archive / scoring", phase_key="exploration", cycle_step="archive")
                ui_message("Archiving/scoring search pool after basin hopping / Monte Carlo endpoints.")
                render_dashboard()
                write_basin_archive(args, combined_results, archive_dir, prefix="post_basin_hop", hop_csv=_bh_csv)

        # ── Stage 4: NMA expand + minimize ────────────────────────────────────
        if args.nma_expand:
            nma_scores = _out / "nma_implicit_minimization_scores.csv"
            if _resume and _resume_csv_done(nma_scores):
                ui_message(f"[resume] NMA + minimize: skipping — loading {nma_scores.name}")
                render_dashboard()
                combined_results.extend(minresults_from_score_csv(nma_scores, out_dir=_out))
            else:
                # Expand from basin-hop minima if present, otherwise from implicit minima.
                source_dir = _out / "basin_hop_minima" if args.basin_hop else implicit_dir
                nma_dir = nma_expand_from_dir(args, source_dir)
                if any(nma_dir.glob("*.pdb")):
                    nma_imp_dir = _out / "aa_nma_implicit_minimized_pdbs"
                    nma_results = run_minimization_batch(args, nma_dir, nma_imp_dir, "implicit", nma_scores, ui_phase_key="exploration", ui_cycle_step="minimize")
                    combined_results.extend(nma_results)
                    print_stage("Post-NMA basin archive / scoring", phase_key="exploration", cycle_step="archive")
                    ui_message("Archiving/scoring search pool after NMA-generated minima.")
                    render_dashboard()
                    write_basin_archive(args, combined_results, archive_dir, prefix="post_nma")

        if args.explore_loop:
            run_adaptive_exploration_loop(args, combined_results)
            print_stage("Post-PCA exploration basin archive / scoring", phase_key="exploration", cycle_step="archive")
            ui_message("Archiving/scoring search pool after adaptive PCA exploration.")
            render_dashboard()
            write_basin_archive(args, combined_results, archive_dir, prefix="post_pca_explore")

        print_stage("Final search-pool basin archive / scoring", phase_key="exploration", cycle_step="archive")
        ui_message("Writing final basin archive before survivor selection.")
        render_dashboard()
        write_basin_archive(args, combined_results, archive_dir, prefix="final_search_pool", hop_csv=Path(args.out) / "basin_hop_minima.csv")
        survivor_dir = select_implicit_survivors(args, combined_results)

        # Atomistic explicit-solvent minimization intentionally removed from this
        # SIRAH-oriented workflow. The final selected implicit minima are the
        # handoff structures for SIRAH CG conversion/equilibration.

    sirah_out = run_sirah_cg_stage(args)

    pca_out = run_integrated_pca_fes(args)

    turbo_summary_path = write_turbo_summary(args, sirah_out=sirah_out, pca_out=pca_out)

    print_stage("done", phase_key="done")
    print()
    print("Done.")
    print(f"Output:             {args.out}")
    print(f"Candidate seeds:    {Path(args.out) / 'candidate_seeds'}")
    if args.two_stage or args.implicit_only:
        print(f"Implicit minima:    {Path(args.out) / 'aa_implicit_minimized_pdbs'}")
        if args.basin_hop:
            print(f"Basin-hop minima:   {Path(args.out) / 'basin_hop_minima'}")
        if args.nma_expand:
            print(f"NMA probes:         {Path(args.out) / 'nma_probe_seeds'}")
        if args.explore_loop:
            print(f"Adaptive PCA:       {Path(args.out) / 'adaptive_pca_exploration'}")
        print(f"Final implicit survivors: {Path(args.out) / 'final_implicit_survivor_seeds'}")
    if 'sirah_out' in locals() and sirah_out is not None:
        print(f"SIRAH CG seeds:     {sirah_out}")
    if 'pca_out' in locals() and pca_out is not None:
        print(f"PCA pseudo-FES:     {pca_out}")
    if 'turbo_summary_path' in locals() and turbo_summary_path is not None:
        print(f"Turbo summary:      {turbo_summary_path}")
        compat_path = Path(args.out) / "GENPEPT_config_compatibility.json"
        if compat_path.exists():
            print(f"Config report:      {compat_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
