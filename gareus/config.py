"""
Configuration loading and reproducibility helpers.

This module provides functions for reading and writing configuration
files (YAML or JSON), normalising nested config mappings into flat
argparse default dictionaries, and generating reproducible payloads
describing the runtime environment and command line.  These functions
were originally defined in ``gareus_peptide.py`` but have been
extracted here to improve modularity.

The central constant ``CONFIG_SCHEMA_VERSION`` identifies the version
of the configuration schema used by the script.  A small alias map
``CONFIG_KEY_ALIASES`` allows abbreviated keys in user‑provided config
files to map onto the longer argparse destination names.

To apply configuration defaults to an ``argparse.ArgumentParser``, use
``_apply_config_defaults_to_parser(parser, path)``.  To write the
effective configuration and command‑line information for
reproducibility, use ``_write_reproducibility_files(args, out_dir)``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any, List

from .io import _json_ready  # type: ignore

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CONFIG_IGNORED_TOP_LEVEL_KEYS",
    "CONFIG_KEY_ALIASES",
    "_yaml_module",
    "_minimal_yaml_scalar",
    "_minimal_yaml_dump",
    "_write_yaml_or_json",
    "_load_config_file",
    "_normalize_config_key",
    "_flatten_config_mapping",
    "_build_known_config_dests",
    "_apply_config_defaults_to_parser",
    "_argv_as_list",
    "_shell_join",
    "_script_sha256",
    "_effective_config_payload",
    "_write_reproducibility_files",
    "_write_config_template",
    "_basic_chignolin_config",
]


# -----------------------------------------------------------------------------
# Configuration schema and key aliasing
# -----------------------------------------------------------------------------

CONFIG_SCHEMA_VERSION: str = "1.0"

# Blocks for adjacent workflow stages that may live in a combined YAML file.
# GAREUS ignores these top-level blocks so one YAML can contain both GENPEPT
# seed-generation settings and GAREUS production settings.
CONFIG_IGNORED_TOP_LEVEL_KEYS: set[str] = {
    "conformer_generation",
    "genpept",
    "seed_generation",
}

# Nested config groups are for human readability only.  Leaf keys normally
# match argparse dest names exactly, e.g. ``gamd_production_steps``.  A small
# alias map accepts a few shorter spellings in hand‑written configs.
CONFIG_KEY_ALIASES: Dict[str, str] = {
    "sequence": "seq",
    "peptide_sequence": "seq",
    "output_dir": "out",
    "output": "out",
    "mode": "window_mode",
    "primary": "cv1",
    "primary_cv_short": "cv1",
    "secondary": "cv2",
    "cv_1": "cv1",
    "cv_2": "cv2",
    # v2.0: aggressiveness, pilot_fraction, validation_steps, adaptive_rounds,
    #        production_steps, equil_steps are now canonical dests — aliases removed.
    "boost_type": "gamd_boost_type",
}


# -----------------------------------------------------------------------------
# YAML/JSON loading and dumping helpers
# -----------------------------------------------------------------------------

def _yaml_module():
    """Return the PyYAML module if available, else None."""
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        return None


def _minimal_yaml_scalar(value: Any) -> str:
    """Return a YAML scalar representation of a Python value.

    This helper emits simple values without quotes where possible and
    quotes strings that YAML might reinterpret.  It falls back to JSON
    encoding for unsupported types.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        # Keep simple strings unquoted, quote strings that YAML might reinterpret.
        if value == "" or any(ch in value for ch in ":#{}[]&,*?|-<>=!%@\\\n\t"):
            return json.dumps(value)
        lower = value.lower()
        if lower in {"true", "false", "null", "none", "yes", "no", "on", "off"}:
            return json.dumps(value)
        return value
    return json.dumps(value)


def _minimal_yaml_dump(obj: Any, indent: int = 0) -> str:
    """Recursively dump a Python object to a minimal YAML string."""
    pad = " " * int(indent)
    if isinstance(obj, dict):
        lines: List[str] = []
        for key, value in obj.items():
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                lines.append(_minimal_yaml_dump(value, indent + 2))
            elif isinstance(value, list):
                if not value:
                    lines.append(f"{pad}{key}: []")
                elif all(not isinstance(x, (dict, list)) for x in value):
                    vals = ", ".join(_minimal_yaml_scalar(x) for x in value)
                    lines.append(f"{pad}{key}: [" + vals + "]")
                else:
                    lines.append(f"{pad}{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            lines.append(f"{pad}  -")
                            lines.append(_minimal_yaml_dump(item, indent + 4))
                        else:
                            lines.append(f"{pad}  - {_minimal_yaml_scalar(item)}")
            else:
                lines.append(f"{pad}{key}: {_minimal_yaml_scalar(value)}")
        return "\n".join(lines)
    return pad + _minimal_yaml_scalar(obj)


def _write_yaml_or_json(path: Path, payload: Any) -> None:
    """Write a payload to disk as YAML or JSON based on the file suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    yaml_mod = _yaml_module()
    if yaml_mod is not None:
        path.write_text(yaml_mod.safe_dump(_json_ready(payload), sort_keys=False, default_flow_style=False), encoding="utf-8")
    else:
        path.write_text(_minimal_yaml_dump(_json_ready(payload)) + "\n", encoding="utf-8")


def _load_config_file(path: Path) -> dict:
    """Load a YAML or JSON configuration file into a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        yaml_mod = _yaml_module()
        if yaml_mod is None:
            raise RuntimeError(
                f"Config file {path} looks like YAML, but PyYAML is not installed. "
                "Install pyyaml or use a .json config."
            )
        data = yaml_mod.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping/object at top level: {path}")
    return data


def _normalize_config_key(key: str) -> str:
    """Normalize a raw config key by stripping and replacing dashes with underscores."""
    key = str(key).strip().replace("-", "_")
    return CONFIG_KEY_ALIASES.get(key, key)


def _flatten_config_mapping(config: Dict[str, Any], known_dests: set[str], path: str = "") -> Tuple[Dict[str, Any], List[str]]:
    """Flatten a nested config mapping into argparse dest defaults.

    Nested group names are ignored.  Leaf keys should match argparse dest names
    after dash‑to‑underscore normalization.  Unknown leaves are reported so
    typos fail early instead of silently starting a very expensive wrong run.
    Returns a tuple of (flat mapping, list of unknown keys).
    """
    flat: Dict[str, Any] = {}
    unknown: List[str] = []
    for raw_key, value in (config or {}).items():
        key = _normalize_config_key(raw_key)
        here = f"{path}.{raw_key}" if path else str(raw_key)
        if key in {"schema_version", "comments", "notes", "description"}:
            continue
        if not path and key in CONFIG_IGNORED_TOP_LEVEL_KEYS:
            continue
        if key in known_dests and not isinstance(value, dict):
            flat[key] = value
        elif isinstance(value, dict):
            sub_flat, sub_unknown = _flatten_config_mapping(value, known_dests, here)
            flat.update(sub_flat)
            unknown.extend(sub_unknown)
        elif key in known_dests:
            flat[key] = value
        else:
            unknown.append(here)
    return flat, unknown


def _build_known_config_dests(parser: argparse.ArgumentParser) -> set[str]:
    """Return the set of destination names for the given argparse parser."""
    return {str(a.dest) for a in parser._actions if getattr(a, "dest", None) and a.dest != argparse.SUPPRESS}


def _apply_config_defaults_to_parser(parser: argparse.ArgumentParser, config_path: Optional[str]) -> dict:
    """Load a config file and apply its defaults to an argparse parser.

    Unknown keys raise a ValueError.  Returns a dictionary with keys
    ``config_path``, ``config_values`` and ``unknown_keys``.
    """
    if not config_path:
        return {"config_path": None, "config_values": {}, "unknown_keys": []}
    raw = _load_config_file(Path(config_path))
    # Effective configs written by this script contain metadata plus an ``args``
    # block.  Treat that block as the actual input config so resume_command.sh can
    # point at config/effective_config.yaml directly.
    config_body = raw.get("args", raw) if isinstance(raw, dict) else raw
    known = _build_known_config_dests(parser)
    flat, unknown = _flatten_config_mapping(config_body, known)
    if unknown:
        raise ValueError(
            "Unknown config key(s): " + ", ".join(unknown) +
            ". Use argparse destination names such as 'gamd_production_steps', "
            "or run --write-config-template to generate a reference file."
        )
    parser.set_defaults(**flat)
    # Required CLI args, especially --seq, can be satisfied by config defaults.
    for action in parser._actions:
        if getattr(action, "dest", None) in flat:
            action.required = False
    return {"config_path": str(config_path), "config_values": flat, "unknown_keys": []}


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def _argv_as_list(argv: Optional[Iterable[str]]) -> List[str]:
    """Return argv as a list of strings, defaulting to sys.argv[1:] if None."""
    return [str(x) for x in (sys.argv[1:] if argv is None else list(argv))]


def _shell_join(argv: List[str]) -> str:
    """Return a shell‑escaped command line string for a list of arguments."""
    try:
        return shlex.join([str(x) for x in argv])
    except Exception:
        return " ".join(shlex.quote(str(x)) for x in argv)


def _script_sha256() -> str:
    """Return the SHA‑256 hash of this script file, or an empty string on error."""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        return ""


def _effective_config_payload(args: Any, argv: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Return a reproducibility payload describing the script, config and args."""
    arg_list = _argv_as_list(argv)
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _script_sha256(),
            "python": sys.version.split()[0],
            "argv": [sys.executable, str(Path(__file__).name)] + arg_list,
            "command_line": _shell_join([sys.executable, str(Path(__file__))] + arg_list),
            "working_directory": str(Path.cwd()),
        },
        "config_source": {
            "config": getattr(args, "config", None),
            "config_values": getattr(args, "_config_values", {}),
        },
        "args": {k: v for k, v in vars(args).items() if not str(k).startswith("_")},
    }


def _write_reproducibility_files(args: Any, out_dir: Path, argv: Optional[Iterable[str]] = None) -> None:
    """Write effective config, command line and resume script for reproducibility."""
    out_dir = Path(out_dir)
    cfg_dir = out_dir / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    payload = _effective_config_payload(args, argv=argv)
    _write_yaml_or_json(cfg_dir / "effective_config.yaml", payload)
    _write_yaml_or_json(cfg_dir / "effective_config.json", payload)
    # Also keep copies at the run root for backward‑compatible discoverability.
    _write_yaml_or_json(out_dir / "effective_config.yaml", payload)
    _write_yaml_or_json(out_dir / "effective_config.json", payload)
    command_text = payload["script"].get("command_line", "")
    (cfg_dir / "command_line.txt").write_text(command_text + "\n", encoding="utf-8")
    (out_dir / "command_line.txt").write_text(command_text + "\n", encoding="utf-8")
    script_name = str(Path(__file__).resolve())
    resume_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(script_name)} --config {shlex.quote(str((cfg_dir / 'effective_config.yaml').resolve()))} --resume\n"
    for path in (cfg_dir / "resume_command.sh", out_dir / "resume_command.sh"):
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + resume_cmd, encoding="utf-8")
        try:
            path.chmod(path.stat().st_mode | 0o111)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Config template and example
# -----------------------------------------------------------------------------

def _basic_chignolin_config() -> Dict[str, Any]:
    """Return a ready-to-edit v2.0 starting config grouped by feature area."""
    return {
        "schema_version": "2.0",
        "sequence": {
            "seq": "GYDPETGTWG",
        },
        "output": {
            "out": "chignolin/",
            "resume": True,
        },
        "platform": {
            "platform": "CUDA",
            "precision": "mixed",
            "device_index": "0,1,2,3",
            "replica_device_mode": "auto",
            "cuda_use_cpu_pme": "auto",
            "cuda_use_blocking_sync": "auto",
            "cpu_threads": 1,
            "setup_platform": "",
            "setup_precision": "",
            "setup_device_index": "",
            "setup_cpu_threads": 0,
        },
        "simulation": {
            "run_mode": "hmr-gamd",
            "timestep_fs": 4.0,
            "box_shape": "dodecahedron",
            "padding_nm": 1.0,
            "water_model": "tip3p",
            "ionic_strength_molar": 0.15,
        },
        "cvs": {
            "cv1": "contacts",
            "cv2": "rama-map",
        },
        "contact_cv": {
            "contact_scheme": "residue-balanced",
            "contact_atom_selection": "backbone-heavy",
            "contact_min_sequence_separation": 3,
            "contact_r0_a": 4.8,
            "contact_beta_a_inv": 4.0,
            "contact_normalize": True,
            # CV1 adaptive (unified; units from cv1 type)
            "cv1_range_min": 0.0,
            "cv1_range_max": 0.6,
            "cv1_target_spacing": 0.05,
            "cv1_k_min": 50.0,
            "cv1_k_max": 100.0,
            "cv1_prescan": True,
            "cv1_prescan_steps": 10000,
            "cv1_frontier": True,
            "cv1_frontier_probe_count": 2,
        },
        "cv2": {
            # CV2 adaptive params
            "cv2_k_mode": "adaptive",
            "cv2_k_min": 20.0,
            "cv2_k_max": 50.0,
            "cv2_adaptive_overlap_sigma": 20.0,
            "cv2_k_scale": 2.0,
        },
        "starting_structures": {
            "seed_conformers_dir": "chignolin_genpept_seeds/",
            "seed_selection_mode": "auto",
            "seed_cv2_weight": 1.0,
            "seed_max_reuse_per_conformer": 0,
            "us_pull_steps_per_window": 50000,
            "us_pull_k": 50.0,
            "us_pull_ramp_stages": 5,
            "us_2d_secondary_k_scale": 5.0,
        },
        "windows": {
            "window_mode": "adaptive-feedback",
            "aggressiveness": "aggressive",
            "adaptive_rounds": 10,
            "pilot_fraction": 0.008,
            "validation_steps": 10000,
            "sparse_2d": True,
            "max_2d_patches": 12,
            "region_memory": True,
        },
        "gamd": {
            "gamd_boost_type": "lower-dual",
            "production_steps": 3000000,
            "sigma0p": 5.0,
            "sigma0d": 5.0,
            "equil_steps": 100000,
        },
        "exchange": {
            "exchange_mode": "gibbs-walk",
            "exchange_interval": 100,
        },
        "output": {
            "traj_interval": 50,
            "traj_format": "xtc",
            "adaptive_pilot_trajectories": False,
            "adaptive_production_trajectories": True,
            "sample_potential_energy": True,
            "report_interval": 50,
            "distance_output_interval": 50,
            "checkpoint_interval": 10,
        },
        "tui": {
            "tui_mode": "dashboard",
        },
    }


def _write_config_template(path: Path, variant: str = "basic") -> None:
    """Write a basic or variant config template to the given path.

    The ``variant`` parameter is reserved for future template variants; currently
    only ``"basic"`` is defined, so all variants resolve to the chignolin template.
    """
    payload = _basic_chignolin_config()
    _write_yaml_or_json(Path(path), payload)