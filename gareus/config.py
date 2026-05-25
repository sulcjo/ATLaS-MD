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
    "adaptive_rounds": "adaptive_feedback_rounds",
    "pilot_fraction": "adaptive_feedback_pilot_fraction",
    "min_total_windows": "adaptive_min_total_windows",
    "max_total_windows": "adaptive_max_total_windows",
    "min_total_replicas": "adaptive_min_total_windows",
    "max_total_replicas": "adaptive_max_total_windows",
    "adaptive_min_total_replicas": "adaptive_min_total_windows",
    "adaptive_max_total_replicas": "adaptive_max_total_windows",
    "contact_adaptive_min_total_replicas": "contact_adaptive_min_total_windows",
    "contact_adaptive_max_total_replicas": "contact_adaptive_max_total_windows",
    "validation_steps": "adaptive_feedback_validation_steps",
    "aggressiveness": "adaptive_window_aggressiveness",
    "secondary_mode": "secondary_cv",
    "boost_type": "gamd_boost_type",
    "production_steps": "gamd_production_steps",
    "cmd_prep_steps": "gamd_cmd_prep_steps",
    "cmd_steps": "gamd_cmd_steps",
    "equil_prep_steps": "gamd_equil_prep_steps",
    "equil_steps": "gamd_equil_steps",
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
        path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")
        return
    yaml_mod = _yaml_module()
    if yaml_mod is not None:
        path.write_text(yaml_mod.safe_dump(_json_ready(payload), sort_keys=False, default_flow_style=False))
    else:
        path.write_text(_minimal_yaml_dump(_json_ready(payload)) + "\n")


def _load_config_file(path: Path) -> dict:
    """Load a YAML or JSON configuration file into a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
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
    (cfg_dir / "command_line.txt").write_text(command_text + "\n")
    (out_dir / "command_line.txt").write_text(command_text + "\n")
    script_name = str(Path(__file__).resolve())
    resume_cmd = f"{shlex.quote(sys.executable)} {shlex.quote(script_name)} --config {shlex.quote(str((cfg_dir / 'effective_config.yaml').resolve()))} --resume\n"
    for path in (cfg_dir / "resume_command.sh", out_dir / "resume_command.sh"):
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + resume_cmd)
        try:
            path.chmod(path.stat().st_mode | 0o111)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Config template and example
# -----------------------------------------------------------------------------

def _basic_chignolin_config() -> Dict[str, Any]:
    """Return a ready‑to‑edit starting config grouped by feature area."""
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
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
            "replica_device_map": "",
            "cuda_use_cpu_pme": "auto",
            "cuda_use_blocking_sync": "auto",
            "cuda_deterministic_forces": "auto",
            "platform_temp_directory": "",
            "cpu_threads": 1,
            "setup_platform": "",
            "setup_precision": "",
            "setup_device_index": "",
            "setup_cpu_threads": 0,
        },
        "simulation": {
            "run_mode": "hmr-gamd",
            "hmr": True,
            "hydrogen_mass_amu": 3.024,
            "timestep_fs": 4.0,
            "box_shape": "dodecahedron",
            "padding_nm": 1.0,
            "water_model": "tip3p",
            "ionic_strength_molar": 0.15,
        },
        "windows": {
            "cv1": "distance",
            "cv2": "rama-map",
            "window_mode": "adaptive-feedback",
            "adaptive_prescan_steps": 50000,
            "adaptive_window_aggressiveness": "aggressive",
            "adaptive_feedback_rounds": 10,
            "adaptive_feedback_pilot_fraction": 0.008,
            "adaptive_feedback_validation_steps": 10000,
        },
        "genpept_prior": {
            "# note": "Optional round-zero adaptive-feedback prior. GENPEPT remains unchanged; GAREUS rescoring writes genpept_prior/genpept_prior_windows_2d.csv and feeds it to the first pilot.",
            "genpept_prior_enabled": False,
            "genpept_prior_dir": "chignolin_genpept_seeds/",
            "genpept_prior_stages": [
                "final_search_pool",
                "post_pca_explore",
                "post_nma",
                "post_basin_hop",
                "initial_implicit",
                "final_survivor_seeds",
            ],
            "genpept_prior_max_structures": 50000,
            "genpept_prior_min_points": 10,
            "genpept_prior_max_windows": 48,
            "genpept_prior_bins": "24,12",
            "genpept_prior_min_hits_per_bin": 2,
            "genpept_prior_absence_means_unknown": True,
            "genpept_prior_snap_secondary_centers": True,
            "genpept_prior_required": False,
        },
        "secondary_cv": {
            "# note": "cv2 above auto-expands to a 2D secondary-center ladder; rama-map defaults to beta/PPII/right-alpha/left-alpha centers. Set secondary_cv_centers here to override.",
        },
        "starting_structures": {
            "seed_conformers_dir": "chignolin_genpept_seeds/",
            "seed_selection_mode": "auto",
            "seed_secondary_weight": 1.0,
            "seed_max_reuse_per_conformer": 0,
        },
        "gamd": {
            "gamd_boost_type": "lower-dual",
            "gamd_production_steps": 3000000,
            "sigma0p_kcal_mol": 5.0,
            "sigma0d_kcal_mol": 5.0,
        },
        "exchange": {
            "exchange_mode": "gibbs-walk",
            "exchange_interval": 100,
        },
        "output_intervals": {
            "traj_interval": 50,
            "traj_format": "xtc",
            "adaptive_pilot_trajectories": False,
            "dashboard_render_interval_sec": 2.0,
            "dashboard_panels": "normal",
            "dashboard_heavy_panels_every": 2,
            "analysis_consolidated_max_elements": 100000000,
            "analysis_array_dtype": "float64",
            "sample_potential_energy": True,
            "flush_every_log": True,
            "report_interval": 50,
            "distance_output_interval": 50,
            "checkpoint_interval": 10,
        },
        "tui": {
            # Terminal UI settings.  A future refactor could include these in tui.py.
        },
    }


def _write_config_template(path: Path, variant: str = "basic") -> None:
    """Write a basic or variant config template to the given path.

    The ``variant`` parameter is reserved for future template variants; currently
    only ``"basic"`` is defined, so all variants resolve to the chignolin template.
    """
    payload = _basic_chignolin_config()
    _write_yaml_or_json(Path(path), payload)