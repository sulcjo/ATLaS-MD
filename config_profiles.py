"""Shared GAREUS/GENPEPT config profiles and a YAML duplicate-key guard.

A ``profile: <name>`` key in a combined YAML expands to a hardcoded bundle of
boilerplate config keys, mirroring GENPEPT's existing ``diversity_bank_preset``
pattern. This lets a per-peptide config carry only ``profile`` + ``seq`` + the
couple of output paths that genuinely vary, instead of copy-pasting ~150
keys (the copy-paste is also how seed_conformers_dir/genpept_prescan_dir
ended up pointing at the wrong peptide's GENPEPT output in most RUNS/runs_rdy
configs).

This module is intentionally dependency-light (stdlib + PyYAML only, no
OpenMM/pandas/gareus) so importing it from GENPEPT.py's standalone loader
stays cheap; gareus/config.py and GENPEPT.py both import from here.

Precedence when a profile is used: defaults < profile bundle < explicit
YAML keys < CLI flags. Each tool's loader merges the bundle underneath its
own flattened explicit keys before applying argparse defaults, so an
explicit key always wins.
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import yaml

__all__ = [
    "GAREUS_PROFILES",
    "GENPEPT_PROFILES",
    "expand_profile",
    "merge_profile",
    "assert_no_duplicate_keys",
]


# Source: RUNS/runs_rdy/chignolin/chignolin.yaml (the byte-identical-across-
# 57/59-files baseline) + RUNS/runs_rdy/chignolin_tica/chignolin_tica_autoswitch.yaml's
# tica: block. Excludes seq/out/seed_conformers_dir/genpept_prescan_dir/description,
# which are per-peptide and must stay explicit or be derived from seq+out.
GAREUS_PROFILES: Dict[str, Dict[str, object]] = {
    "chignolin_tica": {
        "cv1": "contacts",
        "cv2": "rama-map",
        "contact_scheme": "residue-balanced",
        "contact_atom_selection": "heavy",
        "contact_min_sequence_separation": 3,
        "contact_r0_a": 4.8,
        "contact_beta_a_inv": 4.0,
        "contact_normalize": True,
        "cv1_range_min": 0.0,
        "cv1_target_spacing": 0.10,
        "cv1_k_min": 50.0,
        "cv1_k_max": 250.0,
        "cv1_boundary_pull_steps": 100000,
        "cv1_boundary_pull_k": 200.0,
        "cv1_frontier": True,
        "cv1_frontier_probe_count": 2,
        "cv2_k_min": 20.0,
        "cv2_k_max": 100.0,
        "cv2_k_mode": "adaptive",
        "cv2_adaptive_overlap_sigma": 20.0,
        "cv2_k_scale": 2.0,
        "cv2_centers": [-1.0, -0.333, 0.333, 1.0],
        "us_pull_steps_per_window": 50000,
        "us_pull_timestep_fs": 3.0,
        "us_pull_k": 75.0,
        "us_pull_ramp_stages": 5,
        "us_2d_secondary_k_scale": 5.0,
        "window_mode": "double-adaptive",
        "aggressiveness": "aggressive",
        "pilot_fraction": 0.004,
        "pilot_steps": 40000,
        "validation_steps": 50000,
        "adaptive_rounds": 5,
        "sparse_2d": True,
        "max_2d_patches": 32,
        "max_total_windows": 24,
        "min_total_windows": 12,
        "max_replicas": 24,
        "explicit_2d_window_schema": "generic",
        "region_memory": True,
        "region_memory_decay": 0.5,
        "region_probe_unvalidated": True,
        "region_prune_overscanned": True,
        "region_extend_undersampled": True,
        "adaptive_feedback_min_rounds": 3,
        "target_overlap": 0.30,
        "pilot_min_cv1_coverage": 0.80,
        "pilot_min_cv2_coverage": 0.80,
        "delaunay_iterate": True,
        "delaunay_stability_tol": 0.001,
        "delaunay_coverage_scaffold": True,
        "delaunay_coverage_grid_n": 5,
        "md_budget_ns": 20000.0,
        "ap_epochs": 4,
        "ap_final_pool_fraction": 0.50,
        "ap_min_final_pool_ns": 300.0,
        "ap_min_samples": 10000,
        "ap_min_samples_per_window": 200000,
        "ap_resume": True,
        "ap_write_reports": True,
        "ap_write_mbar_inputs": True,
        "ap_run_mbar": True,
        "ap_fes_bins": "40,80",
        "ap_seed_bank": True,
        "ap_seed_bank_max": 1,
        "explicit_2d_primary_cv_mode_column": "primary_cv_mode",
        "explicit_2d_primary_center_column": "primary_cv_center",
        "explicit_2d_primary_k_column": "primary_cv_k_kcal",
        "explicit_2d_secondary_cv_mode_column": "secondary_cv_mode",
        "explicit_2d_secondary_center_column": "secondary_cv_center",
        "explicit_2d_secondary_k_column": "secondary_cv_k_kcal_mol",
        "traj_interval": 150,
        "report_interval": 150,
        "distance_output_interval": 150,
        "parquet_flush_rows": 200000,
        "traj_format": "xtc",
        "traj_solute_only": True,
        "sample_potential_energy": False,
        "platform": "CUDA",
        "setup_cpu_threads": 8,
        "cpu_budget": 32,
        "max_cpu_per_replica": 6,
        "run_mode": "hmr-gamd",
        "timestep_fs": 4.0,
        "nvt_warmup_timestep_fs": 0.5,
        "npt_ramp_timestep_fs": 1.0,
        "production_steps": 10000000,
        "gamd_boost_type": "lower-dual-nonbonded-dihedral",
        "sigma0p": 3.0,
        "sigma0d": 3.0,
        "equil_steps": 1500000,
        "gamd_averaging_window": 5000,
        "exchange_mode": "gibbs-walk",
        "exchange_interval": 500,
        "tui_mode": "interactive",
        "genpept_prescan": True,
        "genpept_prescan_stages": [
            "candidate_seeds",
            "aa_implicit_minimized_pdbs",
            "basin_hop_minima",
            "nma_probe_seeds",
            "final_survivor_seeds",
        ],
        "genpept_prescan_frontier_stages": [
            "aa_implicit_minimized_pdbs",
            "basin_hop_minima",
            "final_survivor_seeds",
        ],
        "genpept_prescan_use_as_window_prior": True,
        "genpept_prescan_use_as_seed_library": True,
        # tica_* deliberately excluded: only 1/59 runs_rdy configs
        # (chignolin_tica_autoswitch.yaml) enables tICA. It stays an explicit
        # per-file override on top of this profile, not a bundle default —
        # baking it in here would silently enable it for every other peptide.
    },
}

# Source: RUNS/runs_rdy/chignolin/chignolin.yaml's genpept: block.
# Excludes seq/out, which are per-peptide.
GENPEPT_PROFILES: Dict[str, Dict[str, object]] = {
    "chignolin_tica": {
        # Not present in the genpept: block itself; every runs_rdy file relied
        # on _apply_gareus_friendly_hints inheriting this from the GAREUS-side
        # platform: block. Kept explicit here since the profile collapse
        # removes that literal block.
        "platform": "CUDA",
        "clean": False,
        "two_stage": True,
        "implicit_only": True,
        "n": 5000000,
        "n_candidate_seeds": 5000,
        "n_final_seeds": 2500,
        "jobs": 16,
        "min_jobs": 1,
        "generation_backend": "fast",
        "rama_sampling": "stratified",
        "initial_diversity_features": "mixed",
        "diversity_bank_preset": "broad",
        "preselection_bin_quota": 3,
        "preselection_rg_bin_A": 1.0,
        "preselection_e2e_bin_A": 1.0,
        "preselection_contact_bin": 2,
        "preselection_include_bank": True,
        "adaptive_min": True,
        "implicit_max_iterations": 300,
        "adaptive_min_chunk_iterations": 25,
        "adaptive_min_energy_tol": 0.05,
        "adaptive_min_force_tol": 500.0,
        "adaptive_min_stall_rounds": 2,
        "basin_hop": True,
        "bh_parent_seeds": 120,
        "bh_steps": 3,
        "bh_md_steps": 350,
        "bh_temperature": 675.0,
        "bh_friction": 5.0,
        "bh_timestep": 0.0035,
        "bh_hmr_mass": 3.014,
        "bh_initial_min_iterations": 200,
        "bh_min_iterations": 150,
        "bh_restart_from_parent": True,
        "nma_expand": True,
        "nma_modes": 2,
        "nma_amplitudes": "0.75,1.35",
        "nma_cutoff": 12.0,
        "explore_loop": True,
        "explore_rounds": 5,
        "explore_proposals": 50000,
        "explore_keep": 250,
        "explore_bins": 45,
        "explore_target_max_count": 0,
        "explore_pca_padding": 0.15,
        "explore_frontier_only": True,
        "explore_allow_outside": True,
        "explore_max_kept_per_bin": 4,
        "explore_bh": False,
    },
}


def expand_profile(table: Dict[str, Dict[str, object]], name: str, *, tool: str) -> Dict[str, object]:
    """Return a defensive copy of the named profile's flat key/value bundle.

    Raises KeyError if `name` is not a known profile for `tool`.
    """
    if name not in table:
        raise KeyError(
            f"Unknown profile {name!r} for tool {tool!r}. Available profiles: {sorted(table)}"
        )
    return copy.deepcopy(table[name])


def merge_profile(
    table: Dict[str, Dict[str, object]],
    explicit: Dict[str, object],
    profile_name: Optional[str],
    *,
    tool: str,
) -> Dict[str, object]:
    """Return `explicit` layered on top of `profile_name`'s bundle.

    Explicit keys always win. Raises ValueError (not KeyError) for an
    unknown profile name, since callers treat this as a config error.
    """
    if not profile_name:
        return dict(explicit)
    try:
        bundle = expand_profile(table, profile_name, tool=tool)
    except KeyError as exc:
        raise ValueError(str(exc)) from None
    return {**bundle, **explicit}


class _DuplicateYamlKeyError(Exception):
    def __init__(self, key: object) -> None:
        self.key = key
        super().__init__(str(key))


def _no_duplicates_constructor(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise _DuplicateYamlKeyError(key)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


class _DuplicateKeyCheckingLoader(yaml.SafeLoader):
    pass


_DuplicateKeyCheckingLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _no_duplicates_constructor,
)


def assert_no_duplicate_keys(text: str, source: object) -> None:
    """Raise ValueError if any YAML mapping in `text` repeats a key.

    PyYAML's default loader silently keeps the last occurrence of a
    duplicate key with no warning; this turns that into a clear, fail-fast
    error naming the file and the offending key.
    """
    try:
        yaml.load(text, Loader=_DuplicateKeyCheckingLoader)
    except _DuplicateYamlKeyError as exc:
        raise ValueError(
            f"Duplicate YAML key {exc.key!r} in {source}. "
            "PyYAML silently keeps the last occurrence; remove the duplicate."
        ) from None
