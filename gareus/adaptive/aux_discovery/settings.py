"""Frozen settings of the adaptive aux-CV discovery hook.

Values are c10's pre-registered rules (docs/_local_docs/c10_aux_diagnosis/prereg_v2.json) and c10
placement.py constants. Rulings on c10 discrepancies: lineage bootstraps keep multiplicity; k in 2..8
for both partitions; z3 scale = training-only sd; placement ported exactly (4-dp rounding, point gates).
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

AUX_SETTINGS_FILENAME = "aux_settings.json"
AUX_SETTINGS_SCHEMA = "atlas-aux-discovery-settings-v1"


@dataclass(frozen=True)
class AuxDiscoverySettings:
    # partitions (prereg_v2)
    k_min: int = 2
    k_max: int = 8
    ari_min: float = 0.5
    n_boot_partition: int = 10
    group_min_share: float = 0.01
    knn_k: int = 100
    pca_var: float = 0.8
    pca_max: int = 12
    sd_drop: float = 1e-3
    sd_floor: float = 0.05
    s_bins: int = 6
    hidden_min: float = 0.5
    cooc_share: float = 0.10
    cooc_bin_min: float = 0.02
    lineage_info_min: float = 0.10
    lineage_dirichlet_c: float = 5.0
    # z3 search
    l1_c_grid: Tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3)
    n_null_z3: int = 20
    max_cv_corr: float = 0.7
    # placement (c10 placement.py)
    quantiles: Tuple[float, ...] = (0.05, 0.10, 0.90, 0.95)
    width_fractions: Tuple[float, ...] = (0.35, 0.5, 0.7)
    n_boot_place: int = 100
    n_null: int = 50
    n_null_boot: int = 10
    min_train_frames: int = 100
    gate_o_q05: float = 0.20
    gate_ess_frames: float = 50.0
    gate_eff_lineages: float = 20.0
    gate_top3_share: float = 0.5
    heldout_min_frames: int = 50
    heldout_o_min: float = 0.15
    max_workers: int = 4
    placement_seed: int = 20261007
    # data
    frame_stride_steps: int = 0          # 0 = one exchange interval
    max_frames: int = 400_000
    partition_seed: int = 0
    temperature_k: float = 300.0

    def to_mapping(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "AuxDiscoverySettings":
        names = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - set(names))
        if unknown:
            raise ValueError(f"unknown aux settings key(s): {unknown}")
        kw = {}
        for k, v in raw.items():
            kw[k] = tuple(float(x) for x in v) if isinstance(v, list) else v
        return cls(**kw)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def resolve_aux_settings(adaptive_dir: Path, settings: AuxDiscoverySettings, *,
                         override: bool) -> Tuple[AuxDiscoverySettings, Dict[str, Any]]:
    """Freeze settings at first use; a recorded file wins unless ``override``."""
    path = Path(adaptive_dir) / AUX_SETTINGS_FILENAME
    if path.exists() and not override:
        rec = json.loads(path.read_text())
        if rec.get("schema") != AUX_SETTINGS_SCHEMA:
            raise ValueError(f"{path}: schema {rec.get('schema')!r} is not {AUX_SETTINGS_SCHEMA}")
        return AuxDiscoverySettings.from_mapping(rec["settings"]), rec
    rec = {"schema": AUX_SETTINGS_SCHEMA, "settings": settings.to_mapping(),
           "written_unix": time.time(), "override": bool(override)}
    _atomic_write_json(path, rec)
    return settings, rec


def aux_discovery_incompatibilities(args, *, topups: bool = False) -> List[str]:
    """Options aux discovery cannot run with (spec Section 11); [] = compatible. One source for the parse-time
    refusal (``gareus.cli``) and the driver-start re-check against a campaign's frozen policy (fix wave I4).
    ``topups``: the effective top-up switch (frozen policy or this job's flag)."""
    bad = []
    window_mode = str(getattr(args, "window_mode", "") or "")
    if window_mode and window_mode != "adaptive-production":
        bad.append("--ap-aux-discovery needs --window-mode adaptive-production")
    if topups or getattr(args, "ap_topups", False):
        bad.append("--ap-aux-discovery cannot run with --ap-topups (out of scope, spec Section 11)")
    if str(getattr(args, "exchange_mode", "") or "") == "neighbor":
        bad.append("--ap-aux-discovery needs an unrestricted exchange mode (gibbs-walk, all-pair-sweep, random-pair)")
    if getattr(args, "us_auto_drop_bad_windows", False):
        bad.append("--ap-aux-discovery refuses --us-auto-drop-bad-windows (aux populations are frozen)")
    run_mode = str(getattr(args, "run_mode", "gamd") or "gamd")
    boost = str(getattr(args, "gamd_boost_type", "") or "")
    if run_mode in ("gamd", "hmr-gamd") and not boost.startswith("pep-gamd"):
        bad.append("--ap-aux-discovery needs a pep-gamd-* boost type")
    traj = int(getattr(args, "traj_interval", 0) or 0)
    dist = int(getattr(args, "distance_output_interval", 0) or 0)
    exch = int(getattr(args, "exchange_interval", 0) or 0)
    if traj <= 0 or traj != dist:
        bad.append("--ap-aux-discovery needs traj_interval == distance_output_interval > 0 (z backfill needs "
                   "one trajectory frame per sample)")
    elif exch <= 0 or exch % traj:
        bad.append("--ap-aux-discovery needs exchange_interval to be a multiple of traj_interval")
    slots = getattr(args, "ap_aux_reserve_slots", None)
    if slots is None:
        slots = getattr(args, "adaptive_production_aux_reserve_slots", 4)
    if int(slots) < 1:
        bad.append("--ap-aux-discovery needs --ap-aux-reserve-slots >= 1")
    return bad
