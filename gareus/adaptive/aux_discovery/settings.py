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
from typing import Any, Dict, Tuple

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
    info_gain_min: float = 0.10
    stability_min: float = 0.8
    max_cv_corr: float = 0.7
    basin_gain_min: float = 0.05
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
