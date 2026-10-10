"""Frozen settings of the adaptive aux-CV discovery hook.

Values are c10's pre-registered rules (docs/_local_docs/c10_aux_diagnosis/prereg_v2.json) and c10
placement.py constants. Rulings on c10 discrepancies: lineage bootstraps keep multiplicity; k in 2..8
for both partitions; z3 scale = training-only sd; placement ported exactly (4-dp rounding, point gates).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

AUX_SETTINGS_FILENAME = "aux_settings.json"
AUX_SETTINGS_SCHEMA = "atlas-aux-discovery-settings-v1"
AUX_CAMPAIGN_OPTIONS_FILENAME = "aux_campaign_options.json"
AUX_CAMPAIGN_OPTIONS_SCHEMA = "atlas-aux-campaign-options-v1"
CONTINUE_STATES_REQUIRED = ("--ap-aux-discovery needs --ap-continue-states: without it every phase re-pulls every "
                            "worker, so the per-carrier worker burn-in drops every worker row and most carriers' "
                            "ordinary rows in every phase")
FEATURE_SPACES = ("backbone", "sidechain", "mixed", "auto")


def _feature_space(value: Any) -> str:
    if not isinstance(value, str) or value not in FEATURE_SPACES:
        raise ValueError(f"aux setting 'feature_space': expected one of {FEATURE_SPACES}, got {value!r}")
    return value


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
    k3_max_unvalidated: float = 10.0     # k3 cap when --ap-aux-validation off (no aux_validation.json k3_max_validated)
    placement_seed: int = 20261007
    # data
    frame_stride_steps: int = 0          # 0 = one exchange interval
    max_frames: int = 400_000
    partition_seed: int = 0
    temperature_k: float = 300.0
    feature_space: str = "backbone"

    def __post_init__(self) -> None:
        _feature_space(self.feature_space)

    def to_mapping(self) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        if self.feature_space == "backbone":
            out.pop("feature_space")
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "AuxDiscoverySettings":
        names = {f.name: f for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - set(names))
        if unknown:
            raise ValueError(f"unknown aux settings key(s): {unknown}")
        kw = {}
        for k, v in raw.items():
            kw[k] = _typed_setting(k, v)
        return cls(**kw)


_NONNEG_ZERO_OK = {"frame_stride_steps", "partition_seed", "placement_seed"}
_FRACTIONS = {"ari_min", "group_min_share", "pca_var", "hidden_min", "cooc_share", "cooc_bin_min",
              "lineage_info_min", "gate_o_q05", "heldout_o_min", "gate_top3_share"}


def _real(name: str, x: Any) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"aux setting {name!r}: expected a finite number, got {x!r}")
    return float(x)


def _typed_setting(name: str, v: Any) -> Any:
    """Type and range check of one frozen aux setting (F03): finite, no booleans, positive counts/sizes."""
    default = _DEFAULTS[name]
    if name == "feature_space":
        return _feature_space(v)
    if isinstance(default, tuple):
        if not isinstance(v, (list, tuple)) or not v:
            raise ValueError(f"aux setting {name!r}: expected a non-empty list, got {v!r}")
        out = tuple(_real(name, x) for x in v)
        if any(x <= 0 for x in out):
            raise ValueError(f"aux setting {name!r}: entries must be > 0, got {v!r}")
        if name in ("quantiles", "width_fractions") and any(x >= 1 for x in out):
            raise ValueError(f"aux setting {name!r}: entries must be < 1, got {v!r}")
        return out
    if isinstance(default, int):
        if isinstance(v, bool) or not isinstance(v, int):
            if isinstance(v, float) and math.isfinite(v) and v == int(v):
                v = int(v)
            else:
                raise ValueError(f"aux setting {name!r}: expected an integer, got {v!r}")
        if v < 0 or (v == 0 and name not in _NONNEG_ZERO_OK):
            raise ValueError(f"aux setting {name!r}: must be > 0, got {v!r}")
        return v
    x = _real(name, v)
    if x < 0 or x == 0 or (name in _FRACTIONS and x > 1):
        raise ValueError(f"aux setting {name!r}: out of range, got {v!r}")
    return x


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def resolve_aux_settings(adaptive_dir: Path, settings: AuxDiscoverySettings, *,
                         override: bool) -> Tuple[AuxDiscoverySettings, Dict[str, Any]]:
    """Freeze settings at first use; ``override`` can replace settings but cannot change the feature space."""
    require_aux_feature_space(adaptive_dir, settings.feature_space)
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


def aux_validation_mode(args) -> str:
    """``--ap-aux-validation`` of this job: ``required`` (default) or ``off``."""
    mode = str(getattr(args, "ap_aux_validation", None) or "required")
    if mode not in ("required", "off"):
        raise ValueError(f"--ap-aux-validation must be required or off, got {mode!r}")
    return mode


def aux_feature_space(args) -> str:
    return _feature_space(getattr(args, "ap_aux_feature_space", "backbone"))


def require_aux_feature_space(adaptive_dir: Path, feature_space: str) -> None:
    """Refuse changing the feature space in either campaign record, including legacy backbone records."""
    current = _feature_space(feature_space)
    for filename, schema, field in ((AUX_CAMPAIGN_OPTIONS_FILENAME, AUX_CAMPAIGN_OPTIONS_SCHEMA, "options"),
                                    (AUX_SETTINGS_FILENAME, AUX_SETTINGS_SCHEMA, "settings")):
        path = Path(adaptive_dir) / filename
        if not path.exists():
            continue
        try:
            rec = json.loads(path.read_text())
            if rec.get("schema") != schema or not isinstance(rec.get(field), dict):
                raise ValueError(f"schema {rec.get('schema')!r} / {field} {rec.get(field)!r}")
            recorded = _feature_space(rec[field].get("feature_space", "backbone"))
        except Exception as exc:
            raise RuntimeError(f"{path} is not a valid {schema} record ({exc})") from exc
        if recorded != current:
            raise RuntimeError(f"{path}: this aux campaign records feature_space={recorded}, this job has "
                               f"feature_space={current}; --ap-aux-feature-space is frozen per campaign")


def resolve_aux_campaign_options(adaptive_dir: Path, args) -> Dict[str, Any]:
    """Freeze continue-states, validation mode and feature space in the aux-specific record
    ``aux_campaign_options.json`` at the campaign's first aux job; every later job must match it. A mismatch or an
    unreadable record raises (never adopted: a recorded False would re-pull every worker). Kept out of
    ``DECISION_SETTINGS_FIELDS`` so a non-aux campaign's decision_settings.json is unchanged."""
    path = Path(adaptive_dir) / AUX_CAMPAIGN_OPTIONS_FILENAME
    current = {"continue_states": bool(getattr(args, "ap_continue_states", False))}
    feature_space = aux_feature_space(args)
    require_aux_feature_space(adaptive_dir, feature_space)
    if feature_space != "backbone":
        current["feature_space"] = feature_space
    mode = aux_validation_mode(args)
    if mode != "required":                   # key written only when off: a required campaign's record is unchanged
        current["aux_validation"] = mode
    if not path.exists():
        rec = {"schema": AUX_CAMPAIGN_OPTIONS_SCHEMA, "options": current, "written_unix": time.time()}
        _atomic_write_json(path, rec)
        return rec
    try:
        rec = json.loads(path.read_text())
        recorded = rec["options"] if rec.get("schema") == AUX_CAMPAIGN_OPTIONS_SCHEMA else None
        if not isinstance(recorded, dict) or not isinstance(recorded.get("continue_states"), bool):
            raise ValueError(f"schema {rec.get('schema')!r} / options {recorded!r}")
    except Exception as exc:
        raise RuntimeError(f"{path} is not a valid {AUX_CAMPAIGN_OPTIONS_SCHEMA} record ({exc})") from exc
    # Records written before the switch carry no key: they were validated, i.e. "required".
    rec_validation = recorded.get("aux_validation", "required")
    if rec_validation != current.get("aux_validation", "required"):
        raise RuntimeError(f"{path}: this aux campaign records aux_validation={rec_validation}, this job has "
                           f"aux_validation={current.get('aux_validation', 'required')}; --ap-aux-validation is frozen per campaign")
    if not recorded["continue_states"] or recorded["continue_states"] != current["continue_states"]:
        raise RuntimeError(f"{path}: this aux campaign records continue_states={recorded['continue_states']}, this "
                           f"job has continue_states={current['continue_states']}; aux discovery runs only with "
                           "--ap-continue-states on every job (a re-pull per phase discards the workers' samples)")
    return rec


def aux_discovery_incompatibilities(args, *, topups: bool = False, reserve_slots=None) -> List[str]:
    """Options aux discovery cannot run with (spec Section 11); [] = compatible. One source for the parse-time
    refusal (``gareus.cli``) and the driver-start re-check against a campaign's frozen policy (fix wave I4).
    ``topups``: the effective top-up switch (frozen policy or this job's flag). ``reserve_slots``: the campaign's
    frozen ``aux_reserve_slots`` on a resume (overrides this job's flag)."""
    bad = []
    if not bool(getattr(args, "ap_continue_states", False)):
        bad.append(CONTINUE_STATES_REQUIRED)
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
    slots = reserve_slots if reserve_slots is not None else getattr(args, "ap_aux_reserve_slots", None)
    if slots is None:
        slots = getattr(args, "adaptive_production_aux_reserve_slots", 4)
    if int(slots) < 1:
        bad.append("--ap-aux-discovery needs --ap-aux-reserve-slots >= 1")
    return bad


_DEFAULTS = {f.name: getattr(AuxDiscoverySettings(), f.name) for f in dataclasses.fields(AuxDiscoverySettings)}
