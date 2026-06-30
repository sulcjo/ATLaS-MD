"""GENPEPT-derived round-zero window priors for adaptive feedback.

This module intentionally lives on the GAREUS side of the workflow.  It reads
GENPEPT output structures, rescoring them in the active GAREUS CV space, then
writes an explicit sparse 2D window table that the existing production path can
consume through ``--windows-2d-csv``.

The prior is a sampling preconditioner, not thermodynamic evidence.  It can
place first-round windows and starting-structure suggestions, but WHAM/MBAR
must still use only equilibrium samples from known production biases.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .cv import (
    adaptive_secondary_force_constants_kcal,
    choose_cv_atoms,
    prepare_primary_cv_definition,
    primary_cv_is_contacts,
    primary_cv_mode,
    primary_cv_units,
    primary_cv_value_from_positions_nm,
    rama_map_definitions,
    secondary_cv_enabled,
    secondary_cv_mode,
    secondary_cv_range,
    secondary_cv_target_angles,
    secondary_structure_score_from_positions_nm,
    secondary_structure_torsions,
)
from .io import _json_ready, write_json
from .seeding import (
    _relative_primary_cv_def_for_conformer,
    _relative_secondary_cv_metadata_for_conformer,
)
from .windows import adaptive_contact_force_constants_kcal, adaptive_force_constants_kcal_a2

__all__ = [
    "ScoredGenpeptPoint",
    "build_genpept_window_prior",
    "build_secondary_cv_metadata_for_prior",
    "genpept_prior_enabled",
    "select_prior_windows_from_points",
]


PDB_PATH_COLUMNS = (
    "survivor_pdb_path",
    "pdb_path",
    "output_pdb",
    "input_pdb",
    "candidate_pdb_path",
    "representative_pdb_path",
)

DEFAULT_PRIOR_STAGES = (
    "final_search_pool",
    "post_pca_explore",
    "post_nma",
    "post_basin_hop",
    "initial_implicit",
    "final_survivor_seeds",
)


@dataclass
class ScoredGenpeptPoint:
    """One GENPEPT structure rescored in active GAREUS CV space."""

    pdb_path: Path
    stage: str
    source_index: int
    primary_cv_value: float
    secondary_cv_value: float
    energy_kj_mol: float = float("nan")
    rg_nm: float = float("nan")
    end_to_end_nm: float = float("nan")
    contact_count: float = float("nan")
    source: str = ""
    seed_name: str = ""


def genpept_prior_enabled(args) -> bool:
    return bool(getattr(args, "genpept_prior_enabled", False))


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _positive_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except Exception:
        return int(default)
    return out if out > 0 else int(default)


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    text = os.path.expandvars(str(value)).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _read_csv_rows(path: Path) -> list[dict]:
    try:
        with Path(path).open(newline="") as handle:
            return [dict(r) for r in csv.DictReader(handle)]
    except Exception:
        return []


def _write_csv(path: Path, rows: list[dict], preferred: Optional[list[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if preferred is None:
        preferred = []
    fieldnames: list[str] = []
    for name in preferred:
        if name not in fieldnames:
            fieldnames.append(name)
    for row in rows:
        for name in row.keys():
            if name not in fieldnames:
                fieldnames.append(name)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _stage_csv_candidates(genpept_dir: Path, stage: str) -> list[Path]:
    stage = str(stage).strip()
    if not stage:
        return []
    p = Path(stage)
    if p.suffix.lower() == ".csv":
        return [p if p.is_absolute() else genpept_dir / p]
    if stage in {"final_survivor_seeds", "final_survivors", "survivors"}:
        return [genpept_dir / "final_survivor_seeds.csv"]
    out = [
        genpept_dir / "basin_archive" / f"{stage}_points.csv",
        genpept_dir / f"{stage}.csv",
    ]
    if not stage.endswith("_points"):
        out.append(genpept_dir / "basin_archive" / f"{stage}.csv")
    return out


def _resolve_pdb_path(genpept_dir: Path, raw_path: Any) -> Optional[Path]:
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    p = Path(os.path.expandvars(text)).expanduser()
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            Path.cwd() / p,
            genpept_dir / p,
            genpept_dir.parent / p,
            genpept_dir / p.name,
        ])
    for cand in candidates:
        try:
            if cand.exists() and cand.is_file():
                return cand.resolve()
        except Exception:
            continue
    return None


def _row_pdb_path(genpept_dir: Path, row: dict) -> Optional[Path]:
    for col in PDB_PATH_COLUMNS:
        if col in row and str(row.get(col, "")).strip():
            path = _resolve_pdb_path(genpept_dir, row.get(col))
            if path is not None:
                return path
    return None


def _collect_genpept_rows(genpept_dir: Path, stages: Iterable[str]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []
    seen_inputs: set[tuple[str, str, str]] = set()
    for stage in stages:
        stage = str(stage).strip()
        if not stage:
            continue
        stage_rows: list[dict] = []
        for csv_path in _stage_csv_candidates(genpept_dir, stage):
            if csv_path.exists():
                stage_rows = _read_csv_rows(csv_path)
                if stage_rows:
                    break
        if stage_rows:
            for i, row in enumerate(stage_rows):
                p = _row_pdb_path(genpept_dir, row)
                if p is None:
                    continue
                key = (stage, str(p), str(row.get("seed_name", "")))
                if key in seen_inputs:
                    continue
                seen_inputs.add(key)
                out = dict(row)
                out["_genpept_prior_stage"] = stage
                out["_genpept_prior_source_index"] = i
                out["_genpept_prior_pdb_path"] = str(p)
                rows.append(out)
            continue

        stage_dir = genpept_dir / stage
        if stage_dir.exists() and stage_dir.is_dir():
            pdbs = sorted(stage_dir.rglob("*.pdb"))
            for i, p in enumerate(pdbs):
                key = (stage, str(p), "")
                if key in seen_inputs:
                    continue
                seen_inputs.add(key)
                rows.append({
                    "_genpept_prior_stage": stage,
                    "_genpept_prior_source_index": i,
                    "_genpept_prior_pdb_path": str(p.resolve()),
                    "pdb_path": str(p),
                    "seed_name": p.stem,
                })
        else:
            warnings.append(f"GENPEPT prior stage {stage!r} had no readable CSV or PDB directory")
    return rows, warnings


def _thin_rows_by_stage(rows: list[dict], max_rows: int) -> list[dict]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    by_stage: dict[str, list[dict]] = {}
    for row in rows:
        by_stage.setdefault(str(row.get("_genpept_prior_stage", "")), []).append(row)
    stages = sorted(by_stage)
    base_quota = max(1, max_rows // max(1, len(stages)))
    selected: list[dict] = []
    leftovers: list[dict] = []
    for stage in stages:
        group = by_stage[stage]
        if len(group) <= base_quota:
            selected.extend(group)
        else:
            idxs = np.linspace(0, len(group) - 1, base_quota)
            chosen = {int(round(float(i))) for i in idxs}
            selected.extend(group[i] for i in sorted(chosen))
            leftovers.extend(group[i] for i in range(len(group)) if i not in chosen)
    if len(selected) < max_rows and leftovers:
        need = max_rows - len(selected)
        idxs = np.linspace(0, len(leftovers) - 1, min(need, len(leftovers)))
        selected.extend(leftovers[int(round(float(i)))] for i in idxs)
    return selected[:max_rows]


def _read_pdb_positions_nm(path: Path) -> np.ndarray:
    coords = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            coords.append((0.1 * x, 0.1 * y, 0.1 * z))
    return np.asarray(coords, dtype=float)


def _row_source(row: dict, stage: str) -> str:
    for col in ("source", "origin", "stage_source", "mode"):
        val = str(row.get(col, "")).strip()
        if val:
            low = val.lower()
            if "bh" in low or "basin" in low:
                return "BH"
            if "nma" in low or "anm" in low:
                return "NMA"
            if "pca" in low or "frontier" in low or "adaptive" in low:
                return "PCA"
            if "implicit" in low or "initial" in low:
                return "initial"
            return val
    low_stage = str(stage).lower()
    if "pca" in low_stage:
        return "PCA"
    if "nma" in low_stage:
        return "NMA"
    if "basin" in low_stage or "hop" in low_stage:
        return "BH"
    if "initial" in low_stage or "implicit" in low_stage:
        return "initial"
    return "GENPEPT"


def _score_genpept_rows(
    rows: list[dict],
    genpept_dir: Path,
    args,
    topology,
    primary_cv_def: dict,
    secondary_cv_metadata: dict,
) -> tuple[list[ScoredGenpeptPoint], list[str]]:
    warnings: list[str] = []
    rel_primary = _relative_primary_cv_def_for_conformer(primary_cv_def, topology)
    if rel_primary is None:
        warnings.append("could not map active primary CV to peptide-only GENPEPT atom indices")
        return [], warnings
    rel_secondary = _relative_secondary_cv_metadata_for_conformer(secondary_cv_metadata, topology)
    if secondary_cv_enabled(args) and rel_secondary is None:
        warnings.append("could not map active secondary CV to peptide-only GENPEPT atom indices")
        return [], warnings

    points: list[ScoredGenpeptPoint] = []
    skipped = 0
    seen_paths: set[str] = set()
    for row in rows:
        path = _resolve_pdb_path(genpept_dir, row.get("_genpept_prior_pdb_path"))
        if path is None:
            skipped += 1
            continue
        # Repeated structures across archive stages are useful stage-support
        # evidence only once per stage.  Exact repeated path within one stage is
        # not useful for window placement.
        stage = str(row.get("_genpept_prior_stage", "GENPEPT"))
        dedup_key = f"{stage}:{path}"
        if dedup_key in seen_paths:
            continue
        seen_paths.add(dedup_key)
        try:
            pos_nm = _read_pdb_positions_nm(path)
            if pos_nm.size == 0:
                skipped += 1
                continue
            primary_value = primary_cv_value_from_positions_nm(pos_nm, rel_primary, args)
            secondary_value = secondary_structure_score_from_positions_nm(pos_nm, rel_secondary) if rel_secondary else float("nan")
        except Exception:
            skipped += 1
            continue
        if not (math.isfinite(float(primary_value)) and math.isfinite(float(secondary_value))):
            skipped += 1
            continue
        energy = _float_or_nan(row.get("energy_kj_mol", row.get("minimized_energy_kj_mol", row.get("best_energy_kj_mol", float("nan")))))
        points.append(ScoredGenpeptPoint(
            pdb_path=path,
            stage=stage,
            source_index=int(row.get("_genpept_prior_source_index", len(points)) or 0),
            primary_cv_value=float(primary_value),
            secondary_cv_value=float(secondary_value),
            energy_kj_mol=energy,
            rg_nm=_float_or_nan(row.get("rg_nm")),
            end_to_end_nm=_float_or_nan(row.get("end_to_end_nm")),
            contact_count=_float_or_nan(row.get("contact_count")),
            source=_row_source(row, stage),
            seed_name=str(row.get("seed_name", path.stem)),
        ))
    if skipped:
        warnings.append(f"skipped {skipped} GENPEPT rows that could not be resolved or scored")
    return points, warnings


def build_secondary_cv_metadata_for_prior(args, topology) -> dict:
    """Build secondary-CV scoring metadata without constructing OpenMM forces."""
    if not secondary_cv_enabled(args):
        return {"enabled": False, "mode": "none"}
    phi_torsions, psi_torsions = secondary_structure_torsions(topology)
    if not phi_torsions and not psi_torsions:
        return {"enabled": False, "mode": secondary_cv_mode(args), "reason": "no peptide phi/psi torsions"}
    mode = secondary_cv_mode(args)
    sigma_deg = float(getattr(args, "secondary_cv_sigma_deg", 35.0) or 35.0)
    range_min, range_max = secondary_cv_range(args)
    base = {
        "enabled": True,
        "mode": mode,
        "range_min": float(range_min),
        "range_max": float(range_max),
        "sigma_deg": float(sigma_deg),
        "n_phi_torsions": int(len(phi_torsions)),
        "n_psi_torsions": int(len(psi_torsions)),
        "phi_torsions": [list(map(int, t)) for t in phi_torsions],
        "psi_torsions": [list(map(int, t)) for t in psi_torsions],
    }
    if mode == "alpha-coil-beta":
        base.update({
            "label": "alpha-coil-beta signed transition coordinate (+alpha, 0 coil, -beta)",
            "coil_center": 0.0,
            "alpha_phi0_deg": -60.0,
            "alpha_psi0_deg": -45.0,
            "beta_phi0_deg": -135.0,
            "beta_psi0_deg": 135.0,
        })
        return base
    if mode == "rama-map":
        regions = rama_map_definitions()
        base.update({
            "label": "explicit Ramachandran basin map",
            "regions": [dict(r) for r in regions],
            "region_values": {str(r["name"]): float(r["value"]) for r in regions},
        })
        return base
    phi0, psi0, label = secondary_cv_target_angles(args)
    base.update({
        "label": label,
        "phi0_deg": math.degrees(float(phi0)),
        "psi0_deg": math.degrees(float(psi0)),
    })
    return base


def _parse_bins(value: Any) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        text = str(value or "24,12").replace("x", ",")
        parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 1:
        n = _positive_int(parts[0], 24)
        return n, n
    if len(parts) >= 2:
        return _positive_int(parts[0], 24), _positive_int(parts[1], 12)
    return 24, 12


def _finite_scale(values: Iterable[float], fallback: float = 1.0) -> float:
    vals = sorted({round(float(v), 10) for v in values if math.isfinite(float(v))})
    if len(vals) <= 1:
        return max(1.0e-12, float(fallback))
    diffs = np.diff(np.asarray(vals, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1.0e-12)]
    if diffs.size:
        return max(1.0e-12, float(np.nanmedian(diffs)))
    return max(1.0e-12, float(vals[-1] - vals[0]) / max(1, len(vals) - 1), float(fallback))


def _range_with_padding(values: np.ndarray, *, min_span: float, lower: Optional[float] = None, upper: Optional[float] = None) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        lo, hi = 0.0, float(min_span)
    else:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    span = max(float(min_span), hi - lo)
    center = 0.5 * (lo + hi)
    lo = center - 0.55 * span
    hi = center + 0.55 * span
    if lower is not None:
        lo = max(float(lower), lo)
    if upper is not None:
        hi = min(float(upper), hi)
    if hi <= lo:
        hi = lo + max(1.0e-6, float(min_span))
        if upper is not None and hi > upper:
            hi = float(upper)
            lo = hi - max(1.0e-6, float(min_span))
    return float(lo), float(hi)


def _category_budgets(args, max_windows: int) -> dict[str, int]:
    explicit = {
        "basin": int(getattr(args, "genpept_prior_basin_windows", 0) or 0),
        "bridge": int(getattr(args, "genpept_prior_bridge_windows", 0) or 0),
        "frontier": int(getattr(args, "genpept_prior_frontier_windows", 0) or 0),
        "probe": int(getattr(args, "genpept_prior_probe_windows", 0) or 0),
    }
    if sum(explicit.values()) > 0:
        total = sum(max(0, v) for v in explicit.values())
        if total > max_windows:
            scale = float(max_windows) / float(total)
            scaled = {k: int(math.floor(max(0, v) * scale)) for k, v in explicit.items()}
            while sum(scaled.values()) < max_windows:
                key = max(explicit, key=lambda k: explicit[k] - scaled[k])
                scaled[key] += 1
            return scaled
        explicit["basin"] += max(0, max_windows - total)
        return explicit
    basin = max(1, int(round(0.30 * max_windows)))
    bridge = max(0, int(round(0.35 * max_windows)))
    frontier = max(0, int(round(0.25 * max_windows)))
    probe = max(0, max_windows - basin - bridge - frontier)
    return {"basin": basin, "bridge": bridge, "frontier": frontier, "probe": probe}


def _nearest_point(points: list[ScoredGenpeptPoint], primary: float, secondary: float, p_scale: float, s_scale: float) -> Optional[ScoredGenpeptPoint]:
    if not points:
        return None
    best = None
    best_score = float("inf")
    for point in points:
        dp = (float(point.primary_cv_value) - float(primary)) / max(1.0e-12, p_scale)
        ds = (float(point.secondary_cv_value) - float(secondary)) / max(1.0e-12, s_scale)
        e = float(point.energy_kj_mol)
        e_term = 0.0 if not math.isfinite(e) else 1.0e-8 * e
        score = dp * dp + ds * ds + e_term
        if score < best_score:
            best_score = score
            best = point
    return best


def _assign_force_constants(rows: list[dict], args) -> None:
    primary_vals = sorted({round(float(r["primary_cv_center"]), 8) for r in rows})
    if primary_cv_is_contacts(args):
        k_vals = adaptive_contact_force_constants_kcal(np.asarray(primary_vals, dtype=float), args)
    else:
        k_vals = adaptive_force_constants_kcal_a2(np.asarray(primary_vals, dtype=float), args)
    p_k = {round(float(c), 8): float(k) for c, k in zip(primary_vals, k_vals)}

    secondary_vals = sorted({round(float(r["secondary_cv_center"]), 8) for r in rows})
    sec_k_vals = adaptive_secondary_force_constants_kcal(secondary_vals, args)
    s_k = {round(float(c), 8): float(k) for c, k in zip(secondary_vals, sec_k_vals)}

    for row in rows:
        pk = p_k.get(round(float(row["primary_cv_center"]), 8), float(getattr(args, "default_window_k_kcal_a2", 1.0) or 1.0))
        sk = s_k.get(round(float(row["secondary_cv_center"]), 8), float(getattr(args, "secondary_cv_k_kcal", 50.0) or 50.0))
        row["primary_cv_k_kcal"] = float(pk)
        row["distance_k_kcal_mol_A2"] = float(pk)
        row["secondary_cv_k_kcal_mol"] = float(sk)


def select_prior_windows_from_points(
    points: list[ScoredGenpeptPoint],
    args,
    *,
    primary_mode_name: Optional[str] = None,
    secondary_mode_name: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """Select basin/bridge/frontier/probe windows from rescored GENPEPT points."""
    finite = [
        p for p in points
        if math.isfinite(float(p.primary_cv_value)) and math.isfinite(float(p.secondary_cv_value))
    ]
    if not finite:
        return [], {"reason": "no finite rescored GENPEPT points"}

    primary_mode_name = str(primary_mode_name or primary_cv_mode(args))
    secondary_mode_name = str(secondary_mode_name or secondary_cv_mode(args))
    n_x, n_y = _parse_bins(getattr(args, "genpept_prior_bins", "24,12"))
    min_hits = max(1, int(getattr(args, "genpept_prior_min_hits_per_bin", 2) or 2))
    max_windows = max(1, int(getattr(args, "genpept_prior_max_windows", 48) or 48))
    budgets = _category_budgets(args, max_windows)

    primary_values = np.asarray([p.primary_cv_value for p in finite], dtype=float)
    if primary_cv_is_contacts(args) and bool(getattr(args, "contact_normalize", True)):
        p_lo, p_hi = _range_with_padding(primary_values, min_span=max(0.05, float(getattr(args, "contact_adaptive_target_spacing", 0.15) or 0.15) * 3.0), lower=0.0, upper=1.0)
    else:
        p_lo, p_hi = _range_with_padding(primary_values, min_span=1.0)
    sec_range_lo, sec_range_hi = secondary_cv_range(args)
    secondary_centers = [
        float(x) for x in (getattr(args, "secondary_cv_centers", None) or [])
        if math.isfinite(float(x))
    ]
    snap_secondary = bool(getattr(args, "genpept_prior_snap_secondary_centers", True)) and bool(secondary_centers)
    secondary_centers = sorted({round(float(x), 8) for x in secondary_centers})
    if snap_secondary:
        n_y = max(1, len(secondary_centers))
        s_lo, s_hi = float(min(secondary_centers)), float(max(secondary_centers))
    else:
        s_lo, s_hi = float(sec_range_lo), float(sec_range_hi)

    dx = max(1.0e-12, (p_hi - p_lo) / float(max(1, n_x)))
    dy = max(1.0e-12, (s_hi - s_lo) / float(max(1, n_y)))
    p_scale = _finite_scale([p.primary_cv_value for p in finite], fallback=dx)
    s_scale = _finite_scale([p.secondary_cv_value for p in finite], fallback=dy)

    cells: dict[tuple[int, int], dict] = {}
    for idx, point in enumerate(finite):
        xb = int(math.floor((float(point.primary_cv_value) - p_lo) / dx))
        xb = max(0, min(n_x - 1, xb))
        if snap_secondary:
            yb = min(range(len(secondary_centers)), key=lambda j: abs(float(point.secondary_cv_value) - float(secondary_centers[j])))
            sec_center = float(secondary_centers[yb])
        else:
            yb = int(math.floor((float(point.secondary_cv_value) - s_lo) / dy))
            yb = max(0, min(n_y - 1, yb))
            sec_center = float(s_lo + (yb + 0.5) * dy)
        key = (xb, yb)
        cell = cells.setdefault(key, {
            "x_bin": xb,
            "y_bin": yb,
            "primary_center": float(p_lo + (xb + 0.5) * dx),
            "secondary_center": sec_center,
            "point_indices": [],
            "stages": set(),
            "sources": set(),
            "best_energy": float("nan"),
            "best_point_index": idx,
        })
        cell["point_indices"].append(idx)
        cell["stages"].add(str(point.stage))
        cell["sources"].add(str(point.source))
        e = float(point.energy_kj_mol)
        best_e = float(cell.get("best_energy", float("nan")))
        if math.isfinite(e) and (not math.isfinite(best_e) or e < best_e):
            cell["best_energy"] = e
            cell["best_point_index"] = idx

    node_rows: list[dict] = []
    key_to_node: dict[tuple[int, int], int] = {}
    for node_id, key in enumerate(sorted(cells)):
        cell = cells[key]
        idxs = list(cell["point_indices"])
        key_to_node[key] = node_id
        counts = len(idxs)
        node_rows.append({
            "node": int(node_id),
            "x_bin": int(cell["x_bin"]),
            "y_bin": int(cell["y_bin"]),
            "primary_cv_center": float(cell["primary_center"]),
            "secondary_cv_center": float(cell["secondary_center"]),
            "count": int(counts),
            "stage_count": int(len(cell["stages"])),
            "source_count": int(len(cell["sources"])),
            "stages": ";".join(sorted(cell["stages"])),
            "sources": ";".join(sorted(cell["sources"])),
            "best_energy_kj_mol": float(cell["best_energy"]) if math.isfinite(float(cell["best_energy"])) else "",
        })

    edge_rows: list[dict] = []
    degrees = {key: 0 for key in cells}
    for key_a in sorted(cells):
        xa, ya = key_a
        for dx_i in (-1, 0, 1):
            for dy_i in (-1, 0, 1):
                if dx_i == 0 and dy_i == 0:
                    continue
                key_b = (xa + dx_i, ya + dy_i)
                if key_b not in cells or key_b <= key_a:
                    continue
                ca, cb = cells[key_a], cells[key_b]
                nd = math.sqrt(
                    ((float(ca["primary_center"]) - float(cb["primary_center"])) / max(1.0e-12, p_scale)) ** 2 +
                    ((float(ca["secondary_center"]) - float(cb["secondary_center"])) / max(1.0e-12, s_scale)) ** 2
                )
                degrees[key_a] += 1
                degrees[key_b] += 1
                edge_rows.append({
                    "node_i": int(key_to_node[key_a]),
                    "node_j": int(key_to_node[key_b]),
                    "edge_type": "occupied_neighbor",
                    "normalized_distance": float(nd),
                })

    components: list[list[tuple[int, int]]] = []
    unseen = set(cells)
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = {k: set() for k in cells}
    for row in edge_rows:
        ki = next(k for k, node in key_to_node.items() if node == int(row["node_i"]))
        kj = next(k for k, node in key_to_node.items() if node == int(row["node_j"]))
        adjacency[ki].add(kj)
        adjacency[kj].add(ki)
    while unseen:
        start = min(unseen)
        stack = [start]
        comp = []
        unseen.remove(start)
        while stack:
            key = stack.pop()
            comp.append(key)
            for nxt in sorted(adjacency.get(key, ())):
                if nxt in unseen:
                    unseen.remove(nxt)
                    stack.append(nxt)
        components.append(comp)

    selected: list[dict] = []
    selected_keys: set[tuple[float, float]] = set()

    def add_window(primary: float, secondary: float, window_type: str, priority: float, reason: str, point: Optional[ScoredGenpeptPoint], cell: Optional[dict] = None) -> bool:
        if len(selected) >= max_windows:
            return False
        if not (math.isfinite(float(primary)) and math.isfinite(float(secondary))):
            return False
        secondary = max(float(sec_range_lo), min(float(sec_range_hi), float(secondary)))
        if snap_secondary and secondary_centers:
            secondary = float(min(secondary_centers, key=lambda v: abs(float(v) - float(secondary))))
        key = (round(float(primary), 5), round(float(secondary), 5))
        if key in selected_keys:
            return False
        selected_keys.add(key)
        if point is None:
            point = _nearest_point(finite, primary, secondary, p_scale, s_scale)
        row = {
            "candidate_window": int(len(selected)),
            "window_type": str(window_type),
            "patch_lifecycle": "genpept_prior_round0",
            "primary_cv_mode": primary_mode_name,
            "primary_cv_center": float(primary),
            "primary_cv_k_kcal": 0.0,
            "distance_center_A": float(primary),
            "distance_k_kcal_mol_A2": 0.0,
            "secondary_cv_mode": secondary_mode_name,
            "secondary_cv_center": float(secondary),
            "secondary_cv_k_kcal_mol": 0.0,
            "priority": float(priority),
            "reason": str(reason),
            "used_by_adaptive_feedback_next_pilot_when_available": True,
            "used_by_adaptive_feedback_final_production": True,
            "requires_future_explicit_2d_window_support": False,
        }
        if cell is not None:
            row.update({
                "genpept_count": int(len(cell.get("point_indices", []))),
                "genpept_stage_count": int(len(cell.get("stages", []))),
                "genpept_source_count": int(len(cell.get("sources", []))),
                "genpept_x_bin": int(cell.get("x_bin", -1)),
                "genpept_y_bin": int(cell.get("y_bin", -1)),
            })
        if point is not None:
            row.update({
                "representative_pdb_path": str(point.pdb_path),
                "representative_stage": str(point.stage),
                "representative_source": str(point.source),
                "representative_seed_name": str(point.seed_name),
                "representative_primary_cv_value": float(point.primary_cv_value),
                "representative_secondary_cv_value": float(point.secondary_cv_value),
                "representative_energy_kj_mol": float(point.energy_kj_mol) if math.isfinite(float(point.energy_kj_mol)) else "",
            })
        selected.append(row)
        return True

    def cell_sort_tuple(cell: dict) -> tuple:
        best_e = float(cell.get("best_energy", float("nan")))
        energy_key = best_e if math.isfinite(best_e) else float("inf")
        return (-len(cell.get("stages", [])), -len(cell.get("point_indices", [])), energy_key)

    occupied_cells = [c for c in cells.values() if len(c.get("point_indices", [])) >= min_hits]
    basin_cells = sorted(occupied_cells, key=cell_sort_tuple)
    for cell in basin_cells[:budgets["basin"]]:
        idx = int(cell.get("best_point_index", 0))
        add_window(
            float(cell["primary_center"]), float(cell["secondary_center"]),
            "genpept_basin_anchor", 100.0 + len(cell["point_indices"]),
            "stable occupied GENPEPT basin/bin", finite[idx], cell,
        )

    bridge_candidates: list[tuple[float, float, float, str, Optional[dict]]] = []
    sparse_occupied = [
        c for key, c in cells.items()
        if len(c.get("point_indices", [])) >= min_hits and degrees.get(key, 0) <= 2
    ]
    for cell in sparse_occupied:
        bridge_candidates.append((
            float(cell["primary_center"]), float(cell["secondary_center"]),
            80.0 + max(0.0, 4.0 - float(degrees.get((cell["x_bin"], cell["y_bin"]), 0))),
            "sparse occupied bridge/frontier cell in GENPEPT topology",
            cell,
        ))
    if len(components) > 1:
        comp_pairs = []
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                best = None
                for ka in components[i]:
                    for kb in components[j]:
                        ca, cb = cells[ka], cells[kb]
                        nd = math.sqrt(
                            ((float(ca["primary_center"]) - float(cb["primary_center"])) / max(1.0e-12, p_scale)) ** 2 +
                            ((float(ca["secondary_center"]) - float(cb["secondary_center"])) / max(1.0e-12, s_scale)) ** 2
                        )
                        if best is None or nd < best[0]:
                            best = (nd, ca, cb)
                if best is not None:
                    comp_pairs.append(best)
        for nd, ca, cb in sorted(comp_pairs, key=lambda x: x[0]):
            bridge_candidates.append((
                0.5 * (float(ca["primary_center"]) + float(cb["primary_center"])),
                0.5 * (float(ca["secondary_center"]) + float(cb["secondary_center"])),
                90.0 - float(nd),
                "probe midpoint between disconnected GENPEPT occupied components",
                None,
            ))
    bridge_candidates = sorted(bridge_candidates, key=lambda x: -float(x[2]))
    for primary, secondary, priority, reason, cell in bridge_candidates[:budgets["bridge"]]:
        point = None
        if cell is not None:
            point = finite[int(cell.get("best_point_index", 0))]
        add_window(primary, secondary, "genpept_bridge", priority, reason, point, cell)

    frontier_cells = []
    for key, cell in cells.items():
        if len(cell.get("point_indices", [])) < min_hits:
            continue
        frontier = False
        x, y = key
        for dx_i in (-1, 0, 1):
            for dy_i in (-1, 0, 1):
                if dx_i == 0 and dy_i == 0:
                    continue
                nx, ny = x + dx_i, y + dy_i
                if nx < 0 or nx >= n_x or ny < 0 or ny >= n_y or (nx, ny) not in cells:
                    frontier = True
        if frontier:
            frontier_cells.append(cell)
    frontier_cells = sorted(frontier_cells, key=lambda c: (-len(c.get("stages", [])), len(c.get("point_indices", []))))
    for cell in frontier_cells[:budgets["frontier"]]:
        point = finite[int(cell.get("best_point_index", 0))]
        add_window(
            float(cell["primary_center"]), float(cell["secondary_center"]),
            "genpept_frontier", 70.0 + len(cell.get("stages", [])),
            "occupied support boundary; expands the first adaptive pilot frontier", point, cell,
        )

    if bool(getattr(args, "genpept_prior_absence_means_unknown", True)) and budgets["probe"] > 0:
        probe_support: dict[tuple[int, int], int] = {}
        for key in cells:
            x, y = key
            for dx_i in (-1, 0, 1):
                for dy_i in (-1, 0, 1):
                    if dx_i == 0 and dy_i == 0:
                        continue
                    nk = (x + dx_i, y + dy_i)
                    if nk in cells:
                        continue
                    if 0 <= nk[0] < n_x and 0 <= nk[1] < n_y:
                        probe_support[nk] = int(probe_support.get(nk, 0)) + 1
        probe_rows = sorted(probe_support.items(), key=lambda kv: (-kv[1], abs(kv[0][0] - n_x // 2) + abs(kv[0][1] - n_y // 2)))
        for (x, y), support in probe_rows[:budgets["probe"]]:
            primary = p_lo + (x + 0.5) * dx
            if snap_secondary and secondary_centers:
                secondary = float(secondary_centers[y])
            else:
                secondary = s_lo + (y + 0.5) * dy
            add_window(
                primary, secondary, "genpept_probe_unknown",
                40.0 + float(support), "empty neighbor of GENPEPT support; absence treated as unknown",
                None, None,
            )

    if len(selected) < max_windows:
        for cell in basin_cells:
            if len(selected) >= max_windows:
                break
            point = finite[int(cell.get("best_point_index", 0))]
            add_window(
                float(cell["primary_center"]), float(cell["secondary_center"]),
                "genpept_backfill_occupied", 10.0 + len(cell.get("point_indices", [])),
                "occupied GENPEPT bin used to fill remaining prior budget", point, cell,
            )

    _assign_force_constants(selected, args)
    for i, row in enumerate(selected):
        row["candidate_window"] = int(i)

    graph = {
        "mode": "genpept-prior-topology",
        "n_points": int(len(finite)),
        "n_cells": int(len(cells)),
        "n_edges": int(len(edge_rows)),
        "n_components": int(len(components)),
        "components": [int(len(c)) for c in components],
        "primary_range": [float(p_lo), float(p_hi)],
        "secondary_range": [float(s_lo), float(s_hi)],
        "bins": [int(n_x), int(n_y)],
        "snap_secondary_centers": bool(snap_secondary),
        "secondary_centers": [float(x) for x in secondary_centers],
        "budgets": budgets,
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "window_type_counts": {
            str(k): int(sum(1 for r in selected if str(r.get("window_type")) == str(k)))
            for k in sorted({str(r.get("window_type")) for r in selected})
        },
    }
    return selected, graph


def _default_prior_dir(args) -> Optional[Path]:
    prior_dir = _as_path(getattr(args, "genpept_prior_dir", None))
    if prior_dir is not None:
        return prior_dir
    return _as_path(getattr(args, "seed_conformers_dir", None))


def build_genpept_window_prior(
    args,
    out_dir: Path,
    topology,
    *,
    primary_cv_def: Optional[dict] = None,
    secondary_cv_metadata: Optional[dict] = None,
) -> dict:
    """Build and write a GENPEPT-derived explicit 2D window prior."""
    prior_out = Path(out_dir) / "genpept_prior"
    prior_out.mkdir(parents=True, exist_ok=True)
    if not genpept_prior_enabled(args):
        return {"enabled": False, "reason": "genpept_prior_enabled is false"}
    genpept_dir = _default_prior_dir(args)
    if genpept_dir is None:
        summary = {"enabled": True, "used": False, "reason": "no genpept_prior_dir or seed_conformers_dir provided"}
        write_json(prior_out / "genpept_prior_report.json", summary)
        return summary
    if not genpept_dir.exists():
        summary = {"enabled": True, "used": False, "genpept_dir": str(genpept_dir), "reason": "GENPEPT prior directory does not exist"}
        write_json(prior_out / "genpept_prior_report.json", summary)
        return summary
    if not secondary_cv_enabled(args):
        summary = {"enabled": True, "used": False, "genpept_dir": str(genpept_dir), "reason": "GENPEPT prior requires an active secondary CV for 2D window injection"}
        write_json(prior_out / "genpept_prior_report.json", summary)
        return summary

    warnings: list[str] = []
    if primary_cv_def is None:
        cv_atom1, cv_atom2, cv_label = choose_cv_atoms(topology, args)
        primary_cv_def = prepare_primary_cv_definition(topology, args, cv_atom1=cv_atom1, cv_atom2=cv_atom2, cv_label=cv_label)
    if secondary_cv_metadata is None:
        secondary_cv_metadata = build_secondary_cv_metadata_for_prior(args, topology)
    if not secondary_cv_metadata.get("enabled"):
        summary = {
            "enabled": True,
            "used": False,
            "genpept_dir": str(genpept_dir),
            "reason": "secondary CV metadata could not be built",
            "secondary_cv": _json_ready(secondary_cv_metadata),
        }
        write_json(prior_out / "genpept_prior_report.json", summary)
        return summary

    stages = list(getattr(args, "genpept_prior_stages", None) or DEFAULT_PRIOR_STAGES)
    input_rows, row_warnings = _collect_genpept_rows(genpept_dir, stages)
    warnings.extend(row_warnings)
    max_structures = int(getattr(args, "genpept_prior_max_structures", 50000) or 50000)
    input_rows = _thin_rows_by_stage(input_rows, max_structures)
    points, score_warnings = _score_genpept_rows(
        input_rows, genpept_dir, args, topology, primary_cv_def, secondary_cv_metadata
    )
    warnings.extend(score_warnings)
    min_points = max(1, int(getattr(args, "genpept_prior_min_points", 10) or 10))
    if len(points) < min_points:
        summary = {
            "enabled": True,
            "used": False,
            "genpept_dir": str(genpept_dir),
            "n_input_rows": int(len(input_rows)),
            "n_scored_points": int(len(points)),
            "min_points": int(min_points),
            "warnings": warnings,
            "reason": "too few GENPEPT structures could be rescored for a window prior",
        }
        write_json(prior_out / "genpept_prior_report.json", _json_ready(summary))
        return summary

    windows, graph = select_prior_windows_from_points(
        points,
        args,
        primary_mode_name=primary_cv_mode(args),
        secondary_mode_name=secondary_cv_mode(args),
    )
    if not windows:
        summary = {
            "enabled": True,
            "used": False,
            "genpept_dir": str(genpept_dir),
            "n_scored_points": int(len(points)),
            "warnings": warnings,
            "selection": graph,
            "reason": "GENPEPT prior selection produced no windows",
        }
        write_json(prior_out / "genpept_prior_report.json", _json_ready(summary))
        return summary

    windows_csv = prior_out / "genpept_prior_windows_2d.csv"
    nodes_csv = prior_out / "genpept_prior_nodes.csv"
    edges_csv = prior_out / "genpept_prior_edges.csv"
    seeds_csv = prior_out / "genpept_prior_seed_assignments.csv"
    _write_csv(windows_csv, windows, preferred=[
        "candidate_window", "window_type", "patch_lifecycle",
        "primary_cv_mode", "primary_cv_center", "primary_cv_k_kcal",
        "distance_center_A", "distance_k_kcal_mol_A2",
        "secondary_cv_mode", "secondary_cv_center", "secondary_cv_k_kcal_mol",
        "priority", "reason", "representative_pdb_path",
    ])
    _write_csv(nodes_csv, list(graph.get("node_rows", [])))
    _write_csv(edges_csv, list(graph.get("edge_rows", [])))
    seed_rows = []
    for row in windows:
        seed_rows.append({
            "window": int(row.get("candidate_window", len(seed_rows))),
            "window_type": str(row.get("window_type", "")),
            "primary_cv_center": float(row.get("primary_cv_center")),
            "secondary_cv_center": float(row.get("secondary_cv_center")),
            "representative_pdb_path": str(row.get("representative_pdb_path", "")),
            "representative_stage": str(row.get("representative_stage", "")),
            "representative_source": str(row.get("representative_source", "")),
            "representative_primary_cv_value": row.get("representative_primary_cv_value", ""),
            "representative_secondary_cv_value": row.get("representative_secondary_cv_value", ""),
        })
    _write_csv(seeds_csv, seed_rows)

    graph_payload = dict(graph)
    graph_payload.pop("node_rows", None)
    graph_payload.pop("edge_rows", None)
    graph_payload.update({
        "nodes_csv": str(nodes_csv),
        "edges_csv": str(edges_csv),
    })
    write_json(prior_out / "genpept_prior_graph.json", _json_ready(graph_payload))

    summary = {
        "enabled": True,
        "used": True,
        "mode": "genpept-window-prior",
        "description": "GENPEPT proposal structures rescored in active GAREUS CV space and converted into a round-zero explicit sparse 2D window table. This is a sampling prior only, not thermodynamic PMF evidence.",
        "genpept_dir": str(genpept_dir),
        "output_dir": str(prior_out),
        "windows_2d_csv": str(windows_csv),
        "nodes_csv": str(nodes_csv),
        "edges_csv": str(edges_csv),
        "seed_assignments_csv": str(seeds_csv),
        "graph_json": str(prior_out / "genpept_prior_graph.json"),
        "n_input_rows": int(len(input_rows)),
        "n_scored_points": int(len(points)),
        "n_windows": int(len(windows)),
        "primary_cv": primary_cv_mode(args),
        "primary_cv_units": primary_cv_units(args),
        "secondary_cv": secondary_cv_mode(args),
        "window_type_counts": graph.get("window_type_counts", {}),
        "warnings": warnings,
        "caveat": "GENPEPT prior windows guide initial coverage and seeding. Final WHAM/MBAR must use only biased MD samples with known reduced potentials.",
    }
    write_json(prior_out / "genpept_prior_report.json", _json_ready(summary))
    return summary
