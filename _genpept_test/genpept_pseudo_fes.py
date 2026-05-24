#!/usr/bin/env python3
"""
genpept_pseudo_fes.py

Build first-guess 2D pseudo-FES / seed-density landscapes from GENPEPT outputs.

This is intentionally NOT a thermodynamic PMF calculator. GENPEPT structures are
selected/generated/minimized proposals, not equilibrium samples. The output is a
useful diagnostic for CV coverage, window planning, basin topology, and comparing
which exploration stage/source found which regions.

Typical examples
----------------
Built-in archive columns:

    python genpept_pseudo_fes.py RUN_DIR \
      --cv1 end_to_end_nm --cv2 contact_count \
      --stage final_search_pool --mode source-balanced

Distance from PDB coordinates plus a built-in column:

    python genpept_pseudo_fes.py RUN_DIR \
      --cv1 distance:1:CA:10:CA --cv2 contact_count \
      --stage final_search_pool --mode source-balanced

All archive stages, same CVs:

    python genpept_pseudo_fes.py RUN_DIR \
      --cv1 end_to_end_nm --cv2 rg_nm --all-stages --write-source-maps

Distance CV syntax
------------------
  distance:<res1>:<atom1>:<res2>:<atom2>
  distance:<chain1>:<res1>:<atom1>:<chain2>:<res2>:<atom2>

Residue numbers use PDB residue sequence numbers. Distances are reported in nm.

Main outputs
------------
  pseudo_fes_2d_grid.csv         long-form bin/count/probability/F grid
  pseudo_fes_2d_metadata.json    full settings and summary
  pseudo_fes_2d.png              heatmap with optional survivor overlay
  suggested_windows_2d.csv       optional candidate window centers
  stage_summary.csv              when --all-stages is used
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    # Optional: use scipy.ndimage gaussian_filter for smoothing if available.
    from scipy.ndimage import gaussian_filter as _gaussian_filter
except Exception:
    _gaussian_filter = None

def smooth_nan_grid(F: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 2D array with NaNs using a Gaussian filter.

    NaN values are ignored in the smoothing operation. If scipy.ndimage
    is unavailable or sigma <= 0, the input array is returned unchanged.
    The output is shifted so that the minimum finite value is zero.
    """
    # Return early if no smoothing requested or unavailable
    if sigma <= 0 or _gaussian_filter is None:
        return F.copy()
    valid = np.isfinite(F)
    # If nothing is finite, return copy
    if not valid.any():
        return F.copy()
    values = np.where(valid, F, 0.0)
    weights = valid.astype(float)
    # Apply Gaussian smoothing separately to values and weights
    smooth_values = _gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = _gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.divide(
        smooth_values,
        smooth_weights,
        out=np.full_like(smooth_values, np.nan),
        where=smooth_weights > 1e-8,
    )
    out[smooth_weights <= 1e-8] = np.nan
    # Shift the minimum to zero for comparable scaling
    finite = np.isfinite(out)
    if finite.any():
        out = out - float(np.nanmin(out[finite]))
        # Remove tiny negative values due to numerical noise
        tiny = np.isfinite(out) & (np.abs(out) < 1.0e-12)
        out[tiny] = 0.0
    return out
import pandas as pd

K_B_KJ_PER_MOL_K = 0.00831446261815324
KJ_TO_KCAL = 1.0 / 4.184

STAGE_ORDER = [
    "initial_implicit",
    "post_basin_hop",
    "post_nma",
    "post_pca_explore",
    "final_search_pool",
]

SOURCE_ORDER = ["initial", "BH", "NMA", "PCA", "other"]


@dataclass
class CvSpec:
    original: str
    kind: str
    label: str
    column: Optional[str] = None
    chain1: Optional[str] = None
    res1: Optional[int] = None
    atom1: Optional[str] = None
    chain2: Optional[str] = None
    res2: Optional[int] = None
    atom2: Optional[str] = None


def slugify(text: str, max_len: int = 80) -> str:
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "value"
    return text[:max_len]


def parse_bins(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(text).replace("x", ",").split(",") if p.strip()]
    if len(parts) == 1:
        n = int(parts[0])
        return n, n
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise ValueError("--bins must be an integer or two comma-separated integers, e.g. 80 or 80,60")


def parse_range(text: Optional[str]) -> Optional[tuple[float, float, float, float]]:
    if not text:
        return None
    parts = [float(p.strip()) for p in str(text).split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("--range must be x_min,x_max,y_min,y_max")
    x0, x1, y0, y1 = parts
    if not (x1 > x0 and y1 > y0):
        raise ValueError("--range needs x_max > x_min and y_max > y_min")
    return x0, x1, y0, y1


def available_stage_files(run_dir: Path) -> dict[str, Path]:
    archive = run_dir / "basin_archive"
    found = {}
    for stage in STAGE_ORDER:
        path = archive / f"{stage}_points.csv"
        if path.exists():
            found[stage] = path
    # Include any non-standard *_points.csv files too, after canonical stages.
    if archive.exists():
        for path in sorted(archive.glob("*_points.csv")):
            stage = path.name[: -len("_points.csv")]
            found.setdefault(stage, path)
    return found


def resolve_stage(run_dir: Path, stage: str) -> str:
    found = available_stage_files(run_dir)
    if not found:
        raise FileNotFoundError(f"No basin_archive/*_points.csv files found under {run_dir}")
    if stage == "auto":
        for s in reversed(STAGE_ORDER):
            if s in found:
                return s
        return next(reversed(found))
    if stage not in found:
        raise FileNotFoundError(
            f"Stage {stage!r} not found. Available stages: {', '.join(found.keys())}"
        )
    return stage


def load_stage_points(run_dir: Path, stage: str) -> pd.DataFrame:
    stage = resolve_stage(run_dir, stage)
    path = available_stage_files(run_dir)[stage]
    df = pd.read_csv(path)
    df["__stage"] = stage
    df["__stage_points_csv"] = str(path)
    return df


def infer_source_from_row(row: pd.Series) -> str:
    for col in ("source", "origin", "stage_source"):
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val:
                low = val.lower()
                if "bh" in low or "basin" in low:
                    return "BH"
                if "nma" in low or "anm" in low:
                    return "NMA"
                if "pca" in low or "frontier" in low or "adaptive" in low:
                    return "PCA"
                if "initial" in low or "implicit" in low:
                    return "initial"
                return val
    text = " ".join(
        str(row.get(c, "")) for c in ("mode", "seed_name", "pdb_path", "input_pdb", "output_pdb")
    ).lower()
    if "nma_" in text or "nma" in str(row.get("mode", "")).lower():
        return "NMA"
    if "bh_" in text or "basin_hop" in text or "implicit_bh" in text:
        return "BH"
    if "pca" in text or "frontier" in text or "adaptive" in text:
        return "PCA"
    if "implicit" in text or "candidate" in text or "initial" in text:
        return "initial"
    return "other"


def add_source_column(df: pd.DataFrame) -> pd.DataFrame:
    if "__source" not in df.columns:
        df = df.copy()
        df["__source"] = [infer_source_from_row(row) for _, row in df.iterrows()]
    return df


def resolve_path(run_dir: Path, value) -> Optional[Path]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    p = Path(text)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    p2 = run_dir / p
    if p2.exists():
        return p2
    return p2


def best_pdb_path_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "pdb_path",
        "output_pdb",
        "output_pdb_path",
        "candidate_pdb_path",
        "input_pdb",
        "input_pdb_path",
        "representative_pdb_path",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def parse_cv_spec(text: str, df_columns: set[str]) -> CvSpec:
    original = str(text).strip()
    if original in df_columns:
        return CvSpec(original=original, kind="column", label=original, column=original)

    low = original.lower()
    if low.startswith("dist:"):
        original = "distance:" + original.split(":", 1)[1]
        low = original.lower()
    if low.startswith("distance:"):
        parts = original.split(":")
        if len(parts) == 6:
            _, res1, atom1, res2, atom2 = parts[:5]
            # Actually len 5 for no-chain; keep fallback below.
        if len(parts) == 5:
            _tag, res1, atom1, res2, atom2 = parts
            label = f"d({res1}:{atom1},{res2}:{atom2}) / nm"
            return CvSpec(
                original=original,
                kind="distance",
                label=label,
                chain1=None,
                res1=int(res1),
                atom1=atom1,
                chain2=None,
                res2=int(res2),
                atom2=atom2,
            )
        if len(parts) == 7:
            _tag, chain1, res1, atom1, chain2, res2, atom2 = parts
            label = f"d({chain1}:{res1}:{atom1},{chain2}:{res2}:{atom2}) / nm"
            return CvSpec(
                original=original,
                kind="distance",
                label=label,
                chain1=chain1 or None,
                res1=int(res1),
                atom1=atom1,
                chain2=chain2 or None,
                res2=int(res2),
                atom2=atom2,
            )
        raise ValueError(
            "Distance CV syntax must be distance:res1:atom1:res2:atom2 "
            "or distance:chain1:res1:atom1:chain2:res2:atom2"
        )

    # Common aliases.
    aliases = {
        "rg": "rg_nm",
        "radius_of_gyration": "rg_nm",
        "e2e": "end_to_end_nm",
        "end_to_end": "end_to_end_nm",
        "contacts": "contact_count",
        "contact": "contact_count",
        "energy": "energy_kj_mol",
        "force": "max_force_kj_mol_nm",
    }
    if low in aliases and aliases[low] in df_columns:
        col = aliases[low]
        return CvSpec(original=original, kind="column", label=col, column=col)

    raise KeyError(
        f"CV {original!r} is neither a column nor a supported CV spec. "
        f"Available columns include: {', '.join(sorted(list(df_columns))[:30])}"
    )


def read_pdb_atom_coord(
    pdb_path: Path,
    resseq: int,
    atom_name: str,
    chain: Optional[str] = None,
) -> Optional[np.ndarray]:
    atom_name = str(atom_name).strip()
    chain = None if chain in (None, "", "*") else str(chain)
    try:
        with Path(pdb_path).open() as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                line_atom = line[12:16].strip()
                line_chain = (line[21:22].strip() or "A")
                try:
                    line_resseq = int(line[22:26])
                except ValueError:
                    continue
                if line_resseq != int(resseq):
                    continue
                if line_atom != atom_name:
                    continue
                if chain is not None and line_chain != chain:
                    continue
                try:
                    return np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=float,
                    )
                except ValueError:
                    return None
    except OSError:
        return None
    return None


def compute_cv_values(run_dir: Path, df: pd.DataFrame, spec: CvSpec) -> np.ndarray:
    if spec.kind == "column":
        return pd.to_numeric(df[spec.column], errors="coerce").to_numpy(dtype=float)

    if spec.kind == "distance":
        pdb_col = best_pdb_path_column(df)
        if pdb_col is None:
            raise KeyError("Distance CV requested, but no PDB path column was found in the stage points CSV.")
        out = np.full(len(df), np.nan, dtype=float)
        cache: dict[tuple[str, int, str, Optional[str]], Optional[np.ndarray]] = {}
        for i, value in enumerate(df[pdb_col].tolist()):
            p = resolve_path(run_dir, value)
            if p is None or not p.exists():
                continue
            key1 = (str(p), int(spec.res1), str(spec.atom1), spec.chain1)
            key2 = (str(p), int(spec.res2), str(spec.atom2), spec.chain2)
            if key1 not in cache:
                cache[key1] = read_pdb_atom_coord(p, spec.res1, spec.atom1, spec.chain1)
            if key2 not in cache:
                cache[key2] = read_pdb_atom_coord(p, spec.res2, spec.atom2, spec.chain2)
            c1, c2 = cache[key1], cache[key2]
            if c1 is None or c2 is None:
                continue
            out[i] = float(np.linalg.norm(c2 - c1) * 0.1)  # A -> nm
        return out

    raise ValueError(f"Unsupported CV kind: {spec.kind}")


def load_survivor_keys(run_dir: Path) -> set[str]:
    candidates = [
        run_dir / "final_survivor_seeds.csv",
        run_dir / "final_survivors.csv",
        run_dir / "final_seed_pdbs.csv",
    ]
    keys: set[str] = set()
    for path in candidates:
        if not path.exists():
            continue
        try:
            sdf = pd.read_csv(path)
        except Exception:
            continue
        for col in sdf.columns:
            if col.lower() in {
                "seed_name",
                "source_seed_name",
                "selected_seed_name",
                "pdb_path",
                "output_pdb",
                "output_pdb_path",
                "final_pdb_path",
                "representative_pdb_path",
            }:
                for val in sdf[col].dropna().astype(str):
                    keys.add(val)
                    keys.add(Path(val).name)
                    keys.add(Path(val).stem)
    return keys


def mark_survivors(run_dir: Path, df: pd.DataFrame) -> np.ndarray:
    keys = load_survivor_keys(run_dir)
    if not keys:
        return np.zeros(len(df), dtype=bool)
    mask = np.zeros(len(df), dtype=bool)
    cols = [c for c in ("seed_name", "pdb_path", "output_pdb", "input_pdb") if c in df.columns]
    for i, row in df.iterrows():
        for col in cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            text = str(val)
            if text in keys or Path(text).name in keys or Path(text).stem in keys:
                mask[i] = True
                break
    return mask


def filter_representatives_only(df: pd.DataFrame) -> pd.DataFrame:
    if "basin_id" not in df.columns:
        return df
    energy_col = "energy_kj_mol" if "energy_kj_mol" in df.columns else None
    if energy_col:
        idx = df.groupby("basin_id")[energy_col].idxmin()
    else:
        idx = df.groupby("basin_id").head(1).index
    return df.loc[idx].copy().reset_index(drop=True)


def compute_weights(df: pd.DataFrame, mode: str, temperature: float, energy_col: str) -> np.ndarray:
    mode = str(mode).lower().replace("_", "-")
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=float)
    if mode == "unweighted":
        return np.ones(n, dtype=float)

    sources = df["__source"].astype(str).to_numpy() if "__source" in df.columns else np.array(["all"] * n)

    if mode == "source-balanced":
        w = np.zeros(n, dtype=float)
        for src in sorted(set(sources)):
            idx = np.where(sources == src)[0]
            if len(idx):
                w[idx] = 1.0 / float(len(idx))
        return w

    if mode in {"energy-weighted", "source-energy-balanced"}:
        if energy_col not in df.columns:
            raise KeyError(f"Energy-weighted mode requested, but column {energy_col!r} is missing.")
        e = pd.to_numeric(df[energy_col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(e)
        w = np.zeros(n, dtype=float)
        if not finite.any():
            return w
        kbt = K_B_KJ_PER_MOL_K * float(temperature)
        if kbt <= 0:
            raise ValueError("Temperature must be positive for energy weighting.")
        e0 = float(np.nanmin(e[finite]))
        expo = -np.clip((e - e0) / kbt, -700.0, 700.0)
        w[finite] = np.exp(expo[finite])
        if mode == "source-energy-balanced":
            for src in sorted(set(sources)):
                idx = np.where((sources == src) & (w > 0))[0]
                total = float(np.sum(w[idx]))
                if total > 0:
                    w[idx] /= total
        return w

    if mode == "basin-balanced":
        if "basin_id" not in df.columns:
            raise KeyError("basin-balanced mode needs a basin_id column.")
        basins = df["basin_id"].astype(str).to_numpy()
        w = np.zeros(n, dtype=float)
        for basin in sorted(set(basins)):
            idx = np.where(basins == basin)[0]
            if len(idx):
                w[idx] = 1.0 / float(len(idx))
        return w

    raise ValueError(
        "Unknown --mode. Use unweighted, source-balanced, energy-weighted, "
        "source-energy-balanced, or basin-balanced."
    )


def auto_range(vals: np.ndarray, clip_percentile: Optional[float]) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No finite values available for histogram range.")
    if clip_percentile is not None and clip_percentile > 0:
        p = float(clip_percentile)
        lo = float(np.nanpercentile(vals, p))
        hi = float(np.nanpercentile(vals, 100.0 - p))
    else:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite histogram range.")
    if hi <= lo:
        pad = max(1e-6, abs(lo) * 0.01, 1e-3)
        lo -= pad
        hi += pad
    else:
        pad = 0.02 * (hi - lo)
        lo -= pad
        hi += pad
    return lo, hi


def histogram_to_fes(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bins: tuple[int, int],
    hist_range: Optional[tuple[float, float, float, float]],
    temperature: float,
    clip_percentile: Optional[float],
    pseudocount: float,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        raise ValueError("No finite x/y values with positive weights.")
    x_use = x[mask]
    y_use = y[mask]
    w_use = weights[mask]
    if hist_range is None:
        x0, x1 = auto_range(x_use, clip_percentile)
        y0, y1 = auto_range(y_use, clip_percentile)
    else:
        x0, x1, y0, y1 = hist_range
    in_range = (x_use >= x0) & (x_use <= x1) & (y_use >= y0) & (y_use <= y1)
    x_use = x_use[in_range]
    y_use = y_use[in_range]
    w_use = w_use[in_range]
    if x_use.size == 0:
        raise ValueError("No points remain after applying histogram range.")

    H, xedges, yedges = np.histogram2d(
        x_use,
        y_use,
        bins=bins,
        range=((x0, x1), (y0, y1)),
        weights=w_use,
    )
    raw_H = H.copy()
    if pseudocount > 0:
        H = H + float(pseudocount)
    total = float(np.sum(H))
    if total <= 0:
        raise ValueError("Histogram total weight is zero.")
    P = H / total
    kbt = K_B_KJ_PER_MOL_K * float(temperature)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -kbt * np.log(P)
    # Empty bins should remain empty if no pseudocount is used.
    if pseudocount <= 0:
        F[raw_H <= 0] = np.nan
    finite = np.isfinite(F)
    if finite.any():
        F = F - float(np.nanmin(F[finite]))
    return {
        "H": raw_H,
        "H_with_pseudocount": H,
        "P": P,
        "F_kJ_mol": F,
        "xedges": xedges,
        "yedges": yedges,
        "mask_used_before_range": mask,
        "n_used_after_range": int(x_use.size),
        "total_weight_after_range": float(np.sum(w_use)),
    }


def write_grid_csv(path: Path, result: dict):
    H = result["H"]
    P = result["P"]
    F = result["F_kJ_mol"]
    xedges = result["xedges"]
    yedges = result["yedges"]
    rows = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            f = float(F[i, j]) if np.isfinite(F[i, j]) else float("nan")
            rows.append({
                "x_bin": i,
                "y_bin": j,
                "x_lo": float(xedges[i]),
                "x_hi": float(xedges[i + 1]),
                "x_center": float(0.5 * (xedges[i] + xedges[i + 1])),
                "y_lo": float(yedges[j]),
                "y_hi": float(yedges[j + 1]),
                "y_center": float(0.5 * (yedges[j] + yedges[j + 1])),
                "weight_count": float(H[i, j]),
                "probability": float(P[i, j]),
                "pseudo_F_kJ_mol": f,
                "pseudo_F_kcal_mol": f * KJ_TO_KCAL if np.isfinite(f) else float("nan"),
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def nearest_representative_for_bin(
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    x_center: float,
    y_center: float,
    bin_mask: np.ndarray,
) -> dict:
    idxs = np.where(bin_mask)[0]
    if idxs.size == 0:
        return {}
    dx = x[idxs] - x_center
    dy = y[idxs] - y_center
    score = dx * dx + dy * dy
    if "energy_kj_mol" in df.columns:
        e = pd.to_numeric(df.iloc[idxs]["energy_kj_mol"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(e).any():
            # Prefer low energy as tie-breaker without overwhelming distance.
            ez = np.zeros_like(e)
            finite = np.isfinite(e)
            if finite.sum() > 1:
                sd = float(np.nanstd(e[finite])) or 1.0
                ez[finite] = (e[finite] - float(np.nanmin(e[finite]))) / sd
            score = score + 1e-6 * ez
    chosen = int(idxs[int(np.nanargmin(score))])
    row = df.iloc[chosen]
    out = {
        "representative_row_index": chosen,
        "representative_source": str(row.get("__source", "")),
        "representative_seed_name": str(row.get("seed_name", "")),
        "representative_pdb_path": str(row.get("pdb_path", row.get("output_pdb", row.get("input_pdb", "")))),
    }
    if "energy_kj_mol" in row:
        out["representative_energy_kj_mol"] = float(row.get("energy_kj_mol")) if pd.notna(row.get("energy_kj_mol")) else float("nan")
    return out


def write_suggested_windows(
    path: Path,
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    result: dict,
    n_windows: int,
    min_bin_sep: int,
    all_occupied: bool,
):
    H = result["H"]
    P = result["P"]
    F = result["F_kJ_mol"]
    xedges = result["xedges"]
    yedges = result["yedges"]
    occupied = np.argwhere((H > 0) & np.isfinite(F))
    if occupied.size == 0:
        pd.DataFrame().to_csv(path, index=False)
        return
    # Sort by pseudo-free energy, then by descending occupancy.
    order = sorted(
        occupied.tolist(),
        key=lambda ij: (float(F[ij[0], ij[1]]), -float(H[ij[0], ij[1]])),
    )
    selected: list[tuple[int, int]] = []
    if all_occupied or n_windows <= 0:
        selected = [(int(i), int(j)) for i, j in order]
    else:
        sep = max(0, int(min_bin_sep))
        for i, j in order:
            i = int(i); j = int(j)
            if all(max(abs(i - a), abs(j - b)) >= sep for a, b in selected):
                selected.append((i, j))
            if len(selected) >= n_windows:
                break
        # If the separation was too strict, fill remaining by score.
        if len(selected) < n_windows:
            existing = set(selected)
            for i, j in order:
                pair = (int(i), int(j))
                if pair not in existing:
                    selected.append(pair)
                    existing.add(pair)
                if len(selected) >= n_windows:
                    break

    rows = []
    for rank, (i, j) in enumerate(selected, start=1):
        xc = float(0.5 * (xedges[i] + xedges[i + 1]))
        yc = float(0.5 * (yedges[j] + yedges[j + 1]))
        bin_mask = (
            np.isfinite(x) & np.isfinite(y) &
            (x >= xedges[i]) & (x < xedges[i + 1]) &
            (y >= yedges[j]) & (y < yedges[j + 1])
        )
        rep = nearest_representative_for_bin(df, x, y, xc, yc, bin_mask)
        rows.append({
            "rank": rank,
            "x_bin": i,
            "y_bin": j,
            "cv1_center": xc,
            "cv2_center": yc,
            "weight_count": float(H[i, j]),
            "probability": float(P[i, j]),
            "pseudo_F_kJ_mol": float(F[i, j]),
            "pseudo_F_kcal_mol": float(F[i, j]) * KJ_TO_KCAL,
            **rep,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def plot_fes(
    path: Path,
    result: dict,
    cv1: CvSpec,
    cv2: CvSpec,
    title: str,
    max_fes: Optional[float],
    overlay_points: Optional[tuple[np.ndarray, np.ndarray, str]] = None,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib not available; skipping plot {path}: {exc}", file=sys.stderr)
        return False

    F = np.array(result["F_kJ_mol"], copy=True)
    if max_fes is not None:
        F = np.where(np.isfinite(F), np.minimum(F, float(max_fes)), np.nan)
    xedges = result["xedges"]
    yedges = result["yedges"]

    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    mesh = ax.pcolormesh(xedges, yedges, F.T, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("pseudo-F / kJ mol$^{-1}$")
    ax.set_xlabel(cv1.label)
    ax.set_ylabel(cv2.label)
    ax.set_title(title)
    if overlay_points is not None:
        ox, oy, label = overlay_points
        ok = np.isfinite(ox) & np.isfinite(oy)
        if ok.any():
            ax.scatter(ox[ok], oy[ok], s=12, marker="o", facecolors="none", label=label)
            ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def plot_source_scatter(
    path: Path,
    df: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    cv1: CvSpec,
    cv2: CvSpec,
    title: str,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib not available; skipping source scatter {path}: {exc}", file=sys.stderr)
        return False
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    sources = df["__source"].astype(str).to_numpy()
    ordered = [s for s in SOURCE_ORDER if s in set(sources)] + sorted(set(sources) - set(SOURCE_ORDER))
    for src in ordered:
        mask = (sources == src) & np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.scatter(x[mask], y[mask], s=10, alpha=0.65, label=f"{src} ({int(mask.sum())})")
    ax.set_xlabel(cv1.label)
    ax.set_ylabel(cv2.label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def source_composition_table(df: pd.DataFrame, used_mask: np.ndarray, weights: np.ndarray) -> list[dict]:
    rows = []
    sources = df["__source"].astype(str).to_numpy()
    total_n = max(1, int(np.sum(used_mask)))
    total_w = float(np.nansum(weights[used_mask])) if len(weights) else 0.0
    for src in [s for s in SOURCE_ORDER if s in set(sources)] + sorted(set(sources) - set(SOURCE_ORDER)):
        mask = used_mask & (sources == src)
        rows.append({
            "source": src,
            "n_used": int(np.sum(mask)),
            "fraction_used": float(np.sum(mask)) / float(total_n),
            "weight_sum": float(np.nansum(weights[mask])) if len(weights) else 0.0,
            "weight_fraction": float(np.nansum(weights[mask]) / total_w) if total_w > 0 else float("nan"),
        })
    return rows


def run_one_stage(args, run_dir: Path, stage: str, base_out_dir: Path) -> dict:
    stage = resolve_stage(run_dir, stage)
    df = load_stage_points(run_dir, stage)
    df = add_source_column(df)

    if args.representatives_only:
        df = filter_representatives_only(df)
    survivor_mask = mark_survivors(run_dir, df)
    df["__is_final_survivor"] = survivor_mask
    if args.survivors_only:
        df = df.loc[survivor_mask].copy().reset_index(drop=True)
        df["__is_final_survivor"] = True
        if len(df) == 0:
            raise ValueError("--survivors-only requested, but no rows matched final_survivor_seeds.csv")

    cv1 = parse_cv_spec(args.cv1, set(df.columns))
    cv2 = parse_cv_spec(args.cv2, set(df.columns))
    x = compute_cv_values(run_dir, df, cv1)
    y = compute_cv_values(run_dir, df, cv2)
    weights = compute_weights(df, args.mode, args.temperature, args.energy_column)
    bins = parse_bins(args.bins)
    hist_range = parse_range(args.range)

    # Compute the base histogram and pseudo-FES
    result = histogram_to_fes(
        x,
        y,
        weights,
        bins=bins,
        hist_range=hist_range,
        temperature=args.temperature,
        clip_percentile=args.clip_percentile,
        pseudocount=args.pseudocount,
    )
    # Optionally smooth the pseudo-F surface. This does not change the underlying
    # probability or histogram; it only affects the F grid used for plotting and
    # output. The smoothing function handles NaNs and normalization internally.
    if getattr(args, "fes_smooth_sigma", 0.0) and float(args.fes_smooth_sigma) > 0.0:
        try:
            F_orig = np.array(result["F_kJ_mol"], copy=True)
            F_smooth = smooth_nan_grid(F_orig, float(args.fes_smooth_sigma))
            result["F_kJ_mol"] = F_smooth
        except Exception:
            # If smoothing fails, leave result unchanged and warn
            print("[warn] smoothing failed or scipy unavailable; using unsmoothed FES", file=sys.stderr)

    if args.all_stages:
        out_dir = base_out_dir / stage
    else:
        out_dir = base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_grid_csv(out_dir / "pseudo_fes_2d_grid.csv", result)

    used_mask = result["mask_used_before_range"]
    # mask_used_before_range is pre-range; create final in-range mask for summary/source comp.
    xedges, yedges = result["xedges"], result["yedges"]
    used_mask = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0) &
        (x >= xedges[0]) & (x <= xedges[-1]) & (y >= yedges[0]) & (y <= yedges[-1])
    )

    source_rows = source_composition_table(df, used_mask, weights)
    pd.DataFrame(source_rows).to_csv(out_dir / "pseudo_fes_source_composition.csv", index=False)

    overlay = None
    if args.overlay_survivors and "__is_final_survivor" in df.columns:
        smask = df["__is_final_survivor"].to_numpy(dtype=bool)
        if smask.any():
            overlay = (x[smask], y[smask], "final survivors")

    plot_fes(
        out_dir / "pseudo_fes_2d.png",
        result,
        cv1,
        cv2,
        title=f"{stage}: {args.mode} pseudo-FES",
        max_fes=args.max_fes,
        overlay_points=overlay,
    )
    plot_source_scatter(
        out_dir / "pseudo_fes_sources_scatter.png",
        df,
        x,
        y,
        cv1,
        cv2,
        title=f"{stage}: source positions in CV space",
    )

    if args.write_source_maps:
        for src in sorted(set(df["__source"].astype(str))):
            src_df = df[df["__source"].astype(str) == src].copy().reset_index(drop=True)
            if len(src_df) == 0:
                continue
            sx = compute_cv_values(run_dir, src_df, cv1)
            sy = compute_cv_values(run_dir, src_df, cv2)
            sw = np.ones(len(src_df), dtype=float)
            try:
                src_result = histogram_to_fes(
                    sx,
                    sy,
                    sw,
                    bins=bins,
                    hist_range=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                    temperature=args.temperature,
                    clip_percentile=None,
                    pseudocount=args.pseudocount,
                )
            except Exception:
                continue
            src_dir = out_dir / "source_maps"
            src_dir.mkdir(exist_ok=True)
            slug = slugify(src)
            write_grid_csv(src_dir / f"{slug}_pseudo_fes_2d_grid.csv", src_result)
            plot_fes(
                src_dir / f"{slug}_pseudo_fes_2d.png",
                src_result,
                cv1,
                cv2,
                title=f"{stage}: {src} only",
                max_fes=args.max_fes,
            )

    if args.suggest_windows > 0 or args.suggest_all_occupied:
        write_suggested_windows(
            out_dir / "suggested_windows_2d.csv",
            df,
            x,
            y,
            result,
            n_windows=args.suggest_windows,
            min_bin_sep=args.suggest_min_bin_sep,
            all_occupied=args.suggest_all_occupied,
        )

    finite_F = result["F_kJ_mol"][np.isfinite(result["F_kJ_mol"])]
    H = result["H"]
    summary = {
        "run_dir": str(run_dir),
        "stage": stage,
        "stage_points_csv": str(available_stage_files(run_dir)[stage]),
        "output_dir": str(out_dir),
        "cv1": asdict(cv1),
        "cv2": asdict(cv2),
        "mode": args.mode,
        "temperature_K": float(args.temperature),
        "kBT_kJ_mol": float(K_B_KJ_PER_MOL_K * args.temperature),
        "bins": [int(bins[0]), int(bins[1])],
        "range": [float(xedges[0]), float(xedges[-1]), float(yedges[0]), float(yedges[-1])],
        "n_rows_loaded": int(len(df)),
        "n_points_used": int(np.sum(used_mask)),
        "n_used_after_range": int(result["n_used_after_range"]),
        "total_weight_after_range": float(result["total_weight_after_range"]),
        "occupied_bins": int(np.count_nonzero(H > 0)),
        "total_bins": int(H.size),
        "occupied_fraction": float(np.count_nonzero(H > 0) / max(1, H.size)),
        "min_pseudo_F_kJ_mol": float(np.nanmin(finite_F)) if finite_F.size else float("nan"),
        "max_finite_pseudo_F_kJ_mol": float(np.nanmax(finite_F)) if finite_F.size else float("nan"),
        "median_finite_pseudo_F_kJ_mol": float(np.nanmedian(finite_F)) if finite_F.size else float("nan"),
        "source_composition": source_rows,
        "fes_smooth_sigma": float(getattr(args, "fes_smooth_sigma", 0.0)),
        "caveat": (
            "This is a GENPEPT pseudo-FES from generated/selected/minimized structures, "
            "not a thermodynamic PMF from equilibrium sampling. Use it for CV coverage, "
            "window planning, and basin-discovery diagnostics."
        ),
    }
    (out_dir / "pseudo_fes_2d_metadata.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build first-guess 2D pseudo-FES / seed-density maps from GENPEPT output directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_dir", type=Path, help="GENPEPT output directory.")
    p.add_argument("--stage", default="auto", help="Archive stage to analyze, or auto for best available final stage.")
    p.add_argument("--all-stages", action="store_true", help="Analyze every available basin_archive/*_points.csv stage.")
    p.add_argument("--cv1", default="end_to_end_nm", help="First CV: archive column or distance spec.")
    p.add_argument("--cv2", default="contact_count", help="Second CV: archive column or distance spec.")
    p.add_argument("--bins", default="80,80", help="Histogram bins: N or Nx,Ny / N,Ny.")
    p.add_argument("--range", default=None, help="Histogram range: x_min,x_max,y_min,y_max. Default uses data range.")
    p.add_argument("--clip-percentile", type=float, default=None, help="Optional percentile clipping for automatic range, e.g. 1.0.")
    p.add_argument("--temperature", type=float, default=300.0, help="Temperature for -kBT ln P and energy weighting.")
    p.add_argument("--mode", default="source-balanced", choices=[
        "unweighted",
        "source-balanced",
        "energy-weighted",
        "source-energy-balanced",
        "basin-balanced",
    ], help="Weighting scheme for the 2D density.")
    p.add_argument("--energy-column", default="energy_kj_mol", help="Energy column for energy-weighted modes.")
    p.add_argument("--pseudocount", type=float, default=0.0, help="Optional histogram pseudocount. Empty bins are NaN when zero.")
    p.add_argument(
        "--fes-smooth-sigma",
        type=float,
        default=0.0,
        help=(
            "Apply a Gaussian blur to the pseudo-FES surface before plotting. "
            "A value of 0 disables smoothing. Increasing sigma values produce "
            "progressively smoother free-energy landscapes (requires scipy)."
        ),
    )
    p.add_argument("--max-fes", type=float, default=50.0, help="Clip plotted pseudo-F values to this kJ/mol; does not affect CSV.")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory. Default: RUN_DIR/pseudo_fes_<stage>_<cv1>_vs_<cv2>.")
    p.add_argument("--representatives-only", action="store_true", help="Use one lowest-energy row per basin_id before histogramming.")
    p.add_argument("--survivors-only", action="store_true", help="Use only rows matching final_survivor_seeds.csv.")
    p.add_argument("--overlay-survivors", action=argparse.BooleanOptionalAction, default=True, help="Overlay final survivor points on the main heatmap when available.")
    p.add_argument("--write-source-maps", action="store_true", help="Also write per-source pseudo-FES maps.")
    p.add_argument("--suggest-windows", type=int, default=0, help="Write this many suggested low-pseudo-F diverse window centers.")
    p.add_argument("--suggest-all-occupied", action="store_true", help="Write all occupied bin centers to suggested_windows_2d.csv.")
    p.add_argument("--suggest-min-bin-sep", type=int, default=2, help="Greedy bin separation for suggested windows.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if args.all_stages:
        stages = list(available_stage_files(run_dir).keys())
        # canonical order first, then extras
        stages = [s for s in STAGE_ORDER if s in stages] + [s for s in stages if s not in STAGE_ORDER]
        if args.out_dir is None:
            base_out = run_dir / f"pseudo_fes_all_stages_{slugify(args.cv1)}_vs_{slugify(args.cv2)}"
        else:
            base_out = args.out_dir
        base_out.mkdir(parents=True, exist_ok=True)
        summaries = []
        for stage in stages:
            print(f"[stage] {stage}")
            summaries.append(run_one_stage(args, run_dir, stage, base_out))
        pd.DataFrame([
            {
                "stage": s["stage"],
                "n_points_used": s["n_points_used"],
                "occupied_bins": s["occupied_bins"],
                "occupied_fraction": s["occupied_fraction"],
                "total_weight_after_range": s["total_weight_after_range"],
                "max_finite_pseudo_F_kJ_mol": s["max_finite_pseudo_F_kJ_mol"],
                "output_dir": s["output_dir"],
            }
            for s in summaries
        ]).to_csv(base_out / "stage_summary.csv", index=False)
        (base_out / "pseudo_fes_all_stages_metadata.json").write_text(json.dumps(summaries, indent=2))
        print(f"Wrote all-stage pseudo-FES outputs to: {base_out}")
        return 0

    stage = resolve_stage(run_dir, args.stage)
    if args.out_dir is None:
        out_dir = run_dir / f"pseudo_fes_{stage}_{slugify(args.cv1)}_vs_{slugify(args.cv2)}"
    else:
        out_dir = args.out_dir
    summary = run_one_stage(args, run_dir, stage, out_dir)
    print(f"Wrote pseudo-FES outputs to: {summary['output_dir']}")
    print(
        f"Used {summary['n_points_used']} structures; occupied "
        f"{summary['occupied_bins']}/{summary['total_bins']} bins."
    )
    print("Caveat: pseudo-FES from GENPEPT seed density, not an equilibrium PMF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
