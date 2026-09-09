#!/usr/bin/env python3
"""
compare_genpept_exploration.py

Tell whether GENPEPT basin hopping and ANM/NMA actually added useful conformational
coverage, or mostly produced local duplicates.

It is designed for output folders produced by GENPEPT.py / rama_to_bh_twostage_seeds.py.
It reads the archive CSVs that GENPEPT already writes:

  <run>/basin_archive/initial_implicit_summary.csv
  <run>/basin_archive/post_basin_hop_summary.csv
  <run>/basin_archive/post_nma_summary.csv
  <run>/basin_archive/final_search_pool_summary.csv
  <run>/basin_archive/*_points.csv
  <run>/basin_archive/*_basins.csv
  <run>/exploration_space_metrics.csv
  <run>/basin_hop_minima.csv
  <run>/nma_implicit_minimization_scores.csv
  <run>/final_survivor_seeds.csv

Example:

  python compare_genpept_exploration.py ACDEFGHIKL_bh
  python compare_genpept_exploration.py ACDEFGHIKL_bh --write-md comparison.md --write-json comparison.json

The verdict is heuristic. Treat it as a triage report, not a thermodynamic or kinetic claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# For plotting
try:
    import matplotlib
    matplotlib.use("Agg")  # Use a non-interactive backend
    import matplotlib.pyplot as plt
except Exception:
    plt = None

STAGE_ORDER = [
    "initial_implicit",
    "post_basin_hop",
    "post_nma",
    "post_pca_explore",
    "final_search_pool",
]

SUMMARY_FIELDS = [
    "n_structures",
    "n_basins",
    "N_eff_count",
    "space_explored_score",
    "space_delta_from_previous",
    "space_rms_radius",
    "space_pc12_area",
    "H_count_dimensionless",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return default
        out = float(text)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    x = to_float(value, math.nan)
    if not math.isfinite(x):
        return default
    return int(round(x))


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def finite_values(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        x = to_float(value, math.nan)
        if math.isfinite(x):
            out.append(x)
    return out


def median_or_nan(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    if not vals:
        return math.nan
    return float(statistics.median(vals))


def min_or_nan(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    if not vals:
        return math.nan
    return float(min(vals))


def pct(numer: float, denom: float) -> float:
    if denom == 0 or not math.isfinite(denom):
        return math.nan
    return 100.0 * numer / denom


def fmt_float(value: Any, digits: int = 3, signed: bool = False) -> str:
    x = to_float(value, math.nan)
    if not math.isfinite(x):
        return "n/a"
    sign = "+" if signed else ""
    return f"{x:{sign}.{digits}f}"


def fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    x = to_float(value, math.nan)
    if not math.isfinite(x):
        return "n/a"
    sign = "+" if signed else ""
    return f"{x:{sign}.{digits}f}%"


def safe_basename(text: str) -> str:
    text = str(text or "").replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def row_text(row: dict[str, Any]) -> str:
    fields = [
        "mode",
        "seed_name",
        "parent_seed",
        "pdb_path",
        "input_pdb",
        "output_pdb",
        "survivor_pdb_path",
        "representative_pdb_path",
        "best_seed_name",
    ]
    return " ".join(str(row.get(f, "")) for f in fields).lower().replace("\\", "/")


def source_of(row: dict[str, Any]) -> str:
    """Classify a row as initial, BH, NMA, PCA/adaptive, or other.

    NMA is checked before BH because NMA files may be generated from BH minima and
    therefore contain both strings in their names.
    """
    text = row_text(row)
    base_bits = " ".join(
        safe_basename(str(row.get(f, ""))).lower()
        for f in ("seed_name", "pdb_path", "input_pdb", "output_pdb", "survivor_pdb_path", "representative_pdb_path", "best_seed_name")
    )
    mode = str(row.get("mode", "")).lower()

    if (
        "aa_nma_implicit_minimized_pdbs" in text
        or "/nma_probe_seeds/" in text
        or "nma_implicit" in text
        or " nma_" in (" " + base_bits)
        or base_bits.startswith("nma_")
    ):
        return "NMA"

    if "adaptive_pca" in text or "pca_explore" in text or "frontier" in text:
        return "PCA"

    if (
        "implicit_bh" in mode
        or "/basin_hop_minima/" in text
        or "basin_hop_minima" in text
        or " bh_" in (" " + base_bits)
        or "_hop_" in base_bits
        or base_bits.startswith("bh_")
    ):
        return "BH"

    if mode in {"implicit", "explicit", ""}:
        return "initial"
    return mode or "other"


def coarse_key(row: dict[str, Any], rg_bin_nm: float, e2e_bin_nm: float) -> tuple[int, int, int] | None:
    rg = to_float(row.get("rg_nm"), math.nan)
    e2e = to_float(row.get("end_to_end_nm"), math.nan)
    ccount = to_float(row.get("contact_count"), math.nan)
    if not (math.isfinite(rg) and math.isfinite(e2e) and math.isfinite(ccount)):
        return None
    rg_bin_nm = max(float(rg_bin_nm), 1e-9)
    e2e_bin_nm = max(float(e2e_bin_nm), 1e-9)
    return (int(round(rg / rg_bin_nm)), int(round(e2e / e2e_bin_nm)), int(round(ccount)))


def keys_for(rows: list[dict[str, Any]], rg_bin_nm: float, e2e_bin_nm: float, source: str | None = None) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for row in rows:
        if source is not None and source_of(row) != source:
            continue
        key = coarse_key(row, rg_bin_nm, e2e_bin_nm)
        if key is not None:
            out.add(key)
    return out


def archive_dir(run: Path) -> Path:
    return run / "basin_archive"


def read_summary(run: Path, prefix: str) -> dict[str, Any]:
    path = archive_dir(run) / f"{prefix}_summary.csv"
    rows = read_csv_rows(path)
    if rows:
        out: dict[str, Any] = dict(rows[0])
        out["archive_prefix"] = prefix
        out["summary_path"] = str(path)
        return out

    # Fallback: root exploration_space_metrics.csv may still have the metrics.
    metrics = read_csv_rows(run / "exploration_space_metrics.csv")
    for row in metrics:
        if str(row.get("archive_prefix", "")) == prefix or str(row.get("step_label", "")) == prefix:
            out = dict(row)
            out["archive_prefix"] = prefix
            out["summary_path"] = str(run / "exploration_space_metrics.csv")
            return out
    return {"archive_prefix": prefix, "missing": True}


def read_points(run: Path, prefix: str) -> list[dict[str, Any]]:
    return read_csv_rows(archive_dir(run) / f"{prefix}_points.csv")


def read_basins(run: Path, prefix: str) -> list[dict[str, Any]]:
    return read_csv_rows(archive_dir(run) / f"{prefix}_basins.csv")


def existing_stage_prefixes(run: Path) -> list[str]:
    prefixes = []
    for prefix in STAGE_ORDER:
        if (archive_dir(run) / f"{prefix}_summary.csv").exists() or any(
            str(row.get("archive_prefix", "")) == prefix for row in read_csv_rows(run / "exploration_space_metrics.csv")
        ):
            prefixes.append(prefix)
    # Also catch custom archive prefixes.
    for path in archive_dir(run).glob("*_summary.csv"):
        prefix = path.name[: -len("_summary.csv")]
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


def summarize_stage_deltas(run: Path) -> list[dict[str, Any]]:
    rows = []
    prefixes = existing_stage_prefixes(run)
    prev: dict[str, Any] | None = None
    for prefix in prefixes:
        cur = read_summary(run, prefix)
        if cur.get("missing"):
            continue
        out: dict[str, Any] = {"stage": prefix}
        for field in SUMMARY_FIELDS:
            out[field] = to_float(cur.get(field), math.nan)
        if prev is not None:
            prev_score = to_float(prev.get("space_explored_score"), math.nan)
            cur_score = to_float(cur.get("space_explored_score"), math.nan)
            out["score_delta_vs_previous"] = cur_score - prev_score if math.isfinite(prev_score) and math.isfinite(cur_score) else math.nan
            out["score_delta_pct_vs_previous"] = pct(out["score_delta_vs_previous"], prev_score)
            prev_basins = to_float(prev.get("n_basins"), math.nan)
            cur_basins = to_float(cur.get("n_basins"), math.nan)
            out["n_basins_delta_vs_previous"] = cur_basins - prev_basins if math.isfinite(prev_basins) and math.isfinite(cur_basins) else math.nan
            prev_neff = to_float(prev.get("N_eff_count"), math.nan)
            cur_neff = to_float(cur.get("N_eff_count"), math.nan)
            out["N_eff_count_delta_vs_previous"] = cur_neff - prev_neff if math.isfinite(prev_neff) and math.isfinite(cur_neff) else math.nan
        else:
            out["score_delta_vs_previous"] = math.nan
            out["score_delta_pct_vs_previous"] = math.nan
            out["n_basins_delta_vs_previous"] = math.nan
            out["N_eff_count_delta_vs_previous"] = math.nan
        rows.append(out)
        prev = cur
    return rows


def source_counts(rows: list[dict[str, Any]]) -> Counter:
    return Counter(source_of(row) for row in rows)


def summarize_source_contribution(
    run: Path,
    prefix: str,
    baseline_prefix: str,
    source: str,
    rg_bin_nm: float,
    e2e_bin_nm: float,
) -> dict[str, Any]:
    points = read_points(run, prefix)
    baseline = read_points(run, baseline_prefix)
    basins = read_basins(run, prefix)

    total_points = len(points)
    src_rows = [row for row in points if source_of(row) == source]
    baseline_keys = keys_for(baseline, rg_bin_nm, e2e_bin_nm)
    src_keys = keys_for(src_rows, rg_bin_nm, e2e_bin_nm)
    new_src_keys = src_keys - baseline_keys

    basin_to_sources: dict[str, Counter] = defaultdict(Counter)
    for row in points:
        bid = str(row.get("basin_id", ""))
        if bid != "":
            basin_to_sources[bid][source_of(row)] += 1

    basins_with_source = 0
    source_only_basins = 0
    for counts in basin_to_sources.values():
        if counts.get(source, 0) > 0:
            basins_with_source += 1
            if sum(counts.values()) == counts.get(source, 0):
                source_only_basins += 1

    representative_basins = 0
    for row in basins:
        if source_of(row) == source:
            representative_basins += 1

    baseline_energies = [to_float(row.get("energy_kj_mol"), math.nan) for row in baseline]
    src_energies = [to_float(row.get("energy_kj_mol"), math.nan) for row in src_rows]
    best_baseline = min_or_nan(baseline_energies)
    best_source = min_or_nan(src_energies)
    energy_improvement = best_source - best_baseline if math.isfinite(best_source) and math.isfinite(best_baseline) else math.nan

    return {
        "source": source,
        "archive_stage": prefix,
        "baseline_stage": baseline_prefix,
        "total_points_in_archive": total_points,
        "source_points": len(src_rows),
        "source_fraction_pct": pct(len(src_rows), total_points),
        "source_unique_coarse_bins": len(src_keys),
        "new_coarse_bins_vs_baseline": len(new_src_keys),
        "basins_with_source": basins_with_source,
        "source_only_basins": source_only_basins,
        "source_representative_basins": representative_basins,
        "best_baseline_energy_kj_mol": best_baseline,
        "best_source_energy_kj_mol": best_source,
        "best_source_minus_baseline_energy_kj_mol": energy_improvement,
    }


def summarize_final_survivors(run: Path) -> dict[str, Any]:
    rows = read_csv_rows(run / "final_survivor_seeds.csv")
    counts = source_counts(rows)
    out: dict[str, Any] = {"total": len(rows)}
    for src in ["initial", "BH", "NMA", "PCA", "other"]:
        out[f"{src}_count"] = int(counts.get(src, 0))
        out[f"{src}_pct"] = pct(counts.get(src, 0), len(rows))
    return out


def summarize_bh(run: Path, rg_bin_nm: float, e2e_bin_nm: float) -> dict[str, Any]:
    rows = read_csv_rows(run / "basin_hop_minima.csv")
    if not rows:
        return {"present": False}

    success = [row for row in rows if truthy(row.get("success"))]
    non_parent = [row for row in success if to_int(row.get("hop_index"), -999) > 0]
    parents = sorted({str(row.get("parent_seed", "")) for row in rows if str(row.get("parent_seed", ""))})

    parent_key: dict[str, tuple[int, int, int]] = {}
    parent_energy: dict[str, float] = {}
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in success:
        parent = str(row.get("parent_seed", ""))
        by_parent[parent].append(row)
        if to_int(row.get("hop_index"), -999) == 0:
            key = coarse_key(row, rg_bin_nm, e2e_bin_nm)
            if key is not None:
                parent_key[parent] = key
            e = to_float(row.get("minimized_energy_kj_mol"), math.nan)
            if math.isfinite(e):
                parent_energy[parent] = e

    changed = 0
    comparable = 0
    energy_deltas: list[float] = []
    unique_endpoint_keys_per_parent = []
    parents_with_any_changed = 0
    for parent, group in by_parent.items():
        keys = set()
        any_changed = False
        for row in group:
            if to_int(row.get("hop_index"), -999) <= 0:
                continue
            key = coarse_key(row, rg_bin_nm, e2e_bin_nm)
            if key is not None:
                keys.add(key)
            if parent in parent_key and key is not None:
                comparable += 1
                if key != parent_key[parent]:
                    changed += 1
                    any_changed = True
            if parent in parent_energy:
                e = to_float(row.get("minimized_energy_kj_mol"), math.nan)
                if math.isfinite(e):
                    energy_deltas.append(e - parent_energy[parent])
        if keys:
            unique_endpoint_keys_per_parent.append(len(keys))
        if any_changed:
            parents_with_any_changed += 1

    return {
        "present": True,
        "rows_total": len(rows),
        "success_total": len(success),
        "success_pct": pct(len(success), len(rows)),
        "parents_total": len(parents),
        "non_parent_success_endpoints": len(non_parent),
        "coarse_bin_changed_endpoints": changed,
        "coarse_bin_comparable_endpoints": comparable,
        "coarse_bin_changed_pct": pct(changed, comparable),
        "parents_with_any_changed_endpoint": parents_with_any_changed,
        "median_unique_endpoint_bins_per_parent": median_or_nan(unique_endpoint_keys_per_parent),
        "median_endpoint_minus_parent_energy_kj_mol": median_or_nan(energy_deltas),
        "best_endpoint_minus_parent_energy_kj_mol": min_or_nan(energy_deltas),
    }


def summarize_nma(run: Path, rg_bin_nm: float, e2e_bin_nm: float, baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = read_csv_rows(run / "nma_implicit_minimization_scores.csv")
    probe_count = len(list((run / "nma_probe_seeds").glob("*.pdb"))) if (run / "nma_probe_seeds").exists() else 0
    if not rows:
        return {"present": False, "probe_pdb_count": probe_count}

    success = [row for row in rows if truthy(row.get("success"))]
    baseline_keys = keys_for(baseline_rows, rg_bin_nm, e2e_bin_nm)
    nma_keys = keys_for(success, rg_bin_nm, e2e_bin_nm)
    return {
        "present": True,
        "probe_pdb_count": probe_count,
        "rows_total": len(rows),
        "success_total": len(success),
        "success_pct": pct(len(success), len(rows)),
        "unique_coarse_bins": len(nma_keys),
        "new_coarse_bins_vs_pre_nma": len(nma_keys - baseline_keys),
        "best_nma_energy_kj_mol": min_or_nan(row.get("minimized_energy_kj_mol") for row in success),
        "median_nma_energy_kj_mol": median_or_nan(row.get("minimized_energy_kj_mol") for row in success),
        "median_nma_max_force_kj_mol_nm": median_or_nan(row.get("max_force_kj_mol_nm") for row in success),
    }


def stage_row(stage_rows: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    for row in stage_rows:
        if row.get("stage") == stage:
            return row
    return None


def stage_delta_pct(stage_rows: list[dict[str, Any]], stage: str) -> float:
    row = stage_row(stage_rows, stage)
    if not row:
        return math.nan
    return to_float(row.get("score_delta_pct_vs_previous"), math.nan)


def stage_delta_abs(stage_rows: list[dict[str, Any]], stage: str) -> float:
    row = stage_row(stage_rows, stage)
    if not row:
        return math.nan
    return to_float(row.get("score_delta_vs_previous"), math.nan)


def score_bh(contrib: dict[str, Any], bh: dict[str, Any], survivors: dict[str, Any], stage_rows: list[dict[str, Any]]) -> tuple[int, list[str], str]:
    if not bh.get("present") and not contrib.get("source_points"):
        return 0, ["No basin_hop_minima.csv/post-BH contribution found."], "not run or no readable BH output"

    score = 0
    reasons: list[str] = []
    gain_pct = stage_delta_pct(stage_rows, "post_basin_hop")
    new_bins = to_int(contrib.get("new_coarse_bins_vs_baseline"), 0)
    reps = to_int(contrib.get("source_representative_basins"), 0)
    src_only = to_int(contrib.get("source_only_basins"), 0)
    surv = to_int(survivors.get("BH_count"), 0)
    changed_pct = to_float(bh.get("coarse_bin_changed_pct"), math.nan)

    if math.isfinite(gain_pct):
        if gain_pct >= 15.0:
            score += 30
            reasons.append(f"post-BH archive score increased strongly ({gain_pct:.1f}%).")
        elif gain_pct >= 5.0:
            score += 18
            reasons.append(f"post-BH archive score increased modestly ({gain_pct:.1f}%).")
        elif gain_pct >= 1.0:
            score += 8
            reasons.append(f"post-BH archive score increased slightly ({gain_pct:.1f}%).")
        else:
            reasons.append(f"post-BH archive score barely moved ({gain_pct:.1f}%).")

    if new_bins >= 10:
        score += 24
        reasons.append(f"BH added {new_bins} new coarse Rg/E2E/contact bins.")
    elif new_bins > 0:
        score += 12
        reasons.append(f"BH added {new_bins} new coarse Rg/E2E/contact bins.")
    else:
        reasons.append("BH added no new coarse Rg/E2E/contact bins at the chosen bin size.")

    if reps > 0:
        score += min(18, 8 + 2 * reps)
        reasons.append(f"BH became the representative for {reps} archive basin(s).")
    if src_only > 0:
        score += min(10, 3 * src_only)
        reasons.append(f"BH created {src_only} BH-only archive basin(s).")

    if surv > 0:
        score += min(25, 12 + 4 * surv)
        reasons.append(f"BH contributed {surv} final survivor(s).")
    else:
        reasons.append("BH contributed no final survivors.")

    if math.isfinite(changed_pct):
        if changed_pct >= 25.0:
            score += 15
            reasons.append(f"{changed_pct:.1f}% of comparable BH endpoints changed coarse bins relative to parent.")
        elif changed_pct >= 10.0:
            score += 8
            reasons.append(f"{changed_pct:.1f}% of comparable BH endpoints changed coarse bins relative to parent.")
        else:
            reasons.append(f"Only {changed_pct:.1f}% of comparable BH endpoints changed coarse bins relative to parent.")

    verdict = verdict_from_score(score)
    return score, reasons, verdict


def score_nma(contrib: dict[str, Any], nma: dict[str, Any], survivors: dict[str, Any], stage_rows: list[dict[str, Any]]) -> tuple[int, list[str], str]:
    if not nma.get("present") and not contrib.get("source_points"):
        return 0, ["No nma_implicit_minimization_scores.csv/post-NMA contribution found."], "not run or no readable NMA output"

    score = 0
    reasons: list[str] = []
    gain_pct = stage_delta_pct(stage_rows, "post_nma")
    new_bins = to_int(contrib.get("new_coarse_bins_vs_baseline"), 0)
    if new_bins == 0:
        new_bins = to_int(nma.get("new_coarse_bins_vs_pre_nma"), 0)
    reps = to_int(contrib.get("source_representative_basins"), 0)
    src_only = to_int(contrib.get("source_only_basins"), 0)
    surv = to_int(survivors.get("NMA_count"), 0)

    if math.isfinite(gain_pct):
        if gain_pct >= 15.0:
            score += 30
            reasons.append(f"post-NMA archive score increased strongly ({gain_pct:.1f}%).")
        elif gain_pct >= 5.0:
            score += 18
            reasons.append(f"post-NMA archive score increased modestly ({gain_pct:.1f}%).")
        elif gain_pct >= 1.0:
            score += 8
            reasons.append(f"post-NMA archive score increased slightly ({gain_pct:.1f}%).")
        else:
            reasons.append(f"post-NMA archive score barely moved ({gain_pct:.1f}%).")

    if new_bins >= 10:
        score += 24
        reasons.append(f"NMA added {new_bins} new coarse Rg/E2E/contact bins.")
    elif new_bins > 0:
        score += 12
        reasons.append(f"NMA added {new_bins} new coarse Rg/E2E/contact bins.")
    else:
        reasons.append("NMA added no new coarse Rg/E2E/contact bins at the chosen bin size.")

    if reps > 0:
        score += min(18, 8 + 2 * reps)
        reasons.append(f"NMA became the representative for {reps} archive basin(s).")
    if src_only > 0:
        score += min(10, 3 * src_only)
        reasons.append(f"NMA created {src_only} NMA-only archive basin(s).")

    if surv > 0:
        score += min(25, 12 + 4 * surv)
        reasons.append(f"NMA contributed {surv} final survivor(s).")
    else:
        reasons.append("NMA contributed no final survivors.")

    success_pct = to_float(nma.get("success_pct"), math.nan)
    if math.isfinite(success_pct) and success_pct < 50.0:
        score -= 5
        reasons.append(f"NMA minimization success was low ({success_pct:.1f}%).")

    verdict = verdict_from_score(score)
    return score, reasons, verdict


def verdict_from_score(score: int) -> str:
    if score >= 60:
        return "meaningfully expanding conformational coverage"
    if score >= 35:
        return "adding some useful coverage"
    if score >= 15:
        return "weak/local contribution"
    return "probably mostly redundant for this run"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(x: Any) -> str:
        return str(x).replace("\n", " ")
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(cell(x) for x in row) + " |")
    return "\n".join(out)


def build_report(data: dict[str, Any]) -> str:
    run = data["run"]
    stage_rows = data["stage_deltas"]
    bh = data["bh"]
    nma = data["nma"]
    bh_contrib = data["bh_contribution"]
    nma_contrib = data["nma_contribution"]
    survivors = data["final_survivors"]
    bh_score, bh_reasons, bh_verdict = data["bh_verdict"]
    nma_score, nma_reasons, nma_verdict = data["nma_verdict"]

    lines: list[str] = []
    lines.append(f"# GENPEPT exploration comparison: `{run}`")
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    lines.append(f"- **Basin hopping:** {bh_verdict}  ")
    lines.append(f"  Evidence score: **{bh_score}/100**")
    lines.append(f"- **NMA:** {nma_verdict}  ")
    lines.append(f"  Evidence score: **{nma_score}/100**")
    lines.append("")

    total_surv = to_int(survivors.get("total"), 0)
    if total_surv:
        lines.append(
            f"Final survivors by source: initial={survivors.get('initial_count', 0)}, "
            f"BH={survivors.get('BH_count', 0)}, NMA={survivors.get('NMA_count', 0)}, "
            f"PCA={survivors.get('PCA_count', 0)} out of {total_surv}."
        )
        lines.append("")

    lines.append("## Archive score changes")
    lines.append("")
    table_rows = []
    for row in stage_rows:
        table_rows.append([
            row.get("stage", ""),
            fmt_float(row.get("n_structures"), 0),
            fmt_float(row.get("n_basins"), 0),
            fmt_float(row.get("N_eff_count"), 2),
            fmt_float(row.get("space_explored_score"), 3),
            fmt_float(row.get("score_delta_vs_previous"), 3, signed=True),
            fmt_pct(row.get("score_delta_pct_vs_previous"), 1, signed=True),
            fmt_float(row.get("space_pc12_area"), 3),
        ])
    lines.append(markdown_table(
        ["stage", "n", "basins", "N_eff", "space score", "delta", "delta %", "PC area"],
        table_rows,
    ))
    lines.append("")

    lines.append("## Basin hopping detail")
    lines.append("")
    for reason in bh_reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append(markdown_table(
        ["metric", "value"],
        [
            ["BH endpoints / rows", f"{bh.get('success_total', 0)} / {bh.get('rows_total', 0)}"],
            ["BH parent seeds", bh.get("parents_total", "n/a")],
            ["non-parent successful endpoints", bh.get("non_parent_success_endpoints", "n/a")],
            ["coarse-bin changed endpoints", f"{bh.get('coarse_bin_changed_endpoints', 'n/a')} / {bh.get('coarse_bin_comparable_endpoints', 'n/a')} ({fmt_pct(bh.get('coarse_bin_changed_pct'))})"],
            ["median unique endpoint bins per parent", fmt_float(bh.get("median_unique_endpoint_bins_per_parent"), 2)],
            ["best endpoint minus parent energy", fmt_float(bh.get("best_endpoint_minus_parent_energy_kj_mol"), 2) + " kJ/mol"],
            ["new coarse bins vs initial", bh_contrib.get("new_coarse_bins_vs_baseline", "n/a")],
            ["BH representative basins", bh_contrib.get("source_representative_basins", "n/a")],
            ["BH-only basins", bh_contrib.get("source_only_basins", "n/a")],
            ["BH final survivors", survivors.get("BH_count", 0)],
        ],
    ))
    lines.append("")

    lines.append("## NMA detail")
    lines.append("")
    for reason in nma_reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append(markdown_table(
        ["metric", "value"],
        [
            ["NMA probes", nma.get("probe_pdb_count", "n/a")],
            ["NMA minimized successes / rows", f"{nma.get('success_total', 0)} / {nma.get('rows_total', 0)}"],
            ["NMA success rate", fmt_pct(nma.get("success_pct"))],
            ["NMA unique coarse bins", nma.get("unique_coarse_bins", "n/a")],
            ["NMA new coarse bins vs pre-NMA", nma.get("new_coarse_bins_vs_pre_nma", "n/a")],
            ["new coarse bins in post-NMA archive", nma_contrib.get("new_coarse_bins_vs_baseline", "n/a")],
            ["NMA representative basins", nma_contrib.get("source_representative_basins", "n/a")],
            ["NMA-only basins", nma_contrib.get("source_only_basins", "n/a")],
            ["NMA final survivors", survivors.get("NMA_count", 0)],
        ],
    ))
    lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append(
        "A stage is probably useful when it increases the archive score, adds new coarse "
        "Rg/end-to-end/contact bins, becomes a basin representative, or contributes final survivors. "
        "A stage is probably redundant when it adds many structures but gives near-zero score delta, "
        "no new coarse bins, no representative basins, and no survivors."
    )
    lines.append("")
    lines.append(
        f"Coarse-bin settings used here: Rg bin = {data['rg_bin_nm']} nm, "
        f"end-to-end bin = {data['e2e_bin_nm']} nm. Change them with --rg-bin-nm and --e2e-bin-nm."
    )
    lines.append("")
    return "\n".join(lines)


def analyze(run: Path, rg_bin_nm: float, e2e_bin_nm: float) -> dict[str, Any]:
    run = run.resolve()
    stage_rows = summarize_stage_deltas(run)

    bh_contrib = {"source": "BH", "archive_stage": "post_basin_hop", "baseline_stage": "initial_implicit"}
    if (archive_dir(run) / "post_basin_hop_points.csv").exists():
        bh_contrib = summarize_source_contribution(run, "post_basin_hop", "initial_implicit", "BH", rg_bin_nm, e2e_bin_nm)

    nma_baseline = "post_basin_hop" if (archive_dir(run) / "post_basin_hop_points.csv").exists() else "initial_implicit"
    nma_contrib = {"source": "NMA", "archive_stage": "post_nma", "baseline_stage": nma_baseline}
    if (archive_dir(run) / "post_nma_points.csv").exists():
        nma_contrib = summarize_source_contribution(run, "post_nma", nma_baseline, "NMA", rg_bin_nm, e2e_bin_nm)

    pre_nma_rows = read_points(run, nma_baseline)
    bh = summarize_bh(run, rg_bin_nm, e2e_bin_nm)
    nma = summarize_nma(run, rg_bin_nm, e2e_bin_nm, pre_nma_rows)
    survivors = summarize_final_survivors(run)

    bh_verdict = score_bh(bh_contrib, bh, survivors, stage_rows)
    nma_verdict = score_nma(nma_contrib, nma, survivors, stage_rows)

    return {
        "run": str(run),
        "rg_bin_nm": float(rg_bin_nm),
        "e2e_bin_nm": float(e2e_bin_nm),
        "stage_deltas": stage_rows,
        "bh": bh,
        "nma": nma,
        "bh_contribution": bh_contrib,
        "nma_contribution": nma_contrib,
        "final_survivors": survivors,
        "bh_verdict": bh_verdict,
        "nma_verdict": nma_verdict,
    }


def json_friendly(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, tuple):
        return [json_friendly(x) for x in obj]
    if isinstance(obj, list):
        return [json_friendly(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_friendly(v) for k, v in obj.items()}
    return obj


def generate_figures(run_dir: Path, data: dict[str, Any], out_dir: Path) -> None:
    """Generate a basic set of diagnostic plots summarizing the exploration process.

    The generated figures include stage-based line charts for archive metrics and a bar
    chart summarizing final survivor source composition. Figures are written into the
    specified output directory.

    Note: This function uses matplotlib if available. If matplotlib is not
    installed or cannot be used, this function does nothing.
    """
    # Only proceed if matplotlib is available
    if plt is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = data.get("stage_deltas", [])
    if not stage_rows:
        return
    # Extract stage names and metrics
    names: list[str] = [str(row.get("stage", "")) for row in stage_rows]
    x = list(range(len(names)))
    scores = [to_float(row.get("space_explored_score"), math.nan) for row in stage_rows]
    basins = [to_float(row.get("n_basins"), math.nan) for row in stage_rows]
    neffs = [to_float(row.get("N_eff_count"), math.nan) for row in stage_rows]
    pc_areas = [to_float(row.get("space_pc12_area"), math.nan) for row in stage_rows]
    # Stage metric plots
    def _plot_metric(values: list[float], ylabel: str, filename: str, title: str) -> None:
        plt.figure()
        plt.plot(x, values, marker="o")
        plt.xticks(x, names, rotation=45, ha="right")
        plt.xlabel("Stage")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / filename)
        plt.close()
    _plot_metric(scores, "Space explored score", "01_space_score_growth.png", "Space explored score across stages")
    _plot_metric(basins, "Number of basins", "02_basin_count_growth.png", "Number of basins across stages")
    _plot_metric(neffs, "Effective basin count", "03_effective_basin_count_growth.png", "Effective basin count across stages")
    _plot_metric(pc_areas, "PC1-2 area", "04_pc_area_growth.png", "PCA spread across stages")
    # Final survivor composition bar chart
    survivors = data.get("final_survivors", {})
    sources = ["initial", "BH", "NMA", "PCA", "other"]
    counts: list[int] = [int(survivors.get(f"{src}_count", 0)) for src in sources]
    plt.figure()
    plt.bar(sources, counts)
    plt.xlabel("Source")
    plt.ylabel("Final survivor count")
    plt.title("Final survivor counts by source")
    plt.tight_layout()
    plt.savefig(out_dir / "05_final_survivor_counts.png")
    plt.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare whether GENPEPT basin hopping and NMA added useful exploration coverage."
    )
    parser.add_argument("run_dir", type=Path, help="GENPEPT output directory")
    parser.add_argument("--rg-bin-nm", type=float, default=0.05, help="Rg coarse-bin width in nm for novelty checks")
    parser.add_argument("--e2e-bin-nm", type=float, default=0.05, help="end-to-end coarse-bin width in nm for novelty checks")
    parser.add_argument("--write-json", type=Path, default=None, help="optional JSON output path")
    parser.add_argument("--write-md", type=Path, default=None, help="optional Markdown report path")
    parser.add_argument("--write-stage-csv", type=Path, default=None, help="optional CSV of stage deltas")
    # Additional reporting options
    parser.add_argument(
        "--auto-final-report",
        action="store_true",
        help=(
            "Write a comprehensive final report to a default path in the run directory. "
            "This report includes the same bottom-line and detailed sections as the Markdown report, "
            "plus a timings and throughput section if timing data can be reconstructed."
        ),
    )
    parser.add_argument(
        "--write-final-report",
        type=Path,
        default=None,
        help="optional path to write a comprehensive final report (Markdown format)",
    )
    parser.add_argument(
        "--write-timing-csv",
        type=Path,
        default=None,
        help="optional path to write a CSV of reconstructed timing information",
    )
    # Accept write-figures for compatibility even if unused
    parser.add_argument(
        "--write-figures",
        type=Path,
        default=None,
        help="optional output directory for figure packs (currently unused)",
    )

    args = parser.parse_args(argv)

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        return 2

    # Run the core analysis
    data = analyze(run_dir, args.rg_bin_nm, args.e2e_bin_nm)
    # Build the basic report string
    report = build_report(data)
    # Always print the report to stdout
    print(report)

    # Write optional JSON file
    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(json_friendly(data), indent=2, sort_keys=True) + "\n")
        print(f"\nWrote JSON: {args.write_json}")

    # Write optional Markdown report
    if args.write_md:
        args.write_md.parent.mkdir(parents=True, exist_ok=True)
        args.write_md.write_text(report + "\n")
        print(f"Wrote Markdown: {args.write_md}")

    # Write optional stage CSV
    if args.write_stage_csv:
        write_csv_rows(args.write_stage_csv, data["stage_deltas"])
        print(f"Wrote stage CSV: {args.write_stage_csv}")

    # Helper: reconstruct stage timing information
    def _reconstruct_timings() -> list[dict[str, Any]]:
        from datetime import datetime
        stage_times: list[tuple[str, float]] = []
        # Collect modification times for known summary files
        for prefix in existing_stage_prefixes(run_dir):
            summary_file = archive_dir(run_dir) / f"{prefix}_summary.csv"
            if summary_file.exists():
                stage_times.append((prefix, summary_file.stat().st_mtime))
        # Also include final survivor and exploration space metrics if present
        final_survivor = run_dir / "final_survivor_seeds.csv"
        if final_survivor.exists():
            stage_times.append(("final_survivors", final_survivor.stat().st_mtime))
        # Sort by timestamp
        stage_times.sort(key=lambda x: x[1])
        timings: list[dict[str, Any]] = []
        # Build a lookup for n_structures per stage
        n_structures_map: dict[str, float] = {row.get("stage"): to_float(row.get("n_structures")) for row in data["stage_deltas"]}
        last_time: float | None = None
        for prefix, ts in stage_times:
            dt = datetime.utcfromtimestamp(ts)
            if last_time is None:
                duration = math.nan
                throughput = math.nan
            else:
                duration = ts - last_time
                ns = n_structures_map.get(prefix, math.nan)
                if math.isfinite(ns) and math.isfinite(duration) and duration > 0:
                    throughput = ns / duration
                else:
                    throughput = math.nan
            timings.append(
                {
                    "stage": prefix,
                    "timestamp": dt.isoformat() + "Z",
                    "duration_seconds_since_previous": duration,
                    "n_structures": n_structures_map.get(prefix, math.nan),
                    "throughput_structures_per_second": throughput,
                }
            )
            last_time = ts
        return timings

    # Compose a comprehensive final report
    def _build_final_report() -> str:
        lines = []
        # Header from the standard report
        lines.append(report)
        # Timing section
        timings = _reconstruct_timings()
        if timings:
            lines.append("## Timings and throughput")
            lines.append("")
            lines.append("| stage | timestamp (UTC) | duration vs previous | n_structures | throughput |")
            lines.append("| --- | --- | --- | --- | --- |")
            for t in timings:
                dur = t["duration_seconds_since_previous"]
                if math.isfinite(dur):
                    # Format duration as HH:MM:SS
                    hours, rem = divmod(int(dur), 3600)
                    minutes, seconds = divmod(rem, 60)
                    dur_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                else:
                    dur_str = "n/a"
                n_structures = t.get("n_structures")
                if math.isfinite(n_structures):
                    ns_str = f"{int(n_structures)}"
                else:
                    ns_str = "n/a"
                throughput = t.get("throughput_structures_per_second")
                if math.isfinite(throughput):
                    thr_str = f"{throughput:.3f}"  # structures per second
                else:
                    thr_str = "n/a"
                lines.append(
                    "| "
                    + t["stage"]
                    + " | "
                    + t["timestamp"]
                    + " | "
                    + dur_str
                    + " | "
                    + ns_str
                    + " | "
                    + thr_str
                    + " |"
                )
            lines.append("")
        else:
            lines.append("## Timings and throughput")
            lines.append("")
            lines.append("Timing data could not be reconstructed from file modification timestamps.")
            lines.append("")
        return "\n".join(lines)

    # If the user requested a final report, write it
    if args.auto_final_report or args.write_final_report:
        final_path: Path
        if args.write_final_report:
            final_path = args.write_final_report
        else:
            # Default final report path within the run directory
            final_path = run_dir / "GENPEPT_exploration_final_report.md"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_content = _build_final_report()
        final_path.write_text(final_content + "\n")
        print(f"Wrote final report: {final_path}")

    # Write timing CSV if requested
    if args.write_timing_csv:
        timing_rows = _reconstruct_timings()
        if timing_rows:
            # Convert to CSV compatible rows
            csv_rows: list[dict[str, Any]] = []
            for t in timing_rows:
                row: dict[str, Any] = {
                    "stage": t["stage"],
                    "timestamp": t["timestamp"],
                    "duration_seconds_since_previous": t["duration_seconds_since_previous"],
                    "n_structures": t["n_structures"],
                    "throughput_structures_per_second": t["throughput_structures_per_second"],
                }
                csv_rows.append(row)
            args.write_timing_csv.parent.mkdir(parents=True, exist_ok=True)
            write_csv_rows(args.write_timing_csv, csv_rows)
            print(f"Wrote timing CSV: {args.write_timing_csv}")
    # Generate figures if requested
    if args.write_figures:
        try:
            generate_figures(run_dir, data, args.write_figures)
            print(f"Wrote figures into {args.write_figures}")
        except Exception as exc:
            print(f"WARNING: failed to generate figures: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
