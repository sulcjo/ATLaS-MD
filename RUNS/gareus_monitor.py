#!/usr/bin/env python3
"""
GAREUS multi-peptide run monitor  —  stdlib only, no deps.

Usage:
  python gareus_monitor.py [RUNS_DIR] [--interval SECS] [--once]
  python gareus_monitor.py RUNS/runs2/ RUNS/v01_runs/

Interactive keys (TTY mode):
  ↑/k  ↓/j   navigate
  Enter/d     drill into selected peptide (detail view)
  m           MBAR-readiness / adaptive diagnostics view (toggles with detail)
  c           connect: live-attach view of one run (fast ~1s refresh)
  ESC/q       back / quit
  r           force refresh
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import datetime
from pathlib import Path
from typing import Optional

KT_KCAL = 0.5962   # k_B T at ~300 K, kcal/mol (matches run beta 0.4009 /kJ)

try:
    import termios
    import tty
    import select as _select
    _HAS_TTY = True
except ImportError:
    _HAS_TTY = False

# ── ANSI helpers ──────────────────────────────────────────────────────────────

class A:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    ITALIC = "\033[3m"

    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = \
        (f"\033[3{i}m" for i in range(8))
    BBLACK, BRED, BGREEN, BYELLOW, BBLUE, BMAGENTA, BCYAN, BWHITE = \
        (f"\033[9{i}m" for i in range(8))

    BG_GREY   = "\033[48;5;236m"
    BG_RESET  = "\033[49m"
    REVERSE   = "\033[7m"

def c(*codes) -> str:
    return "".join(codes)

def plain(s: str) -> str:
    """Strip ANSI codes to measure display width."""
    return re.sub(r"\033\[[0-9;]*m", "", s)

def pad(s: str, width: int, align: str = "<") -> str:
    """Pad an ANSI-colored string to display width."""
    vis = len(plain(s))
    diff = max(0, width - vis)
    if align == ">":
        return " " * diff + s
    if align == "^":
        l = diff // 2; r = diff - l
        return " " * l + s + " " * r
    return s + " " * diff

CLEAR_SCREEN  = "\033[2J\033[H"
HIDE_CURSOR   = "\033[?25l"
SHOW_CURSOR   = "\033[?25h"
MOVE_HOME     = "\033[H"
CLEAR_EOS     = "\033[J"          # clear from cursor to end of screen

# ── Phase styling ─────────────────────────────────────────────────────────────

PHASE_COLOR = {
    "gareus_production": c(A.BOLD, A.BGREEN),
    "adaptive_feedback": c(A.BOLD, A.BCYAN),
    "setup":             c(A.BOLD, A.BYELLOW),
    "equilibration":     c(A.BOLD, A.BYELLOW),
    "genpept":           c(A.BOLD, A.BBLUE),
    "done":              c(A.BOLD, A.BWHITE),
    "converged":         c(A.BOLD, A.BMAGENTA),
    "error":             c(A.BOLD, A.BRED),
    "not_started":       c(A.DIM),
    "unknown":           c(A.DIM),
}

PHASE_LABEL = {
    "gareus_production": "PRODUCTION",
    "adaptive_feedback": "ADAPT-FB",
    "setup":             "SETUP",
    "equilibration":     "EQUIL",
    "genpept":           "GENPEPT",
    "done":              "DONE ✓",
    "converged":         "CONVERGED",
    "error":             "ERROR ✗",
    "not_started":       "PENDING",
    "unknown":           "—",
}

NAME_COLOR = {
    "gareus_production": c(A.BOLD, A.BCYAN),
    "adaptive_feedback": c(A.BOLD, A.BYELLOW),
    "setup":             c(A.BOLD, A.BYELLOW),
    "equilibration":     c(A.BOLD, A.BYELLOW),
    "genpept":           c(A.BOLD, A.BBLUE),
    "done":              c(A.BOLD, A.WHITE),
    "error":             c(A.BOLD, A.BRED),
    "not_started":       c(A.DIM),
    "unknown":           c(A.DIM),
}

# ── JSONL helpers ─────────────────────────────────────────────────────────────

def tail_jsonl(path: Path, n: int = 60, chunk_bytes: int = 65536) -> list:
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        if size == 0:
            return []
        chunk = min(size, chunk_bytes)
        with open(path, "rb") as f:
            f.seek(max(0, size - chunk))
            raw = f.read()
        lines = raw.decode("utf-8", errors="replace").splitlines()
        results = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
                if len(results) >= n:
                    break
            except json.JSONDecodeError:
                continue
        return list(reversed(results))
    except Exception:
        return []


def last_progress_entry(path: Path) -> dict:
    entries = tail_jsonl(path, 60)
    for e in reversed(entries):
        if e.get("event") == "progress":
            return e
    for e in reversed(entries):
        if "percent" in e or "fraction" in e:
            return e
    return entries[-1] if entries else {}


# ── Diagnostics parse layer (pure, stdlib, testable) ───────────────────────────
#
# These functions turn the small per-epoch JSON/CSV artifacts into normalized
# dicts.  They do NO I/O timing and emit NO ANSI, so they can be unit-tested
# directly against real epoch directories.  All staleness/throttle logic lives
# in PeptideState's loaders, NOT here.

MBAR_OVERLAP_CONNECT = 0.03      # connectivity threshold the pipeline reports at
MBAR_OVERLAP_TARGET  = 0.20      # neighbor target overlap (below = weak edge)


def _safe_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _safe_rows(path: Path) -> list:
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y", "t")


def _as_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _percentile(sorted_vals: list, p: float) -> Optional[float]:
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    k = int(round((p / 100.0) * (n - 1)))
    return sorted_vals[max(0, min(n - 1, k))]


def parse_mbar_readiness(validation: Optional[dict]) -> dict:
    """Coverage + neighbor-overlap + connectivity summary.

    Source: epoch_NNN/analysis_metadata_validation.json (window-indexed).
    """
    if not validation:
        return {}
    counts = [c for c in (validation.get("sample_counts_by_window") or [])
              if isinstance(c, (int, float))]
    overlaps = [o for o in (nb.get("overlap") for nb in
                            (validation.get("neighbor_overlaps") or []))
                if isinstance(o, (int, float))]
    cs = sorted(counts)
    osr = sorted(overlaps)
    weak = validation.get("neighbor_pairs_below_target_overlap") or []
    return {
        "status":      validation.get("status"),
        "n_samples":   validation.get("n_samples"),
        "n_windows":   validation.get("n_windows"),
        "connected":   validation.get("overlap_connected_at_0p03"),
        "cov_min":     cs[0] if cs else None,
        "cov_p10":     _percentile(cs, 10),
        "cov_med":     _percentile(cs, 50),
        "cov_max":     cs[-1] if cs else None,
        "cov_zero":    sum(1 for c in counts if c == 0),
        "n_cov":       len(counts),
        "olap_min":    osr[0] if osr else None,
        "olap_p10":    _percentile(osr, 10),
        "olap_med":    _percentile(osr, 50),
        "n_overlaps":  len(osr),
        "n_weak":      len(weak),
        "sparse_2d":   validation.get("sparse_2d"),
        "explicit_2d": validation.get("explicit_2d"),
        "rectangular": validation.get("rectangular_grid"),
        "warnings":    validation.get("warnings") or [],
        "errors":      validation.get("errors") or [],
    }


def parse_epoch_edges(diagnostics: Optional[dict]) -> list:
    """Per-edge exchange/overlap list, sorted worst-overlap first.

    Source: epoch_NNN/adaptive_epoch_diagnostics.json.  Edges carry BOTH the
    epoch-local window index (wi/wj) and the persistent state id (si/sj).
    """
    if not diagnostics:
        return []
    out = []
    for e in diagnostics.get("edges") or []:
        out.append({
            "wi":      e.get("window_i"),
            "wj":      e.get("window_j"),
            "si":      e.get("state_i"),
            "sj":      e.get("state_j"),
            "overlap": e.get("overlap"),
            "acc":     e.get("exchange_acceptance"),
            "att":     e.get("exchange_attempts"),
            "type":    e.get("edge_type"),
            "warn":    e.get("warnings") or [],
        })
    out.sort(key=lambda x: (x["overlap"] is None,
                            x["overlap"] if x["overlap"] is not None else 0.0))
    return out


def parse_convergence(gate: Optional[dict]) -> dict:
    """Adaptive-feedback action pressure from adaptive_convergence_gate.json."""
    if not gate:
        return {}
    actions = gate.get("actions") or []
    return {
        "epoch":             gate.get("epoch"),
        "n_actions":         len(actions),
        "actions":           actions,
        "continue_reasons":  gate.get("continue_reasons") or [],
        "low_sample_states": gate.get("low_sample_states") or [],
        "high_boost_states": gate.get("high_boost_states") or [],
        "errors":            gate.get("errors") or [],
    }


def parse_epoch_progression(drvsumm: Optional[dict]) -> list:
    """Per-epoch trend rows from adaptive_production_driver_summary.json.

    This is the *temporal* half of the diagnostics — a single epoch snapshot
    can't show whether adaptive sampling is converging.  Watch n_active
    stabilize, n_actions fall to zero, and ns-consumed per epoch.
    """
    if not drvsumm:
        return []
    out = []
    for e in drvsumm.get("epoch_summaries") or []:
        gate = e.get("convergence_gate") or {}
        actions = gate.get("actions")
        if actions is None:
            actions = e.get("actions") or []
        epoch = e.get("epoch")
        # the embedded runtime_pool is a CUMULATIVE snapshot (it carries earlier
        # epochs' and even final/* events) — filter to this epoch's own labels
        # so consumed_ns / topups are per-epoch, not running totals.
        prefix = f"epoch_{epoch:03d}" if isinstance(epoch, int) else None
        consumed = 0.0
        n_topups = 0
        for ev in ((e.get("runtime_pool") or {}).get("events") or []):
            label = ev.get("label") or ""
            if prefix is not None and not (label == prefix or label.startswith(prefix + "/")):
                continue
            v = ev.get("consumed_ns")
            if isinstance(v, (int, float)):
                consumed += v
            if "topup" in label:
                n_topups += 1
        out.append({
            "epoch":       epoch,
            "n_active":    e.get("n_active_after_epoch"),
            "n_total":     e.get("n_total_states"),
            "n_actions":   len(actions),
            "n_topups":    n_topups,
            "consumed_ns": consumed,
        })
    out.sort(key=lambda r: (r["epoch"] is None, r["epoch"] if r["epoch"] is not None else 0))
    return out


def parse_pool_timeline(pool: Optional[dict]) -> dict:
    """Phase / topup / final timeline from adaptive_runtime_pool.json events.

    Event kinds in the pool:
      adaptive_epoch   label epoch_000                  initial epoch (1 segment)
      scheduled_epoch  label epoch_NNN/baseline|topup_* adaptive epoch + topups
      scheduled_final  label final/baseline|topup_*     final production (cycles)

    A "topup period" is any segment whose label contains 'topup' — extra
    sampling appended to a baseline pass to repair weak/under-sampled windows.
    Phases are returned in logical pipeline order (epochs ascending, final
    last) so resume-interleaved event ordering doesn't scramble the bar.
    """
    if not pool:
        return {}
    idx = {}
    for e in pool.get("events") or []:
        label = e.get("label") or ""
        kind  = e.get("kind") or ""
        ns    = e.get("consumed_ns")
        ns    = float(ns) if isinstance(ns, (int, float)) else 0.0
        is_topup = "topup" in label
        if kind == "scheduled_final" or label.startswith("final"):
            name, pkind, epnum = "final", "final", None
        else:
            m = re.match(r"epoch_(\d+)", label)
            epnum = int(m.group(1)) if m else None
            name  = f"ep{epnum}" if epnum is not None else (label.split("/")[0] or "epoch")
            pkind = "epoch"
        p = idx.get(name)
        if p is None:
            p = {"name": name, "kind": pkind, "epnum": epnum, "ns": 0.0,
                 "baseline_ns": 0.0, "topup_ns": 0.0,
                 "n_baseline": 0, "n_topups": 0, "n_segments": 0}
            idx[name] = p
        p["ns"] += ns
        p["n_segments"] += 1
        if is_topup:
            p["topup_ns"] += ns
            p["n_topups"] += 1
        else:
            p["baseline_ns"] += ns
            p["n_baseline"] += 1
    phases = sorted(idx.values(), key=lambda p: (
        p["kind"] == "final", p["epnum"] if p["epnum"] is not None else 1 << 30))
    return {
        "total_ns":     pool.get("total_ns"),
        "used_ns":      pool.get("used_ns"),
        "remaining_ns": pool.get("remaining_ns"),
        "phases":       phases,
    }


def parse_distances_samples(entries: list) -> list:
    """Per-replica (CV1, CV2, umbrella bias, GaMD boost) from 'distances' events.

    These instantaneous biased samples are the only live CV-distribution data
    the monitor can read without MBAR/parquet.
    """
    out = []
    for e in entries:
        if e.get("event") != "distances":
            continue
        for w in e.get("distances") or []:
            cv1 = w.get("primary_cv_value")
            if cv1 is None:
                cv1 = w.get("cv_A")
            cv2 = w.get("secondary_cv")
            if cv1 is None or cv2 is None:
                continue
            try:
                out.append({
                    "cv1":   float(cv1),
                    "cv2":   float(cv2),
                    "ubias": float(w.get("umbrella_bias_kcal_mol") or 0.0),
                    "boost": float(w.get("gamd_boost_total_kcal_mol") or 0.0),
                })
            except (TypeError, ValueError):
                continue
    return out


def approx_basins(samples: list, kT: float = KT_KCAL,
                  n1: int = 16, n2: int = 12, n_minima: int = 4) -> dict:
    """APPROXIMATE free-energy basins from biased CV snapshots — NO MBAR.

    Poor-man's unstratified reweight (per GPT): each biased sample is weighted
    w = exp(+β·(GaMD_boost + U_umbrella)) to undo the GaMD and umbrella bias,
    log-weights clipped to tame the exponential tail, then a CV1×CV2 weighted
    histogram gives F_rel = -kT·ln(Σw / Σw_max).  Local minima (8-neighbour) of
    F_rel are returned as basin hints.  This is a *rough* hint, not a PMF: no
    inter-window normalization, sparse snapshots, exponential-weight noise.
    """
    pts = [s for s in samples
           if isinstance(s.get("cv1"), float) and isinstance(s.get("cv2"), float)
           and math.isfinite(s["cv1"]) and math.isfinite(s["cv2"])]
    res = {"minima": [], "n_samples": len(pts), "n_bins": 0}
    if len(pts) < 3:
        return res
    beta = 1.0 / kT
    logw = [beta * (s["boost"] + s["ubias"]) for s in pts]
    sl = sorted(logw)
    cap = min(sl[len(sl) // 2] + 20.0, 50.0)
    logw = [min(x, cap) for x in logw]
    mx = max(logw)
    weights = [math.exp(x - mx) for x in logw]

    c1 = [s["cv1"] for s in pts]
    c2 = [s["cv2"] for s in pts]
    lo1, hi1 = min(c1), max(c1)
    lo2, hi2 = min(c2), max(c2)

    def _bin(v, lo, hi, n):
        if hi <= lo:
            return 0
        return max(0, min(n - 1, int((v - lo) / (hi - lo) * n * 0.9999999)))

    def _center(i, lo, hi, n):
        return lo + (i + 0.5) * (hi - lo) / n if hi > lo else lo

    grid = {}
    for s, w in zip(pts, weights):
        key = (_bin(s["cv1"], lo1, hi1, n1), _bin(s["cv2"], lo2, hi2, n2))
        grid[key] = grid.get(key, 0.0) + w
    if not grid:
        return res
    total = sum(grid.values())
    maxw = max(grid.values())
    minima = []
    for (i, j), sw in grid.items():
        is_min = True
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = grid.get((i + di, j + dj))
                if nb is not None and nb > sw:
                    is_min = False
                    break
            if not is_min:
                break
        if is_min:
            minima.append({
                "cv1":      _center(i, lo1, hi1, n1),
                "cv2":      _center(j, lo2, hi2, n2),
                "F_rel":    -kT * math.log(sw / maxw),
                "pop_frac": sw / total,
            })
    minima.sort(key=lambda m: m["F_rel"])
    res["minima"] = minima[:n_minima]
    res["n_bins"] = len(grid)
    return res


def parse_exchange_stats(dashboard: Optional[dict]) -> dict:
    """Replica-exchange health from a progress.jsonl 'dashboard' payload.

    The run TUI's exchange panel: overall Gibbs-walk acceptance plus per
    jump-distance (dw1..dwN) acceptance.  Used by the live 'connect' view.
    """
    es = (dashboard or {}).get("exchange_stats") or {}
    acc, att = es.get("accepted"), es.get("attempts")
    rate = (acc / att) if isinstance(acc, (int, float)) and att else None
    jumps = []
    for name, v in sorted((es.get("jump_bins") or {}).items()):
        a, t = (v or {}).get("accepted"), (v or {}).get("attempts")
        jumps.append({"name": name, "accepted": a, "attempts": t,
                      "rate": (a / t) if isinstance(a, (int, float)) and t else None})
    return {
        "mode":        es.get("mode"),
        "accepted":    acc,
        "attempts":    att,
        "rate":        rate,
        "gibbs_moves": es.get("gibbs_moves"),
        "gibbs_stays": es.get("gibbs_stays"),
        "jumps":       jumps,
        "n_pairs":     len(es.get("pairs") or {}),
    }


def parse_live_windows(dashboard: Optional[dict]) -> list:
    """Live window centers from the dashboard's secondary_cv.normalized_rows.

    This is the window set the run is *currently* propagating (may differ from
    the last completed epoch's MBAR window set).
    """
    rows = ((dashboard or {}).get("secondary_cv") or {}).get("normalized_rows") or []
    out = []
    for r in rows:
        out.append({
            "window":    r.get("window"),
            "cv1":       r.get("primary_cv_center"),
            "cv2":       r.get("secondary_cv_center"),
            "lifecycle": "",
        })
    return out


def parse_registry(rows: list) -> dict:
    """Window-registry churn from state_registry.csv (persistent state_id space)."""
    n_active = n_retired = n_usable = 0
    for r in rows:
        if _as_bool(r.get("active")):
            n_active += 1
        if str(r.get("retired_epoch", "")).strip():
            n_retired += 1
        if _as_bool(r.get("usable_for_mbar")):
            n_usable += 1
    return {
        "n_total":   len(rows),
        "n_active":  n_active,
        "n_retired": n_retired,
        "n_usable":  n_usable,
    }


def parse_window_centers(rows: list) -> list:
    """Window-indexed (CV1, CV2) centers from umbrella_explicit_windows.csv.

    Aligns 1:1 (by window index) with sample_counts_by_window / neighbor_overlaps.
    """
    out = []
    for r in rows:
        w = _as_float(r.get("window"))
        if w is None:
            continue
        out.append({
            "window":    int(w),
            "cv1":       _as_float(r.get("primary_center")),
            "cv2":       _as_float(r.get("secondary_cv_center")),
            "lifecycle": (r.get("lifecycle") or "").strip(),
        })
    out.sort(key=lambda d: d["window"])
    return out


def cv_topology_grid(window_centers: list, coverage: list,
                     n_rows: int, n_cols: int) -> list:
    """Place windows into an (n_rows x n_cols) grid by (CV1, CV2).

    CV1 → column (left→right, ascending).  CV2 → row (top=high, bottom=low),
    matching Ramachandran intuition.  Returns grid[row][col] = list of cell
    dicts {window, cv1, cv2, cov, lifecycle} or None.  Collisions stack in the
    cell list (renderer shows a count).  Pure: no ANSI, degenerate-axis safe.
    """
    grid = [[None] * n_cols for _ in range(n_rows)]
    pts = [w for w in window_centers
           if w.get("cv1") is not None and w.get("cv2") is not None]
    if not pts:
        return grid
    c1s = [w["cv1"] for w in pts]
    c2s = [w["cv2"] for w in pts]
    c1mn, c1mx = min(c1s), max(c1s)
    c2mn, c2mx = min(c2s), max(c2s)

    def _col(v):
        if c1mx == c1mn:
            return n_cols // 2
        return max(0, min(n_cols - 1,
                          int(round((v - c1mn) / (c1mx - c1mn) * (n_cols - 1)))))

    def _row(v):
        if c2mx == c2mn:
            return n_rows // 2
        # invert: high CV2 at top
        return max(0, min(n_rows - 1,
                          int(round((c2mx - v) / (c2mx - c2mn) * (n_rows - 1)))))

    for w in pts:
        wi = w["window"]
        cov = coverage[wi] if (coverage and 0 <= wi < len(coverage)) else None
        cell = {"window": wi, "cv1": w["cv1"], "cv2": w["cv2"],
                "cov": cov, "lifecycle": w.get("lifecycle", "")}
        r, c = _row(w["cv2"]), _col(w["cv1"])
        if grid[r][c] is None:
            grid[r][c] = [cell]
        else:
            grid[r][c].append(cell)
    return grid


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_dur(s: Optional[float]) -> str:
    if s is None or s < 0:
        return "—"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d{h:02d}h"
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s"


def bar(pct: float, width: int = 14) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    if pct >= 80:
        col = A.BGREEN
    elif pct >= 40:
        col = A.BYELLOW
    elif pct > 0:
        col = A.BRED
    else:
        col = A.DIM
    return (
        c(col) + "█" * filled
        + c(A.DIM) + "░" * (width - filled)
        + A.RESET
    )


# Segmented timeline bar — epochs cycle through this palette; final is distinct.
_PHASE_PALETTE = (A.BCYAN, A.BBLUE, A.BGREEN, A.BYELLOW, A.BWHITE)
_FINAL_COLOR   = A.BMAGENTA


def _phase_color(p: dict, epoch_order_index: int) -> str:
    if p.get("kind") == "final":
        return _FINAL_COLOR
    return _PHASE_PALETTE[epoch_order_index % len(_PHASE_PALETTE)]


def render_progress_bar(timeline: dict, width: int = 60):
    """Segmented total-progress bar over the ns budget.

    Returns (bar, legend).  Each phase is a coloured run; within a phase the
    baseline pass renders as solid '█' and topup periods as shaded '▓', so
    topups are visible as a texture change.  Unused budget is dim '░'.
    """
    phases = (timeline or {}).get("phases") or []
    if not phases:
        return (c(A.DIM) + "░" * width + A.RESET,
                c(A.DIM) + "no runtime-pool data" + A.RESET)

    total = timeline.get("total_ns")
    used  = sum(p["ns"] for p in phases)
    denom = total if isinstance(total, (int, float)) and total > 0 else used
    if not denom:
        return (c(A.DIM) + "░" * width + A.RESET,
                c(A.DIM) + "no consumption yet" + A.RESET)

    cells = []                       # (glyph, color)
    epoch_i = 0
    for p in phases:
        col = _phase_color(p, epoch_i)
        if p["kind"] == "epoch":
            epoch_i += 1
        if p["ns"] <= 0:
            continue
        nchars = max(1, int(round(p["ns"] / denom * width)))
        base_chars = int(round(p["baseline_ns"] / p["ns"] * nchars)) if p["ns"] else nchars
        for i in range(nchars):
            cells.append(("█" if i < base_chars else "▓", col))
    cells = cells[:width]
    while len(cells) < width:
        cells.append(("░", A.DIM))
    bar_str = "".join(c(col) + g for g, col in cells) + A.RESET

    parts = []
    epoch_i = 0
    for p in phases:
        col = _phase_color(p, epoch_i)
        if p["kind"] == "final":
            seg = (c(A.BOLD, col) + "FINAL" + A.RESET
                   + c(A.DIM) + f" {p['ns']:.0f}ns" + A.RESET)
            if p["n_topups"]:
                seg += c(A.DIM) + f" (+{p['n_topups']}↑)" + A.RESET
        else:
            epoch_i += 1
            seg = (c(A.BOLD, col) + p["name"] + A.RESET
                   + c(A.DIM) + f" {p['ns']:.0f}ns" + A.RESET)
            if p["n_topups"]:
                seg += c(A.DIM) + f" (+{p['n_topups']} topup)" + A.RESET
        parts.append(seg)
    rem = timeline.get("remaining_ns")
    if isinstance(rem, (int, float)) and rem > 0.5:
        parts.append(c(A.DIM) + f"free {rem:.0f}ns" + A.RESET)
    return bar_str, "  ".join(parts)


_SPARK = "▁▂▃▄▅▆▇█"

def sparkline(values: list, width: int = 60) -> str:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return c(A.DIM) + "─" * width + A.RESET
    mn, mx = min(clean), max(clean)
    if mx == mn:
        return c(A.DIM) + "─" * width + A.RESET
    n = len(clean)
    idxs = [int(round(i * (n - 1) / (width - 1))) for i in range(width)]
    chars = []
    for i in idxs:
        v = clean[i]
        idx = min(7, int((v - mn) / (mx - mn) * 8))
        chars.append(c(A.BGREEN) + _SPARK[idx])
    return "".join(chars) + A.RESET


# ── Per-peptide state ─────────────────────────────────────────────────────────

class PeptideState:
    def _init_cache(self):
        self._prog: dict          = {}
        self._drvsumm: dict       = {}
        self._pool: dict          = {}
        self._mtime_prog: float   = 0.0
        self._mtime_drv: float    = 0.0
        self._mtime_pool: float   = 0.0
        self._epoch_count         = None
        self._ckpt_count          = None
        self._ckpt_refresh: float = 0.0
        self._genpept_status      = None
        self._genpept_surv        = None
        self._total_ep_cache      = None
        self._total_ep_mtime: float = 0.0
        # diagnostics (MBAR readiness) — parsed from the latest COMPLETED epoch
        self._diag: dict          = {}
        self._diag_epoch_dir      = None
        self._diag_epoch_idx      = None
        self._diag_scan_refresh: float = 0.0
        self._mtime_diag: float   = 0.0
        self._registry: dict      = {}
        self._mtime_reg: float    = 0.0

    def __init__(self, name: str, base_dir: Path):
        self.name        = name
        self.base_dir    = base_dir
        self.run_dir     = base_dir / f"{name}_2d_run"
        self.genpept_dir = base_dir / f"{name}_genpept"
        self._init_cache()

    @classmethod
    def from_rundir(cls, run_dir: Path) -> "PeptideState":
        obj = object.__new__(cls)
        obj.name        = run_dir.name
        obj.base_dir    = run_dir.parent
        obj.run_dir     = run_dir
        obj.genpept_dir = run_dir.parent / f"{run_dir.name}_genpept"
        obj._init_cache()
        return obj

    def _load_progress(self):
        p = self.run_dir / "progress.jsonl"
        if not p.exists():
            return
        mt = p.stat().st_mtime
        if mt == self._mtime_prog:
            return
        self._mtime_prog = mt
        self._prog = last_progress_entry(p)

    def _load_driver_summary(self):
        p = self.run_dir / "adaptive_production" / "adaptive_production_driver_summary.json"
        if not p.exists():
            return
        mt = p.stat().st_mtime
        if mt == self._mtime_drv:
            return
        self._mtime_drv = mt
        try:
            with open(p) as f:
                self._drvsumm = json.load(f)
        except Exception:
            pass

    def _load_epochs(self):
        ap = self.run_dir / "adaptive_production"
        if not ap.exists():
            return
        self._epoch_count = sum(1 for d in ap.iterdir()
                                if d.is_dir() and d.name.startswith("epoch_"))

    def _load_checkpoints(self):
        ap = self.run_dir / "adaptive_production"
        if not ap.exists():
            self._ckpt_count = 0
            return
        now = time.time()
        if self._ckpt_count is not None and now - self._ckpt_refresh < 60:
            return
        self._ckpt_refresh = now
        count = 0
        for root, dirs, files in os.walk(ap):
            for f in files:
                if f == "production_checkpoint_manifest.json":
                    count += 1
        self._ckpt_count = count

    def _load_genpept(self):
        if self._genpept_status is not None:
            return
        if not self.genpept_dir.exists():
            self._genpept_status = "pending"
            return
        seeds = self.genpept_dir / "final_survivor_seeds.csv"
        if seeds.exists():
            self._genpept_status = "done"
            try:
                with open(seeds) as f:
                    self._genpept_surv = max(0, sum(1 for _ in f) - 1)
            except Exception:
                pass
        elif any(self.genpept_dir.iterdir()):
            self._genpept_status = "running"
        else:
            self._genpept_status = "pending"

    def _load_runtime_pool(self):
        p = self.run_dir / "adaptive_production" / "adaptive_runtime_pool.json"
        if not p.exists():
            return
        mt = p.stat().st_mtime
        if mt == self._mtime_pool:
            return
        self._mtime_pool = mt
        try:
            with open(p) as f:
                self._pool = json.load(f)
        except Exception:
            pass

    def _load_diagnostics(self):
        """Parse MBAR-readiness diagnostics from the latest COMPLETED epoch.

        Cost discipline (runs over ~50 peptides every tick):
          * the epoch-dir scan to locate the newest epoch carrying
            analysis_metadata_validation.json is throttled to once / 60 s;
          * the actual JSON/CSV parse is gated on the validation file's mtime,
            so a completed epoch is parsed exactly once and is free thereafter.
        """
        ap = self.run_dir / "adaptive_production"
        if not ap.exists():
            return
        now = time.time()
        if self._diag_epoch_dir is None or (now - self._diag_scan_refresh) >= 60:
            self._diag_scan_refresh = now
            latest = None
            latest_idx = -1
            try:
                for d in ap.iterdir():
                    if not (d.is_dir() and d.name.startswith("epoch_")):
                        continue
                    if not (d / "analysis_metadata_validation.json").exists():
                        continue
                    try:
                        idx = int(d.name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    if idx > latest_idx:
                        latest_idx, latest = idx, d
            except OSError:
                return
            self._diag_epoch_dir = latest
            self._diag_epoch_idx = latest_idx if latest_idx >= 0 else None

        ed = self._diag_epoch_dir
        if ed is None:
            return
        val_path = ed / "analysis_metadata_validation.json"
        try:
            mt = val_path.stat().st_mtime
        except OSError:
            return
        if mt == self._mtime_diag and self._diag:
            return
        self._mtime_diag = mt

        validation = _safe_json(val_path)
        diagnostics = _safe_json(ed / "adaptive_epoch_diagnostics.json")
        gate = _safe_json(ed / "adaptive_convergence_gate.json")
        centers = parse_window_centers(_safe_rows(ed / "umbrella_explicit_windows.csv"))
        self._diag = {
            "epoch":      self._diag_epoch_idx,
            "readiness":  parse_mbar_readiness(validation),
            "edges":      parse_epoch_edges(diagnostics),
            "convergence": parse_convergence(gate),
            "centers":    centers,
            "coverage":   (validation or {}).get("sample_counts_by_window") or [],
        }

    def _load_registry(self):
        p = self.run_dir / "adaptive_production" / "state_registry.csv"
        if not p.exists():
            return
        try:
            mt = p.stat().st_mtime
        except OSError:
            return
        if mt == self._mtime_reg and self._registry:
            return
        self._mtime_reg = mt
        self._registry = parse_registry(_safe_rows(p))

    def refresh(self):
        self._load_progress()
        self._load_driver_summary()
        self._load_epochs()
        self._load_checkpoints()
        self._load_genpept()
        self._load_runtime_pool()
        self._load_diagnostics()
        self._load_registry()

    @property
    def phase(self) -> str:
        if not self.run_dir.exists():
            return "genpept" if self._genpept_status == "running" else "not_started"
        if self._drvsumm.get("status") == "done":
            return "done"
        return self._prog.get("phase") or "unknown"

    @property
    def percent(self) -> Optional[float]:
        v = self._prog.get("percent")
        if v is not None:
            return float(v)
        f = self._prog.get("fraction")
        return float(f) * 100 if f is not None else None

    @property
    def eta_s(self) -> Optional[float]:
        return self._prog.get("eta_s")

    @property
    def elapsed_s(self) -> Optional[float]:
        return self._prog.get("wall_elapsed_s") or self._prog.get("elapsed_s")

    @property
    def ns_per_day(self) -> Optional[float]:
        return self._prog.get("aggregate_ns_per_day") or self._prog.get("ns_per_day")

    @property
    def agg_ns(self) -> Optional[float]:
        return self._prog.get("aggregate_sim_time_ns")

    @property
    def steps_per_s(self) -> Optional[float]:
        return self._prog.get("steps_per_s")

    @property
    def cv_range(self) -> str:
        mn = self._prog.get("cv_min_A")
        mx = self._prog.get("cv_max_A")
        if mn is not None and mx is not None:
            return f"{mn:.3f}–{mx:.3f}"
        return "—"

    @property
    def gamd_boost(self) -> Optional[float]:
        return self._prog.get("gamd_boost_mean_kcal_mol")

    @property
    def n_replicas(self) -> Optional[int]:
        m = re.search(r"(\d+)\s+replica", self._prog.get("message", ""))
        return int(m.group(1)) if m else None

    @property
    def exchange_attempts(self) -> Optional[int]:
        return self._prog.get("exchange_attempts")

    @property
    def epochs_completed(self) -> Optional[int]:
        v = self._drvsumm.get("epochs_completed")
        return v if v is not None else self._epoch_count

    @property
    def total_epochs(self) -> Optional[int]:
        cfg = self.run_dir / "effective_config.json"
        if not cfg.exists():
            return None
        try:
            mt = cfg.stat().st_mtime
            if self._total_ep_cache is not None and mt == self._total_ep_mtime:
                return self._total_ep_cache
            self._total_ep_mtime = mt
            with open(cfg) as f:
                d = json.load(f)
            v = d.get("ap_epochs") or d.get("adaptive_production", {}).get("ap_epochs")
            self._total_ep_cache = int(v) if v is not None else None
            return self._total_ep_cache
        except Exception:
            return None

    @property
    def total_budget_ns(self) -> Optional[float]:
        v = self._pool.get("total_ns")
        return float(v) if v else None

    @property
    def committed_ns(self) -> Optional[float]:
        v = self._pool.get("used_ns")
        return float(v) if v is not None else None

    @property
    def global_percent(self) -> Optional[float]:
        total = self.total_budget_ns
        if total is None or total == 0:
            return None
        committed = self.committed_ns or 0.0
        current   = self.agg_ns or 0.0
        return min(100.0, (committed + current) / total * 100)

    @property
    def global_eta_s(self) -> Optional[float]:
        total    = self.total_budget_ns
        nspd     = self.ns_per_day
        if total is None or not nspd:
            return None
        committed = self.committed_ns or 0.0
        current   = self.agg_ns or 0.0
        remaining_ns = total - committed - current
        if remaining_ns <= 0:
            return 0.0
        return remaining_ns / nspd * 86400

    # ── quality signals ───────────────────────────────────────────────────────

    @property
    def gamd_anharmonicity(self) -> Optional[float]:
        return self._prog.get("gamd_boost_anharmonicity_score")

    @property
    def gamd_boost_sd(self) -> Optional[float]:
        return self._prog.get("gamd_boost_sd_kcal_mol")

    @property
    def ubias_mean(self) -> Optional[float]:
        return self._prog.get("umbrella_bias_mean_kcal_mol")

    @property
    def quality_grade(self) -> str:
        anh = self.gamd_anharmonicity
        sd  = self.gamd_boost_sd
        if anh is None and sd is None:
            return "?"
        bad  = (anh is not None and anh > 0.5) or (sd is not None and sd > 5.0)
        warn = (anh is not None and anh > 0.3) or (sd is not None and sd > 3.0)
        if bad:
            return "BAD"
        if warn:
            return "WARN"
        return "OK"

    @property
    def stale(self) -> bool:
        wt = self._prog.get("wall_time_s")
        return wt is not None and (time.time() - wt) > 900

    @property
    def latest_message(self) -> str:
        return self._prog.get("message", "")

    # ── MBAR-readiness diagnostics (from latest completed epoch) ───────────────

    @property
    def readiness(self) -> dict:
        return self._diag.get("readiness") or {}

    @property
    def diag_epoch(self) -> Optional[int]:
        return self._diag.get("epoch")

    @property
    def mbar_connected(self) -> Optional[bool]:
        return self.readiness.get("connected")

    @property
    def mbar_n_weak(self) -> Optional[int]:
        return self.readiness.get("n_weak")

    @property
    def mbar_min_overlap(self) -> Optional[float]:
        return self.readiness.get("olap_min")

    @property
    def mbar_cov_min(self) -> Optional[int]:
        return self.readiness.get("cov_min")

    @property
    def mbar_grade(self) -> str:
        """One-glance MBAR readiness: OK / WARN / BAD / ? (no data)."""
        r = self.readiness
        if not r:
            return "?"
        conn = r.get("connected")
        n_weak = r.get("n_weak") or 0
        cov_zero = r.get("cov_zero") or 0
        if conn is False or cov_zero > 0:
            return "BAD"
        if n_weak > 0 or (r.get("olap_min") is not None and r["olap_min"] < MBAR_OVERLAP_TARGET):
            return "WARN"
        if conn is True:
            return "OK"
        return "?"

    @property
    def worst_edges(self) -> list:
        return self._diag.get("edges") or []

    @property
    def convergence(self) -> dict:
        return self._diag.get("convergence") or {}

    @property
    def window_centers(self) -> list:
        return self._diag.get("centers") or []

    @property
    def coverage(self) -> list:
        return self._diag.get("coverage") or []

    @property
    def registry(self) -> dict:
        return self._registry or {}

    @property
    def epoch_progression(self) -> list:
        return parse_epoch_progression(self._drvsumm)

    @property
    def pool_timeline(self) -> dict:
        return parse_pool_timeline(self._pool)

    def ns_history(self, n: int = 500) -> list:
        p = self.run_dir / "progress.jsonl"
        if not p.exists():
            return []
        entries = tail_jsonl(p, n)
        return [float(e["aggregate_sim_time_ns"])
                for e in entries
                if e.get("aggregate_sim_time_ns") is not None]

    _LIVE_FIELDS = (
        "phase", "step", "total_steps", "percent", "fraction",
        "aggregate_ns_per_day", "ns_per_day", "aggregate_sim_time_ns",
        "eta_s", "elapsed_s", "wall_time_s", "message", "steps_per_s",
        "cv_min_A", "cv_mean_A", "cv_max_A",
        "secondary_cv_min", "secondary_cv_mean", "secondary_cv_max",
        "gamd_boost_mean_kcal_mol", "gamd_boost_sd_kcal_mol",
        "gamd_boost_anharmonicity_score", "umbrella_bias_mean_kcal_mol",
    )

    def live_snapshot(self, chunk_bytes: int = 1 << 20) -> dict:
        """Freshest-per-field merge of recent progress.jsonl entries (connect view).

        progress.jsonl interleaves lightweight throughput 'progress' lines,
        periodic rich lines (cv/boost/dashboard), and log lines.  No single
        entry is complete, so we walk the tail newest→oldest and take the first
        non-null value for each field, recording its age (in entries) and the
        freshest exchange 'dashboard' payload with its wall-clock age.
        """
        p = self.run_dir / "progress.jsonl"
        if not p.exists():
            return {}
        entries = tail_jsonl(p, 1_000_000, chunk_bytes=chunk_bytes)
        snap = {f: None for f in self._LIVE_FIELDS}
        ages = {f: None for f in self._LIVE_FIELDS}
        dash = None
        dash_wall = None
        newest_wall = None
        for age, e in enumerate(reversed(entries)):
            if newest_wall is None and e.get("wall_time_s") is not None:
                newest_wall = e.get("wall_time_s")
            for f in self._LIVE_FIELDS:
                if snap[f] is None and e.get(f) is not None:
                    snap[f] = e[f]
                    ages[f] = age
            if dash is None and e.get("dashboard"):
                dash = e["dashboard"]
                dash_wall = e.get("wall_time_s")
        snap["_ages"] = ages
        snap["_n"] = len(entries)
        snap["_dist_samples"] = parse_distances_samples(entries)[-6000:]
        snap["_dashboard"] = dash
        snap["_dashboard_age"] = None
        snap["_dashboard_age_s"] = None
        if dash is not None:
            # age in entries
            for age, e in enumerate(reversed(entries)):
                if e.get("dashboard"):
                    snap["_dashboard_age"] = age
                    break
            if newest_wall is not None and dash_wall is not None:
                snap["_dashboard_age_s"] = max(0.0, newest_wall - dash_wall)
        return snap


# ── Discovery ─────────────────────────────────────────────────────────────────

_IGNORE = {
    ".git", ".claude", ".codex", ".remember", ".agents", "__pycache__",
    "alpha_runs", "validation", "gamd-openmm", "gareus",
}


def _is_flat_rundir(d: Path) -> bool:
    """True if d itself is a GAREUS run dir (not a peptide container)."""
    return (d / "adaptive_production").is_dir() or (d / "progress.jsonl").exists()


def discover(runs_dir: Path) -> list:
    out = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        n = d.name
        if n.startswith(".") or n in _IGNORE:
            continue
        if n.startswith("CONV") or n.startswith("alpha"):
            continue

        # peptide container: PEPTIDE/PEPTIDE_2d_run  or  PEPTIDE/PEPTIDE.yaml
        if (d / f"{n}.yaml").exists() or (d / f"{n}_2d_run").exists():
            out.append(PeptideState(n, d))
            continue

        # flat run dir: dir itself IS the run output (e.g. chignolin_2d_run7)
        if _is_flat_rundir(d):
            out.append(PeptideState.from_rundir(d))
    return out


# ── Rendering ─────────────────────────────────────────────────────────────────

def term_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 120


# column widths (display chars)
COL = {
    "name":    14,
    "genpept":  9,
    "phase":   11,
    "bar":     16,
    "gpct":     7,
    "epct":     6,
    "budget":  11,
    "epoch":    5,
    "ckpt":     5,
    "mbar":     7,
    "elapsed":  8,
    "geta":     9,
    "nspd":     7,
    "sps":      7,
    "cv":      14,
    "qual":     4,
    "nrep":     4,
    "gamd_mu":  7,
    "gamd_sd":  6,
    "anharm":   7,
    "ubias":    6,
}

HEADERS = {
    "name":    "Peptide",
    "genpept": "GENPEPT",
    "phase":   "Phase",
    "bar":     "Total progress",
    "gpct":    "G%",
    "epct":    "E%",
    "budget":  "ns/budget",
    "epoch":   "Epoch",
    "ckpt":    "Ckpts",
    "mbar":    "MBAR",
    "elapsed": "Elapsed",
    "geta":    "Global ETA",
    "nspd":    "ns/day",
    "sps":     "Steps/s",
    "cv":      "CV range",
    "qual":    "Qual",
    "nrep":    "Rep",
    "gamd_mu": "Boost μ",
    "gamd_sd": "Boost σ",
    "anharm":  "Anharm",
    "ubias":   "Ubias",
}

ALIGNS = {
    "name":    "<",
    "genpept": "^",
    "phase":   "^",
    "bar":     "<",
    "gpct":    ">",
    "epct":    ">",
    "budget":  ">",
    "epoch":   ">",
    "ckpt":    ">",
    "mbar":    "^",
    "elapsed": ">",
    "geta":    ">",
    "nspd":    ">",
    "sps":     ">",
    "cv":      ">",
    "qual":    "^",
    "nrep":    ">",
    "gamd_mu": ">",
    "gamd_sd": ">",
    "anharm":  ">",
    "ubias":   ">",
}

COLS = list(COL.keys())


def hline(char: str = "─", cross: str = "┼",
          left: str = "├", right: str = "┤") -> str:
    segs = [char * (COL[k] + 2) for k in COLS]
    return left + (cross + "").join(segs) + right


def header_row() -> str:
    cells = []
    for k in COLS:
        h = c(A.BOLD, A.BG_GREY, A.WHITE) + pad(HEADERS[k], COL[k], ALIGNS[k]) + A.RESET
        cells.append(" " + h + " ")
    return "│" + "│".join(cells) + "│"


def data_row(s: PeptideState, selected: bool = False) -> str:
    phase = s.phase
    pcol  = PHASE_COLOR.get(phase, A.DIM)
    ncol  = NAME_COLOR.get(phase, A.DIM)

    gpct = s.global_percent
    epct = s.percent

    stale = c(A.DIM, A.ITALIC, " ~") + A.RESET if s.stale else ""
    name_val = c(ncol) + s.name[:COL["name"]] + A.RESET + stale

    if s._genpept_status == "done":
        gp_val = c(A.BOLD, A.BGREEN) + f"✓ {s._genpept_surv or '?'}" + A.RESET
    elif s._genpept_status == "running":
        gp_val = c(A.BBLUE) + "run…" + A.RESET
    else:
        gp_val = c(A.DIM) + "—" + A.RESET

    plabel = PHASE_LABEL.get(phase, phase.upper()[:10])
    phase_val = c(pcol) + plabel + A.RESET

    display_pct = gpct if gpct is not None else epct
    bar_val = (bar(display_pct, COL["bar"]) if display_pct is not None
               else c(A.DIM) + "░" * COL["bar"] + A.RESET)

    if gpct is not None:
        gcol = A.BGREEN if gpct >= 80 else A.BYELLOW if gpct >= 40 else A.BRED
        gpct_val = c(A.BOLD, gcol) + f"{gpct:.1f}%" + A.RESET
    else:
        gpct_val = c(A.DIM) + "—" + A.RESET

    if epct is not None:
        epct_val = c(A.DIM) + f"{epct:.0f}%" + A.RESET
    else:
        epct_val = c(A.DIM) + "—" + A.RESET

    total     = s.total_budget_ns
    committed = s.committed_ns or 0.0
    current   = s.agg_ns or 0.0
    done_ns   = committed + current
    if total:
        budget_val = f"{done_ns:.0f}/{total:.0f}"
    elif current:
        budget_val = f"{current:.1f} ns"
    else:
        budget_val = "—"

    ep = s.epochs_completed
    ep_val = str(ep) if ep is not None else "—"

    ck = s._ckpt_count
    ck_val = str(ck) if ck is not None else "—"

    mgrade = s.mbar_grade
    if mgrade == "?":
        mbar_val = c(A.DIM) + "—" + A.RESET
    else:
        nw = s.mbar_n_weak or 0
        if mgrade == "BAD":
            sym = "✗" if s.mbar_connected is False else "!"
            mbar_val = c(A.BOLD, A.BRED) + f"{sym}{nw}w" + A.RESET
        elif mgrade == "WARN":
            mbar_val = c(A.BYELLOW) + f"✓{nw}w" + A.RESET
        else:
            mbar_val = c(A.BGREEN) + "✓" + A.RESET

    geta = s.global_eta_s
    if geta is None:
        geta = s.eta_s
    if geta is not None and geta <= 0:
        geta_val = c(A.BOLD, A.BGREEN) + "done" + A.RESET
    else:
        geta_val = fmt_dur(geta)

    nspd = s.ns_per_day
    sps  = s.steps_per_s

    grade = s.quality_grade
    if grade == "BAD":
        qual_val = c(A.BOLD, A.BRED) + "BAD" + A.RESET
    elif grade == "WARN":
        qual_val = c(A.BOLD, A.BYELLOW) + "WARN" + A.RESET
    elif grade == "OK":
        qual_val = c(A.BOLD, A.BGREEN) + "OK" + A.RESET
    else:
        qual_val = c(A.DIM) + "?" + A.RESET

    nrep = s.n_replicas
    nrep_val = str(nrep) if nrep else "—"

    mu = s.gamd_boost
    mu_val = f"{mu:.1f}" if mu is not None else "—"

    sd = s.gamd_boost_sd
    if sd is not None:
        sd_col = A.BRED if sd > 5 else A.BYELLOW if sd > 3 else A.BGREEN
        sd_val = c(sd_col) + f"{sd:.1f}" + A.RESET
    else:
        sd_val = c(A.DIM) + "—" + A.RESET

    anh = s.gamd_anharmonicity
    if anh is not None:
        anh_col = A.BRED if anh > 0.5 else A.BYELLOW if anh > 0.3 else A.BGREEN
        anh_val = c(anh_col) + f"{anh:.2f}" + A.RESET
    else:
        anh_val = c(A.DIM) + "—" + A.RESET

    ub = s.ubias_mean
    ub_val = f"{ub:.1f}" if ub is not None else "—"

    vals = {
        "name":    name_val,
        "genpept": gp_val,
        "phase":   phase_val,
        "bar":     bar_val,
        "gpct":    gpct_val,
        "epct":    epct_val,
        "budget":  budget_val,
        "epoch":   ep_val,
        "ckpt":    ck_val,
        "mbar":    mbar_val,
        "elapsed": fmt_dur(s.elapsed_s),
        "geta":    geta_val,
        "nspd":    f"{nspd:.0f}" if nspd else "—",
        "sps":     f"{sps:.0f}"  if sps  else "—",
        "cv":      s.cv_range,
        "qual":    qual_val,
        "nrep":    nrep_val,
        "gamd_mu": mu_val,
        "gamd_sd": sd_val,
        "anharm":  anh_val,
        "ubias":   ub_val,
    }

    cells = []
    for k in COLS:
        cells.append(" " + pad(vals[k], COL[k], ALIGNS[k]) + " ")

    if selected:
        border = c(A.BOLD, A.BYELLOW) + "▶" + A.RESET
    else:
        border = "│"
    return border + "│".join(cells) + "│"


def summary_line(states: list) -> str:
    n_prod  = sum(1 for s in states if s.phase == "gareus_production")
    n_adapt = sum(1 for s in states if s.phase == "adaptive_feedback")
    n_done  = sum(1 for s in states if s.phase in ("done", "converged"))
    n_err   = sum(1 for s in states if s.phase == "error")
    n_pend  = sum(1 for s in states if s.phase in ("not_started", "genpept", "unknown"))

    total_budget = sum(s.total_budget_ns for s in states if s.total_budget_ns)
    committed    = sum((s.committed_ns or 0) + (s.agg_ns or 0) for s in states)
    global_pct   = committed / total_budget * 100 if total_budget else None

    parts = []
    if n_prod:  parts.append(c(A.BOLD, A.BGREEN)  + f"{n_prod} production" + A.RESET)
    if n_adapt: parts.append(c(A.BOLD, A.BCYAN)   + f"{n_adapt} adapting"  + A.RESET)
    if n_done:  parts.append(c(A.BOLD, A.BWHITE)  + f"{n_done} done"       + A.RESET)
    if n_err:   parts.append(c(A.BOLD, A.BRED)    + f"{n_err} error"       + A.RESET)
    if n_pend:  parts.append(c(A.DIM)             + f"{n_pend} pending"    + A.RESET)
    if total_budget:
        pct_str = f"{global_pct:.1f}%" if global_pct is not None else "?%"
        parts.append(
            c(A.DIM) + f"fleet: {committed:.0f}/{total_budget:.0f} ns  " + A.RESET
            + c(A.BOLD, A.BYELLOW) + pct_str + A.RESET
        )
    return "  ".join(parts)


def render(states: list, selected: Optional[int] = None) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    title = (
        c(A.BOLD, A.BCYAN) + " GAREUS Monitor " + A.RESET
        + c(A.DIM) + now + A.RESET
    )

    top = "┌" + hline("─", "┬", "┌", "┐")[1:-1] + "┐"
    mid = hline("─", "┼", "├", "┤")
    bot = "└" + hline("─", "┴", "└", "┘")[1:-1] + "┘"

    if selected is not None:
        hint = c(A.DIM) + "  ↑↓/jk select   Enter detail   m MBAR diag   c connect (live)   r refresh   q quit" + A.RESET
    else:
        hint = c(A.DIM) + "  q/Ctrl-C quit   r refresh now" + A.RESET

    lines = [
        "",
        "  " + title,
        "",
        top,
        header_row(),
        mid,
    ]
    for i, s in enumerate(states):
        lines.append(data_row(s, selected=(selected is not None and selected == i)))
        if i < len(states) - 1:
            lines.append(mid)
    lines += [
        bot,
        "",
        "  " + summary_line(states),
        "",
        hint,
    ]
    return "\n".join(lines)


def _clip(content: str, width: int) -> str:
    """Truncate an ANSI string to `width` visible chars, preserving escape codes.

    Zero-width ANSI sequences are kept; only visible characters past the limit
    are dropped, and a RESET is appended so a cut inside a colored span can't
    bleed colour into the box border.
    """
    if len(plain(content)) <= width:
        return content
    out, vis, i, n = [], 0, 0, len(content)
    while i < n and vis < width:
        if content[i] == "\033":
            j = content.find("m", i)
            if j == -1:
                break
            out.append(content[i:j + 1])
            i = j + 1
        else:
            out.append(content[i])
            vis += 1
            i += 1
    out.append(A.RESET)
    return "".join(out)


def _fill(content: str, width: int) -> str:
    """Clip (if over) then pad an ANSI string to exactly `width` visible chars."""
    content = _clip(content, width)
    vis = len(plain(content))
    return content + " " * max(0, width - vis)


def render_detail(s: PeptideState) -> str:
    tw    = min(term_width(), 100)
    inner = tw - 2

    phase  = s.phase
    pcol   = PHASE_COLOR.get(phase, A.DIM)
    plabel = PHASE_LABEL.get(phase, phase.upper())

    ep       = s.epochs_completed
    total_ep = s.total_epochs
    if ep is not None and total_ep:
        ep_str = f"epoch {ep}/{total_ep}"
    elif ep is not None:
        ep_str = f"epoch {ep}"
    else:
        ep_str = ""

    grade     = s.quality_grade
    grade_col = A.BRED if grade == "BAD" else A.BYELLOW if grade == "WARN" else A.BGREEN

    stale_tag = c(A.DIM, A.ITALIC) + " [stale]" + A.RESET if s.stale else ""
    header_content = (
        c(A.BOLD) + f" {s.name} " + A.RESET
        + c(pcol) + plabel + A.RESET + "  "
        + c(A.DIM) + ep_str + A.RESET
        + ("  " if ep_str else "")
        + c(A.DIM) + fmt_dur(s.elapsed_s) + A.RESET
        + stale_tag + "  "
        + c(A.BOLD, grade_col) + grade + A.RESET
    )

    top_border  = "┌" + "─" * inner + "┐"
    header_line = "│" + _fill(header_content, inner) + "│"
    sep         = "├" + "─" * inner + "┤"
    bot_border  = "└" + "─" * inner + "┘"

    LW = 14  # label width

    def labeled(label: str, value: str) -> str:
        lpart = c(A.DIM) + f"  {label:<{LW}}" + A.RESET
        return "│" + _fill(lpart + value, inner) + "│"

    def blank() -> str:
        return "│" + " " * inner + "│"

    lines = [top_border, header_line, sep]

    # pool progress
    bar_w     = max(20, inner - 50)
    total     = s.total_budget_ns
    committed = s.committed_ns or 0.0
    current   = s.agg_ns or 0.0
    done_ns   = committed + current
    if total:
        gpct     = min(100.0, done_ns / total * 100)
        eta      = s.global_eta_s
        eta_str  = f"  ETA {fmt_dur(eta)}" if eta is not None else ""
        pool_val = (bar(gpct, bar_w) + "  "
                    + c(A.BOLD) + f"{done_ns:.1f}" + A.RESET
                    + c(A.DIM) + f" / {total:.0f} ns  ({gpct:.1f}%){eta_str}" + A.RESET)
    elif current:
        pool_val = (c(A.DIM) + "─" * bar_w + A.RESET
                    + c(A.DIM) + f"  {current:.1f} ns (no pool)" + A.RESET)
    else:
        pool_val = c(A.DIM) + "─" * bar_w + "  no data" + A.RESET
    lines.append(labeled("Pool", pool_val))

    # epoch progress + throughput
    epct = s.percent
    nspd = s.ns_per_day
    sps  = s.steps_per_s
    nrep = s.n_replicas
    if epct is not None:
        ep_val = bar(epct, bar_w) + "  " + c(A.BOLD) + f"{epct:.1f}%" + A.RESET
    else:
        ep_val = c(A.DIM) + "─" * bar_w + "  —" + A.RESET
    extras = []
    if nspd: extras.append(f"{nspd:.0f} ns/day")
    if sps:  extras.append(f"{sps:.0f} steps/s")
    if nrep: extras.append(f"{nrep} replicas")
    if extras:
        ep_val += c(A.DIM) + "   " + "   ".join(extras) + A.RESET
    lines.append(labeled("Epoch", ep_val))
    lines += [blank(), sep]

    # ── total-progress timeline (epochs · topups · final · free budget) ────────
    tl = s.pool_timeline
    if tl and tl.get("phases"):
        tbar_w = max(20, inner - 6)
        tbar, tlegend = render_progress_bar(tl, width=tbar_w)
        lines.append("│" + _fill(c(A.DIM) + "  Timeline  " + A.RESET
                     + c(A.DIM, A.ITALIC) + "█baseline ▓topup ░free" + A.RESET, inner) + "│")
        lines.append("│" + _fill("   " + tbar, inner) + "│")
        # legend can overflow → soft-wrap parts across up to 2 rows
        parts = tlegend.split("  ")
        rows, cur = [], ""
        for part in parts:
            cand = part if not cur else cur + "  " + part
            if len(plain(cand)) > inner - 5 and cur:
                rows.append(cur)
                cur = part
            else:
                cur = cand
        if cur:
            rows.append(cur)
        for rw in rows[:2]:
            lines.append("│" + _fill("   " + rw, inner) + "│")
        lines += [blank(), sep]

    # CV / biasing metrics
    mu  = s.gamd_boost
    sd  = s.gamd_boost_sd
    anh = s.gamd_anharmonicity
    ub  = s.ubias_mean
    xch = s.exchange_attempts

    bias = []
    if mu  is not None: bias.append(c(A.DIM) + "boost μ=" + A.RESET + f"{mu:.1f}")
    if sd  is not None:
        sc = A.BRED if sd > 5 else A.BYELLOW if sd > 3 else A.BGREEN
        bias.append(c(A.DIM) + "σ=" + A.RESET + c(sc) + f"{sd:.1f}" + A.RESET)
    if anh is not None:
        ac = A.BRED if anh > 0.5 else A.BYELLOW if anh > 0.3 else A.BGREEN
        bias.append(c(A.DIM) + "anharm=" + A.RESET + c(ac) + f"{anh:.2f}" + A.RESET)
    if ub  is not None: bias.append(c(A.DIM) + "ubias=" + A.RESET + f"{ub:.1f}")
    if xch is not None: bias.append(c(A.DIM) + "exchanges=" + A.RESET + str(xch))

    cv_val = c(A.BOLD) + s.cv_range + A.RESET
    if bias:
        cv_val += c(A.DIM) + "   " + "  ".join(bias) + A.RESET
    lines.append(labeled("CV1", cv_val))
    lines += [blank(), sep]

    # ns history sparkline
    hist    = s.ns_history(500)
    spark_w = max(20, inner - 6)
    lines.append("│" + _fill(c(A.DIM) + f"  ns history ({len(hist)} events)" + A.RESET, inner) + "│")
    if len(hist) >= 2:
        lines.append("│" + _fill("  " + sparkline(hist, width=min(spark_w, inner - 4)), inner) + "│")
        gap  = max(0, min(spark_w, inner - 4) - len(f"{hist[0]:.1f} ns") - len(f"{hist[-1]:.1f} ns") - 2)
        axis = c(A.DIM) + f"  {hist[0]:.1f} ns" + " " * gap + f"{hist[-1]:.1f} ns" + A.RESET
        lines.append("│" + _fill(axis, inner) + "│")
    else:
        lines.append("│" + _fill(c(A.DIM) + "  (no history)" + A.RESET, inner) + "│")

    # latest message / alerts
    msg = s.latest_message
    if msg:
        lines += [blank(), sep]
        mc = A.BRED if "!!" in msg else A.BYELLOW if "WARN" in msg.upper() else A.DIM
        for part in msg.split("\n")[:4]:
            part = part.strip()
            if part:
                lines.append("│" + _fill(c(mc) + "  " + part + A.RESET, inner) + "│")

    lines += [blank(), sep]
    footer = c(A.DIM) + "  ESC/q back   ↑↓/jk prev/next   m MBAR diag   c connect   r refresh" + A.RESET
    lines.append("│" + _fill(footer, inner) + "│")
    lines.append(bot_border)
    return "\n".join(lines)


def _cov_glyph(cell: dict, cov_max: float):
    """(glyph, color) for a single window node in the CV1×CV2 map."""
    life = (cell.get("lifecycle") or "").lower()
    if "retire" in life:
        return "×", A.DIM
    cov = cell.get("cov")
    if cov is None:
        return "?", A.DIM
    if cov == 0:
        return "○", A.BRED            # active window with no samples = MBAR poison
    if cov_max <= 0:
        return "█", A.BGREEN
    frac = cov / cov_max
    ramp = "░▒▓█"
    idx = min(len(ramp) - 1, int(frac * len(ramp)))
    col = A.BRED if frac < 0.25 else A.BYELLOW if frac < 0.5 else A.BGREEN
    return ramp[idx], col


def render_diag(s: PeptideState) -> str:
    """Full-screen MBAR-readiness / adaptive-convergence diagnostics view."""
    tw    = min(term_width(), 100)
    inner = tw - 2
    top = "┌" + "─" * inner + "┐"
    sep = "├" + "─" * inner + "┤"
    bot = "└" + "─" * inner + "┘"

    def line(content: str) -> str:
        return "│" + _fill(content, inner) + "│"

    def labeled(label: str, value: str, lw: int = 13) -> str:
        return line(c(A.DIM) + f"  {label:<{lw}}" + A.RESET + value)

    def _f(x, fmt="{:.3f}"):
        return fmt.format(x) if isinstance(x, (int, float)) else "—"

    def _cf(x):
        return f"{int(x):,}" if isinstance(x, (int, float)) else "—"

    r       = s.readiness
    grade   = s.mbar_grade
    gcol    = (A.BRED if grade == "BAD" else A.BYELLOW if grade == "WARN"
               else A.BGREEN if grade == "OK" else A.DIM)

    de   = s.diag_epoch
    live = s.epochs_completed

    lines = [top]
    lines.append(line(
        c(A.BOLD) + f" {s.name} " + A.RESET
        + c(A.DIM) + "MBAR readiness & adaptive diagnostics" + A.RESET
        + "   " + c(A.BOLD, gcol) + grade + A.RESET
    ))
    prov = c(A.DIM) + f"  readiness epoch: {de if de is not None else '—'}"
    if live is not None:
        prov += f"   ·   live/current epoch: {live}"
    prov += A.RESET
    lines.append(line(prov))
    lines.append(sep)

    if not r:
        lines.append(line(c(A.DIM) + "  No completed-epoch diagnostics yet —"
                          " waiting for the first epoch to finish." + A.RESET))
        # still show live exchange health if available
        xch = s.exchange_attempts
        if xch is not None:
            lines.append(line(c(A.DIM) + f"  live exchange attempts: {xch}" + A.RESET))
        lines += [sep, line(c(A.DIM)
                  + "  ESC/q back   ↑↓/jk peptide   m/Enter detail   r refresh" + A.RESET), bot]
        return "\n".join(lines)

    # ── readiness summary ──────────────────────────────────────────────────────
    conn = r.get("connected")
    if conn is True:
        conn_s = c(A.BGREEN) + "✓ connected" + A.RESET + c(A.DIM) + f" @{MBAR_OVERLAP_CONNECT}" + A.RESET
    elif conn is False:
        conn_s = c(A.BOLD, A.BRED) + "✗ DISCONNECTED" + A.RESET + c(A.DIM) + f" @{MBAR_OVERLAP_CONNECT}" + A.RESET
    else:
        conn_s = c(A.DIM) + "—" + A.RESET
    nweak = r.get("n_weak") or 0
    if nweak:
        conn_s += "   " + c(A.BYELLOW) + f"{nweak} weak edge(s) <{MBAR_OVERLAP_TARGET}" + A.RESET
    if r.get("status"):
        conn_s += "   " + c(A.DIM) + f"status={r['status']}" + A.RESET
    lines.append(labeled("Connectivity", conn_s))

    omin = r.get("olap_min")
    ocol = (A.BRED if (isinstance(omin, (int, float)) and omin < MBAR_OVERLAP_CONNECT)
            else A.BYELLOW if (isinstance(omin, (int, float)) and omin < MBAR_OVERLAP_TARGET)
            else A.BGREEN)
    lines.append(labeled("Overlap",
        c(ocol) + f"min {_f(omin)}" + A.RESET
        + c(A.DIM) + f"   p10 {_f(r.get('olap_p10'))}   med {_f(r.get('olap_med'))}"
        + f"   ({r.get('n_overlaps') or 0} edges)" + A.RESET))

    cz   = r.get("cov_zero") or 0
    ccol = A.BRED if cz > 0 else A.BGREEN
    cov_s = (c(ccol) + f"min {_cf(r.get('cov_min'))}" + A.RESET
             + c(A.DIM) + f"   med {_cf(r.get('cov_med'))}   max {_cf(r.get('cov_max'))}"
             + f"   ({r.get('n_cov') or 0} win @ep{de}" + A.RESET)
    if cz:
        cov_s += c(A.BOLD, A.BRED) + f", {cz} EMPTY" + A.RESET
    cov_s += c(A.DIM) + ")" + A.RESET
    lines.append(labeled("Coverage", cov_s))
    lines.append(labeled("Samples",
        c(A.BOLD) + _cf(r.get("n_samples")) + A.RESET
        + c(A.DIM) + f"  across {r.get('n_windows') or '—'} windows" + A.RESET))

    reg = s.registry
    if reg:
        lines.append(labeled("Windows",
            c(A.BOLD) + f"{reg.get('n_active', '—')}" + A.RESET + c(A.DIM) + " active · " + A.RESET
            + f"{reg.get('n_usable', '—')}" + c(A.DIM) + " usable · " + A.RESET
            + f"{reg.get('n_retired', '—')}" + c(A.DIM) + " retired · " + A.RESET
            + f"{reg.get('n_total', '—')}" + c(A.DIM) + " total" + A.RESET))

    # GaMD/reweighting quality (live) — overlap can look fine while reweighting is poor
    mu, sd, anh = s.gamd_boost, s.gamd_boost_sd, s.gamd_anharmonicity
    if any(x is not None for x in (mu, sd, anh)):
        qparts = []
        if mu is not None:
            qparts.append(c(A.DIM) + "μ=" + A.RESET + f"{mu:.1f}")
        if sd is not None:
            sc = A.BRED if sd > 5 else A.BYELLOW if sd > 3 else A.BGREEN
            qparts.append(c(A.DIM) + "σ=" + A.RESET + c(sc) + f"{sd:.1f}" + A.RESET)
        if anh is not None:
            ac = A.BRED if anh > 0.5 else A.BYELLOW if anh > 0.3 else A.BGREEN
            qparts.append(c(A.DIM) + "anharm=" + A.RESET + c(ac) + f"{anh:.2f}" + A.RESET)
        lines.append(labeled("GaMD boost", "  ".join(qparts)
                     + c(A.DIM, A.ITALIC) + "  (live)" + A.RESET))
    lines.append(sep)

    # ── CV1×CV2 window map ──────────────────────────────────────────────────────
    lines.append(line(c(A.DIM) + "  CV1×CV2 window map  " + A.RESET
                 + c(A.DIM, A.ITALIC) + "░low █high coverage · ○=empty · ×=retired · digit=collision" + A.RESET))
    centers = s.window_centers
    cov     = s.coverage
    if centers:
        n_cols = max(12, min(inner - 8, 40))
        distinct_cv2 = sorted({w["cv2"] for w in centers if w.get("cv2") is not None})
        n_rows = max(3, min(len(distinct_cv2) or 4, 10))
        grid = cv_topology_grid(centers, cov, n_rows, n_cols)
        covvals = [v for v in cov if isinstance(v, (int, float))]
        cmax = max(covvals) if covvals else 0
        for grow in grid:
            rowstr = ""
            for cell in grow:
                if not cell:
                    rowstr += c(A.DIM) + "·" + A.RESET
                elif len(cell) > 1:
                    rowstr += c(A.BOLD, A.BWHITE) + str(len(cell) % 10) + A.RESET
                else:
                    g, gc = _cov_glyph(cell[0], cmax)
                    rowstr += c(gc) + g + A.RESET
            lines.append(line("   " + rowstr))
        c1s = [w["cv1"] for w in centers if w.get("cv1") is not None]
        c2s = [w["cv2"] for w in centers if w.get("cv2") is not None]
        if c1s and c2s:
            lines.append(line(c(A.DIM)
                + f"   CV1 {min(c1s):.3f}→{max(c1s):.3f} (cols)   "
                + f"CV2 {max(c2s):.2f}→{min(c2s):.2f} (top→bottom)" + A.RESET))
    else:
        lines.append(line(c(A.DIM) + "   (no window centers)" + A.RESET))
    lines.append(sep)

    # ── worst edges ─────────────────────────────────────────────────────────────
    lines.append(line(c(A.DIM) + "  Worst edges (overlap ↑)" + A.RESET))
    edges = s.worst_edges[:6]
    if edges:
        for e in edges:
            ov_ = e["overlap"]
            oc = (A.BRED if (isinstance(ov_, (int, float)) and ov_ < MBAR_OVERLAP_CONNECT)
                  else A.BYELLOW if (isinstance(ov_, (int, float)) and ov_ < MBAR_OVERLAP_TARGET)
                  else A.BGREEN)
            acc = e["acc"]
            acc_s = f"{acc:.2f}" if isinstance(acc, (int, float)) else "—"
            att = e["att"]
            warn = (" ⚠ " + ",".join(e["warn"])) if e["warn"] else ""
            lines.append(line(
                c(A.DIM) + f"   w{e['wi']}-w{e['wj']}" + A.RESET
                + c(A.DIM) + f"(s{e['si']}-s{e['sj']})" + A.RESET
                + "  ov " + c(oc) + _f(ov_) + A.RESET
                + c(A.DIM) + f"  acc {acc_s}  att {att if att is not None else '—'}"
                + f"  {e.get('type', '')}" + A.RESET
                + c(A.BYELLOW) + warn + A.RESET))
    else:
        lines.append(line(c(A.DIM) + "   (no edge diagnostics)" + A.RESET))
    lines.append(sep)

    # ── adaptive convergence pressure ───────────────────────────────────────────
    conv = s.convergence
    nact = conv.get("n_actions", 0)
    cep  = conv.get("epoch")
    acol = A.BGREEN if nact == 0 else A.BYELLOW
    lines.append(labeled("Adaptive",
        c(acol) + (f"{nact} action(s) proposed" if nact else "0 actions proposed") + A.RESET
        + c(A.DIM) + (f"  @ep{cep}" if cep is not None else "")
        + "  (see progression for trend)" + A.RESET))
    for reason in (conv.get("continue_reasons") or [])[:3]:
        lines.append(line(c(A.DIM) + "   • " + str(reason)[:inner - 6] + A.RESET))

    # ── total-progress timeline + epoch-over-epoch progression ─────────────────
    tl = s.pool_timeline
    prog = s.epoch_progression
    topups_by_ep = {p["epnum"]: p["n_topups"]
                    for p in (tl.get("phases") or []) if p.get("kind") == "epoch"}
    if tl and tl.get("phases"):
        lines.append(sep)
        lines.append(line(c(A.DIM) + "  Timeline  " + A.RESET
                     + c(A.DIM, A.ITALIC) + "█baseline ▓topup ░free" + A.RESET))
        tbar, tlegend = render_progress_bar(tl, width=max(20, inner - 6))
        lines.append(line("   " + tbar))
        lines.append(line("   " + tlegend))
    if prog:
        lines.append(sep)
        lines.append(line(c(A.DIM) + "  Epoch progression  " + A.RESET
                     + c(A.DIM, A.ITALIC) + "windows · actions · topups · ns" + A.RESET))
        for row in prog[-6:]:
            ep = row.get("epoch")
            nact = row.get("n_actions") or 0
            acol = A.BGREEN if nact == 0 else A.BYELLOW
            na = row.get("n_active")
            nt = row.get("n_total")
            win_s = f"{na}" + (f"/{nt}" if nt not in (None, na) else "") + " win"
            ntop = row.get("n_topups") or topups_by_ep.get(ep, 0)
            top_s = (c(A.BCYAN) + f"+{ntop}↑" + A.RESET) if ntop else c(A.DIM) + "  ·" + A.RESET
            lines.append(line(
                c(A.DIM) + f"   ep{ep if ep is not None else '?':<3}" + A.RESET
                + c(A.BOLD) + f"{win_s:<11}" + A.RESET
                + c(acol) + f"{nact} act  " + A.RESET
                + top_s
                + c(A.DIM) + f"   {row.get('consumed_ns', 0):.0f} ns" + A.RESET))

    lines.append(sep)
    lines.append(line(c(A.DIM)
        + "  ESC/q back   ↑↓/jk peptide   m/Enter detail   c connect   r refresh" + A.RESET))
    lines.append(bot)
    return "\n".join(lines)


def render_connected(s: PeptideState) -> str:
    """Live 'attach' view for a single run — reconstructs the run's dashboard
    from a freshest-per-field tail of progress.jsonl, refreshed fast (~1s).
    """
    snap  = s.live_snapshot()
    tw    = min(term_width(), 100)
    inner = tw - 2
    top = "┌" + "─" * inner + "┐"
    sep = "├" + "─" * inner + "┤"
    bot = "└" + "─" * inner + "┘"

    def line(content: str) -> str:
        return "│" + _fill(content, inner) + "│"

    def labeled(label: str, value: str, lw: int = 11) -> str:
        return line(c(A.DIM) + f"  {label:<{lw}}" + A.RESET + value)

    def _age_note(field: str) -> str:
        a = (snap.get("_ages") or {}).get(field)
        if a is None or a <= 3:
            return ""
        return c(A.DIM, A.ITALIC) + f"  ~{a} frames old" + A.RESET

    if not snap:
        return (top + "\n" + line(c(A.DIM)
                + f"  ▸ CONNECTED  {s.name} — no progress.jsonl yet" + A.RESET)
                + "\n" + bot)

    phase  = snap.get("phase") or "—"
    pcol   = PHASE_COLOR.get(phase, A.DIM)
    plabel = PHASE_LABEL.get(phase, str(phase).upper())

    lines = [top]
    lines.append(line(
        c(A.BOLD, A.BCYAN) + f" ▸ CONNECTED  {s.name} " + A.RESET
        + c(pcol) + plabel + A.RESET
        + c(A.DIM) + f"   live · {fmt_dur(snap.get('elapsed_s'))}" + A.RESET
        + (c(A.DIM, A.ITALIC) + "  [stale]" + A.RESET if s.stale else "")))
    lines.append(sep)

    # progress / throughput
    pct = snap.get("percent")
    if pct is None and snap.get("fraction") is not None:
        pct = snap["fraction"] * 100
    bar_w = max(16, inner - 46)
    prog_val = (bar(pct, bar_w) + "  " + c(A.BOLD) + f"{pct:.1f}%" + A.RESET) if pct is not None \
        else (c(A.DIM) + "─" * bar_w + "  —" + A.RESET)
    extras = []
    nspd = snap.get("aggregate_ns_per_day") or snap.get("ns_per_day")
    if nspd:
        extras.append(f"{nspd:.0f} ns/day")
    sps = snap.get("steps_per_s")
    if sps:
        extras.append(f"{sps:.0f} steps/s")
    eta = snap.get("eta_s")
    if eta is not None:
        extras.append("ETA " + fmt_dur(eta))
    if extras:
        prog_val += c(A.DIM) + "   " + "  ".join(extras) + A.RESET
    lines.append(labeled("Progress", prog_val))
    step, tot = snap.get("step"), snap.get("total_steps")
    aggns = snap.get("aggregate_sim_time_ns")
    sub = []
    if step is not None and tot:
        sub.append(f"step {step:,}/{tot:,}")
    if aggns is not None:
        sub.append(f"{aggns:.1f} ns simulated")
    if sub:
        lines.append(labeled("", c(A.DIM) + "   ".join(sub) + A.RESET))

    # segmented timeline (epochs · topups · final · free)
    tl = s.pool_timeline
    if tl and tl.get("phases"):
        tbar, tlegend = render_progress_bar(tl, width=max(20, inner - 6))
        lines.append(sep)
        lines.append(line(c(A.DIM) + "  Timeline  " + A.RESET
                     + c(A.DIM, A.ITALIC) + "█base ▓topup ░free" + A.RESET))
        lines.append(line("   " + tbar))
        lines.append(line("   " + tlegend))

    # live CVs + GaMD boost
    def _rng(lo, me, hi):
        f = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "—"
        return f"min {f(lo)}  mean {f(me)}  max {f(hi)}"
    cv1_on = any(snap.get(k) is not None for k in ("cv_min_A", "cv_mean_A", "cv_max_A"))
    cv2_on = any(snap.get(k) is not None for k in ("secondary_cv_min", "secondary_cv_mean", "secondary_cv_max"))
    mu, sd, anh = (snap.get("gamd_boost_mean_kcal_mol"), snap.get("gamd_boost_sd_kcal_mol"),
                   snap.get("gamd_boost_anharmonicity_score"))
    if cv1_on or cv2_on or any(x is not None for x in (mu, sd, anh)):
        lines.append(sep)
        if cv1_on:
            lines.append(labeled("CV1", c(A.BOLD)
                + _rng(snap.get("cv_min_A"), snap.get("cv_mean_A"), snap.get("cv_max_A"))
                + A.RESET + _age_note("cv_mean_A")))
        if cv2_on:
            lines.append(labeled("CV2", c(A.BOLD)
                + _rng(snap.get("secondary_cv_min"), snap.get("secondary_cv_mean"), snap.get("secondary_cv_max"))
                + A.RESET + _age_note("secondary_cv_mean")))
        if any(x is not None for x in (mu, sd, anh)):
            bparts = []
            if mu is not None:
                bparts.append(c(A.DIM) + "μ=" + A.RESET + f"{mu:.1f}")
            if sd is not None:
                sc = A.BRED if sd > 5 else A.BYELLOW if sd > 3 else A.BGREEN
                bparts.append(c(A.DIM) + "σ=" + A.RESET + c(sc) + f"{sd:.1f}" + A.RESET)
            if anh is not None:
                ac = A.BRED if anh > 0.5 else A.BYELLOW if anh > 0.3 else A.BGREEN
                bparts.append(c(A.DIM) + "anharm=" + A.RESET + c(ac) + f"{anh:.2f}" + A.RESET)
            lines.append(labeled("GaMD boost", "  ".join(bparts) + _age_note("gamd_boost_mean_kcal_mol")))

    # replica-exchange panel (with staleness)
    dash = snap.get("_dashboard")
    xs = parse_exchange_stats(dash)
    if xs["attempts"]:
        age_s = snap.get("_dashboard_age_s")
        age_str = (c(A.DIM) + f"  (snapshot {fmt_dur(age_s)} ago)" + A.RESET) if (age_s and age_s > 5) else ""
        lines.append(sep)
        lines.append(line(c(A.DIM) + f"  Exchange ({xs.get('mode') or '—'})" + A.RESET + age_str))
        rate = xs["rate"]
        rcol = A.BGREEN if (rate is not None and rate >= 0.2) else A.BYELLOW if rate else A.DIM
        lines.append(line(
            "   gibbs acc " + c(rcol) + (f"{rate * 100:.1f}%" if rate is not None else "—") + A.RESET
            + c(A.DIM) + f"  ({xs['accepted']}/{xs['attempts']})"
            + (f"  gibbs stay {xs['gibbs_stays']}" if xs.get("gibbs_stays") is not None else "")
            + f"  pairs {xs['n_pairs']}" + A.RESET))
        for j in xs["jumps"][:6]:
            r = j["rate"]
            bw = 10
            fill = int(round((r or 0) * bw))
            jc = A.BGREEN if (r is not None and r >= 0.5) else A.BYELLOW if r else A.DIM
            jbar = c(jc) + "█" * fill + c(A.DIM) + "░" * (bw - fill) + A.RESET
            lines.append(line(
                c(A.DIM) + f"   {j['name']:<4}" + A.RESET + jbar
                + c(A.DIM) + f" {(r * 100 if r is not None else 0):.0f}%  {j['accepted']}/{j['attempts']}" + A.RESET))

    # live window map (windows currently being propagated)
    lw = parse_live_windows(dash)
    if lw:
        lines.append(sep)
        lines.append(line(c(A.DIM) + f"  Live window map ({len(lw)} windows)" + A.RESET
                     + c(A.DIM, A.ITALIC) + "  ●=window · digit=collision" + A.RESET))
        n_cols = max(12, min(inner - 8, 40))
        distinct = sorted({w["cv2"] for w in lw if w.get("cv2") is not None})
        n_rows = max(3, min(len(distinct) or 4, 8))
        grid = cv_topology_grid(lw, [], n_rows, n_cols)
        for grow in grid:
            rowstr = ""
            for cell in grow:
                if not cell:
                    rowstr += c(A.DIM) + "·" + A.RESET
                elif len(cell) > 1:
                    rowstr += c(A.BOLD, A.BWHITE) + str(len(cell) % 10) + A.RESET
                else:
                    rowstr += c(A.BCYAN) + "●" + A.RESET
            lines.append(line("   " + rowstr))
        c1s = [w["cv1"] for w in lw if w.get("cv1") is not None]
        c2s = [w["cv2"] for w in lw if w.get("cv2") is not None]
        if c1s and c2s:
            lines.append(line(c(A.DIM)
                + f"   CV1 {min(c1s):.3f}→{max(c1s):.3f}   CV2 {max(c2s):.2f}→{min(c2s):.2f}" + A.RESET))

    # approximate basin hints (reweighted biased CV snapshots — NOT a PMF)
    basins = approx_basins(snap.get("_dist_samples") or [])
    if basins["minima"]:
        lines.append(sep)
        lines.append(line(c(A.DIM) + "  Basin hints " + A.RESET
                     + c(A.BYELLOW, A.ITALIC) + "(approx, no MBAR — biased/GaMD-reweighted)" + A.RESET))
        for b in basins["minima"]:
            depth = max(0.0, b["F_rel"])
            dcol = A.BGREEN if depth < 0.3 else A.BCYAN if depth < 1.0 else A.DIM
            lines.append(line(
                c(dcol) + "   ◆ " + A.RESET
                + c(A.BOLD) + f"CV1 {b['cv1']:.3f}  CV2 {b['cv2']:+.2f}" + A.RESET
                + c(A.DIM) + f"   ΔF≈{depth:.2f} kcal/mol   pop {b['pop_frac'] * 100:.0f}%" + A.RESET))
        lines.append(line(c(A.DIM)
            + f"   from {basins['n_samples']} live CV snapshots, {basins['n_bins']} bins" + A.RESET))

    # latest message
    msg = snap.get("message")
    if msg:
        lines.append(sep)
        mc = A.BRED if "!!" in str(msg) else A.BYELLOW if "WARN" in str(msg).upper() else A.DIM
        for part in str(msg).split("\n")[:2]:
            if part.strip():
                lines.append(line(c(mc) + "  " + part.strip() + A.RESET))

    lines.append(sep)
    lines.append(line(c(A.DIM)
        + "  c/ESC/q detach   ↑↓/jk peptide   m diag   Enter detail   r refresh  ·  auto-1s" + A.RESET))
    lines.append(bot)
    return "\n".join(lines)


# ── Keyboard reader ───────────────────────────────────────────────────────────

def _read_key_raw(fd: int, timeout: float) -> Optional[str]:
    """Read one key event from raw fd. Returns key name or None on timeout."""
    r, _, _ = _select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    ch = os.read(fd, 1)
    if ch == b'\x1b':
        r2, _, _ = _select.select([sys.stdin], [], [], 0.05)
        if not r2:
            return "ESC"
        ch2 = os.read(fd, 1)
        if ch2 == b'[':
            r3, _, _ = _select.select([sys.stdin], [], [], 0.05)
            if not r3:
                return "ESC"
            ch3 = os.read(fd, 1)
            if ch3 == b'A': return "UP"
            if ch3 == b'B': return "DOWN"
            if ch3 == b'C': return "RIGHT"
            if ch3 == b'D': return "LEFT"
            # drain numeric sequences (e.g. \x1b[1;5A)
            while _select.select([sys.stdin], [], [], 0.02)[0]:
                os.read(fd, 1)
            return None
        return "ESC"
    if ch in (b'\r', b'\n'):
        return "ENTER"
    if ch == b'\x03':
        raise KeyboardInterrupt
    if ch == b'\x04':
        raise KeyboardInterrupt
    if ch == b'\x7f':
        return "BACKSPACE"
    try:
        return ch.decode("utf-8")
    except Exception:
        return None


# ── Interactive loop ──────────────────────────────────────────────────────────

def _write(s: str):
    sys.stdout.write(s)
    sys.stdout.flush()


CONNECT_INTERVAL = 1.0   # fast refresh cadence for the live 'connect' view


def next_view(view: str, key: str):
    """Pure view/key state machine for the interactive loop.

    Returns (new_view, action), action in {None, 'quit', 'refresh',
    'connect_reset', 'noop'}.  Selection nav (UP/DOWN/k/j) is NOT handled here —
    it changes the cursor, not the view, so the caller handles it.  Extracted as
    a pure function so the navigation surface is unit-testable (a TTY loop is not).
    """
    if view == "list":
        if key == "q":
            return view, "quit"
        if key in ("ENTER", "d", " "):
            return "detail", None
        if key == "m":
            return "diag", None
        if key == "c":
            return "connected", "connect_reset"
        if key == "r":
            return view, "refresh"
        return view, "noop"
    # detail / diag / connected
    if key in ("q", "ESC", "h"):
        return "list", None
    if key == "c":
        return ("list", None) if view == "connected" else ("connected", "connect_reset")
    if key == "m":
        return ("detail" if view == "diag" else "diag"), None
    if key in ("ENTER", "d") and view in ("diag", "connected"):
        return "detail", None
    if key == "r":
        return view, "refresh"
    return view, "noop"


def interactive_loop(states: list, interval: float,
                     start_view: str = "list", start_sel: int = 0):
    if not _HAS_TTY or not sys.stdin.isatty():
        # non-TTY fallback: plain sleep loop
        _write(HIDE_CURSOR)
        try:
            first = True
            while True:
                for s in states:
                    s.refresh()
                out = render(states)
                _write((CLEAR_SCREEN if first else MOVE_HOME + CLEAR_EOS) + out)
                first = False
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            _write(SHOW_CURSOR + "\n")
        return

    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    # cbreak: disable ICANON + ECHO, keep OPOST (so \n → \r\n works in output)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
    new[6][termios.VMIN]  = 1
    new[6][termios.VTIME] = 0

    sel           = start_sel % len(states)
    view          = start_view      # "list" | "detail" | "diag" | "connected"
    last_refresh  = 0.0
    last_connect  = 0.0             # fast-refresh clock for the connected view
    force_refresh = True
    need_redraw   = True
    prev_view     = None            # detect view switch for full clear

    _write(HIDE_CURSOR)
    try:
        termios.tcsetattr(fd, termios.TCSAFLUSH, new)
        first = True
        while True:
            now = time.time()
            if force_refresh or (now - last_refresh) >= interval:
                for s in states:
                    s.refresh()
                last_refresh  = time.time()
                force_refresh = False
                need_redraw   = True
            # connected view tails one run at a faster cadence
            if view == "connected" and (now - last_connect) >= CONNECT_INTERVAL:
                states[sel].refresh()
                last_connect = now
                need_redraw  = True

            if need_redraw:
                if view == "detail":
                    out = render_detail(states[sel])
                elif view == "diag":
                    out = render_diag(states[sel])
                elif view == "connected":
                    out = render_connected(states[sel])
                else:
                    out = render(states, selected=sel)
                view_changed = (view != prev_view)
                prev_view    = view
                if first or view_changed:
                    _write(CLEAR_SCREEN + out)
                    first = False
                else:
                    _write(MOVE_HOME + CLEAR_EOS + out)
                need_redraw = False

            # short poll so the connected view repaints near its cadence
            key = _read_key_raw(fd, timeout=0.15 if view != "connected" else 0.25)
            if key is None:
                continue

            need_redraw = True
            if key in ("UP", "k"):
                sel = (sel - 1) % len(states)
            elif key in ("DOWN", "j"):
                sel = (sel + 1) % len(states)
            else:
                new_view, action = next_view(view, key)
                if action == "quit":
                    break
                view = new_view
                if action == "refresh":
                    force_refresh = True
                elif action == "connect_reset":
                    last_connect = 0.0
                elif action == "noop":
                    need_redraw = False
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        _write(SHOW_CURSOR + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GAREUS multi-peptide live monitor (stdlib only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("runs_dirs", nargs="*", default=["."],
                        help="Root dir(s) containing peptide subdirs (or flat run dirs)")
    parser.add_argument("--interval", "-i", type=float, default=10.0,
                        help="Refresh interval (seconds)")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit")
    parser.add_argument("--extra", "-e", nargs="+", metavar="DIR",
                        help="Additional flat run dirs to include explicitly")
    parser.add_argument("--connect", "-c", metavar="PEPTIDE",
                        help="Start attached to one run's live view (substring match on name)")
    args = parser.parse_args()

    states: list = []
    seen: set = set()

    for rd in args.runs_dirs:
        p = Path(rd).resolve()
        if not p.is_dir():
            print(f"ERROR: {p} is not a directory", file=sys.stderr)
            sys.exit(1)
        for s in discover(p):
            key = str(s.run_dir)
            if key not in seen:
                seen.add(key)
                states.append(s)

    for extra in (args.extra or []):
        p = Path(extra)
        if not p.is_absolute():
            p = Path(args.runs_dirs[0]).resolve() / p
        p = p.resolve()
        if not p.is_dir():
            print(f"WARNING: --extra {extra} not found, skipping", file=sys.stderr)
            continue
        key = str(p)
        if key not in seen:
            seen.add(key)
            states.append(PeptideState.from_rundir(p))

    if not states:
        print("No peptide runs found.", file=sys.stderr)
        sys.exit(1)

    for s in states:
        s.refresh()

    if args.once:
        print(render(states))
        return

    start_view, start_sel = "list", 0
    if args.connect:
        q = args.connect.lower()
        match = next((i for i, s in enumerate(states) if q in s.name.lower()), None)
        if match is None:
            print(f"WARNING: --connect {args.connect!r} matched no run; starting in list view",
                  file=sys.stderr)
        else:
            start_view, start_sel = "connected", match

    interactive_loop(states, args.interval, start_view=start_view, start_sel=start_sel)


if __name__ == "__main__":
    main()
