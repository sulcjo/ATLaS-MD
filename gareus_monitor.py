#!/usr/bin/env python3
"""
GAREUS multi-peptide run monitor  —  stdlib only, no deps.

Usage:
  python gareus_monitor.py [RUNS_DIR] [--interval SECS] [--once]
"""

import argparse
import json
import os
import re
import sys
import time
import datetime
from pathlib import Path
from typing import Optional

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

def tail_jsonl(path: Path, n: int = 60) -> list:
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        if size == 0:
            return []
        chunk = min(size, 65536)
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


# ── Per-peptide state ─────────────────────────────────────────────────────────

class PeptideState:
    def __init__(self, name: str, base_dir: Path):
        self.name        = name
        self.base_dir    = base_dir
        self.run_dir     = base_dir / f"{name}_2d_run"
        self.genpept_dir = base_dir / f"{name}_genpept"

        self._prog: dict         = {}
        self._drvsumm: dict      = {}
        self._pool: dict         = {}
        self._mtime_prog: float  = 0.0
        self._mtime_drv: float   = 0.0
        self._mtime_pool: float  = 0.0
        self._epoch_count: Optional[int] = None
        self._ckpt_count: Optional[int]  = None
        self._ckpt_refresh: float        = 0.0
        self._genpept_status: Optional[str] = None
        self._genpept_surv: Optional[int]   = None

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

    def refresh(self):
        self._load_progress()
        self._load_driver_summary()
        self._load_epochs()
        self._load_checkpoints()
        self._load_genpept()
        self._load_runtime_pool()

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
        return self._drvsumm.get("epochs_completed") or self._epoch_count

    @property
    def total_budget_ns(self) -> Optional[float]:
        """Total MD budget from adaptive_runtime_pool (committed + remaining)."""
        events = self._pool.get("events")
        if not events:
            return None
        last = events[-1]
        return last.get("used_ns_after", 0) + last.get("remaining_ns_after", 0)

    @property
    def committed_ns(self) -> Optional[float]:
        """NS from fully completed pool segments (does not include in-progress topup)."""
        events = self._pool.get("events")
        if not events:
            return None
        return events[-1].get("used_ns_after", 0)

    @property
    def global_percent(self) -> Optional[float]:
        """Total MD budget progress: (committed + in-progress agg) / total × 100."""
        total = self.total_budget_ns
        if total is None or total == 0:
            return None
        committed = self.committed_ns or 0.0
        current   = self.agg_ns or 0.0
        return min(100.0, (committed + current) / total * 100)

    @property
    def global_eta_s(self) -> Optional[float]:
        """Estimated time to finish entire MD budget based on current ns/day."""
        total    = self.total_budget_ns
        nspd     = self.ns_per_day
        if total is None or not nspd:
            return None
        committed = self.committed_ns or 0.0
        current   = self.agg_ns or 0.0
        remaining_ns = total - committed - current
        if remaining_ns <= 0:
            return 0.0
        return remaining_ns / nspd * 86400  # seconds

    @property
    def stale(self) -> bool:
        wt = self._prog.get("wall_time_s")
        return wt is not None and (time.time() - wt) > 900


# ── Discovery ─────────────────────────────────────────────────────────────────

_IGNORE = {
    ".git", ".claude", ".codex", ".remember", ".agents", "__pycache__",
    "alpha_runs", "validation", "gamd-openmm", "gareus",
}


def discover(runs_dir: Path) -> list:
    out = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        n = d.name
        if n.startswith(".") or n in _IGNORE:
            continue
        if n.startswith("CONV") or n.startswith("chignolin") or n.startswith("alpha"):
            continue
        if (d / f"{n}.yaml").exists() or (d / f"{n}_2d_run").exists():
            out.append(PeptideState(n, d))
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
    "bar":     16,   # global progress bar
    "gpct":     7,   # global % (total budget)
    "epct":     6,   # current epoch/topup %
    "budget":  11,   # ns_done/ns_total
    "epoch":    5,
    "ckpt":     5,
    "elapsed":  8,
    "geta":     9,   # global ETA
    "nspd":     7,
    "sps":      7,
    "cv":      14,
    "gamd":     7,
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
    "elapsed": "Elapsed",
    "geta":    "Global ETA",
    "nspd":    "ns/day",
    "sps":     "Steps/s",
    "cv":      "CV range",
    "gamd":    "GaMDμ",
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
    "elapsed": ">",
    "geta":    ">",
    "nspd":    ">",
    "sps":     ">",
    "cv":      ">",
    "gamd":    ">",
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


def data_row(s: PeptideState) -> str:
    phase = s.phase
    pcol  = PHASE_COLOR.get(phase, A.DIM)
    ncol  = NAME_COLOR.get(phase, A.DIM)

    # global and epoch percentages
    gpct = s.global_percent   # total budget progress
    epct = s.percent          # current topup/epoch progress

    # name
    stale = c(A.DIM, A.ITALIC, " ~") + A.RESET if s.stale else ""
    name_val = c(ncol) + s.name[:COL["name"]] + A.RESET + stale

    # genpept
    if s._genpept_status == "done":
        gp_val = c(A.BOLD, A.BGREEN) + f"✓ {s._genpept_surv or '?'}" + A.RESET
    elif s._genpept_status == "running":
        gp_val = c(A.BBLUE) + "run…" + A.RESET
    else:
        gp_val = c(A.DIM) + "—" + A.RESET

    # phase
    plabel = PHASE_LABEL.get(phase, phase.upper()[:10])
    phase_val = c(pcol) + plabel + A.RESET

    # global progress bar (uses gpct; fallback to epct if pool not available)
    display_pct = gpct if gpct is not None else epct
    bar_val = (bar(display_pct, COL["bar"]) if display_pct is not None
               else c(A.DIM) + "░" * COL["bar"] + A.RESET)

    # G% — global percent
    if gpct is not None:
        gcol = A.BGREEN if gpct >= 80 else A.BYELLOW if gpct >= 40 else A.BRED
        gpct_val = c(A.BOLD, gcol) + f"{gpct:.1f}%" + A.RESET
    else:
        gpct_val = c(A.DIM) + "—" + A.RESET

    # E% — current epoch/topup percent (dim, smaller context)
    if epct is not None:
        epct_val = c(A.DIM) + f"{epct:.0f}%" + A.RESET
    else:
        epct_val = c(A.DIM) + "—" + A.RESET

    # ns/budget — e.g. "54/2000"
    total = s.total_budget_ns
    committed = s.committed_ns or 0.0
    current   = s.agg_ns or 0.0
    done_ns   = committed + current
    if total:
        budget_val = f"{done_ns:.0f}/{total:.0f}"
    elif current:
        budget_val = f"{current:.1f} ns"
    else:
        budget_val = "—"

    # epoch
    ep = s.epochs_completed
    ep_val = str(ep) if ep is not None else "—"

    # checkpoints
    ck = s._ckpt_count
    ck_val = str(ck) if ck is not None else "—"

    # global ETA (falls back to per-topup eta if no pool)
    geta = s.global_eta_s
    if geta is None:
        geta = s.eta_s
    if geta is not None and geta <= 0:
        geta_val = c(A.BOLD, A.BGREEN) + "done" + A.RESET
    else:
        geta_val = fmt_dur(geta)

    nspd = s.ns_per_day
    sps  = s.steps_per_s
    gb   = s.gamd_boost

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
        "elapsed": fmt_dur(s.elapsed_s),
        "geta":    geta_val,
        "nspd":    f"{nspd:.0f}" if nspd else "—",
        "sps":     f"{sps:.0f}"  if sps  else "—",
        "cv":      s.cv_range,
        "gamd":    f"{gb:.1f}"   if gb   else "—",
    }

    cells = []
    for k in COLS:
        cells.append(" " + pad(vals[k], COL[k], ALIGNS[k]) + " ")
    return "│" + "│".join(cells) + "│"


def summary_line(states: list) -> str:
    n_prod  = sum(1 for s in states if s.phase == "gareus_production")
    n_adapt = sum(1 for s in states if s.phase == "adaptive_feedback")
    n_done  = sum(1 for s in states if s.phase in ("done", "converged"))
    n_err   = sum(1 for s in states if s.phase == "error")
    n_pend  = sum(1 for s in states if s.phase in ("not_started", "genpept", "unknown"))

    # aggregate global budget across all runs that have pool data
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


def render(states: list) -> str:
    tw = term_width()
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    title = (
        c(A.BOLD, A.BCYAN) + " GAREUS Monitor " + A.RESET
        + c(A.DIM) + now + A.RESET
    )

    top    = "┌" + hline("─", "┬", "┌", "┐")[1:-1] + "┐"
    mid    = hline("─", "┼", "├", "┤")
    bot    = "└" + hline("─", "┴", "└", "┘")[1:-1] + "┘"

    lines = [
        "",
        "  " + title,
        "",
        top,
        header_row(),
        mid,
    ]
    for i, s in enumerate(states):
        lines.append(data_row(s))
        if i < len(states) - 1:
            lines.append(mid)
    lines += [
        bot,
        "",
        "  " + summary_line(states),
        "",
        c(A.DIM) + "  q/Ctrl-C quit   r refresh now" + A.RESET,
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GAREUS multi-peptide live monitor (stdlib only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("runs_dir", nargs="?", default=".",
                        help="Root dir containing peptide subdirs")
    parser.add_argument("--interval", "-i", type=float, default=10.0,
                        help="Refresh interval (seconds)")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        print(f"ERROR: {runs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    states = discover(runs_dir)
    if not states:
        print(f"No peptide runs found in {runs_dir}", file=sys.stderr)
        sys.exit(1)

    for s in states:
        s.refresh()

    if args.once:
        print(render(states))
        return

    # live loop
    print(HIDE_CURSOR, end="", flush=True)
    try:
        first = True
        while True:
            for s in states:
                s.refresh()
            output = render(states)
            if first:
                print(CLEAR_SCREEN + output, flush=True)
                first = False
            else:
                print(MOVE_HOME + output, flush=True)
            # non-blocking wait with 'q' detection (best-effort; no termios)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        print(SHOW_CURSOR, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
