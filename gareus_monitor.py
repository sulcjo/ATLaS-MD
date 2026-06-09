#!/usr/bin/env python3
"""
GAREUS multi-peptide run monitor.

Usage:
  python gareus_monitor.py [RUNS_DIR] [--interval SECS] [--once]

Scans RUNS_DIR for peptide subdirs with {PEP}/{PEP}_2d_run/ layouts.
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

try:
    from rich.live import Live
    from rich.table import Table
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    from rich import box
except ImportError:
    print("rich not installed. Run: pip install rich")
    sys.exit(1)

# ── Phase metadata ────────────────────────────────────────────────────────────

PHASE_COLORS = {
    "gareus_production": "bright_green",
    "adaptive_feedback":  "cyan",
    "setup":              "yellow",
    "equilibration":      "yellow",
    "genpept":            "bright_blue",
    "done":               "bright_white",
    "converged":          "bright_magenta",
    "error":              "bright_red",
    "not_started":        "dim",
    "unknown":            "dim",
}

PHASE_LABELS = {
    "gareus_production": "PRODUCTION",
    "adaptive_feedback":  "ADAPT-FB",
    "setup":              "SETUP",
    "equilibration":      "EQUIL",
    "genpept":            "GENPEPT",
    "done":               "DONE ✓",
    "converged":          "CONVERGED",
    "error":              "ERROR ✗",
    "not_started":        "PENDING",
    "unknown":            "—",
}

# ── JSONL helpers ─────────────────────────────────────────────────────────────

def tail_jsonl(path: Path, n: int = 30) -> list:
    """Read last n JSON lines from a jsonl file via tail-seek (O(1))."""
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        if size == 0:
            return []
        chunk = min(size, 65_536)  # 64 KB tail
        with open(path, "rb") as f:
            f.seek(max(0, size - chunk))
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
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
    """Return the most recent 'progress' event from progress.jsonl."""
    entries = tail_jsonl(path, 60)
    for e in reversed(entries):
        if e.get("event") == "progress":
            return e
    # fallback: any entry that has meaningful fields
    for e in reversed(entries):
        if "percent" in e or "fraction" in e:
            return e
    return entries[-1] if entries else {}


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_dur(s: Optional[float], *, compact: bool = True) -> str:
    if s is None or s < 0:
        return "—"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec  = divmod(rem, 60)
    if h >= 24:
        d, h = divmod(h, 24)
        return f"{d}d{h:02d}h"
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{sec:02d}s"


def progress_bar(pct: float, width: int = 16) -> Text:
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    if pct >= 80:
        col = "bright_green"
    elif pct >= 40:
        col = "yellow"
    elif pct > 0:
        col = "red"
    else:
        col = "dim"
    bar = Text()
    bar.append("█" * filled,          style=col)
    bar.append("░" * (width - filled), style="dim")
    return bar


# ── Per-peptide state ─────────────────────────────────────────────────────────

class PeptideState:
    """Live status for one peptide run, cached by file mtime."""

    def __init__(self, name: str, base_dir: Path):
        self.name        = name
        self.base_dir    = base_dir
        self.run_dir     = base_dir / f"{name}_2d_run"
        self.genpept_dir = base_dir / f"{name}_genpept"

        self._prog: dict      = {}
        self._drvsumm: dict   = {}
        self._mtime_prog: float  = 0.0
        self._mtime_drv: float   = 0.0
        self._epoch_count: Optional[int]  = None
        self._topup_count: Optional[int]  = None
        self._ckpt_count: Optional[int]   = None
        self._ckpt_refresh: float = 0.0
        self._genpept_status: Optional[str] = None
        self._genpept_surv: Optional[int]   = None

    # ── loaders ──────────────────────────────────────────────────────────────

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
        epoch_dirs = sorted(ap.glob("epoch_*"))
        self._epoch_count = len(epoch_dirs)
        if epoch_dirs:
            topup_dirs = sorted(epoch_dirs[-1].glob("topup_*"))
            self._topup_count = len(topup_dirs)

    def _load_checkpoints(self):
        ap = self.run_dir / "adaptive_production"
        if not ap.exists():
            self._ckpt_count = 0
            return
        now = time.time()
        if self._ckpt_count is not None and now - self._ckpt_refresh < 60:
            return
        self._ckpt_refresh = now
        manifests = list(ap.rglob("production_checkpoint_manifest.json"))
        self._ckpt_count = len(manifests)

    def _load_genpept(self):
        if self._genpept_status is not None:
            return
        if not self.genpept_dir.exists():
            self._genpept_status = "pending"
            return
        seeds_csv = self.genpept_dir / "final_survivor_seeds.csv"
        if seeds_csv.exists():
            self._genpept_status = "done"
            try:
                with open(seeds_csv) as f:
                    lines = sum(1 for _ in f) - 1
                self._genpept_surv = max(0, lines)
            except Exception:
                pass
        elif any(self.genpept_dir.iterdir()):
            self._genpept_status = "running"
        else:
            self._genpept_status = "pending"

    # ── public ────────────────────────────────────────────────────────────────

    def refresh(self):
        self._load_progress()
        self._load_driver_summary()
        self._load_epochs()
        self._load_checkpoints()
        self._load_genpept()

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def phase(self) -> str:
        if not self.run_dir.exists():
            if self._genpept_status == "running":
                return "genpept"
            return "not_started"
        if self._drvsumm.get("status") == "done":
            return "done"
        p = self._prog.get("phase", "")
        return p if p else "unknown"

    @property
    def percent(self) -> Optional[float]:
        v = self._prog.get("percent")
        if v is not None:
            return float(v)
        f = self._prog.get("fraction")
        if f is not None:
            return float(f) * 100
        return None

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
        msg = self._prog.get("message", "")
        m = re.search(r"(\d+)\s+replica", msg)
        return int(m.group(1)) if m else None

    @property
    def exchange_attempts(self) -> Optional[int]:
        return self._prog.get("exchange_attempts")

    @property
    def epochs_completed(self) -> Optional[int]:
        return self._drvsumm.get("epochs_completed") or self._epoch_count

    @property
    def stale(self) -> bool:
        """Progress not updated in >15 min."""
        wt = self._prog.get("wall_time_s")
        if wt is None:
            return False
        return (time.time() - wt) > 900

    @property
    def genpept_ok(self) -> bool:
        return self._genpept_status == "done"


# ── Discovery ─────────────────────────────────────────────────────────────────

_IGNORE_DIRS = {
    ".git", ".claude", ".codex", ".remember", ".agents",
    "__pycache__", "alpha_runs", "validation", "gamd-openmm", "gareus",
}


def discover_peptides(runs_dir: Path) -> list:
    results = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if name.startswith(".") or name in _IGNORE_DIRS:
            continue
        if name.startswith("CONV") or name.startswith("chignolin") or name.startswith("alpha"):
            continue
        has_yaml = (d / f"{name}.yaml").exists()
        has_run  = (d / f"{name}_2d_run").exists()
        if has_yaml or has_run:
            results.append(PeptideState(name, d))
    return results


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(states: list) -> Table:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    table = Table(
        title=f"[bold bright_cyan] GAREUS Multi-Peptide Monitor [/]  [dim]{now_str}[/]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold bright_white on grey19",
        expand=True,
        show_edge=True,
        padding=(0, 1),
    )

    table.add_column("Peptide",  style="bold",   min_width=12, no_wrap=True)
    table.add_column("GENPEPT",  justify="center", min_width=9, no_wrap=True)
    table.add_column("Phase",    justify="center", min_width=11, no_wrap=True)
    table.add_column("Progress", min_width=18,   no_wrap=True)
    table.add_column("%",        justify="right", min_width=6,  no_wrap=True)
    table.add_column("Epoch",    justify="right", min_width=5,  no_wrap=True)
    table.add_column("Ckpts",    justify="right", min_width=5,  no_wrap=True)
    table.add_column("Elapsed",  justify="right", min_width=8,  no_wrap=True)
    table.add_column("ETA",      justify="right", min_width=8,  no_wrap=True)
    table.add_column("ns/day",   justify="right", min_width=7,  no_wrap=True)
    table.add_column("Σns",      justify="right", min_width=6,  no_wrap=True)
    table.add_column("Steps/s",  justify="right", min_width=7,  no_wrap=True)
    table.add_column("CV range", justify="right", min_width=14, no_wrap=True)
    table.add_column("GaMD μ",   justify="right", min_width=7,  no_wrap=True)

    for s in states:
        phase  = s.phase
        pcolor = PHASE_COLORS.get(phase, "dim")
        plabel = PHASE_LABELS.get(phase, phase.upper()[:10])
        pct    = s.percent

        # name style
        if phase == "gareus_production":
            name_style = "bold bright_cyan"
        elif phase in ("adaptive_feedback", "setup", "equilibration"):
            name_style = "bold yellow"
        elif phase == "done":
            name_style = "bold bright_white"
        elif phase == "error":
            name_style = "bold red"
        elif phase == "genpept":
            name_style = "bold bright_blue"
        else:
            name_style = "dim"

        stale_tag = " [dim italic](stale)[/]" if s.stale else ""
        name_cell = Text.from_markup(f"[{name_style}]{s.name}[/]{stale_tag}")

        # GENPEPT cell
        if s.genpept_ok:
            gp_cell = Text(f"✓ {s.genpept_surv if s._genpept_surv else '?'}", style="bright_green")
        elif s._genpept_status == "running":
            gp_cell = Text("running…", style="bright_blue")
        else:
            gp_cell = Text("—", style="dim")
        # expose survivor count
        if hasattr(s, '_genpept_surv') and s._genpept_surv:
            gp_cell = Text(f"✓ {s._genpept_surv}", style="bright_green")

        phase_cell = Text(plabel, style=f"bold {pcolor}")

        bar = progress_bar(pct) if pct is not None else Text("░" * 16, style="dim")

        pct_str   = f"[{pcolor}]{pct:.1f}%[/]" if pct is not None else "[dim]—[/]"
        ep        = s.epochs_completed
        ep_str    = str(ep) if ep is not None else "—"
        ck        = s._ckpt_count
        ck_str    = str(ck) if ck is not None else "—"

        eta = s.eta_s
        if eta is not None and eta <= 0:
            eta_cell = Text("done", style="bright_green")
        else:
            eta_cell = Text(fmt_dur(eta), style="white")

        nspd  = s.ns_per_day
        ans   = s.agg_ns
        sps   = s.steps_per_s
        gb    = s.gamd_boost

        table.add_row(
            name_cell,
            gp_cell,
            phase_cell,
            bar,
            Text.from_markup(pct_str),
            ep_str,
            ck_str,
            fmt_dur(s.elapsed_s),
            eta_cell,
            f"{nspd:.0f}" if nspd else "—",
            f"{ans:.2f}"  if ans  else "—",
            f"{sps:.0f}"  if sps  else "—",
            s.cv_range,
            f"{gb:.1f}"   if gb   else "—",
        )

    return table


def build_footer(states: list) -> Text:
    n_prod  = sum(1 for s in states if s.phase == "gareus_production")
    n_adapt = sum(1 for s in states if s.phase == "adaptive_feedback")
    n_done  = sum(1 for s in states if s.phase in ("done", "converged"))
    n_err   = sum(1 for s in states if s.phase == "error")
    n_pend  = sum(1 for s in states if s.phase in ("not_started", "genpept"))
    total_ns = sum(s.agg_ns for s in states if s.agg_ns)

    parts = []
    if n_prod:  parts.append(f"[bright_green]{n_prod} production[/]")
    if n_adapt: parts.append(f"[cyan]{n_adapt} adapting[/]")
    if n_done:  parts.append(f"[bright_white]{n_done} done[/]")
    if n_err:   parts.append(f"[red]{n_err} error[/]")
    if n_pend:  parts.append(f"[dim]{n_pend} pending[/]")
    parts.append(f"[dim]total simulated: {total_ns:.2f} ns[/]")

    return Text.from_markup("   ".join(parts))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GAREUS multi-peptide live monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("runs_dir", nargs="?", default=".",
                        help="Root directory containing peptide subdirs")
    parser.add_argument("--interval", "-i", type=float, default=10.0,
                        help="Refresh interval (seconds)")
    parser.add_argument("--once", action="store_true",
                        help="Print once and exit (no live mode)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    if not runs_dir.is_dir():
        print(f"ERROR: {runs_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    console = Console()
    states  = discover_peptides(runs_dir)

    if not states:
        console.print(f"[red]No peptide runs found in {runs_dir}[/]")
        sys.exit(1)

    console.print(f"[dim]Found {len(states)} peptide(s): {', '.join(s.name for s in states)}[/]")

    def render():
        for s in states:
            s.refresh()
        tbl = build_table(states)
        footer = build_footer(states)
        return Panel(
            tbl,
            subtitle=footer,
            border_style="bright_blue",
            padding=(0, 0),
        )

    if args.once:
        console.print(render())
        return

    try:
        with Live(render(), console=console, refresh_per_second=1, screen=True) as live:
            while True:
                time.sleep(args.interval)
                live.update(render())
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/]")


if __name__ == "__main__":
    main()
