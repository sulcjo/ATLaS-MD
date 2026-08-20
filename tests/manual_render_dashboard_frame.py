#!/usr/bin/env python3
"""Render the live dashboard frame at fixed terminal sizes and report overflow.

Not collected by pytest (no ``test_`` prefix): this is the harness behind the
evidence table in ``docs/superpowers/specs/2026-08-19-live-dashboard-tui-screen-redesign-design.md``
(section 1), and the intended fixture for that design's fit property test.

It drives the real ``DistanceLogger._render_screen_frame`` with synthetic-but-realistic
data — 25 windows, populated per-window CV histories, a full neighbour exchange
table — then compares the produced line count against what the terminal could
actually display (``term_h - 1``, matching ``_safe_tui_frame_text``'s own reserve).

Usage:  python tests/manual_render_dashboard_frame.py [--dump DIR]
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

# Runnable directly from a checkout, not just under pytest's rootdir.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi

N_WINDOWS = 25
CENTERS = [4.0 + 0.55 * i for i in range(N_WINDOWS)]
HISTORY_PER_WINDOW = 300
# (label, term_w, term_h, is_2d) — the three cases quoted in the design spec.
CASES = (
    ("200x55 1D", 200, 55, False),
    ("140x45 1D", 140, 45, False),
    ("200x55 2D", 200, 55, True),
)


def _rows(rng: random.Random) -> list[dict]:
    return [
        {
            "replica": i,
            "window": i,
            "center_A": CENTERS[i],
            "k_kcal_mol_A2": 2.5,
            "cv_A": CENTERS[i] + rng.gauss(0.0, 0.35),
            "potential_energy_kj": -31500.0 + rng.gauss(0.0, 120.0),
            "gamd_boost_total_kj": max(0.0, rng.gauss(10.0, 8.0)),
            "gamd_boost_dihedral_kj": max(0.0, rng.gauss(10.0, 8.0)),
        }
        for i in range(N_WINDOWS)
    ]


def _exchange_stats(rng: random.Random) -> dict:
    stats = {}
    for i in range(N_WINDOWS - 1):
        attempts = rng.randint(20, 60)
        stats[f"{i}-{i+1}"] = {
            "attempts": attempts,
            "accepted": int(attempts * rng.uniform(0.05, 0.55)),
        }
    return stats


def _force_terminal_size(term_w: int, term_h: int) -> None:
    """Pin the terminal size every module the frame path measures it through."""
    import gareus.logger as _logger
    import gareus.tui as _tui

    sized = lambda fallback=None: os.terminal_size((term_w, term_h))  # noqa: E731
    shutil.get_terminal_size = sized
    _logger.shutil.get_terminal_size = sized
    _tui.shutil.get_terminal_size = sized


def render_frame(term_w: int, term_h: int, is_2d: bool, out_dir: Path) -> str:
    rng = random.Random(7)
    args = argparse.Namespace()
    args.timestep_fs = 2.0
    logger = DistanceLogger(out_dir, args, no_file_persistence=True)
    rows = _rows(rng)
    for i in range(N_WINDOWS):
        samples = [CENTERS[i] + rng.gauss(0.0, 0.4) for _ in range(HISTORY_PER_WINDOW)]
        logger.history_by_window[i] = samples
        logger.history_by_replica[i] = list(samples)
        logger.secondary_history_by_replica[i] = [rng.gauss(0.0, 1.2) for _ in range(200)]

    info = {
        "centers_a": CENTERS,
        "n_windows": N_WINDOWS,
        "exchange_stats": _exchange_stats(rng),
        "display_step": 12_450_000,
        "display_total_steps": 37_500_000,
        "eta_start_wall": time.time() - 6 * 3600,
        "primary_cv_label": "nonlocal contacts",
        "primary_cv_units": "A",
        "primary_k_units": "kcal/mol/A^2",
        "adaptive_phase": {"epoch_index": 1, "epoch_total": 2, "segment_name": "epoch_001"},
        "dashboard_panels": "normal",
        "render_heavy_panels": True,
    }
    if is_2d:
        targets = [-2.0, -1.0, 0.0, 1.0, 2.0]
        info["secondary_cv_centers"] = [targets[i % len(targets)] for i in range(N_WINDOWS)]
        info["secondary_cv"] = {"explicit_2d_windows": True, "grid": False, "type": "tica-linear"}
        info["explicit_2d"] = True
        for row in rows:
            row["secondary_cv"] = targets[row["window"] % len(targets)] + rng.gauss(0.0, 0.3)

    logger.tui_view = "windows"          # densest view, closest to the old single frame
    _force_terminal_size(term_w, term_h)
    return logger._render_screen_frame(
        rows, "gareus_production", info["display_step"],
        info["display_total_steps"], logger.summarize(rows), info,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", default="", help="Directory to write ANSI-stripped frames into.")
    opts = parser.parse_args()
    dump_dir = Path(opts.dump) if opts.dump else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'case':<12} {'lines':>7} {'display':>8} {'dropped':>9} {'loss':>6}  max_cols")
    for label, term_w, term_h, is_2d in CASES:
        frame = strip_ansi(render_frame(term_w, term_h, is_2d, Path(os.environ.get("TMPDIR", "/tmp"))))
        lines = frame.splitlines()
        displayable = term_h - 1
        dropped = max(0, len(lines) - displayable)
        loss = 100.0 * dropped / max(1, len(lines))
        widest = max((len(line) for line in lines), default=0)
        print(f"{label:<12} {len(lines):>7} {displayable:>8} {dropped:>9} {loss:>5.0f}%  {widest}")
        if dump_dir is not None:
            (dump_dir / f"frame_{label.replace(' ', '_')}.txt").write_text(frame)


if __name__ == "__main__":
    main()
