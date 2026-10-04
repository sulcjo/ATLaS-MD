"""Swarm dashboard frames fit every terminal and say the right things."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from gareus.dashboard.swarm_view import DeviceStat, Histogram, SwarmSnapshot, render_swarm_screen

WIDTHS = (40, 55, 70, 80, 100, 120, 140, 200, 400)
HEIGHTS = (18, 20, 24, 35, 45, 55, 80)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
GOLDEN = Path(__file__).parent / "golden" / "dashboard" / "swarm_140x45.txt"
NOW = 1_790_000_000.0


def _snap(n=174, stalled=(), slow=False):
    phases, fracs = [], []
    for i in range(n):
        p = ("done", "production", "production", "equilibrating", "grafting", "graft_wait", "queued", "failed")[i % 8]
        phases.append(p)
        fracs.append({"done": 1.0, "production": (i % 10) / 10, "failed": 0.3}.get(p, 0.0))
    devs = tuple(DeviceStat(str(d), 40, 700.0 if not (slow and d == 3) else 200.0, 0.4, slow=slow and d == 3)
                 for d in range(4))
    hists = (Histogram("cv1", -2.2, 2.2, tuple((i * 7) % 13 for i in range(48)), below=3, above=0),
             Histogram("rg", 0.5, 1.3, tuple((i * 5) % 11 for i in range(48))),
             Histogram("e2e", 0.4, 3.0, tuple((i * 3) % 9 for i in range(48)), above=2))
    events = tuple({"member_id": i, "status": "ok" if i % 5 else "md_failed", "cell_id": f"{i % 4}_1_2",
                    "ns_per_day": 17.0 + i, "wall": NOW - 60 * i} for i in range(20))
    return SwarmSnapshot(now=NOW, run_label="chignolin_10", round_index=0, n_members=n, timestep_fs=3.5,
                         n_prod_steps=1428714, bins=(4, 3, 3), member_ids=tuple(range(n)), phases=tuple(phases),
                         production_fraction=tuple(fracs), elapsed_s=7800.0,
                         mean_fraction=sum(fracs) / n if n else 0.0, aggregate_ns_per_day=2900.0,
                         median_member_ns_per_day=17.0, slowest_member_ns_per_day=11.0,
                         eta_mean_s=17400.0, eta_last_s=26000.0, stalled=tuple(stalled), devices=devs,
                         histograms=hists, cells_visited=21, cells_total=36,
                         completions_per_bin=(0, 0, 1, 3, 5, 2, 0, 4, 6, 3, 1, 2), events=events)


def _visible(line: str) -> int:
    return len(ANSI.sub("", line))


@pytest.mark.parametrize("n", [0, 1, 174, 500])
@pytest.mark.parametrize("w", WIDTHS)
@pytest.mark.parametrize("h", HEIGHTS)
def test_frame_fits_the_terminal(n, w, h):
    text = render_swarm_screen(_snap(n), term_w=w, term_h=h)
    lines = text.split("\n")
    assert len(lines) <= h - 1
    assert max(_visible(x) for x in lines) <= w - 2
    assert "nan" not in ANSI.sub("", text).lower()


def test_glyph_grid_has_one_mark_per_member():
    s = _snap(174)
    text = ANSI.sub("", render_swarm_screen(s, term_w=400, term_h=80))
    grid = "".join(ch for ch in text if ch in "·gGe▁▂▃▅▇✓✗")
    # legend contributes 1 of each glyph family; the grid contributes one per member
    assert grid.count("✓") - 1 == s.count("done")
    assert grid.count("✗") - 1 == s.count("failed")


def test_straggler_eta_and_alerts_are_shown():
    text = ANSI.sub("", render_swarm_screen(_snap(stalled=(7, 9), slow=True), term_w=200, term_h=55))
    assert "ETA last" in text and "slowest member" in text
    assert "2 stalled" in text and "slow gpu 3" in text and "SLOW" in text


def test_ascii_glyphs():
    text = render_swarm_screen(_snap(20), term_w=120, term_h=40, glyphs="ascii")
    assert "✓" not in text and "+" in text


def test_golden_snapshot():
    text = "\n".join(ANSI.sub("", x).rstrip() for x in render_swarm_screen(_snap(), term_w=140, term_h=45).split("\n"))
    text = re.sub(r"\d\d:\d\d:\d\d", "00:00:00", text) + "\n"
    if os.environ.get("GAREUS_UPDATE_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(text)
    assert text == GOLDEN.read_text()
