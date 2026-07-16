import argparse
import os
import re

import pytest

from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi

# Matches a bracketed hist3d bar, e.g. "[▁▂▃·◉●...]" — see
# gareus/tui.py's `_hist3d_cell` glyph set plus the │/◉/● markers used for
# umbrella-center/current-value/overlap.
_BAR_RE = re.compile(r"\[([·▁▂▃▄▅▆▇█ │◉●]+)\]")


def _make_logger(tmp_path):
    args = argparse.Namespace()
    return DistanceLogger(tmp_path, args, no_file_persistence=True)


def _sample_rows():
    return [
        {
            "replica": i, "window": i, "center_A": float(i), "k_kcal_mol_A2": 1.0,
            "cv_A": float(i) + 0.1,
        }
        for i in range(4)
    ]


def _render_at(monkeypatch, logger, term_w, term_h):
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((term_w, term_h))
    )
    return logger._render_dashboard(
        _sample_rows(), phase="production", step=100, total_steps=1000,
        summary={}, dashboard_info={"centers_a": [0.0, 1.0, 2.0, 3.0], "n_windows": 4},
    )


def test_cv_histogram_bar_grows_past_old_ceiling_on_large_terminal(tmp_path, monkeypatch):
    # The panel *container* already scaled with terminal width even before
    # this rework (padding fills it regardless). The actual bug was the
    # histogram *bar content* inside it being ceiling-capped at 110/140
    # columns regardless of how much container space was available. Assert
    # directly on the bar glyph count, not on container/line width.
    logger = _make_logger(tmp_path)
    out = strip_ansi(_render_at(monkeypatch, logger, 500, 60))
    bar_matches = _BAR_RE.findall(out)
    assert bar_matches, "expected at least one bracketed histogram bar in dashboard output"
    max_bar_len = max(len(b) for b in bar_matches)
    # Old code capped this at 140 (full-density tier) regardless of terminal
    # width; a 500-wide terminal must clearly exceed that fixed ceiling.
    assert max_bar_len > 140
    logger.close()


def test_dashboard_renders_at_tiny_terminal_without_crashing(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path)
    out = _render_at(monkeypatch, logger, 80, 24)
    assert isinstance(out, str)
    assert len(out) > 0
    logger.close()


def test_dashboard_renders_at_typical_terminal(tmp_path, monkeypatch):
    logger = _make_logger(tmp_path)
    out = _render_at(monkeypatch, logger, 160, 40)
    assert isinstance(out, str)
    assert "production" in strip_ansi(out)
    logger.close()
