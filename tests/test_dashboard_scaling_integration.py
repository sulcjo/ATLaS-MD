import argparse
import collections
import os
import re

import pytest

from gareus.logger import DistanceLogger
from gareus.tui import _weighted_panel_widths, strip_ansi

# Matches a bracketed hist3d bar, e.g. "[▁▂▃·◉●...]" — see
# gareus/tui.py's `_hist3d_cell` glyph set plus the │/◉/● markers used for
# umbrella-center/current-value/overlap.
_BAR_RE = re.compile(r"\[([·▁▂▃▄▅▆▇█ │◉●]+)\]")

# Matches the spine's "cv1 <lo> |<bar>| <hi>" coverage-bar line — see
# gareus/dashboard/spine.py's `spine_lines`, which uses `_coverage_bar`'s glyph
# set (" ░▒▓█" plus "·" for empty bins). Accepts the historical "cov" prefix
# too, in case a fixture/log predating the screen-engine rework is replayed.
_COV_BAR_RE = re.compile(r"^(?:cov|cv1) .*?\|([ ░▒▓█·]+)\|", re.MULTILINE)

# Matches the potential-energy histogram bar's "kJ |<bar>|" suffix — see
# gareus/dashboard/panels.py's `pe_map_panel` (rendered by the WINDOWS view),
# which uses the glyph set " ·░▒▓█" plus the "●" current-value marker.
_PE_BAR_RE = re.compile(r"kJ \|([ ·░▒▓█●]+)\|")


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
    logger.tui_view = "windows"          # the view that owns the CV and PE map panels
    return logger._render_screen_frame(
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


def test_cv1_coverage_bar_grows_past_old_ceiling_on_large_terminal(tmp_path, monkeypatch):
    # Same bug class as the CV histogram bar above, in a different, always-
    # rendered header line: the CV1 coverage bar (now `gareus/dashboard/spine.py`'s
    # `spine_lines`, previously `_render_compact_header`'s `cov_w`) was
    # independently capped at 48 columns regardless of terminal width, unrelated
    # to and missed by the earlier `cov2_w` fix (which only covers the 2D-run
    # secondary-CV coverage line).
    logger = _make_logger(tmp_path)
    # `spine_lines` renders the cv1 bar from accumulated `history_by_window`
    # (matching `_render_screen_frame`'s own not-populated-by-raw-`rows`
    # behavior -- see the identical seeding note on the PE test below); with no
    # history at all it prints "no samples yet" instead of a bar, same as the
    # old `_render_compact_header` did with an empty `history_by_replica`
    # (`_coverage_bar` degrades to `" " * width` rather than omitting the line,
    # so the old test passed without seeding -- the new "no samples yet" text
    # is a deliberate improvement, but it means this test needs real history
    # now to have any bar at all to measure).
    for w in range(4):
        logger.history_by_window[w] = collections.deque([float(w) + 0.05 * (i % 5) for i in range(20)])
    out = strip_ansi(_render_at(monkeypatch, logger, 500, 60))
    matches = _COV_BAR_RE.findall(out)
    assert matches, "expected a 'cov ...|...|' coverage bar line in dashboard output"
    assert len(matches[0]) > 48
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


def test_pe_histogram_bar_width_pinned_at_160x40(tmp_path, monkeypatch):
    # The CV/PE histogram row switched from hand-split ceilings to a weighted
    # proportional split (`_weighted_panel_widths([2.4, 0.9], ...)`). At the
    # most common terminal size (160x40) this narrows the PE histogram bar
    # from the old hand-split 28 columns to 18 -- an intended, reviewed
    # redesign, but nothing pinned the new value down. Pin it here so any
    # future change to the weight/offset arithmetic is a deliberate, visible
    # test change instead of silent drift.
    logger = _make_logger(tmp_path)
    # `_render_screen_frame` doesn't itself populate potential-energy history
    # (that happens in `_update_history`, called by the public logging
    # entrypoint, not exercised by this test's direct `_render_screen_frame`
    # call) -- seed it directly so `gareus/dashboard/panels.py`'s `pe_map_panel`
    # renders real histogram bars instead of its "PE unavailable" early-return.
    for rep in range(4):
        logger.potential_history_by_replica[rep] = collections.deque(
            [100.0 + rep + i for i in range(20)], maxlen=logger.history_limit
        )
    out = strip_ansi(_render_at(monkeypatch, logger, 160, 40))
    matches = _PE_BAR_RE.findall(out)
    assert matches, "expected a 'kJ |<bar>|' potential-energy histogram bar in dashboard output"
    assert len(matches[0]) == 18
    logger.close()


def test_pe_bar_width_arithmetic_pinned_at_160x40():
    # Narrowly-scoped companion to the rendered-output test above: mirrors
    # the exact arithmetic now in `gareus/dashboard/view_windows.py`'s
    # `_map_bar_widths` (weight split then `pe_bar_width = max(10, pe_panel_w - 24)`,
    # ported unchanged from the removed `_render_dashboard`) directly, so this
    # still catches a silent width change even if the rendered-text format
    # around the bar changes for unrelated reasons.
    _cv_panel_w, pe_panel_w = _weighted_panel_widths(
        [2.4, 0.9], term_w=160, gap=3, min_panel_width=30
    )
    pe_bar_width = max(10, pe_panel_w - 24)
    assert pe_panel_w - 24 == 18
    assert pe_bar_width == 18
