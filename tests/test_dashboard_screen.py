import argparse
import collections
import os
import random
import shutil

from gareus.dashboard.context import build_context
from gareus.dashboard.screen import (
    footer_line,
    promotion_reasons,
    render_screen,
    resolve_view,
)
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi, strip_ansi_len

CENTERS = tuple(4.0 + 0.55 * i for i in range(8))
_RNG = random.Random(11)


def _ctx(tmp_path, *, view="auto", term_w=140, term_h=45, exchange=None, sidecar=None):
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=4.0), no_file_persistence=True)
    for w in range(8):
        # A HEALTHY fixture, built deliberately. Windows sit 0.55 A apart and the
        # restraint's own width at k=2.5, 300 K is sigma = 0.488 A, so samples must
        # spread at roughly that scale for neighbours to overlap the way umbrella
        # sampling intends. The obvious narrow ramp spans only 0.2 A, which yields
        # histogram overlap of EXACTLY 0.0 on every pair and ranks all eight windows
        # BAD -- so `auto` promotes away from PROGRESS and the healthy-run tests below
        # fail while looking like defects in the promotion logic. Measured with this
        # seed and spread: overlap 0.35-0.62, every window ok.
        logger.history_by_window[w] = [CENTERS[w] + _RNG.gauss(0.0, 0.35)
                                      for _ in range(40)]
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05, "umbrella_bias_kcal_mol": 0.0,
             "umbrella_pull_kcal_mol_A": 0.0} for w in range(8)]
    stats = exchange if exchange is not None else {
        f"{i}-{i+1}": {"attempts": 40, "accepted": 12} for i in range(7)}
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=100, total_steps=1000,
        summary={}, dashboard_info={"centers_a": list(CENTERS), "n_windows": 8,
                                    "k_list": [2.5] * 8, "exchange_stats": stats,
                                    "primary_cv_label": "contacts", "primary_cv_units": "A",
                                    "primary_k_units": "kcal/mol/A^2"},
        sidecar=sidecar or SidecarSnapshot(), term_w=term_w, term_h=term_h, now=1000.0,
        view=view, glyphs="unicode",
    )


def test_resolve_view_honours_an_explicit_request(tmp_path):
    assert resolve_view(_ctx(tmp_path, view="physics"), "physics") == "physics"
    assert resolve_view(_ctx(tmp_path, view="windows"), "windows") == "windows"


def test_resolve_view_defaults_to_progress_on_a_healthy_run(tmp_path):
    assert resolve_view(_ctx(tmp_path), "auto") == "progress"


def test_resolve_view_promotes_to_windows_when_a_pair_is_dead(tmp_path):
    ctx = _ctx(tmp_path, exchange={"0-1": {"attempts": 40, "accepted": 0},
                                   **{f"{i}-{i+1}": {"attempts": 40, "accepted": 12}
                                      for i in range(1, 7)}})
    assert resolve_view(ctx, "auto") == "windows"
    assert any("dead" in r for r in promotion_reasons(ctx))


def test_auto_does_not_promote_before_any_exchange_is_attempted(tmp_path):
    """Every run starts here. Without a measured pair, every window is its own
    component, so a naive connectivity check would promote to PHYSICS on frame one
    and stay there until exchanges accumulate."""
    ctx = _ctx(tmp_path, exchange={})
    assert resolve_view(ctx, "auto") == "progress"
    assert not any("graph" in r for r in promotion_reasons(ctx))


def test_auto_promotes_to_physics_on_a_proven_split(tmp_path):
    """Every window tried, graph still in pieces -- the one case that justifies
    taking over the screen."""
    ctx = _ctx(tmp_path, exchange={
        "0-1": {"attempts": 40, "accepted": 12}, "1-2": {"attempts": 40, "accepted": 0},
        "2-3": {"attempts": 40, "accepted": 12}, "3-4": {"attempts": 40, "accepted": 12},
        "4-5": {"attempts": 40, "accepted": 12}, "5-6": {"attempts": 40, "accepted": 12},
        "6-7": {"attempts": 40, "accepted": 12}})
    assert any("graph" in r for r in promotion_reasons(ctx))


def test_resolve_view_falls_back_to_progress_for_an_unknown_name(tmp_path):
    assert resolve_view(_ctx(tmp_path), "nonsense") == "progress"


def test_render_screen_gives_unused_spine_lines_to_the_view(tmp_path):
    """A 1D run's spine needs eight of its ten reserved lines; the rest must
    become view content rather than blank rows."""
    lines = strip_ansi(render_screen(_ctx(tmp_path, term_w=140, term_h=45))).splitlines()
    assert [l for l in lines if not l.strip()] == []
    assert len(lines) <= 44


def test_render_screen_never_exceeds_the_terminal_height_or_width(tmp_path):
    out = render_screen(_ctx(tmp_path, term_w=140, term_h=45))
    lines = strip_ansi(out).splitlines()
    assert len(lines) <= 45 - 1
    assert all(strip_ansi_len(l) <= 140 - 2 for l in lines)


def test_render_screen_puts_the_spine_first_and_the_footer_last(tmp_path):
    lines = strip_ansi(render_screen(_ctx(tmp_path))).splitlines()
    assert lines[0].startswith("GaREUS")
    assert "PROGRESS" in lines[-1] or "progress" in lines[-1]


def test_render_screen_degrades_to_spine_only_on_a_very_short_terminal(tmp_path):
    lines = strip_ansi(render_screen(_ctx(tmp_path, term_h=18))).splitlines()
    assert 0 < len(lines) <= 17
    assert lines[0].startswith("GaREUS")


def test_footer_line_names_dropped_panels_instead_of_hiding_them(tmp_path):
    footer = strip_ansi(footer_line(_ctx(tmp_path), ["pe_map", "cv_map"], "windows"))
    assert "dropped" in footer
    assert "pe_map" in footer and "cv_map" in footer


# --- Regressions found while integrating: the brief's given code passed these
# names/signatures but not these three defects. None of the tests above exercise
# a width/height narrow enough to reach them.

def test_render_screen_respects_a_narrow_terminal_width_on_every_view(tmp_path):
    """spine.py's own width floor of 40 (and the same floor copied into
    footer_line) let spine/footer lines run up to 40 columns even when the real
    budget (term_w - 2) is smaller -- reproducible at any term_w < 42. Every
    downstream bar/cell width already floors itself independently (8, 12, ...),
    so the outer width has no legitimate reason to exceed the real budget."""
    for view in ("progress", "physics", "windows"):
        lines = strip_ansi(render_screen(_ctx(tmp_path, view=view, term_w=40))).splitlines()
        assert all(strip_ansi_len(l) <= 40 - 2 for l in lines), view


def test_render_screen_fits_every_view_on_an_18_row_terminal(tmp_path):
    """frame_tiers' short-terminal tier hands back body_budget=0 specifically
    because there is no room for a view body *or* a footer once the spine fills
    it. Reclaiming the spine's unused reservation without a ceiling manufactured
    view+footer room that was never really there -- physics/windows rendered 18
    lines into a 17-line budget at term_h=18, one line over every time,
    regardless of terminal width. Only 'auto' (-> progress on this healthy
    fixture) was covered above; physics/windows need their own explicit check."""
    for view in ("progress", "physics", "windows"):
        lines = strip_ansi(render_screen(_ctx(tmp_path, view=view, term_h=18))).splitlines()
        assert 0 < len(lines) <= 17, (view, len(lines))


def _rich_physics_ctx(tmp_path, term_w, term_h, n=25):
    """25 windows and real GaMD boost data, so overlap/boost panels have enough
    real content to approach their declared want_lines -- the thin 8-window,
    no-GaMD default fixture above renders panels so short (a handful of real
    lines against a much larger min_lines budget) that an allocator mispricing
    a row's cost never surfaces as a visible overflow, masking the defect
    below entirely."""
    centers = [4.0 + 0.55 * i for i in range(n)]
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=4.0),
                            no_file_persistence=True)
    for w in range(n):
        logger.history_by_window[w] = [centers[w] + _RNG.gauss(0.0, 0.35) for _ in range(60)]
    logger.boost_history_all = [2.0 + 0.5 * (i % 7) for i in range(300)]
    rows = [{"replica": w, "window": w, "center_A": centers[w], "k_kcal_mol_A2": 2.5,
             "cv_A": centers[w] + 0.05, "umbrella_bias_kcal_mol": 0.0,
             "umbrella_pull_kcal_mol_A": 0.0} for w in range(n)]
    stats = {f"{i}-{i+1}": {"attempts": 40, "accepted": 12} for i in range(n - 1)}
    info = {"centers_a": centers, "n_windows": n, "k_list": [2.5] * n,
            "exchange_stats": {"mode": "neighbor", "pairs": stats},
            "primary_cv_label": "contacts", "primary_cv_units": "A",
            "primary_k_units": "kcal/mol/A^2"}
    gamd = {"joint_envelope": {"Dihedral": {"sigma0_kj_mol": 12.552,
                                            "sigmaV_kj_mol": 11.039, "k0": 1.0}}}
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=100, total_steps=1000,
        summary={}, dashboard_info=info, sidecar=SidecarSnapshot(gamd=gamd),
        term_w=term_w, term_h=term_h, now=1000.0, view="physics", glyphs="unicode",
    )


def test_render_screen_prices_a_stacking_row_consistently_with_compose_rows(tmp_path):
    """allocate_rows must be given the same term_w as compose_rows. At term_w=45
    the physics view's overlap+boost row stacks (two side-by-side panels no
    longer fit); pricing the allocation without term_w defaults to side-by-side
    cost, under-budgeting the row compose_rows then renders stacked -- the
    under-costed row silently overflows the terminal height. Reproduced directly:
    omitting term_w from the allocate_rows call rendered 31 lines into a 29-line
    budget at term_w=45, term_h=30 with this fixture."""
    ctx = _rich_physics_ctx(tmp_path, term_w=45, term_h=30)
    lines = strip_ansi(render_screen(ctx)).splitlines()
    assert len(lines) <= 30 - 1, len(lines)
    assert all(strip_ansi_len(l) <= 45 - 2 for l in lines)


# --- Task 12: DistanceLogger wired to the screen engine ---

def test_logger_render_screen_frame_fits_and_names_the_phase(tmp_path, monkeypatch):
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=4.0),
                            no_file_persistence=True)
    for w in range(8):
        logger.history_by_window[w] = collections.deque(
            [CENTERS[w] + 0.05 * (i % 5) for i in range(30)])
    monkeypatch.setattr("shutil.get_terminal_size",
                        lambda fallback=None: os.terminal_size((160, 40)))
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05} for w in range(8)]
    frame = logger._render_screen_frame(
        rows, "gareus_production", 100, 1000, {},
        {"centers_a": list(CENTERS), "n_windows": 8, "k_list": [2.5] * 8},
    )
    lines = strip_ansi(frame).splitlines()
    assert len(lines) <= 39
    assert all(strip_ansi_len(l) <= 158 for l in lines)
    assert "gareus_production" in strip_ansi(frame)


def test_logger_reuses_one_sidecar_cache_across_frames(tmp_path, monkeypatch):
    logger = DistanceLogger(tmp_path, argparse.Namespace(), no_file_persistence=True)
    monkeypatch.setattr("shutil.get_terminal_size",
                        lambda fallback=None: os.terminal_size((120, 40)))
    first = logger._sidecar
    logger._render_screen_frame([], "p", 1, 2, {}, {"centers_a": [], "n_windows": 0})
    assert logger._sidecar is first
