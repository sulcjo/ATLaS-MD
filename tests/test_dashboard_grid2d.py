"""2D window map: the (CV1, CV2) umbrella layout drawn as a grid of lambda rungs."""

import argparse

import pytest

from gareus import colors
from gareus.dashboard.grid2d import layout_grid, render_grid, window_grid_panel
from gareus.dashboard.ranking import BAD, OK, WARN, WindowStatus
from gareus.tui import strip_ansi, strip_ansi_len

# chignolin_9's real shape: 16 CV1 centres x 5 CV2 centres, 59 occupied cells,
# 4 lambda rungs per cell = 236 windows (measured 2026-09-24 from
# epoch_000/umbrella_explicit_windows.csv).
CV1 = [0.0697 + 0.0579 * i for i in range(16)]
CV2 = [-1.851, -0.785, -0.252, 0.281, 1.348]
LAMBDAS = (0.0, 0.25, 0.5, 1.0)


def _chignolin_9_layout():
    cells = [(c, r) for c in range(16) for r in (2, 3)]           # 32 dense cells
    cells += [(c, 1) for c in range(0, 16, 2)] + [(c, 4) for c in range(1, 16, 2)]  # 16
    cells += [(c, 0) for c in range(0, 16, 3)][:11]              # 6 -> pad to 59 below
    extra = [(c, 0) for c in range(1, 16, 3)]
    cells = (cells + extra)[:59]
    assert len(set(cells)) == 59
    centers, secondary, lam = [], [], []
    # Deliberately interleave rungs across the flat window index, as the real
    # ladder CSV does, so the grid must sort by lambda rather than by index.
    for rung in (2, 0, 3, 1):
        for c, r in cells:
            centers.append(CV1[c])
            secondary.append(CV2[r])
            lam.append(LAMBDAS[rung])
    return centers, secondary, lam


def test_layout_grid_groups_every_window_into_its_cv1_cv2_cell():
    centers, secondary, lam = _chignolin_9_layout()
    grid = layout_grid(centers, secondary, lam)
    assert len(grid.cv1_values) == 16
    assert len(grid.cv2_values) == 5
    assert len(grid.cells) == 59
    assert sorted(w for ws in grid.cells.values() for w in ws) == list(range(236))


def test_layout_grid_orders_each_cell_by_lambda_not_by_window_index():
    centers, secondary, lam = _chignolin_9_layout()
    grid = layout_grid(centers, secondary, lam)
    for windows in grid.cells.values():
        assert [lam[w] for w in windows] == list(LAMBDAS)


def test_layout_grid_puts_the_highest_cv2_on_the_top_row():
    grid = layout_grid([0.1, 0.2], [-1.0, 2.0], [0.0, 0.0])
    assert grid.cv2_values == (2.0, -1.0)


def test_layout_grid_treats_float_noise_as_the_same_centre():
    grid = layout_grid([0.1, 0.1 + 1e-12], [0.5, 0.5], [0.0, 1.0])
    assert len(grid.cells) == 1


def test_render_grid_draws_one_glyph_per_rung_when_it_fits():
    centers, secondary, lam = _chignolin_9_layout()
    grid = layout_grid(centers, secondary, lam)
    lines = render_grid(grid, {}, set(range(236)), width=136, glyphs="unicode")
    body = "".join(strip_ansi(l) for l in lines[1:6])
    assert body.count("■") == 236
    assert all(strip_ansi_len(l) <= 136 for l in lines)


def test_render_grid_falls_back_to_the_worst_rung_per_cell_when_narrow():
    centers, secondary, lam = _chignolin_9_layout()
    grid = layout_grid(centers, secondary, lam)
    status = {0: BAD}
    lines = render_grid(grid, status, set(range(236)), width=50, glyphs="unicode")
    body = "".join(strip_ansi(l) for l in lines[1:6])
    assert body.count("✗") == 1
    assert body.count("■") == 58
    assert all(strip_ansi_len(l) <= 50 for l in lines)


def test_render_grid_gives_up_rather_than_overflowing():
    centers, secondary, lam = _chignolin_9_layout()
    grid = layout_grid(centers, secondary, lam)
    assert render_grid(grid, {}, set(), width=20, glyphs="unicode") is None


def test_render_grid_status_glyphs_read_without_colour():
    grid = layout_grid([0.1, 0.1, 0.1, 0.1], [0.0] * 4, [0.0, 0.25, 0.5, 1.0])
    lines = render_grid(grid, {1: WARN, 2: BAD}, {0, 1, 2}, width=80, glyphs="unicode")
    assert "■!✗·" in strip_ansi(lines[1])
    ascii_lines = render_grid(grid, {1: WARN, 2: BAD}, {0, 1, 2}, width=80, glyphs="ascii")
    assert "#!x." in ascii_lines[1]


def test_render_grid_colours_status_when_colour_is_on(monkeypatch):
    monkeypatch.setattr(colors, "ASCII_COLOR_ENABLED", True)
    grid = layout_grid([0.1, 0.1], [0.0, 0.0], [0.0, 1.0])
    lines = render_grid(grid, {1: BAD}, {0, 1}, width=80, glyphs="unicode")
    assert "\033[31m" in lines[1] and "\033[32m" in lines[1]


def test_render_grid_legend_counts_add_up_to_the_window_total():
    centers, secondary, lam = _chignolin_9_layout()
    grid = layout_grid(centers, secondary, lam)
    status = {w: BAD for w in range(10)} | {w: WARN for w in range(10, 13)}
    legend = strip_ansi(render_grid(grid, status, set(range(230)), width=136, glyphs="unicode")[-1])
    assert "ok 217" in legend and "warn 3" in legend and "bad 10" in legend and "no samples 6" in legend


def _ctx(tmp_path, *, secondary):
    from gareus.dashboard.context import build_context
    from gareus.dashboard.sidecar import SidecarSnapshot
    from gareus.logger import DistanceLogger

    n = len(secondary)
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    centers = [0.1 + 0.1 * (i // 2) for i in range(n)]
    for w in range(n):
        logger.history_by_window[w] = [centers[w]] * 20
    rows = [{"replica": w, "window": w, "center_A": centers[w], "k_kcal_mol_A2": 2.5,
             "cv_A": centers[w], "gamd_lambda": float(w % 2)} for w in range(n)]
    info = {"centers_a": centers, "n_windows": n, "k_list": [2.5] * n,
            "primary_cv_label": "contacts", "primary_cv_units": "",
            "exchange_stats": {}}
    if any(s is not None for s in secondary):
        info["secondary_cv_centers"] = list(secondary)
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=1, total_steps=10,
        summary={}, dashboard_info=info, sidecar=SidecarSnapshot(), term_w=140, term_h=45,
        now=0.0, view="windows", glyphs="unicode",
    )


def test_window_grid_panel_survives_fewer_centres_than_windows(tmp_path):
    """A mid-update layout (n_windows ahead of centers_a) skips the panel, no IndexError."""
    from dataclasses import replace

    ctx = _ctx(tmp_path, secondary=[0.0, 1.0] * 3)
    if not ctx.is_2d:
        pytest.skip("fixture did not produce a 2D context")
    short = replace(ctx, centers_a=ctx.centers_a[:-2])
    assert window_grid_panel(short, [WindowStatus(w, 0, OK) for w in range(ctx.n_windows)]) is None


def test_window_grid_panel_is_absent_for_a_1d_run(tmp_path):
    ctx = _ctx(tmp_path, secondary=[None] * 6)
    assert window_grid_panel(ctx, [WindowStatus(w, 0, OK) for w in range(6)]) is None


def test_windows_view_shows_the_grid_for_a_2d_run(tmp_path):
    from gareus.dashboard import view_windows

    ctx = _ctx(tmp_path, secondary=[0.0, 1.0] * 3)
    if not ctx.is_2d:
        pytest.skip("fixture did not produce a 2D context")
    keys = [p.key for row in view_windows.build(ctx) for p in row.panels]
    assert "window_grid" in keys


def test_progress_view_offers_the_grid_at_its_lowest_priority(tmp_path):
    from gareus.dashboard import view_progress

    ctx = _ctx(tmp_path, secondary=[0.0, 1.0] * 3)
    if not ctx.is_2d:
        pytest.skip("fixture did not produce a 2D context")
    panels = [p for row in view_progress.build(ctx) for p in row.panels]
    grid = [p for p in panels if p.key == "window_grid"]
    assert len(grid) == 1
    assert grid[0].priority == max(p.priority for p in panels)


def test_unrestrained_windows_get_their_own_row_not_a_fake_grid_column():
    # chignolin_9's zero-k stack: 4 unrestrained rungs parked at CV1 0.502,
    # which would otherwise render as a phantom column next to the real 0.50.
    grid = layout_grid([0.5005, 0.5005, 0.3], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    lines = render_grid(grid, {}, {0, 1, 2, 10, 11}, width=80, glyphs="unicode",
                        free_windows=(10, 11, 12))
    text = [strip_ansi(l) for l in lines]
    assert any(l.startswith("free") and "■■·" in l for l in text)
    assert "ok 5" in text[-1] and "no samples 1" in text[-1]


def test_window_grid_panel_moves_zero_k_windows_off_the_grid(tmp_path):
    from dataclasses import replace

    ctx = _ctx(tmp_path, secondary=[0.0, 1.0] * 3)
    if not ctx.is_2d:
        pytest.skip("fixture did not produce a 2D context")
    n = ctx.n_windows
    ctx = replace(ctx, k_list=(0.0,) + ctx.k_list[1:], secondary_k=(0.0,) + (2.0,) * (n - 1))
    p = window_grid_panel(ctx, [WindowStatus(w, 0, OK) for w in range(n)])
    text = [strip_ansi(l) for l in p.lines]
    free_rows = [l for l in text if l.startswith("free")]
    assert len(free_rows) == 1 and free_rows[0].count("■") == 1
    grid_rows = [l for l in text[1:] if not l.startswith("free") and "ok " not in l]
    assert sum(l.count("■") for l in grid_rows) == n - 1


def test_a_column_with_no_cv1_restraint_is_labelled_free_not_with_a_centre():
    # chignolin_9: 20 CV2-only windows share CV1 0.502 (unrestrained on CV1),
    # right next to the real 0.5005 column -- two "0.50" labels otherwise.
    # 4 rungs per cell, so each column is wide enough to carry its own label.
    centres = [c for c in (0.5005, 0.502063, 0.3) for _ in LAMBDAS]
    grid = layout_grid(centres, [0.0] * 12, list(LAMBDAS) * 3)
    header = strip_ansi(render_grid(grid, {}, set(range(12)), width=80, glyphs="unicode",
                                    cv1_free_columns={2})[0])
    assert header.split() == ["0.30", "0.50", "free"]


def test_a_cv2_only_window_stays_on_the_grid(tmp_path):
    """Primary k = 0 alone is not 'free': that window is restrained on CV2."""
    from dataclasses import replace

    ctx = _ctx(tmp_path, secondary=[0.0, 1.0] * 3)
    if not ctx.is_2d:
        pytest.skip("fixture did not produce a 2D context")
    n = ctx.n_windows
    for secondary_k in ((2.0,) * n, ()):          # known CV2 k, and CV2 k unknown
        c = replace(ctx, k_list=(0.0,) + ctx.k_list[1:], secondary_k=secondary_k)
        text = [strip_ansi(l) for l in window_grid_panel(
            c, [WindowStatus(w, 0, OK) for w in range(n)]).lines]
        assert not any(l.startswith("free") for l in text)
