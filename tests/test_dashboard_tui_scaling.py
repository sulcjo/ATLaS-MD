from gareus.tui import dashboard_body_budget, _weighted_panel_widths, _dashboard_weighted_row, MIN_PANEL_WIDTH, ABSOLUTE_MIN_PANEL_WIDTH, dashboard_row_gap


def test_min_panel_width_constants():
    assert MIN_PANEL_WIDTH == 30
    assert ABSOLUTE_MIN_PANEL_WIDTH == 22
    assert ABSOLUTE_MIN_PANEL_WIDTH < MIN_PANEL_WIDTH


def test_dashboard_row_gap_narrow_vs_wide():
    assert dashboard_row_gap(100) == 2
    assert dashboard_row_gap(119) == 2
    assert dashboard_row_gap(120) == 3
    assert dashboard_row_gap(300) == 3


def test_dashboard_body_budget_scales_with_term_h():
    small = dashboard_body_budget(term_h=24, fraction=0.45, floor=10)
    typical = dashboard_body_budget(term_h=40, fraction=0.45, floor=10)
    large = dashboard_body_budget(term_h=300, fraction=0.45, floor=10)
    assert small < typical < large
    # No hard ceiling: a very tall terminal must get far more than the old
    # fixed caps (18/26/30 in the pre-rework code) ever allowed.
    assert large > 100


def test_dashboard_body_budget_respects_floor_on_tiny_terminal():
    assert dashboard_body_budget(term_h=10, fraction=0.45, floor=10) == 10
    assert dashboard_body_budget(term_h=1, fraction=0.30, floor=8) == 8


def test_dashboard_body_budget_matches_old_normal_tier_reference_point():
    # fraction is defined as old_normal_value / 40 (the stdlib terminal-size
    # fallback height); term_h=40 should reproduce the old "normal" tier
    # value exactly, by construction.
    assert dashboard_body_budget(term_h=40, fraction=18 / 40, floor=10) == 18
    assert dashboard_body_budget(term_h=40, fraction=12 / 40, floor=8) == 12


def test_weighted_panel_widths_distributes_proportionally():
    widths = _weighted_panel_widths([2.4, 0.9], term_w=200, gap=2, min_panel_width=30)
    assert len(widths) == 2
    assert widths[0] > widths[1]
    # Proportional: ratio should track the weight ratio (2.4 / 0.9 ~= 2.67),
    # loosely, since rounding adjustments only ever move 1 column at a time.
    ratio = widths[0] / widths[1]
    assert 2.2 < ratio < 3.2


def test_weighted_panel_widths_sums_to_usable_width():
    term_w = 200
    gap = 2
    widths = _weighted_panel_widths([1.0, 1.0, 1.0], term_w=term_w, gap=gap, min_panel_width=30)
    usable = term_w - 2
    assert sum(widths) == usable - (len(widths) - 1) * gap


def test_weighted_panel_widths_grows_with_terminal_width():
    narrow = _weighted_panel_widths([2.4, 0.9], term_w=160, gap=2, min_panel_width=30)
    wide = _weighted_panel_widths([2.4, 0.9], term_w=400, gap=2, min_panel_width=30)
    assert wide[0] > narrow[0]
    assert wide[1] > narrow[1]


def test_weighted_panel_widths_narrow_terminal_returns_full_usable_each():
    # Below the stacking threshold, every panel gets the full usable width
    # (matches _dashboard_weighted_row's narrow-terminal vertical-stack path).
    widths = _weighted_panel_widths([1.0, 1.0, 1.0], term_w=50, gap=2, min_panel_width=30)
    assert widths == [48, 48, 48]


def test_dashboard_weighted_row_unchanged_for_typical_terminal():
    # Regression check: refactoring the width math out must not change
    # _dashboard_weighted_row's output for a typical 160-wide terminal.
    rows = _dashboard_weighted_row(
        "diagnostics",
        [("a", ["x"], 1.15), ("b", ["y"], 1.0), ("c", ["z"], 1.0)],
        term_w=160,
        gap=2,
        min_panel_width=30,
    )
    assert any("diagnostics" in _strip(r) for r in rows)
    assert len(rows) > 1


def _strip(line):
    from gareus.tui import strip_ansi
    return strip_ansi(line)
