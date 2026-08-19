import pytest

from gareus.tui import strip_ansi_len
from gareus.tui_screen import (
    PANEL_CHROME_LINES,
    Panel,
    Row,
    allocate_rows,
    compose_rows,
    frame_tiers,
    row_min,
    row_want,
    trim_panel,
)


def _panel(key, n_lines, min_lines, want_lines, priority=5):
    return Panel(
        key=key,
        title=key,
        lines=tuple(f"{key}-{i}" for i in range(n_lines)),
        min_lines=min_lines,
        want_lines=want_lines,
        priority=priority,
    )


def test_row_min_and_want_add_chrome_and_take_the_max_over_panels():
    row = Row(panels=(_panel("a", 9, 3, 8), _panel("b", 9, 5, 6)))
    assert row_min(row) == 5 + PANEL_CHROME_LINES
    assert row_want(row) == 8 + PANEL_CHROME_LINES


def test_allocate_rows_fills_budget_without_exceeding_it():
    rows = [Row(panels=(_panel("a", 30, 4, 20, priority=1),)),
            Row(panels=(_panel("b", 30, 4, 20, priority=2),))]
    allocated, dropped = allocate_rows(rows, budget=40)
    assert dropped == ()
    total = sum(body + PANEL_CHROME_LINES for _row, body in allocated)
    assert total <= 40
    # Highest priority row is fed first, so it reaches `want` before the other.
    assert allocated[0][1] == 20


def test_allocate_rows_never_returns_less_than_min_lines():
    rows = [Row(panels=(_panel("a", 30, 6, 20, priority=1),))]
    allocated, dropped = allocate_rows(rows, budget=10)
    assert dropped == ()
    assert allocated[0][1] == 6


def test_allocate_rows_drops_lowest_priority_rows_and_reports_their_keys():
    rows = [Row(panels=(_panel("keep", 30, 8, 20, priority=1),)),
            Row(panels=(_panel("drop", 30, 8, 20, priority=9),))]
    allocated, dropped = allocate_rows(rows, budget=12)
    assert [p.key for row, _body in allocated for p in row.panels] == ["keep"]
    assert dropped == ("drop",)


def test_allocate_rows_returns_nothing_when_even_one_min_row_cannot_fit():
    rows = [Row(panels=(_panel("a", 30, 8, 20, priority=1),))]
    allocated, dropped = allocate_rows(rows, budget=3)
    assert allocated == ()
    assert dropped == ("a",)


def test_allocate_rows_preserves_declaration_order_in_its_output():
    rows = [Row(panels=(_panel("first", 5, 1, 3, priority=9),)),
            Row(panels=(_panel("second", 5, 1, 3, priority=1),))]
    allocated, _dropped = allocate_rows(rows, budget=40)
    assert [p.key for row, _b in allocated for p in row.panels] == ["first", "second"]


def test_trim_panel_keeps_the_worst_first_lines_and_appends_a_generated_tail():
    panel = _panel("a", 10, 1, 10)
    trimmed = trim_panel(panel, 4)
    assert len(trimmed.lines) == 4
    assert trimmed.lines[:3] == ("a-0", "a-1", "a-2")
    assert trimmed.lines[-1] == "… 7 more"


def test_trim_panel_reports_the_severity_composition_of_what_it_hid():
    statuses = ["", "BAD", "WARN", "WARN", "ok", "ok", "ok"]
    panel = Panel(key="a", title="a", lines=tuple(f"l{i}" for i in range(7)),
                  min_lines=1, want_lines=7, line_statuses=tuple(statuses))
    trimmed = trim_panel(panel, 3)
    # Keeps l0, l1; hides l2..l6 -> one WARN kept, one WARN + one BAD... hidden slice is
    # statuses[2:] = WARN, WARN, ok, ok, ok
    assert trimmed.lines[-1] == "+5 more (2 warn)"


def test_trim_panel_says_all_ok_when_nothing_hidden_is_flagged():
    panel = Panel(key="a", title="a", lines=tuple(f"l{i}" for i in range(6)),
                  min_lines=1, want_lines=6, line_statuses=("",) + ("ok",) * 5)
    assert trim_panel(panel, 3).lines[-1] == "+4 more (all ok)"


def test_trim_panel_is_a_noop_when_content_already_fits():
    panel = _panel("a", 3, 1, 3)
    assert trim_panel(panel, 9) is panel


def test_trim_panel_composes_severity_tail_with_both_bad_and_warn():
    statuses = ("", "BAD", "WARN", "BAD", "WARN", "ok", "ok")
    panel = Panel(key="a", title="a", lines=tuple(f"l{i}" for i in range(7)),
                  min_lines=1, want_lines=7, line_statuses=tuple(statuses))
    trimmed = trim_panel(panel, 3)
    # Keep = max(0, 3 - 1) = 2, hidden = 5
    # Hidden statuses (from index 2) = (WARN, BAD, WARN, ok, ok)
    # BAD count = 1, WARN count = 2
    assert trimmed.lines[-1] == "+5 more (1 bad, 2 warn)"


def test_frame_tiers_gives_the_full_spine_on_a_tall_terminal():
    spine, body = frame_tiers(55)
    assert spine == 10
    assert body == 55 - 1 - 10 - 1


def test_frame_tiers_keeps_the_full_spine_down_to_a_28_line_terminal():
    assert frame_tiers(28)[0] == 10
    assert frame_tiers(40)[0] == 10


def test_frame_tiers_switches_to_the_compact_spine_between_20_and_27_lines():
    assert frame_tiers(27)[0] == 5
    assert frame_tiers(20)[0] == 5


def test_frame_tiers_returns_a_spine_only_frame_below_20_lines():
    spine, body = frame_tiers(18)
    assert body == 0
    assert spine == 17


def test_compose_rows_lays_two_panels_side_by_side_within_the_terminal_width():
    row = Row(panels=(
        Panel(key="a", title="left", lines=("l1", "l2"), min_lines=2, want_lines=2, weight=2.4),
        Panel(key="b", title="right", lines=("r1", "r2"), min_lines=2, want_lines=2, weight=0.9),
    ))
    out = compose_rows([(row, 2)], term_w=160)
    assert all(strip_ansi_len(line) <= 160 - 2 for line in out)
    assert len(out) == 2 + PANEL_CHROME_LINES
    joined = "\n".join(out)
    assert "left" in joined and "right" in joined


def test_compose_rows_stacks_panels_vertically_on_a_narrow_terminal():
    row = Row(panels=(
        Panel(key="a", title="left", lines=("l1",), min_lines=1, want_lines=1),
        Panel(key="b", title="right", lines=("r1",), min_lines=1, want_lines=1),
    ))
    out = compose_rows([(row, 1)], term_w=50)
    # Two independently boxed panels, one above the other.
    assert len(out) == 2 * (1 + PANEL_CHROME_LINES)
    assert all(strip_ansi_len(line) <= 50 - 2 for line in out)


def test_compose_rows_trims_panel_content_to_its_allocation():
    row = Row(panels=(Panel(key="a", title="t", lines=tuple(f"line{i}" for i in range(9)),
                            min_lines=1, want_lines=9),))
    out = compose_rows([(row, 3)], term_w=120)
    assert len(out) == 3 + PANEL_CHROME_LINES
    assert "… 7 more" in "\n".join(out)


def test_compose_rows_never_exceeds_a_very_narrow_terminal_two_panel():
    """Verify narrow-terminal stacking doesn't overflow term_w."""
    row = Row(panels=(
        Panel(key="a", title="left", lines=("l1", "l2"), min_lines=2, want_lines=2),
        Panel(key="b", title="right", lines=("r1", "r2"), min_lines=2, want_lines=2),
    ))
    out = compose_rows([(row, 2)], term_w=30)
    # Every line must fit within term_w - 2 = 28 columns
    assert all(strip_ansi_len(line) <= 28 for line in out)


def test_compose_rows_never_exceeds_a_very_narrow_terminal_single_panel():
    """Verify single-panel rows don't overflow narrow terminal."""
    row = Row(panels=(
        Panel(key="a", title="t", lines=("l1", "l2"), min_lines=2, want_lines=2, weight=1.0),
    ))
    out = compose_rows([(row, 2)], term_w=30)
    # Every line must fit within term_w - 2 = 28 columns
    assert all(strip_ansi_len(line) <= 28 for line in out)


def test_allocate_rows_prices_stacked_layouts_correctly():
    """Verify allocator and composer split stacked row budgets across panels.

    This test ensures that when panels stack, each panel gets a portion of the
    body budget, not the full budget. The key is panels with content longer than
    min_lines so truncation reveals whether the split is correct.
    """
    # Two panels: each with min_lines=1, want_lines=3, but 10 real content lines.
    # Side-by-side: max(1, 1) + 4 = 5 lines total
    # Stacked: (1 + 1) + 4 + 4 = 10 lines total
    # With term_w=50 the panels stack.
    row = Row(panels=(
        Panel(key="a", title="left", lines=tuple(f"l{i}" for i in range(10)),
              min_lines=1, want_lines=3),
        Panel(key="b", title="right", lines=tuple(f"r{i}" for i in range(10)),
              min_lines=1, want_lines=3),
    ))
    # Budget of 10 is just enough for stacked at min_lines (1+1+4+4).
    allocated, dropped = allocate_rows([row], budget=10, term_w=50)
    assert len(allocated) == 1
    assert dropped == ()
    row_out, body_lines = allocated[0]
    # Compose and verify line count does not exceed budget.
    # The critical assertion: if the body budget is not split across panels,
    # the second panel would try to trim to 10 lines, causing both panels to
    # render 10+ lines each, and the total would exceed 10.
    out = compose_rows([(row_out, body_lines)], term_w=50)
    assert len(out) <= 10, f"Composed {len(out)} lines, budget was 10"
