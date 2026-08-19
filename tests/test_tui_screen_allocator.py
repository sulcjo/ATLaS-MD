import pytest

from gareus.tui_screen import (
    PANEL_CHROME_LINES,
    Panel,
    Row,
    allocate_rows,
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
