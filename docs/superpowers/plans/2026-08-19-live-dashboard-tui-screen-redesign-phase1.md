# Live-Dashboard TUI Screen Redesign — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the live dashboard's blind-render-then-clip frame with a spine + three views driven by a top-down line allocator, so the frame fits the terminal by construction and hidden content is always the least severe.

**Architecture:** A domain-free screen engine (`gareus/tui_screen.py`) owns `Panel`/`Row` types, row allocation against a line budget, and frame composition on top of today's `gareus/tui.py` primitives. A `gareus/dashboard/` package holds a frozen `DashboardContext` snapshot, mtime-polled sidecar JSON readers, severity ranking, ported content panels, the 10-line spine, and three pure view builders. `DistanceLogger` keeps data collection and loses frame assembly.

**Tech Stack:** Python 3.11+, stdlib + numpy only (no new dependencies), pytest. Frozen dataclasses. ANSI/box-drawing rendering via the existing `gareus.tui` helpers.

**Spec:** `docs/superpowers/specs/2026-08-19-live-dashboard-tui-screen-redesign-design.md`

## Global Constraints

- **Phase 1 uses only data that already exists.** Every panel is sourced from a row the spec's §9 data audit marks *exists*, or from an O(K) derivation of it. New accumulators (per-window cumulative sample counts, per-window σΔV, ns/day history, exchange-stall timing) and interactive key handling are **Phase 2** and must not appear here. Columns that need them are omitted, not faked.
- **No changes to the MD stepping path.** Nothing in `gareus/production.py`'s stepping loop, no new arguments threaded through it. Sidecar data is read from JSON already on disk.
- **The 20 existing layout tests must stay green**: `tests/test_dashboard_tui_scaling.py`, `tests/test_dashboard_scaling_integration.py`. They pin `dashboard_row_gap`, `dashboard_body_budget`, `_weighted_panel_widths`, and `render_distance_ascii`'s separator width — all of which this design keeps using.
- **Rendering stays on the background thread.** `DistanceLogger.log` already offloads frame rendering to a single-worker `ThreadPoolExecutor` and drops a frame if the previous render is still running (`gareus/logger.py:2740-2758`). The context is snapshotted on the caller thread; view builders must not touch live logger state.
- **Status is never colour-only.** Every severity is present as text (`ok`/`WARN`/`BAD`/`DEAD`/`SATURATED`) so an ANSI-stripped frame is fully readable. This is asserted by test.
- **Frozen dataclasses; no mutation of inputs.** Per project coding style. Views are pure functions of the context.
- **Files stay under 800 lines**; target 200-450.
- **Conventional commits** (`feat:`, `refactor:`, `test:`, `fix:`), one per task.
- Constants that already exist are imported, never re-declared: `MIN_PANEL_WIDTH`, `ABSOLUTE_MIN_PANEL_WIDTH`, `dashboard_row_gap`, `_panel_lines`, `_join_columns`, `_weighted_panel_widths`, `strip_ansi`, `strip_ansi_len`, `_ansi_truncate`, `_ansi_pad`, `make_progress_bar`, `format_duration` from `gareus.tui`; `role_text`, `severity_role`, `color_text`, `style_text`, `ROLE_*` from `gareus.colors`; `_hist_overlap` from `gareus.math_helpers`; `K_B_KJ_PER_MOL_K`, `KJ_PER_KCAL` from `gareus.units`.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `gareus/tui_screen.py` | create | Domain-free: `Panel`, `Row`, `allocate_rows`, `trim_panel`, `frame_tiers`, `compose_rows`. Knows nothing about MD. |
| `gareus/dashboard/__init__.py` | create | Package marker; re-exports `render_screen`. |
| `gareus/dashboard/sidecar.py` | create | `SidecarSnapshot`, `SidecarCache` — mtime-polled JSON reads, run-root discovery. |
| `gareus/dashboard/ranking.py` | create | `WindowStatus`, `rank_windows`, `tail_summary`, `Hysteresis`. |
| `gareus/dashboard/context.py` | create | `DashboardContext` frozen snapshot + `build_context`. |
| `gareus/dashboard/panels.py` | create | Panel-producing functions ported from `logger.py`'s `_render_*`. |
| `gareus/dashboard/spine.py` | create | `spine_lines(ctx, lines_budget)` — 10-line and 5-line variants. |
| `gareus/dashboard/view_progress.py` | create | `build(ctx) -> tuple[Row, ...]` — timeline, projection, throughput. |
| `gareus/dashboard/view_physics.py` | create | `build(ctx)` — overlap worst-first, boost envelope, CV2 regime + connectivity. |
| `gareus/dashboard/view_windows.py` | create | `build(ctx)` — worst-first window table + selected-window detail. |
| `gareus/dashboard/screen.py` | create | `render_screen(ctx)` — tier select, spine + allocated view + footer, view resolution/promotion. |
| `gareus/logger.py` | modify | Delete frame assembly and the ported `_render_*` methods; call `render_screen`. |
| `gareus/cli.py` | modify | Add 7 real flags; stop `_shim_output` clobbering them. |
| `gareus/tui.py` | unmodified in Phase 1 | Moving `render_distance_ascii` out is deferred; it is still called by `logger.py`'s non-dashboard path. |
| `tests/test_tui_screen_allocator.py` | create | Tasks 1-2. |
| `tests/test_dashboard_sidecar.py` | create | Task 3. |
| `tests/test_dashboard_ranking.py` | create | Task 4. |
| `tests/test_dashboard_context.py` | create | Task 5. |
| `tests/test_dashboard_panels.py` | create | Task 6. |
| `tests/test_dashboard_spine.py` | create | Task 7. |
| `tests/test_dashboard_views.py` | create | Tasks 8-10. |
| `tests/test_dashboard_screen.py` | create | Tasks 11-12. |
| `tests/test_dashboard_cli_flags.py` | create | Task 13. |
| `tests/test_dashboard_frame_fit.py` | create | Task 14 (property matrix + golden frames). |
| `tests/golden/dashboard/*.txt` | create | Task 14 snapshots. |

---

## Task 1: Panel/Row types and the row allocator

**Files:**
- Create: `gareus/tui_screen.py`
- Test: `tests/test_tui_screen_allocator.py`

**Interfaces:**
- Consumes: `gareus.tui.MIN_PANEL_WIDTH` (int, 30).
- Produces: `Panel` (frozen dataclass: `key: str`, `title: str`, `lines: tuple[str, ...]`, `min_lines: int = 1`, `want_lines: int = 1`, `priority: int = 5`, `weight: float = 1.0`, `tail: str = ""`); `Row` (frozen dataclass: `panels: tuple[Panel, ...]`); `PANEL_CHROME_LINES: int = 4`; `row_priority(row) -> int`; `row_min(row) -> int`; `row_want(row) -> int`; `allocate_rows(rows: Sequence[Row], budget: int) -> tuple[tuple[tuple[Row, int], ...], tuple[str, ...]]`; `trim_panel(panel: Panel, body_lines: int) -> Panel`. Lower `priority` number means more important.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tui_screen_allocator.py
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


def test_trim_panel_prefers_a_caller_supplied_severity_tail():
    panel = Panel(key="a", title="a", lines=tuple("x" * 10), min_lines=1,
                  want_lines=10, tail="+7 more (2 warn, 1 bad)")
    trimmed = trim_panel(panel, 4)
    assert trimmed.lines[-1] == "+7 more (2 warn, 1 bad)"


def test_trim_panel_is_a_noop_when_content_already_fits():
    panel = _panel("a", 3, 1, 3)
    assert trim_panel(panel, 9) is panel
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tui_screen_allocator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.tui_screen'`

- [ ] **Step 3: Write the implementation**

```python
# gareus/tui_screen.py
"""Domain-free TUI screen assembly: panels, row allocation, frame composition.

The live dashboard used to render every panel blind and let the frame writer
clip whatever did not fit, which discarded 33-52% of the frame by *position*
rather than by importance. This module inverts that: the screen hands each row
a line budget, so the frame fits by construction.

Nothing here knows about molecular dynamics. Domain panels live in
``gareus.dashboard``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

# Top border, title, separator, bottom border -- see gareus.tui._panel_lines.
PANEL_CHROME_LINES = 4


@dataclass(frozen=True)
class Panel:
    """One boxed panel's content plus its line-budget declaration.

    ``priority`` is ordinal with 1 as the most important; rows are dropped from
    the least important end when the budget cannot hold every minimum.
    ``tail`` lets a caller supply a severity-aware truncation notice
    ("+9 more (2 warn, 1 bad)") instead of the generic count.
    """

    key: str
    title: str
    lines: tuple[str, ...]
    min_lines: int = 1
    want_lines: int = 1
    priority: int = 5
    weight: float = 1.0
    tail: str = ""


@dataclass(frozen=True)
class Row:
    """Panels rendered side by side. They share one set of border lines."""

    panels: tuple[Panel, ...]


def row_priority(row: Row) -> int:
    """A row is as important as its most important panel."""
    return min((p.priority for p in row.panels), default=9)


def row_min(row: Row) -> int:
    """Total lines a row occupies at its minimum, chrome included."""
    return max((p.min_lines for p in row.panels), default=1) + PANEL_CHROME_LINES


def row_want(row: Row) -> int:
    """Total lines a row would occupy if fully satisfied, chrome included."""
    return max((p.want_lines for p in row.panels), default=1) + PANEL_CHROME_LINES


def allocate_rows(
    rows: Sequence[Row], budget: int
) -> tuple[tuple[tuple[Row, int], ...], tuple[str, ...]]:
    """Fit ``rows`` into ``budget`` lines.

    Returns ``((row, body_lines), ...)`` in declaration order, plus the keys of
    panels that were dropped entirely. Dropped panels are reported so the caller
    can name them in the footer -- silence is what made the old behaviour a bug
    rather than a limitation.
    """
    kept = list(rows)
    dropped: list[str] = []
    while kept and sum(row_min(r) for r in kept) > int(budget):
        victim = max(range(len(kept)), key=lambda i: (row_priority(kept[i]), i))
        dropped.extend(p.key for p in kept.pop(victim).panels)
    if not kept:
        return (), tuple(dropped)

    body = {i: row_min(r) - PANEL_CHROME_LINES for i, r in enumerate(kept)}
    spare = int(budget) - sum(row_min(r) for r in kept)
    for i in sorted(range(len(kept)), key=lambda i: (row_priority(kept[i]), i)):
        if spare <= 0:
            break
        room = row_want(kept[i]) - row_min(kept[i])
        give = min(max(0, room), spare)
        body[i] += give
        spare -= give
    return tuple((r, body[i]) for i, r in enumerate(kept)), tuple(dropped)


def trim_panel(panel: Panel, body_lines: int) -> Panel:
    """Cut a panel's content to ``body_lines``, keeping the first (worst) lines.

    Panel content arrives already ranked worst-first, so a positional cut here
    is a severity cut. The last kept line becomes the truncation notice.
    """
    limit = max(0, int(body_lines))
    if len(panel.lines) <= limit:
        return panel
    keep = max(0, limit - 1)
    hidden = len(panel.lines) - keep
    tail = panel.tail or f"… {hidden} more"
    return replace(panel, lines=tuple(panel.lines[:keep]) + (tail,))


__all__ = [
    "PANEL_CHROME_LINES",
    "Panel",
    "Row",
    "allocate_rows",
    "row_min",
    "row_priority",
    "row_want",
    "trim_panel",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tui_screen_allocator.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/tui_screen.py tests/test_tui_screen_allocator.py
git commit -m "feat(tui): add Panel/Row types and top-down row allocator"
```

---

## Task 2: Frame tiers and row composition

**Files:**
- Modify: `gareus/tui_screen.py`
- Test: `tests/test_tui_screen_allocator.py` (append)

**Interfaces:**
- Consumes: Task 1's `Panel`, `Row`, `trim_panel`; `gareus.tui._panel_lines`, `_join_columns`, `_weighted_panel_widths`, `dashboard_row_gap`, `strip_ansi_len`, `MIN_PANEL_WIDTH`, `ABSOLUTE_MIN_PANEL_WIDTH`.
- Produces: `frame_tiers(term_h: int) -> tuple[int, int]` returning `(spine_lines, body_budget)`; `compose_rows(allocated: Sequence[tuple[Row, int]], term_w: int, gap: int | None = None) -> tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_tui_screen_allocator.py
from gareus.tui import strip_ansi_len
from gareus.tui_screen import compose_rows, frame_tiers


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tui_screen_allocator.py -k "frame_tiers or compose_rows" -v`
Expected: FAIL with `ImportError: cannot import name 'compose_rows'`

- [ ] **Step 3: Write the implementation**

```python
# append to gareus/tui_screen.py (and extend __all__ with "compose_rows", "frame_tiers")
from .tui import (
    ABSOLUTE_MIN_PANEL_WIDTH,
    MIN_PANEL_WIDTH,
    _join_columns,
    _panel_lines,
    _weighted_panel_widths,
    dashboard_row_gap,
)

FULL_SPINE_LINES = 10
COMPACT_SPINE_LINES = 5
FOOTER_LINES = 1
_FULL_SPINE_MIN_USABLE = 27   # a 10-line spine past this point is >36% of the screen
_COMPACT_SPINE_MIN_USABLE = 19


def frame_tiers(term_h: int) -> tuple[int, int]:
    """Split the terminal height into ``(spine_lines, body_budget)``.

    One line is reserved to match ``gareus.tui._safe_tui_frame_text``'s own
    reserve, and one more for the footer. Below 20 rows the view body is
    dropped entirely and only the spine renders.
    """
    usable = max(1, int(term_h) - 1)
    if usable >= _FULL_SPINE_MIN_USABLE:
        return FULL_SPINE_LINES, max(0, usable - FULL_SPINE_LINES - FOOTER_LINES)
    if usable >= _COMPACT_SPINE_MIN_USABLE:
        return COMPACT_SPINE_LINES, max(0, usable - COMPACT_SPINE_LINES - FOOTER_LINES)
    return usable, 0


def compose_rows(
    allocated: Sequence[tuple[Row, int]], term_w: int, gap: int | None = None
) -> tuple[str, ...]:
    """Render allocated rows to text lines, stacking when too narrow for columns."""
    gap_v = dashboard_row_gap(term_w) if gap is None else int(gap)
    usable = max(40, int(term_w) - 2)
    out: list[str] = []
    for row, body in allocated:
        panels = [trim_panel(p, body) for p in row.panels]
        n = len(panels)
        min_w = max(ABSOLUTE_MIN_PANEL_WIDTH, MIN_PANEL_WIDTH)
        if n > 1 and usable < n * min_w + (n - 1) * gap_v:
            for p in panels:
                out.extend(_panel_lines(p.title, list(p.lines), usable))
            continue
        widths = _weighted_panel_widths(
            [p.weight for p in panels], term_w=term_w, gap=gap_v, min_panel_width=min_w
        )
        cols = [_panel_lines(p.title, list(p.lines), w) for p, w in zip(panels, widths)]
        out.extend(_join_columns(cols, gap=gap_v))
    return tuple(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tui_screen_allocator.py -v`
Expected: 16 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/tui_screen.py tests/test_tui_screen_allocator.py
git commit -m "feat(tui): add frame height tiers and row composition"
```

---

## Task 3: Sidecar JSON cache

**Files:**
- Create: `gareus/dashboard/__init__.py`, `gareus/dashboard/sidecar.py`
- Test: `tests/test_dashboard_sidecar.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SidecarSnapshot` (frozen dataclass: `pool: dict | None`, `pool_mtime: float | None`, `gamd: dict | None`, `gamd_mtime: float | None`, `quality_gate: dict | None`, `seeding_quality: dict | None`, `errors: tuple[str, ...]`); `SidecarCache(out_dir: Path, min_interval_s: float = 5.0)` with `snapshot(now: float) -> SidecarSnapshot`; `find_run_root(out_dir: Path) -> Path`; `SIDECAR_MIN_INTERVAL_S: float = 5.0`.

Run-root discovery matters: `DistanceLogger.out_dir` is a *segment* directory such as
`RUNS/chignolin_5/adaptive_production/epoch_001/baseline`, while the pool ledger lives at
`RUNS/chignolin_5/adaptive_production/adaptive_runtime_pool.json`. Note
`global_shared_gamd_setup/` exists at *both* the run root and inside `adaptive_production/`,
so detection keys on `adaptive_production/`, which is unique to the run root.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_sidecar.py
import json

from gareus.dashboard.sidecar import SidecarCache, find_run_root


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_run(tmp_path):
    root = tmp_path / "chignolin_x"
    segment = root / "adaptive_production" / "epoch_001" / "baseline"
    segment.mkdir(parents=True)
    _write(root / "adaptive_production" / "adaptive_runtime_pool.json",
           {"total_ns": 15000.0, "used_ns": 7500.0, "remaining_ns": 7500.0, "events": []})
    _write(root / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json",
           {"joint_envelope": {"Dihedral": {"sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 10.77, "k0": 1.0}}})
    return root, segment


def test_find_run_root_walks_up_from_a_segment_directory(tmp_path):
    root, segment = _make_run(tmp_path)
    assert find_run_root(segment) == root


def test_find_run_root_falls_back_to_the_given_directory(tmp_path):
    lonely = tmp_path / "not_a_run"
    lonely.mkdir()
    assert find_run_root(lonely) == lonely


def test_snapshot_reads_pool_and_gamd_payloads(tmp_path):
    _root, segment = _make_run(tmp_path)
    snap = SidecarCache(segment).snapshot(now=1000.0)
    assert snap.pool["total_ns"] == 15000.0
    assert snap.gamd["joint_envelope"]["Dihedral"]["k0"] == 1.0
    assert snap.errors == ()


def test_snapshot_is_cached_until_the_min_interval_elapses(tmp_path):
    root, segment = _make_run(tmp_path)
    cache = SidecarCache(segment, min_interval_s=5.0)
    first = cache.snapshot(now=1000.0)
    _write(root / "adaptive_production" / "adaptive_runtime_pool.json",
           {"total_ns": 15000.0, "used_ns": 9999.0, "remaining_ns": 5001.0, "events": []})
    assert cache.snapshot(now=1002.0) is first          # inside the interval
    assert cache.snapshot(now=1006.0).pool["used_ns"] == 9999.0


def test_snapshot_reports_missing_files_as_none_not_an_error(tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    snap = SidecarCache(bare).snapshot(now=1.0)
    assert snap.pool is None and snap.gamd is None
    assert snap.errors == ()


def test_snapshot_records_a_corrupt_file_as_an_error_and_keeps_going(tmp_path):
    root, segment = _make_run(tmp_path)
    (root / "adaptive_production" / "adaptive_runtime_pool.json").write_text("{not json")
    snap = SidecarCache(segment).snapshot(now=1.0)
    assert snap.pool is None
    assert any("adaptive_runtime_pool.json" in e for e in snap.errors)
    assert snap.gamd is not None            # one bad file does not poison the rest


def test_snapshot_finds_per_epoch_seeding_quality(tmp_path):
    root, segment = _make_run(tmp_path)
    _write(segment.parent / "setup" / "us_starting_structure_quality.json",
           {"windows": [{"window": 17, "status": "bad", "direction": "pull_crash_fallback_unpulled"}]})
    snap = SidecarCache(segment).snapshot(now=1.0)
    assert snap.seeding_quality["windows"][0]["window"] == 17
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_sidecar.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.dashboard'`

- [ ] **Step 3: Write the implementation**

```python
# gareus/dashboard/__init__.py
"""Live-dashboard screen: context, panels, views, frame assembly."""

__all__: list[str] = []
```

```python
# gareus/dashboard/sidecar.py
"""Slow-changing campaign state, read from JSON already written to disk.

The MD-facing loop must not carry pool budgets or GaMD calibration values
around: they change at phase boundaries (minutes to hours) while CV samples
change every report interval. An mtime-gated read every few seconds is enough,
and keeps the stepping path untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SIDECAR_MIN_INTERVAL_S = 5.0
_MAX_PARENTS_SEARCHED = 5

_POOL_REL = Path("adaptive_production") / "adaptive_runtime_pool.json"
_GAMD_REL = Path("global_shared_gamd_setup") / "shared_gamd_setup_globals.json"
_GATE_REL = Path("adaptive_production") / "adaptive_quality_gate.json"
_SEEDING_NAME = "us_starting_structure_quality.json"


@dataclass(frozen=True)
class SidecarSnapshot:
    pool: Optional[dict] = None
    pool_mtime: Optional[float] = None
    gamd: Optional[dict] = None
    gamd_mtime: Optional[float] = None
    quality_gate: Optional[dict] = None
    seeding_quality: Optional[dict] = None
    errors: tuple[str, ...] = field(default=())


def find_run_root(out_dir: Path) -> Path:
    """Walk up from a segment directory to the run root.

    Keyed on ``adaptive_production/``, which only exists at the run root --
    ``global_shared_gamd_setup/`` also appears one level down and would give a
    false positive.
    """
    here = Path(out_dir)
    for candidate in [here, *list(here.parents)[:_MAX_PARENTS_SEARCHED]]:
        if (candidate / "adaptive_production").is_dir():
            return candidate
    for candidate in [here, *list(here.parents)[:_MAX_PARENTS_SEARCHED]]:
        if (candidate / "global_shared_gamd_setup").is_dir():
            return candidate
    return here


def _read_json(path: Path, errors: list[str]) -> tuple[Optional[dict], Optional[float]]:
    try:
        if not path.is_file():
            return None, None
        mtime = path.stat().st_mtime
        payload = json.loads(path.read_text())
        return (payload if isinstance(payload, dict) else None), mtime
    except Exception as exc:                      # unreadable is a display state
        errors.append(f"{path.name}: {type(exc).__name__}")
        return None, None


class SidecarCache:
    """Re-reads the sidecar files at most every ``min_interval_s`` seconds."""

    def __init__(self, out_dir: Path, min_interval_s: float = SIDECAR_MIN_INTERVAL_S) -> None:
        self.out_dir = Path(out_dir)
        self.run_root = find_run_root(self.out_dir)
        self.min_interval_s = float(min_interval_s)
        self._last_read_wall = float("-inf")
        self._snapshot = SidecarSnapshot()

    def _seeding_candidates(self) -> list[Path]:
        bases = [self.out_dir, self.out_dir.parent]
        return [base / sub / _SEEDING_NAME
                for base in bases for sub in ("setup", "us_starting_structures")]

    def snapshot(self, now: float) -> SidecarSnapshot:
        if float(now) - self._last_read_wall < self.min_interval_s:
            return self._snapshot
        errors: list[str] = []
        pool, pool_mtime = _read_json(self.run_root / _POOL_REL, errors)
        gamd, gamd_mtime = _read_json(self.run_root / _GAMD_REL, errors)
        gate, _gate_mtime = _read_json(self.run_root / _GATE_REL, errors)
        seeding = None
        for candidate in self._seeding_candidates():
            seeding, _mtime = _read_json(candidate, errors)
            if seeding is not None:
                break
        self._snapshot = SidecarSnapshot(
            pool=pool, pool_mtime=pool_mtime, gamd=gamd, gamd_mtime=gamd_mtime,
            quality_gate=gate, seeding_quality=seeding, errors=tuple(errors),
        )
        self._last_read_wall = float(now)
        return self._snapshot


__all__ = ["SIDECAR_MIN_INTERVAL_S", "SidecarCache", "SidecarSnapshot", "find_run_root"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_sidecar.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/dashboard/__init__.py gareus/dashboard/sidecar.py tests/test_dashboard_sidecar.py
git commit -m "feat(dashboard): add mtime-gated sidecar JSON cache"
```

---

## Task 4: Severity ranking, tail summaries and hysteresis

**Files:**
- Create: `gareus/dashboard/ranking.py`
- Test: `tests/test_dashboard_ranking.py`

**Interfaces:**
- Consumes: `gareus.units.K_B_KJ_PER_MOL_K`, `gareus.units.KJ_PER_KCAL`.
- Produces: `OK`/`WARN`/`BAD` string constants; `WindowStatus` (frozen dataclass: `window: int`, `severity: int`, `status: str`, `reasons: tuple[str, ...]`); `restraint_sigma(k_kcal_per_a2: float, temperature_k: float) -> float`; `rank_windows(*, n_windows, centers_a, k_list, acceptance_by_window, overlap_by_pair, delta_by_window, temperature_k) -> tuple[WindowStatus, ...]`; `tail_summary(hidden: Sequence[WindowStatus]) -> str`; `Hysteresis(frames: int = 5)` with `update(bad_keys: Iterable[str]) -> frozenset[str]`.

Severity is the first matching rule; higher `severity` number is worse, so sorting is
`(-severity, window)`. Rules, from the spec §8: dead exchange (<0.02) → overlap < 0.10 →
|Δ| > 2σ → samples below half the median (Phase 2, not evaluated here) → per-window
anharmonicity (Phase 2) → acceptance < 0.15.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_ranking.py
import math

from gareus.dashboard.ranking import (
    BAD,
    OK,
    WARN,
    Hysteresis,
    rank_windows,
    restraint_sigma,
    tail_summary,
)


def _rank(**over):
    kwargs = dict(
        n_windows=4,
        centers_a=(4.0, 4.55, 5.10, 5.65),
        k_list=(2.5, 2.5, 2.5, 2.5),
        acceptance_by_window={0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30},
        overlap_by_pair={(0, 1): 0.45, (1, 2): 0.45, (2, 3): 0.45},
        delta_by_window={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        temperature_k=300.0,
    )
    kwargs.update(over)
    return rank_windows(**kwargs)


def test_restraint_sigma_matches_sqrt_kt_over_k():
    # k in kcal/mol/A^2, kT at 300 K; sigma in A.
    sigma = restraint_sigma(2.5, 300.0)
    assert math.isclose(sigma, 0.4877, rel_tol=2e-3)


def test_restraint_sigma_is_infinite_for_a_zero_force_constant():
    assert math.isinf(restraint_sigma(0.0, 300.0))


def test_all_healthy_windows_rank_ok():
    statuses = _rank()
    assert {s.status for s in statuses} == {OK}
    assert [s.window for s in statuses] == [0, 1, 2, 3]


def test_dead_exchange_outranks_every_other_rule():
    statuses = _rank(acceptance_by_window={0: 0.30, 1: 0.30, 2: 0.30, 3: 0.001})
    assert statuses[0].window == 3
    assert statuses[0].status == BAD
    assert any("dead" in r for r in statuses[0].reasons)


def test_low_overlap_marks_both_windows_of_the_pair_bad():
    statuses = _rank(overlap_by_pair={(0, 1): 0.45, (1, 2): 0.02, (2, 3): 0.45})
    flagged = {s.window for s in statuses if s.status == BAD}
    assert flagged == {1, 2}


def test_delta_beyond_two_sigma_is_bad_and_named_pinned():
    statuses = _rank(delta_by_window={0: 0.0, 1: 0.0, 2: 2.31, 3: 0.0})
    worst = statuses[0]
    assert worst.window == 2 and worst.status == BAD
    assert any("pinned" in r for r in worst.reasons)


def test_low_but_not_dead_acceptance_is_a_warning():
    statuses = _rank(acceptance_by_window={0: 0.30, 1: 0.11, 2: 0.30, 3: 0.30})
    assert [s.status for s in statuses][0] == WARN
    assert statuses[0].window == 1


def test_ties_break_by_window_index_so_order_is_stable():
    statuses = _rank(delta_by_window={0: 3.0, 1: 0.0, 2: 3.0, 3: 0.0})
    assert [s.window for s in statuses[:2]] == [0, 2]


def test_tail_summary_reports_composition_of_hidden_entries():
    hidden = _rank(acceptance_by_window={0: 0.001, 1: 0.11, 2: 0.30, 3: 0.30})
    assert tail_summary(hidden) == "+4 more (1 bad, 1 warn)"
    assert tail_summary([s for s in hidden if s.status == OK]) == "+2 more (all ok)"
    assert tail_summary([]) == ""


def test_hysteresis_keeps_a_recovered_key_for_the_configured_frames():
    h = Hysteresis(frames=3)
    assert h.update(["w17"]) == frozenset({"w17"})
    assert h.update([]) == frozenset({"w17"})     # frame 1 clear
    assert h.update([]) == frozenset({"w17"})     # frame 2 clear
    assert h.update([]) == frozenset()            # frame 3 clear -> released


def test_hysteresis_resets_the_countdown_when_a_key_re_triggers():
    h = Hysteresis(frames=2)
    h.update(["w17"])
    h.update([])
    assert h.update(["w17"]) == frozenset({"w17"})
    assert h.update([]) == frozenset({"w17"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_ranking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.dashboard.ranking'`

- [ ] **Step 3: Write the implementation**

```python
# gareus/dashboard/ranking.py
"""Severity ranking for window-facing panels.

Positional truncation is only acceptable if position encodes severity, so every
list panel sorts through here first. Hysteresis keeps a flagged window in its
slot for a few frames after it recovers: a table that re-sorts on noise moves
the row out from under whoever is reading it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K

OK = "ok"
WARN = "WARN"
BAD = "BAD"

DEAD_ACCEPTANCE = 0.02
LOW_ACCEPTANCE = 0.15
DEAD_OVERLAP = 0.10
PINNED_SIGMA_MULTIPLE = 2.0

_SEV_BAD_DEAD = 40
_SEV_BAD_OVERLAP = 30
_SEV_BAD_PINNED = 20
_SEV_WARN_ACCEPT = 10
_SEV_OK = 0


@dataclass(frozen=True)
class WindowStatus:
    window: int
    severity: int
    status: str
    reasons: tuple[str, ...] = ()


def restraint_sigma(k_kcal_per_a2: float, temperature_k: float) -> float:
    """Gaussian width of a harmonic umbrella, in the CV's own units.

    ``sigma = sqrt(k_B T / k)`` with ``k`` converted from kcal/mol/A^2 to
    kJ/mol/A^2 so it divides a kJ/mol thermal energy.
    """
    k = float(k_kcal_per_a2) * KJ_PER_KCAL
    if not math.isfinite(k) or k <= 0.0:
        return math.inf
    return math.sqrt(K_B_KJ_PER_MOL_K * float(temperature_k) / k)


def rank_windows(
    *,
    n_windows: int,
    centers_a: Sequence[float],
    k_list: Sequence[float],
    acceptance_by_window: Mapping[int, float],
    overlap_by_pair: Mapping[tuple[int, int], float],
    delta_by_window: Mapping[int, float],
    temperature_k: float,
) -> tuple[WindowStatus, ...]:
    """Rank windows worst-first. Ties break by window index for stability."""
    low_overlap: dict[int, float] = {}
    for (a, b), ov in overlap_by_pair.items():
        if math.isfinite(ov) and ov < DEAD_OVERLAP:
            for w in (a, b):
                low_overlap[w] = min(low_overlap.get(w, math.inf), float(ov))

    out: list[WindowStatus] = []
    for w in range(int(n_windows)):
        acc = float(acceptance_by_window.get(w, float("nan")))
        delta = abs(float(delta_by_window.get(w, 0.0)))
        k = float(k_list[w]) if w < len(k_list) else float("nan")
        sigma = restraint_sigma(k, temperature_k)

        if math.isfinite(acc) and acc < DEAD_ACCEPTANCE:
            out.append(WindowStatus(w, _SEV_BAD_DEAD, BAD, (f"dead exchange {acc:.3f}",)))
        elif w in low_overlap:
            out.append(WindowStatus(w, _SEV_BAD_OVERLAP, BAD,
                                    (f"overlap {low_overlap[w]:.2f}",)))
        elif math.isfinite(sigma) and delta > PINNED_SIGMA_MULTIPLE * sigma:
            out.append(WindowStatus(w, _SEV_BAD_PINNED, BAD,
                                    (f"pinned |d|={delta:.2f} > 2s={2 * sigma:.2f}",)))
        elif math.isfinite(acc) and acc < LOW_ACCEPTANCE:
            out.append(WindowStatus(w, _SEV_WARN_ACCEPT, WARN, (f"low accept {acc:.2f}",)))
        else:
            out.append(WindowStatus(w, _SEV_OK, OK, ()))
    return tuple(sorted(out, key=lambda s: (-s.severity, s.window)))


def tail_summary(hidden: Sequence[WindowStatus]) -> str:
    """Describe what a truncated list is not showing, by severity."""
    if not hidden:
        return ""
    bad = sum(1 for s in hidden if s.status == BAD)
    warn = sum(1 for s in hidden if s.status == WARN)
    if not bad and not warn:
        return f"+{len(hidden)} more (all ok)"
    parts = [f"{bad} bad"] if bad else []
    if warn:
        parts.append(f"{warn} warn")
    return f"+{len(hidden)} more ({', '.join(parts)})"


class Hysteresis:
    """Hold a key in its ranked slot until it has been clear for `frames` frames."""

    def __init__(self, frames: int = 5) -> None:
        self.frames = max(1, int(frames))
        self._countdown: dict[str, int] = {}

    def update(self, bad_keys: Iterable[str]) -> frozenset[str]:
        live = {str(k) for k in bad_keys}
        for key in live:
            self._countdown[key] = self.frames
        for key in list(self._countdown):
            if key not in live:
                self._countdown[key] -= 1
                if self._countdown[key] <= 0:
                    del self._countdown[key]
        return frozenset(self._countdown)


__all__ = [
    "BAD", "DEAD_ACCEPTANCE", "DEAD_OVERLAP", "Hysteresis", "LOW_ACCEPTANCE", "OK",
    "PINNED_SIGMA_MULTIPLE", "WARN", "WindowStatus", "rank_windows", "restraint_sigma",
    "tail_summary",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_ranking.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/dashboard/ranking.py tests/test_dashboard_ranking.py
git commit -m "feat(dashboard): add severity ranking, tail summaries and hysteresis"
```

---

## Task 5: DashboardContext frozen snapshot

**Files:**
- Create: `gareus/dashboard/context.py`
- Test: `tests/test_dashboard_context.py`

**Interfaces:**
- Consumes: Task 3's `SidecarSnapshot`; `gareus.math_helpers._hist_overlap`.
- Produces: `DashboardContext` (frozen dataclass, fields below); `build_context(*, logger, rows, phase, step, total_steps, summary, dashboard_info, sidecar, term_w, term_h, now, view, glyphs) -> DashboardContext`; `acceptance_by_pair(exchange_stats) -> dict[tuple[int, int], float]`; `acceptance_by_window(pairs, n_windows) -> dict[int, float]`; `overlap_by_pair(history_by_window, centers_a) -> dict[tuple[int, int], float]`; `delta_by_window(rows) -> dict[int, float]`.

Every field is a `tuple`/`Mapping` copy, never a live logger deque — views run on the
render thread and must not observe mid-flight mutation. `decision` carries the existing
verdict dict from `DistanceLogger._dashboard_decision_state` unchanged; re-deriving the
verdict is out of scope for Phase 1.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_context.py
import argparse
import collections
import math

from gareus.dashboard.context import (
    DashboardContext,
    acceptance_by_pair,
    acceptance_by_window,
    build_context,
    delta_by_window,
    overlap_by_pair,
)
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger


def _rows(n=4):
    return [{"replica": i, "window": i, "center_A": 4.0 + 0.55 * i,
             "k_kcal_mol_A2": 2.5, "cv_A": 4.0 + 0.55 * i + 0.1,
             "umbrella_bias_kcal_mol": 0.01, "umbrella_pull_kcal_mol_A": 0.2}
            for i in range(n)]


def test_acceptance_by_pair_converts_attempt_counts_to_rates():
    pairs = acceptance_by_pair({"0-1": {"attempts": 50, "accepted": 10},
                                "1-2": {"attempts": 0, "accepted": 0}})
    assert pairs[(0, 1)] == 0.2
    assert math.isnan(pairs[(1, 2)])            # no attempts yet is unknown, not zero


def test_acceptance_by_window_takes_the_worst_neighbour_of_each_window():
    pairs = {(0, 1): 0.40, (1, 2): 0.05}
    per_window = acceptance_by_window(pairs, n_windows=3)
    assert per_window[0] == 0.40
    assert per_window[1] == 0.05                # worst of its two sides
    assert per_window[2] == 0.05


def test_overlap_by_pair_is_high_for_identical_and_low_for_disjoint_histories():
    history = {0: (1.0, 1.1, 1.2), 1: (1.0, 1.1, 1.2), 2: (9.0, 9.1, 9.2)}
    ov = overlap_by_pair(history, centers_a=(1.0, 1.1, 9.0))
    assert ov[(0, 1)] > 0.9
    assert ov[(1, 2)] < 0.1


def test_delta_by_window_is_the_signed_distance_from_the_restraint_centre():
    d = delta_by_window(_rows(2))
    assert math.isclose(d[0], 0.1, abs_tol=1e-9)


def test_build_context_copies_histories_into_immutable_tuples(tmp_path):
    args = argparse.Namespace(timestep_fs=2.0, temperature_k=310.0)
    logger = DistanceLogger(tmp_path, args, no_file_persistence=True)
    logger.history_by_window[0] = collections.deque([1.0, 2.0])
    ctx = build_context(
        logger=logger, rows=_rows(), phase="gareus_production", step=100,
        total_steps=1000, summary={}, dashboard_info={"centers_a": [4.0, 4.55, 5.1, 5.65],
                                                     "n_windows": 4},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="progress", glyphs="unicode",
    )
    assert isinstance(ctx, DashboardContext)
    assert ctx.cv_history_by_window[0] == (1.0, 2.0)
    logger.history_by_window[0].append(3.0)
    assert ctx.cv_history_by_window[0] == (1.0, 2.0)     # snapshot, not a live view
    assert ctx.temperature_k == 310.0
    assert ctx.n_windows == 4
    assert ctx.is_2d is False
    # render_distance_ascii needs per-replica history and the ascii knobs too.
    assert ctx.cv_history_by_replica == {}
    assert (ctx.ascii_mode, ctx.ascii_max_replicas) == ("hist3d", 32)


def test_build_context_marks_a_2d_run_and_records_its_secondary_cv_type(tmp_path):
    args = argparse.Namespace(timestep_fs=2.0)
    logger = DistanceLogger(tmp_path, args, no_file_persistence=True)
    ctx = build_context(
        logger=logger, rows=_rows(), phase="gareus_production", step=1, total_steps=10,
        summary={}, dashboard_info={
            "centers_a": [4.0, 4.55, 5.1, 5.65], "n_windows": 4,
            "secondary_cv_centers": [-1.0, 1.0, -1.0, 1.0],
            "secondary_cv": {"explicit_2d_windows": True, "grid": False, "type": "tica-linear"},
        },
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=1.0,
        view="physics", glyphs="unicode",
    )
    assert ctx.is_2d is True
    assert ctx.secondary_cv_type == "tica-linear"
    assert ctx.topology_label == "sparse explicit 2D"


def test_build_context_prefers_the_sidecar_temperature_when_args_lack_one(tmp_path):
    logger = DistanceLogger(tmp_path, argparse.Namespace(), no_file_persistence=True)
    ctx = build_context(
        logger=logger, rows=_rows(), phase="p", step=1, total_steps=2, summary={},
        dashboard_info={"centers_a": [4.0], "n_windows": 1},
        sidecar=SidecarSnapshot(gamd={"temperature_K": 277.0}),
        term_w=100, term_h=30, now=1.0, view="progress", glyphs="ascii",
    )
    assert ctx.temperature_k == 277.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.dashboard.context'`

- [ ] **Step 3: Write the implementation**

```python
# gareus/dashboard/context.py
"""Immutable snapshot every view is a pure function of.

Views run on the render thread. Handing them the live ``DistanceLogger`` would
let them observe deques mutating mid-frame, and would make every view test
require a live logger. Copying into tuples here costs one pass over histories
that are already capped by ``distance_history_limit``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..math_helpers import _hist_overlap
from .sidecar import SidecarSnapshot

DEFAULT_TEMPERATURE_K = 300.0


def acceptance_by_pair(exchange_stats: Mapping[str, Any]) -> dict[tuple[int, int], float]:
    """``{"0-1": {"attempts": n, "accepted": m}}`` -> ``{(0, 1): m / n}``.

    Zero attempts yields NaN, not 0.0: "not tried yet" and "tried and always
    rejected" must not rank the same.
    """
    out: dict[tuple[int, int], float] = {}
    for key, stats in (exchange_stats or {}).items():
        try:
            a_str, b_str = str(key).split("-")[:2]
            a, b = int(a_str), int(b_str)
            attempts = float((stats or {}).get("attempts", 0) or 0)
            accepted = float((stats or {}).get("accepted", 0) or 0)
        except Exception:
            continue
        out[(a, b)] = (accepted / attempts) if attempts > 0 else float("nan")
    return out


def acceptance_by_window(
    pairs: Mapping[tuple[int, int], float], n_windows: int
) -> dict[int, float]:
    """Each window inherits the worst finite acceptance among its own pairs."""
    out: dict[int, float] = {}
    for (a, b), rate in pairs.items():
        if not math.isfinite(rate):
            continue
        for w in (a, b):
            if 0 <= w < int(n_windows):
                out[w] = min(out.get(w, math.inf), float(rate))
    return out


def overlap_by_pair(
    history_by_window: Mapping[int, Sequence[float]], centers_a: Sequence[float]
) -> dict[tuple[int, int], float]:
    """Neighbour histogram overlap on one shared axis spanning all centres."""
    centers = [float(c) for c in centers_a]
    if len(centers) < 2:
        return {}
    lo, hi = min(centers), max(centers)
    if hi <= lo:
        hi = lo + 1.0
    out: dict[tuple[int, int], float] = {}
    for a in range(len(centers) - 1):
        ov = _hist_overlap(
            list(history_by_window.get(a, ())), list(history_by_window.get(a + 1, ())), lo, hi
        )
        if math.isfinite(ov):
            out[(a, a + 1)] = float(ov)
    return out


def delta_by_window(rows: Sequence[Mapping[str, Any]]) -> dict[int, float]:
    """Signed CV offset from each window's restraint centre, latest sample."""
    out: dict[int, float] = {}
    for row in rows:
        try:
            out[int(row["window"])] = float(row["cv_A"]) - float(row["center_A"])
        except Exception:
            continue
    return out


@dataclass(frozen=True)
class DashboardContext:
    # identity / phase
    run_label: str
    phase: str
    segment_name: str
    epoch_index: Optional[int]
    epoch_total: Any
    # progress
    step: int
    total_steps: Optional[int]
    display_step: int
    display_total_steps: Optional[int]
    elapsed_s: float
    eta_s: Optional[float]
    timestep_fs: float
    n_replicas: int
    now_wall: float
    # geometry
    term_w: int
    term_h: int
    # window layout
    n_windows: int
    centers_a: tuple[float, ...]
    k_list: tuple[float, ...]
    secondary_centers: tuple[float, ...]
    secondary_k: tuple[float, ...]
    is_2d: bool
    topology_label: str
    secondary_cv_type: str
    temperature_k: float
    # live data (all copies)
    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    cv_history_by_window: Mapping[int, tuple[float, ...]]
    cv_history_by_replica: Mapping[int, tuple[float, ...]]
    secondary_history_by_window: Mapping[int, tuple[float, ...]]
    pe_history_by_replica: Mapping[int, tuple[float, ...]]
    boost_history_all: tuple[float, ...]
    window_trace_by_replica: Mapping[int, tuple[int, ...]]
    exchange_stats: Mapping[str, Any]
    acceptance_pairs: Mapping[tuple[int, int], float]
    acceptance_windows: Mapping[int, float]
    overlap_pairs: Mapping[tuple[int, int], float]
    deltas: Mapping[int, float]
    decision: Mapping[str, Any]
    # labels + config
    primary_cv_label: str
    primary_cv_units: str
    primary_k_units: str
    ascii_mode: str = "hist3d"
    ascii_max_replicas: int = 32
    sidecar: SidecarSnapshot = field(default_factory=SidecarSnapshot)
    view: str = "progress"
    glyphs: str = "unicode"


def _tuple_map(source: Mapping[int, Any]) -> dict[int, tuple[float, ...]]:
    return {int(k): tuple(float(x) for x in v) for k, v in (source or {}).items()}


def _resolve_temperature(args: Any, sidecar: SidecarSnapshot) -> float:
    for value in (getattr(args, "temperature_k", None),
                  (sidecar.gamd or {}).get("temperature_K")):
        try:
            t = float(value)
            if math.isfinite(t) and t > 0.0:
                return t
        except (TypeError, ValueError):
            continue
    return DEFAULT_TEMPERATURE_K


def build_context(
    *,
    logger: Any,
    rows: Sequence[Mapping[str, Any]],
    phase: str,
    step: int,
    total_steps: Optional[int],
    summary: Mapping[str, Any],
    dashboard_info: Optional[Mapping[str, Any]],
    sidecar: SidecarSnapshot,
    term_w: int,
    term_h: int,
    now: float,
    view: str,
    glyphs: str,
) -> DashboardContext:
    info = dict(dashboard_info or {})
    centers = tuple(float(x) for x in info.get("centers_a", ()))
    n_windows = int(info.get("n_windows", len(centers)) or len(centers))
    sec_centers = tuple(
        float(x) for x in info.get("secondary_cv_centers", ())
        if str(x) not in {"", "None", "nan"}
    )
    sec_meta = dict(info.get("secondary_cv", {}) or {})
    explicit_2d = bool(sec_meta.get("explicit_2d_windows", info.get("explicit_2d", False)))
    n_sec_targets = len(set(round(x, 4) for x in sec_centers))
    is_2d = bool(explicit_2d or n_sec_targets > 1)
    if explicit_2d and not bool(sec_meta.get("grid", False)):
        topology = "sparse explicit 2D"
    elif is_2d:
        topology = "rectangular 2D"
    else:
        topology = "1D/secondary-fixed"

    adaptive = dict(info.get("adaptive_phase", {}) or {})
    display_step = int(info.get("display_step", step) or 0)
    display_total = info.get("display_total_steps", total_steps)
    display_total = int(display_total) if display_total is not None else None
    frac = (float(display_step) / float(display_total)) if display_total else 0.0
    frac = max(0.0, min(1.0, frac))
    eta_start = float(info.get("eta_start_wall", getattr(logger, "start_wall", now)) or now)
    elapsed = max(0.0, float(now) - eta_start)
    eta = (elapsed * (1.0 - frac) / frac) if frac > 0.0 and display_total else None

    hist_windows = _tuple_map(getattr(logger, "history_by_window", {}))
    pairs = acceptance_by_pair(info.get("exchange_stats", {}) or {})
    try:
        decision = logger._dashboard_decision_state(
            list(rows), info.get("exchange_stats", {}) or {}, list(centers), dict(summary),
            str(phase), info,
        )
    except Exception:
        decision = {}

    return DashboardContext(
        run_label=str(getattr(logger.out_dir, "name", "") or ""),
        phase=str(phase),
        segment_name=str(adaptive.get("segment_name", "") or ""),
        epoch_index=adaptive.get("epoch_index"),
        epoch_total=adaptive.get("epoch_total"),
        step=int(step),
        total_steps=int(total_steps) if total_steps is not None else None,
        display_step=display_step,
        display_total_steps=display_total,
        elapsed_s=elapsed,
        eta_s=eta,
        timestep_fs=float(getattr(logger.args, "timestep_fs", 0.0) or 0.0),
        n_replicas=max(1, len(rows)),
        now_wall=float(now),
        term_w=int(term_w),
        term_h=int(term_h),
        n_windows=n_windows,
        centers_a=centers,
        k_list=tuple(float(x) for x in info.get("k_list", ())) or tuple(
            float(r.get("k_kcal_mol_A2", float("nan"))) for r in rows
        ),
        secondary_centers=sec_centers,
        secondary_k=tuple(float(x) for x in info.get("secondary_cv_k_kcal_mol", ())),
        is_2d=is_2d,
        topology_label=topology,
        secondary_cv_type=str(sec_meta.get("type", "") or ""),
        temperature_k=_resolve_temperature(logger.args, sidecar),
        rows=tuple(dict(r) for r in rows),
        summary=dict(summary),
        cv_history_by_window=hist_windows,
        cv_history_by_replica=_tuple_map(getattr(logger, "history_by_replica", {})),
        secondary_history_by_window=_tuple_map(
            getattr(logger, "secondary_history_by_window", {})),
        pe_history_by_replica=_tuple_map(getattr(logger, "potential_history_by_replica", {})),
        boost_history_all=tuple(float(x) for x in getattr(logger, "boost_history_all", ())),
        window_trace_by_replica={
            int(k): tuple(int(x) for x in v)
            for k, v in (getattr(logger, "window_trace_by_replica", {}) or {}).items()
        },
        exchange_stats=dict(info.get("exchange_stats", {}) or {}),
        acceptance_pairs=pairs,
        acceptance_windows=acceptance_by_window(pairs, n_windows),
        overlap_pairs=overlap_by_pair(hist_windows, centers),
        deltas=delta_by_window(rows),
        decision=dict(decision or {}),
        primary_cv_label=str(info.get("primary_cv_label", "primary CV")),
        primary_cv_units=str(info.get("primary_cv_units", "")),
        primary_k_units=str(info.get("primary_k_units", "")),
        ascii_mode=str(getattr(logger, "ascii_mode", "hist3d") or "hist3d"),
        ascii_max_replicas=int(getattr(logger, "ascii_max_replicas", 32) or 32),
        sidecar=sidecar,
        view=str(view),
        glyphs=str(glyphs),
    )


__all__ = [
    "DEFAULT_TEMPERATURE_K", "DashboardContext", "acceptance_by_pair",
    "acceptance_by_window", "build_context", "delta_by_window", "overlap_by_pair",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_context.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/dashboard/context.py tests/test_dashboard_context.py
git commit -m "feat(dashboard): add frozen DashboardContext snapshot"
```

---

## Task 6: Content panels ported off DistanceLogger

**Files:**
- Create: `gareus/dashboard/panels.py`
- Test: `tests/test_dashboard_panels.py`
- Read (source of the ports): `gareus/logger.py:634-1137`

**Interfaces:**
- Consumes: Task 1's `Panel`; Task 4's `rank_windows`, `tail_summary`, `restraint_sigma`, `OK`/`WARN`/`BAD`; Task 5's `DashboardContext`.
- Produces: `panel(key, title, lines, *, min_lines, want_lines, priority, weight=1.0, tail="") -> Panel`; `overlap_panel(ctx) -> Panel`; `boost_envelope_panel(ctx) -> Panel`; `cv_map_panel(ctx, bar_width) -> Panel`; `pe_map_panel(ctx, bar_width) -> Panel`; `exchange_panel(ctx) -> Panel`; `pull_panel(ctx) -> Panel`; `replica_table_panel(ctx, ncols) -> Panel`; `window_table_panel(ctx, statuses) -> Panel`; `window_detail_panel(ctx, window) -> Panel`.

**Port rule.** Each source method returns `list[str]` whose first element is its own title
line, which `_render_dashboard` strips with `[1:]`. Ported functions drop the title line
(the title lives on the `Panel`), take `ctx` instead of `self`, and return a `Panel`.
Attribute substitutions, exhaustively:

| In `logger.py` | In `panels.py` |
|---|---|
| `self.history_by_window` | `ctx.cv_history_by_window` |
| `self.secondary_history_by_window` | `ctx.secondary_history_by_window` |
| `self.potential_history_by_replica` | `ctx.pe_history_by_replica` |
| `self.boost_history_all` | `ctx.boost_history_all` |
| `self.window_trace_by_replica` | `ctx.window_trace_by_replica` |
| `self.args` | `ctx` fields (`temperature_k`, `glyphs`, `primary_*`) |
| `self.ascii_mode` / `self.ascii_max_replicas` | passed as function arguments |
| `centers_a` argument | `ctx.centers_a` |
| `exchange_stats` argument | `ctx.exchange_stats` / `ctx.acceptance_pairs` |

**Panel budgets** (`min_lines`, `want_lines`, `priority`, `weight`), taken from the spec's
per-view panel lists:

| Function | `Panel.key` | Source | min | want | priority | weight |
|---|---|---|---|---|---|---|
| `overlap_panel` | `overlap` | `_render_overlap` (`logger.py:751`) | 5 | 12 | 1 | 1.3 |
| `boost_envelope_panel` | `boost` | `_render_gamd_boost` (`:858`) + sidecar `joint_envelope` | 5 | 11 | 1 | 1.0 |
| `cv_map_panel` | `cv_map` | `render_distance_ascii` via `gareus.tui` | 6 | 18 | 2 | 2.4 |
| `pe_map_panel` | `pe_map` | `_render_potential_energy_map` (`:634`) | 4 | 10 | 3 | 0.9 |
| `exchange_panel` | `exchange` | `_render_exchange_acceptance` (`:680`) | 4 | 10 | 2 | 1.15 |
| `pull_panel` | `pull` | `_render_pull_map` (`:771`) | 4 | 10 | 3 | 1.0 |
| `replica_table_panel` | `replicas` | `_render_replica_table` (`:974`) | 4 | 16 | 2 | 1.5 |
| `window_table_panel` | `windows` | new, from `statuses` + `ctx` | 4 | 16 | 1 | 1.0 |
| `window_detail_panel` | `detail-w<NN>` | new, from `ctx` + selected window | 6 | 11 | 1 | 1.0 |

The keys are load-bearing: Task 10's tests and the footer's dropped-panel list both match on
them, and `cv_map`/`pe_map` sit at priority 2/3 per the amended spec §7.3 so a short
terminal drops them before the ranked table or the detail panel.

Panels that stay behind on `DistanceLogger` for Phase 1 because no view in this plan uses
them: `_render_pmf_preview`, `_render_replica_diffusion`, `_render_sparklines`,
`_render_health`, `_render_recommendations`, `_render_2d_diffusion_map`,
`_render_2d_replica_map`, `_render_2d_sparse_topology_map`. They are deleted or ported in
Phase 2; Task 12 must not remove them.

**Import-cycle prevention.** `panels.py` needs `_mini_bar` and `boost_anharmonicity`,
which currently live in `gareus/logger.py`. Task 12 makes `logger.py` import the screen,
so importing back out of `logger.py` here would create
`logger -> screen -> panels -> logger`. Step 0 moves those helpers to neutral homes first.

- [ ] **Step 0: Move the shared helpers out of `logger.py` (own commit)**

Move `_sparkline`, `_mini_bar`, `_coverage_bar` (`gareus/logger.py:123-161`) verbatim into
`gareus/tui.py`, and `boost_anharmonicity`, `anharmonicity_label`
(`gareus/logger.py:162-197`) verbatim into `gareus/math_helpers.py`. In `logger.py`, replace
the definitions with imports and keep both names in its `__all__` so existing importers of
`gareus.logger.boost_anharmonicity` keep working:

```python
# gareus/logger.py — replacing the moved definitions
from .math_helpers import _hist_overlap, anharmonicity_label, boost_anharmonicity
from .tui import _coverage_bar, _mini_bar, _sparkline
```

Add the moved names to `gareus/tui.py`'s and `gareus/math_helpers.py`'s `__all__`.

Run: `python -m pytest -q tests/ -x -k "logger or tui or dashboard or store"`
Expected: PASS — the move is verbatim, so no behaviour changes.

```bash
git add gareus/logger.py gareus/tui.py gareus/math_helpers.py
git commit -m "refactor(tui): move shared glyph and anharmonicity helpers out of logger"
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_panels.py
import argparse

import pytest

from gareus.dashboard.context import build_context
from gareus.dashboard.panels import (
    boost_envelope_panel,
    overlap_panel,
    panel,
    window_detail_panel,
    window_table_panel,
)
from gareus.dashboard.ranking import BAD, rank_windows
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi

CENTERS = (4.0, 4.55, 5.10, 5.65)


def _ctx(tmp_path, *, sidecar=None, histories=True, exchange=True):
    args = argparse.Namespace(timestep_fs=2.0, temperature_k=300.0)
    logger = DistanceLogger(tmp_path, args, no_file_persistence=True)
    if histories:
        for w, c in enumerate(CENTERS):
            logger.history_by_window[w] = [c + 0.1 * (i % 5 - 2) for i in range(60)]
        logger.boost_history_all = [2.0 + 0.5 * (i % 7) for i in range(200)]
    stats = {f"{i}-{i+1}": {"attempts": 40, "accepted": 12} for i in range(3)} if exchange else {}
    rows = [{"replica": w, "window": w, "center_A": c, "k_kcal_mol_A2": 2.5,
             "cv_A": c + 0.05, "umbrella_bias_kcal_mol": 0.01,
             "umbrella_pull_kcal_mol_A": 0.1} for w, c in enumerate(CENTERS)]
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=10, total_steps=100,
        summary={}, dashboard_info={"centers_a": list(CENTERS), "n_windows": 4,
                                    "k_list": [2.5] * 4, "exchange_stats": stats,
                                    "primary_cv_label": "contacts", "primary_cv_units": "A"},
        sidecar=sidecar or SidecarSnapshot(), term_w=140, term_h=45, now=1000.0,
        view="physics", glyphs="unicode",
    )


def test_panel_helper_builds_a_frozen_panel_with_the_given_budget():
    p = panel("k", "title", ["a", "b"], min_lines=1, want_lines=4, priority=2, weight=1.5)
    assert (p.key, p.title, p.lines, p.min_lines, p.want_lines, p.priority, p.weight) == (
        "k", "title", ("a", "b"), 1, 4, 2, 1.5)


def test_overlap_panel_lists_pairs_worst_first_with_a_severity_tail(tmp_path):
    ctx = _ctx(tmp_path)
    p = overlap_panel(ctx)
    text = strip_ansi("\n".join(p.lines))
    assert "w00-w01" in text
    values = [float(line.split()[1]) for line in strip_ansi("\n".join(p.lines)).splitlines()
              if line.strip().startswith("w")]
    assert values == sorted(values)                 # ascending overlap == worst first
    assert p.priority == 1 and p.min_lines == 5


def test_overlap_panel_states_that_it_has_no_samples_yet(tmp_path):
    p = overlap_panel(_ctx(tmp_path, histories=False))
    assert "insufficient samples" in strip_ansi("\n".join(p.lines))


def test_boost_envelope_panel_reports_sigma_target_and_k0_saturation(tmp_path):
    sidecar = SidecarSnapshot(gamd={"joint_envelope": {"Dihedral": {
        "sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 11.039, "k0": 1.0}}})
    text = strip_ansi("\n".join(boost_envelope_panel(_ctx(tmp_path, sidecar=sidecar)).lines))
    assert "11.04" in text and "12.55" in text
    assert "SATURATED" in text                      # k0 at its ceiling, stated as text


def test_boost_envelope_panel_says_so_when_the_run_has_no_gamd(tmp_path):
    text = strip_ansi("\n".join(boost_envelope_panel(_ctx(tmp_path)).lines))
    assert "no GaMD" in text


def test_window_table_panel_puts_the_worst_window_first(tmp_path):
    ctx = _ctx(tmp_path)
    statuses = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window={0: 0.30, 1: 0.30, 2: 0.001, 3: 0.30},
        overlap_by_pair=ctx.overlap_pairs, delta_by_window=ctx.deltas,
        temperature_k=ctx.temperature_k,
    )
    lines = strip_ansi("\n".join(window_table_panel(ctx, statuses).lines)).splitlines()
    body = [l for l in lines if l.strip().startswith("w")]
    assert body[0].split()[0] == "w02"
    assert BAD in body[0]


def test_window_detail_panel_names_the_window_and_its_restraint(tmp_path):
    ctx = _ctx(tmp_path)
    text = strip_ansi("\n".join(window_detail_panel(ctx, 2).lines))
    assert "5.10" in text                           # its centre
    assert "2.5" in text                            # its k
    assert "w03" in text or "w01" in text           # its exchange partners
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_panels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.dashboard.panels'`

- [ ] **Step 3: Write `panels.py`**

Start from this skeleton, then port the remaining functions from the table above using the
attribute-substitution rule. `overlap_panel` and `boost_envelope_panel` are given in full
as the two worked examples; the rest follow the same shape.

```python
# gareus/dashboard/panels.py
"""Content panels for the live dashboard, as pure functions of DashboardContext.

Ported from DistanceLogger's _render_* methods. Two changes from the originals:
they take `ctx` rather than `self`, and they drop their own leading title line
because the title now lives on the Panel (which is what lets the allocator
account for chrome).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from ..colors import ROLE_BAD, ROLE_GOOD, ROLE_WARN, color_text, role_text
from ..math_helpers import boost_anharmonicity
from ..tui import _mini_bar
from ..tui_screen import Panel
from .context import DashboardContext
from .ranking import BAD, DEAD_OVERLAP, OK, WARN, WindowStatus, restraint_sigma, tail_summary

_K0_SATURATED = 0.999


def panel(
    key: str,
    title: str,
    lines: Iterable[str],
    *,
    min_lines: int,
    want_lines: int,
    priority: int,
    weight: float = 1.0,
    tail: str = "",
) -> Panel:
    return Panel(
        key=key, title=title, lines=tuple(str(x) for x in lines), min_lines=int(min_lines),
        want_lines=int(want_lines), priority=int(priority), weight=float(weight), tail=tail,
    )


def overlap_panel(ctx: DashboardContext) -> Panel:
    """Neighbour histogram overlap, worst pair first.

    Ordering is the whole point: the previous version emitted pairs in window
    order and truncated the tail, so a disconnected pair past the visible cut
    was invisible.
    """
    pairs = sorted(ctx.overlap_pairs.items(), key=lambda kv: (kv[1], kv[0]))
    if not pairs:
        return panel("overlap", "neighbour overlap — worst first",
                     [color_text("insufficient samples", "dim")],
                     min_lines=5, want_lines=12, priority=1, weight=1.3)
    lines = [color_text("pair        overlap   target 0.30", "white", bold=True)]
    for (a, b), ov in pairs:
        if ov < DEAD_OVERLAP:
            role, label = ROLE_BAD, "DEAD  → MBAR graph breaks" if ov < 0.05 else "BAD"
        elif ov < 0.30:
            role, label = ROLE_WARN, "WARN"
        else:
            role, label = ROLE_GOOD, OK
        lines.append(
            f"  w{a:02d}-w{b:02d}   {ov:6.2f}   {_mini_bar(ov / 0.5, 10)}  "
            + role_text(label, role)
        )
    return panel("overlap", "neighbour overlap — worst first", lines,
                 min_lines=5, want_lines=12, priority=1, weight=1.3)


def boost_envelope_panel(ctx: DashboardContext) -> Panel:
    """GaMD envelope: achieved sigma vs calibration target, k0 saturation, shape.

    The sigma/k0 half comes from the sidecar calibration file, the shape half
    from live boost samples. A run with no GaMD says so rather than rendering
    an empty box.
    """
    envelope = ((ctx.sidecar.gamd or {}).get("joint_envelope") or {})
    group = next(iter(envelope.values()), {}) if isinstance(envelope, dict) else {}
    boosts = [b for b in ctx.boost_history_all if math.isfinite(b)]
    if not group and not boosts:
        return panel("boost", "GaMD boost envelope",
                     [color_text("no GaMD boost (plain umbrella run)", "dim")],
                     min_lines=5, want_lines=11, priority=1)

    lines: list[str] = []
    if boosts:
        shape = boost_anharmonicity(boosts)
        score = float(shape.get("anharmonicity_score", float("nan")))
        role = ROLE_BAD if score > 1.5 else (ROLE_WARN if score > 1.0 else ROLE_GOOD)
        lines.append(
            f"  boost {np.mean(boosts):.2f} ± {np.std(boosts):.2f} kcal/mol   "
            f"skew {shape.get('skew', float('nan')):+.2f}  "
            f"kurt {shape.get('excess_kurtosis', float('nan')):+.2f}"
        )
        lines.append("  anharmonicity " + role_text(
            f"{score:.2f} " + ("HIGH" if score > 1.0 else OK), role))
    if group:
        sigma_v = float(group.get("sigmaV_kj_mol", float("nan")))
        sigma_0 = float(group.get("sigma0_kj_mol", float("nan")))
        k0 = float(group.get("k0", float("nan")))
        pct = (100.0 * sigma_v / sigma_0) if sigma_0 else float("nan")
        lines.append(f"  σΔV {sigma_v:.2f} / σ0 {sigma_0:.2f} kJ  ({pct:.0f}%)")
        if math.isfinite(k0):
            k0_label = "SATURATED at ceiling" if k0 >= _K0_SATURATED else OK
            role = ROLE_BAD if k0 >= _K0_SATURATED else ROLE_GOOD
            lines.append(f"  k0 {k0:.2f}  " + role_text(k0_label, role))
    return panel("boost", "GaMD boost envelope", lines,
                 min_lines=5, want_lines=11, priority=1)
```

`cv_map_panel` is the one function that is *not* a `self.X -> ctx.X` substitution: it wraps
`gareus.tui.render_distance_ascii`, whose full signature is `(rows, phase, step,
total_steps, width, max_replicas, mode, history_by_replica, history_by_window,
histogram_source, primary_label, primary_units, primary_k_unit_label)`. Every argument comes
from `ctx` (including the `cv_history_by_replica`, `ascii_mode` and `ascii_max_replicas`
fields added in Task 5), and its first three lines — its own border, title and separator —
are dropped because the Panel now supplies them, exactly as `gareus/logger.py:2497-2500`
does today. Written out:

```python
def cv_map_panel(ctx: DashboardContext, bar_width: int) -> Panel:
    """Per-window CV distributions: the densest panel on the dashboard."""
    if ctx.ascii_mode == "none":
        return panel("cv_map", "per-window CV distributions",
                     [color_text("disabled (--distance-ascii-mode none)", "dim")],
                     min_lines=1, want_lines=2, priority=2, weight=2.4)
    block = render_distance_ascii(
        [dict(r) for r in ctx.rows], phase=ctx.phase, step=ctx.step,
        total_steps=ctx.total_steps, width=int(bar_width),
        max_replicas=ctx.ascii_max_replicas, mode=ctx.ascii_mode,
        history_by_replica={k: list(v) for k, v in ctx.cv_history_by_replica.items()},
        history_by_window={k: list(v) for k, v in ctx.cv_history_by_window.items()},
        histogram_source="window", primary_label=ctx.primary_cv_label,
        primary_units=ctx.primary_cv_units, primary_k_unit_label=ctx.primary_k_units,
    )
    lines = block.splitlines()
    if len(lines) > 4:
        lines = lines[3:]                 # drop its own border/title/separator
    return panel("cv_map", "per-window CV distributions",
                 lines or [color_text("no samples yet", "dim")],
                 min_lines=6, want_lines=18, priority=2, weight=2.4)
```

(add `from ..tui import _mini_bar, render_distance_ascii` to the imports).

Then port, in this order, following the table and the substitution rule: `pe_map_panel`,
`exchange_panel`, `pull_panel`, `replica_table_panel`, and write the two new ones:

```python
def window_table_panel(ctx: DashboardContext, statuses: Sequence[WindowStatus]) -> Panel:
    """One row per window, worst first, with a severity-composition tail."""
    header = color_text(
        "win   cv1 ctr   cv2 ctr    accL   accR   |Δ|max   status", "white", bold=True)
    lines = [header]
    for s in statuses:
        w = s.window
        centre = ctx.centers_a[w] if w < len(ctx.centers_a) else float("nan")
        sec = ctx.secondary_centers[w] if w < len(ctx.secondary_centers) else float("nan")
        acc_l = ctx.acceptance_pairs.get((w - 1, w), float("nan"))
        acc_r = ctx.acceptance_pairs.get((w, w + 1), float("nan"))
        delta = abs(float(ctx.deltas.get(w, float("nan"))))
        role = {BAD: ROLE_BAD, WARN: ROLE_WARN}.get(s.status, ROLE_GOOD)
        lines.append(
            f"  w{w:02d}  {centre:8.2f}  {sec:8.2f}   {acc_l:5.2f}  {acc_r:5.2f}   "
            f"{delta:6.2f}   " + role_text(s.status, role)
            + ("  " + ", ".join(s.reasons) if s.reasons else "")
        )
    return panel("windows", "windows — worst first", lines, min_lines=4, want_lines=16,
                 priority=1, tail=tail_summary(statuses[max(0, len(statuses) - 1):]))


def window_detail_panel(ctx: DashboardContext, window: int) -> Panel:
    """Everything known about one window, for the auto- or key-selected row."""
    w = int(window)
    centre = ctx.centers_a[w] if w < len(ctx.centers_a) else float("nan")
    k = ctx.k_list[w] if w < len(ctx.k_list) else float("nan")
    sigma = restraint_sigma(k, ctx.temperature_k)
    samples = ctx.cv_history_by_window.get(w, ())
    mean_cv = float(np.mean(samples)) if samples else float("nan")
    lines = [
        f"  restraint   cv1 {centre:.2f} {ctx.primary_cv_units}  "
        f"k {k:.2f} {ctx.primary_k_units}   σ {sigma:.2f}",
        f"  sampled     n {len(samples)} in history   mean {mean_cv:.2f}   "
        f"Δ {mean_cv - centre:+.2f}",
    ]
    for (a, b), rate in sorted(ctx.acceptance_pairs.items()):
        if w in (a, b):
            other = b if a == w else a
            label = "DEAD" if math.isfinite(rate) and rate < 0.02 else OK
            lines.append(f"  exchange    w{other:02d} {rate:5.2f}  " + label)
    trace = ctx.window_trace_by_replica.get(w, ())
    if trace:
        lines.append(f"  occupancy   windows visited: {', '.join(f'w{t:02d}' for t in trace[-6:])}")
    return panel(f"detail-w{w:02d}", f"w{w:02d} detail", lines,
                 min_lines=6, want_lines=11, priority=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_panels.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/dashboard/panels.py tests/test_dashboard_panels.py
git commit -m "refactor(dashboard): port content panels onto DashboardContext"
```

---

## Task 7: The spine

**Files:**
- Create: `gareus/dashboard/spine.py`
- Test: `tests/test_dashboard_spine.py`

**Interfaces:**
- Consumes: Task 5's `DashboardContext`; Task 4's `rank_windows`/`BAD`/`WARN`; `gareus.tui.make_progress_bar`, `format_duration`, `_ansi_truncate`, `_coverage_bar` (moved there in Task 6 Step 0).
- Produces: `bucket_strip(values: Sequence[float], statuses: Sequence[str], cells: int, glyphs: str = "unicode") -> str`; `spine_lines(ctx: DashboardContext, lines_budget: int) -> tuple[str, ...]` returning exactly `min(lines_budget, 10)` lines; `FULL_SPINE_LINES = 10`; `COMPACT_SPINE_LINES = 5`.

`bucket_strip` is the load-bearing piece: one glyph per window works to ~120 windows, and
real runs in this repo reach 364. Past that, cells aggregate — **height is the bucket mean,
status is the bucket's worst** — so a bad window is coarsened in position but never hidden.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_spine.py
import argparse

from gareus.dashboard.context import build_context
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.dashboard.spine import COMPACT_SPINE_LINES, FULL_SPINE_LINES, bucket_strip, spine_lines
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi, strip_ansi_len

CENTERS = tuple(4.0 + 0.55 * i for i in range(25))


def _ctx(tmp_path, *, term_w=140, term_h=45, sidecar=None, n=25):
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    for w in range(n):
        logger.history_by_window[w] = [CENTERS[w] + 0.1 * (i % 5 - 2) for i in range(50)]
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05} for w in range(n)]
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=12_450_000,
        total_steps=37_500_000, summary={}, dashboard_info={
            "centers_a": list(CENTERS[:n]), "n_windows": n, "k_list": [2.5] * n,
            "exchange_stats": {f"{i}-{i+1}": {"attempts": 40, "accepted": 12}
                               for i in range(n - 1)},
            "primary_cv_label": "contacts", "primary_cv_units": "A",
            "adaptive_phase": {"epoch_index": 1, "epoch_total": 2,
                               "segment_name": "epoch_001/topup_002"}},
        sidecar=sidecar or SidecarSnapshot(), term_w=term_w, term_h=term_h, now=1000.0,
        view="progress", glyphs="unicode",
    )


def test_bucket_strip_renders_one_cell_per_value_when_it_fits():
    assert strip_ansi_len(bucket_strip([1.0, 2.0, 3.0], ["ok"] * 3, cells=8)) == 3


def test_bucket_strip_aggregates_when_there_are_more_values_than_cells():
    assert strip_ansi_len(bucket_strip([1.0] * 364, ["ok"] * 364, cells=100)) == 100


def test_bucket_strip_keeps_a_bad_window_visible_after_bucketing():
    statuses = ["ok"] * 364
    statuses[200] = "BAD"
    strip = bucket_strip([1.0] * 364, statuses, cells=100, glyphs="ascii")
    assert "X" in strip                     # ascii mode encodes status in the glyph


def test_bucket_strip_handles_an_empty_input():
    assert bucket_strip([], [], cells=20) == ""


def test_spine_lines_returns_exactly_ten_lines_and_fits_the_width(tmp_path):
    lines = spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)
    assert len(lines) == FULL_SPINE_LINES
    assert all(strip_ansi_len(l) <= 140 - 2 for l in lines)


def test_spine_lines_names_the_phase_epoch_and_window_count(tmp_path):
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)))
    assert "gareus_production" in text
    assert "1/2" in text
    assert "25 win" in text


def test_spine_shows_the_pool_budget_when_the_sidecar_has_it(tmp_path):
    sidecar = SidecarSnapshot(pool={"total_ns": 15000.0, "used_ns": 7500.0,
                                    "remaining_ns": 7500.0, "events": []})
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path, sidecar=sidecar), FULL_SPINE_LINES)))
    assert "7500" in text and "15000" in text


def test_spine_says_the_pool_is_unavailable_rather_than_faking_a_bar(tmp_path):
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)))
    assert "pool" in text and "unavailable" in text


def test_spine_reports_gamd_saturation_as_text_not_only_colour(tmp_path):
    sidecar = SidecarSnapshot(gamd={"joint_envelope": {"Dihedral": {
        "sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 11.039, "k0": 1.0}}})
    text = strip_ansi("\n".join(spine_lines(_ctx(tmp_path, sidecar=sidecar), FULL_SPINE_LINES)))
    assert "SATURATED" in text


def test_compact_spine_returns_five_lines_and_keeps_verdict_and_progress(tmp_path):
    lines = spine_lines(_ctx(tmp_path, term_w=100, term_h=24), COMPACT_SPINE_LINES)
    assert len(lines) == COMPACT_SPINE_LINES
    text = strip_ansi("\n".join(lines))
    assert "gareus_production" in text
    assert "%" in text                       # progress survives the compact tier


def test_spine_omits_the_cv2_line_for_a_1d_run(tmp_path):
    lines = spine_lines(_ctx(tmp_path), FULL_SPINE_LINES)
    assert not any(strip_ansi(l).startswith("cv2") for l in lines)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_spine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.dashboard.spine'`

- [ ] **Step 3: Write the implementation**

```python
# gareus/dashboard/spine.py
"""The always-visible header: ten lines that answer all three operator questions.

Every line is `label + elastic middle + fixed right summary`; the middle is sized
from the measured width of the fixed parts, never guessed. Lines 6 and 7 share an
x-axis on purpose — pair *i* sits between windows *i* and *i+1*, so a dead pair
reads as a seam between two window columns.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from ..colors import ROLE_BAD, ROLE_GOOD, ROLE_WARN, color_text, role_text
from ..tui import _ansi_truncate, _coverage_bar, format_duration, make_progress_bar, strip_ansi_len
from .context import DashboardContext
from .ranking import BAD, OK, WARN, rank_windows, restraint_sigma

FULL_SPINE_LINES = 10
COMPACT_SPINE_LINES = 5

_DENSITY_GLYPHS = " ▁▂▃▄▅▆▇█"
_ASCII_GLYPHS = " .:-=+*#%"
_K0_SATURATED = 0.999


def bucket_strip(
    values: Sequence[float], statuses: Sequence[str], cells: int, glyphs: str = "unicode"
) -> str:
    """One glyph per value, or per aggregated bucket when values outnumber cells.

    Bucket height is the mean; bucket status is the worst member. Real 364-window
    runs cannot fit one cell each, and dropping the tail would hide exactly the
    windows worth seeing.
    """
    vals = [float(v) for v in values]
    if not vals:
        return ""
    n_cells = max(1, min(int(cells), len(vals)))
    ramp = _DENSITY_GLYPHS if glyphs != "ascii" else _ASCII_GLYPHS
    edges = np.linspace(0, len(vals), n_cells + 1).astype(int)
    finite = [v for v in vals if math.isfinite(v)]
    hi = max(finite) if finite else 1.0
    lo = min(finite) if finite else 0.0
    span = (hi - lo) or 1.0
    out: list[str] = []
    for i in range(n_cells):
        chunk = vals[edges[i]:edges[i + 1]] or [lo]
        chunk_status = [str(statuses[j]) for j in range(edges[i], edges[i + 1])
                        if j < len(statuses)]
        level = int(round((float(np.mean(chunk)) - lo) / span * (len(ramp) - 1)))
        glyph = ramp[max(0, min(len(ramp) - 1, level))]
        if BAD in chunk_status:
            out.append(role_text("X" if glyphs == "ascii" else glyph, ROLE_BAD))
        elif WARN in chunk_status:
            out.append(role_text("!" if glyphs == "ascii" else glyph, ROLE_WARN))
        else:
            out.append(glyph)
    return "".join(out)


def _statuses_by_window(ctx: DashboardContext) -> list[str]:
    ranked = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k,
    )
    by_window = {s.window: s.status for s in ranked}
    return [by_window.get(w, OK) for w in range(ctx.n_windows)]


def _verdict(ctx: DashboardContext) -> str:
    issues = int(ctx.decision.get("issue_count", 0) or 0)
    status = str(ctx.decision.get("status", "") or "").upper()
    if status.startswith("BAD") or status.startswith("FAIL"):
        return role_text(f"✗ BAD {issues}", ROLE_BAD)
    if issues:
        return role_text(f"⚠ CAUTION {issues}", ROLE_WARN)
    return role_text("✓ OK", ROLE_GOOD)


def _pool_line(ctx: DashboardContext, width: int) -> str:
    pool = ctx.sidecar.pool or {}
    total = float(pool.get("total_ns", 0.0) or 0.0)
    used = float(pool.get("used_ns", 0.0) or 0.0)
    if total <= 0.0:
        return "pool  " + color_text("unavailable (no adaptive_runtime_pool.json yet)", "dim")
    ns_day = _ns_per_day(ctx)
    right = f"  {used:.0f}/{total:.0f} ns   perf {ns_day:.0f} ns/d/rep"
    bar_w = max(12, width - len("pool  ") - len(right))
    return "pool  [" + make_progress_bar(used / total, bar_w) + "]" + right


def _ns_per_day(ctx: DashboardContext) -> float:
    if ctx.timestep_fs <= 0.0 or ctx.elapsed_s <= 0.0:
        return float("nan")
    sim_ns = ctx.display_step * ctx.timestep_fs / 1.0e6
    return sim_ns / ctx.elapsed_s * 86400.0


def _run_line(ctx: DashboardContext, width: int) -> str:
    total = ctx.display_total_steps or 0
    frac = (ctx.display_step / total) if total else 0.0
    right = (f"  {100.0 * frac:5.1f}%   {ctx.display_step / 1e6:.2f}/{total / 1e6:.2f} Msteps"
             f"   wall {format_duration(ctx.elapsed_s)}  eta {format_duration(ctx.eta_s)}")
    bar_w = max(12, width - len("run   ") - len(right))
    return "run   [" + make_progress_bar(frac, bar_w) + "]" + right


def _gamd_line(ctx: DashboardContext) -> str:
    group = next(iter(((ctx.sidecar.gamd or {}).get("joint_envelope") or {}).values()), {})
    if not group:
        return "gamd  " + color_text("no GaMD boost (plain umbrella run)", "dim")
    sigma_v = float(group.get("sigmaV_kj_mol", float("nan")))
    sigma_0 = float(group.get("sigma0_kj_mol", float("nan")))
    k0 = float(group.get("k0", float("nan")))
    pct = (100.0 * sigma_v / sigma_0) if sigma_0 else float("nan")
    k0_txt = role_text("SATURATED", ROLE_BAD) if k0 >= _K0_SATURATED else role_text(OK, ROLE_GOOD)
    return (f"gamd  σΔV {sigma_v:.2f} / σ0 {sigma_0:.2f} kJ ({pct:.0f}%)   "
            f"k0 {k0:.2f} {k0_txt}")


def _alert_line(ctx: DashboardContext) -> str:
    ranked = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k,
    )
    alerts = [f"w{s.window:02d} " + ", ".join(s.reasons) for s in ranked
              if s.status in {BAD, WARN}][:2]
    if not alerts:
        return "alert " + color_text("none", "dim")
    return "alert " + "   |   ".join(alerts)


def spine_lines(ctx: DashboardContext, lines_budget: int) -> tuple[str, ...]:
    """Render the spine at the requested height (10 full, 5 compact)."""
    width = max(40, ctx.term_w - 2)
    statuses = _statuses_by_window(ctx)
    counts = [float(len(ctx.cv_history_by_window.get(w, ()))) for w in range(ctx.n_windows)]
    epoch = ""
    if ctx.epoch_index is not None:
        epoch = f"ep {ctx.epoch_index}/{ctx.epoch_total}  "
    identity = (f"GaREUS  {ctx.run_label}   {ctx.phase}  {epoch}{ctx.segment_name}   "
                f"{ctx.n_windows} win  {ctx.topology_label}")
    if ctx.secondary_cv_type:
        identity += f"  cv2 {ctx.secondary_cv_type}"
    identity += "      " + _verdict(ctx)

    win_strip = bucket_strip(counts, statuses, cells=max(8, width - 60), glyphs=ctx.glyphs)
    accept = [ctx.acceptance_pairs.get((w, w + 1), float("nan")) for w in range(ctx.n_windows - 1)]
    pair_status = [BAD if (math.isfinite(a) and a < 0.02) else OK for a in accept]
    exch_strip = bucket_strip(accept, pair_status, cells=max(8, width - 60), glyphs=ctx.glyphs)
    finite_acc = [a for a in accept if math.isfinite(a)]

    lines = [identity, _run_line(ctx, width), _pool_line(ctx, width)]
    cv_vals = [v for w in range(ctx.n_windows) for v in ctx.cv_history_by_window.get(w, ())]
    if cv_vals and ctx.centers_a:
        lo, hi = min(ctx.centers_a), max(ctx.centers_a)
        lines.append(f"cv1 {lo:6.1f} |{_coverage_bar(cv_vals, lo, hi, max(12, width - 46))}| "
                     f"{hi:6.1f} {ctx.primary_cv_units}   span {hi - lo:.1f}")
    else:
        lines.append("cv1   " + color_text("no samples yet", "dim"))
    if ctx.is_2d and ctx.secondary_centers:
        sec_vals = [v for w in range(ctx.n_windows)
                    for v in ctx.secondary_history_by_window.get(w, ())]
        s_lo, s_hi = min(ctx.secondary_centers), max(ctx.secondary_centers)
        bar = _coverage_bar(sec_vals, s_lo, s_hi, max(12, width - 46)) if sec_vals else ""
        lines.append(f"cv2 {s_lo:+6.1f} |{bar}| {s_hi:+6.1f}   "
                     f"{len(set(round(x, 4) for x in ctx.secondary_centers))} rows")
    lines.append(f"win   |{win_strip}|  samples {min(counts) if counts else 0:.0f}-"
                 f"{max(counts) if counts else 0:.0f}/win")
    lines.append(f"exch   {exch_strip}    accept "
                 + (f"{np.mean(finite_acc):.2f} mean, {min(finite_acc):.2f} min"
                    if finite_acc else "no attempts yet"))
    lines.append(_gamd_line(ctx))
    lines.append(_alert_line(ctx))

    budget = max(1, int(lines_budget))
    if budget >= FULL_SPINE_LINES:
        chosen = lines[:FULL_SPINE_LINES]
    else:
        # Compact tier: identity, run, pool, window strip, top alert.
        chosen = [lines[0], lines[1], lines[2],
                  next((l for l in lines if l.startswith("win   ")), ""),
                  lines[-1]][:budget]
    while len(chosen) < min(budget, FULL_SPINE_LINES):
        chosen.append("")
    return tuple(_ansi_truncate(l, width) for l in chosen)


__all__ = ["COMPACT_SPINE_LINES", "FULL_SPINE_LINES", "bucket_strip", "spine_lines"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_spine.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/dashboard/spine.py tests/test_dashboard_spine.py
git commit -m "feat(dashboard): add the ten-line spine with bucketed window strips"
```

---

## Task 8: PROGRESS view

**Files:**
- Create: `gareus/dashboard/view_progress.py`
- Test: `tests/test_dashboard_views.py`

**Interfaces:**
- Consumes: Task 1's `Panel`/`Row`; Task 6's `panel`; Task 5's `DashboardContext`.
- Produces: `build(ctx: DashboardContext) -> tuple[Row, ...]`; `timeline_panel(ctx) -> Panel`; `projection_panel(ctx) -> Panel`; `throughput_panel(ctx) -> Panel`; `collapse_extension_rounds(events: Sequence[Mapping]) -> tuple[tuple[str, float, int, str], ...]`.

The timeline is the pool ledger's `events[]`, chronological. Repeated
`epoch_NNN/baseline` entries are the quality-gate extension rounds; they collapse into
`↳ ext round N` rows so a 28-event ledger fits a 12-line panel.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_views.py
import argparse

from gareus.dashboard import view_physics, view_progress, view_windows
from gareus.dashboard.context import build_context
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi

CENTERS = tuple(4.0 + 0.55 * i for i in range(6))

POOL = {
    "total_ns": 15000.0, "used_ns": 7500.0, "remaining_ns": 7500.0, "timestep_fs": 4.0,
    "events": [
        {"label": "epoch_000", "kind": "adaptive_epoch", "consumed_ns": 1875.0, "n_states": 27},
        {"label": "final/baseline", "kind": "scheduled_final", "consumed_ns": 14.5, "n_states": 29},
        {"label": "epoch_001/baseline", "kind": "scheduled_epoch", "consumed_ns": 1391.7, "n_states": 29},
        {"label": "epoch_001/baseline", "kind": "scheduled_epoch", "consumed_ns": 1029.3, "n_states": 29},
        {"label": "epoch_001/baseline", "kind": "scheduled_epoch", "consumed_ns": 757.5, "n_states": 29},
    ],
}
GAMD = {"joint_envelope": {"Dihedral": {"sigma0_kj_mol": 12.552, "sigmaV_kj_mol": 11.039, "k0": 1.0}}}


def _ctx(tmp_path, *, view="progress", sidecar=None, exchange=None, n=6, term_w=140, term_h=45):
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=4.0), no_file_persistence=True)
    for w in range(n):
        logger.history_by_window[w] = [CENTERS[w] + 0.1 * (i % 5 - 2) for i in range(40)]
    logger.boost_history_all = [2.0 + 0.4 * (i % 6) for i in range(120)]
    rows = [{"replica": w, "window": w, "center_A": CENTERS[w], "k_kcal_mol_A2": 2.5,
             "cv_A": CENTERS[w] + 0.05, "umbrella_bias_kcal_mol": 0.0,
             "umbrella_pull_kcal_mol_A": 0.0} for w in range(n)]
    stats = exchange if exchange is not None else {
        f"{i}-{i+1}": {"attempts": 40, "accepted": 12} for i in range(n - 1)}
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=1_000_000,
        total_steps=4_000_000, summary={}, dashboard_info={
            "centers_a": list(CENTERS[:n]), "n_windows": n, "k_list": [2.5] * n,
            "exchange_stats": stats, "primary_cv_label": "contacts",
            "primary_cv_units": "A", "primary_k_units": "kcal/mol/A^2"},
        sidecar=sidecar or SidecarSnapshot(), term_w=term_w, term_h=term_h, now=5000.0,
        view=view, glyphs="unicode",
    )


def _text(rows):
    return strip_ansi("\n".join(line for row in rows for p in row.panels for line in p.lines))


def test_progress_view_lists_pool_events_chronologically(tmp_path):
    rows = view_progress.build(_ctx(tmp_path, sidecar=SidecarSnapshot(pool=POOL)))
    text = _text(rows)
    assert text.index("epoch_000") < text.index("epoch_001")
    assert "1875" in text


def test_progress_view_collapses_repeated_extension_rounds(tmp_path):
    rows = view_progress.build(_ctx(tmp_path, sidecar=SidecarSnapshot(pool=POOL)))
    text = _text(rows)
    assert "ext round" in text
    assert text.count("epoch_001/baseline") == 1        # first occurrence only


def test_progress_view_explains_a_missing_pool_ledger(tmp_path):
    text = _text(view_progress.build(_ctx(tmp_path)))
    assert "adaptive_runtime_pool.json" in text


def test_progress_view_projects_remaining_budget_in_gpu_days(tmp_path):
    text = _text(view_progress.build(_ctx(tmp_path, sidecar=SidecarSnapshot(pool=POOL))))
    assert "7500" in text
    assert "GPU-day" in text


def test_progress_view_returns_rows_of_panels(tmp_path):
    rows = view_progress.build(_ctx(tmp_path, sidecar=SidecarSnapshot(pool=POOL)))
    assert rows and all(row.panels for row in rows)
    assert {p.key for row in rows for p in row.panels} >= {"timeline", "projection", "throughput"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_views.py -k progress -v`
Expected: FAIL with `ImportError: cannot import name 'view_progress'`

- [ ] **Step 3: Write the implementation**

```python
# gareus/dashboard/view_progress.py
"""PROGRESS view: where the campaign is, and how long the rest will take.

Everything here comes from the pool ledger on disk, which is the only record of
the quality-gate extension rounds -- they were previously invisible live.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from ..colors import color_text
from ..tui import format_duration, make_progress_bar
from ..tui_screen import Panel, Row
from .context import DashboardContext
from .panels import panel

_SECONDS_PER_DAY = 86400.0


def collapse_extension_rounds(
    events: Sequence[Mapping[str, object]]
) -> tuple[tuple[str, float, int, str], ...]:
    """Collapse repeated same-label segments into ``↳ ext round N`` rows.

    A quality-gate extension loop re-invokes the same segment with a shrinking
    remaining gap, so the ledger holds several `epoch_001/baseline` entries. The
    first keeps its name; later ones are numbered.
    """
    seen: dict[str, int] = {}
    out: list[tuple[str, float, int, str]] = []
    for event in events:
        label = str(event.get("label", "") or "")
        consumed = float(event.get("consumed_ns", 0.0) or 0.0)
        states = int(event.get("n_states", 0) or 0)
        seen[label] = seen.get(label, 0) + 1
        display = label if seen[label] == 1 else f" ↳ ext round {seen[label]}"
        out.append((display, consumed, states, str(event.get("kind", "") or "")))
    return tuple(out)


def _ledger_age(ctx: DashboardContext) -> str:
    """How stale the ledger is. It is written at phase boundaries, so early in a
    segment it legitimately lags reality -- say so rather than implying it is live."""
    if ctx.sidecar.pool_mtime is None:
        return ""
    age_s = max(0.0, ctx.now_wall - float(ctx.sidecar.pool_mtime))
    return f" (ledger {format_duration(age_s)} old)"


def timeline_panel(ctx: DashboardContext) -> Panel:
    pool = ctx.sidecar.pool or {}
    events = pool.get("events") or []
    if not events:
        return panel("timeline", "campaign timeline",
                     [color_text("unavailable: no adaptive_runtime_pool.json yet "
                                 "(non-adaptive run, or first epoch still open)", "dim")],
                     min_lines=1, want_lines=2, priority=1)
    rows = collapse_extension_rounds(events)
    peak = max((ns for _l, ns, _s, _k in rows), default=1.0) or 1.0
    bar_w = max(10, (ctx.term_w - 2) - 60)
    lines = []
    for label, consumed, states, _kind in rows:
        filled = int(round(bar_w * consumed / peak))
        lines.append(f" {label:<22.22} " + "█" * filled + " " * (bar_w - filled)
                     + f" {consumed:9.1f} ns  {states:2d} st  ✓ done")
    remaining = float(pool.get("remaining_ns", 0.0) or 0.0)
    if remaining > 0:
        lines.append(f" {'final (reserve)':<22.22} " + "░" * bar_w
                     + f" {remaining:9.1f} ns          ○ scheduled")
    return panel("timeline", "campaign timeline" + _ledger_age(ctx), lines,
                 min_lines=6, want_lines=min(12, len(lines)), priority=1)


def _ns_per_day(ctx: DashboardContext) -> float:
    if ctx.timestep_fs <= 0.0 or ctx.elapsed_s <= 0.0:
        return float("nan")
    return ctx.display_step * ctx.timestep_fs / 1.0e6 / ctx.elapsed_s * _SECONDS_PER_DAY


def projection_panel(ctx: DashboardContext) -> Panel:
    pool = ctx.sidecar.pool or {}
    remaining = float(pool.get("remaining_ns", 0.0) or 0.0)
    ns_day = _ns_per_day(ctx)
    aggregate = ns_day * ctx.n_replicas if math.isfinite(ns_day) else float("nan")
    lines = [f" perf now        {ns_day:.0f} ns/day/rep   ({aggregate:.0f} aggregate)"]
    if remaining > 0 and math.isfinite(aggregate) and aggregate > 0:
        lines.append(f" pool remaining  {remaining:.0f} ns = "
                     f"{remaining / aggregate:.1f} GPU-days at this rate")
    elif remaining > 0:
        lines.append(f" pool remaining  {remaining:.0f} ns (GPU-days unknown until "
                     f"throughput is measured)")
    lines.append(f" this segment    eta {format_duration(ctx.eta_s)}")
    lines.append(f" elapsed         {format_duration(ctx.elapsed_s)}")
    return panel("projection", "projection", lines,
                 min_lines=4, want_lines=7, priority=2, weight=1.0)


def throughput_panel(ctx: DashboardContext) -> Panel:
    total_samples = sum(len(v) for v in ctx.cv_history_by_window.values())
    lines = [
        f" replicas        {ctx.n_replicas}",
        f" cv samples      {total_samples} in live history",
        f" step            {ctx.display_step} / {ctx.display_total_steps or '?'}",
    ]
    if ctx.timestep_fs > 0:
        lines.append(f" sim time        {ctx.display_step * ctx.timestep_fs / 1e6:.3g} ns/rep")
    return panel("throughput", "throughput", lines,
                 min_lines=3, want_lines=6, priority=3, weight=1.0)


def build(ctx: DashboardContext) -> tuple[Row, ...]:
    return (
        Row(panels=(timeline_panel(ctx),)),
        Row(panels=(projection_panel(ctx), throughput_panel(ctx))),
    )


__all__ = ["build", "collapse_extension_rounds", "projection_panel", "throughput_panel",
           "timeline_panel"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_views.py -k progress -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/dashboard/view_progress.py tests/test_dashboard_views.py
git commit -m "feat(dashboard): add PROGRESS view with pool-ledger timeline"
```

---

## Task 9: PHYSICS view

**Files:**
- Create: `gareus/dashboard/view_physics.py`
- Test: `tests/test_dashboard_views.py` (append)

**Interfaces:**
- Consumes: Task 6's `overlap_panel`, `boost_envelope_panel`, `panel`; Task 4's `restraint_sigma`.
- Produces: `build(ctx) -> tuple[Row, ...]`; `connected_components(n_windows: int, acceptance_pairs: Mapping[tuple[int, int], float], min_acceptance: float = 0.02) -> tuple[frozenset[int], ...]`; `sigma_spacing_panel(ctx) -> Panel`; `connectivity_panel(ctx) -> Panel`.

Two new analyses, both O(K): union-find connectivity (MBAR disconnection as a live
verdict) and the σ-vs-spacing check (`σ = sqrt(k_B T / k)` against real window spacing, so
a mistuned `k` shows on day 1 rather than in a post-run overlap report).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_dashboard_views.py
from gareus.dashboard.view_physics import connected_components


def test_connected_components_returns_one_set_for_a_fully_coupled_chain():
    pairs = {(i, i + 1): 0.30 for i in range(5)}
    comps = connected_components(6, pairs)
    assert len(comps) == 1 and comps[0] == frozenset(range(6))


def test_connected_components_splits_at_a_dead_pair():
    pairs = {(0, 1): 0.30, (1, 2): 0.001, (2, 3): 0.30}
    comps = connected_components(4, pairs)
    assert sorted(len(c) for c in comps) == [2, 2]


def test_connected_components_treats_unattempted_pairs_as_unlinked():
    pairs = {(0, 1): float("nan")}
    assert len(connected_components(2, pairs)) == 2


def test_physics_view_reports_connectivity_and_names_isolated_windows(tmp_path):
    ctx = _ctx(tmp_path, view="physics", exchange={
        "0-1": {"attempts": 40, "accepted": 12}, "1-2": {"attempts": 40, "accepted": 0},
        "2-3": {"attempts": 40, "accepted": 12}, "3-4": {"attempts": 40, "accepted": 12},
        "4-5": {"attempts": 40, "accepted": 12}})
    text = _text(view_physics.build(ctx))
    assert "connected" in text
    assert "w02" in text or "w01" in text


def test_physics_view_compares_restraint_sigma_with_window_spacing(tmp_path):
    text = _text(view_physics.build(_ctx(tmp_path, view="physics")))
    assert "spacing" in text
    assert "σ" in text or "sigma" in text


def test_physics_view_includes_overlap_and_boost_panels(tmp_path):
    rows = view_physics.build(_ctx(tmp_path, view="physics",
                                   sidecar=SidecarSnapshot(gamd=GAMD)))
    assert {p.key for row in rows for p in row.panels} >= {"overlap", "boost"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_views.py -k physics -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.dashboard.view_physics'`

- [ ] **Step 3: Write the implementation**

```python
# gareus/dashboard/view_physics.py
"""PHYSICS view: will MBAR work on what is being collected right now?

Both derived analyses here already exist post-run in analyze_gareus_mbar.py. The
cheap live versions are what let a mistuned k or a disconnected state graph be
fixed on day 1 instead of discovered after a week of MD.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ..colors import ROLE_BAD, ROLE_GOOD, ROLE_WARN, color_text, role_text
from ..tui_screen import Panel, Row
from .context import DashboardContext
from .panels import boost_envelope_panel, overlap_panel, panel
from .ranking import DEAD_ACCEPTANCE, OK, restraint_sigma

TARGET_OVERLAP = 0.30


def connected_components(
    n_windows: int,
    acceptance_pairs: Mapping[tuple[int, int], float],
    min_acceptance: float = DEAD_ACCEPTANCE,
) -> tuple[frozenset[int], ...]:
    """Union-find over pairs that actually exchange.

    A NaN rate means "not attempted yet", which is not evidence of a link, so it
    does not join two components.
    """
    parent = list(range(int(n_windows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), rate in acceptance_pairs.items():
        if not (0 <= a < n_windows and 0 <= b < n_windows):
            continue
        if math.isfinite(rate) and rate >= float(min_acceptance):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups: dict[int, set[int]] = {}
    for w in range(int(n_windows)):
        groups.setdefault(find(w), set()).add(w)
    return tuple(sorted((frozenset(g) for g in groups.values()), key=lambda s: (-len(s), min(s))))


def sigma_spacing_panel(ctx: DashboardContext) -> Panel:
    """Is k tuned for the window spacing actually in use?"""
    if len(ctx.centers_a) < 2 or not ctx.k_list:
        return panel("sigma", "σ-vs-spacing check",
                     [color_text("needs at least two windows", "dim")],
                     min_lines=1, want_lines=2, priority=3)
    spacings = np.diff(np.asarray(ctx.centers_a, dtype=float))
    median_spacing = float(np.median(np.abs(spacings)))
    k_median = float(np.median(np.asarray(ctx.k_list, dtype=float)))
    sigma = restraint_sigma(k_median, ctx.temperature_k)
    observed = [v for v in ctx.overlap_pairs.values() if math.isfinite(v)]
    obs_median = float(np.median(observed)) if observed else float("nan")
    ratio = (sigma / median_spacing) if median_spacing else float("nan")
    if not math.isfinite(ratio):
        verdict = color_text("not yet measurable", "dim")
    elif ratio < 0.5:
        verdict = role_text(f"k likely too stiff (σ/spacing {ratio:.2f})", ROLE_WARN)
    elif ratio > 1.5:
        verdict = role_text(f"k likely too soft (σ/spacing {ratio:.2f})", ROLE_WARN)
    else:
        verdict = role_text(f"σ/spacing {ratio:.2f} {OK}", ROLE_GOOD)
    return panel("sigma", "σ-vs-spacing check", [
        f" k median {k_median:.2f} {ctx.primary_k_units} → σ {sigma:.2f} "
        f"{ctx.primary_cv_units}   spacing {median_spacing:.2f}",
        f" observed median overlap {obs_median:.2f}   target {TARGET_OVERLAP:.2f}",
        " " + verdict,
    ], min_lines=3, want_lines=3, priority=2)


def connectivity_panel(ctx: DashboardContext) -> Panel:
    """MBAR readiness: is the state graph one piece?"""
    comps = connected_components(ctx.n_windows, ctx.acceptance_pairs)
    largest = len(comps[0]) if comps else 0
    lines = []
    if ctx.secondary_cv_type:
        lines.append(f" cv2 {ctx.secondary_cv_type}   "
                     f"{len(set(round(x, 4) for x in ctx.secondary_centers))} rows")
    if largest >= ctx.n_windows and ctx.n_windows > 0:
        lines.append(" graph " + role_text(f"{largest}/{ctx.n_windows} connected {OK}", ROLE_GOOD))
    else:
        isolated = sorted(w for comp in comps[1:] for w in comp)
        shown = ", ".join(f"w{w:02d}" for w in isolated[:8])
        lines.append(" graph " + role_text(
            f"{largest}/{ctx.n_windows} connected", ROLE_BAD)
            + f"   isolated: {shown}"
            + ("" if len(isolated) <= 8 else f" (+{len(isolated) - 8} more)"))
        lines.append(role_text(
            f" → union MBAR drops {ctx.n_windows - largest} states unless a bridge is added",
            ROLE_WARN))
    return panel("connectivity", "CV2 regime + graph connectivity", lines,
                 min_lines=2, want_lines=4, priority=1)


def build(ctx: DashboardContext) -> tuple[Row, ...]:
    return (
        Row(panels=(overlap_panel(ctx), boost_envelope_panel(ctx))),
        Row(panels=(sigma_spacing_panel(ctx),)),
        Row(panels=(connectivity_panel(ctx),)),
    )


__all__ = ["TARGET_OVERLAP", "build", "connected_components", "connectivity_panel",
           "sigma_spacing_panel"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_views.py -k physics -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/dashboard/view_physics.py tests/test_dashboard_views.py
git commit -m "feat(dashboard): add PHYSICS view with connectivity and sigma-spacing checks"
```

---

## Task 10: WINDOWS view

**Files:**
- Create: `gareus/dashboard/view_windows.py`
- Test: `tests/test_dashboard_views.py` (append)

**Interfaces:**
- Consumes: Task 6's `window_table_panel`, `window_detail_panel`, `cv_map_panel`, `pe_map_panel`; Task 4's `rank_windows`.
- Produces: `build(ctx) -> tuple[Row, ...]`; `select_window(ctx, statuses) -> int` (auto-selects the worst-ranked window when no key selection is available — Phase 1 always auto-selects, since key handling is Phase 2).

Row order and priorities, per the amended spec §7.3: ranked table (1), CV+PE maps
(2 / 3, weights 2.4 / 0.9), selected-window detail (1). The CV/PE row keeps today's
`pe_bar_width = max(10, pe_panel_w - 24)` arithmetic so the pinned 160×40 width holds.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_dashboard_views.py
from gareus.dashboard.ranking import rank_windows as _rank
from gareus.dashboard.view_windows import select_window


def _statuses(ctx, acceptance):
    return _rank(n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
                 acceptance_by_window=acceptance, overlap_by_pair=ctx.overlap_pairs,
                 delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k)


def test_select_window_picks_the_worst_ranked_window(tmp_path):
    ctx = _ctx(tmp_path, view="windows")
    statuses = _statuses(ctx, {0: 0.30, 1: 0.30, 2: 0.001, 3: 0.30, 4: 0.30, 5: 0.30})
    assert select_window(ctx, statuses) == 2


def test_select_window_falls_back_to_window_zero_when_all_are_healthy(tmp_path):
    ctx = _ctx(tmp_path, view="windows")
    statuses = _statuses(ctx, {w: 0.30 for w in range(6)})
    assert select_window(ctx, statuses) == 0


def test_windows_view_orders_rows_table_then_maps_then_detail(tmp_path):
    rows = view_windows.build(_ctx(tmp_path, view="windows"))
    keys = [tuple(p.key for p in row.panels) for row in rows]
    assert keys[0] == ("windows",)
    assert keys[1] == ("cv_map", "pe_map")
    assert keys[2][0].startswith("detail-w")


def test_windows_view_gives_the_maps_row_lower_priority_than_table_and_detail(tmp_path):
    rows = view_windows.build(_ctx(tmp_path, view="windows"))
    by_key = {p.key: p for row in rows for p in row.panels}
    assert by_key["windows"].priority == 1
    assert by_key["cv_map"].priority == 2
    assert by_key["pe_map"].priority == 3
    assert min(p.priority for p in rows[2].panels) == 1


def test_windows_view_keeps_the_historical_cv_pe_weights(tmp_path):
    rows = view_windows.build(_ctx(tmp_path, view="windows"))
    by_key = {p.key: p for row in rows for p in row.panels}
    assert (by_key["cv_map"].weight, by_key["pe_map"].weight) == (2.4, 0.9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_views.py -k windows -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.dashboard.view_windows'`

- [ ] **Step 3: Write the implementation**

```python
# gareus/dashboard/view_windows.py
"""WINDOWS view: which window is misbehaving, and why.

With no tty there are no keys, so the detail panel auto-selects the worst-ranked
window: a SLURM log tail then always shows expanded exactly the window someone
would have pressed `j` to reach.
"""

from __future__ import annotations

from typing import Sequence

from ..tui import _weighted_panel_widths, dashboard_row_gap
from ..tui_screen import Row
from .context import DashboardContext
from .panels import cv_map_panel, pe_map_panel, window_detail_panel, window_table_panel
from .ranking import OK, WindowStatus, rank_windows

CV_WEIGHT = 2.4
PE_WEIGHT = 0.9
_PE_LABEL_OVERHEAD = 24     # pinned by tests/test_dashboard_scaling_integration.py
_CV_LABEL_OVERHEAD = 72     # measured prefix + "] nw=<n>" suffix, see gareus/logger.py history


def select_window(ctx: DashboardContext, statuses: Sequence[WindowStatus]) -> int:
    """Worst-ranked window, or window 0 when everything is healthy."""
    for status in statuses:
        if status.status != OK:
            return int(status.window)
    return int(statuses[0].window) if statuses else 0


def _map_bar_widths(ctx: DashboardContext) -> tuple[int, int]:
    gap = dashboard_row_gap(ctx.term_w)
    cv_w, pe_w = _weighted_panel_widths([CV_WEIGHT, PE_WEIGHT], term_w=ctx.term_w, gap=gap)
    return max(24, cv_w - _CV_LABEL_OVERHEAD), max(10, pe_w - _PE_LABEL_OVERHEAD)


def build(ctx: DashboardContext) -> tuple[Row, ...]:
    statuses = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k,
    )
    cv_bar_w, pe_bar_w = _map_bar_widths(ctx)
    return (
        Row(panels=(window_table_panel(ctx, statuses),)),
        Row(panels=(cv_map_panel(ctx, cv_bar_w), pe_map_panel(ctx, pe_bar_w))),
        Row(panels=(window_detail_panel(ctx, select_window(ctx, statuses)),)),
    )


__all__ = ["CV_WEIGHT", "PE_WEIGHT", "build", "select_window"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_views.py -v`
Expected: 16 passed (5 progress + 6 physics + 5 windows)

- [ ] **Step 5: Commit**

```bash
git add gareus/dashboard/view_windows.py tests/test_dashboard_views.py
git commit -m "feat(dashboard): add WINDOWS view with ranked table and auto-selected detail"
```

---

## Task 11: Screen assembly and view resolution

**Files:**
- Create: `gareus/dashboard/screen.py`
- Modify: `gareus/dashboard/__init__.py` (re-export `render_screen`)
- Test: `tests/test_dashboard_screen.py`

**Interfaces:**
- Consumes: Task 2's `frame_tiers`, `compose_rows`; Task 1's `allocate_rows`; Task 7's `spine_lines`; Tasks 8-10's `build`.
- Produces: `VIEWS: Mapping[str, Callable]`; `promotion_reasons(ctx) -> tuple[str, ...]`; `resolve_view(ctx, requested: str) -> str`; `footer_line(ctx, dropped: Sequence[str], view: str) -> str`; `render_screen(ctx) -> str`.

`auto` shows PROGRESS and promotes to another view only while a promotion rule holds:
any BAD-ranked window or dead pair → `windows`; disconnected graph or anharmonicity > 1.0 →
`physics`. Promotion is a warning light, not a rotation.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_screen.py
import argparse

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


def _ctx(tmp_path, *, view="auto", term_w=140, term_h=45, exchange=None, sidecar=None):
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=4.0), no_file_persistence=True)
    for w in range(8):
        logger.history_by_window[w] = [CENTERS[w] + 0.1 * (i % 5 - 2) for i in range(40)]
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


def test_resolve_view_falls_back_to_progress_for_an_unknown_name(tmp_path):
    assert resolve_view(_ctx(tmp_path), "nonsense") == "progress"


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_screen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.dashboard.screen'`

- [ ] **Step 3: Write the implementation**

```python
# gareus/dashboard/screen.py
"""Assemble the frame: spine, one view's allocated rows, footer.

The frame is built to fit the terminal, so gareus.tui._safe_tui_frame_text's clip
becomes a safety net that should never fire. Panels that could not be given their
minimum are named in the footer -- silence about them is what made the old
overflow a bug rather than a limitation.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Mapping, Sequence

from ..colors import ROLE_SECTION, color_text, role_text
from ..math_helpers import boost_anharmonicity
from ..tui import _ansi_truncate
from ..tui_screen import allocate_rows, compose_rows, frame_tiers
from . import view_physics, view_progress, view_windows
from .context import DashboardContext
from .ranking import BAD, rank_windows
from .spine import spine_lines
from .view_physics import connected_components

VIEWS: Mapping[str, Callable[[DashboardContext], tuple]] = {
    "progress": view_progress.build,
    "physics": view_physics.build,
    "windows": view_windows.build,
}
DEFAULT_VIEW = "progress"
ANHARMONICITY_PROMOTE = 1.0


def promotion_reasons(ctx: DashboardContext) -> tuple[str, ...]:
    """Conditions that justify taking over the screen, worst first."""
    reasons: list[str] = []
    ranked = rank_windows(
        n_windows=ctx.n_windows, centers_a=ctx.centers_a, k_list=ctx.k_list,
        acceptance_by_window=ctx.acceptance_windows, overlap_by_pair=ctx.overlap_pairs,
        delta_by_window=ctx.deltas, temperature_k=ctx.temperature_k,
    )
    for status in ranked:
        if status.status == BAD:
            reasons.append(f"windows: w{status.window:02d} " + ", ".join(status.reasons))
            break
    comps = connected_components(ctx.n_windows, ctx.acceptance_pairs)
    if comps and len(comps[0]) < ctx.n_windows:
        reasons.append(f"physics: graph {len(comps[0])}/{ctx.n_windows} connected")
    boosts = [b for b in ctx.boost_history_all if math.isfinite(b)]
    if boosts:
        score = float(boost_anharmonicity(boosts).get("anharmonicity_score", float("nan")))
        if math.isfinite(score) and score > ANHARMONICITY_PROMOTE:
            reasons.append(f"physics: anharmonicity {score:.2f}")
    return tuple(reasons)


def resolve_view(ctx: DashboardContext, requested: str) -> str:
    """Explicit request wins; `auto` promotes only while a rule holds."""
    name = str(requested or "").lower()
    if name in VIEWS:
        return name
    if name != "auto":
        return DEFAULT_VIEW
    for reason in promotion_reasons(ctx):
        target = reason.split(":", 1)[0].strip()
        if target in VIEWS:
            return target
    return DEFAULT_VIEW


def footer_line(ctx: DashboardContext, dropped: Sequence[str], view: str) -> str:
    tabs = "   ".join(
        (role_text(f"[{i}] {name.upper()}", ROLE_SECTION) if name == view
         else color_text(f"[{i}] {name}", "dim"))
        for i, name in enumerate(VIEWS, start=1)
    )
    clock = time.strftime("%H:%M:%S", time.localtime(ctx.now_wall))
    parts = [tabs]
    if dropped:
        parts.append(color_text("dropped: " + ", ".join(dict.fromkeys(dropped)), "dim"))
    parts.append(color_text(clock, "dim"))
    return _ansi_truncate("──── " + "   ".join(parts), max(40, ctx.term_w - 2))


def render_screen(ctx: DashboardContext) -> str:
    """Full frame as one string, guaranteed to fit `ctx.term_h - 1` lines."""
    spine_budget, body_budget = frame_tiers(ctx.term_h)
    spine = spine_lines(ctx, spine_budget)
    if body_budget <= 0:
        return "\n".join(spine)
    view = resolve_view(ctx, ctx.view)
    rows = VIEWS[view](ctx)
    allocated, dropped = allocate_rows(rows, body_budget)
    body = compose_rows(allocated, ctx.term_w)
    return "\n".join((*spine, *body, footer_line(ctx, dropped, view)))


__all__ = ["ANHARMONICITY_PROMOTE", "DEFAULT_VIEW", "VIEWS", "footer_line",
           "promotion_reasons", "render_screen", "resolve_view"]
```

```python
# gareus/dashboard/__init__.py
"""Live-dashboard screen: context, panels, views, frame assembly."""

from .context import DashboardContext, build_context
from .screen import render_screen
from .sidecar import SidecarCache, SidecarSnapshot

__all__ = [
    "DashboardContext", "SidecarCache", "SidecarSnapshot", "build_context", "render_screen",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_screen.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add gareus/dashboard/screen.py gareus/dashboard/__init__.py tests/test_dashboard_screen.py
git commit -m "feat(dashboard): assemble spine, view and footer into a fitting frame"
```

---

## Task 12: Wire the screen into DistanceLogger and remove the old frame

**Files:**
- Modify: `gareus/logger.py` (`__init__` ~`:219`; delete `_render_dashboard` `:2398-2652` and the ported/obsolete renderers; add `_render_screen_frame`)
- Modify: `tests/test_dashboard_scaling_integration.py:40-49` and `:17`
- Test: `tests/test_dashboard_screen.py` (append integration checks)

**Interfaces:**
- Consumes: Task 11's `render_screen`; Task 5's `build_context`; Task 3's `SidecarCache`.
- Produces: `DistanceLogger._render_screen_frame(rows, phase, step, total_steps, summary, dashboard_info) -> str` — same call signature as the deleted `_render_dashboard`, so `log()`'s threading block is unchanged.

**Delete** (all now live in `gareus/dashboard/`): `_render_dashboard`, `_render_compact_header`,
`_render_dashboard_context_panel`, `_render_dashboard_decision_panel`,
`_render_compact_decision_inline`, `_render_overlap`, `_render_gamd_boost`,
`_render_potential_energy_map`, `_render_exchange_acceptance`, `_render_pull_map`,
`_render_replica_table`.

**Keep** (not ported in Phase 1; deleting them would lose content with no replacement):
`_dashboard_decision_state` (feeds `ctx.decision`), `_render_pmf_preview`,
`_render_replica_diffusion`, `_render_sparklines`, `_render_health`,
`_render_recommendations`, `_render_2d_diffusion_map`, `_render_2d_replica_map`,
`_render_2d_sparse_topology_map`, `_unique_axis_values`, `_nearest_index`,
`_primary_axis_bounds_for_2d_map`, `_primary_axis_tick_label`, `_acb_y_label`.

`gareus.tui.dashboard_body_budget` becomes unused by the frame path. Keep it and its four
tests: it is still correct, and removing a tested public helper is a separate decision.

`dashboard_heavy_panels_every` and the `render_heavy_panels` flag `log()` computes from it
(`gareus/logger.py:2739`) also go inert — nothing in the new frame reads that flag, since the
allocator drops panels by priority and budget instead of by frame parity, exactly as spec §15
requires. Leave both in place for now: the only panels they ever gated are the three 2D
renderers that are still stranded on `DistanceLogger`, so knob and panels are removed
together in Phase 2 rather than half here.

- [ ] **Step 1: Write the failing integration test**

```python
# append to tests/test_dashboard_screen.py
import collections
import os
import shutil


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_dashboard_screen.py -k logger -v`
Expected: FAIL with `AttributeError: 'DistanceLogger' object has no attribute '_render_screen_frame'`

- [ ] **Step 3: Wire it up**

In `DistanceLogger.__init__`, after `self._dashboard_render_count = 0`:

```python
        self._sidecar = SidecarCache(self.out_dir)
        self.tui_view = str(getattr(args, "tui_view", "auto") or "auto").lower()
        self.tui_glyphs = _resolve_glyphs(str(getattr(args, "tui_glyphs", "auto") or "auto"))
```

Add `import sys` to `logger.py`'s stdlib imports (`gareus/logger.py:13-20` currently has
`collections`, `concurrent.futures`, `csv`, `math`, `shutil`, `time` — **not** `sys`), then
add near the module's other helpers:

```python
def _resolve_glyphs(mode: str) -> str:
    """Box-drawing/block glyphs, or an ASCII ramp for terminals that mangle them."""
    if mode in {"unicode", "ascii"}:
        return mode
    encoding = str(getattr(sys.stdout, "encoding", "") or "").lower()
    return "unicode" if "utf" in encoding else "ascii"
```

Replace the whole of `_render_dashboard` with:

```python
    def _render_screen_frame(
        self, rows: list[dict], phase: str, step: int, total_steps: Optional[int],
        summary: dict, dashboard_info: Optional[dict],
    ) -> str:
        """Build one dashboard frame through the screen engine.

        Same signature as the removed `_render_dashboard` so `log()`'s render-thread
        offload is untouched.
        """
        term_size = shutil.get_terminal_size((160, 40))
        now = time.time()
        ctx = build_context(
            logger=self, rows=rows, phase=phase, step=step, total_steps=total_steps,
            summary=summary, dashboard_info=dashboard_info,
            sidecar=self._sidecar.snapshot(now),
            term_w=int(term_size.columns or 160), term_h=int(term_size.lines or 40),
            now=now, view=self.tui_view, glyphs=self.tui_glyphs,
        )
        return render_screen(ctx)
```

Add the imports at the top of `logger.py` (after the existing `from .tui import ...`):

```python
from .dashboard.context import build_context
from .dashboard.screen import render_screen
from .dashboard.sidecar import SidecarCache
```

In `log()`'s `_do_render` closure (`gareus/logger.py:2752-2756`), change the one call:

```python
                        block = self._render_screen_frame(rows_s, phase_s, step_s, total_s,
                                                          summary_s, info_s)
```

Then delete the methods listed under **Delete** above.

- [ ] **Step 4: Update the two frame-shape assertions in the existing integration test**

The frame layout changed by design, so two of the six integration tests need their target
updated. Nothing about their intent changes — they still pin bar widths.

In `tests/test_dashboard_scaling_integration.py`, make `_render_at` request the view that
contains the CV/PE maps, and rename the coverage-bar prefix the spine now uses:

```python
# was: _COV_BAR_RE = re.compile(r"^cov .*?\|([ ░▒▓█·]+)\|", re.MULTILINE)
_COV_BAR_RE = re.compile(r"^(?:cov|cv1) .*?\|([ ░▒▓█·]+)\|", re.MULTILINE)


def _render_at(monkeypatch, logger, term_w, term_h):
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback=None: os.terminal_size((term_w, term_h))
    )
    logger.tui_view = "windows"          # the view that owns the CV and PE map panels
    return logger._render_screen_frame(
        _sample_rows(), phase="production", step=100, total_steps=1000,
        summary={}, dashboard_info={"centers_a": [0.0, 1.0, 2.0, 3.0], "n_windows": 4},
    )
```

- [ ] **Step 5: Run the full dashboard suite**

Run: `python -m pytest tests/test_dashboard_tui_scaling.py tests/test_dashboard_scaling_integration.py tests/test_dashboard_screen.py -v`
Expected: all pass (20 pre-existing + 10 new)

- [ ] **Step 6: Point the evidence harness at the new entrypoint**

`tests/manual_render_dashboard_frame.py` calls `logger._render_dashboard`, which this task
deletes, so it must be updated in the same commit or it breaks. In its `render_frame`, set
the view and call the new method:

```python
    logger.tui_view = "windows"          # densest view, closest to the old single frame
    _force_terminal_size(term_w, term_h)
    return logger._render_screen_frame(
        rows, "gareus_production", info["display_step"],
        info["display_total_steps"], logger.summarize(rows), info,
    )
```

Then run it: `python tests/manual_render_dashboard_frame.py`
Expected: `dropped` is `0` and `loss` is `0%` for all three cases — the whole point of the
change. Before this rework the same harness reported 27/33%, 47/52% and 43/44%.

- [ ] **Step 7: Commit**

```bash
git add gareus/logger.py tests/test_dashboard_scaling_integration.py \
        tests/test_dashboard_screen.py tests/manual_render_dashboard_frame.py
git commit -m "refactor(dashboard): render the live frame through the screen engine"
```

---

## Task 13: Promote the hardcoded TUI knobs to real flags

**Files:**
- Modify: `gareus/cli.py` — `_add_output_args` (`:514-533`), `_shim_output` (`:1053-1072`)
- Test: `tests/test_dashboard_cli_flags.py`

**Interfaces:**
- Consumes: nothing.
- Produces: seven argparse flags — `--tui-view`, `--tui-glyphs`, `--color`, `--tui-clear-mode`, `--dashboard-density`, `--distance-ascii-max-replicas`, `--dashboard-render-interval-sec` — whose defaults reproduce today's hardcoded values exactly.

Two shipped strings currently name flags that do not exist: `render_distance_ascii` prints
"raise `--distance-ascii-max-replicas`" when it omits replicas, and `tui_clear_enabled`'s
docstring (`gareus/tui.py:53`) cites `--tui-clear-mode never` while `_shim_output` pins it
to `"always"`, making the `never` branch unreachable. This task makes both real.

**The clobber trap.** `_shim_output` assigns these names unconditionally, so adding a flag
without editing the shim would leave the flag silently overridden — the same failure mode
CLAUDE.md records for `--sambar-*`. The seven promoted assignments must be *deleted* from
`_shim_output`, not guarded, since argparse now always supplies them.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dashboard_cli_flags.py
import pytest

from gareus import cli

# Verified against this checkout: `cli.parse_args(["--seq", "AAAA", "--out", "out"])`
# succeeds and yields tui_mode='dashboard', color='auto', dashboard_density='auto',
# distance_ascii_max_replicas=32 -- i.e. the shim's values, pre-change.
BASE = ["--seq", "AAAA", "--out", "out"]      # minimal accepted invocation


def _parse(extra):
    return cli.parse_args(BASE + extra)


@pytest.mark.parametrize("attr,default", [
    ("tui_view", "auto"),
    ("tui_glyphs", "auto"),
    ("color", "auto"),
    ("tui_clear_mode", "always"),
    ("dashboard_density", "auto"),
    ("distance_ascii_max_replicas", 32),
    ("dashboard_render_interval_sec", 0.0),
])
def test_promoted_flag_defaults_match_the_previously_hardcoded_values(attr, default):
    assert getattr(_parse([]), attr) == default


@pytest.mark.parametrize("flag,attr,value,expected", [
    ("--tui-view", "tui_view", "physics", "physics"),
    ("--tui-glyphs", "tui_glyphs", "ascii", "ascii"),
    ("--color", "color", "never", "never"),
    ("--tui-clear-mode", "tui_clear_mode", "never", "never"),
    ("--dashboard-density", "dashboard_density", "compact", "compact"),
    ("--distance-ascii-max-replicas", "distance_ascii_max_replicas", "64", 64),
    ("--dashboard-render-interval-sec", "dashboard_render_interval_sec", "2.5", 2.5),
])
def test_promoted_flag_survives_the_compat_shims(flag, attr, value, expected):
    """A flag the shim used to overwrite must reach the parsed namespace intact."""
    assert getattr(_parse([flag, value]), attr) == expected


def test_shim_no_longer_assigns_the_promoted_names():
    import inspect
    source = inspect.getsource(cli._shim_output)
    for name in ("args.color", "args.tui_clear_mode", "args.dashboard_density",
                 "args.distance_ascii_max_replicas", "args.dashboard_render_interval_sec"):
        assert name not in source, f"{name} is still clobbered by _shim_output"


def test_shim_still_assigns_the_knobs_that_stayed_internal():
    import inspect
    source = inspect.getsource(cli._shim_output)
    assert "args.distance_history_limit" in source
    assert "args.dashboard_min_panel_width" in source


def test_repeated_parse_args_does_not_drift_a_promoted_default():
    """Guards the sticky-default pattern recorded for --sambar-* in CLAUDE.md."""
    assert _parse(["--distance-ascii-max-replicas", "64"]).distance_ascii_max_replicas == 64
    assert _parse([]).distance_ascii_max_replicas == 32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dashboard_cli_flags.py -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'tui_view'`

- [ ] **Step 3: Add the flags**

In `gareus/cli.py`'s `_add_output_args`, after the existing `--tui-mode` argument:

```python
    p.add_argument("--tui-view", choices=["auto", "progress", "physics", "windows"],
                   default="auto",
                   help="Which dashboard view to show. 'auto' shows PROGRESS and switches "
                        "to another view only while a problem is present (dead exchange "
                        "pair, disconnected state graph, high boost anharmonicity).")
    p.add_argument("--tui-glyphs", choices=["auto", "unicode", "ascii"], default="auto",
                   help="Glyph set for density strips and panel borders. 'auto' picks "
                        "unicode on a UTF-8 stdout, ascii otherwise.")
    p.add_argument("--color", choices=["auto", "always", "never"], default="auto",
                   help="ANSI colour output. 'auto' enables it when stdout is a TTY.")
    p.add_argument("--tui-clear-mode", choices=["always", "never"], default="always",
                   help="Whether full-frame TUI redraws clear the visible terminal. "
                        "'never' appends frames instead (log-style).")
    p.add_argument("--dashboard-density", choices=["auto", "compact", "normal", "full"],
                   default="auto",
                   help="Panel density hint. 'auto' derives it from terminal size.")
    p.add_argument("--distance-ascii-max-replicas", type=int, default=32,
                   help="Maximum replica rows drawn in the per-window CV distribution "
                        "panel (default 32).")
    p.add_argument("--dashboard-render-interval-sec", type=float, default=0.0,
                   help="Minimum wall seconds between dashboard repaints (0 = every "
                        "logging step).")
```

In `_shim_output`, delete exactly these five lines (the two new names were never there):

```python
    args.color = "auto"
    args.tui_clear_mode = "always"
    args.dashboard_density = "auto"
    args.dashboard_render_interval_sec = 0.0
    args.distance_ascii_max_replicas = 32
```

Leave every other assignment in `_shim_output` untouched — `progress_jsonl`,
`progress_update_interval_sec`, `progress_bar_width`, `dashboard_wide_threshold`,
`dashboard_min_panel_width`, `dashboard_max_height`, `dashboard_panels`,
`dashboard_heavy_panels_every`, `distance_csv`, `distance_jsonl`,
`no_distance_gui_events`, `distance_ascii_mode`, `distance_ascii_width`,
`distance_history_limit`, and the rest stay internal in Phase 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dashboard_cli_flags.py -v`
Expected: 17 passed

- [ ] **Step 5: Smoke-test the parser and the help text**

Run: `python -m gareus --help | grep -E "tui-view|tui-glyphs|color|tui-clear-mode|dashboard-density|distance-ascii-max-replicas|dashboard-render-interval-sec"`
Expected: all seven flags listed.

- [ ] **Step 6: Commit**

```bash
git add gareus/cli.py tests/test_dashboard_cli_flags.py
git commit -m "feat(cli): promote seven hardcoded TUI knobs to real flags"
```

---

## Task 14: Frame-fit property matrix and golden frames

**Files:**
- Create: `tests/test_dashboard_frame_fit.py`
- Create: `tests/golden/dashboard/` (three `.txt` snapshots, generated in Step 3)
- Modify: `tests/manual_render_dashboard_frame.py` (report the post-fix zero-loss result)

**Interfaces:**
- Consumes: everything above.
- Produces: the invariant no test asserts today — for every terminal size and every view, the rendered frame fits.

- [ ] **Step 1: Write the failing property test**

```python
# tests/test_dashboard_frame_fit.py
"""The invariant the old dashboard violated: a frame must fit its terminal.

Measured before this rework, `_render_dashboard` emitted 81-97 lines for a 44-54
line terminal (see tests/manual_render_dashboard_frame.py). Nothing asserted the
fit, so the loss was invisible.
"""

import argparse
import os

import pytest

from gareus.dashboard.context import build_context
from gareus.dashboard.screen import render_screen
from gareus.dashboard.sidecar import SidecarSnapshot
from gareus.logger import DistanceLogger
from gareus.tui import strip_ansi, strip_ansi_len

WIDTHS = (70, 80, 100, 120, 140, 200, 400)
HEIGHTS = (18, 20, 24, 35, 45, 55, 80)
VIEWS = ("progress", "physics", "windows")
N_WINDOWS = 25

POOL = {"total_ns": 15000.0, "used_ns": 7500.0, "remaining_ns": 7500.0,
        "events": [{"label": f"seg_{i}", "kind": "scheduled_epoch",
                    "consumed_ns": 100.0 * (i + 1), "n_states": 29} for i in range(12)]}
GAMD = {"joint_envelope": {"Dihedral": {"sigma0_kj_mol": 12.552,
                                        "sigmaV_kj_mol": 11.039, "k0": 1.0}}}


def _ctx(tmp_path, term_w, term_h, view, *, is_2d=False, n=N_WINDOWS, rich=True):
    centers = [4.0 + 0.55 * i for i in range(n)]
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=4.0),
                            no_file_persistence=True)
    if rich:
        for w in range(n):
            logger.history_by_window[w] = [centers[w] + 0.1 * (i % 5 - 2) for i in range(60)]
            logger.potential_history_by_replica[w] = [-31500.0 + i for i in range(40)]
            logger.secondary_history_by_window[w] = [0.1 * (i % 7) - 0.3 for i in range(40)]
        logger.boost_history_all = [2.0 + 0.5 * (i % 7) for i in range(300)]
    rows = [{"replica": w, "window": w, "center_A": centers[w], "k_kcal_mol_A2": 2.5,
             "cv_A": centers[w] + 0.05, "umbrella_bias_kcal_mol": 0.01,
             "umbrella_pull_kcal_mol_A": 0.1, "potential_kj_mol": -31500.0,
             "gamd_boost_total_kcal_mol": 2.4} for w in range(n)]
    info = {"centers_a": centers, "n_windows": n, "k_list": [2.5] * n,
            "exchange_stats": {f"{i}-{i+1}": {"attempts": 40, "accepted": 12}
                               for i in range(n - 1)},
            "primary_cv_label": "nonlocal contacts", "primary_cv_units": "A",
            "primary_k_units": "kcal/mol/A^2",
            "adaptive_phase": {"epoch_index": 1, "epoch_total": 2,
                               "segment_name": "epoch_001/topup_002"}}
    if is_2d:
        targets = [-2.0, -1.0, 0.0, 1.0, 2.0]
        info["secondary_cv_centers"] = [targets[i % len(targets)] for i in range(n)]
        info["secondary_cv"] = {"explicit_2d_windows": True, "grid": False,
                                "type": "tica-linear"}
    return build_context(
        logger=logger, rows=rows, phase="gareus_production", step=12_450_000,
        total_steps=37_500_000, summary={}, dashboard_info=info,
        sidecar=SidecarSnapshot(pool=POOL, gamd=GAMD), term_w=term_w, term_h=term_h,
        now=1_700_000_000.0, view=view, glyphs="unicode",
    )


@pytest.mark.parametrize("term_w", WIDTHS)
@pytest.mark.parametrize("term_h", HEIGHTS)
@pytest.mark.parametrize("view", VIEWS)
def test_frame_fits_every_terminal_size_and_view(tmp_path, term_w, term_h, view):
    lines = strip_ansi(render_screen(_ctx(tmp_path, term_w, term_h, view))).splitlines()
    assert len(lines) <= term_h - 1, f"{len(lines)} lines in a {term_h}-row terminal"
    for line in lines:
        assert strip_ansi_len(line) <= term_w - 2, f"line wider than {term_w - 2}: {line!r}"


@pytest.mark.parametrize("view", VIEWS)
def test_frame_fits_for_a_2d_run(tmp_path, view):
    lines = strip_ansi(render_screen(_ctx(tmp_path, 140, 45, view, is_2d=True))).splitlines()
    assert len(lines) <= 44
    assert all(strip_ansi_len(l) <= 138 for l in lines)


@pytest.mark.parametrize("view", VIEWS)
def test_frame_fits_with_364_windows(tmp_path, view):
    lines = strip_ansi(render_screen(_ctx(tmp_path, 140, 45, view, n=364))).splitlines()
    assert len(lines) <= 44
    assert all(strip_ansi_len(l) <= 138 for l in lines)


@pytest.mark.parametrize("view", VIEWS)
def test_frame_renders_before_any_samples_exist(tmp_path, view):
    frame = render_screen(_ctx(tmp_path, 140, 45, view, rich=False))
    assert frame.strip()


def test_every_status_word_survives_ansi_stripping(tmp_path):
    """Colour must never be the only carrier of severity."""
    text = strip_ansi(render_screen(_ctx(tmp_path, 200, 55, "physics")))
    assert "SATURATED" in text          # k0 at its ceiling
    assert "ok" in text or "OK" in text


@pytest.mark.parametrize("view", VIEWS)
def test_golden_frame_matches_the_committed_snapshot(view):
    """Layout drift must be a deliberate, reviewed diff."""
    import pathlib
    import tempfile
    golden = pathlib.Path(__file__).parent / "golden" / "dashboard" / f"{view}_140x45.txt"
    with tempfile.TemporaryDirectory() as tmp:
        actual = strip_ansi(render_screen(_ctx(pathlib.Path(tmp), 140, 45, view)))
    if os.environ.get("GAREUS_UPDATE_GOLDEN") == "1":
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(actual)
    assert golden.is_file(), f"missing golden file; regenerate with GAREUS_UPDATE_GOLDEN=1"
    assert actual == golden.read_text()
```

- [ ] **Step 2: Run to verify the fit tests pass and the golden tests fail**

Run: `python -m pytest tests/test_dashboard_frame_fit.py -v`
Expected: the 147 parametrised fit cases and the four robustness cases PASS; the three
golden cases FAIL with "missing golden file". If any *fit* case fails, that is a real defect
in Tasks 1-12 — fix it there, do not relax the assertion.

- [ ] **Step 3: Generate and inspect the golden frames**

```bash
GAREUS_UPDATE_GOLDEN=1 python -m pytest tests/test_dashboard_frame_fit.py -k golden -q
head -30 tests/golden/dashboard/windows_140x45.txt
```

Read all three files before committing them. They are the reviewable artefact of this whole
plan: check the spine reads as designed, the panels are the intended ones, and no line is
truncated mid-word where a width calculation is off by a few columns.

- [ ] **Step 4: Confirm the original defect is gone**

Run: `python tests/manual_render_dashboard_frame.py`
Expected: `dropped 0`, `loss 0%` on all three cases (was 27/33%, 47/52%, 43/44%).

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: no new failures. `tests/test_validation_common.py` fails on a clean checkout for
an unrelated pre-existing reason (`ModuleNotFoundError: No module named 'common'`) — that
one is expected, per CLAUDE.md.

- [ ] **Step 6: Commit**

```bash
git add tests/test_dashboard_frame_fit.py tests/golden/dashboard tests/manual_render_dashboard_frame.py
git commit -m "test(dashboard): assert frames fit every terminal size, add golden frames"
```

---

## Out of scope for this plan (Phase 2)

Deliberately excluded so Phase 1 is a working, reviewable replacement. Each needs its own
plan:

1. **New accumulators** — per-window cumulative sample counts (surviving resume via
   `restore_from_previous_outputs`), per-window σΔV and anharmonicity, ns/day history,
   exchange-stall timing. These unlock the `samples`, `σΔV` and `throughput` sparkline
   columns the spec's mockups show; Phase 1 omits those columns rather than faking them.
2. **Interactive keys** — termios raw mode for `1`/`2`/`3`/`Tab`/`j`/`k`, reusing the
   pattern at `gareus_monitor.py:4490-4542`, with terminal-state restore on every exit path
   including SIGTERM. This is what finally makes `--tui-mode interactive` distinct from
   `dashboard`.
3. **Hysteresis wired into the live path** — `ranking.Hysteresis` is implemented and tested
   in Task 4 but not yet applied, because it needs per-frame state that belongs with the
   Phase 2 accumulators.
4. **The remaining 2D panels** — `_render_2d_diffusion_map`, `_render_2d_replica_map`,
   `_render_2d_sparse_topology_map` still live on `DistanceLogger` and are unreachable from
   the new frame. Either port them into a 2D view or delete them; leaving them stranded is
   not an acceptable end state.
5. **`render_distance_ascii` relocation** out of `gareus/tui.py`, which is what makes that
   module stdlib-only (spec §12).
6. **Unifying `gareus_monitor.py`** onto this engine (spec §3 non-goal, still worth doing).
7. **The `reading` and `options` advisory lines** in the spec's §7.3 window-detail mockup.
   They are rule-derived text keyed to the severity rule that fired, and the rules that would
   drive most of them (per-window σΔV, starved-sample counts) are Phase 2 items 1. Phase 1's
   detail panel shows measurements only — no inferred verdict — rather than inferring from
   rules it cannot yet evaluate.
8. **Removing `dashboard_heavy_panels_every`** together with the three stranded 2D panels it
   gated (see Task 12).
