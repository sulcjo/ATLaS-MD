# Live Dashboard TUI Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace gareus's live run-facing dashboard's hardcoded panel-width/body-line ceilings with continuous terminal-size-driven scaling, and make its color/spacing choices consistent, without changing what data any panel shows.

**Architecture:** All changes are in three existing files — `gareus/tui.py` (layout primitives + new scaling/color-role helpers), `gareus/logger.py` (the `DistanceLogger._render_dashboard` method that assembles the frame), `gareus/colors.py` (semantic color roles). No new files, no new dependency. Each new helper is a pure function (terminal dims / weights in, ints/strings out) so it can be unit-tested without constructing a full `DistanceLogger`.

**Tech Stack:** Python 3, stdlib only (`shutil`, `math`), `pytest`.

## Global Constraints

- No Rich/Textual/blessed — keep the hand-rolled ANSI architecture (spec non-goal).
- No change to `RUNS/gareus_monitor.py` or `gareus/helptext.py`.
- No change to what data any panel shows, CLI flag semantics, or `dashboard_render_interval_sec` throttling.
- Color role migration scope is limited to panel/section title coloring and the CV-delta severity coloring — not all 184 `color_text`/`style_text` call sites.
- `DistanceLogger` is constructed in tests as `DistanceLogger(tmp_path, argparse.Namespace(), no_file_persistence=True)` — see `tests/test_store.py:414-426` for the existing pattern. `argparse.Namespace()` works because every config value is read via `getattr(args, name, default)`.
- Terminal size is read via `shutil.get_terminal_size(...)` — both `gareus/tui.py` and `gareus/logger.py` do `import shutil` (not `from shutil import ...`), so a single `monkeypatch.setattr(shutil, "get_terminal_size", lambda fallback=None: os.terminal_size((cols, lines)))` (patching the shared `shutil` module object) affects both modules in tests.
- Existing frame-level safety net: `gareus/tui.py`'s `_safe_tui_frame_text` (lines 74-102) already truncates any frame taller than the terminal with a "truncated" message. New body-line budgets do not need to sum exactly to `term_h` — this net already prevents real overflow/corruption.

---

### Task 1: `dashboard_body_budget()` — continuous body-line budget helper

**Files:**
- Modify: `gareus/tui.py` (add function near `_dashboard_density`, ~line 342)
- Test: `tests/test_dashboard_tui_scaling.py` (new file)

**Interfaces:**
- Produces: `dashboard_body_budget(term_h: int, fraction: float, floor: int = 10) -> int` — importable from `gareus.tui`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dashboard_tui_scaling.py`:

```python
from gareus.tui import dashboard_body_budget


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_tui_scaling.py -v`
Expected: FAIL with `ImportError: cannot import name 'dashboard_body_budget'`

- [ ] **Step 3: Implement**

In `gareus/tui.py`, add after `_dashboard_density` (after line 341):

```python
def dashboard_body_budget(term_h: int, fraction: float, floor: int = 10) -> int:
    """Proportional body-line budget for one dashboard row/panel.

    Scales continuously with terminal height instead of a fixed ceiling tied
    to a coarse density bucket, so a tall terminal shows more content rather
    than the same capped amount plus blank margin. `_safe_tui_frame_text`
    already truncates a frame that ends up taller than the terminal, so this
    does not need to sum exactly across rows to stay safe.
    """
    return max(int(floor), int(round(float(term_h) * float(fraction))))
```

Add `"dashboard_body_budget"` to the `__all__` list in `gareus/tui.py` (near line 32, alongside `"render_distance_ascii"`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_tui_scaling.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add gareus/tui.py tests/test_dashboard_tui_scaling.py
git commit -m "feat(tui): add continuous body-line budget helper"
```

---

### Task 2: `_weighted_panel_widths()` — standalone proportional-width computation

**Files:**
- Modify: `gareus/tui.py` (extract width math out of `_dashboard_weighted_row`, lines 257-312)
- Test: `tests/test_dashboard_tui_scaling.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `_weighted_panel_widths(weights: list[float], term_w: Optional[int] = None, gap: int = 2, min_panel_width: int = 30) -> list[int]` — importable from `gareus.tui`. `_dashboard_weighted_row` is refactored to call this internally instead of duplicating the width math, so its observable behavior for existing callers is unchanged (verified by Step 1's regression test).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard_tui_scaling.py`:

```python
from gareus.tui import _weighted_panel_widths, _dashboard_weighted_row


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_tui_scaling.py -v`
Expected: FAIL with `ImportError: cannot import name '_weighted_panel_widths'`

- [ ] **Step 3: Implement**

In `gareus/tui.py`, replace the body of `_dashboard_weighted_row` (lines 257-312) with an extracted helper plus a thinner caller:

```python
def _weighted_panel_widths(
    weights: List[float],
    term_w: Optional[int] = None,
    gap: int = 2,
    min_panel_width: int = 30,
) -> List[int]:
    """Compute proportional panel widths for a weighted dashboard row.

    Exposed standalone (not just inline in `_dashboard_weighted_row`) so a
    caller can pre-render panel content — e.g. an ASCII histogram bar — at
    the exact width the row will allocate, instead of guessing a width up
    front and letting the row silently pad or truncate the mismatch away.
    """
    term_w_val = int(term_w or shutil.get_terminal_size((160, 40)).columns or 160)
    usable = max(40, term_w_val - 2)
    clean_weights = [max(0.05, float(w)) for w in weights]
    n = len(clean_weights)
    if n <= 0:
        return []
    gap_total = max(0, n - 1) * gap
    min_w = max(22, int(min_panel_width))
    if n > 1 and usable < n * min_w + gap_total:
        return [usable] * n
    available = max(n * min_w, usable - gap_total)
    total_weight = sum(clean_weights) or 1.0
    raw_widths = [max(min_w, int(round(available * w / total_weight))) for w in clean_weights]
    delta = available - sum(raw_widths)
    order = sorted(range(n), key=lambda i: raw_widths[i], reverse=True)
    idx = 0
    while delta != 0 and order:
        i = order[idx % len(order)]
        if delta > 0:
            raw_widths[i] += 1
            delta -= 1
        elif raw_widths[i] > min_w:
            raw_widths[i] -= 1
            delta += 1
        idx += 1
        if idx > 10000:
            break
    return raw_widths


def _dashboard_weighted_row(
    title: str,
    panels: List[tuple[str, List[str] | tuple]],
    term_w: Optional[int] = None,
    gap: int = 2,
    max_panel_body_lines: Optional[int] = None,
    min_panel_width: int = 30,
) -> List[str]:
    """A themed row with proportional panel widths and safe narrow stacking.

    Each panel may be ``(name, body)`` or ``(name, body, weight)``.  Weighted
    rows make wide terminals useful instead of giving every diagnostic
    the same cramped column even when one panel is visually dominant.
    """
    clean: List[tuple[str, List[str], float]] = []
    for item in panels:
        if len(item) >= 3:
            name, body, weight = item[0], item[1], float(item[2])
        else:
            name, body, weight = item[0], item[1], 1.0
        clean.append((str(name), list(body), max(0.05, weight)))
    n = len(clean)
    if n <= 0:
        return [color_text(title, "magenta", bold=True)]
    widths = _weighted_panel_widths(
        [w for _n, _b, w in clean], term_w=term_w, gap=gap, min_panel_width=min_panel_width
    )
    term_w_val = int(term_w or shutil.get_terminal_size((160, 40)).columns or 160)
    usable = max(40, term_w_val - 2)
    min_w = max(22, int(min_panel_width))
    gap_total = max(0, n - 1) * gap
    if n > 1 and usable < n * min_w + gap_total:
        out: List[str] = [color_text(title, "magenta", bold=True)]
        for name, body, _weight in clean:
            out.extend(_panel_lines(name, body, usable, max_panel_body_lines))
        return out
    cols = [
        _panel_lines(name, body, width, max_panel_body_lines)
        for (name, body, _w), width in zip(clean, widths)
    ]
    return [color_text(title, "magenta", bold=True)] + _join_columns(cols, gap=gap)
```

Add `"_weighted_panel_widths"` to `__all__`'s internal-helpers section (near `"replica_marker"`, `"replica_fg256"`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_tui_scaling.py -v`
Expected: PASS (7 tests total: 3 from Task 1 + 4 new, plus the regression test — 8 total)

- [ ] **Step 5: Commit**

```bash
git add gareus/tui.py tests/test_dashboard_tui_scaling.py
git commit -m "refactor(tui): extract weighted-panel-width math into standalone helper"
```

---

### Task 3: Unify min-panel-width and row-gap constants

**Files:**
- Modify: `gareus/tui.py` — `_dashboard_row` (line 246), `_dashboard_weighted_row` (default param, now via Task 2's version), new `dashboard_row_gap()` function
- Test: `tests/test_dashboard_tui_scaling.py`

**Interfaces:**
- Produces: `MIN_PANEL_WIDTH: int = 30`, `ABSOLUTE_MIN_PANEL_WIDTH: int = 22` (module-level constants in `gareus.tui`), `dashboard_row_gap(term_w: int) -> int` — all importable from `gareus.tui`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard_tui_scaling.py`:

```python
from gareus.tui import MIN_PANEL_WIDTH, ABSOLUTE_MIN_PANEL_WIDTH, dashboard_row_gap


def test_min_panel_width_constants():
    assert MIN_PANEL_WIDTH == 30
    assert ABSOLUTE_MIN_PANEL_WIDTH == 22
    assert ABSOLUTE_MIN_PANEL_WIDTH < MIN_PANEL_WIDTH


def test_dashboard_row_gap_narrow_vs_wide():
    assert dashboard_row_gap(100) == 2
    assert dashboard_row_gap(119) == 2
    assert dashboard_row_gap(120) == 3
    assert dashboard_row_gap(300) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_tui_scaling.py -v`
Expected: FAIL with `ImportError: cannot import name 'MIN_PANEL_WIDTH'`

- [ ] **Step 3: Implement**

In `gareus/tui.py`, add near the top (after the `__all__` list, before `tui_clear_enabled`):

```python
MIN_PANEL_WIDTH = 30
ABSOLUTE_MIN_PANEL_WIDTH = 22


def dashboard_row_gap(term_w: int) -> int:
    """Inter-panel gap for a dashboard row, shared by every row layout call."""
    return 2 if int(term_w) < 120 else 3
```

Add `"MIN_PANEL_WIDTH"`, `"ABSOLUTE_MIN_PANEL_WIDTH"`, `"dashboard_row_gap"` to `__all__`.

In `_dashboard_row` (line 246), change:

```python
    min_panel_w = 28
```

to:

```python
    min_panel_w = MIN_PANEL_WIDTH
```

In `_dashboard_weighted_row`'s signature (from Task 2), change the default:

```python
def _dashboard_weighted_row(
    title: str,
    panels: List[tuple[str, List[str] | tuple]],
    term_w: Optional[int] = None,
    gap: int = 2,
    max_panel_body_lines: Optional[int] = None,
    min_panel_width: int = 30,
) -> List[str]:
```

to:

```python
def _dashboard_weighted_row(
    title: str,
    panels: List[tuple[str, List[str] | tuple]],
    term_w: Optional[int] = None,
    gap: int = 2,
    max_panel_body_lines: Optional[int] = None,
    min_panel_width: int = MIN_PANEL_WIDTH,
) -> List[str]:
```

and inside both `_weighted_panel_widths` and `_dashboard_weighted_row`, change every `max(22, int(min_panel_width))` to `max(ABSOLUTE_MIN_PANEL_WIDTH, int(min_panel_width))`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_tui_scaling.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add gareus/tui.py tests/test_dashboard_tui_scaling.py
git commit -m "refactor(tui): unify min-panel-width and row-gap constants"
```

---

### Task 4: Semantic color roles + `_severity_color` in `colors.py`

**Files:**
- Modify: `gareus/colors.py`
- Test: `tests/test_colors_roles.py` (new file)

**Interfaces:**
- Produces: `ROLE_TITLE`, `ROLE_SECTION`, `ROLE_GOOD`, `ROLE_WARN`, `ROLE_BAD`, `ROLE_MUTED`, `ROLE_VALUE` (string constants), `role_text(text: object, role: str, bold: Optional[bool] = None) -> str`, `severity_role(value: float, warn_at: float, bad_at: float) -> str` — all importable from `gareus.colors`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_colors_roles.py`:

```python
import gareus.colors as colors_mod
from gareus.colors import (
    role_text,
    severity_role,
    ROLE_TITLE,
    ROLE_SECTION,
    ROLE_GOOD,
    ROLE_WARN,
    ROLE_BAD,
    ROLE_MUTED,
    ROLE_VALUE,
)


def test_role_text_disabled_returns_plain(monkeypatch):
    monkeypatch.setattr(colors_mod, "ASCII_COLOR_ENABLED", False)
    assert role_text("hello", ROLE_TITLE) == "hello"


def test_role_text_enabled_applies_style(monkeypatch):
    monkeypatch.setattr(colors_mod, "ASCII_COLOR_ENABLED", True)
    out = role_text("hello", ROLE_TITLE)
    assert out != "hello"
    assert "hello" in out


def test_role_text_unknown_role_falls_back_to_muted(monkeypatch):
    monkeypatch.setattr(colors_mod, "ASCII_COLOR_ENABLED", True)
    out = role_text("x", "not-a-real-role")
    fallback = role_text("x", ROLE_MUTED)
    assert out == fallback


def test_severity_role_thresholds():
    assert severity_role(0.0, warn_at=0.5, bad_at=1.5) == ROLE_GOOD
    assert severity_role(0.5, warn_at=0.5, bad_at=1.5) == ROLE_GOOD
    assert severity_role(0.51, warn_at=0.5, bad_at=1.5) == ROLE_WARN
    assert severity_role(1.5, warn_at=0.5, bad_at=1.5) == ROLE_WARN
    assert severity_role(1.51, warn_at=0.5, bad_at=1.5) == ROLE_BAD
    assert severity_role(-0.51, warn_at=0.5, bad_at=1.5) == ROLE_WARN
    assert severity_role(-1.51, warn_at=0.5, bad_at=1.5) == ROLE_BAD
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_colors_roles.py -v`
Expected: FAIL with `ImportError: cannot import name 'role_text'`

- [ ] **Step 3: Implement**

In `gareus/colors.py`, add after the `ANSI_COLORS` dict (after line 26):

```python
ROLE_TITLE = "title"
ROLE_SECTION = "section"
ROLE_GOOD = "good"
ROLE_WARN = "warn"
ROLE_BAD = "bad"
ROLE_MUTED = "muted"
ROLE_VALUE = "value"

_ROLE_STYLE: dict[str, tuple[str, bool]] = {
    ROLE_TITLE: ("cyan", True),
    ROLE_SECTION: ("magenta", True),
    ROLE_GOOD: ("green", True),
    ROLE_WARN: ("yellow", True),
    ROLE_BAD: ("red", True),
    ROLE_MUTED: ("dim", False),
    ROLE_VALUE: ("white", True),
}
```

Add after `color_text` (after line 99):

```python
def role_text(text: object, role: str, bold: Optional[bool] = None) -> str:
    """Style `text` using a named semantic role instead of a raw color name.

    Centralizing role -> color mapping here means every dashboard panel
    that asks for "a title" or "a warning" gets the same color, instead of
    each call site picking cyan/magenta/yellow ad hoc.
    """
    color, default_bold = _ROLE_STYLE.get(role, _ROLE_STYLE[ROLE_MUTED])
    use_bold = default_bold if bold is None else bool(bold)
    if color == "dim":
        return style_text(text, dim=True, bold=use_bold)
    return style_text(text, color=color, bold=use_bold)


def severity_role(value: float, warn_at: float, bad_at: float) -> str:
    """Map a magnitude to ROLE_GOOD/ROLE_WARN/ROLE_BAD by absolute thresholds."""
    magnitude = abs(float(value))
    if magnitude <= float(warn_at):
        return ROLE_GOOD
    if magnitude <= float(bad_at):
        return ROLE_WARN
    return ROLE_BAD
```

Update `__all__` (lines 102-108) to include the new names:

```python
__all__ = [
    "ANSI_COLORS",
    "ASCII_COLOR_ENABLED",
    "configure_color",
    "style_text",
    "color_text",
    "role_text",
    "severity_role",
    "ROLE_TITLE",
    "ROLE_SECTION",
    "ROLE_GOOD",
    "ROLE_WARN",
    "ROLE_BAD",
    "ROLE_MUTED",
    "ROLE_VALUE",
]
```

Add `from typing import Optional` to the existing `from typing import Optional` import line (already present at line 12 — no change needed there, just confirming `Optional` is available for `role_text`'s signature).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_colors_roles.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add gareus/colors.py tests/test_colors_roles.py
git commit -m "feat(colors): add semantic color roles and severity helper"
```

---

### Task 5: Wire roles into `tui.py` panel/row titles

**Files:**
- Modify: `gareus/tui.py` — `_panel_lines` (line 201), `_dashboard_row` (line 248), `_dashboard_weighted_row` (title line), `_dashboard_full_width_panel` (line 325)
- Test: `tests/test_dashboard_tui_scaling.py`

**Interfaces:**
- Consumes: `role_text`, `ROLE_TITLE`, `ROLE_SECTION` from `gareus.colors` (Task 4).
- Produces: no new public names — same functions, now using roles internally instead of raw `color_text(..., "cyan"/"magenta", bold=True)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard_tui_scaling.py`:

```python
def test_panel_title_uses_title_role(monkeypatch):
    import gareus.tui as tui_mod
    import gareus.colors as colors_mod

    monkeypatch.setattr(colors_mod, "ASCII_COLOR_ENABLED", True)
    from gareus.colors import role_text, ROLE_TITLE

    lines = tui_mod._panel_lines("my panel", ["line1"], width=30)
    expected_title = role_text("my panel", ROLE_TITLE)
    assert any(expected_title in line for line in lines)


def test_dashboard_row_section_title_uses_section_role(monkeypatch):
    import gareus.tui as tui_mod
    import gareus.colors as colors_mod

    monkeypatch.setattr(colors_mod, "ASCII_COLOR_ENABLED", True)
    from gareus.colors import role_text, ROLE_SECTION

    lines = tui_mod._dashboard_row("my section", [("p", ["x"])], term_w=160)
    expected = role_text("my section", ROLE_SECTION)
    assert lines[0] == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_tui_scaling.py -v -k "title_role or section_role"`
Expected: FAIL — current output still uses raw `color_text(..., "cyan"/"magenta", bold=True)`, which produces the same-looking ANSI bytes as `role_text` with today's mapping, so this only fails if the call site hasn't been switched to `role_text` yet... 

**Note for implementer:** because `ROLE_TITLE` maps to `("cyan", True)` and `ROLE_SECTION` to `("magenta", True)` — identical to what `color_text` already produces — this test will actually PASS even before Step 3, since the byte output is the same either way. That is expected and fine: the test's job is to lock in the *mapping* so a future change to `_ROLE_STYLE` in `colors.py` automatically propagates here, not to catch a visible behavior change today. Proceed to Step 3 regardless (do the substitution), then confirm Step 4 still passes.

- [ ] **Step 3: Implement**

In `gareus/tui.py`, add to the existing import line:

```python
from .colors import ASCII_COLOR_ENABLED, color_text, style_text
```

change to:

```python
from .colors import ASCII_COLOR_ENABLED, ROLE_SECTION, ROLE_TITLE, color_text, role_text, style_text
```

In `_panel_lines` (line 201), change:

```python
    title_line = "│" + _ansi_pad(color_text(title, "cyan", bold=True), inner) + "│"
```

to:

```python
    title_line = "│" + _ansi_pad(role_text(title, ROLE_TITLE), inner) + "│"
```

In `_dashboard_row` (line 248), change:

```python
        out: List[str] = [color_text(title, "magenta", bold=True)]
```

to:

```python
        out: List[str] = [role_text(title, ROLE_SECTION)]
```

and the return line:

```python
    return [color_text(title, "magenta", bold=True)] + _join_columns(cols, gap=gap)
```

to:

```python
    return [role_text(title, ROLE_SECTION)] + _join_columns(cols, gap=gap)
```

In `_dashboard_weighted_row` (both the empty-panels early return and the two return statements added/kept from Task 2), change every `color_text(title, "magenta", bold=True)` to `role_text(title, ROLE_SECTION)`.

In `_dashboard_full_width_panel` (line 325), change:

```python
    prefix = [color_text(title, "magenta", bold=True)] if title else []
```

to:

```python
    prefix = [role_text(title, ROLE_SECTION)] if title else []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_tui_scaling.py -v`
Expected: PASS (all tests so far, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add gareus/tui.py tests/test_dashboard_tui_scaling.py
git commit -m "refactor(tui): use semantic color roles for panel/section titles"
```

---

### Task 6: Fix `render_distance_ascii` header separator + dedupe severity coloring

**Files:**
- Modify: `gareus/tui.py` — `render_distance_ascii` (lines 445-631)
- Test: `tests/test_dashboard_tui_scaling.py`

**Interfaces:**
- Consumes: `severity_role`, `ROLE_GOOD`/`ROLE_WARN`/`ROLE_BAD`, `role_text` from `gareus.colors` (Task 4).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard_tui_scaling.py`:

```python
def test_render_distance_ascii_header_separator_matches_width():
    from gareus.tui import render_distance_ascii, strip_ansi

    rows = [{"replica": 0, "window": 0, "center_A": 1.0, "k_kcal_mol_A2": 1.0, "cv_A": 1.0}]
    out = render_distance_ascii(rows, phase="test", step=1, width=80, mode="hist3d")
    header_line = [l for l in strip_ansi(out).splitlines() if l.startswith("+--")][0]
    assert header_line.count("-") == 80


def test_render_distance_ascii_short_width_separator_also_matches():
    from gareus.tui import render_distance_ascii, strip_ansi

    rows = [{"replica": 0, "window": 0, "center_A": 1.0, "k_kcal_mol_A2": 1.0, "cv_A": 1.0}]
    out = render_distance_ascii(rows, phase="test", step=1, width=24, mode="hist3d")
    header_line = [l for l in strip_ansi(out).splitlines() if l.startswith("+--")][0]
    assert header_line.count("-") == 24
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_tui_scaling.py -v -k header_separator`
Expected: FAIL — current output always has exactly 58 dashes regardless of `width`.

- [ ] **Step 3: Implement**

In `gareus/tui.py`'s `render_distance_ascii`, change the header block (lines 547-559) from:

```python
    header = (
        "\n"
        + color_text(f"+-- CV {primary_label_val} coverage ", "cyan", bold=True)
        + color_text("-" * 58, "cyan")
        + "\n"
```

to:

```python
    header = (
        "\n"
        + color_text(f"+-- CV {primary_label_val} coverage ", "cyan", bold=True)
        + color_text("-" * width_int, "cyan")
        + "\n"
```

Then, dedupe the delta-severity coloring. Change (lines 593-594, hist3d path):

```python
            delta = r["cv_A"] - r["center_A"]
            delta_col = "green" if abs(delta) <= 0.5 else "yellow" if abs(delta) <= 1.5 else "red"
```

and the identical pair (lines 619-620, table path) to both instead read:

```python
            delta = r["cv_A"] - r["center_A"]
            delta_col = _delta_severity_color(delta)
```

Add a small local helper near the top of `render_distance_ascii` (after the `width_int = max(20, int(width or 54))` line, ~line 466), importing what's needed:

```python
    from .colors import severity_role

    def _delta_severity_color(delta: float) -> str:
        role = severity_role(delta, warn_at=0.5, bad_at=1.5)
        return {"good": "green", "warn": "yellow", "bad": "red"}[role]
```

(Using the raw color-name string here, not `role_text`, keeps this change local to the two `f"{color_text(f'{delta:+7.2f}', delta_col, bold=True)}"` call sites unchanged — only the *derivation* of `delta_col` is deduplicated, matching the spec's scoped color-role migration.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_tui_scaling.py -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add gareus/tui.py tests/test_dashboard_tui_scaling.py
git commit -m "fix(tui): scale histogram header separator to actual width, dedupe severity coloring"
```

---

### Task 7: Rewire `_render_dashboard` in `logger.py` to use the new helpers

**Files:**
- Modify: `gareus/logger.py` — imports (lines 27-44), `_render_dashboard` (lines 2417-2596)
- Test: `tests/test_dashboard_scaling_integration.py` (new file)

**Interfaces:**
- Consumes: `dashboard_body_budget`, `_weighted_panel_widths`, `MIN_PANEL_WIDTH`, `dashboard_row_gap` from `gareus.tui` (Tasks 1-3).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dashboard_scaling_integration.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dashboard_scaling_integration.py -v`
Expected: `test_cv_histogram_bar_grows_past_old_ceiling_on_large_terminal` FAILs (`max_bar_len` tops out at 140 under the current hardcoded ceiling regardless of the 500-column terminal). The other two tests may already PASS against current code; that's fine, they're regression guards for the refactor in Step 3.

- [ ] **Step 3: Implement**

In `gareus/logger.py`, update the `from .tui import (...)` block (lines 27-44) to add the new names:

```python
from .tui import (
    _ansi_pad,
    _ansi_truncate,
    _ascii_position,
    _dashboard_density,
    _dashboard_full_width_panel,
    _dashboard_weighted_row,
    _join_columns,
    _panel_lines,
    _weighted_panel_widths,
    dashboard_body_budget,
    dashboard_row_gap,
    format_duration,
    make_progress_bar,
    render_distance_ascii,
    replica_fg256,
    replica_marker,
    strip_ansi,
    strip_ansi_len,
    write_tui_frame,
    MIN_PANEL_WIDTH,
)
```

Replace the block at lines 2449-2459:

```python
        usable_w = max(40, term_w - 2)
        row_gap = 2 if term_w < 120 else 3
        min_panel_w = max(24, int(getattr(self.args, "dashboard_min_panel_width", 30) or 30))
        wide_threshold = int(getattr(self.args, "dashboard_wide_threshold", 120) or 120)
        pe_panel_w = usable_w if usable_w < 105 else max(34, min(58, usable_w // 3))
        cv_panel_w = usable_w if usable_w < 105 else max(50, usable_w - pe_panel_w - row_gap)
        cv_bar_width = max(24, min(140 if full_dashboard else 110, cv_panel_w - 62))
        pe_bar_width = max(10, min(36 if full_dashboard else 28, pe_panel_w - 24))
        row1_max = (10 if compact_dashboard else 18 if density == "normal" else 30)
        row23_max = (8 if compact_dashboard else 12 if density == "normal" else 18)
        row4_max = (8 if compact_dashboard else 14 if density == "normal" else 18)
```

with:

```python
        usable_w = max(40, term_w - 2)
        row_gap = dashboard_row_gap(term_w)
        min_panel_w = max(
            ABSOLUTE_MIN_PANEL_WIDTH,
            int(getattr(self.args, "dashboard_min_panel_width", MIN_PANEL_WIDTH) or MIN_PANEL_WIDTH),
        )
        wide_threshold = int(getattr(self.args, "dashboard_wide_threshold", 120) or 120)
        if usable_w < 105:
            cv_panel_w = pe_panel_w = usable_w
        else:
            cv_panel_w, pe_panel_w = _weighted_panel_widths(
                [2.4, 0.9], term_w=term_w, gap=row_gap, min_panel_width=min_panel_w
            )
        cv_bar_width = max(24, cv_panel_w - 62)
        pe_bar_width = max(10, pe_panel_w - 24)
        row1_max = dashboard_body_budget(term_h, 18 / 40, floor=10)
        row23_max = dashboard_body_budget(term_h, 12 / 40, floor=8)
        row4_max = dashboard_body_budget(term_h, 14 / 40, floor=8)
```

This also requires adding `ABSOLUTE_MIN_PANEL_WIDTH` to the `from .tui import (...)` block above (add it alongside `MIN_PANEL_WIDTH`).

Note `full_dashboard`/`compact_dashboard` are no longer read by this block — they are still used elsewhere in `_render_dashboard` (topology/2D panel body-line caps, gating heavy panels), so do not remove those variables themselves, only their use in the block above.

Next, replace the other density-tied body-line ladders with `dashboard_body_budget`, keeping each site's existing reference value at `term_h=40` as the fraction numerator (so behavior at a typical 160x40 terminal is unchanged):

Line 2515 (`topology_lines` full-width panel), change:

```python
                max_body_lines=(10 if compact_dashboard else 18 if density == "normal" else 26),
```

to:

```python
                max_body_lines=dashboard_body_budget(term_h, 18 / 40, floor=10),
```

Line 2528 (CV+PE weighted row), change:

```python
                max_panel_body_lines=max(row1_max, min(32, int(getattr(self.args, "distance_ascii_max_replicas", 32) or 32) + 4)),
```

to:

```python
                max_panel_body_lines=max(row1_max, int(getattr(self.args, "distance_ascii_max_replicas", 32) or 32) + 4),
```

(dropping the `min(32, ...)` ceiling — `row1_max` on its own already grows with the terminal, and capping the replica-count-driven term at 32 defeats showing more replicas on a taller terminal).

Lines 2548 and 2566 (`diff2d`/`map2d` full-width panels), change both occurrences of:

```python
                max_body_lines=(14 if compact_dashboard else 22 if density == "normal" else 28),
```

to:

```python
                max_body_lines=dashboard_body_budget(term_h, 22 / 40, floor=14),
```

Line 2541 (`map2d_max_body`), change:

```python
        map2d_max_body = max(10, min(24, n_sec + 8))
```

to:

```python
        map2d_max_body = max(10, n_sec + 8)
```

Line 2490 (`cov2_w`), change:

```python
                cov2_w = max(12, min(48, term_w // 3))
```

to:

```python
                cov2_w = max(12, term_w // 3)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dashboard_scaling_integration.py tests/test_dashboard_tui_scaling.py tests/test_store.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add gareus/logger.py tests/test_dashboard_scaling_integration.py
git commit -m "refactor(logger): drive dashboard panel/body sizing from continuous terminal-size scaling"
```

---

### Task 8: Full regression pass and manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the full existing test suite**

Run: `pytest -q`
Expected: all tests pass (no regressions in unrelated modules from the `gareus/tui.py`/`gareus/logger.py`/`gareus/colors.py` changes).

- [ ] **Step 2: Byte-compile check**

Run: `python -m py_compile gareus/tui.py gareus/logger.py gareus/colors.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Manual smoke at three terminal sizes**

Run (from repo root, in a real terminal, resizing between runs — or use `COLUMNS`/`LINES` env vars if the terminal emulator doesn't resize interactively):

```bash
python - <<'EOF'
import argparse, os
from gareus.logger import DistanceLogger

for cols, lines in [(80, 24), (160, 40), (300, 90)]:
    os.environ["COLUMNS"] = str(cols)
    os.environ["LINES"] = str(lines)
    args = argparse.Namespace()
    logger = DistanceLogger(".", args, no_file_persistence=True)
    rows = [
        {"replica": i, "window": i, "center_A": float(i), "k_kcal_mol_A2": 1.0, "cv_A": float(i) + 0.1}
        for i in range(6)
    ]
    print(f"\n=== {cols}x{lines} ===")
    print(logger._render_dashboard(
        rows, phase="production", step=100, total_steps=1000, summary={},
        dashboard_info={"centers_a": [float(i) for i in range(6)], "n_windows": 6},
    ))
    logger.close()
EOF
```

Expected: at 300x90, panels are visibly wider/taller than at 160x40 (not the same size with blank margin around it); at 80x24, output still renders without crashing (narrow-terminal vertical stacking kicks in); colors/spacing look consistent panel-to-panel across all three.

- [ ] **Step 4: Confirm no leftover references to removed magic numbers**

Run: `grep -n "min(58\|min(140\|min(110\|min(36 if\|min(28 if\|min(48, term_w" gareus/logger.py gareus/tui.py`
Expected: no output (all the ceilings this rework targeted are gone).

- [ ] **Step 5: Final commit (if Step 3/4 surfaced any fixups)**

```bash
git add -A
git commit -m "test: confirm dashboard scaling rework end to end"
```

(Skip this commit if no files changed — Steps 1-4 are verification only.)
