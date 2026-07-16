# Live Dashboard TUI Rework

Date: 2026-07-16
Target: `gareus/tui.py`, `gareus/logger.py`, `gareus/colors.py`

## Goal

Rework gareus's own live run-facing dashboard (the hand-rolled ANSI TUI shown
during `gareus` runs — not `RUNS/gareus_monitor.py`, a separate multi-run
monitor script covered by a different spec, and not the `-hh` help
encyclopedia) so it:

1. Uses the full available terminal on large/ultrawide terminals instead of
   capping out and leaving blank margin.
2. Has a consistent color scheme and consistent spacing/alignment across
   panels, instead of ad hoc per-call-site choices.

## Background / Root Cause

`gareus/tui.py` already measures `shutil.get_terminal_size()` every frame and
reflows correctly in general: `_dashboard_weighted_row` distributes panel
width proportionally by weight and patches rounding error into the widest
panel so the total width is always exact, and both `_dashboard_row` and
`_dashboard_weighted_row` fall back to vertical stacking below a minimum
panel width instead of producing unreadably narrow columns.

The bug is that some dashboard rows bypass this proportional splitter.
`_render_dashboard` (`gareus/logger.py:2417+`) hand-computes fixed pixel
splits for the CV-histogram / potential-energy row with hardcoded ceilings:

```
pe_panel_w   = max(34, min(58, usable_w // 3))
cv_panel_w   = max(50, usable_w - pe_panel_w - row_gap)
cv_bar_width = max(24, min(140 if full_dashboard else 110, cv_panel_w - 62))
pe_bar_width = max(10, min(36 if full_dashboard else 28, pe_panel_w - 24))
cov2_w       = max(12, min(48, term_w // 3))
```

and vertical body-line budgets are three hardcoded integers keyed to a
3-bucket density classifier (`_dashboard_density`, `gareus/tui.py:329`):

```
row1_max  = 10 if compact else 18 if normal else 30
row23_max = 8  if compact else 12 if normal else 18
row4_max  = 8  if compact else 14 if normal else 18
```

`_dashboard_density` itself tops out at one threshold (`term_w>=150 and
term_h>=48` → `"full"`) — any terminal bigger than that still gets the same
caps, so extra space renders as blank margin. Separately,
`render_distance_ascii`'s section header always draws a literal
`"-" * 58` separator regardless of the panel's actual computed width.

Cosmetically: `gareus/colors.py` has no semantic color scheme, just raw ANSI
names (`cyan`, `magenta`, `dim`, `green`, `yellow`, `red`, `white`). 184
call sites across `tui.py`/`logger.py` pick colors ad hoc. The red/yellow/
green severity thresholding for CV delta (`abs(delta) <= 0.5` → green,
`<= 1.5` → yellow, else red) is duplicated verbatim in two places in
`render_distance_ascii`. Minimum-panel-width and inter-panel gap constants
also drift across call sites (28 vs. 30 vs. 22 for min width; `row_gap = 2
if term_w < 120 else 3` is a one-off in `logger.py` not shared with `tui.py`).

## Non-Goals

- No swap to Rich/Textual/blessed. The code's own comments state the design
  principle ("ugly but stable beats beautiful-but-sentient terminal
  confetti") and this dashboard runs unattended for multi-day MD jobs;
  reflow correctness matters more than layout-engine convenience here, and
  domain-specific rendering (histograms, replica markers, 2D density maps)
  would still be hand-written under Rich — only the box/layout math would
  change, which is the cheap part to fix in place.
- No change to `RUNS/gareus_monitor.py` (separate tool, separate spec) or
  to `gareus/helptext.py` (`-hh` encyclopedia, unrelated).
- No change to what data any panel shows, CLI flag semantics, or dashboard
  render throttling (`dashboard_render_interval_sec`).
- No full retroactive migration of all 184 color call sites to semantic
  roles — scope is headers/section titles and the good/warn/bad severity
  coloring, where inconsistency is most visible and where the duplicated
  threshold logic lives. Purely decorative `dim`/`white` uses elsewhere are
  left as-is.

## Design

### 1. Continuous width/height scaling

Replace the hand-split, ceiling-capped row computation in `_render_dashboard`
with a call through `_dashboard_weighted_row` (or a new small helper on the
same pattern), passing weights for the CV-histogram vs. PE-histogram panels
instead of pre-dividing `usable_w`. Bar widths within each panel
(`cv_bar_width`, `pe_bar_width`, `cov2_w`) are derived from the *resulting*
panel width the splitter assigns minus fixed label overhead (e.g. `cv_panel_w
- 62`), with a floor but no independent upper ceiling — so wide terminals
grow bar width instead of hitting a wall.

`_dashboard_density`'s 3-bucket classifier (`compact`/`normal`/`full`) stays,
but its role narrows to "which panels are shown at all" (a real categorical
call — e.g. hiding heavy 2D panels on a tiny terminal). Any value it
currently gates purely for *sizing* (`row1_max`, `row23_max`, `row4_max`,
the `cv_bar_width`/`pe_bar_width` ceilings) is replaced by a formula against
the actual `term_h`/`term_w` for that frame — proportional to available
vertical space minus fixed chrome (header/decision panel/legend lines),
with the same floors already in place today so small terminals don't
regress.

`render_distance_ascii`'s header separator changes from a literal `"-" *
58` to match the panel's actual rendered width.

### 2. Semantic color roles

Add a small role layer to `gareus/colors.py`: named roles (e.g.
`ROLE_TITLE`, `ROLE_SECTION`, `ROLE_GOOD`, `ROLE_WARN`, `ROLE_BAD`,
`ROLE_MUTED`, `ROLE_VALUE`) each mapped to one color/style choice in one
place, plus a `role_text(text, role, bold=None)` helper alongside the
existing `color_text`/`style_text` (kept, not removed — other call sites
keep working unchanged). Migrate: panel/section title coloring in
`tui.py`'s `_panel_lines`/`_dashboard_row`/`_dashboard_weighted_row`/
`_dashboard_full_width_panel`, and the delta-severity coloring in
`render_distance_ascii`.

Extract the duplicated green/yellow/red delta threshold logic into one
`_severity_color(value, warn_at, bad_at) -> role` helper in `tui.py`, used
by both `render_distance_ascii` code paths (hist3d and table modes).

### 3. Alignment/spacing consistency

Unify the drifted magic numbers:
- One shared `MIN_PANEL_WIDTH` constant in `tui.py`, used by both
  `_dashboard_row` (currently hardcodes 28) and `_dashboard_weighted_row`
  (currently defaults to 30, floors at 22).
- One shared row-gap formula in `tui.py` (e.g. `dashboard_row_gap(term_w)`),
  replacing the one-off `row_gap = 2 if term_w < 120 else 3` inline in
  `logger.py`.

## Testing

No existing tests cover `tui.py`/`logger.py` layout functions — this is a
gap independent of this rework. TDD per project convention: write tests
first for the functions being touched, covering:

- Panel/bar widths scale up as `term_w` grows past 150 columns (no
  regression to a flat value above any threshold), down to a documented
  sane floor.
- Vertical body-line budgets scale with `term_h` similarly.
- `_dashboard_weighted_row`'s existing proportional-split and
  narrow-terminal stacking behavior is unchanged for typical 160×40
  terminals (regression protection while refactoring the CV/PE row to use
  it).
- `_severity_color` returns the correct role at and around both thresholds
  (boundary cases at exactly 0.5 and 1.5).
- Role-color mapping in `colors.py` (role → correct ANSI/256 style,
  disabled when `ASCII_COLOR_ENABLED` is `False`).

Target: the touched functions reach the project's 80% coverage bar: this is
not a retroactive full-file test backfill for pre-existing untouched code.

## Files

- `gareus/tui.py`
- `gareus/logger.py`
- `gareus/colors.py`
- new/updated tests under `tests/` (file TBD at implementation-plan time,
  following existing `tests/test_*` naming convention)

## Verification

- New/updated unit tests pass.
- `python -m py_compile gareus/tui.py gareus/logger.py gareus/colors.py`
- Manual smoke: run gareus (or a minimal harness driving `_render_dashboard`
  directly) at a small (80×24), typical (160×40), and large (300×80+)
  terminal size and visually confirm panels scale rather than capping with
  blank margin, and colors/spacing look consistent across panels.
