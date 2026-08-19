# GAREUS TUI — code inventory and design spec

Extracted 2026-08-19 from `gareus/` at `main` (post `ab9d62e`). Purpose: hand over the
terminal-UI subsystem as a liftable unit — what the files are, what the design rules
are, and exactly where the extraction boundary cuts.

---

## 1. Inventory — what "the TUI" actually is

The TUI is **not** one file. It is a 3-layer stack; only the bottom layer is generic.

| Layer | File | Lines | Role | Generic? |
|---|---|---|---|---|
| Palette | `gareus/colors.py` | 159 | ANSI codes, semantic **roles**, severity mapping, on/off gate | yes — zero domain coupling |
| Primitives | `gareus/tui.py` | 696 | frame lifecycle, ANSI-safe pad/truncate, box panels, weighted column layout, progress bar, histogram renderer | mostly — `render_distance_ascii` is umbrella-sampling-specific |
| Composition | `gareus/logger.py` | 2779 | `DistanceLogger` — all content panels + full dashboard frame assembly | no — GAREUS domain (windows, replicas, GaMD boost, exchange) |
| Line mode | `gareus/progress.py` | 234 | `GuiProgressSink` — single-line progress + JSONL event stream | yes, apart from OpenMM cleanup helper |
| Help surface | `gareus/helptext.py` | 1814 | `-hh` encyclopedia renderer + pager (`page_text`) | yes — reusable builder |
| CV formatting | `gareus/formatting.py` | 87 | primary-CV value/unit strings | thin, half-stubbed |

Import graph (no cycles):

```
colors.py        (stdlib: sys)
   ↑
tui.py           (stdlib: math, re, shutil, sys  + numpy)
   ↑        ↑
progress.py   logger.py      helptext.py (stdlib only: argparse, os, re, subprocess, sys, textwrap)
```

`gareus/tui.py` imports **only** `colors` from the package. That is the clean cut line.

---

## 2. Design system

### 2.1 Semantic colour roles, not raw colours

`colors.py` deliberately routes every call through a **role**, so "a title" is the same
colour in every panel instead of each call site picking cyan/magenta ad hoc
(`gareus/colors.py:117`).

| Role | Colour | Bold |
|---|---|---|
| `ROLE_TITLE` | cyan | yes |
| `ROLE_SECTION` | magenta | yes |
| `ROLE_GOOD` | green | yes |
| `ROLE_WARN` | yellow | yes |
| `ROLE_BAD` | red | yes |
| `ROLE_MUTED` | dim | no |
| `ROLE_VALUE` | white | yes |

`severity_role(value, warn_at, bad_at)` maps a magnitude → good/warn/bad by absolute
thresholds (`gareus/colors.py:131`). Style entry points: `style_text` (supports
`fg256`/`bg256`), `color_text` (basic), `role_text` (semantic).

**Colour gate:** module global `ASCII_COLOR_ENABLED`, default **`False`** — deliberate, HPC
launchers and log parsers mangle ANSI. `configure_color("always"|"never"|"auto")` flips it;
`auto` = `sys.stdout.isatty()`. Called once, `gareus/cli.py:1567`.

### 2.2 Character vocabulary

- **Panel box:** `┌ ─ ┐ ├ ┤ └ ┘ │` (`_panel_lines`, `gareus/tui.py:206`) — title row, separator, body, no bottom title.
- **Density ramp (9 levels):** `" ▁▂▃▄▅▆▇█"` with paired 256-colour fg/bg ramps
  (`_hist3d_cell`, `gareus/tui.py:458`): fg `246→249→81→87→117→123→159→231`,
  bg `236→236→237→238→239→24→25→31`; bold at level ≥ 6. Level 0 renders dim `·`, not blank.
- **2D map ramp:** `░▒▓█` (log-density panels in `logger.py`).
- **Markers:** `●` current CV (coloured by replica id), `│` umbrella centre (fg 226 / bg 236),
  `◉` current-overlaps-centre (fg 231 / bg 53), `*`/`C`/`@` in the plain-ASCII fallback mode.
- **Progress bar:** `#` filled (green bold) / `-` empty (dim) — plain ASCII, not blocks
  (`make_progress_bar`, `gareus/tui.py:412`).
- **Truncation marker:** `…`.

### 2.3 Replica identity palette

`replica_fg256` (`gareus/tui.py:423`) — 16 bright 256-colour indices
`[46,51,226,201,208,117,171,82,214,159,99,197,45,220,141,33]`, `replica % 16`, fallback 231.
Stable per replica id so an exchange is visually trackable across frames.
`replica_marker(text, replica, bg256=236, bold=True)` is the wrapper.

---

## 3. Layout engine (the real design content)

### 3.1 Constants and rules

- `MIN_PANEL_WIDTH = 30`, `ABSOLUTE_MIN_PANEL_WIDTH = 22` (`gareus/tui.py:44-45`).
- `dashboard_row_gap(term_w)` → `2` if `term_w < 120` else `3` (`:48`). One shared rule for every row.
- Usable width is always `max(40, term_w - 2)`; terminal size falls back to `(160, 40)`.

### 3.2 Proportional widths

`_weighted_panel_widths(weights, term_w, gap, min_panel_width)` (`gareus/tui.py:271`):

1. clamp weights to `≥ 0.05`;
2. if `usable < n*min_w + gap_total` → **narrow mode**: return `[usable]*n` (caller stacks vertically);
3. else distribute `available = usable - gap_total` proportionally, `min_w` floor;
4. redistribute the rounding remainder ±1 at a time, widest-first, never below `min_w`, hard-capped at 10 000 iterations.

Exposed standalone (not just inline in the row builder) specifically so a caller can
pre-render content — e.g. an ASCII bar — at the exact width the row will allocate instead
of guessing and letting the panel silently pad/truncate the mismatch.

### 3.3 Vertical budget

`dashboard_body_budget(term_h, fraction, floor)` → `max(floor, round(term_h*fraction))`
(`gareus/tui.py:386`). Continuous in terminal height rather than a fixed ceiling per
density bucket, so a tall terminal shows more instead of the same amount plus margin.
Rows do **not** need to sum exactly — `_safe_tui_frame_text` truncates the whole frame anyway.

### 3.4 Density tiers

`_dashboard_density(args, term_w, term_h)` (`gareus/tui.py:371`) — honours an explicit
`compact|normal|full`, else auto: `compact` if `w<110 or h<34`; `full` if `w>=150 and h>=48`; else `normal`.

### 3.5 Row builders

| Builder | Line | Behaviour |
|---|---|---|
| `_dashboard_row` | `:243` | equal widths, stacks vertically when too narrow |
| `_dashboard_weighted_row` | `:314` | panels as `(name, body)` or `(name, body, weight)`; stacks when too narrow |
| `_dashboard_full_width_panel` | `:357` | one panel across the usable width |
| `_join_columns` | `:226` | side-by-side join of already-fixed-width panels, gap ≥ 1, short columns padded |

Every row emits its title via `ROLE_SECTION` as a bare line above the boxes.

### 3.6 ANSI-safe geometry

All width maths goes through `strip_ansi` / `strip_ansi_len` (`:174`, `:179`,
regex `\x1b\[[0-9;]*m`). `_ansi_truncate` (`:184`) **strips colour from a clipped line** —
documented as intentional: stable geometry beats colour on a clipped edge.
`_ansi_pad` (`:200`) pads/truncates to exact visible width.

---

## 4. Frame lifecycle

- `tui_clear_enabled(args)` (`:53`) — **does not consult `isatty()`**, by design: MPI
  launchers, `tee`, SLURM wrappers and some IDE terminals report non-TTY while ANSI cursor
  control still works. Full-frame clear is on unless `tui_clear_mode == "never"` or
  `tui_mode ∉ {dashboard, interactive}`.
- `clear_tui_screen` (`:71`) — `\033[?25l\033[H\033[J` (hide cursor, home, clear-down).
  Uses `J` not a newline to avoid the one-line scroll some terminals do when a frame ends on the last row.
- `write_tui_frame(text, args)` (`:136`) — full repaint
  `\033[?25l\033[H\033[2J\033[3J` + frame + `\033[0m\033[J`. Setup phases (NVT/NPT/prescan/
  pulling) stay full-frame so the TUI never degrades into a log waterfall.
  On `BlockingIOError`/`BrokenPipeError`/`OSError` the repaint is **dropped**, not replaced
  by a fallback line (`_write_stdout_best_effort`, `:122`). Log-style mode appends with `\n`.
- `_safe_tui_frame_text` (`:88`) — clamps every physical line to `term_w-2` and the frame to
  `min(term_h-1, dashboard_max_height)`, replacing the last kept line with a dim
  "dashboard truncated to terminal height …" notice. This is what makes the frame
  non-wrapping, hence non-jumping.
- `restore_tui_cursor()` (`:162`) — `\033[?25h`. Registered defensively at
  `gareus_peptide.py:25`, `gareus/core.py:18`, `gareus/legacy.py:39`.

Module global `_TUI_LAST_FRAME_LINES` tracks the last frame height (reuse-relevant state).

---

## 5. Dashboard composition (`DistanceLogger._render_dashboard`, `gareus/logger.py:2398`)

Section order of the assembled frame:

1. **Header** — `_render_compact_header` (`:2283`): phase, progress bar, wall/ETA, ns/day per
   replica + aggregate, step/total, window topology label (`sparse explicit 2D` /
   `rectangular 2D` / `1D/secondary-fixed`). Every line `_ansi_truncate`d to usable width.
2. **Decision line** — `_render_compact_decision_inline` (`:2150`), from
   `_dashboard_decision_state` (`:1920`).
3. **CV2 coverage bar** — 2D runs only, `_coverage_bar`, width `term_w // 3`.
4. `panel_mode == "minimal"` short-circuits here with one full-width CV map panel.
5. **2D topology** — full-width, `_render_2d_sparse_topology_map` (`:1761`), heavy-panel gated.
6. **CV + potential-energy row** — weighted `[2.4, 0.9]`; below `usable_w < 105` both panels go
   full width. Bar widths are pre-computed for the allocated panel width:
   `cv_bar_width = max(24, cv_panel_w - 72)` (the 72 accounts for the hist3d row's own
   `] nw=<n>` suffix, measured — not the naive prefix width), `pe_bar_width = max(10, pe_panel_w - 24)`.
7. **2D landscape** — side-by-side log-density map (56 % of width) + replica grid when
   `term_w >= dashboard_wide_threshold` and not compact; otherwise two stacked full-width panels.
8. **Diagnostics row** — weighted: exchange acceptance `1.15`, umbrella pull `1.00`,
   GaMD boost `1.00`, plus histogram overlap `0.85` inserted at index 1 for 1D runs only
   (in 2D the overlap lives in the topology panel).
9. **Motion + recommendations row** — replica table `1.5` / recommendations `1.0`. The table
   auto-columnizes: render at `ncols=1`, measure the natural line width, then
   `ncols = (inner_avail + 3) // (line_w + 3)`.

Body budgets used: row 1 `18/40`, rows 2-3 `12/40`, row 4 `14/40`, 2D panels `22/40`
(fractions of `term_h`, via `dashboard_body_budget`).

Content renderers on `DistanceLogger` (all return `list[str]`, all width-parameterised):
`_render_potential_energy_map`, `_render_exchange_acceptance`, `_render_overlap`,
`_render_pull_map`, `_render_gamd_boost`, `_render_pmf_preview`, `_render_replica_diffusion`,
`_render_replica_table`, `_render_sparklines`, `_render_health`, `_render_recommendations`,
`_render_2d_diffusion_map`, `_render_2d_replica_map`. Plus local mini-primitives
`_sparkline`, `_mini_bar`, `_coverage_bar`, `_theme_header`, `_short_status`.

Render throttling: `dashboard_render_interval_sec`, `dashboard_heavy_panels_every`
(heavy panels every N frames), `dashboard_panels` (`minimal|normal|…`).

---

## 6. The umbrella histogram renderer (`render_distance_ascii`, `gareus/tui.py:499`)

Lives in `tui.py` but is domain-shaped — it consumes rows with
`replica / window / center_A / k_kcal_mol_A2 / cv_A` (+ optional
`umbrella_bias_kcal_mol`, `umbrella_pull_kcal_mol_A`, both recomputed from `k` and `Δ` if absent).

Modes: `none` (empty string), `compact` (single-row strip), `hist` / `hist3d` (per-replica table
plus shaded bar). Axis range = min/max over current CVs, window centres **and** history, padded
8 % each side. Histogram source selectable `window` (default) or `replica`.

`_ascii_position` (`:437`) uses `floor(frac*width)` capped at `width-1` — deliberately matching
`np.histogram`'s binning so the centre marker, the current marker and the shaded cell can never
land in adjacent bins near a boundary. Rows beyond `max_replicas` (32) are dropped with an
explicit "raise `--distance-ascii-max-replicas`" note rather than silently.

Note `_delta_severity_color` (inside `render_distance_ascii`) **re-encodes** the role→colour map
instead of calling `role_text`, with an in-code comment saying so: it keeps the ANSI output
byte-for-byte stable against future `_ROLE_STYLE` bold/format changes. Intentional duplication.

---

## 7. Line mode + machine-readable stream (`GuiProgressSink`, `gareus/progress.py`)

Two independent sinks from one `progress()` call:

- **JSONL** — one JSON object per line to `<out>/progress.jsonl`, throttled by
  `progress_update_interval_sec` (0.25 s). Fields: `event/phase/step/total_steps/fraction/
  percent/message/elapsed_s/eta_s/wall_elapsed_s` plus, when `timestep_fs > 0`:
  `sim_time_ps`, `sim_time_ns`, `aggregate_sim_time_ns`, `ns_per_day`, `aggregate_ns_per_day`,
  `wall_s_per_ns`, `wall_h_per_us`, `wall_ms_per_step`, `steps_per_s`.
  Deliberately boring so a GUI can tail it without importing OpenMM.
- **Console** — `[phase] [bar] step/total pct wall … eta … sim … perf …` joined by `" | "`,
  ANSI-length-truncated to `term_w-1`. In `dashboard`/`interactive` mode it routes through
  `write_tui_frame`; in line mode it prints with `\r` (or `\n` when `force`).

Handoff rule: in dashboard/interactive mode the sink **returns early** for phases
`gamd_calibration` and `gareus_production` — `DistanceLogger` owns the frame there, so the
progress bar can't flicker underneath the dashboard.

---

## 8. Help / pager surface (`gareus/helptext.py`)

Same house style, non-dashboard half of it (details already in `CLAUDE.md`):

- `render_encyclopedia_help(parser, encyclopedia_text, topic=None, color=None)` (`:1728`) —
  shared builder: banner + auto-numbered TOC + topic bodies + `parser.format_help()`.
- `_parse_encyclopedia` (`:1648`) — headings discovered by `Title\n----` underline convention
  (`_HEADING_RE`, `:1624`, underline length must match), numbered **at render time** so the TOC
  can never drift from the prose.
- `_render_toc` / `_render_section` / `_render_topic` (`:1686`, `:1698`, `:1705`);
  `heavy_help_text` (`:1762`) is a one-line wrapper. `gareus/energy_decomposition.py`'s `-hh`
  uses the same builder — "unite tui design completely".
- `page_text(text, parser=None)` (`:1767`) — pipes through `$PAGER` (default `less`, `$LESS`
  defaulted to `FRX` only if unset) when `sys.stdout.isatty()`; plain print when redirected;
  `GAREUS_NO_PAGER=1` forces plain.
- Colour auto-detected from `isatty()`, disabled by `NO_COLOR` — same gate as paging.

---

## 9. Config surface — and a live finding

Only **two** TUI knobs are real argparse flags (`gareus/cli.py:515-516`):

```
--progress-mode  none|console|jsonl|both   (default console)
--tui-mode       dashboard|interactive|line|none   (default dashboard)
```

Everything else the TUI reads is **hardcoded in `_shim_output`** (`gareus/cli.py:1053-1072`) and
unreachable from the CLI or YAML:

`color="auto"`, `progress_jsonl`, `progress_update_interval_sec=0.25`, `progress_bar_width=36`,
`tui_clear_mode="always"`, `dashboard_density="auto"`, `dashboard_wide_threshold=132`,
`dashboard_min_panel_width=30`, `dashboard_max_height=0`, `dashboard_render_interval_sec=0.0`,
`dashboard_panels="normal"`, `dashboard_heavy_panels_every=1`, `distance_ascii_mode="hist3d"`,
`distance_ascii_width=54`, `distance_ascii_max_replicas=32`, `distance_history_limit=4000`.

This is the same "dropped tuning knob" pattern `CLAUDE.md` records for
`--ap-epoch0-step-fraction` and `require_convergence_before_final`. Live consequences worth
knowing before reuse — all three verified by grep against `gareus/cli.py`:

- **`--color` does not exist.** Colour is always `auto` (= `isatty()`); there is no way to force
  it on for a redirected log or off for a colour-hostile TTY.
- **`--distance-ascii-max-replicas` does not exist**, yet the dashboard prints
  "raise `--distance-ascii-max-replicas` to show them" when it omits replicas
  (`gareus/tui.py`, `render_distance_ascii`) — a user-facing instruction to type a flag that
  will error as unrecognised.
- **`--tui-clear-mode` does not exist**, yet `tui_clear_enabled`'s own docstring
  (`gareus/tui.py:53`) says clearing happens "unless the user explicitly requests
  `--tui-clear-mode never`". `_shim_output` pins it to `"always"`, so the `never` branch at
  `gareus/tui.py:60` is dead code from the shipped CLI. Same shape as the
  `require_convergence_before_final` guard `CLAUDE.md` records as unreachable.
Also note `dashboard_wide_threshold` is shimmed to `132` while `logger.py`'s `getattr` fallback
is `120` — the fallback only bites if the shim is bypassed.

---

## 10. Extraction boundary — what to lift, and the warts

**Lift cleanly:** `colors.py` + `tui.py` (minus `render_distance_ascii`) + `helptext.py`.
`progress.py` needs `io.BufferedJsonlWriter` and drops `release_openmm_contexts`.
`logger.py` is GAREUS-specific composition — treat it as the reference *example* of using the
primitives, not as portable code.

Warts to plan around:

1. **`args` duck-typing.** `tui_clear_enabled`, `clear_tui_screen`, `write_tui_frame`,
   `_dashboard_density`, `GuiProgressSink` all read attributes off an argparse-shaped
   namespace via `getattr(args, name, default)`. A reuser must supply a namespace (or a small
   config shim exposing those names). This is the main coupling in an otherwise clean module.
2. **Module-level mutable state.** `colors.ASCII_COLOR_ENABLED` (set once by
   `configure_color`) and `tui._TUI_LAST_FRAME_LINES`. Not thread-safe, not per-instance.
3. **numpy in the TUI layer.** `tui.py` imports numpy, and — verified by
   `grep -n "np\." gareus/tui.py` — **exactly two functions touch it**:
   `_render_histogram_row` (`:468`, `np.histogram`/`np.max`) and `render_distance_ascii`
   (`:499`, axis min/max + summary stats). Their pure-stdlib helpers `_ascii_position` (`:437`)
   and `_hist3d_cell` (`:458`) can stay. Move those two functions out and the primitives layer
   is stdlib-only (`math`, `re`, `shutil`, `sys`).
4. **`formatting.py` is half-stubbed** — its docstrings admit the real logic still lives in
   `gareus.core`; `primary_cv_unit_suffix` guesses `"nm"` for legacy namespaces. Don't treat
   it as the CV-formatting authority.
5. **Truncation drops colour** on the clipped line, by design (§3.6) — expected, not a bug.

---

## 11. Executable contract

`tests/test_dashboard_tui_scaling.py` + `tests/test_dashboard_scaling_integration.py` —
**20 tests, all passing** (`python -m pytest -q`, verified 2026-08-19). They pin exactly the
invariants above and are what any port should be re-run against:

- `dashboard_row_gap`: 100→2, 119→2, 120→3, 300→3;
- `dashboard_body_budget`: monotone in `term_h`, floor honoured on a tiny terminal, and matches
  the historical fixed-tier reference points (`h=40, 18/40 → 18`; `12/40 → 12`);
- `_weighted_panel_widths`: proportional split, widths+gaps sum to usable width, grows with
  terminal width, narrow terminal returns full usable per panel;
- `render_distance_ascii`: header separator length matches the requested width at both wide and
  short widths (the classic off-by-one geometry bug).

---

## 12. Adjacent, not part of this stack

`gareus_monitor.py` (4961 lines, untracked) is a **separate, independent** TUI —
Textual → Rich → ANSI fallback, its own `render*` / `_rich_*_panel` functions, `KT_KCAL`
constant, and no import of `gareus.tui` / `gareus.colors`. It duplicates the design intent
(panels, replica colouring, boost histograms) with none of the shared code. Unifying it onto
this stack is available work, not part of this extraction.
