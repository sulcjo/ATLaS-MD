# GAREUS live-dashboard TUI screen redesign — design spec

Date: 2026-08-19
Status: design approved in chat, awaiting spec review
Scope: the in-run live dashboard frame (`DistanceLogger._render_dashboard`, `gareus/logger.py:2398`).
Companion reference: `docs/GAREUS_tui_design.md` (inventory of the TUI subsystem as it exists today).
Predecessor: `docs/superpowers/specs/2026-07-16-live-dashboard-tui-rework-design.md` — implemented;
see §17 for what it delivered and where this design supersedes it.

---

## 1. Problem

The live dashboard produces far more content than the terminal can display, and the
overflow is discarded by position rather than by importance. Measured by driving the real
`DistanceLogger._render_dashboard` with synthetic-but-realistic data — 25 windows,
populated per-window CV histories, a full neighbour exchange table — at fixed terminal
sizes. Reproduce with `python tests/manual_render_dashboard_frame.py`; that harness is also
the intended fixture for the fit property test in §14.1:

| Terminal | Content lines produced | Lines displayable | Silently discarded |
|---|---|---|---|
| 200×55 | 81 | 54 | 27 (33%) |
| 140×45 | 91 | 44 | 47 (52%) |
| 200×55, 2D run | 97 | 54 | 43 (44%) |

Five structural defects, all reproducible in that render:

1. **Truncation is positional, not severity-ranked.** Every list panel shows its first
   *k* entries and hides the rest (`histogram overlap` shows `w00-w01 … w14-w15` then
   `… 9 more`; `umbrella pull` shows `w00 … w13` then `… 12 more`). A failing window at
   w20 is invisible. This is worse than crowding: the screen is confidently wrong.
2. **Panel chrome costs 4 lines each** (top border, title, separator, bottom). Eight
   panels ≈ 32 of 81 lines are borders.
3. **`_join_columns` pads every column to the tallest panel** (`gareus/tui.py:226`), so a
   one-line panel reserves a full-height column of blank space next to a 21-line neighbour.
4. **Repetition consumes the width.** The CV table prints an identical `k` value 25 times,
   `nw=300` 25 times, and the window index twice per row (`r00 w00`).
5. **No visual hierarchy.** The verdict is one plain line above ~70 lines of
   identically-weighted boxes.

Root cause is control inversion: panels render blind, then `_safe_tui_frame_text`
(`gareus/tui.py:88`) clips the assembled frame at write time. `dashboard_body_budget`
(`gareus/tui.py:386`) was an attempt at top-down allocation, but the fractions in use
(`18/40 + 12/40 + 14/40 + 22/40`) sum to 165% of the screen — the mechanism exists and is
over-subscribed.

## 2. Goals

- The frame fits the terminal **by construction**; `_safe_tui_frame_text`'s clip becomes a
  safety net that never fires in normal operation.
- What is hidden is always the *least severe* content, and the fact that something is
  hidden is stated with its severity composition.
- Serve the three decisions the operator actually makes mid-run (§3) with one screen each,
  rather than one screen attempting all three.
- Surface, live, several run-invalidating conditions that today are only visible in
  post-run analysis (GaMD envelope saturation, MBAR graph disconnection, mistuned `k`).
- Reduce `gareus/logger.py` (2779 lines) to data collection; move rendering into focused
  modules of 150-450 lines each.

## 3. Non-goals

- The multi-run fleet monitor (`gareus_monitor.py`, 4961 lines) is out of scope. It is a
  separate TUI with its own Textual/Rich/ANSI stack and no import of `gareus.tui`.
  Unifying the two is deliberately deferred.
- No new post-run analysis. Where a live panel reuses an analysis that exists in
  `analyze_gareus_mbar.py`, it implements the cheap live version only.
- No permanent legacy rendering path.

## 4. Operator jobs this screen serves

Confirmed with the user, in priority order:

1. **Watch progress / plan time** — how far in, how long left, epoch schedule, MD-pool
   spend, when the final phase starts.
2. **Confirm physics is valid** — neighbour overlap, boost σΔV against calibration target,
   anharmonicity, CV2 coverage, MBAR connectivity.
3. **Diagnose a specific window** — which window or replica is stuck, pinned, NaN'd, or
   never exchanging, and why.

Explicitly *not* a job for this screen: deciding whether to kill the run.

## 5. Architecture

### 5.1 Frame anatomy

```
spine    10 lines, identical in every view (5 lines in the compact tier)
body     H - 11 lines, owned by exactly one view
footer    1 line, view tabs + clock + names of any dropped panels
```

Render target height is `term_h - 1`, matching `_safe_tui_frame_text`'s own reserve.

### 5.2 Top-down allocator

Each panel declares `(min_lines, want_lines, priority)`. Allocation:

1. Grant every panel its `min_lines` in priority order, where **priority 1 is highest**
   and ties break by declaration order within the view.
2. Panels that do not fit are dropped and **named in the footer** — never silently.
3. Distribute the remaining budget as `want_lines - min_lines`, in priority order.

The frame therefore fits exactly. This inverts today's control flow: the screen tells
panels their budget instead of panels rendering blind.

### 5.3 Severity-ranked truncation

Every list panel carries a rank key. Truncation keeps the *worst* k entries; the tail line
reports composition, e.g. `+24 more (all ok, min 0.31)` versus
`+9 more (2 warn, 1 bad)`.

### 5.4 Two data tiers

- **Per-frame, in memory** — replica rows, exchange stats, CV/PE histories. As today.
- **Sidecar, mtime-polled at most every 5 s** — campaign state that changes at epoch
  boundaries, read from JSON already written to disk:
  - `adaptive_production/adaptive_runtime_pool.json` — `total_ns`, `used_ns`,
    `remaining_ns`, and an `events[]` ledger (per segment: `label`, `kind`,
    `consumed_ns`, `n_states`, `steps_per_state`, `remaining_ns_after`).
  - `global_shared_gamd_setup/shared_gamd_setup_globals.json` — `sigma0_*` (target),
    `sigmaV_*` (achieved), `k0_*`, `joint_envelope`, `recalibration_history`.
  - `adaptive_production/adaptive_quality_gate.json`, and the umbrella-seeding quality
    report (for `pull_crash_fallback_unpulled` flags).

Rationale: campaign state moves on a timescale of minutes to hours while CV samples move
every report interval. An mtime check costs nothing and buys a whole view; threading the
same values through `gareus/production.py`'s hot loop would touch the MD path for data
that barely changes.

### 5.5 View selection

New flag `--tui-view auto|progress|physics|windows`, default `auto`.

- `auto` shows PROGRESS, and **promotes** to another view only while a promotion rule
  holds — warning-light behaviour, so a headless SLURM log surfaces problems by itself.
  Promotion rules: any BAD-ranked window; a dead exchange pair; boost anharmonicity > 1.0;
  a disconnected MBAR graph.
- Keys `1`/`2`/`3`/`Tab` are enabled only when `sys.stdin.isatty()`, using the termios
  raw-mode pattern already proven in this repo at `gareus_monitor.py:4490-4542`. This
  finally gives `--tui-mode interactive` a distinct meaning; today it is a dead alias
  appearing only inside `{"dashboard", "interactive"}` sets.
- No timed rotation. A screen that moves while being read is worse than one that needs a
  keypress.

## 6. The spine (10 lines)

**On the numbers in this spec's mockups**: values are real `chignolin_5` measurements
wherever §9's data audit marks the source as *exists* (pool ledger `7500/15000 ns` and its
per-segment durations, `sigma0 12.552`, `sigmaV 11.039`, `k0 1.00`, `cv2 tica-linear`
switched after epoch 0, 29 states). Everything sourced from a row §9 marks as *new* is
illustrative and shaped to be plausible — per-window `σΔV`, per-CV2-row sample share,
exchange-stall percentage, GPU-day conversion, and the `w17`/`w18` failure narrative.
Layout geometry (line counts, column widths) is measured in all cases.

Measured at 140 cols: max line 121 of a 138-col budget, leaving slack that the elastic
middles consume on wider terminals.

```
GaREUS  chignolin_5   ep 1/2  epoch_001/topup_002_8178000   29 win  2D sparse  cv2 tica-linear      ⚠ CAUTION 2
run   [##############################------------------------------]  33.2%   12.45/37.50 Msteps   wall 6h00m  eta 12h04m
pool  [##############################------------------------------]  7500/15000 ns   perf 104 ns/d/rep   3.0k aggregate
cv1   3.9 |▁▂▃▅▆▇███▇▆▅▄▃▄▅▆▇██▇▆▅▄▃▂▁▁▂▃▄▅▆▇█▇▆▅▄▃▂▁·▁▂▃▄▅▆▇▆▅▄▃▂▁| 17.2 A     span 13.3   min ovl w17-w18 0.02
cv2  -2.4 |▂▃▅▇█▇▅▃▂·▁▂▄▆█▆▄▂▁·▁▃▅▇█▇▅▃▁·▁▂▄▆█▆▄▂▁·▁▂▃▅▇█▇▅▃▂▁·▁▂▃| +2.4       5 rows   worst row 3  share 9%
win   |▅▆▇█▇▆▅▄▃▄▅▆▇█▇▆▅▄▃▄▅▆▇█▇▆▅▄▃|  samples 4.2-9.8 M/win   min w17 4.2M (43% of median)
exch   ╵▃▄▅▄▃▂▁·▁▂▃▄▅▄▃▂▁·▁▂▃▄▅▄▃▂▁╵   accept 0.31 mean, 0.04 min   1 dead pair w17-w18
gamd  σΔV 11.04 / σ0 12.55 kJ (88%)   k0 1.00 SATURATED   anharm 1.74 HIGH   boost 2.4±1.9 kcal/mol
alert w17 pinned 3.2 ns, no exchange   |   gamd anharm 1.74 > 1.0 -> cumulant reweighting unreliable
──── [1] PROGRESS   [2] physics   [3] windows ─────────────────────────── epoch 1 t+6h00m ─── 20:07:14 ────
```

| # | Line | Job |
|---|---|---|
| 1 | run identity, epoch/segment, window count, topology, CV2 type, verdict + issue count | orientation |
| 2 | run progress bar, percent, steps, wall, ETA | plan time |
| 3 | MD-pool spend bar, ns used/total, throughput | plan time |
| 4 | CV1 sample density over the CV axis, span, worst neighbour overlap | physics |
| 5 | CV2 sample density over the CV2 axis, row count, worst row (2D runs only) | physics |
| 6 | per-window sample-count strip, range, starved window | diagnose |
| 7 | per-neighbour-pair acceptance strip, mean/min, dead-pair count | physics |
| 8 | GaMD envelope: σΔV vs σ0, k0 saturation, anharmonicity, boost magnitude | physics |
| 9 | top-2 ranked alerts, worst first, naming the window | diagnose |
| 10 | view tabs, epoch clock, wall clock | navigation |

Two mechanics that constrain implementation:

**Strip bucketing.** Lines 6-7 draw one glyph per window/pair, which exceeds the width
past ~120 windows; this repo has real 364-window runs. When `n_windows > available cells`,
cells aggregate: glyph height is the bucket mean, **status is the bucket's worst**. A bad
window is never hidden, only coarsened in position. Lines 4-5 are unaffected — they bin a
continuous axis already.

**Width elasticity.** Every spine line is `label + elastic middle + fixed-width right
summary`. The middle is sized from `usable_w` minus the *measured* fixed parts, following
the pre-measure discipline that today's `cv_bar_width = cv_panel_w - 72` gets right.

**Line 7 is deliberately aligned under line 6**: pair *i* is offset half a cell so it sits
between windows *i* and *i+1*. A dead pair then reads as a visual seam between two window
columns. Today these live in separate boxes ~20 lines apart, destroying the spatial
relationship that shows where the MBAR graph is about to disconnect.

## 7. Views

Each view must fit its worst case in 24 body lines (a 35-line terminal) and expand to 44
(a 55-line terminal). Measured drafts: 21 / 18 / 21 lines, ≤ 114 cols.

### 7.1 View 1 — PROGRESS

```
PROGRESS   28 pool events   projection at current throughput                              [2] physics  [3] windows
┌ campaign timeline ─────────────────────────────────────────────────────────────────────────────────────────┐
│ epoch_000          ████████████████████████                              1875.0 ns  27 st  ✓ done         │
│ final/baseline     ▏                                                       14.5 ns  29 st  ✓ done         │
│ final/topup_001    ▎                                                       43.5 ns  29 st  ✓ done         │
│ epoch_001/baseline ██████████████████                                    1391.7 ns  29 st  ✓ done         │
│  ↳ ext round 2     █████████████                                         1029.3 ns  29 st  ✓ done         │
│  ↳ ext round 3     █████████                                              757.5 ns  29 st  ✓ done         │
│  ↳ ext round 4-7   ██████████                                            1105.9 ns  29 st  ✓ done         │
│ ep1/topup_001      ████                                                   365.9 ns  27 st  ✓ done         │
│ ep1/topup_002      ▌                                            ◀ NOW       40.7 ns   2 st  ▶ running     │
│ final (reserve)    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  7500.0 ns  29 st  ○ scheduled    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┌ projection ──────────────────────────────────┐  ┌ throughput ───────────────────────────────────────────┐
│ perf now        104 ns/day/rep               │  │ ns/day/rep, last 2h    ▅▆▇█▇▆▅▆▇█▇▆▅▄▅▆▇█▇▆  μ 98    │
│ pool remaining  7500 ns = 2.4 GPU-days       │  │ slowest replica  r07 at 91 ns/day (-12%)              │
│ this segment    eta 12h04m                   │  │ exchange stall   0.4% of wall                         │
│ epoch 1 ends    eta ~14h                     │  │ samples          8.18M total, +2.1M this segment      │
│ final phase     ~2.9 d after that            │  │ write backlog    (deferred, see §9)                   │
│ budget check    OK — 0 ns overcommitted      │  └───────────────────────────────────────────────────────┘
└──────────────────────────────────────────────┘
```

Panels: `campaign timeline` (priority 1, min 6, want 12), `projection` (2, min 4, want 7),
`throughput` (3, min 3, want 6).

The timeline is the pool `events[]` ledger, chronological, including quality-gate
extension rounds — visible live for the first time. Consecutive extension rounds of the
same segment collapse into one row (`↳ ext round 4-7`) when the budget is tight.

### 7.2 View 2 — PHYSICS

```
PHYSICS   MBAR readiness 27/29 states connected  ⚠ 2 isolated                          [1] progress  [3] windows
┌ neighbour overlap — worst first ────────────────────────────────┐  ┌ GaMD boost envelope ──────────────────┐
│ pair        overlap   target 0.30                               │  │ ΔV vs Gaussian reference              │
│ w17-w18       0.02   ░░░░░░░░░░  DEAD  → MBAR graph breaks     │  │        ▁▃▆█▇▅▃▂▁                      │
│ w23-w24       0.11   ███░░░░░░░  BAD                           │  │     ·▁▂▄▇███▆▄▂▁·                     │
│ w05-w06       0.19   ██████░░░░  WARN                          │  │  ▁▂▄▆████████▆▄▂▁      observed       │
│ w11-w12       0.27   █████████░  ok                            │  │  ┈┈┈╌╌━━━━━━━━╌╌┈┈┈    gauss(2.4,1.9)│
│ +24 more (all ok, min 0.31, median 0.44)                       │  │ skew +0.96  kurt +1.13  anharm 1.74 ⚠ │
│                                                                 │  │ σΔV 11.04 / σ0 12.55 kJ  (88%)        │
│ σ-vs-spacing check                                              │  │ k0 1.00  SATURATED at ceiling         │
│   k 2.5 → σ 0.44 A   spacing 0.55 A   predicted overlap 0.42    │  │ recal after ep0: k0 1.00→1.00 no-op   │
│   observed median 0.31  → k ~12% too stiff for this spacing     │  │ boost active 96% of steps             │
└─────────────────────────────────────────────────────────────────┘  └───────────────────────────────────────┘
┌ CV2 regime + graph connectivity ───────────────────────────────────────────────────────────────────────────┐
│ cv2 tica-linear   switched at epoch_000 end (cv_version 2)   prior torsion-pca: 3.13M samples, split out   │
│ rows -2.4 .. +2.4 (5 targets)   sample share 22% 26% 9%◀ 23% 20%   row 3 starved, k2 5.0 likely too stiff  │
│ graph 27/29 connected   isolated: w17 w18   → union MBAR drops 2 states unless a bridge window is added    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Panels: `neighbour overlap` (1, min 5, want 12), `GaMD boost envelope` (1, min 5, want 11),
`CV2 regime + connectivity` (2, min 3, want 4).

Two panels are new *analysis*, not new rendering:

- **σ-vs-spacing check.** From the harmonic restraint, `σ = sqrt(k_B T / k)`; compare with
  actual neighbour spacing to predict overlap, and report the ratio against observed
  median overlap. This tells the operator on day 1 that `k` is mistuned, instead of after
  the run from an overlap report.
- **Graph connectivity.** Union-find over `exchange_stats` edges with non-negligible
  acceptance, reporting largest-connected-component vs total states. MBAR disconnection is
  a documented real failure mode for this pipeline; this makes it a live verdict.

### 7.3 View 3 — WINDOWS

```
WINDOWS   sorted worst-first   auto-selected w17   [j/k] move  [s] sort  [f] filter    [1] progress  [2] physics
┌ windows — worst first ─────────────────────────────────────────────────────────────────────────────────────┐
│ win   cv1 ctr   cv2 ctr   samples   accL   accR   |Δ|max   σΔV    CV drift        status                   │
│ w17     13.35     -1.00     4.2M   0.04   0.00     2.31   17.2   ▂▂▁▁▁▁▁▁▁▁  ●   BAD  pinned, dead R      │
│ w18     13.90     -1.00     4.4M   0.00   0.21     1.88   16.9   ▁▁▂▂▂▃▃▃▂▂      BAD  dead L              │
│ w23     16.65     +1.00     5.1M   0.11   0.19     0.94   12.1   ▃▄▅▄▃▄▅▄▃▄      WARN low accept          │
│ w05      6.75     -2.00     6.8M   0.19   0.24     0.61   11.4   ▄▅▆▅▄▅▆▅▄▅      WARN                     │
│ w11     10.05      0.00     7.2M   0.27   0.31     0.44   10.8   ▅▆▇▆▅▆▇▆▅▆      ok                       │
│ +24 more ok   samples 6.1-9.8M   accept 0.28-0.55   |Δ| < 0.7   σΔV 9.8-12.4                              │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
┌ w17 detail ────────────────────────────────────────────────────────────────────────────────────────────────┐
│ restraint   cv1 13.35 A  k 2.5 kcal/mol/A²      cv2 -1.00  k2 5.0 kcal/mol                                │
│ occupancy   replica r17 for 3.2 ns, never swapped   windows visited: 17 only                               │
│ cv1 dist    11.0 |·····▁▂▃▅███▇▅▃▂▁·······| 15.5    mean 13.90  Δ +0.55  drifting into w18 range          │
│ cv2 dist    -2.4 |······▁▃▆█▅▂▁···········| +2.4    mean -1.42  Δ -0.42                                   │
│ boost       ΔV 3.1 ± 2.4 kcal/mol   σΔV 17.2 kJ (56% above run median)   anharm 2.31 HIGH                 │
│ exchange     w16 0.04 (2/48 attempts)    w18 0.00 (0/47) DEAD                                             │
│ seeding     start structure: pull ok, no crash fallback   (flag would appear here)                         │
│ reading     pinned against a steric barrier; contact target likely unreachable at cv2 = -1.0              │
│ options     retire or split w17  ·  loosen k2  ·  add bridge window at cv1 13.6 / cv2 -0.5                │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Panels: `windows — worst first` (1, min 4, want 16), `<win> detail` (1, min 6, want 11).

Selection: keys `j`/`k` when a tty is present; otherwise the **worst-ranked window is
auto-selected**, so a headless log always shows the problem window expanded.

`reading` and `options` are rule-derived text, not free-form prose: each is a lookup from
the matched severity rule to a fixed sentence with the window's own numbers interpolated.

## 8. Ranking and hysteresis

Severity is the first matching rule, evaluated per window:

| Order | Condition | Status |
|---|---|---|
| 1 | either neighbour acceptance < 0.02 | BAD (`dead L`/`dead R`) |
| 2 | either neighbour overlap < 0.10 | BAD |
| 3 | \|Δ\| from restraint centre > 2σ, σ = sqrt(k_B T / k) | BAD (`pinned`) |
| 4 | cumulative samples < 50% of the per-window median | WARN (`starved`) |
| 5 | per-window boost anharmonicity > 1.5 | WARN |
| 6 | acceptance < 0.15 | WARN (`low accept`) |
| — | none of the above | ok |

Ties break by window index, so ordering is stable frame to frame. **Hysteresis:** once a
window occupies a ranked slot it keeps it until its trigger has been clear for N
consecutive frames (default 5). Without this the table re-sorts on noise and the row being
read moves out from under the reader — the usual reason live-sorted TUIs get ignored.

## 9. Data audit

| Requirement | Status |
|---|---|
| pool ledger; σ0/σΔV/k0 + recalibration history; seeding-quality flags; quality gate | **exists** — sidecar JSON on disk |
| neighbour overlap, per-pair acceptance, \|Δ\|, CV/PE histories and sparklines, CV2 type/version, window centres and k | **exists** in `gareus/logger.py` / `dashboard_info` (built at `gareus/production.py:4459`) |
| per-window cumulative sample counts | new counter in `_update_history`, restored on resume (§11) |
| per-window σΔV and anharmonicity | new bounded per-window boost deque |
| ns/day history; per-replica window-visit set; exchange stall timing | new small in-memory accumulators |
| σ-vs-spacing prediction; graph connectivity; per-CV2-row sample share | new derived computations, all O(K) |
| parquet write backlog | **deferred** — needs `gareus/store.py` writer state plumbed to the logger; not implemented in this design |

No item requires changes to the MD stepping path.

## 10. Degradation

By height:

| Height | Behaviour |
|---|---|
| ≥ 40 | full 10-line spine, view body at `H − 11` |
| 28-39 | full spine; lowest-priority panels dropped and named in the footer |
| 20-27 | **compact spine (5 lines)**: identity+verdict, run bar, pool bar, window strip, top alert |
| < 20 | spine only, one line per essential — close to today's `line` mode |

The 28-line boundary exists because a 10-line spine is 36% of a 28-line terminal; past
that ratio a spine stops being a spine.

By width: below ~100 cols, side-by-side rows stack (today's `_dashboard_weighted_row`
narrow-stacking rule, `gareus/tui.py:314`); strips bucket more aggressively; below ~70
cols the right-hand fixed summaries drop before the bars shrink further.

**Colour.** Status always appears as text (`BAD`/`WARN`/`ok`/`DEAD`/`SATURATED`), never
colour-only; every mockup in this spec is ANSI-stripped and remains readable, which is the
acceptance criterion. In density strips, where colour normally carries status, a flagged
cell swaps its density glyph for `!` (warn) or `X` (bad) when colour is disabled.
Optional `--tui-glyphs unicode|ascii` (auto-detected from locale) for terminals that
mangle box-drawing characters.

**Missing data is a normal state, never an error.** Absent `adaptive_runtime_pool.json`
(non-adaptive run, or before the first epoch closes) replaces the timeline panel with a
one-line explanation; absent GaMD globals (plain umbrella run) replaces the boost panel
with `no GaMD boost (plain umbrella run)` and drops σΔV from the spine; a 1D run drops the
CV2 spine line and the CV2 panel; unreadable or partial JSON shows `unreadable` and never
raises.

## 11. Resume correctness

The WINDOWS table's `samples` column must be cumulative from disk, not since process
start, or it misreports after every SIGTERM resume — and this pipeline resumes routinely.
The new per-window counters and boost accumulators hook
`DistanceLogger.restore_from_previous_outputs` (`gareus/logger.py:408`), which already
rebuilds histories from prior output. Any value that genuinely cannot be restored is
labelled `since resume` rather than presented as a total.

## 12. Module layout and API

| File | ~lines | Contents |
|---|---|---|
| `gareus/tui.py` | 696 → ~500 | unchanged primitives; `render_distance_ascii` and `_render_histogram_row` move out, leaving the module stdlib-only |
| `gareus/tui_screen.py` | ~250 | `Panel`, `allocate()`, spine renderer, view registry, degradation tiers. No domain knowledge |
| `gareus/dashboard/context.py` | ~200 | `DashboardContext` frozen snapshot |
| `gareus/dashboard/sidecar.py` | ~150 | mtime-polled JSON readers |
| `gareus/dashboard/panels.py` | ~450 | reusable content panels ported from `logger.py`'s `_render_*` |
| `gareus/dashboard/view_progress.py` | ~250 | pure `build(ctx)` |
| `gareus/dashboard/view_physics.py` | ~250 | pure `build(ctx)` |
| `gareus/dashboard/view_windows.py` | ~250 | pure `build(ctx)` |
| `gareus/dashboard/ranking.py` | ~150 | severity rules, hysteresis state, tail-composition strings |
| `gareus/logger.py` | 2779 → ~1000 | data collection, history, restore, summarize; no frame assembly |

Measured basis for the split: of `logger.py`'s 2779 lines, 1756 sit in 30
rendering functions and 961 in data collection/restore.

Core types (frozen dataclasses, per the project's immutability convention):

```python
@dataclass(frozen=True)
class Panel:
    key: str
    title: str
    lines: tuple[str, ...]       # already ranked worst-first
    min_lines: int
    want_lines: int
    priority: int
    tail: str = ""               # "+24 more (2 warn, 1 bad)" when trimmed

def allocate(
    panels: Sequence[Panel], budget: int
) -> tuple[tuple[Panel, ...], tuple[str, ...]]:
    """Trim panels to fit `budget` lines; return them plus dropped panel names."""

def build(ctx: DashboardContext) -> tuple[Row, ...]:
    """Each view module implements this. Pure function of a frozen snapshot."""
```

`DashboardContext` holds the per-frame data, the sidecar snapshot, and config derived from
`args`. Because views are pure functions of it, they are testable without OpenMM — the
same property today's tests exploit by passing a bare `argparse.Namespace()`.

## 13. New CLI flags

The redesign needs real flags, which also closes an existing defect class: 16 TUI knobs
are hardcoded in `_shim_output` (`gareus/cli.py:1053-1072`) and unreachable, while two of
them are named to the user in shipped strings that would error if typed
(`--distance-ascii-max-replicas`, printed by `render_distance_ascii`; `--tui-clear-mode
never`, in `tui_clear_enabled`'s docstring at `gareus/tui.py:53`).

Added: `--tui-view auto|progress|physics|windows`, `--tui-glyphs unicode|ascii|auto`,
`--color always|never|auto`, `--tui-clear-mode always|never`,
`--dashboard-density auto|compact|normal|full`, `--distance-ascii-max-replicas N`,
`--dashboard-render-interval-sec S`. Defaults reproduce today's hardcoded values.

## 14. Testing

1. **Fit property test** — for `W ∈ {70,80,100,120,140,200,400} × H ∈ {18,20,24,35,45,55,80}`,
   every view renders ≤ `H − 1` lines and ≤ `W − 2` visible columns. This invariant is
   asserted nowhere today and is the defect that discards 33-52% of the frame.
2. **Allocator** — exact fill, priority order honoured, `min_lines` respected, dropped
   panels reported rather than swallowed.
3. **Ranking** — worst-first ordering, tail-composition counts, and hysteresis across a
   frame sequence (bad → recovered must persist N frames).
4. **Strip bucketing** — 364 windows into 100 cells: a single BAD window still reads BAD.
5. **Sidecar** — missing, stale, truncated, and corrupt JSON each degrade gracefully.
6. **Colour-off** — ANSI-stripped output still contains every status word.
7. **Golden frames** — three views at 140×45 from a fixed synthetic `ctx`, ANSI-stripped,
   committed as snapshots to catch unintended layout drift.
8. **Degradation tiers** — `H=20` selects the compact spine; `H=18` selects spine-only.

All tests reuse the existing harness shape (bare `Namespace` plus monkeypatched
`shutil.get_terminal_size`, as in `tests/test_dashboard_scaling_integration.py`), so no
new test infrastructure is required.

## 15. Migration

`_render_dashboard` is replaced outright. No permanent `--tui-legacy` path (YAGNI); the
golden-frame tests are the regression safety net. The existing 20 tests in
`tests/test_dashboard_tui_scaling.py` and `tests/test_dashboard_scaling_integration.py`
must continue to pass, since they pin the primitives (`dashboard_row_gap`,
`dashboard_body_budget`, `_weighted_panel_widths`, histogram separator width) that this
design keeps using.

`dashboard_heavy_panels_every` (render expensive panels only every Nth frame) is
**superseded** and removed: under the allocator, expensive panels are dropped by priority
and available budget, not by frame parity. `dashboard_render_interval_sec` is kept and
promoted to a real flag, since it throttles whole-frame repaints rather than panel
selection.

## 16. Risks

- **Sidecar staleness.** Pool and GaMD JSON are written at phase boundaries, so early in a
  segment the timeline lags reality by up to one segment. Mitigation: display the file's
  own mtime next to the panel title, so a stale ledger is visibly stale.
- **Per-window boost accumulators cost memory.** 364 windows × a bounded deque; the bound
  must be set from `distance_history_limit`-style config rather than unbounded growth.
- **Rule-derived `reading`/`options` text can be confidently wrong** when a window fails
  for a reason outside the rule table. Mitigation: every such line names the rule that
  fired, so an operator can see the inference rather than trusting a verdict.
- **Keyboard handling under a run.** Raw-mode termios on a process that is also driving
  OpenMM must restore terminal state on every exit path, including SIGTERM. The existing
  `restore_tui_cursor` hooks (`gareus/core.py:18`, `gareus/legacy.py:39`,
  `gareus_peptide.py:25`) are the precedent to extend.

## 17. Relationship to the 2026-07-16 live-dashboard rework

`docs/superpowers/specs/2026-07-16-live-dashboard-tui-rework-design.md` (plan:
`docs/superpowers/plans/2026-07-16-live-dashboard-tui-rework.md`) was implemented in
commits `f57c846`..`05c2b71`. It delivered the foundation this design builds on:

- continuous width/height scaling — `_weighted_panel_widths`, `dashboard_body_budget`,
  removal of the hardcoded bar-width ceilings and the 3-bucket sizing constants;
- semantic colour roles — `role_text`, `severity_role`, `_ROLE_STYLE`;
- unified `MIN_PANEL_WIDTH` and `dashboard_row_gap` constants;
- `render_distance_ascii`'s header separator scaled to the real panel width;
- the 20 layout tests in `tests/test_dashboard_tui_scaling.py` and
  `tests/test_dashboard_scaling_integration.py`, which this design must keep green.

That rework made each panel scale correctly with the terminal. It did not address total
content exceeding the screen, because it explicitly scoped that out: "No change to what
data any panel shows, CLI flag semantics, or dashboard render throttling". The measured
33-52% frame loss in §1 is what remains after its fix landed — every panel now sizes
itself correctly, and there are still more panels than lines.

This design supersedes three of its decisions:

| Prior decision | Now |
|---|---|
| `_dashboard_density` keeps a sizing role narrowed to "which panels are shown at all" | sizing moves entirely to the allocator; density survives only as the compact-spine tier hint (§10) |
| `dashboard_heavy_panels_every` untouched | removed — the allocator drops panels by priority and budget, not by frame parity (§15) |
| "No change to what data any panel shows" | intentionally lifted — new panels and sidecar-sourced data (§7, §9) |

It remains consistent with the rest of that spec's position: no Rich/Textual (hand-rolled
ANSI stays, for the reasons its non-goals give), no changes to `gareus_monitor.py` or
`gareus/helptext.py`, and `dashboard_render_interval_sec` retained.
