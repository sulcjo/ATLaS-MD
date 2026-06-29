# GAREUS Monitor TUI Redesign

Date: 2026-06-29
Target: `RUNS/gareus_monitor.py`

## Goal

Redo the monitor UI for clearer fleet status, stronger diagnostics, and better
graphs while preserving current run discovery, progress semantics, and cluster
portability.

The monitor should be useful in three environments:

- `textual` installed: full interactive TUI.
- `rich` installed but not `textual`: live dashboard with richer tables/panels.
- neither installed: current-style stdlib ANSI fallback.

## Non-Goals

- Do not change simulation outputs or producer schemas.
- Do not require `textual` or `rich` for basic cluster use.
- Do not replace MBAR or analysis workflows. Monitor graphs are status and
  diagnostics only.
- Do not remove `--once`, `--interval`, `--extra`, or `--connect`.

## Constraints

- Use lazy optional imports so missing UI libraries never break fallback mode.
- Keep whole-budget progress first-class:
  `G% = (adaptive_runtime_pool.used_ns + live aggregate ns) / total_ns`.
- Keep current `E%` semantics as current segment/epoch progress.
- Read large `progress.jsonl` files by tail scan, not full-file parsing on every
  refresh.
- Cache file mtimes and throttle expensive recursive scans.
- Treat missing, stale, or contradictory data as diagnostics, not silent blanks.
- Keep run discovery compatible with canonical peptide directories and flat
  explicit `--extra` run dirs.

## CLI

Existing commands remain valid:

```bash
python RUNS/gareus_monitor.py RUNS/v02_runs --interval 10
python RUNS/gareus_monitor.py RUNS/v02_runs --once
python RUNS/gareus_monitor.py RUNS/v02_runs --extra chignolin_2d_run7
python RUNS/gareus_monitor.py RUNS/v02_runs --connect chignolin
```

New UI selector:

```bash
python RUNS/gareus_monitor.py RUNS/v02_runs --ui auto
python RUNS/gareus_monitor.py RUNS/v02_runs --ui textual
python RUNS/gareus_monitor.py RUNS/v02_runs --ui rich
python RUNS/gareus_monitor.py RUNS/v02_runs --ui ansi
```

Default: `--ui auto`.

Selection rules:

- `auto`: prefer Textual, then Rich, then ANSI.
- `textual`: require Textual or fail with clear install/fallback message.
- `rich`: require Rich or fail with clear install/fallback message.
- `ansi`: force stdlib fallback.

## Architecture

Keep `RUNS/gareus_monitor.py` as the user-facing executable. Internally split it
into focused layers in the same file first, so deployment remains one-file:

- `MonitorConfig`: CLI-derived options.
- `RunDiscovery`: canonical and flat run discovery.
- `RunLoader`: cached reads from progress, pool, driver summary, diagnostics,
  registry, and checkpoints.
- `RunSnapshot`: normalized per-run state used by all UI modes.
- `FleetSnapshot`: aggregate state for fleet summary.
- `Diagnostic`: structured issue with severity, code, message, source path, and
  suggested action.
- `GraphData`: pure data for budget timelines, sparklines, heatmaps, exchange
  bars, and epoch progression.
- UI adapters:
  - `TextualMonitorApp`
  - `RichMonitorView`
  - `AnsiMonitorView`

Parsing and diagnostic logic must stay pure and testable. UI adapters render
snapshots only.

## Textual Layout

Primary screen:

```text
+--------------------------+-----------------------------------------------+
| Fleet status             | Selected run                                  |
| phase counts             | budget timeline, current phase, ETA           |
| total ns / budget / ETA  | throughput sparkline, latest message          |
+--------------------------+-----------------------------------------------+
| Runs table                                                               |
| name phase G% E% budget epoch ckpt MBAR qual stale top_issue             |
+-------------------------------------------------------------------------+
| Diagnostics                                                             |
| severity run code message source/action                                  |
+-------------------------------------------------------------------------+
```

Navigation:

- Up/down or j/k: select run.
- Enter/d: detail view.
- m: MBAR/adaptive diagnostics.
- c: live attach.
- f: filter by severity/status.
- s: sort by top issue, G%, ETA, phase, name.
- r: refresh.
- q/Esc: back or quit.

## Views

Fleet view:

- Compact at-a-glance table.
- Top issue column instead of leaving users to infer from raw numbers.
- Fleet progress summary across runs with known total budgets.
- Phase counts: production, adaptive feedback, done, error, pending.

Detail view:

- Global budget bar and segmented epoch/topup/final timeline.
- Current epoch progress and ETA.
- Throughput sparkline from recent aggregate simulated ns.
- CV1/CV2 ranges.
- GaMD/reweighting panel: Boost mu, Boost sigma, anharmonicity, umbrella bias.
- Latest message with stale data marker.

MBAR/adaptive diagnostics view:

- Connectivity status at overlap threshold.
- Weak edge count and worst edges.
- Coverage summary: min, p10, median, max, zero-window count.
- Window registry: active, usable, retired, total.
- CV1/CV2 coverage heatmap.
- Epoch progression: windows, actions, topups, consumed ns.

Live attach view:

- Faster refresh cadence for selected run.
- Freshest-per-field tail merge from `progress.jsonl`.
- Exchange acceptance bars by jump distance.
- Live window map.
- Approximate basin hints must remain labelled as approximate and not MBAR PMF.

## Diagnostics

Diagnostics are structured and color-coded:

- `error`: likely run failure or scientifically unsafe analysis state.
- `warn`: degraded sampling, weak overlap, stale data, or incomplete metadata.
- `info`: normal but useful status.

Initial diagnostic codes:

- `missing_progress`: no `progress.jsonl`.
- `stale_progress`: latest wall time older than threshold.
- `missing_pool`: no `adaptive_runtime_pool.json`.
- `pool_overrun`: used/live ns exceeds total budget beyond tolerance.
- `pool_blank_epoch0`: top-level pool exists but no events yet.
- `mbar_disconnected`: overlap graph disconnected.
- `mbar_weak_edges`: neighbor overlap below target.
- `mbar_zero_coverage`: at least one active window has zero samples.
- `gamd_anharm_high`: anharmonicity above warning/error threshold.
- `gamd_sigma_high`: boost sigma above warning/error threshold.
- `checkpoint_missing`: production progress exists but checkpoint manifests are
  absent or unexpectedly old.
- `driver_done`: adaptive driver reports done.
- `no_completed_epoch_diag`: first epoch diagnostics not available yet.

Each diagnostic includes source path when known.

## Graphs

Use Textual/Rich primitives for terminal-safe graphs:

- Budget timeline: baseline, topup, final, free budget segments.
- Throughput sparkline: recent aggregate ns over time.
- CV1/CV2 heatmap: coverage glyphs by window center.
- Exchange bars: accepted/attempted by jump distance.
- Epoch progression strip: actions/topups/consumed ns by epoch.

ANSI fallback uses existing glyph-based renderers and shares the same graph data.

## Testing

Use test-first implementation.

Pure tests:

- UI mode selection falls back correctly.
- `RunSnapshot` preserves G% and E% semantics.
- Diagnostics emit expected codes for missing/stale/pool/MBAR/GaMD cases.
- Graph data handles epoch 0 with empty pool events.
- Flat `--extra` directories still work.
- ANSI width handling strips escape codes.
- Textual/Rich imports are lazy and optional.

Render smoke tests:

- `--once --ui ansi` produces stable output without optional dependencies.
- `--once --ui rich` works when Rich is installed.
- Textual app can instantiate and build widgets when Textual is installed.

Manual verification:

- Run `python RUNS/gareus_monitor.py RUNS/v02_runs --once --ui ansi`.
- Run `python RUNS/gareus_monitor.py RUNS/v02_runs --once --ui rich` if Rich is installed.
- Run `python RUNS/gareus_monitor.py RUNS/v02_runs --ui textual` if Textual is installed.

## Migration Plan

1. Add failing tests around UI selection, snapshot semantics, diagnostics, and
   graph data.
2. Extract current pure parse/load behavior into snapshot/diagnostic helpers
   without changing CLI behavior.
3. Add `--ui` mode selection and keep ANSI path passing.
4. Add Rich view.
5. Add Textual app.
6. Improve diagnostics and graph presentation.
7. Verify syntax, tests, ANSI once render, and optional Rich/Textual paths when
   installed.

## Risks

- Textual may be absent on cluster. Mitigation: lazy imports and ANSI fallback.
- Large run trees can make refresh slow. Mitigation: tail reads, mtime caches,
  throttled checkpoint/epoch scans.
- UI can hide scientific failure signals behind colors. Mitigation: top issue
  column and structured diagnostics list with source paths.
- One-file implementation can remain large. Mitigation: internal layer split
  first; move to modules only if size blocks maintainability or user accepts
  multi-file deployment.
