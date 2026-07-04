# GaMD Boost Detail Monitor Design

Date: 2026-07-04
Target: `RUNS/gareus_monitor.py`

## Goal

Extend selected-run detail view in monitor to show live GaMD boost distribution
from recent `progress.jsonl` `distances` events, with compact plot and numeric
summary that helps judge reweighting quality at glance.

## Non-Goals

- Do not change fleet table, YAML view, MBAR diagnostics view, or run discovery.
- Do not require new dependencies.
- Do not fabricate distribution shape from `mu`/`sigma` alone.
- Do not read historical analysis artifacts or parquet outputs.
- Do not change producer schema or simulation logging.

## Constraints

- Use live values only.
- Source distribution from recent `distances` events in `progress.jsonl`.
- Keep existing summary metrics (`boost mu`, `boost sigma`, `anharmonicity`,
  `umbrella bias`) intact.
- Read by tail scan, not full-file parse beyond existing monitor approach.
- Keep fallback explicit when no recent boost samples exist.
- Keep render compact enough for ANSI detail panel width budget.
- Scope stays inside monitor script and monitor tests.

## Data Source

Monitor already extracts per-sample biased CV data through
`parse_distances_samples(entries)`, including `gamd_boost_total_kcal_mol` as
`boost`.

New detail-view distribution must reuse same live source class:

- scan recent `progress.jsonl` tail
- select `distances` events
- collect recent `boost` values
- cap retained sample count for speed and deterministic layout

Preferred retention:

- parse enough tail to find recent mixed monitor events
- keep at most last ~2000-6000 boost samples in memory for detail rendering

## User-Facing Behavior

Selected-run detail view gains new block after existing CV/bias summary:

- title line: `GaMD boost dist`
- one compact ANSI histogram using recent live boost samples
- one numeric summary line:
  - `n`
  - `min`
  - `p10`
  - `p50`
  - `p90`
  - `max`
- one quality line reusing current live summary metrics when present:
  - `mu`
  - `sigma`
  - `anharm`

Fallback behavior:

- if no recent live boost samples exist, render explicit
  `(no live boost samples)`
- existing CV/bias row still renders current summary metrics

## Rendering

ANSI detail panel only for this change.

Histogram requirements:

- fixed-width terminal-safe glyph plot
- use monitor's existing ANSI palette
- no extra labels beyond x-axis endpoints if space too tight
- tolerate narrow terminal widths by clipping gracefully

Numeric summary requirements:

- values formatted in kcal/mol
- percentiles use existing nearest-rank helper style
- if fewer than 2 samples, histogram may collapse but summary still shows exact
  available values

Quality coloring:

- reuse current sigma thresholds: warn `> 3`, bad `> 5`
- reuse current anharmonicity thresholds: warn `> 0.3`, bad `> 0.5`
- raw histogram itself can stay neutral/cyan unless future tuning needed

## Architecture

Keep implementation local to `RUNS/gareus_monitor.py`:

- add pure helper to summarize boost values:
  - sample extraction
  - percentile stats
  - histogram binning/render input
- add `PeptideState` accessor for live boost sample tail
- update `render_detail()` to render new block from accessor output

No change needed to `RunSnapshot` or fleet summary because feature is
selected-run detail only.

## Error Handling

- Missing `progress.jsonl`: behave same as current monitor, render fallback.
- Mixed malformed entries: skip bad values silently, keep valid samples.
- Empty or constant distributions: render stable degenerate histogram.
- Terminal width pressure: prefer dropping axis text before wrapping panel.

## Testing

Use test-first flow.

Add tests in `RUNS/test_gareus_monitor.py` for:

- boost samples extracted from mixed `progress.jsonl` entries
- summary stats from known boost list
- detail render includes `GaMD boost dist` and percentile labels when samples
  exist
- detail render includes `(no live boost samples)` when samples absent

Verification after implementation:

- `python -m unittest ../RUNS/test_gareus_monitor.py`
- `python -m py_compile ../RUNS/gareus_monitor.py ../RUNS/test_gareus_monitor.py`
- `git diff --check -- ../RUNS/gareus_monitor.py ../RUNS/test_gareus_monitor.py ../docs/superpowers/specs/2026-07-04-gamd-boost-detail-monitor-design.md`

## Scope Check

Single subsystem only:

- one monitor detail-view enhancement
- one data path
- one test file

No further decomposition needed.
