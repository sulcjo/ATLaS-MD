# Monitor Per-Window Boost Design

Date: 2026-07-05
Target: `RUNS/gareus_monitor.py`

## Goal

Extend monitor detail views in ANSI, Rich, and Textual modes to show
per-window live GaMD boost distributions when explicit window or replica ids
are present in live `progress.jsonl` `distances` payloads.

Existing aggregate live boost distribution remains available.

## Non-Goals

- Do not infer window assignment from CV coordinates.
- Do not change producer schema.
- Do not change fleet table, YAML view, or MBAR diagnostics view.
- Do not require `rich` or `textual`; ANSI fallback remains first-class.
- Do not remove aggregate boost distribution when explicit ids are absent.

## Constraints

- Use live values only.
- Source per-window grouping only from explicit ids carried in live payloads.
- If explicit ids are absent, show aggregate distribution plus explicit
  unavailable notice.
- Keep existing summary metrics (`boost mu`, `boost sigma`, `anharmonicity`,
  `umbrella bias`) intact.
- Read by tail scan, not full-file parse beyond existing monitor approach.
- Keep scope inside monitor script and monitor tests.
- Preserve optional Textual/Rich import behavior.

## Data Source

Current live source is `parse_distances_samples(entries)`, which already reads:

- `primary_cv_value` / `cv_A`
- `secondary_cv`
- `umbrella_bias_kcal_mol`
- `gamd_boost_total_kcal_mol`

This change extends sample preservation to include explicit identity keys when
present, for example:

- `window`
- `window_index`
- `state_id`
- `replica`

Implementation must not invent ids. It may normalize any explicit id into one
stable display key, but only from fields actually present in payload rows.

## User-Facing Behavior

All detail views keep existing aggregate `GaMD boost dist` block.

If explicit ids are present in live samples:

- show new `per-window boost dist` section
- one compact row per explicit id
- each row contains:
  - window label
  - compact histogram
  - small numeric summary such as `n`, `p50`, `p90`

If explicit ids are absent:

- aggregate boost section still renders
- per-window section renders one clear note:
  `per-window boost dist unavailable: no explicit window ids in live payload`

## Rendering By UI

### ANSI

- keep current aggregate histogram block
- append compact `per-window boost dist` block below it
- each window row must remain width-safe under current `_fill()` clipping
- if too many windows exist, show only top window rows that fit current panel
  budget and note hidden count

### Rich

- add structured per-window boost panel or table inside selected-run detail
- each row shows label, histogram text, and brief stats
- preserve current selected-run layout and diagnostics ordering

### Textual

- Textual detail reuses Rich detail content path, so per-window section should
  appear there automatically once Rich detail panel is updated
- do not add Textual-only state or widgets unless required

## Architecture

Keep implementation local to `RUNS/gareus_monitor.py`:

- extend `parse_distances_samples()` to preserve explicit ids
- add pure grouping helper:
  - consumes parsed live samples
  - emits ordered per-window boost sample groups
  - ignores samples without explicit ids
- add pure render helpers for compact per-window summaries reusable by ANSI and
  Rich
- update ANSI `render_detail()`
- update Rich `_rich_detail_panel()`

Textual should inherit Rich detail changes through existing shared adapter flow.

## Fallback And Error Handling

- missing `progress.jsonl`: same as current monitor, no live samples
- malformed ids: ignore affected sample, keep valid ones
- explicit ids missing on all live samples: show unavailable note, not empty
  histogram row pretending success
- constant or tiny distributions: render stable degenerate histogram
- many windows: clip output rather than widen panel

## Testing

Use test-first flow.

Add tests in `RUNS/test_gareus_monitor.py` for:

- parser preserves explicit id fields when present
- grouping helper builds per-window boost lists only from explicit ids
- grouping helper ignores samples with missing ids
- ANSI detail render shows per-window section and window rows when ids exist
- ANSI detail render shows explicit unavailable note when ids do not exist
- Rich detail panel text includes per-window section when ids exist
- Rich detail panel text includes unavailable note when ids do not exist

Verification after implementation:

- `python -m unittest ../RUNS/test_gareus_monitor.py`
- `python -m py_compile ../RUNS/gareus_monitor.py ../RUNS/test_gareus_monitor.py`
- `git diff --check -- ../RUNS/gareus_monitor.py ../RUNS/test_gareus_monitor.py ../docs/superpowers/specs/2026-07-05-monitor-per-window-boost-design.md`
- `python ../RUNS/gareus_monitor.py --ui ansi --once v02_runs`
- if available locally: `python ../RUNS/gareus_monitor.py --ui rich --once v02_runs`

## Scope Check

Single subsystem only:

- one monitor detail enhancement
- shared live-sample/grouping path
- ANSI and Rich render updates
- Textual inherits Rich path

No decomposition needed.
