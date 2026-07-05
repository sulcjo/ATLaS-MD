# Task 2 Report: Render ANSI Per-Window Boost Section

## What I implemented

- Added ANSI detail-view per-window boost section in `RUNS/gareus_monitor.py`.
- Kept existing aggregate summary metrics intact: `boost mu`, `boost sigma`, `anharmonicity`, `umbrella bias`, plus existing aggregate `GaMD boost dist`.
- Added `_live_boost_window_groups(state, n=4000, limit_per_window=400)` to source grouping from `live_snapshot()` `_dist_samples`, which already comes from a tail scan of `progress.jsonl`.
- Render behavior:
  - If explicit live ids exist (`window`, `window_index`, `state_id`), show `per-window boost dist` rows with label, histogram, `n`, `p50`, `p90`.
  - If explicit ids are absent, show `per-window boost dist unavailable: no explicit window ids in live payload`.
  - If more than 6 groups exist, hide remainder behind `... N window row(s) hidden`.
- Preserved optional Textual/Rich import behavior. Scope stayed inside monitor script and monitor tests.

## TDD evidence

### RED

- Added failing tests in `RUNS/test_gareus_monitor.py`:
  - `test_render_detail_shows_per_window_boost_rows_when_explicit_ids_exist`
  - `test_render_detail_reports_unavailable_when_no_explicit_ids_exist`
- Ran:

```bash
python -m unittest test_gareus_monitor.GaMDBoostSummaryTests.test_render_detail_shows_per_window_boost_rows_when_explicit_ids_exist test_gareus_monitor.GaMDBoostSummaryTests.test_render_detail_reports_unavailable_when_no_explicit_ids_exist
```

- Result: `FAILED (failures=2)`
- Failure cause: `render_detail()` output did not contain `per-window boost dist`.

### GREEN

- Implemented minimal production change in `RUNS/gareus_monitor.py`.
- Re-ran same focused tests:

```bash
python -m unittest test_gareus_monitor.GaMDBoostSummaryTests.test_render_detail_shows_per_window_boost_rows_when_explicit_ids_exist test_gareus_monitor.GaMDBoostSummaryTests.test_render_detail_reports_unavailable_when_no_explicit_ids_exist
```

- Result: `OK`

## Tests and results

- Focused RED check:
  - `python -m unittest test_gareus_monitor.GaMDBoostSummaryTests.test_render_detail_shows_per_window_boost_rows_when_explicit_ids_exist test_gareus_monitor.GaMDBoostSummaryTests.test_render_detail_reports_unavailable_when_no_explicit_ids_exist`
  - Result: `FAILED (failures=2)` before implementation
- Focused GREEN check:
  - Same command
  - Result: `OK`
- Full required suite:
  - `python -m unittest test_gareus_monitor.py`
  - Result: `Ran 21 tests in 0.007s` / `OK`
- Compile check:
  - `PYTHONPYCACHEPREFIX=/tmp/pycache-gareus python -m py_compile gareus_monitor.py test_gareus_monitor.py`
  - Result: success
  - Note: plain `python -m py_compile ...` failed first because `RUNS/__pycache__` was not writable in this environment.
- Diff whitespace check:
  - `git diff --check -- gareus_monitor.py test_gareus_monitor.py`
  - Result: clean

## Files changed

- `RUNS/gareus_monitor.py`
- `RUNS/test_gareus_monitor.py`

## Self-review findings

- Grouping source stays fail-closed: only explicit live ids carried in payload samples are used.
- No synthetic grouping from replica ids or inferred ordering.
- Tail-scan requirement preserved by reusing `live_snapshot()` instead of adding a full-file parser.
- Aggregate boost block remains unchanged aside from adding the new section below it.
- Per-window rows use live histograms and percentiles only; no stored or inferred historical grouping.

## Concerns

- Per-window section currently uses `plain(render_boost_histogram(...))`, so row histograms render uncolored glyphs inside a dim line. This keeps row width stable and readable in ANSI detail, but the per-window mini-histograms are less visually prominent than the aggregate histogram.

## Fix report

- Changed `RUNS/gareus_monitor.py` so the per-window ANSI section is rendered only when progress is still live, matching the aggregate boost block's live-only behavior.
- Split the unavailable message by cause:
  - no explicit window ids in live payload
  - explicit window ids present but no usable boost values
- Added regressions in `RUNS/test_gareus_monitor.py` for:
  - live explicit ids with usable rows
  - live explicit ids with no usable boost values
  - terminal progress hiding the per-window section entirely
- Tests run:
  - `cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS && python -m unittest test_gareus_monitor.GaMDBoostSummaryTests`
  - Result: `OK` (`Ran 15 tests`)
  - `cd /run/media/sulcjo/sulcjo-data/IOCB/md/2026_peptide_sampler/RUNS && env PYTHONPYCACHEPREFIX=/tmp/pycache-gareus python -m py_compile gareus_monitor.py test_gareus_monitor.py`
  - Result: success
- Files changed:
  - `RUNS/gareus_monitor.py`
  - `RUNS/test_gareus_monitor.py`
  - `.superpowers/sdd/task-2-report-per-window.md`
- Concerns:
  - None beyond the existing choice to render per-window histograms as uncolored ASCII inside the dim ANSI frame.
