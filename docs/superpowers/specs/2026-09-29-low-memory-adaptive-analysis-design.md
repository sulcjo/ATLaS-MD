# Low-memory adaptive analysis design

## Goal

Make `gareus-analyze --low-memory` capable of loading and analyzing the
`chignolin_9` adaptive campaign without an OS OOM kill, while preserving sample
alignment, per-replica stride semantics, and existing MBAR values when no
stride is requested.

## Scope

Four changes ship together:

1. **Early stride.** Add loader-side `analysis_stride` and
   `analysis_stride_offset` parameters. Select rows per `(epoch, replica)` from
   production step before reconstructing each `u_nk` block. Keep existing
   post-load stride path for non-low-memory callers until behavior is proven
   equivalent; low-memory callers pass the selection into loaders.
2. **Streaming `u_nk`.** In low-memory Parquet mode, write each reconstructed
   block directly to a temporary `float64` memmap. Do not retain
   `all_unk_blocks`. Return memmap-backed `Data.u_nk`; close/unlink temporary
   storage on success and exceptions.
3. **Header-only NPZ inspection.** Read `.npy` headers inside compressed NPZ
   files to discover sample/window shapes. Never access `f[key]` for shape-only
   checks. Low-memory union-NPZ loading reads only
   `umbrella_reduced_bias_nk` and required vectors; redundant stored bias
   matrices remain unopened.
4. **Opt-in RSS reporting.** Add `--memory-report PATH` (or equivalent
   environment-free CLI control). Emit JSONL records containing timestamp,
   phase, PID, RSS bytes, available memory when readable, and swap usage when
   readable. Report at startup, source selection, each epoch boundary, after
   `u_nk` write, after clean, after stride, and before/after first MBAR solve.
   Reporting failure must never abort analysis.

## Invariants

- `u_nk.shape[0] == cv.size == every per-sample array size` at every returned
  boundary.
- Stride selection matches `_apply_analysis_stride`: independent per replica,
  ordered by production step, with offset semantics unchanged; adaptive epoch
  source remains part of the grouping key.
- `u_nk` retains `float64` dtype and exact column order.
- Low-memory mode may trade RAM for disk, never silently drop samples except
  explicit stride or existing invalid-row filtering.
- Temporary files are created under the adaptive output directory or its
  configured temporary location, use restrictive permissions, and are removed
  on normal and exceptional exits.
- Memory reporting is diagnostic only; no analysis result depends on it.

## Acceptance tests

- Small two-epoch fixture: early-stride output equals current full-load then
  `_apply_analysis_stride` output for all per-sample arrays and `u_nk`.
- Low-memory fixture: loader peak resident memory does not include all
  reconstructed blocks simultaneously; returned matrix is memmap-backed and
  cleanup occurs on success and injected failure.
- NPZ shape checks do not invoke array decompression; valid and malformed NPY
  headers produce expected results.
- Memory report contains required phase records and survives an unwritable or
  malformed `/proc` source without changing return status.
- Chignolin-9 smoke: `--low-memory --analysis-stride 10 --memory-report ...`
  reaches post-load and records peak RSS; full analysis then runs with the
  lightweight diagnostic flags used for the memory probe.
