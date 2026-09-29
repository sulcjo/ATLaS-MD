# Low-memory adaptive analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make low-memory adaptive analysis stream `u_nk`, reduce rows before matrix construction, avoid NPZ decompression during metadata checks, and expose RSS checkpoints.

**Architecture:** Pass stride and memory-report settings from CLI into `load_data`. The Parquet union loader computes per-epoch keep masks before bias reconstruction and writes blocks to a temporary memmap. NPZ shape checks parse ZIP member headers. A small diagnostic helper writes best-effort JSONL records at lifecycle boundaries.

**Tech Stack:** Python, NumPy, compressed NPZ/NPY headers, `mmap`, JSONL, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-low-memory-adaptive-analysis-design.md`

## Global Constraints

- Preserve existing sample alignment and MBAR column order.
- Preserve `_apply_analysis_stride` per-replica production-step semantics.
- Keep `u_nk` `float64`.
- Never make diagnostics affect analysis results.
- Remove temporary memmap files on success and exception.

## Review Focus

- Offset and replica ordering: test exact equivalence with post-load stride.
- Multi-epoch grouping: test same replica IDs in different epochs are independent.
- Exception cleanup: test injected reconstruction failure removes temporary files.
- Compressed NPZ: test shape inspection without array data access.
- Missing `/proc` data: test memory reporting remains non-fatal.

---

### Task 1: Add failing tests for header-only NPZ inspection

**Files:**
- Modify: `tests/test_mbar_analysis_loaders.py`
- Modify: `gareus/mbar_analysis/loaders.py:64-86`

**Interfaces:**
- Preserve `_npz_sample_count_open` and `_npz_window_count_open` signatures.
- Shape helpers may inspect ZIP/NPY headers but must return same counts.

- [ ] **Step 1: Write tests** for a fake NPZ wrapper whose `__getitem__` raises on matrix keys while `.files` and header metadata remain available, plus a real compressed NPZ shape check.
- [ ] **Step 2: Run focused tests and verify failure** with `pytest -q tests/test_mbar_analysis_loaders.py`.
- [ ] **Step 3: Implement header parser** using `zipfile.ZipFile` and NumPy format header readers; route only union snapshot shape checks through it.
- [ ] **Step 4: Run focused tests and verify pass.**
- [ ] **Step 5: Commit** `fix: inspect union npz shapes without decompression`.

### Task 2: Add loader-side early stride

**Files:**
- Modify: `gareus/mbar_analysis/data.py:341-385`
- Modify: `gareus/mbar_analysis/loaders.py:1039-1096`
- Modify: `gareus/mbar_analysis/loaders_union_parquet.py:261-531`
- Test: `tests/test_mbar_analysis_data_part_b.py`
- Test: `tests/test_mbar_analysis_low_memory.py`

**Interfaces:**
- `load_data(..., analysis_stride: int = 1, analysis_stride_offset: int = 0, ...) -> Data`.
- `load_parquet_adaptive_union(..., analysis_stride: int = 1, analysis_stride_offset: int = 0, ...) -> Data`.
- New helper computes epoch-local keep mask from replica and step while including epoch identity in grouping.

- [ ] **Step 1: Write failing equivalence tests** covering two epochs, repeated replica IDs, stride 2, and nonzero offset.
- [ ] **Step 2: Run tests and verify failure.**
- [ ] **Step 3: Implement shared keep-mask helper** matching `_sample_block_ids` and `_apply_analysis_stride` ordering; apply mask to all epoch sample vectors before bias reconstruction.
- [ ] **Step 4: Thread parameters through `load_data` and CLI call.**
- [ ] **Step 5: Run focused tests and verify pass.**
- [ ] **Step 6: Commit** `feat: apply analysis stride before adaptive bias reconstruction`.

### Task 3: Stream low-memory Parquet `u_nk` to memmap

**Files:**
- Modify: `gareus/mbar_analysis/loaders_union_parquet.py:37-64,338-532`
- Test: `tests/test_mbar_analysis_low_memory.py`

**Interfaces:**
- Replace `_spool_u_nk_blocks(blocks, directory)` with a writer receiving block rows incrementally and returning `(memmap, path)`.
- `Data.u_nk` remains array-compatible and `float64`.

- [ ] **Step 1: Write failing tests** asserting no block list retention, memmap-backed output, and cleanup after injected failure.
- [ ] **Step 2: Run tests and verify failure.**
- [ ] **Step 3: Implement two-pass or known-count memmap allocation**; write each block immediately and delete block before next epoch. Keep scalar arrays independent.
- [ ] **Step 4: Add `try/finally` cleanup ownership** so returned Data keeps path alive during analysis and final cleanup occurs through an explicit lifecycle hook or process cleanup.
- [ ] **Step 5: Run focused loader and regression suites.**
- [ ] **Step 6: Commit** `fix: stream adaptive union bias blocks to memmap`.

### Task 4: Add opt-in RSS JSONL reporting

**Files:**
- Create: `gareus/mbar_analysis/memory.py`
- Modify: `analyze_gareus_mbar.py:5876-5895` and MBAR entry boundary.
- Modify: `gareus/mbar_analysis/cli.py` or parser section for `--memory-report`.
- Test: `tests/test_mbar_analysis_memory_report.py`

**Interfaces:**
- `MemoryReporter(path: Optional[Path])` with `record(phase: str) -> None` and `close() -> None`.
- `--memory-report PATH` defaults to disabled.

- [ ] **Step 1: Write failing tests** for required JSON fields, disabled no-op behavior, and `/proc` read failure tolerance.
- [ ] **Step 2: Run tests and verify failure.**
- [ ] **Step 3: Implement best-effort reporter** using `/proc/self/status`, `/proc/meminfo`, and `/proc/swaps`; write one JSON object per line and swallow diagnostic I/O errors.
- [ ] **Step 4: Add lifecycle records** for startup, source selection, epoch boundaries, u_nk write, clean, stride, and first MBAR boundary.
- [ ] **Step 5: Run focused tests and verify pass.**
- [ ] **Step 6: Commit** `feat: add opt-in analysis memory checkpoints`.

### Task 5: End-to-end verification

**Files:**
- Test: `tests/test_mbar_analysis_low_memory.py`
- Artifact: `RUNS/chignolin_9/pmf_analysis/memory_report.jsonl` (runtime output only; do not commit)

- [ ] **Step 1: Run focused regression suite.**
- [ ] **Step 2: Run clean-checkout loader smoke with `NUMBA_CACHE_DIR=/tmp/atlas-md-numba-cache`.**
- [ ] **Step 3: Run chignolin-9 low-memory stride-10 smoke with lightweight analysis flags and memory report.**
- [ ] **Step 4: Inspect final JSONL for last completed phase and peak RSS.**
- [ ] **Step 5: Run full relevant suite and report unrelated failures separately.**
