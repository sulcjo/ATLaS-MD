# MBAR Analysis Modularization — Plan A2 (Data Loading)

## Goal

Relocate everything in `analyze_gareus_mbar.py` responsible for turning a run
directory on disk into the in-memory `Data` object every later analysis stage
consumes — the `Data` dataclass itself, format auto-detection (`load_data`),
and every format-specific loader (NPZ, CSV, Parquet, adaptive-production
union-Parquet, legacy union-NPZ, epoch-CSV, multi-round augmentation) — into
the `gareus/mbar_analysis/` subpackage established by Plan A1. This is Plan
A2 of the 6-plan sequence (A1-A6; see A1's own design doc and Out Of Scope).
No analysis logic changes: every moved function keeps its exact current
behavior, signature, and name.

## Problem

`analyze_gareus_mbar.py`'s data-loading subsystem is large (~35 functions/one
dataclass, ~1,650 lines once traced to closure) and entirely embedded in the
9,946-line script, alongside MBAR solvers, PMF math, trajectory observables,
and plotting it has nothing to do with. Concretely, in scope for this plan
(current names — grep to re-locate; this file has been edited many times
this session and none of these line numbers are stable):

- **The `Data` dataclass** (currently ~line 147) and its lifecycle: `clean`,
  `_masked_data`, `_apply_analysis_stride`, `_filter_epoch_source`,
  `_sample_block_ids`, `_skip_first_n_frames`.
- **Format-specific loaders**, each turning one on-disk layout into a `Data`:
  `load_npz` (+ `_load_merged_arrays`/`_append_npz_arrays`/
  `_discover_analysis_chunk_paths`/`_npz_sample_count_open`/
  `_npz_window_count_open`/`_Arrays`/`_ANALYSIS_VECTOR_KEYS`), `load_csv`
  (+ `_window_float_array`), `load_parquet`, `load_union_npz`
  (+ `_find_adaptive_final_run_dirs`), `load_epoch_csv_adaptive`
  (+ `_find_adaptive_epoch_csv_sources`/`_has_epoch_csv_layout`),
  `load_parquet_adaptive_union` (+ `_is_usable_for_mbar`/
  `_merge_missing_usable_states`/`_load_epoch_task`), and
  `_augment_with_adaptive_rounds` (+ `_find_gareus_round_dirs`/
  `_load_round_raw`/`_build_union_window_table`/`_round_window_to_union_map`/
  `_vectorized_map_lookup`/`_vectorized_map_lookup_or_self`/
  `_vectorized_map_index`).
- **`prod_dir_of`** and **`load_data`** — the resolution/dispatch pair that
  decides, per run directory, which loader above actually runs. `load_data`
  is the single most important function in this domain: every entry point
  into the analysis pipeline goes through it.
- **Cross-cutting metadata/bookkeeping helpers**: `infer_temp_beta`,
  `_load_secondary_cv_from_csv`, `_epoch_run_manifest_secondary_cv_type`,
  `_epoch_number_for_run_dir`, `_epoch_dir_index`, `_epoch_zero_split_masks`,
  `_secondary_cv_epoch_regime_masks`, `rjson`/`wjson`/`_json_default`,
  `read_windows`, `jvec`.

None of this has a natural home outside the script today, so every one of
these ~35 functions is untestable except by importing the entire
9,946-line module, and every future change to (say) the NPZ loader risks
merge conflicts with unrelated work on GaMD boost math or trajectory
plotting happening in the same file.

### Overlap with the existing `gareus` package

The `gareus` package already has real, load-bearing Parquet/metadata
infrastructure. Four overlaps were investigated in depth; each gets an
explicit decision below (not just a "these look similar" footnote).

**1. `gareus/query.py`'s Parquet reading is already reused, not duplicated —
but its bias-reconstruction math is duplicated four different ways.**
`gareus.query.load_samples`/`load_windows` (DuckDB + `segments.json`-aware
segment filtering, via `SegmentRegistry` in `gareus/store.py`) are **already**
the actual Parquet-reading mechanism behind every Parquet-based loader in
`analyze_gareus_mbar.py` — confirmed directly by reading the code:
`load_parquet` calls `from gareus.query import load_samples, load_windows,
reconstruct_bias_matrix` and uses all three; `_load_epoch_task` (used by
`load_parquet_adaptive_union`) calls `from gareus.query import load_samples`.
There is no independent DuckDB/pyarrow code anywhere in
`analyze_gareus_mbar.py`'s Parquet path — this part of the "duplication"
concern raised in A1 does not exist for the loaders in scope here.

What **is** real: `gareus.query.reconstruct_bias_matrix` computes
`U_k(cv) = 0.5*k1*(cv1-center1)^2 [+ 0.5*k2*(cv2-center2)^2]`, and this exact
formula is reimplemented, independently, four times:
1. `load_parquet` — uses the shared `reconstruct_bias_matrix` directly. This
   is the **already-reconciled** case; it should be the target pattern.
2. `load_parquet_adaptive_union`'s helper cluster
   (`_parse_epoch_window_map_native_params`/`_epoch_bias_param_vectors`/
   `_reconstruct_union_bias_block`) reimplements the same formula, generalized
   to accept **per-epoch-native** window params instead of one static
   snapshot — this generalization is not something `reconstruct_bias_matrix`
   can do today (it takes one fixed `windows` list; the union loader needs a
   different center/k per epoch for a state that was recentered mid-campaign,
   the exact bug documented in this repo's `CLAUDE.md` "Union-MBAR bias
   matrix used a stale global registry snapshot" entry).
3. `_augment_with_adaptive_rounds`'s `_compute_u_nk_analytical` reimplements
   the same formula a third time, over a hand-rolled `union_windows` list of
   dicts instead of `gareus.query`'s window-dict shape.
4. `load_csv` and `load_epoch_csv_adaptive` **each** reimplement the formula
   a fourth and fifth time, **inline** (not as separately named functions —
   `u[:,k]=scale*total` inside a `for k in range(K)` loop in both), as a
   fallback for samples.csv-only runs with no all-window bias vectors on
   disk. These two inline copies are easy to miss with a
   function-name grep, which is exactly why they are called out by name
   here for whoever picks up Plan A3.

**2. `gareus/store.py`'s `SegmentRegistry`: not duplicated, but a known,
deliberately-preserved precision gap.** `analyze_gareus_mbar.py` never
reimplements `SegmentRegistry` or `segments.json` parsing — its own
`_parquet_sample_count` (used only by `load_data`'s `auto` heuristic to
decide "does Parquet or NPZ/CSV have more rows") does a raw
`SELECT count(*) FROM read_parquet(...)` over every `*.parquet` file with
**no segment-status filtering** — it doesn't consult `segments.json` at all,
so it counts `abandoned`/stale `running`-non-last-segment rows that
`gareus.query.load_samples` would correctly exclude. This means the `auto`
heuristic can, in principle, overestimate how many *real* rows a Parquet
source has relative to what will actually load. It is pre-existing behavior,
narrow in effect (a fast source-selection estimate, not the actual loaded
data — the real load always goes through the correct `load_samples`), and is
preserved as-is by this plan (see Error Handling).

**3. `gareus/io.py`'s `resolve_run_temperature_k` vs. `infer_temp_beta`: real
duplication, not reconcilable inside a plain-relocation plan.**
`resolve_run_temperature_k` is the fix this repo's `CLAUDE.md` documents for
a real historical bug (`gareus_metadata.json` never has a temperature field;
silently defaulting to 300 K corrupted reduced-bias energies for any run not
actually at 300 K). `infer_temp_beta` independently implements a **superset**
of that same fallback chain: it already checks `gareus_metadata.json`/
`run_manifest.json`-shaped `meta` dicts at the same nested key paths
(`resolved_args.temperature_k`, `method_settings.temperature_k`) that
`resolve_run_temperature_k` checks — so it already carries the equivalent of
that fix, independently — but it *also* probes fields `resolve_run_temperature_k`
never looks at at all: live in-memory NPZ arrays (`beta_1_over_kJ_mol`/
`temperature_K`, checked **before** any file read, for the `load_npz`/
`load_union_npz` callers that already have the array open), and
`run_args.json` at three directory levels (`prod`, `prod.parent`,
`prod.parent.parent`, covering epoch-dir/adaptive_production/run-root
nesting). It also returns `(temp, beta)` as a tuple (`resolve_run_temperature_k`
returns only `temp`), and it already warns via `warnings.warn` **and**
appends to `meta['load_notes']` before falling back to 300 K — the same
non-silent behavior `CLAUDE.md` documents being *added* to
`_ensure_analysis_arrays_npz` as a fix, already present here independently.
Decision: **keep `infer_temp_beta` as-is** (relocate unchanged); delegating
to `resolve_run_temperature_k` would either lose the array-probe and
`run_args.json` capabilities or require redesigning
`resolve_run_temperature_k`'s own signature/scope, both out of scope for a
plan whose mandate is relocation, not new logic. Flagged in Out Of Scope for
a future consolidation decision, since two independently-maintained
temperature-fallback implementations is a real risk (a future bug fix to one
is not guaranteed to reach the other).
**4. `gareus/analysis.py`'s `validate_analysis_metadata_readiness`/
`_ensure_analysis_arrays_npz`: a different pipeline stage, not a competing
loader.** This module is called exclusively from the **live production**
side (`gareus/cv_discovery.py`, `gareus/adaptive_feedback.py`,
`gareus/production.py`) to gate a run's own pipeline on MBAR-readiness,
auto-regenerating `analysis_arrays.npz` from Parquet if missing.
`analyze_gareus_mbar.py`'s `load_data` never calls into it and doesn't need
to: `load_data`'s auto-detection compares actual row counts across whichever
formats exist and picks the richest one, including going straight to
`load_parquet` without ever materializing an NPZ — a strictly more capable
selection than "ensure one specific legacy artifact exists." Decision: no
change; these are genuinely different concerns (pre-analysis live-run gating
vs. post-hoc richest-source selection), not a duplication needing
reconciliation.

## Recommended Approach

Relocate the ~35 functions/dataclass above, verbatim (same names, same
bodies, same signatures — a pure move), into four new modules under
`gareus/mbar_analysis/`, split along real dependency and responsibility
boundaries rather than forced into one file:

- **`gareus/mbar_analysis/data.py`** — the `Data` dataclass, generic
  JSON/CSV primitives (`rjson`/`wjson`/`_json_default`/`read_windows`/
  `jvec`), temperature/beta resolution (`infer_temp_beta`), and every
  function that transforms or inspects an already-constructed `Data`
  (`clean`, `_masked_data`, `_apply_analysis_stride`, `_filter_epoch_source`,
  `_sample_block_ids`, `_skip_first_n_frames`, `_epoch_dir_index`,
  `_epoch_number_for_run_dir`, `_epoch_run_manifest_secondary_cv_type`,
  `_epoch_zero_split_masks`, `_secondary_cv_epoch_regime_masks`). This module
  has no dependency on any other new module — it is the bottom of the
  dependency graph.
- **`gareus/mbar_analysis/loaders_adaptive.py`** — adaptive-production
  directory-layout discovery (`_find_adaptive_epoch_dirs`,
  `_find_selfcontained_epoch_dirs`, `_find_adaptive_epoch_csv_sources`,
  `_has_epoch_csv_layout`, `_find_adaptive_final_run_dirs`,
  `_find_gareus_round_dirs`), the shared vectorized index-remapping utilities
  (`_vectorized_map_lookup`, `_vectorized_map_lookup_or_self`,
  `_vectorized_map_index`), and the two **legacy/fallback** adaptive loaders
  plus rounds augmentation (`load_epoch_csv_adaptive`, `load_union_npz`,
  `_load_round_raw`, `_build_union_window_table`, `_round_window_to_union_map`,
  `_augment_with_adaptive_rounds`). Depends only on `data.py`.
- **`gareus/mbar_analysis/loaders_union_parquet.py`** — the **primary**
  adaptive-production loader and its private registry/task helpers
  (`_is_usable_for_mbar`, `_merge_missing_usable_states`, `_load_epoch_task`,
  `load_parquet_adaptive_union`). Kept separate from
  `loaders_adaptive.py` despite both being "adaptive-production" loaders:
  this is the loader this repo's `CLAUDE.md` documents an entire historical
  bug narrative around (the per-epoch-native-bias fix), it is independently
  the single largest function in the whole domain (~200 lines), and it has
  its own dedicated test coverage (see Testing) — cohesion and
  reviewability both favor isolating it rather than folding it into the
  smaller legacy-format file. Depends on `data.py` and `loaders_adaptive.py`
  (for `_find_adaptive_epoch_dirs`/`_vectorized_map_lookup`/
  `_vectorized_map_index`).
- **`gareus/mbar_analysis/loaders.py`** — the single-run format loaders
  (NPZ family: `_npz_sample_count_open`, `_npz_window_count_open`, `_Arrays`,
  `_ANALYSIS_VECTOR_KEYS`, `_discover_analysis_chunk_paths`,
  `_append_npz_arrays`, `_load_merged_arrays`, `load_npz`,
  `_load_secondary_cv_from_csv`; CSV family: `_csv_row_count_fast`,
  `_npz_sample_count`, `_analysis_binary_sample_count`, `_window_float_array`,
  `load_csv`; Parquet family: `_parquet_sample_count`, `load_parquet`), plus
  the top-level **`prod_dir_of`** resolution function and the **`load_data`**
  dispatcher that ties every loader in all four modules together. This is
  deliberately the top of the dependency graph — it is the only module that
  needs to know about every other one.

This gives a clean, acyclic dependency chain:
`data.py` ← `loaders_adaptive.py` ← `loaders_union_parquet.py` ← `loaders.py`
(`loaders.py` also depends directly on `data.py` and `loaders_adaptive.py`).

**Four bias-reconstruction functions stay behind in `analyze_gareus_mbar.py`
for now, called via a lazy, function-body-local reverse import**:
`_compute_u_nk_analytical` (needed by `_augment_with_adaptive_rounds`, moving
to `loaders_adaptive.py`) and `_parse_epoch_window_map_native_params`/
`_epoch_bias_param_vectors`/`_reconstruct_union_bias_block` (needed by
`_load_epoch_task`/`load_parquet_adaptive_union`, moving to
`loaders_union_parquet.py`). These four are explicitly Plan A3's scope per
A1's own Out Of Scope section ("relocate MBAR solvers and bias
reconstruction, reconciling with `gareus/query.py`'s `reconstruct_bias_matrix`
— including fixing the already-documented NaN-guard gap between the two
existing copies... gets its own audit-driven verification pass, not a plain
code move"). Moving them here as a side effect of moving their one caller
would pre-empt that audit. See Architecture for exactly why this reverse
import must be lazy (function-body-local), not a normal module-top import.

`analyze_gareus_mbar.py` itself keeps every name unchanged via a re-export
import block at the top of the file (mirroring A1 Task 2's
`from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K` pattern exactly) —
every one of the ~9,900 remaining lines that reference `Data`, `load_data`,
`clean`, etc. by bare name need zero changes.

## Why This Approach

- **Matches A1's own established transitional pattern exactly.** A1's
  `cli.py` already imports `analyze_gareus_mbar` (package importing the
  script) as a deliberate strangler-fig placeholder; A2's lazy reverse
  imports for the four bias-math functions are the same pattern, in the same
  direction, for the same reason (the callee hasn't moved yet).
- **Four files, not one or two, because the real 800-line-per-file guideline
  this project's own contributors follow elsewhere (see Alternatives
  Considered) is a real constraint, not a suggestion to route around.** A
  naive single `loading.py` would be ~1,650+ core lines before counting the
  substantial docstrings/comments this code already carries (e.g.
  `load_parquet_adaptive_union`'s threading/memory-optimization comments,
  `_secondary_cv_epoch_regime_masks`'s regime-selection rationale) — likely
  2,000+ lines realized. The four-way split was chosen by tracing actual
  call dependencies (not arbitrary function-count buckets), and each
  resulting file lands comfortably under 800 real lines (estimated: `data.py`
  ~480-520, `loaders_adaptive.py` ~680-750, `loaders_union_parquet.py`
  ~320-380, `loaders.py` ~650-720).
- **The split boundary between `loaders_adaptive.py` and
  `loaders_union_parquet.py` tracks a real, already-documented risk
  boundary, not just a line-count target.** The union-Parquet loader is
  where the per-epoch-native-bias bug lived and was fixed; isolating it
  means any future audit or regression of that specific subsystem touches
  one ~350-line file, not a ~1,300-line grab-bag of every adaptive-format
  loader.
- **Already-reconciled `load_parquet` is called out explicitly as the target
  pattern**, so whoever picks up Plan A3 has a concrete "make the other three
  copies look like this one" reference rather than starting from zero.
- **Every overlap decision is a real decision, not a deferral disguised as
  one**: `gareus.query`'s Parquet reading is confirmed already reused (no
  action needed); `SegmentRegistry`'s precision gap is confirmed narrow and
  preserved with a documented reason; `resolve_run_temperature_k` is
  confirmed a strict subset of `infer_temp_beta`'s capability (relocate
  as-is, flag for a future consolidation plan); `gareus.analysis` is
  confirmed a different pipeline stage entirely (no action needed).

## Alternatives Considered

### One `loading.py` file for the whole domain

Rejected. As sized above, this would land at 1,650-2,000+ lines, well past
this project's own established per-file guideline. It would also force
`load_data`'s dispatcher logic, the NPZ chunk-merging internals, and the
union-Parquet per-epoch-native-bias bookkeeping into one undifferentiated
scroll — exactly the "everything in one file" problem this whole 6-plan
effort exists to fix, just at a smaller scale.

### Two files (`data.py` + one `loaders.py`), per the task's own example split

Considered first, matching the illustrative split named in this plan's brief.
Rejected once actually sized: a single `loaders.py` holding every
format-specific loader plus `prod_dir_of`/`load_data` totals ~1,320-1,600
realized lines (NPZ + CSV + Parquet-single-run + adaptive-union-Parquet +
legacy-union-NPZ + epoch-CSV + rounds-augmentation + dispatcher) — still over
the 800-line guideline. A three-file variant (`data.py` +
`loaders.py`-with-dispatcher + one combined `loaders_adaptive.py` for
everything adaptive-production-flavored) was sized next and found to put the
combined adaptive file at ~790-900 realized lines once
`load_parquet_adaptive_union`'s own ~200-line, heavily-commented cluster is
included — marginal-to-over even before accounting for docstring overhead.
The four-file split in Architecture is the smallest number of files that
keeps every one comfortably under 800 real lines without cutting a
dependency edge awkwardly.

### Splitting further, one file per on-disk format

Considered (e.g. `loaders_npz.py`, `loaders_csv.py`, `loaders_parquet.py`
separately). Rejected: `load_data`'s dispatcher inherently needs direct,
frequent access to sibling row-count helpers across NPZ/CSV/Parquet to
implement its `auto` best-source comparison (`_parquet_sample_count` vs.
`_analysis_binary_sample_count` vs. `_csv_row_count_fast`) — these three
single-run counters have no meaningful cohesion boundary between them (they
exist purely to be compared against each other in one function), so
splitting them into three separate files would multiply cross-file imports
for zero cohesion benefit. `loaders.py` groups exactly the loaders/counters
that `load_data`'s single-run branch needs side by side.

### Reconciling `infer_temp_beta`/`resolve_run_temperature_k` now

Considered doing the consolidation as part of this plan, since both were
fully read and compared. Rejected: `infer_temp_beta` is a strict capability
superset (npz-array probing, `run_args.json` directory-walk) with a
different return signature (`(temp, beta)` vs. `temp`); folding one into the
other is a real design decision (does `resolve_run_temperature_k` grow a new
signature and new file-probing responsibility, or does `infer_temp_beta`
lose capability by delegating a subset of its checks?) that this plan's
"relocate, don't redesign" mandate explicitly excludes. Flagged in Out Of
Scope.

### Reconciling the four bias-reconstruction copies as part of this plan

Considered, since three of the four copies are moving as part of this plan's
loaders anyway (only staying behind as functions, not as call sites).
Rejected: A1 explicitly reserves "bias reconstruction, reconciling with
`gareus/query.py`'s `reconstruct_bias_matrix`" for Plan A3, calling it out as
needing "its own audit-driven verification pass, not a plain code move." Bias
energies feed `u_nk` directly — the single most scientifically consequential
number in this whole pipeline — exactly the kind of change this session's
own prior physics audits (see `CLAUDE.md`) show needs dedicated,
non-bundled verification.

## Architecture

### Files touched

```
gareus/mbar_analysis/data.py                  CREATE  (~480-520 lines: Data dataclass, JSON/CSV
                                                        primitives, temp/beta inference, Data-transform
                                                        and epoch/regime metadata helpers)
gareus/mbar_analysis/loaders_adaptive.py      CREATE  (~680-750 lines: adaptive-production directory
                                                        discovery, vectorized index remapping, legacy
                                                        union-NPZ + epoch-CSV loaders, rounds augmentation)
gareus/mbar_analysis/loaders_union_parquet.py CREATE  (~320-380 lines: the primary adaptive-production
                                                        union-Parquet loader and its registry/task helpers)
gareus/mbar_analysis/loaders.py               CREATE  (~650-720 lines: NPZ/CSV/Parquet single-run
                                                        loaders, prod_dir_of, load_data dispatcher)
analyze_gareus_mbar.py                        MODIFY  (delete ~35 relocated definitions; add one
                                                        re-export import block per new module, placed
                                                        after the KJ_PER_KCAL import Plan A1 adds)
```

No `pyproject.toml` change: these are plain internal modules of the
`gareus.mbar_analysis` subpackage A1 already registers for discovery: no new
console script, no new extras.

### Dependency graph (acyclic)

```
                 data.py
                    ^
                    |
          loaders_adaptive.py
                    ^
                    |
       loaders_union_parquet.py
                    ^
                    |
                loaders.py  (also imports data.py and loaders_adaptive.py directly)
                    ^
                    |
       analyze_gareus_mbar.py  (re-exports everything; the ONLY consumer
                                 outside gareus/mbar_analysis/ during A2-A5)
```

### Why the reverse import (package → script) must be lazy, not module-top

`loaders_adaptive.py`'s `_augment_with_adaptive_rounds` and
`loaders_union_parquet.py`'s `_load_epoch_task`/`load_parquet_adaptive_union`
each need one of the four bias-reconstruction functions that stay behind in
`analyze_gareus_mbar.py` (see Recommended Approach). A naive **module-top**
`from analyze_gareus_mbar import _compute_u_nk_analytical` in
`loaders_adaptive.py` would create a real circular import that fails, not
just a style smell — traced explicitly:

1. Something does `import analyze_gareus_mbar` (or runs it as `__main__`).
2. Python registers the (empty) module in `sys.modules` and starts executing
   it top to bottom.
3. Near the top (mirroring A1's constants-import placement), it reaches
   `from gareus.mbar_analysis.loaders import Data, load_data, ...` — this
   transitively imports `loaders_adaptive.py`.
4. `loaders_adaptive.py`, if it did `from analyze_gareus_mbar import
   _compute_u_nk_analytical` at **its own module top**, would need that name
   to already exist as an attribute of `analyze_gareus_mbar` — but
   `analyze_gareus_mbar` is only partially executed (stopped at step 3,
   which is near the top of a 9,900-line file); `_compute_u_nk_analytical`
   is defined much further down and has not run yet.
5. Result: `ImportError: cannot import name '_compute_u_nk_analytical' from
   partially initialized module 'analyze_gareus_mbar' (most likely due to a
   circular import)`.

Fix: the four reverse-direction imports are all **inside the function body**
of their one call site (`_augment_with_adaptive_rounds`,
`_load_epoch_task`, `load_parquet_adaptive_union`), executed lazily the
first time that function is actually *called* — by which point
`analyze_gareus_mbar.py` has finished executing top to bottom (the module
that triggered the whole import chain is only ever imported once, and by the
time any analysis function runs, module-level execution is long done). This
is not a new idiom introduced by this plan — `_load_epoch_task` and
`load_parquet_adaptive_union` already do exactly this today for
`from gareus.query import load_samples` (a real, already-shipped example of
the same lazy-import-inside-function-body pattern in this exact file), and
A2 simply keeps doing it for the two now-reversed-direction imports.

**Handoff note for Plan A3**: once A3 relocates
`_compute_u_nk_analytical`/`_parse_epoch_window_map_native_params`/
`_epoch_bias_param_vectors`/`_reconstruct_union_bias_block` to their own new
home, exactly three lazy-import lines need their source module string
updated (one inside `_augment_with_adaptive_rounds` in
`loaders_adaptive.py`, importing `_compute_u_nk_analytical`; two inside
`loaders_union_parquet.py` — one inside `_load_epoch_task` importing
`_parse_epoch_window_map_native_params`, one inside
`load_parquet_adaptive_union` importing `_epoch_bias_param_vectors` and
`_reconstruct_union_bias_block`) — a small, mechanical follow-up, not a
redesign.

### `gareus/mbar_analysis/data.py` (imports)

```python
from __future__ import annotations

import csv
import json
import math
import re
import warnings as _warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K
```

### `gareus/mbar_analysis/loaders_adaptive.py` (imports)

```python
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K
from .data import Data, clean, infer_temp_beta, rjson, read_windows, _epoch_dir_index
```

### `gareus/mbar_analysis/loaders_union_parquet.py` (imports)

```python
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.units import K_B_KJ_PER_MOL_K
from .data import Data, clean, infer_temp_beta, rjson
from .loaders_adaptive import _find_adaptive_epoch_dirs, _vectorized_map_lookup, _vectorized_map_index
```

### `gareus/mbar_analysis/loaders.py` (imports)

```python
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.units import KJ_PER_KCAL
from .data import Data, clean, infer_temp_beta, rjson, read_windows, jvec
from .loaders_adaptive import (
    _find_adaptive_epoch_dirs, _find_selfcontained_epoch_dirs,
    _find_adaptive_epoch_csv_sources, _has_epoch_csv_layout,
    load_epoch_csv_adaptive, load_union_npz,
    _find_gareus_round_dirs, _augment_with_adaptive_rounds,
)
from .loaders_union_parquet import load_parquet_adaptive_union
```

(`gareus.query`'s `load_samples`/`load_windows`/`reconstruct_bias_matrix` and
`from gareus.query import load_samples` stay as the existing
function-body-local imports inside `load_parquet`/`_load_epoch_task`/
`load_parquet_adaptive_union` respectively — unchanged, since that is
pre-existing, already-correct style, not something this plan touches.)

### `analyze_gareus_mbar.py` (relevant lines, after Plan A1's constants swap)

```python
from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K          # Plan A1
from gareus.mbar_analysis.data import (                          # Plan A2
    Data, rjson, wjson, read_windows, jvec, infer_temp_beta,
    clean, _masked_data, _apply_analysis_stride, _filter_epoch_source,
    _sample_block_ids, _skip_first_n_frames, _epoch_dir_index,
    _epoch_number_for_run_dir, _epoch_run_manifest_secondary_cv_type,
    _epoch_zero_split_masks, _secondary_cv_epoch_regime_masks,
)
from gareus.mbar_analysis.loaders_adaptive import (               # Plan A2
    _find_adaptive_epoch_dirs, _find_selfcontained_epoch_dirs,
    _find_adaptive_epoch_csv_sources, _has_epoch_csv_layout,
    _find_adaptive_final_run_dirs, _find_gareus_round_dirs,
    _vectorized_map_lookup, _vectorized_map_lookup_or_self, _vectorized_map_index,
    load_epoch_csv_adaptive, load_union_npz, _load_round_raw,
    _build_union_window_table, _round_window_to_union_map,
    _augment_with_adaptive_rounds,
)
from gareus.mbar_analysis.loaders_union_parquet import (          # Plan A2
    _is_usable_for_mbar, _merge_missing_usable_states,
    load_parquet_adaptive_union,
)
from gareus.mbar_analysis.loaders import (                        # Plan A2
    _npz_sample_count_open, _npz_window_count_open, _Arrays,
    _ANALYSIS_VECTOR_KEYS, _discover_analysis_chunk_paths,
    _append_npz_arrays, _load_merged_arrays, load_npz,
    _load_secondary_cv_from_csv, _csv_row_count_fast, _npz_sample_count,
    _analysis_binary_sample_count, _window_float_array, load_csv,
    _parquet_sample_count, load_parquet, prod_dir_of, load_data,
)
```

`_json_default` (used only by `wjson`, itself already re-exported) and
`_load_epoch_task` (used only inside `load_parquet_adaptive_union`, itself
already re-exported) are intentionally **not** re-exported into
`analyze_gareus_mbar`'s namespace: grep confirms neither name is referenced
anywhere else in the 9,946-line file outside its own module. Re-exporting
unused names would be dead weight, not a compatibility requirement.

## Error Handling

No new failure modes: every moved function keeps its existing error handling
verbatim (e.g. `load_data`'s `FileNotFoundError`/`ValueError` messages,
`prod_dir_of`'s detailed "checked X, Y, Z" `FileNotFoundError`, `clean`'s
`ValueError('u_nk/sample count mismatch')`). Two pre-existing behaviors are
explicitly preserved, not hardened, by this plan:

- `_parquet_sample_count`'s lack of `segments.json` status filtering (see
  Problem, overlap #2) continues to feed `load_data`'s `auto` source-count
  comparison exactly as it does today. This is a narrow, pre-existing
  precision gap in a heuristic, not a correctness bug in what actually
  loads — flagged for awareness, not fixed here (fixing it would mean
  duplicating `SegmentRegistry`-aware filtering into a fast-count function
  whose whole purpose is to avoid the cost of a real segment-aware read; a
  real fix should share code with `gareus.query` rather than reimplement its
  filtering a second time, which is design work beyond a relocation plan).
- `infer_temp_beta`'s final fallback (default to 300 K with a `RuntimeWarning`
  and a `load_notes` entry) is preserved exactly; no attempt is made to
  route it through `resolve_run_temperature_k` (see Recommended Approach
  overlap #3).

## Testing

Since this is a pure relocation of already-tested, already-working code (not
new logic), the test strategy is aimed at proving the move introduced zero
behavioral change, not at re-deriving new correctness properties:

1. **Object-identity checks, not re-derived behavioral tests, are the
   primary proof of a clean relocation.** For every one of the ~35 moved
   names, `analyze_gareus_mbar.<name> is gareus.mbar_analysis.<module>.<name>`
   after the re-export import is the load-bearing assertion — it fails
   loudly (not silently) if the old definition in `analyze_gareus_mbar.py`
   was left in place (a later `def` at the same name would simply shadow the
   earlier import, and nothing else would notice). This matters more here
   than a re-derived behavioral test would: several of these functions
   mutate their `Data` argument in place (`clean`, `_apply_analysis_stride`,
   `_skip_first_n_frames`, `_filter_epoch_source` all reassign `d.cv`/`d.window`/
   etc. on the object they were passed rather than returning a fresh copy),
   so a naive "call it once via the old path, call it again via the new
   path, compare outputs" test would be comparing a mutated object against
   itself and could pass even if the two callables were genuinely different
   functions with different bugs.
2. **Every relocated function/class is actually invoked (not just imported)
   at module-creation time**, before `analyze_gareus_mbar.py` is rewired to
   import from it. No static linter is available in this environment for a
   mechanical undefined-name check (`pyflakes`/`ruff` both absent — verified:
   `ModuleNotFoundError`/`which` failure); `python -m py_compile` only
   catches syntax errors, not a `NameError` buried inside a function body
   that's never called. (This is exactly the kind of bug a naive read of
   this domain could introduce silently — e.g. relocating `_load_merged_arrays`
   without noticing its return statement, `return _Arrays(result), notes`,
   needs the `_Arrages` dict-subclass and `_ANALYSIS_VECTOR_KEYS` set moved
   alongside it, since neither is a separately-named "loader" and both are
   easy to miss on a first pass.) Requiring a real invocation against a
   minimal synthetic fixture for every moved function, before any rewiring
   happens, closes this gap without new tooling.
3. **The existing test suite that already covers this domain runs unchanged
   against the new location** (via the re-export, it's the same objects):
   `tests/test_prod_dir_resolution.py` (`prod_dir_of`),
   `tests/test_mbar_epoch_filter.py` (`_find_adaptive_epoch_csv_sources`,
   `_find_adaptive_epoch_dirs`, `load_epoch_csv_adaptive`),
   `tests/test_perf_loading_and_solver_memory.py` (`load_parquet_adaptive_union`,
   `load_parquet`, `Data`), `tests/test_pmf_bootstrap_uncertainty.py`
   (`_sample_block_ids`, `Data`), `tests/test_infer_temp_beta.py`
   (`infer_temp_beta`), `tests/test_final_registry_merge.py`
   (`_merge_missing_usable_states`, `_is_usable_for_mbar`),
   `tests/test_epoch0_pmf_gamd_split.py` (`_epoch_number_for_run_dir`,
   `_epoch_zero_split_masks`, `_masked_data` — also imports
   `run_pmf_and_gamd_boost_report`/`parse_args`, owned by a different plan;
   this one test file straddles two domains, see below),
   `tests/test_masked_logw_subset_pmf.py` (`_epoch_zero_split_masks`,
   `_masked_data`, `Data` — also imports `_subset_logw_from_global_fk`, not
   in this plan's scope), `tests/test_secondary_cv_regime_split.py`
   (`_epoch_run_manifest_secondary_cv_type`, `_masked_data`,
   `_secondary_cv_epoch_regime_masks` — also imports `_regime_slug`,
   `_secondary_cv_regions`, `_slug`, `run_secondary_cv_analyses`, not in this
   plan's scope), `tests/test_union_mbar_per_epoch_bias.py`
   (`load_parquet_adaptive_union` — also imports `_epoch_bias_param_vectors`,
   `_parse_epoch_window_map_native_params`, `_reconstruct_union_bias_block`,
   which are Plan A3's, staying behind in `analyze_gareus_mbar.py`).
   The three straddling test files above are the concrete reason the
   re-export convention (rather than, say, having tests import from the new
   submodules directly) matters: each continues to `from analyze_gareus_mbar
   import (...)` a mix of names now backed by up to three different physical
   modules (this plan's, and whichever plan ends up owning
   `run_secondary_cv_analyses`/the bias-math cluster), and none of them need
   to change at all as long as every plan in the sequence keeps re-exporting
   into `analyze_gareus_mbar`'s namespace the way A1 established.
4. Full-suite regression: `pytest -q tests/ --ignore=tests/test_validation_common.py`
   shows the same pre-existing failures as `main` before this plan (see A1's
   own Testing section) and no new ones.
5. `python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/data.py gareus/mbar_analysis/loaders_adaptive.py gareus/mbar_analysis/loaders_union_parquet.py gareus/mbar_analysis/loaders.py`
6. `python -c "import gareus.mbar_analysis.loaders"` — the real circular-import
   smoke test: this single import transitively pulls in all four new
   modules and, if any reverse import were accidentally placed at module top
   instead of function-body-local, would fail immediately with the exact
   `ImportError` traced in Architecture.

## Out Of Scope

- **The four bias-reconstruction functions** (`_compute_u_nk_analytical`,
  `_parse_epoch_window_map_native_params`, `_epoch_bias_param_vectors`,
  `_reconstruct_union_bias_block`) and reconciling all four
  now-duplicated-four-ways copies of the bias formula (including the two
  inline copies inside `load_csv`/`load_epoch_csv_adaptive` newly documented
  above) with `gareus.query.reconstruct_bias_matrix` — Plan A3, per A1's Out
  Of Scope. This plan's loaders call the four staying-behind functions via a
  lazy, function-body-local reverse import (see Architecture); Plan A3 only
  needs to repoint two import lines once it relocates them.
- **`infer_temp_beta` vs. `gareus.io.resolve_run_temperature_k`
  consolidation** — flagged as a real, confirmed duplication (see Problem/
  Recommended Approach) but requires a signature/scope redesign of one or
  both functions, which is new design work, not relocation. No plan in the
  current A1-A6 sequence explicitly claims this; whoever eventually touches
  either function again should be aware both exist.
- **`_parquet_sample_count`'s missing `segments.json`-status filtering** —
  a narrow, pre-existing precision gap in a fast row-count heuristic (see
  Error Handling); not fixed here.
- **`_secondary_cv_label`, `_secondary_cv_regions`, `_subset_logw_from_global_fk`,
  `_regime_slug`, `run_secondary_cv_analyses`** — these read/derive from a
  `Data`'s metadata but produce analysis outputs (PMF splits, human-readable
  plot labels), not a `Data` object; they belong with whichever plan owns
  secondary-CV PMF/2D-FES analysis (referenced but not claimed by this plan).
- **`_epoch_source_annotations`, `_epoch_source_pooling_table`,
  `_skipped_empty_epochs`, `_add_epoch_annotations_to_axes`,
  `_epoch_source_aggregate_ns_info`, `_aggregate_ns_for_fracs`,
  `_add_aggregate_ns_secondary_axis`** — convergence-plotting helpers that
  consume `Data.meta['_epoch_source']` for axis annotations; plotting/
  convergence domain (Plan A6), not data loading.
- **`_prepare_adaptive_merged_traj_dir`, `_adjusted_steps_for_merged_traj`,
  `_read_traj_interval_from_epoch_dirs`, `_get_adaptive_epoch_traj_dirs`** —
  trajectory-file (XTC/DCD) merging for Rg/PCA/Poincaré analyses, not
  sample-array loading; trajectory-observables domain (Plan A6).
- **`analyze()`, `parse_args()`, `main()`** — the top-level CLI glue that
  currently calls `load_data`/`_skip_first_n_frames` and everything else in
  this file; untouched by this plan (they keep calling the same names,
  unchanged, via the re-export).
- Any change to loader **behavior**, output format, or the `Data` dataclass's
  fields — this plan changes only where ~35 definitions physically live.
