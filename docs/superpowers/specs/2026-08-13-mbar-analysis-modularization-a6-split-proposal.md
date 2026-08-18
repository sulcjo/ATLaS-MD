# MBAR Analysis Modularization — Plan A6 Split Proposal

## Summary

Plan A6 ("Trajectory Observables + Convergence + Reporting; retire
`analyze_gareus_mbar.py`") is too large for one plan. A precise, function-by-
function line count (script below, cross-checked against four independent
domain investigations) shows A6's assigned scope is **~5,071 lines across 81
named functions** before counting supporting helpers, and **~6,070 lines
across 121 functions** once every helper that has no other owner is added in.
That is **61% of the 9,946-line file** — bigger than Plans A2, A3, A4, and A5
combined (~3,880 lines total, ~1,000-1,300 lines apiece). A1's own stated
approach ("a single giant move has no natural checkpoint to catch a mistake
before it's buried under five more domains' worth of changes") applies to A6
at least as strongly as it applied to A1 rejecting a one-shot move for the
whole file.

**Recommendation: split A6 into five sequential sub-plans, in this order:**

| Order | Plan | Scope | Lines into `gareus/mbar_analysis/` | Functions |
|---|---|---|---|---|
| 1 | **A6a** | Plotting & reporting (leaf/foundation) | ~1,297 | 42 + whole-file `plotstyle.py` (12 exports) |
| 2 | **A6b** | Convergence orchestration | ~1,427 | 19 |
| 3 | **A6c** | Trajectory I/O infra + structural observables (Rg/PCA/distance-Rg-FES) | ~1,090 (+284 to a new top-level chignolin file) | 36 into package + 5 into `chignolin_fes_extension.py` |
| 4 | **A6d** | Residue/topology observables + Poincaré + CV1×CV2 FES | ~1,730 | 26 |
| 5 | **A6e** | Orchestrator retirement (`analyze`/`parse_args`/`main`) | ~350 (+ cleanup) | 3 |

This **inverts** the task's own suggested example grouping
("A6a=observables, A6b=convergence, A6c=reporting+retirement"). That example
groups reporting with retirement because both sound like "final glue" — but
the actual call graph makes that combination structurally impossible: see
"Why this order" below. This is not a stylistic disagreement with the
example; it's what four independent code investigations of the actual call
sites found.

This document is the umbrella proposal. **A6a is fully speced and planned**
(`2026-08-13-mbar-analysis-modularization-a6a-design.md` /
`...-a6a.md`) — ready to execute now. **A6b, A6c, A6d, A6e are scoped here**
(function lists, sizes, dependencies, open decisions) but not yet
speced/planned in bite-sized-task form; a future session should read this
document plus the four raw investigation findings summarized below before
writing each one's design spec.

## Why this order: the dependency graph is a DAG with one source and one sink

Four domain investigations (reporting, trajectory observables, convergence
orchestration, and the `analyze`/`parse_args`/`main` orchestrator) each
independently traced every call made *out of* their own function list. The
combined picture:

- **Reporting calls out to nothing else in A6.** Every one of its ~41
  functions receives already-computed arrays/dicts (`pmf` dicts, `fes` dicts,
  `conv_rows` lists, the `s` summary dict) as plain parameters and only reads
  them — confirmed by reading every body, not just names. It is a pure sink
  for A3/A4 output and a pure *source* for everything else in A6 that writes
  a file.
- **Convergence orchestration calls reporting directly, inline.**
  `run_epoch_pmf_convergence` calls `write_convergence_plots` and
  `_write_epoch_ess_plot`; `run_observable_pmf_convergence` calls
  `write_observable_convergence_plots`, `write_observable_convergence_report`,
  and `_plot_basin_population_convergence` — all inside the same function
  body that does the MBAR re-solve, not returned as data for some later
  caller to render. Convergence cannot be relocated cleanly without
  reporting's functions already existing somewhere importable.
- **Trajectory observables call out to reporting *and* convergence.**
  `analyze_rg` and `analyze_extra_observable_pmfs` call
  `run_observable_pmf_convergence` (convergence) directly, and every
  `analyze_*` orchestrator in this sub-domain calls its matching
  `write_*`/`plot_*` function (reporting) directly.
- **The top-level `analyze()`/`parse_args()`/`main()` retirement step depends
  on literally everything** — A2-A5's modules, plus all of A6a-A6d. It also
  has two hazards found nowhere else in A6 (see "Two hazards only retirement
  can fix" below), which is reason enough on its own to keep it last and
  separate.

So the true shape is: `reporting ← convergence ← observables ← retirement`
(arrows point from dependent to dependency). Reporting is the one node with
zero outgoing edges into the rest of A6 — the textbook place to start, for
exactly the reason A1 started with constants: prove the plumbing on the
lowest-risk, most self-contained material first. Retirement is the one node
every other node feeds into — it cannot go anywhere but last.

This also minimizes churn from the interim "script-into-package" imports
every plan in this sequence uses (see "Cross-plan import direction" below):
if reporting lands first, convergence and observables can import the *final*
`gareus.mbar_analysis.*` location from day one instead of doing a temporary
`from analyze_gareus_mbar import ...` that A6a would later have to sweep and
rewrite.

## Precise ownership map

Every function below was resolved to exactly one plan; `git grep` confirms
no function name appears in two buckets. Numbers are line-inclusive
`(start_line, end_line, count)` as of `analyze_gareus_mbar.py` at commit
`024cea7` (current HEAD of this worktree) — **re-run the discovery script
below before executing any of A6b-A6e**, since A2-A5 landing first will not
move these line numbers (they touch different functions) but any bugfix
landing on `main` in the meantime could.

Discovery script used throughout this investigation (safe to re-run):

```python
import re
with open("analyze_gareus_mbar.py") as f:
    lines = f.readlines()
starts = [(i, m.group(1)) for i, l in enumerate(lines, 1)
          if (m := re.match(r'^def ([A-Za-z0-9_]+)\(', l))]
sizes = {name: (line, (starts[idx+1][0]-1 if idx+1 < len(starts) else len(lines)))
         for idx, (line, name) in enumerate(starts)}
```

### A6a — Plotting & Reporting (fully speced; see companion documents)

42 relocated functions + whole-file `gareus_plotstyle.py` → 5 new files under
`gareus/mbar_analysis/`. Full function list, exact line ranges, and file
grouping are in `2026-08-13-mbar-analysis-modularization-a6a-design.md`.
**Explicitly excludes** the three chignolin-kJ plotting functions
(`_chignolin_fes_range_label_kj`, `_plot_chignolin_fes_kj_range`,
`_plot_chignolin_fes_kj_multirange`, 101 lines) — see "Chignolin decision"
below; these move with A6c instead, not A6a, so the chignolin bundle relocates
as one atomic unit rather than being split across two plans. Also excludes
`write_convergence_report` (29 lines, **confirmed dead code** — `grep -n
"write_convergence_report" analyze_gareus_mbar.py` returns only its own `def`
line, zero call sites anywhere in the file or repo) — **delete, don't
relocate**, following this codebase's own stated convention of dropping
confirmed-unused code during an adjacent change rather than carrying it
forward.

**`_write_csv_rows` belongs here, in A6a — not in A6b as an earlier pass
through this investigation concluded.** Direct verification
(`grep -n "_write_csv_rows(" analyze_gareus_mbar.py`) shows 15 call sites,
not 3: `run_epoch_pmf_convergence`/`run_observable_pmf_convergence` (A6b,
3 sites) — but also `analyze_rg` (A6c, 1 site), `analyze_extra_observable_pmfs`
(A6d, 3 sites), `analyze_poincare_map` (A6d, 2 sites), and
`analyze_poincare_residue_torsions` (A6d, 1 site). A claim that its "only
callers anywhere in the file" are the two A6b convergence drivers does not
survive a direct grep. Since it is needed by A6b, A6c, *and* A6d, leaf-first
ordering requires it live in the one sub-plan that lands before all three of
its consumer domains — A6a, the true leaf — not in A6b, which is itself a
consumer, not the provider. A6a's own design spec is the source of truth for
this function's exact new location and signature.

### A6b — Convergence Orchestration (~1,427 lines, 19 functions)

```
checkpoint_steps_from_data          _get_convergence_mbar_cache
_convergence_mask_digest            _convergence_mbar_cache_key
run_pmf_convergence                 run_epoch_pmf_convergence
run_observable_pmf_convergence      _observable_pmf_from_logw
_epoch_source_annotations           _epoch_source_pooling_table
_epoch_source_aggregate_ns_info     _analyze_epoch_cv_exploration
_analyze_tica_epochs                _analyze_torsion_pca_scree
_load_epoch_dihedral_features       _get_checkpoint_steps_cache
_checkpoint_steps_cache_key         _aggregate_ns_for_fracs
_short_source_label                 _skipped_empty_epochs
```

`_write_csv_rows` (a generic list-of-dicts-to-CSV writer, 17 lines) is
called from `run_epoch_pmf_convergence`/`run_observable_pmf_convergence`
here — **but it is owned by A6a, not A6b**. Direct verification
(`grep -n "_write_csv_rows(" analyze_gareus_mbar.py`) shows 15 call sites,
not the 3 in this domain: also `analyze_rg` (A6c, 1 site),
`analyze_extra_observable_pmfs` (A6d, 3 sites), `analyze_poincare_map`
(A6d, 2 sites), and `analyze_poincare_residue_torsions` (A6d, 1 site).
Leaf-first ordering requires it live in the one sub-plan that lands before
all three of its consumer domains — A6a, the true leaf — not here, which is
itself a consumer. See A6a's design spec for its exact new location
(`gareus/mbar_analysis/writers.py`) and signature. A6b simply imports it
from there.

Key facts for whoever specs A6b:
- **Re-solves MBAR directly** (`solve_mbar`, from A3) inside
  `run_epoch_pmf_convergence` and `run_observable_pmf_convergence` — this is
  the domain's core behavior, not a bias to design around.
- **Two in-memory caches**, both keyed off the `args` namespace object
  (`args._convergence_mbar_cache`, `args._checkpoint_steps_cache` — created
  lazily, never persisted to disk, live only for one `analyze()` run). Keys
  are content-based tuples (shape, checkpoint step, `blake2b`-hashed sample
  mask, solver params); the checkpoint-steps cache additionally requires `is`
  identity on the cached `d.step` array before trusting a hit, closing a
  content-key collision gap.
- `gareus/diagnostics.py`'s `_sample_counts_by_window(window_arr, n_windows)`
  does **not** overlap with anything here — confirmed by direct comparison.
  It's a one-shot per-window census used once, as a pre-flight readiness
  gate (`validate_us_mbar_inputs`), over the whole dataset. Convergence's own
  counting is a scalar `n = count_nonzero(mask)` per growing time-prefix
  checkpoint, along a completely different axis (sampling time, not window
  index). **Keep separate — this is a "no action needed" finding**, not a
  missed dedup.
- Calls reporting (A6a, already landed by the time A6b executes) directly:
  `write_convergence_plots`, `_write_epoch_ess_plot`,
  `write_observable_convergence_plots`, `write_observable_convergence_report`,
  `_plot_basin_population_convergence`.
- Calls PMF-comparison functions from what is presumably A5's diagnostics
  domain: `pmf_probability`, `js_divergence_1d`, `pmf_rmse_1d`,
  `barrier_error_1d`, `identify_basins_1d`, `_compute_basin_populations`,
  `ess`, `norm_logw`. **Confirm A5's actual module name/import path before
  writing A6b's spec** — see "Interfaces A6 expects from A2-A5" below.
- Two convergence drivers (`run_epoch_pmf_convergence`,
  `run_observable_pmf_convergence`) are structurally parallel but not
  unified: the "generic" observable engine won on the PMF-building side
  (`run_pmf_convergence` is an 8-line wrapper delegating to
  `run_observable_pmf_convergence`), but `run_epoch_pmf_convergence` is a
  genuinely separate third driver (epoch-cumulative x-axis, not
  prefix-by-raw-step) that still calls the *legacy* `write_convergence_plots`
  rather than the generic `write_observable_convergence_plots`. This is a
  real pre-existing asymmetry (also: only `run_epoch_pmf_convergence`
  computes per-row `mbar_ess`/`gamd_reweight_ess`). Not this plan's job to
  unify — a pure relocation should carry the asymmetry over unchanged and
  flag it in scope notes, exactly as this document does.
- **`_bridge()`'s branch order is broken and `run_observable_pmf_convergence`
  is the exact function the pattern was written to protect — fix this before
  reusing `_bridge()` here, do not just copy it forward.** Plan A4's final
  whole-plan review (`docs/superpowers/sdd/.../2026-08-13-mbar-analysis-modularization-a4/final-review-report.md`
  — see Finding F1) found and reproduced that `gareus/mbar_analysis/pmf.py`'s
  `_bridge()` (`sys.modules.get('analyze_gareus_mbar')` checked *before*
  `__main__`) does **not** deliver the guarantee the A4 design spec claims
  for it. Under `python analyze_gareus_mbar.py <run_dir>`, several already-landed
  sibling modules (`gareus/mbar_analysis/loaders_union_parquet.py`,
  `loaders_adaptive.py`, `writers.py`, `plotting.py`) do a bare
  `from analyze_gareus_mbar import ...`, which registers a *second*,
  independent script copy under `sys.modules['analyze_gareus_mbar']` before
  any `_bridge()`-using function ever runs (`loaders_union_parquet.py`'s
  fires unconditionally on every adaptive-Parquet data load). `_bridge()`
  then matches branch 1 and returns that second, never-`parse_args()`'d
  copy — never `__main__`, the copy the CLI user's actual flags landed on.
  Today (A4 only) this is bounded: `pmf.py`'s three script-local bridged
  names (`ess`, `analyze_cv1_cv2_2d_fes`, `run_observable_pmf_convergence`)
  only diverge on `--sambar-*`/`--mbar-anderson-history`-derived state, and
  only when the paired `--convergence-sambar-*`/`--mbar-anderson-history`
  flag is left falsy. **A6b is exactly the plan that changes that**:
  `run_observable_pmf_convergence` (and its siblings above, once relocated
  here) becomes the module that itself resolves `_agm.<name>` for whatever
  still lives in the script at that point, and per A4's own design-spec
  rationale (`...-a4-design.md:143-152`), this exact function is the one the
  whole resolver was written to protect against a copy-divergent read. Do
  not carry `_bridge()`'s current branch order forward into A6b's own bridge
  calls (if any) or leave A4's copy unfixed and load-bearing by the time A6b
  ships. Recommended fix (already verified correct and given verbatim in the
  A4 final-review report): swap the two branches so `__main__` is checked
  first, then the module name, then fresh-import — strictly better in all
  three invocation shapes (`python analyze_gareus_mbar.py`, pytest/notebook,
  `python -m analyze_gareus_mbar`). One existing test needs re-stating
  (`test_bridge_prefers_already_loaded_real_module` in
  `tests/test_mbar_analysis_pmf_module.py`); the other two `_bridge()` branch
  tests remain valid unchanged. Fix this in `gareus/mbar_analysis/pmf.py`
  directly (not a new copy) as A6b's first task, before any new bridged call
  into `run_observable_pmf_convergence`'s new home is written against the
  broken ordering.

### A6c — Trajectory I/O Infra + Structural Observables (~1,090 lines into package + 284 to a new top-level file, 41 functions total)

**Into `gareus/mbar_analysis/`:**
```
# shared trajectory-finding/alignment infra (~453 lines, used by A6c and A6d)
_prepare_adaptive_merged_traj_dir      _adjusted_steps_for_merged_traj
_find_replica_trajectory               _find_all_replica_trajectory_segments
_trajectory_extensions                 _find_data_topology_path
_find_rg_topology_path                 _find_pca_topology_path
_find_extra_topology_path              _sample_aligned_trajectory_frames
_base_segment_resume_start             _sample_to_segment_frame
_saved_frame_index_from_step           _read_traj_interval
_read_traj_interval_from_epoch_dirs    _get_adaptive_epoch_traj_dirs
_find_first_existing_path              _find_explicit_topology_path
_find_default_topology_path            _find_trajectory_companion_topology_path
_trajectory_pattern_hint

# structural/Rg/PCA observables (~632 lines, chignolin excluded — see below)
_compute_rg_from_trajectories          analyze_rg
analyze_distance_rg_2d_fes             _fit_and_project_pca_from_trajectories
analyze_pca_2d_fes                     _pca_replica_plan
_load_pca_reference                    _aligned_flattened_coords_A
_trajectory_frame_count                _rg_from_segment_chunked
_chunk_local_frame_selection           _weighted_mean_std
_pmf_distribution_mean_std
```

**Into a new top-level `chignolin_fes_extension.py` (not `gareus/`):**
```
_compute_chignolin_distances (108 lines)     analyze_chignolin_fes (75 lines)
_chignolin_fes_range_label_kj (8 lines)      _plot_chignolin_fes_kj_range (66 lines)
_plot_chignolin_fes_kj_multirange (27 lines)
```

This sub-domain owns the trajectory-finding/alignment infra that A6d also
consumes — A6d must import it from here, not duplicate it (A6d executes
after A6c, so this is a forward reference in the right direction).

`_weighted_mean_std`/`_pmf_distribution_mean_std` are small stats helpers
physically embedded in the Rg block; confirm at implementation time that
their only consumer is `analyze_rg`/`write_rg_*` (adjacent code) before
finalizing ownership — if A4 or A5 also want them, that's a five-minute
coordination check, not a redesign.

**Real bugs found here, explicitly out of scope for this relocation plan**
(same convention as every other CLAUDE.md entry in this repo — found and
named, not silently fixed inside an unrelated relocation):
- `_compute_chignolin_distances` and `analyze_poincare_residue_torsions`
  (A6d) use raw segment `resume_start`/`seg_start` directly instead of
  correcting through `_base_segment_resume_start` the way
  `_compute_rg_from_trajectories`/`_pca_replica_plan`/`_extra_replica_plan`
  do — an inconsistency in the GaMD-calibration-step offset correction across
  trajectory-alignment call sites.
- `gareus/traj_consolidate.py` (a *different* tool, permanent disk
  consolidation via MDAnalysis, not the ephemeral symlink-merge this plan's
  `_prepare_adaptive_merged_traj_dir` does) independently has the same bug
  class the analysis script was fixed for on 2026-08-05: its
  `_replica_trajs` looks for exactly one literal `replica_{id:03d}.{ext}`
  file per baseline/topup directory with **no handling for
  `_resume_from_NNNNNNNN` segments at all** — any sub-run that was itself
  SIGTERM-interrupted-and-resumed will silently lose data when consolidated.
  Zero test coverage on this file. Not this plan's job to fix — flagging for
  a dedicated fix plan.

### A6d — Residue/Topology Observables + Poincaré + CV1×CV2 FES (~1,733 lines, 26 functions)

```
analyze_extra_observable_pmfs (399)     _sasa_total_A2
_internal_contact_counts                _peptide_solvent_contact_counts
_build_peptide_solvent_classification   _dssp_fraction_arrays
_dssp_valid_counts                      _dssp_residue_probability_rows
_extra_replica_plan                     _slug
_md_residue_label                       _wrap_degrees
_protein_traj                           _bin_indices_1d
_make_acc_1d                            _accumulate_1d
_pmf_from_probability_centers           _finalize_acc_1d
_make_acc_2d                            _accumulate_2d
_pmf2d_from_probability                 _finalize_acc_2d
analyze_poincare_map (434)              analyze_poincare_residue_torsions (451)
_poincare_torsions_chunked              analyze_cv1_cv2_2d_fes
```

Notes for whoever specs A6d:
- The streaming-accumulator family (`_bin_indices_1d` through
  `_finalize_acc_2d`, ~160 lines) implements incremental/memory-bounded
  histogram accumulation for a single chunked trajectory pass — a different
  algorithm from A4's whole-array `pmf_from_weights`/cumulant-expansion
  family, used only by `analyze_extra_observable_pmfs`. **Decision: A6d owns
  these** (private implementation detail of one orchestrator, not
  general-purpose PMF math another domain would want) — stated explicitly so
  A4's cross-check can confirm no conflicting claim.
- `analyze_cv1_cv2_2d_fes` (57 lines) and `analyze_poincare_map` (434 lines)
  are **not actually trajectory-based** — grep-confirmed zero
  `mdtraj`/`md.load`/`md.compute*` calls in either. Both are pure `d.cv`/
  `d.cv2`-array consumers. They stay in this sub-domain because they're
  thematically paired with `analyze_poincare_residue_torsions` (which *is*
  trajectory-based and takes `analyze_poincare_map`'s output as a required
  input) — flagging only so a future session doesn't add an unnecessary
  mdtraj import guard to code that never needed one.
- Imports the shared trajectory-finding infra from A6c's module(s).
- Calls reporting (A6a) and convergence (A6b), both already landed.

### A6e — Orchestrator Retirement (~348 lines, 3 functions, plus cleanup)

`analyze` (121 lines), `parse_args` (165 lines), `main` (62 lines), plus the
actual retirement mechanics (see "Retirement mechanics" below).

`analyze()` is **not** a giant function — it is already a thin, 121-line
orchestrator: solve MBAR once, optionally split into an epoch_000/rest
subset (`_epoch_zero_split_masks`/`_masked_data`/`_subset_logw_from_global_fk`
— **not in any A6 list**, presumably A4's territory since their sole other
caller is `run_pmf_and_gamd_boost_report`; confirm with A4's spec), call
~24 distinct domain functions, assemble one `s` dict, write
`pmf_summary.json`/`.md`. The `s` dict pattern is **"each sub-call returns
its own dict; `analyze()` is the sole place that does `s[key] = info`"** —
never "pass `s` in and let the callee mutate it." Any relocation must
preserve this contract exactly: A2-A6d's functions must keep returning plain
dicts, not accept and mutate a shared summary object.

**Two hazards only A6e can fix — call them out explicitly, don't defer them:**

1. **`parse_args()`'s `globals()` mutation.** After parsing, 9 MBAR-solver
   defaults (`DEFAULT_MBAR_BACKEND`, `SAMBAR_EPOCHS`,
   `SAMBAR_INITIAL_BATCH_SIZE`, `SAMBAR_BATCH_PATIENCE`, `SAMBAR_SEED`,
   `SAMBAR_LR_SCALE`, `SAMBAR_DELTA_F_MAX`, `SAMBAR_POLISH_BACKEND`,
   `MBAR_ANDERSON_HISTORY`) are written via a bare `globals()` call, wrapped
   in a swallow-all `try/except Exception: pass`. Today this mutates
   `analyze_gareus_mbar`'s own module namespace, which is exactly where the
   solver functions that read these defaults currently live — so it works.
   Once `parse_args` moves into `gareus/mbar_analysis/cli.py` but the solver
   defaults live in A3's module, `globals()` will silently mutate `cli.py`'s
   own namespace instead, and every solver call that relies on an
   unspecified CLI flag falling back to its module default will silently use
   the wrong value with **no error** (the `except Exception: pass` swallows
   everything, including an `AttributeError` if someone later tries to
   detect the mismatch). **A6e must replace this with an explicit
   `import gareus.mbar_analysis.<solver_module> as solvers; solvers.DEFAULT_MBAR_BACKEND = ...`
   (or an equivalent explicit-namespace mechanism) — not carry the
   `globals()` pattern forward as-is.** State this constraint to whoever
   specs A3 too, so A3's design doesn't assume its own defaults are
   externally immutable.
2. **`main()`'s dynamic sibling-script import.** `main()` conditionally loads
   `plot_adaptive_diagnostics.py` via
   `importlib.util.spec_from_file_location` resolved relative to
   `Path(__file__).parent` — i.e. "the directory this script lives in."
   Once this code lives in `gareus/mbar_analysis/cli.py`, `__file__` points
   inside the package, not next to `plot_adaptive_diagnostics.py` at the
   repo root. A6e must resolve this path a different way (e.g. relative to
   the repo root, or via an explicit config/env value) — a silent path
   change here means the adaptive-diagnostics companion figures silently
   stop being generated for every adaptive-production run, with only a
   swallowed one-line skip message (the existing bare `try/except Exception`
   around this call) as any indication.

**Retirement mechanics** (the actual "stop `python analyze_gareus_mbar.py`
being primary" step, requested explicitly in the A1 design spec):
- Move `analyze`/`parse_args`/`main` bodies into
  `gareus/mbar_analysis/cli.py`, replacing A1's placeholder delegation.
- `analyze_gareus_mbar.py` becomes a thin backward-compat shim: keep it
  importable (many test files still do
  `from analyze_gareus_mbar import <name>` per the 21 test files found — see
  "Backward compatibility" below) by re-exporting every name from its new
  `gareus.mbar_analysis.*` home, plus a `main`/`analyze`/`parse_args` that
  delegate into `gareus.mbar_analysis.cli` (the *mirror image* of A1's
  original delegation direction — package now owns the logic, script
  delegates into package).
- `pyproject.toml`'s `gareus-analyze` script (already registered by A1)
  becomes the documented primary entry point; update `README.md`/any
  onboarding docs that currently say `python analyze_gareus_mbar.py` (grep
  for it before writing A6e's task list).
- Sweep every temporary `from analyze_gareus_mbar import ...` left behind by
  A2-A5 and A6a-A6d's "script hosts the not-yet-relocated symbol" interim
  imports, pointing them at final `gareus.mbar_analysis.*` locations. This is
  mechanical but must be done via a repo-wide grep at execution time, not
  assumed from this document (which cannot see what A2-A5 actually shipped).

## Chignolin decision (applies to A6a, A6c)

`_compute_chignolin_distances`/`analyze_chignolin_fes` (183 lines) hardcode
`resSeq 3/7/8` atom selections and CLN025 native-hairpin H-bond distance
definitions; `_chignolin_fes_range_label_kj`/`_plot_chignolin_fes_kj_range`/
`_plot_chignolin_fes_kj_multirange` (101 lines) exist solely to plot that
data in kJ/mol with hardcoded distance-axis labels. All five are gated behind
`--chignolin_fes`, off by default, and are explicitly not portable to any
other peptide (the code's own error text: *"ensure topology matches chignolin
residue numbering (resSeq 1-10)"*).

**Recommendation: none of this moves into `gareus/mbar_analysis/`.**
`gareus/` is the general peptide-simulation package (per its own
`pyproject.toml` description, "Modular GaREUS/GaMD peptide simulation
workflow"); baking one 10-residue test peptide's H-bond geometry into it
would be the same category of coupling this whole modularization is meant to
remove. Instead: extract the full 284-line bundle into a new top-level
`chignolin_fes_extension.py` (repo root, same tier as `plot_adaptive_diagnostics.py`
and `gareus_report.py` — project-specific extensions that sit alongside the
package rather than inside it), imported by `analyze()` only when
`--chignolin_fes` is passed. This is a single decision applied once, honored
by both A6a (which must *not* try to relocate the kJ-plotting trio into
`gareus/mbar_analysis/reporting`-family modules) and A6c (which performs the
actual extraction, since it owns `_compute_chignolin_distances`/
`analyze_chignolin_fes`, the two functions that make the bundle worth moving
as a unit rather than piecemeal).

## `gareus_plotstyle.py` decision (A6a)

`grep -rl "gareus_plotstyle" --include='*.py' .` returns exactly two files:
`analyze_gareus_mbar.py` and its own `tests/test_gareus_plotstyle.py` — no
other script (`plot_adaptive_diagnostics.py`, the thermodynamic-validity
report generator, etc.) imports it. Migration cost is therefore two
call-site updates, both owned by A6a. **Recommendation: relocate the whole
128-line file, unchanged, into `gareus/mbar_analysis/plotstyle.py`** as part
of A6a (detailed in the companion A6a spec) rather than leaving it as a
stray top-level module once everything that consumes it lives inside the
package. `gareus_report.py` (a similarly-shaped top-level module — health
verdict / warning classification, imported by `analyze()`/`main()`) is
**not** part of this decision: it isn't in A6's assigned function list, has
its own consumers beyond this file (`tests/test_gareus_report.py`,
`tests/test_cumulant_shared_computation.py`), and A6e's retirement step
should simply keep importing it from its current top-level location.

## `gareus/colors.py` and `gareus/formatting.py` — no overlap found (A6a)

Both were checked in full against the plotting/reporting function list.
`gareus/colors.py` is ANSI terminal-escape styling for TUI console output
(different color space — named ANSI colors vs. matplotlib hex RGB — and a
different consumer, stdout dashboards vs. PNG rendering);
`analyze_gareus_mbar.py` never imports it. `gareus/formatting.py` has
functions that overlap `analyze_gareus_mbar.py`'s own CV-label helpers
(`_primary_cv_label`/`_primary_cv_units`/`_primary_cv_axis_label`) in
*purpose* only (both format "a primary CV value with units") — no shared
code, no shared constants, two independent implementations for two different
surfaces (this script's plot axes vs. the live TUI). Neither is imported by
`analyze_gareus_mbar.py` today. **No consolidation recommended** — this is a
"checked, no action" finding, included here because the task required
checking it explicitly.

## Interfaces A6 expects from A2-A5 (for the cross-check)

These are the exact names/signatures A6's five sub-plans call, as found by
direct code reading — **the coordinating cross-check should confirm each
still exists with a matching signature in whatever A2-A5 actually ship**,
since those specs were written in parallel and may have renamed things:

- **A2 (loading)**: the `Data` class/dataclass with attributes `d.cv`,
  `d.cv2`, `d.window`, `d.replica`, `d.step`, `d.beta`, `d.boost_kj`,
  `d.u_nk`, `d.centers`, `d.k_kcal`, `d.rg_A`, `d.meta` (dict), `d.prod_dir`
  (Path) — every A6 function receiving `d: Data` reads these directly.
  `_masked_data(d, mask)` and `_epoch_zero_split_masks(d)` (used by
  `analyze()`/A6e) — confirm which plan claims these (candidate: A4, per its
  sole other caller `run_pmf_and_gamd_boost_report`).
- **A3 (solvers/bias)**: `solve_mbar(u_nk, window, tol=, maxiter=,
  progress=, backend=, threads=, f_init=, sambar_epochs=,
  sambar_initial_batch_size=, sambar_batch_patience=, sambar_seed=,
  sambar_lr_scale=, sambar_delta_f_max=, sambar_polish_backend=) -> dict`
  (keys used downstream: `f_k`, `logw`, `converged`, `iterations`,
  `max_delta`, `backend`, `sambar_warmstart_epochs`,
  `sambar_polish_backend`) — called directly by A6b's convergence drivers.
  `norm_logw`, `ess`, `logsumexp*` — called by A6b/A6e.
  `reconstruct_bias_matrix` (in `gareus/query.py` today) — confirmed **not**
  called by anything in A6; only used at data-load time. Module-level
  solver-default globals (`DEFAULT_MBAR_BACKEND`, `SAMBAR_*`,
  `MBAR_ANDERSON_HISTORY`) — A6e needs to know their exact final module path
  to fix the `globals()` hazard above.
- **A4 (pmf)**: `make_bins`, `pmf_from_weights`, `pmf2d_from_weights`,
  `cumulant2`/`cumulant3`/`cumulant2_2d`/`cumulant3_2d`,
  `_cumulant_expansion_both`/`_cumulant_expansion_2d_both`, `boost_stats`,
  `_window_moments`, `_window_cv_mean_std` (confirmed via
  `2026-08-13-mbar-analysis-modularization-a4-design.md`, and independently
  by `grep -n "_window_cv_mean_std(" analyze_gareus_mbar.py`, which shows its
  *only* call site is inside A4's own `run_pmf_and_gamd_boost_report` at
  L9576 — an earlier pass through this document had mistakenly attributed it
  to A5 and claimed A6b calls it; A6b does not, corrected here),
  `run_pmf_and_gamd_boost_report` (calls A6a's `write_pmf`/`write_all`/
  `plot_outputs`/`_eff_smooth` directly — **A6a must update this import once
  it lands, wherever A4 puts this function**; also calls `_window_moments`
  indirectly via `_per_window_gamd_boost_stats`, which A6a's own
  `plot_gamd_boost` also needs — see A6a's design spec for the exact
  interim-import handling), `run_secondary_cv_analyses`/
  `analyze_secondary_cv_pmf` (calls plotting logic inline today per A6a's
  investigation — flag for A4's spec to either delegate to A6a's
  `_write_scalar_pmfs` or keep its own inline block, but decide explicitly).
- **A5 (diagnostics)**: `overlap_matrix`, `pmf_probability`,
  `js_divergence_1d`, `pmf_rmse_1d`, `barrier_error_1d`,
  `identify_basins_1d`, `_compute_basin_populations` — called by A6b
  (convergence, for reference-PMF comparison and basin tracking) and A4.
  `gareus/diagnostics.py`'s existing `_sample_counts_by_window` — confirmed
  no overlap with A6b, see above.

## Cross-plan import direction (applies to every A6 sub-plan)

Every sub-plan in this sequence follows A1's established strangler-fig
convention, applied in whichever direction is live at the time:

- **Before a name has been relocated**, any A6 sub-plan that needs it
  imports it from `analyze_gareus_mbar` (script → package direction,
  matching A1's `cli.py`).
- **After a name has been relocated**, `analyze_gareus_mbar.py` keeps a
  `from gareus.mbar_analysis.<module> import <name>` re-export so its own
  internal call sites (and the ~21 existing test files that still do
  `from analyze_gareus_mbar import <name>`) keep working unmodified, exactly
  as A1 did for `KJ_PER_KCAL`/`K_B_KJ_PER_MOL_K` (package → script direction
  for consumption, re-export for backward compatibility).
- Whichever plan relocates a function is responsible for updating every
  *other already-landed* A2-A6 module's import of that function — this is
  explicit, assigned work in A6a's plan (for A4's `run_pmf_and_gamd_boost_report`)
  and must be repeated in A6b-A6e's future specs for whatever they land on
  top of.
- Only A6e removes the re-export shims (or leaves them permanently, if
  `analyze_gareus_mbar.py`'s continued importability is still wanted as a
  compatibility surface — decide at A6e-spec time, not here).

## Out of Scope (all of A6a-A6e)

- Any behavior/numeric change. Every sub-plan is a pure relocation; identical
  inputs must produce byte-identical outputs before and after.
- The two real bugs found during this investigation (chignolin/Poincaré
  skipping `_base_segment_resume_start`; `traj_consolidate.py`'s unhandled
  `_resume_from_` segments) — named above, not fixed here.
- Unifying `run_epoch_pmf_convergence`'s legacy plotting path with
  `run_observable_pmf_convergence`'s generic one (A6b notes above) — a real
  duplication, not this sequence's job to remove.
- Consolidating the six near-identical 2D-FES CSV/NPZ writer pairs, or
  `_write_generic_2d_fes` vs. `_write_rama_2d` — noted in A6a's own spec as a
  found-but-deferred simplification opportunity.
- Full A6b/A6c/A6d/A6e design specs and bite-sized plans — this document
  scopes them; a future session writes them following A6a's template.
