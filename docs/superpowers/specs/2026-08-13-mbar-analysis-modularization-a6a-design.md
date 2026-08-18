# MBAR Analysis Modularization — Plan A6a (Plotting & Reporting)

## Goal

Relocate `analyze_gareus_mbar.py`'s plotting and file-output layer — 48
functions and 1 module-level constant (`_OPT_IN_GAMD_METHODS`, ~1,218 lines)
plus the standalone `gareus_plotstyle.py` (128 lines) — into five new,
size-bounded modules under `gareus/mbar_analysis/`,
with zero behavior change. This is Plan A6a of a five-plan split of the
original Plan A6 (see
`2026-08-13-mbar-analysis-modularization-a6-split-proposal.md` for the full
split rationale) and the **first** of those five to execute: this domain
calls out to nothing else in A6, and everything else in A6 calls into it
directly, so landing it first lets every later A6 sub-plan import its final
location from day one instead of a temporary script-import that would later
need sweeping.

## Problem

`analyze_gareus_mbar.py`'s plotting/reporting code is scattered across five
disjoint line ranges (roughly 3757-4003, 4282-4910, 5648-5684, 6060-6527,
7067-7409), interleaved with PMF-building, diagnostics, and trajectory
analysis functions throughout the middle 74% of a 9,946-line file. Concretely:

- There is no module boundary between "compute a PMF" and "write it to
  disk/render it" anywhere in this file — a reader has to know which of ~40
  functions with names like `write_*`/`plot_*`/`_write_*` exist and where,
  rather than importing one module.
- `gareus_plotstyle.py` (Okabe-Ito palette, per-method line styles, pretty
  names) lives as a bare top-level module next to the script that is its
  only consumer (confirmed: `grep -rl "gareus_plotstyle" --include='*.py' .`
  returns only `analyze_gareus_mbar.py` and its own test) — dead weight at
  the repo root once its consumer moves into the package.
- One function, `write_convergence_report` (29 lines), has zero callers
  anywhere in the repo (`grep -rn write_convergence_report .` finds only its
  own `def` line) — confirmed dead code, carried along by every earlier pass
  over this file.
- Real, if modest, near-duplication exists inside this domain itself: six
  2D-FES CSV/NPZ writer pairs (`write_2d_fes_csv`/`_npz`,
  `write_cv1_cv2_2d_fes_csv`/`_npz`, `write_pca_2d_fes_csv`/`_npz`) are
  field-renames of the same shape; `_write_generic_2d_fes` and
  `_write_rama_2d` are near-identical modulo hardcoded axis labels; the
  "legacy" CV1-specific `write_convergence_plots` duplicates the "generic"
  `write_observable_convergence_plots` almost panel-for-panel. None of these
  are fixed here (Out Of Scope) but the relocation groups them so a future
  consolidation pass has an obvious home to work in.
- Two functions (`_chignolin_fes_range_label_kj`, and its siblings) exist
  purely to plot one specific 10-residue test peptide's hardcoded H-bond
  geometry in the general analysis pipeline's plotting section — a
  project-specific leak into what should be a general package.

## Recommended Approach

**Five new modules under `gareus/mbar_analysis/`, each under 450 lines,
grouped by what they do rather than preserving the original file's
interleaved order; delete confirmed-dead code; defer chignolin-specific code
to a later plan instead of relocating it into the general package.**

1. `gareus/mbar_analysis/plotstyle.py` — the entire contents of
   `gareus_plotstyle.py`, unchanged, plus import-site updates at all **four**
   `gareus_plotstyle` call sites in `analyze_gareus_mbar.py` (L5671 inside
   `plot_rg_outputs`, L6425 inside `_write_scalar_pmfs`, L7313 inside
   `plot_outputs` — all three relocate with this plan — **and L7654 inside
   `analyze_secondary_cv_pmf`, which does NOT relocate with this plan** but
   still needs its bare `import gareus_plotstyle as ps` rewritten to
   `from gareus.mbar_analysis import plotstyle as ps`) plus
   `tests/test_gareus_plotstyle.py`. Verified via
   `grep -n "gareus_plotstyle" analyze_gareus_mbar.py`: exactly these 4 call
   sites exist, no more, no fewer — confirm this count is still 4 before
   deleting the file, in case another change added a 5th between when this
   spec was written and when it executes.

   **The L7654 fix is not optional, and the failure mode if it's skipped is
   worse than a crash: it is silent.** That import sits inside a
   `try: ... except Exception as e: warnings.append(...)` block
   (L7652-7670) that already exists to degrade gracefully when matplotlib is
   unavailable. Deleting `gareus_plotstyle.py` without fixing this line
   turns `ModuleNotFoundError: No module named 'gareus_plotstyle'` into just
   another caught exception: `analyze_secondary_cv_pmf` returns normally,
   `plot_file` stays `None`, `info['files']` never gets a `cv2_pmf_png` key,
   and `cv2_pmf_unbiased.png` silently stops being written for every
   secondary-CV-regime-split run — with only a warning string added to the
   run's `warnings` list as any trace, and zero existing test failure:
   `tests/test_secondary_cv_regime_split.py::test_regime_meta_preserves_dict_shape_with_regions`
   does call through to the real `analyze_secondary_cv_pmf` (confirmed by
   reading it — `_spy` wraps and calls `real_analyze`, it does not
   short-circuit), so it would execute this exact code path today, but it
   only asserts on `d_regime.meta['secondary_cv']` dict shape, never on
   `info['files']` or on whether the PNG was produced — so it stays green
   regardless of whether this plot silently disappears. Fix the import site;
   do not rely on any existing test to catch a regression here. The old
   top-level `gareus_plotstyle.py` is deleted outright once all 4 call sites
   (not 3) are updated, so no compatibility shim is needed.
2. `gareus/mbar_analysis/plotting.py` (~474 lines) — every function whose
   job is "render a matplotlib figure," plus the small numeric/label helpers
   that only plotting needs: the six CV-label formatters
   (`_secondary_cv_label`, `_secondary_cv_regions`, `_regime_slug`,
   `_primary_cv_label`, `_primary_cv_units`, `_primary_cv_axis_label` — see
   the CORRECTION below); smoothing (`_smooth_pmf_1d`,
   `_smooth_masked_grid`), the master-vs-specific sigma resolver
   (`_eff_smooth`), the 2D-FES multi-vmax-range machinery
   (`_fes_range_tag`/`_fes_range_label`/`_fes_variant_path`/
   `_plot_2d_fes_range`/`_plot_2d_fes_multirange` and its three thin
   wrappers `plot_2d_fes`/`plot_cv1_cv2_2d_fes`/`plot_pca_2d_fes`), and the
   three top-level output plotters (`plot_outputs`, `plot_gamd_boost`,
   `plot_rg_outputs`) together with `plot_gamd_boost`'s private stats helper
   (`_per_window_gamd_boost_stats`) and the two small GaMD-method-visibility
   filters (`_visible_pmfs`, `_want_gamd_method`).

   > **CORRECTION (cross-plan reconciliation pass) — the six CV-label
   > helpers are this plan's, and were missing from an earlier draft of this
   > list.** `_secondary_cv_label` (L323-340), `_secondary_cv_regions`
   > (L341-347), `_regime_slug` (L510-512), `_primary_cv_label` (L589-595),
   > `_primary_cv_units` (L596-602), and `_primary_cv_axis_label` (L603-608)
   > were claimed by **none** of the ten A1-A6a plans. This spec itself
   > recorded `_primary_cv_axis_label` as "unclaimed by any A2-A6 domain
   > description found so far" and bridged it back to
   > `analyze_gareus_mbar` with a local import (see the External Symbols
   > table below, now corrected); the reconciliation pass found the same
   > gap applied to all six. `...-a2-design.md`'s Out Of Scope explicitly
   > disclaims four of them ("they belong with whichever plan owns
   > secondary-CV PMF/2D-FES analysis — referenced but not claimed by this
   > plan"), and `...-a4-design.md`'s dependency table marks
   > `_secondary_cv_label`/`_secondary_cv_regions` `unowned` while resolving
   > them, and `_regime_slug`, through its call-time `_bridge()` helper.
   >
   > The coordinator assigned all six to `plotting.py`: each is a pure,
   > stateless `meta: dict -> str/list` formatter whose sole output is
   > matplotlib axis/annotation text, which is exactly this module's stated
   > domain, and two of their callers (`plot_outputs`, `plot_rg_outputs`)
   > are already relocated here. `_regime_slug` is included with them
   > despite producing a directory-name slug rather than an axis label: it
   > is the same one-line `meta`-derived-string shape, its only caller in
   > the file is A4's `run_secondary_cv_analyses`, and no other plan claims
   > it. **Consequences**: `_primary_cv_axis_label` stops being an interim
   > bridged import (it now lives beside its callers); `_regime_slug` brings
   > a new interim `_slug` dependency into `plotting.py`, handled exactly
   > like `writers.py`'s existing one; A4's `_bridge()` resolution is
   > unaffected, since it goes through `analyze_gareus_mbar`'s namespace,
   > which this plan's re-export block keeps populated.
3. `gareus/mbar_analysis/writers.py` (~265 lines) — every function whose job
   is "serialize an already-built PMF/FES dict (or plain list-of-dicts) to
   CSV or NPZ," including the two consolidating writers that already call
   into `plotting.py` (`_write_generic_2d_fes`, `_write_rama_2d`,
   `_write_scalar_pmfs`), and the generic `_write_csv_rows` helper. This last
   one is not itself a "reporting" function by name, but direct verification
   (`grep -n "_write_csv_rows(" analyze_gareus_mbar.py`) shows it has 15 call
   sites spanning three domains that all execute *after* A6a — 10 inside
   A6b's two convergence drivers, 1 inside A6c's `analyze_rg`, and 4 across
   A6d's `analyze_extra_observable_pmfs`/`analyze_poincare_map`/
   `analyze_poincare_residue_torsions` — so it must land in the one plan
   that precedes all three consumers, which is this one.
4. `gareus/mbar_analysis/convergence_reporting.py` (~390 lines) — the
   convergence-specific plot/report writers
   (`write_convergence_plots`, `write_observable_convergence_plots`,
   `write_observable_convergence_report`, `_write_epoch_ess_plot`,
   `_plot_basin_population_convergence`) and their two shared matplotlib-axis
   helpers (`_add_epoch_annotations_to_axes`,
   `_add_aggregate_ns_secondary_axis`). `write_convergence_report` (the dead
   sibling of `write_observable_convergence_report`) is **deleted**, not
   relocated.
5. `gareus/mbar_analysis/summary.py` (~85 lines) — the three functions that
   render `pmf_summary.md` from the already-fully-assembled `s` dict
   (`summary_md`, `_render_health_section_md`, `_key_diagnostics_md`).

**Explicitly excluded from this plan and left untouched in
`analyze_gareus_mbar.py`:** the three chignolin-kJ plotting functions
(`_chignolin_fes_range_label_kj`, `_plot_chignolin_fes_kj_range`,
`_plot_chignolin_fes_kj_multirange`, 101 lines). Per the split proposal's
chignolin decision, these move together with `_compute_chignolin_distances`/
`analyze_chignolin_fes` into a new top-level `chignolin_fes_extension.py`
when Plan A6c executes (A6c owns the two functions that make extracting the
bundle worthwhile) — moving only the plotting third of that bundle here would
split one project-specific feature across two unrelated plans and two
different final homes for no benefit.

`analyze_gareus_mbar.py` keeps every relocated name resolvable at its
original qualified path via `from gareus.mbar_analysis.<module> import
<name>` — so its own internal callers (e.g. `run_pmf_and_gamd_boost_report`,
which calls `write_pmf`/`write_all`/`plot_outputs`) and the ~21 existing test
files that do `from analyze_gareus_mbar import <name>` need zero changes.

## Why This Approach

- **Reporting has zero outgoing calls into the rest of A6** (confirmed by
  reading every one of the 47 function bodies — each receives already-
  computed `pmf`/`fes` dicts, `warnings` lists, and `d`/`args` objects as
  plain parameters and only reads them). The single exception, unchanged in
  kind by the six-CV-label-helper addition, is `_slug` — a pure
  string-slugifier in A6d's scope, now called from two relocated bodies
  (`_regime_slug` in `plotting.py`, `_write_rama_2d` in `writers.py`)
  instead of one; see the External Symbols table. It is the one A6
  sub-domain with essentially no
  dependency on anything else still left in `analyze_gareus_mbar.py`,
  matching A1's own "prove the plumbing on the lowest-risk material" logic:
  a pure file-I/O-and-matplotlib layer has no numerical-correctness stakes
  of its own to get wrong.
- **Landing this first, rather than last (as the un-investigated example
  split suggested), avoids double work.** Convergence orchestration
  (A6b) and trajectory observables (A6c/A6d) both call reporting functions
  *directly, inline* — not through some later top-level assembly step — so
  if reporting moved last, every one of those plans would need a temporary
  `from analyze_gareus_mbar import write_pmf, ...` that A6a would then have
  to find and rewrite once it finally landed. Landing reporting first means
  A6b/A6c/A6d import the real, final location immediately.
- **Five files instead of one** keeps every new file under this repo's
  800-line ceiling (`coding-style.md`: "200-400 lines typical, 800 max") and
  groups by what a reader would actually go looking for ("how does a PMF get
  written to CSV" vs. "how does a PMF get plotted" vs. "how does a
  convergence report get built" are three different questions with three
  different answers today, scattered across the same file).
- **Deleting `write_convergence_report` instead of relocating it** avoids
  moving 29 lines of code with a real hazard (a plausible-looking but
  never-actually-exercised markdown renderer) into a location where its
  deadness would be less obvious than "sits three lines from its only
  living relative in the original file."
- **Deferring chignolin extraction to A6c** keeps this plan's chignolin
  decision to exactly one sentence ("we're not touching those three
  functions") instead of half-relocating a project-specific feature.

## Alternatives Considered

### One `reporting.py` file for everything

Rejected: 47 functions plus a 128-line style module would be ~1,328 lines in
one file, well past this repo's stated 800-line ceiling, and would bury five
genuinely different concerns (style constants, matplotlib rendering, CSV/NPZ
serialization, convergence-specific reporting, markdown summary rendering)
behind one name.

### Move the chignolin-kJ plotting trio here now, since it's "plotting"

Considered, since the three functions are textually plotting code and this
is the plotting plan. Rejected: `_compute_chignolin_distances`/
`analyze_chignolin_fes` (the functions that actually decide *whether* this
code is worth having) don't move until A6c, and splitting one feature's
extraction across two plans/two final destinations (a `gareus/mbar_analysis/`
module now, a top-level `chignolin_fes_extension.py` later) creates a dangling
intermediate state with no benefit over doing it once, atomically, in A6c.

### Keep `gareus_plotstyle.py` at the top level, just update its import path

Considered, since it has only one real consumer file and moving it is
optional work. Rejected once that consumer (`analyze_gareus_mbar.py`,
`tests/test_gareus_plotstyle.py`) was confirmed to be the only one: leaving
a single-consumer-file top-level module next to a script that's being
retired (per A6e) just relocates the untidiness one plan later for no
reason, and the move costs exactly five import-line edits today (four call
sites inside `analyze_gareus_mbar.py` — three relocating with this plan, one,
L7654 inside `analyze_secondary_cv_pmf`, not relocating but still needing its
import target updated — plus one in the test file).

### Fix the 2D-FES writer near-duplication (six pairs → one generic writer) while relocating

Considered, since the relocation naturally groups all six pairs together and
the duplication is real. Rejected for this plan: consolidating call sites
changes what CSV columns exist and in what order for three of the six pairs
(the ones with axis-specific field names), which is a behavior-adjacent
change this plan's own success criterion ("byte-identical output
before/after") explicitly rules out. Left as a named, deferred finding
(Out Of Scope) instead.

## Architecture

### Files touched

```
gareus/mbar_analysis/plotstyle.py               CREATE  (~128 lines, from gareus_plotstyle.py)
gareus/mbar_analysis/plotting.py                CREATE  (~474 lines)
gareus/mbar_analysis/writers.py                 CREATE  (~265 lines)
gareus/mbar_analysis/convergence_reporting.py   CREATE  (~390 lines)
gareus/mbar_analysis/summary.py                 CREATE  (~85 lines)
gareus_plotstyle.py                             DELETE  (128 lines, fully relocated)
analyze_gareus_mbar.py                          MODIFY  (delete 48 relocated function bodies
                                                           + 1 dead function; add 6 re-export
                                                           import blocks in their place; PLUS one
                                                           1-line fix at L7654, inside
                                                           analyze_secondary_cv_pmf, which is NOT
                                                           relocated — see Recommended Approach)
tests/test_gareus_plotstyle.py                  MODIFY  (1-line import path update)
tests/test_perf_plotting_redundancy.py          MODIFY  (rewrite one monkeypatch target —
                                                           see Error Handling)
gareus/mbar_analysis/pmf.py (or wherever A4      MODIFY  (update write_pmf/write_all/
  landed run_pmf_and_gamd_boost_report —          plot_outputs/_eff_smooth import —
  discover exact path at execution time)          see Task 9)
```

### Function-to-file map (exact current line ranges, `analyze_gareus_mbar.py` @ `024cea7`)

**`plotstyle.py`** — whole-file relocation of `gareus_plotstyle.py`:
`OKABE_ITO`, `OKABE_ITO_LINES`, `METHOD_STYLE`, `_PRETTY`, `BAR_COLOR`,
`REFERENCE_GREY`, `method_style()`, `method_color()`, `pretty_method()`,
`plot_method_curve()`, `style_line_axes()`, `annotate_minimum()`.

**`plotting.py`**:
```
_secondary_cv_label           L323-340  (18)
_secondary_cv_regions         L341-347  ( 7)
_regime_slug                  L510-512  ( 3)
_primary_cv_label             L589-595  ( 7)
_primary_cv_units             L596-602  ( 7)
_primary_cv_axis_label        L603-608  ( 6)
_smooth_masked_grid           L3822-3841 (20)
_smooth_pmf_1d                L3842-3862 (21)
_eff_smooth                   L3863-3871 ( 9)
FES_PLOT_VMAX_VALUES (const)  L3870      ( 1, module-level tuple constant)
_fes_range_tag                L3872-3879 ( 8)
_fes_range_label              L3880-3886 ( 7)
_fes_variant_path              L3887-3891 ( 5)
_plot_2d_fes_range             L3892-3961 (70)
_plot_2d_fes_multirange        L3962-3986 (25)
plot_2d_fes                    L3987-3994 ( 8)
plot_cv1_cv2_2d_fes            L3995-4002 ( 8)
plot_pca_2d_fes                L6078-6085 ( 8)
_per_window_gamd_boost_stats   L7119-7182 (64)
plot_gamd_boost                L7183-7307 (125)
plot_outputs                   L7308-7324 (17)
plot_rg_outputs                L5665-5683 (19)
_OPT_IN_GAMD_METHODS (const)   L6386      ( 1, module-level dict constant)
_want_gamd_method              L6388-6394 ( 7)
_visible_pmfs                  L6395-6399 ( 5)
```
(Note: `FES_PLOT_VMAX_VALUES = (2.0, 5.0, 10.0, 20.0, None)` is defined
in-line at L3870, between `_eff_smooth` and `_fes_range_tag` — move it with
this group, not as a separate step. Likewise `_OPT_IN_GAMD_METHODS =
{'gamd_exponential': 'plot_gamd_exponential', 'gamd_cumulant3':
'plot_gamd_cumulant3'}` sits at L6386, immediately before `_want_gamd_method`
— its only reader — and must move with it, or `_want_gamd_method` raises
`NameError` in its new home.

**`_window_moments` (L7104-7116) does NOT move with this plan**, despite
sitting immediately above `_per_window_gamd_boost_stats`, its sole caller:
`docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a4-design.md`
explicitly claims it (alongside `boost_stats` and `_window_cv_mean_std`) for
`gareus/mbar_analysis/pmf.py`. `_per_window_gamd_boost_stats` must import it
from there — or, if A4 hasn't landed yet at this plan's execution time, fall
back to importing it from `analyze_gareus_mbar` — never define its own copy.
See the implementation plan's Task 2 for the exact defensive
`try/except ImportError` pattern.)

**`writers.py`**:
```
write_2d_fes_csv               L3757-3779 (23)
write_2d_fes_npz                L3780-3793 (14)
write_cv1_cv2_2d_fes_csv        L3794-3807 (14)
write_cv1_cv2_2d_fes_npz        L3808-3821 (14)
write_pca_2d_fes_csv            L6060-6073 (14)
write_pca_2d_fes_npz            L6074-6077 ( 4)
_want_gamd_method-dependent:
_write_scalar_pmfs              L6407-6440 (34)
_write_generic_2d_fes           L6441-6483 (43)
_write_rama_2d                  L6484-6527 (44)
write_rg_pmf                    L5648-5657 (10)
write_rg_all                    L5658-5664 ( 7)
write_pmf                       L7067-7076 (10)
write_all                       L7077-7083 ( 7)
write_cv2_pmf                   L7084-7093 (10)
_write_csv_rows                 L4440-4456 (17)
```
(`_write_csv_rows` is a generic list-of-dicts-to-CSV writer with no
FES/PMF-shape assumptions — it has no call-site dependency on anything else
in this file. It is grouped into `writers.py` on ownership grounds, not
adjacency: its 15 call sites are spread across Plans A6b/A6c/A6d, all of
which execute after this plan, so it must live wherever lands first.)
(`_write_scalar_pmfs`/`_write_generic_2d_fes`/`_write_rama_2d` call
`_smooth_pmf_1d`/`_plot_2d_fes_multirange`/`_want_gamd_method` from
`plotting.py` — `writers.py` imports from `plotting.py`, never the reverse.
Land `plotting.py` first, per the task order below.)

**`convergence_reporting.py`**:
```
_add_epoch_annotations_to_axes       L4282-4297 (16)
_write_epoch_ess_plot                 L4382-4425 (44)
write_convergence_plots               L4457-4575 (119)
[write_convergence_report — DEAD, DELETE, do not relocate — L4576-4604]
_add_aggregate_ns_secondary_axis      L4353-4381 (29)
write_observable_convergence_plots    L4631-4745 (115)
_plot_basin_population_convergence    L4746-4783 (38)
write_observable_convergence_report   L4881-4909 (29)
```

**`summary.py`**:
```
_render_health_section_md     L7325-7333 ( 9)
_key_diagnostics_md           L7334-7368 (35)
summary_md                    L7369-7409 (41)
```

### Cross-references between the five new files

```
writers.py               imports from plotting.py: _smooth_pmf_1d, _plot_2d_fes_multirange,
                                                     _want_gamd_method
plotting.py              imports from plotstyle.py (as `gareus.mbar_analysis.plotstyle`,
                                                      matching the original `import
                                                      gareus_plotstyle as ps` alias)
convergence_reporting.py imports from plotting.py: _smooth_pmf_1d
summary.py               imports nothing from the other four (pure dict-to-markdown renderer)
```

No cycles. `plotting.py` and `plotstyle.py` must exist before `writers.py`
and `convergence_reporting.py` are written (reflected in the task order).

### External symbols consumed but not owned by this plan

Two names, found by reading the relocated function bodies rather than by
name alone, are called by code this plan moves but are not themselves part
of this plan's scope. Each is handled with a local, function-body import
(never a module-level import, to avoid a circular import back into
`analyze_gareus_mbar.py`, which itself imports from these new modules):

| Symbol | Called by (this plan) | Owner | Where defined at this plan's execution time |
|---|---|---|---|
| `_window_moments` | `_per_window_gamd_boost_stats` (`plotting.py`) | Plan A4 (`gareus/mbar_analysis/pmf.py`, per A4's own design spec) | `gareus.mbar_analysis.pmf` if A4 has landed, else `analyze_gareus_mbar.py` (L7104-7116) — handled with a defensive `try/except ImportError` since A4's landing status at this plan's execution time is not guaranteed |
| `_slug` | `_regime_slug` (`plotting.py`), `_write_rama_2d` (`writers.py`) | Plan A6d (its own future "residue/topology observables" scope) | `analyze_gareus_mbar.py` (L6165-6177) |

Neither is relocated by this plan. The implementation plan's Tasks 2
and 3 give the exact import statements and placement.

`_primary_cv_axis_label` was a third row here in an earlier draft, listed as
"unclaimed by any A2-A6 domain description found so far" and bridged back to
the script from `plot_outputs`/`plot_rg_outputs`. It is no longer an external
symbol: the cross-plan reconciliation pass confirmed it and its five siblings
were unowned across all ten A1-A6a plans, and the coordinator assigned all six
to this plan's `plotting.py` — see the CORRECTION under Recommended Approach.
`_regime_slug` arriving with them is what added `plotting.py` to `_slug`'s
"called by" cell above.

### Signatures preserved exactly (no parameter changes)

```python
def _secondary_cv_label(meta: dict) -> str: ...
def _secondary_cv_regions(meta: dict) -> list: ...
def _regime_slug(regime: str) -> str: ...
def _primary_cv_label(meta: dict) -> str: ...
def _primary_cv_units(meta: dict) -> str: ...
def _primary_cv_axis_label(meta: dict) -> str: ...
def _smooth_pmf_1d(pmf, sigma): ...
def _smooth_masked_grid(grid, sigma=1.0): ...
def _eff_smooth(args, attr: str) -> float: ...
def _fes_range_tag(vmax) -> str: ...
def _fes_range_label(vmax, actual_max: float) -> str: ...
def _fes_variant_path(base_path, vmax) -> Path: ...
def _plot_2d_fes_range(F, xedges, yedges, xc, yc, out_png, title, xlabel, ylabel, warnings, *,
                        smooth_sigma=1.0, range_vmax=None, figsize=(8.8,6.6), dpi=220,
                        cmap_name='viridis', contour=True, y_annotation_lines=None) -> bool: ...
def _plot_2d_fes_multirange(F, xedges, yedges, xc, yc, out_png, title, xlabel, ylabel, warnings, *,
                             smooth_sigma=1.0, figsize=(8.8,6.6), dpi=220, cmap_name='viridis',
                             contour=True, y_annotation_lines=None) -> dict[str,str]: ...
def plot_2d_fes(fes, method, out_png, title, warnings, smooth_sigma=1.0): ...
def plot_cv1_cv2_2d_fes(fes, method, out_png, title, warnings, smooth_sigma=1.0,
                         cv2_label='Secondary CV', cv1_label='CV distance (A)', regions=None): ...
def plot_pca_2d_fes(fes: dict, method: str, out_png: Path, title: str, warnings: list[str],
                     smooth_sigma: float = 1.0) -> dict[str,str]: ...
def plot_outputs(d, pmfs, selected, O, out, warnings, smooth_sigma=0.0, args=None): ...
def plot_gamd_boost(d, out, warnings): ...
def plot_rg_outputs(d: Data, rg_pmfs: dict, selected: str, out: Path, warnings: list[str],
                     smooth_sigma: float = 0.0, args=None): ...
def _per_window_gamd_boost_stats(window, comb_kcal, kbt_kcal, K, dih_kcal=None): ...
_OPT_IN_GAMD_METHODS = {"gamd_exponential": "plot_gamd_exponential", "gamd_cumulant3": "plot_gamd_cumulant3"}  # module constant, not a function
def _visible_pmfs(pmfs: dict, chosen: str, args) -> dict: ...
def _want_gamd_method(method: str, chosen: str, args) -> bool: ...
def write_pmf(path, pmf, method, extra=None): ...
def write_all(path, pmfs): ...
def write_cv2_pmf(path, pmf, method, extra=None): ...
def _write_csv_rows(path: Path, rows: list[dict]) -> None: ...
def write_rg_pmf(path, pmf, method, extra=None): ...
def write_rg_all(path, pmfs): ...
def write_2d_fes_csv(path, fes, method): ...
def write_2d_fes_npz(path, fes, method): ...
def write_cv1_cv2_2d_fes_csv(path, fes, method): ...
def write_cv1_cv2_2d_fes_npz(path, fes, method): ...
def write_pca_2d_fes_csv(path: Path, fes: dict, method: str) -> None: ...
def write_pca_2d_fes_npz(path: Path, fes: dict, method: str) -> None: ...
def _write_generic_2d_fes(out_dir: Path, prefix: str, title: str, xlabel: str, ylabel: str,
                           pmfs2d: dict, selected_method: str, warnings: list[str],
                           x_field: str='x', y_field: str='y', x_unit: str='', y_unit: str='',
                           smooth_sigma: float = 1.0, args=None) -> dict: ...
def _write_rama_2d(out_dir: Path, residue_label: str, pmfs2d: dict, selected_method: str,
                    warnings: list[str], smooth_sigma: float = 1.0, args=None) -> dict: ...
def _write_scalar_pmfs(out_dir: Path, prefix: str, label: str, xlabel: str, pmfs: dict,
                        selected_method: str, warnings: list[str], smooth_sigma: float = 0.0,
                        args=None) -> dict: ...
def write_convergence_plots(conv_rows: list[dict], pmf_rows: list[dict], summary_rows: list[dict],
                             out: Path, args, warnings: list[str], *, epoch_annotations: list = None,
                             aggregate_ns: Optional[np.ndarray] = None) -> list[str]: ...
def write_observable_convergence_plots(conv_rows: list[dict], pmf_rows: list[dict],
                                        summary_rows: list[dict], out: Path, args, warnings: list[str],
                                        *, prefix: str, metric_label: str, x_label: str,
                                        smooth_sigma: float = 0.0, epoch_annotations: list = None,
                                        aggregate_ns: Optional[np.ndarray] = None) -> list[str]: ...
def write_observable_convergence_report(path: Path, row: dict, conv_rows: list[dict],
                                         warnings: list[str], metric_label: str) -> None: ...
def _write_epoch_ess_plot(conv_rows: list, out: Path, *, ea: list = None,
                           aggregate_ns: Optional[np.ndarray] = None) -> Optional[str]: ...
def _plot_basin_population_convergence(basin_pop_rows: list, basins: list, final_pmf: dict,
                                        out: Path, file_prefix: str, cv_label: str, warnings: list,
                                        smooth_sigma: float = 0.0) -> list: ...
def _add_epoch_annotations_to_axes(axes_list: list, epoch_annotations: list) -> None: ...
def _add_aggregate_ns_secondary_axis(ax, x_frac: np.ndarray, agg_ns: np.ndarray) -> None: ...
def summary_md(path, s): ...
def _render_health_section_md(s): ...
def _key_diagnostics_md(s): ...
```

### Imports each new file needs

None of these functions import anything from elsewhere in
`analyze_gareus_mbar.py` except each other (within the same new file) and
the `plotting.py`→`writers.py`/`convergence_reporting.py` edges above.
Standard-library/third-party imports needed per file:

- `plotstyle.py`: `matplotlib` colors only (unchanged from the original file).
- `plotting.py`: `from __future__ import annotations`; `numpy as np`;
  `math`; `shutil`; `from pathlib import Path`; `from typing import
  Optional`; lazy per-function `import matplotlib.pyplot as plt` (and
  `matplotlib.patches`/`matplotlib.cm` where the original used them) —
  **preserve the lazy-import-inside-function pattern exactly**, do not hoist
  these to module level (the original file does this deliberately so a
  missing matplotlib degrades gracefully per-call rather than failing
  import of the whole module); `from . import plotstyle` (or
  `from gareus.mbar_analysis import plotstyle as ps` to keep the original
  `ps.` call-site prefix unchanged).
- `writers.py`: `numpy as np`; `csv`; `from pathlib import Path`; `from
  .plotting import _smooth_pmf_1d, _plot_2d_fes_multirange,
  _want_gamd_method`.
- `convergence_reporting.py`: `numpy as np`; `from pathlib import Path`;
  `from typing import Optional`; lazy `import matplotlib.pyplot as plt`
  (plus `matplotlib.ticker`, `matplotlib.cm` where used); `from .plotting
  import _smooth_pmf_1d`.
- `summary.py`: no third-party imports beyond what plain dict/string
  formatting needs (confirm exact set by reading the three function bodies
  at implementation time — they were not fully transcribed into this spec).

**Important existing convention to preserve**: `analyze_gareus_mbar.py`
aliases the stdlib `warnings` module as `_warnings` at module scope because
so many of its functions use `warnings` as a parameter name (a list of
strings to append to) — see its own top-of-file comment. None of A6a's 45
functions need the stdlib `warnings` module (they only ever *receive* a
`warnings: list[str]` parameter and `.append()` to it) — so the new files
should simply never import the stdlib `warnings` module, sidestepping the
shadowing issue entirely rather than replicating the `_warnings` alias.

### `analyze_gareus_mbar.py` after relocation (representative excerpt)

```python
from gareus.mbar_analysis.plotstyle import (
    OKABE_ITO, OKABE_ITO_LINES, METHOD_STYLE, method_style, method_color,
    pretty_method, plot_method_curve, style_line_axes, annotate_minimum,
)
# ...  (the block below lands at the old L323, where _secondary_cv_label was;
#       the next one at the old L3822 -- two separate plotting re-exports, one
#       per relocated cluster, each where its definitions lived)
from gareus.mbar_analysis.plotting import (
    _secondary_cv_label, _secondary_cv_regions, _regime_slug,
    _primary_cv_label, _primary_cv_units, _primary_cv_axis_label,
)
from gareus.mbar_analysis.plotting import (
    _smooth_masked_grid, _smooth_pmf_1d, _eff_smooth, FES_PLOT_VMAX_VALUES,
    _fes_range_tag, _fes_range_label, _fes_variant_path, _plot_2d_fes_range,
    _plot_2d_fes_multirange, plot_2d_fes, plot_cv1_cv2_2d_fes, plot_pca_2d_fes,
    _per_window_gamd_boost_stats, plot_gamd_boost, plot_outputs, plot_rg_outputs,
    _OPT_IN_GAMD_METHODS, _want_gamd_method, _visible_pmfs,
)
from gareus.mbar_analysis.writers import (
    write_2d_fes_csv, write_2d_fes_npz, write_cv1_cv2_2d_fes_csv, write_cv1_cv2_2d_fes_npz,
    write_pca_2d_fes_csv, write_pca_2d_fes_npz, _write_scalar_pmfs, _write_generic_2d_fes,
    _write_rama_2d, write_rg_pmf, write_rg_all, write_pmf, write_all, write_cv2_pmf,
    _write_csv_rows,
)
from gareus.mbar_analysis.convergence_reporting import (
    _add_epoch_annotations_to_axes, _write_epoch_ess_plot, write_convergence_plots,
    _add_aggregate_ns_secondary_axis, write_observable_convergence_plots,
    _plot_basin_population_convergence, write_observable_convergence_report,
)
from gareus.mbar_analysis.summary import _render_health_section_md, _key_diagnostics_md, summary_md
```
placed where the original function definitions were (so a reader diffing the
file sees an import block exactly where the code used to live, not a jump to
the top of the file), with the 47 function bodies plus the 1 dead function
(`write_convergence_report`) deleted from their original locations.

## Error Handling

Nothing new to guard: every relocated function's error handling (the
`try/except` around lazy `matplotlib` imports, the graceful-degradation
`warnings.append(...); return False/None` patterns) moves verbatim. This
plan introduces exactly one new failure mode to watch for — **a relocated
function silently changing behavior because it lost access to a name it used
to resolve via the same module's global namespace** — guarded against by the
identity tests in Testing below (every relocated name must be proven to be
literally the same object, not a coincidentally-equal reimplementation) and
by running every existing test that exercises this domain unmodified.

**A second, distinct failure mode this plan must explicitly guard against:
a monkeypatch that stops being effective after a relocation, without
raising any error at all.** A re-export shim (`from
gareus.mbar_analysis.plotting import _plot_2d_fes_range` inside
`analyze_gareus_mbar.py`) keeps `agm._plot_2d_fes_range` resolvable and keeps
direct calls to it working — but it does **not** make
`monkeypatch.setattr(agm, "_plot_2d_fes_range", spy)` visible to
`_plot_2d_fes_multirange` once both functions live together in
`gareus/mbar_analysis/plotting.py`: `_plot_2d_fes_multirange`'s internal call
to `_plot_2d_fes_range(...)` resolves the name via `plotting.py`'s own
`__globals__` after the move, not `analyze_gareus_mbar`'s, so patching the
old module's attribute silently has no effect and the test's call-count
assertion will read as if the wrapped function was never invoked — a false
negative, not a crash, and easy to miss in a large pytest run.
**Confirmed live instance**: `tests/test_perf_plotting_redundancy.py`
(L286-331) has exactly two tests of this shape —
`test_plot_2d_fes_multirange_renders_each_range_exactly_once` (patches
`agm._plot_2d_fes_range`, L302, asserts
`calls == list(agm.FES_PLOT_VMAX_VALUES)` at L308) and
`test_plot_chignolin_fes_kj_multirange_renders_each_range_exactly_once`
(patches `agm._plot_chignolin_fes_kj_range`, L325) — both of which import
`analyze_gareus_mbar as agm` locally and call
`agm._plot_2d_fes_multirange(...)` / `agm._plot_chignolin_fes_kj_multirange(...)`.
Only the first of these two tests is affected by this plan (both
`_plot_2d_fes_range` and `_plot_2d_fes_multirange` relocate together into
`plotting.py`); the second is unaffected here since
`_plot_chignolin_fes_kj_range`/`_plot_chignolin_fes_kj_multirange` are
explicitly out of scope for this plan (Out Of Scope) and stay in
`analyze_gareus_mbar.py` together, so their mutual monkeypatch keeps working
unchanged until Plan A6c relocates that pair — which must repeat this exact
check when it does.

The affected test will **fail loudly, not pass silently**, once
`_plot_2d_fes_range`/`_plot_2d_fes_multirange` both live in
`gareus/mbar_analysis/plotting.py`: `_plot_2d_fes_multirange`'s internal call
to `_plot_2d_fes_range(...)` resolves via `plotting.py`'s own `__globals__`,
so `monkeypatch.setattr(agm, "_plot_2d_fes_range", counting)` — which only
rebinds `agm`'s own attribute — never intercepts it; the real
`_plot_2d_fes_range` runs directly, `calls` stays `[]`, and
`assert calls == list(agm.FES_PLOT_VMAX_VALUES)` fails
(`[] != [2.0, 5.0, 10.0, 20.0, None]`). This is a real, deterministic test
failure this plan will introduce if left unaddressed — not a stealthy
false-negative — so it must be fixed as part of this plan's own task list,
not discovered later as a surprise CI failure. Task 4 below (`plotting.py`)
must rewrite `test_plot_2d_fes_multirange_renders_each_range_exactly_once`
to patch `gareus.mbar_analysis.plotting._plot_2d_fes_range` instead of
`agm._plot_2d_fes_range` (still importing `agm` for `agm._plot_2d_fes_multirange`/
`agm.FES_PLOT_VMAX_VALUES`, both still resolvable through A6a's re-export).

## Testing

- **Per-file import-identity tests** (one new test file per new module,
  mirroring A1's `is`-identity pattern): for every name listed in "Signatures
  preserved exactly," assert
  `analyze_gareus_mbar.<name> is gareus.mbar_analysis.<module>.<name>` —
  proves the re-export actually replaced the local definition rather than
  merely shadowing it, exactly as A1 did for `KJ_PER_KCAL`.
- **Whole-module compile check**: `python -m py_compile
  gareus/mbar_analysis/plotstyle.py gareus/mbar_analysis/plotting.py
  gareus/mbar_analysis/writers.py gareus/mbar_analysis/convergence_reporting.py
  gareus/mbar_analysis/summary.py analyze_gareus_mbar.py`.
- **Full existing regression suite must be unaffected, with one named,
  deliberate exception**: the ~21 test files that import from
  `analyze_gareus_mbar` today must pass unmodified, **except**
  `tests/test_perf_plotting_redundancy.py::test_plot_2d_fes_multirange_renders_each_range_exactly_once`,
  which this plan's own task list must update in place (see Error Handling's
  monkeypatch-vs-shim finding above) — its sibling test in the same file,
  `test_plot_chignolin_fes_kj_multirange_renders_each_range_exactly_once`,
  is genuinely unaffected and must stay unmodified, since the pair it patches
  does not move until Plan A6c. Run `pytest -q tests/
  --ignore=tests/test_validation_common.py` before and after and diff the
  pass/fail set — it must be identical apart from this one named test, which
  must still be *passing* after this plan's fix, just via a rewritten
  monkeypatch target rather than by being left alone.
- **`gareus_plotstyle.py` deletion check**: `test -f gareus_plotstyle.py`
  must fail (file gone) after Task 1; `tests/test_gareus_plotstyle.py` must
  still pass, now importing from `gareus.mbar_analysis.plotstyle`.
- **Fourth `gareus_plotstyle` call site check**: after deleting
  `gareus_plotstyle.py`, `grep -n "import gareus_plotstyle" analyze_gareus_mbar.py`
  must return zero matches — this must include the L7654 call site inside
  `analyze_secondary_cv_pmf` (a function this plan does not otherwise touch;
  see Recommended Approach's Step 1), not just the three call sites inside
  functions this plan relocates. As detailed above, missing this one does
  not fail loudly: the surrounding `try/except Exception` silently downgrades
  it to a warning and drops `cv2_pmf_unbiased.png` from the output, and
  `test_secondary_cv_regime_split.py`'s own real-code-path test does not
  assert on that file's presence — so this must be verified by grep after
  the fact, not inferred from a green test run. Consider also adding one
  assertion to `test_regime_meta_preserves_dict_shape_with_regions` (or a new
  small test) that `info['files']` contains `cv2_pmf_png` after this plan
  lands, so this class of regression has a real automated guard going
  forward and not just a one-time grep.
- **Dead-code deletion check**: `grep -rn write_convergence_report .` must
  return zero matches anywhere in the repo after Task 2 (re-confirming the
  pre-deletion finding still holds, in case something landed a new caller
  since this spec was written).
- **A4 cross-import sweep verification**: after relocating `write_pmf`/
  `write_all`/`plot_outputs`/`_eff_smooth`, `grep -rn
  "write_pmf\|write_all\|plot_outputs\|_eff_smooth" gareus/mbar_analysis/*.py`
  (excluding the four files this plan itself creates) must show any A4-owned
  module importing these from `gareus.mbar_analysis.plotting`/`.writers`, not
  from `analyze_gareus_mbar` — see Task 9.

Verification commands (once implemented):

```bash
pytest -q tests/test_mbar_analysis_plotstyle.py tests/test_mbar_analysis_plotting.py \
          tests/test_mbar_analysis_writers.py tests/test_mbar_analysis_convergence_reporting.py \
          tests/test_mbar_analysis_summary.py
python -m py_compile gareus/mbar_analysis/*.py analyze_gareus_mbar.py
pytest -q tests/ --ignore=tests/test_validation_common.py
grep -rn write_convergence_report .   # expect: zero matches
test ! -f gareus_plotstyle.py         # expect: file absent
```

## Out Of Scope

- **The chignolin-kJ plotting trio** (`_chignolin_fes_range_label_kj`,
  `_plot_chignolin_fes_kj_range`, `_plot_chignolin_fes_kj_multirange`) —
  deferred to Plan A6c, which owns the rest of the chignolin bundle.
- **Consolidating the six 2D-FES CSV/NPZ writer pairs** into one generic
  writer, or **replacing `_write_rama_2d` with a parameterized call to
  `_write_generic_2d_fes`** — both are real, named duplication findings
  (see Problem) left for a dedicated follow-up, since either change would
  alter output column layout for some callers, which this plan's
  byte-identical-output bar forbids.
- **Unifying `write_convergence_plots` (legacy, CV1-specific) with
  `write_observable_convergence_plots` (generic)** — `run_epoch_pmf_convergence`
  (Plan A6b) still calls the legacy one; unifying the plotting side without
  also touching A6b's caller is out of scope here, and touching both is out
  of scope for a pure-relocation plan.
- **Plans A6b (convergence orchestration), A6c (trajectory infra +
  structural observables + chignolin extraction), A6d (residue/topology
  observables + Poincaré), A6e (orchestrator retirement)** — scoped in the
  split proposal, not planned here.
- Any change to `pmf_summary.json`/`pmf_summary.md`'s actual content,
  `PNG`/`CSV`/`NPZ` file formats, or CLI flags — this plan moves *where code
  lives*, never *what it produces*.
