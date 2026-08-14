# Plan A6 Split — Verification Findings (supplement, not a standalone spec)

A concurrent process independently authored the live
`2026-08-13-mbar-analysis-modularization-a6-split-proposal.md` and
`2026-08-13-mbar-analysis-modularization-a6a-design.md` in this same
worktree while this investigation was in progress, overwriting this
session's own drafts of both. Both analyses converged independently on the
same core call — split A6, land plotting/reporting first because it is the
verified leaf of the sub-domain dependency graph — for the same reason
(traced call graph, not stylistic preference). Rather than re-editing files
another active process owns and risking a second clobber, this document
records only what this session verified independently and did not find
already reflected in the live documents at the time of writing. Treat it as
a diff to check against the final merged version, not a competing proposal.

## Where the other analysis was right and this session's draft was wrong

- **`write_convergence_report` (L4576-4604) is confirmed dead code** — `grep
  -n "write_convergence_report" analyze_gareus_mbar.py` returns only its own
  `def` line, zero call sites anywhere in the file. This session's own A6a
  draft had it as a relocation target; it should be deleted, not moved.
- **Deleting `gareus_plotstyle.py` outright beats leaving a re-export shim.**
  Both of its consumers (`analyze_gareus_mbar.py`, `tests/test_gareus_plotstyle.py`)
  are known in full and can be updated in the same change — a shim adds a
  permanent indirection for zero remaining benefit, and this codebase's own
  `coding-style.md` argues against backwards-compatibility shims when the
  code can just be changed directly. This session's draft proposed a shim
  out of excess caution; the direct-delete approach is the better call.
- **The chignolin bundle should move entirely out of `gareus/mbar_analysis/`**
  into a new top-level extension file, not into an isolated file inside the
  package. `plot_adaptive_diagnostics.py` and `gareus_report.py` already
  establish the working convention in this codebase: project-specific
  extensions live at the repo root, general-purpose code lives inside the
  package. This session's draft moved chignolin into the package (isolated
  but still inside it); the top-level-extension call is more consistent with
  existing precedent and should stand.

## Where this session found a factual error in the live documents

**`_write_csv_rows` (L4440-4456) is misassigned to A6b (convergence) in the
live split proposal — it belongs in A6a (reporting/leaf), and the "only
callers" claim behind that assignment does not survive a direct grep.**

Verified call sites, `grep -n "_write_csv_rows(" analyze_gareus_mbar.py`:

```
2182, 2194, 2198   run_epoch_pmf_convergence          (A6b, convergence)
5065, 5070, 5074,
5098, 5099, 5137,
5139               run_observable_pmf_convergence     (A6b, convergence)
5730               analyze_rg                         (observables)
7011, 7012, 7024   analyze_extra_observable_pmfs      (observables)
7889, 7906         analyze_poincare_map               (observables)
8438               analyze_poincare_residue_torsions  (observables)
```

15 call sites across 6 functions, not 3 call sites across 2 functions. It is
needed by convergence *and* every observables sub-plan (however the
observables domain ultimately gets split). Leaf-first ordering requires it
live in whichever sub-plan lands before **all** of its consumers — that is
A6a, not A6b, which is itself one of the consumers. Whoever finalizes the
merged split proposal/A6a spec should move this function's ownership back to
A6a and correct the reasoning attached to it.

## Findings not present in the live documents at time of writing

### 1. The monkeypatch-vs-shim hazard (layout-independent — applies regardless of which module-file layout wins)

A re-export/shim keeps `from analyze_gareus_mbar import X` and `agm.X(...)`
working after `X` relocates. It does **not** make
`monkeypatch.setattr(agm, "X", spy)` transparent when some other *moved*
function calls `X` internally — that internal call resolves `X` in the new
module's own `__globals__`, not `analyze_gareus_mbar`'s, so the patch
silently has no effect.

**Confirmed live instance, directly in A6a's scope, verified by reading the
test file in full:** `tests/test_perf_plotting_redundancy.py` has two tests
(`test_plot_2d_fes_multirange_renders_each_range_exactly_once`,
`test_plot_chignolin_fes_kj_multirange_renders_each_range_exactly_once`,
L286-331) that do `import analyze_gareus_mbar as agm` locally (L292, L314),
then `monkeypatch.setattr(agm, "_plot_2d_fes_range", counting)` (L302) /
`monkeypatch.setattr(agm, "_plot_chignolin_fes_kj_range", counting)` (L325),
then call `agm._plot_2d_fes_multirange(...)` (L306) /
`agm._plot_chignolin_fes_kj_multirange(...)` (L329) and assert the counting
wrapper was invoked once per `FES_PLOT_VMAX_VALUES` /
`CHIGNOLIN_FES_PLOT_VMAX_VALUES_KJ` entry (L308, L331). Once the patched
function and its caller move together into the same new plotting module —
true under both this session's four-file layout and the live document's
five-file layout — patching `agm._plot_2d_fes_range` no longer affects what
the moved `_plot_2d_fes_multirange` actually calls. **This will break on the
first CI run after whichever A6a lands, unless a task explicitly repoints
both tests at the new module.** This finding appears in neither live
document as of this writing; it must land as an explicit task in whichever
A6a implementation plan is finalized, not left to be discovered by a failing
test.

**Other confirmed instances, outside A6a's scope, for the relevant plans'
own cross-checks** (same `grep -rn "monkeypatch.setattr(agm\|monkeypatch.setattr(analyze_gareus_mbar" tests/*.py`):
- `tests/test_perf_loading_and_solver_memory.py:493` patches
  `agm.infer_temp_beta` — A2 (loading) territory.
- `tests/test_cumulant_shared_computation.py:541-543` patches
  `agm._cumulant_expansion_both`, `agm.cumulant2`, `agm.cumulant3` — A4 (PMF/
  GaMD cumulant math) territory.
- `tests/test_secondary_cv_regime_split.py:389` patches
  `agm.analyze_secondary_cv_pmf` — see the unassigned-domain gap below.

### 2. `_sample_aligned_trajectory_frames` (L5399-5464) is genuinely dead code, verified — not "appears orphaned"

`grep -rn "_sample_aligned_trajectory_frames" --include="*.py" .` returns
only its own `def` line — zero call sites anywhere in the repository,
including its own file. This is narrow, not a sign of broader rot: its
neighbors `_base_segment_resume_start` and `_sample_to_segment_frame` are
demonstrably live (`tests/test_traj_alignment_offset.py:3` imports both
directly). Whichever plan ends up owning trajectory-observables helpers
should either delete this one 66-line function outright or explicitly
justify keeping it — this session did not find a reason to keep it.

### 3. `run_secondary_cv_analyses` / `analyze_secondary_cv_pmf` — confirmed unassigned across every A2-A6 plan, with its full dependency edge list

> **[SUPERSEDED — corrected by the later cross-plan reconciliation pass.]**
> The "confirmed unassigned" headline below is no longer true. **Plan A4
> claims both functions outright**, with real relocation tasks, not a
> passing mention: `...-a4-design.md`'s Goal lists them as #23 and #24 of
> its "24 functions move, verbatim" into `gareus/mbar_analysis/pmf.py`, and
> `...-a4.md`'s Task 4 (L742-961) relocates both with exact extraction
> ranges, a full bridge-substitution list, and its own end-to-end test. The
> dependency edge list below is still accurate and still useful — in
> particular the `gareus_plotstyle` import-site edge, which A6a's own spec
> independently found and assigned to itself (A6a Task 1) — but read it as
> "here is what A4's relocation must carry with it," not as an unowned-
> function report. Nothing needs a new owner here.

`analyze_secondary_cv_pmf` (L7619-7685) and its caller
`run_secondary_cv_analyses` (L514-588, alongside
`_secondary_cv_epoch_regime_masks`/`_masked_data`/`_subset_logw_from_global_fk`)
are not named in any A2-A6 function list this session has seen, including
the live A6 documents (which mention it only in passing, as an A4 interface
note). Its dependency edges, verified directly:

- Calls `write_cv2_pmf` five times (A6a's domain, whichever layout wins).
- Calls `run_observable_pmf_convergence` (A6b's domain).
- Does its own inline matplotlib plotting via a function-local
  `import gareus_plotstyle as ps` at ~L7654 — the fourth of `gareus_plotstyle`'s
  four call sites and the only one not owned by any A6a variant. If A6a
  deletes `gareus_plotstyle.py` outright (per the live document's better
  call, conceded above), **this import site must be updated too, by
  whoever ends up owning `analyze_secondary_cv_pmf`** — it is not
  automatically covered by A6a's own two-consumer cleanup, since this is a
  third, unowned consumer.
- Is directly monkeypatch-tested at `tests/test_secondary_cv_regime_split.py:389`
  (`monkeypatch.setattr(analyze_gareus_mbar, "analyze_secondary_cv_pmf", _spy)`)
  — this one is a same-module patch of the function itself (not an internal
  call inside it), so it is not subject to the hazard in finding 1 above,
  but it does mean whichever plan relocates `analyze_secondary_cv_pmf` must
  keep this test passing by preserving the re-export path the test's
  `monkeypatch.setattr(analyze_gareus_mbar, ...)` call depends on.

This function needs an explicit owner before any A6 sub-plan touching
`write_cv2_pmf` or `run_observable_pmf_convergence` (or, once
`gareus_plotstyle.py` is deleted, anything touching that cleanup) can be
considered complete. Flagging for the coordinating session to assign, not
deciding it here.

## Interfaces this session expects from A2-A5 (for the cross-check, scoped to what this session verified directly)

- **A2 (loading)**: `_primary_cv_axis_label(meta: dict) -> str` (L603,
  consumed by `plot_outputs`/`plot_rg_outputs`). **[CORRECTED by the
  cross-plan reconciliation pass: A2 does not claim this.** Neither
  `...-a2-design.md` nor `...-a2.md` names `_primary_cv_axis_label`
  anywhere — grep-confirmed — and A6a's own design independently records it
  as "Unclaimed by any A2-A6 domain description found so far"
  (`...-a6a-design.md`, L368). It is genuinely unowned, together with
  `_primary_cv_label`, `_primary_cv_units`, `_secondary_cv_label`,
  `_secondary_cv_regions`, and `_regime_slug`; see the reconciliation
  report's unowned-helpers finding. Do not expect A2 to deliver it.**
- **A4 (PMF/GaMD cumulant math)**: `_per_window_gamd_boost_stats(window,
  comb_kcal, kbt_kcal, K, dih_kcal=None) -> dict` and `_window_moments(a) ->
  tuple` (L7104-7182, consumed by `plot_gamd_boost`); also
  `run_pmf_and_gamd_boost_report`, which calls `write_pmf`/`write_all`/
  `plot_outputs` directly and will need its import of those three names
  re-pointed once A6a lands, wherever A4's own module ends up.
- **A5 (diagnostics)**: `overlap_matrix`, `_window_cv_mean_std`,
  `pmf_probability`, `js_divergence_1d`, `pmf_rmse_1d`, `barrier_error_1d`,
  `identify_basins_1d`, `_compute_basin_populations` — called by convergence
  orchestration (checkpoint-vs-reference PMF comparison, basin tracking),
  physically positioned in the file's PMF-math block but used almost
  exclusively by convergence code, not by A4's own cumulant-expansion
  machinery — confirm with A4 whether these land in A4's or the convergence
  plan's module before finalizing either.
