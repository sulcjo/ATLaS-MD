# MBAR Analysis Modularization — Plan A4 (PMF / GaMD Cumulant Expansion / Bootstrap Uncertainty)

## Goal

Relocate `analyze_gareus_mbar.py`'s core PMF-construction math — weighted
histograms, the GaMD cumulant expansion (2nd/3rd-order reweighting
correction), boost-distribution statistics, the fixed-`f_k` block-bootstrap
uncertainty machinery, and the two orchestrating functions that tie them
together for the headline CV1 and secondary-CV2 PMFs — into
`gareus/mbar_analysis/pmf.py`, faithfully and without re-deriving any of the
physics. This is Plan A4 of the 6-plan sequence started by Plan A1 (see
`docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a1-design.md`).

24 functions move, verbatim:

```
make_bins, _bin_indices, pmf_from_weights,
_cumulant_shared_stats, _cumulant_from_shared, _cumulant_expansion, _cumulant_expansion_both,
cumulant2, cumulant3,
pmf2d_from_weights,
_cumulant_shared_stats_2d, _cumulant_from_shared_2d, _cumulant_expansion_2d, _cumulant_expansion_2d_both,
cumulant2_2d, cumulant3_2d,
_bootstrap_pmf_uncertainty_1d, _bootstrap_pmf_uncertainty_2d,
_window_cv_mean_std, boost_stats, _window_moments,
run_pmf_and_gamd_boost_report, run_secondary_cv_analyses, analyze_secondary_cv_pmf
```

This code was extensively audited and hardened earlier this session (the
masked-logw PMF-tilt fix, the GaMD NaN-boost-bin guard, the cumulant
shared-stats refactor, the just-shipped fixed-`f_k` block bootstrap). Nothing
here is re-derived, re-verified, or "improved" — this plan is a pure,
byte-faithful relocation.

## Problem

`analyze_gareus_mbar.py` is a 9,946-line, ~260-function top-level script.
Its PMF/cumulant/uncertainty math is scientifically the highest-value code
in the file — it produces every headline free-energy number the pipeline
reports — but it lives interleaved with trajectory-observable analysis,
plotting, data loading, and CLI parsing, with no module boundary separating
"the math that turns weighted samples into a PMF" from everything else. A
change to, say, the Rg-PMF plotting code (A6's domain) currently sits in the
same file, at the same indentation level, as `_cumulant_from_shared`'s NaN
guard — nothing enforces that a plotting change can't accidentally reach
into the cumulant math, and nothing makes the cumulant math importable on
its own (e.g. for a notebook cross-check, or a future non-CLI consumer)
without dragging in matplotlib, mdtraj, and DuckDB.

A secondary, concrete problem: the just-shipped PMF bootstrap-uncertainty
feature (`docs/superpowers/specs/2026-08-13-pmf-bootstrap-uncertainty-design.md`)
landed its two core helpers, `_bootstrap_pmf_uncertainty_1d`/`_2d`, directly
into this same file, growing exactly the file this whole 6-plan effort exists
to shrink. `_bootstrap_pmf_uncertainty_2d` is fully implemented, tested, and
already correct — but genuinely unwired: `grep` confirms zero call sites
outside its own definition and its own test file (`tests/test_pmf_bootstrap_uncertainty.py`).
Plan 3 of that separate 4-plan effort (wiring it into the four 2D-FES
families) and Plan 4 (plotting bands + `pmf_summary.json` scalar) were never
executed. This plan relocates the helper as-is; wiring it up is still
someone else's future work, not this plan's.

## Recommended Approach

**New file `gareus/mbar_analysis/pmf.py`, populated via mechanical,
byte-verified extraction from `analyze_gareus_mbar.py` — never hand-retyped
— with `analyze_gareus_mbar.py` re-importing every relocated name back at
module scope so its ~40 other call sites (Rg, PCA, extra-observable PMFs,
chignolin FES, every 2D-FES family — none of them in this plan's scope)
keep working completely unchanged.**

The 24 functions fall into four groups, moved in four tasks in dependency
order (leaf math first, orchestrators last — see Alternatives Considered for
why the orchestrators aren't moved first):

1. **The histogram/cumulant math block** (18 functions) — `make_bins`
   through `cumulant3_2d`, plus the two bootstrap-uncertainty helpers, which
   live inside the same block in the current file. This is one single
   **contiguous** 466-line range in the current file (confirmed by
   `grep -n "^def "` — every one of these 18 `def`s appears back-to-back
   with nothing else interleaved), so it moves as one mechanical
   `sed -n` extraction. Its only external dependency is `norm_logw`
   (used by the two bootstrap helpers only) — everything else is pure
   NumPy/optional-SciPy.
2. **Boost/window statistics** (3 functions) — `_window_cv_mean_std`
   (zero external deps), `boost_stats` and `_window_moments` (`boost_stats`
   needs `norm_logw`/`ess`).
3. **`run_pmf_and_gamd_boost_report`** (1 function) — the orchestrator
   behind both the main CV1 report and the `epoch_000_separate/` report.
   Heaviest bridge surface of the four groups (7 external names — see
   Architecture).
4. **`run_secondary_cv_analyses` + `analyze_secondary_cv_pmf`** (2
   functions) — the secondary-CV2 PMF path. Second-heaviest bridge surface
   (14 external names).

Every group that needs a name not yet relocated (by this plan or any
already-executed plan) resolves it through a small `_bridge()` helper — see
Architecture — rather than a plain top-level `import analyze_gareus_mbar`.
`analyze_gareus_mbar.py` gains one `from gareus.mbar_analysis.pmf import
(...)` block (grown by each task) restoring every relocated name as a
module-level attribute, so the file's own remaining ~40 call sites and
`analyze()`'s own direct calls need zero changes.

## Why This Approach

- **Byte-faithful extraction, not re-implementation.** This code was hardened
  through real, hard-won bug fixes this session (the NaN-vs-silent-zero
  cumulant guard, the two-pass mean-centered kappa3/variance accumulation
  that avoids catastrophic cancellation, the shift-to-main-minimum bootstrap
  anchoring). Retyping ~1,000 lines by hand risks silently reintroducing any
  of these. `sed -n 'A,Bp' analyze_gareus_mbar.py` is exactly reproducible
  and diffable against `git show HEAD:analyze_gareus_mbar.py` for a real
  byte-identity check (see Testing) — strictly safer than hand-transcription
  for a plan whose own instructions say "do not re-derive or re-verify the
  physics."
- **The contiguous-block discovery is a genuine, verified simplification.**
  All 18 histogram/cumulant/bootstrap functions already sit back-to-back in
  the source (lines 3290-3755, confirmed via `grep -n "^def "` showing no
  other function's `def` line falls inside that range) — one extraction, one
  task, instead of 18 individually-located cut-and-paste operations.
- **Re-importing into the script (not the reverse) matches the direction
  every later call site needs.** Unlike Plan A1's `cli.py` (which only
  needed a delegating *entry point*, so the import ran one way — package
  calling into script), this plan's relocated functions are called from
  ~40 other places still living inside `analyze_gareus_mbar.py` (Rg, PCA,
  chignolin-FES, extra-observable PMFs, secondary-CV 2D FES — see the
  Architecture "who else calls this" table). Those call sites are bare
  global-name lookups (`make_bins(...)`, `cumulant2(...)`, etc.) that this
  plan is explicitly not touching (out of scope: A6 owns most of them).
  The only way to relocate the *definitions* without also editing ~40
  unrelated call sites is for `analyze_gareus_mbar.py` to import the names
  back in, exactly as Plan A1's Task 2 did for two constants — this plan
  just does it for 24 names instead of 2.
- **The `_bridge()` resolver, not a bare `import analyze_gareus_mbar`,
  because of a real correctness hazard.** `python analyze_gareus_mbar.py
  <run_dir>` (the still-current, unchanged invocation) loads the script as
  `__main__`, not as a module named `analyze_gareus_mbar` — `sys.modules`
  has no entry under that name. A bare `import analyze_gareus_mbar` from
  inside `gareus/mbar_analysis/pmf.py` would then load a **second,
  independent copy** of the whole 9,946-line module. This isn't just
  wasteful: `parse_args()` (in the script, untouched by this plan) mutates
  module globals in place (`DEFAULT_MBAR_BACKEND`, `SAMBAR_EPOCHS`,
  `SAMBAR_INITIAL_BATCH_SIZE`, `SAMBAR_BATCH_PATIENCE`, `SAMBAR_SEED`,
  `SAMBAR_LR_SCALE`, `SAMBAR_DELTA_F_MAX`, `SAMBAR_POLISH_BACKEND`,
  `MBAR_ANDERSON_HISTORY`) so that later `solve_mbar()` calls without an
  explicit `--mbar-backend`/`--sambar-*` override pick up the user's CLI
  choice. Those mutations land in the `__main__` copy's globals. A
  `run_observable_pmf_convergence` call bridged through a second, freshly
  re-imported `analyze_gareus_mbar` copy (as this plan's `analyze_secondary_cv_pmf`
  does) would read that second copy's never-mutated, pre-`parse_args`
  defaults instead — a real, silent behavior divergence for exactly the
  kind of MBAR-backend/SAMBAR-tuning flag this session's earlier work
  cared about getting right. `_bridge()` (Architecture) prefers whichever
  copy is already loaded, `__main__` included, so this never happens.
- **Task ordering (leaf math, then stats, then the two orchestrators last)
  isolates risk.** If the bridge pattern needs adjustment partway through
  (e.g. a name the plan assumed lives in one place turns out to live
  somewhere else once A2/A3/A5 actually execute), Tasks 1-2 (21 of the 24
  functions, zero or near-zero bridge surface) still land as clean,
  independently valuable relocations. The two heaviest-bridge functions
  (Tasks 3-4) are exactly the ones most exposed to another plan's still-only-
  assumed final module layout.

## Alternatives Considered

### Move the two orchestrators first (they're the "interesting" part)

Rejected: `run_pmf_and_gamd_boost_report` and `run_secondary_cv_analyses`/
`analyze_secondary_cv_pmf` carry every one of this plan's 20 bridge
dependencies between them, resting on assumptions about where A2/A3/A5/A6
will eventually put things (see Architecture's bridge table) that cannot be
fully verified until those plans actually execute. Moving them last means a
wrong assumption is caught after 21 of 24 functions (all bridge-light or
bridge-free) have already landed safely, not before.

### Hard-import from A2's/A3's assumed final module paths now

Considered importing `Data` and `_sample_block_ids` directly from
`gareus.mbar_analysis.loading` (A2's assumed home) and `norm_logw`/`ess`
from wherever A3 eventually re-exports them, on the theory that stating the
real intended dependency is more honest than a runtime bridge. Rejected:
Plans A1-A6 are being authored in parallel by independent agents in this
session, and there is no guarantee A2/A3/A5/A6 execute before A4, or in any
particular order at all. A hard import against a module that doesn't exist
yet makes `gareus/mbar_analysis/pmf.py` unimportable the moment someone
tries to use it standalone. The `_bridge()` pattern degrades gracefully
regardless of execution order — see Architecture and Error Handling.

### Reconcile `overlap_matrix`/`norm_logw`/`ess`/`_eff_smooth` ownership now, rather than bridging

Considered folding `overlap_matrix` (used once, by `run_pmf_and_gamd_boost_report`)
and `norm_logw`/`ess` (used by three of this plan's functions) directly into
`gareus/mbar_analysis/pmf.py` now, since they're small and this plan already
touches every call site that needs them. Rejected: `overlap_matrix` is
explicitly assigned to Plan A5 in A1's own Out Of Scope list ("histogram-
overlap statistics vs. `gareus/math_helpers.py`/`gareus/diagnostics.py` —
deferred to Plan A5"), and `norm_logw`/`ess` sit immediately beside
`solve_mbar` in the source (A3's explicit territory: "MBAR solvers and bias
reconstruction"). Taking them here would be scope creep into two other
agents' concurrently-being-written plans, and would collide the moment A3
or A5 also tries to relocate them. `_eff_smooth`/`_smooth_pmf_1d` are a
different, harder case — genuinely unowned by any of A1-A6's explicit
scoping (used by ~20 call sites across Rg, PCA, extra-observable PMFs,
secondary-CV, chignolin-FES, and this plan's own two orchestrators) — see
Architecture and Out Of Scope for why this plan bridges them rather than
claiming them unilaterally.

### Retype the moved code by hand instead of `sed -n` extraction

Rejected outright per Recommended Approach/Why This Approach: hand-
transcription of ~1,000 lines of audited numerical code is strictly worse
than a mechanical extraction that can be byte-verified against
`git show HEAD:analyze_gareus_mbar.py`, for no offsetting benefit.

## Architecture

### Files touched

```
gareus/mbar_analysis/pmf.py          CREATE  (24 relocated functions + _bridge() helper, ~1,050 lines)
analyze_gareus_mbar.py               MODIFY  (delete 24 function bodies; add one growing import block)
tests/test_mbar_analysis_pmf_module.py  CREATE  (scaffolding, identity, bridge-resolver, byte-identity tests)
```

No existing test file is modified — every existing test that touches these
24 functions imports them via `from analyze_gareus_mbar import ...` (or
`import analyze_gareus_mbar as agm; agm.foo(...)`), which keeps resolving
correctly once `analyze_gareus_mbar.py` re-imports the relocated names as
its own module attributes.

### `gareus/mbar_analysis/pmf.py` — the `_bridge()` resolver

```python
from __future__ import annotations

import sys
from typing import Any


def _bridge() -> Any:
    """Resolve the not-yet-migrated remainder of analyze_gareus_mbar.py.

    Temporary, explicit strangler-fig bridge (see
    docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a1-design.md):
    this module's functions call back into analyze_gareus_mbar.py for names
    that belong to a different, not-yet-executed plan in this sequence (data
    loading/masking -> A2, MBAR solving -> A3, histogram-overlap diagnostics
    -> A5, writers/plotting/2D-FES-family/convergence -> A6) or that no plan
    in the A1-A6 decomposition explicitly owns (_eff_smooth, _smooth_pmf_1d
    -- see this spec's Out Of Scope).

    A plain `import analyze_gareus_mbar` is NOT safe here: running
    `python analyze_gareus_mbar.py <run_dir>` (the still-current, unchanged
    invocation) loads that file as `__main__`, not as a module named
    `analyze_gareus_mbar` -- `sys.modules` has no entry under that name in
    that case. A bare import would then load a SECOND, independent copy of
    the whole script, with its own never-`parse_args`-mutated globals
    (DEFAULT_MBAR_BACKEND, SAMBAR_*, MBAR_ANDERSON_HISTORY) -- silently
    diverging from whatever the user actually passed on the command line.
    This resolver prefers whichever copy is already loaded, `__main__`
    included, and only falls back to a fresh import (which correctly
    resolves to the SAME module object on any subsequent call, since Python
    caches it in sys.modules under its real name from then on) when neither
    is present -- e.g. a test or notebook that imports
    `gareus.mbar_analysis.pmf` directly without ever loading the script.
    """
    mod = sys.modules.get('analyze_gareus_mbar')
    if mod is not None:
        return mod
    main_mod = sys.modules.get('__main__')
    if str(getattr(main_mod, '__file__', '')).endswith('analyze_gareus_mbar.py'):
        return main_mod
    import analyze_gareus_mbar as mod  # noqa: PLC0415 (intentional: see docstring)
    return mod
```

Every function in this module that needs a not-yet-relocated name calls
`_bridge()` as its first line and references members off the result
(`_agm.norm_logw(...)`, `_agm.write_pmf(...)`) — never a per-name top-level
`from analyze_gareus_mbar import X`, which would force eager resolution at
`gareus/mbar_analysis/pmf.py`'s own import time, before `analyze_gareus_mbar.py`
has necessarily loaded at all.

`from __future__ import annotations` at the top of this module also solves
a real, otherwise-blocking problem for `analyze_secondary_cv_pmf`'s existing
signature, `d: Data` (unquoted, no import of `Data` in this module):
`analyze_gareus_mbar.py` already relies on the exact same PEP 563 postponed-
evaluation behavior (it has the identical future-import at its own line 2)
to make bare `Data`/`Progress` annotations resolve lazily; carrying the same
future-import into `pmf.py` makes the unquoted annotation equally safe here,
with no per-signature quoting/rewriting needed and no runtime import of
`Data` at all (it is never evaluated, and nothing in this module does
`isinstance(d, Data)` or otherwise needs the real class at runtime).

### Function groups and their bridge dependencies

| Group | Functions | External (bridged) names needed | Assumed future home (unverified until that plan executes) |
|---|---|---|---|
| 1. Histogram/cumulant math (contiguous, lines 3290-3755) | `make_bins`, `_bin_indices`, `pmf_from_weights`, `_cumulant_shared_stats`, `_cumulant_from_shared`, `_cumulant_expansion`, `_cumulant_expansion_both`, `cumulant2`, `cumulant3`, `pmf2d_from_weights`, `_cumulant_shared_stats_2d`, `_cumulant_from_shared_2d`, `_cumulant_expansion_2d`, `_cumulant_expansion_2d_both`, `_bootstrap_pmf_uncertainty_1d`, `_bootstrap_pmf_uncertainty_2d`, `cumulant2_2d`, `cumulant3_2d` | `norm_logw` (bootstrap helpers only) | A3 (`solve_mbar` territory) |
| 2. Boost/window statistics | `_window_cv_mean_std`, `boost_stats`, `_window_moments` | `norm_logw`, `ess` (`boost_stats` only) | A3 |
| 3. Main PMF orchestrator | `run_pmf_and_gamd_boost_report` | `norm_logw`, `_eff_smooth`, `overlap_matrix`, `_sample_block_ids`, `write_pmf`, `write_all`, `plot_outputs` | A3 (`norm_logw`, `overlap_matrix`), A2 (`_sample_block_ids`), A6 (`write_pmf`/`write_all`/`plot_outputs`), A6a (`_eff_smooth`) |
| 4. Secondary-CV2 orchestrator | `run_secondary_cv_analyses`, `analyze_secondary_cv_pmf` | `norm_logw`, `_eff_smooth`, `_smooth_pmf_1d`, `_secondary_cv_epoch_regime_masks`, `_masked_data`, `_subset_logw_from_global_fk`, `_regime_slug`, `_secondary_cv_label`, `_secondary_cv_regions`, `_visible_pmfs`, `analyze_cv1_cv2_2d_fes`, `run_observable_pmf_convergence`, `write_cv2_pmf`, `wjson` | A3 (`norm_logw`), A2 (`_masked_data`, `_subset_logw_from_global_fk`, `_secondary_cv_epoch_regime_masks`, `wjson`), A6 (`analyze_cv1_cv2_2d_fes`, `run_observable_pmf_convergence`, `write_cv2_pmf`), A6a `plotting.py` (`_eff_smooth`, `_smooth_pmf_1d`, `_regime_slug`, `_secondary_cv_label`, `_secondary_cv_regions`, `_visible_pmfs`) |

**CORRECTION** (post-cross-check): `_regime_slug`, `_secondary_cv_label`,
`_secondary_cv_regions`, `_visible_pmfs`, `_eff_smooth`, `_smooth_pmf_1d`, and
`overlap_matrix` were unresolved/misattributed when this table was first
written (A4 was drafted in parallel with A2/A3/A5/A6a, before their final
ownership settled). A whole-set cross-check has since resolved all of them:
`overlap_matrix` belongs to A3's `solvers.py` (not A5, which explicitly
disclaims it); the six label/smoothing helpers all landed in A6a's
`plotting.py`. This changes no code in this plan — every one of these is
already accessed through `_bridge()`/`_agm.<name>` dynamic lookup precisely
so the plan doesn't need to know which module truly owns a name, only that
it's reachable on the loaded `analyze_gareus_mbar` object once its owning
plan's re-export runs. The table above is corrected for documentation
accuracy only.
(`_want_gamd_method` is a related helper but is used only by
`analyze_cv1_cv2_2d_fes`, itself bridged as A6's — not a direct dependency of
either function this plan relocates, so it is not in this table.)

### `analyze_gareus_mbar.py` (relevant change, grown across Tasks 1-4)

Before (representative excerpt; the real file has these 24 `def`s scattered
across two regions):

```python
def make_bins(cv,bins,lo,hi):
    ...
def cumulant2(cv,base_w,boost,bins,beta,kbt_kcal,smooth_logfac_sigma=0.0):
    ...
def run_pmf_and_gamd_boost_report(d: 'Data', args, ...):
    ...
```

After (one import block, grown by each task; final state after Task 4):

```python
from gareus.mbar_analysis.pmf import (
    make_bins, _bin_indices, pmf_from_weights,
    _cumulant_shared_stats, _cumulant_from_shared, _cumulant_expansion, _cumulant_expansion_both,
    cumulant2, cumulant3,
    pmf2d_from_weights,
    _cumulant_shared_stats_2d, _cumulant_from_shared_2d, _cumulant_expansion_2d, _cumulant_expansion_2d_both,
    cumulant2_2d, cumulant3_2d,
    _bootstrap_pmf_uncertainty_1d, _bootstrap_pmf_uncertainty_2d,
    _window_cv_mean_std, boost_stats, _window_moments,
    run_pmf_and_gamd_boost_report,
    run_secondary_cv_analyses, analyze_secondary_cv_pmf,
)
```

Placed immediately after the existing `from gareus.units import KJ_PER_KCAL,
K_B_KJ_PER_MOL_K` line (Plan A1's own cross-import, so this plan's import
sits next to the one precedent for "package name imported into the script").
Every one of the ~40 other call sites in `analyze_gareus_mbar.py` (Rg, PCA,
extra-observable PMFs, chignolin FES, secondary-CV 2D FES, the main
`analyze()` function itself) references these as bare global names and is
completely unchanged — Python resolves them from the module's own namespace,
which now contains them via this import instead of a local `def`.

### `write_pmf`/`plot_outputs`/`overlap_matrix` — the requested temporary-bridge decision

`run_pmf_and_gamd_boost_report` (Task 3) calls `write_pmf`, `write_all`, and
`plot_outputs` (CSV writers and the plotting entry point, both squarely A6's
future "reporting" territory per A1's Out Of Scope list) and `overlap_matrix`
(A5's future territory). **Decision: these calls stay inline in
`run_pmf_and_gamd_boost_report`, which itself now lives in
`gareus/mbar_analysis/pmf.py`, resolved through `_bridge()`** — i.e.
`_bridge().write_pmf(...)`, `_bridge().plot_outputs(...)`,
`_bridge().overlap_matrix(...)` — exactly the explicit, temporary,
clearly-labeled strangler-fig bridge this plan's brief called for, since A6
has not been planned yet (it is being authored in parallel by another agent
right now) and this plan cannot import from a module/name that may not exist
under that final name. **A6 is expected to clean this up** when it relocates
`write_pmf`/`write_all`/`plot_outputs` for real: at that point, the bridge
call sites in `pmf.py` become direct top-level imports from A6's actual
module path instead of `_bridge()` lookups, and `_bridge()` itself may
shrink or disappear entirely once every one of its current uses has a real
home. Until then, correctness only requires that `analyze_gareus_mbar.py`
(or whatever eventually re-exports `write_pmf`/`plot_outputs`/`overlap_matrix`
under those names) is loaded by the time `run_pmf_and_gamd_boost_report` is
actually *called* — guaranteed today because the only caller, `analyze()`,
lives in the very same script.

### Who else calls these 24 functions (for the cross-check against A3/A5/A6)

Confirmed via `grep -c "\bfn(" analyze_gareus_mbar.py` for each relocated
name: `make_bins`, `pmf_from_weights`, `pmf2d_from_weights`,
`_cumulant_expansion_both`, `_cumulant_expansion_2d_both`, and `boost_stats`
are each called from several more places in the file beyond this plan's own
two orchestrators — Rg analysis (~5699-5767), PCA 2D FES (~6098-6109),
extra-observable PMFs (~4620-4628), chignolin FES (~8675-8685), and
`plot_gamd_boost`'s own `_per_window_gamd_boost_stats` helper (`_window_moments`,
~7170). None of these call sites are touched by this plan; they are exactly
the ~40 "everything else" call sites the re-export block above exists to
keep working. **A6, when it relocates those functions, will need to import
`make_bins`/`pmf_from_weights`/`pmf2d_from_weights`/`_cumulant_expansion_both`/
`_cumulant_expansion_2d_both`/`_window_moments` from `gareus.mbar_analysis.pmf`**
(the real final home this plan creates) rather than continuing to reach into
`analyze_gareus_mbar.py`'s re-export.

## Error Handling

- **`_bridge()`'s two-copy hazard is the one new failure mode this plan
  introduces**, and it is guarded by construction (preferring an
  already-loaded copy, `__main__` included) rather than by a runtime check
  that could fail — see Architecture. Task-level testing includes a direct,
  monkeypatched-`sys.modules` unit test of `_bridge()` itself (see Testing)
  precisely because none of the existing integration tests exercise the
  `__main__` code path (they all import `analyze_gareus_mbar` under its real
  name, which is the easy case `_bridge()` already handled correctly on the
  first `if`).
- **A bridged name that genuinely doesn't exist yet** (e.g. if a future
  reader runs `gareus/mbar_analysis/pmf.py`'s functions against a version of
  `analyze_gareus_mbar.py` from before some dependency was added) surfaces as
  a plain `AttributeError` from `_bridge()`'s returned module object — no new
  silent-failure mode; this is the same failure any direct call to a missing
  function would already produce.
- **No change to any of the 24 functions' own error handling.** The NaN-vs-
  silent-zero cumulant guards, the low-block-count bootstrap warning, the
  `ValueError` on an unrecognized cumulant `order`/`selected_method` — all
  relocate verbatim (see Testing's byte-identity check) and are not
  re-examined by this plan, per the brief's explicit instruction not to
  re-derive or re-verify the physics.

## Testing

Since this is a pure, audited-and-already-correct relocation, testing
focuses on **behavioral equivalence** and **faithful transcription**, not
new physics coverage:

- **Byte-identity of the extracted blocks.** For each `sed -n 'A,Bp'`
  extraction, diff the extracted text against
  `git show HEAD:analyze_gareus_mbar.py | sed -n 'A,Bp'` — must be
  byte-identical. This is the direct check that the two hardest-won fixes
  in this code (the `logfac[(~nz)&(p0>0)]=np.nan` NaN-vs-silent-zero guard,
  both 1D and 2D; the two-pass mean-centered kappa3/variance accumulation
  that avoids catastrophic cancellation) survive the move unperturbed.
- **Identity, not just equality, of every re-exported name.**
  `analyze_gareus_mbar.cumulant2 is gareus.mbar_analysis.pmf.cumulant2`
  (and so on for all 24) — proves the import actually replaced the local
  `def`, matching Plan A1's own `is`-based verification convention for its
  two relocated constants.
- **A direct, monkeypatched unit test of `_bridge()`** covering all three
  branches: `analyze_gareus_mbar` already in `sys.modules` under its real
  name (the common case); absent from `sys.modules` but `__main__.__file__`
  ends with `analyze_gareus_mbar.py` (the `python analyze_gareus_mbar.py
  <run_dir>` invocation); neither present (falls back to a fresh import).
- **The existing, substantial test suite for this code must pass unchanged**
  — confirmed present via direct `grep`. **CORRECTION (found during Plan
  A6a's final whole-plan review): this claim is false for 3 of the tests
  below** (`test_run_pmf_and_gamd_boost_report_uses_combined_path_exactly_once`,
  `test_regime_meta_preserves_dict_shape_with_regions`,
  `test_run_secondary_cv_analyses_uses_corrected_reweight_when_f_k_global_given`)
  — each monkeypatches/reassigns a name whose caller this plan relocates
  into the same module as the callee (Task 3 for the first, Task 4 for the
  other two), the identical monkeypatch-vs-shim hazard Plan A6a's Task 2/
  Task 6 already hit once. These 3 tests must be REPOINTED (not left
  unchanged) as part of Tasks 3/4 themselves, mirroring A6a's Task 6 fix
  exactly. Full detail, exact line numbers, and the required fix shape are
  in the implementation plan's own Global Constraints CORRECTION block —
  read that before executing Task 3 or Task 4.
  - `tests/test_cumulant_expansion.py` (86 lines) — `cumulant2`/`cumulant3`/
    `cumulant2_2d`/`cumulant3_2d` hand-computed-value and symmetric-boost
    checks, `_cumulant_expansion`'s bad-order `ValueError`.
  - `tests/test_cumulant_shared_computation.py` (548 lines) — the shared-
    stats refactor's own regression suite: combined-vs-separate-calls
    bit-identity (1D and 2D), `pmf_from_weights`/bincount-vs-histogram
    equivalence, alias-safety between order-2/order-3 results, and an
    end-to-end `run_pmf_and_gamd_boost_report` call site test
    (`test_run_pmf_and_gamd_boost_report_call_site_end_to_end`,
    `test_run_pmf_and_gamd_boost_report_uses_combined_path_exactly_once`).
  - `tests/test_pmf_gamd_nan_bin_handling.py` (204 lines) — the NaN-vs-
    silent-zero guard, 1D and 2D, order 2 and order 3, plus a convergence-
    metric non-poisoning check.
  - `tests/test_pmf_bootstrap_uncertainty.py` (495 lines) — the full
    bootstrap-uncertainty suite: block-vs-naive validity, shift-anchor
    correctness, low-block-count flagging, seeded determinism, 2D/1D
    cross-consistency, CLI flag parsing, and `run_pmf_and_gamd_boost_report`
    wiring (on/off/warns-on-low-blocks).
  - `tests/test_epoch0_pmf_gamd_split.py` (249 lines),
    `tests/test_masked_logw_subset_pmf.py` (386 lines),
    `tests/test_perf_report_redundancy.py` (530 lines) — all exercise
    `run_pmf_and_gamd_boost_report` end-to-end (epoch_000/rest split
    independence, corrected-subset-logw reweighting,
    `_window_cv_mean_std`'s vectorized-vs-loop equivalence, the
    `precomputed_base_w` fast path).
  - `tests/test_secondary_cv_regime_split.py` (430 lines) — regime-detection,
    dominant-regime selection, and single-vs-multi-regime `run_secondary_cv_analyses`
    output-directory splitting, including a not-blended cross-regime check.
- **Full-suite regression run** (`pytest -q tests/ --ignore=tests/test_validation_common.py`)
  must show the same pre-existing failures as `main` today and no new ones —
  same convention as Plan A1's own Task 2, Step 5.

Verification commands (once implemented):

```bash
pytest -q tests/test_mbar_analysis_pmf_module.py
pytest -q tests/test_cumulant_expansion.py tests/test_cumulant_shared_computation.py \
          tests/test_pmf_gamd_nan_bin_handling.py tests/test_pmf_bootstrap_uncertainty.py \
          tests/test_epoch0_pmf_gamd_split.py tests/test_masked_logw_subset_pmf.py \
          tests/test_perf_report_redundancy.py tests/test_secondary_cv_regime_split.py
python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/pmf.py
python -c "import gareus.mbar_analysis.pmf"  # confirms standalone importability with zero script dependency at import time
```

## Out Of Scope

- **Plan A2**: data loading/masking (`Data`, `load_csv`/`load_npz`/
  `load_parquet*`, `clean()`, `_masked_data`, `_subset_logw_from_global_fk`,
  `_secondary_cv_epoch_regime_masks`, `_regime_slug`, `wjson`,
  `_sample_block_ids`). This plan bridges to all of these via `_bridge()`
  and assumes (unverified until A2 executes) they land in
  `gareus/mbar_analysis/loading.py`.
- **Plan A3**: MBAR solvers and bias reconstruction (`solve_mbar`,
  `norm_logw`, `ess`, `logsumexp`), reconciling with `gareus/query.py`'s
  `reconstruct_bias_matrix`. This plan bridges to `norm_logw`/`ess`.
- **Plan A5**: statistical diagnostics (`overlap_matrix`, reconciling with
  `gareus/math_helpers.py`'s `_hist_overlap`/`_adaptive_hist_overlap`
  family). This plan bridges to `overlap_matrix`.
- **Plan A6**: trajectory observables, convergence analysis, writers, and
  plotting (`write_pmf`, `write_all`, `write_cv2_pmf`, `plot_outputs`,
  `analyze_cv1_cv2_2d_fes`, `run_observable_pmf_convergence`, and every
  Rg/PCA/chignolin-FES/extra-observable call site that also calls this
  plan's relocated `make_bins`/`pmf_from_weights`/`pmf2d_from_weights`/
  `_cumulant_expansion_both`/`_cumulant_expansion_2d_both`/`_window_moments`).
  This plan bridges to the writers/plotting/2D-FES/convergence names and
  flags, for A6's own cross-check, that it should eventually import the six
  names above from `gareus.mbar_analysis.pmf` directly.
- **`_eff_smooth`/`_smooth_pmf_1d`/`_secondary_cv_label`/`_secondary_cv_regions`/
  `_visible_pmfs`/`_want_gamd_method`: genuinely unowned by the A1-A6
  decomposition.** `_eff_smooth`/`_smooth_pmf_1d` in particular are called
  from roughly 20 sites spanning nearly every domain (Rg, PCA,
  extra-observable PMFs, secondary-CV, chignolin-FES, and this plan's own
  two orchestrators) — no single plan's scope cleanly contains them. Left
  bridged rather than claimed unilaterally; **flagging for the coordinating
  session that a `gareus/mbar_analysis/common.py` (or similar shared-utility
  module) may be needed as a follow-up**, since A5 and A6 are likely to hit
  the identical "which plan owns this generic helper" question independently
  and three agents each inventing a different home for the same handful of
  functions would itself become a new duplication to reconcile.
- **Wiring `_bootstrap_pmf_uncertainty_2d` into any call site.** It moves
  as-is (fully implemented, tested, unwired) — wiring it into the four
  2D-FES families is Plan 3 of the separate PMF-bootstrap-uncertainty
  4-plan effort, never executed, not part of this plan.
- **Any change to the CLI flags that control this code**
  (`--pmf-uncertainty`, `--pmf-uncertainty-n-boot`, `--pmf-uncertainty-seed`,
  `--selected-method`, `--smooth-sigma`/`--gamd-smooth-sigma`/
  `--pmf-smooth-sigma`, `--min-neighbor-overlap`) — these stay in
  `parse_args()`, which stays in `analyze_gareus_mbar.py`, untouched.
- **Any numeric, behavioral, or output-format change** to any of the 24
  relocated functions. This plan's only content change to any of them is
  replacing a bare global-name reference to a not-yet-relocated dependency
  with a `_bridge().name` call — never a formula, guard, default, or return
  value.
