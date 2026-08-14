# MBAR Analysis Modularization — Plan A3 (MBAR Solvers + Bias Reconstruction)

## Goal

Relocate `analyze_gareus_mbar.py`'s MBAR numerical core — the self-consistent
MBAR solver family and the analytic harmonic-umbrella bias-reconstruction
math — into the `gareus` package, as Plan A3 of the 6-plan
`analyze_gareus_mbar.py` modularization sequence (see Plan A1's design,
`docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a1-design.md`).
Unlike A1 (pure scaffolding), this plan also **fixes a real, currently-live
correctness bug**: `gareus/query.py`'s `reconstruct_bias_matrix` — which is
called directly both by `analyze_gareus_mbar.py`'s own primary Parquet loader
and by the live adaptive-production validation pipeline — silently fabricates
a wrong bias energy for any sample whose secondary CV value is missing under
a window that has a real secondary restraint, instead of correctly excluding
that sample. This plan is scientifically the highest-stakes of the six.

## Problem

### Three implementations of the same physics formula, one of them wrong

The harmonic-umbrella reduced-bias formula

```
U_k(cv) = 0.5*k1_k*(cv1-center1_k)^2 [+ 0.5*k2_k*(cv2-center2_k)^2, 2D windows]
reduced_bias[n,k] = beta * KJ_PER_KCAL * U_k(cv[n])
```

is implemented three separate times in this codebase:

1. **`gareus/query.py`'s `reconstruct_bias_matrix(cv_A, cv2, windows, beta)`**
   — the live package's own copy, documented in its module docstring as *the*
   analytic-reconstruction implementation.
2. **`analyze_gareus_mbar.py`'s `_compute_u_nk_analytical(cv1, cv2,
   union_windows, beta)`** — used by the legacy
   `adaptive_feedback_round_*`-pooling path (`_augment_with_adaptive_rounds`).
3. **`analyze_gareus_mbar.py`'s `_reconstruct_union_bias_block(cv, cv2, beta,
   primary_centers, primary_ks, sec_centers, sec_ks)`** — used by
   `load_parquet_adaptive_union`'s per-epoch bias reconstruction (the fix
   documented in CLAUDE.md's "Union-MBAR bias matrix used a stale global
   registry snapshot" entry).

**Implementations 2 and 3 are already correct** — a prior audit this session
fixed both and pinned the fix with `tests/test_bias_reconstruction_nan_handling.py`
(24 lines of docstring explaining exactly this bug class, 7 passing tests).
Both correctly guard the secondary term two ways:

- **Window-side guard** (does this window restrain CV2 at all?):
  `_compute_u_nk_analytical` spells this out as three clauses,
  `math.isfinite(sec_center) and math.isfinite(sec_k) and sec_k > 0`;
  `_reconstruct_union_bias_block` writes only
  `math.isfinite(sec_centers[k]) and sec_ks[k] > 0` (two clauses) — these
  are **equivalent, not a discrepancy**: `NaN > 0` is always `False` in both
  Python and NumPy, so a NaN `sec_k` already fails the `> 0` half without
  needing a separate `isfinite` check on it. The unified implementation
  (below) uses the explicit three-clause form for readability, matching
  `_compute_u_nk_analytical`'s spelling. If not satisfied, the CV2 term is
  skipped entirely — a window with no real secondary restraint gets a
  purely CV1 bias, regardless of what `cv2` holds for its samples.
- **Sample-side exclusion** (did we actually measure this sample's CV2?): for
  a window that *does* restrain CV2, a sample with non-finite `cv2` gets a
  **NaN** bias entry for that `(sample, window)` pair — not a fabricated
  zero deviation. `clean()` (every `Data` loader's shared post-processing
  step, `analyze_gareus_mbar.py`) then drops any sample row with a NaN
  anywhere in its `u_nk` row via `mask = np.isfinite(d.cv) &
  np.all(np.isfinite(d.u_nk), axis=1)` — i.e. "NaN-propagate, then
  mask-and-drop at the boundary" is this codebase's established, correct
  convention for "we don't have a real number for this."

**Implementation 1 (`gareus/query.py`'s `reconstruct_bias_matrix`) has both
of the same two bugs this session's prior audit already fixed in
implementations 2 and 3, independently reintroduced** — verified directly
against the current code (not just cited from an old audit summary):

```python
for k, w in enumerate(windows):
    d1 = cv_A - float(w["center1"])
    nk[:, k] = 4.184 * 0.5 * float(w["k1"]) * d1 * d1  # kcal → kJ
    if cv2 is not None and "center2" in w and "k2" in w:
        c2 = np.asarray(cv2, dtype=np.float64)
        valid = np.isfinite(c2)
        d2 = np.where(valid, c2 - float(w["center2"]), 0.0)   # <-- BUG
        nk[:, k] += 4.184 * 0.5 * float(w["k2"]) * d2 * d2
```

- **Missing window-side guard**: the `if` only checks that the dict *keys*
  `"center2"`/`"k2"` are present, never that their *values* are finite and
  `k2 > 0`. A window with a real key but a NaN `center2` (which happens: see
  `test_finite_positive_k_with_nan_center_still_produces_finite_bias` in the
  existing audit's test file, documenting exactly this case for
  implementation 2) is completely unguarded here and NaN-poisons that
  window's whole column regardless of the sample's own `cv2`.
- **Fabricated zero deviation (the primary bug)**: for a window that *does*
  have a real secondary restraint, `np.where(valid, c2-center2, 0.0)`
  substitutes **exactly 0.0** — "assume this sample was perfectly on
  target" — for any sample whose `cv2` was never measured, instead of
  letting NaN propagate so the sample gets excluded downstream. This
  silently *keeps* a should-be-excluded sample in the pool with a
  wrong (too-low) bias energy, rather than dropping it — the opposite
  failure mode from a crash: it looks like more data, not less.

No existing test in `tests/test_query.py` (5 direct `reconstruct_bias_matrix`
tests: shape, 1D formula, 2D formula, zero-at-center, non-negative) exercises
either guard — all use fully-finite, fully-restrained-or-fully-1D inputs — so
nothing currently pins the buggy behavior as intentional, and nothing needs
updating to fix it.

### This is not confined to the offline analysis script

`reconstruct_bias_matrix` is called directly from **two real places**, one of
which is the live production pipeline, not just this analysis script:

```
analyze_gareus_mbar.py:2276  from gareus.query import ... reconstruct_bias_matrix
analyze_gareus_mbar.py:2313  u_nk = reconstruct_bias_matrix(cv, cv2_for_nk, windows, beta)
    (inside load_parquet() -- the primary, non-union Parquet loader)

gareus/query.py:310, 316     export_analysis_arrays_npz() calls it directly
    -> gareus/analysis.py:35  _ensure_analysis_arrays_npz() calls export_analysis_arrays_npz
    -> gareus/analysis.py:59  validate_analysis_metadata_readiness() calls _ensure_analysis_arrays_npz
    -> gareus/production.py:2509        validate_analysis_metadata_readiness(...)  [LIVE production pipeline]
    -> gareus/adaptive_feedback.py:31   imports validate_analysis_metadata_readiness
    -> gareus/cv_discovery.py:21,135    imports + calls validate_analysis_metadata_readiness
```

So the bug reaches: (a) `analyze_gareus_mbar.py`'s own main
non-union-adaptive PMF report path (`load_parquet`), and (b) the *live*
adaptive-production/CV-discovery readiness validation whenever
`analysis_arrays.npz` needs reconstructing from Parquet (resumed/incomplete
runs — exactly the scenario `_ensure_analysis_arrays_npz`'s own docstring
names). This raises the stakes above a pure offline-script fix: the fix
changes real output for a live-pipeline consumer, not only for
`analyze_gareus_mbar.py`'s own reports.

`gareus/helptext.py`'s `-hh` encyclopedia (lines 229-232, 884-887) also
documents `reconstruct_bias_matrix(samples['cv1'], samples['cv2'], windows,
beta)` verbatim as a stable, directly-callable public API for ad hoc
user scripts — not executed code, but an advertised call signature that
must not change shape.

### `validate_analysis_metadata_readiness`'s NaN tolerance — verified, not a blocker

`export_analysis_arrays_npz` (unlike `analyze_gareus_mbar.py`'s `clean()`)
does **not** row-mask non-finite `u_nk`/`umbrella_reduced_bias_nk` entries
before persisting the NPZ. Once `reconstruct_bias_matrix` is fixed, a
2D run with any genuinely-unmeasured `cv2` sample will, for the first time,
produce NaN entries in the persisted `analysis_arrays.npz`. Checked directly
(`gareus/analysis.py:195-196`):

```python
if required <= keys and ub.ndim == 2 and not np.all(np.isfinite(ub)):
    warnings.append("umbrella_reduced_bias_nk contains non-finite values")
```

This is appended to `warnings`, **not** `errors` — `validate_analysis_metadata_readiness`
already tolerates this exact scenario as an informational warning, not a
validation failure. (The one other NaN-aware check nearby,
`bias_matrix_finite_fraction` at line 169-176, is a finite&finite consistency
smoke-check between `umbrella_bias_kj_mol_nk` and `umbrella_reduced_bias_nk`,
also non-fatal.) The fix therefore does not require any new masking logic in
`export_analysis_arrays_npz` for this consumer to keep behaving sanely — it
requires only a dedicated regression test proving this tolerance actually
holds today, since nothing currently exercises it (Task 2 below).

### The unmigrated MBAR solver family, and a migration-specific hazard

`solve_mbar` (the backend dispatcher) and its four backends
(`solve_mbar_numba`, `solve_mbar_numba_anderson`, `solve_mbar_sambar` +
`solve_mbar_sambar_warmstart`, `solve_mbar_lbfgs`), plus their shared
`logsumexp`/`logsumexp_axis1_finite`/`logsumexp_axis0_finite`/`norm_logw`/
`ess`/`_anderson_step`/`overlap_matrix`/`_subset_logw_from_global_fk` helpers
and two `@njit`-decorated kernels (`_numba_mbar_update`,
`_numba_mbar_logdenom`), are ~750 lines of scientifically load-bearing code
still living in the top-level script — the last major numerical subsystem
blocking A6's eventual retirement of `analyze_gareus_mbar.py`. This part has
no known correctness bug (it duplicates nothing in the package today), but
moving it surfaces a **migration-specific hazard that does not exist for any
other A-plan**:

`analyze_gareus_mbar.py`'s `parse_args()` applies `--sambar-*`/
`--mbar-anderson-history`/`--mbar-backend` CLI overrides by directly mutating
its **own module globals** after argparse runs:

```python
g = globals()
g['DEFAULT_MBAR_BACKEND'] = str(getattr(args, 'mbar_backend', ...) or ...)
g['SAMBAR_EPOCHS'] = int(getattr(args, 'sambar_epochs', ...))
...  # 6 more, wrapped in try/except Exception: pass
```

This only works today because `solve_mbar`/`solve_mbar_sambar`/
`solve_mbar_sambar_warmstart`/`solve_mbar_numba_anderson` read
`SAMBAR_EPOCHS` etc. as bare names resolved against **the module they are
defined in** (Python's ordinary global-lookup semantics) — which today is
the same module doing the mutating. Confirmed load-bearing, not
hypothetical: `analyze()`'s main PMF report call,
`solve_mbar(d.u_nk, d.window, ..., backend=getattr(args,'mbar_backend','auto'), ...)`
(line ~9602), never passes `sambar_epochs=` explicitly, so `solve_mbar`'s own
`sambar_epochs=None` default flows through to `solve_mbar_sambar`, which
resolves it via `epochs = SAMBAR_EPOCHS if epochs is None else int(epochs)` —
i.e. **the global mutation is the only mechanism by which `--sambar-epochs`
(and five sibling flags) reach the default `analyze()` call path at all.**

If these functions and constants move to `gareus/mbar_analysis/solvers.py`
and `analyze_gareus_mbar.py` merely does `from
gareus.mbar_analysis.solvers import SAMBAR_EPOCHS, ...` (a plain name
import, identical in form to A1's `KJ_PER_KCAL` pattern), the override block
above keeps "succeeding" with **zero exception** (the surrounding `except
Exception: pass` would swallow one anyway) while only ever rebinding
`analyze_gareus_mbar.py`'s own now-orphaned copy of the name. The actual
solver functions, now defined in `gareus.mbar_analysis.solvers`, would keep
reading *that* module's untouched default — **every `--sambar-*` and
`--mbar-anderson-history` CLI flag would silently become a no-op**, with no
error, warning, or test failure to reveal it (confirmed: `grep -rn
"SAMBAR_EPOCHS\|sambar_epochs\|mbar_anderson_history" tests/` finds zero
references — there is no existing coverage of this override mechanism at
all). This is a bug the *migration itself* would introduce if the override
block is moved verbatim; it must be explicitly retargeted as part of this
plan, not treated as an unrelated follow-up.

## Recommended Approach

**(a) Unify into one, correct, shared implementation — fixed in place in
`gareus/query.py`.** Concretely, four coordinated moves:

1. **Fix `gareus/query.py`'s `reconstruct_bias_matrix` in place.** Add the
   missing window-side guard (`isfinite(center2) and isfinite(k2) and k2 >
   0`, matching implementations 2/3 exactly) and remove the
   `np.where(valid, ..., 0.0)` sample-side fabrication so a non-finite
   sample `cv2` under a real secondary restraint produces a NaN entry
   instead. Also replace the inline `4.184` literal with `KJ_PER_KCAL`
   imported from `gareus.units` (Plan A1's new named constant) — a small,
   free deduplication noticed while already touching this function; not
   previously called out by A1 (A1 only addressed `gareus/units.py`'s own
   inline literals and `analyze_gareus_mbar.py`'s copy, not this separate
   occurrence in `query.py`). **Public signature
   `reconstruct_bias_matrix(cv_A, cv2, windows, beta)` is unchanged** — every
   real caller (`analyze_gareus_mbar.load_parquet`,
   `export_analysis_arrays_npz`, the `-hh` documented example, all of
   `tests/test_query.py`'s and the real-OpenMM thermodynamic-validity oracle
   suite's direct calls) needs zero changes.
2. **New `gareus/mbar_analysis/bias.py`**: relocate
   `_compute_u_nk_analytical`, `_parse_epoch_window_map_native_params`,
   `_epoch_bias_param_vectors`, `_reconstruct_union_bias_block`
   out of `analyze_gareus_mbar.py`. The two
   bias-math functions become **thin delegating wrappers** around the
   now-fixed `gareus.query.reconstruct_bias_matrix` (translating their own
   array/dict-shaped inputs into `reconstruct_bias_matrix`'s `windows:
   list[dict]` convention) rather than duplicating the formula a third and
   fourth time — collapsing three formula copies to one canonical
   implementation. Their own guard logic was **already correct** (per the
   existing `tests/test_bias_reconstruction_nan_handling.py`), so this is a
   pure deduplication with zero intended behavior change, verified by a
   before/after identity check.

   > **CORRECTION (cross-plan reconciliation pass) — this list was wrong in
   > an earlier draft, in two directions.**
   >
   > **Removed: `_is_usable_for_mbar` and `_merge_missing_usable_states` are
   > Plan A2's, not A3's.** An earlier draft of this spec named both here
   > (and gave them bodies in the `bias.py` sketch below, plus `__all__` and
   > re-import entries). That double-claimed them:
   > `docs/superpowers/plans/2026-08-13-mbar-analysis-modularization-a2.md`'s
   > Task 7 already relocates both into
   > `gareus/mbar_analysis/loaders_union_parquet.py`, with full bodies, its
   > own re-import block, its own delete-from-script grep anchor, and its own
   > identity test (`_UNION_NAMES`). The reconciliation pass confirmed **no
   > caller of either function exists outside A2's union-Parquet loading
   > cluster** (A2's Task 7 Interfaces line records the same grep result:
   > "internal to this module only"), so A2 is the correct owner and this
   > plan leaves both definitions untouched in `analyze_gareus_mbar.py`.
   > `tests/test_final_registry_merge.py` is correspondingly A2's regression
   > surface, not this plan's.
   >
   > **Added: `_parse_epoch_window_map_native_params` is A3's.** It had no
   > relocation task in *any* plan — A2 explicitly deferred it to A3 ("stay
   > behind in the script, Plan A3's"), and A3's own Task 3 explicitly
   > excluded it — leaving it permanently unowned, which would have forced
   > `loaders_union_parquet.py` to keep a lazy
   > `from analyze_gareus_mbar import _parse_epoch_window_map_native_params`
   > (package importing the retiring script) for A6e to resolve. It belongs
   > here, next to `_epoch_bias_param_vectors`: it parses one epoch's own
   > `epoch_window_map.csv` rows into exactly the
   > `{state_id: {primary_center, primary_k, secondary_center, secondary_k}}`
   > dict that `_epoch_bias_param_vectors` consumes, and the two are always
   > used together (`analyze_gareus_mbar.py:1611`; single call site `:1707`,
   > inside A2's `_load_epoch_task`).
3. **New `gareus/mbar_analysis/solvers.py`**: relocate the entire MBAR
   solver family, `logsumexp`* helpers, `overlap_matrix`,
   `_subset_logw_from_global_fk`, the `@njit` kernels, the
   NUMBA/SCIPY-optional-dependency shims, and the `DEFAULT_MBAR_BACKEND`/
   `SAMBAR_*`/`MBAR_ANDERSON_HISTORY` module constants, verbatim (no
   behavior change to the numerics themselves).
4. **Retarget the CLI-override block**: change
   `g = globals(); g['SAMBAR_EPOCHS'] = ...` (and its 7 siblings) in
   `analyze_gareus_mbar.py`'s `parse_args()` to mutate
   `gareus.mbar_analysis.solvers`'s own module namespace directly
   (`import gareus.mbar_analysis.solvers as _mbar_solvers;
   _mbar_solvers.SAMBAR_EPOCHS = ...`), with a dedicated two-level
   regression test (module-attribute check, and an actual-solver-call
   check via monkeypatch) proving the flags still functionally work
   post-move — since, as established above, nothing currently tests this
   mechanism at all.

`analyze_gareus_mbar.py` imports every relocated name back by simple
`from gareus.mbar_analysis.X import name1, name2, ...` (the exact pattern
A1 established for `KJ_PER_KCAL`/`K_B_KJ_PER_MOL_K`), so its ~20+ existing
call sites across the file need zero changes beyond the import line itself.

**Verification is feasibility-gated, not assumed**: Task 1 (the
`reconstruct_bias_matrix` fix) and its dedicated numeric test land and pass
*before* Task 3/4 relocate any code around it, so the highest-stakes change
is isolated in its own commit with its own before/after evidence, independent
of the lower-risk pure-relocation work.

## Why This Approach

- **Eliminates the divergence at its root**, matching A1's own stated intent
  for this exact duplication ("Plan A3: relocate MBAR solvers and bias
  reconstruction, reconciling with `gareus/query.py`'s
  `reconstruct_bias_matrix` — including fixing the already-documented
  NaN-guard gap"). Fixing `query.py` in place — rather than moving the
  canonical formula into the new `mbar_analysis` subpackage and having
  `query.py` import it — keeps the live package's own dependency direction
  intact: `gareus/analysis.py` → `gareus/production.py` already depend on
  `gareus/query.py` today, and `gareus/query.py` is not part of the
  `analyze_gareus_mbar.py`-retirement sequence (A1-A6 folds the *script*
  into the package; `query.py` already lives in the package and stays there
  after A6 too).
- **Preserves every real call site's signature and behavior except the two
  intentional bug fixes.** `_compute_u_nk_analytical`/
  `_reconstruct_union_bias_block` keep their exact existing signatures (so
  `_augment_with_adaptive_rounds`/`load_parquet_adaptive_union` — both A2's
  domain, not yet relocated — need zero changes), and their own
  already-correct guard behavior is preserved bit-for-bit by delegating to
  an equivalently-correct shared core rather than being deleted and
  replaced with call sites that would need reshaping to a different
  calling convention.
- **The CLI-override retarget is not optional polish — it is the one place
  this specific migration can silently break user-facing behavior with zero
  error surfaced**, unlike every other constant/function relocated in this
  plan (which are read the same way regardless of which module defines
  them). No other A-plan's domain has an equivalent "the answer changes
  post-hoc via `globals()` mutation" mechanism, so this hazard is unique to
  A3 and must be handled explicitly here.
- **Reuses this session's own prior audit work as the regression baseline**
  rather than re-deriving it: `tests/test_bias_reconstruction_nan_handling.py`,
  `tests/test_union_mbar_per_epoch_bias.py`, and
  `tests/test_final_registry_merge.py` already encode exactly the guard
  semantics implementations 2 and 3 must keep after relocation; this plan
  leans on them being importable unchanged (they import from
  `analyze_gareus_mbar`, which keeps re-exporting the same names) rather
  than writing parallel duplicate tests.

## Alternatives Considered

### Delete `_compute_u_nk_analytical`/`_reconstruct_union_bias_block`, call `reconstruct_bias_matrix` directly at their former call sites

Rejected: this forces `_augment_with_adaptive_rounds` and
`load_parquet_adaptive_union` (both A2's domain, not yet relocated, out of
scope for A3 to redesign) to change their calling convention from
parallel-array parameters to a `windows: list[dict]` shape, and discards two
already-audited, already-passing regression test files
(`tests/test_bias_reconstruction_nan_handling.py`,
`tests/test_union_mbar_per_epoch_bias.py`) for no scientific benefit — they'd
need rewriting to test a different call shape, not because anything they
check today is wrong.

### New `gareus/mbar_analysis/bias.py` owns the canonical formula instead of `gareus/query.py`

Rejected: `gareus/query.py`'s own module docstring already declares this
formula as its responsibility ("The N×K bias matrix ... is reconstructed
analytically ... This is 10-100x smaller than storing the full matrix"), and
it is already imported by the live pipeline
(`gareus/analysis.py`→`gareus/production.py`,
`gareus/adaptive_feedback.py`, `gareus/cv_discovery.py`). Moving the
canonical formula *out* of `query.py` into a subpackage whose whole purpose
is absorbing the *script* would invert the dependency the live pipeline
already has — `gareus/analysis.py` would need to start importing from
`gareus.mbar_analysis`, a package conceptually downstream of it.

### Decouple `_subset_logw_from_global_fk` from the `Data` dataclass (accept raw `u_nk`/`window` arrays instead of a `Data` object)

Considered, for a "solvers.py never depends on a specific data container
shape" ideal. Rejected: `from __future__ import annotations` (already active
in `analyze_gareus_mbar.py`, carried into the new module) makes the
`d_subset: 'Data'` annotation a lazily-evaluated string with **zero runtime
import requirement** — there is no real circular-import risk to avoid.
Meanwhile `tests/test_masked_logw_subset_pmf.py` calls
`_subset_logw_from_global_fk(d_subset, m_global["f_k"])` with a real `Data`
object in two places today; changing the signature would force edits to an
already-passing, physics-critical regression test for a purely aesthetic
gain. Keep the signature exactly as-is.

### Fix the CLI-override hazard by threading SAMBAR/backend values through explicit function parameters instead of globals

Rejected as a larger, riskier diff than this plan needs: every one of
`solve_mbar`/`solve_mbar_sambar`/`solve_mbar_sambar_warmstart`/
`solve_mbar_numba_anderson`'s internal `if X is None: X = MODULE_DEFAULT`
resolution logic would need touching, plus every call site that currently
relies on the implicit default would need an explicit value threaded from
`args`. A single retargeted `globals()`-mutation block plus one new
regression test achieves the same user-facing correctness with a much
smaller, more auditable diff. Left as a plausible future cleanup (arguably
this global-mutation-for-CLI-override pattern is itself not great style),
but out of scope for a plan whose job is relocation, not redesign.

### Reconcile `overlap_matrix` with `_hist_overlap`/`_adaptive_hist_overlap`/`_hist_overlap_np`

Investigated (see Architecture below) and explicitly rejected as an A3-scope
change — see "Histogram-overlap: related but not reconcilable here" below.

## Architecture

### Files touched

```
gareus/query.py                       MODIFY  (fix reconstruct_bias_matrix's two guards; KJ_PER_KCAL import)
gareus/mbar_analysis/bias.py          CREATE  (4 relocated functions; 2 become thin delegating wrappers)
gareus/mbar_analysis/solvers.py       CREATE  (MBAR solver family + logsumexp/overlap/subset-logw helpers + constants)
analyze_gareus_mbar.py                MODIFY  (delete relocated bodies, import by name; retarget globals-override block)
tests/test_query_reconstruct_bias_matrix_nan_guard.py       CREATE
tests/test_analysis_readiness_nonfinite_bias_warning.py     CREATE
tests/test_mbar_analysis_bias_module.py                     CREATE
tests/test_mbar_analysis_solvers_module.py                  CREATE
```

**Precondition**: this plan assumes Plan A1 has already executed —
`gareus/mbar_analysis/__init__.py` and `gareus/mbar_analysis/cli.py` already
exist, and `gareus.units.KJ_PER_KCAL` already exists. If A1 has not run,
`from gareus.units import KJ_PER_KCAL` (Task 1) and `import
gareus.mbar_analysis.bias`/`gareus.mbar_analysis.solvers` (Tasks 3-4) will
fail immediately and loudly at import time — the correct fail-fast behavior,
not a new guard this plan needs to add.

### `gareus/query.py` (after) — the fixed `reconstruct_bias_matrix`

```python
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .io import read_json_file
from .store import SegmentRegistry
from .units import KJ_PER_KCAL

...

def reconstruct_bias_matrix(
    cv_A: np.ndarray,
    cv2: Optional[np.ndarray],
    windows: list,
    beta: float,
) -> np.ndarray:
    """Reconstruct umbrella_reduced_bias_nk analytically.

    A window is "secondary-restrained" iff it has both a "center2" and "k2"
    key AND those values are finite with k2 > 0 -- such a window's CV2 term
    is included only when that holds; otherwise the CV2 term is omitted
    entirely (a 1D run, or an unrestrained 2D state, gets a purely-CV1 bias
    regardless of what cv2 holds for its samples).

    For a secondary-restrained window, a sample whose own cv2 value is not
    finite gets NaN for that (sample, window) entry -- this is intentional
    exclusion-by-propagation (the same convention analyze_gareus_mbar.py's
    clean() and every other bias-reconstruction site in this codebase uses),
    not a bug: fabricating a zero deviation would silently claim the sample
    was on-target for a coordinate that was never actually measured.

    Parameters
    ----------
    cv_A : (N,) array of primary CV values
    cv2  : (N,) array of secondary CV values, or None for 1D runs
    windows : list of window dicts with center1, k1 (and optionally center2, k2)
    beta : 1/(kB*T) in mol/kJ (e.g. 1 / (8.314462618e-3 * T_K))

    Returns
    -------
    nk : (N, K) float64 array of dimensionless reduced umbrella biases
    """
    cv_A = np.asarray(cv_A, dtype=np.float64)
    N = len(cv_A)
    K = len(windows)
    nk = np.zeros((N, K), dtype=np.float64)
    cv2_arr = np.asarray(cv2, dtype=np.float64) if cv2 is not None else None

    for k, w in enumerate(windows):
        d1 = cv_A - float(w["center1"])
        nk[:, k] = KJ_PER_KCAL * 0.5 * float(w["k1"]) * d1 * d1  # kcal -> kJ
        if cv2_arr is not None and "center2" in w and "k2" in w:
            k2 = float(w["k2"])
            c2 = float(w["center2"])
            if math.isfinite(c2) and math.isfinite(k2) and k2 > 0.0:
                d2 = cv2_arr - c2
                nk[:, k] += KJ_PER_KCAL * 0.5 * k2 * d2 * d2

    return beta * nk
```

Module-docstring formula block (top of file) also updated to spell out the
two guards, matching the function docstring above (documentation-only diff).

### `gareus/mbar_analysis/bias.py` (new)

```python
"""Bias-reconstruction helpers relocated from analyze_gareus_mbar.py.

`_compute_u_nk_analytical` and `_reconstruct_union_bias_block` delegate to
the canonical `gareus.query.reconstruct_bias_matrix` (fixed as part of this
same modularization plan) rather than duplicating its formula a third and
fourth time. `_parse_epoch_window_map_native_params` and
`_epoch_bias_param_vectors` are pure supporting glue for
`load_parquet_adaptive_union`'s per-epoch-native bias reconstruction (see
CLAUDE.md's "Union-MBAR bias matrix used a stale global registry snapshot"
entry) and have no formula of their own to reconcile: the first parses one
epoch's own `epoch_window_map.csv` rows into a `{state_id: {...}}`
native-params dict, the second consumes exactly that dict.

`_is_usable_for_mbar` / `_merge_missing_usable_states` deliberately do NOT
live here -- Plan A2 relocates them into
`gareus/mbar_analysis/loaders_union_parquet.py`, the only cluster that calls
them (see the CORRECTION in Recommended Approach).
"""
from __future__ import annotations

import math

import numpy as np

from gareus import query

# Module-level `from gareus import query` (not `from gareus.query import
# reconstruct_bias_matrix`) so `monkeypatch.setattr(gareus.query,
# "reconstruct_bias_matrix", fake)` in a test actually intercepts calls made
# from this module too -- a plain name-import would bind a local reference
# at import time that a later monkeypatch on gareus.query's own attribute
# would not affect.


def _compute_u_nk_analytical(cv1: np.ndarray, cv2: np.ndarray,
                              union_windows: list, beta: float) -> np.ndarray:
    """Compute N x K_union reduced bias matrix analytically from CV values."""
    windows = [
        {
            "center1": w["primary_center"], "k1": w["primary_k_kcal"],
            "center2": w["secondary_cv_center"], "k2": w["secondary_k_kcal"],
        }
        for w in union_windows
    ]
    return query.reconstruct_bias_matrix(cv1, cv2, windows, beta)


def _parse_epoch_window_map_native_params(rows: list) -> dict:
    """Return ``{state_id: {primary_center, primary_k, secondary_center, secondary_k}}``.

    (docstring unchanged from analyze_gareus_mbar.py -- see original for the
    full rationale about per-epoch native window params vs. the live
    registry's recentered ones)
    """
    def _f(row: dict, key: str, default: float) -> float:
        v = row.get(key, '')
        if v in ('', 'None', 'nan', None):
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    out: dict = {}
    for r in rows:
        if 'state_id' not in r:
            continue
        sid = int(r['state_id'])
        out[sid] = {
            'primary_center': _f(r, 'primary_center', float('nan')),
            'primary_k': _f(r, 'primary_k', float('nan')),
            'secondary_center': _f(r, 'secondary_center', float('nan')),
            'secondary_k': _f(r, 'secondary_k', float('nan')),
        }
    return out


def _epoch_bias_param_vectors(native_params: dict, state_ids: list,
                               global_primary_centers: np.ndarray, global_primary_ks: np.ndarray,
                               global_sec_centers: np.ndarray, global_sec_ks: np.ndarray) -> tuple:
    """Per-epoch (primary_center, primary_k, secondary_center, secondary_k) vectors.

    (unchanged from analyze_gareus_mbar.py)
    """
    pc = global_primary_centers.copy()
    pk = global_primary_ks.copy()
    sc = global_sec_centers.copy()
    sk = global_sec_ks.copy()
    for k, sid in enumerate(state_ids):
        row = native_params.get(sid)
        if row is None:
            continue
        if math.isfinite(row['primary_center']):
            pc[k] = row['primary_center']
        if math.isfinite(row['primary_k']):
            pk[k] = row['primary_k']
        if math.isfinite(row['secondary_center']):
            sc[k] = row['secondary_center']
        if math.isfinite(row['secondary_k']):
            sk[k] = row['secondary_k']
    return pc, pk, sc, sk


def _reconstruct_union_bias_block(cv: np.ndarray, cv2: np.ndarray, beta: float,
                                   primary_centers: np.ndarray, primary_ks: np.ndarray,
                                   sec_centers: np.ndarray, sec_ks: np.ndarray) -> np.ndarray:
    """Build one epoch-block's N x K reduced-bias-energy matrix.

    Thin delegation to reconstruct_bias_matrix: converts the array-based
    calling convention this function's callers already use into the
    windows:list[dict] shape reconstruct_bias_matrix expects. K is typically
    tens to low hundreds (real runs: K=91, K=364) and this is called once per
    epoch block, not per sample, so the K-dict-object construction cost here
    is negligible next to the O(N*K) numpy arithmetic reconstruct_bias_matrix
    itself performs.
    """
    windows = [
        {"center1": primary_centers[k], "k1": primary_ks[k],
         "center2": sec_centers[k], "k2": sec_ks[k]}
        for k in range(len(primary_centers))
    ]
    return query.reconstruct_bias_matrix(cv, cv2, windows, beta)


__all__ = [
    "_compute_u_nk_analytical",
    "_parse_epoch_window_map_native_params",
    "_epoch_bias_param_vectors",
    "_reconstruct_union_bias_block",
]
```

### `gareus/mbar_analysis/solvers.py` (new) — skeleton

```python
"""MBAR self-consistent solver family, relocated verbatim from
analyze_gareus_mbar.py.

Includes the backend dispatcher (solve_mbar), four solver backends, their
shared logsumexp/Anderson-mixing/overlap helpers, the two @njit-decorated
hot-loop kernels, and the MBAR configuration constants
(DEFAULT_MBAR_BACKEND, SAMBAR_*, MBAR_ANDERSON_HISTORY) that
analyze_gareus_mbar.py's CLI overrides at parse time -- see that file's
parse_args() for the override block, which targets THIS module's globals()
directly (gareus.mbar_analysis.solvers), not analyze_gareus_mbar.py's own.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None
    prange = range
    set_num_threads = None
    get_num_threads = None

try:
    from scipy.optimize import minimize as _scipy_minimize
    SCIPY_AVAILABLE = True
except Exception:
    _scipy_minimize = None
    SCIPY_AVAILABLE = False

# -----------------------------------------------------------------------
# MBAR solver configuration defaults -- mutated at runtime by
# analyze_gareus_mbar.py's parse_args() when --sambar-*/--mbar-backend/
# --mbar-anderson-history are passed. See that module's retargeted
# override block (this file's own globals(), not analyze_gareus_mbar's).
DEFAULT_MBAR_BACKEND = 'sambar'
SAMBAR_EPOCHS = 30
SAMBAR_INITIAL_BATCH_SIZE = 1024
SAMBAR_BATCH_PATIENCE = 5
SAMBAR_SEED = 12345
SAMBAR_LR_SCALE = 1.0
SAMBAR_DELTA_F_MAX = 10.0
SAMBAR_POLISH_BACKEND = 'numba-anderson'
MBAR_ANDERSON_HISTORY = 5

# ... logsumexp, logsumexp_axis1_finite, logsumexp_axis0_finite, norm_logw,
# _numba_mbar_update, _numba_mbar_logdenom, _anderson_step,
# solve_mbar_numba, solve_mbar_numba_anderson, solve_mbar_sambar_warmstart,
# solve_mbar_sambar, solve_mbar_lbfgs, solve_mbar, overlap_matrix,
# _subset_logw_from_global_fk moved verbatim (bodies unchanged; see Plan A3
# implementation plan Task 4 for the exact current-line-range extraction).

__all__ = [
    "NUMBA_AVAILABLE", "SCIPY_AVAILABLE",
    "DEFAULT_MBAR_BACKEND", "SAMBAR_EPOCHS", "SAMBAR_INITIAL_BATCH_SIZE",
    "SAMBAR_BATCH_PATIENCE", "SAMBAR_SEED", "SAMBAR_LR_SCALE",
    "SAMBAR_DELTA_F_MAX", "SAMBAR_POLISH_BACKEND", "MBAR_ANDERSON_HISTORY",
    "logsumexp", "logsumexp_axis1_finite", "logsumexp_axis0_finite",
    "norm_logw", "solve_mbar_numba", "solve_mbar_numba_anderson",
    "solve_mbar_sambar_warmstart", "solve_mbar_sambar", "solve_mbar_lbfgs",
    "solve_mbar", "overlap_matrix", "_subset_logw_from_global_fk",
]
```

`Progress` and `Data` (used only as `Optional[Progress]`/`'Data'` type
hints) are never imported — `from __future__ import annotations` (present in
both the source and destination files) makes every annotation a lazily
evaluated string, so no import is required and none of these functions ever
`isinstance`-checks or otherwise runtime-resolves either name.

### `analyze_gareus_mbar.py` (relevant diffs)

Constant/function bodies deleted, replaced with:

```python
from gareus.mbar_analysis.bias import (
    _compute_u_nk_analytical,
    _parse_epoch_window_map_native_params,
    _epoch_bias_param_vectors,
    _reconstruct_union_bias_block,
)
from gareus.mbar_analysis.solvers import (
    NUMBA_AVAILABLE, SCIPY_AVAILABLE,
    DEFAULT_MBAR_BACKEND, SAMBAR_EPOCHS, SAMBAR_INITIAL_BATCH_SIZE,
    SAMBAR_BATCH_PATIENCE, SAMBAR_SEED, SAMBAR_LR_SCALE, SAMBAR_DELTA_F_MAX,
    SAMBAR_POLISH_BACKEND, MBAR_ANDERSON_HISTORY,
    logsumexp, logsumexp_axis1_finite, logsumexp_axis0_finite,
    norm_logw, solve_mbar_numba, solve_mbar_numba_anderson,
    solve_mbar_sambar_warmstart, solve_mbar_sambar, solve_mbar_lbfgs,
    solve_mbar, overlap_matrix, _subset_logw_from_global_fk,
)
```

(`njit`/`prange`/`set_num_threads`/`get_num_threads`/`_scipy_minimize` are
*not* re-imported into `analyze_gareus_mbar.py` — nothing outside
`solvers.py` itself references them once the `@njit` kernels move with the
rest of the solver family.)

And the retargeted CLI-override block:

```python
try:
    import gareus.mbar_analysis.solvers as _mbar_solvers
    _mbar_solvers.DEFAULT_MBAR_BACKEND = str(getattr(args, 'mbar_backend', _mbar_solvers.DEFAULT_MBAR_BACKEND) or _mbar_solvers.DEFAULT_MBAR_BACKEND)
    _mbar_solvers.SAMBAR_EPOCHS = int(getattr(args, 'sambar_epochs', _mbar_solvers.SAMBAR_EPOCHS))
    _mbar_solvers.SAMBAR_INITIAL_BATCH_SIZE = int(getattr(args, 'sambar_initial_batch_size', _mbar_solvers.SAMBAR_INITIAL_BATCH_SIZE))
    _mbar_solvers.SAMBAR_BATCH_PATIENCE = int(getattr(args, 'sambar_batch_patience', _mbar_solvers.SAMBAR_BATCH_PATIENCE))
    _mbar_solvers.SAMBAR_SEED = int(getattr(args, 'sambar_seed', _mbar_solvers.SAMBAR_SEED))
    _mbar_solvers.SAMBAR_LR_SCALE = float(getattr(args, 'sambar_lr_scale', _mbar_solvers.SAMBAR_LR_SCALE))
    _mbar_solvers.SAMBAR_DELTA_F_MAX = float(getattr(args, 'sambar_delta_f_max', _mbar_solvers.SAMBAR_DELTA_F_MAX))
    _mbar_solvers.SAMBAR_POLISH_BACKEND = str(getattr(args, 'sambar_polish_backend', _mbar_solvers.SAMBAR_POLISH_BACKEND) or _mbar_solvers.SAMBAR_POLISH_BACKEND)
    _mbar_solvers.MBAR_ANDERSON_HISTORY = int(getattr(args, 'mbar_anderson_history', _mbar_solvers.MBAR_ANDERSON_HISTORY))
    # Keep this module's own re-exported names in sync too, since some
    # internal call sites (and the `from gareus.mbar_analysis.solvers import
    # DEFAULT_MBAR_BACKEND` binding at module scope) still read the local
    # name for e.g. argparse's default= display; solve_mbar* functions
    # themselves only ever read _mbar_solvers' copy, which is what matters
    # for actual behavior.
    globals()['DEFAULT_MBAR_BACKEND'] = _mbar_solvers.DEFAULT_MBAR_BACKEND
    globals()['SAMBAR_EPOCHS'] = _mbar_solvers.SAMBAR_EPOCHS
    globals()['SAMBAR_INITIAL_BATCH_SIZE'] = _mbar_solvers.SAMBAR_INITIAL_BATCH_SIZE
    globals()['SAMBAR_BATCH_PATIENCE'] = _mbar_solvers.SAMBAR_BATCH_PATIENCE
    globals()['SAMBAR_SEED'] = _mbar_solvers.SAMBAR_SEED
    globals()['SAMBAR_LR_SCALE'] = _mbar_solvers.SAMBAR_LR_SCALE
    globals()['SAMBAR_DELTA_F_MAX'] = _mbar_solvers.SAMBAR_DELTA_F_MAX
    globals()['SAMBAR_POLISH_BACKEND'] = _mbar_solvers.SAMBAR_POLISH_BACKEND
    globals()['MBAR_ANDERSON_HISTORY'] = _mbar_solvers.MBAR_ANDERSON_HISTORY
except Exception:
    pass
return args
```

(The trailing "keep this module's own re-exported names in sync too" half is
belt-and-suspenders: no known current call site reads
`analyze_gareus_mbar.SAMBAR_EPOCHS` etc. directly after `parse_args()`
returns rather than via a fresh `solve_mbar(...)` call, but it costs nine
cheap assignments to close off the possibility entirely rather than argue
it's unreachable.)

### Histogram-overlap: related but not reconcilable here

`overlap_matrix(cv, window, bins, K)` (moved into `solvers.py` unchanged)
computes a full **K x K** window-window overlap matrix in one vectorized
pass: a single `np.bincount` over a `window*B+bin` linear index builds all K
windows' histograms at once (shape `(K, B)`), normalizes each row, then a
`H[:,None,:]`/`H[None,:,:]` broadcast-min-sum produces every pairwise overlap
simultaneously.

`gareus/math_helpers.py`'s `_hist_overlap`/`_adaptive_hist_overlap` and
`gareus/diagnostics.py`'s `_hist_overlap_np` are conceptually the same
"sum of per-bin min(P_a, P_b)" statistic, but structurally different: each
takes exactly **two** 1D sample arrays and returns one scalar, with three
mutually inconsistent conventions between them (`_hist_overlap`: no padding,
requires >=5 samples each, `bins=24` default; `_adaptive_hist_overlap`: pads
by `max(0.1, 0.02*(hi-lo))`, only requires non-empty; `_hist_overlap_np`:
same padding, requires >=5 samples each). Rebuilding `overlap_matrix` on top
of any of the two-sample functions would mean calling one of them
`O(K^2)` times, each re-histogramming its two slices from scratch — throwing
away the one-pass vectorized-`bincount` design `overlap_matrix`'s own
docstring calls out ("vectorized histogram assembly"). Conversely,
rebuilding any of the two-sample functions on `overlap_matrix` would mean
constructing a fake `window` array of all-0s/all-1s and a `K=2` call just to
get one scalar out of a K×K matrix — solving a harder problem to get a
simpler answer.

**Decision: move `overlap_matrix` into `solvers.py` unchanged; do not
attempt to reconcile it with the two-sample family.** This mirrors A1's own
framing of this as a genuine three-way divergence, not a simple duplication
— the three padding/min-count conventions are themselves inconsistent with
each other even before `overlap_matrix`'s structural difference is
considered, and Plan A5 (which owns
`gareus/diagnostics.py`/`gareus/math_helpers.py`'s side of this) is the
right place to decide whether the two-sample family should eventually be
reimplemented in terms of a shared K-way core (with `K=2` as a trivial case)
or left alone. Flagging explicitly here rather than silently leaving it
undiscussed, per this plan's mandate to surface every divergence found.

### `gareus/mbar_subsample.py` — checked, not applicable

`equilibrated_subsample_indices` (33 lines) does equilibration-discard +
autocorrelation-length subsampling via `pymbar.timeseries`, deciding **which
raw samples get pooled** before MBAR ever runs — a live/adaptive-production
sampling-selection concern, not post-hoc MBAR solving or bias reconstruction.
It shares no formula, no data shape, and no call-graph edge with anything in
this plan's domain (confirmed via `grep -rn
"equilibrated_subsample_indices"` across the repo: only used from
`gareus/adaptive_production.py`, never from `analyze_gareus_mbar.py` or
`gareus/query.py`). No action needed; noted per this plan's explicit mandate
to check it.

## Error Handling

- **`reconstruct_bias_matrix`'s fixed NaN-propagation is not an error path**
  — it is this codebase's established "exclude via NaN, let the caller
  decide" convention (see Problem section). The fixed function never raises
  for a non-finite sample; it returns NaN entries. Callers that route through
  `analyze_gareus_mbar.py`'s `clean()` (every `Data` loader) get automatic
  row-exclusion; `export_analysis_arrays_npz` does not row-mask and persists
  the NaN as-is, which `validate_analysis_metadata_readiness` already
  tolerates as a warning (verified above, pinned by Task 2's new test).
- **The two relocated bias-math wrappers add zero new error handling** —
  they are pure delegations; any exception `reconstruct_bias_matrix` could
  raise (malformed `windows` dicts missing required keys — unchanged,
  pre-existing `KeyError` behavior) propagates unchanged.
- **The MBAR solver family's existing error handling is untouched by
  relocation**: `RuntimeError` for a requested-but-unavailable numba/scipy
  backend, `ValueError` for zero active states, the bare bare `except
  Exception: pass` around the CLI-override block (kept, now guarding
  9 assignments against `_mbar_solvers` instead of 8 against `globals()` —
  functionally identical risk profile, not worsened).
- **The retargeted override block's silent-failure risk is the one thing
  this plan adds defense against**, via the dedicated regression test in
  Task 4 rather than a code-level change to the `except Exception: pass`
  itself (changing that swallowing behavior is a separate, larger design
  question — e.g. should a typo'd CLI flag ever hard-fail the whole
  analysis run? — explicitly out of scope here).

## Testing

New tests (this plan):

- `tests/test_query_reconstruct_bias_matrix_nan_guard.py` — the
  `reconstruct_bias_matrix` fix, mirroring
  `tests/test_bias_reconstruction_nan_handling.py`'s existing structure/
  naming so the three bias-reconstruction test files read as one consistent
  family: window-side guard (NaN/zero `k2`, NaN `center2` each independently
  produce a finite CV1-only bias, not NaN-poisoned), sample-side exclusion
  (a NaN sample `cv2` under a real secondary restraint produces NaN for that
  entry only, a neighboring finite-`cv2` sample in the same window
  unaffected), aggregate all-NaN-column edge case, and a positive check that
  a real secondary restraint still applies correctly (the fix must not
  disable the term globally).
- `tests/test_analysis_readiness_nonfinite_bias_warning.py` — a synthetic
  `analysis_arrays.npz` with a NaN entry in `umbrella_reduced_bias_nk`
  (npz constructed directly, not through the full Parquet pipeline, to
  isolate this from Task 1's fix) drives `validate_analysis_metadata_readiness`
  to append the "contains non-finite values" warning, with `errors` still
  empty for that specific check — closing the "does this consumer tolerate
  the newly-possible NaN" question with a pinned test rather than a
  point-in-time code read.
- `tests/test_mbar_analysis_bias_module.py` — `gareus.mbar_analysis.bias`
  is importable; `analyze_gareus_mbar._compute_u_nk_analytical is
  gareus.mbar_analysis.bias._compute_u_nk_analytical` (`is`, not `==` — same
  identity test style as A1's constants check) for all 4 relocated names
  (`_compute_u_nk_analytical`, `_parse_epoch_window_map_native_params`,
  `_epoch_bias_param_vectors`, `_reconstruct_union_bias_block`);
  `_compute_u_nk_analytical`/`_reconstruct_union_bias_block`'s outputs on
  the exact fixture data from `tests/test_bias_reconstruction_nan_handling.py`
  are identical to their pre-relocation selves **within floating-point
  tolerance** (`np.testing.assert_allclose`/`pytest.approx`, not
  `np.array_equal`) — proving zero *behavior* change for these two, since
  they were already correct. Not literally bitwise: the pre-relocation
  originals associate the multiplication differently (`_compute_u_nk_analytical`
  precomputed `scale = beta*KJ_PER_KCAL` once and multiplied it through;
  `_reconstruct_union_bias_block` multiplied `beta*KJ_PER_KCAL` into each
  term separately; the unified `reconstruct_bias_matrix` accumulates in
  kcal and multiplies by `beta` once at the end) — floating-point addition
  and multiplication are not associative, so this changes results at the
  ~1e-16-relative-error level, not the formula itself.
- `tests/test_mbar_analysis_solvers_module.py` — `gareus.mbar_analysis.solvers`
  is importable; identity checks for all relocated names; the two-level
  CLI-override regression (module-attribute value check post-`parse_args`,
  and a monkeypatch-based check that `solve_mbar(..., backend='sambar')`
  actually receives the overridden `SAMBAR_EPOCHS` value through the real
  call chain, not just that the module attribute changed).

Existing tests serving as the regression gate (must pass unmodified after
relocation, since they import from `analyze_gareus_mbar` which keeps
re-exporting the same names):

- `tests/test_bias_reconstruction_nan_handling.py`,
  `tests/test_union_mbar_per_epoch_bias.py` — bias-math guard regression (2, 3
  above). (`tests/test_final_registry_merge.py` was listed here in an earlier
  draft; it only imports `_is_usable_for_mbar`/`_merge_missing_usable_states`,
  which this plan no longer relocates — it is Plan A2's regression gate, per
  the CORRECTION in Recommended Approach.)
- `tests/test_masked_logw_subset_pmf.py`,
  `tests/test_mbar_lbfgs_convergence.py`,
  `tests/test_perf_loading_and_solver_memory.py` — solver-family regression.
- `tests/test_query.py` (5 `reconstruct_bias_matrix` tests + 5
  `export_analysis_arrays_npz` tests) — the fixed-in-place formula's own
  direct regression gate.
- `tests/test_thermodynamic_validity_2d.py`,
  `tests/test_thermodynamic_validity_2d_rough.py`,
  `tests/test_thermodynamic_validity_real_md.py`,
  `tests/test_physics_oracle.py` — real-OpenMM-MD end-to-end oracle suite
  exercising `reconstruct_bias_matrix` + `solve_mbar` together against known
  analytic ground truth; slow (real MD), run separately from the fast unit
  suite, not part of every-commit CI-speed iteration but must be run once
  before this plan is considered done given its stakes.

Verification commands (once implemented). Every new test file below
`pytest.importorskip`s its target module — under `-q` a skip and a pass both
print `.`/`s` quietly, so **check the printed summary line's exact pass
count**, not just "no failures": a summary reading `9 skipped` (e.g. because
`gareus.query` failed to import for an unrelated reason) is not the same as
`9 passed` and must not be read as green.

```bash
pytest -q tests/test_query_reconstruct_bias_matrix_nan_guard.py tests/test_analysis_readiness_nonfinite_bias_warning.py tests/test_mbar_analysis_bias_module.py tests/test_mbar_analysis_solvers_module.py
# Expected: "9 passed, 2 passed, 7 passed, 15 passed" (one line per file) --
# NOT "... skipped". A skip means an importorskip target failed to import;
# investigate that failure, do not treat the run as green.
pytest -q tests/test_bias_reconstruction_nan_handling.py tests/test_union_mbar_per_epoch_bias.py tests/test_final_registry_merge.py tests/test_masked_logw_subset_pmf.py tests/test_mbar_lbfgs_convergence.py tests/test_query.py
python -m py_compile analyze_gareus_mbar.py gareus/query.py gareus/mbar_analysis/bias.py gareus/mbar_analysis/solvers.py
pytest -q tests/test_thermodynamic_validity_2d.py tests/test_thermodynamic_validity_2d_rough.py tests/test_physics_oracle.py  # slower, real-MD-adjacent
```

## Out Of Scope

- **Plan A2** (data loading: `load_csv`/`load_npz`/`load_parquet`/
  `load_parquet_adaptive_union`/`_augment_with_adaptive_rounds`/`clean()`/
  the `Data` dataclass) — this plan's `bias.py`/`solvers.py` are consumed by
  A2's loaders (unchanged call sites), not the reverse.
- **Plan A4** (PMF/GaMD cumulant/uncertainty math) — consumes `solve_mbar`,
  `norm_logw`, `ess`, `overlap_matrix`, `_subset_logw_from_global_fk` from
  this plan's `solvers.py`; exact signatures listed under Testing/Interfaces
  above for cross-plan verification.
- **Plan A5** (statistical diagnostics, `gareus/diagnostics.py`
  reconciliation) — owns the decision on whether
  `_hist_overlap`/`_adaptive_hist_overlap`/`_hist_overlap_np` should be
  reimplemented in terms of a shared K-way core; this plan only flags the
  divergence (see Architecture), does not resolve it.
- **`ess` is Plan A5's, not this plan's** (cross-check correction). An
  earlier draft of this spec listed `ess` among the solver-family helpers
  relocated into `solvers.py`; Plan A5's design
  (`...-a5-design.md`, "Why This Approach") claims it for
  `gareus/math_helpers.py` instead, as a shared leaf-level primitive both
  this plan's domain and A5's need. Verified directly against the source:
  no `solve_mbar*`/`logsumexp*` body calls `ess` at all — its only call
  sites (`analyze_gareus_mbar.py:2131`, `:2134`, `:7101`, `:9512`, `:9677`)
  are post-hoc reporting after a solve or a reweighting. So `ess` is simply
  left where it is by this plan (A5 relocates it), which also keeps A3 and
  A5 order-independent: neither has to land before the other. `norm_logw`
  (which `ess` is always called *with* at every call site, but never *by*)
  still moves here.
- **`gareus/mbar_subsample.py`** — confirmed unrelated (see Architecture);
  no change.
- **The `NUMEXPR_NUM_THREADS`/`NUMBA_NUM_THREADS` environment-variable cap**
  `analyze_gareus_mbar.py` sets at the very top of the file (lines 3-9, before
  any other import) to bound thread usage. This still applies whenever
  `analyze_gareus_mbar.py` itself is the entry point (it imports
  `gareus.mbar_analysis.solvers`, which imports `numba`, only after those
  environment variables are already set) — but it does **not** apply if
  something imports `gareus.mbar_analysis.solvers` directly without going
  through `analyze_gareus_mbar.py` first (e.g. a future direct consumer, or
  this plan's own `tests/test_mbar_analysis_solvers_module.py`, which numba
  may pick up with its own default thread count instead of the capped one).
  This affects performance/thread count only, never numerical results — not
  fixed here; flagging so it isn't rediscovered later as a mystery
  performance variance between the two entry paths.
- **Any change to `solve_mbar`'s numerical algorithm, default tolerances, or
  backend-selection heuristics** — this plan relocates the code, it does not
  redesign it.
- **Any change to argparse flag names, help text, or the CLI surface** —
  only the *storage location* of the constants those flags override, and the
  *target* of the post-parse override-application block, change.
- **Regenerating `write_physics_oracle_report.py`/
  `write_thermodynamic_validity_2d_rough_report.py`'s `.docx` output** — both
  scripts call `reconstruct_bias_matrix` directly and their existing
  narrative text describes specific numeric mutation-battery results; the
  fix should not change any *passing* oracle result's direction (none of
  their fixtures exercise the NaN-cv2 case), but re-running them to confirm
  and refresh the generated report is a documentation task for whoever owns
  that report, not a code change this plan makes.
- **Redesigning the `globals()`-mutation-for-CLI-override pattern itself**
  (e.g. replacing it with explicit parameter threading) — this plan
  retargets it to the correct module so it keeps working, it does not
  question whether it should exist; see Alternatives Considered.
