# MBAR Analysis Modularization — Plan A5 (Statistical Diagnostics)

## Goal

Relocate `analyze_gareus_mbar.py`'s standalone statistical-comparison
metrics — the pure-math functions used to judge PMF quality, basin
structure, and convergence, as distinct from PMF-*construction* math
(Plan A4) and convergence *orchestration* (Plan A6) — into the existing
`gareus/diagnostics.py`, and resolve the histogram-overlap-statistic
duplication that already exists inside the `gareus` package itself,
independent of anything `analyze_gareus_mbar.py` does.

## Problem

`analyze_gareus_mbar.py` defines eight small, pure functions that compare
or characterize already-computed PMF/probability arrays:

- `pmf_probability(pmf)` — normalize a raw `prob` array to a finite,
  non-negative distribution.
- `js_divergence_1d(P, Q)`, `pmf_rmse_1d(F, Fref, P, Pref, min_prob)`,
  `barrier_error_1d(F, Fref, P, Pref)` — three PMF-vs-reference comparison
  metrics, used identically by both the epoch-convergence loop and the
  frame-count convergence loop to score how close a partial-data PMF is to
  the final one.
- `identify_basins_1d(cv_A, F, min_depth_kcal)` and
  `_compute_basin_populations(prob, basins)` — partition a 1D PMF into
  basins and integrate probability mass within each.
- `_weighted_mean_std(values, weights)` and
  `_pmf_distribution_mean_std(pmf)` — mean/std of a value array under
  arbitrary importance weights, or of a PMF's own probability distribution.

None of these live in `gareus/` today; every one is a plain function
operating only on NumPy arrays and Python dicts/lists it's handed, with no
I/O, no MBAR-solve dependency, and no GaMD-specific physics knowledge —
exactly the profile `gareus/diagnostics.py` already exists to hold.

Separately, and only tangentially connected to `analyze_gareus_mbar.py`:
`gareus/diagnostics.py`'s own `_hist_overlap_np` (used by
`validate_us_mbar_inputs`'s neighbor-pair overlap check) and
`gareus/math_helpers.py`'s `_adaptive_hist_overlap` (used throughout
`gareus/adaptive_feedback.py`) are the same statistic implemented twice
inside the package that was supposed to be the single source of truth.
Both compute the padded, auto-ranging histogram-overlap
`sum(min(P_i, Q_i))`; the *only* behavioral difference between them is a
finite-sample floor (`_hist_overlap_np` returns NaN below 5 finite samples
per side; `_adaptive_hist_overlap` only guards the empty-array case).
Confirmed directly:

```
>>> _hist_overlap_np([1,1,1], [1,1,1], 0.0, 2.0)          # 3 samples/side
nan
>>> _adaptive_hist_overlap([1,1,1], [1,1,1], 0.0, 2.0)    # same input
1.0
>>> # both agree once there are enough samples:
>>> _hist_overlap_np(a50, b50, -3, 3, bins=40)
0.56
>>> _adaptive_hist_overlap(a50, b50, -3, 3, bins=40)
0.56
```

This third implementation (`overlap_matrix`, `analyze_gareus_mbar.py`'s own
K×K window-pair overlap matrix) and `gareus/logger.py`'s `_hist_overlap`
(no padding, no auto-range, `bins=24` default, list-only input) are
related but out of scope here — see Alternatives Considered and Out Of
Scope.

## Recommended Approach

**Extend `gareus/diagnostics.py` in place** with the eight functions above
(five made public via `__all__`, three kept private/unlisted, matching the
file's existing convention), leave `analyze_gareus_mbar.py`'s call sites
untouched by replacing each relocated function's *definition* with a
`from gareus.diagnostics import ...` (same strangler-fig pattern Plan A1
already used for `KJ_PER_KCAL`/`K_B_KJ_PER_MOL_K` — same object, not a
coincidentally-equal copy, so every one of the file's ~15 existing call
sites needs zero changes).

Two functions outside that eight-function list are pulled into the same
recommendation, each for a concrete, investigated reason (see Why This
Approach):

- **`ess(w)`** (currently at `analyze_gareus_mbar.py:2572`) relocates to
  **`gareus/math_helpers.py`**, not `gareus/diagnostics.py` — it's a
  general-purpose statistical primitive (Kish effective sample size from an
  importance-weight array) used across both this plan's domain and MBAR
  solver reporting (Plan A3's territory), and `math_helpers.py` is the
  package's existing zero-gareus-dependency leaf module, whereas
  `diagnostics.py` is the PMF-quality-diagnostics *domain* module. Kept
  without the module's usual leading underscore (documented exception,
  since `agm.ess` is already real public API exercised directly by three
  real-MD test files).
- **`_hist_overlap_np`** (already in `gareus/diagnostics.py`) is refactored
  to delegate to `gareus/math_helpers.py`'s `_adaptive_hist_overlap`, which
  gains one new optional parameter (`min_samples: int = 0`, default
  preserves every existing `adaptive_feedback.py` call site unchanged) so
  the 5-sample floor becomes a parameter instead of a second, drifting
  implementation.

## Why This Approach

- **`gareus/diagnostics.py` is already the right home, with real content,
  not a stub.** It already computes ESS-shaped ratios
  (`compute_gamd_reweighting_diagnostics`) and window-histogram-overlap
  connectivity checks (`validate_us_mbar_inputs`) — conceptually the same
  neighborhood as PMF-comparison metrics and basin population math. Its
  `__all__` convention (two real public entry points; small helpers
  `_read_csv_dicts`/`_safe_float`/`_sample_counts_by_window`/
  `_hist_overlap_np` unlisted but present) is exactly the shape this
  plan's five-public/three-private split follows.
- **No circular-import risk, verified by reading its actual imports, not
  assuming.** `gareus/diagnostics.py`'s only module-level imports are
  `pathlib`/`csv`/`math`/`os`/`json`/`typing`/`numpy` — it already uses
  NumPy freely throughout every relocated function's likely neighbors. Its
  one `gareus`-internal import (`from .logger import boost_anharmonicity`)
  is deliberately *lazy* — inside the function body, not at module top —
  because `gareus/logger.py` pulls in `.colors`/`.io`/`.progress`/`.tui`/
  `.cv`/`.windows`/`.math_helpers`, a real dependency chain `diagnostics.py`
  is clearly avoiding for its own top-level footprint. `gareus/math_helpers.py`
  itself has zero `gareus`-internal imports (only `math`/`typing`/`numpy`) —
  a true leaf module. Adding `from .math_helpers import _adaptive_hist_overlap`
  to `diagnostics.py`'s top level introduces no such chain: `math_helpers.py`
  imports nothing from `gareus` at all, so this one new import cannot create
  a cycle and does not compromise the lazy-import discipline that motivated
  avoiding `.logger`.
- **`identify_basins_1d`'s optional `scipy.signal.find_peaks` dependency is
  safe to replicate.** `scipy` is already a real, declared dependency
  (`pyproject.toml`'s `analysis` extra), the function already degrades
  gracefully to a pure-Python peak search when scipy is unavailable
  (confirmed: `analyze_gareus_mbar.py:4129-4144`), and `gareus/diagnostics.py`
  already tolerates an optional dependency being absent (the `.logger`
  import is wrapped in `try/except Exception`). Replicating the same
  `try/except` pattern at module top for `SCIPY_SIGNAL_AVAILABLE`/
  `_scipy_find_peaks` costs nothing new.
- **`pmf_probability` is claimed even though it isn't in the prompt's
  explicit function list, because leaving it behind is structurally
  worse.** It's called *from inside* `js_divergence_1d`'s own body
  (`analyze_gareus_mbar.py:4074-4075`) and `_pmf_distribution_mean_std`'s
  body (`:5636`) — both of which this plan relocates. Leaving
  `pmf_probability` in the script would force the *package* to import
  *back* from the script for something two of its own newly-relocated
  functions need — exactly the backwards direction Plan A1 frames as a
  temporary, script-importing-package transition, never the other way.
  Relocating it also serves other callers unmodified via the same
  `analyze_gareus_mbar.pmf_probability` re-export; its other call sites
  (`:2081`, `:2139`, `:4960`, `:5024`) are all in convergence-orchestration
  code (Plan A6's domain) that consumes it as a normalization utility with
  no PMF-construction logic of its own — it does not touch cumulant
  expansion, GaMD boost weighting, or anything else Plan A4 owns. **Flagged
  explicitly for cross-check**: Plan A4 also produces `pmf` dicts
  (`pmf_from_weights`, `_cumulant_expansion`) that `pmf_probability`
  normalizes, so A4 has a plausible adjacent claim; the tie-break argument
  above (it's an internal dependency of two comparison metrics, not a
  PMF-construction step itself) should be read against A4's own reasoning
  before either plan finalizes.
- **`_compute_basin_populations` is claimed alongside `identify_basins_1d`
  for the same reason**: it's the one two-line pure-math function that
  turns `identify_basins_1d`'s output into basin population numbers, has no
  I/O or orchestration content, and its only call site
  (`analyze_gareus_mbar.py:5134`) sits right next to `identify_basins_1d`'s
  own only call site (`:5127`) inside the same convergence-orchestration
  loop. Splitting the pair across two plans would force Plan A6 to import
  from two different places for two lines of code that are always used
  together.
- **`ess` is deliberately *not* claimed for `gareus/diagnostics.py` itself**,
  despite living in this plan's design doc, because it's genuinely
  cross-domain: grep confirms it is never called inside any of
  `solve_mbar`/`solve_mbar_numba`/`solve_mbar_numba_anderson`/
  `solve_mbar_sambar*`/`solve_mbar_lbfgs`'s own fixed-point iterations
  (Plan A3's core solvers) — every call site
  (`boost_stats`, the epoch-convergence loop, `run_pmf_and_gamd_boost_report`,
  and the final `analyze()` summary's `base_ess`) is post-hoc reporting
  *after* a solve or *after* a reweighting, on an already-produced weight
  array. Both Plan A3 (MBAR-solve diagnostics) and this plan (GaMD-boost /
  convergence diagnostics) need it, so it belongs at the shared leaf level
  both already import from for other primitives, not nested one level
  deeper inside either domain's own module. Note for the A3 cross-check:
  every real call site is `ess(norm_logw(...))`, and `norm_logw` sits in
  the same `logsumexp`/`logsumexp_axis0_finite` cluster at
  `analyze_gareus_mbar.py:2560-2570` that Plan A3 almost certainly claims —
  this plan deliberately separates `ess` from that cluster since `ess`
  itself never calls `norm_logw` and has no other dependency on it, but the
  two are always used together at call sites, so A3 should know the pair is
  being split on purpose, not by oversight.
- **The histogram-overlap reconciliation is explicitly this plan's job**,
  per Plan A1's own Out Of Scope section ("Plan A5: relocate statistical
  diagnostics, reconciling with `gareus/diagnostics.py`'s existing
  `_hist_overlap_np` family"), and the fix is low-risk because the two
  implementations are provably almost identical (same pad formula, same
  `max(8, int(bins))` clamp, same auto-range fallback, same
  normalize-then-`sum(min(...))` finish) with exactly one behavioral knob
  differing between them.

## Alternatives Considered

### New `gareus/mbar_analysis/diagnostics.py` module instead of extending `gareus/diagnostics.py`

Rejected. `gareus/diagnostics.py` already exists with the exact right name,
real production content in the same conceptual neighborhood, an
established `__all__` convention, and — per the import-graph check above —
no circular-import obstacle to extending it. Creating a second,
similarly-named diagnostics module would immediately re-introduce the kind
of confusing multi-location duplication this whole modularization effort
exists to eliminate (see Plan A1's own framing of the histogram-overlap
mess as the cautionary example). Per the task's own default expectation,
this alternative would need a real reason to win, and investigation found
none.

### Reconcile `overlap_matrix` (A3's K×K window-pair matrix) here too, since it's "the same statistic"

Rejected — explicitly out of scope per the task assignment: `overlap_matrix`
is a parallel agent's (Plan A3's) function, vectorized very differently
(bincount over `K` windows and `B` bins at once, producing a full K×K
matrix, not a pairwise scalar), and reconciling three implementations at
once (`overlap_matrix`, `_hist_overlap_np`, `_adaptive_hist_overlap`) in one
plan risks exactly the kind of unreviewable, high-blast-radius change this
6-plan sequence is structured to avoid. This plan touches only the two
diagnostics-level pairwise implementations explicitly named in its own
scope.

### Also reconcile `gareus/logger.py`'s `_hist_overlap` into the same delegation

Considered and rejected for now, coexistence with a documented reason
instead: `_hist_overlap` (`bins=24` default, list-typed inputs) has a real
semantic difference beyond its bin count — **no padding term and no
auto-range fallback** (it uses `lo`/`hi` exactly as given, no
`math.isfinite` guard, no ±2%-or-0.1 padding). Folding it into
`_adaptive_hist_overlap`'s auto-ranging/padded behavior would change
`gareus/logger.py`'s real-time per-window overlap numbers during a live
adaptive-feedback run, not just relocate code — a behavioral change with
no request behind it and its own verification burden. Left untouched.

### Rename `ess` to `_ess` for `math_helpers.py`'s naming convention

Rejected. `analyze_gareus_mbar.ess` (soon `gareus.math_helpers.ess`, via
the same identity-preserving import swap as everything else in this plan)
is exercised directly, by its exact public name, in three real-MD
physics-oracle test files (`tests/test_thermodynamic_validity_2d.py`,
`tests/test_thermodynamic_validity_2d_rough.py`,
`tests/test_thermodynamic_validity_real_md.py`, all calling `agm.ess(w)`).
Renaming it would be a gratuitous breaking change to already-passing,
scientifically-validated tests for a purely cosmetic convention match.
Documented as an explicit, one-function exception instead.

### Give `_adaptive_hist_overlap` a `min_samples` default of 5 (match `_hist_overlap_np`'s stricter floor) instead of 0

Rejected. `adaptive_feedback.py` calls `_adaptive_hist_overlap` from many
live-run call sites, none passing `min_samples`, relying on today's
effectively-zero floor (only the fully-empty-array case is guarded). Making
5 the default would silently change real adaptive-feedback overlap
decisions mid-run. Default `0` (via `max(1, min_samples)`, so the
zero-array guard is never lost) preserves every existing call site
exactly; only `_hist_overlap_np`'s explicit `min_samples=5` opts into the
stricter floor it always had.

## Architecture

### Files touched

```
gareus/diagnostics.py       MODIFY  (add 8 relocated functions -- 5 public, 3 private --
                                      plus a module-level optional scipy.signal import;
                                      _hist_overlap_np body replaced with a delegating call;
                                      __all__ and module docstring extended)
gareus/math_helpers.py      MODIFY  (add ess(); _adaptive_hist_overlap gains min_samples
                                      param; __all__ extended)
analyze_gareus_mbar.py      MODIFY  (8 function bodies + the ess() def + the now-dead
                                      scipy.signal try/except block replaced with 3 import
                                      statements; zero changes to any call site)
```

No new files, no `pyproject.toml` change (this plan adds no new external
dependency — `scipy` is already declared).

### `gareus/diagnostics.py` (new additions, appended after `validate_us_mbar_inputs`)

Module-top addition (after `import numpy as np`):

```python
try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    SCIPY_SIGNAL_AVAILABLE = True
except Exception:  # scipy is optional; identify_basins_1d has a pure-Python fallback.
    _scipy_find_peaks = None
    SCIPY_SIGNAL_AVAILABLE = False

from .math_helpers import _adaptive_hist_overlap
```

`_hist_overlap_np` body replaced with a delegating call (signature
unchanged):

```python
def _hist_overlap_np(a: np.ndarray, b: np.ndarray, lo: float, hi: float, bins: int = 80) -> float:
    """Estimate the histogram overlap between two distributions.

    Delegates to gareus.math_helpers._adaptive_hist_overlap -- this
    module's own pre-existing 5-finite-sample floor is preserved via
    min_samples=5, so validate_us_mbar_inputs's exact prior behavior is
    unchanged: a window pair with fewer than 5 finite samples on either
    side reads as NaN, not a statistically meaningless finite number.
    """
    return _adaptive_hist_overlap(a, b, lo, hi, bins=bins, min_samples=5)
```

New public functions (verbatim bodies from `analyze_gareus_mbar.py`, added
to `__all__`): `pmf_probability`, `js_divergence_1d`, `pmf_rmse_1d`,
`barrier_error_1d`, `identify_basins_1d`.

New private functions (unlisted in `__all__`, matching
`_sample_counts_by_window`'s existing convention): `_compute_basin_populations`,
`_weighted_mean_std`, `_pmf_distribution_mean_std`.

Exact function bodies are reproduced unchanged (only their `analyze_gareus_mbar.py`
line numbers move) in the implementation plan's Task steps.

### `gareus/math_helpers.py` (new/modified)

```python
def _adaptive_hist_overlap(values_a, values_b, lo, hi, bins: int = 80, min_samples: int = 0) -> float:
    ...  # unchanged body, except:
    floor = max(1, int(min_samples))
    if a.size < floor or b.size < floor:
        return float("nan")
    ...  # rest unchanged


def ess(w: np.ndarray) -> float:
    """Effective sample size (Kish estimator) from importance weights.

    Named without a leading underscore -- a documented exception to this
    module's convention -- because it is already public API on
    analyze_gareus_mbar (agm.ess), exercised directly by three real-MD
    physics-oracle tests.
    """
    w = np.asarray(w, dtype=np.float64)
    w = w[np.isfinite(w) & (w >= 0)]
    if w.size == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return 0.0 if s2 <= 0 else s1 * s1 / s2
```

### `analyze_gareus_mbar.py` (relevant lines)

Immediately after the Plan-A1-added
`from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K` line (this plan
requires Plan A1 to have landed first — see Global Constraints in the
implementation plan):

```python
from gareus.diagnostics import (
    pmf_probability, js_divergence_1d, pmf_rmse_1d, barrier_error_1d,
    identify_basins_1d, _compute_basin_populations,
    _weighted_mean_std, _pmf_distribution_mean_std,
)
from gareus.math_helpers import ess
```

The eight function *definitions* at their original locations
(`ess` at `:2572`, `pmf_probability`/`js_divergence_1d`/`pmf_rmse_1d`/
`barrier_error_1d`/`identify_basins_1d`/`_compute_basin_populations` at
`:4066-4165`, `_weighted_mean_std`/`_pmf_distribution_mean_std` at
`:5623-5646`) are deleted, along with the now-dead
`scipy.signal.find_peaks` optional-import block (`:36-41` — its only two
consumers, both inside `identify_basins_1d`, move with the function).
Every one of the ~15 existing call sites across the file (epoch
convergence, frame-count convergence, `boost_stats`,
`run_pmf_and_gamd_boost_report`, the Rg-PMF analysis, the final `analyze()`
summary) is unchanged: same names, same objects, same values.

## Error Handling

Nothing new: every relocated function is already defensive against
non-finite/empty/mismatched-length input (NaN fallbacks, size-mismatch
truncation to the shorter array, `np.errstate` guards around `log`) and
that behavior moves unchanged. The one real behavioral surface this plan
touches — `_adaptive_hist_overlap`'s new `min_samples` parameter — is
additive with a default that reproduces today's behavior exactly (verified
directly, see Problem section), so no existing caller's error/NaN behavior
changes.

## Testing

- **Relocation identity, not just value equality**, for every symbol this
  plan moves (mirrors Plan A1's `is`-identity tests): e.g.
  `analyze_gareus_mbar.js_divergence_1d is gareus.diagnostics.js_divergence_1d`
  and `analyze_gareus_mbar.ess is gareus.math_helpers.ess` — proves the
  import actually replaced the local definition, not just produced a
  same-valued shadow.
- **Known-answer values**, verified directly against the current
  (pre-relocation) implementation before writing them into tests, so none
  are hand-derived-and-hoped:
  - `js_divergence_1d(P, P) == 0.0`; `js_divergence_1d([1,0], [0,1]) == log(2) ≈ 0.6931471805599453`.
  - `pmf_rmse_1d(F, F+2.0, P, P) == 2.0` exactly (constant-offset PMF, uniform positive probability).
  - `barrier_error_1d([5.0], [3.0], [1.0], [1.0]) == 2.0`.
  - `_weighted_mean_std([1,2,3], [1,1,1]) == (2.0, sqrt(2/3))`; `_pmf_distribution_mean_std({'cv_A':[1,2,3],'prob':[1,1,1]})` gives the identical pair (uniform weights and uniform probability agree, as they must).
  - `identify_basins_1d` on a symmetric double well (`cv_A=0..6`, `F=[4,2,0,3,0,2,4]`, `min_depth_kcal=1.0`) returns exactly 2 basins: `{left_bin:0, right_bin:3, center_cv_A:2.0, min_F_kcal:0.0}` and `{left_bin:4, right_bin:6, center_cv_A:4.0, min_F_kcal:0.0}` — verified directly, and also asserts the docstring's own partition contract (`left_bin`/`right_bin` cover `[0, n-1]` with no gap or overlap).
  - `_compute_basin_populations` on that same basin pair with `prob=[0.1]*6+[0.4]` gives `[0.4, 0.6]` (sums to 1.0).
  - `ess(np.ones(5)) == 5.0`; `ess([1,0,0,0]) == 1.0`.
- **Histogram-overlap reconciliation regression**: a 3-finite-sample fixture
  must still return NaN through `gareus.diagnostics._hist_overlap_np` (its
  pre-existing 5-sample floor, now expressed via `min_samples=5`) and a
  finite value (`1.0` for identical samples) through
  `gareus.math_helpers._adaptive_hist_overlap` (default `min_samples=0`,
  unchanged) — proving the merge unified the *implementation* without
  unifying the two call sites' *behavior*. A ≥50-sample fixture must give
  bit-identical output through both entry points (they now share one code
  path).
- **Existing test suite must be unaffected** (not modified, still passing):
  `tests/test_pmf_gamd_nan_bin_handling.py` (imports
  `js_divergence_1d`/`pmf_rmse_1d`/`barrier_error_1d`/`pmf_probability`
  from `analyze_gareus_mbar` directly, alongside A4-owned
  `_cumulant_expansion`/`pmf_from_weights`/`norm_logw` which this plan does
  not touch), `tests/test_thermodynamic_validity_2d.py`,
  `tests/test_thermodynamic_validity_2d_rough.py`,
  `tests/test_thermodynamic_validity_real_md.py` (all call `agm.ess(w)`
  directly), `tests/test_2d_overlap_neighbor_graph.py` (imports
  `_hist_overlap_np` from `gareus.diagnostics` and calls it directly with
  real overlap fixtures), `tests/test_package_smoke.py`
  (`test_adaptive_hist_overlap_accepts_numpy_arrays`, calls
  `_adaptive_hist_overlap` via `gareus.adaptive_feedback`'s re-export with
  no `min_samples` argument), `tests/test_diagnostics.py`
  (`compute_gamd_reweighting_diagnostics` — untouched; its own inline ESS
  formula is a *fourth*, pre-existing near-duplicate of the Kish estimator
  this plan does not touch or unify — see Out Of Scope).

Verification commands (once implemented):

```bash
pytest -q tests/test_diagnostics_pmf_metrics.py tests/test_diagnostics_basins.py \
          tests/test_diagnostics_mean_std.py tests/test_math_helpers_ess.py \
          tests/test_hist_overlap_reconciliation.py
pytest -q tests/test_pmf_gamd_nan_bin_handling.py tests/test_diagnostics.py \
          tests/test_2d_overlap_neighbor_graph.py tests/test_package_smoke.py \
          tests/test_thermodynamic_validity_2d.py tests/test_thermodynamic_validity_2d_rough.py
python -m py_compile analyze_gareus_mbar.py gareus/diagnostics.py gareus/math_helpers.py
```

## Out Of Scope

- **`_plot_basin_population_convergence`** (`analyze_gareus_mbar.py:4746`) —
  investigated and deliberately **not** claimed. It's called from inside
  `run_observable_pmf_convergence` (`:4910`), is matplotlib-based (writes
  PNGs, appends to a `warnings` list passed down from convergence
  orchestration), and its inputs (`basin_pop_rows` keyed by `frac_total`/
  `checkpoint_index`, `file_prefix`, `out`) are all convergence-sweep
  bookkeeping, not pure math on already-computed arrays. This is Plan A6's
  "convergence analysis and plotting/reporting" territory per Plan A1's own
  scope split, not this plan's. **Cross-domain dependency for A6 to note**:
  A6's `run_observable_pmf_convergence` will need to import
  `identify_basins_1d`, `_compute_basin_populations`, `js_divergence_1d`,
  `pmf_rmse_1d`, `barrier_error_1d`, and `pmf_probability` from
  `gareus.diagnostics` once this plan lands.
- **`overlap_matrix`** (Plan A3) and **`gareus/logger.py`'s `_hist_overlap`**
  (left coexisting, see Alternatives Considered) — not touched.
- **`solve_mbar`/`solve_mbar_numba*`/`solve_mbar_sambar*`/`solve_mbar_lbfgs`**
  and the `logsumexp`/`logsumexp_axis0_finite`/`norm_logw` cluster
  (Plan A3) — not touched, despite `ess` always being called as
  `ess(norm_logw(...))` at every real call site (see Why This Approach).
- **`boost_stats`** (`analyze_gareus_mbar.py:7094`) and **`_window_moments`**
  (`:7104`) — investigated, deliberately not claimed. Both characterize the
  GaMD boost-energy distribution's own shape (mean/std/skew/kurtosis/
  anharmonicity), which is Plan A4's PMF/GaMD domain, not this plan's
  PMF-vs-reference comparison and basin-structure domain — the two are
  conceptually adjacent (both are "distribution statistics") but
  characterize different things for different purposes. `boost_stats` does
  call `ess` internally (line `:7101`); it will pick up the relocated
  `gareus.math_helpers.ess` automatically through this plan's import swap
  without needing to move itself.
- **`compute_gamd_reweighting_diagnostics`'s own inline ESS formula**
  (`gareus/diagnostics.py:186-189`, `:216-222`) — a fourth,
  independently-written copy of the same `sum(w)^2/sum(w^2)` estimator,
  already living in this plan's target file. Noted as seen, not silently
  missed: it differs from `ess()` in one edge case (returns NaN rather than
  `ess()`'s `0.0` when every weight is non-positive) and is already covered
  by `tests/test_diagnostics.py`'s passing tests, which don't exercise that
  edge case either way. Not refactored here — doing so would mean editing
  an already-shipped, already-tested function that was never named in this
  plan's scope, for a marginal-edge-case cleanup with its own separate risk
  profile. Flagged for a future decision, not fixed.
- **`_window_cv_mean_std`** (`analyze_gareus_mbar.py:4022`) and
  **`_sample_counts_by_window`** (`gareus/diagnostics.py:70`) — investigated
  for overlap with `_weighted_mean_std`/`_pmf_distribution_mean_std`; none
  found. `_window_cv_mean_std` computes unweighted per-window CV mean/std
  via the same `bincount`-over-linear-index idiom as `overlap_matrix` (A3's
  territory) for `window_diagnostics.csv`; `_sample_counts_by_window`
  counts raw samples per window index. Neither takes arbitrary importance
  weights or a PMF probability array — different inputs, different
  consumers, no real duplication with this plan's functions.
- Any change to `analyze_gareus_mbar.py`'s argument parsing, output format,
  or numeric behavior beyond where these ten names are *defined* — matching
  every other plan in this sequence.
