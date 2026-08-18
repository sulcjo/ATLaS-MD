# MBAR Analysis Modularization — Plan A3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the real correctness bug in `gareus/query.py`'s `reconstruct_bias_matrix` (it fabricates a zero secondary-CV deviation for samples with a missing/NaN secondary CV value, instead of correctly excluding them), then relocate `analyze_gareus_mbar.py`'s MBAR solver family and bias-reconstruction helpers into the `gareus` package as Plan A3 of the 6-plan `analyze_gareus_mbar.py` modularization sequence.

**Architecture:** `gareus/query.py`'s `reconstruct_bias_matrix` is fixed in place (same public signature) to become the one canonical bias-reconstruction implementation; `analyze_gareus_mbar.py`'s own `_compute_u_nk_analytical`/`_reconstruct_union_bias_block` relocate into new `gareus/mbar_analysis/bias.py` as thin delegating wrappers around it (their own guard logic was already correct — a prior audit fixed them — so this is pure deduplication, not a behavior change). The MBAR solver family (`solve_mbar` + 4 backends + shared helpers) relocates verbatim into new `gareus/mbar_analysis/solvers.py`. `analyze_gareus_mbar.py` imports every relocated name back by simple `from gareus.mbar_analysis.X import name` (the same pattern Plan A1 established for `KJ_PER_KCAL`), so its ~20+ existing call sites need zero changes beyond the import line. A migration-specific hazard — a `globals()`-mutation block that applies `--sambar-*` CLI overrides by rewriting `analyze_gareus_mbar.py`'s own module dict — is explicitly retargeted to `gareus.mbar_analysis.solvers`'s own namespace, since after the move the solver functions read these names from that module, not from `analyze_gareus_mbar.py`.

**Tech Stack:** Python 3.10+, pytest, NumPy, optional numba/scipy (already-optional dependencies, unchanged).

**Spec:** `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a3-design.md`

## Global Constraints

- **Depends on Plan A1 having already executed**: `gareus/mbar_analysis/__init__.py` and `gareus/mbar_analysis/cli.py` must already exist, and `gareus.units.KJ_PER_KCAL` must already exist. If either is missing, Task 1 Step 3 and Task 3/4 Step 3 will fail immediately with `ImportError`/`ModuleNotFoundError` — run Plan A1 first.
- **`reconstruct_bias_matrix(cv_A, cv2, windows, beta)`'s public signature never changes** — every real caller (`analyze_gareus_mbar.load_parquet`, `gareus.query.export_analysis_arrays_npz`, the `-hh` documented example in `gareus/helptext.py`, all of `tests/test_query.py`) keeps working unchanged.
- **`_compute_u_nk_analytical`, `_parse_epoch_window_map_native_params`, `_epoch_bias_param_vectors`, `_reconstruct_union_bias_block`, and every MBAR-solver-family function keep their exact existing signatures** after relocation — `analyze_gareus_mbar.py`'s own call sites, and `tests/test_bias_reconstruction_nan_handling.py`/`tests/test_union_mbar_per_epoch_bias.py`/`tests/test_masked_logw_subset_pmf.py`/`tests/test_mbar_lbfgs_convergence.py` (which import from `analyze_gareus_mbar`, not the new modules directly), must all keep passing unmodified.
- **No change to `solve_mbar`'s numerical algorithm, tolerances, or backend-selection heuristics** — this is a pure relocation of the solver family, not a redesign.
- **The `--sambar-*`/`--mbar-anderson-history`/`--mbar-backend` CLI flags must keep functionally working after the move** — Task 4 explicitly retargets the `globals()`-override block and adds a regression test proving it, since nothing currently tests this mechanism (`grep -rn "SAMBAR_EPOCHS\|sambar_epochs\|mbar_anderson_history" tests/` returns nothing today).
- **CORRECTION — `ess` does NOT move with this plan; it is claimed by Plan A5, not A3.** An earlier draft of this plan listed `ess` (`analyze_gareus_mbar.py:2572`) among the solver-family helpers relocated into `gareus/mbar_analysis/solvers.py`. That is wrong: `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a5-design.md` explicitly claims it for `gareus/math_helpers.py` (the package's zero-`gareus`-dependency leaf module), and `...-a5.md`'s Task 4 relocates it there with its own identity test (`agm.ess is gareus.math_helpers.ess`). Verified directly against the source: **no `solve_mbar*`/`logsumexp*`/`_anderson_step` body calls `ess` at all** — its only call sites (`:2131`, `:2134`, `:7101`, `:9512`, `:9677`) are all post-hoc reporting after a solve or a reweighting, none inside this plan's relocated functions. So this plan simply leaves `ess`'s definition untouched in `analyze_gareus_mbar.py` for A5 to relocate: do not extract it into `solvers.py`, do not list it in `solvers.py`'s `__all__`, do not delete it from `analyze_gareus_mbar.py`, and do not add it to this plan's re-import block. This also keeps A3 and A5 order-independent — neither has to land before the other. `norm_logw` (which `ess` is always called *with* at every call site, but never *by*) is unaffected and still moves here.
- **CORRECTION — `_is_usable_for_mbar` and `_merge_missing_usable_states` do NOT move with this plan; they are claimed by Plan A2, not A3.** An earlier draft of this plan listed both among the helpers relocated into `gareus/mbar_analysis/bias.py` (function bodies, `__all__` entries, re-import block, identity-test parametrize list). That is wrong, and it was a genuine double-claim: `docs/superpowers/plans/2026-08-13-mbar-analysis-modularization-a2.md`'s Task 7 already relocates both into `gareus/mbar_analysis/loaders_union_parquet.py` with their full bodies, its own re-import block (`from gareus.mbar_analysis.loaders_union_parquet import _is_usable_for_mbar, _merge_missing_usable_states, load_parquet_adaptive_union`), its own delete-from-script grep anchor, and its own identity test (`_UNION_NAMES`). The cross-plan reconciliation pass confirmed **no caller of either function exists outside A2's union-Parquet loading cluster** — A2's own Task 7 Interfaces line records the same grep result ("internal to this module only"), and `_merge_missing_usable_states`'s only caller is `load_parquet_adaptive_union`, which is A2's. So this plan simply leaves both definitions untouched for A2 to relocate: do not extract them into `bias.py`, do not list them in `bias.py`'s `__all__`, do not delete them from `analyze_gareus_mbar.py`, and do not add them to this plan's re-import block or identity-test list. `tests/test_final_registry_merge.py` (which imports both `from analyze_gareus_mbar`) is therefore A2's regression surface, not this plan's.
- **CORRECTION — `_parse_epoch_window_map_native_params` DOES move with this plan** (Task 3), resolving the `UNRESOLVED OWNERSHIP` gap an earlier cross-plan reconciliation pass flagged. Neither A2 nor A3 previously gave it a real relocation task: A2 explicitly defers it ("stay behind in the script, Plan A3's"), and A3's Task 3 explicitly excluded it — leaving it owned by nobody, which would have forced `loaders_union_parquet.py` to keep a permanent lazy `from analyze_gareus_mbar import _parse_epoch_window_map_native_params` (package importing the retiring script) that A6e's retirement step would have to resolve anyway. It belongs here: it parses `epoch_window_map.csv` rows into the exact `{state_id: {primary_center, primary_k, secondary_center, secondary_k}}` dict that `_epoch_bias_param_vectors` (already this task's) consumes, and the two are always used together (`analyze_gareus_mbar.py:1611` and its single call site at `:1707`, inside A2's `_load_epoch_task`). A2 needs no change: its lazy `from analyze_gareus_mbar import _parse_epoch_window_map_native_params` and its accompanying "when Plan A3 relocates this function, update the module path in this line" comment are both still accurate — A2 lands first and the script still owns the definition at A2's execution time; this plan is what makes that comment's instruction actionable, repointing it to `gareus.mbar_analysis.bias`.
- Re-confirm every exact current line range with `grep -n` immediately before editing `analyze_gareus_mbar.py` in every task below — this file has been edited many times this session; line numbers shift.
- **"Identical" output for the relocated bias-math wrappers (Task 3) means identical within floating-point tolerance (`np.testing.assert_allclose`/`pytest.approx`), never bitwise (`np.array_equal`).** The pre-relocation originals associate the `beta*KJ_PER_KCAL` multiplication differently from the unified `reconstruct_bias_matrix` (one precomputes a combined scale factor and multiplies once; the other multiplies per-term; the unified version accumulates in kcal and applies `beta` once at the end) — floating-point arithmetic is not associative, so results differ at the ~1e-16-relative level. This is expected and is not a formula change; do not "fix" it by reordering the arithmetic to force bitwise equality.
- **`_compute_u_nk_analytical`'s three-clause window-side guard (`isfinite(center2) and isfinite(k2) and k2 > 0`) and `_reconstruct_union_bias_block`'s two-clause guard (`isfinite(center2) and k2 > 0`) are equivalent, not a discrepancy** — `NaN > 0` is always `False` in both Python and NumPy, so a NaN `k2` already fails the `k2 > 0` clause without needing a separate `isfinite(k2)` check. The unified `reconstruct_bias_matrix` (Task 1) uses the explicit three-clause form.
- **Numba thread-count capping is entry-point-dependent, not code-dependent, and this plan does not change that.** `analyze_gareus_mbar.py`'s top-of-file `NUMEXPR_NUM_THREADS`/`NUMBA_NUM_THREADS` environment-variable cap (lines 3-9) only takes effect when `analyze_gareus_mbar.py` itself is imported/run first (it imports `gareus.mbar_analysis.solvers`, which imports `numba`, only after those env vars are set). Anything that imports `gareus.mbar_analysis.solvers` directly — including this plan's own `tests/test_mbar_analysis_solvers_module.py` — does not get that cap and may see numba pick a different default thread count. This affects performance only, never numerical results; not fixed here (see design spec's Out Of Scope).

---

### Task 1: Fix `gareus/query.py`'s `reconstruct_bias_matrix` NaN-guard bug

**Files:**
- Modify: `gareus/query.py` (imports + the `reconstruct_bias_matrix` function body + module docstring)
- Test: `tests/test_query_reconstruct_bias_matrix_nan_guard.py` (new file)

**Interfaces:**
- Produces: a fixed `gareus.query.reconstruct_bias_matrix(cv_A, cv2, windows, beta) -> np.ndarray` — unchanged signature, corrected two-guard behavior. Consumed unchanged by `analyze_gareus_mbar.load_parquet`, `gareus.query.export_analysis_arrays_npz`, and (via Task 3) `gareus.mbar_analysis.bias`'s two delegating wrappers.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_query_reconstruct_bias_matrix_nan_guard.py`:

```python
"""Regression tests for a NaN-guard bug in gareus.query.reconstruct_bias_matrix,
found by direct comparison against analyze_gareus_mbar.py's own two
bias-reconstruction copies (_compute_u_nk_analytical,
_reconstruct_union_bias_block), which a prior audit this session already
fixed for exactly this bug class -- see tests/test_bias_reconstruction_nan_handling.py.

Bug (window-side): the `if cv2 is not None and "center2" in w and "k2" in w`
guard only checked dict-key *presence*, never that the values were finite and
k2 > 0. A window with a real "center2"/"k2" key holding NaN (or a
non-positive k2) was completely unguarded and could NaN-poison, or wrongly
apply, a secondary term that should have been skipped entirely.

Bug (sample-side, the primary one): for a window that DOES have a real
secondary restraint, `d2 = np.where(np.isfinite(c2), c2-center2, 0.0)`
fabricated a deviation of exactly 0.0 -- "assume this sample was perfectly
on-target" -- for any sample whose secondary CV was never measured (NaN),
instead of letting NaN propagate so downstream code (e.g.
analyze_gareus_mbar.py's clean()) can exclude just that sample. This
silently kept a should-be-excluded sample in the pool with a fabricated,
too-low bias energy.

Fixed: window-side guard now checks isfinite(center2) and isfinite(k2) and
k2 > 0 before adding any secondary term at all; the np.where(..., 0.0)
sample-side fallback is removed so NaN propagates correctly.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("gareus.query")

from gareus.query import reconstruct_bias_matrix


# --- window-side guard: a window with no REAL secondary restraint must
#     never poison or wrongly restrain, regardless of what the keys hold ---

def test_nan_k2_produces_finite_cv1_only_bias():
    cv1 = np.array([0.0, 1.0, 2.0])
    cv2 = np.array([5.0, -3.0, np.nan])  # secondary values irrelevant/unmeasured
    windows = [{"center1": 0.0, "k1": 10.0, "center2": -1.0, "k2": np.nan}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1 = cv1 - 0.0
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(nk[:, 0], expected)


def test_zero_k2_produces_finite_cv1_only_bias():
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([100.0, -100.0])  # would blow up if the secondary term applied
    windows = [{"center1": 0.0, "k1": 10.0, "center2": 0.0, "k2": 0.0}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1 = cv1 - 0.0
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(nk[:, 0], expected)


def test_negative_k2_produces_finite_cv1_only_bias():
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([2.0, -2.0])
    windows = [{"center1": 0.0, "k1": 10.0, "center2": 0.0, "k2": -5.0}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1 = cv1 - 0.0
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(nk[:, 0], expected)


def test_nan_center2_with_positive_finite_k2_produces_finite_cv1_only_bias():
    """Deliberate edge case: a real, positive k2 but a NaN/missing center is
    the OPPOSITE half of the guard from the k2 checks above -- both isfinite
    checks are required, neither alone is sufficient.
    """
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([5.0, -3.0])  # finite -- would previously NaN-poison via c2-NaN
    windows = [{"center1": 0.0, "k1": 10.0, "center2": np.nan, "k2": 11.87}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1 = cv1 - 0.0
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(nk[:, 0], expected)


# --- positive control: a real secondary restraint must still apply ---------

def test_real_secondary_restraint_still_applies_correctly():
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([-2.0, -2.0])
    windows = [{"center1": 0.0, "k1": 10.0, "center2": -2.1337, "k2": 11.87}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    d1 = cv1 - 0.0
    d2 = cv2 - (-2.1337)
    expected = (0.4 * 4.184 * 0.5 * 10.0 * d1 ** 2
                + 0.4 * 4.184 * 0.5 * 11.87 * d2 ** 2)
    np.testing.assert_allclose(nk[:, 0], expected)


# --- sample-side exclusion: the primary bug ---------------------------------

def test_nan_cv2_sample_under_restrained_window_produces_nan_for_that_sample_only():
    cv1 = np.array([0.0, 0.0])
    cv2 = np.array([np.nan, -2.0])  # sample 0 unmeasured, sample 1 valid
    windows = [{"center1": 0.0, "k1": 10.0, "center2": -2.1337, "k2": 11.87}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert math.isnan(nk[0, 0])
    assert math.isfinite(nk[1, 0])
    d1 = cv1[1] - 0.0
    d2 = cv2[1] - (-2.1337)
    expected = 0.4 * 4.184 * 0.5 * 10.0 * d1 * d1 + 0.4 * 4.184 * 0.5 * 11.87 * d2 * d2
    assert nk[1, 0] == pytest.approx(expected)


def test_all_nan_cv2_under_restrained_window_yields_all_nan_column():
    cv1 = np.array([0.0, 0.1, -0.1])
    cv2 = np.array([np.nan, np.nan, np.nan])
    windows = [{"center1": 0.0, "k1": 10.0, "center2": -2.1337, "k2": 11.87}]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isnan(nk[:, 0]))


def test_mixed_windows_one_restrained_one_not_isolated_correctly():
    """A no-restraint window's NaN k2 must not affect another window's
    column in the same call (regression against a cross-window broadcast bug)."""
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([-2.0, 3.0])
    windows = [
        {"center1": 0.0, "k1": 10.0, "center2": np.nan, "k2": np.nan},
        {"center1": 0.5, "k1": 20.0, "center2": 1.0, "k2": 5.0},
    ]

    nk = reconstruct_bias_matrix(cv1, cv2, windows, beta=0.4)

    assert np.all(np.isfinite(nk))
    d1_0 = cv1 - 0.0
    expected_col0 = 0.4 * 4.184 * 0.5 * 10.0 * d1_0 ** 2
    np.testing.assert_allclose(nk[:, 0], expected_col0)

    d1_1 = cv1 - 0.5
    d2_1 = cv2 - 1.0
    expected_col1 = (0.4 * 4.184 * 0.5 * 20.0 * d1_1 ** 2
                     + 0.4 * 4.184 * 0.5 * 5.0 * d2_1 ** 2)
    np.testing.assert_allclose(nk[:, 1], expected_col1)


def test_1d_window_with_no_center2_k2_keys_at_all_is_unaffected():
    """Baseline: a window dict with no secondary keys at all (the ordinary 1D
    case) must be completely untouched by the guard changes."""
    cv1 = np.array([0.1, 0.2, 0.3])
    windows = [{"center1": 0.15, "k1": 100.0}]

    nk = reconstruct_bias_matrix(cv1, None, windows, beta=1.0 / (8.314462618e-3 * 300.0))

    beta = 1.0 / (8.314462618e-3 * 300.0)
    expected = beta * 4.184 * 0.5 * 100.0 * (cv1 - 0.15) ** 2
    np.testing.assert_allclose(nk[:, 0], expected, rtol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_query_reconstruct_bias_matrix_nan_guard.py -v`
Expected: `test_nan_k2_produces_finite_cv1_only_bias`,
`test_nan_center2_with_positive_finite_k2_produces_finite_cv1_only_bias`,
`test_nan_cv2_sample_under_restrained_window_produces_nan_for_that_sample_only`
(and its all-NaN sibling) FAIL — the current buggy implementation either
NaN-poisons a should-be-CV1-only column or fabricates a finite value where
NaN is expected. `test_negative_k2_...`, `test_real_secondary_restraint_...`,
and `test_1d_window_with_no_center2_k2_keys_at_all_...` should already PASS
(they don't exercise either bug) — confirms the new test file isn't
accidentally testing something already broken elsewhere.

- [ ] **Step 3: Write the fix**

First re-confirm current imports and function location:

Run: `grep -n "^from __future__\|^import\|^from \.\|^def reconstruct_bias_matrix" gareus/query.py`

Add `import math` and `from .units import KJ_PER_KCAL` to the import block
(after `from .store import SegmentRegistry`):

```python
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .io import read_json_file
from .store import SegmentRegistry
from .units import KJ_PER_KCAL
```

Replace the full body of `reconstruct_bias_matrix` (and its docstring) with:

```python
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

Also update the module docstring's formula block (top of `gareus/query.py`,
the `U_k(cv) = ...` comment) to note the two guards, for documentation
consistency:

```python
"""
Parquet-based data loading and MBAR bias reconstruction for GAREUS.

The N×K bias matrix (umbrella_reduced_bias_nk) is reconstructed analytically
from stored CV values and window parameters rather than being persisted:

    U_k(cv) = 0.5 * k1_k * (cv1 - center1_k)^2  [kcal/mol]
            + 0.5 * k2_k * (cv2 - center2_k)^2  [kcal/mol, only when window k
                                                  has isfinite(center2_k) and
                                                  isfinite(k2_k) and k2_k > 0]
    reduced_bias[n,k] = beta * KJ_PER_KCAL * U_k(cv[n])

A sample whose own cv2 is non-finite gets NaN for any window that DOES
restrain CV2 (exclusion-by-propagation), rather than a fabricated zero
deviation -- see reconstruct_bias_matrix's docstring.

This is 10-100x smaller than storing the full matrix.
"""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_query_reconstruct_bias_matrix_nan_guard.py -v`
Expected: `9 passed`. This file's `pytest.importorskip("gareus.query")`
means an import failure reports as `9 skipped`, not a failure — a skip is
NOT an acceptable outcome here; if you see it, `gareus.query` failed to
import (most likely: Plan A1 hasn't landed yet and `from .units import
KJ_PER_KCAL` — added in this task's own Step 3 — can't resolve) and must be
investigated before continuing.

- [ ] **Step 5: Confirm existing `reconstruct_bias_matrix`/`export_analysis_arrays_npz` tests are unaffected**

Run: `pytest tests/test_query.py -v -k "reconstruct_bias_matrix or export_npz"`
Expected: PASS (10 tests) — none of these fixtures exercise a NaN `k2`/`center2`
or a NaN sample `cv2`, so none of their expected values change.

Run: `pytest tests/test_perf_loading_and_solver_memory.py -v -k "load_parquet_narrows"`
Expected: PASS — this test monkeypatches `reconstruct_bias_matrix` entirely
with its own fake implementation, so it is unaffected by this fix.

- [ ] **Step 6: Commit**

```bash
git add gareus/query.py tests/test_query_reconstruct_bias_matrix_nan_guard.py
git commit -m "fix: reconstruct_bias_matrix now excludes (not zeroes) samples with unmeasured secondary CV"
```

---

### Task 2: Verify `validate_analysis_metadata_readiness` tolerates a non-finite bias matrix

**Files:**
- Test only: `tests/test_analysis_readiness_nonfinite_bias_warning.py` (new file)
- No production code changes — this task confirms and pins existing behavior in `gareus/analysis.py` (verified by direct code reading during design: `gareus/analysis.py:195-196` already appends to `warnings`, not `errors`, for this exact case).

**Interfaces:**
- Consumes: `gareus.analysis.validate_analysis_metadata_readiness(out_dir, target_overlap=0.30)` (pre-existing, unchanged).
- Confirms: once Task 1's fix lands, a run with any genuinely-unmeasured secondary CV sample under a 2D window will, for the first time, produce a non-finite `umbrella_reduced_bias_nk` — and this consumer already tolerates that as a warning, not a validation failure. This closes the one open design question from the spec (does the fix need extra masking in `export_analysis_arrays_npz`? No.) with a pinned regression test rather than a point-in-time code read.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis_readiness_nonfinite_bias_warning.py`:

```python
"""Confirms gareus.analysis.validate_analysis_metadata_readiness tolerates a
non-finite umbrella_reduced_bias_nk as a warning, not a validation failure.

This matters because of a fix landing in the same modularization plan:
gareus.query.reconstruct_bias_matrix used to fabricate a zero deviation for
any sample with an unmeasured (NaN) secondary CV under a restrained 2D
window, instead of correctly excluding it via NaN. Once fixed,
export_analysis_arrays_npz (which does NOT row-mask, unlike
analyze_gareus_mbar.py's clean()) can, for the first time, persist NaN
entries in analysis_arrays.npz's umbrella_reduced_bias_nk for a real run with
genuinely-unmeasured secondary-CV samples. This test constructs that npz
directly (bypassing the Parquet/export pipeline, to isolate this consumer's
own tolerance from the reconstruct_bias_matrix fix itself) and confirms
validate_analysis_metadata_readiness surfaces it as an informational warning
rather than an error or a crash.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gareus.analysis")

from gareus.analysis import validate_analysis_metadata_readiness


def _write_minimal_1d_window_csv(out_dir, n_windows: int) -> None:
    with (out_dir / "umbrella_windows.csv").open("w") as f:
        f.write("window,distance_center_A,distance_k_kcal_mol_A2\n")
        for i in range(n_windows):
            f.write(f"{i},{i * 0.5},200.0\n")


def test_nonfinite_bias_matrix_produces_warning_not_error(tmp_path):
    n_samples, n_windows = 6, 2
    cv_a = np.linspace(-1.0, 1.0, n_samples)
    window = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    bias = np.full((n_samples, n_windows), 1.0)
    bias[2, 0] = np.nan  # one sample's reduced bias entry is unmeasured/excluded

    np.savez_compressed(
        tmp_path / "analysis_arrays.npz",
        cv_A=cv_a, window=window, umbrella_reduced_bias_nk=bias,
    )
    _write_minimal_1d_window_csv(tmp_path, n_windows)

    result = validate_analysis_metadata_readiness(tmp_path)

    assert any("umbrella_reduced_bias_nk contains non-finite values" in w
               for w in result["warnings"]), result["warnings"]
    assert not any("umbrella_reduced_bias_nk" in e for e in result["errors"]), result["errors"]


def test_fully_finite_bias_matrix_produces_no_such_warning(tmp_path):
    """Negative control: the warning must not fire when there is nothing
    non-finite -- proves the check above is specific, not a false positive
    that always fires."""
    n_samples, n_windows = 6, 2
    cv_a = np.linspace(-1.0, 1.0, n_samples)
    window = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    bias = np.full((n_samples, n_windows), 1.0)

    np.savez_compressed(
        tmp_path / "analysis_arrays.npz",
        cv_A=cv_a, window=window, umbrella_reduced_bias_nk=bias,
    )
    _write_minimal_1d_window_csv(tmp_path, n_windows)

    result = validate_analysis_metadata_readiness(tmp_path)

    assert not any("umbrella_reduced_bias_nk contains non-finite values" in w
                   for w in result["warnings"]), result["warnings"]
```

- [ ] **Step 2: Run tests to verify their current status**

Run: `pytest tests/test_analysis_readiness_nonfinite_bias_warning.py -v`
Expected: `2 passed` — this task is a **verification-only** task; the
behavior it pins already exists in `gareus/analysis.py` today (confirmed by
direct code reading during design), so no implementation step is needed.
This file's `pytest.importorskip("gareus.analysis")` means an import
failure reports as `2 skipped`, not a failure — treat a skip the same as a
failure and investigate, don't read it as green. If either test
unexpectedly FAILS here, stop and re-read `gareus/analysis.py`'s
window-table/NPZ validation block before continuing — that would mean the
design's premise (this consumer already tolerates NaN as a warning) is
wrong, which changes Task 1's risk assessment materially.

- [ ] **Step 3: (No implementation step — this task only adds test coverage.)**

- [ ] **Step 4: Commit**

```bash
git add tests/test_analysis_readiness_nonfinite_bias_warning.py
git commit -m "test: pin validate_analysis_metadata_readiness's NaN-bias-matrix tolerance ahead of the reconstruct_bias_matrix fix"
```

---

### Task 3: Relocate bias-reconstruction helpers into `gareus/mbar_analysis/bias.py`

**Files:**
- Create: `gareus/mbar_analysis/bias.py`
- Modify: `analyze_gareus_mbar.py` (delete 4 function bodies, add one import block)
- Test: `tests/test_mbar_analysis_bias_module.py` (new file)

**Interfaces:**
- Consumes: `gareus.query.reconstruct_bias_matrix` (Task 1, fixed).
- Produces: `gareus.mbar_analysis.bias._compute_u_nk_analytical(cv1, cv2, union_windows, beta)`, `gareus.mbar_analysis.bias._parse_epoch_window_map_native_params(rows)`, `gareus.mbar_analysis.bias._epoch_bias_param_vectors(native_params, state_ids, global_primary_centers, global_primary_ks, global_sec_centers, global_sec_ks)`, `gareus.mbar_analysis.bias._reconstruct_union_bias_block(cv, cv2, beta, primary_centers, primary_ks, sec_centers, sec_ks)` — all re-exported at `analyze_gareus_mbar.<name>` unchanged, consumed by Plan A2's `load_parquet_adaptive_union`/`_load_epoch_task`/`_augment_with_adaptive_rounds` (once A2 relocates those) exactly as they consume them today. `_is_usable_for_mbar`/`_merge_missing_usable_states` are **not** produced here — Plan A2 owns them (see the CORRECTION in Global Constraints).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_bias_module.py`:

```python
"""Confirms the bias-reconstruction helpers relocated from
analyze_gareus_mbar.py into gareus.mbar_analysis.bias are real delegations
(same object, not a coincidentally-equal reimplementation) and that the two
formula-bearing wrappers (_compute_u_nk_analytical,
_reconstruct_union_bias_block) still produce output identical -- within
floating-point tolerance, NOT bitwise -- to their pre-relocation selves on
the exact fixture data from tests/test_bias_reconstruction_nan_handling.py.
Not bitwise because the pre-relocation originals associate the
multiplication differently (_compute_u_nk_analytical precomputed
scale=beta*KJ_PER_KCAL once; _reconstruct_union_bias_block multiplied
beta*KJ_PER_KCAL into each term separately; the unified
reconstruct_bias_matrix accumulates in kcal and multiplies by beta once at
the end) -- floating-point arithmetic is not associative, so this differs
at the ~1e-16-relative level. Use assert_allclose/pytest.approx below, never
np.array_equal -- proving the delegation to
gareus.query.reconstruct_bias_matrix changed zero behavior for these two
(they were already correct before this relocation)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

import analyze_gareus_mbar as agm
import gareus.mbar_analysis.bias as bias_mod


def test_bias_module_is_importable():
    import gareus.mbar_analysis.bias  # noqa: F401


@pytest.mark.parametrize("name", [
    "_compute_u_nk_analytical",
    "_parse_epoch_window_map_native_params",
    "_epoch_bias_param_vectors",
    "_reconstruct_union_bias_block",
])
def test_analyze_gareus_mbar_reexports_the_same_object(name):
    assert getattr(agm, name) is getattr(bias_mod, name)


def test_compute_u_nk_analytical_matches_pre_relocation_formula():
    beta = 0.4
    cv1 = np.array([0.0, 1.0])
    cv2 = np.array([-2.0, -2.0])
    union_windows = [{
        'primary_center': 0.0, 'primary_k_kcal': 10.0,
        'secondary_cv_center': -2.1337, 'secondary_k_kcal': 11.87,
    }]

    u = agm._compute_u_nk_analytical(cv1, cv2, union_windows, beta)

    d1 = cv1 - 0.0
    d2 = cv2 - (-2.1337)
    expected = (beta * agm.KJ_PER_KCAL * 0.5 * 10.0 * d1 ** 2
                + beta * agm.KJ_PER_KCAL * 0.5 * 11.87 * d2 ** 2)
    np.testing.assert_allclose(u[:, 0], expected)


def test_compute_u_nk_analytical_no_restraint_window_still_finite():
    """Regression: the delegation must preserve the already-correct
    window-side guard (NaN secondary_k_kcal must not poison the row)."""
    beta = 0.4
    cv1 = np.array([0.0, 1.0, 2.0])
    cv2 = np.array([5.0, -3.0, np.nan])
    union_windows = [{
        'primary_center': 0.0, 'primary_k_kcal': 10.0,
        'secondary_cv_center': np.nan, 'secondary_k_kcal': np.nan,
    }]

    u = agm._compute_u_nk_analytical(cv1, cv2, union_windows, beta)

    assert np.all(np.isfinite(u))
    d1 = cv1 - 0.0
    expected = beta * agm.KJ_PER_KCAL * 0.5 * 10.0 * d1 ** 2
    np.testing.assert_allclose(u[:, 0], expected)


def test_reconstruct_union_bias_block_matches_pre_relocation_formula():
    beta = 0.4
    cv = np.array([0.0, 0.0])
    cv2 = np.array([np.nan, -2.0])
    pc = np.array([0.0]); pk = np.array([10.0])
    sc = np.array([-2.1337]); sk = np.array([11.87])

    u = agm._reconstruct_union_bias_block(cv, cv2, beta, pc, pk, sc, sk)

    assert np.isnan(u[0, 0])
    assert np.isfinite(u[1, 0])
    d1 = cv[1] - pc[0]
    d2 = cv2[1] - sc[0]
    expected = beta * agm.KJ_PER_KCAL * 0.5 * pk[0] * d1 * d1 + beta * agm.KJ_PER_KCAL * 0.5 * sk[0] * d2 * d2
    assert u[1, 0] == pytest.approx(expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_bias_module.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.mbar_analysis.bias'`

- [ ] **Step 3: Create `gareus/mbar_analysis/bias.py`**

```python
"""Bias-reconstruction helpers relocated from analyze_gareus_mbar.py.

_compute_u_nk_analytical and _reconstruct_union_bias_block delegate to the
canonical gareus.query.reconstruct_bias_matrix (fixed as part of this same
modularization plan -- see Plan A3's Task 1) rather than duplicating its
formula a third and fourth time. Their own guard logic was already correct
(a prior audit this session fixed both -- see
tests/test_bias_reconstruction_nan_handling.py), so this relocation is a
pure deduplication with zero intended behavior change.

_parse_epoch_window_map_native_params and _epoch_bias_param_vectors are pure
supporting glue for load_parquet_adaptive_union's per-epoch-native bias
reconstruction (see CLAUDE.md's "Union-MBAR bias matrix used a stale global
registry snapshot" entry) and have no formula of their own to reconcile:
the first parses one epoch's own epoch_window_map.csv rows into a
{state_id: {...}} native-params dict, the second consumes exactly that dict.
They are always used together, which is why both live here.

_is_usable_for_mbar / _merge_missing_usable_states deliberately do NOT live
here -- Plan A2 relocates them into
gareus/mbar_analysis/loaders_union_parquet.py, the only cluster that calls
them (see Plan A3's Global Constraints CORRECTION).
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

__all__ = [
    "_compute_u_nk_analytical",
    "_parse_epoch_window_map_native_params",
    "_epoch_bias_param_vectors",
    "_reconstruct_union_bias_block",
]


def _compute_u_nk_analytical(cv1: np.ndarray, cv2: np.ndarray,
                              union_windows: list, beta: float) -> np.ndarray:
    """Compute N×K_union reduced bias matrix analytically from CV values."""
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

    These are the window parameters that were *actually in effect* for that
    specific epoch, as recorded in that epoch's own ``epoch_window_map.csv``
    snapshot at the time it ran — as opposed to whatever a state's row in the
    live/final registry says today. A state's secondary (or even primary)
    center/k can be recentered by later epochs (e.g. the tICA CV2 auto-switch
    in ``gareus/adaptive_production.py``, which overwrites
    ``state.secondary_center`` in place); this dict preserves the
    epoch-specific ground truth so bias energies for that epoch's own samples
    can be reconstructed against what was really applied, not what the state
    looks like now.
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

    Starts from the global (final-registry) arrays — the existing fallback
    behavior — then overrides entries for any state_id this epoch's own
    ``epoch_window_map.csv`` snapshot actually covers. States created in a
    later epoch (absent from this epoch's snapshot) keep the global fallback,
    which is correct: they didn't exist yet, so there is no "native" value to
    prefer, and no sample from this epoch can be assigned to them anyway.
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
```

- [ ] **Step 4: Point `analyze_gareus_mbar.py` at the new module**

Re-confirm current locations:

Run: `grep -n "^def _compute_u_nk_analytical\|^def _is_usable_for_mbar\|^def _merge_missing_usable_states\|^def _parse_epoch_window_map_native_params\|^def _epoch_bias_param_vectors\|^def _reconstruct_union_bias_block" analyze_gareus_mbar.py`

The grep deliberately also anchors `_is_usable_for_mbar`/
`_merge_missing_usable_states` even though they do **not** move — they sit
in the middle of this neighborhood, so you need their exact boundaries to
avoid sweeping them into a deletion.

Delete the full bodies of all 4 relocated functions. In the current source
they form **two** non-adjacent regions, in this order:

1. `_compute_u_nk_analytical` alone (`:1380`, ending just before the next
   top-level `def`).
2. `_parse_epoch_window_map_native_params` (`:1611`) through the end of
   `_reconstruct_union_bias_block` (`:1679`) — contiguous, since
   `_epoch_bias_param_vectors` (`:1648`) sits between them and also moves.

**`_is_usable_for_mbar` (`:1583`) and `_merge_missing_usable_states`
(`:1587`) — sitting immediately *above* `_parse_epoch_window_map_native_params`
— stay behind, untouched. They are Plan A2's** (`gareus/mbar_analysis/
loaders_union_parquet.py`, its Task 7), not part of this relocation; see the
CORRECTION in Global Constraints. Do not delete them, and do not let region 2
start any earlier than `_parse_epoch_window_map_native_params`'s own `def`
line.

`_parse_epoch_window_map_native_params` is relocated by this task alongside
`_epoch_bias_param_vectors` for the reason spelled out in Global
Constraints' second CORRECTION: it parses `epoch_window_map.csv` rows into
exactly the native-params dict `_epoch_bias_param_vectors` consumes, the two
are always used together, and its single call site (`:1707`, inside A2's
`_load_epoch_task`) already reaches it through a lazy import that this plan
repoints to `gareus.mbar_analysis.bias`.

In their place, immediately after the existing
`from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K` import line (added
by Plan A1's Task 2), add:

```python
from gareus.mbar_analysis.bias import (
    _compute_u_nk_analytical,
    _parse_epoch_window_map_native_params,
    _epoch_bias_param_vectors,
    _reconstruct_union_bias_block,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mbar_analysis_bias_module.py -v`
Expected: `8 passed` (1 importability + 4 parametrized identity cases + 3
formula tests — count them off Step 1's file, don't assume: the parametrize
list is 4, not 5, because `_is_usable_for_mbar`/`_merge_missing_usable_states`
are Plan A2's, per the CORRECTION in Global Constraints). This file's
`pytest.importorskip("analyze_gareus_mbar")` means an import failure reports
as `8 skipped` — not acceptable; treat a skip the same as a failure and
investigate.

- [ ] **Step 6: Confirm the pre-existing regression suite still passes unmodified**

Run: `pytest tests/test_bias_reconstruction_nan_handling.py tests/test_union_mbar_per_epoch_bias.py -v`
Expected: PASS (all tests, same count as before this task — these files
import `_compute_u_nk_analytical`/`_parse_epoch_window_map_native_params`/
`_epoch_bias_param_vectors`/`_reconstruct_union_bias_block`
directly `from analyze_gareus_mbar import ...`, which now resolves through
the Step 4 re-export instead of a local definition — same names, same
values, same behavior).

`tests/test_final_registry_merge.py` is deliberately **not** in that command:
its two imports (`_is_usable_for_mbar`, `_merge_missing_usable_states`) are
untouched local definitions after this task, so it is Plan A2's regression
surface, not this plan's. Running it here is harmless but proves nothing
about this task.

Run: `python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/bias.py`
Expected: clean compile.

- [ ] **Step 7: Commit**

```bash
git add gareus/mbar_analysis/bias.py analyze_gareus_mbar.py tests/test_mbar_analysis_bias_module.py
git commit -m "refactor: relocate bias-reconstruction helpers into gareus.mbar_analysis.bias, delegate to fixed reconstruct_bias_matrix"
```

---

### Task 4: Relocate the MBAR solver family into `gareus/mbar_analysis/solvers.py`, retarget the CLI-override block

**Files:**
- Create: `gareus/mbar_analysis/solvers.py`
- Modify: `analyze_gareus_mbar.py` (delete relocated bodies/constants/optional-import shims, add import block, retarget the `parse_args()` override block)
- Test: `tests/test_mbar_analysis_solvers_module.py` (new file)

**Interfaces:**
- Produces: `gareus.mbar_analysis.solvers.solve_mbar(u_nk, window, tol=1e-10, maxiter=10000, progress=None, backend='auto', threads=0, f_init=None, sambar_epochs=None, sambar_initial_batch_size=None, sambar_batch_patience=None, sambar_seed=None, sambar_lr_scale=None, sambar_delta_f_max=None, sambar_polish_backend=None) -> dict` (keys: `f_k, n_k, active, logw, converged, iterations, max_delta, backend, threads`) — the dispatcher, consumed by Plan A4's PMF-report functions and by `analyze_gareus_mbar.py`'s own convergence-check call sites. `gareus.mbar_analysis.solvers.norm_logw(lw) -> np.ndarray`, `gareus.mbar_analysis.solvers.overlap_matrix(cv, window, bins, K) -> np.ndarray` (shape `(K, K)`), `gareus.mbar_analysis.solvers.logsumexp(a, axis=None)`, `gareus.mbar_analysis.solvers._subset_logw_from_global_fk(d_subset: 'Data', f_k_global: np.ndarray) -> np.ndarray` — all consumed the same way by Plan A4's `run_pmf_and_gamd_boost_report`/`run_secondary_cv_analyses` once those relocate.
- Consumes: nothing from this plan's own `bias.py` or from A2's not-yet-relocated `Data` dataclass (the `'Data'` type hint in `_subset_logw_from_global_fk` is a lazily-evaluated string, never imported at runtime — see design spec's Alternatives Considered).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_solvers_module.py`:

```python
"""Confirms the MBAR solver family relocated from analyze_gareus_mbar.py
into gareus.mbar_analysis.solvers is a real delegation (same object), and
that the CLI-override block in analyze_gareus_mbar.py's parse_args() -- which
used to mutate analyze_gareus_mbar.py's own globals() to apply --sambar-*
flags -- was correctly retargeted to gareus.mbar_analysis.solvers' own
namespace. Before this retarget, the override would silently become a no-op
after relocation: solve_mbar_sambar/solve_mbar_sambar_warmstart read
SAMBAR_EPOCHS etc. as bare names resolved against the module they are
DEFINED in, not the module that happens to also import the same name."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

import analyze_gareus_mbar as agm
import gareus.mbar_analysis.solvers as solvers_mod


def test_solvers_module_is_importable():
    import gareus.mbar_analysis.solvers  # noqa: F401


@pytest.mark.parametrize("name", [
    "logsumexp", "logsumexp_axis1_finite", "logsumexp_axis0_finite",
    "norm_logw", "solve_mbar_numba", "solve_mbar_numba_anderson",
    "solve_mbar_sambar_warmstart", "solve_mbar_sambar", "solve_mbar_lbfgs",
    "solve_mbar", "overlap_matrix", "_subset_logw_from_global_fk",
])
def test_analyze_gareus_mbar_reexports_the_same_object(name):
    assert getattr(agm, name) is getattr(solvers_mod, name)


def test_solve_mbar_still_solves_a_real_two_window_problem():
    """Smoke-level numeric regression: the relocated dispatcher must still
    produce a converged, sane MBAR solve after the move."""
    rng = np.random.default_rng(0)
    beta = 0.5
    centers = [0.0, 1.0]
    k_spring = [30.0, 30.0]
    n_per_window = 3000
    cv_parts, win_parts = [], []
    for k, (c, ks) in enumerate(zip(centers, k_spring)):
        sigma = 1.0 / np.sqrt(beta * ks)
        cv_parts.append(rng.normal(c, sigma, n_per_window))
        win_parts.append(np.full(n_per_window, k, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(win_parts)
    u_nk = beta * 0.5 * np.asarray(k_spring)[None, :] * (cv[:, None] - np.asarray(centers)[None, :]) ** 2

    res = solvers_mod.solve_mbar(u_nk, window, backend="numpy", tol=1e-10, maxiter=20000)

    assert res["converged"]
    assert np.all(np.isfinite(res["f_k"]))


# --- the CLI-override retarget regression ----------------------------------

def test_sambar_epochs_cli_override_updates_the_solvers_module_attribute():
    """Exercises the real, unmodified parse_args() code path end-to-end --
    argparse's `input` positional has no filesystem-existence type=
    validator (confirmed: `p.add_argument('input', help=...)`, no `type=`),
    so a nonexistent dummy path parses fine; parse_args() applies the
    override block as its last step before returning, so this reaches
    exactly the mechanism under test without needing to extract any new
    helper function out of parse_args()."""
    original = solvers_mod.SAMBAR_EPOCHS

    agm.parse_args(["dummy_run_dir", "--sambar-epochs", "7"])

    try:
        assert solvers_mod.SAMBAR_EPOCHS == 7
    finally:
        solvers_mod.SAMBAR_EPOCHS = original


def test_sambar_epochs_cli_override_actually_reaches_a_solver_call(monkeypatch):
    """The stronger check: prove the overridden value is what a real
    solve_mbar(backend='sambar') call receives, not just that the module
    attribute changed (a weaker check that would pass even if a later
    refactor stopped reading that global at all)."""
    received = {}

    def fake_warmstart(u_nk, window, epochs=None, **kwargs):
        received["epochs"] = epochs
        K = u_nk.shape[1]
        return np.zeros(K, dtype=np.float64)

    monkeypatch.setattr(solvers_mod, "solve_mbar_sambar_warmstart", fake_warmstart)
    monkeypatch.setattr(solvers_mod, "SAMBAR_EPOCHS", 7)
    monkeypatch.setattr(solvers_mod, "SAMBAR_POLISH_BACKEND", "numpy")

    u_nk = np.zeros((10, 2), dtype=np.float64)
    window = np.array([0] * 5 + [1] * 5, dtype=np.int64)

    solvers_mod.solve_mbar(u_nk, window, backend="sambar", maxiter=1)

    # solve_mbar_sambar (the wrapper solve_mbar dispatches to for
    # backend='sambar') passes its own `epochs` parameter through to
    # solve_mbar_sambar_warmstart -- when solve_mbar's own `sambar_epochs`
    # argument is left at its None default (the common case, matching the
    # main analyze() call site), solve_mbar_sambar's internal
    # `epochs = SAMBAR_EPOCHS if epochs is None else int(epochs)` resolves
    # the module-level SAMBAR_EPOCHS -- monkeypatched to 7 above -- before
    # ever calling solve_mbar_sambar_warmstart, so the warmstart call itself
    # already receives 7, not None.
    assert received["epochs"] == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_solvers_module.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.mbar_analysis.solvers'`

- [ ] **Step 3: Create `gareus/mbar_analysis/solvers.py`**

First re-confirm every function's current exact boundaries (line numbers
shift as this file gets edited):

```bash
grep -n "^def logsumexp\b\|^def logsumexp_axis1_finite\|^def logsumexp_axis0_finite\|^def norm_logw\|^def ess\b\|^if NUMBA_AVAILABLE:\|^def _anderson_step\|^def solve_mbar_numba\b\|^def solve_mbar_numba_anderson\|^def solve_mbar_sambar_warmstart\|^def solve_mbar_sambar\b\|^def solve_mbar_lbfgs\|^def solve_mbar\b\|^def overlap_matrix\|^def _subset_logw_from_global_fk\|^def make_bins" analyze_gareus_mbar.py
```

Each function's end is the line immediately before the next top-level `def`/
comment-block start reported by the grep above (e.g. `solve_mbar`'s body
runs from its own `def solve_mbar(` line through the line before
`def make_bins(...)`, which must stay in `analyze_gareus_mbar.py` — it is
PMF-domain, Plan A4's job, not moved here).

`^def ess\b` is included in that grep **only as a boundary marker** (it sits
between `norm_logw` and the `if NUMBA_AVAILABLE:` kernel block, so it is
needed to find where `norm_logw`'s body ends) — `ess` itself is **not**
extracted: it stays in `analyze_gareus_mbar.py` for Plan A5, per the
CORRECTION in Global Constraints.

Create `gareus/mbar_analysis/solvers.py` with the following complete content
(constants/shims/docstring are new/modified text below; every function body
is moved verbatim, unchanged, from its current location in
`analyze_gareus_mbar.py` per the boundaries confirmed above):

```python
"""MBAR self-consistent solver family, relocated from analyze_gareus_mbar.py.

Includes the backend dispatcher (solve_mbar), four solver backends
(solve_mbar_numba, solve_mbar_numba_anderson, solve_mbar_sambar +
solve_mbar_sambar_warmstart, solve_mbar_lbfgs), their shared
logsumexp/Anderson-mixing/overlap/subset-reweight helpers, the two
@njit-decorated hot-loop kernels, and the MBAR configuration constants
(DEFAULT_MBAR_BACKEND, SAMBAR_*, MBAR_ANDERSON_HISTORY).

analyze_gareus_mbar.py's parse_args() applies --sambar-*/--mbar-backend/
--mbar-anderson-history CLI overrides by mutating THIS module's globals()
directly (gareus.mbar_analysis.solvers.SAMBAR_EPOCHS = ...), not its own --
every solve_mbar*/logsumexp* function here resolves these names as bare
globals against the module they are defined in (this one), so the override
must target this module's namespace to actually take effect.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

try:
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_AVAILABLE = True
except Exception:  # numba is optional; NumPy backend remains the safe fallback.
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

# -----------------------------------------------------------------------------
# MBAR solver configuration defaults
#
# The following module-level variables control advanced MBAR solver behavior.
# They are exposed as command‑line options in analyze_gareus_mbar.py and can
# be overridden at runtime -- see that module's parse_args(), which mutates
# THIS module's globals() directly after argparse runs. See the argparse
# section in analyze_gareus_mbar.py for the corresponding --sambar-* and
# --mbar-anderson-history flags.

# Default MBAR backend.  "sambar" uses a stochastic warm‑start followed by
# deterministic polishing (usually L-BFGS).  Other choices include
# "numba", "numpy", "anderson", "numba-anderson", "lbfgs", etc.  The
# solve_mbar() function dispatches to the appropriate solver based on this
# string.
DEFAULT_MBAR_BACKEND = 'sambar'

# Stochastic SAMBAR warm‑start parameters.  The SAMBAR algorithm performs
# several epochs of mini‑batch MBAR fixed‑point updates to produce a good
# initial estimate for the free energy offsets f_k.  These parameters
# control the stochastic batching and learning rate.  See
# solve_mbar_sambar_warmstart() for details.
SAMBAR_EPOCHS = 30
SAMBAR_INITIAL_BATCH_SIZE = 1024
SAMBAR_BATCH_PATIENCE = 5
SAMBAR_SEED = 12345
SAMBAR_LR_SCALE = 1.0
SAMBAR_DELTA_F_MAX = 10.0

# Which deterministic backend to use to polish the SAMBAR warm‑start.  This
# should be one of the accepted backends for solve_mbar(): 'lbfgs',
# 'numba', 'numba-anderson', 'anderson', or 'numpy'.  It must not be
# 'sambar' to avoid infinite recursion.
SAMBAR_POLISH_BACKEND = 'numba-anderson'

# Anderson/DIIS mixing history length for the Numba‑accelerated solver.
# A larger history can accelerate convergence but increases memory and
# susceptibility to ill‑conditioning.  See solve_mbar_numba_anderson().
MBAR_ANDERSON_HISTORY = 5


def logsumexp(a, axis=None):
    """Small dependency-free logsumexp.

    The analysis data are cleaned before use, so the hot MBAR path can avoid
    the slower nan-aware reductions.  This fallback still tolerates non-finite
    values outside the hot path.
    """
    a = np.asarray(a, dtype=np.float64)
    if axis is None:
        finite = np.isfinite(a)
        if not np.any(finite): return float('-inf')
        x = a[finite]; m = float(np.max(x)); return float(m + np.log(np.sum(np.exp(x - m))))
    finite = np.isfinite(a)
    safe = np.where(finite, a, -np.inf)
    m = np.max(safe, axis=axis, keepdims=True)
    all_bad = ~np.isfinite(m)
    shifted = np.exp(safe - m)
    shifted = np.where(np.isfinite(shifted), shifted, 0.0)
    out = m + np.log(np.sum(shifted, axis=axis, keepdims=True))
    out = np.where(all_bad, -np.inf, out)
    return np.squeeze(out, axis=axis)


def logsumexp_axis1_finite(a):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=1)
    return m + np.log(np.sum(np.exp(a - m[:, None]), axis=1))


def logsumexp_axis0_finite(a):
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=0)
    return m + np.log(np.sum(np.exp(a - m[None, :]), axis=0))


def norm_logw(lw):
    lw = np.asarray(lw, float); out = np.zeros_like(lw); mask = np.isfinite(lw)
    if not np.any(mask): return out
    x = lw[mask]; x = x - logsumexp(x); out[mask] = np.exp(x); return out


if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_mbar_update(u, logn, f, ld, nf):
        N = u.shape[0]
        K = u.shape[1]
        for n in prange(N):
            m = -1.0e300
            for k in range(K):
                v = logn[k] + f[k] - u[n, k]
                if v > m:
                    m = v
            ss = 0.0
            for k in range(K):
                ss += math.exp(logn[k] + f[k] - u[n, k] - m)
            ld[n] = m + math.log(ss)
        for k in prange(K):
            m = -1.0e300
            for n in range(N):
                v = -u[n, k] - ld[n]
                if v > m:
                    m = v
            ss = 0.0
            for n in range(N):
                ss += math.exp(-u[n, k] - ld[n] - m)
            nf[k] = -(m + math.log(ss))

    @njit(parallel=True, fastmath=True, cache=True)
    def _numba_mbar_logdenom(u, logn, f, ld):
        N = u.shape[0]
        K = u.shape[1]
        for n in prange(N):
            m = -1.0e300
            for k in range(K):
                v = logn[k] + f[k] - u[n, k]
                if v > m:
                    m = v
            ss = 0.0
            for k in range(K):
                ss += math.exp(logn[k] + f[k] - u[n, k] - m)
            ld[n] = m + math.log(ss)
else:
    _numba_mbar_update = None
    _numba_mbar_logdenom = None


def _anderson_step(F_hist: list, G_hist: list, m: int = 5) -> np.ndarray:
    """Anderson/DIIS mixing step.

    Given history lists F_hist (input iterates) and G_hist (fixed-point outputs),
    return the next Anderson-mixed iterate.  Falls back to the last G value when
    the system is ill-conditioned or history is length-1.

    The constrained LS problem min||Σ c_i r_i||² s.t. Σ c_i=1 is solved by
    translating the constraint: c_0 = 1 - Σ c_red, leading to
    dR @ c_red = -r_0 where dR[:,j] = r_{j+1} - r_0.
    """
    mk = min(len(F_hist), m)
    Rk = np.array(G_hist[-mk:]) - np.array(F_hist[-mk:])   # (mk, Ka)
    Gk = np.array(G_hist[-mk:])
    if mk == 1:
        return Gk[0].copy()
    r0 = Rk[0]
    dR = (Rk[1:] - r0).T                                     # (Ka, mk-1)
    try:
        c_red, _, _, _ = np.linalg.lstsq(dR, -r0, rcond=None)
    except Exception:
        return Gk[-1].copy()
    c0 = 1.0 - float(np.sum(c_red))
    coeffs = np.concatenate([[c0], c_red])
    if np.any(np.abs(coeffs) > 1e3) or not np.all(np.isfinite(coeffs)):
        return Gk[-1].copy()
    return coeffs @ Gk


def solve_mbar_numba(u_nk, window, tol=1e-10, maxiter=10000, progress: Optional['Progress'] = None, threads: int = 0, f_init: Optional[np.ndarray] = None):
    """Parallel MBAR fixed-point solve using optional Numba kernels.

    This accelerates the two hot reductions in each iteration:
      logsum_k N_k exp(f_k-u_nk) for every sample, and
      logsum_n exp(-u_nk-logdenom_n) for every state.
    It falls back before import-time if numba is unavailable.
    """
    if not NUMBA_AVAILABLE or _numba_mbar_update is None:
        raise RuntimeError('numba backend requested but numba is not available')
    if int(threads or 0) > 0 and set_num_threads is not None:
        set_num_threads(int(threads))
    used_threads = int(get_num_threads()) if get_num_threads is not None else None
    u_nk = np.asarray(u_nk, dtype=np.float64, order='C')
    window = np.asarray(window, dtype=np.int64)
    N, K = u_nk.shape
    nk = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where(nk > 0)[0]
    if active.size == 0:
        raise ValueError('no samples assigned to any state')
    u = u_nk if active.size == K else np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
    n = nk[active]
    logn = np.log(n)
    f = np.zeros(active.size, dtype=np.float64)
    if f_init is not None:
        fi = np.asarray(f_init, dtype=np.float64)
        if fi.size == K:
            for _i, _a in enumerate(active):
                if _a < fi.size and np.isfinite(fi[_a]):
                    f[_i] = fi[_a]
            f -= f[0]
    nf = np.zeros_like(f)
    ld = np.empty(N, dtype=np.float64)
    conv = False
    md = float('inf')
    if progress is not None:
        progress.step('MBAR backend', f'numba parallel backend; threads={used_threads if used_threads is not None else "auto"}; compiling kernels')
    _numba_mbar_update(u, logn, f, ld, nf)
    for it in range(1, maxiter + 1):
        if progress is not None and (it == 1 or it % 25 == 0):
            progress.bar('MBAR iterations', it, maxiter, f'numba delta {md:.2e}')
        _numba_mbar_update(u, logn, f, ld, nf)
        nf -= nf[0]
        md = float(np.max(np.abs(nf - f)))
        f, nf = nf, f
        if md < tol:
            conv = True
            break
    if progress is not None:
        progress.bar('MBAR iterations', 1, 1, f'backend=numba converged={conv} iter={it} delta={md:.2e}', force=True)
    _numba_mbar_logdenom(u, logn, f, ld)
    lw = -ld
    lw -= logsumexp(lw)
    fall = np.full(K, np.nan, dtype=np.float64)
    fall[active] = f
    return {'f_k': fall, 'n_k': nk, 'active': active, 'logw': lw, 'converged': conv, 'iterations': it, 'max_delta': md, 'backend': 'numba', 'threads': used_threads}


def solve_mbar_numba_anderson(u_nk, window, tol: float = 1e-10, maxiter: int = 10000,
                              progress: Optional['Progress'] = None, threads: int = 0,
                              f_init: Optional[np.ndarray] = None,
                              history: int = MBAR_ANDERSON_HISTORY) -> dict:
    """Parallel MBAR fixed‑point solve using Numba kernels with Anderson/DIIS mixing.

    (docstring unchanged from analyze_gareus_mbar.py -- full parameter/return
    documentation preserved verbatim)
    """
    if not NUMBA_AVAILABLE or _numba_mbar_update is None:
        raise RuntimeError('numba-anderson backend requested but numba is not available')
    if int(threads or 0) > 0 and set_num_threads is not None:
        set_num_threads(int(threads))
    used_threads = int(get_num_threads()) if get_num_threads is not None else None

    u_nk = np.asarray(u_nk, dtype=np.float64, order='C')
    window = np.asarray(window, dtype=np.int64)
    N, K = u_nk.shape
    nk = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where(nk > 0)[0]
    if active.size == 0:
        raise ValueError('no samples assigned to any state')
    u = u_nk if active.size == K else np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
    n = nk[active]
    logn = np.log(n)
    Ka = active.size
    f = np.zeros(Ka, dtype=np.float64)
    if f_init is not None:
        fi = np.asarray(f_init, dtype=np.float64)
        if fi.size == K:
            for _i, _a in enumerate(active):
                if _a < fi.size and np.isfinite(fi[_a]):
                    f[_i] = fi[_a]
            f -= f[0]
    nf = np.zeros_like(f)
    ld = np.empty(N, dtype=np.float64)
    F_hist: list = []
    G_hist: list = []
    conv = False
    md = float('inf')
    if progress is not None:
        progress.step('MBAR backend', f'numba-anderson backend; threads={used_threads if used_threads is not None else "auto"}; compiling kernels')
    _numba_mbar_update(u, logn, f, ld, nf)
    for it in range(1, maxiter + 1):
        if progress is not None and (it == 1 or it % 25 == 0):
            progress.bar('MBAR iterations', it, maxiter, f'numba-anderson delta {md:.2e}')
        _numba_mbar_update(u, logn, f, ld, nf)
        nf -= nf[0]
        md = float(np.max(np.abs(nf - f)))
        F_hist.append(f.copy())
        G_hist.append(nf.copy())
        if history is not None and history > 0:
            while len(F_hist) > history:
                F_hist.pop(0); G_hist.pop(0)
        if len(F_hist) > 1:
            try:
                f_new = _anderson_step(F_hist, G_hist, m=history if history is not None else 5)
                f_new -= f_new[0]
                f = f_new
            except Exception:
                f = nf.copy()
        else:
            f = nf.copy()
        if md < tol:
            conv = True
            break
    if progress is not None:
        progress.bar('MBAR iterations', 1, 1, f'backend=numba-anderson converged={conv} iter={it} delta={md:.2e}', force=True)
    _numba_mbar_logdenom(u, logn, f, ld)
    lw = -ld
    lw -= logsumexp(lw)
    fall = np.full(K, np.nan, dtype=np.float64)
    fall[active] = f
    return {'f_k': fall, 'n_k': nk, 'active': active, 'logw': lw, 'converged': conv,
            'iterations': it, 'max_delta': md, 'backend': 'numba-anderson', 'threads': used_threads}


def solve_mbar_sambar_warmstart(u_nk, window,
                                epochs: int = None,
                                initial_batch_size: int = None,
                                batch_patience: int = None,
                                seed: Optional[int] = None,
                                lr_scale: float = None,
                                delta_f_max: float = None,
                                progress: Optional['Progress'] = None,
                                f_init: Optional[np.ndarray] = None) -> np.ndarray:
    """Stochastic SAMBAR mini‑batch warm‑start for MBAR.

    (docstring unchanged from analyze_gareus_mbar.py -- full parameter/return
    documentation preserved verbatim)
    """
    if epochs is None:
        epochs = SAMBAR_EPOCHS
    if initial_batch_size is None:
        initial_batch_size = SAMBAR_INITIAL_BATCH_SIZE
    if batch_patience is None:
        batch_patience = SAMBAR_BATCH_PATIENCE
    if seed is None:
        seed = SAMBAR_SEED
    if lr_scale is None:
        lr_scale = SAMBAR_LR_SCALE
    if delta_f_max is None:
        delta_f_max = SAMBAR_DELTA_F_MAX
    u_nk = np.asarray(u_nk, dtype=np.float64)
    window = np.asarray(window, dtype=np.int64)
    N, K = u_nk.shape
    nk = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where(nk > 0)[0]
    if active.size == 0:
        raise ValueError('no samples assigned to any state')
    u = u_nk if active.size == K else np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
    n = nk[active]
    logn = np.log(n)
    Ka = active.size
    f = np.zeros(Ka, dtype=np.float64)
    if f_init is not None:
        fi = np.asarray(f_init, dtype=np.float64)
        if fi.size == K:
            for _i, _a in enumerate(active):
                if _a < fi.size and np.isfinite(fi[_a]):
                    f[_i] = fi[_a]
            f -= f[0]
    rng = np.random.default_rng(seed)
    batch_size = int(initial_batch_size)
    patience_counter = 0
    for epoch in range(int(epochs)):
        if progress is not None and (epoch == 0 or epoch % 10 == 0):
            progress.step('MBAR backend', f'SAMBAR warm‑start epoch {epoch + 1}/{epochs}, batch_size={batch_size}')
        if batch_size >= N:
            idx = np.arange(N)
        else:
            idx = rng.integers(0, N, batch_size)
        tmp = logn[None, :] + f[None, :] - u[idx, :]
        ld = logsumexp_axis1_finite(tmp)
        tmp2 = -u[idx, :] - ld[:, None]
        nf = -logsumexp_axis0_finite(tmp2)
        nf -= nf[0]
        delta = nf - f
        lr = float(lr_scale) * math.sqrt(float(len(idx)) / float(N)) if N > 0 else float(lr_scale)
        if delta_f_max is not None and float(delta_f_max) > 0:
            delta = np.clip(delta, -float(delta_f_max), float(delta_f_max))
        f = f + lr * delta
        f -= f[0]
        patience_counter += 1
        if patience_counter >= int(batch_patience) and batch_size < N:
            new_size = min(int(batch_size * 2), N)
            if new_size > batch_size:
                batch_size = new_size
                patience_counter = 0
    f_full = np.full(K, np.nan, dtype=np.float64)
    f_full[active] = f
    return f_full


def solve_mbar_sambar(u_nk, window, tol: float = 1e-10, maxiter: int = 10000,
                      progress: Optional['Progress'] = None, threads: int = 0,
                      f_init: Optional[np.ndarray] = None,
                      polish_backend: Optional[str] = None,
                      epochs: Optional[int] = None,
                      initial_batch_size: Optional[int] = None,
                      batch_patience: Optional[int] = None,
                      seed: Optional[int] = None,
                      lr_scale: Optional[float] = None,
                      delta_f_max: Optional[float] = None) -> dict:
    """Full SAMBAR MBAR solver: stochastic warm‑start followed by deterministic polish.

    (docstring unchanged from analyze_gareus_mbar.py -- full parameter/return
    documentation preserved verbatim)
    """
    if polish_backend is None or not polish_backend:
        polish_backend = SAMBAR_POLISH_BACKEND
    epochs = SAMBAR_EPOCHS if epochs is None else int(epochs)
    initial_batch_size = SAMBAR_INITIAL_BATCH_SIZE if initial_batch_size is None else int(initial_batch_size)
    batch_patience = SAMBAR_BATCH_PATIENCE if batch_patience is None else int(batch_patience)
    seed = SAMBAR_SEED if seed is None else int(seed)
    lr_scale = SAMBAR_LR_SCALE if lr_scale is None else float(lr_scale)
    delta_f_max = SAMBAR_DELTA_F_MAX if delta_f_max is None else float(delta_f_max)
    if progress is not None:
        progress.step('MBAR backend', f'sambar warm-start epochs={epochs}')
    warm_f = solve_mbar_sambar_warmstart(u_nk, window,
                                         epochs=epochs,
                                         initial_batch_size=initial_batch_size,
                                         batch_patience=batch_patience,
                                         seed=seed,
                                         lr_scale=lr_scale,
                                         delta_f_max=delta_f_max,
                                         progress=progress,
                                         f_init=f_init)
    if progress is not None:
        progress.step('MBAR backend', f'sambar polish ({polish_backend})')
    if str(polish_backend).lower() == 'sambar':
        raise RuntimeError('sambar backend cannot polish another sambar solve')
    res = solve_mbar(u_nk, window, tol=tol, maxiter=maxiter, progress=progress,
                     backend=polish_backend, threads=threads, f_init=warm_f)
    res['sambar_warmstart_epochs'] = int(epochs)
    res['sambar_initial_batch_size'] = int(initial_batch_size)
    res['sambar_batch_patience'] = int(batch_patience)
    res['sambar_polish_backend'] = str(polish_backend)
    return res


def solve_mbar_lbfgs(u_nk, window, tol=1e-10, maxiter=10000, progress: Optional['Progress'] = None, f_init: Optional[np.ndarray] = None):
    """MBAR via L-BFGS-B on the negated log-likelihood.

    (docstring unchanged from analyze_gareus_mbar.py -- full derivation
    preserved verbatim)
    """
    if not SCIPY_AVAILABLE:
        raise RuntimeError('lbfgs backend requires scipy')
    u_nk = np.asarray(u_nk, dtype=np.float64, order='C')
    window = np.asarray(window, dtype=np.int64)
    N, K = u_nk.shape
    nk = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where(nk > 0)[0]
    if active.size == 0:
        raise ValueError('no samples assigned to any state')
    Ka = active.size
    u = u_nk if active.size == K else np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
    n = nk[active]
    logn = np.log(n)

    f0 = np.zeros(Ka, dtype=np.float64)
    if f_init is not None:
        fi = np.asarray(f_init, dtype=np.float64)
        if fi.size == K:
            for _i, _a in enumerate(active):
                if _a < fi.size and np.isfinite(fi[_a]):
                    f0[_i] = fi[_a]
            f0 -= f0[0]

    tmp = np.empty((N, Ka), dtype=np.float64)

    def neg_loglik_and_grad(f_red):
        f = np.empty(Ka, dtype=np.float64)
        f[0] = 0.0
        f[1:] = f_red
        np.add(logn[None, :] + f[None, :], -u, out=tmp)
        ld = logsumexp_axis1_finite(tmp)
        neg_L = -(float(np.dot(n, f)) - float(np.sum(ld)))
        tmp2 = tmp - ld[:, None]
        grad_L = n - np.sum(np.exp(tmp2), axis=0)
        neg_grad_red = -grad_L[1:]
        return neg_L, neg_grad_red

    if progress is not None:
        progress.step('MBAR backend', 'L-BFGS-B (scipy)')

    result = _scipy_minimize(
        neg_loglik_and_grad,
        f0[1:],
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': maxiter, 'gtol': tol, 'ftol': 0.0},
    )

    f = np.empty(Ka, dtype=np.float64)
    f[0] = 0.0
    f[1:] = result.x
    conv = result.success or result.status == 0
    it = int(result.nit)
    grad_norm = float(np.max(np.abs(result.jac))) if result.jac is not None else float('nan')

    np.add(logn[None, :] + f[None, :], -u, out=tmp)
    ld = logsumexp_axis1_finite(tmp)
    lw = -ld
    lw -= logsumexp(lw)

    fall = np.full(K, np.nan, dtype=np.float64)
    fall[active] = f

    if progress is not None:
        progress.bar('MBAR iterations', 1, 1, f'backend=lbfgs converged={conv} iter={it} grad_norm={grad_norm:.2e}', force=True)

    return {'f_k': fall, 'n_k': nk, 'active': active, 'logw': lw,
            'converged': conv, 'iterations': it, 'max_delta': grad_norm, 'backend': 'lbfgs', 'threads': None}


def solve_mbar(u_nk, window, tol=1e-10, maxiter=10000, progress: Optional['Progress'] = None, backend: str = 'auto', threads: int = 0, f_init: Optional[np.ndarray] = None,
               sambar_epochs: Optional[int] = None, sambar_initial_batch_size: Optional[int] = None,
               sambar_batch_patience: Optional[int] = None, sambar_seed: Optional[int] = None,
               sambar_lr_scale: Optional[float] = None, sambar_delta_f_max: Optional[float] = None,
               sambar_polish_backend: Optional[str] = None):
    """Solve MBAR self-consistency with selectable backends.

    (docstring unchanged from analyze_gareus_mbar.py -- backend list
    preserved verbatim)
    """
    if backend is None or backend == '':
        backend = DEFAULT_MBAR_BACKEND
    backend = str(backend).lower()
    try:
        problem_size = int(np.asarray(u_nk).shape[0]) * int(np.asarray(u_nk).shape[1])
    except Exception:
        problem_size = 0

    if backend == 'sambar':
        return solve_mbar_sambar(u_nk, window, tol=tol, maxiter=maxiter, progress=progress, threads=threads, f_init=f_init,
                                 polish_backend=sambar_polish_backend, epochs=sambar_epochs,
                                 initial_batch_size=sambar_initial_batch_size, batch_patience=sambar_batch_patience,
                                 seed=sambar_seed, lr_scale=sambar_lr_scale, delta_f_max=sambar_delta_f_max)
    if backend in ('numba-anderson', 'numba-diis'):
        res = solve_mbar_numba_anderson(u_nk, window, tol=tol, maxiter=maxiter, progress=progress, threads=threads, f_init=f_init, history=MBAR_ANDERSON_HISTORY)
        res['backend'] = backend
        return res

    use_lbfgs = (backend == 'lbfgs') or (backend == 'auto' and SCIPY_AVAILABLE and problem_size < 1_000_000)
    if use_lbfgs:
        try:
            return solve_mbar_lbfgs(u_nk, window, tol=tol, maxiter=maxiter, progress=progress, f_init=f_init)
        except Exception as exc:
            if backend == 'lbfgs':
                raise
            if progress is not None:
                progress.step('MBAR backend', f'lbfgs failed ({exc}); trying next backend')

    use_numba = (backend == 'numba') or (backend == 'auto' and NUMBA_AVAILABLE and problem_size >= 200_000)
    if use_numba:
        try:
            return solve_mbar_numba(u_nk, window, tol=tol, maxiter=maxiter, progress=progress, threads=threads, f_init=f_init)
        except Exception as exc:
            if backend == 'numba':
                raise
            if progress is not None:
                progress.step('MBAR backend', f'numba unavailable/failed ({exc}); falling back to anderson/numpy')

    use_anderson = backend in ('anderson', 'auto')
    if progress is not None:
        progress.step('MBAR backend', f'{"anderson" if use_anderson else "numpy"} vectorized backend')
    u_nk = np.asarray(u_nk, dtype=np.float64, order='C')
    window = np.asarray(window, dtype=np.int64)
    N, K = u_nk.shape
    nk = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where(nk > 0)[0]
    if active.size == 0: raise ValueError('no samples assigned to any state')
    u = u_nk if active.size == K else np.ascontiguousarray(u_nk[:, active], dtype=np.float64)
    n = nk[active]
    logn = np.log(n)
    f = np.zeros(active.size, dtype=np.float64)
    if f_init is not None:
        fi = np.asarray(f_init, dtype=np.float64)
        if fi.size == K:
            for _i, _a in enumerate(active):
                if _a < fi.size and np.isfinite(fi[_a]):
                    f[_i] = fi[_a]
            f -= f[0]
    conv = False
    md = float('inf')
    tmp = np.empty_like(u)
    F_hist: list = []
    G_hist: list = []
    for it in range(1, maxiter + 1):
        if progress is not None and (it == 1 or it % 25 == 0):
            progress.bar('MBAR iterations', it, maxiter, f'delta {md:.2e}')
        np.subtract(logn[None, :] + f[None, :], u, out=tmp)
        ld = logsumexp_axis1_finite(tmp)
        np.negative(u, out=tmp)
        tmp -= ld[:, None]
        nf = -logsumexp_axis0_finite(tmp)
        nf -= nf[0]
        md = float(np.max(np.abs(nf - f)))
        if use_anderson:
            F_hist.append(f.copy())
            G_hist.append(nf.copy())
            if len(F_hist) > 5:
                F_hist.pop(0); G_hist.pop(0)
            f = _anderson_step(F_hist, G_hist)
            f -= f[0]
        else:
            f = nf
        if md < tol:
            conv = True
            break
    bname = 'anderson' if use_anderson else 'numpy'
    if progress is not None:
        progress.bar('MBAR iterations', 1, 1, f'backend={bname} converged={conv} iter={it} delta={md:.2e}', force=True)
    np.subtract(logn[None, :] + f[None, :], u, out=tmp)
    ld = logsumexp_axis1_finite(tmp)
    lw = -ld
    lw -= logsumexp(lw)
    fall = np.full(K, np.nan, dtype=np.float64)
    fall[active] = f
    return {'f_k': fall, 'n_k': nk, 'active': active, 'logw': lw, 'converged': conv, 'iterations': it, 'max_delta': md, 'backend': bname, 'threads': None}


def overlap_matrix(cv, window, bins, K):
    """Window-window histogram overlap with vectorized histogram assembly."""
    cv = np.asarray(cv, dtype=np.float64)
    window = np.asarray(window, dtype=np.int64)
    B = len(bins) - 1
    H = np.zeros((K, B), dtype=np.float64)
    if cv.size and K > 0 and B > 0:
        bi = np.searchsorted(bins, cv, side='right') - 1
        bi[cv == bins[-1]] = B - 1
        mask = (window >= 0) & (window < K) & (bi >= 0) & (bi < B)
        if np.any(mask):
            linear = window[mask] * B + bi[mask]
            H = np.bincount(linear, minlength=K * B).reshape(K, B).astype(np.float64)
            row_sums = H.sum(axis=1)
            nz = row_sums > 0
            H[nz] /= row_sums[nz, None]
    return np.minimum(H[:, None, :], H[None, :, :]).sum(axis=2)


def _subset_logw_from_global_fk(d_subset: 'Data', f_k_global: np.ndarray) -> np.ndarray:
    """Correct per-sample MBAR log-weights for a SUBSET of the full sample
    population (e.g. an epoch_000/rest split, or a secondary-CV regime
    split), reusing the already-solved GLOBAL free energies ``f_k_global``
    but the subset's own per-state sample counts ``N_k^subset`` in the MBAR
    self-consistency denominator.

    (docstring unchanged from analyze_gareus_mbar.py -- full derivation and
    degenerate-case discussion preserved verbatim)
    """
    K = int(f_k_global.size)
    u_nk = np.asarray(d_subset.u_nk, dtype=np.float64)
    window = np.asarray(d_subset.window, dtype=np.int64)
    f_k_global = np.asarray(f_k_global, dtype=np.float64)
    n_k_subset = np.bincount(window[(window >= 0) & (window < K)], minlength=K).astype(np.float64)
    active = np.where((n_k_subset > 0) & np.isfinite(f_k_global[:K]))[0]
    if active.size == 0:
        return np.full(u_nk.shape[0], -np.inf, dtype=np.float64)
    log_n = np.log(n_k_subset[active])
    f_active = f_k_global[active]
    if active.size == K and u_nk.shape[1] == K:
        tmp = log_n[None, :] + f_active[None, :] - u_nk
    else:
        tmp = log_n[None, :] + f_active[None, :] - u_nk[:, active]
    ld = logsumexp_axis1_finite(tmp)
    logw_s = -ld
    logw_s -= logsumexp(logw_s)
    return logw_s


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

(`'Progress'` and `'Data'` type hints are never resolved at runtime —
`from __future__ import annotations` at the top of this file, matching
`analyze_gareus_mbar.py`'s own convention, makes every annotation a lazily
evaluated string. Neither name is imported.)

- [ ] **Step 4: Point `analyze_gareus_mbar.py` at the new module and retarget the CLI-override block**

Delete from `analyze_gareus_mbar.py`:
- The numba optional-import block (`try: from numba import njit, prange, set_num_threads, get_num_threads ... NUMBA_AVAILABLE = ...`) — but leave the adjacent `scipy.signal`/`find_peaks` optional-import block (`SCIPY_SIGNAL_AVAILABLE`) untouched; it is unrelated (PMF-domain, not moved).
- The `scipy.optimize.minimize` optional-import block (`_scipy_minimize`/`SCIPY_AVAILABLE`).
- The "MBAR solver configuration defaults" constants block (`DEFAULT_MBAR_BACKEND` through `MBAR_ANDERSON_HISTORY`).
- The full bodies of `logsumexp`, `logsumexp_axis1_finite`, `logsumexp_axis0_finite`, `norm_logw`, the `if NUMBA_AVAILABLE: ... else: ...` njit-kernel block, `_anderson_step`, `solve_mbar_numba`, `solve_mbar_numba_anderson`, `solve_mbar_sambar_warmstart`, `solve_mbar_sambar`, `solve_mbar_lbfgs`, `solve_mbar`, `overlap_matrix`, `_subset_logw_from_global_fk`.

Add, immediately after the `gareus.mbar_analysis.bias` import added in Task
3 Step 4:

```python
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
import gareus.mbar_analysis.solvers as _mbar_solvers
```

Then re-confirm and retarget the CLI-override block:

Run: `grep -n "Apply command" analyze_gareus_mbar.py` (locates the comment
immediately preceding the override block inside `parse_args()`).

Replace the block (currently `g = globals(); g['DEFAULT_MBAR_BACKEND'] = ...`
through `except Exception: pass`) with:

```python
    try:
        _mbar_solvers.DEFAULT_MBAR_BACKEND = str(getattr(args, 'mbar_backend', _mbar_solvers.DEFAULT_MBAR_BACKEND) or _mbar_solvers.DEFAULT_MBAR_BACKEND)
        _mbar_solvers.SAMBAR_EPOCHS = int(getattr(args, 'sambar_epochs', _mbar_solvers.SAMBAR_EPOCHS))
        _mbar_solvers.SAMBAR_INITIAL_BATCH_SIZE = int(getattr(args, 'sambar_initial_batch_size', _mbar_solvers.SAMBAR_INITIAL_BATCH_SIZE))
        _mbar_solvers.SAMBAR_BATCH_PATIENCE = int(getattr(args, 'sambar_batch_patience', _mbar_solvers.SAMBAR_BATCH_PATIENCE))
        _mbar_solvers.SAMBAR_SEED = int(getattr(args, 'sambar_seed', _mbar_solvers.SAMBAR_SEED))
        _mbar_solvers.SAMBAR_LR_SCALE = float(getattr(args, 'sambar_lr_scale', _mbar_solvers.SAMBAR_LR_SCALE))
        _mbar_solvers.SAMBAR_DELTA_F_MAX = float(getattr(args, 'sambar_delta_f_max', _mbar_solvers.SAMBAR_DELTA_F_MAX))
        _mbar_solvers.SAMBAR_POLISH_BACKEND = str(getattr(args, 'sambar_polish_backend', _mbar_solvers.SAMBAR_POLISH_BACKEND) or _mbar_solvers.SAMBAR_POLISH_BACKEND)
        _mbar_solvers.MBAR_ANDERSON_HISTORY = int(getattr(args, 'mbar_anderson_history', _mbar_solvers.MBAR_ANDERSON_HISTORY))
        # Keep this module's own re-exported copies in sync too, in case any
        # call site reads the local name directly rather than calling
        # solve_mbar()/solve_mbar_sambar() afresh (none identified today, but
        # the assignments are cheap and remove the possibility entirely).
        g = globals()
        g['DEFAULT_MBAR_BACKEND'] = _mbar_solvers.DEFAULT_MBAR_BACKEND
        g['SAMBAR_EPOCHS'] = _mbar_solvers.SAMBAR_EPOCHS
        g['SAMBAR_INITIAL_BATCH_SIZE'] = _mbar_solvers.SAMBAR_INITIAL_BATCH_SIZE
        g['SAMBAR_BATCH_PATIENCE'] = _mbar_solvers.SAMBAR_BATCH_PATIENCE
        g['SAMBAR_SEED'] = _mbar_solvers.SAMBAR_SEED
        g['SAMBAR_LR_SCALE'] = _mbar_solvers.SAMBAR_LR_SCALE
        g['SAMBAR_DELTA_F_MAX'] = _mbar_solvers.SAMBAR_DELTA_F_MAX
        g['SAMBAR_POLISH_BACKEND'] = _mbar_solvers.SAMBAR_POLISH_BACKEND
        g['MBAR_ANDERSON_HISTORY'] = _mbar_solvers.MBAR_ANDERSON_HISTORY
    except Exception:
        # Ignore errors during assignment and retain existing defaults
        pass
    return args
```

Note: `parse_args()`'s override block is exercised end-to-end by Step 1's
test via a real `agm.parse_args([...])` call (see that test's docstring) —
no separate helper function needs extracting from `parse_args()` for this
plan's purposes; the retarget above is the only change this step makes to
that block's own logic.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_mbar_analysis_solvers_module.py -v`
Expected: `15 passed` (1 import + 12 identity + 1 numeric smoke + 2
CLI-override; the identity list is 12, not 13, because `ess` is not
relocated by this plan — see the CORRECTION in Global Constraints). This
file's `pytest.importorskip("analyze_gareus_mbar")`
means an import failure reports as `15 skipped` — not acceptable; treat a
skip the same as a failure and investigate (most likely cause: Task 4 Step
4's deletions/imports left `analyze_gareus_mbar.py` with a `NameError` or
`ImportError` at module load time).

- [ ] **Step 6: Confirm the pre-existing regression suite still passes unmodified**

Run: `pytest tests/test_masked_logw_subset_pmf.py tests/test_mbar_lbfgs_convergence.py tests/test_perf_loading_and_solver_memory.py -v`
Expected: PASS (same counts as before this task).

Run: `python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/solvers.py`
Expected: clean compile.

Run (slower, real-MD-adjacent oracle suite — run once before considering
this plan done, per its stakes):
`pytest -q tests/test_query.py tests/test_thermodynamic_validity_2d.py tests/test_thermodynamic_validity_2d_rough.py tests/test_physics_oracle.py`
Expected: same pass/fail profile as on `main` before this plan (no new
failures attributable to this relocation).

- [ ] **Step 7: Commit**

```bash
git add gareus/mbar_analysis/solvers.py analyze_gareus_mbar.py tests/test_mbar_analysis_solvers_module.py
git commit -m "refactor: relocate MBAR solver family into gareus.mbar_analysis.solvers, retarget CLI-override block"
```

---

## Plan Self-Review Notes

- **Spec coverage**: the design spec's Recommended Approach lists four
  coordinated moves — all four map onto Tasks 1/3/4 exactly (Task 2 is an
  added verification-only task the spec's Problem section motivates but
  doesn't number separately). The spec's explicit Out Of Scope items
  (A2/A4/A5 domains, `mbar_subsample.py`, algorithm/tolerance changes, CLI
  surface changes, report regeneration, redesigning the globals-mutation
  pattern itself) are correctly untouched by every task — Task 4 retargets
  *where* the override writes, never *whether* it exists or what it
  overrides.
- **Placeholder scan**: every task's implementation step contains complete,
  runnable code transcribed from the actual current file contents (verified
  during design via direct `Read`/`grep`, not reconstructed from memory).
  Task 4's function-relocation step is the one place using a
  grep-boundary-confirmation instruction rather than re-deriving line
  numbers that will have shifted by execution time — this mirrors the
  project's own established convention (see this repo's CLAUDE.md: "grep to
  relocate, do not trust stale line numbers") and every function's full body
  is given verbatim in the file content immediately below that instruction,
  so nothing is actually left for the executor to invent.
- **Signature consistency across tasks**: `gareus.mbar_analysis.bias`'s two
  formula-bearing wrappers (Task 3) both call
  `gareus.query.reconstruct_bias_matrix` (Task 1) with the exact
  `(cv_A, cv2, windows, beta)` positional order Task 1 fixes and Task 1's
  own test file exercises — no drift between what Task 1 tests and what
  Task 3 calls. `gareus.mbar_analysis.solvers.solve_mbar`'s signature (Task
  4) is unchanged from its pre-relocation form, so Plan A4's consumption
  (`solve_mbar(d.u_nk, d.window, tol=..., progress=..., backend=...,
  threads=..., ...)`) needs no adjustment once A4 relocates its own call
  sites. `_subset_logw_from_global_fk(d_subset: 'Data', f_k_global:
  np.ndarray)` (Task 4) keeps the exact signature
  `tests/test_masked_logw_subset_pmf.py` already calls today.
- **Cross-plan interface surface for A2/A4/A5**: A2's data loaders
  (`load_parquet_adaptive_union`, `_load_epoch_task`,
  `_augment_with_adaptive_rounds`) consume
  `gareus.mbar_analysis.bias._parse_epoch_window_map_native_params(rows:
  list) -> dict`,
  `._epoch_bias_param_vectors(native_params: dict, state_ids: list,
  global_primary_centers: np.ndarray, global_primary_ks: np.ndarray,
  global_sec_centers: np.ndarray, global_sec_ks: np.ndarray) -> tuple`,
  `._reconstruct_union_bias_block(cv: np.ndarray, cv2: np.ndarray, beta:
  float, primary_centers: np.ndarray, primary_ks: np.ndarray, sec_centers:
  np.ndarray, sec_ks: np.ndarray) -> np.ndarray`, and
  `._compute_u_nk_analytical(cv1: np.ndarray, cv2: np.ndarray,
  union_windows: list, beta: float) -> np.ndarray`.
  `_is_usable_for_mbar`/`_merge_missing_usable_states` are **not** on this
  surface — A2 owns and relocates them itself, into
  `gareus.mbar_analysis.loaders_union_parquet`; see the CORRECTION in Global
  Constraints. A4's PMF-report
  functions consume `gareus.mbar_analysis.solvers.solve_mbar(...) -> dict`
  (see Task 4's Interfaces line for the full keyword-argument list and
  return-dict key set), `.norm_logw(lw) -> np.ndarray` (plus `ess(w) ->
  float`, which A4 also consumes but which this plan does NOT relocate —
  Plan A5 moves it to `gareus.math_helpers.ess`; see the CORRECTION in
  Global Constraints),
  `.overlap_matrix(cv, window, bins, K) -> np.ndarray` (shape `(K,K)`), and
  `._subset_logw_from_global_fk(d_subset, f_k_global) -> np.ndarray`. A5
  inherits the explicit, documented, unresolved divergence between
  `overlap_matrix` and `_hist_overlap`/`_adaptive_hist_overlap`/
  `_hist_overlap_np` (design spec, Architecture section) as a flagged
  decision point, not a task this plan hands off half-finished.
