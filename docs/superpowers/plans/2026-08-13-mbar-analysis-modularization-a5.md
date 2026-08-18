# MBAR Analysis Modularization — Plan A5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate `analyze_gareus_mbar.py`'s statistical-comparison and
basin-structure diagnostics (`pmf_probability`, `js_divergence_1d`,
`pmf_rmse_1d`, `barrier_error_1d`, `identify_basins_1d`,
`_compute_basin_populations`, `_weighted_mean_std`,
`_pmf_distribution_mean_std`) into `gareus/diagnostics.py`, relocate `ess`
into `gareus/math_helpers.py` as a shared low-level utility, and reconcile
`gareus/diagnostics.py`'s `_hist_overlap_np` with
`gareus/math_helpers.py`'s `_adaptive_hist_overlap` (two implementations of
the same histogram-overlap statistic) into one delegating implementation.

**Architecture:** `gareus/diagnostics.py` gains eight relocated functions
(five public, three private, matching its existing `__all__` convention)
plus a module-level optional `scipy.signal` import for
`identify_basins_1d`'s peak-finding fallback. `gareus/math_helpers.py`
gains `ess()` (a documented naming-convention exception, since it is
already public API on `analyze_gareus_mbar`) and a new `min_samples`
parameter on `_adaptive_hist_overlap` (default `0`, preserves every
existing call site). `analyze_gareus_mbar.py`'s own definitions of all nine
names are replaced with imports from the two modules — same strangler-fig,
identity-preserving pattern Plan A1 used for its two physical constants.

**Tech Stack:** Python 3.10+, pytest, NumPy, optional SciPy
(`scipy.signal.find_peaks`, already a declared dependency).

**Spec:** `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a5-design.md`

## Global Constraints

- **Plan A1 must land first.** This plan's import-insertion point in
  `analyze_gareus_mbar.py` (immediately after
  `from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K`) assumes that
  line already exists. If A1 hasn't landed yet, insert immediately after
  `import numpy as np` instead and re-confirm with
  `grep -n "^import numpy as np\|^from gareus.units import"
  analyze_gareus_mbar.py` before editing.
- The old invocation (`python analyze_gareus_mbar.py <run_dir> ...`) must
  keep working unchanged throughout — every relocated name stays a valid,
  identically-behaving `analyze_gareus_mbar.<name>` attribute.
- No change to any relocated function's numeric behavior. The one
  intentional behavioral surface (`_adaptive_hist_overlap`'s new
  `min_samples` parameter) must default to reproducing today's behavior
  exactly for every existing caller.
- Every line-number reference below was correct at the time this plan was
  written; the file has been edited many times this session. Re-confirm
  with the stated `grep` command immediately before each edit — do not
  trust the line numbers alone.
- `identify_basins_1d`'s `scipy.signal.find_peaks` usage is optional
  (falls back to a pure-Python peak search) — do not make `scipy` a hard
  new dependency of `gareus/diagnostics.py`.

---

### Task 1: Relocate PMF-vs-reference comparison metrics

**Files:**
- Modify: `gareus/diagnostics.py` (add 4 functions + `__all__`/docstring update)
- Modify: `analyze_gareus_mbar.py:4066-4102` (exact current lines — re-confirm with `grep -n "^def pmf_probability\|^def js_divergence_1d\|^def pmf_rmse_1d\|^def barrier_error_1d" analyze_gareus_mbar.py` before editing)
- Test: `tests/test_diagnostics_pmf_metrics.py` (new file)

**Interfaces:**
- Produces: `gareus.diagnostics.pmf_probability(pmf: dict) -> np.ndarray`,
  `gareus.diagnostics.js_divergence_1d(P, Q) -> float`,
  `gareus.diagnostics.pmf_rmse_1d(F, Fref, P, Pref, min_prob=0.0) -> float`,
  `gareus.diagnostics.barrier_error_1d(F, Fref, P, Pref) -> float`.
- Consumed by: Plan A6's convergence-orchestration code (epoch/frame-count
  convergence loops) once it relocates; consumed internally by
  `js_divergence_1d` (calls `pmf_probability`) and by Task 3's
  `_pmf_distribution_mean_std` (also calls `pmf_probability`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_diagnostics_pmf_metrics.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.diagnostics as gd
import analyze_gareus_mbar as agm


def test_relocated_functions_are_the_same_object_not_a_copy():
    # `is`, not `==` -- proves the import actually replaced the local
    # definition rather than merely producing a coincidentally-equal copy.
    assert agm.pmf_probability is gd.pmf_probability
    assert agm.js_divergence_1d is gd.js_divergence_1d
    assert agm.pmf_rmse_1d is gd.pmf_rmse_1d
    assert agm.barrier_error_1d is gd.barrier_error_1d


def test_pmf_probability_normalizes_and_zero_fills_non_finite():
    out = gd.pmf_probability({'prob': [1, 2, 3, float('nan'), -1]})
    np.testing.assert_allclose(out, [0.2, 0.4, 0.6, 0.0, 0.0])


def test_js_divergence_identical_distributions_is_zero():
    assert gd.js_divergence_1d([1, 2, 3], [1, 2, 3]) == 0.0


def test_js_divergence_disjoint_distributions_is_log2():
    # P=[1,0] and Q=[0,1] normalize to two disjoint point masses; JS
    # divergence between disjoint distributions is exactly ln(2).
    js = gd.js_divergence_1d([1, 0], [0, 1])
    assert js == pytest.approx(np.log(2), abs=1e-12)


def test_pmf_rmse_constant_offset_gives_exact_rmse():
    F = np.array([1.0, 3.0])
    Fref = F + 2.0
    P = np.array([0.5, 0.5])
    assert gd.pmf_rmse_1d(F, Fref, P, P) == pytest.approx(2.0, abs=1e-12)


def test_pmf_rmse_empty_after_masking_is_nan():
    F = np.array([1.0, 3.0])
    P = np.array([0.0, 0.0])  # nothing passes P > min_prob
    assert np.isnan(gd.pmf_rmse_1d(F, F, P, P))


def test_barrier_error_is_abs_difference_of_masked_maxima():
    F = np.array([5.0])
    Fref = np.array([3.0])
    P = np.array([1.0])
    assert gd.barrier_error_1d(F, Fref, P, P) == pytest.approx(2.0, abs=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostics_pmf_metrics.py -v`
Expected: FAIL on the `is`-identity test with
`AttributeError: module 'gareus.diagnostics' has no attribute 'pmf_probability'`
(the value-only tests would pass against the still-in-script originals if
run in isolation, but the import doesn't exist yet, so collection/the
identity test fails first).

- [ ] **Step 3: Write minimal implementation**

In `gareus/diagnostics.py`, add after the existing `_hist_overlap_np`
function (before `compute_gamd_reweighting_diagnostics`):

```python
def pmf_probability(pmf: dict) -> np.ndarray:
    """Normalize a raw PMF ``prob`` array to a finite, non-negative distribution."""
    prob = np.asarray(pmf.get('prob', []), dtype=np.float64)
    total = float(np.nansum(prob))
    if total > 0:
        prob = prob / total
    return np.where(np.isfinite(prob) & (prob >= 0), prob, 0.0)


def js_divergence_1d(P, Q) -> float:
    """Jensen-Shannon divergence (nats) between two (unnormalized) 1D distributions."""
    P = pmf_probability({'prob': P})
    Q = pmf_probability({'prob': Q})
    if P.size != Q.size:
        n = min(P.size, Q.size)
        P = P[:n]
        Q = Q[:n]
    M = 0.5 * (P + Q)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(P > 0, P * np.log(P / np.maximum(M, 1e-300)), 0.0)
        b = np.where(Q > 0, Q * np.log(Q / np.maximum(M, 1e-300)), 0.0)
    return float(0.5 * (np.sum(a) + np.sum(b)))


def pmf_rmse_1d(F, Fref, P, Pref, min_prob=0.0) -> float:
    """RMSE between two PMFs, restricted to bins with probability above ``min_prob`` on both sides."""
    F = np.asarray(F, dtype=np.float64)
    Fref = np.asarray(Fref, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    Pref = np.asarray(Pref, dtype=np.float64)
    n = min(F.size, Fref.size, P.size, Pref.size)
    if n <= 0:
        return float('nan')
    F = F[:n]
    Fref = Fref[:n]
    P = P[:n]
    Pref = Pref[:n]
    mask = np.isfinite(F) & np.isfinite(Fref) & (P > float(min_prob)) & (Pref > float(min_prob))
    if not np.any(mask):
        return float('nan')
    d = F[mask] - Fref[mask]
    return float(np.sqrt(np.mean(d * d)))


def barrier_error_1d(F, Fref, P, Pref) -> float:
    """Absolute difference between two PMFs' maxima, restricted to bins with positive probability on both sides."""
    F = np.asarray(F, dtype=np.float64)
    Fref = np.asarray(Fref, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    Pref = np.asarray(Pref, dtype=np.float64)
    n = min(F.size, Fref.size, P.size, Pref.size)
    if n <= 0:
        return float('nan')
    mask = np.isfinite(F[:n]) & np.isfinite(Fref[:n]) & (P[:n] > 0) & (Pref[:n] > 0)
    if not np.any(mask):
        return float('nan')
    return float(abs(np.nanmax(F[:n][mask]) - np.nanmax(Fref[:n][mask])))
```

Update `__all__` at the top of `gareus/diagnostics.py`:

```python
__all__ = [
    "compute_gamd_reweighting_diagnostics",
    "validate_us_mbar_inputs",
    "pmf_probability",
    "js_divergence_1d",
    "pmf_rmse_1d",
    "barrier_error_1d",
]
```

In `analyze_gareus_mbar.py`, delete the `pmf_probability`, `js_divergence_1d`,
`pmf_rmse_1d`, and `barrier_error_1d` function definitions (the exact block
found by the Files-section grep above), and add, immediately after the
Plan-A1 `from gareus.units import ...` line:

```python
from gareus.diagnostics import pmf_probability, js_divergence_1d, pmf_rmse_1d, barrier_error_1d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_diagnostics_pmf_metrics.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Confirm nothing downstream broke**

Run: `pytest -q tests/test_pmf_gamd_nan_bin_handling.py -v`
Expected: PASS, unchanged — this file imports these four names directly
from `analyze_gareus_mbar` (`from analyze_gareus_mbar import (...,
pmf_probability, js_divergence_1d, pmf_rmse_1d, barrier_error_1d)`
alongside A4-owned names this plan does not touch); the import-swap must
make those names resolve identically.

Run: `python -m py_compile analyze_gareus_mbar.py gareus/diagnostics.py`
Expected: clean compile.

- [ ] **Step 6: Commit**

```bash
git add gareus/diagnostics.py analyze_gareus_mbar.py tests/test_diagnostics_pmf_metrics.py
git commit -m "refactor: relocate PMF comparison metrics into gareus.diagnostics"
```

---

### Task 2: Relocate basin identification and population math

**Files:**
- Modify: `gareus/diagnostics.py` (add module-top optional scipy import + 2 functions + `__all__`)
- Modify: `analyze_gareus_mbar.py:36-41,4104-4165` (exact current lines — re-confirm with `grep -n "^try:$\|find_peaks\|^def identify_basins_1d\|^def _compute_basin_populations" analyze_gareus_mbar.py` before editing; the `try:` at line 36 is the third of three near-identical optional-import blocks at the top of the file — confirm you have the `scipy.signal` one, not the `numba` or `scipy.optimize` ones just above it)
- Test: `tests/test_diagnostics_basins.py` (new file)

**Interfaces:**
- Produces: `gareus.diagnostics.identify_basins_1d(cv_A, F, min_depth_kcal=0.5) -> list[dict]`,
  `gareus.diagnostics._compute_basin_populations(prob, basins) -> list[float]`.
- Consumed by: Plan A6's `run_observable_pmf_convergence` (basin-tracking
  block) once it relocates.

- [ ] **Step 1: Write the failing test**

Create `tests/test_diagnostics_basins.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.diagnostics as gd
import analyze_gareus_mbar as agm


def test_relocated_functions_are_the_same_object_not_a_copy():
    assert agm.identify_basins_1d is gd.identify_basins_1d
    assert agm._compute_basin_populations is gd._compute_basin_populations


def test_scipy_signal_optional_import_flag_exists():
    # Whichever value it resolves to in this environment, the flag and the
    # (possibly-None) function reference must both exist as module attrs.
    assert hasattr(gd, "SCIPY_SIGNAL_AVAILABLE")
    assert hasattr(gd, "_scipy_find_peaks")


def _symmetric_double_well():
    cv_A = np.arange(7.0)
    F = np.array([4.0, 2.0, 0.0, 3.0, 0.0, 2.0, 4.0])
    return cv_A, F


def test_identify_basins_finds_exactly_two_basins_with_correct_boundaries():
    cv_A, F = _symmetric_double_well()
    basins = gd.identify_basins_1d(cv_A, F, min_depth_kcal=1.0)
    assert len(basins) == 2
    assert basins[0]['left_bin'] == 0
    assert basins[0]['right_bin'] == 3
    assert basins[0]['center_cv_A'] == pytest.approx(2.0)
    assert basins[0]['min_F_kcal'] == pytest.approx(0.0)
    assert basins[1]['left_bin'] == 4
    assert basins[1]['right_bin'] == 6
    assert basins[1]['center_cv_A'] == pytest.approx(4.0)


def test_identify_basins_partition_covers_full_range_with_no_gap_or_overlap():
    cv_A, F = _symmetric_double_well()
    basins = gd.identify_basins_1d(cv_A, F, min_depth_kcal=1.0)
    basins = sorted(basins, key=lambda b: b['left_bin'])
    assert basins[0]['left_bin'] == 0
    assert basins[-1]['right_bin'] == len(F) - 1
    for a, b in zip(basins, basins[1:]):
        assert b['left_bin'] == a['right_bin'] + 1  # no gap, no overlap


def test_identify_basins_too_few_points_returns_empty():
    assert gd.identify_basins_1d(np.array([0.0, 1.0]), np.array([1.0, 2.0])) == []


def test_compute_basin_populations_sums_to_one_and_matches_known_split():
    cv_A, F = _symmetric_double_well()
    basins = gd.identify_basins_1d(cv_A, F, min_depth_kcal=1.0)
    prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])
    pops = gd._compute_basin_populations(prob, basins)
    assert pops == pytest.approx([0.4, 0.6], abs=1e-9)
    assert sum(pops) == pytest.approx(1.0, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostics_basins.py -v`
Expected: FAIL — `AttributeError: module 'gareus.diagnostics' has no attribute 'identify_basins_1d'`.

- [ ] **Step 3: Write minimal implementation**

In `gareus/diagnostics.py`, add near the top, immediately after
`import numpy as np` (before the `__all__` list):

```python
try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    SCIPY_SIGNAL_AVAILABLE = True
except Exception:  # scipy is optional; identify_basins_1d has a pure-Python fallback.
    _scipy_find_peaks = None
    SCIPY_SIGNAL_AVAILABLE = False
```

Add after Task 1's new functions:

```python
def identify_basins_1d(cv_A: np.ndarray, F: np.ndarray, min_depth_kcal: float = 0.5) -> list:
    """Find basins in a 1D PMF and partition the CV axis into basin domains.

    Returns a list of dicts (sorted by CV position):
      basin_id, center_cv_A, min_F_kcal, prominence_kcal,
      left_bin, right_bin (inclusive bin indices into cv_A/F),
      left_cv_A, right_cv_A.

    Basin boundaries sit at the local maximum between adjacent minima so that
    every bin belongs to exactly one basin and populations sum to 1.
    """
    cv_A = np.asarray(cv_A, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    n = len(F)
    if n < 3:
        return []
    F_safe = np.where(np.isfinite(F), F, np.inf)
    neg_F = np.where(np.isfinite(F_safe), -F_safe, -np.inf)
    minima_idx: list = []
    prominences: list = []
    if SCIPY_SIGNAL_AVAILABLE and _scipy_find_peaks is not None:
        try:
            peaks, props = _scipy_find_peaks(neg_F, prominence=min_depth_kcal)
            minima_idx = list(peaks)
            prominences = list(props['prominences'])
        except Exception:
            pass
    if not minima_idx:
        finite_mask = np.isfinite(F)
        for i in range(1, n - 1):
            if not finite_mask[i]:
                continue
            left_ok = finite_mask[:i]
            right_ok = finite_mask[i + 1:]
            if not (left_ok.any() and right_ok.any()):
                continue
            if F[i] >= F[i - 1] or F[i] >= F[i + 1]:
                continue
            lmax = float(np.max(F[:i][left_ok]))
            rmax = float(np.max(F[i + 1:][right_ok]))
            prom = min(lmax, rmax) - F[i]
            if prom >= min_depth_kcal:
                minima_idx.append(i)
                prominences.append(float(prom))
    if not minima_idx:
        finite_idx = np.where(np.isfinite(F))[0]
        if len(finite_idx) == 0:
            return []
        gmin = int(finite_idx[np.argmin(F[finite_idx])])
        minima_idx = [gmin]
        prominences = [0.0]
    order = np.argsort(minima_idx)
    minima_idx = [minima_idx[i] for i in order]
    prominences = [prominences[i] for i in order]
    # barrier_bins[i] = argmax between minima_idx[i] and minima_idx[i+1]
    barrier_bins = []
    for i in range(len(minima_idx) - 1):
        lo, hi = minima_idx[i], minima_idx[i + 1]
        barrier_bins.append(lo + int(np.argmax(F_safe[lo:hi + 1])))
    # Partition: basin i covers [left_edges[i], right_edges[i]] inclusive.
    # Barrier bin belongs to the left basin so the union is [0, n-1] without gaps or overlap.
    left_edges = [0] + [b + 1 for b in barrier_bins]
    right_edges = barrier_bins + [n - 1]
    basins = []
    for i, mi in enumerate(minima_idx):
        lb, rb = left_edges[i], right_edges[i]
        basins.append({
            'basin_id': i,
            'center_cv_A': float(cv_A[mi]),
            'min_F_kcal': float(F[mi]) if np.isfinite(F[mi]) else float('nan'),
            'prominence_kcal': float(prominences[i]),
            'left_bin': int(lb),
            'right_bin': int(rb),
            'left_cv_A': float(cv_A[lb]),
            'right_cv_A': float(cv_A[rb]),
        })
    return basins


def _compute_basin_populations(prob: np.ndarray, basins: list) -> list:
    prob = np.asarray(prob, dtype=np.float64)
    return [float(np.sum(prob[b['left_bin']:b['right_bin'] + 1])) for b in basins]
```

Update `__all__`, adding `"identify_basins_1d"` (`_compute_basin_populations`
stays unlisted, matching `_sample_counts_by_window`'s existing convention).

In `analyze_gareus_mbar.py`:
1. Delete the `scipy.signal` optional-import block (`try: from scipy.signal
   import find_peaks as _scipy_find_peaks; SCIPY_SIGNAL_AVAILABLE = True;
   except Exception: _scipy_find_peaks = None; SCIPY_SIGNAL_AVAILABLE =
   False`) — confirm via `grep -n "SCIPY_SIGNAL_AVAILABLE\|_scipy_find_peaks"
   analyze_gareus_mbar.py` that no reference to either name remains outside
   this block before deleting it.
2. Delete the `identify_basins_1d` and `_compute_basin_populations` function definitions.
3. Add, next to Task 1's new import line:

```python
from gareus.diagnostics import identify_basins_1d, _compute_basin_populations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_diagnostics_basins.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Confirm nothing downstream broke**

Run: `grep -n "SCIPY_SIGNAL_AVAILABLE\|_scipy_find_peaks" analyze_gareus_mbar.py`
Expected: no output (both names fully removed, not just their definitions —
if this prints anything, some other code still references the deleted
names and Step 3's deletion needs to be revisited).

Run: `python -m py_compile analyze_gareus_mbar.py gareus/diagnostics.py`
Expected: clean compile.

- [ ] **Step 6: Commit**

```bash
git add gareus/diagnostics.py analyze_gareus_mbar.py tests/test_diagnostics_basins.py
git commit -m "refactor: relocate basin identification math into gareus.diagnostics"
```

---

### Task 3: Relocate weighted/PMF-distribution mean-std helpers

**Files:**
- Modify: `gareus/diagnostics.py` (add 2 private functions)
- Modify: `analyze_gareus_mbar.py:5623-5646` (exact current lines — re-confirm with `grep -n "^def _weighted_mean_std\|^def _pmf_distribution_mean_std" analyze_gareus_mbar.py` before editing)
- Test: `tests/test_diagnostics_mean_std.py` (new file)

**Interfaces:**
- Produces: `gareus.diagnostics._weighted_mean_std(values, weights) -> tuple[float, float]`,
  `gareus.diagnostics._pmf_distribution_mean_std(pmf: dict) -> tuple[float, float]`.
- Consumed by: Plan A6's Rg-PMF summary computation once it relocates
  (currently `analyze_gareus_mbar.py:5718`, the only call site for either
  function).

- [ ] **Step 1: Write the failing test**

Create `tests/test_diagnostics_mean_std.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.diagnostics as gd
import analyze_gareus_mbar as agm


def test_relocated_functions_are_the_same_object_not_a_copy():
    assert agm._weighted_mean_std is gd._weighted_mean_std
    assert agm._pmf_distribution_mean_std is gd._pmf_distribution_mean_std


def test_weighted_mean_std_known_answer():
    mean, std = gd._weighted_mean_std([1, 2, 3], [1, 1, 1])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx((2.0 / 3.0) ** 0.5)


def test_pmf_distribution_mean_std_agrees_with_weighted_mean_std_for_uniform_prob():
    # Uniform prob over [1,2,3] is the same distribution as uniform weights
    # over [1,2,3] -- the two functions must agree exactly.
    pmf = {'cv_A': [1, 2, 3], 'prob': [1, 1, 1]}
    mean, std = gd._pmf_distribution_mean_std(pmf)
    mean_w, std_w = gd._weighted_mean_std([1, 2, 3], [1, 1, 1])
    assert mean == pytest.approx(mean_w)
    assert std == pytest.approx(std_w)


def test_weighted_mean_std_all_zero_weight_is_nan():
    mean, std = gd._weighted_mean_std([1, 2, 3], [0, 0, 0])
    assert np.isnan(mean)
    assert np.isnan(std)


def test_pmf_distribution_mean_std_empty_is_nan():
    mean, std = gd._pmf_distribution_mean_std({'cv_A': [], 'prob': []})
    assert np.isnan(mean)
    assert np.isnan(std)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_diagnostics_mean_std.py -v`
Expected: FAIL — `AttributeError: module 'gareus.diagnostics' has no attribute '_weighted_mean_std'`.

- [ ] **Step 3: Write minimal implementation**

In `gareus/diagnostics.py`, add after Task 2's functions (ensure `import math`
is already present at module top — it is, from the existing `_safe_float`):

```python
def _weighted_mean_std(values, weights):
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    if not np.any(mask):
        return float('nan'), float('nan')
    v = v[mask]
    w = w[mask]
    sw = float(np.sum(w))
    if sw <= 0:
        return float('nan'), float('nan')
    w = w / sw
    mean = float(np.sum(w * v))
    var = float(np.sum(w * (v - mean) ** 2))
    return mean, float(math.sqrt(max(0.0, var)))


def _pmf_distribution_mean_std(pmf: dict):
    x = np.asarray(pmf.get('cv_A', []), dtype=np.float64)
    p = pmf_probability(pmf)
    if x.size == 0 or p.size == 0:
        return float('nan'), float('nan')
    n = min(x.size, p.size)
    x = x[:n]
    p = p[:n]
    mask = np.isfinite(x) & np.isfinite(p) & (p >= 0)
    if not np.any(mask):
        return float('nan'), float('nan')
    x = x[mask]
    p = p[mask]
    sp = float(np.sum(p))
    if sp <= 0:
        return float('nan'), float('nan')
    p = p / sp
    mean = float(np.sum(p * x))
    var = float(np.sum(p * (x - mean) ** 2))
    return mean, float(math.sqrt(max(0.0, var)))
```

Neither is added to `__all__` (both stay private helpers, matching
`_sample_counts_by_window`'s convention).

In `analyze_gareus_mbar.py`, delete the `_weighted_mean_std` and
`_pmf_distribution_mean_std` function definitions, and add, next to the
previous tasks' import lines:

```python
from gareus.diagnostics import _weighted_mean_std, _pmf_distribution_mean_std
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_diagnostics_mean_std.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Confirm nothing downstream broke**

Run: `python -m py_compile analyze_gareus_mbar.py gareus/diagnostics.py`
Expected: clean compile.

- [ ] **Step 6: Commit**

```bash
git add gareus/diagnostics.py analyze_gareus_mbar.py tests/test_diagnostics_mean_std.py
git commit -m "refactor: relocate weighted/PMF-distribution mean-std helpers into gareus.diagnostics"
```

---

### Task 4: Relocate `ess` into `gareus/math_helpers.py`

**Files:**
- Modify: `gareus/math_helpers.py` (add `ess`, extend `__all__`)
- Modify: `analyze_gareus_mbar.py:2572-2575` (exact current lines — re-confirm with `grep -n "^def ess" analyze_gareus_mbar.py` before editing)
- Test: `tests/test_math_helpers_ess.py` (new file)

**Interfaces:**
- Produces: `gareus.math_helpers.ess(w: np.ndarray) -> float`.
- Consumed by: `analyze_gareus_mbar.py`'s own `boost_stats`, epoch
  convergence loop, and `run_pmf_and_gamd_boost_report` (all pick this up
  automatically through the import swap, no call-site changes); flagged in
  the design spec as a function Plan A3 (MBAR solver reporting) should also
  import from here rather than reimplementing.

- [ ] **Step 1: Write the failing test**

Create `tests/test_math_helpers_ess.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.math_helpers as gmh
import analyze_gareus_mbar as agm


def test_relocated_ess_is_the_same_object_not_a_copy():
    assert agm.ess is gmh.ess


def test_ess_uniform_weights_equals_sample_count():
    assert gmh.ess(np.ones(5)) == pytest.approx(5.0)


def test_ess_single_dominant_weight_is_one():
    assert gmh.ess(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_ess_empty_after_filtering_is_zero():
    assert gmh.ess(np.array([-1.0, float('nan'), -5.0])) == 0.0


def test_ess_used_by_thermodynamic_validity_tests_still_resolves():
    # Sanity check for the exact call pattern
    # tests/test_thermodynamic_validity_2d.py and friends use directly.
    w = np.array([0.5, 0.5, 0.5, 0.5])
    assert agm.ess(w) / w.size == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_math_helpers_ess.py -v`
Expected: FAIL — `AttributeError: module 'gareus.math_helpers' has no attribute 'ess'`.

- [ ] **Step 3: Write minimal implementation**

In `gareus/math_helpers.py`, add after `_adaptive_hist_overlap` (before
`_batch_torsion_angles_rad`):

```python
def ess(w: np.ndarray) -> float:
    """Effective sample size (Kish estimator) from an array of importance weights.

    ``sum(w)**2 / sum(w**2)``. Non-finite and negative entries are dropped
    before computing; an all-dropped or all-non-positive input returns 0.0.

    Kept without this module's usual leading underscore because it is
    already public API on ``analyze_gareus_mbar`` (``agm.ess``), called
    directly by three real-MD physics-oracle test files
    (tests/test_thermodynamic_validity_2d.py,
    tests/test_thermodynamic_validity_2d_rough.py,
    tests/test_thermodynamic_validity_real_md.py).
    """
    w = np.asarray(w, dtype=np.float64)
    w = w[np.isfinite(w) & (w >= 0)]
    if w.size == 0:
        return 0.0
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return 0.0 if s2 <= 0 else s1 * s1 / s2
```

Update `__all__` at the bottom of `gareus/math_helpers.py`:

```python
__all__ = [
    "_hist_overlap",
    "_adaptive_hist_overlap",
    "ess",
    "_batch_torsion_angles_rad",
    "_mean_torsion_score_from_angles",
    "_torsion_angle_rad_from_positions",
]
```

In `analyze_gareus_mbar.py`, delete the `ess` function definition, and add,
next to the previous tasks' import lines:

```python
from gareus.math_helpers import ess
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_math_helpers_ess.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Confirm nothing downstream broke**

Run: `pytest -q tests/test_thermodynamic_validity_2d.py -k "not slow" -v 2>&1 | tail -30`
(or the fastest available subset — these are real-MD tests; if the full
file is too slow to run in this step, at minimum run:
`python -c "import analyze_gareus_mbar as agm; import numpy as np; assert agm.ess(np.ones(3)) == 3.0"`
to confirm the re-exported name is callable with the exact signature these
tests use.)

Run: `python -m py_compile analyze_gareus_mbar.py gareus/math_helpers.py`
Expected: clean compile.

- [ ] **Step 6: Commit**

```bash
git add gareus/math_helpers.py analyze_gareus_mbar.py tests/test_math_helpers_ess.py
git commit -m "refactor: relocate ess into gareus.math_helpers as a shared utility"
```

---

### Task 5: Reconcile `_hist_overlap_np` with `_adaptive_hist_overlap`

**Files:**
- Modify: `gareus/math_helpers.py:52-83` (`_adaptive_hist_overlap` — exact current lines, re-confirm with `grep -n "^def _adaptive_hist_overlap" gareus/math_helpers.py`)
- Modify: `gareus/diagnostics.py:80-100` (`_hist_overlap_np` — exact current lines, re-confirm with `grep -n "^def _hist_overlap_np" gareus/diagnostics.py`; line numbers here have shifted from the original file since Tasks 1-3 added functions above this one)
- Test: `tests/test_hist_overlap_reconciliation.py` (new file)

**Interfaces:**
- Modifies: `gareus.math_helpers._adaptive_hist_overlap` gains
  `min_samples: int = 0` (backward compatible — no existing call site in
  `gareus/adaptive_feedback.py` passes it).
- Produces: `gareus.diagnostics._hist_overlap_np` now delegates to
  `gareus.math_helpers._adaptive_hist_overlap(..., min_samples=5)` instead
  of maintaining an independent implementation.

**Note:** this task is entirely internal to the `gareus` package — it does
not touch `analyze_gareus_mbar.py` at all (its `overlap_matrix` is a
different, vectorized K×K implementation, out of this plan's scope).

- [ ] **Step 1: Write the failing test**

Create `tests/test_hist_overlap_reconciliation.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gareus.diagnostics import _hist_overlap_np
from gareus.math_helpers import _adaptive_hist_overlap


def test_diagnostics_floor_still_returns_nan_below_five_samples():
    # 3 identical-valued samples per side: below _hist_overlap_np's
    # pre-existing 5-sample floor -> NaN, even though the underlying
    # histograms would show perfect (1.0) overlap.
    a = [1.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0]
    assert np.isnan(_hist_overlap_np(a, b, 0.0, 2.0))


def test_math_helpers_default_has_no_five_sample_floor():
    # Same 3-sample input, default min_samples=0 (unchanged
    # adaptive_feedback.py behavior): full overlap, not NaN.
    a = [1.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0]
    assert _adaptive_hist_overlap(a, b, 0.0, 2.0) == pytest.approx(1.0)


def test_math_helpers_explicit_min_samples_matches_diagnostics_floor():
    a = [1.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0]
    assert np.isnan(_adaptive_hist_overlap(a, b, 0.0, 2.0, min_samples=5))


def test_both_entry_points_agree_bit_for_bit_above_the_floor():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 50)
    b = rng.normal(0.2, 1, 50)
    via_diagnostics = _hist_overlap_np(a, b, -3, 3, bins=40)
    via_math_helpers = _adaptive_hist_overlap(a, b, -3, 3, bins=40, min_samples=5)
    assert via_diagnostics == via_math_helpers


def test_adaptive_feedback_default_call_shape_unaffected():
    # Mirrors gareus/adaptive_feedback.py's real call shape: positional
    # a, b, lo, hi, no min_samples argument at all.
    a = [0.1, 0.2, 0.3, 0.4, 0.5]
    b = [0.15, 0.25, 0.35, 0.45, 0.55]
    result = _adaptive_hist_overlap(a, b, 0.0, 1.0)
    assert np.isfinite(result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hist_overlap_reconciliation.py -v`
Expected: `test_math_helpers_explicit_min_samples_matches_diagnostics_floor`
FAILS with `TypeError: _adaptive_hist_overlap() got an unexpected keyword
argument 'min_samples'` (the other four tests describe today's already-true
behavior and would pass even before this task's change — they exist to
pin that behavior as a regression guard once the delegation lands).

- [ ] **Step 3: Write minimal implementation**

In `gareus/math_helpers.py`, modify `_adaptive_hist_overlap`'s signature
and its second sample-count guard:

Before:
```python
def _adaptive_hist_overlap(values_a: List[float] | np.ndarray, values_b: List[float] | np.ndarray, lo: float, hi: float, bins: int = 80) -> float:
    ...
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
```

After:
```python
def _adaptive_hist_overlap(values_a: List[float] | np.ndarray, values_b: List[float] | np.ndarray, lo: float, hi: float, bins: int = 80, min_samples: int = 0) -> float:
    """Histogram overlap sum(min(P_i, P_j)) for two 1D CV samples.

    Accepts lists or already-sliced NumPy arrays.  Keeping bootstrap resamples as
    arrays avoids thousands of temporary Python-list conversions in adaptive
    feedback without changing the histogram definition.

    ``min_samples`` is an optional post-finite-filter floor on each side's
    sample count (default 0, i.e. only the empty-array case returns NaN --
    unchanged behavior for every existing call site in adaptive_feedback.py).
    gareus.diagnostics._hist_overlap_np delegates here with min_samples=5,
    its own pre-existing floor.
    """
    ...
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    floor = max(1, int(min_samples))
    if a.size < floor or b.size < floor:
        return float("nan")
```

(Only the signature line, the docstring, and the one guard line change;
everything else in the function body — the `lo`/`hi` auto-range fallback,
padding, histogramming, normalization — is untouched.)

In `gareus/diagnostics.py`, add `from .math_helpers import
_adaptive_hist_overlap` to the top-level imports (next to the
`SCIPY_SIGNAL_AVAILABLE` block added in Task 2), and replace
`_hist_overlap_np`'s body:

Before:
```python
def _hist_overlap_np(a: np.ndarray, b: np.ndarray, lo: float, hi: float, bins: int = 80) -> float:
    """Estimate the histogram overlap between two distributions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        allv = np.concatenate([a, b])
        if allv.size < 2:
            return float("nan")
        lo, hi = float(np.min(allv)), float(np.max(allv))
    pad = max(0.1, 0.02 * (hi - lo))
    ha, _ = np.histogram(a, bins=max(8, int(bins)), range=(lo - pad, hi + pad))
    hb, _ = np.histogram(b, bins=max(8, int(bins)), range=(lo - pad, hi + pad))
    if ha.sum() <= 0 or hb.sum() <= 0:
        return float("nan")
    pa = ha.astype(float) / float(ha.sum())
    pb = hb.astype(float) / float(hb.sum())
    return float(np.minimum(pa, pb).sum())
```

After:
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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hist_overlap_reconciliation.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Confirm nothing downstream broke**

Run: `pytest -q tests/test_2d_overlap_neighbor_graph.py tests/test_package_smoke.py -v`
Expected: PASS, unchanged — the former calls `_hist_overlap_np` from
`gareus.diagnostics` directly with real ≥5-sample fixtures (now routed
through the shared implementation, but bit-identical per Step 4's test 4);
the latter calls `_adaptive_hist_overlap` via `gareus.adaptive_feedback`'s
re-export with no `min_samples` argument (default `0`, unchanged).

Run: `python -m py_compile gareus/diagnostics.py gareus/math_helpers.py`
Expected: clean compile.

- [ ] **Step 6: Commit**

```bash
git add gareus/diagnostics.py gareus/math_helpers.py tests/test_hist_overlap_reconciliation.py
git commit -m "refactor: unify _hist_overlap_np and _adaptive_hist_overlap via a shared min_samples parameter"
```

---

## Plan Self-Review Notes

- **Spec coverage**: all eight `analyze_gareus_mbar.py` functions named in
  scope (Tasks 1-3), `ess`'s relocation to the shared leaf module (Task 4),
  and the `_hist_overlap_np`/`_adaptive_hist_overlap` reconciliation
  explicitly assigned to this plan by A1's Out Of Scope section (Task 5)
  are all covered. The spec's explicit non-goals — `_plot_basin_population_convergence`,
  `overlap_matrix`, `solve_mbar*`/`logsumexp*`/`norm_logw`, `boost_stats`/
  `_window_moments`, `compute_gamd_reweighting_diagnostics`'s own inline ESS,
  `_window_cv_mean_std`, `gareus/logger.py`'s `_hist_overlap` — are correctly
  left untouched by every task; none of the five tasks edits any of them.
- **Placeholder scan**: every task's Step 3 contains the complete,
  verbatim-or-precisely-modified function body (confirmed against a live
  read of the current file, not reconstructed from memory), not a
  "..." stand-in — the one intentional elision (Task 5's `_adaptive_hist_overlap`
  Before/After blocks) is explicitly scoped to "only the signature line,
  the docstring, and the one guard line change" with the exact before/after
  text of that one line given.
- **Signature/type consistency checked**: `pmf_probability`'s signature
  (`dict -> np.ndarray`) matches what `js_divergence_1d` and
  `_pmf_distribution_mean_std` call it with after relocation (both moved in
  Tasks 1/3, both already call `pmf_probability` by name, resolved via
  Python's normal module-level name lookup once all three live in
  `gareus/diagnostics.py`); `_adaptive_hist_overlap`'s new `min_samples`
  parameter is appended after `bins` (its last existing parameter), so no
  existing positional call site anywhere in `gareus/adaptive_feedback.py`
  (verified: none pass a 5th positional argument) is affected.
- **Known-answer values were computed by executing the current
  implementation directly**, not hand-derived, before being written into
  any test in this plan (`js_divergence_1d([1,0],[0,1])`,
  `pmf_rmse_1d`/`barrier_error_1d` constant-offset cases,
  `identify_basins_1d`'s exact basin boundaries on the symmetric
  double-well fixture, `_compute_basin_populations`'s resulting split,
  `ess`'s two edge cases, and the `_hist_overlap_np` vs
  `_adaptive_hist_overlap` 3-sample/50-sample discrepancy-then-agreement) —
  every one of these was run against the pre-relocation code during this
  plan's authoring and matched exactly what's written into the test files
  above.
