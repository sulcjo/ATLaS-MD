# PMF Bootstrap Uncertainty (Core Algorithm + CV1 Wiring) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, per-bin statistical uncertainty (`pmf_std_kcal_mol`) to the main CV1 PMF that `analyze_gareus_mbar.py` produces, via a fixed-`f_k` block bootstrap that respects replica-trajectory autocorrelation.

**Architecture:** Solve MBAR once (already happening, unchanged). Add a block-id helper that groups samples by `(epoch_source, replica)`. Add a per-window block-bootstrap helper that resamples each window's blocks with replacement, reuses each drawn sample's already-computed weight (no MBAR re-solve), rebuilds the selected PMF method on the resampled set, and reports the per-bin std across replicates. Wire this into `run_pmf_and_gamd_boost_report` (the shared function behind both the main and epoch_000-separate reports) behind a new opt-in flag.

**Tech Stack:** Python, NumPy (`np.random.Generator` for seeded resampling), existing `pmf_from_weights`/`_cumulant_expansion`/`norm_logw` functions in `analyze_gareus_mbar.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-pmf-bootstrap-uncertainty-design.md`

This is Plan 1 of 4 (per the spec's scope decomposition, agreed during brainstorming). This plan covers the block-id helper, the 1D and 2D bootstrap-uncertainty helpers, the CLI flags, and wiring into the main CV1 PMF only. Plan 2 extends to Rg and secondary-CV2 (1D). Plan 3 extends to the four 2D-FES families (using this plan's 2D helper, built but not yet wired here). Plan 4 adds plotting bands and the `pmf_summary.json` scalar.

## Global Constraints

- Method: fixed-`f_k` block bootstrap (Approach B in the spec) — never re-solve MBAR per replicate.
- Block definition: one block = one `(epoch_source, replica)` pair; falls back to `replica` alone when `d.meta['_epoch_source']` is absent.
- Every bootstrap replicate must be shifted so it reads exactly `0` at the SAME bin index the main PMF's own minimum uses — never each replicate's own minimum.
- Feature is opt-in via `--pmf-uncertainty` (default off). With the flag off, every touched function's output must be byte-for-byte unchanged from before this plan.
- Uncertainty is computed only for the `selected` (headline) PMF method, not all four estimator variants.
- Seeded via `--pmf-uncertainty-seed` (default `0`) for reproducibility.
- A window with fewer than 3 blocks cannot be meaningfully block-bootstrapped; this must be surfaced as a warning, not silently reported as a precise error bar.

---

### Task 1: Per-sample block-id helper

**Files:**
- Modify: `analyze_gareus_mbar.py` — add `_sample_block_ids` immediately after `_filter_epoch_source` (currently ends around line 2365; use `grep -n "^def _filter_epoch_source"` to find the current location, since line numbers shift as earlier plans/PRs land).
- Test: `tests/test_pmf_bootstrap_uncertainty.py` (new file)

**Interfaces:**
- Produces: `_sample_block_ids(d: Data) -> np.ndarray` — an `int64` array, same length as `d.cv`/`d.window`/`d.replica`, giving each sample a compact `0..M-1` block id. Used by Task 2 and Task 3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pmf_bootstrap_uncertainty.py`:

```python
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze_gareus_mbar as agm


def _make_data_stub(replica, epoch_source=None):
    """Minimal object exposing only the fields _sample_block_ids reads."""
    class _Stub:
        pass
    d = _Stub()
    d.replica = np.asarray(replica)
    d.meta = {}
    if epoch_source is not None:
        d.meta['_epoch_source'] = list(epoch_source)
    return d


def test_block_ids_group_by_replica_when_no_epoch_source():
    d = _make_data_stub(replica=[0, 0, 1, 1, 2])
    block_ids = agm._sample_block_ids(d)
    # Same replica -> same block id; different replica -> different block id.
    assert block_ids[0] == block_ids[1]
    assert block_ids[2] == block_ids[3]
    assert len({block_ids[0], block_ids[2], block_ids[4]}) == 3


def test_block_ids_distinguish_same_replica_across_epoch_sources():
    # Replica 0 in epoch source 0 and replica 0 in epoch source 1 must be
    # DIFFERENT blocks -- adaptive-production runs don't guarantee trajectory
    # continuity across epoch boundaries.
    d = _make_data_stub(replica=[0, 0, 0, 0], epoch_source=[0, 0, 1, 1])
    block_ids = agm._sample_block_ids(d)
    assert block_ids[0] == block_ids[1]
    assert block_ids[2] == block_ids[3]
    assert block_ids[0] != block_ids[2]


def test_block_ids_are_compact_zero_based():
    d = _make_data_stub(replica=[5, 5, 9, 9, 5])
    block_ids = agm._sample_block_ids(d)
    assert set(np.unique(block_ids)) == {0, 1}
    assert block_ids.dtype == np.int64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v`
Expected: FAIL with `AttributeError: module 'analyze_gareus_mbar' has no attribute '_sample_block_ids'`

- [ ] **Step 3: Write minimal implementation**

Find `_filter_epoch_source` (`grep -n "^def _filter_epoch_source" analyze_gareus_mbar.py`) and add immediately after its closing line:

```python
def _sample_block_ids(d: 'Data') -> np.ndarray:
    """Per-sample block id for the fixed-f_k block bootstrap (see
    docs/superpowers/specs/2026-08-13-pmf-bootstrap-uncertainty-design.md):
    one block = one (epoch_source, replica) pair, matching one replica's
    samples within one epoch/phase -- treating the same replica index reused
    in a later epoch as a NEW block, since adaptive-production runs don't
    guarantee trajectory continuity across epoch boundaries. Falls back to
    `replica` alone when `d.meta['_epoch_source']` is absent (single-source/
    non-adaptive-production runs).

    Returns a compact 0..M-1 int64 array, same length as d.replica.
    """
    replica = np.asarray(d.replica)
    epoch_src = d.meta.get('_epoch_source')
    if epoch_src is None:
        keys = replica.reshape(-1, 1)
    else:
        keys = np.stack([np.asarray(epoch_src), replica], axis=1)
    _, block_ids = np.unique(keys, axis=0, return_inverse=True)
    return block_ids.reshape(-1).astype(np.int64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add analyze_gareus_mbar.py tests/test_pmf_bootstrap_uncertainty.py
git commit -m "feat: add per-sample block-id helper for PMF bootstrap uncertainty"
```

---

### Task 2: 1D fixed-`f_k` block bootstrap helper

**Files:**
- Modify: `analyze_gareus_mbar.py` — add `_bootstrap_pmf_uncertainty_1d` immediately after `_cumulant_expansion_2d_both` (`grep -n "^def _cumulant_expansion_2d_both"` — its body ends where `cumulant2_2d` begins).
- Test: `tests/test_pmf_bootstrap_uncertainty.py`

**Interfaces:**
- Consumes: `_sample_block_ids` (Task 1, for building test fixtures — not called internally by this function, which takes `block_ids` as a parameter so it can be computed once and reused across every analysis in a run); `pmf_from_weights(cv, w, bins, kbt_kcal)`, `_cumulant_expansion(cv, base_w, boost, bins, beta, kbt_kcal, order)`, `norm_logw(lw)` — all pre-existing.
- Produces: `_bootstrap_pmf_uncertainty_1d(cv, logw, boost, bins, beta, kbt_kcal, window, block_ids, selected_method, main_pmf, n_boot, rng) -> dict` with keys `pmf_std` (`ndarray`, length `len(bins)-1`), `blocks_per_window` (`ndarray`, length `K`), `low_block_windows` (`list[int]`). Used by Task 5 and (as the 1D reference in a cross-check) by Task 3's test.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pmf_bootstrap_uncertainty.py`:

```python
KBT_KCAL = 0.596  # ~300K, matches the file's own documented round-trip check


def _synthetic_blocked_cv(rng, n_blocks=5, samples_per_block=200,
                           between_block_std=1.0, within_block_std=0.1):
    """One window's worth of samples: `n_blocks` blocks, each block a tight
    cluster around its own randomly-offset center. between_block_std >>
    within_block_std means the block structure carries real information a
    naive per-sample bootstrap would miss."""
    block_offsets = rng.normal(0.0, between_block_std, size=n_blocks)
    cv_parts, block_id_parts = [], []
    for b in range(n_blocks):
        cv_parts.append(block_offsets[b] + rng.normal(0.0, within_block_std, size=samples_per_block))
        block_id_parts.append(np.full(samples_per_block, b, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    block_ids = np.concatenate(block_id_parts)
    window = np.zeros_like(cv, dtype=np.int64)
    logw = np.zeros_like(cv)  # uniform weight
    boost = np.full_like(cv, np.nan)  # no GaMD boost -> exercise umbrella_only
    return cv, block_ids, window, logw, boost


def _naive_per_sample_bootstrap_std(cv, logw, bins, n_boot, rng):
    """Reference implementation for the block-bootstrap validity test only
    -- NOT the production code path. Resamples individual samples, ignoring
    block structure entirely."""
    n = cv.size
    reps = np.full((n_boot, len(bins) - 1), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        w = agm.norm_logw(logw[idx])
        reps[b] = agm.pmf_from_weights(cv[idx], w, bins, KBT_KCAL)['pmf']
    return np.nanstd(reps, axis=0)


def test_block_bootstrap_reports_wider_uncertainty_than_naive_per_sample():
    rng = np.random.default_rng(1)
    cv, block_ids, window, logw, boost = _synthetic_blocked_cv(rng)
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    main_pmf = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)

    block_std = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=200, rng=np.random.default_rng(0),
    )['pmf_std']
    naive_std = _naive_per_sample_bootstrap_std(cv, logw, bins, n_boot=200, rng=np.random.default_rng(0))

    finite = np.isfinite(block_std) & np.isfinite(naive_std)
    assert finite.sum() >= 5
    assert np.mean(block_std[finite]) > np.mean(naive_std[finite])


def test_bootstrap_replicates_anchor_to_main_pmf_minimum_not_their_own():
    rng = np.random.default_rng(2)
    cv, block_ids, window, logw, boost = _synthetic_blocked_cv(rng)
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    main_pmf = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)
    minidx = int(np.nanargmin(main_pmf['pmf']))

    result = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=100, rng=np.random.default_rng(0),
    )
    # By construction every replicate is re-anchored to read exactly 0 at
    # the bin the MAIN pmf calls its minimum -- so the std AT THAT BIN must
    # be exactly 0. If replicates were instead anchored to their own
    # individual minima, this would generally be nonzero.
    assert result['pmf_std'][minidx] == pytest.approx(0.0, abs=1e-12)


def test_low_block_count_window_is_flagged():
    rng = np.random.default_rng(3)
    cv0, block_ids0, _, logw0, boost0 = _synthetic_blocked_cv(rng, n_blocks=5)
    cv1, block_ids1, _, logw1, boost1 = _synthetic_blocked_cv(rng, n_blocks=1, samples_per_block=50)
    cv = np.concatenate([cv0, cv1 + 10.0])  # push window-1 samples into their own CV range
    block_ids = np.concatenate([block_ids0, block_ids1 + 1000])  # keep block ids globally distinct
    window = np.concatenate([np.zeros_like(cv0, dtype=np.int64), np.ones_like(cv1, dtype=np.int64)])
    logw = np.concatenate([logw0, logw1])
    boost = np.concatenate([boost0, boost1])
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    main_pmf = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)

    result = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=50, rng=np.random.default_rng(0),
    )
    assert result['blocks_per_window'].tolist() == [5, 1]
    assert result['low_block_windows'] == [1]


def test_bootstrap_uncertainty_is_deterministic_given_same_seed():
    rng = np.random.default_rng(4)
    cv, block_ids, window, logw, boost = _synthetic_blocked_cv(rng)
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    main_pmf = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)

    result_a = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=50, rng=np.random.default_rng(7),
    )['pmf_std']
    result_b = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=50, rng=np.random.default_rng(7),
    )['pmf_std']
    result_c = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf, n_boot=50, rng=np.random.default_rng(8),
    )['pmf_std']
    assert np.array_equal(result_a, result_b, equal_nan=True)
    assert not np.array_equal(result_a, result_c, equal_nan=True)


def test_bootstrap_handles_gamd_cumulant2_selected_method():
    rng = np.random.default_rng(5)
    cv, block_ids, window, _, _ = _synthetic_blocked_cv(rng)
    logw = np.zeros_like(cv)
    boost = rng.normal(50.0, 8.0, size=cv.size)  # realistic GaMD boost scale, kJ/mol
    bins = agm.make_bins(cv, 15, None, None)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    base_w = agm.norm_logw(logw)
    main_pmf, _ = agm._cumulant_expansion(cv, base_w, boost, bins, beta, KBT_KCAL, order=2)

    result = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'gamd_cumulant2', main_pmf, n_boot=30, rng=np.random.default_rng(0),
    )
    assert result['pmf_std'].shape == main_pmf['pmf'].shape
    assert np.any(np.isfinite(result['pmf_std']))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v -k bootstrap`
Expected: FAIL with `AttributeError: module 'analyze_gareus_mbar' has no attribute '_bootstrap_pmf_uncertainty_1d'`

- [ ] **Step 3: Write minimal implementation**

Find `_cumulant_expansion_2d_both` (`grep -n "^def _cumulant_expansion_2d_both"`) and add immediately after its closing line:

```python
def _bootstrap_pmf_uncertainty_1d(cv, logw, boost, bins, beta, kbt_kcal,
                                   window, block_ids, selected_method,
                                   main_pmf, n_boot, rng):
    """Fixed-f_k block bootstrap uncertainty for a 1D PMF (see
    docs/superpowers/specs/2026-08-13-pmf-bootstrap-uncertainty-design.md).

    For each window, resamples that window's own blocks (see
    `_sample_block_ids`) with replacement -- same number of blocks drawn as
    the window originally had, different composition -- reusing each drawn
    sample's already-computed per-sample log-weight `logw` (f_k stays
    fixed; no MBAR re-solve). Rebuilds `selected_method`'s PMF on the
    resampled set for each of `n_boot` replicates, re-anchors every
    replicate to read exactly 0 at the SAME bin `main_pmf` uses as its own
    minimum (never the replicate's own minimum -- anchoring to each
    replicate's own minimum would artificially erase uncertainty exactly at
    that bin and distort every other bin's uncertainty relative to it), and
    returns the per-bin std across replicates.

    `selected_method` must be one of 'umbrella_only', 'gamd_exponential',
    'gamd_cumulant2', 'gamd_cumulant3' -- the same keys as the `pmfs` dict
    built in `run_pmf_and_gamd_boost_report`.

    Returns {'pmf_std': ndarray (len(bins)-1,),
             'blocks_per_window': ndarray (K,),
             'low_block_windows': list[int]} -- windows with fewer than 3
    blocks, whose contribution to the estimate is unreliable.
    """
    cv = np.asarray(cv, dtype=np.float64)
    logw = np.asarray(logw, dtype=np.float64)
    boost = np.asarray(boost, dtype=np.float64)
    window = np.asarray(window)
    block_ids = np.asarray(block_ids)
    K = int(np.max(window)) + 1 if window.size else 0
    minidx = int(np.nanargmin(main_pmf['pmf']))

    window_block_map = []
    blocks_per_window = np.zeros(K, dtype=np.int64)
    for k in range(K):
        idx_k = np.where(window == k)[0]
        blocks_k = block_ids[idx_k]
        uniq = np.unique(blocks_k)
        blocks_per_window[k] = uniq.size
        grouped = {b: idx_k[blocks_k == b] for b in uniq}
        window_block_map.append((uniq, grouped))
    low_block_windows = [k for k in range(K) if 0 < blocks_per_window[k] < 3]

    n_bins = len(bins) - 1
    reps = np.full((n_boot, n_bins), np.nan, dtype=np.float64)
    for b in range(n_boot):
        resampled_parts = []
        for k in range(K):
            uniq, grouped = window_block_map[k]
            if uniq.size == 0:
                continue
            chosen = rng.choice(uniq, size=uniq.size, replace=True)
            for blk in chosen:
                resampled_parts.append(grouped[blk])
        if not resampled_parts:
            continue
        resampled_idx = np.concatenate(resampled_parts)
        cv_b = cv[resampled_idx]
        logw_b = logw[resampled_idx]
        boost_b = boost[resampled_idx]
        if selected_method == 'umbrella_only':
            w_b = norm_logw(logw_b)
            rep_pmf = pmf_from_weights(cv_b, w_b, bins, kbt_kcal)
        elif selected_method == 'gamd_exponential':
            w_b = norm_logw(logw_b + beta * boost_b)
            rep_pmf = pmf_from_weights(cv_b, w_b, bins, kbt_kcal)
        elif selected_method in ('gamd_cumulant2', 'gamd_cumulant3'):
            base_w_b = norm_logw(logw_b)
            order = 2 if selected_method == 'gamd_cumulant2' else 3
            rep_pmf, _ = _cumulant_expansion(cv_b, base_w_b, boost_b, bins, beta, kbt_kcal, order=order)
        else:
            raise ValueError(f"unknown selected_method {selected_method!r}")
        rep_arr = np.asarray(rep_pmf['pmf'], dtype=np.float64)
        anchor = rep_arr[minidx]
        reps[b] = rep_arr - anchor

    with np.errstate(invalid='ignore'):
        pmf_std = np.nanstd(reps, axis=0)
    return {'pmf_std': pmf_std, 'blocks_per_window': blocks_per_window, 'low_block_windows': low_block_windows}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v -k bootstrap`
Expected: PASS (6 tests). If `test_block_bootstrap_reports_wider_uncertainty_than_naive_per_sample` is flaky (it uses randomness), re-run once — the effect size (between-block std 1.0 vs within-block std 0.1) is large enough that this should pass reliably, but if it doesn't, increase `n_boot` in that test to 400 rather than changing the production code.

- [ ] **Step 5: Commit**

```bash
git add analyze_gareus_mbar.py tests/test_pmf_bootstrap_uncertainty.py
git commit -m "feat: add 1D fixed-f_k block bootstrap PMF uncertainty helper"
```

---

### Task 3: 2D fixed-`f_k` block bootstrap helper

**Files:**
- Modify: `analyze_gareus_mbar.py` — add `_bootstrap_pmf_uncertainty_2d` immediately after `_bootstrap_pmf_uncertainty_1d` (Task 2).
- Test: `tests/test_pmf_bootstrap_uncertainty.py`

**Interfaces:**
- Consumes: `pmf2d_from_weights(x, y, w, xbins, ybins, kbt_kcal)`, `_cumulant_expansion_2d(x, y, base_w, boost, xbins, ybins, beta, kbt_kcal, order)`, `norm_logw` — all pre-existing.
- Produces: `_bootstrap_pmf_uncertainty_2d(x, y, logw, boost, xbins, ybins, beta, kbt_kcal, window, block_ids, selected_method, main_fes, n_boot, rng) -> dict` with keys `pmf_std` (`ndarray`, shape `(Bx, By)`), `blocks_per_window`, `low_block_windows` — same meaning as the 1D helper. Not wired to any call site in this plan (Plan 3 wires it into the four 2D-FES families); this task's test is the only consumer for now, verifying it against the 1D helper.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pmf_bootstrap_uncertainty.py`:

```python
def test_2d_bootstrap_collapses_to_1d_case_with_degenerate_y():
    rng = np.random.default_rng(6)
    cv, block_ids, window, logw, boost = _synthetic_blocked_cv(rng)
    y = np.zeros_like(cv)
    bins = agm.make_bins(cv, 15, None, None)
    ybins = np.array([-0.5, 0.5])
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)

    main_pmf_1d = agm.pmf_from_weights(cv, agm.norm_logw(logw), bins, KBT_KCAL)
    main_fes_2d = agm.pmf2d_from_weights(cv, y, agm.norm_logw(logw), bins, ybins, KBT_KCAL)

    result_1d = agm._bootstrap_pmf_uncertainty_1d(
        cv, logw, boost, bins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_pmf_1d, n_boot=100, rng=np.random.default_rng(42),
    )
    result_2d = agm._bootstrap_pmf_uncertainty_2d(
        cv, y, logw, boost, bins, ybins, beta, KBT_KCAL, window, block_ids,
        'umbrella_only', main_fes_2d, n_boot=100, rng=np.random.default_rng(42),
    )
    # Same seed, same window/block structure -> both helpers draw the exact
    # same sequence of resampled index sets, so the collapsed (single-y-bin)
    # 2D result must match the 1D result bit-for-bit.
    assert np.allclose(result_2d['pmf_std'][:, 0], result_1d['pmf_std'], atol=1e-10, equal_nan=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v -k collapses`
Expected: FAIL with `AttributeError: module 'analyze_gareus_mbar' has no attribute '_bootstrap_pmf_uncertainty_2d'`

- [ ] **Step 3: Write minimal implementation**

```python
def _bootstrap_pmf_uncertainty_2d(x, y, logw, boost, xbins, ybins, beta, kbt_kcal,
                                   window, block_ids, selected_method,
                                   main_fes, n_boot, rng):
    """2D counterpart of `_bootstrap_pmf_uncertainty_1d`; see that
    docstring. Iterates windows/blocks identically, so with the same `rng`
    state and the same `window`/`block_ids`, it draws the exact same
    resampled index sets per replicate as the 1D helper -- verified by the
    degenerate-y collapse test.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    logw = np.asarray(logw, dtype=np.float64)
    boost = np.asarray(boost, dtype=np.float64)
    window = np.asarray(window)
    block_ids = np.asarray(block_ids)
    K = int(np.max(window)) + 1 if window.size else 0
    main_arr = np.asarray(main_fes['pmf'], dtype=np.float64)
    minidx_flat = int(np.nanargmin(main_arr.ravel()))
    minidx = np.unravel_index(minidx_flat, main_arr.shape)

    window_block_map = []
    blocks_per_window = np.zeros(K, dtype=np.int64)
    for k in range(K):
        idx_k = np.where(window == k)[0]
        blocks_k = block_ids[idx_k]
        uniq = np.unique(blocks_k)
        blocks_per_window[k] = uniq.size
        grouped = {b: idx_k[blocks_k == b] for b in uniq}
        window_block_map.append((uniq, grouped))
    low_block_windows = [k for k in range(K) if 0 < blocks_per_window[k] < 3]

    Bx, By = main_arr.shape
    reps = np.full((n_boot, Bx, By), np.nan, dtype=np.float64)
    for b in range(n_boot):
        resampled_parts = []
        for k in range(K):
            uniq, grouped = window_block_map[k]
            if uniq.size == 0:
                continue
            chosen = rng.choice(uniq, size=uniq.size, replace=True)
            for blk in chosen:
                resampled_parts.append(grouped[blk])
        if not resampled_parts:
            continue
        resampled_idx = np.concatenate(resampled_parts)
        x_b = x[resampled_idx]
        y_b = y[resampled_idx]
        logw_b = logw[resampled_idx]
        boost_b = boost[resampled_idx]
        if selected_method == 'umbrella_only':
            w_b = norm_logw(logw_b)
            rep_fes = pmf2d_from_weights(x_b, y_b, w_b, xbins, ybins, kbt_kcal)
        elif selected_method == 'gamd_exponential':
            w_b = norm_logw(logw_b + beta * boost_b)
            rep_fes = pmf2d_from_weights(x_b, y_b, w_b, xbins, ybins, kbt_kcal)
        elif selected_method in ('gamd_cumulant2', 'gamd_cumulant3'):
            base_w_b = norm_logw(logw_b)
            order = 2 if selected_method == 'gamd_cumulant2' else 3
            rep_fes, _ = _cumulant_expansion_2d(x_b, y_b, base_w_b, boost_b, xbins, ybins, beta, kbt_kcal, order=order)
        else:
            raise ValueError(f"unknown selected_method {selected_method!r}")
        rep_arr = np.asarray(rep_fes['pmf'], dtype=np.float64)
        anchor = rep_arr[minidx]
        reps[b] = rep_arr - anchor

    with np.errstate(invalid='ignore'):
        pmf_std = np.nanstd(reps, axis=0)
    return {'pmf_std': pmf_std, 'blocks_per_window': blocks_per_window, 'low_block_windows': low_block_windows}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v -k collapses`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyze_gareus_mbar.py tests/test_pmf_bootstrap_uncertainty.py
git commit -m "feat: add 2D fixed-f_k block bootstrap PMF uncertainty helper"
```

---

### Task 4: CLI flags

**Files:**
- Modify: `analyze_gareus_mbar.py`, `parse_args()` (`grep -n "^def parse_args"`) — add immediately after the existing `p.add_argument('--min-neighbor-overlap', ...)` line (`grep -n "min-neighbor-overlap"`).
- Test: `tests/test_pmf_bootstrap_uncertainty.py`

**Interfaces:**
- Produces: `args.pmf_uncertainty` (bool, default `False`), `args.pmf_uncertainty_n_boot` (int, default `100`), `args.pmf_uncertainty_seed` (int, default `0`). Consumed by Task 5.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pmf_bootstrap_uncertainty.py`:

```python
def test_pmf_uncertainty_flags_default_off_and_parse_correctly():
    args = agm.parse_args(['some_run_dir'])
    assert args.pmf_uncertainty is False
    assert args.pmf_uncertainty_n_boot == 100
    assert args.pmf_uncertainty_seed == 0

    args_on = agm.parse_args(['some_run_dir', '--pmf-uncertainty',
                               '--pmf-uncertainty-n-boot', '250',
                               '--pmf-uncertainty-seed', '7'])
    assert args_on.pmf_uncertainty is True
    assert args_on.pmf_uncertainty_n_boot == 250
    assert args_on.pmf_uncertainty_seed == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v -k flags_default_off`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'pmf_uncertainty'` (or an argparse error if `--pmf-uncertainty` is unrecognized)

- [ ] **Step 3: Write minimal implementation**

Find the `--min-neighbor-overlap` line and add immediately after it:

```python
    p.add_argument('--pmf-uncertainty', action='store_true', help='Compute per-bin statistical uncertainty (std, kcal/mol) for the selected/headline PMF via a fixed-f_k block bootstrap (blocks = one replica within one epoch/phase). Off by default: adds real compute cost (roughly n_boot resample-and-rebuild passes) with no MBAR re-solve.')
    p.add_argument('--pmf-uncertainty-n-boot', type=int, default=100, help='Number of block-bootstrap replicates for --pmf-uncertainty. Higher is more precise but slower; below ~20 the uncertainty-of-the-uncertainty is itself noisy.')
    p.add_argument('--pmf-uncertainty-seed', type=int, default=0, help='Seed for --pmf-uncertainty block resampling, for reproducible error bars across runs.')
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v -k flags_default_off`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add analyze_gareus_mbar.py tests/test_pmf_bootstrap_uncertainty.py
git commit -m "feat: add --pmf-uncertainty CLI flags"
```

---

### Task 5: Wire uncertainty into the main CV1 PMF report

**Files:**
- Modify: `analyze_gareus_mbar.py`, `run_pmf_and_gamd_boost_report` (`grep -n "^def run_pmf_and_gamd_boost_report"`).
- Test: `tests/test_pmf_bootstrap_uncertainty.py`

**Interfaces:**
- Consumes: `_sample_block_ids` (Task 1), `_bootstrap_pmf_uncertainty_1d` (Task 2), `args.pmf_uncertainty`/`args.pmf_uncertainty_n_boot`/`args.pmf_uncertainty_seed` (Task 4).
- Produces: `run_pmf_and_gamd_boost_report(...)`'s returned dict gains a new key `'pmf_uncertainty_std'` (`Optional[np.ndarray]`, `None` when the flag is off). The written `pmf_unbiased.csv` gains an optional `pmf_std_kcal_mol` column (via the function's existing `extra=` dict mechanism on `write_pmf` — no changes needed to `write_pmf` itself).

Since `run_pmf_and_gamd_boost_report` is the shared function behind BOTH the main report and the `epoch_000_separate/` report (see `analyze()`, both call sites use the same function), this one change covers both automatically.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pmf_bootstrap_uncertainty.py`. These build a minimal real `Data` object and a temp output directory, so they exercise the actual integration point rather than the helpers in isolation:

```python
import csv as _csv
import tempfile
from pathlib import Path as _Path


def _make_real_data(tmp_path, rng, n_windows=3, samples_per_window=300, n_blocks_per_window=5):
    """A minimal but real analyze_gareus_mbar.Data with a genuine multi-window
    harmonic-umbrella u_nk, suitable for a real solve_mbar + run_pmf_and_gamd_boost_report call."""
    centers = np.linspace(-1.0, 1.0, n_windows)
    k_kcal = np.full(n_windows, 5.0)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)

    cv_parts, window_parts, replica_parts, epoch_src_parts = [], [], [], []
    for k in range(n_windows):
        cv_parts.append(centers[k] + rng.normal(0.0, 0.3, size=samples_per_window))
        window_parts.append(np.full(samples_per_window, k, dtype=np.int64))
        # Spread this window's samples across n_blocks_per_window (replica, epoch_source=0) blocks.
        replica_parts.append(np.repeat(np.arange(n_blocks_per_window), samples_per_window // n_blocks_per_window))
        epoch_src_parts.append(np.zeros(samples_per_window, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(window_parts)
    replica = np.concatenate([p[:len(w)] for p, w in zip(replica_parts, window_parts)])
    epoch_src = np.concatenate(epoch_src_parts)
    n = cv.size

    u_nk = np.zeros((n, n_windows), dtype=np.float64)
    for k in range(n_windows):
        u_nk[:, k] = beta * agm.KJ_PER_KCAL * 0.5 * k_kcal[k] * (cv - centers[k]) ** 2

    out_dir = _Path(tmp_path) / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)
    d = agm.Data(
        prod_dir=_Path(tmp_path), out_dir=out_dir, cv=cv, cv2=np.full(n, np.nan),
        rg_A=np.full(n, np.nan), window=window, replica=replica, step=np.arange(n),
        u_nk=u_nk, centers=centers, k_kcal=k_kcal, beta=beta, temp=300.0,
        boost_kj=np.full(n, np.nan), potential_kj=None, source='test',
        meta={'_epoch_source': epoch_src.tolist()},
    )
    return d


class _Args:
    def __init__(self, **kw):
        self.bins = 20
        self.min_neighbor_overlap = 0.0
        self.selected_method = 'auto'
        self.gamd_smooth_sigma = 0.0
        self.pmf_smooth_sigma = 0.0
        self.pmf_uncertainty = False
        self.pmf_uncertainty_n_boot = 30
        self.pmf_uncertainty_seed = 0
        self.__dict__.update(kw)


def test_pmf_uncertainty_off_by_default_leaves_csv_unchanged(tmp_path):
    rng = np.random.default_rng(10)
    d = _make_real_data(tmp_path, rng)
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    args_off = _Args(pmf_uncertainty=False)
    info_off = agm.run_pmf_and_gamd_boost_report(d, args_off, logw, bins, kbt_kcal, d.out_dir, [], None)
    assert info_off.get('pmf_uncertainty_std') is None

    with (d.out_dir / 'pmf_unbiased.csv').open() as f:
        header = next(_csv.reader(f))
    assert 'pmf_std_kcal_mol' not in header


def test_pmf_uncertainty_on_adds_std_column_and_return_value(tmp_path):
    rng = np.random.default_rng(11)
    d = _make_real_data(tmp_path, rng)
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    args_on = _Args(pmf_uncertainty=True)
    warnings = []
    info_on = agm.run_pmf_and_gamd_boost_report(d, args_on, logw, bins, kbt_kcal, d.out_dir, warnings, None)
    assert info_on.get('pmf_uncertainty_std') is not None
    assert info_on['pmf_uncertainty_std'].shape == info_on['pmfs'][info_on['selected']]['pmf'].shape

    with (d.out_dir / 'pmf_unbiased.csv').open() as f:
        header = next(_csv.reader(f))
    assert 'pmf_std_kcal_mol' in header


def test_pmf_uncertainty_warns_on_low_block_count_windows(tmp_path):
    rng = np.random.default_rng(12)
    # n_blocks_per_window=1 -> every window is a "low block count" window.
    d = _make_real_data(tmp_path, rng, n_blocks_per_window=1)
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    args_on = _Args(pmf_uncertainty=True)
    warnings = []
    agm.run_pmf_and_gamd_boost_report(d, args_on, logw, bins, kbt_kcal, d.out_dir, warnings, None)
    assert any('uncertainty' in w.lower() and 'block' in w.lower() for w in warnings)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v -k pmf_uncertainty_o`
Expected: FAIL — `info_off.get('pmf_uncertainty_std')`/`info_on.get(...)` will be `None` in both cases (key doesn't exist yet) and the CSV will never contain `pmf_std_kcal_mol`, so `test_pmf_uncertainty_on_adds_std_column_and_return_value` and `test_pmf_uncertainty_warns_on_low_block_count_windows` fail; `test_pmf_uncertainty_off_by_default_leaves_csv_unchanged` passes vacuously already (nothing to break yet) — that's fine, it becomes a real regression guard once Step 3 lands.

- [ ] **Step 3: Write minimal implementation**

In `run_pmf_and_gamd_boost_report`, find the line `sel = pmfs[selected]` (comes right after the `_force_method` block) and, immediately before the existing `write_pmf(out / 'pmf_unbiased.csv', sel, selected, {...})` line, insert:

```python
    pmf_uncertainty_std = None
    _uncertainty_extra = {}
    if getattr(args, 'pmf_uncertainty', False):
        block_ids = _sample_block_ids(d)
        boot_rng = np.random.default_rng(int(getattr(args, 'pmf_uncertainty_seed', 0)))
        n_boot = int(getattr(args, 'pmf_uncertainty_n_boot', 100))
        boot_result = _bootstrap_pmf_uncertainty_1d(
            d.cv, logw, d.boost_kj, bins, d.beta, kbt_kcal, d.window, block_ids,
            selected, sel, n_boot, boot_rng,
        )
        pmf_uncertainty_std = boot_result['pmf_std']
        _uncertainty_extra = {'pmf_std_kcal_mol': pmf_uncertainty_std}
        if boot_result['low_block_windows']:
            warnings.append(f"{warning_prefix}PMF uncertainty is unreliable near windows {boot_result['low_block_windows']} (fewer than 3 independent trajectory blocks).")
```

Then change the existing `write_pmf(out / 'pmf_unbiased.csv', sel, selected, {...})` call to merge `_uncertainty_extra` into its `extra` dict:

```python
    write_pmf(out / 'pmf_unbiased.csv', sel, selected, {'boost_mean_kj_mol': _sel_diag.get('boost_mean_kj', np.full(args.bins, np.nan)), 'boost_var_kj2_mol2': _sel_diag.get('boost_var_kj2', np.full(args.bins, np.nan)), **_uncertainty_extra})
```

Finally, add `pmf_uncertainty_std` to the function's returned dict (in the `return { ... }` block at the end of the function):

```python
    return {
        'pmfs': pmfs, 'selected': selected, 'boost_ok': boost_ok, 'boost': bs,
        'pmf_span_kcal_mol': span, 'pmf_minimum_cv_A': float(sel['cv_A'][minidx]) if minidx >= 0 else None,
        'neighbor_overlap': neigh, 'n_samples': N, 'O': O,
        'pmf_uncertainty_std': pmf_uncertainty_std,
        'files': {
            'pmf_unbiased_csv': str(out / 'pmf_unbiased.csv'), 'pmf_all_methods_csv': str(out / 'pmf_all_methods.csv'),
            'pmf_umbrella_only_csv': str(out / 'pmf_umbrella_only.csv'), 'pmf_gamd_exponential_csv': str(out / 'pmf_gamd_exponential.csv'),
            'pmf_gamd_cumulant2_csv': str(out / 'pmf_gamd_cumulant2.csv'), 'pmf_gamd_cumulant3_csv': str(out / 'pmf_gamd_cumulant3.csv'),
            'overlap_matrix_csv': str(out / 'overlap_matrix.csv'), 'window_diagnostics_csv': str(out / 'window_diagnostics.csv'),
        },
    }
```

(Only the `'pmf_uncertainty_std': pmf_uncertainty_std,` line is new; the rest of the `return` block is shown for placement context — do not otherwise change it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pmf_bootstrap_uncertainty.py -v`
Expected: PASS (all tests in the file, including every test from Tasks 1-4)

- [ ] **Step 5: Run the full pre-existing test suite for this function to confirm no regression**

Run: `pytest -q tests/test_epoch0_pmf_gamd_split.py tests/test_masked_logw_subset_pmf.py tests/test_cumulant_shared_computation.py tests/test_pmf_gamd_nan_bin_handling.py tests/test_perf_report_redundancy.py`
Expected: PASS — these are the existing test files that exercise `run_pmf_and_gamd_boost_report`; none of them pass `--pmf-uncertainty`/set `args.pmf_uncertainty=True`, so this confirms the off-by-default path is unaffected. If any of these fail, do not proceed — the most likely cause is an `AttributeError` from a test's own `_Args`-style stub not having the three new attributes; the fix is `getattr(args, 'pmf_uncertainty', False)` (already used above) doing its job everywhere `args.pmf_uncertainty` is read, never a bare `args.pmf_uncertainty`.

- [ ] **Step 6: Compile check**

Run: `python -m py_compile analyze_gareus_mbar.py`
Expected: no output (clean compile)

- [ ] **Step 7: Commit**

```bash
git add analyze_gareus_mbar.py tests/test_pmf_bootstrap_uncertainty.py
git commit -m "feat: wire --pmf-uncertainty into the main CV1 PMF report"
```

---

## Plan Self-Review Notes

- **Spec coverage**: block definition (Task 1), fixed-f_k resampling algorithm + shift-anchor rule (Task 2), 2D twin (Task 3), CLI flags (Task 4), CV1 wiring + output column + low-block warning (Task 5) all covered. Deferred to later plans per the agreed scope split: Rg/secondary-CV2 wiring (Plan 2), 2D-FES family wiring (Plan 3), plot bands + `pmf_summary.json` scalar (Plan 4) — none of these are silently dropped, they're explicitly out of scope for Plan 1.
- **Type/signature consistency checked**: `_bootstrap_pmf_uncertainty_1d`'s and `_2d`'s parameter order and the `{'pmf_std', 'blocks_per_window', 'low_block_windows'}` return-dict keys are identical between Task 2, Task 3, and their use in Task 5 and the Task 3 test. `run_pmf_and_gamd_boost_report`'s new `pmf_uncertainty_std` return key name matches what Task 5's own tests assert on.
- **No placeholders**: every step has real, complete code; no "add appropriate handling"-style steps.
