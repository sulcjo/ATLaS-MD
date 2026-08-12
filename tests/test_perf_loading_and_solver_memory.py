"""Regression tests for confirmed-safe performance fixes to
analyze_gareus_mbar.py (a performance/dtype audit's lead findings):

  Fix 1a: solve_mbar_* backends (and _subset_logw_from_global_fk) skip the
          u_nk[:, active] fancy-index copy when `active` already covers every
          window -- the common case, since nothing ever actually gets
          dropped in practice.
  Fix 1b: clean() skips its mask-copy of every per-sample field when the
          finite-mask is already all-True, but must still run the
          boost_dih_kj shape-normalization regardless (that check is
          independent of sample finiteness).
  Fix 1c: load_parquet_adaptive_union frees each per-epoch block list right
          after concatenation (not directly unit-testable as a memory fact;
          see the note at the bottom of this file).
  Fix 2:  per-sample Python dict-lookup loops (wmap.get(int(w), -1),
          state_id_to_k[int(s)], ...) replaced with a vectorized dense
          lookup table.
  Fix 3:  window/replica columns kept at their on-disk-matching narrow dtype
          (int16) instead of being widened to int32/int64 on load.
  Fix 4:  cast-then-filter reordered to filter-then-cast for cv1/cv2/step in
          load_parquet_adaptive_union, avoiding a full-epoch-length float64
          transient that's mostly discarded by the very next line.

Every fix is meant to be zero-behavior-change. Where a literal "before vs
after" comparison isn't available (the "before" code no longer exists after
editing in place), these tests instead cross-check two independent code
paths that must agree if the fix didn't change any actual arithmetic (e.g.
two different MBAR backends on the same problem, or a narrow-dtype vs
wide-dtype window array run through the same solver).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

agm = pytest.importorskip(
    "analyze_gareus_mbar",
    reason="estimator module (numpy-only) must be importable from repo root",
)

KB_KCAL = 0.0019872041  # Boltzmann constant, kcal/(mol*K)
TEMP_K = 300.0
KT = KB_KCAL * TEMP_K
BETA = 1.0 / KT


def _generate_umbrella_oracle(k0, k_bias, centers, n_per_window, beta, seed):
    """Same harmonic-umbrella oracle construction as
    test_mbar_lbfgs_convergence.py / test_physics_oracle.py -- a real,
    non-degenerate MBAR problem (not a toy with a trivial optimum)."""
    rng = np.random.RandomState(seed)
    centers = np.asarray(centers, dtype=np.float64)
    K = centers.size
    sigma = np.sqrt(1.0 / (beta * (k0 + k_bias)))
    means = k_bias * centers / (k0 + k_bias)
    win_parts = []
    cv_parts = []
    for i in range(K):
        cv_parts.append(rng.normal(means[i], sigma, size=n_per_window))
        win_parts.append(np.full(n_per_window, i, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(win_parts)
    diff = cv[:, None] - centers[None, :]
    u_nk = beta * 0.5 * k_bias * diff * diff
    return window, u_nk


def _all_active_problem():
    centers = np.linspace(-2.0, 2.0, 9)
    return _generate_umbrella_oracle(2.0, 10.0, centers, 400, BETA, seed=7)


# ---------------------------------------------------------------------------
# Fix 1a -- solve_mbar_* backends' u_nk[:, active] fast path.
# ---------------------------------------------------------------------------

def test_solve_mbar_all_active_anderson_matches_lbfgs():
    """Every window has samples (the fast-path case: active == arange(K)).
    Two independent solvers must agree -- if either fast path silently used
    the wrong columns, they wouldn't."""
    window, u_nk = _all_active_problem()
    K = int(window.max()) + 1
    assert np.unique(window).size == K  # sanity: genuinely all-active

    res_anderson = agm.solve_mbar(u_nk, window, backend="anderson", tol=1e-12, maxiter=20000)
    assert res_anderson["active"].size == K
    assert np.array_equal(res_anderson["active"], np.arange(K))

    if agm.SCIPY_AVAILABLE:
        res_lbfgs = agm.solve_mbar(u_nk, window, backend="lbfgs", tol=1e-10, maxiter=2000)
        np.testing.assert_allclose(res_anderson["f_k"], res_lbfgs["f_k"], atol=1e-4)


@pytest.mark.skipif(not getattr(agm, "NUMBA_AVAILABLE", False), reason="numba backend not available")
def test_solve_mbar_numba_all_active_matches_anderson():
    window, u_nk = _all_active_problem()
    res_numba = agm.solve_mbar_numba(u_nk, window, tol=1e-12, maxiter=20000)
    res_anderson = agm.solve_mbar(u_nk, window, backend="anderson", tol=1e-12, maxiter=20000)
    np.testing.assert_allclose(res_numba["f_k"], res_anderson["f_k"], atol=1e-4)


@pytest.mark.skipif(not getattr(agm, "NUMBA_AVAILABLE", False), reason="numba backend not available")
def test_solve_mbar_numba_anderson_all_active_matches_plain_anderson():
    window, u_nk = _all_active_problem()
    res = agm.solve_mbar_numba_anderson(u_nk, window, tol=1e-12, maxiter=20000)
    res_anderson = agm.solve_mbar(u_nk, window, backend="anderson", tol=1e-12, maxiter=20000)
    np.testing.assert_allclose(res["f_k"], res_anderson["f_k"], atol=1e-4)


def test_solve_mbar_some_inactive_windows_still_correct():
    """The fast path must NOT trigger when some windows have zero samples:
    `active` must reflect exactly the populated states, and f_k for the
    missing ones must stay NaN -- proving the fast path's guard doesn't
    accidentally fire, and that the (unchanged) slow path still works."""
    window, u_nk = _all_active_problem()
    K = int(window.max()) + 1
    keep = ~np.isin(window, [3, 6])
    window_sub = window[keep]
    u_nk_sub = u_nk[keep]

    res = agm.solve_mbar(u_nk_sub, window_sub, backend="anderson", tol=1e-12, maxiter=20000)
    assert set(res["active"].tolist()) == set(range(K)) - {3, 6}
    assert np.isnan(res["f_k"][3])
    assert np.isnan(res["f_k"][6])
    assert np.all(np.isfinite(res["f_k"][res["active"]]))

    if agm.SCIPY_AVAILABLE:
        res_lbfgs = agm.solve_mbar_lbfgs(u_nk_sub, window_sub, tol=1e-10, maxiter=2000)
        both_finite = np.isfinite(res["f_k"]) & np.isfinite(res_lbfgs["f_k"])
        np.testing.assert_allclose(res["f_k"][both_finite], res_lbfgs["f_k"][both_finite], atol=1e-4)


def test_solve_mbar_sambar_warmstart_all_active_and_partial_active():
    """SAMBAR warm-start is stochastic/not fully converged by construction,
    so this only checks the fast path runs cleanly on all-active data, and
    that the active/inactive split is preserved with a window dropped."""
    window, u_nk = _all_active_problem()
    K = int(window.max()) + 1
    f_full = agm.solve_mbar_sambar_warmstart(u_nk, window, epochs=25, initial_batch_size=200, seed=3)
    assert f_full.shape == (K,)
    assert np.all(np.isfinite(f_full))

    keep = ~np.isin(window, [3])
    f_partial = agm.solve_mbar_sambar_warmstart(
        u_nk[keep], window[keep], epochs=25, initial_batch_size=200, seed=3
    )
    assert np.isnan(f_partial[3])
    assert np.all(np.isfinite(np.delete(f_partial, 3)))


def test_subset_logw_from_global_fk_all_active_fast_path():
    """_subset_logw_from_global_fk's fast path needs BOTH active.size==K and
    u_nk.shape[1]==K (K here comes from f_k_global.size, not from u_nk.shape
    directly, unlike the solve_mbar_* backends above)."""
    window, u_nk = _all_active_problem()
    res = agm.solve_mbar(u_nk, window, backend="anderson", tol=1e-12, maxiter=20000)
    f_k_global = res["f_k"]

    class _Subset:
        pass

    sub = _Subset()
    sub.u_nk = u_nk
    sub.window = window

    logw = agm._subset_logw_from_global_fk(sub, f_k_global)
    assert logw.shape[0] == u_nk.shape[0]
    assert np.all(np.isfinite(logw))
    # Degenerate case (per the function's own docstring): subset == the full
    # population the global f_k was solved from -> this must reduce exactly
    # to that global solve's own (normalized) logw.
    np.testing.assert_allclose(np.exp(logw), agm.norm_logw(res["logw"]), atol=1e-8)


def test_subset_logw_from_global_fk_extra_u_nk_columns_not_shortcut():
    """If u_nk has MORE columns than f_k_global (K), active.size==K must not
    trigger the u_nk-identity shortcut -- that would silently broadcast
    against the wrong (padded) column count instead of the K columns that
    match f_k_global. The extra columns must never affect the result."""
    window, u_nk = _all_active_problem()
    res = agm.solve_mbar(u_nk, window, backend="anderson", tol=1e-12, maxiter=20000)
    f_k_global = res["f_k"]

    pad = np.full((u_nk.shape[0], 2), 1.0e6)
    u_nk_padded = np.concatenate([u_nk, pad], axis=1)

    class _Subset:
        pass

    sub_padded = _Subset()
    sub_padded.u_nk = u_nk_padded
    sub_padded.window = window

    sub_plain = _Subset()
    sub_plain.u_nk = u_nk
    sub_plain.window = window

    logw_padded = agm._subset_logw_from_global_fk(sub_padded, f_k_global)
    logw_plain = agm._subset_logw_from_global_fk(sub_plain, f_k_global)
    np.testing.assert_allclose(logw_padded, logw_plain, atol=1e-10)


# ---------------------------------------------------------------------------
# Fix 1b -- clean()'s all-finite fast path.
# ---------------------------------------------------------------------------

def _make_data(n=6, k=2, boost_dih_size=None):
    cv = np.linspace(-1.0, 1.0, n)
    cv2 = np.zeros(n)
    rg = np.zeros(n)
    window = np.zeros(n, dtype=np.int64)
    replica = np.zeros(n, dtype=np.int64)
    step = np.arange(n, dtype=np.int64)
    u_nk = np.zeros((n, k))
    boost_kj = np.zeros(n)
    boost_dih_kj = np.zeros(n if boost_dih_size is None else boost_dih_size)
    return agm.Data(
        prod_dir=Path("."), out_dir=Path("."),
        cv=cv, cv2=cv2, rg_A=rg, window=window, replica=replica, step=step,
        u_nk=u_nk, centers=np.zeros(k), k_kcal=np.zeros(k),
        beta=BETA, temp=TEMP_K, boost_kj=boost_kj, potential_kj=None,
        source="test", meta={}, boost_dih_kj=boost_dih_kj,
    )


def test_clean_fast_path_preserves_identity_when_all_finite():
    """No copy should happen when nothing needs filtering: every returned
    per-sample array must be the exact same object as before clean()."""
    d = _make_data()
    cv_before, u_nk_before, window_before = d.cv, d.u_nk, d.window
    replica_before, step_before, boost_before = d.replica, d.step, d.boost_kj

    out = agm.clean(d)

    assert out is d
    assert out.cv is cv_before
    assert out.u_nk is u_nk_before
    assert out.window is window_before
    assert out.replica is replica_before
    assert out.step is step_before
    assert out.boost_kj is boost_before


def test_clean_fast_path_still_normalizes_mismatched_boost_dih_kj():
    """The exact ordering hazard flagged for this fix: even when the mask is
    all-True (nothing to filter), a boost_dih_kj whose length doesn't match
    the sample count must still be reset to None -- that normalization is
    independent of sample finiteness and must run regardless of the fast
    path taken above."""
    d = _make_data(n=6, boost_dih_size=4)  # deliberately mismatched length
    out = agm.clean(d)
    assert out.boost_dih_kj is None
    assert out.cv.size == 6  # fast path did NOT filter anything else


def test_clean_fast_path_leaves_correctly_sized_boost_dih_kj_alone():
    d = _make_data(n=6, boost_dih_size=6)
    boost_dih_before = d.boost_dih_kj
    out = agm.clean(d)
    assert out.boost_dih_kj is boost_dih_before


def test_clean_slow_path_filters_non_finite_rows():
    """Regression guard: the ordinary (some non-finite samples) path must
    still filter exactly as before."""
    d = _make_data(n=6)
    d.cv[2] = np.nan
    d.u_nk[4, 0] = np.inf
    out = agm.clean(d)
    assert out.cv.size == 4
    assert np.all(np.isfinite(out.cv))
    assert np.all(np.isfinite(out.u_nk))
    assert out.window.size == 4
    assert out.boost_dih_kj.size == 4


def test_clean_slow_path_boost_dih_kj_mismatch_still_dropped_to_none():
    d = _make_data(n=6, boost_dih_size=4)
    d.cv[2] = np.nan
    out = agm.clean(d)
    assert out.boost_dih_kj is None
    assert out.cv.size == 5


# ---------------------------------------------------------------------------
# Fix 2 -- vectorized dict-lookup replacements.
# ---------------------------------------------------------------------------

def test_vectorized_map_lookup_matches_naive_with_missing_key():
    mapping = {0: 10, 1: 11, 5: 15}
    arr = np.array([0, 1, 2, 5, 99])  # 2 and 99 are missing keys
    expected = np.array([mapping.get(int(x), -1) for x in arr])
    got = agm._vectorized_map_lookup(arr, mapping, default=-1, dtype=np.int32)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.int32


def test_vectorized_map_lookup_key_larger_than_every_map_key():
    """Exercises the `hi = max(arr.max(), map_keys.max())` LUT sizing."""
    mapping = {0: 1, 1: 2}
    arr = np.array([0, 1, 1000])
    expected = np.array([mapping.get(int(x), -1) for x in arr])
    got = agm._vectorized_map_lookup(arr, mapping, default=-1)
    np.testing.assert_array_equal(got, expected)


def test_vectorized_map_lookup_empty_mapping():
    arr = np.array([0, 1, 2])
    got = agm._vectorized_map_lookup(arr, {}, default=7, dtype=np.int64)
    np.testing.assert_array_equal(got, np.full(3, 7))


def test_vectorized_map_lookup_negative_key_falls_back_correctly():
    mapping = {-1: 99, 0: 1}
    arr = np.array([-1, 0, 2])
    expected = np.array([mapping.get(int(x), -5) for x in arr])
    got = agm._vectorized_map_lookup(arr, mapping, default=-5)
    np.testing.assert_array_equal(got, expected)


def test_vectorized_map_lookup_huge_key_falls_back_without_allocating():
    """Beyond the LUT size cap, the naive path must still be used -- and
    still be correct -- rather than allocating a huge dense table."""
    mapping = {0: 1}
    arr = np.array([0, 20_000_000])
    expected = np.array([mapping.get(int(x), -1) for x in arr])
    got = agm._vectorized_map_lookup(arr, mapping, default=-1)
    np.testing.assert_array_equal(got, expected)


def test_vectorized_map_lookup_or_self_matches_naive():
    mapping = {0: 10, 2: 12}
    arr = np.array([0, 1, 2, 3])
    expected = np.array([mapping.get(int(x), int(x)) for x in arr])
    got = agm._vectorized_map_lookup_or_self(arr, mapping)
    np.testing.assert_array_equal(got, expected)


def test_vectorized_map_lookup_or_self_empty_mapping_is_identity():
    arr = np.array([3, 7, 9])
    got = agm._vectorized_map_lookup_or_self(arr, {})
    np.testing.assert_array_equal(got, arr)


def test_vectorized_map_index_matches_naive():
    mapping = {0: 100, 1: 101, 2: 102}
    arr = np.array([2, 0, 1, 2])
    expected = np.array([mapping[int(x)] for x in arr])
    got = agm._vectorized_map_index(arr, mapping, dtype=np.int16)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.int16


def test_vectorized_map_index_missing_key_raises_key_error():
    """state_id_to_k[int(s)] has no default -- a missing key must raise
    KeyError, exactly like plain dict subscripting, not silently produce a
    sentinel that would then be misused as a real window index."""
    mapping = {0: 100, 1: 101}
    arr = np.array([0, 1, 2])  # 2 is missing
    with pytest.raises(KeyError):
        agm._vectorized_map_index(arr, mapping)


# ---------------------------------------------------------------------------
# Fix 3 / Fix 4 -- load_parquet_adaptive_union / load_parquet.
# ---------------------------------------------------------------------------

def _write_epoch(epoch_dir: Path, samples, state_id=0, primary_center=0.0,
                  primary_k=44.3, secondary_center=0.0, secondary_k=0.0):
    """Write one epoch's worth of real Parquet samples + window snapshot +
    epoch_window_map.csv, using the real gareus.store writers (same pattern
    as tests/test_union_mbar_per_epoch_bias.py's _write_epoch)."""
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    epoch_dir.mkdir(parents=True, exist_ok=True)
    reg = SegmentRegistry(epoch_dir)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(epoch_dir).snapshot(
        seg_id, [{"window_id": 0, "center1": primary_center, "k1": primary_k,
                  "center2": secondary_center, "k2": secondary_k}],
        cv1_type="contacts", cv2_type="torsion-pca",
    )
    writer = ParquetSampleWriter(epoch_dir / "samples" / seg_id, flush_rows=1000)
    max_step = 0
    for step, replica, cv1, cv2 in samples:
        writer.write_sample(step, replica, 0, cv1, cv2, -100.0, 5.0, 2.0, 0.4)
        max_step = max(max_step, step)
    writer.close()
    reg.close_segment(seg_id, end_step=max_step)

    (epoch_dir / "epoch_window_map.csv").write_text(
        "epoch_window,state_id,primary_center,primary_k,secondary_center,secondary_k\n"
        f"0,{state_id},{primary_center},{primary_k},{secondary_center},{secondary_k}\n",
        encoding="utf-8",
    )


def _write_registry(adaptive_dir: Path, rows: list) -> None:
    with (adaptive_dir / "state_registry.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "state_id", "active", "created_epoch", "retired_epoch", "parent_state_id",
            "primary_center", "primary_k", "secondary_center", "secondary_k",
            "usable_for_mbar", "burnin_steps",
        ])
        w.writeheader()
        for row in rows:
            w.writerow({
                "active": "True", "created_epoch": 0, "retired_epoch": "",
                "parent_state_id": "", "usable_for_mbar": "True", "burnin_steps": 0,
                **row,
            })


def test_load_parquet_adaptive_union_narrows_dtype_and_matches_wide_solve(tmp_path):
    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir(parents=True)

    samples0 = [(i * 50, i % 3, float(np.sin(i) * 0.3), 0.0) for i in range(20)]
    samples1 = [(i * 50, i % 3, 0.5 + float(np.cos(i) * 0.3), 0.0) for i in range(20)]
    _write_epoch(adaptive_dir / "epoch_000", samples0, state_id=0, primary_center=0.0)
    _write_epoch(adaptive_dir / "epoch_001", samples1, state_id=1, primary_center=0.5)
    _write_registry(adaptive_dir, [
        {"state_id": 0, "primary_center": 0.0, "primary_k": 44.3,
         "secondary_center": 0.0, "secondary_k": 0.0},
        {"state_id": 1, "primary_center": 0.5, "primary_k": 44.3,
         "secondary_center": 0.0, "secondary_k": 0.0},
    ])

    data = agm.load_parquet_adaptive_union(adaptive_dir)

    assert data.cv.size == 40
    # Fix 3: narrowed to int16 (on-disk source is uint16; downstream
    # consumers already defensively re-cast to int64 before use).
    assert data.window.dtype == np.int16
    assert data.replica.dtype == np.int16
    # Fix 4: cv1/cv2/step still end up float64/int64 as before -- only the
    # cast/filter order around [valid] changed, not the resulting dtype.
    assert data.cv.dtype == np.float64
    assert data.cv2.dtype == np.float64
    assert data.step.dtype == np.int64
    assert set(np.unique(data.replica).tolist()) == {0, 1, 2}

    # Real downstream consumer: solve_mbar re-casts window to int64 itself,
    # so the narrowed on-disk dtype must not change the actual MBAR solve.
    res_narrow = agm.solve_mbar(data.u_nk, data.window, backend="anderson", tol=1e-12, maxiter=5000)
    res_wide = agm.solve_mbar(
        data.u_nk, data.window.astype(np.int64), backend="anderson", tol=1e-12, maxiter=5000
    )
    np.testing.assert_allclose(res_narrow["f_k"], res_wide["f_k"], atol=1e-10)
    np.testing.assert_array_equal(res_narrow["n_k"], res_wide["n_k"])


def test_load_parquet_narrows_window_replica_dtype(tmp_path, monkeypatch):
    """Same Fix 3 check for the sibling (non-union) loader, load_parquet --
    mocked at the gareus.query boundary so this doesn't need a full run
    directory (windows/, umbrella_windows.csv, etc.)."""
    import gareus.query as query_mod

    n = 8
    fake_samples = {
        "cv1": np.linspace(-1.0, 1.0, n),
        "cv2": np.zeros(n),
        "window_id": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint16),
        "step": np.arange(n, dtype=np.int64) * 50,
        "replica": np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint16),
        "gamd_boost_total": np.full(n, 5.0),
        "gamd_boost_dihedral": np.full(n, 2.0),
        "potential": np.full(n, -100.0),
    }
    fake_windows = [
        {"center1": -0.5, "k1": 20.0},
        {"center1": 0.5, "k1": 20.0},
    ]

    def fake_reconstruct_bias_matrix(cv, cv2, windows, beta):
        u = np.zeros((cv.size, len(windows)))
        for j, w in enumerate(windows):
            diff = cv - w["center1"]
            u[:, j] = beta * 4.184 * 0.5 * w["k1"] * diff * diff
        return u

    monkeypatch.setattr(query_mod, "load_samples", lambda prod: fake_samples)
    monkeypatch.setattr(query_mod, "load_windows", lambda prod: fake_windows)
    monkeypatch.setattr(query_mod, "reconstruct_bias_matrix", fake_reconstruct_bias_matrix)
    monkeypatch.setattr(agm, "infer_temp_beta", lambda prod, meta: (TEMP_K, BETA))

    d = agm.load_parquet(tmp_path)

    assert d.window.dtype == np.int16
    assert d.replica.dtype == np.int16

    res_narrow = agm.solve_mbar(d.u_nk, d.window, backend="anderson", tol=1e-12, maxiter=5000)
    res_wide = agm.solve_mbar(
        d.u_nk, d.window.astype(np.int64), backend="anderson", tol=1e-12, maxiter=5000
    )
    np.testing.assert_allclose(res_narrow["f_k"], res_wide["f_k"], atol=1e-10)


# ---------------------------------------------------------------------------
# Fix 1c note: load_parquet_adaptive_union frees each per-epoch block list
# (all_cv, all_cv2, all_window, ...) via `del` immediately after its
# np.concatenate call, so the per-epoch copies aren't held alive (pending GC)
# alongside the pooled arrays for the rest of the function. This is a peak
# memory footprint fact, not a value the loader's return type exposes, and
# there's no lightweight way to assert "this list was deleted from the
# function's local namespace after it returned" from outside the function
# without brittle frame/gc introspection that would rot with any unrelated
# refactor. No dedicated test is added for this one; it's covered
# functionally by test_load_parquet_adaptive_union_narrows_dtype_and_matches_wide_solve
# above (which exercises the exact code path the `del` statements sit in and
# would fail immediately if any `del` referenced the wrong / already-rebound
# name).
# ---------------------------------------------------------------------------
