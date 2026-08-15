"""Regression tests for two performance fixes found by a perf audit of
analyze_gareus_mbar.py, both required to be bit-identical / byte-identical
to the pre-fix code:

1. `plot_gamd_boost` used to rebuild the same O(N) `d.window==k` boolean
   mask 8-12x per umbrella window across several separate list
   comprehensions (`win_means_comb`, `win_stds_comb`, `win_varbdv`,
   `win_means_dih`, `win_frac_dih`, `win_groups`, and a triple-redundant
   `_window_moments` call for skew/kurtosis/anharmonicity). Measured at
   N=8.18M/K=364: ~31.5s. Fixed by extracting the whole per-window
   statistics computation into `_per_window_gamd_boost_stats`, which sorts
   sample indices by window ONCE (`np.argsort(..., kind='stable')`) and
   slices each window's contiguous block out of that single sorted view --
   measured 0.76s for the same scale. A stable sort preserves each window's
   samples in their original relative order, so every per-window slice is
   element-for-element identical to the boolean-masked equivalent, and every
   statistic computed on it (mean/std/var/skew/kurtosis) is bit-identical,
   not just numerically close.

2. `_plot_2d_fes_multirange` / `_plot_chignolin_fes_kj_multirange` fully
   re-ran the entire render pipeline (figure/contour/colorbar/PNG encode) a
   second time to produce the "main"/untagged output file, even though that
   file is byte-for-byte identical (confirmed via SHA256) to one of the
   already-rendered tagged range variants. Fixed by copying the
   already-rendered file (`shutil.copyfile`) instead of re-rendering.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("analyze_gareus_mbar")

from analyze_gareus_mbar import (  # noqa: E402
    Data,
    KJ_PER_KCAL,
    _per_window_gamd_boost_stats,
    _plot_2d_fes_multirange,
    _plot_chignolin_fes_kj_multirange,
    _window_moments,
    plot_gamd_boost,
)


# --- Fix 1: per-window GaMD boost statistics (sort+slice vs boolean mask) --

def _old_way_per_window_stats(window, comb_kcal, kbt_kcal, K, dih_kcal=None):
    """Reconstruction of the pre-fix boolean-masking approach, used only as
    an independent reference to compare the new sort+slice implementation
    against. Deliberately NOT calling into the production code path."""
    win_means_comb = np.array([
        float(np.nanmean(comb_kcal[window == k])) if np.any(window == k) else np.nan
        for k in range(K)
    ])
    win_stds_comb = np.array([
        float(np.nanstd(comb_kcal[window == k])) if np.any(window == k) else np.nan
        for k in range(K)
    ])
    win_varbdv = np.array([
        float(np.var(comb_kcal[window == k] / kbt_kcal)) if np.any(window == k) else np.nan
        for k in range(K)
    ])
    win_groups = [comb_kcal[window == k] for k in range(K)]
    win_groups = [g[np.isfinite(g)] for g in win_groups]
    win_skew = np.array([_window_moments(comb_kcal[window == k])[0] for k in range(K)])
    win_kurt = np.array([_window_moments(comb_kcal[window == k])[1] for k in range(K)])
    win_anh = np.array([_window_moments(comb_kcal[window == k])[2] for k in range(K)])
    out = dict(means_comb=win_means_comb, stds_comb=win_stds_comb, varbdv=win_varbdv,
               groups=win_groups, skew=win_skew, kurt=win_kurt, anharmonicity=win_anh)
    if dih_kcal is not None:
        win_means_dih = np.array([
            float(np.nanmean(dih_kcal[window == k])) if np.any(window == k) else np.nan
            for k in range(K)
        ])
        _c_pos = comb_kcal.copy(); _c_pos[_c_pos <= 0] = np.nan
        win_frac_dih = np.array([
            float(np.nanmedian(dih_kcal[window == k] / _c_pos[window == k])) if np.any(window == k) else np.nan
            for k in range(K)
        ])
        out['means_dih'] = win_means_dih
        out['frac_dih'] = win_frac_dih
    return out


def _assert_nan_aware_array_equal(old, new, label):
    old = np.asarray(old); new = np.asarray(new)
    assert old.shape == new.shape, label
    both_nan = np.isnan(old) & np.isnan(new)
    assert np.array_equal(np.isnan(old), np.isnan(new)), f"{label}: NaN positions differ"
    assert np.array_equal(old[~both_nan], new[~both_nan]), f"{label}: non-NaN values differ"


def _make_synthetic_window_dataset(rng, K=37, n=6000, empty_window=5, include_dih=True):
    window = rng.integers(0, K, size=n)
    # Force a genuinely empty window (no samples at all) for the edge case.
    window = np.where(window == empty_window, (empty_window + 1) % K, window)
    # A handful of out-of-range window indices: plot_outputs elsewhere
    # explicitly filters `(d.window>=0)&(d.window<K)`, confirming the
    # codebase expects these to occur in real data. The searchsorted-based
    # boundary lookup (over k in range(K) only) must silently exclude them
    # from every window's group, exactly like the old `window==k` masking
    # did (neither -1 nor K+3 ever equals a k in range(K)).
    oor_idx = rng.choice(n, size=min(10, n), replace=False)
    window[oor_idx[: len(oor_idx) // 2]] = -1
    window[oor_idx[len(oor_idx) // 2:]] = K + 3
    comb_kcal = rng.normal(loc=3.0, scale=1.7, size=n)
    # Sprinkle some NaNs and some tiny (<4-sample) windows happen naturally
    # from the random assignment above -- both exercise real edge cases in
    # the stat functions (nanmean/nanstd on small arrays, _window_moments'
    # own size<4 guard).
    nan_idx = rng.choice(n, size=max(1, n // 100), replace=False)
    comb_kcal[nan_idx] = np.nan
    dih_kcal = comb_kcal * 0.4 + rng.normal(0, 0.2, size=n) if include_dih else None
    return window, comb_kcal, dih_kcal


@pytest.mark.parametrize("include_dih", [False, True])
def test_per_window_gamd_boost_stats_matches_boolean_masking(include_dih):
    rng = np.random.default_rng(20260812)
    K = 37
    window, comb_kcal, dih_kcal = _make_synthetic_window_dataset(rng, K=K, include_dih=include_dih)
    kbt_kcal = 0.62

    old = _old_way_per_window_stats(window, comb_kcal, kbt_kcal, K, dih_kcal)
    new = _per_window_gamd_boost_stats(window, comb_kcal, kbt_kcal, K, dih_kcal)

    for key in ["means_comb", "stds_comb", "varbdv", "skew", "kurt", "anharmonicity"]:
        _assert_nan_aware_array_equal(old[key], new[key], key)
    if include_dih:
        for key in ["means_dih", "frac_dih"]:
            _assert_nan_aware_array_equal(old[key], new[key], key)
        assert "means_dih" in new and "frac_dih" in new
    else:
        assert "means_dih" not in new and "frac_dih" not in new

    assert len(new["groups"]) == K
    for k in range(K):
        assert np.array_equal(old["groups"][k], new["groups"][k]), f"groups[{k}] differ"


def test_per_window_gamd_boost_stats_empty_window_is_all_nan_and_empty_group():
    rng = np.random.default_rng(7)
    K = 10
    n = 500
    window = rng.integers(1, K, size=n)  # window index 0 never occurs
    comb_kcal = rng.normal(2.0, 1.0, size=n)
    kbt_kcal = 0.6

    stats = _per_window_gamd_boost_stats(window, comb_kcal, kbt_kcal, K)
    assert np.isnan(stats["means_comb"][0])
    assert np.isnan(stats["stds_comb"][0])
    assert np.isnan(stats["varbdv"][0])
    assert np.isnan(stats["skew"][0])
    assert np.isnan(stats["kurt"][0])
    assert np.isnan(stats["anharmonicity"][0])
    assert stats["groups"][0].size == 0
    # a genuinely present window is not all-nan
    assert not np.isnan(stats["means_comb"][1])


def test_per_window_gamd_boost_stats_matches_boolean_masking_at_larger_scale():
    # A bigger, more window-dense case (closer in spirit to the real
    # N=8.18M/K=364 scale, scaled down for test speed) to make sure the
    # sort+slice boundary bookkeeping (searchsorted start/end) doesn't just
    # happen to work on tiny inputs.
    rng = np.random.default_rng(99)
    K = 120
    n = 40000
    window, comb_kcal, dih_kcal = _make_synthetic_window_dataset(rng, K=K, n=n, include_dih=True)
    kbt_kcal = 0.593

    old = _old_way_per_window_stats(window, comb_kcal, kbt_kcal, K, dih_kcal)
    new = _per_window_gamd_boost_stats(window, comb_kcal, kbt_kcal, K, dih_kcal)
    for key in ["means_comb", "stds_comb", "varbdv", "means_dih", "frac_dih", "skew", "kurt", "anharmonicity"]:
        _assert_nan_aware_array_equal(old[key], new[key], key)
    for k in range(K):
        assert np.array_equal(old["groups"][k], new["groups"][k])


@pytest.mark.parametrize("with_dih", [False, True])
def test_plot_gamd_boost_end_to_end_smoke(with_dih):
    """Full plot_gamd_boost call through the refactored code path: confirms
    the wiring (Data -> _per_window_gamd_boost_stats -> matplotlib panels)
    still produces every expected output file without error, for both the
    2-panel (no dihedral split) and 3-panel (dihedral split) layouts --
    the `if has_dih:` guards around stats['means_dih']/['frac_dih'] are only
    exercised when with_dih=True."""
    rng = np.random.default_rng(3)
    n = 4000
    K = 25
    window = rng.integers(0, K, size=n).astype(np.int64)
    boost_kj = rng.normal(10.0, 4.0, size=n) * KJ_PER_KCAL
    boost_dih_kj = (boost_kj * 0.5 + rng.normal(0, 1.0, size=n) * KJ_PER_KCAL) if with_dih else None
    # kB in kJ/(mol K); d.beta is documented/used as 1/kJ-mol (kbt_kcal =
    # 1/(d.beta * KJ_PER_KCAL)), so beta must be built from kB in kJ, not kcal.
    kb_kj_mol_k = 8.314462618e-3
    d = Data(
        prod_dir=Path("."), out_dir=Path("."),
        cv=np.zeros(n), cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=window, replica=np.zeros(n, dtype=np.int32), step=np.arange(n, dtype=np.int64),
        u_nk=np.zeros((n, K)), centers=np.zeros(K), k_kcal=np.zeros(K),
        beta=1.0 / (kb_kj_mol_k * 300.0), temp=300.0,
        boost_kj=boost_kj, potential_kj=None, source="test", meta={},
        boost_dih_kj=boost_dih_kj,
    )
    warnings = []
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        plot_gamd_boost(d, out, warnings)
        for fname in [
            "gamd_boost_diagnostics.png",
            "gamd_dv_distribution_per_window.png",
            "gamd_reweight_quality.png",
            "gamd_cumulant_quality.png",
        ]:
            fpath = out / fname
            assert fpath.exists() and fpath.stat().st_size > 0, f"missing/empty {fname}"


# --- Fix 2: multirange 2D-FES plotting duplicate-render elimination -------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_valid_png(path: Path):
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
    arr = mpimg.imread(str(path))
    assert arr.size > 0


def _synthetic_2d_grid():
    xedges = np.linspace(0, 10, 21)
    yedges = np.linspace(0, 8, 17)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    F = np.outer((xc - 5) ** 2, (yc - 4) ** 2) * 0.05
    return F, xedges, yedges, xc, yc


def test_plot_2d_fes_multirange_main_is_byte_identical_to_first_tagged_variant():
    F, xedges, yedges, xc, yc = _synthetic_2d_grid()
    warnings = []
    with tempfile.TemporaryDirectory() as td:
        out_png = Path(td) / "test_fes.png"
        files = _plot_2d_fes_multirange(F, xedges, yedges, xc, yc, out_png, "Test FES", "x", "y", warnings)

        assert "main" in files
        first_tag = "0_2"  # FES_PLOT_VMAX_VALUES[0] == 2.0
        assert first_tag in files

        _assert_valid_png(files["main"])
        _assert_valid_png(files[first_tag])
        assert _sha256(files["main"]) == _sha256(files[first_tag])
        # sanity: a differently-ranged tagged variant need not match (not a
        # strict requirement, just confirms we're not comparing two copies
        # of the same trivial degenerate output).
        assert Path(files["0_all"]).exists()


def test_plot_chignolin_fes_kj_multirange_main_is_byte_identical_to_main_vmax_variant():
    F, xedges, yedges, xc, yc = _synthetic_2d_grid()
    fes_kj = {"pmf": F, "cv_edges_A": xedges, "rg_edges_A": yedges, "cv_A": xc, "rg_A": yc}
    warnings = []
    with tempfile.TemporaryDirectory() as td:
        out_png = Path(td) / "chignolin_fes.png"
        files = _plot_chignolin_fes_kj_multirange(fes_kj, out_png, "gamd_cumulant2", warnings)

        assert "main" in files
        main_tag = "0_20"  # main_vmax == 20.0 in CHIGNOLIN_FES_PLOT_VMAX_VALUES_KJ
        assert main_tag in files

        _assert_valid_png(files["main"])
        _assert_valid_png(files[main_tag])
        assert _sha256(files["main"]) == _sha256(files[main_tag])


def test_plot_2d_fes_multirange_renders_each_range_exactly_once(monkeypatch):
    """SHA256 equality alone can't fail if the second full render is
    reinstated (it would just re-produce the same bytes, slower). The
    regression this fix actually targets is the *render count*: assert
    `_plot_2d_fes_range` is called exactly once per FES_PLOT_VMAX_VALUES
    entry, with no extra call to (re-)produce the 'main' file.

    Patches and calls through `gareus.mbar_analysis.plotting` directly, not
    `analyze_gareus_mbar`'s re-export -- `_plot_2d_fes_multirange` resolves
    `_plot_2d_fes_range` in its own module's globals (Plan A6a relocated
    both), so patching the re-exported `analyze_gareus_mbar` name would
    silently patch a different binding than the one the real call site
    reads.
    """
    import gareus.mbar_analysis.plotting as plotting

    F, xedges, yedges, xc, yc = _synthetic_2d_grid()
    real = plotting._plot_2d_fes_range
    calls = []

    def counting(*a, **kw):
        calls.append(kw.get("range_vmax"))
        return real(*a, **kw)

    monkeypatch.setattr(plotting, "_plot_2d_fes_range", counting)
    warnings = []
    with tempfile.TemporaryDirectory() as td:
        out_png = Path(td) / "test_fes.png"
        files = plotting._plot_2d_fes_multirange(F, xedges, yedges, xc, yc, out_png, "Test FES", "x", "y", warnings)
        assert "main" in files
    assert calls == list(plotting.FES_PLOT_VMAX_VALUES)


def test_plot_chignolin_fes_kj_multirange_renders_each_range_exactly_once(monkeypatch):
    """Same render-count regression check as above, for the chignolin
    contact-FES multirange path."""
    import analyze_gareus_mbar as agm

    F, xedges, yedges, xc, yc = _synthetic_2d_grid()
    fes_kj = {"pmf": F, "cv_edges_A": xedges, "rg_edges_A": yedges, "cv_A": xc, "rg_A": yc}
    real = agm._plot_chignolin_fes_kj_range
    calls = []

    def counting(*a, **kw):
        calls.append(kw.get("range_vmax"))
        return real(*a, **kw)

    monkeypatch.setattr(agm, "_plot_chignolin_fes_kj_range", counting)
    warnings = []
    with tempfile.TemporaryDirectory() as td:
        out_png = Path(td) / "chignolin_fes.png"
        files = agm._plot_chignolin_fes_kj_multirange(fes_kj, out_png, "gamd_cumulant2", warnings)
        assert "main" in files
    assert calls == list(agm.CHIGNOLIN_FES_PLOT_VMAX_VALUES_KJ)
