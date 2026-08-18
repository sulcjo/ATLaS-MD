import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_pmf_module_is_importable_standalone():
    """Must import with ZERO dependency on analyze_gareus_mbar having been
    loaded first -- proves _bridge()'s deferred-import design (no eager
    cross-import at gareus/mbar_analysis/pmf.py's own module level)."""
    import gareus.mbar_analysis.pmf  # noqa: F401


def test_bridge_prefers_already_loaded_real_module(monkeypatch):
    import gareus.mbar_analysis.pmf as pmfmod
    sentinel = object()

    class _FakeModule:
        marker = sentinel

    monkeypatch.setitem(sys.modules, 'analyze_gareus_mbar', _FakeModule())
    assert pmfmod._bridge().marker is sentinel


def test_bridge_prefers_main_module_when_it_is_the_script(monkeypatch):
    import gareus.mbar_analysis.pmf as pmfmod
    monkeypatch.delitem(sys.modules, 'analyze_gareus_mbar', raising=False)

    class _FakeMain:
        __file__ = '/some/path/analyze_gareus_mbar.py'
        marker = 'main-copy'

    monkeypatch.setitem(sys.modules, '__main__', _FakeMain())
    assert pmfmod._bridge().marker == 'main-copy'


def test_bridge_falls_back_to_fresh_import_when_neither_present(monkeypatch):
    import gareus.mbar_analysis.pmf as pmfmod
    monkeypatch.delitem(sys.modules, 'analyze_gareus_mbar', raising=False)

    class _FakeMainUnrelated:
        __file__ = '/some/other/script.py'

    monkeypatch.setitem(sys.modules, '__main__', _FakeMainUnrelated())
    result = pmfmod._bridge()
    assert result.__name__ == 'analyze_gareus_mbar'


_TASK1_NAMES = [
    'make_bins', '_bin_indices', 'pmf_from_weights',
    '_cumulant_shared_stats', '_cumulant_from_shared', '_cumulant_expansion', '_cumulant_expansion_both',
    'cumulant2', 'cumulant3',
    'pmf2d_from_weights',
    '_cumulant_shared_stats_2d', '_cumulant_from_shared_2d', '_cumulant_expansion_2d', '_cumulant_expansion_2d_both',
    '_bootstrap_pmf_uncertainty_1d', '_bootstrap_pmf_uncertainty_2d',
    'cumulant2_2d', 'cumulant3_2d',
]


def test_task1_names_are_reexported_identically_by_analyze_gareus_mbar():
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    for name in _TASK1_NAMES:
        assert hasattr(pmfmod, name), f'{name} missing from gareus.mbar_analysis.pmf'
        assert getattr(agm, name) is getattr(pmfmod, name), (
            f'analyze_gareus_mbar.{name} is not the SAME object as '
            f'gareus.mbar_analysis.pmf.{name} -- re-export import did not replace the local def'
        )


_TASK2_NAMES = ['_window_cv_mean_std', 'boost_stats', '_window_moments']


def test_task2_names_are_reexported_identically_by_analyze_gareus_mbar():
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    for name in _TASK2_NAMES:
        assert hasattr(pmfmod, name), f'{name} missing from gareus.mbar_analysis.pmf'
        assert getattr(agm, name) is getattr(pmfmod, name)


def test_boost_stats_still_resolves_norm_logw_and_ess_via_bridge():
    """boost_stats calls norm_logw/ess internally (via _bridge()); a real,
    non-degenerate boost array must still produce a finite ESS -- this is
    the one behavioral check that _bridge() is actually wired into this
    function's body, not just present in the module."""
    import gareus.mbar_analysis.pmf as pmfmod
    import numpy as np
    rng = np.random.default_rng(0)
    boost = rng.normal(50.0, 8.0, size=2000)
    beta = 1.0 / (0.00831446261815324 * 300.0)
    out = pmfmod.boost_stats(boost, beta)
    assert out['available'] is True
    assert np.isfinite(out['boost_reweight_ess'])
    assert 0.0 < out['boost_reweight_ess_fraction'] <= 1.0


def test_run_pmf_and_gamd_boost_report_is_reexported_identically():
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    assert agm.run_pmf_and_gamd_boost_report is pmfmod.run_pmf_and_gamd_boost_report


def test_run_pmf_and_gamd_boost_report_end_to_end_still_works(tmp_path):
    """Real end-to-end call through the relocated function, proving every
    bridged name (norm_logw, _eff_smooth, overlap_matrix, _sample_block_ids,
    write_pmf, write_all, plot_outputs) resolves correctly via _bridge()."""
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    import numpy as np
    from pathlib import Path as _Path

    rng = np.random.default_rng(20)
    n_windows, samples_per_window = 3, 300
    centers = np.linspace(-1.0, 1.0, n_windows)
    k_kcal = np.full(n_windows, 5.0)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)

    cv_parts, window_parts, replica_parts, epoch_src_parts = [], [], [], []
    for k in range(n_windows):
        cv_parts.append(centers[k] + rng.normal(0.0, 0.3, size=samples_per_window))
        window_parts.append(np.full(samples_per_window, k, dtype=np.int64))
        replica_parts.append(np.zeros(samples_per_window, dtype=np.int64) + k)
        epoch_src_parts.append(np.zeros(samples_per_window, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(window_parts)
    replica = np.concatenate(replica_parts)
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
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    bins = agm.make_bins(d.cv, 20, None, None)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    class _Args:
        bins = 20
        min_neighbor_overlap = 0.0
        selected_method = 'auto'
        gamd_smooth_sigma = 0.0
        pmf_smooth_sigma = 0.0
        smooth_sigma = 0.0
        pmf_uncertainty = False
        pmf_uncertainty_n_boot = 30
        pmf_uncertainty_seed = 0

    info = pmfmod.run_pmf_and_gamd_boost_report(d, _Args(), logw, bins, kbt_kcal, out_dir, [], None)
    assert info['selected'] == 'umbrella_only'  # no finite boost in this fixture
    assert (out_dir / 'pmf_unbiased.csv').exists()
    assert (out_dir / 'window_diagnostics.csv').exists()
    assert (out_dir / 'overlap_matrix.csv').exists()


def test_task4_names_are_reexported_identically():
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    assert agm.run_secondary_cv_analyses is pmfmod.run_secondary_cv_analyses
    assert agm.analyze_secondary_cv_pmf is pmfmod.analyze_secondary_cv_pmf


def test_analyze_secondary_cv_pmf_end_to_end_still_works(tmp_path):
    """Real end-to-end call proving every bridged name in this function
    (norm_logw, _eff_smooth, write_cv2_pmf, _secondary_cv_label,
    _secondary_cv_regions, _visible_pmfs, _smooth_pmf_1d,
    run_observable_pmf_convergence, wjson) resolves via _bridge()."""
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    import numpy as np
    from pathlib import Path as _Path

    rng = np.random.default_rng(30)
    n_windows, samples_per_window = 3, 300
    centers = np.linspace(-1.0, 1.0, n_windows)
    k_kcal = np.full(n_windows, 5.0)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    cv_parts, cv2_parts, window_parts = [], [], []
    for k in range(n_windows):
        cv_parts.append(centers[k] + rng.normal(0.0, 0.3, size=samples_per_window))
        cv2_parts.append(rng.normal(0.0, 1.0, size=samples_per_window))
        window_parts.append(np.full(samples_per_window, k, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    cv2 = np.concatenate(cv2_parts)
    window = np.concatenate(window_parts)
    n = cv.size
    u_nk = np.zeros((n, n_windows), dtype=np.float64)
    for k in range(n_windows):
        u_nk[:, k] = beta * agm.KJ_PER_KCAL * 0.5 * k_kcal[k] * (cv - centers[k]) ** 2

    out_dir = _Path(tmp_path) / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)
    d = agm.Data(
        prod_dir=_Path(tmp_path), out_dir=out_dir, cv=cv, cv2=cv2,
        rg_A=np.full(n, np.nan), window=window, replica=window.copy(), step=np.arange(n),
        u_nk=u_nk, centers=centers, k_kcal=k_kcal, beta=beta, temp=300.0,
        boost_kj=np.full(n, np.nan), potential_kj=None, source='test', meta={},
    )
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    class _Args:
        bins = 20
        cv2_bins = None
        cv2_min = None
        cv2_max = None
        gamd_smooth_sigma = 0.0
        pmf_smooth_sigma = 0.0
        smooth_sigma = 0.0
        plot_gamd_exponential = False
        plot_gamd_cumulant3 = False
        no_convergence = True

    info = pmfmod.analyze_secondary_cv_pmf(d, _Args(), logw, 'umbrella_only', False, kbt_kcal, out_dir, [], None)
    assert info['available'] is True
    assert (out_dir / 'cv2_pmf_unbiased.csv').exists()
    assert (out_dir / 'cv2_pmf_summary.json').exists()


def test_analyze_secondary_cv_pmf_boost_branch_end_to_end(tmp_path):
    """Regression: the existing end-to-end test above always passes
    boost_ok=False with an all-NaN boost_kj, so `if boost_ok and
    np.isfinite(boost_sel).sum()>10 and np.nanstd(boost_sel)>1e-12:`'s whole
    branch (a second _agm.norm_logw call, plus the _agm._eff_smooth kwarg
    feeding _cumulant_expansion_both) is never reached by any existing test
    -- found during this plan's own final-task review as a real coverage
    gap structurally invisible to both the byte-identity diff and the
    undefined-name checker (neither can see whether a bridged attribute
    access actually executes correctly, only whether the bare name is
    resolvable). Separately, the plotting try-block below (_eff_smooth,
    _visible_pmfs, _smooth_pmf_1d) runs unconditionally either way but
    silently swallows any exception into `warnings` -- no existing test
    asserts `warnings == []`, so a broken bridge there would produce a
    missing PNG with no test failure. This test closes both gaps at once:
    real finite, non-trivial-variance boost_kj with boost_ok=True exercises
    the boost branch for real, and asserting `warnings == []` (not just
    `info['available']`) means a swallowed AttributeError from any of the
    7 bridge points reachable from this call would now fail the test."""
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    import numpy as np
    from pathlib import Path as _Path

    rng = np.random.default_rng(31)
    n_windows, samples_per_window = 3, 300
    centers = np.linspace(-1.0, 1.0, n_windows)
    k_kcal = np.full(n_windows, 5.0)
    beta = 1.0 / (agm.K_B_KJ_PER_MOL_K * 300.0)
    cv_parts, cv2_parts, window_parts = [], [], []
    for k in range(n_windows):
        cv_parts.append(centers[k] + rng.normal(0.0, 0.3, size=samples_per_window))
        cv2_parts.append(rng.normal(0.0, 1.0, size=samples_per_window))
        window_parts.append(np.full(samples_per_window, k, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    cv2 = np.concatenate(cv2_parts)
    window = np.concatenate(window_parts)
    n = cv.size
    u_nk = np.zeros((n, n_windows), dtype=np.float64)
    for k in range(n_windows):
        u_nk[:, k] = beta * agm.KJ_PER_KCAL * 0.5 * k_kcal[k] * (cv - centers[k]) ** 2
    boost_kj = rng.normal(20.0, 5.0, size=n)  # real, finite, non-trivial variance

    out_dir = _Path(tmp_path) / 'out'
    out_dir.mkdir(parents=True, exist_ok=True)
    d = agm.Data(
        prod_dir=_Path(tmp_path), out_dir=out_dir, cv=cv, cv2=cv2,
        rg_A=np.full(n, np.nan), window=window, replica=window.copy(), step=np.arange(n),
        u_nk=u_nk, centers=centers, k_kcal=k_kcal, beta=beta, temp=300.0,
        boost_kj=boost_kj, potential_kj=None, source='test', meta={},
    )
    m = agm.solve_mbar(d.u_nk, d.window)
    logw = np.asarray(m['logw'], dtype=np.float64)
    kbt_kcal = (1.0 / d.beta) / agm.KJ_PER_KCAL

    class _Args:
        bins = 20
        cv2_bins = None
        cv2_min = None
        cv2_max = None
        gamd_smooth_sigma = 0.0
        pmf_smooth_sigma = 0.0
        smooth_sigma = 0.0
        plot_gamd_exponential = False
        plot_gamd_cumulant3 = False
        no_convergence = True

    warnings_list = []
    info = pmfmod.analyze_secondary_cv_pmf(d, _Args(), logw, 'gamd_cumulant2', True, kbt_kcal, out_dir, warnings_list, None)
    assert info['available'] is True
    assert warnings_list == [], f"boost-branch/plotting bridge silently failed: {warnings_list}"
    assert info['selected_unbiased_method'] != 'umbrella_only', "boost branch was not actually taken"
    assert (out_dir / 'cv2_pmf_unbiased.csv').exists()
    assert (out_dir / 'cv2_pmf_summary.json').exists()
    assert (out_dir / 'cv2_pmf_unbiased.png').exists()
