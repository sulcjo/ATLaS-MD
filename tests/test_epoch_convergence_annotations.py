"""Epoch/topup convergence annotation + pooling diagnostics.

Regression guard for the "labels are opposite" bug: boundary annotation lines
on the epoch-convergence plots were positioned by sorting samples on their
per-epoch-LOCAL step counter (which resets every epoch).  Because epoch_000
runs longest, its max local step is largest, so its boundary landed at the far
RIGHT while short late top-ups landed at the LEFT -- reversed vs the actual
convergence curve.

The convergence x-axis is frac_total = cum_samples(sources 0..k)/N, so each
source's boundary must sit at its cumulative-sample fraction in *source order*,
independent of step.  These tests pin that contract and the pooling table that
confirms every epoch + final source was pooled into the PMF.
"""
import numpy as np
from types import SimpleNamespace

from analyze_gareus_mbar import (
    _epoch_source_annotations,
    _epoch_source_pooling_table,
    _short_source_label,
)

_AP = '/r/adaptive_production'


def _make_data(src, step, window=None, run_dirs=None):
    src = np.asarray(src, dtype=np.int64)
    step = np.asarray(step, dtype=np.int64)
    if window is None:
        window = np.zeros_like(src)
    meta = {'_epoch_source': src.tolist()}
    if run_dirs is not None:
        meta['adaptive_epoch_run_dirs'] = list(run_dirs)
    return SimpleNamespace(step=step, window=np.asarray(window, dtype=np.int64), meta=meta)


def test_short_source_label_derivation():
    assert _short_source_label(f'{_AP}/epoch_000') == 'epoch_000'
    assert _short_source_label(f'{_AP}/epoch_001/baseline') == 'epoch_001/baseline'
    assert _short_source_label(f'{_AP}/final/baseline') == 'final/baseline'
    # topup step suffix shortened to @<k>k
    assert _short_source_label(f'{_AP}/final/topup_001_500000') == 'final/topup_001@500k'


def test_annotation_fracs_follow_source_order_not_local_step():
    # epoch_000 runs LONGEST (largest local step) and is ~47% of all samples.
    # The two later sources reset to small local steps.  Pre-fix the global
    # step-sort pushed epoch_000's boundary to ~1.0; it must sit at ~0.47.
    n0, n1, n2 = 47, 33, 20  # 100 samples
    src = [0] * n0 + [1] * n1 + [2] * n2
    step = list(range(1000, 1000 + n0)) + list(range(0, n1)) + list(range(0, n2))
    run_dirs = [f'{_AP}/epoch_000', f'{_AP}/epoch_001/baseline',
                f'{_AP}/epoch_001/topup_001_500000']
    ann = _epoch_source_annotations(_make_data(src, step, run_dirs=run_dirs))

    assert len(ann) == 3
    label_to_frac = {lbl: frac for lbl, frac, _ in ann}
    # epoch_000 boundary at its cumulative fraction (~0.47), NOT near 1.0
    assert abs(label_to_frac['epoch_000'] - 0.47) < 0.02
    assert label_to_frac['epoch_000'] < 0.6  # the bug put this ~1.0
    # monotonic in source order and last source closes the plot at 1.0
    fracs = [frac for _, frac, _ in ann]
    assert fracs == sorted(fracs)
    assert abs(fracs[-1] - 1.0) < 1e-9


def test_annotation_independent_of_step_values():
    # Placement must depend only on per-source sample counts, never step values.
    src = [0] * 40 + [1] * 60
    run_dirs = [f'{_AP}/epoch_000', f'{_AP}/final/baseline']
    a1 = [f for _, f, _ in _epoch_source_annotations(
        _make_data(src, list(range(100, 140)) + list(range(0, 60)), run_dirs=run_dirs))]
    a2 = [f for _, f, _ in _epoch_source_annotations(
        _make_data(src, list(range(0, 40)) + list(range(900, 960)), run_dirs=run_dirs))]
    assert np.allclose(a1, a2)
    assert abs(a1[0] - 0.40) < 1e-9


def test_pooling_table_reports_every_source():
    src = [0] * 47 + [1] * 33 + [2] * 20
    window = [0] * 47 + [5] * 33 + [0] * 20  # source 1 uses a distinct window
    run_dirs = [f'{_AP}/epoch_000', f'{_AP}/epoch_001/baseline',
                f'{_AP}/final/topup_001_500000']
    d = _make_data(src, np.arange(100), window=window, run_dirs=run_dirs)

    rows = _epoch_source_pooling_table(d)
    assert len(rows) == 3
    assert sum(r['n_samples'] for r in rows) == 100
    assert rows[0]['label'] == 'epoch_000' and rows[0]['n_samples'] == 47
    assert rows[1]['n_windows'] == 1
    assert abs(rows[-1]['cum_frac'] - 1.0) < 1e-9
    # final/ source must appear -> proves final is pooled, not dropped
    assert any('final' in r['label'] for r in rows)


def test_annotations_empty_without_epoch_metadata():
    d = SimpleNamespace(step=np.arange(10), window=np.zeros(10, dtype=np.int64), meta={})
    assert _epoch_source_annotations(d) == []
    assert _epoch_source_pooling_table(d) == []


def _synthetic_adaptive_data(tmp_path):
    """Tiny 2-window / 3-source harmonic umbrella Data for a real MBAR solve.

    u_nk = beta * 0.5 * k * (cv - center)^2 (reduced bias), with a varying GaMD
    boost so the GaMD-reweight ESS path is exercised (not NaN).
    """
    import analyze_gareus_mbar as A

    rng = np.random.default_rng(0)
    beta = 0.4  # 1/(kJ/mol), ~300 K
    centers = np.array([0.0, 2.0])
    k_spring = np.array([50.0, 50.0])  # kJ/mol/unit^2
    sigma = 0.18
    per_src = 120
    cv_parts, win_parts, src_parts = [], [], []
    for s in range(3):
        for w in (0, 1):
            x = rng.normal(centers[w], sigma, per_src // 2)
            cv_parts.append(x)
            win_parts.append(np.full(x.size, w, dtype=np.int64))
            src_parts.append(np.full(x.size, s, dtype=np.int64))
    cv = np.concatenate(cv_parts)
    window = np.concatenate(win_parts)
    src = np.concatenate(src_parts)
    N = cv.size
    u_nk = beta * 0.5 * k_spring[None, :] * (cv[:, None] - centers[None, :]) ** 2
    boost_kj = rng.uniform(0.5, 6.0, N)  # varying boost -> finite GaMD-reweight ESS
    run_dirs = ['/r/adaptive_production/epoch_000',
                '/r/adaptive_production/epoch_001/baseline',
                '/r/adaptive_production/final/baseline']
    meta = {'_epoch_source': src.tolist(), 'adaptive_epoch_run_dirs': run_dirs,
            'timestep_fs': 4.0}
    d = A.Data(prod_dir=tmp_path, out_dir=tmp_path / 'out',
               cv=cv, cv2=np.full(N, np.nan), rg_A=np.full(N, np.nan),
               window=window, replica=window.copy(), step=np.arange(N, dtype=np.int64),
               u_nk=u_nk, centers=centers, k_kcal=k_spring / A.KJ_PER_KCAL,
               beta=beta, temp=300.0, boost_kj=boost_kj, potential_kj=None,
               source='synthetic', meta=meta)
    return d


def test_epoch_convergence_writes_ess_and_pooling(tmp_path):
    """Exercise the real run_epoch_pmf_convergence write-path (numpy backend, no
    pymbar): the ESS columns and the pooling CSV must be produced with finite
    values, and every source (incl. final) must be pooled."""
    import csv
    import analyze_gareus_mbar as A

    d = _synthetic_adaptive_data(tmp_path)
    kbt_kcal = (1.0 / d.beta) / A.KJ_PER_KCAL
    bins = A.make_bins(d.cv, 24, float(d.cv.min()), float(d.cv.max()))
    args = SimpleNamespace(no_convergence=False, convergence_mbar_backend='numpy',
                           convergence_mbar_tol=1e-8, convergence_mbar_maxiter=5000,
                           mbar_threads=0, gamd_smooth_sigma=0.0,
                           convergence_js_threshold=0.01, convergence_rmse_threshold=0.1)
    m_full = A.solve_mbar(d.u_nk, d.window, tol=1e-8, maxiter=5000, backend='numpy')
    final_pmf, _diag, _method = A._observable_pmf_from_logw(
        d.cv, m_full['logw'], d.boost_kj, bins, 'umbrella_only', d.beta, kbt_kcal,
        smooth_logfac_sigma=0.0)

    res = A.run_epoch_pmf_convergence(d, args, bins, 'umbrella_only', final_pmf, tmp_path / 'out')

    assert res['enabled'] is True
    assert res['n_sources_pooled'] == 3
    assert any('final' in lbl for lbl in res['sources'])

    rows = list(csv.DictReader(open(tmp_path / 'out' / 'epoch_convergence' / 'epoch_pmf_convergence.csv')))
    finite = [r for r in rows if r.get('JS') not in ('', 'nan', None)]
    assert finite, 'expected at least one finite convergence checkpoint'
    for r in finite:
        assert 'mbar_ess' in r and np.isfinite(float(r['mbar_ess'])) and float(r['mbar_ess']) > 0
        assert np.isfinite(float(r['mbar_ess_frac']))
        # boost varies -> GaMD-reweight ESS must be finite (and tiny, but real)
        assert np.isfinite(float(r['gamd_reweight_ess'])) and float(r['gamd_reweight_ess']) > 0

    pool_rows = list(csv.DictReader(open(tmp_path / 'out' / 'epoch_convergence' / 'epoch_source_pooling.csv')))
    assert len(pool_rows) == 3
    assert sum(int(p['n_samples']) for p in pool_rows) == d.cv.size
    # summary carries the final-checkpoint ESS values
    s = res['summary']
    assert np.isfinite(s['final_mbar_ess']) and np.isfinite(s['final_gamd_reweight_ess'])
