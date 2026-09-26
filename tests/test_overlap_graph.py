"""Overlap graph for gareus-analyze: states placed at (CV1, CV2, lambda),
edges and a CV-space density of the pairwise MBAR overlap integral.

The density is the per-sample integrand of the pairwise overlap
``sqrt(O_ij*O_ji) = sqrt(N_i N_j) * sum_n W_ni W_nj`` binned on the CV grid,
so it must integrate back to exactly the number
``gareus.mbar_analysis.ladder.pairwise_state_overlap`` reports.
"""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from gareus.mbar_analysis.ladder import pair_log_weights, pairwise_state_overlap
from gareus.mbar_analysis.overlap_graph import (
    PAIR_KIND_CV,
    PAIR_KIND_GAP,
    PAIR_KIND_RUNG,
    graph_pairs,
    layer_value,
    pair_overlap_densities,
    write_pairs_csv,
)

KT_300 = 0.0019872043 * 300.0   # kcal/mol


def _grid_union(centers1, centers2, lambdas, k1=50.0, k2=50.0, n_per=400, seed=0):
    """Harmonic states on a (CV1, CV2) grid, replicated per lambda rung.

    The rung term is a constant per-state energy offset (lambda * 1 kT) so
    rung pairs overlap strongly but not trivially. u_nk is reduced (kT).
    """
    rng = np.random.default_rng(seed)
    states = [(c1, c2, lam) for lam in lambdas for c1 in centers1 for c2 in centers2]
    K = len(states)
    cv1, cv2, window = [], [], []
    for s, (c1, c2, _lam) in enumerate(states):
        cv1.append(rng.normal(c1, math.sqrt(KT_300 / k1), n_per))
        cv2.append(rng.normal(c2, math.sqrt(KT_300 / k2), n_per))
        window.append(np.full(n_per, s))
    cv1, cv2, window = np.concatenate(cv1), np.concatenate(cv2), np.concatenate(window)
    u = np.empty((cv1.size, K))
    for s, (c1, c2, lam) in enumerate(states):
        u[:, s] = (0.5 * k1 * (cv1 - c1) ** 2 + 0.5 * k2 * (cv2 - c2) ** 2) / KT_300 + lam
    f_k = np.array([-lam for (_c1, _c2, lam) in states])     # exact: harmonic Z identical per centre
    n_k = np.bincount(window, minlength=K).astype(float)
    c1s = np.array([s[0] for s in states]); c2s = np.array([s[1] for s in states])
    lams = np.array([s[2] for s in states])
    return cv1, cv2, window, u, f_k, n_k, c1s, c2s, lams


def test_pair_log_weights_reproduces_pairwise_state_overlap():
    cv1, cv2, window, u, f_k, n_k, *_ = _grid_union([0.0, 0.3], [0.0], [0.0])
    rows = (window == 0) | (window == 1)
    log_w = pair_log_weights(u[np.ix_(rows, [0, 1])], f_k[[0, 1]], n_k[[0, 1]])
    shared = float(np.exp(log_w[:, 0] + log_w[:, 1]).sum())
    expected = pairwise_state_overlap(u, window, f_k, n_k, 0, 1)
    assert math.sqrt(n_k[0] * n_k[1]) * shared == pytest.approx(expected, rel=1e-12)


def test_graph_pairs_two_centres_two_rungs():
    # states: rung 0 -> (0: c=0), (1: c=0.3); rung 0.5 -> (2: c=0), (3: c=0.3)
    c1 = [0.0, 0.3, 0.0, 0.3]
    pairs = graph_pairs(c1, [0.0] * 4, [0.0, 0.0, 0.5, 0.5], [50.0] * 4, [0.0] * 4, 300.0)
    assert sorted(p for p in pairs if p[2] == PAIR_KIND_CV) == [(0, 1, PAIR_KIND_CV), (2, 3, PAIR_KIND_CV)]
    assert sorted(p for p in pairs if p[2] == PAIR_KIND_RUNG) == [(0, 2, PAIR_KIND_RUNG), (1, 3, PAIR_KIND_RUNG)]


def test_graph_pairs_rung_edges_join_adjacent_rungs_only():
    c1 = [0.0, 0.0, 0.0]
    pairs = graph_pairs(c1, [0.0] * 3, [0.0, 0.5, 1.0], [50.0] * 3, [0.0] * 3, 300.0)
    assert sorted(pairs) == [(0, 1, PAIR_KIND_RUNG), (1, 2, PAIR_KIND_RUNG)]


def test_graph_pairs_without_secondary_or_ladder():
    # CV1-only, no ladder: NaN secondary centres and lambdas=None must not break pairing
    pairs = graph_pairs([0.0, 0.3, 0.6], [math.nan] * 3, None, [50.0] * 3, [math.nan] * 3, 300.0)
    assert sorted(pairs) == [(0, 1, PAIR_KIND_CV), (1, 2, PAIR_KIND_CV)]


def test_graph_pairs_bridges_disconnected_clusters_on_each_rung():
    # two CV1 clusters {0, 0.1} and {0.9, 1.0}: the neighbour rule alone leaves them unconnected
    c1 = [0.0, 0.1, 0.9, 1.0] * 2
    lam = [0.0] * 4 + [0.5] * 4
    pairs = graph_pairs(c1, [0.0] * 8, lam, [50.0] * 8, [0.0] * 8, 300.0)
    gaps = sorted(p for p in pairs if p[2] == PAIR_KIND_GAP)
    assert gaps == [(1, 2, PAIR_KIND_GAP), (5, 6, PAIR_KIND_GAP)]   # closest pair across the gap, per rung


def test_graph_pairs_adds_no_gap_edge_when_connected():
    pairs = graph_pairs([0.0, 0.3, 0.6], [0.0] * 3, None, [50.0] * 3, [0.0] * 3, 300.0)
    assert not [p for p in pairs if p[2] == PAIR_KIND_GAP]


def test_layer_value_same_rung_and_midpoint():
    assert layer_value(0.25, 0.25) == 0.25
    assert layer_value(0.0, 0.5) == pytest.approx(0.25)


def test_density_integrates_to_pairwise_overlap():
    cv1, cv2, window, u, f_k, n_k, c1s, c2s, lams = _grid_union([0.0, 0.3], [0.0, 0.3], [0.0, 0.5])
    pairs = graph_pairs(c1s, c2s, lams, [50.0] * len(c1s), [50.0] * len(c1s), 300.0)
    edges1 = np.linspace(cv1.min() - 1.0, cv1.max() + 1.0, 31)
    edges2 = np.linspace(cv2.min() - 1.0, cv2.max() + 1.0, 21)
    results = pair_overlap_densities(u, window, f_k, n_k, cv1, cv2, pairs, edges1, edges2)
    assert {(r['i'], r['j']) for r in results} == {(i, j) for i, j, _k in pairs}
    for r in results:
        expected = pairwise_state_overlap(u, window, f_k, n_k, r['i'], r['j'])
        assert r['overlap'] == pytest.approx(expected, rel=1e-10)
        assert r['density'].shape == (30, 20)
        assert float(r['density'].sum()) == pytest.approx(expected, rel=1e-10)


def test_density_is_one_dimensional_without_cv2():
    cv1, _cv2, window, u, f_k, n_k, c1s, _c2s, lams = _grid_union([0.0, 0.3], [0.0], [0.0])
    pairs = graph_pairs(c1s, [math.nan] * 2, lams, [50.0] * 2, [math.nan] * 2, 300.0)
    edges1 = np.linspace(-1.0, 1.5, 26)
    (r,) = pair_overlap_densities(u, window, f_k, n_k, cv1, None, pairs, edges1, None)
    assert r['density'].shape == (25,)
    assert float(r['density'].sum()) == pytest.approx(r['overlap'], rel=1e-10)


def test_samples_outside_grid_still_count_in_overlap():
    cv1, cv2, window, u, f_k, n_k, c1s, c2s, lams = _grid_union([0.0, 0.3], [0.0], [0.0])
    pairs = [(0, 1, PAIR_KIND_CV)]
    narrow = np.linspace(0.1, 0.2, 5)            # clips most samples
    (r,) = pair_overlap_densities(u, window, f_k, n_k, cv1, cv2, pairs, narrow, np.linspace(-1, 1, 5))
    assert r['overlap'] == pytest.approx(pairwise_state_overlap(u, window, f_k, n_k, 0, 1), rel=1e-10)
    assert float(r['density'].sum()) < r['overlap']


def test_empty_state_gives_nan_overlap_not_crash():
    cv1, cv2, window, u, f_k, n_k, *_ = _grid_union([0.0, 0.3], [0.0], [0.0])
    keep = window == 0
    n_k2 = n_k.copy(); n_k2[1] = 0.0
    (r,) = pair_overlap_densities(u[keep], window[keep], f_k, n_k2, cv1[keep], cv2[keep],
                                  [(0, 1, PAIR_KIND_CV)], np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    assert math.isnan(r['overlap'])


def test_write_pairs_csv(tmp_path):
    rows = [{'i': 0, 'j': 1, 'kind': PAIR_KIND_CV, 'overlap': 0.3, 'layer': 0.0},
            {'i': 0, 'j': 2, 'kind': PAIR_KIND_RUNG, 'overlap': 0.1, 'layer': 0.25},
            {'i': 1, 'j': 3, 'kind': PAIR_KIND_RUNG, 'overlap': math.nan, 'layer': 0.25}]
    path = tmp_path / 'overlap_pairs_mbar.csv'
    write_pairs_csv(rows, path, threshold=0.15)
    got = list(csv.DictReader(path.open()))
    assert [g['below_threshold'] for g in got] == ['0', '1', '']
    assert got[1]['kind'] == PAIR_KIND_RUNG and float(got[1]['overlap']) == pytest.approx(0.1)


@pytest.mark.parametrize('with_cv2,with_ladder', [(True, True), (False, True), (True, False)])
def test_render_writes_figures(tmp_path, with_cv2, with_ladder):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    from gareus.mbar_analysis.plotting_overlap import render_overlap_graph

    lambdas = [0.0, 0.5] if with_ladder else [0.0]
    cv1, cv2, window, u, f_k, n_k, c1s, c2s, lams = _grid_union([0.0, 0.3], [0.0, 0.3] if with_cv2 else [0.0], lambdas)
    K = len(c1s)
    sec = c2s if with_cv2 else np.full(K, math.nan)
    k2 = [50.0] * K if with_cv2 else [math.nan] * K
    info = render_overlap_graph(
        tmp_path, u_nk=u, window=window, f_k=f_k, n_k=n_k, cv1=cv1, cv2=cv2 if with_cv2 else None,
        centers=c1s, secondary_centers=sec, lambdas=lams if with_ladder else None,
        k1=[50.0] * K, k2=k2, temperature_k=300.0, threshold=0.15, bins=20,
        cv1_label='CV1', cv2_label='CV2', write_html=False)
    assert info['available'] is True
    for name in ('overlap_matrix.png', 'overlap_density_layers.png', 'overlap_pairs_mbar.csv'):
        assert (tmp_path / name).is_file() and (tmp_path / name).stat().st_size > 0
    assert info['n_pairs'] > 0


def test_render_skips_cleanly_without_centres(tmp_path):
    from gareus.mbar_analysis.plotting_overlap import render_overlap_graph
    info = render_overlap_graph(
        tmp_path, u_nk=np.zeros((4, 2)), window=np.array([0, 0, 1, 1]), f_k=np.zeros(2), n_k=np.array([2.0, 2.0]),
        cv1=np.zeros(4), cv2=None, centers=np.array([math.nan, math.nan]), secondary_centers=np.full(2, math.nan),
        lambdas=None, k1=[50.0, 50.0], k2=[math.nan, math.nan], temperature_k=300.0, threshold=0.15, bins=10,
        cv1_label='CV1', cv2_label='CV2', write_html=False)
    assert info['available'] is False and info['reason']


def _agg():
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')


def test_render_cv1_only_without_ladder(tmp_path):
    _agg()
    from gareus.mbar_analysis.plotting_overlap import render_overlap_graph
    cv1, _cv2, window, u, f_k, n_k, c1s, _c2s, _lams = _grid_union([0.0, 0.3, 0.6], [0.0], [0.0])
    info = render_overlap_graph(
        tmp_path, u_nk=u, window=window, f_k=f_k, n_k=n_k, cv1=cv1, cv2=None, centers=c1s,
        secondary_centers=np.full(3, math.nan), lambdas=None, k1=[50.0] * 3, k2=[math.nan] * 3,
        temperature_k=300.0, threshold=0.15, bins=20, cv1_label='CV1', cv2_label='CV2', write_html=False)
    assert info['available'] is True and info['dimensions'] == 'cv1_lambda'


def test_unmeasured_edge_is_not_counted_weak(tmp_path):
    _agg()
    from gareus.mbar_analysis.plotting_overlap import render_overlap_graph
    # 0.1 apart = ~0.9 restraint widths: edge (0, 1) is well above threshold
    cv1, _cv2, window, u, f_k, n_k, c1s, _c2s, _lams = _grid_union([0.0, 0.1, 0.2], [0.0], [0.0])
    keep = window != 2                           # state 2 has no samples -> edge (1, 2) unmeasured
    n_k = n_k.copy(); n_k[2] = 0.0
    info = render_overlap_graph(
        tmp_path, u_nk=u[keep], window=window[keep], f_k=f_k, n_k=n_k, cv1=cv1[keep], cv2=None, centers=c1s,
        secondary_centers=np.full(3, math.nan), lambdas=None, k1=[50.0] * 3, k2=[math.nan] * 3,
        temperature_k=300.0, threshold=0.15, bins=20, cv1_label='CV1', cv2_label='CV2', write_html=False)
    assert info['n_unmeasured'] == 1 and info['n_below_threshold'] == 0


def test_render_skips_when_states_are_indistinguishable(tmp_path):
    # a 2D layout whose secondary centres were lost: states share (CV1 centre, lambda)
    from gareus.mbar_analysis.plotting_overlap import render_overlap_graph
    cv1, _cv2, window, u, f_k, n_k, c1s, _c2s, _lams = _grid_union([0.0, 0.3], [0.0, 0.3], [0.0])
    K = c1s.size
    info = render_overlap_graph(
        tmp_path, u_nk=u, window=window, f_k=f_k, n_k=n_k, cv1=cv1, cv2=None, centers=c1s,
        secondary_centers=np.full(K, math.nan), lambdas=None, k1=[50.0] * K, k2=[math.nan] * K,
        temperature_k=300.0, threshold=0.15, bins=20, cv1_label='CV1', cv2_label='CV2', write_html=False)
    assert info['available'] is False and 'share' in info['reason']


def test_render_removes_stale_outputs_before_writing(tmp_path):
    from gareus.mbar_analysis.plotting_overlap import OUTPUT_FILES, render_overlap_graph
    for name in OUTPUT_FILES:
        (tmp_path / name).write_text('stale')
    info = render_overlap_graph(
        tmp_path, u_nk=np.zeros((4, 2)), window=np.array([0, 0, 1, 1]), f_k=np.zeros(2), n_k=np.array([2.0, 2.0]),
        cv1=np.zeros(4), cv2=None, centers=np.array([math.nan, math.nan]), secondary_centers=np.full(2, math.nan),
        lambdas=None, k1=[50.0, 50.0], k2=[math.nan, math.nan], temperature_k=300.0, threshold=0.15, bins=10,
        cv1_label='CV1', cv2_label='CV2', write_html=False)
    assert info['available'] is False
    assert not any((tmp_path / name).exists() for name in OUTPUT_FILES)


def test_rung_pair_sheet_never_merges_into_a_same_rung_sheet():
    from gareus.mbar_analysis.plotting_overlap import _layers
    d = np.ones(3)
    results = [{'layer': 0.25, 'kind': PAIR_KIND_CV, 'density': d},        # same-rung pair on rung 0.25
               {'layer': 0.25, 'kind': PAIR_KIND_RUNG, 'density': 2 * d}]  # rung 0 <-> 0.5 (0.25 missing here)
    layers = _layers(results)
    assert set(layers) == {(0.25, 'same'), (0.25, 'rung')}
    assert float(layers[(0.25, 'rung')].sum()) == 6.0
