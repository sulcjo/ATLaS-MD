"""Dashboard overlap on a 2D layout: compare spatial neighbours, not flat-index ones.

The flat (w, w+1) pairing is only meaningful for a 1D ladder. On a sparse 2D
lambda-ladder layout, consecutive window indices sit in different cells or on
different rungs, so their CV1 histograms barely touch and the window ranking
flagged ~a third of chignolin_9's windows BAD on "overlap" that was never a
real neighbour relation. Neighbours are now the nearest windows on the same
rung in restraint-width units, which also covers diagonal layouts where the
nearest window is neither in the same row nor the same column.
"""

import random

from gareus.dashboard.context import overlap_by_pair
from gareus.dashboard.neighbours import (
    best_neighbour_overlaps,
    overlap_by_pair_2d,
    spatial_neighbour_pairs,
)
from gareus.dashboard.ranking import BAD, DEAD_OVERLAP, rank_windows, restraint_sigma

T = 300.0
K1, K2 = 300.0, 1.2            # chignolin_9-like force constants (kcal/mol/CV^2)
S1, S2 = restraint_sigma(K1, T), restraint_sigma(K2, T)
# 3 CV1 columns x 2 CV2 rows x 2 rungs, rungs interleaved across the flat index
# the way the real ladder CSV writes them; spacing 2 (CV1) and 1.5 (CV2)
# restraint widths, so the column neighbour is each window's nearest.
CV1 = tuple(0.1 + 2.0 * S1 * i for i in range(3))
CV2 = (-0.75 * S2, 0.75 * S2)
LAYOUT = [(c1, c2, lam) for lam in (0.0, 1.0) for c2 in CV2 for c1 in CV1]


def _cols(layout):
    return [w[0] for w in layout], [w[1] for w in layout], [w[2] for w in layout]


def _pairs(layout, k1=None, k2=None):
    c1, c2, lam = _cols(layout)
    n = len(layout)
    return spatial_neighbour_pairs(c1, c2, lam, k1 or [K1] * n, k2 or [K2] * n, T)


def test_each_window_is_paired_with_its_nearest_neighbour_on_the_same_rung():
    pairs = _pairs(LAYOUT)
    lam = _cols(LAYOUT)[2]
    assert all(lam[a] == lam[b] for a, b in pairs)
    touched = {w for p in pairs for w in p}
    assert touched == set(range(len(LAYOUT)))
    assert (0, 3) in pairs          # same column, 1.5 sigma apart on CV2: nearest


def test_flat_index_neighbours_in_far_cells_or_other_rungs_are_not_paired():
    pairs = set(_pairs(LAYOUT))
    assert (5, 6) not in pairs      # flat neighbours across the rung boundary
    assert (0, 2) not in pairs      # two columns apart, a nearer window exists


def test_a_diagonal_chain_pairs_along_the_diagonal():
    # The dashboard golden fixture's shape: each window's nearest neighbour is
    # the next one diagonally; the same-row window is five steps away.
    layout = [(4.0 + 0.55 * i, float(i % 5 - 2), 0.0) for i in range(25)]
    c1, c2, lam = _cols(layout)
    pairs = set(spatial_neighbour_pairs(c1, c2, lam, [2.5] * 25, [], T))
    assert {(i, i + 1) for i in range(24) if i % 5 != 4} <= pairs
    assert (0, 5) not in pairs


def test_fully_unrestrained_windows_are_left_out():
    layout = [(0.1, 0.0, 0.0), (0.1 + S1, 0.0, 0.0), (0.1, 0.0, 0.0)]
    pairs = _pairs(layout, k1=[K1, K1, 0.0], k2=[K2, K2, 0.0])
    assert pairs == [(0, 1)]


def test_windows_pair_only_with_the_same_restraint_pattern():
    # CV2-only windows (k1 = 0) sit at a parking CV1 value; letting them pair
    # with fully restrained windows at zero CV1 distance would make them every
    # window's "nearest" neighbour and hide real CV1 gaps. They pair with each
    # other, along CV2 only.
    layout = [(0.1, 0.0, 0.0), (0.1 + S1, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, S2, 0.0)]
    pairs = _pairs(layout, k1=[K1, K1, 0.0, 0.0], k2=[K2] * 4)
    assert pairs == [(0, 1), (2, 3)]


def test_unknown_lambdas_are_treated_as_one_rung():
    layout = [(0.1, 0.0, float("nan")), (0.1 + S1, 0.0, float("nan"))]
    assert _pairs(layout) == [(0, 1)]


def _samples(layout, seed=3):
    rng = random.Random(seed)
    h1 = {w: [c1 + rng.gauss(0, S1) for _ in range(400)] for w, (c1, _c2, _l) in enumerate(layout)}
    h2 = {w: [c2 + rng.gauss(0, S2) for _ in range(400)] for w, (_c1, c2, _l) in enumerate(layout)}
    return h1, h2


def test_a_healthy_2d_layout_has_no_dead_neighbour_overlap_but_flat_pairing_does():
    h1, h2 = _samples(LAYOUT)
    c1, c2, lam = _cols(LAYOUT)
    n = len(LAYOUT)
    fixed = overlap_by_pair_2d(h1, h2, c1, c2, lam, [K1] * n, [K2] * n, T)
    assert fixed and min(fixed.values()) >= DEAD_OVERLAP
    flat = overlap_by_pair(h1, c1)
    assert min(flat.values()) < DEAD_OVERLAP            # the artefact this replaces


def test_a_genuine_gap_is_still_caught():
    layout = [(0.1, -6 * S2, 0.0), (0.1, 6 * S2, 0.0)]
    h1, h2 = _samples(layout)
    ov = overlap_by_pair_2d(h1, h2, *_cols(layout), [K1] * 2, [K2] * 2, T)
    assert ov[(0, 1)] < DEAD_OVERLAP


def test_a_window_is_judged_by_its_best_neighbour_not_its_worst():
    pairs = {(0, 1): 0.45, (0, 2): 0.02, (1, 2): 0.40, (3, 4): 0.01}
    edges, per_window = best_neighbour_overlaps(pairs, 5)
    assert per_window == {0: 0.45, 1: 0.45, 2: 0.40, 3: 0.01, 4: 0.01}
    assert edges == {(0, 1): 0.45, (1, 2): 0.40, (3, 4): 0.01}


def test_ranking_flags_only_isolated_windows_when_given_per_window_overlap():
    statuses = rank_windows(
        n_windows=5, centers_a=[0.0] * 5, k_list=[K1] * 5, acceptance_by_window={},
        overlap_by_pair={(0, 2): 0.02, (3, 4): 0.01}, delta_by_window={},
        temperature_k=T, overlap_by_window={0: 0.45, 1: 0.45, 2: 0.40, 3: 0.01, 4: 0.01})
    bad = {s.window for s in statuses if s.status == BAD}
    assert bad == {3, 4}


def test_build_context_uses_spatial_pairs_for_a_2d_run(tmp_path):
    import argparse

    from gareus.dashboard.context import build_context
    from gareus.dashboard.sidecar import SidecarSnapshot
    from gareus.logger import DistanceLogger

    h1, h2 = _samples(LAYOUT)
    c1, c2, lam = _cols(LAYOUT)
    n = len(LAYOUT)
    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0, temperature_k=T),
                            no_file_persistence=True)
    logger.history_by_window.update(h1)
    logger.secondary_history_by_window.update(h2)
    rows = [{"replica": w, "window": w, "center_A": c1[w], "k_kcal_mol_A2": K1,
             "cv_A": c1[w], "gamd_lambda": lam[w], "secondary_cv_k_kcal_mol": K2}
            for w in range(n)]
    ctx = build_context(
        logger=logger, rows=rows, phase="gareus_production", step=1, total_steps=10,
        summary={}, dashboard_info={"centers_a": list(c1), "n_windows": n, "k_list": [K1] * n,
                                    "secondary_cv_centers": list(c2), "exchange_stats": {}},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=0.0, view="windows",
        glyphs="unicode")
    assert ctx.is_2d
    assert (5, 6) not in ctx.overlap_pairs and (0, 3) in ctx.overlap_pairs
    assert min(ctx.overlap_pairs.values()) >= DEAD_OVERLAP
    assert set(ctx.overlap_windows) == set(range(n))
    assert min(ctx.overlap_windows.values()) >= DEAD_OVERLAP


def test_build_context_keeps_the_flat_pairs_for_a_1d_run(tmp_path):
    import argparse

    from gareus.dashboard.context import build_context
    from gareus.dashboard.sidecar import SidecarSnapshot
    from gareus.logger import DistanceLogger

    logger = DistanceLogger(tmp_path, argparse.Namespace(timestep_fs=2.0), no_file_persistence=True)
    for w in range(4):
        logger.history_by_window[w] = [1.0 + 0.5 * w + 0.01 * i for i in range(30)]
    rows = [{"replica": w, "window": w, "center_A": 1.0 + 0.5 * w, "k_kcal_mol_A2": 2.5,
             "cv_A": 1.0} for w in range(4)]
    ctx = build_context(
        logger=logger, rows=rows, phase="gareus_production", step=1, total_steps=10,
        summary={}, dashboard_info={"centers_a": [1.0, 1.5, 2.0, 2.5], "n_windows": 4,
                                    "k_list": [2.5] * 4, "exchange_stats": {}},
        sidecar=SidecarSnapshot(), term_w=140, term_h=45, now=0.0, view="windows",
        glyphs="unicode")
    assert not ctx.is_2d
    assert set(ctx.overlap_pairs) <= {(0, 1), (1, 2), (2, 3)}
    assert ctx.overlap_windows == {}
