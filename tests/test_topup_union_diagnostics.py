import math
from unittest import mock

import numpy as np

from gareus.adaptive.union_diagnostics import union_diagnostics_from_npz

KT = 0.596  # kcal/mol at 300 K


def _write(tmp_path, centers, k, n_per, seed=0, drift_state=None, name="u.npz"):
    rng = np.random.default_rng(seed)
    x, sid = [], []
    for s, c in enumerate(centers):
        sig = 1.0 / math.sqrt(k)
        xs = rng.normal(c, sig, n_per)
        if drift_state == s:                       # second half sits somewhere else
            xs[n_per // 2:] += 4 * sig
        x.append(xs); sid += [s] * n_per
    x = np.concatenate(x)
    u = 0.5 * k * (x[:, None] - np.asarray(centers)[None, :]) ** 2   # reduced (kT units)
    p = tmp_path / name
    np.savez(p, umbrella_reduced_bias_nk=u, state_ids=np.arange(len(centers)),
             sampled_state_ids=np.asarray(sid))
    return p


def _chain_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def test_healthy_chain_has_small_local_sigma_and_all_edges_measured(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0, 3.0], k=4.0, n_per=2000)
    d = union_diagnostics_from_npz(p, _chain_edges(4), kt_kcal=KT)
    assert d is not None
    assert all(math.isfinite(d.sigma_kcal[s]) and d.sigma_kcal[s] > 0 for s in range(4))
    assert d.sigma_kcal[3] < 2.0 * d.sigma_kcal[1]          # local sigma: no penalty for distance from state 0
    assert set(d.edge_overlap) == set(_chain_edges(4)) and min(d.edge_overlap.values()) > 0.15
    assert not d.unconverged


def test_fewer_samples_means_larger_sigma(tmp_path):
    big = union_diagnostics_from_npz(_write(tmp_path, [0.0, 1.0], 4.0, 4000, name="b.npz"), [(0, 1)], kt_kcal=KT)
    small = union_diagnostics_from_npz(_write(tmp_path, [0.0, 1.0], 4.0, 250, name="s.npz"), [(0, 1)], kt_kcal=KT)
    assert small.sigma_kcal[1] > big.sigma_kcal[1]


def test_a_state_whose_halves_disagree_is_flagged(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0], k=4.0, n_per=3000, drift_state=1)
    d = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT)
    assert 1 in d.unconverged


def test_false_alarms_are_controlled_at_production_scale(tmp_path):
    # 236 iid states: a fixed 2-sigma rule flags ~10 states; Bonferroni keeps the family error ~alpha.
    p = _write(tmp_path, [0.5 * i for i in range(236)], k=4.0, n_per=160, seed=7)
    d = union_diagnostics_from_npz(p, _chain_edges(236), kt_kcal=KT)
    assert d is not None and len(d.unconverged) <= 2


def test_inefficiency_comes_from_the_builder_meta(tmp_path):
    p = _write(tmp_path, [0.0, 1.0], k=4.0, n_per=500)
    counts = {"0": {"raw": 5000, "t0": 500, "kept": 500, "g": 9.0, "status": "subsampled"},
              "1": {"raw": 5000, "t0": 0, "kept": 500, "g": float("nan"), "status": "detect_failed"}}
    d = union_diagnostics_from_npz(p, [(0, 1)], kt_kcal=KT, subsample_counts=counts)
    assert d.inefficiency[0] == 9.0
    assert d.inefficiency[1] == 10.0                        # fallback: (raw - t0) / kept
    assert d.n_k == {0: 500, 1: 500}


def test_a_warm_start_reproduces_the_cold_solution(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0], k=4.0, n_per=1500)
    cold = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT)
    warm = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT, f_init=cold.f_kT)
    assert all(abs(cold.f_kT[s] - warm.f_kT[s]) < 1e-6 for s in range(3))


def test_missing_or_empty_input_returns_none(tmp_path):
    assert union_diagnostics_from_npz(tmp_path / "nope.npz", [], kt_kcal=KT) is None
    p = tmp_path / "e.npz"
    np.savez(p, umbrella_reduced_bias_nk=np.full((3, 2), np.nan), state_ids=np.arange(2),
             sampled_state_ids=np.array([0, 1, 1]))
    assert union_diagnostics_from_npz(p, [(0, 1)], kt_kcal=KT) is None


def test_a_state_with_zero_samples_gets_nan_sigma_not_a_crash(tmp_path):
    p = _write(tmp_path, [0.0, 1.0, 2.0], k=4.0, n_per=1000)
    with np.load(p) as z:
        u, ids, sid = z["umbrella_reduced_bias_nk"], z["state_ids"], z["sampled_state_ids"]
    keep = sid != 2
    np.savez(p, umbrella_reduced_bias_nk=u[keep], state_ids=ids, sampled_state_ids=sid[keep])
    d = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT)
    assert d is not None and math.isnan(d.sigma_kcal[2]) and d.n_k[2] == 0


def test_split_halves_is_specific_not_leaking(tmp_path):
    """9-state chain with only state 4 drifting → only state 4 flagged."""
    p = _write(tmp_path, [0.5 * i for i in range(9)], k=4.0, n_per=3000, drift_state=4)
    d = union_diagnostics_from_npz(p, _chain_edges(9), kt_kcal=KT)
    assert d is not None
    assert d.unconverged == frozenset({4})


def test_warm_start_is_actually_passed(tmp_path):
    """Verify that initial_f_k is passed to MBAR when f_init is given."""
    from pymbar import MBAR as RealMBAR

    p = _write(tmp_path, [0.0, 1.0, 2.0], k=4.0, n_per=1500)
    cold = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT)

    # Save the real MBAR init
    real_init = RealMBAR.__init__
    recorded_kwargs = {}

    def mock_mbar_init(self, *args, **kwargs):
        recorded_kwargs.update(kwargs)
        return real_init(self, *args, **kwargs)

    with mock.patch('pymbar.MBAR.__init__', mock_mbar_init):
        # Call with warm start
        recorded_kwargs.clear()
        warm = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT, f_init=cold.f_kT)

        # Verify initial_f_k was passed
        assert "initial_f_k" in recorded_kwargs
        assert recorded_kwargs["initial_f_k"] is not None
        assert warm is not None


def test_unsampled_endpoint_edge_absent_from_overlap(tmp_path):
    """An edge with an unsampled endpoint is absent from edge_overlap."""
    p = _write(tmp_path, [0.0, 1.0, 2.0], k=4.0, n_per=1000)
    with np.load(p) as z:
        u, ids, sid = z["umbrella_reduced_bias_nk"], z["state_ids"], z["sampled_state_ids"]
    keep = sid != 2
    np.savez(p, umbrella_reduced_bias_nk=u[keep], state_ids=ids, sampled_state_ids=sid[keep])
    d = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT)
    assert d is not None
    assert (1, 2) not in d.edge_overlap


def test_non_contiguous_state_ids(tmp_path):
    """State ids [3, 7, 20] should work and map correctly."""
    rng = np.random.default_rng(42)
    centers = [0.0, 1.0, 2.0]
    states = [3, 7, 20]
    k = 4.0
    n_per = 1000

    x, sid = [], []
    for s, c in zip(states, centers):
        sig = 1.0 / math.sqrt(k)
        xs = rng.normal(c, sig, n_per)
        x.append(xs)
        sid += [s] * n_per

    x = np.concatenate(x)
    u = 0.5 * k * (x[:, None] - np.asarray(centers)[None, :]) ** 2
    p = tmp_path / "noncontig.npz"
    np.savez(p, umbrella_reduced_bias_nk=u, state_ids=np.array(states),
             sampled_state_ids=np.array(sid))

    d = union_diagnostics_from_npz(p, [(3, 7), (7, 20)], kt_kcal=KT)
    assert d is not None
    assert set(d.n_k.keys()) == {3, 7, 20}
    assert set(d.sigma_kcal.keys()) == {3, 7, 20}
    assert (3, 7) in d.edge_overlap
    assert (7, 20) in d.edge_overlap


def test_warning_is_logged_on_missing_file(tmp_path, caplog):
    """A warning is logged when the input file is missing."""
    import logging
    with caplog.at_level(logging.WARNING):
        result = union_diagnostics_from_npz(tmp_path / "nope.npz", [], kt_kcal=KT)
    assert result is None
    assert "not found" in caplog.text.lower() or "missing" in caplog.text.lower()


def test_warning_is_logged_on_all_nan_input(tmp_path, caplog):
    """A warning is logged when all rows are excluded (non-finite)."""
    import logging
    p = tmp_path / "e.npz"
    np.savez(p, umbrella_reduced_bias_nk=np.full((3, 2), np.nan), state_ids=np.arange(2),
             sampled_state_ids=np.array([0, 1, 1]))
    with caplog.at_level(logging.WARNING):
        result = union_diagnostics_from_npz(p, [(0, 1)], kt_kcal=KT)
    assert result is None
    assert "no finite rows" in caplog.text.lower() or "remain" in caplog.text.lower()


def _write_seeded(tmp_path, centers, k, n_per, name):
    """Per-state RNG streams: a state's samples do not depend on which other states exist."""
    x, sid = [], []
    for s, c in enumerate(centers):
        rng = np.random.default_rng([11, s])
        x.append(rng.normal(c, 1.0 / math.sqrt(k), n_per)); sid += [s] * n_per
    x = np.concatenate(x)
    u = 0.5 * k * (x[:, None] - np.asarray(centers)[None, :]) ** 2
    p = tmp_path / name
    np.savez(p, umbrella_reduced_bias_nk=u, state_ids=np.arange(len(centers)), sampled_state_ids=np.asarray(sid))
    return p


def test_edge_overlap_does_not_shrink_when_the_pair_sits_in_a_bigger_union(tmp_path):
    # The same pair (states 0, 1) alone with one neighbour, then crowded by 27 more states
    # overlapping it; the full-union overlap matrix would dilute ~1/degree, the pairwise one must not.
    small = [0.0, 1.0, 2.0]
    crowded = small + list(np.linspace(-0.5, 1.5, 27))
    a = union_diagnostics_from_npz(_write_seeded(tmp_path, small, 4.0, 2000, "a.npz"), [(0, 1)], kt_kcal=KT)
    b = union_diagnostics_from_npz(_write_seeded(tmp_path, crowded, 4.0, 2000, "b.npz"), [(0, 1)], kt_kcal=KT)
    assert abs(a.edge_overlap[(0, 1)] - b.edge_overlap[(0, 1)]) < 2e-3


def test_two_identical_states_have_pairwise_overlap_one_half(tmp_path):
    d = union_diagnostics_from_npz(_write_seeded(tmp_path, [0.0, 0.0], 4.0, 1000, "c.npz"), [(0, 1)], kt_kcal=KT)
    assert math.isclose(d.edge_overlap[(0, 1)], 0.5, rel_tol=1e-9)


def test_a_sparse_state_is_not_rescued_by_a_well_sampled_rung_twin(tmp_path):
    # Ruling 23: sigma_k = max dDelta_f over the state's same-rung spatial neighbours (rung
    # partners only as fallback). State 0 is sparse; state 1 is its rung twin (same centre,
    # a slightly tilted Hamiltonian) and well sampled; state 2 is 0's spatial neighbour.
    k, tilt = 4.0, 0.05
    rng = np.random.default_rng(3)
    n = {0: 30, 1: 3000, 2: 3000}
    centre = {0: 0.0, 1: 0.0, 2: 1.0}
    x = np.concatenate([rng.normal(centre[s], 1.0 / math.sqrt(k), n[s]) for s in (0, 1, 2)])
    sid = np.concatenate([[s] * n[s] for s in (0, 1, 2)])
    u = np.column_stack([0.5 * k * x ** 2, 0.5 * k * x ** 2 + tilt * x, 0.5 * k * (x - 1.0) ** 2])
    p = tmp_path / "twin.npz"
    np.savez(p, umbrella_reduced_bias_nk=u, state_ids=np.arange(3), sampled_state_ids=sid)
    edges = [(0, 1), (0, 2)]
    d = union_diagnostics_from_npz(p, edges, kt_kcal=KT, sigma_neighbours={0: [2], 1: [0], 2: [0]})
    via_twin = union_diagnostics_from_npz(p, edges, kt_kcal=KT, sigma_neighbours={0: [1], 1: [0], 2: [0]})
    assert d.sigma_kcal[0] > 3.0 * via_twin.sigma_kcal[0]       # the twin no longer hides the sparse state
    default = union_diagnostics_from_npz(p, edges, kt_kcal=KT)      # edge neighbours {1, 2}: max picks the spatial link
    assert math.isclose(default.sigma_kcal[0], d.sigma_kcal[0], rel_tol=1e-9)


def test_pair_overlap_with_f_held_fixed_is_identical_in_a_3_and_a_30_state_union(tmp_path):
    from gareus.adaptive.union_diagnostics import _pair_overlap
    small = [0.0, 1.0, 2.0]
    crowded = small + list(np.linspace(-0.5, 1.5, 27))
    got = []
    for centers, name in ((small, "s.npz"), (crowded, "l.npz")):
        with np.load(_write_seeded(tmp_path, centers, 4.0, 500, name)) as z:
            u, w = z["umbrella_reduced_bias_nk"], z["sampled_state_ids"]
        f = np.zeros(len(centers)); f[1] = 0.3                      # the pair's f, held fixed
        got.append(_pair_overlap(u, w, f, np.bincount(w, minlength=len(centers)), 0, 1))
    assert got[0] == got[1]


def test_sigma_argmax_names_the_neighbour_that_sets_sigma(tmp_path):
    """State 1 sits between a well-sampled 0 and a sparse 2: the 1-2 link is the least certain."""
    rng = np.random.default_rng(3)
    centers, k, n = [0.0, 1.0, 2.0], 4.0, [4000, 4000, 60]
    x = np.concatenate([rng.normal(c, 0.5, m) for c, m in zip(centers, n)])
    sid = np.concatenate([np.full(m, s) for s, m in enumerate(n)])
    u = 0.5 * k * (x[:, None] - np.asarray(centers)[None, :]) ** 2
    p = tmp_path / "a.npz"
    np.savez(p, umbrella_reduced_bias_nk=u, state_ids=np.arange(3), sampled_state_ids=sid)
    d = union_diagnostics_from_npz(p, _chain_edges(3), kt_kcal=KT, split_halves=False)
    assert d.sigma_argmax[1] == 2 and d.sigma_argmax[0] == 1 and d.sigma_argmax[2] == 1


def test_sigma_argmax_is_absent_on_the_no_neighbour_fallback(tmp_path):
    p = _write(tmp_path, [0.0, 1.0], k=4.0, n_per=500)
    d = union_diagnostics_from_npz(p, [(0, 1)], kt_kcal=KT, sigma_neighbours={0: [], 1: [0]})
    assert 0 not in d.sigma_argmax and d.sigma_argmax[1] == 0
