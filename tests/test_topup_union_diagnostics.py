import math

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
