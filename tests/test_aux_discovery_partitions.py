import numpy as np
from gareus.adaptive.aux_discovery import partitions as P
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings


def _planted(n_lin=40, per=60, hidden=True, seed=0):
    """Two hidden states at fixed (cv1, cv2): descriptor family 'a' separates them, cv does not."""
    rng = np.random.default_rng(seed)
    n = n_lin * per
    lineage = np.repeat([f"p:{i}" for i in range(n_lin)], per)
    step = np.tile(np.arange(per) * 3000, n_lin)
    cv = rng.normal(size=(n, 2)).astype(np.float32)
    state = (np.repeat(rng.integers(0, 2, n_lin), per) if hidden else np.zeros(n, int))
    flip = rng.random(n) < 0.05
    state = np.where(flip, 1 - state, state)
    Xa = rng.normal(size=(n, 6)) + (3.0 * state[:, None] if hidden else 0.0)
    Xb = rng.normal(size=(n, 6))
    X = np.hstack([Xa, Xb]); fam = ["a"] * 6 + ["b"] * 6
    train = np.repeat(np.arange(n_lin) < 28, per); holdout = ~train
    return X, fam, cv, train, holdout, lineage, step, state


def test_planted_hidden_mode_found():
    X, fam, cv, tr, ho, lin, step, state = _planted()
    res = P.fit_partition(X, fam, cv, tr, ho, lin, step, AuxDiscoverySettings(k_max=4), seed=0)
    assert res.status == "ok" and res.choice.k >= 2
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(state[ho], res.choice.labels[ho]) > 0.8
    np.testing.assert_array_equal(res.frozen.predict(X, cv), res.choice.labels)


def test_negative_control_insufficient_evidence():
    X, fam, cv, tr, ho, lin, step, _ = _planted(hidden=False)
    res = P.fit_partition(X, fam, cv, tr, ho, lin, step, AuxDiscoverySettings(k_max=4), seed=0)
    assert res.status in ("insufficient_evidence", "keep")


def test_preprocess_drops_constant_and_floors_sd():
    X = np.c_[np.ones(50), np.linspace(0, 1e-2, 50), np.random.default_rng(0).normal(size=50)]
    tr = np.ones(50, bool)
    Z, keep, mean, sd = P.preprocess(X, ["a", "a", "b"], tr, AuxDiscoverySettings())
    assert keep.tolist() == [False, True, True]
    assert np.isclose(sd[1], 0.05)                          # floored
    assert Z.shape == (50, 2)


def test_lineage_bootstrap_keeps_multiplicity():
    rng = np.random.default_rng(0)
    idx = P.lineage_bootstrap_rows(np.array(["a", "a", "b", "c"]), rng)
    assert len(idx) >= 1
    lin = np.repeat(["x", "y"], 3)
    counts = [len(P.lineage_bootstrap_rows(lin, np.random.default_rng(s))) for s in range(20)]
    assert set(counts) == {6}                               # 2 lineages drawn with replacement -> 6 rows
    # a lineage drawn twice contributes its rows twice
    seen_repeat = any(len(set(P.lineage_bootstrap_rows(lin, np.random.default_rng(s)).tolist())) == 3
                      for s in range(20))
    assert seen_repeat


def test_info_gain_positive_for_informative_z():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000); z = y + 0.3 * rng.normal(size=4000); bins = np.zeros(4000, int)
    tr = np.arange(4000) < 3000
    assert P.info_gain(z, y, bins, tr, ~tr, 2, 1) > 0.3
    assert abs(P.info_gain(rng.normal(size=4000), y, bins, tr, ~tr, 2, 1)) < 0.02


def test_lineage_info_matches_loop_reference():
    rng = np.random.default_rng(3)
    n, K, nb, c = 600, 3, 4, 5.0
    lineage = np.repeat([f"l{i}" for i in range(12)], 50)
    step = np.tile(np.arange(50), 12)
    lab = rng.integers(0, K, n); bins = rng.integers(0, nb, n)
    got = P.lineage_info(lab, bins, lineage, step, K, nb, c)
    # reference: straightforward python loop
    lin_id = np.unique(lineage, return_inverse=True)[1]
    order = np.lexsort((step, lin_id))
    ls, ys, bs = lin_id[order], lab[order], bins[order]
    first = np.zeros(n, bool)
    for l in np.unique(ls):
        idx = np.nonzero(ls == l)[0]; first[idx[: idx.size // 2]] = True
    pA = P._cond_table(ys[first], bs[first], K, nb)
    cnt = {}
    for l, b, y in zip(ls[first], bs[first], ys[first]):
        cnt.setdefault((l, b), np.zeros(K))[y] += 1
    lt = [np.log(((cnt.get((l, b), np.zeros(K)) + c * pA[b]) / (cnt.get((l, b), np.zeros(K)).sum() + c))[y])
          for l, b, y in zip(ls[~first], bs[~first], ys[~first])]
    ref = float(np.mean(lt) - P._ll(pA, ys[~first], bs[~first]))
    assert np.isclose(got, ref, atol=1e-12)
