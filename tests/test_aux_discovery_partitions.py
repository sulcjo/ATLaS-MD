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
    Xlow = 0.01 * rng.normal(size=(n, 1))   # sd ~0.01: kept but sd-floored
    X = np.hstack([Xa, Xb, Xlow]); fam = ["a"] * 6 + ["b"] * 7
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


def test_planted_pins_sd_floor():
    X, fam, cv, tr, ho, lin, step, _ = _planted()
    res = P.fit_partition(X, fam, cv, tr, ho, lin, step, AuxDiscoverySettings(k_max=3), seed=0)
    assert res.frozen.keep[-1] and res.frozen.sd[-1] == 0.05


def test_hidden_fraction_hand_computed():
    # label fully determined by bin -> hidden fraction ~ 0 ; label independent of bin -> ~ 1
    lab = np.array([0, 1] * 200); bins = lab.copy()
    tr = np.arange(400) < 200
    assert P.hidden_fraction(lab, bins, tr, ~tr, 2, 2) < 0.1
    bins2 = np.tile([0, 0, 1, 1], 100)
    assert abs(P.hidden_fraction(lab, bins2, tr, ~tr, 2, 2) - 1.0) < 0.05


def test_co_occurrence_floors():
    s = AuxDiscoverySettings()
    # bin 0: 100 rows 50/50 -> reported; bin 1: 100 rows 95/5 -> one group above 10 %; bin 2: 1 row (<2 %)
    lab = np.r_[np.tile([0, 1], 50), np.zeros(95, int), np.ones(5, int), [1]]
    bins = np.r_[np.zeros(100, int), np.ones(100, int), [2]]
    out = P.co_occurrence(lab, bins, np.ones(201, bool), 2, s)
    assert [o["bin"] for o in out] == [0] and out[0]["groups"] == [0, 1]


def test_too_few_rows_is_insufficient_evidence():
    X, fam, cv, tr, ho, lin, step, _ = _planted()
    tiny = np.zeros_like(ho); tiny[np.nonzero(ho)[0][:10]] = True
    res = P.fit_partition(X, fam, cv, tr, tiny, lin, step, AuxDiscoverySettings(k_max=3), seed=0)
    assert res.status == "insufficient_evidence" and res.choice.reason == "too_few_rows"


def test_frozen_roundtrip_and_chunked_predict(tmp_path):
    X, fam, cv, tr, ho, lin, step, _ = _planted()
    res = P.fit_partition(X, fam, cv, tr, ho, lin, step, AuxDiscoverySettings(k_max=3), seed=0)
    f = tmp_path / "p.pkl"; res.frozen.to_file(f)
    g = P.FrozenPartition.from_file(f)
    np.testing.assert_array_equal(g.predict(X, cv), res.choice.labels)


# ---- F05: conditioning coordinate contract -------------------------------------------------------------------
import pytest


def _fit(cv, names=None, **kw):
    X, fam, _cv, tr, ho, lin, step, _ = _planted()
    return P.fit_partition(X, fam, cv, tr, ho, lin, step, AuxDiscoverySettings(k_max=4), seed=0, cv_names=names, **kw), \
        (X, fam, tr, ho, lin, step)


def test_two_d_bins_equal_the_former_formula():
    X, fam, cv, tr, ho, lin, step, _ = _planted()
    res, _ = _fit(cv)
    S = (cv - cv[tr].mean(0)) / cv[tr].std(0)
    edges = [np.quantile(S[tr, a], np.linspace(0, 1, 7)[1:-1]) for a in range(2)]
    legacy = np.digitize(S[:, 0], edges[0]) * 6 + np.digitize(S[:, 1], edges[1])
    np.testing.assert_array_equal(res.bins, legacy)
    assert res.n_cond_bins == 36 and res.conditioning["selected"] == ["cv1", "cv2"]


def test_one_dimensional_campaign_uses_n_bins_not_squared():
    X, fam, cv, *_ = _planted()
    res, _ = _fit(cv[:, :1], names=["cv1"])
    assert res.status in ("ok", "keep") and res.conditioning["selected"] == ["cv1"]
    assert res.n_cond_bins == 6 and res.bins.max() < 6 and res.frozen.conditioning.n_bins == 6


def test_constant_cv2_is_dropped_and_equals_the_one_d_fit():
    X, fam, cv, *_ = _planted()
    both = np.c_[cv[:, 0], np.full(len(cv), 0.3, np.float32)]
    res, _ = _fit(both)
    one, _ = _fit(cv[:, :1], names=["cv1"])
    assert res.conditioning["dropped"] == {"cv2": "constant"} and res.conditioning["selected"] == ["cv1"]
    np.testing.assert_array_equal(res.bins, one.bins)
    np.testing.assert_array_equal(res.frozen.predict(X, both), one.frozen.predict(X, cv[:, :1]))


def test_constant_cv1_with_useful_cv2():
    X, fam, cv, *_ = _planted()
    both = np.c_[np.full(len(cv), 5.0, np.float32), cv[:, 1]]
    res, _ = _fit(both)
    assert res.conditioning["dropped"] == {"cv1": "constant"} and res.conditioning["selected"] == ["cv2"]
    assert res.status in ("ok", "keep") and res.n_cond_bins == 6


def test_tiny_finite_variance_is_usable_and_the_threshold_is_scale_aware():
    X, fam, cv, tr, *_ = _planted()
    tiny = np.c_[cv[:, 0], (1e-6 * cv[:, 1]).astype(np.float32)]
    assert _fit(tiny)[0].conditioning["selected"] == ["cv1", "cv2"]
    # a huge mean with sd below 1e-12 * |mean| is constant; with a real spread it is not
    spec, bad = P.fit_conditioning(np.c_[cv[:, 0], np.full(len(cv), 1e9, np.float32)], tr, 6)
    assert bad is None and spec.dropped == {"cv2": "constant"}


def test_one_unexpected_nan_is_invalid_input_not_a_crash():
    X, fam, cv, *_ = _planted()
    bad = cv.copy(); bad[17, 1] = np.nan
    res, _ = _fit(bad)
    assert res.status == "invalid_conditioning_input" and res.frozen is None
    assert res.conditioning["nonfinite"] == ["cv2"]
    inf = cv.copy(); inf[3, 0] = np.inf
    assert _fit(inf)[0].status == "invalid_conditioning_input"


def test_all_conditioning_constant_is_insufficient_conditioning_evidence():
    X, fam, cv, *_ = _planted()
    res, _ = _fit(np.ones_like(cv))
    assert res.status == "insufficient_conditioning_evidence" and res.frozen is None
    assert res.conditioning["dropped"] == {"cv1": "constant", "cv2": "constant"}


def test_too_few_training_rows_stays_too_few_rows():
    X, fam, cv, tr, ho, lin, step, _ = _planted()
    few = np.zeros_like(tr); few[np.nonzero(tr)[0][:3]] = True
    res = P.fit_partition(X, fam, cv, few, ho, lin, step, AuxDiscoverySettings(k_max=3), seed=0)
    assert res.status == "insufficient_evidence" and res.choice.reason == "too_few_rows"


def test_tied_quantiles_create_no_phantom_bins():
    X, fam, cv, tr, *_ = _planted()
    tied = np.c_[np.where(np.arange(len(cv)) % 10 == 0, 1.0, 0.0).astype(np.float32), cv[:, 1]]
    spec, bad = P.fit_conditioning(tied, tr, 6)
    assert bad is None and len(spec.edges[0]) == 1 and spec.n_bins == 2 * 6
    bins = P.apply_bins(spec.transform(tied), spec.edges)
    assert len(np.unique(bins)) <= spec.n_bins and bins.max() < spec.n_bins
    assert P.s_bins(tied[:, :1], tr, 6).max() <= 1          # five identical edges collapsed to one, not 6 bins


def test_holdout_never_refits_scales_and_schema_mismatch_raises():
    X, fam, cv, tr, ho, lin, step, _ = _planted()
    res, _ = _fit(cv)
    spec = res.frozen.conditioning
    np.testing.assert_allclose(spec.mean, cv[tr].mean(0), rtol=1e-6)
    np.testing.assert_allclose(spec.sd, cv[tr].std(0), rtol=1e-6)
    before = (spec.mean.copy(), spec.sd.copy(), [e.copy() for e in spec.edges])
    shifted = cv.copy(); shifted[ho] = shifted[ho] * 50 + 100          # wildly different holdout distribution
    res.frozen.predict(X, shifted)
    assert np.array_equal(spec.mean, before[0]) and np.array_equal(spec.sd, before[1])
    assert all(np.array_equal(a, b) for a, b in zip(spec.edges, before[2]))
    S = spec.transform(shifted)
    np.testing.assert_allclose(S[tr], (shifted[tr] - spec.mean) / spec.sd, rtol=1e-6)
    with pytest.raises(ValueError, match="schema mismatch"):
        res.frozen.predict(X, cv[:, :1])
    nan = cv.copy(); nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        res.frozen.predict(X, nan)


def test_legacy_frozen_partition_without_the_field_still_predicts_two_d(tmp_path):
    X, fam, cv, *_ = _planted()
    res, _ = _fit(cv)
    f = res.frozen
    del f.__dict__["conditioning"]                       # a pickle written before the contract
    np.testing.assert_array_equal(f.predict(X, cv), res.choice.labels)
    path = tmp_path / "p.pkl"; f.to_file(path)
    np.testing.assert_array_equal(P.FrozenPartition.from_file(path).predict(X, cv), res.choice.labels)
