"""Respring confidence is measured on the variance, not on the CV2 series (verification F2).

The decision reads c_real = k2 Var(z) / RT, a second moment. Its correlation time is that of
q = (z - mean z)^2, which a slowly switching amplitude makes far longer than z's own. The
decision uses g_variance = max(g_cv2, g_q); bootstrap blocks never cross a sample source.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gareus.adaptive import cv2_respring as rs  # noqa: E402
from gareus.swarm.ladder_design import R_KCAL_MOL_K  # noqa: E402

RT = R_KCAL_MOL_K * 300.0
S = rs.RespringSettings()


def _view(k2):
    return {"state_id": 0, "c1": 0.2, "k1": 500.0, "c2": 0.5, "k2": k2, "lam": 0.0, "created_epoch": 0,
            "metadata": {}, "members": [0], "mandatory": False}


def _amplitude_switching(n, dwell, seed):
    """z = sigma(t) eps: iid sign/phase, two widths, rare switches (mean dwell ``dwell`` rows)."""
    rng = np.random.default_rng(seed)
    flips = rng.random(n) < 1.0 / dwell
    state = np.cumsum(flips) % 2
    sigma = np.where(state == 0, 0.3, 1.0)
    return sigma * rng.normal(size=n)


def test_iid_gaussian_interval_contains_the_generating_compression():
    k2, f2_true = 1.18, 2.0
    var_true = RT / (k2 + f2_true)
    z = np.random.default_rng(11).normal(0.0, math.sqrt(var_true), 8000)
    m = rs.measure_window(var=float(np.var(z)), n=z.size, k2=k2, rt=RT, z=z, g=None)
    assert m["g_variance_status"] == "ok" and m["bootstrap_status"] == "ok"
    assert 1.0 <= m["g_variance"] < 1.5
    assert m["n_eff_variance"] == pytest.approx(z.size / m["g_variance"]) and m["n_eff_variance"] > 5000
    c_true = k2 / (k2 + f2_true)
    assert m["c_interval"][0] < c_true < m["c_interval"][1]


def test_slow_amplitude_switching_is_skipped_not_proposed():
    z = _amplitude_switching(20000, dwell=2000, seed=4)
    k2 = 0.3                                              # c_real ~ 0.27: would trigger
    g_z = 1.05                                            # what X5 reports for this z
    m = rs.measure_window(var=float(np.var(z)), n=z.size, k2=k2, rt=RT, z=z, g=g_z)
    assert m["g_q"] > 50 * g_z
    assert m["g_variance"] == m["g_q"] and m["g_variance_source"] == "variance"
    assert m["n_eff_variance"] < S.respring_min_neff
    cand = rs.evaluate(_view(k2), m, S, epoch=1, touched=[])
    assert cand["decision"] == "skipped" and cand["reason"] == "too_few_effective_samples"
    # the pre-fix rule (n_eff = n / g_z) would have acted on it
    old = dict(m, n_eff=z.size / g_z)
    assert rs.evaluate(_view(k2), old, S, epoch=1, touched=[])["decision"] == "proposed"


def test_gaussian_g_q_is_below_g_z_so_the_max_keeps_g_z():
    rng = np.random.default_rng(2)
    rho, n = 0.9, 20000
    z = np.empty(n)
    z[0] = rng.normal()
    e = rng.normal(size=n) * math.sqrt(1 - rho * rho)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + e[i]
    m = rs.measure_window(var=float(np.var(z)), n=n, k2=1.0, rt=RT, z=z, g=None)
    assert m["g_q"] < m["g_cv2"]                          # (1 + r^2)/(1 - r^2) = 9.5 < (1 + r)/(1 - r) = 19
    assert m["g_variance"] == m["g_cv2"] and m["g_variance_source"] == "cv2"


def test_failed_variance_estimate_is_never_read_as_g_one():
    z = np.tile([0.1, -0.1], 10)                          # q constant: zero variance of q
    m = rs.measure_window(var=float(np.var(z)), n=z.size, k2=1.0, rt=RT, z=z, g=1.0)
    assert m["g_variance"] is None and m["n_eff"] is None
    assert m["g_variance_status"].startswith("variance_inefficiency_")
    cand = rs.evaluate(_view(1.0), m, S, epoch=1, touched=[])
    assert cand["decision"] == "skipped" and cand["reason"] == "variance_inefficiency_unavailable"


def _spy_block_ids(monkeypatch):
    calls = []
    real = rs.block_ids

    def spy(n, source_index, block_len):
        ids = real(n, source_index, block_len)
        calls.append((None if source_index is None else np.asarray(source_index).copy(), ids))
        return ids
    monkeypatch.setattr(rs, "block_ids", spy)
    return calls


def test_no_block_crosses_a_source_including_the_guard(monkeypatch):
    src = np.repeat([0, 1, 2], 7)                          # n = 21
    z = np.random.default_rng(0).normal(size=src.size)
    calls = _spy_block_ids(monkeypatch)
    _ratio, info = rs.bootstrap_var_ratio(z, src, block_len=50)      # forces the guard
    assert info["source_boundary_guard"] is True and info["bootstrap_status"] == "ok"
    assert len(calls) == 2
    for s, ids in calls:
        assert s is not None                              # the guard keeps source_index
        for b in np.unique(ids):
            assert np.unique(src[ids == b]).size == 1
    assert info["n_blocks"] == 3 * math.ceil(7 / (21 // rs.BOOT_MIN_BLOCKS))    # 6 source-local, not 5


def test_stratified_replicates_keep_each_sources_row_count(monkeypatch):
    src = np.repeat([0, 1], [40, 10])
    z = np.where(src == 0, 0.0, 5.0) + np.random.default_rng(1).normal(size=src.size) * 0.1
    ratio, info = rs.bootstrap_var_ratio(z, src, block_len=5, reps=100)
    # both sources keep their share in every replicate, so the between-source term never
    # vanishes and never doubles: var_b / var stays near 1
    assert info["bootstrap_status"] == "ok"
    assert ratio.min() > 0.7 and ratio.max() < 1.3


def test_too_few_source_blocks_gives_no_interval():
    z = np.array([0.1, -0.2, 0.3, 0.05])
    ratio, info = rs.bootstrap_var_ratio(z, np.zeros(4, dtype=int), block_len=10)
    assert ratio.size == 0 and info["bootstrap_status"] == "too_few_source_blocks"
    m = {"n": 5000, "var": 0.1, "sd": math.sqrt(0.1), "c_real": 0.2, "f2_prod": 4.0, "n_eff": 1000.0,
         "g_variance_status": "ok", "bootstrap_status": "too_few_source_blocks", "c_interval": None}
    cand = rs.evaluate(_view(1.0), m, S, epoch=1, touched=[])
    assert cand["decision"] == "skipped" and cand["reason"] == "too_few_source_blocks"


def test_repeated_calls_are_deterministic():
    src = np.repeat([0, 1, 2], 500)
    z = _amplitude_switching(src.size, dwell=100, seed=9)
    a = rs.measure_window(var=float(np.var(z)), n=z.size, k2=1.0, rt=RT, z=z, source_index=src, g=None, seed=3)
    b = rs.measure_window(var=float(np.var(z)), n=z.size, k2=1.0, rt=RT, z=z, source_index=src, g=None, seed=3)
    assert a == b


def test_source_row_order_does_not_change_the_measurement():
    src = np.repeat([0, 1, 2], 400)
    z = _amplitude_switching(src.size, dwell=80, seed=12)
    order = np.concatenate([np.flatnonzero(src == s) for s in (2, 0, 1)])      # same labels, other order
    kw = dict(var=float(np.var(z)), n=z.size, k2=1.0, rt=RT, g=None, seed=5)
    a = rs.measure_window(z=z, source_index=src, **kw)
    b = rs.measure_window(z=z[order], source_index=src[order], **kw)
    assert a.keys() == b.keys() and a["bootstrap"] == b["bootstrap"]
    for k in a:                                            # equal up to summation-order rounding
        if isinstance(a[k], (float, list)):
            assert b[k] == pytest.approx(a[k], rel=1e-12, abs=1e-12), k
        else:
            assert a[k] == b[k], k
