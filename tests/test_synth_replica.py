"""Replica-count study: rugged-1d landscape + fair-k calibration + replica_run."""
from __future__ import annotations

import numpy as np
import pytest

from gareus.synth.landscapes import LANDSCAPES, rugged_1d
from gareus.synth.oracle import reference_pmf


def test_rugged_1d_has_six_basins_and_a_large_barrier():
    lp = rugged_1d()
    assert lp.name == "rugged-1d"
    assert len(lp.basins) == 6
    c1, c2, f = lp.grid(res=80)
    assert np.isclose(f.min(), 0.0) and np.all(np.isfinite(f))
    x, g = reference_pmf(lp, axis="cv1", res=400)
    mins = [i for i in range(1, len(g) - 1) if g[i] < g[i - 1] and g[i] < g[i + 1]]
    assert len(mins) == 6                       # six CV1 basins
    barrier = g[(x >= 0.36) & (x <= 0.56)].max()
    assert barrier > 8.0                        # a large (>8 kBT) barrier between basins


def test_calibrate_k_rises_with_R_and_meets_overlap():
    pytest.importorskip("scipy")
    from gareus.synth.replica import calibrate_k
    lp = LANDSCAPES["rugged-1d"]
    ks = []
    for R in (2, 4, 8, 12):
        centers, k, min_ov, valid = calibrate_k(lp, R, overlap_target=0.30)
        assert valid and len(centers) == R
        assert min_ov >= 0.30 - 0.02            # connected ladder at the target
        ks.append(k)
    assert ks == sorted(ks)                     # tighter k needed for more (closer) windows


def test_replica_run_u_curve_mid_beats_extreme():
    pytest.importorskip("scipy")
    from gareus.synth.replica import replica_run
    lp = LANDSCAPES["rugged-1d"]
    B = 50000
    r4 = replica_run(lp, 4, B, seed=0)
    r12 = replica_run(lp, 12, B, seed=0)
    assert r4["gated_ok"] and r12["gated_ok"]
    # at fixed budget a moderate replica count beats spreading too thin
    assert r4["pmf_rmse_lowf"] < r12["pmf_rmse_lowf"]
    # more replicas -> fewer effective samples per window + more burn-in overhead
    assert r12["min_per_window_ess"] < r4["min_per_window_ess"]
    assert r12["burnin_fraction"] > r4["burnin_fraction"]


def test_replica_run_single_window_cannot_bridge_barrier():
    pytest.importorskip("scipy")
    from gareus.synth.replica import replica_run
    lp = LANDSCAPES["rugged-1d"]
    r1 = replica_run(lp, 1, 50000, seed=0)
    r4 = replica_run(lp, 4, 50000, seed=0)
    # one unbiased window samples but cannot reconstruct relative basin depths
    # across the tall barrier -> worse than a proper ladder
    assert np.isfinite(r1["pmf_rmse_lowf"])
    assert r1["pmf_rmse_lowf"] > r4["pmf_rmse_lowf"]
