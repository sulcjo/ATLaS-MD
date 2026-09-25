"""Ruling 26: rung/ladder edges must be scored by the PAIRWISE MBAR state
overlap, not by indexing the full-union ``mbar_state_overlap`` matrix.

The full-union matrix normalises every sample's weight over EVERY state in
the union, so one edge's overlap is diluted by roughly how many other states
share its region. Measured on RUNS/chignolin_7's 64-state union (16 CV1
centres x 4 rungs): the 48 adjacent-rung pairs came out at a full-union
sqrt(O_ij*O_ji) median of 0.089 (38/48 below the 0.15 floor) versus a
pairwise median of 0.258 (0/48 below). ``gareus.mbar_analysis.ladder.
pairwise_state_overlap`` (moved here from the private
``gareus.adaptive.union_diagnostics._pair_overlap``) is the fix; this file
covers its two campaign-end consumers:

- ``gareus.adaptive_production.rung_mbar_overlap_from_union``
- ``gareus.mbar_analysis.ladder_overlap.ladder_overlap_by_axis`` /
  ``ladder_overlap_health_checks`` (via the new ``pair_overlap`` callback)
"""
from __future__ import annotations

import math

import numpy as np

from gareus.mbar_analysis.ladder import mbar_state_overlap, pairwise_state_overlap
from gareus.mbar_analysis.ladder_overlap import (
    LADDER_STATE_OVERLAP_MIN,
    ladder_overlap_by_axis,
    ladder_overlap_health_checks,
)
from gareus.mbar_analysis.solvers import solve_mbar
from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    WindowStateRegistry,
    rung_mbar_overlap_from_union,
)


def _harmonic_union(centers, k, n_per, seed=0):
    """Per-state seeded RNG (independent streams): a state's own samples do
    not depend on which other states exist in the union -- same convention as
    tests/test_topup_union_diagnostics.py's ``_write_seeded``.
    """
    sig = 1.0 / math.sqrt(k)
    x, sid = [], []
    for s, c in enumerate(centers):
        rng = np.random.default_rng([seed, s])
        x.append(rng.normal(c, sig, n_per))
        sid += [s] * n_per
    x = np.concatenate(x)
    window = np.asarray(sid, dtype=np.int64)
    u_nk = 0.5 * k * (x[:, None] - np.asarray(centers)[None, :]) ** 2
    n_k = np.bincount(window, minlength=len(centers))
    return u_nk, window, n_k


# ---------------------------------------------------------------------------
# (a) pairwise value is invariant to union size; mbar_state_overlap's is not
# ---------------------------------------------------------------------------

def test_pairwise_value_identical_in_a_4_and_a_40_state_union_while_full_matrix_drops():
    pair = [0.0, 1.0]
    small = pair + [2.0, 3.0]
    crowded = pair + list(np.linspace(-0.5, 1.5, 36)) + [10.0, 11.0]
    k = 4.0

    u_small, w_small, n_small = _harmonic_union(small, k, n_per=800)
    u_crowd, w_crowd, n_crowd = _harmonic_union(crowded, k, n_per=800)

    # f held fixed for the pair (states 0, 1): pairwise_state_overlap only
    # ever reads f_k[[i, j]], so this is exact regardless of how many other
    # states/columns exist in u_nk -- no MBAR solve needed for this half.
    f_pair = np.zeros(len(small))
    f_pair[1] = 0.3
    f_pair_big = np.zeros(len(crowded))
    f_pair_big[1] = 0.3

    ov_small = pairwise_state_overlap(u_small, w_small, f_pair, n_small, 0, 1)
    ov_crowd = pairwise_state_overlap(u_crowd, w_crowd, f_pair_big, n_crowd, 0, 1)
    assert math.isclose(ov_small, ov_crowd, rel_tol=1e-12)

    # The full-union matrix, by contrast, normalises over every state in the
    # union and so is NOT invariant: crowding the same pair with many nearby
    # decoy states dilutes its reported overlap.
    f_small = np.asarray(solve_mbar(u_small, w_small)["f_k"], dtype=float)
    f_crowd = np.asarray(solve_mbar(u_crowd, w_crowd)["f_k"], dtype=float)
    full_small = mbar_state_overlap(u_small, f_small, n_small)
    full_crowd = mbar_state_overlap(u_crowd, f_crowd, n_crowd)
    full_ov_small = math.sqrt(float(full_small[0, 1]) * float(full_small[1, 0]))
    full_ov_crowd = math.sqrt(float(full_crowd[0, 1]) * float(full_crowd[1, 0]))
    assert full_ov_crowd < 0.5 * full_ov_small


# ---------------------------------------------------------------------------
# (b) rung_mbar_overlap_from_union returns the pairwise values
# ---------------------------------------------------------------------------

def _rung_registry():
    registry = WindowStateRegistry()
    registry.add_state(primary_center=0.0, primary_k=4.0, gamd_lambda=0.0)   # state 0
    registry.add_state(primary_center=0.0, primary_k=4.0, gamd_lambda=1.0)   # state 1 (rung of 0)
    registry.add_state(primary_center=2.0, primary_k=4.0, gamd_lambda=0.0)   # state 2
    registry.add_state(primary_center=2.0, primary_k=4.0, gamd_lambda=1.0)   # state 3 (rung of 2)
    return registry


def test_rung_mbar_overlap_from_union_returns_the_pairwise_values(tmp_path):
    registry = _rung_registry()
    # 4 states, one rung pair per centre: (0.0, tiny lambda offset), (2.0, tiny lambda offset).
    # A small per-lambda energy offset differentiates rungs, matching how a
    # real Pep-GaMD rung's boost is a small perturbation on the same restraint.
    centers = [0.0, 0.0, 2.0, 2.0]
    offsets = [0.0, 0.15, 0.0, 0.15]
    k = 4.0
    sig = 1.0 / math.sqrt(k)
    x, sid = [], []
    for s, c in enumerate(centers):
        rng = np.random.default_rng([7, s])
        x.append(rng.normal(c, sig, 1500))
        sid += [s] * 1500
    x = np.concatenate(x)
    window = np.asarray(sid, dtype=np.int64)
    u_nk = 0.5 * k * (x[:, None] - np.asarray(centers)[None, :]) ** 2 + np.asarray(offsets)[None, :]
    n_k = np.bincount(window, minlength=4)

    npz_path = tmp_path / "adaptive_union_mbar.npz"
    np.savez(npz_path, umbrella_reduced_bias_nk=u_nk, state_ids=np.arange(4),
             sampled_state_ids=window)
    union_meta = {"arrays_npz": str(npz_path)}

    f_k = np.asarray(solve_mbar(u_nk, window)["f_k"], dtype=float)
    expected = {
        (0, 1): pairwise_state_overlap(u_nk, window, f_k, n_k, 0, 1),
        (2, 3): pairwise_state_overlap(u_nk, window, f_k, n_k, 2, 3),
    }

    got = rung_mbar_overlap_from_union(tmp_path, union_meta, registry)

    assert set(got) == set(expected)
    for pair, value in expected.items():
        assert math.isclose(got[pair], value, rel_tol=1e-9)
    # And it should agree with the S3-pilot-calibrated floor/target band for a
    # genuinely healthy rung, not the diluted full-matrix number.
    for value in got.values():
        assert value > AdaptiveDecisionPolicy().min_rung_overlap


# ---------------------------------------------------------------------------
# (c) ladder_overlap health: pairwise no longer FAILs a well-overlapping
#     many-state ladder that the full-matrix version would FAIL.
# ---------------------------------------------------------------------------

def test_ladder_overlap_health_pairwise_does_not_fail_where_full_matrix_does(tmp_path):
    # 16 rungs at ONE centre, deliberately overlapping heavily with each
    # other (small pairwise separation relative to sigma) -- the worst case
    # for the full-union normalisation, which spreads each sample's weight
    # across every one of the 16 competing columns.
    n_rungs = 16
    k = 4.0
    sig = 1.0 / math.sqrt(k)
    lam = np.linspace(0.0, 1.0, n_rungs)
    # small offset per rung so adjacent rungs are distinguishable but still
    # strongly overlapping -- a healthy pairwise chain.
    means = 0.15 * sig * np.arange(n_rungs)
    x, sid = [], []
    n_per = 1500
    for s in range(n_rungs):
        rng = np.random.default_rng([13, s])
        x.append(rng.normal(means[s], sig, n_per))
        sid += [s] * n_per
    x = np.concatenate(x)
    window = np.asarray(sid, dtype=np.int64)
    centers = means  # harmonic restraint centred on each rung's own mean
    u_nk = 0.5 * k * (x[:, None] - centers[None, :]) ** 2
    n_k = np.bincount(window, minlength=n_rungs)
    f_k = np.asarray(solve_mbar(u_nk, window)["f_k"], dtype=float)

    cen = np.zeros(n_rungs)  # all one CV1 centre -> every pair is lambda_direction

    overlap_full = mbar_state_overlap(u_nk, f_k, n_k)
    lo_full, warn_full = ladder_overlap_by_axis(overlap_full, lam, cen, thr=LADDER_STATE_OVERLAP_MIN, n_k=n_k)
    assert warn_full == []
    checks_full = {c["name"]: c["status"] for c in ladder_overlap_health_checks(lo_full, LADDER_STATE_OVERLAP_MIN)}

    def _pair_ov(a, b):
        v = pairwise_state_overlap(u_nk, window, f_k, n_k, a, b)
        return float(v) if math.isfinite(v) and v >= 0.0 else None

    lo_pair, warn_pair = ladder_overlap_by_axis(None, lam, cen, thr=LADDER_STATE_OVERLAP_MIN, n_k=n_k,
                                                 pair_overlap=_pair_ov)
    assert warn_pair == []
    checks_pair = {c["name"]: c["status"] for c in ladder_overlap_health_checks(lo_pair, LADDER_STATE_OVERLAP_MIN)}

    # The full-matrix worst value is diluted well below the pairwise one.
    assert lo_pair["lambda_direction"]["worst"] > lo_full["lambda_direction"]["worst"]
    assert checks_full["Overlap along λ"] == "fail"
    assert checks_pair["Overlap along λ"] in ("pass", "caution")
