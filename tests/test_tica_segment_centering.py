"""Whether a tICA fit on umbrella data finds dynamics or finds the restraint.

``_tica_covariance_matrices`` already refuses to build lagged pairs across
segment boundaries, so C(tau) is honest about which frames are consecutive. The
mean it removes, however, is global:

    mean = X.mean(axis=0)

On unbiased trajectories that is correct. On umbrella-sampled data it is not.
Each segment is a replica held near its own window centre, so the spread BETWEEN
segments is the restraint, not motion: a coordinate that is constant inside every
segment and merely differs from window to window has a lagged autocorrelation of
essentially 1 at any lag. tICA maximises exactly that quantity, so it selects the
restraint direction and reports it as the slowest mode.

Measured on chignolin_6 (37 umbrella windows, 132k frames): a tICA refit under
global centring produced a timescale ratio of 22 whose implied timescale grew
linearly with lag (12.5 ns at lag 0.08 ns to 231 ns at lag 2.4 ns) while lambda
sat at ~0.990 for every lag -- the signature of a correlation that never decays
because the coordinate never moves. Removing each segment's own mean collapsed
that to 2.71 and made the timescale lag-stable.

Global centring stays the default: for a single long unbiased trajectory,
per-segment centring would remove real slow variation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gareus.tica import (  # noqa: E402
    _tica_covariance_matrices,
    compute_tica,
    project_tica1,
)

N_SEG = 40
SEG_LEN = 500
LAG = 20
OU_TAU = 50.0

RESTRAINT, SLOW, FAST = 0, 1, 2


def _umbrella_like(seed: int = 0):
    """Three features with known roles, laid out like umbrella-sampled data.

    RESTRAINT: constant within a segment, spread widely across segments. This is
        the window centre. It carries no dynamics at all.
    SLOW:      an Ornstein-Uhlenbeck process that relaxes WITHIN each segment.
        This is the real slow mode a CV should find.
    FAST:      white noise.
    """
    rng = np.random.default_rng(seed)
    a = np.exp(-1.0 / OU_TAU)
    offsets = np.linspace(-3.0, 3.0, N_SEG)
    blocks = []
    for s in range(N_SEG):
        x = np.empty(SEG_LEN)
        x[0] = rng.normal()
        for t in range(1, SEG_LEN):
            x[t] = a * x[t - 1] + np.sqrt(1.0 - a * a) * rng.normal()
        blocks.append(np.column_stack([
            offsets[s] + 0.05 * rng.normal(size=SEG_LEN),
            x,
            rng.normal(size=SEG_LEN),
        ]))
    X = np.vstack(blocks)
    return X, np.full(N_SEG, SEG_LEN)


def _dominant(res):
    return int(np.argmax(np.abs(np.asarray(res.weights, dtype=float))))


# --- the defect ------------------------------------------------------------

def test_global_centering_selects_the_restraint_not_the_dynamics():
    """The motivating failure, reproduced: with a global mean the widest thing
    in the data is the spread between window centres, and it never moves, so it
    wins the autocorrelation contest outright."""
    X, seg = _umbrella_like()
    res = compute_tica(X, LAG, segments=seg)
    assert _dominant(res) == RESTRAINT


def test_per_segment_centering_recovers_the_real_slow_mode():
    X, seg = _umbrella_like()
    res = compute_tica(X, LAG, segments=seg, center="per_segment")
    assert _dominant(res) == SLOW, (
        'per-segment centring must select the OU coordinate, not the window offset'
    )


def test_per_segment_centering_does_not_pick_the_fast_coordinate():
    X, seg = _umbrella_like()
    res = compute_tica(X, LAG, segments=seg, center="per_segment")
    assert _dominant(res) != FAST


# --- the implied timescale is what actually gets misread --------------------

def _its(res, lag):
    lam = float(np.clip(abs(res.eigenvalue), 1e-12, 1 - 1e-12))
    return -lag / np.log(lam)


def test_global_centering_gives_a_lag_dependent_timescale():
    """A converged implied timescale is roughly lag-independent. The restraint
    artifact is not: lambda stays pinned near 1, so -lag/ln(lambda) simply
    tracks the lag. This is the diagnostic that exposed the real run."""
    X, seg = _umbrella_like()
    t = [_its(compute_tica(X, L, segments=seg), L) for L in (10, 20, 40)]
    assert t[2] / t[0] > 2.0, f'expected the artifact to scale with lag, got {t}'


def test_per_segment_centering_gives_a_lag_stable_timescale():
    X, seg = _umbrella_like()
    t = [_its(compute_tica(X, L, segments=seg, center="per_segment"), L)
         for L in (10, 20, 40)]
    assert max(t) / min(t) < 1.6, f'timescale should be roughly lag-stable, got {t}'
    # and it should land near the OU time it was built from
    assert 0.5 * OU_TAU < np.median(t) < 2.0 * OU_TAU, t


# --- contracts that must not move ------------------------------------------

def test_global_is_the_default():
    """Unbiased single trajectories must keep today's behaviour."""
    X, seg = _umbrella_like()
    a = compute_tica(X, LAG, segments=seg)
    b = compute_tica(X, LAG, segments=seg, center="global")
    np.testing.assert_allclose(a.weights, b.weights)


def test_one_segment_makes_the_two_modes_agree():
    """With a single segment there is no between-segment offset to remove, so
    per-segment centring degenerates to global centring."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 3))
    seg = np.array([2000])
    a, _ct_a, ma, _d = _tica_covariance_matrices(X, LAG, segments=seg)
    b, _ct_b, mb, _d2 = _tica_covariance_matrices(X, LAG, segments=seg,
                                                  center="per_segment")
    np.testing.assert_allclose(a, b, atol=1e-10)
    np.testing.assert_allclose(ma, mb, atol=1e-10)


def test_centering_is_validated():
    X, seg = _umbrella_like()
    with pytest.raises(ValueError, match="center"):
        _tica_covariance_matrices(X, LAG, segments=seg, center="nonsense")


def test_projection_still_round_trips():
    """center changes the fit, not the projection contract."""
    X, seg = _umbrella_like()
    res = compute_tica(X, LAG, segments=seg, center="per_segment")
    cv = project_tica1(X, res)
    np.testing.assert_allclose(cv, X @ res.weights + res.offset, rtol=1e-10)


def test_per_segment_centering_works_with_weights():
    """The reweighted estimator must accept the same option."""
    X, seg = _umbrella_like()
    w = np.full(len(X), 1.0 / len(X))
    res = compute_tica(X, LAG, segments=seg, weights=w, center="per_segment")
    assert _dominant(res) == SLOW
