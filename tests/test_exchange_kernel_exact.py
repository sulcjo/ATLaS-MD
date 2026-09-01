"""Exact proof that the shipped exchange kernel preserves the Boltzmann target.

The REUS move swaps *state labels*, never configurations, so against a frozen
bias matrix the exchange chain is a finite Markov chain on the N! permutations.
That means it does not have to be sampled and histogrammed -- the transition
matrix can be built exactly, by driving the shipped kernel, and checked to
machine precision. No sampling error, no autocorrelation, infinite power.

pi-invariance is preserved under composition and under state-independent
mixtures, and every mode's *selection* is state-independent (fixed parity lists,
or an rng.shuffle over fixed labels). So it suffices to verify the elementary
kernels:

* the pair-swap kernel for each window pair -- covers `neighbor`,
  `random-pair` and `all-pair-sweep`, which differ only in which pairs they
  offer;
* the single-replica gibbs kernel -- covers `gibbs-walk`.

What this closes that `tests/test_gibbs_walk.py` does not: that file checks the
detailed-balance *identity* on hand-built arrays, so it never executes the
bookkeeping update

    assignments[i], assignments[j] = assignments[j], assignments[i]
    replica_of_window[assignments[i]] = i
    replica_of_window[assignments[j]] = j

which `gibbs-walk` then reads back to build its next proposal. A desync between
those two arrays silently corrupts the proposal distribution. Here the rows are
built by actually applying the move, so the bookkeeping is on the path.

What this canNOT conclude: nothing about ergodicity (a kernel that never moves
is perfectly pi-invariant -- see the irreducibility test at the end), nothing
about whether the bias matrix handed to the kernel is itself right, and nothing
about sample labelling.

Mutation battery run against these tests (2026-09-01), each applied to an
isolated copy of the tree and reverted afterwards:

    CAUGHT  force_accept in place of the MH correction        4 failed
    CAUGHT  beta doubled in the Metropolis probability        5 failed
    CAUGHT  beta sign flipped                                 5 failed
    CAUGHT  replica_of_window update omitted                  4 failed
    CAUGHT  old_e / new_e swapped                             5 failed
    passes  only one window pair ever offered                 by design
    passes  old upper clip a_max=0.0 in the gibbs proposal    by design

The last two are controls, not misses. Invariance is a per-move property, so it
is blind to *which* moves are offered -- restricting the pair list cannot break
it. And Metropolis-Hastings is valid for ANY proposal distribution as long as
the correction uses the matching forward and reverse probabilities, which it
does (`q_reverse` comes from the same function evaluated against the post-swap
holders). Clipping the heat-bath weights therefore changes efficiency, not
correctness. The historical bug this clip came from (prior audit, N1) was the
clip *combined with* force-accept, and the force_accept mutation above is what
catches that.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from gareus.production import (
    _gibbs_mh_acceptance_probability,
    _gibbs_window_proposal_distribution,
    apply_window_swap,
)

BETA = 0.4009  # mol/kJ, ~300 K
TOL = 1e-12


def _bias(n: int, seed: int = 20260901) -> np.ndarray:
    """A deliberately asymmetric bias matrix: B[window, replica], kJ/mol.

    Asymmetric on purpose -- a symmetric matrix makes pi uniform, and a uniform
    target hides sign errors and missing MH corrections alike.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 6.0, size=(n, n))


def _states(n: int) -> list[tuple[int, ...]]:
    """Every assignment of replicas to windows: state[replica] = window."""
    return list(itertools.permutations(range(n)))


def _pi(states, bias, beta=BETA) -> np.ndarray:
    """pi(sigma) proportional to exp(-beta * sum_r B[sigma(r), r])."""
    e = np.array([sum(bias[w, r] for r, w in enumerate(s)) for s in states])
    p = np.exp(-beta * (e - e.min()))
    return p / p.sum()


def _arrays(state):
    """(assignments, replica_of_window) as production maintains them."""
    n = len(state)
    assignments = np.asarray(state, dtype=np.int64)
    replica_of_window = np.full(n, -1, dtype=np.int64)
    for r, w in enumerate(state):
        replica_of_window[w] = r
    return assignments, replica_of_window


def _pair_kernel(states, bias, wi, wj, beta=BETA) -> np.ndarray:
    """Transition matrix of the shipped pair-swap move for one window pair."""
    index = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for a, s in enumerate(states):
        # uniform=1.0 can never satisfy `uniform < pacc` (pacc <= 1), so this
        # reads pacc out of the real kernel without mutating anything.
        asg, row = _arrays(s)
        peek = apply_window_swap(bias, beta, asg, row, wi, wj, 1.0)
        assert peek is not None and not peek.accepted
        assert tuple(asg) == s, "a rejected swap must not mutate the assignment"
        pacc = peek.pacc

        # uniform=0.0 accepts whenever pacc > 0, so this drives the bookkeeping.
        asg2, row2 = _arrays(s)
        got = apply_window_swap(bias, beta, asg2, row2, wi, wj, 0.0)
        assert got is not None
        moved = tuple(int(x) for x in asg2)
        if got.accepted:
            # the two arrays must stay consistent -- gibbs reads them back
            for w, r in enumerate(row2):
                assert moved[int(r)] == w, "replica_of_window desynced from assignments"
        P[a, index[moved]] += pacc
        P[a, a] += 1.0 - pacc
    return P


def _gibbs_kernel(states, bias, rep, beta=BETA) -> np.ndarray:
    """Transition matrix of the shipped single-replica gibbs move.

    Mirrors the composition in run_gareus's `gibbs-walk` branch exactly: real
    heat-bath proposal, real reverse proposal against the post-swap holders,
    real MH correction, real swap application.
    """
    index = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for a, s in enumerate(states):
        asg, row = _arrays(s)
        wi = int(asg[rep])
        prop = _gibbs_window_proposal_distribution(
            beta=beta, bias_matrix_kj=bias, replica_index=rep,
            current_window=wi, replica_of_window=row,
        )
        wins, probs, deltas = prop["windows"], prop["probabilities"], prop["deltas_kj"]
        for c in range(int(wins.size)):
            wj = int(wins[c])
            q_fwd = float(probs[c])
            if q_fwd <= 0.0:
                continue
            if wj == wi:                      # the stay candidate: no MH test
                P[a, a] += q_fwd
                continue
            holders_after = row.astype(np.int64, copy=True)
            target_rep = int(holders_after[wj])
            holders_after[wi] = target_rep
            holders_after[wj] = rep
            rev = _gibbs_window_proposal_distribution(
                beta=beta, bias_matrix_kj=bias, replica_index=rep,
                current_window=wj, replica_of_window=holders_after,
            )
            ridx = np.where(rev["windows"] == wi)[0]
            q_rev = float(rev["probabilities"][int(ridx[0])]) if ridx.size else 0.0
            pacc = _gibbs_mh_acceptance_probability(
                delta_kj=float(deltas[c]), beta=beta, q_forward=q_fwd, q_reverse=q_rev,
            )
            asg2, row2 = _arrays(s)
            got = apply_window_swap(bias, beta, asg2, row2, wi, wj, 0.0, p_override=pacc)
            if got is None:
                P[a, a] += q_fwd
                continue
            moved = tuple(int(x) for x in asg2)
            P[a, index[moved]] += q_fwd * pacc
            P[a, a] += q_fwd * (1.0 - pacc)
    return P


def _assert_kernel_ok(P, pi, label):
    rows = P.sum(axis=1)
    assert np.allclose(rows, 1.0, atol=TOL), f"{label}: rows do not sum to 1 ({rows.min()}..{rows.max()})"
    resid = np.abs(pi @ P - pi).max()
    assert resid < TOL, f"{label}: pi is not stationary, max|piP-pi| = {resid:.3e}"
    # Detailed balance is strictly stronger than invariance AND localises the
    # offending pair, which invariance alone cannot.
    flux = pi[:, None] * P
    asym = np.abs(flux - flux.T)
    i, j = np.unravel_index(int(asym.argmax()), asym.shape)
    assert asym.max() < TOL, (
        f"{label}: detailed balance violated at (sigma={i}, sigma'={j}): "
        f"pi_i P_ij = {flux[i, j]:.6e} vs pi_j P_ji = {flux[j, i]:.6e}"
    )


@pytest.mark.parametrize("n", [3, 4, 5])
def test_pair_swap_kernel_satisfies_detailed_balance(n):
    """Covers neighbor / random-pair / all-pair-sweep: they differ only in pairing."""
    states, bias = _states(n), _bias(n)
    pi = _pi(states, bias)
    for wi, wj in itertools.combinations(range(n), 2):
        _assert_kernel_ok(_pair_kernel(states, bias, wi, wj), pi, f"pair({wi},{wj}) n={n}")


@pytest.mark.parametrize("n", [3, 4, 5])
def test_gibbs_kernel_satisfies_detailed_balance(n):
    states, bias = _states(n), _bias(n)
    pi = _pi(states, bias)
    for rep in range(n):
        _assert_kernel_ok(_gibbs_kernel(states, bias, rep), pi, f"gibbs(rep={rep}) n={n}")


def test_a_full_sweep_of_either_kernel_is_also_invariant():
    """Composition: what production actually runs is a sequence of these."""
    n = 4
    states, bias = _states(n), _bias(n)
    pi = _pi(states, bias)

    # Only stationarity here, NOT detailed balance: a product of DB kernels is
    # pi-invariant but is not itself pi-reversible, so asserting DB on the sweep
    # would be wrong rather than strict.
    sweep = np.eye(len(states))
    for wi, wj in itertools.combinations(range(n), 2):
        sweep = sweep @ _pair_kernel(states, bias, wi, wj)
    assert np.allclose(sweep.sum(axis=1), 1.0, atol=TOL)
    assert np.abs(pi @ sweep - pi).max() < TOL, "all-pair sweep is not pi-invariant"

    gsweep = np.eye(len(states))
    for rep in range(n):
        gsweep = gsweep @ _gibbs_kernel(states, bias, rep)
    # Composition preserves invariance but NOT pairwise detailed balance, so
    # check stationarity only -- asserting DB here would be wrong, not strict.
    assert np.allclose(gsweep.sum(axis=1), 1.0, atol=TOL)
    assert np.abs(pi @ gsweep - pi).max() < TOL


def test_the_composed_sweep_is_irreducible():
    """pi-invariance is free for a kernel that never moves; ergodicity is not.

    Guards the failure mode the balance tests structurally cannot see.
    """
    n = 4
    states, bias = _states(n), _bias(n)
    sweep = np.eye(len(states))
    for rep in range(n):
        sweep = sweep @ _gibbs_kernel(states, bias, rep)

    reach = (sweep > 1e-14).astype(np.int64)
    closure = reach.copy()
    for _ in range(len(states)):
        closure = ((closure @ reach) > 0).astype(np.int64)
    assert closure.all(), "chain is reducible: some permutation is unreachable"

    ev = np.sort(np.abs(np.linalg.eigvals(sweep)))[::-1]
    assert ev[0] == pytest.approx(1.0, abs=1e-9)
    assert ev[1] < 1.0 - 1e-6, f"subdominant eigenvalue {ev[1]:.6f} -- chain barely mixes"


def test_the_reported_acceptance_matches_metropolis_by_hand():
    """Anchors pacc to the closed form, so the kernel cannot drift silently."""
    n = 3
    bias = _bias(n)
    asg, row = _arrays((0, 1, 2))
    out = apply_window_swap(bias, BETA, asg, row, 0, 1, 1.0)
    delta = (bias[1, 0] + bias[0, 1]) - (bias[0, 0] + bias[1, 1])
    assert out.delta_kj == pytest.approx(delta, rel=1e-12)
    assert out.pacc == pytest.approx(min(1.0, math.exp(-BETA * delta)), rel=1e-12)


def test_a_swap_that_cannot_be_attempted_returns_none():
    """Same window, out-of-range, or an unheld window: caller must not count it."""
    n = 3
    bias = _bias(n)
    asg, row = _arrays((0, 1, 2))
    assert apply_window_swap(bias, BETA, asg, row, 1, 1, 0.0) is None
    assert apply_window_swap(bias, BETA, asg, row, 0, 99, 0.0) is None
    row[2] = -1                                    # window 2 held by nobody
    assert apply_window_swap(bias, BETA, asg, row, 0, 2, 0.0) is None


def test_force_accept_does_not_consume_a_random_number():
    """Production short-circuits rng.random() away; drawing one shifts every
    later stream. The kernel must accept without being handed a uniform."""
    n = 3
    bias = _bias(n)
    asg, row = _arrays((0, 1, 2))
    out = apply_window_swap(bias, BETA, asg, row, 0, 1, None, force_accept=True)
    assert out is not None and out.accepted
    assert tuple(asg) == (1, 0, 2)
