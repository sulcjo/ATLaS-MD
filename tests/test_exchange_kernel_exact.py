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

Mutation battery. Each mutation applied to the shipped file, tests run, file
restored. Counts are over the 20 tests in this file plus
`tests/test_sample_before_exchange_ordering.py`.

HARMFUL mutations -- all caught:

    force_accept in place of the pair-swap MH test              5 failed
    beta doubled in the Metropolis probability                  5 failed
    old_e / new_e swapped                                       6 failed
    replica_of_window update omitted                            8 failed
    gibbs reverse proposal against PRE-swap holders             4 failed
    gibbs q_forward / q_reverse swapped                         4 failed
    gibbs call site force_accept instead of p_override          1 failed
    holders_after missing `holders_after[wi] = target_rep`      4 failed
    q_forward/q_reverse swapped in the returned GibbsProposal   5 failed

HARMLESS mutations -- these SHOULD pass, and do. Each was checked to be
semantically inert rather than assumed to be:

    only one window pair ever offered
        Invariance is a per-move property, blind to which moves are offered.
        Restricting the pair list breaks ergodicity, which the irreducibility
        test guards, not balance.
    the two replica_of_window writes reordered
        They target `assignments[i]` and `assignments[j]`, which differ whenever
        the swap is attempted at all, so the writes are independent.
    the `pacc = max(0, min(1, pacc))` clamp removed
        Provably unreachable: both producers of pacc are bounded to [0,1] and
        finite on every path -- `_exchange_probability` returns 0.0 for
        non-finite input, 1.0 for delta <= 0, else exp(x) with x < 0; and
        `_gibbs_mh_acceptance_probability` likewise. The clamp is defensive
        redundancy, and no production caller supplies p_override from elsewhere.
    gibbs iterating `order[:-1]` instead of `order`
        `order` is reshuffled every sweep, so this drops a uniformly random
        replica. Every remaining move is still a valid MH move (balance holds),
        and each replica is still attempted with probability (n-1)/n per sweep
        (ergodicity holds). An efficiency change, not a correctness one.

One mutation was caught that SHOULD NOT have been, and the test was loosened
rather than left over-sensitive:

    `uniform < pacc` -> `uniform <= pacc`     was 4 failed, now passes
        Measure-zero for a real draw: rng.random() returns [0,1) and so never
        equals a pacc of 1.0. The old probe passed exactly 1.0 to read pacc
        without accepting, which made it sensitive to the boundary convention.
        It now passes 1.5, which can never be accepted under either comparison.
        Reporting a physically irrelevant edit as a correctness break would
        overstate what this battery proves.

Historical note: the gibbs mutations above previously PASSED, because this file
re-implemented the proposal sequence inline -- self-consistent and
production-blind. The call-site force_accept mutation still passes against this
file alone (the numeric kernel cannot see a substitution made between the two
functions it drives) and is caught by an AST pin in the ordering test file.

"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import gareus.production as production
from gareus.production import (
    apply_window_swap,
    gibbs_propose_one_replica,
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
        # A uniform strictly ABOVE 1 can never be accepted, whether the kernel
        # tests `uniform < pacc` or `uniform <= pacc`, so this reads pacc out of
        # the real kernel without mutating anything.
        #
        # It used to probe with exactly 1.0, which made the test fail if `<` were
        # changed to `<=` -- a difference that is measure-zero for a real
        # rng.random() draw (which returns [0,1) and so never equals a pacc of
        # 1.0 anyway). That was an over-constraint: the test would have reported
        # a physically irrelevant edit as a correctness break.
        asg, row = _arrays(s)
        peek = apply_window_swap(bias, beta, asg, row, wi, wj, 1.5)
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

    Every number here comes out of `gibbs_propose_one_replica` and
    `apply_window_swap` -- the same two functions run_gareus's `gibbs-walk`
    branch calls. An earlier version of this helper re-implemented the
    proposal / reverse-proposal / MH-correction sequence inline, which made the
    test self-consistent and production-blind: three separate defects injected
    into the production branch passed all 15 assertions, because the test never
    executed the mutated code. The `choose` callback is the ONLY thing supplied
    locally, and only because it stands in for the RNG.
    """
    index = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for a, s in enumerate(states):
        # Enumerate every candidate the real proposal offers, weighting each
        # branch by the q_forward the real proposal assigned it.
        asg0, row0 = _arrays(s)
        n_cand_seen = {}

        def _count(k, probs, _store=n_cand_seen):
            _store["k"] = int(k)
            _store["p"] = np.asarray(probs, dtype=float).copy()
            return 0

        gibbs_propose_one_replica(bias, beta, asg0, row0, rep, _count)
        assert tuple(asg0) == s, "proposing must not mutate the assignment"
        if not n_cand_seen:
            P[a, a] += 1.0
            continue

        for c in range(n_cand_seen["k"]):
            asg, row = _arrays(s)
            prop = gibbs_propose_one_replica(
                bias, beta, asg, row, rep, lambda k, p, _c=c: _c
            )
            q_fwd = float(prop.q_forward)
            if q_fwd <= 0.0:
                continue
            if prop.stayed or prop.no_candidates:
                P[a, a] += q_fwd
                continue
            got = apply_window_swap(
                bias, beta, asg, row, prop.current_window, prop.proposed_window,
                0.0, p_override=prop.pacc,
            )
            if got is None:
                P[a, a] += q_fwd
                continue
            moved = tuple(int(x) for x in asg)
            if got.accepted:
                for w, r in enumerate(row):
                    assert moved[int(r)] == w, "replica_of_window desynced from assignments"
            # got.pacc, not prop.pacc: read the acceptance back out of the
            # applying function, so a caller that ignored p_override is visible.
            P[a, index[moved]] += q_fwd * got.pacc
            P[a, a] += q_fwd * (1.0 - got.pacc)
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
    out = apply_window_swap(bias, BETA, asg, row, 0, 1, 1.5)   # >1: never accepted
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


class _CountingChoice:
    """Records how many times the gibbs proposal asked for a random draw."""

    def __init__(self, index: int = 0):
        self.calls = 0
        self.index = index

    def __call__(self, k, probs):
        self.calls += 1
        return min(int(self.index), int(k) - 1)


def _proposal_with_counted_draws(bias, state, rep, index=0):
    asg, row = _arrays(state)
    ch = _CountingChoice(index)
    prop = gibbs_propose_one_replica(bias, BETA, asg, row, rep, ch)
    return prop, ch.calls


def test_the_gibbs_proposal_consumes_exactly_one_draw_and_only_when_it_can_choose():
    """Draw accounting, pinned per branch.

    Extracting a decision out of a closure is exactly where an RNG stream gets
    shifted: one extra or one missing draw desynchronises every later decision
    in the run, and NOTHING else in this file would notice -- the transition
    matrices are built from scripted choices, and the mutation battery and the
    full suite both pass regardless of draw count.

    This is not hypothetical. The sibling extraction in this same refactor
    (`apply_window_swap`) shipped a first version that drew a uniform on the
    un-attemptable path where the original closure had returned early. It was
    caught by counting, not by reading.

    Production called `rng.choice` once, after the empty-candidate guard. So:
    0 draws when there are no candidates, exactly 1 otherwise -- whether the
    proposal lands on the current window (a "stay") or on a different one.
    """
    n = 4
    bias = _bias(n)
    state = (0, 1, 2, 3)

    # a real proposal: exactly one draw, whichever candidate is selected
    for index in range(n):
        prop, calls = _proposal_with_counted_draws(bias, state, rep=1, index=index)
        assert calls == 1, (
            f"candidate {index} consumed {calls} draws, not 1 -- the RNG stream "
            "has shifted relative to the closure this was extracted from"
        )
        assert not prop.no_candidates

    # the stay branch must not consume a second draw on its early return
    stays = [
        _proposal_with_counted_draws(bias, state, rep=r, index=i)
        for r in range(n) for i in range(n)
    ]
    assert any(p.stayed for p, _ in stays), "no candidate set offered the current window"
    for p, calls in stays:
        assert calls == 1, f"stayed={p.stayed} consumed {calls} draws, not 1"


def test_the_gibbs_proposal_consumes_no_draw_when_there_are_no_candidates():
    """The no-candidate short-circuit must return before asking for randomness.

    Production's `if valid_windows.size <= 0 ... continue` fired before
    `rng.choice`, so this path must cost zero draws.

    Reaching it took some finding, and the finding is worth recording: **an
    all-NaN bias matrix does NOT empty the candidate list.** The proposal forces
    the stay candidate's delta to exactly 0.0 (`delta[stay_idx] = 0.0`) before
    the finite-mask is applied, so as long as the replica holds a window there is
    always at least one finite candidate. The list empties only when NO window
    has a holder at all.

    That matters beyond this test: the `gibbs_all_nan_skips` counter cannot be
    incremented by NaN bias values, despite its name. A non-zero value there
    would mean an empty holder table, not a NaN CV.
    """
    n = 3
    bias = _bias(n)
    asg = np.array([0, 1, 2], dtype=np.int64)
    row = np.full(n, -1, dtype=np.int64)      # no window has a holder
    ch = _CountingChoice()
    prop = gibbs_propose_one_replica(bias, BETA, asg, row, 0, ch)
    assert prop.no_candidates, "an unheld window table should offer no candidates"
    assert ch.calls == 0, (
        f"consumed {ch.calls} draws on the no-candidate path; production returned "
        "before drawing, so every later decision in the run would shift"
    )


def test_an_all_nan_bias_row_still_offers_the_stay_candidate():
    """Pins the surprise above, because a future change could quietly alter it.

    If the stay candidate ever stopped being force-zeroed, an all-NaN row would
    start emptying the candidate list, `gibbs-walk` would begin skipping
    replicas, and the only visible symptom would be a counter whose name already
    suggests the wrong cause.
    """
    n = 3
    prop, calls = _proposal_with_counted_draws(np.full((n, n), np.nan), (0, 1, 2), rep=0)
    assert not prop.no_candidates
    assert prop.stayed, "the only finite candidate should be the current window"
    assert calls == 1


def test_apply_window_swap_draws_nothing_on_a_swap_it_cannot_attempt():
    """The sibling property, for the extraction that actually got this wrong.

    `swap_candidate_replicas` is checked BEFORE the uniform is drawn, so the
    caller's `None if force_accept else rng.random()` is never evaluated for an
    un-attemptable pair -- matching the original closure, which returned early.
    """
    n = 3
    bias = _bias(n)
    asg, row = _arrays((0, 1, 2))

    drawn = []

    def _rng_random():
        drawn.append(1)
        return 0.5

    # same-window, out-of-range, and unheld-window: all must short-circuit
    for wi, wj in ((1, 1), (0, 99)):
        if production.swap_candidate_replicas(row, wi, wj) is None:
            continue
        apply_window_swap(bias, BETA, asg, row, wi, wj, _rng_random())
    assert not drawn, "a uniform was consumed for a swap that cannot be attempted"
