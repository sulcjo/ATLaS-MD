from __future__ import annotations

import math
from itertools import permutations

import numpy as np


def test_gibbs_mh_acceptance_can_reject_heatbath_choice() -> None:
    from gareus.production import _gibbs_mh_acceptance_probability

    alpha = _gibbs_mh_acceptance_probability(
        delta_kj=4.0,
        beta=1.0,
        q_forward=0.80,
        q_reverse=0.10,
    )

    assert 0.0 < alpha < 1.0


def test_gibbs_window_proposal_distribution_normalizes_stay_candidate() -> None:
    from gareus.production import _gibbs_window_proposal_distribution

    bias_matrix_kj = np.asarray(
        [
            [0.0, 5.0, 7.0],
            [1.0, 0.0, 4.0],
            [6.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    holders = np.asarray([0, 1, 2], dtype=np.int64)

    proposal = _gibbs_window_proposal_distribution(
        beta=0.5,
        bias_matrix_kj=bias_matrix_kj,
        replica_index=0,
        current_window=0,
        replica_of_window=holders,
    )

    assert proposal["windows"].tolist() == [0, 1, 2]
    assert math.isclose(float(proposal["probabilities"].sum()), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    stay_idx = int(np.where(proposal["windows"] == 0)[0][0])
    assert float(proposal["deltas_kj"][stay_idx]) == 0.0
    assert float(proposal["probabilities"][stay_idx]) > 0.0


def test_gibbs_proposal_weights_favourable_moves_above_stay() -> None:
    """After removing the upper clip, a favourable move (delta<0) must get strictly
    higher proposal weight than the stay candidate (delta=0)."""
    from gareus.production import _gibbs_window_proposal_distribution

    # B[w, r]: replica 0 in window 0.  Moving to window 1 is favourable (delta < 0).
    # delta(w1) = B[1,0] + B[0,1] - B[0,0] - B[1,1]
    #           = 0.0    + 8.0  -  5.0   -  5.0   = -2.0   (negative → favourable)
    bias = np.asarray(
        [[5.0, 8.0], [0.0, 5.0]],
        dtype=float,
    )
    holders = np.asarray([0, 1], dtype=np.int64)
    proposal = _gibbs_window_proposal_distribution(
        beta=1.0,
        bias_matrix_kj=bias,
        replica_index=0,
        current_window=0,
        replica_of_window=holders,
    )
    assert math.isclose(float(proposal["probabilities"].sum()), 1.0, abs_tol=1e-12)
    stay_idx = int(np.where(proposal["windows"] == 0)[0][0])
    move_idx = int(np.where(proposal["windows"] == 1)[0][0])
    delta_move = float(proposal["deltas_kj"][move_idx])
    assert delta_move < 0.0, f"expected delta < 0 (favourable), got {delta_move}"
    p_stay = float(proposal["probabilities"][stay_idx])
    p_move = float(proposal["probabilities"][move_idx])
    assert p_move > p_stay, (
        f"favourable move (delta={delta_move:.3f}) should have higher proposal prob "
        f"than stay: p_move={p_move:.6f} p_stay={p_stay:.6f}"
    )


def test_gibbs_walk_detailed_balance_3windows() -> None:
    """Numerical detailed-balance check for 3-window, 3-replica system.

    Constructs the exact Boltzmann target π(σ) over all 6 permutations and
    verifies that for every pair (σ, σ') differing by a single swap and every
    replica r:

        π(σ) · q_forward · α_forward  ==  π(σ') · q_reverse · α_reverse

    This holds for *any* valid proposal distribution (clipped or not) when the
    MH correction uses the same q for both directions — confirming the scheme is
    thermodynamically correct.  The test also guards against regressions in the
    proposal or acceptance functions.
    """
    from gareus.production import (
        _gibbs_mh_acceptance_probability,
        _gibbs_window_proposal_distribution,
    )

    rng = np.random.default_rng(0)
    n = 3
    beta = 0.5

    # Fixed asymmetric bias matrix  B[window, replica] in kJ/mol
    bias = rng.uniform(-6.0, 6.0, size=(n, n))

    # Enumerate all n! permutation states.
    # Convention: perm[w] = replica currently in window w.
    all_perms = list(permutations(range(n)))

    # Compute unnormalised Boltzmann weight: exp(-β · Σ_w B[w, perm[w]])
    def _log_pi(perm: tuple) -> float:
        return -beta * sum(float(bias[w, perm[w]]) for w in range(n))

    log_pis = {p: _log_pi(p) for p in all_perms}
    log_max = max(log_pis.values())
    raw = {p: math.exp(lp - log_max) for p, lp in log_pis.items()}
    z = sum(raw.values())
    pi = {p: v / z for p, v in raw.items()}

    def _proposal_and_acceptance(sigma: tuple, r: int, wj: int):
        """Return (q_forward, q_reverse, alpha) for replica r moving wi→wj."""
        replica_of_window = np.array(sigma, dtype=np.int64)
        # assignments[replica] = window
        assignments = np.zeros(n, dtype=np.int32)
        for w, rep in enumerate(sigma):
            assignments[rep] = w
        wi = int(assignments[r])
        assert wi != wj

        fwd = _gibbs_window_proposal_distribution(
            beta=beta,
            bias_matrix_kj=bias,
            replica_index=r,
            current_window=wi,
            replica_of_window=replica_of_window,
        )
        fwd_idx = np.where(fwd["windows"] == wj)[0]
        assert fwd_idx.size, f"window {wj} missing from proposal for σ={sigma} r={r}"
        q_fwd = float(fwd["probabilities"][int(fwd_idx[0])])
        delta = float(fwd["deltas_kj"][int(fwd_idx[0])])

        # Hypothetical post-swap holders
        holders_after = replica_of_window.copy()
        target_rep = int(holders_after[wj])
        holders_after[wi] = target_rep
        holders_after[wj] = r
        sigma_prime = tuple(int(x) for x in holders_after)

        rev = _gibbs_window_proposal_distribution(
            beta=beta,
            bias_matrix_kj=bias,
            replica_index=r,
            current_window=wj,
            replica_of_window=holders_after,
        )
        rev_idx = np.where(rev["windows"] == wi)[0]
        assert rev_idx.size, f"window {wi} missing from reverse proposal"
        q_rev = float(rev["probabilities"][int(rev_idx[0])])

        alpha_fwd = _gibbs_mh_acceptance_probability(
            delta_kj=delta, beta=beta, q_forward=q_fwd, q_reverse=q_rev
        )
        alpha_rev = _gibbs_mh_acceptance_probability(
            delta_kj=-delta, beta=beta, q_forward=q_rev, q_reverse=q_fwd
        )
        return q_fwd, q_rev, alpha_fwd, alpha_rev, sigma_prime

    # Check DB for every (σ, replica r, target wj≠wi)
    violations = []
    for sigma in all_perms:
        replica_of_window = np.array(sigma, dtype=np.int64)
        assignments = np.zeros(n, dtype=np.int32)
        for w, rep in enumerate(sigma):
            assignments[rep] = w
        for r in range(n):
            wi = int(assignments[r])
            for wj in range(n):
                if wj == wi:
                    continue
                q_fwd, q_rev, alpha_fwd, alpha_rev, sigma_prime = _proposal_and_acceptance(sigma, r, wj)
                lhs = pi[sigma] * q_fwd * alpha_fwd
                rhs = pi[sigma_prime] * q_rev * alpha_rev
                if not math.isclose(lhs, rhs, rel_tol=1e-9, abs_tol=1e-15):
                    violations.append(
                        f"σ={sigma} r={r} wi={wi}→wj={wj}: "
                        f"lhs={lhs:.6e} rhs={rhs:.6e} ratio={lhs/max(rhs,1e-30):.6f}"
                    )

    assert not violations, (
        f"Detailed balance violated for {len(violations)} (σ,r,wj) triples:\n"
        + "\n".join(violations[:10])
    )
