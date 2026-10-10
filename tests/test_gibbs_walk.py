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


# ── F09: log-space Gibbs proposal + MH for auxiliary runs (gibbs_softmax_log_v2) ──────────────────

import hashlib

import pytest

F09_LEGACY_TRACE_SHA256 = "2e8fda76b0824074b2d644f818fbfbd8887ec23226db17e3f86c84a852b7bebd"


def _f09_bias() -> np.ndarray:
    """u[s, r] = 50 (c_s - z_r)^2 with beta = 1 (repair finding F09's counterexample)."""
    c = np.array([math.sqrt(10.0), 0.0, 0.1])
    z = np.array([0.0, math.sqrt(10.0), -0.1])
    return 50.0 * (c[:, None] - z[None, :]) ** 2


def _legacy_fixed_seed_trace(propose):
    """Fixed-seed legacy gibbs-walk loop in run_gareus' draw order: shuffle, rng.choice, rng.random.

    Covers the ±745 clip (one entry offset by 900 so delta, not a cancelling row/column, is huge), the
    NaN-masked candidate regime, stays, accepts and rejects. The digest pins proposals, acceptances,
    final assignment and the RNG state after the loop.
    """
    from gareus.production import apply_window_swap, swap_candidate_replicas
    rng = np.random.default_rng(20261010)
    n = 5
    out = []
    counts = {"clip": 0, "nan": 0, "accept": 0, "reject": 0, "stay": 0}
    asg = np.arange(n, dtype=np.int64)
    row = np.arange(n, dtype=np.int64)
    for sweep in range(60):
        base = rng.normal(0.0, 3.0, size=(n, n))
        if sweep % 3 == 1:
            base[int(rng.integers(n)), int(rng.integers(n))] += 900.0
        if sweep % 11 == 5:
            base[2, 3] = np.nan
        order = list(range(n))
        rng.shuffle(order)
        for rep in order:
            rep = int(rep)
            wi = int(asg[rep])
            d = base[:, rep] + base[wi, row] - base[wi, rep] - base[np.arange(n), row]
            d[wi] = 0.0
            counts["clip"] += int(np.sum(np.abs(d[np.isfinite(d)]) > 745.0))
            counts["nan"] += int(np.sum(~np.isfinite(d)))
            p = propose(base, 1.0, asg, row, rep, lambda k, pr: int(rng.choice(k, p=pr)))
            rec = [p.current_window, p.proposed_window, float(p.delta_kj).hex(), float(p.q_forward).hex(),
                   float(p.q_reverse).hex(), float(p.pacc).hex(), p.stayed, p.no_candidates]
            counts["stay"] += int(p.stayed)
            if not (p.no_candidates or p.stayed) and swap_candidate_replicas(row, p.current_window, p.proposed_window):
                o = apply_window_swap(base, 1.0, asg, row, p.current_window, p.proposed_window, rng.random(),
                                      p_override=p.pacc)
                rec += [o.accepted, float(o.pacc).hex()]
                counts["accept" if o.accepted else "reject"] += 1
            out.append(rec)
    out.append(repr(rng.bit_generator.state))
    out.append(asg.tolist())
    return hashlib.sha256(repr(out).encode()).hexdigest(), counts


def test_legacy_gibbs_walk_is_byte_identical_on_a_fixed_seed() -> None:
    """Aux off: the selected proposer is the unchanged legacy function and its fixed-seed trace (proposals,
    q's, pacc, accept decisions, RNG state) equals the digest recorded on the unmodified code at 1cd6eab."""
    from gareus.production import gibbs_propose_one_replica, select_gibbs_proposer
    propose = select_gibbs_proposer(aux_active=False)
    assert propose is gibbs_propose_one_replica
    digest, counts = _legacy_fixed_seed_trace(propose)
    assert counts["clip"] > 0 and counts["nan"] > 0 and counts["stay"] > 0
    assert counts["accept"] > 0 and counts["reject"] > 0
    assert digest == F09_LEGACY_TRACE_SHA256


def test_f09_counterexample_accepts_with_one_over_one_plus_e_on_the_aux_path() -> None:
    from gareus.production import (GIBBS_PROPOSAL_ALGORITHM_V2, apply_window_swap, gibbs_propose_one_replica,
                                   select_gibbs_proposer)
    bias = _f09_bias()
    expected = 1.0 / (1.0 + math.e)
    propose = select_gibbs_proposer(aux_active=True)
    assert propose is not gibbs_propose_one_replica
    asg, row = np.arange(3), np.arange(3)
    prop = propose(bias, 1.0, asg, row, 0, lambda k, p: 1)        # carrier 0 proposes state 1
    assert prop.proposal_algorithm == GIBBS_PROPOSAL_ALGORITHM_V2
    assert (prop.current_window, prop.proposed_window, prop.stayed) == (0, 1, False)
    assert math.isfinite(prop.log_q_reverse) and prop.log_q_reverse < -745.0   # exp underflows; log does not
    assert prop.log_q_reverse == pytest.approx(-1000.0 - math.log1p(math.e), rel=1e-12)
    assert math.exp(prop.log_p_accept) == pytest.approx(expected, rel=1e-9)
    assert prop.pacc == pytest.approx(expected, rel=1e-9)
    # The legacy algorithm rejects this move outright: its reverse q underflowed to zero.
    assert gibbs_propose_one_replica(bias, 1.0, asg, row, 0, lambda k, p: 1).pacc == 0.0
    # One uniform decides, in log space, through the production swap kernel.
    for u, want in ((expected * (1 - 1e-9), True), (expected * (1 + 1e-9), False), (0.0, True)):
        a, r = np.arange(3), np.arange(3)
        out = apply_window_swap(bias, 1.0, a, r, 0, 1, u, p_override=prop.pacc, log_p_accept=prop.log_p_accept)
        assert out.accepted is want and out.pacc == pytest.approx(expected, rel=1e-9)
        assert a.tolist() == ([1, 0, 2] if want else [0, 1, 2])


def test_v2_refuses_nonfinite_reduced_energies_with_provenance() -> None:
    from gareus.production import select_gibbs_proposer
    bias = _f09_bias()
    bias[2, 1] = np.nan
    with pytest.raises(ValueError, match=r"window 2.*replica 1"):
        select_gibbs_proposer(aux_active=True)(bias, 1.0, np.arange(3), np.arange(3), 0, lambda k, p: 0)


def _log_db_violations(propose, bias, beta=1.0):
    """Every (sigma, r, wj) pair: log pi(s) + log q(s->s') + log alpha == the reverse flux, exactly in log space."""
    n = bias.shape[0]

    def arrays(sigma):
        row = np.array(sigma, dtype=np.int64)
        asg = np.empty(n, dtype=np.int64)
        asg[row] = np.arange(n)
        return asg, row

    def log_pi(sigma):
        return -beta * sum(float(bias[w, sigma[w]]) for w in range(n))

    def log_flux(sigma, r, wj):
        asg, row = arrays(sigma)
        prop = propose(bias, beta, asg, row, r, lambda k, p: wj)
        assert prop.proposed_window == wj
        lq = getattr(prop, "log_q_forward", float("nan"))
        la = getattr(prop, "log_p_accept", float("nan"))
        if not math.isfinite(lq):        # legacy: no log fields, take logs of its rounded probabilities
            lq = math.log(prop.q_forward) if prop.q_forward > 0 else -math.inf
            la = math.log(prop.pacc) if prop.pacc > 0 else -math.inf
        return log_pi(sigma) + lq + la

    bad = []
    for sigma in permutations(range(n)):
        asg, _ = arrays(sigma)
        for r in range(n):
            wi = int(asg[r])
            for wj in range(n):
                if wj == wi:
                    continue
                after = list(sigma)
                after[wi], after[wj] = sigma[wj], r
                fwd, rev = log_flux(sigma, r, wj), log_flux(tuple(after), r, wi)
                if not (math.isfinite(fwd) and math.isfinite(rev) and abs(fwd - rev) < 1e-9):
                    bad.append((sigma, r, wj, fwd, rev))
    return bad


def test_v2_detailed_balance_exact_enumeration_on_the_f09_matrix() -> None:
    from gareus.production import gibbs_propose_one_replica, select_gibbs_proposer
    bias = _f09_bias()
    assert _log_db_violations(gibbs_propose_one_replica, bias), "legacy must fail here or the test is blind"
    assert _log_db_violations(select_gibbs_proposer(aux_active=True), bias) == []
    mild = np.random.default_rng(3).uniform(-6.0, 6.0, size=(4, 4))
    assert _log_db_violations(select_gibbs_proposer(aux_active=True), mild, beta=0.5) == []


def test_gibbs_walk_call_site_selects_the_proposer_and_log_decision_by_aux() -> None:
    import ast
    import inspect
    import gareus.production as production
    src = ast.unparse(ast.parse(inspect.getsource(production.run_gareus)))
    block = src[src.index("if mode == 'gibbs-walk'"):]
    block = block[: block.index("raise ValueError")]
    assert "select_gibbs_proposer(aux_active=_aux_io is not None)" in block
    assert "lambda k, p: int(rng.choice(k, p=p))" in block
    assert "p_override=prop.pacc" in block
    assert "log_p_accept=prop.log_p_accept if _aux_io is not None else None" in block
