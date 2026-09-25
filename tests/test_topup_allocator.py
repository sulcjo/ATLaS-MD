import math

from gareus.adaptive.throughput import wall_hours
from gareus.adaptive.topup_allocator import plan_topup
from gareus.adaptive.union_diagnostics import UnionDiagnostics
from gareus.adaptive_production import AdaptiveDecisionPolicy

POL = AdaptiveDecisionPolicy(topups_enabled=True)          # target 0.10, weak 0.15, cap 0.3, 2 attempts
IV = 500                                                   # report interval


def _diag(sigma, edges=None, unconverged=(), n=None):
    ids = tuple(sorted(sigma))
    return UnionDiagnostics(state_ids=ids, n_k=n or {s: 100 for s in ids}, sigma_kcal=dict(sigma),
                            unconverged=frozenset(unconverged), inefficiency={s: 2.0 for s in ids},
                            edge_overlap=edges or {}, f_kT={})


def _chain(n):
    # n centres on one rung (0..n-1) and the same centres on a second rung (n..2n-1)
    nb = {s: [x for x in (s - 1, s + 1) if 0 <= x < n] for s in range(n)}
    nb.update({s + n: [x + n for x in (s - 1, s + 1) if 0 <= x < n] for s in range(n)})
    rp = {s: [s + n] for s in range(n)}; rp.update({s + n: [s] for s in range(n)})
    return nb, rp


def _plan(diag, n=4, budget=100.0, correction=None, attempts=None, nb=None, rp=None):
    nb0, rp0 = _chain(n)
    return plan_topup(diag, state_ids_in_order=list(diag.state_ids), neighbours=nb or nb0,
                      rung_partners=rp or rp0, policy=POL, report_interval=IV, timestep_fs=4.0,
                      n_gpus=4, budget_hours=budget, correction=correction, edge_attempts=attempts)


def _one_deficit(sigma1=0.15):
    s = {x: 0.05 for x in range(8)}; s[1] = sigma1
    return s


def test_no_diagnostics_means_no_topup():
    nb, rp = _chain(4)
    p = plan_topup(None, state_ids_in_order=list(range(8)), neighbours=nb, rung_partners=rp, policy=POL,
                   report_interval=IV, timestep_fs=4.0, n_gpus=4, budget_hours=100.0)
    assert p.reason == "no_diagnostics" and p.state_ids == ()


def test_a_healthy_campaign_gets_no_topup():
    p = _plan(_diag({s: 0.05 for s in range(8)}))
    assert p.reason == "healthy" and p.state_ids == () and p.steps == 0


def test_unmeasured_edges_never_make_a_state_deficient():
    assert _plan(_diag({s: 0.05 for s in range(8)}, edges={})).reason == "healthy"


def test_one_deficit_state_gets_minimal_partners_and_the_right_length():
    p = _plan(_diag(_one_deficit()))
    assert p.reason == "planned" and p.deficit_state_ids == (1,)
    assert len(p.partner_state_ids) == 2 and 5 in p.partner_state_ids   # one same-rung, one rung partner
    # n=100 decorrelated, g=2, sigma 0.15 -> 0.10 needs 125 more decorrelated = 125*500*2 steps
    assert p.steps == 125_000
    assert p.predicted_sigma[1] <= POL.topup_target_sigma + 1e-9


def test_a_weak_edge_between_healthy_states_is_structural_not_md():
    p = _plan(_diag({s: 0.05 for s in range(8)}, edges={(1, 2): 0.02}))
    assert p.structural_edges == ((1, 2),) and p.reason == "healthy"


def test_a_weak_edge_touching_a_deficit_tops_up_both_endpoints():
    p = _plan(_diag(_one_deficit(), edges={(1, 2): 0.02}))
    assert {1, 2} <= set(p.deficit_state_ids) and p.structural_edges == ()
    assert p.weak_edges_topped == ((1, 2),)


def test_an_edge_that_stayed_weak_after_max_attempts_becomes_structural():
    p = _plan(_diag(_one_deficit(), edges={(1, 2): 0.02}), attempts={(1, 2): 2})
    assert p.structural_edges == ((1, 2),) and 2 not in p.deficit_state_ids


def test_nan_sigma_or_zero_samples_is_not_a_topup_target():
    s = {x: 0.05 for x in range(8)}; s[3] = math.nan
    assert _plan(_diag(s)).reason == "healthy"


def test_a_never_sampled_state_is_never_a_partner():
    s = _one_deficit(); s[0] = math.nan; s[2] = math.nan          # both same-rung neighbours of 1 unsampled
    n = {x: 100 for x in range(8)}; n[0] = 0; n[2] = 0
    p = _plan(_diag(s, n=n))
    assert p.reason == "planned" and not ({0, 2} & set(p.partner_state_ids))


def test_a_budget_smaller_than_one_report_interval_plans_nothing():
    p = _plan(_diag(_one_deficit()), budget=1e-9)
    assert p.reason == "cap_too_small" and p.state_ids == () and p.steps == 0


def test_a_moderate_budget_funds_a_partial_topup():
    budget = wall_hours(60_000, 3, 4.0, 4, POL.topup_throughput_table)   # half of the 125k need, 3-state patch
    p = _plan(_diag(_one_deficit()), budget=budget)
    assert p.reason == "planned" and 0 < p.steps <= 60_000
    assert POL.topup_target_sigma < p.predicted_sigma[1] < 0.15


def test_a_deficit_at_the_layout_edge_still_gets_a_partner():
    s = {x: 0.05 for x in range(8)}; s[0] = 0.15
    nb = {0: [], 1: [0], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    rp = {0: [4], 4: [0], 1: [5], 5: [1], 2: [6], 6: [2], 3: [7], 7: [3]}
    p = _plan(_diag(s), nb=nb, rp=rp)
    assert p.reason == "planned" and 4 in p.partner_state_ids


def test_uniformly_deficient_states_degenerate_to_all_states():
    p = _plan(_diag({s: 0.15 for s in range(8)}))
    assert set(p.state_ids) == set(range(8)) and p.partner_state_ids == ()


def test_a_state_that_underdelivered_needs_more_steps():
    assert _plan(_diag(_one_deficit()), correction={1: 0.5}).steps > _plan(_diag(_one_deficit())).steps


def test_a_bad_correction_factor_is_clamped_not_propagated():
    p = _plan(_diag(_one_deficit()), correction={1: -3.0})
    assert p.reason == "planned" and math.isfinite(p.predicted_sigma[1]) and p.steps > 0


def test_the_plan_is_deterministic():
    assert _plan(_diag(_one_deficit())) == _plan(_diag(_one_deficit()))
