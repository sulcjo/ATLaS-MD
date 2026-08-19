import math

from gareus.dashboard.ranking import (
    BAD,
    OK,
    WARN,
    Hysteresis,
    rank_windows,
    restraint_sigma,
)


def _rank(**over):
    kwargs = dict(
        n_windows=4,
        centers_a=(4.0, 4.55, 5.10, 5.65),
        k_list=(2.5, 2.5, 2.5, 2.5),
        acceptance_by_window={0: 0.30, 1: 0.30, 2: 0.30, 3: 0.30},
        overlap_by_pair={(0, 1): 0.45, (1, 2): 0.45, (2, 3): 0.45},
        delta_by_window={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        temperature_k=300.0,
    )
    kwargs.update(over)
    return rank_windows(**kwargs)


def test_restraint_sigma_matches_sqrt_kt_over_k():
    # k in kcal/mol/A^2, kT at 300 K; sigma in A.
    sigma = restraint_sigma(2.5, 300.0)
    assert math.isclose(sigma, 0.4877, rel_tol=2e-3)


def test_restraint_sigma_is_infinite_for_a_zero_force_constant():
    assert math.isinf(restraint_sigma(0.0, 300.0))


def test_all_healthy_windows_rank_ok():
    statuses = _rank()
    assert {s.status for s in statuses} == {OK}
    assert [s.window for s in statuses] == [0, 1, 2, 3]


def test_dead_exchange_outranks_every_other_rule():
    statuses = _rank(acceptance_by_window={0: 0.30, 1: 0.30, 2: 0.30, 3: 0.001})
    assert statuses[0].window == 3
    assert statuses[0].status == BAD
    assert any("dead" in r for r in statuses[0].reasons)


def test_low_overlap_marks_both_windows_of_the_pair_bad():
    statuses = _rank(overlap_by_pair={(0, 1): 0.45, (1, 2): 0.02, (2, 3): 0.45})
    flagged = {s.window for s in statuses if s.status == BAD}
    assert flagged == {1, 2}


def test_delta_beyond_two_sigma_is_bad_and_named_pinned():
    statuses = _rank(delta_by_window={0: 0.0, 1: 0.0, 2: 2.31, 3: 0.0})
    worst = statuses[0]
    assert worst.window == 2 and worst.status == BAD
    assert any("pinned" in r for r in worst.reasons)


def test_low_but_not_dead_acceptance_is_a_warning():
    statuses = _rank(acceptance_by_window={0: 0.30, 1: 0.11, 2: 0.30, 3: 0.30})
    assert [s.status for s in statuses][0] == WARN
    assert statuses[0].window == 1


def test_ties_break_by_window_index_so_order_is_stable():
    statuses = _rank(delta_by_window={0: 3.0, 1: 0.0, 2: 3.0, 3: 0.0})
    assert [s.window for s in statuses[:2]] == [0, 2]


def test_hysteresis_keeps_a_recovered_key_for_the_configured_frames():
    h = Hysteresis(frames=3)
    assert h.update(["w17"]) == frozenset({"w17"})
    assert h.update([]) == frozenset({"w17"})     # frame 1 clear
    assert h.update([]) == frozenset({"w17"})     # frame 2 clear
    assert h.update([]) == frozenset()            # frame 3 clear -> released


def test_hysteresis_resets_the_countdown_when_a_key_re_triggers():
    h = Hysteresis(frames=2)
    h.update(["w17"])
    h.update([])
    assert h.update(["w17"]) == frozenset({"w17"})
    assert h.update([]) == frozenset({"w17"})
