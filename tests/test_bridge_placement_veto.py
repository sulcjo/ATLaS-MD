"""A weak-edge bridge that provably reconnects nothing must not outrank one that does.

Follow-up to the A8 bridge-prediction work (see
``tests/test_adaptive_new_state_guards.py`` and
``docs/chignolin_6_low_ess_root_cause.md``).  The prediction was already
computed there and already recorded in the new state's ``reason`` -- but it
never influenced the *allocation*.  ``bridges_needed`` was floored at 1 and the
per-epoch window budget was handed out flat round-robin, so on chignolin_6 the
single worst edge (14-20, one midpoint measuring 0.055 against the 0.127 it was
built to repair) always took a round-1 slot ahead of an edge whose second
bridge would have finished a real repair.

What changed, and the one thing that deliberately did NOT:

* Edges that CAN be reconnected inside this epoch's budget are now served
  first, each seeded with the number of bridges it actually takes to reach an
  endpoint (``min_useful_bridges``), before any budget reaches an edge that
  cannot be reconnected this epoch.
* An edge that cannot be reconnected this epoch still gets whatever budget is
  left over, one window at a time, exactly as before -- because an evenly
  spaced placement HALVES the gap, so that window is a bisection step, not a
  wasted slot.  ``test_a_single_bridge_on_a_hopeless_edge_is_a_bisection_step``
  measures that directly through the real prediction.  Refusing outright
  (``bridge_skip_unreachable=True``) is available, and is opt-in precisely
  because it makes an under-budgeted run add zero windows forever.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import gareus.adaptive_production as ap
from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    WindowStateRegistry,
    _bridge_placement_prediction,
    policy_from_args,
    propose_actions_from_diagnostics,
    write_epoch_action_report,
)

# kT at the temperature every test below uses, kcal/mol.
KBT_300 = 1.987204e-3 * 300.0


class _Args:
    """Minimal stand-in for the argparse namespace the driver threads around."""

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


def _adds(actions):
    return [a for a in actions if a[0] == "add"]


def _edge_label(action) -> str:
    """``weak edge 0-1: ...`` -> ``0-1`` (the reason string is the only place
    the placement records which edge it was for)."""
    return action[3].split(":")[0].replace("weak edge ", "").strip()


# ---------------------------------------------------------------------------
# the prediction: which endpoints can a bridge chain actually reach
# ---------------------------------------------------------------------------


def test_the_chignolin_6_edge_reaches_neither_endpoint_with_one_bridge():
    """Real registry values for the edge that produced the phantom repair.

    State 14 (secondary +0.12642021, k2 113.0943) and state 20 (secondary
    -1.62819669, k2 89.8823): the midpoint is ~12.1 and ~10.8 sigma from the
    two endpoints, so it reaches NEITHER.  16 bridges close the gap for both
    endpoints; 14 already reach the softer one, which is what the allocation
    now needs to know.
    """
    registry = WindowStateRegistry()
    s14 = registry.add_state(0.0654, 200.0, 0.12642021030596196, 113.0943250553548,
                             epoch=0, source="epoch0_windows")
    s20 = registry.add_state(0.114085585747846, 200.0, -1.6281966885808175,
                             89.88231226289835, epoch=1, source="tica_coverage")

    pred = _bridge_placement_prediction(s14, s20, temperature_K=300.0)

    assert pred["single_bridge_reaches_i"] is False
    assert pred["single_bridge_reaches_j"] is False
    assert pred["single_bridge_reaches_either"] is False
    assert pred["bridges_needed"] == 16
    assert pred["min_useful_bridges"] == 14


def test_min_useful_is_one_when_the_midpoint_reaches_the_softer_endpoint():
    """Reaching ONE endpoint is enough to make a single bridge worth placing.

    A chain hanging off one endpoint is a partial repair the next epoch can
    extend; a chain hanging off nothing is an island.  Here the soft endpoint
    (k2 = 10, sigma 0.244) is within reach of the midpoint and the stiff one
    (k2 = 400, sigma 0.0386) is nowhere near it.
    """
    registry = WindowStateRegistry()
    a = registry.add_state(0.0, 200.0, 0.0, 10.0, epoch=0, source="seed")
    b = registry.add_state(0.0, 200.0, 0.5, 400.0, epoch=0, source="seed")

    pred = _bridge_placement_prediction(a, b, temperature_K=300.0)

    assert pred["single_bridge_reaches_i"] is True
    assert pred["single_bridge_reaches_j"] is False
    assert pred["single_bridge_reaches_either"] is True
    assert pred["min_useful_bridges"] == 1
    # It is still not a full repair -- both endpoints are not reachable.
    assert pred["single_bridge_sufficient"] is False
    assert pred["bridges_needed"] > 1


def test_reachability_reduces_per_endpoint_over_all_axes_not_per_axis():
    """The trap fixture: each axis has one reachable endpoint, but not the same one.

    Endpoint i is soft on the primary axis and stiff on the secondary; endpoint
    j is the mirror image.  So `min` over the two endpoints *inside* each axis,
    then `max` over axes -- the obvious-looking reduction -- reports a
    comfortably reachable 1.02 sigma and would place a bridge connected to
    nothing.  The correct reduction fixes one endpoint, takes the worst axis
    for it, and only then compares the two endpoints.
    """
    registry = WindowStateRegistry()
    a = registry.add_state(0.0, 10.0, 0.0, 1000.0, epoch=0, source="seed")
    b = registry.add_state(0.5, 1000.0, 0.5, 10.0, epoch=0, source="seed")

    pred = _bridge_placement_prediction(a, b, temperature_K=300.0)

    # Fixture check, read off the function's own per-axis output: each axis
    # really does have exactly one endpoint within the healthy band.
    healthy = float(pred["healthy_spacing_sigma"])
    prim, sec = pred["axes"]["primary"], pred["axes"]["secondary"]
    assert prim["midpoint_spacing_sigma_i"] <= healthy < prim["midpoint_spacing_sigma_j"]
    assert sec["midpoint_spacing_sigma_j"] <= healthy < sec["midpoint_spacing_sigma_i"]

    # ...and neither endpoint is reachable on BOTH axes, which is what counts.
    assert pred["single_bridge_reaches_i"] is False
    assert pred["single_bridge_reaches_j"] is False
    assert pred["single_bridge_reaches_either"] is False
    assert pred["min_useful_bridges"] == 13


def test_min_useful_never_exceeds_the_full_repair_count():
    """Property over random geometries, both numbers straight from the function.

    ``min_useful_bridges`` (reach ONE endpoint) can never ask for more windows
    than ``bridges_needed`` (reach both), and neither may drop below 1.  A
    swapped sigma_min/sigma_max in either formula breaks this immediately.
    """
    rng = np.random.default_rng(20260826)
    registry = WindowStateRegistry()
    for _ in range(400):
        c1, c2 = rng.normal(scale=2.0, size=2)
        k1, k2 = rng.uniform(5.0, 500.0, size=2)
        s1, s2 = rng.normal(scale=2.0, size=2)
        ks1, ks2 = rng.uniform(5.0, 500.0, size=2)
        a = registry.add_state(float(c1), float(k1), float(s1), float(ks1),
                               epoch=0, source="seed")
        b = registry.add_state(float(c2), float(k2), float(s2), float(ks2),
                               epoch=0, source="seed")
        pred = _bridge_placement_prediction(a, b, temperature_K=300.0)
        assert pred["min_useful_bridges"] >= 1
        assert pred["min_useful_bridges"] <= pred["bridges_needed"]
        # The two booleans must agree with the count they are derived from.
        assert pred["single_bridge_reaches_either"] == (pred["min_useful_bridges"] == 1)


def test_an_unmeasurable_geometry_is_never_declared_unreachable():
    """No springs on either axis: nothing is proven, so nothing is vetoed.

    The veto is evidence-based -- it fires only where the geometry PROVES a
    chain reaches neither endpoint.  An unmeasurable pair is not proof, so it
    keeps the historical single midpoint even under the hard-refuse policy.
    """
    registry = WindowStateRegistry()
    a = registry.add_state(0.0, 0.0, None, None, epoch=0, source="seed")
    b = registry.add_state(0.5, 0.0, None, None, epoch=0, source="seed")

    pred = _bridge_placement_prediction(a, b, temperature_K=300.0)
    assert pred["axes"] == {}
    assert pred["min_useful_bridges"] == 1
    assert pred["single_bridge_reaches_either"] is True

    diagnostics = {
        "states": [{"state_id": 0, "sample_count": 100_000},
                   {"state_id": 1, "sample_count": 100_000}],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.05}],
    }
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=1,
                                    bridge_skip_unreachable=True)
    adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))
    assert len(adds) == 1


# ---------------------------------------------------------------------------
# allocation: a hopeless edge must not outrank a repairable one
# ---------------------------------------------------------------------------


def _hopeless_and_repairable():
    """chignolin_6's own edge 14-20 (hopeless at any realistic budget) plus one
    edge that two bridges genuinely repair, and a budget of exactly two.

    Overlaps are the real ones: the hopeless edge is also the WORST-overlap
    edge, so it sorts first and, under flat round-robin, always took the first
    slot.
    """
    registry = WindowStateRegistry()
    registry.add_state(0.0654, 200.0, 0.12642021030596196, 113.0943250553548,
                       epoch=0, source="epoch0_windows")            # 0
    registry.add_state(0.114085585747846, 200.0, -1.6281966885808175,
                       89.88231226289835, epoch=1, source="tica_coverage")  # 1
    registry.add_state(0.5, 200.0, 0.0, 100.0, epoch=0, source="seed")      # 2
    registry.add_state(0.5, 200.0, 0.30, 100.0, epoch=0, source="seed")     # 3
    diagnostics = {
        "states": [{"state_id": i, "sample_count": 100_000} for i in range(4)],
        "edges": [
            {"state_i": 0, "state_j": 1, "overlap": 0.12670336064083917},
            {"state_i": 2, "state_j": 3, "overlap": 0.20},
        ],
    }
    return registry, diagnostics


def test_the_repairable_edge_is_fully_funded_before_the_hopeless_one():
    """The reviewer's scenario, end to end through the real proposer.

    Budget 2.  Flat round-robin gave one window to each edge: the hopeless
    edge's window reconnects nothing (it needs 14 to reach even the softer
    endpoint) and the repairable edge is left one short of the two it needs.
    Both windows cost real MD.  Serving the repairable edge first spends the
    same two windows on a repair that actually completes.
    """
    registry, diagnostics = _hopeless_and_repairable()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=2)

    adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))

    assert len(adds) == 2
    assert sorted(_edge_label(a) for a in adds) == ["2-3", "2-3"]


def test_the_funded_edge_really_is_reconnected_by_what_was_placed():
    """Not just "two windows on edge 2-3" -- measure the result.

    Each placed centre is scored against the endpoint it sits next to using
    that endpoint's own sigma, as reported by the prediction itself (never
    recomputed here), and must land inside the healthy band.
    """
    registry, diagnostics = _hopeless_and_repairable()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=2)

    adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))
    centers = sorted(float(a[2][2]) for a in adds)

    s2, s3 = registry.get_state(2), registry.get_state(3)
    pred = _bridge_placement_prediction(s2, s3, temperature_K=300.0)
    healthy = float(pred["healthy_spacing_sigma"])
    sigma_i = float(pred["axes"]["secondary"]["sigma_i"])
    sigma_j = float(pred["axes"]["secondary"]["sigma_j"])

    assert (centers[0] - float(s2.secondary_center)) <= healthy * sigma_i + 1e-12
    assert (float(s3.secondary_center) - centers[-1]) <= healthy * sigma_j + 1e-12
    # ...and this is exactly what a single midpoint would NOT have achieved.
    midpoint = 0.5 * (float(s2.secondary_center) + float(s3.secondary_center))
    assert (midpoint - float(s2.secondary_center)) > healthy * sigma_i


def test_flat_round_robin_is_restorable_and_is_the_behaviour_being_replaced():
    """``--no-ap-bridge-repairable-first`` reproduces the old allocation exactly.

    This is also the fixture's proof that the test above has teeth: with the
    ordering switched off, the same inputs put one window on each edge --
    including the hopeless one.
    """
    registry, diagnostics = _hopeless_and_repairable()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=2,
                                    bridge_repairable_first=False)

    adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))

    assert len(adds) == 2
    assert sorted(_edge_label(a) for a in adds) == ["0-1", "2-3"]


def test_leftover_budget_still_reaches_an_edge_that_cannot_be_repaired_yet():
    """Priority, not exclusion: spare windows still bisect the hopeless edge.

    Budget 3 on the same fixture -- the repairable edge takes its two, and the
    third window goes to the hopeless edge as a bisection step rather than
    being dropped on the floor.
    """
    registry, diagnostics = _hopeless_and_repairable()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=3)

    adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))

    assert sorted(_edge_label(a) for a in adds) == ["0-1", "2-3", "2-3"]


def _one_expensive_and_three_cheap_repairs():
    """Budget 4. One overlap-0.02 edge needs 4 windows; three weaker-overlap
    edges are each fully repaired by ONE midpoint.

    Weakness is measured, not geometric, so "overlap-weak but geometrically
    almost fine" is the common case rather than an exotic one -- these three
    edges sit 1 sigma apart on CV2 and still fail the overlap target.
    """
    registry = WindowStateRegistry()
    sigma = math.sqrt(KBT_300 / 100.0)
    registry.add_state(0.0, 200.0, 0.0, 100.0, epoch=0, source="seed")     # 0
    registry.add_state(0.0, 200.0, 0.55, 100.0, epoch=0, source="seed")    # 1
    for i in range(3):                                                     # 2..7
        registry.add_state(float(i + 1), 200.0, 0.0, 100.0, epoch=0, source="seed")
        registry.add_state(float(i + 1), 200.0, sigma, 100.0, epoch=0, source="seed")
    diagnostics = {
        "states": [{"state_id": i, "sample_count": 100_000} for i in range(8)],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.02},
                  {"state_i": 2, "state_j": 3, "overlap": 0.10},
                  {"state_i": 4, "state_j": 5, "overlap": 0.15},
                  {"state_i": 6, "state_j": 7, "overlap": 0.20}],
    }
    return registry, diagnostics


def test_one_expensive_edge_cannot_eat_the_budget_of_three_cheap_repairs():
    """The round-2 fairness property, in the regime its own fixture cannot reach.

    Serving repairable edges worst-overlap-first re-creates exactly the
    regression the flat round-robin was introduced to fix: the single worst
    edge seeds all 4 windows and three edges that one midpoint each would have
    FULLY repaired get nothing. Cheapest-first completes all three and leaves
    the expensive edge its bisection step.
    """
    registry, diagnostics = _one_expensive_and_three_cheap_repairs()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=4)

    adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))

    assert sorted(_edge_label(a) for a in adds) == ["0-1", "2-3", "4-5", "6-7"]
    # Fixture check: the expensive edge really did want all four windows, so
    # the assertion above is about priority and not about a budget that was
    # never contested.
    pred = _bridge_placement_prediction(registry.get_state(0), registry.get_state(1),
                                        temperature_K=300.0)
    assert pred["min_useful_bridges"] == 4
    # ...and the three cheap ones are complete repairs, not partial ones.
    for action in adds:
        if _edge_label(action) != "0-1":
            assert "UNDER-BRIDGED" not in action[3], action[3]
            assert "BISECTION-ONLY" not in action[3], action[3]


def test_a_budget_starved_repairable_edge_is_not_reported_as_unreachable():
    """Wrong-instruction guard: never tell an operator to raise a budget that
    was already big enough.

    Under the hard-refusal policy an edge that IS repairable within the budget
    but lost the race to cheaper edges must be reported as budget contention,
    not as geometry -- ``--ap-max-new-windows >= 4`` is useless advice on a run
    already configured for 4.
    """
    registry, diagnostics = _one_expensive_and_three_cheap_repairs()
    # Budget 4 fits the expensive edge exactly, but the three cheap repairs are
    # seeded first and leave it 1.
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=4,
                                    bridge_skip_unreachable=True)
    plan: list = []
    logged = _captured_warnings(lambda: propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0, bridge_plan_out=plan))

    row = next(r for r in plan if (r["state_i"], r["state_j"]) == (0, 1))
    assert row["repairable_this_epoch"] is True
    assert row["funded_repairable"] is False
    assert row["outcome"] == "skipped_no_budget"
    lines = [line for line in logged if "0-1" in line]
    assert lines and not any("REFUSED" in line for line in lines), lines
    assert any("spent on worse edges" in line for line in lines), lines


def test_a_single_bridge_on_a_hopeless_edge_is_a_bisection_step():
    """Why the default is priority and not refusal, measured not asserted.

    One evenly spaced bridge halves the gap, so the follow-up edge the next
    epoch sees needs strictly fewer bridges.  Both numbers come from the real
    prediction, and the placement comes from the real proposer.
    """
    registry = WindowStateRegistry()
    a = registry.add_state(0.0, 200.0, 0.0, 100.0, epoch=0, source="seed")
    b = registry.add_state(0.0, 200.0, 1.0, 100.0, epoch=0, source="seed")
    diagnostics = {
        "states": [{"state_id": 0, "sample_count": 100_000},
                   {"state_id": 1, "sample_count": 100_000}],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.05}],
    }
    before = _bridge_placement_prediction(a, b, temperature_K=300.0)
    assert before["single_bridge_reaches_either"] is False

    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=1)
    adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))
    assert len(adds) == 1, "an under-budgeted run must still make progress"

    params = adds[0][2]
    bridge = registry.add_state(params[0], params[1], params[2], params[3],
                                epoch=1, source="bridge")
    after = _bridge_placement_prediction(a, bridge, temperature_K=300.0)
    assert after["bridges_needed"] < before["bridges_needed"]
    assert after["min_useful_bridges"] < before["min_useful_bridges"]


def test_the_hard_refusal_policy_is_opt_in_and_places_nothing():
    """``bridge_skip_unreachable=True`` refuses rather than bisects.

    Off by default (a run whose budget is too small for every weak edge would
    then add zero windows every epoch, forever); available for an operator who
    would rather spend the whole budget on repairable edges only.
    """
    registry = WindowStateRegistry()
    registry.add_state(0.0, 200.0, 0.0, 100.0, epoch=0, source="seed")
    registry.add_state(0.0, 200.0, 1.0, 100.0, epoch=0, source="seed")
    diagnostics = {
        "states": [{"state_id": 0, "sample_count": 100_000},
                   {"state_id": 1, "sample_count": 100_000}],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.05}],
    }

    default_adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics,
        AdaptiveDecisionPolicy(max_new_windows_per_epoch=1),
        temperature_K=300.0))
    assert len(default_adds) == 1

    refused_adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics,
        AdaptiveDecisionPolicy(max_new_windows_per_epoch=1,
                               bridge_skip_unreachable=True),
        temperature_K=300.0))
    assert refused_adds == []


def test_the_hard_refusal_hands_the_freed_budget_to_a_repairable_edge():
    registry, diagnostics = _hopeless_and_repairable()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=3,
                                    bridge_skip_unreachable=True)

    adds = _adds(propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))

    # Edge 2-3 needs (and is capped at) two; the third window is not spent on
    # the edge that cannot use it.
    assert sorted(_edge_label(a) for a in adds) == ["2-3", "2-3"]


# ---------------------------------------------------------------------------
# an unrepairable weak edge is information, not noise
# ---------------------------------------------------------------------------


def _captured_warnings(fn):
    logged: list = []
    orig = ap.logging.warning
    ap.logging.warning = lambda fmt, *a: logged.append(fmt % a if a else fmt)
    try:
        fn()
    finally:
        ap.logging.warning = orig
    return logged


def test_an_edge_that_only_got_a_bisection_step_says_so_with_a_fix():
    registry, diagnostics = _hopeless_and_repairable()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=3)

    logged = _captured_warnings(lambda: propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))

    lines = [line for line in logged if "0-1" in line]
    assert lines, f"the under-repaired edge must be reported; got {logged}"
    line = lines[0]
    assert "14" in line, line          # min_useful_bridges, the actionable number
    assert "--ap-max-new-windows" in line, line


def test_a_refused_edge_is_reported_loudly_rather_than_dropped():
    registry, diagnostics = _hopeless_and_repairable()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=3,
                                    bridge_skip_unreachable=True)

    logged = _captured_warnings(lambda: propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0))

    lines = [line for line in logged if "0-1" in line]
    assert lines, f"a refused edge must never be silent; got {logged}"
    assert any("no bridge windows" in line or "refus" in line.lower() for line in lines), lines
    assert any("14" in line for line in lines), lines


def test_the_bridge_plan_is_returned_as_a_structured_record():
    registry, diagnostics = _hopeless_and_repairable()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=2)
    plan: list = []

    propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0, bridge_plan_out=plan)

    by_edge = {f"{r['state_i']}-{r['state_j']}": r for r in plan}
    assert set(by_edge) == {"0-1", "2-3"}

    hopeless = by_edge["0-1"]
    assert hopeless["bridges_placed"] == 0
    assert hopeless["min_useful_bridges"] == 14
    assert hopeless["bridges_needed"] == 16
    assert hopeless["repairable_this_epoch"] is False
    assert hopeless["outcome"] == "skipped_no_budget"

    repaired = by_edge["2-3"]
    assert repaired["bridges_placed"] == 2
    assert repaired["min_useful_bridges"] == 2
    assert repaired["repairable_this_epoch"] is True
    assert repaired["outcome"] == "bridged"


def test_the_bridge_plan_reaches_the_epoch_action_report(tmp_path):
    """An unrepairable edge must survive the epoch as a written artefact.

    A log line scrolls away; ``adaptive_epoch_actions.json`` is what an
    operator (or a later investigation) actually reads.
    """
    registry, diagnostics = _hopeless_and_repairable()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=2)
    plan: list = []
    actions = propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0, bridge_plan_out=plan)

    paths = write_epoch_action_report(tmp_path, 1, registry, diagnostics, actions,
                                      policy, bridge_plan=plan)

    payload = json.loads(open(paths["json"], encoding="utf-8").read())
    rows = {f"{r['state_i']}-{r['state_j']}": r for r in payload["bridge_plan"]}
    assert rows["0-1"]["outcome"] == "skipped_no_budget"
    assert rows["0-1"]["min_useful_bridges"] == 14
    assert rows["2-3"]["outcome"] == "bridged"

    md = open(paths["md"], encoding="utf-8").read()
    assert "0-1" in md and "skipped_no_budget" in md


def test_the_written_plan_is_strictly_parseable_json(tmp_path):
    """An unmeasurable geometry leaves the spacing ratio non-finite.

    ``write_json`` -> ``json.dumps`` keeps ``allow_nan`` on, so a NaN would go
    to disk as the bare token ``NaN``: Python reads it back, every strict
    parser (jq, a browser, most other languages) refuses the whole file. One
    unmeasurable weak edge would take the entire epoch report with it.
    """
    registry = WindowStateRegistry()
    registry.add_state(0.0, 0.0, None, None, epoch=0, source="seed")
    registry.add_state(0.5, 0.0, None, None, epoch=0, source="seed")
    diagnostics = {
        "states": [{"state_id": 0, "sample_count": 100_000},
                   {"state_id": 1, "sample_count": 100_000}],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.05}],
    }
    plan: list = []
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=1)
    actions = propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0, bridge_plan_out=plan)
    assert not math.isfinite(
        plan[0]["worst_midpoint_spacing_sigma"] if plan[0]["worst_midpoint_spacing_sigma"]
        is not None else float("nan"))

    paths = write_epoch_action_report(tmp_path, 0, registry, diagnostics, actions,
                                      policy, bridge_plan=plan)
    raw = Path(paths["json"]).read_text(encoding="utf-8")

    def _reject(token):
        raise AssertionError(f"non-JSON constant {token!r} was written to disk")

    payload = json.loads(raw, parse_constant=_reject)
    assert payload["bridge_plan"][0]["worst_midpoint_spacing_sigma"] is None


# ---------------------------------------------------------------------------
# the knobs must be reachable from the real CLI (the dropped-knob pattern)
# ---------------------------------------------------------------------------


def test_the_new_bridge_knobs_are_reachable_from_the_real_cli():
    from gareus.cli import parse_args

    base = ["--seq", "GYDPETGTWG", "--out", "o"]
    args = parse_args(base)
    assert args.adaptive_production_bridge_repairable_first is True
    assert args.adaptive_production_bridge_skip_unreachable is False

    tuned = parse_args(base + ["--no-ap-bridge-repairable-first",
                               "--ap-bridge-skip-unreachable"])
    assert tuned.adaptive_production_bridge_repairable_first is False
    assert tuned.adaptive_production_bridge_skip_unreachable is True

    policy = policy_from_args(tuned)
    assert policy.bridge_repairable_first is False
    assert policy.bridge_skip_unreachable is True


def test_the_policy_defaults_keep_the_shipped_behaviour():
    defaults = policy_from_args(_Args())
    assert defaults.bridge_repairable_first is True
    assert defaults.bridge_skip_unreachable is False

    overridden = policy_from_args(_Args(
        adaptive_production_bridge_repairable_first=False,
        adaptive_production_bridge_skip_unreachable=True,
    ))
    assert overridden.bridge_repairable_first is False
    assert overridden.bridge_skip_unreachable is True
