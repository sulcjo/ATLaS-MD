"""Guards on the states adaptive production creates for itself (A7/A8/A9).

All three regressions were root-caused on the real run ``RUNS/chignolin_6``
(base MBAR ESS 0.74%, health FAIL) -- see
``docs/chignolin_6_low_ess_root_cause.md``.  The numbers hard-coded below are
that run's own registry values (``adaptive_production/final_registry_used_for_mbar.csv``)
and its own final-phase schedule/runtime-pool records, so each test fails on
the pre-fix code for exactly the reason the run did.
"""

import math

import numpy as np
import pytest

import gareus.adaptive_production as ap
from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    AdaptiveRuntimePool,
    WindowStateRegistry,
    _apply_tica_cv2_switch,
    _bridge_placement_prediction,
    _clamp_secondary_k,
    _final_extension_steps,
    _propose_tica_coverage_actions,
    _resolve_secondary_k_max,
    _scheduled_final_default_steps,
    build_adaptive_epoch_schedule,
    policy_from_args,
    propose_actions_from_diagnostics,
)

# kT at the temperature every test below uses, kcal/mol.
KBT_300 = 1.987204e-3 * 300.0


class _Args:
    """Minimal stand-in for the argparse namespace the driver threads around."""

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


# ---------------------------------------------------------------------------
# A7 -- cv2_k_max must also bind states the controller creates itself
# ---------------------------------------------------------------------------


def test_clamp_helper_passes_through_and_reports_only_when_it_bites():
    kept, warning = _clamp_secondary_k(150.0, 200.0, context="unit")
    assert kept == pytest.approx(150.0)
    assert warning is None

    clamped, warning = _clamp_secondary_k(402.4728194252458, 200.0, context="unit")
    assert clamped == pytest.approx(200.0)
    assert warning is not None
    assert "402.47" in warning and "200" in warning and "unit" in warning


@pytest.mark.parametrize("no_cap", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_clamp_helper_is_a_no_op_without_a_usable_cap(no_cap):
    """Backward compatibility: no configured ceiling means no behavior change."""
    kept, warning = _clamp_secondary_k(402.4728194252458, no_cap, context="unit")
    assert kept == pytest.approx(402.4728194252458)
    assert warning is None


def test_clamp_helper_leaves_cv1_only_states_alone():
    kept, warning = _clamp_secondary_k(None, 200.0, context="unit")
    assert kept is None
    assert warning is None


def test_secondary_k_max_is_read_live_so_the_tica_cv2_switch_moves_it():
    """The cap must not be a loop-start snapshot.

    ``_apply_tica_cv2_switch`` rewrites ``args.cv2_k_max`` to
    ``tica_linear_k_max`` mid-campaign, and every state created afterwards
    belongs to the new CV2 regime with the new ceiling.  chignolin_6's state 22
    was created *after* its switch fired, so a pre-switch snapshot would clamp
    against the wrong number.
    """
    args = _Args(
        cv2_k_max=200.0,
        cv2_k_min=5.0,
        secondary_cv="torsion-pca",
        secondary_cv_adaptive_max_k_kcal=200.0,
        tica_switch_cv2=True,
        tica_linear_k_min=10.0,
        tica_linear_k_max=60.0,
        tica_state_file="",
    )
    assert _resolve_secondary_k_max(args) == pytest.approx(200.0)

    report = {"status": "updated", "state_file": __file__}
    assert _apply_tica_cv2_switch(args, report, next_epoch=1) is True
    assert _resolve_secondary_k_max(args) == pytest.approx(60.0)


def test_resolve_secondary_k_max_falls_back_to_the_adaptive_max_alias():
    args = _Args(secondary_cv_adaptive_max_k_kcal=175.0)
    assert _resolve_secondary_k_max(args) == pytest.approx(175.0)
    assert _resolve_secondary_k_max(_Args()) is None


def test_tica_coverage_add_clamps_secondary_k_to_the_configured_max():
    """Reproduces chignolin_6 state 22: k2=402.47 under cv2_k_max=200.

    The coverage path derives the new spring from the observed spread of the
    frames populating the uncovered region (k = kT/Var), clipped only against
    the parent's own k and ``coverage_k_stiffen_cap`` -- never against
    ``cv2_k_max``.  A tight tIC1 group therefore produced a restraint stiffer
    than the run ever asked for.
    """
    registry = WindowStateRegistry()
    registry.add_state(0.0, 200.0, 0.0, 200.0, epoch=0, source="epoch0_windows")
    policy = AdaptiveDecisionPolicy(coverage_k_stiffen_cap=4.0)

    # A very tight populated group far outside every coverage radius: its own
    # std is small, so kT/Var lands well above the cap.
    primary = np.concatenate([np.full(900, 0.0), np.full(60, 0.15)])
    tic1 = np.concatenate([
        np.zeros(900),
        np.linspace(1.34866, 1.49289, 60),
    ])

    uncapped = _propose_tica_coverage_actions(
        registry, primary, tic1, policy, temperature_K=300.0)
    assert len(uncapped) == 1
    assert uncapped[0][2][3] > 200.0, "fixture must exercise the cap"

    capped = _propose_tica_coverage_actions(
        registry, primary, tic1, policy, temperature_K=300.0, secondary_k_max=200.0)
    assert len(capped) == 1
    assert capped[0][2][3] == pytest.approx(200.0)
    assert "clamped" in capped[0][3]


def test_weak_edge_midpoint_clamps_secondary_k_to_the_configured_max():
    """Reproduces chignolin_6 state 25: 0.5*(148.46 + 402.47) = 275.47.

    Even with the coverage path capped, a midpoint average over any pair whose
    endpoints sit at the cap can only reach the cap -- but a legacy registry
    (or a hand-edited window table) can still hold an over-cap endpoint, so the
    bridge path needs its own clamp rather than trusting its inputs.
    """
    registry = WindowStateRegistry()
    registry.add_state(0.15170159836921232, 200.0, -1.437854818262977,
                       148.45990012351126, epoch=1, source="tica_coverage")
    registry.add_state(0.15306649388009563, 200.0, 1.4070028636924448,
                       402.4728194252458, epoch=1, source="tica_coverage")
    diagnostics = {
        "states": [
            {"state_id": 0, "sample_count": 100_000},
            {"state_id": 1, "sample_count": 100_000},
        ],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.28585979586370924}],
    }
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=1, bridge_multi_window=False)

    uncapped = propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0)
    adds = [a for a in uncapped if a[0] == "add"]
    assert len(adds) == 1
    assert adds[0][2][3] == pytest.approx(275.46635977437853)

    capped = propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0, secondary_k_max=200.0)
    adds = [a for a in capped if a[0] == "add"]
    assert len(adds) == 1
    assert adds[0][2][3] == pytest.approx(200.0)
    assert "clamped" in adds[0][3]


# ---------------------------------------------------------------------------
# A8 -- a bridge placement must be checked against each endpoint's own sigma
# ---------------------------------------------------------------------------


def test_bridge_prediction_reproduces_the_chignolin_6_edge_14_20_numbers():
    """State 24 was built to repair edge 14-20 and made it worse.

    Registry values: state 14 (primary 0.0654, k1 200, secondary +0.12642021,
    k2 113.0943) and state 20 (primary 0.11408559, k1 200, secondary
    -1.62819669, k2 89.8823).  The midpoint at -0.75088824 is *correct* as a
    midpoint, but sigma = sqrt(kT/k2) is 0.0726 / 0.0814 there, so the
    single window sits ~12 and ~11 sigma from the two endpoints it was
    supposed to bridge.
    """
    registry = WindowStateRegistry()
    s14 = registry.add_state(0.0654, 200.0, 0.12642021030596196, 113.0943250553548,
                             epoch=0, source="epoch0_windows")
    s20 = registry.add_state(0.114085585747846, 200.0, -1.6281966885808175,
                             89.88231226289835, epoch=1, source="tica_coverage")

    pred = _bridge_placement_prediction(s14, s20, temperature_K=300.0)

    # Secondary axis is the catastrophic one; primary is comfortably fine.
    sec = pred["axes"]["secondary"]
    assert sec["midpoint_spacing_sigma_i"] == pytest.approx(12.1, abs=0.15)
    assert sec["midpoint_spacing_sigma_j"] == pytest.approx(10.8, abs=0.15)
    prim = pred["axes"]["primary"]
    assert prim["worst_midpoint_spacing_sigma"] < 1.5

    assert pred["worst_axis"] == "secondary"
    assert pred["worst_midpoint_spacing_sigma"] == pytest.approx(12.1, abs=0.15)
    assert pred["single_bridge_sufficient"] is False
    # gap 1.7546 / (1.5 * sigma_min 0.07259) -> 17 intervals -> 16 bridges.
    assert pred["bridges_needed"] == 16


def test_bridge_prediction_accepts_a_genuinely_close_pair():
    registry = WindowStateRegistry()
    a = registry.add_state(0.0, 200.0, 0.00, 200.0, epoch=0, source="seed")
    # 1.0 sigma apart on the secondary axis at k2 = 200 (sigma = 0.0546).
    b = registry.add_state(0.0, 200.0, 0.0546, 200.0, epoch=0, source="seed")

    pred = _bridge_placement_prediction(a, b, temperature_K=300.0)
    assert pred["single_bridge_sufficient"] is True
    assert pred["bridges_needed"] == 1
    assert pred["worst_midpoint_spacing_sigma"] < 1.5


def test_bridge_prediction_handles_cv1_only_states():
    registry = WindowStateRegistry()
    a = registry.add_state(0.0, 200.0, None, None, epoch=0, source="seed")
    b = registry.add_state(0.5, 200.0, None, None, epoch=0, source="seed")

    pred = _bridge_placement_prediction(a, b, temperature_K=300.0)
    assert "secondary" not in pred["axes"]
    assert pred["worst_axis"] == "primary"
    assert pred["bridges_needed"] >= 1


def test_weak_edge_places_the_number_of_bridges_the_prediction_calls_for():
    """One midpoint cannot span 1.75 CV2 units at sigma ~ 0.073."""
    registry = WindowStateRegistry()
    registry.add_state(0.0654, 200.0, 0.12642021030596196, 113.0943250553548,
                       epoch=0, source="epoch0_windows")
    registry.add_state(0.114085585747846, 200.0, -1.6281966885808175,
                       89.88231226289835, epoch=1, source="tica_coverage")
    diagnostics = {
        "states": [
            {"state_id": 0, "sample_count": 423_957},
            {"state_id": 1, "sample_count": 855_517},
        ],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.12670336064083917}],
    }
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=32)

    actions = propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0)
    adds = [a for a in actions if a[0] == "add"]
    assert len(adds) == 16, "the prediction asks for 16 bridges, all of them fit"

    centers = sorted(float(a[2][2]) for a in adds)
    gaps = np.diff(centers)
    assert np.allclose(gaps, gaps[0]), "bridges must be evenly spaced"
    # Every new sub-interval is now inside the healthy spacing/sigma band.
    sigma_min = math.sqrt(KBT_300 / 113.0943250553548)
    assert gaps[0] <= 1.5 * sigma_min + 1e-12
    # 17 intervals is odd, so the old plain midpoint is *not* one of the new
    # centers -- it is bracketed by two of them instead.
    assert min(centers) > -1.6281966885808175
    assert max(centers) < 0.12642021030596196


def test_multi_bridge_interpolates_springs_and_inherits_the_nearer_endpoint():
    """Each bridge takes the spring and the parent of the side it sits on.

    The historical single midpoint used the plain endpoint average for the
    spring and always s1 as parent; at frac == 0.5 linear interpolation gives
    the identical spring and resolves to s1, so that case is unchanged.
    """
    registry = WindowStateRegistry()
    registry.add_state(0.0, 100.0, 0.0, 20.0, epoch=0, source="seed")       # soft
    registry.add_state(0.0, 300.0, 1.0, 180.0, epoch=0, source="seed")      # stiff
    diagnostics = {
        "states": [
            {"state_id": 0, "sample_count": 100_000},
            {"state_id": 1, "sample_count": 100_000},
        ],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.05}],
    }
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=3)

    actions = propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0)
    adds = [a for a in actions if a[0] == "add"]
    assert len(adds) == 3

    by_center = sorted(adds, key=lambda a: float(a[2][2]))
    centers = [float(a[2][2]) for a in by_center]
    assert centers == pytest.approx([0.25, 0.5, 0.75])
    # Springs ramp from the soft endpoint to the stiff one.
    springs = [float(a[2][3]) for a in by_center]
    assert springs == pytest.approx([60.0, 100.0, 140.0])
    assert [float(a[2][1]) for a in by_center] == pytest.approx([150.0, 200.0, 250.0])
    # Nearer endpoint becomes the parent; the exact midpoint stays on s1.
    assert [a[1] for a in by_center] == [0, 0, 1]


def test_single_bridge_placement_is_bit_for_bit_the_historical_midpoint():
    """Exact ``==``, not ``pytest.approx``.

    These particular endpoint values happen to agree under either formula, so
    the ULP-sensitive fixture that gives this claim real teeth lives in
    ``test_single_bridge_placement_params_are_exactly_the_historical_average``.
    """
    registry = WindowStateRegistry()
    registry.add_state(0.0, 100.0, 0.0, 20.0, epoch=0, source="seed")
    registry.add_state(0.4, 300.0, 1.0, 180.0, epoch=0, source="seed")
    diagnostics = {
        "states": [
            {"state_id": 0, "sample_count": 100_000},
            {"state_id": 1, "sample_count": 100_000},
        ],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.05}],
    }
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=4, bridge_multi_window=False)

    adds = [a for a in propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0) if a[0] == "add"]
    assert len(adds) == 1
    parent, params = adds[0][1], adds[0][2]
    assert parent == 0
    assert params[0] == 0.5 * (0.0 + 0.4)
    assert params[1] == 0.5 * (100.0 + 300.0)
    assert params[2] == 0.5 * (0.0 + 1.0)
    assert params[3] == 0.5 * (20.0 + 180.0)


def test_weak_edge_bridge_reason_records_the_prediction_and_any_shortfall():
    registry = WindowStateRegistry()
    registry.add_state(0.0654, 200.0, 0.12642021030596196, 113.0943250553548,
                       epoch=0, source="epoch0_windows")
    registry.add_state(0.114085585747846, 200.0, -1.6281966885808175,
                       89.88231226289835, epoch=1, source="tica_coverage")
    diagnostics = {
        "states": [
            {"state_id": 0, "sample_count": 423_957},
            {"state_id": 1, "sample_count": 855_517},
        ],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.12670336064083917}],
    }
    # chignolin_6's own cap: 4 new windows per epoch against 16 needed.
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=4)

    actions = propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0)
    adds = [a for a in actions if a[0] == "add"]
    assert len(adds) == 4

    reason = adds[0][3]
    assert "weak edge 0-1" in reason
    # The prediction that justified the placement is auditable from the row.
    assert "spacing/sigma" in reason
    assert "12.1" in reason or "12.0" in reason
    assert "bridge 1/4" in reason
    assert "16 needed" in reason
    assert "UNDER-BRIDGED" in reason


def test_weak_edge_multi_bridge_can_be_switched_off():
    """`bridge_multi_window=False` restores the historical single midpoint."""
    registry = WindowStateRegistry()
    registry.add_state(0.0654, 200.0, 0.12642021030596196, 113.0943250553548,
                       epoch=0, source="epoch0_windows")
    registry.add_state(0.114085585747846, 200.0, -1.6281966885808175,
                       89.88231226289835, epoch=1, source="tica_coverage")
    diagnostics = {
        "states": [
            {"state_id": 0, "sample_count": 423_957},
            {"state_id": 1, "sample_count": 855_517},
        ],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.12670336064083917}],
    }
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=4, bridge_multi_window=False)

    actions = propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0)
    adds = [a for a in actions if a[0] == "add"]
    assert len(adds) == 1
    assert float(adds[0][2][2]) == pytest.approx(-0.7508882391374277)
    # Even when placement is unchanged, the warning is still recorded.
    assert "UNDER-BRIDGED" in adds[0][3]


# ---------------------------------------------------------------------------
# A9 -- the scheduled final phase never sized itself from the runtime pool
# ---------------------------------------------------------------------------


def _chignolin_6_pool() -> AdaptiveRuntimePool:
    """The pool exactly as chignolin_6 recorded it entering the final phase."""
    pool = AdaptiveRuntimePool(total_ns=10_000.0, timestep_fs=4.0)
    pool.used_ns = 2469.486
    return pool


def test_scheduled_final_default_steps_come_from_the_pool_not_the_gamd_fallback():
    """chignolin_6 spent 54 ns of a 7,530 ns remaining budget on its final phase.

    ``build_adaptive_epoch_schedule`` was called with ``default_steps`` still
    equal to the pool-independent fallback ``gamd_production_steps // 20`` =
    500,000, because ``_final_steps_from_pool`` was only reached inside the
    *non*-scheduled final branch.  27 states * 500,000 steps * 4 fs = 54 ns.
    The scheduled *epoch* path does recompute from the pool
    (``_epoch_steps_from_pool``); the final path did not.
    """
    pool = _chignolin_6_pool()
    policy = AdaptiveDecisionPolicy(total_md_pool_ns=10_000.0, final_pool_fraction=0.5)

    steps = _scheduled_final_default_steps(
        pool, policy, n_states=27, timestep_fs=4.0,
        fallback_steps=500_000, explicit_final_steps=0,
    )

    # Whole remaining budget, spread over the 27 active states.
    expected = int(pool.remaining_ns() * 1.0e6 / 4.0 / 27)
    assert steps == expected
    assert steps > 60_000_000
    assert steps > 100 * 500_000, "the fallback was ~2 orders of magnitude too small"


def test_scheduled_final_default_steps_honours_an_explicit_override():
    pool = _chignolin_6_pool()
    policy = AdaptiveDecisionPolicy(total_md_pool_ns=10_000.0)
    steps = _scheduled_final_default_steps(
        pool, policy, n_states=27, timestep_fs=4.0,
        fallback_steps=500_000, explicit_final_steps=1_234_000,
    )
    assert steps == 1_234_000


@pytest.mark.parametrize("pool", [
    AdaptiveRuntimePool(total_ns=0.0, timestep_fs=4.0),
    None,
])
def test_scheduled_final_default_steps_falls_back_without_a_pool(pool):
    policy = AdaptiveDecisionPolicy()
    steps = _scheduled_final_default_steps(
        pool, policy, n_states=27, timestep_fs=4.0,
        fallback_steps=500_000, explicit_final_steps=0,
    )
    assert steps == 500_000


def test_pool_sized_final_default_steps_actually_reaches_the_new_states():
    """End-to-end on the allocator: the floor for new states was never the issue.

    chignolin_6's ``final_state_schedule.csv`` already gave states 24/25/26 the
    three *largest* allocations in the run (1.02M/1.20M/1.02M steps vs 341k
    typical) and they were honoured to within a rounding step -- yet they came
    out with ~10k samples each.  The cause is the size of the pie, not the
    slice: rebuilding the same schedule with a pool-derived ``default_steps``
    lifts those same states by orders of magnitude.
    """
    registry = WindowStateRegistry()
    for _ in range(24):
        registry.add_state(0.0654, 200.0, 0.1, 100.0, epoch=0, source="epoch0_windows")
    for _ in range(3):
        registry.add_state(0.09, 200.0, -0.75, 100.0, epoch=2, source="adaptive_production")
    diagnostics = {"states": [{"state_id": s.state_id, "sample_count": 300_000}
                              for s in registry.active_states()[:24]]}
    policy = AdaptiveDecisionPolicy(total_md_pool_ns=10_000.0, final_pool_fraction=0.5)
    pool = _chignolin_6_pool()

    old = build_adaptive_epoch_schedule(
        registry, diagnostics, policy, epoch=2, default_steps=500_000, final=True)
    new_steps = _scheduled_final_default_steps(
        pool, policy, n_states=27, timestep_fs=4.0,
        fallback_steps=500_000, explicit_final_steps=0,
    )
    new = build_adaptive_epoch_schedule(
        registry, diagnostics, policy, epoch=2, default_steps=new_steps, final=True)

    def _new_state_steps(rows):
        return [int(r["requested_steps"]) for r in rows
                if "new_state" in str(r["allocation_reason"])]

    old_new = _new_state_steps(old)
    new_new = _new_state_steps(new)
    assert len(old_new) == len(new_new) == 3
    # The old numbers are the real run's order of magnitude ...
    assert max(old_new) < 3_000_000
    # ... and the pool-sized schedule is at least 20x bigger for the same states.
    assert min(new_new) > 20 * max(old_new)


def test_frozen_final_extension_steps_never_inherit_the_pool_widened_target():
    """The pool-derived final target must not leak into the extension rounds.

    Extension rounds run after the final phase has already drawn its share of
    the pool, so seeding them from "the whole remaining budget" turns a top-up
    into a request for the entire pool -- a hard failure under
    ``pool_hard_stop``.  chignolin_6 had ``final_quality_extension_rounds = 0``
    so the regression would have been silent there.
    """
    policy = AdaptiveDecisionPolicy(final_quality_extension_steps=0)
    assert _final_extension_steps(policy, 500_000) == 500_000

    explicit = AdaptiveDecisionPolicy(final_quality_extension_steps=250_000)
    assert _final_extension_steps(explicit, 69_726_981) == 250_000


def test_scheduled_final_branch_does_not_rebind_final_steps():
    """Source-level guard for the leak above.

    The scheduled-final branch computes its pool-derived target into a separate
    local (``final_schedule_steps``).  Only the *non*-scheduled branch is
    allowed to rebind ``final_steps`` itself (it has always done so, and it has
    no extension-round successor reading it through a widened value).
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(ap.run_adaptive_production_auto_loop))
    rebinds = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "final_steps" for t in node.targets)
    ]
    # 1: the initial fallback definition. 2: the non-scheduled branch's own
    # _final_steps_from_pool recompute. Anything more means the scheduled
    # branch started widening the shared local again.
    assert len(rebinds) == 2, [ast.unparse(n) for n in rebinds]


def test_bridge_policy_fields_are_reachable_from_args():
    """Both new knobs must be readable off the args namespace, not defaults-only."""
    defaults = policy_from_args(_Args())
    assert defaults.bridge_multi_window is True
    assert defaults.bridge_healthy_spacing_sigma == pytest.approx(1.5)

    overridden = policy_from_args(_Args(
        adaptive_production_bridge_multi_window=False,
        adaptive_production_bridge_healthy_spacing_sigma=1.1,
    ))
    assert overridden.bridge_multi_window is False
    assert overridden.bridge_healthy_spacing_sigma == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# Review round 2 -- the driver read a temperature attribute that never exists
# ---------------------------------------------------------------------------


def test_args_temperature_k_reads_the_flag_the_cli_actually_defines():
    """``--temperature-k`` is the only temperature flag ``gareus/cli.py`` defines.

    The three driver call sites used to read ``getattr(args, "temperature",
    298.0)``; there is no ``--temperature`` flag and no compat shim that sets
    ``args.temperature``, so every one of them silently took the 298.0 default.
    """
    assert ap._args_temperature_k(_Args(temperature_k=350.0)) == pytest.approx(350.0)
    # Legacy/programmatic namespaces that only carry `temperature` still work.
    assert ap._args_temperature_k(_Args(temperature=277.0)) == pytest.approx(277.0)
    # The real flag wins when both are somehow present.
    assert ap._args_temperature_k(
        _Args(temperature_k=350.0, temperature=277.0)) == pytest.approx(350.0)
    # No temperature anywhere -> the CLI's own default, not 298.
    assert ap._args_temperature_k(_Args()) == pytest.approx(300.0)
    for junk in (None, "", "warm", float("nan"), 0.0, -5.0):
        assert ap._args_temperature_k(_Args(temperature_k=junk)) == pytest.approx(300.0)


def test_no_call_site_reads_the_nonexistent_args_temperature_attribute():
    """Source-level guard: nothing in this module may read ``args.temperature``.

    ``sigma = sqrt(kB*T/k)`` drives ``bridges_needed``, so a wrong T silently
    changes a window-creation decision.  Reverting any of the three call sites
    to ``getattr(args, "temperature", ...)`` must fail here.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(ap))
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            continue
        if len(node.args) < 2:
            continue
        key = node.args[1]
        if isinstance(key, ast.Constant) and key.value == "temperature":
            bad.append(ast.unparse(node))
    assert bad == [], f"reads a temperature attribute the CLI never sets: {bad}"

    # ... and the replacement really is wired in at every site that needs it.
    called = [
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_args_temperature_k"
    ]
    assert len(called) >= 3, f"expected >=3 _args_temperature_k call sites, found {len(called)}"


def test_a_non_300k_run_changes_the_bridge_count_the_prediction_asks_for():
    """A 350 K run must reach the prediction with 350 K, not the 298 K default.

    sigma scales as sqrt(T), so the same geometry needs a different number of
    bridges at a different temperature.  Endpoints chosen so the count actually
    flips: gap 1.0 CV2 units at k2 = 100 kcal/mol gives
    ceil(1.0/(1.5*sigma)) - 1 = 8 bridges at 298 K and 7 at 350 K.
    """
    def _needed_at(temp_k: float) -> int:
        registry = WindowStateRegistry()
        a = registry.add_state(0.0, 200.0, 0.0, 100.0, epoch=0, source="seed")
        b = registry.add_state(0.0, 200.0, 1.0, 100.0, epoch=0, source="seed")
        return int(_bridge_placement_prediction(a, b, temperature_K=temp_k)["bridges_needed"])

    assert _needed_at(298.0) == 8
    assert _needed_at(350.0) == 7

    # Now the same geometry through the real proposer, with the temperature
    # resolved off an args namespace exactly the way the driver does it.
    registry = WindowStateRegistry()
    registry.add_state(0.0, 200.0, 0.0, 100.0, epoch=0, source="seed")
    registry.add_state(0.0, 200.0, 1.0, 100.0, epoch=0, source="seed")
    diagnostics = {
        "states": [
            {"state_id": 0, "sample_count": 100_000},
            {"state_id": 1, "sample_count": 100_000},
        ],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.05}],
    }
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=32)
    args = _Args(temperature_k=350.0)

    adds = [a for a in propose_actions_from_diagnostics(
        registry, diagnostics, policy,
        temperature_K=ap._args_temperature_k(args)) if a[0] == "add"]
    assert len(adds) == 7, "a 350 K run must place the 350 K bridge count, not the 298 K one"


# ---------------------------------------------------------------------------
# Review round 2 -- "bit-for-bit" must actually be bit-for-bit
# ---------------------------------------------------------------------------


def test_exact_midpoint_interpolation_is_bitwise_the_historical_average():
    """``a + 0.5*(b-a)`` is not ``0.5*(a+b)``: ~6.7% of double pairs differ by 1 ULP.

    Physically irrelevant for a restraint centre, but the multi-bridge code
    claims the single-midpoint case is unchanged bit-for-bit, so make that
    claim true instead of asserting it through ``pytest.approx``.
    """
    rng = np.random.default_rng(20260825)
    pairs = rng.normal(scale=50.0, size=(20_000, 2))
    differing = 0
    for a, b in pairs:
        a, b = float(a), float(b)
        assert ap._interpolate_bridge_axis(a, b, 0.5) == 0.5 * (a + b)
        if (a + 0.5 * (b - a)) != 0.5 * (a + b):
            differing += 1
    # Confirms the naive form really does disagree -- i.e. this test has teeth.
    assert differing > 200, f"only {differing} of 20000 pairs differed; check the fixture"

    # Off-midpoint fractions still interpolate linearly.
    assert ap._interpolate_bridge_axis(0.0, 1.0, 0.25) == pytest.approx(0.25)
    assert ap._interpolate_bridge_axis(2.0, 6.0, 0.75) == pytest.approx(5.0)


def test_single_bridge_placement_params_are_exactly_the_historical_average():
    """The whole 4-tuple, asserted with ``==`` rather than ``pytest.approx``.

    Centres chosen so every axis is a pair whose naive interpolation differs
    from the plain average by one ULP; ``pytest.approx`` cannot see that, which
    is why the original version of this test could not fail.
    """
    c1, k1 = 0.01, 100.3
    c2, k2 = 0.03, 300.1
    s1, s2 = -1.63, 1.3
    ks1, ks2 = 20.1, 180.3
    # Fixture check: the naive form really is one ULP off on every axis here.
    for a, b in ((c1, c2), (k1, k2), (s1, s2), (ks1, ks2)):
        assert (a + 0.5 * (b - a)) != 0.5 * (a + b), (a, b)

    registry = WindowStateRegistry()
    registry.add_state(c1, k1, s1, ks1, epoch=0, source="seed")
    registry.add_state(c2, k2, s2, ks2, epoch=0, source="seed")
    diagnostics = {
        "states": [
            {"state_id": 0, "sample_count": 100_000},
            {"state_id": 1, "sample_count": 100_000},
        ],
        "edges": [{"state_i": 0, "state_j": 1, "overlap": 0.05}],
    }
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=4, bridge_multi_window=False)

    adds = [a for a in propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0) if a[0] == "add"]
    assert len(adds) == 1
    params = adds[0][2]
    assert params[0] == 0.5 * (c1 + c2)
    assert params[1] == 0.5 * (k1 + k2)
    assert params[2] == 0.5 * (s1 + s2)
    assert params[3] == 0.5 * (ks1 + ks2)


# ---------------------------------------------------------------------------
# Review round 2 -- one greedy edge must not eat the whole per-epoch budget
# ---------------------------------------------------------------------------


def _five_weak_edges_registry():
    """Six states in three well-separated CV2 pairs -> five weak edges.

    Every pair is far enough apart that the prediction asks for many bridges,
    so under the old greedy loop the worst edge alone consumed the whole
    ``max_new_windows_per_epoch`` budget.
    """
    registry = WindowStateRegistry()
    for i in range(6):
        registry.add_state(0.0, 200.0, float(i), 100.0, epoch=0, source="seed")
    diagnostics = {
        "states": [{"state_id": i, "sample_count": 100_000} for i in range(6)],
        # Deliberately worst-first-out-of-order so the sort is exercised.
        "edges": [
            {"state_i": 2, "state_j": 3, "overlap": 0.09},
            {"state_i": 0, "state_j": 1, "overlap": 0.02},
            {"state_i": 4, "state_j": 5, "overlap": 0.11},
            {"state_i": 1, "state_j": 2, "overlap": 0.05},
            {"state_i": 3, "state_j": 4, "overlap": 0.07},
        ],
    }
    return registry, diagnostics


def test_bridge_budget_is_shared_across_weak_edges_not_eaten_by_the_worst():
    """chignolin_6's own budget: 4 new windows against 5 weak edges.

    Every edge here needs 8 bridges, so the greedy loop gave all 4 to edge 0-1
    and nothing to the other four -- strictly worse than the historical
    behaviour of one midpoint per edge.  Fair-share must restore that:
    the four worst edges get one bridge each.
    """
    registry, diagnostics = _five_weak_edges_registry()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=4)

    adds = [a for a in propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0) if a[0] == "add"]
    assert len(adds) == 4

    # One bridge on each of the four worst edges (0-1, 1-2, 3-4, 2-3), none on
    # the least-bad one (4-5).
    bridged_edges = sorted(a[3].split(":")[0].replace("weak edge ", "") for a in adds)
    assert bridged_edges == ["0-1", "1-2", "2-3", "3-4"]


def test_spare_bridge_budget_goes_to_the_neediest_edges_first():
    """With room for more than one each, the extras follow the worst overlap."""
    registry, diagnostics = _five_weak_edges_registry()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=7)

    adds = [a for a in propose_actions_from_diagnostics(
        registry, diagnostics, policy, temperature_K=300.0) if a[0] == "add"]
    assert len(adds) == 7

    per_edge = {}
    placed_indices = {}
    for action in adds:
        label = action[3].split(":")[0].replace("weak edge ", "")
        per_edge[label] = per_edge.get(label, 0) + 1
        # "; bridge k/N of M needed (" -> (k, N)
        frag = action[3].split("; bridge ", 1)[1].split(" of ", 1)[0]
        k, n = (int(x) for x in frag.split("/"))
        placed_indices.setdefault(label, (set(), n))[0].add(k)
    # Round 1 gives all five edges one; the two spares go to the two worst.
    assert per_edge == {"0-1": 2, "1-2": 2, "2-3": 1, "3-4": 1, "4-5": 1}

    # Assert the counts above are the *allocation*, not an allocation minus
    # some silently skipped placements: registry.has_near_duplicate can drop a
    # bridge mid-loop (its tolerances are 1e-4, far below any spacing here, but
    # the fixture must not depend on that), which would shrink `adds` without
    # shrinking the allocation the reason strings report.
    for label, (indices, n_alloc) in placed_indices.items():
        assert indices == set(range(1, n_alloc + 1)), (label, sorted(indices), n_alloc)
        assert per_edge[label] == n_alloc, (label, per_edge[label], n_alloc)


def test_under_bridged_shortfall_is_the_true_number_not_the_clamped_one():
    """The fifth edge gets nothing, and says so with the right number.

    ``shortfall`` used to be computed from a ``max(1, min(needed, budget_left))``
    clamped count, so a budget-starved edge reported "under-bridged by
    needed - 1" while placing zero bridges.  The message must now distinguish
    "placed some, still short" from "placed none at all".
    """
    registry, diagnostics = _five_weak_edges_registry()
    policy = AdaptiveDecisionPolicy(max_new_windows_per_epoch=4)

    logged = []
    orig = ap.logging.warning
    ap.logging.warning = lambda fmt, *a: logged.append(fmt % a if a else fmt)
    try:
        adds = [a for a in propose_actions_from_diagnostics(
            registry, diagnostics, policy, temperature_K=300.0) if a[0] == "add"]
    finally:
        ap.logging.warning = orig

    # Every edge here needs 8. The four that got one bridge are short by 7.
    for action in adds:
        assert "UNDER-BRIDGED by 7" in action[3], action[3]

    starved = [line for line in logged if "4-5" in line]
    assert starved, f"the starved edge must be reported; got {logged}"
    assert "no bridge windows could be placed" in starved[0], starved[0]
    assert "needs 8" in starved[0], starved[0]
    # And it must NOT claim the off-by-one shortfall the clamp used to produce.
    assert "under-bridged by 7" not in starved[0].lower(), starved[0]


# ---------------------------------------------------------------------------
# Review round 2 -- the new policy knobs must be reachable from the CLI
# ---------------------------------------------------------------------------


def test_new_bridge_knobs_are_reachable_from_the_real_cli():
    """The dropped-knob pattern CLAUDE.md already documents twice.

    ``policy_from_args`` reading ``adaptive_production_bridge_*`` is not enough:
    if no argparse flag ever sets those attributes, an operator cannot express
    the old one-midpoint-per-edge allocation without hand-editing the policy
    dataclass.  Same for ``--ap-final-quality-extension-*``, whose value
    ``_final_extension_steps`` now has a documented contract for.
    """
    from gareus.cli import parse_args

    base = ["--seq", "GYDPETGTWG", "--out", "o"]
    args = parse_args(base)
    # Defaults must leave behaviour exactly as it is today.
    assert args.adaptive_production_bridge_multi_window is True
    assert args.adaptive_production_bridge_healthy_spacing_sigma == pytest.approx(1.5)
    assert args.adaptive_production_final_quality_extension_rounds == 0
    assert args.adaptive_production_final_quality_extension_steps == 0

    tuned = parse_args(base + [
        "--no-ap-bridge-multi-window",
        "--ap-bridge-healthy-spacing-sigma", "1.1",
        "--ap-final-quality-extension-rounds", "2",
        "--ap-final-quality-extension-steps", "250000",
    ])
    assert tuned.adaptive_production_bridge_multi_window is False
    assert tuned.adaptive_production_bridge_healthy_spacing_sigma == pytest.approx(1.1)
    assert tuned.adaptive_production_final_quality_extension_rounds == 2
    assert tuned.adaptive_production_final_quality_extension_steps == 250_000

    # ... and the whole chain flag -> args -> policy -> placement holds.
    policy = policy_from_args(tuned)
    assert policy.bridge_multi_window is False
    assert policy.bridge_healthy_spacing_sigma == pytest.approx(1.1)
    assert _final_extension_steps(policy, fallback_steps=999) == 250_000


# ---------------------------------------------------------------------------
# Review round 2 (handover) -- resume must not clobber a drop-corrected map
# ---------------------------------------------------------------------------


def _fake_production_checkpoint(phase_dir, n_replicas: int = 3) -> None:
    """Make ``production_checkpoint_available(phase_dir)`` true.

    That is the exact predicate ``run_segment``/the driver use to decide whether
    a phase resumes, so it is the one the map-preservation guard has to agree
    with.  Equal file sizes clear its median-outlier torn-write check.
    """
    import json

    chk = phase_dir / "checkpoints"
    chk.mkdir(parents=True, exist_ok=True)
    names = [f"replica_{i:03d}.chk" for i in range(n_replicas)]
    for name in names:
        (chk / name).write_bytes(b"x" * 4096)
    (chk / "production_checkpoint_manifest.json").write_text(
        json.dumps({"replica_checkpoint_files": names, "prod_done": 1_000_000}),
        encoding="utf-8",
    )


# The map final/baseline really carried after --us-auto-drop-bad-windows pruned
# local windows 1 and 3: survivors renumbered 0..2, each keeping its own
# state_id and its own restraint centers.
_CORRECTED_MAP_CSV = (
    "epoch_window,state_id,primary_center,primary_k,secondary_center,secondary_k\n"
    "0,0,0.0,200.0,-1.4379,148.46\n"
    "1,2,0.0654,200.0,0.1264,113.09\n"
    "2,4,0.1141,200.0,-1.6282,89.88\n"
)


def _five_state_registry() -> WindowStateRegistry:
    registry = WindowStateRegistry()
    registry.add_state(0.0, 200.0, -1.4379, 148.46, epoch=0, source="seed")
    registry.add_state(0.02, 200.0, -1.0, 148.46, epoch=0, source="seed")
    registry.add_state(0.0654, 200.0, 0.1264, 113.09, epoch=0, source="seed")
    registry.add_state(0.09, 200.0, -0.5, 120.0, epoch=0, source="seed")
    registry.add_state(0.1141, 200.0, -1.6282, 89.88, epoch=0, source="seed")
    return registry


def test_resuming_phase_keeps_its_drop_corrected_epoch_window_map(tmp_path):
    """SIGTERM mid-phase, then resume: the corrected map must survive.

    ``--us-auto-drop-bad-windows`` prunes windows post-pull and the drop rewrites
    this phase's ``epoch_window_map.csv`` to match the survivors.  The drop only
    runs on a fresh start, so on resume nothing recreates that correction -- and
    the driver used to unconditionally overwrite the map with a fresh identity
    map off the (still un-pruned) registry before the phase re-ran, putting the
    run straight back into the chignolin_6 mis-attribution.
    """
    seg_dir = tmp_path / "final" / "baseline"
    seg_dir.mkdir(parents=True)
    map_path = seg_dir / "epoch_window_map.csv"
    map_path.write_text(_CORRECTED_MAP_CSV, encoding="utf-8")
    _fake_production_checkpoint(seg_dir)
    before = map_path.read_bytes()

    registry = _five_state_registry()
    written = ap._write_phase_window_map(registry, map_path, resume_requested=True)

    assert written is None, "a resuming phase's own map must be left alone"
    assert map_path.read_bytes() == before


def test_a_fresh_phase_still_gets_its_map_written(tmp_path):
    """The negative case: preserving here would be the *new* bug.

    No resume requested means run_gareus starts fresh, re-pulls, and may drop a
    different window set -- the map on disk is then not this run's map at all.
    """
    seg_dir = tmp_path / "final" / "baseline"
    seg_dir.mkdir(parents=True)
    map_path = seg_dir / "epoch_window_map.csv"
    map_path.write_text(_CORRECTED_MAP_CSV, encoding="utf-8")
    _fake_production_checkpoint(seg_dir)

    registry = _five_state_registry()
    written = ap._write_phase_window_map(registry, map_path, resume_requested=False)

    assert written == map_path
    rows = [r for r in map_path.read_text(encoding="utf-8").splitlines() if r.strip()]
    assert len(rows) == 6, "header + 5 registry states"


def test_no_checkpoint_means_nothing_was_sampled_against_the_old_map(tmp_path):
    """A map with no production behind it is a stale plan, not a record."""
    seg_dir = tmp_path / "final" / "baseline"
    seg_dir.mkdir(parents=True)
    map_path = seg_dir / "epoch_window_map.csv"
    map_path.write_text(_CORRECTED_MAP_CSV, encoding="utf-8")

    registry = _five_state_registry()
    assert ap._write_phase_window_map(registry, map_path, resume_requested=True) == map_path
    rows = [r for r in map_path.read_text(encoding="utf-8").splitlines() if r.strip()]
    assert len(rows) == 6


def test_a_resuming_phase_with_no_map_yet_still_gets_one(tmp_path):
    """Epoch 0's bootstrap writes the first map *after* the phase ran.

    The guard must key off an existing file, never off "this phase resumed" --
    otherwise the ``registry is None`` bootstrap site leaves the phase with no
    map at all and the MBAR loader falls back to the registry.
    """
    seg_dir = tmp_path / "epoch_000"
    seg_dir.mkdir(parents=True)
    _fake_production_checkpoint(seg_dir)
    map_path = seg_dir / "epoch_window_map.csv"

    registry = _five_state_registry()
    assert ap._write_phase_window_map(registry, map_path, resume_requested=True) == map_path
    assert map_path.exists()


def test_subset_window_csv_still_written_when_the_map_is_preserved(tmp_path):
    """run_segment's shape: the windows table is written, only the map is skipped.

    The windows CSV is inert on a fast resume (``load_resume_run_definition``
    rebuilds the window set from the phase's own recorded post-drop tables), so
    it stays a faithful record of what the driver asked for.
    """
    seg_dir = tmp_path / "epoch_001" / "topup_001_12000"
    seg_dir.mkdir(parents=True)
    map_path = seg_dir / "epoch_window_map.csv"
    map_path.write_text(_CORRECTED_MAP_CSV, encoding="utf-8")
    _fake_production_checkpoint(seg_dir)
    before = map_path.read_bytes()

    registry = _five_state_registry()
    windows_csv = tmp_path / "epoch_001" / "topup_001_12000_windows.csv"
    preserve = ap._phase_window_map_is_owned_by_a_resuming_phase(
        seg_dir, map_path, resume_requested=True, expected_rows=5)
    assert preserve is True
    ap.write_state_subset_window_csv(
        registry, windows_csv, [0, 1, 2, 3, 4],
        map_path=None if preserve else map_path)

    assert windows_csv.exists()
    assert map_path.read_bytes() == before


def test_every_driver_map_write_goes_through_the_resume_guard():
    """Structural guard on all five write sites.

    Getting this wrong is silent and only shows up as a wrong PMF weeks later,
    so the funnel is enforced at source level: ``write_epoch_window_map`` may be
    called from exactly two places (its own ``write_active_window_csv`` sibling
    and ``_write_phase_window_map``), and any driver call that hands a writer a
    ``map_path`` must hand it a guarded conditional rather than a bare path.
    """
    import ast
    import inspect

    src = inspect.getsource(ap)
    tree = ast.parse(src)

    guarded_scopes = {"write_active_window_csv", "_write_phase_window_map"}
    unguarded = []
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(scope):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "write_epoch_window_map"
                    and scope.name not in guarded_scopes):
                unguarded.append(f"{scope.name}: {ast.unparse(node)}")
    assert unguarded == [], f"map writes bypassing the resume guard: {unguarded}"

    # Every `map_path=` naming a *phase's* epoch_window_map.csv must be a
    # conditional gated on the guard, never a bare path.  (The campaign-level
    # window_map_epoch_NNN.csv planning artifacts under adaptive_dir are a
    # different file in a directory that never holds production output, so they
    # are legitimately unguarded.)
    # Locals assigned a phase map path count as phase map paths -- otherwise a
    # site that hoists the path into a variable first slips through the literal
    # check (verified: it did, before this clause was added).
    map_locals = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and "epoch_window_map.csv" in ast.unparse(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    map_locals.add(target.id)
    assert map_locals, "expected at least one hoisted epoch_window_map.csv path local"

    bare = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "map_path":
                continue
            rendered = ast.unparse(kw.value)
            names = {n.id for n in ast.walk(kw.value) if isinstance(n, ast.Name)}
            if "epoch_window_map.csv" not in rendered and not (names & map_locals):
                continue
            if (isinstance(kw.value, ast.IfExp)
                    and ast.unparse(kw.value.test).endswith("_map_preserved")):
                continue
            bare.append(f"{rendered} in {ast.unparse(node)}")
    assert bare == [], f"unguarded phase map_path= writes: {bare}"
    assert len(
        [n for n in ast.walk(tree)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
         and n.func.id == "_phase_window_map_is_owned_by_a_resuming_phase"]
    ) >= 3, "expected the guard to be consulted directly at the two csv-writer sites"


# ---------------------------------------------------------------------------
# Review round 3 -- the flat (non-scheduled) epoch's own map was never refreshed
#
# The round-2 no-clobber guard above is a write-side *veto*; it presupposes that
# something writes the phase's map when the phase is NOT resuming.  For the
# scheduled path (run_segment), the frozen final phase and each extension round
# that is true -- each writes its own map immediately before the phase runs.  A
# flat (non-scheduled) numbered epoch >= 1 was the exception: its map is written
# by the *previous* iteration's next_epoch_dir site, a whole epoch earlier, and
# nothing touched it again at loop entry.
#
# That leaves the map on disk describing the wrong window set in exactly one
# reachable combination: attempt 1 at this epoch pulled, dropped windows and had
# production compact the map, then died before a usable production checkpoint
# existed.  Attempt 2 re-enters the epoch with a compacted (3-row) map on disk
# against the un-pruned 5-window table it is about to re-pull from, and neither
# write-side correction can repair that -- the drop-time rewrite refuses with
# skipped_row_count_mismatch, repair_epoch_window_map_from_surviving_windows with
# skipped_map_shorter_than_window_set.  Both refusals are correct: neither can
# distinguish "attempt 1's compaction, now stale" from "the right map for a
# different drop set".  The driver has to stop producing the ambiguity.
#
# These tests drive the real loop (run_adaptive_production_auto_loop) with a fake
# run_gareus, so what is under test is the wiring at the flat-epoch call site,
# not a helper called directly.
# ---------------------------------------------------------------------------


class _StopBeforeMD(Exception):
    """Raised by the fake run_gareus to end the drive at the point of interest."""


def _resumed_flat_epoch_campaign(tmp_path, *, with_checkpoint: bool):
    """A campaign whose flat epoch 1 already ran once, dropped windows, and died.

    Writes the on-disk state the driver would actually find: a 5-state registry,
    the un-pruned ``windows_epoch_001.csv`` epoch 1 re-pulls from, a driver
    summary saying epoch 0 completed (so the loop resumes at epoch 1), and
    ``epoch_001/`` holding attempt 1's *compacted* 3-row map plus the rewrite
    ledger entry that compaction recorded.

    ``with_checkpoint`` decides which half of the guard is exercised: a usable
    production checkpoint means attempt 2 fast-resumes and the corrected map must
    survive; none means attempt 2 re-pulls from scratch and the map must be
    refreshed to the pre-drop window set the drop indices are expressed in.
    """
    import json

    from gareus.cli import parse_args

    out = tmp_path / "run"
    adaptive = out / "adaptive_production"
    adaptive.mkdir(parents=True)

    registry = _five_state_registry()
    registry.save(adaptive)
    registry.write_active_window_csv(adaptive / "windows_epoch_001.csv")
    (adaptive / "adaptive_production_driver_summary.json").write_text(
        json.dumps({
            "schema_version": "adaptive_production_driver_summary_v1",
            "epochs_completed": 1,
            "epoch_summaries": [{"epoch": 0}],
        }),
        encoding="utf-8",
    )

    epoch_dir = adaptive / "epoch_001"
    epoch_dir.mkdir()
    (epoch_dir / "epoch_window_map.csv").write_text(_CORRECTED_MAP_CSV, encoding="utf-8")
    # Exactly what production._append_epoch_window_map_rewrite_ledger recorded
    # when attempt 1's post-pull drop compacted the map above.
    (epoch_dir / "epoch_window_map_rewrites.json").write_text(
        json.dumps({
            "schema": "gareus_epoch_window_map_rewrite_ledger_v1",
            "applied": [{
                "status": "rewritten",
                "source": "post_pull_auto_drop",
                "n_windows_before": 5,
                "n_windows_after": 3,
                "dropped_window_indices": [1, 3],
                "dropped_state_ids": [1, 3],
                "surviving_state_ids": [0, 2, 4],
            }],
        }),
        encoding="utf-8",
    )
    if with_checkpoint:
        _fake_production_checkpoint(epoch_dir)

    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(out)])
    args.adaptive_production_resume = True
    # Flat epoch: the scheduled (baseline + topup) path has its own per-segment
    # map write and is not what is under test here.
    args.adaptive_production_allocation_scheduler = False
    args.adaptive_production_epochs = 2
    args.adaptive_production_epoch_steps = 1000
    args.adaptive_production_global_shared_gamd = False
    return args, out, epoch_dir


def _drive_until_run_gareus(monkeypatch, args, out):
    """Run the real driver loop up to the flat epoch's ``run_gareus`` call.

    Returns the map rows as they stood on disk at that instant -- i.e. what the
    phase is about to sample against.  ``run_gareus`` is resolved inside
    ``run_adaptive_production_auto_loop`` (``from .production import
    run_gareus``), so patching the module attribute is enough.
    """
    import csv as _csv
    from pathlib import Path

    import gareus.production as _prod

    captured = {}

    def _fake_run_gareus(_args, run_dir, *rest, **kw):
        with (Path(run_dir) / "epoch_window_map.csv").open(newline="") as handle:
            captured["rows"] = [dict(r) for r in _csv.DictReader(handle)]
        captured["dir"] = Path(run_dir)
        raise _StopBeforeMD()

    monkeypatch.setattr(_prod, "run_gareus", _fake_run_gareus)
    with pytest.raises(_StopBeforeMD):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert captured["dir"].name == "epoch_001", "the drive must reach the flat epoch, not another phase"
    return captured["rows"]


def test_a_re_pulling_flat_epoch_gets_the_pre_drop_map_it_is_about_to_re_pull(tmp_path, monkeypatch):
    """No usable checkpoint: attempt 2 re-pulls all 5 windows, so the map must
    describe all 5 windows again.

    Leaving attempt 1's 3-row compaction in place makes every later correction
    refuse (the drop indices are positions in the 5-window set; a 3-row map
    cannot say which row is which local window), and the epoch samples against a
    map that covers fewer windows than it runs.
    """
    args, out, epoch_dir = _resumed_flat_epoch_campaign(tmp_path, with_checkpoint=False)

    rows = _drive_until_run_gareus(monkeypatch, args, out)

    assert [int(r["state_id"]) for r in rows] == [0, 1, 2, 3, 4]
    assert [int(r["epoch_window"]) for r in rows] == [0, 1, 2, 3, 4]
    # The registry's own centers, not the compacted map's: this is a fresh write.
    assert float(rows[1]["secondary_center"]) == pytest.approx(-1.0)


def test_a_fast_resuming_flat_epoch_still_keeps_its_drop_corrected_map(tmp_path, monkeypatch):
    """The other half of the guard, at the same new call site.

    A usable production checkpoint means attempt 2 fast-resumes: it rebuilds its
    window set from this phase's own recorded post-drop tables and never re-pulls,
    so the 3-row corrected map is the record of what was sampled and refreshing
    it would be the chignolin_6 mis-attribution all over again.  (This one passes
    with the refresh removed as well -- it is the safety half of the same change,
    proving the new write cannot clobber.)
    """
    args, out, epoch_dir = _resumed_flat_epoch_campaign(tmp_path, with_checkpoint=True)

    rows = _drive_until_run_gareus(monkeypatch, args, out)

    assert [int(r["state_id"]) for r in rows] == [0, 2, 4]
    assert [int(r["epoch_window"]) for r in rows] == [0, 1, 2]


def test_the_flat_epoch_branch_refreshes_its_map_before_calling_run_gareus():
    """Source-level guard, so a future reordering cannot silently undo the fix.

    The refresh only helps if it happens *before* the phase runs: written after
    ``run_gareus`` returns it would overwrite the very correction the drop just
    made.  Cheap insurance next to the loop-drive tests above, which need the
    whole campaign fixture to stay valid to keep testing anything.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(ap.run_adaptive_production_auto_loop))
    branches = [
        node.orelse for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "run_gareus"
                for n in ast.walk(node))
        and node.orelse
    ]
    flat = [body for body in branches
            if any("run_gareus(epoch_args" in ast.unparse(stmt) for stmt in body)]
    assert len(flat) == 1, "expected exactly one flat-epoch else-branch calling run_gareus(epoch_args, ...)"
    body = flat[0]

    def _index(predicate):
        return next(i for i, stmt in enumerate(body) if predicate(ast.unparse(stmt)))

    refresh_at = _index(lambda s: "_write_phase_window_map" in s and "epoch_window_map.csv" in s)
    run_at = _index(lambda s: s.startswith("run_gareus(epoch_args"))
    assert refresh_at < run_at, "the flat epoch must refresh its own map before it runs"
    refresh = ast.unparse(body[refresh_at])
    assert "epoch_dir" in refresh, "the refresh must target THIS epoch's directory"
    assert "resume_requested" in refresh, "the refresh must go through the resume no-clobber guard"


def test_every_driver_phase_sets_args_resume_to_the_guards_own_predicate():
    """The invariant ``_phase_window_map_is_owned_by_a_resuming_phase``'s docstring
    now asserts, pinned in code.

    That docstring used to claim the guard and ``run_gareus``'s ``fast_resume``
    can diverge for a driver-invoked phase (a torn checkpoint giving
    ``args.resume=True`` with ``production_checkpoint_available() == False``) and
    told future maintainers to preserve the guard's shape because of it.  They
    cannot diverge: ``run_gareus`` reads the checkpoint manifest only when
    ``args.resume`` is set, and every driver site sets ``args.resume`` to
    ``resume_requested and production_checkpoint_available(<phase dir>)`` --
    strictly the stronger reading of the same manifest.  So guard-preserves =>
    args.resume => fast_resume, by construction.  This test is what makes that
    claim checkable: break the chain at any phase site and it fails here, where
    the docstring can only go quietly out of date.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(ap))

    # Locals whose value is itself the conjunction (run_segment hoists it into
    # _seg_will_resume so the guard and the worker cannot be given two different
    # answers).
    conjunction_locals = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        rendered = ast.unparse(node.value)
        if "resume_requested" in rendered and "production_checkpoint_available" in rendered:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    conjunction_locals.add(target.id)

    # Per-phase worker namespaces only (seg_args, epoch_args, final_args,
    # ext_args).  The one bare `args.resume = True` in the module belongs to
    # `--extend --extend-mode regular`, which resumes a plain non-adaptive run
    # through the CLI's own --resume branch -- that is precisely the guard-free
    # standalone path the docstring names, so it must stay out of this check
    # while still being accounted for.
    sites = []
    bare = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Attribute) and target.attr == "resume"):
                continue
            owner = ast.unparse(target.value)
            if owner.endswith("_args") and owner != "args":
                sites.append((ast.unparse(target), ast.unparse(node.value)))
            else:
                bare.append(f"{ast.unparse(target)} = {ast.unparse(node.value)}")

    assert bare == ["args.resume = True"], (
        "a new non-phase args.resume assignment appeared; decide whether it is a "
        f"driver phase (then give it the guard's predicate) or a standalone path: {bare}"
    )
    assert len(sites) >= 4, f"expected the four driver phase sites, found {sites}"
    bad = [
        f"{lhs} = {rhs}" for lhs, rhs in sites
        if not (("resume_requested" in rhs and "production_checkpoint_available" in rhs)
                or rhs in conjunction_locals)
    ]
    assert bad == [], f"driver phases whose args.resume is not the guard's predicate: {bad}"
