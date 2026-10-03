"""CV2 bridge sets (spec docs/superpowers/specs/2026-10-03-cv2-bridge-sets-design.md).

chignolin_10 epoch_000 (2026-10-03): a cap-bound shape layout kept only the two well windows of
every CV1 column; the barrier between them (2.6-5.7 kT) needs 4-7 narrow fills to connect, one
midpoint window overlaps its neighbours at ~0. Bridges are therefore granted whole or not at all,
one CV1-free barrier chain connects the wells campaign-wide, and R1 adds whole sets.
"""
import copy
import json
import math

import numpy as np
import pytest

from gareus.adaptive.cv2_shape import (MixtureComponent, bridge_fills, fit_cv2_mixture, fit_from_record,
                                       place_cv2_centres, predicted_overlaps)
from gareus.swarm.ladder_design import R_KCAL_MOL_K

T = 300.0
RT = R_KCAL_MOL_K * T
WELLS = ((-0.8, 0.15), (1.0, 0.15))


def _members(rng, specs):
    z, ids, m = [], [], 0
    for n_members, n_frames, mean, sd in specs:
        for _ in range(n_members):
            z.append(rng.normal(mean, sd, n_frames)); ids.append(np.full(n_frames, m)); m += 1
    return np.concatenate(z), np.concatenate(ids)


def _double_well_fit(seed=3, wells=WELLS, n=30):
    rng = np.random.default_rng(seed)
    z, ids = _members(rng, [(n, 80, mu, sd) for mu, sd in wells])
    return fit_cv2_mixture(z, ids, np.ones_like(z), max_components=3, min_mode_members=8), z


def _placement(fit, sigma_t=0.78, env=(-1.7, 1.8)):
    return place_cv2_centres(fit, env, sigma_w_target=sigma_t, temperature_k=T, k_min=1e-3, k_max=1000.0)


# ---- helpers ------------------------------------------------------------------------------

def test_predicted_overlap_matches_independent_gaussian_integral():
    s = 0.5
    fit_one = fit_from_record({"n_components": 1, "components": [
        {"mean": 0.0, "variance": s * s, "weight": 1.0, "n_members": 50, "accepted": True, "reason": "accepted"}],
        "bic": {}, "regularisation": 0.0, "cv_scores": {}, "cv_folds": 0, "n_frames": 100, "n_members": 50,
        "weight_total": 100.0, "pooled_mean": 0.0, "pooled_variance": s * s, "min_mode_members": 8, "reasons": []})
    centres, ks = [-0.3, 0.4], [6.0, 9.0]
    (ov,) = predicted_overlaps(fit_one, centres, ks, T)
    # independent: F = RT z^2/(2 s^2) plus the spring -> Gaussian with K = RT/s^2 + k, mean k c / K
    x = np.linspace(-8, 8, 200001)
    dens = []
    for c, k in zip(centres, ks):
        K = RT / s ** 2 + k
        m, v = k * c / K, RT / K
        dens.append(np.exp(-0.5 * (x - m) ** 2 / v) / math.sqrt(2 * math.pi * v))
    ref = float(np.sum(dens[0] * dens[1] / (dens[0] + dens[1] + 1e-300)) * (x[1] - x[0]))
    assert ov == pytest.approx(ref, abs=1e-3)


def test_designed_barrier_fills_connect_but_a_single_midpoint_does_not():
    fit, _ = _double_well_fit()
    pl = _placement(fit)
    kinds = list(pl.kinds)
    i_lo, i_hi = [i for i, k in enumerate(kinds) if k == "mode"][:2]
    assert i_hi - i_lo >= 3, "the double well must need several fills"
    ov = predicted_overlaps(fit, pl.centres[i_lo:i_hi + 1], pl.k2[i_lo:i_hi + 1], T)
    assert min(ov) >= 0.25
    mid = 0.5 * (pl.centres[i_lo] + pl.centres[i_hi])
    k_mid = pl.k2[(i_lo + i_hi) // 2]
    single = predicted_overlaps(fit, [pl.centres[i_lo], mid, pl.centres[i_hi]], [pl.k2[i_lo], k_mid, pl.k2[i_hi]], T)
    assert max(single) < 0.1


def test_fit_record_round_trip():
    fit, _ = _double_well_fit()
    back = fit_from_record(json.loads(json.dumps(fit.as_record())))
    assert back.as_record() == fit.as_record()


def test_bridge_fills_equal_the_placement_interior_fills():
    fit, _ = _double_well_fit()
    pl = _placement(fit)
    i_lo, i_hi = [i for i, k in enumerate(pl.kinds) if k == "mode"][:2]
    fills = bridge_fills(fit, pl.centres[i_lo], pl.centres[i_hi], sigma_w_target=0.78, temperature_k=T,
                         k_min=1e-3, k_max=1000.0, spacing_sigma=1.5)
    assert [round(z, 9) for z in fills["centres"]] == [round(z, 9) for z in pl.centres[i_lo + 1:i_hi]]
    assert [round(k, 9) for k in fills["k2"]] == [round(k, 9) for k in pl.k2[i_lo + 1:i_hi]]


# ---- layout ------------------------------------------------------------------------------

def _swarm(columns, seed=11):
    """columns: [(c1, wells)] -> (cv1, z2, member ids); every member stays in one well."""
    rng = np.random.default_rng(seed)
    parts, m = [], 0
    for c1, wells in columns:
        for mu, sd in wells:
            for _ in range(24):
                parts.append((rng.normal(c1, 0.01, 60), rng.normal(mu, sd, 60), np.full(60, m))); m += 1
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


COLUMNS = [(0.2, WELLS), (0.6, ((-0.5, 0.15), (0.6, 0.15))), (1.0, WELLS)]


def _layout(max_replicas, bridge_sets, columns=COLUMNS, reserve_fraction=0.0):
    from gareus.swarm.cv2_shape_layout import build_shape_pair_layout
    cv1, z2, ids = _swarm(columns)
    lo, hi = np.quantile(z2, [0.02, 0.98])
    uniform = np.linspace(lo, hi, 4)
    sigma_t = (hi - lo) / 3 / 1.5
    kw = {} if bridge_sets is None else {"bridge_sets": bridge_sets}
    return build_shape_pair_layout(
        cv1=cv1, z2=z2, member_ids=ids, deltav_kj=np.zeros_like(z2), centers1=[c for c, _ in columns],
        ks1=[800.0] * len(columns), lambdas=[0.0, 1.0], temperature_k=T, uniform_centres2=uniform,
        uniform_ks2=[RT / sigma_t ** 2] * 4, overlap_sigma=1.5, k_min=1e-3, k_max=1000.0,
        max_replicas=max_replicas, region_centre_indices=[0], reserve_fraction=reserve_fraction, **kw)


def _sets(out):
    return out["record"]["bridge_sets"]


def test_flag_off_is_byte_identical_to_today():
    a, b = _layout(2 * 30, None), _layout(2 * 30, False)
    assert json.dumps(a["layout"], sort_keys=True, default=str) == json.dumps(b["layout"], sort_keys=True, default=str)
    assert "bridge_sets" not in a["record"] and "barrier_chain" not in a["record"]


@pytest.mark.parametrize("max_replicas", [2 * 14, 2 * 22, 2 * 30, 2 * 45])
def test_bridge_sets_are_granted_whole_or_not_at_all(max_replicas):
    out = _layout(max_replicas, True)
    rec = out["record"]
    by_pos = {(q["column"], q["position"]): q for q in rec["requests"] if q["kind"] in ("mode", "fill")}
    for bs in _sets(out):
        flags = {by_pos[(bs["column"], p)]["granted"] for p in bs["fill_positions"]}
        assert len(flags) == 1, f"partial bridge set {bs}"
        assert flags == {bs["granted"]}
        if not bs["granted"]:
            assert all(by_pos[(bs["column"], p)]["reason"] in ("bridge_set_does_not_fit", "cap")
                       for p in bs["fill_positions"])
    chain = rec["barrier_chain"]
    chain_reqs = [q for q in rec["requests"] if q["kind"] == "barrier_axis"]
    assert len({q["granted"] for q in chain_reqs}) <= 1
    if chain_reqs and chain_reqs[0]["granted"]:
        assert chain["granted"] and all(q["cell"][0] is None for q in chain_reqs)
    assert out["layout"]["spatial_states"] * 2 <= max_replicas


def test_tiers_order_chain_then_connectivity_then_sets_by_score_then_tails():
    out = _layout(2 * 40, True)
    rec = out["record"]
    granted = sorted((q for q in rec["requests"] if q["granted"]), key=lambda q: q["rank"])
    tiers = [q["tier"] for q in granted]
    order = ["mode", "barrier_chain", "connectivity", "bridge_set", "fill"]
    seen = [t for i, t in enumerate(tiers) if i == 0 or tiers[i - 1] != t]
    assert [order.index(t) for t in seen] == sorted(order.index(t) for t in seen), seen
    gsets = [bs for bs in _sets(out) if bs["granted"]]
    assert [bs["score"] for bs in gsets] == sorted((bs["score"] for bs in gsets), reverse=True)
    tails = [q for q in granted if q["tier"] == "fill"]
    assert all(q.get("bridge_set") is None for q in tails)


def test_a_set_that_does_not_fit_is_skipped_for_a_smaller_one():
    from gareus.swarm.cv2_shape_layout import _grant_bridge_tiers
    fill = [{"kind": "fill", "column": c, "position": p, "cell": (c, 10 * c + p), "bridge_set": i}
            for i, (c, ps) in enumerate(((0, range(1, 6)), (1, range(1, 3))))
            for p in ps]
    sets = [{"index": 0, "column": 0, "lower_position": 0, "fill_positions": [1, 2, 3, 4, 5], "n_fills": 5, "score": 1.0},
            {"index": 1, "column": 1, "lower_position": 0, "fill_positions": [1, 2], "n_fills": 2, "score": 0.5}]
    cells, roles = [(None, None)], ["unrestrained_anchor"]
    _grant_bridge_tiers([], [], fill, [], sets, 1 + 3, cells, roles)
    assert sets[0]["reason"] == "bridge_set_does_not_fit" and not sets[0]["granted"]
    assert sets[1]["granted"] and len(cells) == 3
    assert all(q.get("granted") is False for q in fill if q["bridge_set"] == 0)
    assert all(q["granted"] and q["tier"] == "bridge_set" for q in fill if q["bridge_set"] == 1)


def test_bridge_set_records_carry_predicted_overlap_and_internal_barriers():
    out = _layout(2 * 200, True)
    for bs in _sets(out):
        assert 0.0 <= bs["predicted_min_overlap"] <= 0.5
        assert bs["n_fills"] == len(bs["fill_positions"]) == len(bs["internal_barrier_kT"]) >= 1
        assert bs["max_internal_barrier_kT"] == max(bs["internal_barrier_kT"])
    rec = out["record"]
    assert "UPPER bound" in rec["predicted_overlap_note"]
    chain = rec["barrier_chain"]
    assert chain["min_gain"] > 0 and "cv1_free_bottleneck_without" in chain
    if chain["centres"]:
        assert chain["gain"] >= chain["min_gain"] and chain["predicted_min_overlap"] >= 0.15
    else:
        assert chain["not_proposed"] in ("no_barrier", "all_duplicates", "no_gain")
        assert not any(q["kind"] == "barrier_axis" for q in rec["requests"])

def test_joint_branch_grants_everything_including_the_chain():
    out = _layout(2 * 400, True)
    rec = out["record"]
    assert out["layout"]["kind"] == "joint"
    assert all(q["granted"] for q in rec["requests"])
    assert rec["barrier_chain"]["granted"] == bool(rec["barrier_chain"]["centres"])

def _interior_without_near_duplicates(pl, i_lo, i_hi):
    lo, hi = pl.centres[i_lo], pl.centres[i_hi]
    return [q for q in range(i_lo + 1, i_hi)
            if abs(pl.centres[q] - lo) >= 0.5 * pl.sampled_sigma[q] and abs(pl.centres[q] - hi) >= 0.5 * pl.sampled_sigma[q]]


def test_r1_bridge_set_children_and_cost():
    from gareus.adaptive import cv2_resolution as cr
    from gareus.adaptive.cv2_resolution_rules import bridge_set_children
    fit, _ = _double_well_fit()
    pl = _placement(fit)
    i_lo, i_hi = [i for i, k in enumerate(pl.kinds) if k == "mode"][:2]
    keep = _interior_without_near_duplicates(pl, i_lo, i_hi)
    column = {"fit": fit, "sigma_w_target": 0.78, "spacing_sigma": 1.5, "k_min": 1e-3, "k_max": 1000.0,
              "min_mean_compression": 0.5}
    a = cr.StateView(1, 0.2, 10.0, pl.centres[i_lo], pl.k2[i_lo], 0.0)
    b = cr.StateView(2, 0.2, 10.0, pl.centres[i_hi], pl.k2[i_hi], 0.0)
    settings = cr.ResolutionSettings(temperature_k=T, cv2_bridge_sets=True)
    out = bridge_set_children(a, b, column, settings)
    assert out["bridge_mode"] == "set" and len(out["children"]) == len(keep)
    assert [round(c["secondary_center"], 9) for c in out["children"]] == [round(pl.centres[q], 9) for q in keep]
    assert out["predicted_min_overlap"] >= 0.25
    # each child seeds from its nearer endpoint
    for c, src in zip(out["children"], out["seed_sources"]):
        near = 1 if abs(c["secondary_center"] - a.c2) <= abs(c["secondary_center"] - b.c2) else 2
        assert src == near == c["seed_source_state_id"]
    assert {1, 2} <= set(out["seed_sources"])
    cand = cr.new_candidate("R1", "edge", [1, 2], "proposed", "weak", cls="weak",
                            proposal={"parent_state_id": 1, "children": out["children"],
                                      "seed_sources": out["seed_sources"]})
    act = cr.candidate_action(cand, 0)
    assert act[0] == "insert" and len(act[2]) == len(out["children"])
    assert act[4][cr.METADATA_KEY]["seed_source_by_child"] == out["seed_sources"]
    funded, rec = cr.allocate([cand], None, 3, ignore_budget=True)
    assert funded[0]["cost_states"] == 3 * len(out["children"])

def test_r1_budget_never_funds_part_of_a_set():
    from gareus.adaptive import cv2_resolution as cr

    class _Allow:
        governed, reason, resolution_slots = True, "ok", 7
        free_slots = add_rung_slots = n_rungs = add_rung_centres = resolution_centres = 0
    kids = [{"primary_center": 0.2, "primary_k": 10.0, "secondary_center": z, "k2": 5.0} for z in (0.0, 0.3, 0.6)]
    cand = cr.new_candidate("R1", "edge", [1, 2], "proposed", "weak", cls="weak",
                            proposal={"parent_state_id": 1, "children": kids})
    (out,), _rec = cr.allocate([cand], _Allow(), 3)
    assert out["decision"] == "refused" and out["refusal"] == "resolution_budget"     # 9 > 7: none of it


def test_r1_without_layout_falls_back_to_today_midpoint():
    from gareus.adaptive import cv2_resolution as cr
    from gareus.adaptive.cv2_resolution_rules import bridge_set_children
    a = cr.StateView(1, 0.2, 10.0, -0.8, 6.0, 0.0, mean2=-0.8, var2=0.02)
    b = cr.StateView(2, 0.2, 10.0, 1.0, 6.0, 0.0, mean2=1.0, var2=0.02)
    settings = cr.ResolutionSettings(temperature_k=T, cv2_bridge_sets=True)
    out = bridge_set_children(a, b, None, settings)
    assert out["bridge_mode"] == "midpoint_fallback"
    assert out["children"] == [cr.bridge_child(a, b, settings)]


def test_r1_flag_default_off():
    from gareus.adaptive import cv2_resolution as cr
    assert cr.ResolutionSettings(temperature_k=T).cv2_bridge_sets is False
    assert cr.DEFAULTS["cv2_bridge_sets"] is False


# ---- R1 end to end: proposal -> insert -> applier --------------------------------------

def test_r1_bridge_set_end_to_end_through_the_applier():
    import gareus.adaptive_production as ap
    from gareus.adaptive import cv2_resolution as cr
    from gareus.adaptive import cv2_resolution_io as cio
    fit, _ = _double_well_fit()
    pl = _placement(fit)
    i_lo, i_hi = [i for i, k in enumerate(pl.kinds) if k == "mode"][:2]
    keep = _interior_without_near_duplicates(pl, i_lo, i_hi)
    za, zb, ka, kb = pl.centres[i_lo], pl.centres[i_hi], pl.k2[i_lo], pl.k2[i_hi]
    reg = ap.WindowStateRegistry()
    reg.add_state(0.2, 800.0, secondary_center=zb, secondary_k=kb, gamd_lambda=0.0, epoch=0, source="seed")
    reg.add_state(0.2, 800.0, secondary_center=za, secondary_k=ka, gamd_lambda=0.0, epoch=0, source="seed")
    upper, lower = (s.state_id for s in reg.active_states())      # edge listed upper-first on purpose
    pm = {"status": "ok", "overlap": 0.0, "overlap_lower": 0.0, "overlap_upper": 0.01, "pattern_pair": "same",
          "n_eff": [500, 500]}
    payload = {"states": [{"state_id": sid, "sample_count": 5000, "paired_cv": {"n_pairs": 2000}} for sid in (upper, lower)],
               "edges": [{"state_i": upper, "state_j": lower, "edge_type": "secondary_chain", "normalized_distance": 1.0,
                          "overlap": 0.8, "mbar_overlap": None, "pairwise_mbar": pm, "warnings": []}],
               "edge_metric": {"status": "ok", "metric": "pairwise-mbar", "stage": "pre_union",
                               "components": {"components": [[upper, lower]]}}}
    columns = [{"centre1": 0.2, "fit": fit, "sigma_w_target": 0.78, "spacing_sigma": 1.5, "k_min": 1e-3,
                "k_max": 1000.0, "min_mean_compression": 0.5}]
    settings = cr.ResolutionSettings(temperature_k=T, cv2_bridge_sets=True)
    policy = ap.AdaptiveDecisionPolicy(cv2_resolution=True, edge_metric="pairwise-mbar", max_replicas_budget=80,
                                       cv2_bridge_sets=True)
    actions, report, _h = cio.propose_cv2_resolution(reg, payload, [], settings, policy, epoch=0, history={},
                                                     subsamples={}, reserve={"fraction": 0.5, "max_replicas": 80},
                                                     bridge_columns=columns)
    (cand,) = [c for c in report["candidates"] if c["rule"] == "R1"]
    n = len(keep)
    assert cand["decision"] == "proposed" and cand["metrics"]["bridge_mode"] == "set"
    assert cand["metrics"]["n_children"] == n and cand["cost_states"] == n
    (ins,) = [x for x in actions if x[0] == "insert"]
    ctl = ap.AdaptiveProductionController(reg, policy=policy)
    ctl.apply_actions(0, [ins])
    new = sorted((s for s in reg.active_states() if s.state_id not in (upper, lower)),
                 key=lambda s: float(s.secondary_center))
    assert [round(float(s.secondary_center), 9) for s in new] == [round(pl.centres[q], 9) for q in keep]
    for s in new:          # nearer endpoint seeds each child, whatever the edge order
        src = (s.metadata or {})[cr.METADATA_KEY]["seed_source_state_id"]
        near = lower if abs(float(s.secondary_center) - za) <= abs(float(s.secondary_center) - zb) else upper
        assert src == near
        assert "seed_source_by_child" not in (s.metadata or {})[cr.METADATA_KEY]

def test_swarm_dispatch_honours_the_flag():
    import types
    from gareus.swarm.cv2_shape_layout import select_pair_layout
    cv1, z2, ids = _swarm(COLUMNS)
    lo, hi = np.quantile(z2, [0.02, 0.98])
    uniform = np.linspace(lo, hi, 4)
    sigma_t = (hi - lo) / 3 / 1.5
    inputs = dict(cv1=cv1, z2=z2, member_ids=ids, deltav_kj=np.zeros_like(z2), centers1=[c for c, _ in COLUMNS],
                  ks1=[800.0] * len(COLUMNS), temperature_k=T, overlap_sigma=1.5, k_min=1e-3)
    for flag in (False, True):
        args = types.SimpleNamespace(max_replicas=2 * 30, swarm_adaptive_reserve_fraction=0.0, swarm_cv2_layout="shape",
                                     cv2_k_max=1000.0, swarm_cv2_min_mode_members=8, swarm_cv2_bridge_sets=flag)
        _l, _c, _k, shaped = select_pair_layout(args, len(COLUMNS), 4, [0.0, 1.0], [0], uniform,
                                                [RT / sigma_t ** 2] * 4, **inputs)
        assert ("bridge_sets" in shaped["record"]) is flag


# ---- review fixes (2026-10-03 adversarial reviews) ----------------------------------------

def test_refusal_reasons_survive_the_final_pass():
    out = _layout(2 * 24, True)
    rec = out["record"]
    reasons = {q["reason"] for q in rec["requests"] if not q["granted"]}
    assert reasons <= {"cap", "bridge_set_does_not_fit", "barrier_chain_does_not_fit", "cv1_loose_column",
                       "near_duplicate"}
    for bs in _sets(out):
        if not bs["granted"]:
            assert bs["reason"] in ("bridge_set_does_not_fit", "cv1_loose_column", "cap")


def test_cv1_loose_columns_are_not_eligible():
    from gareus.swarm import cv2_shape_layout as L
    out = _layout(2 * 200, True)
    cols = out["record"]["columns"]
    widths = [c["sigma_w1"] for c in cols]
    med = float(np.median(widths))
    for bs in _sets(out):
        loose = widths[bs["column"]] > L.CV1_LOOSE_FACTOR * med
        assert bs["eligible"] is (not loose)


def test_loose_column_set_is_never_granted_under_a_cap():
    from gareus.swarm.cv2_shape_layout import _grant_bridge_tiers
    fill = [{"kind": "fill", "column": 0, "position": p, "cell": (0, p), "bridge_set": 0} for p in (1, 2)]
    sets = [{"index": 0, "column": 0, "lower_position": 0, "fill_positions": [1, 2], "n_fills": 2, "score": 9.0,
             "eligible": False, "ineligible_reason": "cv1_loose_column"}]
    cells, roles = [(None, None)], ["unrestrained_anchor"]
    _grant_bridge_tiers([], [], fill, [], sets, 10, cells, roles)
    assert not sets[0]["granted"] and sets[0]["reason"] == "cv1_loose_column"
    assert all(q["reason"] == "cv1_loose_column" and not q["granted"] for q in fill) and len(cells) == 1


def test_internal_barrier_flags_a_window_straddling_two_wells():
    from gareus.adaptive.cv2_shape import internal_barriers_kT
    fit, _ = _double_well_fit()
    soft_mid = internal_barriers_kT(fit, [0.1], [0.5], T)[0]
    stiff_well = internal_barriers_kT(fit, [-0.8], [30.0], T)[0]
    assert soft_mid > 1.0 and stiff_well == 0.0


def test_chain_not_proposed_without_gain():
    """Re-planned with the chain's own windows already among the CV1-free set, every candidate is a
    duplicate and the chain is not proposed."""
    from gareus.swarm import cv2_shape_layout as L
    fit, _ = _double_well_fit()
    from gareus.swarm.cv2_shape_layout import ShapeColumn
    pl = _placement(fit)
    col = ShapeColumn(0, 0.2, 800.0, 0.05, 1.0, (-1.7, 1.8), fit, pl, tuple([0.1] * len(pl.centres)))
    first = L.barrier_chain([col], (-1.7, 1.8), temperature_k=T, sigma_w_target=0.78, k_min=1e-3, k_max=1000.0,
                            spacing_sigma=1.5, existing=[(-1.7, 0.97), (1.8, 0.97)])
    assert first["centres"], first
    again = L.barrier_chain([col], (-1.7, 1.8), temperature_k=T, sigma_w_target=0.78, k_min=1e-3, k_max=1000.0,
                            spacing_sigma=1.5, existing=[(-1.7, 0.97), (1.8, 0.97),
                                                         *zip(first["centres"], first["k2"])])
    assert not again["centres"] and again["not_proposed"] == "all_duplicates"


def test_r1_set_over_budget_falls_back_to_the_midpoint():
    from gareus.adaptive import cv2_resolution as cr

    class _Allow:
        governed, reason, resolution_slots = True, "ok", 7
        free_slots = add_rung_slots = n_rungs = add_rung_centres = resolution_centres = 0
    kids = [{"primary_center": 0.2, "primary_k": 10.0, "secondary_center": z, "k2": 5.0} for z in (0.0, 0.3, 0.6)]
    mid = {"parent_state_id": 1, "children": [{"primary_center": 0.2, "primary_k": 10.0, "secondary_center": 0.3,
                                               "k2": 5.0}]}
    cand = cr.new_candidate("R1", "edge", [1, 2], "proposed", "weak", cls="weak",
                            proposal={"parent_state_id": 1, "children": kids, "seed_sources": [1, 1, 2]})
    cand["fallback_proposal"] = mid
    (out,), _rec = cr.allocate([cand], _Allow(), 3)
    assert out["decision"] == "proposed" and out["cost_states"] == 3
    assert out["proposal"] == mid and out["metrics"]["bridge_mode"] == "midpoint_after_budget"
    assert "fallback_proposal" not in out
    assert cr.candidate_action(out, 0)[0] == "add"
