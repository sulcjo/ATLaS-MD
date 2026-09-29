"""Spec 3.2 (+ X7): CV2 windows from the local free-energy shape of the swarm's design measure.

T1 items: a shoulder mixture (80 % N(0, 0.71) + 20 % N(mu, 0.15), mu = 1.2..1.6) gets a
centre at the narrow mode; a single Gaussian reproduces a uniform-equivalent layout; the
min_mode_members refusal; F'' shrinkage and floor; the k2 floor/clamp; budget ranking.
"""
import json
import math

import numpy as np
import pytest

from gareus.adaptive.cv2_shape import (estimate_f2, fit_cv2_mixture, mode_depth, mode_pair_resolvable,
                                       place_cv2_centres, predicted_sampled_sigma, shape_rule_k2)
from gareus.swarm.ladder_design import R_KCAL_MOL_K

T = 300.0
RT = R_KCAL_MOL_K * T


def _members(rng, specs):
    """specs: [(n_members, n_frames, mean, sd)] -> (z, member_ids); each member stays in one basin."""
    z, ids, m = [], [], 0
    for n_members, n_frames, mean, sd in specs:
        for _ in range(n_members):
            z.append(rng.normal(mean, sd, n_frames)); ids.append(np.full(n_frames, m)); m += 1
    return np.concatenate(z), np.concatenate(ids)


def _shoulder(mu, n_narrow=12, seed=1):
    rng = np.random.default_rng(seed)
    return _members(rng, [(48, 60, 0.0, 0.71), (n_narrow, 60, mu, 0.15)])


def _uniform_sigma(z, n=4, overlap_sigma=1.5):
    lo, hi = np.quantile(z, [0.02, 0.98])
    return (float(lo), float(hi)), (hi - lo) / (n - 1) / overlap_sigma


@pytest.mark.parametrize("mu", [1.2, 1.4, 1.6])
def test_shoulder_mixture_gets_a_centre_at_the_narrow_mode(mu):
    z, ids = _shoulder(mu)
    fit = fit_cv2_mixture(z, ids, max_components=3, min_mode_members=8)
    narrow = [c for c in fit.accepted_components if abs(c.mean - mu) < 0.05]
    assert len(narrow) == 1, fit.as_record()
    assert 0.5 * 0.15 ** 2 < narrow[0].variance < 2.0 * 0.15 ** 2
    assert narrow[0].n_members >= 8
    envelope, sigma_t = _uniform_sigma(z)
    pl = place_cv2_centres(fit, envelope, sigma_w_target=sigma_t, temperature_k=T, k_min=1e-3, k_max=1000.0)
    c = np.asarray(pl.centres)
    assert np.all(np.diff(c) > 0)
    at_mode = [i for i, x in enumerate(c) if abs(x - narrow[0].mean) < 1e-12]
    assert at_mode and pl.kinds[at_mode[0]] == "mode"
    # the narrow mode's predicted sampled width is below the broad one's (F'' shrunk, not ignored)
    sig = np.asarray(pl.sampled_sigma)
    assert sig[at_mode[0]] < sig[int(np.argmin(np.abs(c)))]
    # every adjacent gap respects 1.5 x the smaller predicted sampled sigma of the pair
    assert np.all(np.diff(c) <= 1.5 * np.minimum(sig[:-1], sig[1:]) + 1e-9)


def _narrow_mode_fit():
    from gareus.adaptive.cv2_shape import CV2MixtureFit, MixtureComponent
    comps = (MixtureComponent(0.0, 0.71 ** 2, 0.8, 10 ** 7, True, "accepted"),
             MixtureComponent(1.4, 0.15 ** 2, 0.2, 10 ** 7, True, "accepted"))
    return CV2MixtureFit(comps, 2, {2: 0.0}, 1e-3, {}, 4, 10 ** 4, 10 ** 7, 1e4, 0.28, 0.8, 8)


def test_placement_steps_by_the_smallest_sampled_sigma_ahead():
    """Plain width rule (compression floor off), shrinkage negligible: the fill resolves the
    narrow mode at 1.5 x its width and approaches it from the broad side without overshooting."""
    pl = place_cv2_centres(_narrow_mode_fit(), (-1.4, 2.0), sigma_w_target=0.7, temperature_k=T, k_min=1e-3,
                           k_max=1000.0, min_mean_compression=0.0)
    c, sig = np.asarray(pl.centres), np.asarray(pl.sampled_sigma)
    near = c[np.abs(c - 1.4) < 0.35]
    assert near.size >= 3
    assert np.all(np.diff(near) <= 1.5 * 0.15 + 1e-9)
    after = c[c > 1.4 - 1e-12][:2]                           # the first step out of the mode is 1.5 sigma
    np.testing.assert_allclose(after[1] - after[0], 1.5 * 0.15, rtol=1e-4)
    i0 = int(np.argmin(np.abs(c)))
    assert np.diff(near).min() < 0.5 * np.diff(c)[i0]
    assert np.all(np.diff(c) <= 1.5 * np.minimum(sig[:-1], sig[1:]) + 1e-9)
    assert pl.n_at_k_floor >= near.size                       # without the floor the narrow mode floors k2
    assert pl.n_at_compression_floor == 0
    assert max(pl.mean_compression[i] for i, x in enumerate(c) if abs(x - 1.4) < 0.35) < 0.01


def test_compression_floor_restrains_the_narrow_mode():
    """Default floor: at the narrow mode k2 = F'' (window means move half-way to their centre),
    so the sampled width there is sqrt(RT / 2F''), below the mode's own width, and no centre
    is left at cv2_k_min."""
    pl = place_cv2_centres(_narrow_mode_fit(), (-1.4, 2.0), sigma_w_target=0.7, temperature_k=T, k_min=1e-3,
                           k_max=1000.0)
    c, k2, f2 = np.asarray(pl.centres), np.asarray(pl.k2), np.asarray(pl.f2)
    at = f2 > 0.5 * RT / 0.15 ** 2                          # centres governed by the narrow mode
    assert np.all(np.abs(c[at] - 1.4) < 0.35)
    assert pl.n_at_k_floor == 0
    assert pl.n_at_compression_floor >= int(at.sum()) > 0
    np.testing.assert_allclose(k2[at], f2[at], rtol=1e-9)
    assert min(pl.mean_compression) >= 0.5 - 1e-12
    np.testing.assert_allclose(np.asarray(pl.sampled_sigma)[at], np.sqrt(RT / (2 * f2[at])), rtol=1e-9)
    assert np.all(np.asarray(pl.sampled_sigma)[at] < 0.15)
    assert pl.as_record()["min_mean_compression"] == 0.5


def test_single_gaussian_reproduces_a_uniform_equivalent_layout():
    rng = np.random.default_rng(3)
    z, ids = _members(rng, [(40, 100, 0.0, 1.0)])
    fit = fit_cv2_mixture(z, ids, max_components=3, min_mode_members=8)
    assert len(fit.accepted_components) == 1 and fit.n_components == 1
    mu = fit.accepted_components[0].mean
    n, half = 9, 2.0
    spacing = 2 * half / (n - 1)
    sigma_t = spacing / 1.5
    pl = place_cv2_centres(fit, (mu - half, mu + half), sigma_w_target=sigma_t, temperature_k=T,
                           k_min=1e-3, k_max=1000.0)
    np.testing.assert_allclose(pl.centres, np.linspace(mu - half, mu + half, n), atol=1e-9)
    f2 = pl.f2[0]
    assert f2 < RT / sigma_t ** 2                          # no floor: the sampled width is the target
    np.testing.assert_allclose(pl.k2, RT / sigma_t ** 2 - f2, rtol=1e-12)
    np.testing.assert_allclose(pl.sampled_sigma, sigma_t, rtol=1e-12)
    assert pl.kinds.count("mode") == 1


def test_a_mode_with_too_few_members_is_refused():
    z, ids = _shoulder(1.4, n_narrow=5)
    fit = fit_cv2_mixture(z, ids, max_components=3, min_mode_members=8)
    narrow = [c for c in fit.components if abs(c.mean - 1.4) < 0.1]
    assert narrow and not narrow[0].accepted and "min_mode_members" in narrow[0].reason
    assert narrow[0].n_members == 5
    envelope, sigma_t = _uniform_sigma(z)
    pl = place_cv2_centres(fit, envelope, sigma_w_target=sigma_t, temperature_k=T, k_min=1e-3, k_max=1000.0)
    assert not any(k == "mode" and abs(x - 1.4) < 0.1 for x, k in zip(pl.centres, pl.kinds))


def test_member_blocked_cross_validation_picks_a_regularisation_and_bic_is_reported():
    z, ids = _shoulder(1.4)
    fit = fit_cv2_mixture(z, ids, max_components=3, min_mode_members=8)
    assert set(fit.bic) == {1, 2, 3} and fit.n_components == min(fit.bic, key=fit.bic.get)
    assert fit.regularisation in fit.cv_scores and fit.cv_folds >= 2
    rec = fit.as_record()
    json.dumps(rec)
    assert rec["n_members"] == 60


def test_weights_act_as_a_cv1_kernel():
    z, ids = _shoulder(1.4)
    w = np.where(z > 1.0, 0.0, 1.0)                        # a kernel that excludes the narrow mode
    fit = fit_cv2_mixture(z, ids, weights=w, max_components=3, min_mode_members=8)
    assert not any(abs(c.mean - 1.4) < 0.1 for c in fit.accepted_components)


def test_f2_shrinks_toward_the_pooled_variance_and_is_floored():
    assert estimate_f2(0.01, 1.0, 8, T) == pytest.approx(RT / (0.5 * 0.01 + 0.5 * 1.0))
    assert estimate_f2(0.01, 1.0, 0, T) == pytest.approx(RT / 1.0)
    assert estimate_f2(0.01, 1.0, 10 ** 9, T) == pytest.approx(RT / 0.01, rel=1e-6)
    assert estimate_f2(float("nan"), 1.0, 8, T) == 0.0
    assert estimate_f2(-1.0, -1.0, 8, T) == 0.0
    assert estimate_f2(0.04, float("nan"), 8, T) == pytest.approx(RT / 0.04)


def test_shape_rule_k2_floor_and_clamp():
    assert shape_rule_k2(0.5, 1.0, T, 1e-3, 1000.0) == pytest.approx(RT / 0.25 - 1.0)
    assert shape_rule_k2(0.5, 50.0, T, 0.1, 1000.0, min_mean_compression=0.0) == 0.1   # width rule alone: floor
    assert shape_rule_k2(0.5, 50.0, T, 0.1, 1000.0) == pytest.approx(50.0)          # compression floor: k2 = F''
    assert shape_rule_k2(0.5, 50.0, T, 0.1, 1000.0, min_mean_compression=0.75) == pytest.approx(150.0)
    assert shape_rule_k2(0.5, 50.0, T, 0.1, 20.0) == 20.0                            # cv2_k_max wins over the floor
    with pytest.raises(ValueError):
        shape_rule_k2(0.5, 1.0, T, 0.1, 1000.0, min_mean_compression=1.0)
    assert shape_rule_k2(1e-3, 0.0, T, 0.1, 1000.0) == 1000.0         # clamp at cv2_k_max
    assert predicted_sampled_sigma(1.0, 3.0, T) == pytest.approx(math.sqrt(RT / 4.0))
    assert predicted_sampled_sigma(0.0, 0.0, T) == math.inf


def test_mode_depth_and_resolvability():
    from gareus.adaptive.cv2_shape import MixtureComponent
    two = (MixtureComponent(-2.0, 0.25, 0.5, 20, True, "accepted"),
           MixtureComponent(2.0, 0.25, 0.5, 20, True, "accepted"))
    d = mode_depth(two, 0, 1)
    assert d["bimodal"] and d["depth_kT"] > 5.0 and abs(d["barrier_z"]) < 0.05
    assert d["weight_a"] == d["weight_b"] == 0.5
    assert mode_pair_resolvable(d)
    lopsided = (MixtureComponent(-2.0, 0.25, 0.95, 20, True, "accepted"),
                MixtureComponent(2.0, 0.25, 0.05, 20, True, "accepted"))
    assert not mode_pair_resolvable(mode_depth(lopsided, 0, 1))          # the minor mode is < 10 %
    merged = (MixtureComponent(0.0, 1.0, 0.5, 20, True, "accepted"),
              MixtureComponent(0.5, 1.0, 0.5, 20, True, "accepted"))
    flat = mode_depth(merged, 0, 1)
    assert flat["depth_kT"] == 0.0 and not flat["bimodal"] and not mode_pair_resolvable(flat)


# ---------------------------------------------------------------------------------------
# swarm wiring: per-CV1-window fits, X7 axis windows, budget ranking, layout_plan record
# ---------------------------------------------------------------------------------------

def _two_column_swarm(seed=5):
    rng = np.random.default_rng(seed)
    parts = []
    m = 0
    for c1 in (0.2, 0.6):
        for n_members, mean, sd in ((48, 0.0, 0.71), (12, 1.4, 0.15)):
            for _ in range(n_members):
                parts.append((rng.normal(c1, 0.01, 60), rng.normal(mean, sd, 60), np.full(60, m)))
                m += 1
    cv1, z2, ids = (np.concatenate([p[i] for p in parts]) for i in range(3))
    return cv1, z2, ids


def _shape_layout(max_replicas, reserve_fraction=0.0):
    from gareus.swarm.cv2_shape_layout import build_shape_pair_layout
    cv1, z2, ids = _two_column_swarm()
    lo, hi = np.quantile(z2, [0.02, 0.98])
    uniform = np.linspace(lo, hi, 4)
    sigma_t = (hi - lo) / 3 / 1.5
    return build_shape_pair_layout(
        cv1=cv1, z2=z2, member_ids=ids, deltav_kj=np.zeros_like(z2), centers1=[0.2, 0.6], ks1=[800.0, 800.0],
        lambdas=[0.0, 1.0], temperature_k=T, uniform_centres2=uniform, uniform_ks2=[RT / sigma_t ** 2] * 4,
        overlap_sigma=1.5, k_min=1e-3, k_max=1000.0, max_replicas=max_replicas, region_centre_indices=[0],
        reserve_fraction=reserve_fraction, min_mode_members=8)


def test_shape_layout_grants_everything_when_it_fits():
    out = _shape_layout(max_replicas=400)
    plan, rec = out["layout"], out["record"]
    assert plan["status"] == "PROPOSED" and plan["kind"] == "joint"
    assert len(rec["columns"]) == 2
    for col in rec["columns"]:
        assert any(abs(m["mean"] - 1.4) < 0.05 for m in col["fit"]["components"] if m["accepted"])
    assert all(r["granted"] for r in rec["requests"])
    # X7: one CV1-free window per accepted CV2 mode (deduplicated across columns)
    x7 = [r for r in rec["requests"] if r["kind"] == "axis2_mode"]
    assert x7 and all(r["cell"][0] is None for r in x7)
    cv1_free = [out["centres2"][j] for j in range(4)] + [r["center2"] for r in x7]
    for mode in (0.0, 1.4):                      # a uniform row or an X7 window sits on every mode
        assert min(abs(z - mode) for z in cv1_free) < 0.25
    roles = set(plan["state_roles"])
    assert roles <= {"unrestrained_anchor", "region_representative", "axis", "joint"}
    assert len(out["centres2"]) == len(out["ks2"])


def test_shape_layout_under_a_tight_cap_keeps_modes_then_ranks_the_rest():
    full = _shape_layout(max_replicas=400)
    shaped = {"mode", "fill", "axis2_mode"}
    n_requests = sum(1 for r in full["record"]["requests"] if r["kind"] in shaped)
    tight = _shape_layout(max_replicas=2 * 8)           # 8 spatial states
    plan, rec = tight["layout"], tight["record"]
    assert plan["kind"] == "sparse" and plan["spatial_states"] == 8
    granted = [r for r in rec["requests"] if r["granted"]]
    dropped = [r for r in rec["requests"] if not r["granted"]]
    assert sum(1 for r in rec["requests"] if r["kind"] in shaped) == n_requests and dropped
    assert all(r["reason"] == "cap" for r in dropped)
    assert all(r["granted"] for r in rec["requests"] if r["kind"] == "mode")
    ranks = sorted(r["rank"] for r in granted)
    assert ranks == list(range(len(granted)))
    # dropped cells never become states: every plan cell is the anchor, a representative or granted
    assert plan["spatial_states"] == 1 + 1 + len(granted)


def test_shape_layout_honours_the_reserve():
    out = _shape_layout(max_replicas=40, reserve_fraction=0.25)
    plan = out["layout"]
    rec = plan["adaptive_reserve"]
    assert rec["reserved_replicas_requested"] == 10
    assert plan["spatial_states"] * 2 <= 40 - 10 and rec["free_slots"] >= 10


def test_swarm_analyze_shape_layout_writes_a_consistent_plan(synthetic_swarm, swarm_args):
    import csv
    from gareus.swarm.analyze import analyze_swarm_stage
    out = synthetic_swarm(with_features=True, wide_anchor=True)
    report = analyze_swarm_stage(out, swarm_args(secondary_cv="auto", swarm_cv2_layout="shape"))
    assert report["status"] == "pass", report.get("reasons")
    an = out / "swarm" / "analysis"
    plan = json.loads((an / "layout_plan.json").read_text())
    assert plan["cv2_shape"]["layout"] == "shape" and plan["cv2_shape"]["columns"]
    with (an / "windows_lambda_ladder.csv").open() as fh:
        recs = list(csv.DictReader(fh))
    assert len(recs) == plan["n_states"] == report["n_states"]
    assert all(s["role"] in {"unrestrained_anchor", "region_representative", "axis", "joint"}
               for s in plan["states"])
    assert set(plan["mandatory_state_ids"]) == {s["state_id"] for s in plan["states"]
                                                if s["role"] in ("unrestrained_anchor", "region_representative")}
    sel = json.loads((an / "cv_selection_report.json").read_text())
    assert sel["cv2_layout"] == "shape" and sel["cv2_shape_summary"]["n_columns"] >= 1
    assert "cv2_shape" not in sel["layout"]                   # the full record lives in layout_plan.json only
    # consumable by the P7b geometry: one rung's states chain into a single connected graph
    from gareus.adaptive.neighbour_rule import NeighbourPoint, chain_edges, components
    from gareus.layout_plan import read_layout_plan
    _raw, states = read_layout_plan(an / "layout_plan.json")
    rung0 = [s for s in states if s.gamd_lambda == 0.0]
    pts = [NeighbourPoint(s.center1, s.k1, s.center2, s.k2, 0.0) for s in rung0]
    edges = chain_edges(pts)
    assert len(components(len(pts), [(i, j) for i, j, _t, _d in edges])) == 1


def test_two_accepted_components_on_one_bump_are_one_mode():
    from gareus.adaptive.cv2_shape import CV2MixtureFit, MixtureComponent
    comps = (MixtureComponent(-0.06, 0.25, 0.3, 30, True, "accepted"),
             MixtureComponent(-0.04, 0.26, 0.7, 40, True, "accepted"))
    fit = CV2MixtureFit(comps, 2, {2: 0.0}, 1e-3, {}, 4, 5000, 70, 5000.0, -0.05, 0.26, 8)
    pl = place_cv2_centres(fit, (-1.5, 1.5), sigma_w_target=0.2, temperature_k=T, k_min=1e-3, k_max=1000.0)
    assert pl.kinds.count("mode") == 1 and -0.04 in pl.centres     # the heavier one is kept
    assert len(pl.dropped_modes) == 1 and pl.dropped_modes[0]["reason"].startswith("merged")
