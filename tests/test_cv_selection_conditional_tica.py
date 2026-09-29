"""Conditional tICA candidates and slowness ranking in automatic CV2 selection.

chignolin_9 deployed residual PC1, a fast, single-peaked torsion direction: its rows only
tilted the ensemble and it added no exploration over CV1-only chignolin_7. A retrospective
check found the slowest CV1-residual direction (conditional tIC1, psi(D3)+psi(P4)) was
bimodal at fixed CV1 and far slower in production. These tests pin the new behaviour: slow
directions of the CV1-residualised torsions are candidates, and a deployable candidate is
ranked by slowness at fixed CV1, gated on bimodality and on agreement between seed halves.
"""
from __future__ import annotations

import numpy as np
import pytest

from gareus.cv_selection import contracts as C
from gareus.cv_selection.anchor import AnchorCandidate
from gareus.cv_selection.models import evaluate_component, from_candidate_set
from gareus.cv_selection.residual_runtime import compile_component
from gareus.cv_selection.select_pair import SelectionConfig, SwarmDataset, select_cv_pair
from gareus.cv_selection.slowness import lag_pairs

ARGS = dict(physical_system_sha256="b" * 64, training_rows_sha256="c" * 64,
            library_versions={"numpy": np.__version__}, genpept_preset="broad")
D = 8


def _anchor(values):
    return AnchorCandidate("nonlocal-contact-fraction",
                           {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0, "min_sequence_separation": 4,
                            "atom_selection": "heavy", "pair_rule": "all-pairs-min-sep",
                            "normalize": True, "pair_list_sha256": "d" * 64, "norm": 45.0},
                           values)


def _schema():
    rows = []
    for t in range(D // 2):
        name = "phi" if t < D // 4 else "psi"
        for trig in ("sin", "cos"):
            rows.append({"index": len(rows), "name": f"{name}-{t}-{trig}", "torsion_name": f"{name}-{t}",
                         "residue_index": t, "atom_indices": [4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3],
                         "trig": trig, "dihedral_sign_convention": "negated"})
    return C.FeatureSchema.from_mapping({"schema": C.FEATURE_SCHEMA_VERSION,
                                         "topology_sha256": "a" * 64, "features": rows})


def _ar1(rng, n, rho, scale):
    x = np.empty(n)
    x[0] = rng.normal() * scale
    e = rng.normal(size=n) * scale * np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def _telegraph(rng, n, p_switch, amp, noise):
    s = np.empty(n)
    s[0] = rng.choice([-1.0, 1.0])
    flips = rng.random(n) < p_switch
    for t in range(1, n):
        s[t] = -s[t - 1] if flips[t] else s[t - 1]
    return amp * s + rng.normal(size=n) * noise


def _trajectories(n_members=80, n_frames=250, seed=0, slow="telegraph", slow_dim=5,
                  alt_slow_dim=None, drop_every=None):
    """Members = independent trajectories (2 ps frames). Dim 0 is fast and high variance;
    ``slow_dim`` carries the slow hidden mode (low variance, orthogonal to the anchor)."""
    rng = np.random.default_rng(seed)
    X, a, g, mid, fidx, shape = [], [], [], [], [], []
    for m in range(n_members):
        am = np.clip(0.25 + _ar1(rng, n_frames, 0.98, 0.08), 0.0, 0.5)
        F = np.column_stack([_ar1(rng, n_frames, 0.3, 1.0 if k != 0 else 3.0) for k in range(D)])
        dim = slow_dim if (alt_slow_dim is None or m % 2 == 0) else alt_slow_dim
        if slow == "telegraph":
            F[:, dim] = _telegraph(rng, n_frames, 0.004, 0.8, 0.2)
        elif slow == "gaussian":
            F[:, dim] = _ar1(rng, n_frames, 0.995, 0.8)
        a_std = (am - 0.25) / 0.1
        F += np.outer(a_std, np.linspace(0.3, -0.3, D))            # every feature leans on CV1
        keep = np.ones(n_frames, bool)
        if drop_every:
            keep[::drop_every] = False
        X.append(F[keep]); a.append(am[keep]); g.append(np.full(keep.sum(), m // 2))
        mid.append(np.full(keep.sum(), m)); fidx.append(np.arange(n_frames)[keep])
        shape.append(np.column_stack([0.6 + 0.3 * am, 1.0 + 0.1 * F[:, 1]])[keep])
    return SwarmDataset(np.vstack(X), _schema(), _anchor(np.concatenate(a)), np.vstack(shape),
                        np.concatenate(g), member_ids=np.concatenate(mid),
                        frame_index=np.concatenate(fidx), frame_dt_ps=2.0)


def _slow_cfg(**kw):
    base = dict(ranking="slowness", tica_lag_ps=20.0, slowness_lag_ps=40.0, min_gain_nats=-1.0)
    base.update(kw)
    return SelectionConfig(**base)


@pytest.fixture(scope="module")
def slow_selection():
    data = _trajectories()
    return data, select_cv_pair(data, _slow_cfg(), **ARGS)


def _selected_vector(sel):
    fit = from_candidate_set(sel.candidate_set)
    return fit.right_vectors[sel.pair_model.selected_component_index - 1]


def test_lag_pairs_never_cross_members_or_gaps():
    members = np.array([0, 0, 0, 0, 1, 1, 1])
    frames = np.array([0, 1, 3, 4, 0, 1, 2])
    i, j = lag_pairs(members, frames, 1)
    assert sorted(zip(i.tolist(), j.tolist())) == [(0, 1), (2, 3), (4, 5), (5, 6)]


def test_conditional_tica_finds_and_selects_the_slow_bimodal_mode(slow_selection):
    _, sel = slow_selection
    assert sel.status == "pair", sel.report["selection_reason"]
    assert sel.report["ranking"] == "slowness"
    j = sel.pair_model.selected_component_index
    assert j > C.MAX_PCA_COMPONENTS, f"selected residual PC{j}, not a conditional tICA mode"
    comp = next(c for c in sel.candidate_set.components if c.component_index == j)
    assert comp.family == "conditional_tica" and comp.tica_eigenvalue > 0.5
    v = _selected_vector(sel)
    assert abs(v[5]) / np.linalg.norm(v) > 0.9
    scores = sel.report["components"][str(j)]
    assert scores["slowness_rho"] > sel.report["components"]["1"]["slowness_rho"]
    assert scores["bimodality_max"] >= 0.555
    assert sel.pair_model.certificate["half_split_agrees"] is True


def test_a_slow_but_single_peaked_mode_fails_the_bimodality_gate():
    sel = select_cv_pair(_trajectories(slow="gaussian"), _slow_cfg(), **ARGS)
    reasons = " ".join(" ".join(s["reasons"]) for s in sel.report["components"].values())
    assert "bimodality" in reasons
    if sel.status == "pair":
        j = sel.pair_model.selected_component_index
        assert sel.report["components"][str(j)]["bimodality_max"] >= 0.555


def test_a_candidate_not_reproduced_in_both_seed_halves_is_not_deployed(slow_selection, monkeypatch):
    # The halves' refits hold only the residual PCs, which do not contain the slow mode:
    # the full-data tICA winner is then not reproduced and must not be deployed.
    from dataclasses import replace
    import gareus.cv_selection.select_pair as sp
    data, sel = slow_selection
    full = from_candidate_set(sel.candidate_set)
    k = C.MAX_PCA_COMPONENTS
    pca_only = replace(full, singular_values=full.singular_values[:k], right_vectors=full.right_vectors[:k],
                       projection_mean=full.projection_mean[:k], projection_std=full.projection_std[:k],
                       families=full.families[:k], tica_lag_frames=full.tica_lag_frames[:k],
                       tica_eigenvalues=full.tica_eigenvalues[:k])
    winner = sel.pair_model.selected_component_index
    z = evaluate_component(full, winner, data.features, np.asarray(data.anchor.values))
    best = max(abs(np.corrcoef(evaluate_component(full, j, data.features, np.asarray(data.anchor.values)), z)[0, 1])
               for j in range(1, k + 1))
    assert best < 0.8, "precondition: the slow mode is not a residual PC"
    monkeypatch.setattr(sp, "_half_fits", lambda *args, **kw: [pca_only, pca_only])
    again = select_cv_pair(data, _slow_cfg(), **ARGS)
    assert "not reproduced" in " ".join(again.report["components"][str(winner)]["reasons"])
    assert again.status == "cv1_only"


def test_the_winner_records_its_half_split_correlations(slow_selection):
    _, sel = slow_selection
    corr = sel.report["half_split"]["winner_abs_corr"]
    assert len(corr) == 2 and min(corr) >= 0.8


def test_dropped_frames_are_handled_and_selection_still_works():
    sel = select_cv_pair(_trajectories(drop_every=7), _slow_cfg(), **ARGS)
    assert sel.status == "pair", sel.report["selection_reason"]
    assert sel.pair_model.selected_component_index > C.MAX_PCA_COMPONENTS


def test_without_trajectory_order_the_ranking_falls_back_to_gain(slow_selection):
    data, _ = slow_selection
    plain = SwarmDataset(data.features, data.feature_schema, data.anchor, data.shape_features, data.groups)
    sel = select_cv_pair(plain, _slow_cfg(), **ARGS)
    assert sel.report["ranking"] == "gain"
    assert "trajectory order" in sel.report["ranking_fallback_reason"]
    assert all(c.family == "pca" for c in sel.candidate_set.components)


def test_gain_ranking_is_the_legacy_pca_only_rule(slow_selection):
    data, _ = slow_selection
    sel = select_cv_pair(data, SelectionConfig(ranking="gain", min_gain_nats=-1.0), **ARGS)
    # trajectory order must not leak into the legacy rule's artifacts
    plain = SwarmDataset(data.features, data.feature_schema, data.anchor, data.shape_features, data.groups)
    legacy = select_cv_pair(plain, SelectionConfig(ranking="gain", min_gain_nats=-1.0), **ARGS)
    assert sel.candidate_set.sha256 == legacy.candidate_set.sha256
    assert sel.pair_model.sha256 == legacy.pair_model.sha256
    assert all("family" not in r["scores"] for r in sel.pair_model.runner_ups)
    assert sel.report["ranking"] == "gain"
    assert len(sel.candidate_set.components) == C.MAX_PCA_COMPONENTS
    assert sel.pair_model.selected_component_index <= C.MAX_PCA_COMPONENTS


def test_candidate_set_with_tica_components_round_trips_and_pca_components_carry_no_new_keys(slow_selection):
    _, sel = slow_selection
    cs = sel.candidate_set
    again = C.CandidateSet.from_json_bytes(cs.to_json_bytes())
    assert again.sha256 == cs.sha256
    for comp in cs.components:
        mapping = comp.to_mapping()
        if comp.family == "pca":
            assert "family" not in mapping and "tica_eigenvalue" not in mapping
        else:
            assert mapping["family"] == "conditional_tica" and mapping["tica_lag_frames"] >= 1


def test_family_is_bound_to_its_index_range(slow_selection):
    _, sel = slow_selection
    raw = sel.candidate_set.to_mapping()
    pca = dict(raw["components"][0])
    pca.update(family="conditional_tica", tica_lag_frames=5, tica_eigenvalue=0.9)
    raw["components"] = [pca] + raw["components"][1:]
    with pytest.raises(Exception):
        C.CandidateSet.from_mapping(raw)


def test_component_index_cap(slow_selection):
    _, sel = slow_selection
    raw = sel.candidate_set.to_mapping()
    comp = dict(raw["components"][0])
    comp["component_index"] = C.MAX_COMPONENT_INDEX + 1
    raw["components"] = [comp]
    with pytest.raises(Exception):
        C.CandidateSet.from_mapping(raw)
    assert C.MAX_COMPONENT_INDEX == C.MAX_PCA_COMPONENTS + C.MAX_TICA_COMPONENTS


def test_compiled_tica_component_matches_the_selection_evaluator(slow_selection):
    data, sel = slow_selection
    fit = from_candidate_set(sel.candidate_set)
    j = sel.pair_model.selected_component_index
    a = np.asarray(data.anchor.values)
    ref = evaluate_component(fit, j, data.features, a)
    got = compile_component(fit, j, norm=1.0).evaluate_features(np.asarray(data.features), a)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-10)
    assert abs(ref.mean()) < 0.2 and 0.7 < ref.std() < 1.3
