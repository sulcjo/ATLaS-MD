"""Task 5: deterministic CV2 selection for a fixed contact anchor."""
from __future__ import annotations

import numpy as np
import pytest

from gareus.cv_selection import contracts as C
from gareus.cv_selection.anchor import AnchorCandidate
from gareus.cv_selection.pair_model import PairModel
from gareus.cv_selection.select_pair import (SelectionConfig, SwarmDataset, balanced_weights,
                                             select_cv_pair)

ARGS = dict(physical_system_sha256="b" * 64, training_rows_sha256="c" * 64,
            library_versions={"numpy": np.__version__}, genpept_preset="broad")


def _contact_anchor(values):
    return AnchorCandidate("nonlocal-contact-fraction",
                           {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0, "min_sequence_separation": 4,
                            "atom_selection": "heavy", "pair_rule": "all-pairs-min-sep",
                            "normalize": True, "pair_list_sha256": "d" * 64, "norm": 45.0},
                           values)


def _schema(d):
    rows = []
    for t in range(d // 2):
        name = "phi" if t % 2 == 0 else "psi"
        for trig in ("sin", "cos"):
            rows.append({"index": len(rows), "name": f"{name}-{t}-{trig}", "torsion_name": f"{name}-{t}",
                         "residue_index": t, "atom_indices": [4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3],
                         "trig": trig, "dihedral_sign_convention": "negated"})
    return C.FeatureSchema.from_mapping({"schema": C.FEATURE_SCHEMA_VERSION,
                                         "topology_sha256": "a" * 64, "features": rows})


def _dataset(n=6000, seed=0, wide=True, quadratic_mode=False):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 0.5 if wide else 0.069, n)
    a_std = (a - a.mean()) / a.std()
    d = 8
    hidden = rng.normal(size=(n, d)) @ np.diag(np.linspace(3.0, 0.3, d))
    X = np.outer(a_std, rng.normal(size=d) * 0.3) + hidden
    if quadratic_mode:
        X[:, 0] += 3.0 * (a_std ** 2 - 1.0)
    shape = np.column_stack([0.6 + 0.3 * a + 0.05 * rng.normal(size=n),
                             1.0 + hidden[:, 1] * 0.1])
    groups = rng.integers(0, 60, n)
    return SwarmDataset(X, _schema(d), _contact_anchor(a), shape, groups)


def test_selects_a_residual_component_with_an_exact_weighted_certificate():
    sel = select_cv_pair(_dataset(), SelectionConfig(), **ARGS)
    assert sel.status == "pair", sel.report["selection_reason"]
    cert = sel.pair_model.certificate
    assert abs(cert["cov_q_weighted"]) < 1e-8
    assert cert["coupling_fraction_of_k1"] <= 0.25
    assert cert["selected_gain_nats"] >= 0.02
    assert isinstance(cert["half_split_agrees"], bool)
    assert sel.pair_model.anchor["kind"] == "nonlocal-contact-fraction"
    assert sel.candidate_set.kind == C.CANDIDATE_KIND_QUADRATIC_RESIDUAL


def test_the_frozen_pair_model_round_trips_and_binds_to_its_candidate_set():
    sel = select_cv_pair(_dataset(), SelectionConfig(), **ARGS)
    back = PairModel.from_mapping(sel.pair_model.to_mapping())
    assert back == sel.pair_model
    assert back.candidate_set_sha256 == sel.candidate_set.sha256


def test_a_quadratic_hidden_mode_is_set_aside_with_the_fold_spread_recorded():
    sel = select_cv_pair(_dataset(quadratic_mode=True), SelectionConfig(), **ARGS)
    runner_ups = sel.report["runner_ups"]
    assert any("nonlinear" in r["reason"] for r in runner_ups)
    assert all("r2_se" in r["scores"] for r in runner_ups)
    if sel.status == "pair":
        assert sel.pair_model.selected_component_index not in {
            r["component_index"] for r in runner_ups if "nonlinear" in r["reason"]}


def test_a_component_that_couples_too_hard_into_cv1_is_set_aside():
    sel = select_cv_pair(_dataset(), SelectionConfig(max_coupling_fraction=1e-12), **ARGS)
    assert sel.status == "cv1_only"
    assert all("coupling" in r["reason"] for r in sel.report["runner_ups"])


def test_an_information_floor_prevents_crowning_noise_by_tie_break():
    sel = select_cv_pair(_dataset(), SelectionConfig(min_gain_nats=1e9), **ARGS)
    assert sel.status == "cv1_only"
    assert "no component adds information" in sel.report["selection_reason"]


def test_narrow_anchor_is_no_deployable_anchor():
    sel = select_cv_pair(_dataset(wide=False), SelectionConfig(), **ARGS)
    assert sel.status == "no_deployable_anchor" and sel.pair_model is None
    assert any("resolvable" in r for r in sel.anchor.reasons)


def test_selection_is_deterministic():
    a = select_cv_pair(_dataset(), SelectionConfig(), **ARGS)
    b = select_cv_pair(_dataset(), SelectionConfig(), **ARGS)
    assert a.pair_model.sha256 == b.pair_model.sha256


def test_balanced_weights_give_every_cell_equal_mass():
    cells = np.array([0] * 90 + [1] * 10)
    w = balanced_weights(cells)
    assert abs(w[cells == 0].sum() - w[cells == 1].sum()) < 1e-12 and abs(w.sum() - 1.0) < 1e-12


def test_a_fold_specific_genpept_preset_is_refused():
    with pytest.raises(ValueError, match="native-blind"):
        select_cv_pair(_dataset(), SelectionConfig(), **{**ARGS, "genpept_preset": "chignolin"})


def test_a_stored_pair_whose_certificate_is_not_orthogonal_is_a_broken_artifact():
    sel = select_cv_pair(_dataset(), SelectionConfig(), **ARGS)
    payload = sel.pair_model.to_mapping()
    payload.pop("sha256")
    payload["certificate"]["cov_q_weighted"] = 0.05
    with pytest.raises(C.IntegrityError) as excinfo:
        PairModel.from_mapping(payload)
    assert excinfo.value.reason is C.ReasonCode.INVALID_ESTIMATE
