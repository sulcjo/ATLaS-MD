"""F02/P03: versioned residual artifacts -- v2 declares its basis, v1 is read as the old force.

Digest validity is necessary, not sufficient: these tests mutate v2 artifacts into
digest-valid but semantically inconsistent ones and require rejection (spec F02).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from gareus.cv_selection import contracts as C
from gareus.cv_selection._base import ContractError
from gareus.cv_selection.models import (PairModelRuntime, evaluate_component, fit_residual_components,
                                        from_candidate_set, to_candidate_set, validate_pair_semantics)
from gareus.cv_selection.pair_model import CERTIFICATE_VERSION_V2, PAIR_MODEL_VERSION_V1, PAIR_MODEL_VERSION_V2, PairModel


def _schema8():
    rows = []
    quads = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    for k, block in enumerate(["phi", "phi", "psi", "psi"]):
        for trig in ("sin", "cos"):
            rows.append({"index": len(rows), "name": f"{block}-{k % 2}-{trig}", "torsion_name": f"{block}-{k % 2}",
                         "residue_index": k % 2, "atom_indices": quads[k], "trig": trig,
                         "dihedral_sign_convention": "negated"})
    return C.FeatureSchema.from_mapping({"schema": C.FEATURE_SCHEMA_VERSION, "topology_sha256": "1" * 64, "features": rows})


def _anchor():
    return {"kind": "nonlocal-contact-fraction", "units": "dimensionless",
            "definition": {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0, "min_sequence_separation": 4,
                           "atom_selection": "heavy", "normalize": True, "norm": 1256.0}}


def _dataset(degree, seed=3, n=600):
    rng = np.random.default_rng(seed)
    a = np.r_[rng.uniform(0.2, 0.5, n - 10), rng.uniform(0.8, 0.9, 10)]
    angles = np.column_stack([3 * a * a + 0.1 * rng.normal(size=n), rng.normal(size=n),
                              rng.normal(size=n), rng.normal(size=n)])
    X = np.stack([np.sin(angles), np.cos(angles)], axis=-1).reshape(n, -1)
    return X, a


def _certificate_v2(cov_t=0.0, cov_raw=-0.01):
    return {"certificate_version": CERTIFICATE_VERSION_V2, "cov_transformed_anchor_weighted": cov_t,
            "cov_raw_anchor_weighted": cov_raw, "r2_z2_given_z1_mean": 0.01, "r2_z2_given_z1_se": 0.005,
            "r2_z1_given_z2_mean": 0.02, "coupling_curvature_kcal": 0.3, "coupling_fraction_of_k1": 0.001,
            "std_unweighted_z2": 0.9, "design_measure": "balanced_frame_cells_8", "n_frames": 600,
            "n_seed_families": 20, "half_split_agrees": False, "selected_gain_nats": 0.05, "max_gain_nats": 0.05}


def _pair_v2(cs, schema, *, component=1, deployable=True, transform=None, anchor=None, degree=None):
    return PairModel.from_mapping({
        "schema": PAIR_MODEL_VERSION_V2, "candidate_set_sha256": cs.sha256, "feature_schema_sha256": schema.sha256,
        "anchor": anchor or _anchor(), "selected_component_index": component,
        "degree": cs.degree if degree is None else degree, "certificate": _certificate_v2(),
        "runner_ups": [], "genpept_preset": "broad",
        "basis_transform": transform or dict(cs.basis_transform),
        "deployment": {"topology_sha256": "2" * 64 if deployable else None,
                       "physical_system_sha256": "3" * 64 if deployable else None,
                       "contact_pair_list_sha256": "4" * 64 if deployable else None, "deployable": deployable},
    })


@pytest.mark.parametrize("degree", [1, 2])
def test_v2_round_trip_preserves_float64_evaluation_and_declares_the_basis(degree):
    X, a = _dataset(degree)
    fit = fit_residual_components(X, a, degree=degree)
    schema = _schema8()
    cs = to_candidate_set(fit, schema, _anchor(), "a" * 64, "b" * 64, {"numpy": np.__version__},
                          design_measure="balanced_frame_cells_8")
    assert cs.schema_version == C.CANDIDATE_SET_VERSION_V2 and not cs.legacy
    assert cs.basis_transform["kind"] == ("hard_clip" if degree == 2 else "identity")
    assert cs.design_measure == "balanced_frame_cells_8"
    back = from_candidate_set(C.CandidateSet.from_json_bytes(cs.to_json_bytes()))
    assert back.transform == fit.transform
    np.testing.assert_array_equal(evaluate_component(back, 1, X, a), evaluate_component(fit, 1, X, a))


def test_digest_valid_but_semantically_inconsistent_v2_artifacts_are_rejected():
    X, a = _dataset(2)
    fit = fit_residual_components(X, a, degree=2)
    schema = _schema8()
    cs = to_candidate_set(fit, schema, _anchor(), "a" * 64, "b" * 64, {"numpy": np.__version__})
    payload = cs.to_mapping(); payload.pop("sha256")
    # 1. declared identity while the coefficients are quadratic
    bad = json.loads(json.dumps(payload)); bad["basis_transform"] = {"kind": "identity", "lo": -1.0, "hi": 1.0}
    with pytest.raises(ContractError, match="coefficients imply degree 2"):
        C.CandidateSet.from_mapping(bad)
    # 2. clip bounds differ from the components' own clamp
    bad = json.loads(json.dumps(payload)); bad["basis_transform"]["hi"] += 0.5
    with pytest.raises(ContractError, match="differ from the components' anchor_clamp"):
        C.CandidateSet.from_mapping(bad)
    # 3. components disagree on the shared regression
    bad = json.loads(json.dumps(payload)); bad["components"][1]["regression_coefficients"][1][0] += 1e-3
    with pytest.raises(ContractError, match="shared regression/scaling parameters"):
        C.CandidateSet.from_mapping(bad)
    # 4. pair anchor != candidate anchor, selected component absent, degree mismatch, transform mismatch
    other = dict(_anchor()); other["definition"] = dict(other["definition"], r0_angstrom=10.0)
    with pytest.raises(RuntimeError, match="anchor differs"):
        validate_pair_semantics(_pair_v2(cs, schema, anchor=other), cs, schema)
    small = to_candidate_set(fit_residual_components(X, a, degree=2, n_components=3), schema, _anchor(),
                             "a" * 64, "b" * 64, {"numpy": np.__version__})
    with pytest.raises(RuntimeError, match="does not contain"):
        validate_pair_semantics(_pair_v2(small, schema, component=6), small, schema)
    with pytest.raises(ContractError):                       # degree 1 with a clip transform is refused at parse
        _pair_v2(cs, schema, degree=1)
    wrong_t = dict(cs.basis_transform); wrong_t["hi"] = cs.basis_transform["hi"] + 0.25
    with pytest.raises(RuntimeError, match="basis transform"):
        validate_pair_semantics(_pair_v2(cs, schema, transform=wrong_t), cs, schema)


def test_certificate_v2_certifies_the_transformed_anchor_and_only_reports_the_raw_one():
    X, a = _dataset(2)
    fit = fit_residual_components(X, a, degree=2)
    schema = _schema8()
    cs = to_candidate_set(fit, schema, _anchor(), "a" * 64, "b" * 64, {"numpy": np.__version__})
    ok = _pair_v2(cs, schema)
    assert ok.certificate["cov_raw_anchor_weighted"] == -0.01          # nonzero raw covariance is fine
    payload = ok.to_mapping(); payload.pop("sha256")
    payload["certificate"]["cov_transformed_anchor_weighted"] = 1e-3
    with pytest.raises(ContractError, match="certified covariance"):
        PairModel.from_mapping(payload)


def test_placeholder_bindings_cannot_be_deployable_and_undeployable_pairs_are_refused_by_production_load(tmp_path):
    X, a = _dataset(1)
    fit = fit_residual_components(X, a, degree=1)
    schema = _schema8()
    cs = to_candidate_set(fit, schema, _anchor(), "a" * 64, "b" * 64, {"numpy": np.__version__})
    payload = _pair_v2(cs, schema).to_mapping(); payload.pop("sha256")
    payload["deployment"]["topology_sha256"] = None
    with pytest.raises(ContractError, match="required for a deployable pair"):
        PairModel.from_mapping(payload)
    discovery = _pair_v2(cs, schema, deployable=False)
    (tmp_path / "p.json").write_bytes(discovery.to_json_bytes())
    (tmp_path / "c.json").write_bytes(cs.to_json_bytes())
    (tmp_path / "f.json").write_bytes(schema.to_json_bytes())
    with pytest.raises(RuntimeError, match="not deployable"):
        PairModelRuntime.load(tmp_path / "p.json", tmp_path / "c.json", tmp_path / "f.json")
    rt = PairModelRuntime.load(tmp_path / "p.json", tmp_path / "c.json", tmp_path / "f.json", require_deployable=False)
    assert rt.deployable is False and rt.legacy is False


def test_legacy_v1_artifacts_read_as_the_old_force_definition_with_bytes_and_digest_untouched(tmp_path):
    X, a = _dataset(2)
    fit = fit_residual_components(X, a, degree=2)
    schema = _schema8()
    v2 = to_candidate_set(fit, schema, _anchor(), "a" * 64, "b" * 64, {"numpy": np.__version__})
    # write the same components as a v1 artifact (no basis transform, no design measure)
    v1_payload = v2.to_mapping(); v1_payload.pop("sha256"); v1_payload.pop("basis_transform"); v1_payload.pop("design_measure")
    v1_payload["schema"] = C.CANDIDATE_SET_VERSION_V1
    cs1 = C.CandidateSet.from_mapping(v1_payload)
    raw = cs1.to_json_bytes()
    again = C.CandidateSet.from_json_bytes(raw)
    assert again.legacy and again.sha256 == cs1.sha256 and again.to_json_bytes() == raw
    assert again.basis_transform == {"kind": "hard_clip", "lo": fit.anchor_clamp[0], "hi": fit.anchor_clamp[1]}
    legacy_fit = from_candidate_set(again)
    # the deployed (clipped) coordinate, exactly -- not a refit
    np.testing.assert_array_equal(evaluate_component(legacy_fit, 1, X, a), evaluate_component(fit, 1, X, a))
    pair1 = PairModel.from_mapping({
        "schema": PAIR_MODEL_VERSION_V1, "candidate_set_sha256": cs1.sha256, "feature_schema_sha256": schema.sha256,
        "anchor": _anchor(), "selected_component_index": 1, "degree": 2,
        "certificate": {"cov_q_weighted": 0.0, "r2_z2_given_z1_mean": 0.01, "r2_z2_given_z1_se": 0.005,
                        "r2_z1_given_z2_mean": 0.02, "coupling_curvature_kcal": 0.3, "coupling_fraction_of_k1": 0.001,
                        "std_unweighted_z2": 0.9, "design_measure": "balanced_frame_cells_8", "n_frames": 600,
                        "n_seed_families": 20, "half_split_agrees": False, "selected_gain_nats": 0.05, "max_gain_nats": 0.05},
        "runner_ups": [], "genpept_preset": "broad"})
    assert pair1.legacy and pair1.legacy_training_certificate and not pair1.deployable
    for name, art in (("p.json", pair1), ("c.json", cs1), ("f.json", schema)):
        (tmp_path / name).write_bytes(art.to_json_bytes())
    with pytest.raises(RuntimeError, match="legacy v1"):
        PairModelRuntime.load(tmp_path / "p.json", tmp_path / "c.json", tmp_path / "f.json", require_deployable=False)
    rt = PairModelRuntime.load(tmp_path / "p.json", tmp_path / "c.json", tmp_path / "f.json",
                               require_deployable=False, allow_legacy_v1=True)
    assert rt.legacy is True and rt.fit.transform == "hard_clip"


def test_unsupported_feature_layout_is_refused_at_load(tmp_path):
    from gareus.cv_selection.residual_runtime import check_feature_schema_layout
    rows = _schema8().to_mapping()["features"]
    swapped = [dict(r) for r in rows]
    swapped[0], swapped[1] = dict(swapped[1], index=0), dict(swapped[0], index=1)    # cos before sin
    schema = C.FeatureSchema.from_mapping({"schema": C.FEATURE_SCHEMA_VERSION, "topology_sha256": "1" * 64, "features": swapped})
    with pytest.raises(ValueError, match="expected \\(sin, cos\\)"):
        check_feature_schema_layout(schema)
    direct = [dict(r, dihedral_sign_convention="direct") for r in rows]
    schema = C.FeatureSchema.from_mapping({"schema": C.FEATURE_SCHEMA_VERSION, "topology_sha256": "1" * 64, "features": direct})
    with pytest.raises(ValueError, match="negated"):
        check_feature_schema_layout(schema)
