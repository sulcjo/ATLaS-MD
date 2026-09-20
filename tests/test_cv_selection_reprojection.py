"""Task 6 (analysis side): a frozen pair model reprojects stored observations exactly."""
from __future__ import annotations

import numpy as np
import pytest

from gareus.cv_selection import contracts as C
from gareus.cv_selection.anchor import AnchorCandidate
from gareus.cv_selection.models import evaluate_component, from_candidate_set
from gareus.cv_selection.select_pair import SelectionConfig, SwarmDataset, select_cv_pair
from gareus.mbar_analysis.cv2_reprojection import (RESIDUAL_REGIME, load_cv2_model, project_cv2,
                                                   reproject_stored_obs)


def _schema(d):
    rows = []
    for t in range(d // 2):
        name = "phi" if t < d // 4 else "psi"
        for trig in ("sin", "cos"):
            rows.append({"index": len(rows), "name": f"{name}-{t}-{trig}", "torsion_name": f"{name}-{t}",
                         "residue_index": t, "atom_indices": [4 * t, 4 * t + 1, 4 * t + 2, 4 * t + 3],
                         "trig": trig, "dihedral_sign_convention": "negated"})
    return C.FeatureSchema.from_mapping({"schema": C.FEATURE_SCHEMA_VERSION,
                                         "topology_sha256": "a" * 64, "features": rows})


@pytest.fixture
def frozen_pair_dir(tmp_path):
    rng = np.random.default_rng(0)
    n, d = 4000, 8
    a = rng.uniform(0.0, 0.5, n)
    a_std = (a - a.mean()) / a.std()
    hidden = rng.normal(size=(n, d)) @ np.diag(np.linspace(3.0, 0.3, d))
    X = np.outer(a_std, rng.normal(size=d) * 0.3) + hidden
    shape = np.column_stack([0.6 + 0.3 * a, 1.0 + 0.1 * hidden[:, 1]])
    anchor = AnchorCandidate("nonlocal-contact-fraction",
                             {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0, "min_sequence_separation": 4,
                              "atom_selection": "heavy", "pair_rule": "all-pairs-min-sep", "normalize": True,
                              "pair_list_sha256": "d" * 64, "norm": 45.0}, a)
    data = SwarmDataset(X, _schema(d), anchor, shape, rng.integers(0, 60, n))
    sel = select_cv_pair(data, SelectionConfig(), physical_system_sha256="b" * 64,
                         training_rows_sha256="c" * 64, library_versions={"numpy": np.__version__},
                         genpept_preset="broad")
    assert sel.status == "pair", sel.report["selection_reason"]
    (tmp_path / "cv_pair_model.json").write_bytes(sel.pair_model.to_json_bytes())
    (tmp_path / "cv_candidate_set.json").write_bytes(sel.candidate_set.to_json_bytes())
    (tmp_path / "cv_feature_schema.json").write_bytes(data.feature_schema.to_json_bytes())
    return tmp_path, sel, X, a


def test_pair_model_loads_as_the_residual_regime_with_the_phi_psi_split(frozen_pair_dir):
    path, sel, X, a = frozen_pair_dir
    model = load_cv2_model(path / "cv_pair_model.json")
    assert model.regime == RESIDUAL_REGIME and model.is_residual
    assert model.n_features == X.shape[1]
    phi, psi = model.torsion_indices
    assert len(phi) == 2 and len(psi) == 2 and phi[0] == [0, 1, 2, 3]


def test_projection_reproduces_the_selector_exactly_and_needs_the_primary_cv(frozen_pair_dir):
    path, sel, X, a = frozen_pair_dir
    model = load_cv2_model(path / "cv_pair_model.json")
    fit = from_candidate_set(sel.candidate_set)
    expected = evaluate_component(fit, sel.pair_model.selected_component_index, X, a)
    got = project_cv2(model, X, a)
    assert np.array_equal(got, expected)
    with pytest.raises(ValueError, match="primary CV"):
        project_cv2(model, X)


def test_reproject_stored_obs_uses_the_recorded_primary_cv(frozen_pair_dir):
    path, sel, X, a = frozen_pair_dir
    model = load_cv2_model(path / "cv_pair_model.json")
    obs = {"features": X[:50], "steps": np.arange(50), "window": np.zeros(50, dtype=int),
           "primary_cv": a[:50], "secondary_cv": np.full(50, np.nan)}
    out = reproject_stored_obs([model], obs)
    assert out[f"cv2_{RESIDUAL_REGIME}"].shape == (50,)
    assert np.array_equal(out[f"cv2_{RESIDUAL_REGIME}"], project_cv2(model, X[:50], a[:50]))


def test_a_missing_companion_artifact_is_refused(frozen_pair_dir):
    path, *_ = frozen_pair_dir
    (path / "cv_feature_schema.json").unlink()
    with pytest.raises(FileNotFoundError, match="cv_feature_schema.json"):
        load_cv2_model(path / "cv_pair_model.json")


def test_a_tampered_candidate_set_breaks_the_digest_binding(frozen_pair_dir):
    path, *_ = frozen_pair_dir
    import json
    cs = json.loads((path / "cv_candidate_set.json").read_text())
    cs.pop("sha256")
    cs["components"][0]["projection_std"] *= 1.5
    (path / "cv_candidate_set.json").write_text(json.dumps(cs))
    with pytest.raises(RuntimeError, match="binds candidate set"):
        load_cv2_model(path / "cv_pair_model.json")
