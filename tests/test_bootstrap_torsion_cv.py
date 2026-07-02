import argparse

import numpy as np
import pytest

from gareus.cv import (
    secondary_cv_enabled,
    secondary_cv_is_transition,
    secondary_cv_mode,
    secondary_cv_range,
)
from gareus.tica import (
    TICAResult,
    compute_bootstrap_torsion_pca,
    project_tica1,
)


def _args(**kw) -> argparse.Namespace:
    ns = argparse.Namespace()
    for key, value in kw.items():
        setattr(ns, key, value)
    return ns


def test_torsion_pca_mode_aliases_and_range():
    assert secondary_cv_mode("torsion-pca") == "torsion-pca"
    assert secondary_cv_mode("bootstrap-torsion") == "torsion-pca"
    assert secondary_cv_mode("bootstrap_linear") == "torsion-pca"
    assert secondary_cv_mode("torsion-linear") == "torsion-pca"
    assert secondary_cv_mode(_args(secondary_cv="bootstrap-torsion")) == "torsion-pca"
    assert secondary_cv_mode({"secondary_cv": "bootstrap-torsion"}) == "torsion-pca"
    assert secondary_cv_enabled(_args(secondary_cv="torsion-pca")) is True
    assert secondary_cv_is_transition("torsion-pca") is True
    assert secondary_cv_range("torsion-pca") == pytest.approx((-6.0, 6.0))


def test_bootstrap_pca_returns_ticaresult_compatible_state():
    rng = np.random.default_rng(123)
    latent = np.linspace(-2.0, 2.0, 80)
    X = np.column_stack(
        [
            np.sin(latent),
            np.cos(latent),
            0.5 * latent,
            rng.normal(0.0, 0.02, size=latent.shape),
        ]
    )
    result = compute_bootstrap_torsion_pca(
        X,
        residualize=False,
        component=1,
        phi_torsion_indices=[(0, 1, 2, 3)],
        psi_torsion_indices=[(1, 2, 3, 4)],
    )
    assert isinstance(result, TICAResult)
    assert result.method == "pca"
    assert result.lag == 0
    assert result.weights.shape == (4,)
    assert result.mean.shape == (4,)
    assert result.n_samples == 80
    assert result.eigenvalue > 0.0
    projected = project_tica1(X, result)
    assert projected.shape == (80,)
    assert np.isfinite(projected).all()


def test_bootstrap_pca_residualizes_linear_cv1_signal():
    cv1 = np.linspace(-1.0, 1.0, 120)
    orthogonal = np.sin(np.linspace(0.0, 4.0 * np.pi, 120))
    X = np.column_stack(
        [
            3.0 * cv1 + 0.05 * orthogonal,
            -2.0 * cv1 + 0.10 * orthogonal,
            orthogonal,
            np.cos(np.linspace(0.0, 4.0 * np.pi, 120)),
        ]
    )
    result = compute_bootstrap_torsion_pca(X, cv1=cv1, residualize=True, component=1)
    projected = project_tica1(X, result)
    corr = np.corrcoef(projected, cv1)[0, 1]
    assert abs(corr) < 0.20


def test_bootstrap_pca_fail_closed_on_zero_variance():
    X = np.ones((30, 4), dtype=float)
    with pytest.raises(ValueError, match="zero bootstrap torsion PCA variance"):
        compute_bootstrap_torsion_pca(X, residualize=False)


def test_bootstrap_pca_fail_closed_on_zero_component_variance():
    X = np.column_stack([np.linspace(-1.0, 1.0, 12), np.zeros(12), np.zeros(12), np.zeros(12)])
    with pytest.raises(ValueError, match="zero bootstrap torsion PCA variance"):
        compute_bootstrap_torsion_pca(X, residualize=False, component=2)


def test_ticaresult_preserves_method_roundtrip(tmp_path):
    result = compute_bootstrap_torsion_pca(np.eye(6, dtype=float), residualize=False)
    path = tmp_path / "bootstrap.json"
    result.save(path)
    loaded = TICAResult.load(path)
    assert loaded.method == "pca"
    assert loaded.lag == 0
    assert loaded.weights.shape == result.weights.shape
