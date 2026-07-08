"""Regression: torsion-PCA cv2 window centers must span the seed distribution.

The chignolin collapse: cv2 `torsion-pca` centers are runtime-derived from the
GENPEPT seed torsion PC1, but (a) they were taken from the inner [0.15,0.85]
quantiles and (b) residualize-against-cv1 (default True) stripped the
cv1-correlated torsion variance -- which for a folder IS the fold signal --
collapsing PC1 to a thin band. Result: window centers spanned ~0.09 while the
runtime projection spanned ~3.8, so no window covered the folded basin.

These pin the two-part fix: span the full seed distribution, and stop
residualizing by default. Pure numpy -- no OpenMM / PeptideBuilder.
"""
import numpy as np

from gareus.production import _seed_projection_centers
from gareus.tica import compute_bootstrap_torsion_pca, project_tica1


def test_seed_projection_centers_span_full_distribution():
    # Outer centers must reach the tails (~2/98%), not the inner 70% (~15/85%),
    # so the umbrella ladder brackets the folded/extended extremes of the seeds.
    arr = np.linspace(-3.0, 3.0, 400)
    lo, mid, hi = _seed_projection_centers(arr)
    assert lo < -2.5 and hi > 2.5      # inner-70% would give only ~+-1.8
    assert abs(mid) < 0.2


def test_residualize_against_cv1_collapses_fold_correlated_span():
    # Build torsion features whose DOMINANT mode is correlated with cv1 (the fold)
    # plus a small orthogonal mode. Residualizing removes the fold mode -> PC1
    # projection span collapses; without residualization PC1 keeps the fold and
    # spans wide. This is why the run-facing default is now residualize=False.
    rng = np.random.default_rng(0)
    n = 120
    cv1 = np.linspace(-1.0, 1.0, n)
    dir_fold = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]); dir_fold /= np.linalg.norm(dir_fold)
    dir_orth = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0]); dir_orth /= np.linalg.norm(dir_orth)
    X = (
        (3.0 * cv1)[:, None] * dir_fold[None, :]
        + (0.2 * rng.standard_normal(n))[:, None] * dir_orth[None, :]
        + 0.01 * rng.standard_normal((n, 6))
    )
    res_on = compute_bootstrap_torsion_pca(X, cv1=cv1, residualize=True, component=1)
    res_off = compute_bootstrap_torsion_pca(X, cv1=cv1, residualize=False, component=1)
    span_on = float(np.ptp(project_tica1(X, res_on)))
    span_off = float(np.ptp(project_tica1(X, res_off)))
    assert span_off > 3.0 * span_on
