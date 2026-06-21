import numpy as np
import pytest

pytest.importorskip("pymbar")
from gareus.mbar_subsample import equilibrated_subsample_indices


def test_short_series_returns_all():
    s = np.arange(5, dtype=float)
    idx = equilibrated_subsample_indices(s, min_samples=10)
    assert list(idx) == [0, 1, 2, 3, 4]


def test_correlated_series_is_thinned():
    rng = np.random.default_rng(0)
    n = 5000
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.97 * x[i - 1] + rng.normal()  # strongly autocorrelated AR(1)
    idx = equilibrated_subsample_indices(x)
    assert idx.ndim == 1 and idx.dtype.kind == "i"
    assert 0 < idx.size < n           # genuinely subsampled
    assert np.all(np.diff(idx) >= 1)  # strictly increasing, valid indices
    assert idx.max() < n
