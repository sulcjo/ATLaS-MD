"""Equilibration-discard + autocorrelation subsampling for MBAR pooling.

pyMBAR assumes each state's samples are uncorrelated and drawn from
equilibrium. Pilot and short-epoch traces violate that, so we discard the
pre-equilibration prefix and subsample to the statistical-inefficiency
spacing before pooling samples across states.
"""
from __future__ import annotations

import numpy as np


def equilibrated_subsample_indices(series, *, min_samples: int = 10):
    """Return increasing integer indices into ``series`` after equilibration
    discard + autocorrelation subsampling.

    Falls back to all indices when pymbar is unavailable or the series is
    too short to estimate equilibration reliably.
    """
    arr = np.asarray(series, dtype=float).ravel()
    n = arr.size
    if n < int(min_samples):
        return np.arange(n, dtype=np.int64)
    try:
        from pymbar import timeseries
    except Exception:
        return np.arange(n, dtype=np.int64)
    try:
        t0, g, _neff = timeseries.detect_equilibration(arr)
        kept = timeseries.subsample_correlated_data(arr[t0:], g=g)
        return (np.asarray(kept, dtype=np.int64) + int(t0))
    except Exception:
        return np.arange(n, dtype=np.int64)
