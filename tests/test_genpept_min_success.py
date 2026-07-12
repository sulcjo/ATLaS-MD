"""GENPEPT minimization success must require a finite energy (NaN-energy crash).

With real-sequence conformers, a small fraction of minimizations diverge to a
non-finite energy.  minimize_one previously returned success=True regardless, so
NaN-energy results reached write_basin_archive; a basin containing only NaN-energy
members made np.nanargmin raise "All-NaN slice encountered".  A minimization that
yields a non-finite energy is a failure and must be excluded downstream.
"""

import pytest

import GENPEPT


def test_finite_energy_is_success():
    assert GENPEPT._minimization_succeeded(-1526.19) is True
    assert GENPEPT._minimization_succeeded(0.0) is True


def test_non_finite_energy_is_failure():
    assert GENPEPT._minimization_succeeded(float("nan")) is False
    assert GENPEPT._minimization_succeeded(float("inf")) is False
    assert GENPEPT._minimization_succeeded(-float("inf")) is False


def test_garbage_energy_is_failure():
    assert GENPEPT._minimization_succeeded(None) is False
    assert GENPEPT._minimization_succeeded("not-a-number") is False


def test_best_energy_index_handles_all_nan():
    # The write_basin_archive selection must not crash on an all-NaN basin.
    assert GENPEPT._best_energy_index([float("nan"), float("nan")]) is None
    assert GENPEPT._best_energy_index([]) is None


def test_best_energy_index_ignores_nan():
    assert GENPEPT._best_energy_index([float("nan"), -10.0, -5.0]) == 1
