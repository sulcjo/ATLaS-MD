"""The λ-ladder over the stock ``lower-dihedral`` boost: one channel, no auxiliary force.

A dihedral-only envelope has no ``*_Total`` globals.  The ladder must then scale only
``k0_Dihedral``, compute the rung boost from ``v_dih`` alone, and treat an all-NaN
``v_pep`` column as "no Total channel", not as missing data."""
import math
from types import SimpleNamespace

import numpy as np

from gareus.pep_gamd import (
    PepGamdEnvelope, pep_gamd_boost_kj, pep_gamd_boost_matrix_kj, k0max_from_globals,
    set_replica_lambda, ladder_supports_boost_type,
)
from gareus.mbar_analysis.ladder import apply_ladder_boost_to_u

DIH_ONLY = {"Vmax_Dihedral": 530.0, "Vmin_Dihedral": 377.0, "threshold_energy_Dihedral": 530.0, "k0_Dihedral": 1.0,
            "sigma0_Dihedral": 25.1}
DUAL = dict(DIH_ONLY, Vmax_Total=-2442.0, Vmin_Total=-3455.0, threshold_energy_Total=-2442.0, k0_Total=0.3)


def test_envelope_without_total_channel_loads_and_flags_it():
    env = PepGamdEnvelope.from_integrator_globals(DIH_ONLY)
    assert env.has_total is False
    assert env.k0max_dih == 1.0 and env.vmax_dih == 530.0
    assert PepGamdEnvelope.from_integrator_globals(DUAL).has_total is True


def test_dihedral_only_boost_ignores_v_pep_and_equals_dihedral_channel():
    env = PepGamdEnvelope.from_integrator_globals(DIH_ONLY)
    v_dih = 450.0
    expected = 0.5 * 1.0 * (530.0 - 450.0) ** 2 / (530.0 - 377.0)   # lower bound, E = Vmax, lambda = 1
    assert math.isclose(pep_gamd_boost_kj(float("nan"), v_dih, 1.0, env), expected, rel_tol=1e-12)
    assert math.isclose(pep_gamd_boost_kj(-3000.0, v_dih, 0.5, env), 0.5 * expected, rel_tol=1e-12)
    assert pep_gamd_boost_kj(float("nan"), v_dih, 0.0, env) == 0.0
    m = pep_gamd_boost_matrix_kj(np.full(3, np.nan), np.array([450.0, 500.0, 600.0]), [0.0, 1.0], env)
    assert m.shape == (2, 3) and np.isfinite(m).all() and m[0].max() == 0.0 and m[1, 2] == 0.0  # above E: no boost


def test_dual_envelope_still_nan_propagates_missing_v_pep():
    env = PepGamdEnvelope.from_integrator_globals(DUAL)
    u = np.zeros((2, 2)); meta = {}
    out = apply_ladder_boost_to_u(u, np.array([np.nan, -3000.0]), np.array([450.0, 450.0]), np.array([0.0, 1.0]), env, 1.0, meta)
    assert np.isnan(out[0, 1]) and np.isfinite(out[1, 1]) and meta["gamd_ladder_samples_without_raw_energies"] == 1


def test_apply_ladder_boost_dihedral_only_accepts_all_nan_v_pep():
    env = PepGamdEnvelope.from_integrator_globals(DIH_ONLY)
    u = np.zeros((3, 2)); meta = {}
    v_pep = np.full(3, np.nan); v_dih = np.array([450.0, 500.0, 600.0])
    out = apply_ladder_boost_to_u(u, v_pep, v_dih, np.array([0.0, 1.0]), env, 1.0, meta)
    assert np.isfinite(out).all(), "all-NaN v_pep is not missing data for a dihedral-only envelope"
    assert out[:, 0].max() == 0.0 and out[0, 1] > 0.0 and out[2, 1] == 0.0
    assert meta["gamd_ladder"] is True and meta["gamd_ladder_samples_without_raw_energies"] == 0


def test_k0max_from_globals_without_total():
    assert k0max_from_globals(DIH_ONLY) == {"Total": 0.0, "Dihedral": 1.0}
    assert k0max_from_globals(DUAL) == {"Total": 0.3, "Dihedral": 1.0}


class _FakeIntegrator:
    def __init__(self, names):
        self.vals = {n: 0.0 for n in names}
    def getNumGlobalVariables(self): return len(self.vals)
    def getGlobalVariableName(self, i): return list(self.vals)[i]
    def setGlobalVariableByName(self, n, v):
        if n not in self.vals: raise RuntimeError(f"no global {n}")
        self.vals[n] = v


def test_set_replica_lambda_skips_absent_total_global():
    integ = _FakeIntegrator(["k0_Dihedral", "Vmax_Dihedral"])
    set_replica_lambda(integ, 0.25, {"Total": 0.0, "Dihedral": 1.0})
    assert integ.vals["k0_Dihedral"] == 0.25
    both = _FakeIntegrator(["k0_Dihedral", "k0_Total"])
    set_replica_lambda(both, 0.5, {"Total": 0.3, "Dihedral": 1.0})
    assert both.vals == {"k0_Dihedral": 0.5, "k0_Total": 0.15}


def test_ladder_accepts_pep_gamd_and_lower_dihedral_only():
    assert ladder_supports_boost_type(SimpleNamespace(gamd_boost_type="pep-gamd-lower-dual"))
    assert ladder_supports_boost_type(SimpleNamespace(gamd_boost_type="lower-dihedral"))
    for bad in ("lower-dual", "lower-total", "upper-dihedral", "gamd-cmd-base", ""):
        assert not ladder_supports_boost_type(SimpleNamespace(gamd_boost_type=bad)), bad
