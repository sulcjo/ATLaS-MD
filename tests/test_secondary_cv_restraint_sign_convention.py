"""Regression test: the tica-linear/torsion-pca restraint force must agree
with project_tica1() on the same structure.

OpenMM's CustomTorsionForce ``theta`` is the negative of the dihedral
convention used by tica._dihedral_rad (that function does not flip the first
bond vector the way the standard praxeolitic formula, and OpenMM, both do).
Every stored TICAResult -- weights, offset, window centers, seed-bank rescore
values -- is expressed in _dihedral_rad's basis. If
_add_weighted_trig_torsion_force builds its CustomTorsionForce from the raw
OpenMM ``theta`` instead of ``-theta``, every sin(phi)/sin(psi) feature term
flips sign (cos terms are unaffected, since cos is even) and the restraint
silently pulls toward a different value than project_tica1 reports for the
same positions -- reproducibly, regardless of pull duration or restraint
strength, since the restraint converges correctly to the wrong number.
"""
from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit as openmm_unit  # noqa: E402

from gareus.tica import TICAResult, project_tica1, backbone_dihedral_features  # noqa: E402
from gareus.production import _add_linear_torsion_cv_force  # noqa: E402


def _random_positions_nm(n_atoms: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(n_atoms, 3))


def _make_result(weights, offset, phi_torsions, psi_torsions, method: str) -> TICAResult:
    return TICAResult(
        weights=np.asarray(weights, dtype=np.float64),
        eigenvalue=0.5,
        mean=np.zeros(len(weights), dtype=np.float64),
        offset=float(offset),
        lag=60,
        phi_torsion_indices=[tuple(t) for t in phi_torsions],
        psi_torsion_indices=[tuple(t) for t in psi_torsions],
        n_samples=100,
        method=method,
    )


def _force_value_vs_official(phi_torsions, psi_torsions, mode: str, seed: int):
    """Build the real restraint force and compare it against project_tica1."""
    n_phi, n_psi = len(phi_torsions), len(psi_torsions)
    n_feats = 2 * n_phi + 2 * n_psi
    rng = np.random.default_rng(seed + 1000)
    weights = rng.uniform(-1.0, 1.0, size=n_feats)
    offset = float(rng.uniform(-2.0, 2.0))
    method = "tica" if mode == "tica-linear" else "pca"
    result = _make_result(weights, offset, phi_torsions, psi_torsions, method=method)

    all_atoms = [a for quart in (phi_torsions + psi_torsions) for a in quart]
    n_atoms = max(all_atoms) + 1
    pos_nm = _random_positions_nm(n_atoms, seed)

    system = openmm.System()
    for _ in range(n_atoms):
        system.addParticle(1.0)
    from pathlib import Path
    _add_linear_torsion_cv_force(
        openmm, system, phi_torsions, psi_torsions, result,
        mode=mode, state_path=Path("dummy_state.json"), force_group=29,
    )
    cv_force = next(
        system.getForce(i) for i in range(system.getNumForces())
        if isinstance(system.getForce(i), openmm.CustomCVForce)
    )
    integrator = openmm.VerletIntegrator(1.0 * openmm_unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(pos_nm * openmm_unit.nanometer)
    cv_values = cv_force.getCollectiveVariableValues(context)
    force_value = float(sum(cv_values) + offset)

    feats = backbone_dihedral_features(pos_nm, phi_torsions, psi_torsions)
    official_value = float(project_tica1(feats.reshape(1, -1), result)[0])
    return force_value, official_value


PHI = [(0, 1, 2, 3), (3, 4, 5, 6)]
PSI = [(1, 2, 3, 4), (4, 5, 6, 7)]


class TestSecondaryCVRestraintSignConvention:
    def test_tica_linear_both_phi_and_psi(self):
        force_value, official_value = _force_value_vs_official(PHI, PSI, "tica-linear", seed=1)
        assert force_value == pytest.approx(official_value, abs=1e-9)

    def test_torsion_pca_both_phi_and_psi(self):
        force_value, official_value = _force_value_vs_official(PHI, PSI, "torsion-pca", seed=2)
        assert force_value == pytest.approx(official_value, abs=1e-9)

    def test_phi_only_skip_branch(self):
        """No psi torsions: sum_sin_psi/sum_cos_psi sub-CVs are skipped entirely."""
        force_value, official_value = _force_value_vs_official(PHI, [], "tica-linear", seed=3)
        assert force_value == pytest.approx(official_value, abs=1e-9)

    def test_psi_only_skip_branch(self):
        """No phi torsions: sum_sin_phi/sum_cos_phi sub-CVs are skipped entirely."""
        force_value, official_value = _force_value_vs_official([], PSI, "tica-linear", seed=4)
        assert force_value == pytest.approx(official_value, abs=1e-9)

    @pytest.mark.parametrize("seed", range(5, 10))
    def test_multiple_independent_geometries(self, seed):
        """Agreement holds across independent random geometries, not just one lucky angle."""
        force_value, official_value = _force_value_vs_official(PHI, PSI, "tica-linear", seed=seed)
        assert force_value == pytest.approx(official_value, abs=1e-9)

    def test_would_have_caught_the_sign_bug(self):
        """Sanity check on the test itself: reverting the fix must fail this test.

        Directly reconstructs the pre-fix expression ("w*sin(theta)" instead of
        "w*sin(-theta)") to confirm it disagrees with project_tica1 by a
        nonzero, sin-weight-dependent amount -- i.e. this test is not
        vacuously true.
        """
        phi_torsions, psi_torsions = PHI, PSI
        n_phi, n_psi = len(phi_torsions), len(psi_torsions)
        rng = np.random.default_rng(1042)
        weights = rng.uniform(-1.0, 1.0, size=2 * n_phi + 2 * n_psi)
        offset = 0.0
        result = _make_result(weights, offset, phi_torsions, psi_torsions, method="tica")
        n_atoms = max(a for quart in (phi_torsions + psi_torsions) for a in quart) + 1
        pos_nm = _random_positions_nm(n_atoms, seed=1042)

        system = openmm.System()
        for _ in range(n_atoms):
            system.addParticle(1.0)
        cv_force = openmm.CustomCVForce("0")
        groups = [
            ("sum_sin_phi", phi_torsions, weights[0:2 * n_phi:2], "sin"),
            ("sum_cos_phi", phi_torsions, weights[1:2 * n_phi:2], "cos"),
            ("sum_sin_psi", psi_torsions, weights[2 * n_phi:2 * n_phi + 2 * n_psi:2], "sin"),
            ("sum_cos_psi", psi_torsions, weights[2 * n_phi + 1:2 * n_phi + 2 * n_psi:2], "cos"),
        ]
        names = []
        for fname, torsions, gw, trig in groups:
            f = openmm.CustomTorsionForce(f"w*{trig}(theta)")  # pre-fix: no negation
            f.addPerTorsionParameter("w")
            for (a, b, c, d), w in zip(torsions, gw):
                f.addTorsion(int(a), int(b), int(c), int(d), [float(w)])
            cv_force.addCollectiveVariable(fname, f)
            names.append(fname)
        cv_force.addGlobalParameter("ss_k", 0.0)
        cv_force.addGlobalParameter("ss0", 0.0)
        cv_force.setEnergyFunction(f"0.5*ss_k*(({' + '.join(names)})-ss0)^2")
        system.addForce(cv_force)

        integrator = openmm.VerletIntegrator(1.0 * openmm_unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName("Reference")
        context = openmm.Context(system, integrator, platform)
        context.setPositions(pos_nm * openmm_unit.nanometer)
        buggy_value = float(sum(cv_force.getCollectiveVariableValues(context)) + offset)

        feats = backbone_dihedral_features(pos_nm, phi_torsions, psi_torsions)
        official_value = float(project_tica1(feats.reshape(1, -1), result)[0])

        assert abs(buggy_value - official_value) > 1e-3
