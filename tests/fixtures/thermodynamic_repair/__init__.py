"""Deterministic regression fixtures for the thermodynamic repair (plan P00).

Every builder is a pure function of its documented seed; values are float64 and units
are stated per fixture. See README.md for the baseline behaviour each one pins.
"""
from __future__ import annotations

import types

import numpy as np

# Values reproduced by the adversarial review at the reviewed baseline (OpenMM Reference
# platform, Python 3.12, NumPy 2.3). They are pinned to 10 significant figures.
REVIEW_TRUE_Z = 0.8396162907771245
REVIEW_FAULTY_Z = 0.06761176499910337
REVIEW_CONTEXT_ENERGY_KJ = 1.4559287063603097     # k = 10 kJ/mol/CV^2, centre 0.3
REVIEW_FAULTY_ENERGY_KJ = 0.27002145883415973


def residual_fast_path_case():
    """The review's I01 fixture: 16 particles, degree-1 residual model, anchor polynomial.

    Returns a namespace with positions (nm, shape (16, 3), rng seed 5), a second configuration
    (positions + N(0, 0.03) noise, rng seed 8), phi/psi quadruplets, contact pairs, the frozen
    ``ResidualFit``, its ``PairModelRuntime`` and the contact ``args``. ``k_kj = 10`` and
    ``center = 0.3`` reproduce the review's energies.
    """
    from gareus.cv_selection.models import PairModelRuntime, ResidualFit, contact_pair_list_digest

    rng = np.random.default_rng(5)
    pos = np.cumsum(rng.normal(scale=0.12, size=(16, 3)), axis=0) + 2.0
    pos2 = pos + np.random.default_rng(8).normal(scale=0.03, size=pos.shape)
    phi = [(0, 1, 2, 3), (4, 5, 6, 7)]
    psi = [(8, 9, 10, 11), (12, 13, 14, 15)]
    pairs = [(0, 15, 1.0), (1, 14, 1.0), (2, 13, 1.0), (3, 12, 1.0)]
    v = np.array([[0.2, 0.3, -0.1, 0.4, 0.2, -0.3, 0.5, 0.1]])
    coefficients = np.array([[0.1] * 8, [0.03] * 8, [0.0] * 8])
    fit = ResidualFit(coefficients, np.zeros(8), np.array([1.0]), v, 0.5, 0.2, (-3.0, 3.0),
                      np.array([0.0]), np.array([0.7]), 1)
    args = types.SimpleNamespace(contact_r0_a=12.0, contact_beta_a_inv=3.0, contact_normalize=True,
                                 contact_min_sequence_separation=4, contact_atom_selection="heavy")
    definition = {"r0_angstrom": 12.0, "beta_per_angstrom": 3.0, "min_sequence_separation": 4,
                  "atom_selection": "heavy", "normalize": True,
                  "pair_list_sha256": contact_pair_list_digest(pairs), "norm": 4.0}
    runtime = PairModelRuntime(fit, 1, "nonlocal-contact-fraction", definition, "f" * 64, tuple(phi + psi))
    return types.SimpleNamespace(positions_nm=pos, positions2_nm=pos2, phi=phi, psi=psi, pairs=pairs,
                                 fit=fit, runtime=runtime, args=args, k_kj=10.0, center=0.3,
                                 n_particles=16, particle_mass=12.0)


def residual_degree2_case(*, clamp=(-1.0, 1.0)):
    """A degree-2 variant of the fast-path fixture whose anchor leaves the clamp for one of the
    two configurations, so inside/outside-clamp behaviour is both exercised."""
    base = residual_fast_path_case()
    from gareus.cv_selection.models import PairModelRuntime, ResidualFit
    coefficients = np.array([[0.1] * 8, [0.03] * 8, [0.02] * 8])
    fit = ResidualFit(coefficients, np.zeros(8), np.array([1.0]), base.fit.right_vectors, 0.5, 0.05,
                      tuple(clamp), np.array([0.0]), np.array([0.7]), 2)
    base.fit = fit
    base.runtime = PairModelRuntime(fit, 1, base.runtime.anchor_kind, base.runtime.anchor_definition,
                                    "e" * 64, base.runtime.feature_atoms)
    return base


def separated_support_case():
    """Discovery CV1 values: 10,000 in [0.1, 0.3] and 200 in [0.8, 0.82]; every value is a seed."""
    x = np.r_[np.linspace(0.1, 0.3, 10000), np.linspace(0.8, 0.82, 200)]
    return types.SimpleNamespace(values=x, seeds=x, n_windows=6, k_max_kcal=1200.0, temperature_k=300.0,
                                 upper_cluster=(0.8, 0.82))


def cap_transition_case():
    return types.SimpleNamespace(n1=6, n2=4, n_rungs=4, caps=(92, 96))


def _tip3p_water(origin_nm):
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0957, 0.0], [0.0, -0.024, 0.0927]]) + np.asarray(origin_nm)


def solvent_collision_case():
    """Peptide heavy atoms at x = 0 and 0.50 nm; two TIP3P waters with oxygens at x = 0.05 and 0.45."""
    pos = np.vstack([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], _tip3p_water([0.05, 0.0, 0.0]), _tip3p_water([0.45, 0.0, 0.0])])
    heavy = np.array([1, 1, 1, 0, 0, 1, 0, 0], dtype=bool)
    return types.SimpleNamespace(positions_nm=pos, box_nm=np.eye(3) * 4.0, peptide=np.array([0, 1]),
                                 groups=[np.arange(2, 5), np.arange(5, 8)], heavy=heavy, initial_oo_nm=0.40)


def oscillating_displacement_case():
    """Peptide atoms at x = 0 and 0.30 nm; a single heavy solvent atom at x = 0.15 nm."""
    pos = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.15, 0.0, 0.0]])
    return types.SimpleNamespace(positions_nm=pos, box_nm=np.eye(3) * 4.0, peptide=np.array([0, 1]),
                                 groups=[np.array([2])], heavy=np.ones(3, dtype=bool), required_nm=0.22)


def dependent_dual_boost_case():
    from gareus.pep_gamd import PepGamdEnvelope
    env = PepGamdEnvelope(20.0, -20.0, 20.0, 1.0, 10.0, -10.0, 10.0, 1.0)
    return types.SimpleNamespace(envelope=env, v_pep_kj=-10.0, v_dih_kj=-5.0, b_half_kj=7.4322509765625,
                                 b_one_kj=13.0517578125, linear_half_kj=6.52587890625)


def quadratic_curvature_case():
    from gareus.cv_selection.models import ResidualFit
    fit = ResidualFit(np.array([[0.0], [0.0], [2.0]]), np.array([0.0]), np.array([1.0]), np.array([[1.0]]),
                      0.0, 1.0, (-3.0, 3.0), np.array([0.0]), np.array([1.0]), 2)
    return types.SimpleNamespace(fit=fit, center=1.0, k2=1.0, expected_curvature_at_zero=4.0)


def dependence_diagnostic_case():
    x = np.tile(np.linspace(-1.0, 1.0, 1001), 20)
    groups = np.repeat(np.arange(20), 1001)
    return types.SimpleNamespace(z2=x, z1=x * x, groups=groups)
