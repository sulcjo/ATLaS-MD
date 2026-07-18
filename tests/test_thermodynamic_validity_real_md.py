"""Tier A thermodynamic-validity oracle: real OpenMM MD, exact ground truth.

tests/test_physics_oracle.py already validates the MBAR *estimator* math
(solve_mbar / pmf_from_weights) against a hand-built, bias-only ``u_nk`` -
deliberately with "no OpenMM and no MD" so it runs in ~1 second. Its own
docstring names the gap this file closes:

    "Units are kept self-consistently in kcal/mol throughout, so u_nk needs
    no kJ<->kcal (4.184) conversion. The production code's 4.184 factor
    lives in ``reconstruct_bias_matrix`` ... validating *that* conversion
    is a separate construction-guard test, deliberately decoupled from
    this estimator-math test."

This file is that construction-guard test. Every link between "real MD
samples" and "recovered PMF" is now the pipeline's own real code, not a
hand-built matrix:

  real integrator      gareus.system_setup.make_langevin_integrator
  real umbrella force  gareus.forces.add_umbrella_force
  real per-window set  gareus.windows.set_window
  real CV extraction   gareus.state.cv_distance_nm
  real bias-matrix     gareus.query.reconstruct_bias_matrix  <- the untested link
  real reweighting     analyze_gareus_mbar.solve_mbar / pmf_from_weights

The system: two OpenMM particles, no force field, no solvent. The anchor
(particle 0) has mass 0 (OpenMM convention for "fixed"). The mobile particle
(particle 1) is pinned to the x-axis by a stiff transverse CustomExternalForce
and feels a "true" harmonic well (fixed d0 > 0, off the r=0 fold) plus the
real swept umbrella bias on the SAME anchor-mobile distance. Pinning to a
line makes that distance an exact 1-D flat-measure coordinate, so this
oracle inherits the same "PMF == potential, no Jacobian" identity
test_physics_oracle.py relies on - see its own docstring trap #2.

Deliberately NOT covered (separate follow-ups, not bugs in scope here):
  * A genuine 3-D inter-atom distance CV carries a radial r^2 Jacobian
    (a `-2kT*ln(r)` term) that ``pmf_from_weights`` does not correct for
    anywhere in this codebase. Pinning to 1-D here sidesteps that question
    rather than answering it - a real "3-D distance CV Jacobian" oracle
    would be a distinct, separate test.
  * GaMD boost. Reweighting a GaMD-boosted ensemble needs the cumulant-
    expansion machinery (analyze_gareus_mbar._cumulant_expansion), not
    reconstruct_bias_matrix - a materially different, larger validity
    test. This file is REUS-only.

Two independent invariants are checked, not one - a translation off the
origin (d0 != 0) makes both meaningful:
  * curvature 2a is translation-invariant -> compare directly to k0_true.
  * linear term b is NOT expected near zero here (unlike the origin-
    centered L0 oracle) - for a well at x=d0, b_true = -k0_true*d0. Getting
    this right also confirms window *positions*, not just spring constants,
    were threaded through correctly.

Mutation battery (ported directly from test_physics_oracle.py's table,
applied to the REAL sampled data): a passing control proves nothing until
shown to fail under the defect class it targets. Each mutation re-runs only
the cheap NumPy construction+reweighting chain on the one real MD dataset -
it does not re-run MD.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

agm = pytest.importorskip(
    "analyze_gareus_mbar", reason="estimator module must be importable from repo root"
)
openmm = pytest.importorskip("openmm", reason="Tier A needs a real OpenMM stack")
unit = pytest.importorskip("openmm.unit", reason="Tier A needs a real OpenMM stack")

from gareus.units import kcal_a2_to_kj_nm2  # noqa: E402
from gareus.system_setup import make_langevin_integrator  # noqa: E402
from gareus.state import cv_distance_nm  # noqa: E402
from gareus.forces import add_umbrella_force  # noqa: E402
from gareus.windows import set_window  # noqa: E402
from gareus.query import reconstruct_bias_matrix  # noqa: E402
from test_physics_oracle import _fit_curvature  # noqa: E402

# --- ground-truth parameters, kcal/mol/A^2 convention -----------------------
# (matches reconstruct_bias_matrix's native units AND test_physics_oracle.py's
# K0_TRUE/K_BIAS, for direct comparability across the two oracles.)
TEMP_K = 300.0
K0_TRUE_KCAL_A2 = 2.0     # true well curvature
K_BIAS_KCAL_A2 = 10.0     # umbrella spring constant
K_PIN_KCAL_A2 = 200.0     # transverse pin - stiff vs k0/k_bias, not "very large"
D0_A = 3.0                # true well center, well off the r=0 fold (~5.5 sigma0)
N_WINDOWS = 9
EQUIL_STEPS = 2000
PRODUCTION_STEPS = 20000
RECORD_STRIDE = 20
TIMESTEP_FS = 0.5
FRICTION_PER_PS = 5.0
SEED = 12345
MOBILE_MASS_AMU = 12.0


def _beta_kj_per_mol() -> float:
    """1/(kB*T) in mol/kJ - the exact convention reconstruct_bias_matrix expects,
    computed the same way gareus.production does (via unit.MOLAR_GAS_CONSTANT_R),
    not from a hand-copied Boltzmann-constant literal."""
    r_kj_mol_k = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)
    return 1.0 / (r_kj_mol_k * TEMP_K)


def _kt_kcal() -> float:
    return (1.0 / _beta_kj_per_mol()) / 4.184


def _build_system() -> "openmm.System":
    """Anchor (fixed, mass 0) + mobile particle pinned to the x-axis, under a
    fixed 'true' harmonic well plus the real, per-window-swept umbrella bias
    on the same anchor-mobile CustomBondForce distance add_umbrella_force
    already implements. The true well and the pin each bake their constant
    directly into the force expression (no addGlobalParameter) specifically
    so they cannot collide with the umbrella's own global "k"/"r0" namespace -
    only the swept umbrella may use those.
    """
    system = openmm.System()
    system.addParticle(0.0)              # anchor: mass 0 -> fixed by OpenMM
    system.addParticle(MOBILE_MASS_AMU)  # mobile

    d0_nm = D0_A * 0.1
    k0_kj_nm2 = kcal_a2_to_kj_nm2(K0_TRUE_KCAL_A2)
    k_pin_kj_nm2 = kcal_a2_to_kj_nm2(K_PIN_KCAL_A2)

    true_well = openmm.CustomBondForce(f"0.5*{k0_kj_nm2:.17g}*(r-{d0_nm:.17g})^2")
    true_well.addBond(0, 1, [])
    true_well.setForceGroup(20)
    system.addForce(true_well)

    pin = openmm.CustomExternalForce(f"0.5*{k_pin_kj_nm2:.17g}*(y^2+z^2)")
    pin.addParticle(1, [])
    pin.setForceGroup(21)
    system.addForce(pin)

    add_umbrella_force(openmm, system, 0, 1, force_group=31)
    return system


def _run_one_window(centers_a: np.ndarray, window_index: int) -> np.ndarray:
    """Run real Langevin MD for one umbrella window; return recorded CV samples (A)."""
    system = _build_system()
    integrator = make_langevin_integrator(
        openmm, unit, types.SimpleNamespace(seed=SEED),
        timestep_fs=TIMESTEP_FS, temperature_k=TEMP_K, friction_per_ps=FRICTION_PER_PS,
        seed_offset=window_index,
    )
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))

    centers_nm = [float(c) * 0.1 for c in centers_a]
    ks_kj_nm2 = [kcal_a2_to_kj_nm2(K_BIAS_KCAL_A2)] * len(centers_a)
    set_window(context, centers_nm, ks_kj_nm2, window_index)

    start_x_nm = centers_nm[window_index]
    context.setPositions([openmm.Vec3(0.0, 0.0, 0.0), openmm.Vec3(start_x_nm, 0.0, 0.0)])
    context.setVelocitiesToTemperature(TEMP_K * unit.kelvin, SEED + 1000 + window_index)

    integrator.step(EQUIL_STEPS)

    n_blocks = PRODUCTION_STEPS // RECORD_STRIDE
    samples_a = np.empty(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        integrator.step(RECORD_STRIDE)
        samples_a[i] = cv_distance_nm(context, 0, 1, unit) * 10.0
    return samples_a


@pytest.fixture(scope="module")
def real_md_windows():
    """Run all N_WINDOWS real umbrella windows once; every test in this module
    reuses the same real samples (mutations only re-run the cheap NumPy
    construction+reweighting chain, not MD)."""
    sigma0_a = np.sqrt(_kt_kcal() / K0_TRUE_KCAL_A2)
    centers_a = np.linspace(D0_A - 2.0 * sigma0_a, D0_A + 2.0 * sigma0_a, N_WINDOWS)
    assert centers_a.min() > 0.5, "windows too close to the r=0 fold - widen D0_A"

    cv_parts, win_parts = [], []
    for i in range(N_WINDOWS):
        samples_a = _run_one_window(centers_a, i)
        cv_parts.append(samples_a)
        win_parts.append(np.full(samples_a.shape, i, dtype=np.int64))
    cv_a = np.concatenate(cv_parts)
    window = np.concatenate(win_parts)
    windows = [{"center1": float(c), "k1": K_BIAS_KCAL_A2} for c in centers_a]
    return {"cv_a": cv_a, "window": window, "windows": windows, "centers_a": centers_a}


def _recover_curvature(cv_a, window, windows, beta):
    u_nk = reconstruct_bias_matrix(cv_a, None, windows, beta)
    res = agm.solve_mbar(u_nk, window, backend="anderson", tol=1e-10)
    w = agm.norm_logw(res["logw"])
    bins = np.linspace(cv_a.min() - 0.3, cv_a.max() + 0.3, 61)
    pmf = agm.pmf_from_weights(cv_a, w, bins, _kt_kcal())
    # _fit_curvature masks |cv_centers| <= fit_halfwidth, i.e. it assumes the
    # well sits at the coordinate origin. Ours sits at D0_A, so shift into the
    # well's own frame before fitting: curvature (2a) is translation-invariant,
    # and in this shifted frame the linear term is expected ~0 again, exactly
    # like test_physics_oracle.py's origin-centered case - no separate
    # "expected nonzero b" bookkeeping needed.
    curvature, linear, resid = _fit_curvature(
        pmf["cv_A"] - D0_A, pmf["pmf"], pmf["counts"],
        fit_halfwidth=1.4, min_counts=max(5, PRODUCTION_STEPS // RECORD_STRIDE // 20),
    )
    ess_frac = agm.ess(w) / w.size
    return curvature, linear, resid, ess_frac, res["converged"]


def test_real_md_recovers_true_curvature(real_md_windows):
    """The control: real integrator + real umbrella force + real CV extraction +
    real reconstruct_bias_matrix + real MBAR must recover k0_true."""
    d = real_md_windows
    beta = _beta_kj_per_mol()
    curvature, linear, resid, ess_frac, converged = _recover_curvature(
        d["cv_a"], d["window"], d["windows"], beta
    )
    assert converged, "MBAR did not converge on real-MD samples"

    assert ess_frac > 0.05, (
        f"ESS fraction too low ({ess_frac:.2%}) - a passing curvature from a "
        f"badly-overlapping real-MD run is not validation"
    )

    rel_err = abs(curvature - K0_TRUE_KCAL_A2) / K0_TRUE_KCAL_A2
    assert rel_err < 0.20, (
        f"recovered curvature {curvature:.3f} vs true {K0_TRUE_KCAL_A2} "
        f"(rel err {rel_err:.1%}); real reconstruct_bias_matrix/MBAR path likely wrong"
    )

    # Fit is done in the well's own frame (cv_A - D0_A, see _recover_curvature),
    # so - same as the origin-centered L0 oracle - a large |linear| signals a
    # sign/centering bug, this time specifically in how window *positions*
    # (not just spring constants) were threaded through set_window/reconstruct.
    assert abs(linear) < 0.5, f"unexpected linear PMF term b={linear:.3f}"

    assert resid < 0.30, f"PMF shape residual too large: {resid:.3f} kcal/mol"


@pytest.mark.parametrize(
    "label,transform,analog",
    [
        ("half-scaled bias (k1 x0.5 for every window)",
         lambda windows: [dict(w, k1=w["k1"] * 0.5) for w in windows],
         "mis-scaled umbrella-force construction"),
        ("bias dropped (beta=0)", None, "reweighting term omitted entirely"),
        ("sign-flipped beta", "flip_beta", "sign error in the reduced potential"),
    ],
)
def test_mutation_is_caught(real_md_windows, label, transform, analog):
    """Ported directly from test_physics_oracle.py's mutation table: a passing
    control proves nothing until each targeted defect is shown to fail loudly
    on the SAME real MD data. No re-running MD - only the cheap NumPy
    construction+reweighting chain changes per mutation."""
    d = real_md_windows
    beta = _beta_kj_per_mol()
    windows = d["windows"]
    if transform == "flip_beta":
        beta = -beta
    elif transform is None:
        beta = 0.0
    else:
        windows = transform(windows)

    curvature, _linear, _resid, _ess, converged = _recover_curvature(
        d["cv_a"], d["window"], windows, beta
    )
    rel_err = abs(curvature - K0_TRUE_KCAL_A2) / K0_TRUE_KCAL_A2 if converged else float("inf")
    assert rel_err > 0.20, (
        f"mutation '{label}' (real-world analog: {analog}) was NOT caught - "
        f"recovered curvature {curvature:.3f} still within the 20% pass band "
        f"(rel err {rel_err:.1%}); the construction-guard chain isn't sensitive "
        f"to this defect class"
    )
