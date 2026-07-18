"""Tier A, 2-D extension: a coupled, multi-basin known potential.

test_thermodynamic_validity_real_md.py (Tier A / 1-D) validates the real
``reconstruct_bias_matrix`` construction-guard on a single harmonic well with
only a primary CV. ``reconstruct_bias_matrix`` also has a *second* branch -
``if cv2 is not None and "center2" in w and "k2" in w: ...`` - that adds a
secondary harmonic bias term. Nothing in the test suite exercised that branch,
or the matching secondary-CV branch in ``gareus.windows.set_window``
(``secondary_centers``/``secondary_ks_kj``), before this file. Both are
generic (CV-shape-agnostic) and reachable from real 2-D REUS runs
(``cv2: torsion-pca`` etc.), so a defect there would previously have gone
undetected.

The ground truth: two OpenMM particles pinned to independent axes off a
shared fixed anchor, so their anchor distances (r1, r2) are exact 1-D
flat-measure coordinates, same trick Tier A uses (see its own docstring on
the r^2-Jacobian question this sidesteps). The "true" potential coupling
them is a closed-form, non-separable double well:

    u = (r1 - R1_0) / L
    Y = r2 - R2_0
    V(r1, r2) = A*(u^2 - 1)^2 + 0.5*KY*Y^2 + C*u*Y

This is genuinely 2-D "complex" in the sense that matters for a validity
oracle: two minima (not one), connected through a real saddle, with a
bilinear u*Y coupling term that makes the surface non-separable (its
Hessian off-diagonal is a nonzero constant everywhere, C/L != 0) - so
recovering it can't be reduced to two independent 1-D problems.

Exact closed-form ground truth (derived once, cross-checked against a
finite-difference Hessian of this exact V during test design - see the
comment on ``_true_potential_basins``):
  * two minima, symmetric under (u, Y) -> (-u, -Y), hence exactly
    degenerate in energy (delta_G_true = 0 - a meaningful check on its own:
    a badly broken recovery pipeline would not reliably land near 0 here).
  * an identical local Hessian (Kxx, Kyy, Kxy) at both minima (by that same
    symmetry) - the primary quantitative check below.
  * the saddle sits exactly at (R1_0, R2_0), where the true potential's own
    Hessian is indefinite along u (that's what makes it a saddle). The
    real, per-window umbrella (K_BIAS) must dominate that negative
    curvature everywhere a window sits, or a single window could sample two
    basins at once and break the "one window, one local free-energy
    estimate" assumption REUS/MBAR stratification relies on. This was
    checked explicitly at design time: true+umbrella combined Hessian at
    the saddle is [[2.89, 0.67], [0.67, 13.0]] for the constants below -
    positive-definite, so every window in the grid (not just the two
    basins) samples a single, unimodal local distribution.

Real code exercised end-to-end, same list as Tier A plus one line:
  real integrator      gareus.system_setup.make_langevin_integrator
  real primary umbrella gareus.forces.add_umbrella_force            (CV1)
  real per-window set   gareus.windows.set_window                   (both CVs)
  real CV extraction    gareus.state.cv_distance_nm                 (both CVs)
  real bias-matrix      gareus.query.reconstruct_bias_matrix         <- cv2 branch, untested until now
  real reweighting      analyze_gareus_mbar.solve_mbar / norm_logw / ess

Deliberately NOT real production code, and why that's the right call here:
  * The secondary umbrella force (``_add_secondary_umbrella_force`` below)
    is a minimal hand-rolled clone of ``add_umbrella_force`` under the
    ``ss0``/``ss_k`` global-parameter names ``set_window`` already assigns
    for an optional secondary CV. No generic (non-torsion) secondary
    distance-umbrella builder exists in ``gareus.forces`` - production's
    real CV2 machinery (``gareus/production.py``'s inline
    ``cv_force.addGlobalParameter("ss_k"/"ss0", ...)`` calls) is
    torsion/rama-specific, which is periodic and would break the exact,
    flat-measure ground truth this whole design relies on. This file
    guards the generic 2-D reweighting/bias-matrix *path*, not the real
    (torsion-based) CV2 *force construction* - that would need its own,
    separately-scoped oracle with a periodic potential and a Jacobian
    correction, not a drop-in extension of this one.
  * ``k``/``r0`` and ``ss0``/``ss_k`` are genuinely separate OpenMM Context
    global parameters (not per-force), which is exactly why production
    needs the second name pair at all: two ``add_umbrella_force``-style
    forces sharing the same Context can't both use "k"/"r0" independently.
  * No replica exchange (REUS-only, matching Tier A) and no GaMD boost
    (same cumulant-expansion scope note as Tier A) - both out of scope.

Window grid: a dense, uniform 13x7 grid (91 windows) spanning both basins
plus a 2-sigma margin, sigma = sqrt(kT/K_BIAS) at the shared per-CV bias
stiffness. Spacing (~1.2-1.4 sigma on each axis) and stiffness were checked
against the saddle-point Hessian above specifically so every window -
including ones sitting near the saddle - is a stiff-spring/single-basin
stratification, not a barrier-crossing free-MD problem. At 3 particles and
Reference-platform MD this whole grid runs in single-digit seconds.

Per-basin curvature is recovered via a local weighted 2-D quadratic surface
fit (``_fit_2d_curvature``) over a real, jointly-solved 2-D PMF
(``_pmf2d_from_weights`` - the same weighted-histogram/-kT*ln(prob)
convention as the real ``analyze_gareus_mbar.pmf_from_weights``, just 2-D;
no 2-D PMF helper exists in the pipeline to reuse, same as Tier A's
``_fit_curvature`` was itself test-local verification code, not production
code). The fit half-width (0.4 A) was tuned empirically during test design:
wider windows pull in the double well's quartic curvature falloff toward
the saddle and systematically bias Kxx low (checked directly: 0.9 A gave a
~40% low bias, 0.4 A gave ~4%) - same "stay inside the harmonic regime"
lesson as any local curvature fit on a non-quadratic surface.

Mutation battery: the three ported from Tier A (mis-scaled primary bias,
bias dropped, bias sign-flipped) plus one new, CV2-specific defect (center2/
k2 stripped from the windows fed to reconstruct_bias_matrix, simulating "the
analysis forgot the secondary restraint term"). That fourth mutation is the
whole point of this file: verified during design that it does NOT
meaningfully perturb Kxx (the real MD still felt the real secondary bias;
only the primary-CV curvature stays cleanly recovered) but DOES corrupt Kyy
and Kxy sharply - so a test that only checked Kxx would silently miss a
dead CV2 reconstruction path. All three curvature checks (Kxx, Kyy, Kxy)
must be evaluated jointly for this file to have real teeth on the 2-D path.
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

from gareus.units import kcal_a2_to_kj_nm2, kcal_to_kj  # noqa: E402
from gareus.system_setup import make_langevin_integrator  # noqa: E402
from gareus.state import cv_distance_nm  # noqa: E402
from gareus.forces import add_umbrella_force  # noqa: E402
from gareus.windows import set_window  # noqa: E402
from gareus.query import reconstruct_bias_matrix  # noqa: E402
from test_thermodynamic_validity_real_md import _beta_kj_per_mol, _kt_kcal, TEMP_K  # noqa: E402

# --- ground-truth parameters, kcal/mol/A^2-family convention -----------------
# TEMP_K is imported (not redefined) so ground-truth beta and MD temperature
# can't silently diverge if one file's constant is edited without the other.
A_KCAL = 4.0                   # double-well quartic-coefficient (kcal/mol)
KY_KCAL_A2 = 3.0                # harmonic stiffness along the Y (CV2) arm
C_KCAL_A = 1.0                  # u-Y bilinear coupling (kcal/mol/A) - the non-separable term
L_A = 1.5                       # double-well length scale (A)
R1_0_A = 4.0                    # CV1 (X) arm shift, off the r=0 fold
R2_0_A = 3.0                    # CV2 (Y) arm shift, off the r=0 fold
K_BIAS_KCAL_A2 = 10.0            # shared per-window umbrella stiffness, both CVs
K_PIN_KCAL_A2 = 200.0            # transverse pin stiffness (both arms)
N1, N2 = 13, 7                  # window grid: (CV1, CV2) points -> 91 windows
EQUIL_STEPS = 2000
PRODUCTION_STEPS = 20000
RECORD_STRIDE = 20
TIMESTEP_FS = 0.5
FRICTION_PER_PS = 5.0
SEED = 12345
MOBILE_MASS_AMU = 12.0
FIT_HALFWIDTH_A = 0.4            # empirically tuned - see module docstring
MIN_2D_BIN_COUNTS = 5


def _true_potential_basins():
    """Exact stationary points of V(u, Y) = A*(u^2-1)^2 + 0.5*KY*Y^2 + C*u*Y.

    dV/dY = KY*Y + C*u = 0            -> Y*(u) = -C*u/KY   (valid for any u)
    dV/du = 4A*u*(u^2-1) + C*Y = 0, substituting Y*(u):
        u*[4A*(u^2-1) - C^2/KY] = 0
        -> u = 0 (the saddle) or u^2 = 1 + C^2/(4*A*KY)

    Both nonzero roots (+-u0) give exactly-degenerate minima under the
    V(u,Y) = V(-u,-Y) symmetry (checked: (u^2-1)^2 is even in u, the KY term
    is even in Y, and C*u*Y is even under simultaneous sign flip of both).
    Hessian (from direct differentiation, cross-checked against a finite
    difference of this exact formula at test-design time):
        Kxx = d^2V/dX^2 = 4*A*(3*u0^2 - 1) / L^2
        Kyy = d^2V/dY^2 = KY                      (Y is pure quadratic)
        Kxy = d^2V/dXdY = C / L                    (bilinear term -> constant)
    identical at both basins since they only depend on u0^2.
    """
    u0 = np.sqrt(1.0 + C_KCAL_A**2 / (4.0 * A_KCAL * KY_KCAL_A2))
    x_min = L_A * u0
    y_min = C_KCAL_A * u0 / KY_KCAL_A2
    r1_plus, r2_plus = R1_0_A + x_min, R2_0_A - y_min
    r1_minus, r2_minus = R1_0_A - x_min, R2_0_A + y_min
    kxx_true = 4.0 * A_KCAL * (3.0 * u0**2 - 1.0) / L_A**2
    kyy_true = KY_KCAL_A2
    kxy_true = C_KCAL_A / L_A
    return {
        "plus": (r1_plus, r2_plus),
        "minus": (r1_minus, r2_minus),
        "kxx": kxx_true,
        "kyy": kyy_true,
        "kxy": kxy_true,
    }


_BASINS = _true_potential_basins()
R1_PLUS, R2_PLUS = _BASINS["plus"]
R1_MINUS, R2_MINUS = _BASINS["minus"]
KXX_TRUE, KYY_TRUE, KXY_TRUE = _BASINS["kxx"], _BASINS["kyy"], _BASINS["kxy"]


def _add_secondary_umbrella_force(openmm, system, atom1: int, atom2: int, force_group: int = 23):
    """Minimal ``ss0``/``ss_k`` distance umbrella - see module docstring's
    "Deliberately NOT real production code" note for why this is a hand-
    rolled clone of ``add_umbrella_force`` rather than reused production
    code (no generic secondary-distance builder exists; production's real
    CV2 force construction is torsion-specific and periodic)."""
    force = openmm.CustomBondForce("0.5*ss_k*(r-ss0)^2")
    force.addGlobalParameter("ss_k", 0.0)
    force.addGlobalParameter("ss0", 0.0)
    force.addBond(int(atom1), int(atom2), [])
    force.setForceGroup(int(force_group))
    system.addForce(force)
    return force


def _build_system() -> "openmm.System":
    """Anchor (fixed) + two mobile particles, each pinned to its own axis, so
    their anchor distances are independent 1-D flat-measure coordinates. The
    true coupled double well and both transverse pins bake their constants
    directly into the force expression (no addGlobalParameter) so they can't
    collide with the umbrellas' own "k"/"r0" and "ss0"/"ss_k" namespaces.
    """
    system = openmm.System()
    system.addParticle(0.0)              # anchor: mass 0 -> fixed by OpenMM
    system.addParticle(MOBILE_MASS_AMU)  # CV1 (X) arm
    system.addParticle(MOBILE_MASS_AMU)  # CV2 (Y) arm

    l_nm = L_A * 0.1
    r1_0_nm = R1_0_A * 0.1
    r2_0_nm = R2_0_A * 0.1
    a_kj = kcal_to_kj(A_KCAL)
    ky_kj_nm2 = kcal_a2_to_kj_nm2(KY_KCAL_A2)
    # kcal/mol/A -> kJ/mol/nm: 4.184 converts kcal->kJ; dividing by 0.1
    # converts "per A" -> "per nm" (1/A = 10/nm). No gareus.units helper
    # exists for this specific per-length^1 combination (only the per-
    # length^2 case, kcal_a2_to_kj_nm2, does) - unlike that helper this one
    # isn't reused elsewhere, so it's kept local rather than added there.
    c_kj_nm = C_KCAL_A * 4.184 / 0.1

    true_potential_expr = (
        f"{a_kj:.17g}*(u^2-1)^2 + 0.5*{ky_kj_nm2:.17g}*y^2 + {c_kj_nm:.17g}*u*y; "
        f"u=(distance(p1,p2)-{r1_0_nm:.17g})/{l_nm:.17g}; "
        f"y=(distance(p1,p3)-{r2_0_nm:.17g})"
    )
    true_potential = openmm.CustomCompoundBondForce(3, true_potential_expr)
    true_potential.addBond([0, 1, 2], [])
    true_potential.setForceGroup(20)
    system.addForce(true_potential)

    k_pin_kj_nm2 = kcal_a2_to_kj_nm2(K_PIN_KCAL_A2)
    pin1 = openmm.CustomExternalForce(f"0.5*{k_pin_kj_nm2:.17g}*(y^2+z^2)")
    pin1.addParticle(1, [])
    pin1.setForceGroup(21)
    system.addForce(pin1)
    pin2 = openmm.CustomExternalForce(f"0.5*{k_pin_kj_nm2:.17g}*(x^2+z^2)")
    pin2.addParticle(2, [])
    pin2.setForceGroup(22)
    system.addForce(pin2)

    add_umbrella_force(openmm, system, 0, 1, force_group=31)
    _add_secondary_umbrella_force(openmm, system, 0, 2, force_group=23)
    return system


def _run_one_window(centers1_a: np.ndarray, centers2_a: np.ndarray, window_index: int):
    """Run real Langevin MD for one 2-D umbrella window; return (cv1, cv2) samples (A)."""
    system = _build_system()
    integrator = make_langevin_integrator(
        openmm, unit, types.SimpleNamespace(seed=SEED),
        timestep_fs=TIMESTEP_FS, temperature_k=TEMP_K, friction_per_ps=FRICTION_PER_PS,
        seed_offset=window_index,
    )
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))

    c1_nm = float(centers1_a[window_index]) * 0.1
    c2_nm = float(centers2_a[window_index]) * 0.1
    k_kj_nm2 = kcal_a2_to_kj_nm2(K_BIAS_KCAL_A2)
    set_window(
        context, [c1_nm], [k_kj_nm2], 0,
        secondary_centers=[c2_nm], secondary_ks_kj=[k_kj_nm2],
    )

    context.setPositions([
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(c1_nm, 0.0, 0.0),
        openmm.Vec3(0.0, c2_nm, 0.0),
    ])
    context.setVelocitiesToTemperature(TEMP_K * unit.kelvin, SEED + 1000 + window_index)

    integrator.step(EQUIL_STEPS)

    n_blocks = PRODUCTION_STEPS // RECORD_STRIDE
    cv1_a = np.empty(n_blocks, dtype=np.float64)
    cv2_a = np.empty(n_blocks, dtype=np.float64)
    for i in range(n_blocks):
        integrator.step(RECORD_STRIDE)
        cv1_a[i] = cv_distance_nm(context, 0, 1, unit) * 10.0
        cv2_a[i] = cv_distance_nm(context, 0, 2, unit) * 10.0
    return cv1_a, cv2_a


@pytest.fixture(scope="module")
def real_md_2d_windows():
    """Run the full 91-window grid once; every test in this module reuses
    the same real samples (mutations only re-run the cheap NumPy
    construction+reweighting chain, not MD)."""
    sigma = np.sqrt(_kt_kcal() / K_BIAS_KCAL_A2)
    r1_lo = min(R1_MINUS, R1_PLUS) - 2.0 * sigma
    r1_hi = max(R1_MINUS, R1_PLUS) + 2.0 * sigma
    r2_lo = min(R2_MINUS, R2_PLUS) - 2.0 * sigma
    r2_hi = max(R2_MINUS, R2_PLUS) + 2.0 * sigma
    centers1_1d = np.linspace(r1_lo, r1_hi, N1)
    centers2_1d = np.linspace(r2_lo, r2_hi, N2)
    assert min(r1_lo, r2_lo) > 0.5, "windows too close to the r=0 fold - widen R1_0_A/R2_0_A"

    grid = [(c1, c2) for c1 in centers1_1d for c2 in centers2_1d]
    centers1_a = np.array([g[0] for g in grid])
    centers2_a = np.array([g[1] for g in grid])

    cv1_parts, cv2_parts, win_parts = [], [], []
    windows = []
    for i, (c1, c2) in enumerate(grid):
        cv1_samples, cv2_samples = _run_one_window(centers1_a, centers2_a, i)
        cv1_parts.append(cv1_samples)
        cv2_parts.append(cv2_samples)
        win_parts.append(np.full(cv1_samples.shape, i, dtype=np.int64))
        windows.append({
            "center1": float(c1), "k1": K_BIAS_KCAL_A2,
            "center2": float(c2), "k2": K_BIAS_KCAL_A2,
        })
    cv1 = np.concatenate(cv1_parts)
    cv2 = np.concatenate(cv2_parts)
    window = np.concatenate(win_parts)
    return {"cv1": cv1, "cv2": cv2, "window": window, "windows": windows}


def _pmf2d_from_weights(cv1, cv2, w, bins1, bins2, kbt_kcal):
    """2-D analog of analyze_gareus_mbar.pmf_from_weights: same weighted-
    histogram / normalize / -kT*ln(prob) / shift-min-to-0 convention, just
    over a joint (cv1, cv2) histogram. No 2-D PMF helper exists in the
    pipeline - this is test-local verification code, same role as
    test_physics_oracle.py's own _fit_curvature."""
    prob, e1, e2 = np.histogram2d(cv1, cv2, bins=[bins1, bins2], weights=w)
    counts, _, _ = np.histogram2d(cv1, cv2, bins=[bins1, bins2])
    prob = np.asarray(prob, dtype=np.float64)
    if prob.sum() > 0:
        prob /= prob.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmf = -kbt_kcal * np.log(prob)
    finite = np.isfinite(pmf)
    if np.any(finite):
        pmf -= np.nanmin(pmf[finite])
    centers1 = 0.5 * (e1[:-1] + e1[1:])
    centers2 = 0.5 * (e2[:-1] + e2[1:])
    grid1, grid2 = np.meshgrid(centers1, centers2, indexing="ij")
    return grid1, grid2, pmf, counts.astype(int)


def _fit_2d_curvature(x, y, pmf, counts, center1, center2, halfwidth, min_counts):
    """Weighted local 2-D quadratic-surface fit around (center1, center2):
    F(dx,dy) ~ c + b1*dx + b2*dy + 0.5*(Kxx*dx^2 + 2*Kxy*dx*dy + Kyy*dy^2).
    Returns None if too few bins survive the mask (caller decides how to
    treat that - a mutation that guts overlap enough to starve a basin's
    own fit is itself a form of "caught")."""
    dx = x - center1
    dy = y - center2
    mask = (np.abs(dx) <= halfwidth) & (np.abs(dy) <= halfwidth) & (counts > min_counts) & np.isfinite(pmf)
    n = int(mask.sum())
    if n < 8:
        return None
    dx_m, dy_m, f_m = dx[mask], dy[mask], pmf[mask]
    w_m = np.sqrt(counts[mask].astype(np.float64))  # PMF bin error ~ 1/sqrt(count)
    design = np.stack([np.ones_like(dx_m), dx_m, dy_m, dx_m**2, dx_m * dy_m, dy_m**2], axis=1)
    coeffs, *_ = np.linalg.lstsq(design * w_m[:, None], f_m * w_m, rcond=None)
    _c, b1, b2, a_xx, a_xy, a_yy = coeffs
    resid = f_m - design @ coeffs
    return {
        "kxx": 2.0 * a_xx, "kyy": 2.0 * a_yy, "kxy": a_xy,
        "b1": float(b1), "b2": float(b2),
        "resid": float(np.sqrt(np.mean(resid * resid))), "n": n,
    }


def _recover_2d(cv1, cv2, window, windows, beta):
    """Real reconstruct_bias_matrix (cv2 branch) + real solve_mbar, then the
    test-local 2-D PMF/fit helpers above. Returns per-basin fits, their
    average (both basins are analytically equivalent - see module
    docstring), the recovered basin free-energy gap, and MBAR diagnostics."""
    u_nk = reconstruct_bias_matrix(cv1, cv2, windows, beta)
    res = agm.solve_mbar(u_nk, window, backend="anderson", tol=1e-10)
    w = agm.norm_logw(res["logw"])
    ess_frac = agm.ess(w) / w.size

    bins1 = np.linspace(cv1.min() - 0.2, cv1.max() + 0.2, 91)
    bins2 = np.linspace(cv2.min() - 0.2, cv2.max() + 0.2, 61)
    grid1, grid2, pmf, counts = _pmf2d_from_weights(cv1, cv2, w, bins1, bins2, _kt_kcal())

    fit_plus = _fit_2d_curvature(grid1, grid2, pmf, counts, R1_PLUS, R2_PLUS, FIT_HALFWIDTH_A, MIN_2D_BIN_COUNTS)
    fit_minus = _fit_2d_curvature(grid1, grid2, pmf, counts, R1_MINUS, R2_MINUS, FIT_HALFWIDTH_A, MIN_2D_BIN_COUNTS)

    def _pmf_nearest(center1, center2):
        d2 = (grid1 - center1) ** 2 + (grid2 - center2) ** 2
        d2 = np.where(counts > MIN_2D_BIN_COUNTS, d2, np.inf)
        idx = np.unravel_index(np.argmin(d2), d2.shape)
        value = pmf[idx]
        return float(value) if np.isfinite(value) else float("nan")

    delta_g = _pmf_nearest(R1_PLUS, R2_PLUS) - _pmf_nearest(R1_MINUS, R2_MINUS)

    avg = None
    if fit_plus is not None and fit_minus is not None:
        avg = {
            "kxx": 0.5 * (fit_plus["kxx"] + fit_minus["kxx"]),
            "kyy": 0.5 * (fit_plus["kyy"] + fit_minus["kyy"]),
            "kxy": 0.5 * (fit_plus["kxy"] + fit_minus["kxy"]),
        }

    return {
        "converged": res["converged"], "ess_frac": ess_frac,
        "fit_plus": fit_plus, "fit_minus": fit_minus, "avg": avg, "delta_g": delta_g,
    }


def test_real_md_recovers_coupled_basins(real_md_2d_windows):
    """The control: real integrator + real primary/secondary umbrella forces
    + real CV extraction + real reconstruct_bias_matrix (both branches) +
    real MBAR must recover both basins' curvature and their (here: zero)
    free-energy gap."""
    d = real_md_2d_windows
    beta = _beta_kj_per_mol()
    result = _recover_2d(d["cv1"], d["cv2"], d["window"], d["windows"], beta)

    assert result["converged"], "MBAR did not converge on real 2-D MD samples"
    assert result["ess_frac"] > 0.15, (
        f"ESS fraction too low ({result['ess_frac']:.2%}) - a passing fit from a "
        f"badly-overlapping real-MD 2-D grid is not validation"
    )
    assert result["avg"] is not None, "per-basin 2-D curvature fit failed (too few usable bins)"

    for label, recovered, true_value, tol in (
        ("Kxx", result["avg"]["kxx"], KXX_TRUE, 0.20),
        ("Kyy", result["avg"]["kyy"], KYY_TRUE, 0.20),
        ("Kxy", result["avg"]["kxy"], KXY_TRUE, 0.35),
    ):
        rel_err = abs(recovered - true_value) / true_value
        assert rel_err < tol, (
            f"recovered {label}={recovered:.3f} vs true {true_value:.3f} "
            f"(rel err {rel_err:.1%}); real 2-D reconstruct_bias_matrix/MBAR path likely wrong"
        )

    # Both basins are exactly degenerate by construction (delta_G_true = 0).
    # Loose, kT-scale band: this is a diagnostic sanity check, not the primary
    # discriminator (that's the three curvature checks above and the
    # mutation battery below).
    dg_over_kt = abs(result["delta_g"]) / _kt_kcal()
    assert dg_over_kt < 0.5, (
        f"recovered basin free-energy gap {result['delta_g']:.3f} kcal/mol "
        f"({dg_over_kt:.2f} kT) - true gap is exactly 0 by symmetry"
    )

    for label, fit in (("plus", result["fit_plus"]), ("minus", result["fit_minus"])):
        assert fit["resid"] < 0.3, f"basin-{label} 2-D PMF fit residual too large: {fit['resid']:.3f} kcal/mol"


# Each mutation names the SPECIFIC metric(s) its own defect is expected to
# corrupt (verified empirically at design time), not "any of Kxx/Kyy/Kxy" -
# a blanket "any" would let e.g. the cv2-stripped case pass via an unrelated
# Kxx deviation without ever actually checking that Kyy/Kxy are what broke.
@pytest.mark.parametrize(
    "label,transform,analog,expected_metrics",
    [
        ("half-scaled primary bias (k1 x0.5 for every window)",
         lambda windows: [dict(w, k1=w["k1"] * 0.5) for w in windows],
         "mis-scaled primary umbrella-force construction",
         ("Kxx",)),
        ("bias dropped (beta=0)", None, "reweighting term omitted entirely",
         ("Kxx", "Kyy", "Kxy")),
        ("sign-flipped beta", "flip_beta", "sign error in the reduced potential",
         ("Kxx", "Kyy", "Kxy")),
        ("secondary bias stripped (center2/k2 removed from windows)",
         lambda windows: [{"center1": w["center1"], "k1": w["k1"]} for w in windows],
         "reconstruct_bias_matrix's cv2 branch silently skipped",
         ("Kyy", "Kxy")),
    ],
)
def test_mutation_is_caught(real_md_2d_windows, label, transform, analog, expected_metrics):
    """Each defect is applied to the SAME real 2-D MD data (no re-running MD)
    and must push its OWN claimed metric(s) outside the control pass band -
    not just some metric. The fourth mutation is the one this file exists
    for: verified at design time that stripping center2/k2 leaves Kxx (the
    primary-CV curvature) nearly untouched - the real MD still felt the real
    secondary bias - but corrupts Kyy and Kxy sharply (the positive half of
    that claim, that Kxx stays clean, is checked separately by
    test_cv2_stripped_leaves_primary_curvature_clean below).
    """
    d = real_md_2d_windows
    beta = _beta_kj_per_mol()
    windows = d["windows"]
    if transform == "flip_beta":
        beta = -beta
    elif transform is None:
        beta = 0.0
    else:
        windows = transform(windows)

    result = _recover_2d(d["cv1"], d["cv2"], d["window"], windows, beta)

    if result["avg"] is None:
        # A total fit breakdown (too few usable 2-D bins in a basin) is still
        # a valid form of "caught", but a materially different one than a
        # curvature deviation - skip (visibly distinct in pytest output)
        # rather than silently returning, so this path can never look
        # identical to a real curvature-based catch.
        pytest.skip(
            f"mutation '{label}' broke 2-D curvature recovery outright "
            f"(too few usable bins survived in a basin) - caught by total "
            f"breakdown rather than by a curvature-deviation threshold"
        )

    rel_errs = {
        "Kxx": (abs(result["avg"]["kxx"] - KXX_TRUE) / KXX_TRUE, 0.20),
        "Kyy": (abs(result["avg"]["kyy"] - KYY_TRUE) / KYY_TRUE, 0.20),
        "Kxy": (abs(result["avg"]["kxy"] - KXY_TRUE) / KXY_TRUE, 0.35),
    }
    caught = any(rel_errs[m][0] > rel_errs[m][1] for m in expected_metrics)
    detail = ", ".join(f"{k}={v:.1%}" for k, (v, _tol) in rel_errs.items())
    assert caught, (
        f"mutation '{label}' (real-world analog: {analog}) was NOT caught via "
        f"its claimed mechanism ({', '.join(expected_metrics)}) - all metrics: "
        f"{detail}; the 2-D construction-guard chain isn't sensitive to this "
        f"defect class the way this test claims"
    )


def test_cv2_stripped_leaves_primary_curvature_clean(real_md_2d_windows):
    """The cv2-stripped mutation's claim (see test_mutation_is_caught's
    docstring) is two-sided: Kyy/Kxy break (checked above), but Kxx - the
    primary-CV curvature, which the stripped secondary bias term never
    touches - should stay within the same pass band the control test uses.
    Only asserting "some metric broke" (as test_mutation_is_caught now does,
    restricted to Kyy/Kxy) would not verify this other half of the claim;
    this test checks it directly and positively.
    """
    d = real_md_2d_windows
    beta = _beta_kj_per_mol()
    windows = [{"center1": w["center1"], "k1": w["k1"]} for w in d["windows"]]
    result = _recover_2d(d["cv1"], d["cv2"], d["window"], windows, beta)
    assert result["avg"] is not None, "cv2-stripped mutation broke curvature recovery outright"
    rel_err_kxx = abs(result["avg"]["kxx"] - KXX_TRUE) / KXX_TRUE
    assert rel_err_kxx < 0.20, (
        f"Kxx should stay clean when only the secondary bias is stripped from "
        f"reconstruction (rel err {rel_err_kxx:.1%}) - if this fails, dropping "
        f"center2/k2 is somehow corrupting the PRIMARY reconstruction too, a "
        f"different and more serious bug than the one this mutation targets"
    )
