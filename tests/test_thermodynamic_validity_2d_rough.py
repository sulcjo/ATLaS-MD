"""Tier A, 2-D extension #2: a noisy, multi-barrier known potential with
false minima and peaky maxima.

test_thermodynamic_validity_2d.py already validates the real 2-D
reconstruct_bias_matrix/set_window path on a single coupled double well (2
basins, smooth, curvature-fit as the recovery metric). This file asks a
harder question: does the SAME real pipeline still recover a genuinely rough
free-energy surface - textured noise, sharp isolated peaks, and spurious
local minima that are not part of the "real" basins - not just a clean
quadratic-ish landscape?

Ground truth is still exact and closed-form (same flat-measure-CV trick as
every file in this family: two mobile particles pinned to independent axes
off a shared anchor, so their anchor distances are exact 1-D coordinates).
What changed is the METRIC, not the trick: with real sub-basin structure,
per-basin curvature fitting would average the roughness away and check
nothing about it. Instead this file compares the recovered 2-D PMF to the
known V(x, y) POINTWISE, bin by bin, over the whole sampled domain - the
same flat-measure "PMF == potential" identity the curvature-fit files rely
on, just checked directly instead of via a local quadratic proxy. That
directly tests every feature the surface has: noise, false minima, peaky
maxima, and the extra barriers between them.

V(r1, r2) = V_backbone(r1, r2) + V_noise(r1, r2) + sum_i V_bump_i(r1, r2)

  V_backbone: the exact same coupled double well as test_thermodynamic_
              validity_2d.py (2 real basins through a real saddle) -
              reused, not redesigned, so this file is purely additive.
  V_noise:    three fixed, incommensurate-period cosine terms, sub-kT
              amplitude - textural roughness that perturbs the surface
              everywhere without creating new stationary points on its own.
  V_bump_i:   six fixed, isolated Gaussian bumps (three positive = peaky
              maxima, three negative = false minima), consequential
              amplitude (~2.5-3.4 kT) and narrow width (0.35 A, sharper
              than the backbone's own ~0.26 A curvature scale) - real
              extra stationary points of the true potential, not noise
              that averages out. All constants are fixed literals (no
              real randomness anywhere) - "noisy-looking", not random.

Three-scale budget (checked at design time, not incidentally): for a window
grid to correctly sample this surface without any window going bimodal, the
per-window umbrella must dominate the sharpest NEGATIVE curvature the true
surface produces anywhere a window sits - including at a peaky maximum's own
center, where the bump alone contributes Hessian eigenvalues of
-amp/w^2 = -1.8/0.35^2 ~ -14.7 kcal/mol/A^2 in isotropic directions, on top
of the backbone's own saddle instability nearby. This was checked directly
(finite-difference Hessian at every bump center, the backbone saddle, both
basins, AND a fine sweep of the whole domain, not just the "obvious" points -
the actual worst case turned up at a bump's shoulder, not its center) before
picking K_BIAS: the combined true+umbrella Hessian must be positive-definite
everywhere sampled. K_BIAS=35 was the smallest value clearing that sweep
with a comfortable margin (worst-case min eigenvalue +7.1, vs +2.1 at
K_BIAS=30 and negative at K_BIAS<=25). That, in turn, sets sigma = sqrt(kT/
K_BIAS) ~ 0.13 A, which sets the required window spacing (~0.15 A, dense
enough to resolve the 0.35 A bumps) and hence the window count (26x14=364 -
about 4x test_thermodynamic_validity_2d.py's grid, for a domain of
comparable size, because "peaky" forces both stiffer umbrellas and finer
spacing at once).

MBAR backend: "numba-anderson", not the "anderson" (pure-NumPy) backend the
sibling files use - a K=364, N=182,000 problem takes minutes in pure NumPy
and seconds with the Numba kernels. Real solve_mbar code either way; this is
a backend choice, not a shortcut.

Deliberately NOT covered, same scope boundaries as the rest of this family:
no torsion-based real CV2 force construction (periodic, would break the
flat-measure ground truth - see test_thermodynamic_validity_2d.py's longer
note on this), no replica exchange, no GaMD boost.

Runtime note: this file is the heaviest oracle in the family (~150s: ~30s
real MD across 364 windows, plus MBAR+histogram+pointwise-comparison run
five times - once for the control, once per mutation). That cost is the
direct, load-bearing consequence of "peaky" (see the budget paragraph
above), not padding.
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
# numba isn't a declared dependency anywhere in pyproject.toml (analyze_gareus_mbar.py
# itself treats it as optional, NumPy remaining the fallback) - but MBAR_BACKEND below
# is pinned to "numba-anderson" (not "auto"), which raises RuntimeError with no
# fallback if numba is missing. Skip before paying for the ~30s real-MD fixture
# rather than running it and then hard-failing all 5 tests.
pytest.importorskip("numba", reason="MBAR_BACKEND='numba-anderson' requires numba")

from gareus.units import kcal_a2_to_kj_nm2, kcal_to_kj  # noqa: E402
from gareus.system_setup import make_langevin_integrator  # noqa: E402
from gareus.state import cv_distance_nm  # noqa: E402
from gareus.forces import add_umbrella_force  # noqa: E402
from gareus.windows import set_window  # noqa: E402
from gareus.query import reconstruct_bias_matrix  # noqa: E402
from test_thermodynamic_validity_real_md import _beta_kj_per_mol, _kt_kcal, TEMP_K  # noqa: E402
from test_thermodynamic_validity_2d import _add_secondary_umbrella_force  # noqa: E402

# --- backbone: identical constants to test_thermodynamic_validity_2d.py -----
A_KCAL = 4.0
KY_KCAL_A2 = 3.0
C_KCAL_A = 1.0
L_A = 1.5
R1_0_A = 4.0
R2_0_A = 3.0

# --- textural noise: fixed, incommensurate-period cosines, sub-kT amplitude -
NOISE_TERMS = (
    # (amplitude_kcal, period_a, phase, cv_combo)  cv_combo: 1 -> r1, 2 -> r2, 3 -> r1+0.5*r2
    (0.15, 0.83, 0.0, 1),
    (0.12, 0.61, 0.7, 2),
    (0.10, 0.97, 1.9, 3),
)

# --- consequential, isolated bumps: (amplitude_kcal, x0_a, y0_a, width_a) ---
# Positive = peaky maxima, negative = false minima. Positions were chosen to
# sit away from the two backbone basins AND away from the backbone saddle
# (4.0, 3.0) - stacking a peaky maximum directly on the saddle's own existing
# instability was the first thing tried and it broke the Hessian budget at
# every K_BIAS tried up to 45; see the module docstring's budget paragraph.
BUMPS = (
    (+1.8, 3.2, 2.55, 0.35),
    (+1.5, 4.6, 3.45, 0.35),
    (+2.0, 4.5, 2.55, 0.35),
    (-1.6, 2.9, 3.55, 0.35),
    (-1.7, 5.1, 2.45, 0.35),
    (-1.4, 3.5, 3.45, 0.35),
)

K_BIAS_KCAL_A2 = 35.0
K_PIN_KCAL_A2 = 200.0
EQUIL_STEPS = 1000
PRODUCTION_STEPS = 10000
RECORD_STRIDE = 20
TIMESTEP_FS = 0.5
FRICTION_PER_PS = 5.0
SEED = 12345
MOBILE_MASS_AMU = 12.0
GRID_SPACING_TARGET_A = 0.15
GRID_MARGIN_SIGMAS = 3.0
MIN_2D_BIN_COUNTS = 5
MBAR_BACKEND = "numba-anderson"


def true_potential_kcal(r1_a, r2_a):
    """Exact ground truth, in the SAME literal constants baked into the
    OpenMM force expression below (cross-checked at design time: OpenMM's
    own potential energy at 30 random points matched this function to
    ~1e-14 kcal/mol, floating-point noise, not approximation error)."""
    r1_a = np.asarray(r1_a, dtype=np.float64)
    r2_a = np.asarray(r2_a, dtype=np.float64)
    u = (r1_a - R1_0_A) / L_A
    y = r2_a - R2_0_A
    v = A_KCAL * (u**2 - 1.0) ** 2 + 0.5 * KY_KCAL_A2 * y**2 + C_KCAL_A * u * y
    cv_combo_values = {1: r1_a, 2: r2_a, 3: r1_a + 0.5 * r2_a}
    for amp, period, phase, combo in NOISE_TERMS:
        v = v + amp * np.cos(2.0 * np.pi * cv_combo_values[combo] / period + phase)
    for amp, x0, y0, w in BUMPS:
        v = v + amp * np.exp(-((r1_a - x0) ** 2 + (r2_a - y0) ** 2) / (2.0 * w**2))
    return v


def _build_true_potential_expr() -> str:
    """OpenMM CustomCompoundBondForce expression for true_potential_kcal,
    every constant converted to kJ/mol-nm and baked in as a literal - same
    pattern as every other force in this file family."""
    l_nm, r1_0_nm, r2_0_nm = L_A * 0.1, R1_0_A * 0.1, R2_0_A * 0.1
    a_kj = kcal_to_kj(A_KCAL)
    ky_kj_nm2 = kcal_a2_to_kj_nm2(KY_KCAL_A2)
    c_kj_nm = C_KCAL_A * 4.184 / 0.1  # kcal/mol/A -> kJ/mol/nm, see test_thermodynamic_validity_2d.py

    cv_combo_exprs = {
        1: "distance(p1,p2)",
        2: "distance(p1,p3)",
        3: "(distance(p1,p2)+0.5*distance(p1,p3))",
    }
    terms = [f"{a_kj:.17g}*(u^2-1)^2", f"0.5*{ky_kj_nm2:.17g}*y^2", f"{c_kj_nm:.17g}*u*y"]
    for amp, period, phase, combo in NOISE_TERMS:
        amp_kj, period_nm = kcal_to_kj(amp), period * 0.1
        terms.append(f"{amp_kj:.17g}*cos(2*3.14159265358979*{cv_combo_exprs[combo]}/{period_nm:.17g}+{phase:.17g})")
    for amp, x0, y0, w in BUMPS:
        amp_kj, x0_nm, y0_nm, w_nm = kcal_to_kj(amp), x0 * 0.1, y0 * 0.1, w * 0.1
        terms.append(
            f"{amp_kj:.17g}*exp(-((distance(p1,p2)-{x0_nm:.17g})^2+"
            f"(distance(p1,p3)-{y0_nm:.17g})^2)/(2*{w_nm:.17g}^2))"
        )
    return " + ".join(terms) + f"; u=(distance(p1,p2)-{r1_0_nm:.17g})/{l_nm:.17g}; y=(distance(p1,p3)-{r2_0_nm:.17g})"


def _build_system() -> "openmm.System":
    system = openmm.System()
    system.addParticle(0.0)
    system.addParticle(MOBILE_MASS_AMU)
    system.addParticle(MOBILE_MASS_AMU)

    true_potential = openmm.CustomCompoundBondForce(3, _build_true_potential_expr())
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
    set_window(context, [c1_nm], [k_kj_nm2], 0, secondary_centers=[c2_nm], secondary_ks_kj=[k_kj_nm2])

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


def _window_grid():
    """Dense uniform grid spanning both backbone basins and all 6 bumps, plus
    a sigma-scaled margin - see module docstring's three-scale budget for
    why the spacing/K_BIAS pair is what it is."""
    sigma = np.sqrt(_kt_kcal() / K_BIAS_KCAL_A2)
    u0 = np.sqrt(1.0 + C_KCAL_A**2 / (4.0 * A_KCAL * KY_KCAL_A2))
    x_min, y_min = L_A * u0, C_KCAL_A * u0 / KY_KCAL_A2
    basins_r1 = [R1_0_A + x_min, R1_0_A - x_min]
    basins_r2 = [R2_0_A - y_min, R2_0_A + y_min]
    r1_feats = basins_r1 + [b[1] for b in BUMPS]
    r2_feats = basins_r2 + [b[2] for b in BUMPS]
    margin = GRID_MARGIN_SIGMAS * sigma
    r1_lo, r1_hi = min(r1_feats) - margin, max(r1_feats) + margin
    r2_lo, r2_hi = min(r2_feats) - margin, max(r2_feats) + margin
    assert min(r1_lo, r2_lo) > 0.5, "windows too close to the r=0 fold - widen R1_0_A/R2_0_A or shrink margin"
    n1 = int(round((r1_hi - r1_lo) / GRID_SPACING_TARGET_A)) + 1
    n2 = int(round((r2_hi - r2_lo) / GRID_SPACING_TARGET_A)) + 1
    return np.linspace(r1_lo, r1_hi, n1), np.linspace(r2_lo, r2_hi, n2)


@pytest.fixture(scope="module")
def real_md_2d_rough_windows():
    """Run the full window grid once; every test in this module reuses the
    same real samples (mutations only re-run the cheap NumPy construction+
    reweighting chain, not MD)."""
    centers1_1d, centers2_1d = _window_grid()
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
    return {
        "cv1": np.concatenate(cv1_parts), "cv2": np.concatenate(cv2_parts),
        "window": np.concatenate(win_parts), "windows": windows,
    }


def _pmf2d_from_weights(cv1, cv2, w, bins1, bins2, kbt_kcal):
    """Adapted from test_thermodynamic_validity_2d.py's helper of the same
    name (duplicated, not imported - this file's comparison logic is
    pointwise, not a local fit, different enough that sharing a helper
    module would add an import for one function with no other shared
    state). Unlike the sibling's version, this one does NOT shift pmf so
    its own minimum is 0 - _pointwise_recovery below re-aligns both pmf and
    the true potential to their minimum over the SAME adequately-sampled
    mask anyway, so an unconditional shift here would just be immediately
    overwritten, not a missing step."""
    prob, e1, e2 = np.histogram2d(cv1, cv2, bins=[bins1, bins2], weights=w)
    counts, _, _ = np.histogram2d(cv1, cv2, bins=[bins1, bins2])
    prob = np.asarray(prob, dtype=np.float64)
    if prob.sum() > 0:
        prob /= prob.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmf = -kbt_kcal * np.log(prob)
    centers1 = 0.5 * (e1[:-1] + e1[1:])
    centers2 = 0.5 * (e2[:-1] + e2[1:])
    grid1, grid2 = np.meshgrid(centers1, centers2, indexing="ij")
    return grid1, grid2, pmf, counts.astype(int)


def _pointwise_recovery(cv1, cv2, window, windows, beta, min_counts=MIN_2D_BIN_COUNTS):
    """Real reconstruct_bias_matrix (cv2 branch) + real solve_mbar, then a
    direct pointwise comparison of the recovered 2-D PMF against the exact
    known V(x, y) - the flat-measure "PMF == potential" identity this whole
    file family relies on, checked directly instead of via a local
    curvature proxy (see module docstring for why that switch matters
    here). Both surfaces are aligned to their own minimum over the SAME
    adequately-sampled mask, so an unsampled high-V region (the peaky
    maxima are, by construction, the least-visited part of the domain -
    see the module docstring) can't bias the comparison.
    """
    u_nk = reconstruct_bias_matrix(cv1, cv2, windows, beta)
    res = agm.solve_mbar(u_nk, window, backend=MBAR_BACKEND, tol=1e-10)
    w = agm.norm_logw(res["logw"])
    ess_frac = agm.ess(w) / w.size

    bins1 = np.linspace(cv1.min() - 0.15, cv1.max() + 0.15, 121)
    bins2 = np.linspace(cv2.min() - 0.15, cv2.max() + 0.15, 71)
    grid1, grid2, pmf, counts = _pmf2d_from_weights(cv1, cv2, w, bins1, bins2, _kt_kcal())

    mask = (counts > min_counts) & np.isfinite(pmf)
    n_used, n_total = int(mask.sum()), int(pmf.size)
    v_true = true_potential_kcal(grid1, grid2)
    f_aligned = pmf[mask] - pmf[mask].min()
    v_aligned = v_true[mask] - v_true[mask].min()
    resid = f_aligned - v_aligned
    rms = float(np.sqrt(np.mean(resid**2)))

    return {
        "converged": res["converged"], "ess_frac": ess_frac, "rms": rms,
        "n_used": n_used, "n_total": n_total, "frac_used": n_used / n_total,
    }


def test_real_md_recovers_noisy_multibarrier_surface(real_md_2d_rough_windows):
    """The control: real integrator + real primary/secondary umbrella forces
    + real CV extraction + real reconstruct_bias_matrix (both branches) +
    real MBAR must recover the FULL noisy, multi-barrier, false-minima,
    peaky-maxima surface pointwise - not just its coarse basin curvature."""
    d = real_md_2d_rough_windows
    beta = _beta_kj_per_mol()
    result = _pointwise_recovery(d["cv1"], d["cv2"], d["window"], d["windows"], beta)

    assert result["converged"], "MBAR did not converge on real noisy-2D MD samples"
    assert result["ess_frac"] > 0.10, (
        f"ESS fraction too low ({result['ess_frac']:.2%}) on a landscape this rough - "
        f"a passing pointwise match from badly-overlapping real MD is not validation"
    )
    assert result["frac_used"] > 0.40, (
        f"only {result['frac_used']:.1%} of 2-D bins had enough samples to compare - "
        f"too little of the surface was actually checked"
    )
    assert result["rms"] < 0.5, (
        f"recovered PMF disagrees with the known noisy/multi-barrier/false-minima/"
        f"peaky-maxima surface by RMS={result['rms']:.3f} kcal/mol pointwise; "
        f"real reconstruct_bias_matrix/MBAR path likely wrong on a rough landscape"
    )


@pytest.mark.parametrize(
    "label,transform,analog",
    [
        ("half-scaled primary bias (k1 x0.5 for every window)",
         lambda windows: [dict(w, k1=w["k1"] * 0.5) for w in windows],
         "mis-scaled primary umbrella-force construction"),
        ("bias dropped (beta=0)", None, "reweighting term omitted entirely"),
        ("sign-flipped beta", "flip_beta", "sign error in the reduced potential"),
        ("secondary bias stripped (center2/k2 removed from windows)",
         lambda windows: [{"center1": w["center1"], "k1": w["k1"]} for w in windows],
         "reconstruct_bias_matrix's cv2 branch silently skipped"),
    ],
)
def test_mutation_is_caught(real_md_2d_rough_windows, label, transform, analog):
    """Each defect is applied to the SAME real MD data (no re-running MD) and
    must push the pointwise RMS well outside the control's pass band. Margins
    here are large by construction: the control recovers RMS ~0.16 kcal/mol
    and every one of these four mutations was verified at design time to land
    at RMS >= 1.3 kcal/mol (8x-400x the control), so 0.5 kcal/mol is a
    comfortable, non-brittle line between them.
    """
    d = real_md_2d_rough_windows
    beta = _beta_kj_per_mol()
    windows = d["windows"]
    if transform == "flip_beta":
        beta = -beta
    elif transform is None:
        beta = 0.0
    else:
        windows = transform(windows)

    result = _pointwise_recovery(d["cv1"], d["cv2"], d["window"], windows, beta)
    assert result["rms"] > 0.5, (
        f"mutation '{label}' (real-world analog: {analog}) was NOT caught - "
        f"pointwise RMS={result['rms']:.3f} kcal/mol stayed within the control's "
        f"pass band; the noisy-2D construction-guard chain isn't sensitive to "
        f"this defect class"
    )
