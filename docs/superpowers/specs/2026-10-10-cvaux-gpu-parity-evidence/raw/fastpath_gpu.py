"""Fast-path GPU aux z parity: CustomCVForce value (getCollectiveVariableValues) vs NumPy z from the same positions.

Run from the CODE_DIR root:  PYTHONPATH=.:tests python scripts/cvaux_parity/fastpath_gpu.py OUT.json
Fixture: solvated GA dipeptide (tests/pep_gamd_fixture), model as in
tests/test_aux_cv_observation.py (random coefficients, offset -0.3; plain, mixed conventions, scale 3.181).
An ACTIVE restraint (k = K_KCAL, centre = initial z) is applied so the GPU force acts, then Langevin MD;
at each sample both observation paths are read from one Context state.
"""
import json
import sys
import types

import numpy as np

from aux_cv_fixture import dipeptide, model_payload
from pep_gamd_fixture import _fresh_system
from gareus.auxiliary_cv.force import set_aux_parameters
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.offline import parity_violation
from gareus.auxiliary_cv.runtime import add_aux_cv_force, observe_aux_z
from gareus.auxiliary_cv.sample_schema import PARITY_TOLERANCE
from gareus.auxiliary_cv.state_table import AuxStateTable

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)
K_KCAL = 1.2            # W24's k_max
T_K = 300.0
BETA = 1.0 / (0.0083144626 * T_K)
import os
N_SAMPLES, STRIDE = int(os.environ.get("FP_N", 200)), 50
KJ = 4.184


def build_model(variant):
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    conv = (["negated" if k % 2 == 0 else "direct" for k in range(len(d["quads"]))]
            if variant == "mixed_conventions" else None)
    scale = 3.181 if variant == "scale_3.181" else 1.0
    rng = np.random.default_rng(5)
    return d, AuxModel.from_mapping(model_payload(d["quads"], rng.normal(size=2 * len(d["quads"])), offset=-0.3,
                                                  conventions=conv, blocks=blocks, scale=scale))


def run(platform_name, precision, variant):
    import openmm as mm
    d, model = build_model(variant)
    system = _fresh_system()
    rt = add_aux_cv_force(mm, system, AuxStateTable(model, (0.0,), (0.0,), (None,)), ARGS)
    force = system.getForce(rt.force_index)
    integ = mm.LangevinMiddleIntegrator(T_K, 1.0, 0.002)
    integ.setRandomNumberSeed(11)
    plat = mm.Platform.getPlatformByName(platform_name)
    props = {"Precision": precision} if platform_name in ("CUDA", "OpenCL") else {}
    ctx = mm.Context(system, integ, plat, props)
    ctx.setPositions(d["positions_nm"])
    mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 200)
    ctx.setVelocitiesToTemperature(T_K, 3)
    pos0 = ctx.getState(getPositions=True).getPositions(asNumpy=True)._value
    z0 = observe_aux_z(ctx, rt, positions_nm=pos0)
    set_aux_parameters(ctx, rt.info, center=z0, k_kcal=K_KCAL)
    group = rt.info.force_group
    fast, slow, e_gpu = [], [], []
    for _ in range(N_SAMPLES):
        integ.step(STRIDE)
        st = ctx.getState(getPositions=True, getEnergy=True, groups={group})
        pos = st.getPositions(asNumpy=True)._value
        fast.append(observe_aux_z(ctx, rt, force=force))
        slow.append(observe_aux_z(ctx, rt, positions_nm=pos))
        e_gpu.append(st.getPotentialEnergy()._value)
    fast, slow, e_gpu = map(np.asarray, (fast, slow, e_gpu))
    e_ref = 0.5 * K_KCAL * KJ * (slow - z0) ** 2
    dz = np.abs(fast - slow)
    red = parity_violation(fast, slow, beta=BETA, k_max_kcal=K_KCAL, centers=[z0])
    tol = PARITY_TOLERANCE.get(precision if platform_name in ("CUDA", "OpenCL") else
                               ("double" if platform_name == "Reference" else "mixed"))
    worst = float(red.max())
    return {"platform": platform_name, "precision": precision, "variant": variant,
            "actual_precision": plat.getPropertyValue(ctx, "Precision") if props else None,
            "n": int(len(dz)), "z_range": [float(slow.min()), float(slow.max())], "z_sd": float(slow.std()),
            "max_abs_dz": float(dz.max()), "median_abs_dz": float(np.median(dz)),
            "max_reduced": worst, "tolerance": tol, "ok": bool(np.isfinite(worst) and worst <= tol),
            "max_abs_dE_kj": float(np.max(np.abs(e_gpu - e_ref))), "max_E_kj": float(e_ref.max())}


def main(out):
    import openmm as mm
    names = {mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())}
    cases = [("Reference", "double")] if os.environ.get("FP_REF", "1") == "1" else []
    for p in ("CUDA", "OpenCL"):
        if p in names:
            cases += [(p, "mixed"), (p, "single"), (p, "double")]
    rows = []
    for p, prec in cases:
        for v in ("plain", "mixed_conventions", "scale_3.181"):
            try:
                r = run(p, prec, v)
            except Exception as exc:  # record, keep going
                r = {"platform": p, "precision": prec, "variant": v, "ok": False, "error": repr(exc)}
            print(json.dumps(r), flush=True)
            rows.append(r)
    json.dump({"rows": rows, "all_ok": all(r["ok"] for r in rows), "k_kcal": K_KCAL, "T_K": T_K}, open(out, "w"),
              indent=2)
    print("ALL_OK" if all(r["ok"] for r in rows) else "NOT_ALL_OK")


if __name__ == "__main__":
    main(sys.argv[1])
