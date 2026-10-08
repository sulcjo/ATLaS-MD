import ast
import inspect
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.windows import set_window

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def _ctx_and_runtime():
    import openmm as mm
    from openmm import unit
    from pep_gamd_fixture import _fresh_system
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.4] * (2 * len(d["quads"]) - 1),
                                            offset=0.1, blocks=blocks))
    table = AuxStateTable(m, (0.0, 0.7, 0.0), (0.0, 3.0, 0.0), (None, None, None))
    system = _fresh_system()
    umb = mm.CustomExternalForce("0.5*k*(x-r0)^2")
    umb.addGlobalParameter("k", 0.0)
    umb.addGlobalParameter("r0", 0.0)
    umb.setForceGroup(31)
    system.addForce(umb)
    rt = add_aux_cv_force(mm, system, table, ARGS)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    return ctx, rt, unit


def _aux_energy(ctx, rt, unit):
    return ctx.getState(getEnergy=True, groups={rt.info.force_group}).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


def test_active_ordinary_active_is_exact():
    ctx, rt, unit = _ctx_and_runtime()
    centers, ks = [0.0] * 3, [0.0] * 3
    set_window(ctx, centers, ks, 1, aux_state=rt)
    e_active = _aux_energy(ctx, rt, unit)
    assert e_active > 0.0
    set_window(ctx, centers, ks, 0, aux_state=rt)
    assert _aux_energy(ctx, rt, unit) == 0.0
    assert ctx.getParameter(rt.info.global_k) == 0.0 and ctx.getParameter(rt.info.global_c) == 0.0
    set_window(ctx, centers, ks, 1, aux_state=rt)
    assert _aux_energy(ctx, rt, unit) == e_active


def test_missing_aux_globals_raise():
    import openmm as mm
    s = mm.System()
    s.addParticle(1.0)
    f = mm.CustomExternalForce("0.5*k*(x-r0)^2")
    f.addGlobalParameter("k", 0.0)
    f.addGlobalParameter("r0", 0.0)
    f.addParticle(0, [])
    s.addForce(f)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    _ctx, rt, _unit = _ctx_and_runtime()
    with pytest.raises(Exception):
        set_window(ctx, [0.0, 0.0], [0.0, 0.0], 1, aux_state=rt)


def _production_tree():
    import gareus.production as production
    return ast.parse(inspect.getsource(production))


def test_every_window_application_passes_aux_state():
    """Direct set_window(...) calls AND calls that pass set_window as a callable (e.g. _run(r, set_window, ...))."""
    tree = _production_tree()
    sites = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        direct = isinstance(n.func, ast.Name) and n.func.id == "set_window"
        passed = any(isinstance(a, ast.Name) and a.id == "set_window" for a in n.args)
        if direct or passed:
            sites.append(n)
    assert len(sites) >= 6, "set_window applications moved; re-anchor this test"
    missing = [n.lineno for n in sites if not any(k.arg == "aux_state" for k in n.keywords)]
    assert not missing, f"set_window applications without aux_state= at lines {missing}"
    # Every bare Name reference to set_window is one of the sites above (no hidden callable use).
    names = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "set_window"]
    assert len(names) == len(sites), "a set_window reference is neither a call nor a callable argument of one"


def _func_source(name):
    import gareus.production as production
    return inspect.getsource(getattr(production, name))


@pytest.mark.parametrize("fn", ["run_multiwindow_gamd_recon", "apply_joint_envelope_gamd_calibration"])
def test_gamd_recon_and_calibration_keep_aux_inactive(fn):
    tree = ast.parse(_func_source(fn))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "set_window"]
    assert calls and all(any(k.arg == "aux_state" and isinstance(k.value, ast.Constant) and k.value.value is None
                             for k in c.keywords) for c in calls)
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "deactivate_aux_parameters"
               for n in ast.walk(tree)), f"{fn} must deactivate the auxiliary restraint (D7)"
