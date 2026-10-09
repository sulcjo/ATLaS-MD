import numpy as np
import pytest

pytest.importorskip("openmm")
from aux_pull_fixture import build_aux_reference_context


def test_ramp_moves_z_toward_center_and_clears_parameters():
    from gareus.seeding import ramp_aux_restraint
    sim, rt, z_of = build_aux_reference_context()
    z0 = z_of(sim)
    target = z0 - 1.0  # z0 sits at the model's upper limit; only downward is reachable
    row = ramp_aux_restraint(sim, rt, center=target, k_kcal=2000.0, stages=5, steps_per_stage=200)
    z1 = z_of(sim)
    assert abs(z1 - target) < abs(z0 - target) - 0.3
    assert row["aux_ramp_stages"] == 5 and np.isfinite(row["aux_z_end"])
    assert sim.context.getParameter(rt.info.global_k) == 0.0
    assert sim.context.getParameter(rt.info.global_c) == 0.0


def test_parameters_cleared_on_exception():
    from gareus.seeding import ramp_aux_restraint
    sim, rt, _ = build_aux_reference_context()
    calls = {"n": 0}
    real = sim.step

    def boom(n):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("md failed")
        real(n)
    sim.step = boom
    with pytest.raises(RuntimeError, match="md failed"):
        ramp_aux_restraint(sim, rt, center=1.0, k_kcal=20.0, stages=5, steps_per_stage=5)
    assert sim.context.getParameter(rt.info.global_k) == 0.0


def test_aux_pull_ramp_helper_none_and_subset():
    import types
    from gareus.production import _aux_pull_ramp
    assert _aux_pull_ramp(types.SimpleNamespace(_aux_runtime=None), None) is None
    tbl = types.SimpleNamespace(n=3, centers=(0.0, 0.7, 1.5), k_kcal=(0.0, 3.0, 4.0))
    rt = types.SimpleNamespace(info="I")
    args = types.SimpleNamespace(_aux_runtime=rt)
    full = _aux_pull_ramp(args, tbl)
    assert full["k_kcal"] == [0.0, 3.0, 4.0] and full["stages"] == 5
    sub = _aux_pull_ramp(args, tbl, [2, 1])
    assert sub["centers"] == [1.5, 0.7] and sub["k_kcal"] == [4.0, 3.0]
