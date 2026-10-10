import math
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import z_from_positions
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force, observe_aux_z
from gareus.auxiliary_cv.state_table import AuxStateTable

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def _setup(conventions=None, scale=1.0):
    import openmm as mm
    from pep_gamd_fixture import _fresh_system
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    rng = np.random.default_rng(5)
    m = AuxModel.from_mapping(model_payload(d["quads"], rng.normal(size=2 * len(d["quads"])), offset=-0.3,
                                            conventions=conventions, blocks=blocks, scale=scale))
    system = _fresh_system()
    rt = add_aux_cv_force(mm, system, AuxStateTable(m, (0.0,), (0.0,), (None,)), ARGS)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    return ctx, rt, system.getForce(rt.force_index), d


@pytest.mark.parametrize("mixed, scale", [(False, 1.0), (True, 1.0), (False, 3.181)])
def test_fast_and_slow_paths_agree_for_an_ordinary_state(mixed, scale):
    """k = 0 everywhere: an ordinary carrier still yields its z for cross-evaluation (spec 3.1)."""
    d0 = dipeptide()
    conv = ["negated" if k % 2 == 0 else "direct" for k in range(len(d0["quads"]))] if mixed else None
    ctx, rt, force, d = _setup(conv, scale)
    fast = observe_aux_z(ctx, rt, force=force)
    slow = observe_aux_z(ctx, rt, positions_nm=d["positions_nm"])
    assert fast == pytest.approx(slow, abs=1e-9)
    assert slow == pytest.approx(z_from_positions(d["positions_nm"], rt.table.model)[0], abs=1e-12)


def test_exactly_one_source():
    ctx, rt, force, d = _setup()
    with pytest.raises(ValueError):
        observe_aux_z(ctx, rt)
    with pytest.raises(ValueError):
        observe_aux_z(ctx, rt, force=force, positions_nm=d["positions_nm"])


def test_degenerate_geometry_is_recorded_not_raised():
    """Slow path: NaN is returned (fatality belongs to aux_bias_matrix_kcal, only for active states)."""
    ctx, rt, _force, d = _setup()
    bad = d["positions_nm"].copy()
    q = rt.table.model.feature_schema.features[0].atom_indices
    bad[q[0]] = bad[q[1]]                       # collapse two torsion atoms -> undefined dihedral
    assert math.isnan(observe_aux_z(ctx, rt, positions_nm=bad))


def _with_table(rt, k_kcal):
    """Same runtime, one-state table with the given strength (0 = sham/ordinary, > 0 = active)."""
    import dataclasses
    return dataclasses.replace(rt, table=AuxStateTable(rt.table.model, (0.5,), (float(k_kcal),), (None,)))


def _degenerate(ctx, rt, d):
    bad = d["positions_nm"].copy()
    q = rt.table.model.feature_schema.features[0].atom_indices
    bad[q[0]] = bad[q[1]]
    ctx.setPositions(bad)
    return bad


def test_live_degenerate_geometry_with_an_active_state_raises_on_both_paths():
    """Board condition 1 / spec 3.3: the fast path alone sees a finite theta, so the observer checks geometry."""
    from openmm import unit
    from gareus.auxiliary_cv.runtime import AuxObservationError, make_aux_z_observer
    ctx, rt, force, d = _setup()
    _degenerate(ctx, rt, d)
    active = _with_table(rt, 2.0)
    observe = make_aux_z_observer(active, aux_forces=[force], unit=unit)
    with pytest.raises(AuxObservationError, match="degenerate"):
        observe(0, types.SimpleNamespace(context=ctx))


def test_sham_only_population_never_raises_on_degenerate_geometry():
    from openmm import unit
    from gareus.auxiliary_cv.runtime import make_aux_z_observer
    ctx, rt, force, d = _setup()
    _degenerate(ctx, rt, d)
    sham = _with_table(rt, 0.0)
    z = make_aux_z_observer(sham, aux_forces=[force], unit=unit)(0, types.SimpleNamespace(context=ctx))
    # F07: z is the force's own value on every CV1 path, and the force sees a finite theta at degenerate
    # geometry; the slow path used to record the NumPy evaluator's NaN here. Recorded, never raised.
    assert math.isfinite(z)


def test_regular_geometry_with_an_active_state_returns_the_path_z():
    from openmm import unit
    from gareus.auxiliary_cv.runtime import make_aux_z_observer
    ctx, rt, force, d = _setup()
    active = _with_table(rt, 2.0)
    z = make_aux_z_observer(active, aux_forces=[force], unit=unit)(0, types.SimpleNamespace(context=ctx))
    assert z == pytest.approx(z_from_positions(d["positions_nm"], rt.table.model)[0], abs=1e-9)


def test_set_aux_parameters_absolute_energy_is_kcal_times_4184():
    """Board dissent (thinker): pin the kcal -> kJ conversion of set_aux_parameters in absolute terms."""
    from openmm import unit
    from gareus.auxiliary_cv.force import set_aux_parameters
    ctx, rt, _force, d = _setup()
    z = z_from_positions(d["positions_nm"], rt.table.model)[0]
    k_kcal, c = 2.5, z - 0.4
    set_aux_parameters(ctx, rt.info, center=c, k_kcal=k_kcal)
    e = ctx.getState(getEnergy=True, groups={rt.info.force_group}).getPotentialEnergy() \
           .value_in_unit(unit.kilojoule_per_mole)
    assert e == pytest.approx(0.5 * k_kcal * 4.184 * (z - c) ** 2, rel=1e-9)
