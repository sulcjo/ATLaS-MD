"""Reference-platform GA-dipeptide Simulation carrying the aux force (shared by aux pull-ramp tests)."""
import types

import numpy as np

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import add_aux_cv_force, observe_aux_z
from gareus.auxiliary_cv.state_table import AuxStateTable

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def build_aux_reference_context():
    import openmm as mm
    from openmm import app, unit
    from pep_gamd_fixture import _fresh_system
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.4] * (2 * len(d["quads"]) - 1),
                                            offset=0.1, blocks=blocks))
    table = AuxStateTable(m, (0.0,), (0.0,), (None,))
    system = _fresh_system()
    rt = add_aux_cv_force(mm, system, table, ARGS)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 5 / unit.picosecond, 0.001 * unit.picoseconds)
    integ.setRandomNumberSeed(7)
    ctx = mm.Context(system, integ, mm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(d["positions_nm"])
    sim = types.SimpleNamespace(context=ctx, step=integ.step)

    def z_of(s):
        pos = s.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        return observe_aux_z(s.context, rt, positions_nm=np.asarray(pos))
    return sim, rt, z_of
