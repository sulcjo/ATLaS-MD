# tests/test_aux_cv_runtime.py
import types

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.force import AUX_FORCE_NAME
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import (RESERVED_PHYSICAL_GROUPS, add_aux_cv_force,
                                         allocate_free_force_group, aux_snapshot_rows,
                                         canonical_topology_sha256, deactivate_aux_parameters,
                                         force_group_audit, refuse_aux_population_change)
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.correctness._io import IntegrityError

ARGS = types.SimpleNamespace(umbrella_force_group=31, secondary_cv_force_group=29)


def _table(k=(0.0, 2.0), c=(0.0, 1.0)):
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.3] * (2 * len(d["quads"]) - 1),
                                            offset=0.2, blocks=blocks))
    return AuxStateTable(m, tuple(c), tuple(k), (None,) * len(k))


def _bare_system(groups):
    import openmm as mm
    s = mm.System()
    for _ in range(4):
        s.addParticle(1.0)
    for g in groups:
        f = mm.CustomExternalForce("0")
        f.setForceGroup(g)
        s.addForce(f)
    return s


def test_allocator_skips_used_and_reserved_groups():
    s = _bare_system([0, 3, 4])
    assert allocate_free_force_group(s, reserved=(5,)) == 6
    assert not ({allocate_free_force_group(s)} & RESERVED_PHYSICAL_GROUPS)


def test_allocator_raises_when_full():
    with pytest.raises(IntegrityError, match="free force group"):
        allocate_free_force_group(_bare_system(range(32)))


def test_audit_lists_every_force():
    s = _bare_system([0, 7])
    audit = force_group_audit(s)
    assert [a["group"] for a in audit] == [0, 7] and all(a["class"] == "CustomExternalForce" for a in audit)


def test_add_aux_cv_force_allocates_and_records():
    from pep_gamd_fixture import _fresh_system
    import openmm as mm
    system = _fresh_system()
    before = {a["group"] for a in force_group_audit(system)}
    rt = add_aux_cv_force(mm, system, _table(), ARGS, topology_sha256="b" * 64)
    assert system.getForce(rt.force_index).getName() == AUX_FORCE_NAME
    assert rt.info.force_group not in before | RESERVED_PHYSICAL_GROUPS | {29, 31}
    assert rt.topology_sha256 == "b" * 64


def test_explicit_group_must_be_free():
    from pep_gamd_fixture import _fresh_system
    import openmm as mm
    system = _fresh_system()
    used = force_group_audit(system)[0]["group"]
    with pytest.raises(IntegrityError, match="force group"):
        add_aux_cv_force(mm, system, _table(), ARGS, force_group=used)


def test_same_audit_gives_same_group_on_a_copy():
    """Starting-structure systems must receive the base system's group (production passes it explicitly)."""
    from pep_gamd_fixture import _fresh_system
    import openmm as mm
    a, b = _fresh_system(), _fresh_system()
    assert add_aux_cv_force(mm, a, _table(), ARGS).info.force_group == \
        add_aux_cv_force(mm, b, _table(), ARGS).info.force_group


def test_deactivate_sets_k_and_center_to_zero_and_ignores_none():
    from pep_gamd_fixture import _fresh_system
    import openmm as mm
    from gareus.auxiliary_cv.force import set_aux_parameters
    system = _fresh_system()
    rt = add_aux_cv_force(mm, system, _table(), ARGS)
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    set_aux_parameters(ctx, rt.info, center=1.0, k_kcal=2.0)
    deactivate_aux_parameters(ctx, rt)
    assert ctx.getParameter(rt.info.global_k) == 0.0 and ctx.getParameter(rt.info.global_c) == 0.0
    deactivate_aux_parameters(ctx, None)        # off path: no-op


@pytest.mark.parametrize("n_expected, n_now", [(4, 3), (4, 5)])
def test_population_change_is_refused(n_expected, n_now):
    with pytest.raises(RuntimeError, match="--max-replicas"):
        refuse_aux_population_change(n_expected, n_now, cause="--max-replicas")


def test_unchanged_population_passes():
    refuse_aux_population_change(4, 4, cause="seed-reachability filter")


def test_snapshot_rows_carry_aux_fields_and_strict_reconstruction_fails_closed():
    from gareus.correctness.bias import MissingCoordinateError, reconstruct_bias_matrix
    table = _table()
    legacy = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0},
              {"window_id": 1, "center1": 0.4, "k1": 10.0, "gamd_lambda": 0.0}]
    rows = aux_snapshot_rows(legacy, table)
    assert "aux_k" not in legacy[0], "input rows must not be mutated"
    assert rows[0]["aux_k"] == 0.0 and rows[0]["aux_model_sha256"] is None
    assert rows[1]["aux_k"] == 2.0 and rows[1]["aux_model_sha256"] == table.model.model_sha256
    with pytest.raises(MissingCoordinateError):
        reconstruct_bias_matrix(np.array([0.3]), None, rows, 0.4)


def test_snapshot_rows_length_must_match():
    with pytest.raises(RuntimeError, match="rows"):
        aux_snapshot_rows([{"window_id": 0, "center1": 0.0, "k1": 1.0, "gamd_lambda": 0.0}], _table())


def test_canonical_topology_sha_is_deterministic_and_content_sensitive():
    import openmm.app as app
    d = dipeptide()
    m = _table().model
    a = canonical_topology_sha256(d["topology"], m)
    assert a == canonical_topology_sha256(d["topology"], m) and len(a) == 64

    def _toy(name):
        top = app.Topology()
        ch = top.addChain()
        res = top.addResidue("ALA", ch)
        for nm in ("N", "CA", "C", name):
            top.addAtom(nm, app.element.carbon, res)
        return top

    m4 = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
    assert canonical_topology_sha256(_toy("O"), m4) != canonical_topology_sha256(_toy("OXT"), m4)
