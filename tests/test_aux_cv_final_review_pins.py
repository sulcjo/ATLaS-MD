"""Stage B final-review pins: window-order key (T1) and the C1 snapshot/instance composition (T2)."""
import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime import aux_snapshot_rows
from gareus.auxiliary_cv.state_table import AuxStateTable
from gareus.correctness._io import IntegrityError
from gareus.correctness.bias import normalize_windows
from gareus.correctness.state_identity import _check_instances
from gareus.production import _aux_window_order_key, snapshot_window_rows


def test_window_order_key_equal_for_identical_tables_and_changed_by_reordering():
    c, k = [0.1, 0.2, 0.3], [10.0, 20.0, 30.0]
    c2, k2 = [1.0, 2.0, 3.0], [1.5, 2.5, 3.5]
    a = _aux_window_order_key(c, k, c2, k2, 3)
    assert a == _aux_window_order_key(list(c), list(k), list(c2), list(k2), 3)
    swapped = _aux_window_order_key([c[1], c[0], c[2]], [k[1], k[0], k[2]],
                                    [c2[1], c2[0], c2[2]], [k2[1], k2[0], k2[2]], 3)
    assert swapped != a
    assert _aux_window_order_key(c, k, None, None, 3) != a


def _inst(sid, role, parent=None, slot=None):
    return {"state_instance_id": sid, "state_role": role, "spawn_parent_state_id": parent,
            "spawn_source_observation": None, "matched_additional_slot_id": slot}


def _table(aux_role, sham_role, aux_k):
    model = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
    insts = (_inst("ord-0", "ordinary"), _inst("aux-0", aux_role, "ord-0", "slot-0"),
             _inst("sham-0", sham_role, "ord-0", "slot-0"))
    return AuxStateTable(model, (0.0, 1.5, 0.0), (0.0, aux_k, 0.0), insts)


def _compose(table):
    rows = snapshot_window_rows([0.2] * 3, [25.0] * 3, [0.0] * 3, [1.0] * 3, None, n=3)
    snap = aux_snapshot_rows(rows, table)
    tr = table.window_rows()
    _check_instances(normalize_windows([dict(r, instance=t["instance"]) for r, t in zip(snap, tr)]))


def test_c1_composition_accepts_valid_ordinary_auxiliary_sham_table():
    _compose(_table("auxiliary", "sham", 1.2))


def test_c1_composition_refuses_role_k_mismatch():
    with pytest.raises(IntegrityError, match="aux"):
        _compose(_table("ordinary", "sham", 1.2))
    with pytest.raises(IntegrityError, match="aux"):
        _compose(_table("auxiliary", "sham", 0.0))
