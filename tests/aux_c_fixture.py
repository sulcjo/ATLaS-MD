# tests/aux_c_fixture.py
"""Stage C test builders: v2 rows with instance blocks (Stage A D3) and definitions."""
from __future__ import annotations

from gareus.correctness.state_identity import make_state_definition

BOX = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
CV1 = {"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}}


def instance(i, role, *, parent=None, slot=None, obs=None):
    return {"state_instance_id": f"s{i}", "state_role": role, "spawn_parent_state_id": parent,
            "spawn_source_observation": obs, "matched_additional_slot_id": slot}


def rows(spec, model_sha=None):
    out = []
    for i, (c1, k1, aux_k, aux_c, role) in enumerate(spec):
        row = {"window_id": i, "center1": c1, "k1": k1, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
               "aux_k": aux_k, "instance": instance(i, role, parent="s0" if role != "ordinary" else None,
                                                    slot="slot-0" if role != "ordinary" else None)}
        if aux_k > 0:
            row.update(aux_model_sha256=model_sha, aux_center=aux_c)
        out.append(row)
    return out


def definition(row_list, model, *, cv2=None):
    return make_state_definition(row_list, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                                 fixed_box_vectors_nm=BOX, cv1=CV1, cv2=cv2,
                                 aux_models={model.model_sha256: model.to_mapping()})
