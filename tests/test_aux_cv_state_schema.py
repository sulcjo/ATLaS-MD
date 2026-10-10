import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError
from gareus.correctness._io import json_loads
from gareus.correctness.state_identity import (STATE_SCHEMA_V2, canonical_state_definition,
                                               freeze_snapshot, hamiltonian_sha256,
                                               make_state_definition, state_definition_hash,
                                               validate_fixed_state_segments, write_frozen_snapshot)

V1_PINNED = "dc0acf77e275ed0d1b357238e16e3f1f8248859b06ab647d7e6578817e4c8725"
BOX = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
CV1 = {"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}}
CV2 = {"kind": "residual", "units": "dimensionless", "definition": {"v": [1.0]}}
MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
SHA = MODEL.model_sha256
OBS = {"run": "r", "segment": "s", "carrier": 3, "state": 1, "checkpoint": "c", "step": 100}


def _legacy_windows():
    return [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0},
            {"window_id": 1, "center1": 0.4, "k1": 10.0, "center2": 1.0, "k2": 2.0, "gamd_lambda": 0.0}]


def _defn(windows, aux_models=..., energy_unit="kcal/mol"):
    kw = {} if aux_models is ... else {"aux_models": aux_models}
    return make_state_definition(windows, physical_system_sha256="a" * 64, ensemble="NVT",
                                 temperature_k=300.0, fixed_box_vectors_nm=BOX, cv1=CV1, cv2=CV2,
                                 energy_unit=energy_unit, **kw)


def _v2_windows():
    base = _legacy_windows()
    for k, row in enumerate(base):
        row.update(aux_k=0.0, instance={"state_instance_id": f"ord-{k}", "state_role": "ordinary",
                                        "spawn_parent_state_id": None, "spawn_source_observation": None,
                                        "matched_additional_slot_id": None})
    base.append({"window_id": 2, "center1": 0.4, "k1": 10.0, "center2": 1.0, "k2": 2.0,
                 "gamd_lambda": 0.0, "aux_model_sha256": SHA, "aux_center": 1.5, "aux_k": 1.2,
                 "instance": {"state_instance_id": "aux-0", "state_role": "auxiliary",
                              "spawn_parent_state_id": "ord-1", "spawn_source_observation": dict(OBS),
                              "matched_additional_slot_id": "slot-0"}})
    base.append({"window_id": 3, "center1": 0.4, "k1": 10.0, "center2": 1.0, "k2": 2.0,
                 "gamd_lambda": 0.0, "aux_k": 0.0,
                 "instance": {"state_instance_id": "sham-0", "state_role": "sham",
                              "spawn_parent_state_id": "ord-1", "spawn_source_observation": dict(OBS),
                              "matched_additional_slot_id": "slot-0"}})
    return base


def _relabelled():
    payload = MODEL.to_mapping()
    payload["label"] = "other"
    payload["provenance"] = {"z": 1}
    return payload


def test_legacy_v1_hash_is_unchanged():
    # Regression guard (passes before and after this task): v1 bytes must not change.
    assert state_definition_hash(_defn(_legacy_windows())) == V1_PINNED


def test_v2_round_trip_and_canonical_inactive():
    d = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    assert d["schema"] == STATE_SCHEMA_V2
    rows = {r["window_id"]: r for r in d["windows"]}
    assert rows[0]["aux_model_sha256"] is None and rows[0]["aux_center"] == 0.0 and rows[0]["aux_k"] == 0.0
    assert rows[2]["aux_model_sha256"] == SHA and rows[2]["aux_k"] == 1.2
    assert state_definition_hash(d) == state_definition_hash(_defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()}))


def test_registry_embeds_identity_only():
    d = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    assert d["aux_models"][SHA] == MODEL.identity_mapping()
    assert "label" not in d["aux_models"][SHA] and "provenance" not in d["aux_models"][SHA]


def test_relabelling_the_model_changes_no_hash():
    d1 = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    d2 = _defn(_v2_windows(), aux_models={SHA: _relabelled()})
    assert state_definition_hash(d1) == state_definition_hash(d2)
    for wid in (0, 1, 2, 3):
        assert hamiltonian_sha256(d1, wid) == hamiltonian_sha256(d2, wid)


def test_negative_zero_center_hashes_like_zero():
    w1, w2 = _v2_windows(), _v2_windows()
    w1[2]["aux_center"] = 0.0
    w2[2]["aux_center"] = -0.0
    m = {SHA: MODEL.to_mapping()}
    assert state_definition_hash(_defn(w1, aux_models=m)) == state_definition_hash(_defn(w2, aux_models=m))


def test_sham_and_parent_share_hamiltonian_but_not_auxiliary():
    d = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    assert hamiltonian_sha256(d, 3) == hamiltonian_sha256(d, 1)
    assert hamiltonian_sha256(d, 2) != hamiltonian_sha256(d, 1)


def test_inactive_v2_row_hashes_like_v1_physics():
    v1 = _defn(_legacy_windows())
    v2 = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    assert hamiltonian_sha256(v1, 1) == hamiltonian_sha256(v2, 1)


def test_hamiltonian_hash_rejects_boolean_window_id():
    d = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    with pytest.raises(IntegrityError, match="window_id"):
        hamiltonian_sha256(d, True)


def test_kj_tables_convert_aux_k():
    w = _v2_windows()
    for row in w:
        row["k1"] *= 4.184
        row["k2"] *= 4.184
        row["aux_k"] *= 4.184
    d = _defn(w, aux_models={SHA: MODEL.to_mapping()}, energy_unit="kJ/mol")
    assert {r["window_id"]: r for r in d["windows"]}[2]["aux_k"] == pytest.approx(1.2, abs=1e-12)


@pytest.mark.parametrize("mutate, message", [
    (lambda w, m: w[0].pop("aux_k"), "aux_k"),
    (lambda w, m: w[2].update(aux_k=-1.0), "aux_k"),
    (lambda w, m: w[2].update(aux_k=float("nan")), "aux_k|Nonfinite"),
    (lambda w, m: w[2].pop("aux_center"), "aux_center"),
    (lambda w, m: w[2].update(aux_model_sha256="b" * 64), "unknown auxiliary model"),
    (lambda w, m: w[2]["instance"].update(state_role="worker"), "state_role"),
    (lambda w, m: w[2]["instance"].pop("matched_additional_slot_id"), "instance"),
    (lambda w, m: w[3]["instance"].update(state_instance_id="aux-0"), "duplicate state_instance_id"),
    (lambda w, m: w[3]["instance"].update(state_instance_id=""), "state_instance_id"),
    (lambda w, m: w[2]["instance"]["spawn_source_observation"].pop("step"), "spawn_source_observation"),
    (lambda w, m: w[2]["instance"]["spawn_source_observation"].update(step="notastep"), "spawn_source_observation"),
    (lambda w, m: w[2]["instance"]["spawn_source_observation"].update(carrier=True), "spawn_source_observation"),
    (lambda w, m: w[2]["instance"].update(spawn_parent_state_id=1), "spawn_parent_state_id"),
    (lambda w, m: w[2]["instance"].update(spawn_parent_state_id="ord-99"), "not a state_instance_id"),
    (lambda w, m: w[2]["instance"].update(spawn_parent_state_id="aux-0"), "own parent"),
    (lambda w, m: w[0].pop("instance"), "every row"),
    # role <-> energy consistency (spec 1.1 / 11.1: a sham must never carry an active bias)
    (lambda w, m: w[3].update(aux_model_sha256=SHA, aux_center=0.0, aux_k=1.0), "sham requires aux_k == 0"),
    (lambda w, m: w[0].update(aux_model_sha256=SHA, aux_center=0.0, aux_k=1.0), "ordinary requires aux_k == 0"),
    (lambda w, m: w[2].update(aux_k=0.0), "auxiliary requires aux_k > 0"),
    (lambda w, m: m.update({"c" * 64: m.pop(SHA)}), "key"),
    # W/B matched-slot rules (spec 5, 11.1)
    (lambda w, m: w[0]["instance"].update(matched_additional_slot_id="slot-0"), "ordinary state cannot carry"),
    (lambda w, m: (w[3].update(aux_model_sha256=SHA, aux_center=0.0, aux_k=1.0),
                   w[3]["instance"].update(state_role="auxiliary")), "more than one auxiliary"),
    (lambda w, m: w[3].update(k1=11.0), "baseline"),
])
def test_v2_refusals(mutate, message):
    w = _v2_windows()
    m = {SHA: MODEL.to_mapping()}
    mutate(w, m)
    with pytest.raises(IntegrityError, match=message):
        _defn(w, aux_models=m)


def test_more_than_one_model_is_stage_f():
    """Stricter than spec 1.2 (which refuses >1 ACTIVE model): MVP refuses >1 registered model."""
    other = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [0.0, 1.0]))
    with pytest.raises(IntegrityError, match="Stage F"):
        _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping(), other.model_sha256: other.to_mapping()})


def test_v1_rejects_aux_fields():
    # Regression guard (passes before and after): v1 keeps refusing unknown physics fields.
    w = _legacy_windows()
    w[0]["aux_k"] = 0.0
    with pytest.raises(IntegrityError, match="unknown physics fields"):
        _defn(w)


def test_v2_definition_round_trips_through_frozen_snapshots(tmp_path):
    """Spec 5: snapshot hashes and restart comparisons cover the aux fields and instance metadata.

    validate_fixed_state_segments re-normalises the visible windows with normalize_windows; this pins
    that the instance block passes through it unchanged and the aux fields re-canonicalise identically.
    """
    d = _defn(_v2_windows(), aux_models={SHA: MODEL.to_mapping()})
    snap = freeze_snapshot("seg_001", d, equilibrium_analysis_eligible=True, phase_kind="production")
    table = validate_fixed_state_segments({"seg_001": snap})
    assert table.definition == d and table.definition_sha256 == state_definition_hash(d)
    rows = {r["window_id"]: r for r in table.windows}
    assert rows[2]["instance"]["state_instance_id"] == "aux-0" and rows[2]["aux_model_sha256"] == SHA
    path = tmp_path / "seg_001.json"
    write_frozen_snapshot(path, snap)
    again = validate_fixed_state_segments({"seg_001": json_loads(path.read_bytes())})
    assert again.definition_sha256 == table.definition_sha256
    other = freeze_snapshot("seg_002", _defn(_v2_windows(), aux_models={SHA: _relabelled()}),
                            equilibrium_analysis_eligible=True, phase_kind="production")
    validate_fixed_state_segments({"seg_001": snap, "seg_002": other})   # relabel: still fixed-state compatible
    changed = _v2_windows()
    changed[2]["aux_center"] = 1.6
    moved = freeze_snapshot("seg_003", _defn(changed, aux_models={SHA: MODEL.to_mapping()}),
                            equilibrium_analysis_eligible=True, phase_kind="production")
    with pytest.raises(IntegrityError, match="aux_center"):
        validate_fixed_state_segments({"seg_001": snap, "seg_003": moved})


def test_kj_canonicalisation_is_idempotent_through_the_hash_paths():
    """energy_unit convention (inherited from k1/k2): canonicalisation divides kJ tables by 4.184 once
    and relabels the table "kcal/mol", so re-canonicalising (as every hash function does) never divides
    again."""
    w = _v2_windows()
    for row in w:
        row["k1"] *= 4.184
        row["k2"] *= 4.184
        row["aux_k"] *= 4.184
    d = _defn(w, aux_models={SHA: MODEL.to_mapping()}, energy_unit="kJ/mol")
    assert d["energy_unit"] == "kcal/mol"
    assert canonical_state_definition(d) == d
    assert state_definition_hash(d) == state_definition_hash(canonical_state_definition(d))
    for wid in (0, 1, 2, 3):
        assert hamiltonian_sha256(d, wid) == hamiltonian_sha256(canonical_state_definition(d), wid)
    assert {r["window_id"]: r for r in canonical_state_definition(d)["windows"]}[2]["aux_k"] == \
        pytest.approx(1.2, abs=1e-12)


def test_negative_aux_centre_is_accepted():
    w = _v2_windows()
    w[2]["aux_center"] = -1.5
    d = _defn(w, aux_models={SHA: MODEL.to_mapping()})
    assert {r["window_id"]: r for r in d["windows"]}[2]["aux_center"] == -1.5
