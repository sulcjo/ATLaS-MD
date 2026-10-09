import csv
from pathlib import Path

from aux_cv_fixture import model_payload
from gareus.adaptive_production import (
    AUX_METADATA_KEY, DECISION_SETTINGS_FIELDS, AdaptiveDecisionPolicy, WindowStateRegistry,
    _hamiltonian_snapshot, _resolve_decision_settings, aux_params, is_auxiliary_state,
    registry_has_aux, write_state_subset_window_csv)
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.state_table import load_aux_state_table
from gareus.windows import AUX_CSV_COLUMNS, INSTANCE_CSV_COLUMNS

SHA = "a" * 64


def _registry():
    r = WindowStateRegistry()
    r.add_state(0.1, 10.0, 0.5, 2.0, gamd_lambda=0.0)
    r.add_state(0.3, 10.0, 0.5, 2.0, gamd_lambda=0.0)
    return r


def _add_worker(r, parent=0, c3=1.5, k3=2.0, sha=SHA):
    p = r.get_state(parent)
    meta = {AUX_METADATA_KEY: {"role": "auxiliary", "aux_center": c3, "aux_k_kcal_mol": k3,
                               "aux_model_sha256": sha, "state_instance_id": None,
                               "spawn_parent_state_id": parent, "admitted_epoch": 1}}
    return r.add_state(p.primary_center, p.primary_k, p.secondary_center, p.secondary_k,
                       gamd_lambda=0.0, parent_state_id=parent, epoch=1, source="adaptive_production_aux",
                       reason="aux_discovery", burnin_steps=1000, metadata=meta)


def test_no_aux_csv_has_no_aux_columns(tmp_path: Path):
    r = _registry()
    a = tmp_path / "a.csv"
    r.write_active_window_csv(a)
    assert not registry_has_aux(r)
    header = a.read_text().splitlines()[0].split(",")
    assert not set(AUX_CSV_COLUMNS) & set(header)
    assert not set(INSTANCE_CSV_COLUMNS) & set(header)
    s = tmp_path / "s.csv"
    write_state_subset_window_csv(r, s, [0, 1])
    assert not set(AUX_CSV_COLUMNS) & set(s.read_text().splitlines()[0].split(","))


def test_aux_columns_on_every_row(tmp_path: Path):
    r = _registry()
    w = _add_worker(r)
    path = tmp_path / "w.csv"
    r.write_active_window_csv(path)
    rows = list(csv.DictReader(path.open()))
    assert all(set(AUX_CSV_COLUMNS) | set(INSTANCE_CSV_COLUMNS) <= set(row) for row in rows)
    by_id = {int(row["state_id"]): row for row in rows}
    assert by_id[0]["state_role"] == "ordinary" and float(by_id[0]["aux_k_kcal_mol"]) == 0.0
    assert by_id[0]["aux_center"] == "" and by_id[0]["aux_model_sha256"] == ""
    wr = by_id[w.state_id]
    assert wr["state_role"] == "auxiliary" and float(wr["aux_center"]) == 1.5
    assert wr["aux_model_sha256"] == SHA and wr["spawn_parent_state_id"] == "s0"
    assert wr["state_instance_id"] == f"s{w.state_id}"


def test_subset_writer_has_aux_columns(tmp_path: Path):
    r = _registry()
    w = _add_worker(r)
    path = tmp_path / "sub.csv"
    write_state_subset_window_csv(r, path, [0, w.state_id])
    rows = {int(x["state_id"]): x for x in csv.DictReader(path.open())}
    assert rows[w.state_id]["state_role"] == "auxiliary"
    assert rows[w.state_id]["spawn_parent_state_id"] == "s0"
    assert rows[0]["state_role"] == "ordinary"
    # parent not in the written subset -> blank
    path2 = tmp_path / "sub2.csv"
    write_state_subset_window_csv(r, path2, [1, w.state_id])
    rows2 = {int(x["state_id"]): x for x in csv.DictReader(path2.open())}
    assert rows2[w.state_id]["spawn_parent_state_id"] == ""


def test_parent_written_blank_when_not_active(tmp_path: Path):
    r = _registry()
    w = _add_worker(r, parent=1)
    r.retire_state(1, epoch=2, reason="test")
    path = tmp_path / "w.csv"
    r.write_active_window_csv(path)
    row = {int(x["state_id"]): x for x in csv.DictReader(path.open())}[w.state_id]
    assert row["spawn_parent_state_id"] == ""


def test_worker_ignored_by_duplicate_check():
    r = _registry()
    w = _add_worker(r)
    pol = AdaptiveDecisionPolicy()
    p = r.get_state(0)
    assert r.has_near_duplicate(p.primary_center, p.secondary_center, pol, gamd_lambda=0.0,
                                primary_k=p.primary_k, secondary_k=p.secondary_k)
    assert is_auxiliary_state(w) and not is_auxiliary_state(p)
    assert aux_params(p) is None
    # a registry holding ONLY the worker: its centre is not occupied for ordinary identity
    only = WindowStateRegistry()
    only.add_state(0.9, 10.0, 0.5, 2.0, metadata=dict(w.metadata))
    assert is_auxiliary_state(only.get_state(0))
    assert not only.has_near_duplicate(0.9, 0.5, pol, gamd_lambda=0.0, primary_k=10.0, secondary_k=2.0)


def test_hamiltonian_snapshot_includes_aux():
    r = _registry()
    w = _add_worker(r)
    snap = _hamiltonian_snapshot(r)
    assert snap[w.state_id][-3:] == (1.5, 2.0, SHA)
    assert snap[0][-3:] == (None, 0.0, None)


def test_registry_json_roundtrip_keeps_aux(tmp_path: Path):
    r = _registry()
    w = _add_worker(r)
    r.save_json(tmp_path / "r.json")
    back = WindowStateRegistry.load_json(tmp_path / "r.json")
    assert is_auxiliary_state(back.get_state(w.state_id))


def test_written_csv_parses_through_aux_state_table(tmp_path: Path):
    model = AuxModel.from_mapping(model_payload([(0, 1, 2, 3), (1, 2, 3, 4)], [0.5, -0.25, 0.0, 1.0]))
    mpath = tmp_path / "aux_model.json"
    model.write(mpath)
    r = _registry()
    _add_worker(r, sha=model.model_sha256)
    path = tmp_path / "w.csv"
    r.write_active_window_csv(path)
    rows = list(csv.DictReader(path.open()))
    table = load_aux_state_table(mpath, rows)
    assert table.n == 3
    assert sorted(table.k_kcal) == [0.0, 0.0, 2.0]
    assert table.centers[[int(x["state_id"]) for x in rows].index(2)] == 1.5


def test_flag_off_decision_settings_only_adds_two_keys(tmp_path: Path):
    _pol, record = _resolve_decision_settings(tmp_path, AdaptiveDecisionPolicy())
    new = {"aux_discovery", "aux_reserve_slots"}
    assert set(record["settings"]) - (set(DECISION_SETTINGS_FIELDS) - new) == new
    assert record["settings"]["aux_discovery"] is False
    assert record["settings"]["aux_reserve_slots"] == 4
    assert set(record["settings"]) == set(DECISION_SETTINGS_FIELDS)
