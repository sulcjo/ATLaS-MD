import json

import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.state_table import load_aux_state_table, parse_aux_csv_row
from gareus.correctness._io import IntegrityError

QUADS = [(0, 1, 2, 3)]


def _model(tmp_path):
    m = AuxModel.from_mapping(model_payload(QUADS, [1.0, 0.5], offset=0.1))
    path = tmp_path / "aux.json"
    m.write(path)
    return m, path


def _csv(tmp_path, rows, header):
    path = tmp_path / "w.csv"
    path.write_text("\n".join([",".join(header)] + [",".join(map(str, r)) for r in rows]) + "\n")
    return path


def _args(tmp_path, model_path):
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GA", "--out", str(tmp_path / "o"), "--cv1", "contacts"])
    a.aux_cv_model = str(model_path)
    return a


HEADER = ["primary_cv_center", "primary_cv_k_kcal", "aux_center", "aux_k_kcal_mol", "state_instance_id",
          "state_role", "spawn_parent_state_id", "matched_additional_slot_id"]
ROWS = [[0.2, 25.0, "", 0.0, "ord-0", "ordinary", "", ""],
        [0.4, 25.0, "", 0.0, "ord-1", "ordinary", "", ""],
        [0.4, 25.0, 1.5, 1.2, "aux-0", "auxiliary", "ord-1", "slot-0"],
        [0.4, 25.0, "", 0.0, "sham-0", "sham", "ord-1", "slot-0"]]


def _inst(**over):
    base = {"aux_k_kcal_mol": "0", "state_instance_id": "x", "state_role": "ordinary",
            "spawn_parent_state_id": "", "matched_additional_slot_id": "", "spawn_source_observation_json": ""}
    base.update(over)
    return base


def test_loader_collects_aux_rows_and_table_parses(tmp_path):
    from gareus.windows import load_explicit_2d_window_csv
    m, mpath = _model(tmp_path)
    *_rest, wmeta = load_explicit_2d_window_csv(_args(tmp_path, mpath), _csv(tmp_path, ROWS, HEADER))
    table = load_aux_state_table(mpath, wmeta["aux_rows"])
    assert table.n == 4
    assert table.k_kcal == (0.0, 0.0, 1.2, 0.0) and table.centers == (0.0, 0.0, 1.5, 0.0)
    assert [i["state_instance_id"] for i in table.instances] == ["ord-0", "ord-1", "aux-0", "sham-0"]
    assert table.instances[2]["spawn_parent_state_id"] == "ord-1"
    rows = table.window_rows()
    assert rows[2]["aux_model_sha256"] == m.model_sha256 and rows[0]["aux_model_sha256"] is None


def test_loaded_rows_keep_aux_cells_of_every_row(tmp_path):
    """Row 0 is ordinary (no aux_center cell); the active row's aux_center must survive normalisation."""
    from gareus.windows import load_explicit_2d_window_csv
    _m, mpath = _model(tmp_path)
    *_rest, wmeta = load_explicit_2d_window_csv(_args(tmp_path, mpath), _csv(tmp_path, ROWS, HEADER))
    keys = set().union(*(r.keys() for r in wmeta["normalized_rows"]))
    assert {"aux_center", "aux_k_kcal_mol", "state_instance_id"} <= keys


def test_legacy_table_has_no_aux_rows_and_does_not_import_the_aux_package(tmp_path):
    import subprocess, sys, textwrap
    code = textwrap.dedent(f"""
        import sys
        from gareus.cli import parse_args
        from gareus.windows import load_explicit_2d_window_csv
        from pathlib import Path
        p = Path({str(tmp_path)!r}) / "legacy.csv"
        p.write_text("primary_cv_center,primary_cv_k_kcal\\n0.2,25.0\\n0.4,25.0\\n")
        a = parse_args(["--seq", "GA", "--out", {str(tmp_path / "o")!r}, "--cv1", "contacts"])
        *_r, wmeta = load_explicit_2d_window_csv(a, p)
        assert "aux_rows" not in wmeta
        assert "aux_k_kcal_mol" not in wmeta["normalized_rows"][0]
        assert "gareus.auxiliary_cv" not in sys.modules, "off path imported the aux package"
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_aux_columns_without_model_flag_are_refused(tmp_path):
    from gareus.cli import parse_args
    from gareus.windows import load_explicit_2d_window_csv
    a = parse_args(["--seq", "GA", "--out", str(tmp_path / "o"), "--cv1", "contacts"])
    with pytest.raises(ValueError, match="--aux-cv-model"):
        load_explicit_2d_window_csv(a, _csv(tmp_path, ROWS, HEADER))


def test_model_flag_without_aux_columns_is_refused(tmp_path):
    from gareus.windows import load_explicit_2d_window_csv
    _m, mpath = _model(tmp_path)
    with pytest.raises(ValueError, match="aux_k_kcal_mol"):
        load_explicit_2d_window_csv(_args(tmp_path, mpath),
                                    _csv(tmp_path, [[0.2, 25.0]], ["primary_cv_center", "primary_cv_k_kcal"]))


@pytest.mark.parametrize("row, message", [
    ({"aux_k_kcal_mol": ""}, "aux_k_kcal_mol"),
    ({"aux_k_kcal_mol": "-1"}, "aux_k"),
    ({"aux_k_kcal_mol": "1.0"}, "aux_center"),
    ({"aux_k_kcal_mol": "1.0", "aux_center": "0.2", "aux_model_sha256": "a" * 64}, "model"),
    (_inst(aux_k_kcal_mol="1.0", aux_center="0.2", state_role="sham"), "role"),
    (_inst(aux_k_kcal_mol="1.0", aux_center="0.2", state_role="ordinary"), "role"),
    (_inst(state_role="auxiliary"), "role"),
    (_inst(spawn_source_observation_json='{"run": "r"}'), "spawn_source_observation"),
])
def test_row_refusals(row, message, tmp_path):
    m, _ = _model(tmp_path)
    with pytest.raises((IntegrityError, ValueError), match=message):
        parse_aux_csv_row(row, 2, m.model_sha256)


def test_spawn_source_observation_json_round_trips(tmp_path):
    m, _ = _model(tmp_path)
    obs = {"run": "r", "segment": "s", "carrier": 3, "state": 1, "checkpoint": "c", "step": 100}
    out = parse_aux_csv_row(_inst(aux_k_kcal_mol="1", aux_center="0.5", state_instance_id="aux-0",
                                  state_role="auxiliary", spawn_parent_state_id="ord-1",
                                  matched_additional_slot_id="slot-0",
                                  spawn_source_observation_json=json.dumps(obs)), 2, m.model_sha256)
    assert out["instance"]["spawn_source_observation"] == obs
    assert out["instance"]["spawn_parent_state_id"] == "ord-1"


def test_table_refuses_partial_instances_duplicate_ids_and_unknown_parents(tmp_path):
    _m, mpath = _model(tmp_path)
    with pytest.raises(IntegrityError, match="duplicate"):
        load_aux_state_table(mpath, [_inst(), _inst()])
    with pytest.raises(IntegrityError, match="every row"):
        load_aux_state_table(mpath, [_inst(), {"aux_k_kcal_mol": "0"}])
    with pytest.raises(IntegrityError, match="parent"):
        load_aux_state_table(mpath, [_inst(state_instance_id="a"),
                                     _inst(state_instance_id="b", spawn_parent_state_id="nope")])


def test_table_refuses_a_state_that_is_its_own_parent(tmp_path):
    _m, mpath = _model(tmp_path)
    with pytest.raises(IntegrityError, match="own parent"):
        load_aux_state_table(mpath, [_inst(state_instance_id="a", spawn_parent_state_id="a")])
