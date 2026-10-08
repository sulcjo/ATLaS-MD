import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.runtime_definition import (build_runtime_state_definition, embed_cv_definition,
                                                    physical_system_sha256)
from gareus.correctness._io import IntegrityError

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
BOX = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
ORD = {"state_instance_id": "w0000", "state_role": "ordinary", "spawn_parent_state_id": None,
       "spawn_source_observation": None, "matched_additional_slot_id": None}
AUX = {"state_instance_id": "aux-0", "state_role": "auxiliary", "spawn_parent_state_id": "w0000",
       "spawn_source_observation": None, "matched_additional_slot_id": "slot-0"}
SNAP = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0},
        {"window_id": 1, "center1": 0.2, "k1": 10.0, "gamd_lambda": 0.0}]
KEY = ((0.2, 10.0), (0.2, 10.0))
CV1 = {"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}}


def _table(k=(0.0, 2.0), inst=(ORD, AUX)):
    from gareus.auxiliary_cv.state_table import AuxStateTable
    return AuxStateTable(MODEL, (0.0, 1.5), tuple(k), tuple(inst))


def _build(**kw):
    base = dict(physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0, pressure_bar=None,
                box_vectors_nm=BOX, cv1=CV1, cv2=None, boost=None, snapshot_rows=SNAP, aux_table=_table(),
                applied_window_key=KEY)
    base.update(kw)
    return build_runtime_state_definition(**base)


def test_definition_merges_snapshot_rows_and_aux_table():
    d = _build()
    rows = {r["window_id"]: r for r in d["windows"]}
    assert rows[0]["instance"]["state_role"] == "ordinary" and rows[0]["aux_k"] == 0.0
    assert rows[1]["aux_k"] == 2.0 and rows[1]["instance"]["state_instance_id"] == "aux-0"
    assert list(d["aux_models"]) == [MODEL.model_sha256]


def test_row_without_instance_is_refused():
    with pytest.raises(IntegrityError, match="instance"):
        _build(aux_table=_table(inst=(None, AUX)))


def test_definition_must_match_the_restraints_set_window_applies():
    with pytest.raises(IntegrityError, match="set_window applies"):
        _build(applied_window_key=((0.2, 10.0), (0.2, 11.0)))
    with pytest.raises(IntegrityError, match="set_window applies"):
        _build(applied_window_key=((0.2, 10.0), (0.3, 10.0)))
    # an inactive axis is inactive whatever placeholder centre (or NaN -> None) it carries
    snap = [dict(SNAP[0], k1=0.0), SNAP[1]]
    _build(snapshot_rows=snap, applied_window_key=((None, 0.0), (0.2, 10.0)))


def test_instance_less_table_is_refused_at_load(tmp_path):
    from gareus.auxiliary_cv.state_table import load_aux_state_table
    path = tmp_path / "m.json"
    MODEL.write(path)
    with pytest.raises(IntegrityError, match="instance"):
        load_aux_state_table(path, [{"aux_k_kcal_mol": "0"}, {"aux_k_kcal_mol": "2.0", "aux_center": "1.5"}])


def test_embed_cv_definition_replaces_paths_by_content(tmp_path):
    f = tmp_path / "m.json"; f.write_text("{}")
    out = embed_cv_definition("contact-map", "dimensionless", {"cv1_model_path": str(f), "r0": 8.0, "note_file": None})
    assert "cv1_model_content_sha256" in out["definition"] and "cv1_model_path" not in out["definition"]
    assert "note_file" not in out["definition"]
    with pytest.raises(IntegrityError, match="missing"):
        embed_cv_definition("x", "dimensionless", {"model_file": str(tmp_path / "nope.json")})


def test_physical_system_sha_is_deterministic_and_bias_sensitive():
    import openmm as mm
    from pep_gamd_fixture import _fresh_system
    s1, s2 = _fresh_system(), _fresh_system()
    assert physical_system_sha256(mm, s1) == physical_system_sha256(mm, s2)
    s3 = _fresh_system()
    a, b, c = s3.getDefaultPeriodicBoxVectors()
    s3.setDefaultPeriodicBoxVectors(a * 1.0001, b * 1.0001, c * 1.0001)       # PDB-rounded box on resume
    assert physical_system_sha256(mm, s3) == physical_system_sha256(mm, s1)
    s2.addForce(mm.CustomExternalForce("0"))
    assert physical_system_sha256(mm, s1) != physical_system_sha256(mm, s2)


def test_cli_drops_unpersisted_flag_and_adds_phase_inputs(tmp_path, capsys):
    from gareus.cli import parse_args
    base = ["--seq", "GA", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        parse_args(base + ["--aux-cv-allow-unpersisted"])
    assert "unrecognized arguments" in capsys.readouterr().err
    a = parse_args(base)
    assert a.aux_phase_kind == "pilot" and a.aux_equilibrium_eligible is False
    with pytest.raises(SystemExit):
        parse_args(base + ["--aux-cv-model", "m.json", "--aux-equilibrium-eligible"])   # needs production
    assert "aux-phase-kind production" in capsys.readouterr().err


def test_resume_is_no_longer_refused_at_parse(tmp_path):
    import inspect
    import gareus.cli as cli
    src = inspect.getsource(cli._validate_aux_cv_args)
    assert "allow_unpersisted" not in src and "cannot --resume" not in src
