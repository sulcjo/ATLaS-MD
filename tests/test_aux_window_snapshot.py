# tests/test_aux_window_snapshot.py
import hashlib
import json

import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError
from gareus.correctness.state_identity import validate_fixed_state_segments

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))


def _defn():
    return definition(rows([(0.2, 10.0, 0.0, 0.0, "ordinary"), (0.2, 10.0, 2.0, 1.0, "auxiliary")],
                           MODEL.model_sha256), MODEL)


def test_legacy_snapshot_bytes_pinned(tmp_path):
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [{"window_id": 0, "center1": 0.1, "k1": 1.0}], "contacts", None)
    assert hashlib.sha256((tmp_path / "windows" / "seg_001.json").read_bytes()).hexdigest() == \
        "45a1984cff98cceb61f95a4ecaba684571e0a172f691f9f61769670d2f1d8cda"
    WindowSnapshot(tmp_path).snapshot("seg_002", [{"window_id": 0, "center1": 0.1, "k1": 1.0}], "contacts", None,
                                      kernel_identity={"kernel_identity_version": "kernel_identity_v1", "digest": "d"})
    assert hashlib.sha256((tmp_path / "windows" / "seg_002.json").read_bytes()).hexdigest() == \
        "23b5638ed577d59eab82e811c101b30e8d13063397551e18590f1e7053840af8"


def test_aux_snapshot_is_a_frozen_v2_snapshot(tmp_path):
    from gareus.query import load_windows
    from gareus.store import WindowSnapshot
    WindowSnapshot(tmp_path).snapshot("seg_001", [], "contacts", None, state_definition=_defn(),
                                      equilibrium_analysis_eligible=True)
    payload = json.loads((tmp_path / "windows" / "seg_001.json").read_text())
    table = validate_fixed_state_segments({"seg_001": payload})
    assert table.definition["aux_models"]
    assert load_windows(tmp_path, "seg_001")[1]["aux_k"] == 2.0


def test_aux_snapshot_cv_kinds_must_match_definition(tmp_path):
    from gareus.store import WindowSnapshot
    with pytest.raises(ValueError, match="cv2"):
        WindowSnapshot(tmp_path).snapshot("seg_001", [], "contacts", "residual-torsion-pc", state_definition=_defn(),
                                          equilibrium_analysis_eligible=True)
    with pytest.raises(ValueError, match="cv1"):
        WindowSnapshot(tmp_path).snapshot("seg_001", [], "distance", None, state_definition=_defn(),
                                          equilibrium_analysis_eligible=True)


def test_aux_snapshot_is_immutable(tmp_path):
    from gareus.store import WindowSnapshot
    s = WindowSnapshot(tmp_path)
    s.snapshot("seg_001", [], "contacts", None, state_definition=_defn(), equilibrium_analysis_eligible=True)
    s.snapshot("seg_001", [], "contacts", None, state_definition=_defn(), equilibrium_analysis_eligible=True)
    with pytest.raises(IntegrityError, match="immutable"):
        s.snapshot("seg_001", [], "contacts", None, state_definition=_defn(), equilibrium_analysis_eligible=False)
