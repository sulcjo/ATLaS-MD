import json

import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AUX_MODEL_SCHEMA, AuxModel, AuxModelError
from gareus.correctness._io import IntegrityError

QUADS = [(0, 1, 2, 3), (1, 2, 3, 4)]


def test_round_trip_keeps_identity(tmp_path):
    m = AuxModel.from_mapping(model_payload(QUADS, [0.5, -0.25, 0.0, 1.0], offset=0.3, scale=3.181))
    path = tmp_path / "aux.json"
    m.write(path)
    again = AuxModel.load(path)
    assert again == m
    assert again.model_sha256 == m.model_sha256
    assert json.loads(path.read_text())["model_sha256"] == m.model_sha256


def test_label_and_provenance_are_not_identity():
    a = AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0], label="a", provenance={"x": 1}))
    b = AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0], label="b", provenance={"y": 2}))
    assert a.model_sha256 == b.model_sha256
    assert a.identity_mapping() == b.identity_mapping()
    assert "label" not in a.identity_mapping() and "provenance" not in a.identity_mapping()


def test_coefficients_offset_scale_and_features_are_identity():
    base = AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0])).model_sha256
    assert AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 1e-9])).model_sha256 != base
    assert AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0], offset=1e-9)).model_sha256 != base
    assert AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0], scale=2.0)).model_sha256 != base
    assert AuxModel.from_mapping(model_payload(QUADS, [1, 0, 0, 0],
                                               conventions=["direct", "negated"])).model_sha256 != base


def test_negative_zero_does_not_change_identity():
    a = AuxModel.from_mapping(model_payload(QUADS, [1.0, 0.0, 0.0, 0.0], offset=0.0))
    b = AuxModel.from_mapping(model_payload(QUADS, [1.0, -0.0, -0.0, 0.0], offset=-0.0))
    assert a.model_sha256 == b.model_sha256


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p.update(coefficients=[1.0, 0.0, 0.0]), "coefficients"),
    (lambda p: p.update(coefficients=[0.0, 0.0, 0.0, 0.0]), "nonzero"),
    (lambda p: p.update(coefficients=[1.0, True, 0.0, 0.0]), "number"),
    (lambda p: p.update(offset=float("nan")), "offset|Nonfinite"),
    (lambda p: p.update(scale=0.0), "scale"),
    (lambda p: p.update(scale=-1.0), "scale"),
    (lambda p: p.update(periodic_imaging="minimum_image"), "periodic_imaging"),
    (lambda p: p.pop("scale"), "missing"),
    (lambda p: p.pop("periodic_imaging"), "missing"),
    (lambda p: p.update(units="angstrom"), "units"),
    (lambda p: p.update(extra=1), "unknown"),
    (lambda p: p.pop("offset"), "missing"),
    (lambda p: p.update(schema="atlas-aux-cv-model-v0"), "schema"),
])
def test_invalid_payloads_are_refused(mutate, message):
    payload = model_payload(QUADS, [1.0, 0.0, 0.0, 0.0])
    mutate(payload)
    with pytest.raises(IntegrityError, match=message):
        AuxModel.from_mapping(payload)


def test_tampered_claimed_digest_is_refused():
    payload = model_payload(QUADS, [1.0, 0.0, 0.0, 0.0])
    payload["model_sha256"] = "f" * 64
    with pytest.raises(AuxModelError, match="digest"):
        AuxModel.from_mapping(payload)


def test_schema_constant():
    assert AUX_MODEL_SCHEMA == "atlas-aux-cv-model-v1"
