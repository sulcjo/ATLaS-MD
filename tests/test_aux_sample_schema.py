# tests/test_aux_sample_schema.py
import json

import numpy as np
import pytest

from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.evaluate import z_from_positions
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.sample_schema import (AUX_SAMPLES_SCHEMA, PARITY_TOLERANCE, AuxSampleSchema,
                                               build_sample_schema, observe_carrier, payload_with_runtime,
                                               runtime_from_payload, runtime_info, runtime_precision)
from gareus.correctness._io import IntegrityError


def _model(d, which):
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    coeffs = np.zeros(2 * len(d["quads"]))
    coeffs[which] = 1.0
    return AuxModel.from_mapping(model_payload(d["quads"], coeffs, offset=0.1, blocks=blocks))


def _peptide_atoms(d):
    return sorted({a for q in d["quads"] for a in q})


def test_schema_round_trip_and_identity():
    d = dipeptide()
    m = _model(d, 0)
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [m])
    again = AuxSampleSchema.from_payload(s.to_payload())
    assert again == s and s.to_payload()["schema"] == AUX_SAMPLES_SCHEMA
    assert s.model_shas == (m.model_sha256,)
    assert s.torsion_columns == tuple(f"tor_{i:03d}" for i in range(len(s.torsion_quads)))
    assert s.z_columns == ("aux_z_00",)


def test_payloads_are_json_canonical():
    d = dipeptide()
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [_model(d, 0)])
    for m in (s.to_payload(), payload_with_runtime(s, runtime_info("CUDA", "mixed"))):
        assert json.loads(json.dumps(m)) == m
    assert all(isinstance(a, int) for q in s.to_payload()["torsion_quads"] for a in q)


def test_runtime_block_is_not_schema_identity():
    d = dipeptide()
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [_model(d, 0)])
    p = payload_with_runtime(s, runtime_info("CUDA", "mixed"))
    assert AuxSampleSchema.from_payload(p) == s
    assert runtime_from_payload(p) == {"platform": "CUDA", "precision": "mixed"}
    assert runtime_from_payload(s.to_payload()) is None
    with pytest.raises(IntegrityError, match="precision"):
        runtime_info("CUDA", "quad")


def test_basis_sha_changes_with_basis():
    d = dipeptide()
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [_model(d, 0)])
    payload = s.to_payload()
    payload["torsion_labels"] = list(reversed(payload["torsion_labels"]))
    with pytest.raises(IntegrityError, match="basis_sha256"):
        AuxSampleSchema.from_payload(payload)


def test_observation_matches_stage_a_evaluator():
    d = dipeptide()
    m = _model(d, 3)
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [m])
    obs = observe_carrier(d["positions_nm"], s, {m.model_sha256: m})
    assert obs.torsions.dtype == np.float64 and obs.z.dtype == np.float64
    assert obs.torsions.shape == (len(s.torsion_quads),) and obs.z.shape == (1,)
    assert obs.z[0] == z_from_positions(d["positions_nm"], m)[0]


def test_model_basis_index_maps_first_appearance_order_explicitly():
    d = dipeptide()
    quads = list(reversed(d["quads"]))
    blocks = list(reversed([lab.split("-")[0] for lab in d["labels"]]))
    coeffs = np.arange(1.0, 2 * len(quads) + 1)
    m = AuxModel.from_mapping(model_payload(quads, coeffs, blocks=blocks))
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [m])
    idx = s.model_basis_index(m)
    assert [s.torsion_quads[i] for i in idx] == [tuple(q) for q in quads]
    obs = observe_carrier(d["positions_nm"], s, {m.model_sha256: m})
    assert obs.z[0] == z_from_positions(d["positions_nm"], m)[0]


def test_model_outside_basis_is_refused():
    d = dipeptide()
    other = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
    with pytest.raises(IntegrityError, match="basis"):
        build_sample_schema(d["topology"], _peptide_atoms(d), [other])


def test_more_than_one_model_is_stage_f():
    d = dipeptide()
    with pytest.raises(IntegrityError, match="Stage F"):
        build_sample_schema(d["topology"], _peptide_atoms(d), [_model(d, 0), _model(d, 1)])


def test_observation_requires_every_schema_model():
    d = dipeptide()
    m = _model(d, 0)
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [m])
    with pytest.raises(IntegrityError, match="model"):
        observe_carrier(d["positions_nm"], s, {})


def test_runtime_precision_of_reference_and_cpu_contexts():
    import openmm as mm
    s = mm.System(); s.addParticle(1.0)
    for name, expected in (("Reference", "double"), ("CPU", "mixed")):
        plat = mm.Platform.getPlatformByName(name)
        ctx = mm.Context(s, mm.VerletIntegrator(0.001), plat)
        assert runtime_precision(plat.getName(), plat, ctx) == expected
    assert runtime_precision("SomethingElse") == "single"
    assert PARITY_TOLERANCE == {"double": 1e-6, "mixed": 1e-4, "single": 1e-4}


def test_runtime_block_carries_the_z_source_through_the_payload():
    """F07: the force/positions sources survive payload_with_runtime -> runtime_from_payload; without them the
    block is byte-identical to before (pinned dicts elsewhere), and a lone or unknown source is refused."""
    from gareus.auxiliary_cv.sample_schema import runtime_from_payload
    d = dipeptide()
    s = build_sample_schema(d["topology"], _peptide_atoms(d), [_model(d, 0)])
    info = runtime_info("CUDA", "mixed", z_source="force", z_reference="positions")
    assert info == {"platform": "CUDA", "precision": "mixed", "aux_z_source": "force", "aux_z_reference": "positions"}
    assert runtime_info("CUDA", "mixed") == {"platform": "CUDA", "precision": "mixed"}
    assert runtime_from_payload(payload_with_runtime(s, info)) == info
    assert runtime_from_payload({"runtime": info}) == info
    for bad in ({"z_source": "force"}, {"z_source": "positions", "z_reference": "positions"},
                {"z_source": "force", "z_reference": "force"}):
        with pytest.raises(IntegrityError, match="source"):
            runtime_info("CUDA", "mixed", **bad)
