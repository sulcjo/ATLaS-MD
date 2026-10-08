# tests/test_aux_checkpoint.py
import numpy as np
import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import dipeptide, model_payload
from gareus.auxiliary_cv.checkpoint import (aux_checkpoint_block, aux_table_from_checkpoint,
                                            expected_aux_parameters, read_aux_parameters,
                                            topology_identity_sha256, verify_aux_ledger, verify_aux_resume)
from gareus.auxiliary_cv.force import AuxForceInfo, build_aux_force, set_aux_parameters
from gareus.auxiliary_cv.ledger import EXCHANGE_EVENT_SCHEMA
from gareus.auxiliary_cv.model import AuxModel
from gareus.correctness._io import IntegrityError

MODEL = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5], offset=0.1))
INFO = AuxForceInfo("ATLaSAuxCVUmbrella", 7, MODEL.model_sha256, "aux_k", "aux_c", ("aux_cos_neg", "aux_sin_neg"))
TOPO = "c" * 64
KID = "d" * 64


def _defn(model=MODEL, aux_k=1.2):
    spec = [(0.0, 10.0, 0.0, 0.0, "ordinary"), (0.2, 10.0, 0.0, 0.0, "ordinary"), (0.4, 10.0, aux_k, 1.5, "auxiliary")]
    return definition(rows(spec, model.model_sha256), model)


def _observed(defn, assignments):
    return [expected_aux_parameters(defn, w) for w in assignments]


def _block(d, a, **kw):
    base = dict(state_definition=d, force_info=INFO, assignments=a, observed_params=_observed(d, a),
                topology_sha256=TOPO, kernel_identity_digest=KID, segment_id="seg_001",
                ledger_anchor={"segment_id": "seg_001", "start_step": 0, "start_assignments": [0, 1, 2]},
                data_boundary={"samples": {"generation": 3, "n_rows": 30}, "exchanges": {"generation": 2, "n_rows": 9}})
    base.update(kw)
    return aux_checkpoint_block(**base)


def _verify_kw(d, a):
    return dict(aux_enabled=True, state_definition=d, force_info=INFO, assignments=a,
                observed_params=_observed(d, a), topology_sha256=TOPO, kernel_identity_digest=KID)


def test_expected_parameters_units():
    d = _defn()
    assert expected_aux_parameters(d, 0) == (0.0, 0.0)
    assert expected_aux_parameters(d, 2) == pytest.approx((1.2 * 4.184, 1.5), rel=1e-15)


def test_save_time_verification_refuses_a_bad_context():
    d, a = _defn(), [2, 0, 1]
    with pytest.raises(IntegrityError, match="save"):
        _block(d, a, observed_params=[(0.0, 0.0)] * 3)


def test_block_round_trip_verifies():
    d, a = _defn(), [2, 0, 1]
    verify_aux_resume({"aux": _block(d, a)}, **_verify_kw(d, a))


@pytest.mark.parametrize("case, match", [
    ("changed_model", "state_definition_sha256"),
    ("force_group", "force_info"),
    ("carrier_count", "replica"),
    ("pre_apply_params", "before re-apply"),
    ("assignments", "assignment"),
    ("aux_off", "capability"),
    ("topology", "topology"),
    ("kernel", "kernel"),
])
def test_resume_refusals(case, match):
    d, a = _defn(), [2, 0, 1]
    manifest = {"aux": _block(d, a)}
    kw = _verify_kw(d, a)
    if case == "changed_model":
        other = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.5000001], offset=0.1))
        kw["state_definition"] = _defn(other)
    elif case == "force_group":
        kw["force_info"] = AuxForceInfo(INFO.name, 8, INFO.model_sha256, INFO.global_k, INFO.global_c, INFO.sub_cv_names)
    elif case == "carrier_count":
        kw["assignments"], kw["observed_params"] = a[:2], _observed(d, a[:2])
    elif case == "pre_apply_params":
        kw["observed_params"] = [(0.0, 0.0)] * 3        # checkpoint restored an inactive aux force
    elif case == "assignments":
        kw["assignments"] = [0, 2, 1]
        kw["observed_params"] = _observed(d, [0, 2, 1])
    elif case == "aux_off":
        kw["aux_enabled"] = False
    elif case == "topology":
        kw["topology_sha256"] = "e" * 64
    elif case == "kernel":
        kw["kernel_identity_digest"] = "f" * 64
    with pytest.raises(IntegrityError, match=match):
        verify_aux_resume(manifest, **kw)


def test_aux_enabled_against_legacy_manifest_refuses():
    d, a = _defn(), [2, 0, 1]
    with pytest.raises(IntegrityError, match="capability"):
        verify_aux_resume({}, **_verify_kw(d, a))


def test_ledger_cross_check(tmp_path):
    from gareus.query import load_exchanges
    from gareus.store import ParquetExchangeWriter
    d, a = _defn(), [1, 0, 2]
    w = ParquetExchangeWriter(tmp_path / "exchanges" / "seg_001", flush_rows=1, event_schema=EXCHANGE_EVENT_SCHEMA)
    w.write_event(step=100, attempt_seq=0, selected_replica=0, replica_i=0, replica_j=1, window_i=0, window_j=1,
                  kind="swap", delta_e_kj=0.0, accepted=True, log_q_forward=0.0, log_q_reverse=0.0, p_accept=1.0,
                  energy_version="v3", assignments_after=[1, 0, 2])
    w.close()
    manifest = {"aux": _block(d, a), "absolute_step": 100, "assignments": a}
    verify_aux_ledger(manifest, load_exchanges(tmp_path))
    with pytest.raises(IntegrityError, match="ledger"):
        verify_aux_ledger(dict(manifest, assignments=[0, 1, 2]), load_exchanges(tmp_path))


def test_topology_identity_is_stable_and_sensitive():
    d = dipeptide()
    assert topology_identity_sha256(d["topology"]) == topology_identity_sha256(d["topology"])
    from openmm import app
    t2 = app.Topology()
    c = t2.addChain(); r = t2.addResidue("ALA", c); t2.addAtom("CA", app.element.carbon, r)
    assert topology_identity_sha256(t2) != topology_identity_sha256(d["topology"])


def test_aux_table_from_checkpoint_round_trip():
    d, a = _defn(), [2, 0, 1]
    table = aux_table_from_checkpoint({"aux": _block(d, a)}, model=MODEL)
    assert table.n == 3 and table.k_kcal == (0.0, 0.0, 1.2) and table.centers[2] == 1.5
    assert table.instances[2]["state_role"] == "auxiliary"
    other = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [0.0, 1.0]))
    with pytest.raises(IntegrityError, match="model"):
        aux_table_from_checkpoint({"aux": _block(d, a)}, model=other)


def test_read_parameters_from_a_real_context():
    import openmm as mm
    d = dipeptide()
    blocks = [lab.split("-")[0] for lab in d["labels"]]
    m = AuxModel.from_mapping(model_payload(d["quads"], [1.0] + [0.0] * (2 * len(d["quads"]) - 1), blocks=blocks))
    s = mm.System()
    for _ in range(len(d["positions_nm"])):
        s.addParticle(1.0)
    force, info = build_aux_force(mm, m, force_group=5)
    s.addForce(force)
    ctx = mm.Context(s, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    set_aux_parameters(ctx, info, center=0.7, k_kcal=2.0)
    assert read_aux_parameters(ctx, info) == pytest.approx((2.0 * 4.184, 0.7), rel=1e-15)
    chk = ctx.createCheckpoint()
    set_aux_parameters(ctx, info, center=0.0, k_kcal=0.0)
    ctx.loadCheckpoint(chk)                               # checkpoints restore global parameters
    assert read_aux_parameters(ctx, info) == pytest.approx((2.0 * 4.184, 0.7), rel=1e-15)


def test_state_change_names_the_differing_field():
    """M3: a resume-time drift names its field, so a non-reproducible input is diagnosable."""
    d, a = _defn(), [2, 0, 1]
    kw = _verify_kw(d, a)
    kw["state_definition"] = _defn(aux_k=1.3)
    with pytest.raises(IntegrityError, match=r"windows\[2\]\.aux_k"):
        verify_aux_resume({"aux": _block(d, a)}, **kw)


def test_aux_block_survives_publish_generation(tmp_path):
    """Preflight T6: the published root manifest keeps the aux block (checkpoint_store preserves caller fields)."""
    import json
    from gareus.correctness.checkpoint_store import checkpoint_manifest_path, publish_generation
    d, a = _defn(), [2, 0, 1]
    block = _block(d, a)
    fields = {"assignments": a, "prod_done": 0, "absolute_step": 0, "parity": 0, "attempt": 0,
              "next_exchange": 0, "next_log": 0, "rng_state": np.random.default_rng(0).bit_generator.state,
              "rng_bit_generator": "PCG64", "exchange_stats": {}, "aux": block}
    publish_generation(tmp_path, [b"r0", b"r1", b"r2"], fields)
    root = json.loads(checkpoint_manifest_path(tmp_path).read_text())
    assert root["aux"] == json.loads(json.dumps(block))
    verify_aux_resume(root, **_verify_kw(d, a))


def test_production_checkpoint_hooks_are_optional_and_ordered():
    import inspect
    import gareus.production as production
    save = inspect.getsource(production.save_production_checkpoint)
    assert "if aux_block is not None:\n        manifest[\"aux\"] = aux_block" in save
    load = inspect.getsource(production.load_production_checkpoint)
    assert load.index("aux_pre_apply(manifest, assignments)") < load.index("def _apply_assignment")
