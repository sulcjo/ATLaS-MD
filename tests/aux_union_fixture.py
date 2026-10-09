"""Small three-phase adaptive campaign with one admitted aux worker (Tasks 13/14 union pooling).

epoch_000 = pre-admission (ordinary states 0, 1; z only in aux_z_backfill.parquet);
epoch_001 and final = post-admission (states 0, 1 and worker 2; z recorded as aux_z_00).
The worker's burnin_phase_epoch is 1, so its epoch_001 samples are burn-in.
Steps are unique across the whole campaign so (step) alone identifies a sample.
"""
from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.adaptive_production import WindowStateRegistry
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha

QUADS = ((0, 1, 2, 3), (1, 2, 3, 4))
LABELS = ("phi-A1", "psi-A1")
MODEL = AuxModel.from_mapping(model_payload(list(QUADS), [1.0, 0.0, 0.0, -0.7], offset=0.2, blocks=["phi", "psi"]))
SCHEMA = AuxSampleSchema(QUADS, LABELS, (MODEL.model_sha256,), _basis_sha(QUADS, LABELS))
RUNTIME = {"platform": "Reference", "precision": "double"}
KI = {"kernel_identity_version": "kernel_identity_v1", "exchange_energy_version": "state_bias_matrix_v3_aux",
      "aux_model_sha256": MODEL.model_sha256}
CENTRES = (0.2, 0.5, 0.2)          # primary centres of states 0, 1 and the worker (2, child of 0)
K1 = 10.0
AUX_CENTER, AUX_K = 0.4, 2.0
ROWS_PER_WINDOW = 40
BURNIN_PHASE_EPOCH = 1


def _registry():
    reg = WindowStateRegistry()
    reg.add_state(CENTRES[0], K1)
    reg.add_state(CENTRES[1], K1)
    reg.add_state(CENTRES[2], K1, parent_state_id=0, epoch=0, source="adaptive_production_aux", metadata={"aux": {
        "role": "auxiliary", "aux_center": AUX_CENTER, "aux_k_kcal_mol": AUX_K,
        "aux_model_sha256": MODEL.model_sha256, "state_instance_id": None, "spawn_parent_state_id": 0,
        "admitted_epoch": 0, "burnin_phase_epoch": BURNIN_PHASE_EPOCH}})
    return reg


def _write_map(phase, n_windows):
    with (phase / "epoch_window_map.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch_window", "state_id", "primary_center", "primary_k",
                                           "secondary_center", "secondary_k"])
        w.writeheader()
        for i in range(n_windows):
            w.writerow({"epoch_window": i, "state_id": i, "primary_center": CENTRES[i], "primary_k": K1,
                        "secondary_center": "", "secondary_k": ""})


def _write_phase(phase, n_windows, aux, step0, rng, truth, label):
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot
    phase.mkdir(parents=True, exist_ok=True)
    (phase / "gareus_metadata.json").write_text(json.dumps({"temperature_K": 300.0}))
    seg_reg = SegmentRegistry(phase)
    seg = seg_reg.open_segment("run_001", None, 1)
    writer = ParquetSampleWriter(phase / "samples" / seg, aux_schema=SCHEMA if aux else None,
                                 aux_runtime=RUNTIME if aux else None, flush_rows=10000)
    step, last = step0, step0
    for w in range(n_windows):
        for i in range(ROWS_PER_WINDOW):
            step += 10
            replica = i % 2
            cv = float(rng.normal(CENTRES[w], 0.03))
            z = float(rng.normal(0.4, 0.3))
            truth[(label, replica, step)] = z
            kw = {}
            if aux:
                kw = dict(torsions=[0.1, 0.2], aux_z=[z])
            writer.write_sample(step=step, replica=replica, window_id=w, cv1=cv, cv2=None, potential=0.0,
                                boost_total=None, boost_dihedral=None, boost_nonbonded=None, **kw)
            last = step
    writer.close()
    seg_reg.close_segment(seg, last)
    if aux:
        states = rows([(CENTRES[0], K1, 0.0, 0.0, "ordinary"), (CENTRES[1], K1, 0.0, 0.0, "ordinary"),
                       (CENTRES[2], K1, AUX_K, AUX_CENTER, "auxiliary")], MODEL.model_sha256)
        WindowSnapshot(phase).snapshot(seg, [], "contacts", None, kernel_identity=KI,
                                       state_definition=definition(states, MODEL),
                                       equilibrium_analysis_eligible=True, phase_kind="production")
    else:
        WindowSnapshot(phase).snapshot(seg, [{"window_id": i, "center1": CENTRES[i], "k1": K1}
                                             for i in range(n_windows)], "contacts", None)
    _write_map(phase, n_windows)
    return last


def _write_backfill(phase, label, truth):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from gareus.adaptive.aux_backfill import BACKFILL_FILENAME
    recs = sorted((r, s, z) for (lab, r, s), z in truth.items() if lab == label)
    table = pa.table({"replica": pa.array([r for r, _, _ in recs], pa.int64()),
                      "step": pa.array([s for _, s, _ in recs], pa.int64()),
                      "aux_z": pa.array([z for _, _, z in recs], pa.float64())})
    pq.write_table(table.replace_schema_metadata({b"aux_model_sha256": MODEL.model_sha256.encode()}),
                   phase / BACKFILL_FILENAME)


def build_campaign(tmp_path, *, backfill=True, admission=True):
    """Adaptive dir with state_registry files, frozen aux files and three phases."""
    ad = tmp_path / "adaptive"
    ad.mkdir(parents=True)
    rng = np.random.default_rng(7)
    truth: dict = {}
    reg = _registry()
    reg.save(ad)
    _write_phase(ad / "epoch_000", 2, False, 0, rng, truth, "epoch_000")
    _write_phase(ad / "epoch_001", 3, True, 100_000, rng, truth, "epoch_001")
    _write_phase(ad / "final", 3, True, 200_000, rng, truth, "final")
    if backfill:
        _write_backfill(ad / "epoch_000", "epoch_000", truth)
    MODEL.write(ad / "aux_model.json")
    if admission:
        (ad / "aux_admission.json").write_text(json.dumps({
            "schema": "atlas-aux-admission-v1", "epoch": 0, "model_sha256": MODEL.model_sha256,
            "workers": [{"parent_state_id": 0, "aux_center": AUX_CENTER, "aux_k_kcal_mol": AUX_K,
                         "placement_rank": 0, "burnin_phase_epoch": BURNIN_PHASE_EPOCH}], "backfill": []}))
    return SimpleNamespace(ad=ad, registry=reg, truth=truth, model=MODEL, worker_id=2)


def expected_reduced(cv, z, beta):
    """Reduced bias of every state at the given samples, via the strict shared reconstruction."""
    from gareus.correctness.bias import reconstruct_bias_matrix
    windows = [{"window_id": 0, "center1": CENTRES[0], "k1": K1, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0},
               {"window_id": 1, "center1": CENTRES[1], "k1": K1, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0},
               {"window_id": 2, "center1": CENTRES[2], "k1": K1, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0,
                "aux_k": AUX_K, "aux_center": AUX_CENTER, "aux_model_sha256": MODEL.model_sha256}]
    return reconstruct_bias_matrix(np.asarray(cv, float), None, windows, beta, aux_z={MODEL.model_sha256: np.asarray(z, float)})
