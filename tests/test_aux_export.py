# tests/test_aux_export.py
import json

import numpy as np
import pytest

from aux_c_fixture import definition, rows
from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.evaluate import z_from_dihedrals
from gareus.auxiliary_cv.model import AuxModel
from gareus.auxiliary_cv.offline import (aux_z_from_samples, exclusion_report, parity_context,
                                         parity_violation)
from gareus.auxiliary_cv.sample_schema import AuxSampleSchema, _basis_sha
from gareus.correctness._io import IntegrityError
from gareus.correctness.export import R_KJ_MOL_K, build_export_arrays
from gareus.correctness.state_identity import freeze_snapshot, make_state_definition

QUADS = ((0, 1, 2, 3), (1, 2, 3, 4))
LABELS = ("phi-A1", "psi-A1")
MODEL = AuxModel.from_mapping(model_payload(list(QUADS), [1.0, 0.0, 0.0, -0.7], offset=0.2,
                                            blocks=["phi", "psi"]))
SCHEMA = AuxSampleSchema(QUADS, LABELS, (MODEL.model_sha256,), _basis_sha(QUADS, LABELS))
BETA = 1.0 / (R_KJ_MOL_K * 300.0)          # the exporter's own constant (C8)
VIEW = {"kind": "immutable", "boundary_id": "b1"}
PREC = {"seg_001": "double"}


def _defn():
    return definition(rows([(0.2, 10.0, 0.0, 0.0, "ordinary"), (0.2, 10.0, 2.0, 0.5, "auxiliary")],
                           MODEL.model_sha256), MODEL)


def _samples(n=8, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=(n, 2))
    z = z_from_dihedrals(theta, MODEL)
    return {"cv1": rng.uniform(0, 0.4, n), "window_id": np.array([0, 1] * (n // 2)),
            "segment_id": np.array(["seg_001"] * n), "step": np.arange(n) * 100,
            "replica": np.array([0, 1] * (n // 2)), "gamd_lambda": np.zeros(n),
            "tor_000": theta[:, 0], "tor_001": theta[:, 1], "aux_z_00": z}


def _snap():
    return {"seg_001": freeze_snapshot("seg_001", _defn(), equilibrium_analysis_eligible=True, phase_kind="production")}


def _export(s, **kw):
    return build_export_arrays(s, _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA,
                               aux_segment_precision=PREC, **kw)


def test_ordinary_origin_rows_get_auxiliary_cross_energy_in_lambda_zero_table():
    s = _samples()
    u = _export(s)["umbrella_reduced_bias_nk"]
    z = z_from_dihedrals(np.stack([s["tor_000"], s["tor_001"]], axis=1), MODEL)
    np.testing.assert_allclose(u[:, 1] - u[:, 0], BETA * 4.184 * 0.5 * 2.0 * (z - 0.5) ** 2, rtol=1e-12)
    assert np.all(u[s["window_id"] == 0, 1] > u[s["window_id"] == 0, 0])


def test_missing_features_fail_closed():
    s = _samples()
    for key in ("tor_000", "tor_001", "aux_z_00"):
        s.pop(key)
    with pytest.raises(IntegrityError, match="auxiliary features"):
        _export(s)


def test_schema_and_precision_required():
    with pytest.raises(IntegrityError, match="aux_sample_schema"):
        build_export_arrays(_samples(), _snap(), BETA, sample_view=VIEW)
    other = AuxSampleSchema(QUADS, LABELS, ("f" * 64,), _basis_sha(QUADS, LABELS))
    with pytest.raises(IntegrityError, match="model"):
        build_export_arrays(_samples(), _snap(), BETA, sample_view=VIEW, aux_sample_schema=other,
                            aux_segment_precision=PREC)
    with pytest.raises(IntegrityError, match="precision"):
        build_export_arrays(_samples(), _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA)


def test_parity_tolerance_follows_segment_precision():
    s = _samples()
    s["aux_z_00"] = s["aux_z_00"] + 3e-6           # GPU-scale disagreement
    with pytest.raises(IntegrityError, match="parity"):
        _export(s)                                    # double: 1e-6 reduced
    a = build_export_arrays(s, _snap(), BETA, sample_view=VIEW, aux_sample_schema=SCHEMA,
                            aux_segment_precision={"seg_001": "mixed"})
    assert a["umbrella_reduced_bias_nk"].shape == (8, 2)


def test_parity_violation_is_conservative_and_dimensionless():
    du = parity_violation(np.array([1.0 + 1e-3]), np.array([1.0]), beta=BETA, k_max_kcal=2.0, centers=[0.5])
    exact = BETA * 4.184 * 0.5 * 2.0 * abs((1.0 + 1e-3 - 0.5) ** 2 - 0.5 ** 2)
    assert du[0] >= exact and du[0] < 2.5 * exact


def test_refuse_vs_exclude_and_report():
    s = _samples(n=8)
    s["tor_000"] = s["tor_000"].copy(); s["aux_z_00"] = s["aux_z_00"].copy()
    s["tor_000"][[0, 2]] = np.nan
    s["aux_z_00"][[0, 2]] = np.nan
    with pytest.raises(IntegrityError, match="Incomplete"):
        _export(s)
    a = _export(s, on_incomplete="exclude_and_report", time_block_steps=400)
    assert a["umbrella_reduced_bias_nk"].shape[0] == 6 and int(a["N_k"].sum()) == 6
    rep = json.loads(str(a["exclusion_report_json"].item()))
    assert rep["n_excluded"] == 2 and rep["by_origin_state"] == {"0": 2}
    assert rep["by_carrier"] == {"0": 2} and rep["by_time_block"] == {"seg_001:0": 2}
    assert rep["by_z_range"][MODEL.model_sha256]["nonfinite"] == 2
    assert rep["by_structural_group"].startswith("unavailable")
    man = json.loads(str(a["export_manifest_json"].item()))
    assert man["n_samples"] == 8 and man["n_kept"] == 6 and man["n_excluded"] == 2
    # Board condition 9: every exported per-sample array is aligned with the kept rows.
    n_kept = man["n_kept"]
    for key, value in a.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] in (8, n_kept) and key not in ("N_k",):
            assert arr.shape[0] == n_kept, f"exported array {key} has {arr.shape[0]} rows, expected {n_kept}"


def test_registry_lookup_is_integrity_error_not_keyerror():
    from gareus.auxiliary_cv.offline import registry_model
    with pytest.raises(IntegrityError, match="missing from the frozen state definition"):
        registry_model({"aux_models": {}}, "f" * 64, where="test")


def test_exclusion_report_bins_finite_z():
    excluded = np.array([True, False, True, False])
    rep = exclusion_report(excluded, origin_ids=np.array([1, 1, 1, 0]), replicas=np.array([3, 3, 4, 4]),
                           steps=np.array([0, 100, 900, 950]), segment_ids=np.array(["s"] * 4),
                           aux_z={"e" * 64: np.array([0.0, 1.0, 2.0, 3.0])}, time_block_steps=500, z_bins=3)
    assert rep["by_time_block"] == {"s:0": 1, "s:1": 1} and rep["by_carrier"] == {"3": 1, "4": 1}
    assert sum(rep["by_z_range"]["e" * 64]["counts"]) == 2 and rep["fraction"] == 0.5
    assert rep["above_audit_threshold"] is True


def test_legacy_v1_export_signature_pinned():
    rows_v1 = [{"window_id": 0, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": 0.0}]
    d = make_state_definition(rows_v1, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                              fixed_box_vectors_nm=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                              cv1={"kind": "contacts", "units": "dimensionless", "definition": {"r0": 4.5}}, cv2=None)
    snaps = {"seg_001": freeze_snapshot("seg_001", d, equilibrium_analysis_eligible=True, phase_kind="production")}
    s = {"cv1": np.array([0.1, 0.3]), "window_id": np.array([0, 0]), "segment_id": np.array(["seg_001"] * 2)}
    a = build_export_arrays(s, snaps, BETA, sample_view=VIEW)
    man = json.loads(str(a["export_manifest_json"].item()))
    assert man["input_signature"] == "4bb8e35b09a426d57c9723fc06ef70d9864b2553d4af28a49942a5e3be707b91"
    assert sorted(a) == ["N_k", "column_window_ids", "cv_A", "export_manifest_json", "original_window_id",
                         "segment_id", "umbrella_reduced_bias_nk", "window"]


# F10: observation identity and stored-dtype lambda comparison ---------------------------------------------

def test_aux_export_refuses_duplicate_conflicting_missing_keys():
    s = _samples()
    _export(s)
    dup = {k: np.asarray(v).copy() for k, v in s.items()}
    dup["step"][1] = dup["step"][3] = 100; dup["replica"][3] = dup["replica"][1]
    with pytest.raises(IntegrityError, match="duplicate"):
        _export(dup)
    conflict = dict(dup); conflict["cv1"] = np.asarray(dup["cv1"]).copy(); conflict["cv1"][3] += 1.0
    with pytest.raises(IntegrityError, match="duplicate"):
        _export(conflict)
    for key in ("step", "replica"):
        missing = {k: v for k, v in s.items() if k != key}
        with pytest.raises(IntegrityError, match=key):
            _export(missing)


def test_aux_export_nk_not_silently_changed_by_added_duplicate():
    s = _samples()
    base = _export(s)["N_k"].tolist()
    grown = {k: np.concatenate([np.asarray(v), np.asarray(v)[:1]]) for k, v in s.items()}
    with pytest.raises(IntegrityError, match="duplicate"):
        _export(grown)
    assert base == [4, 4]


def test_aux_export_equal_steps_in_different_phases_are_not_duplicates():
    s = _samples()
    s["step"] = np.zeros(8, dtype=np.int64); s["replica"] = np.zeros(8, dtype=np.int64)
    s["phase_id"] = np.array([f"p{i}" for i in range(8)])
    assert int(_export(s)["N_k"].sum()) == 8
    s["phase_id"] = np.array(["p0"] * 8)
    with pytest.raises(IntegrityError, match="duplicate"):
        _export(s)


def _lam_export(stored_lambda, frozen_lambda, dtype):
    from aux_c_fixture import definition, rows
    r = rows([(0.2, 10.0, 0.0, 0.0, "ordinary")], MODEL.model_sha256)
    r = [dict(x, gamd_lambda=frozen_lambda) for x in r]
    from aux_c_fixture import BOX, CV1
    d = make_state_definition(r, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                              fixed_box_vectors_nm=BOX, cv1=CV1, cv2=None, boost={"kind": "pep-gamd", "envelope": {"vmax": 1.0}},
                              aux_models={MODEL.model_sha256: MODEL.to_mapping()})
    n = 4
    s = {"cv1": np.linspace(0.1, 0.3, n), "window_id": np.zeros(n, dtype=np.int64),
         "segment_id": np.array(["seg_001"] * n), "step": np.arange(n), "replica": np.zeros(n, dtype=np.int64),
         "gamd_lambda": np.full(n, stored_lambda, dtype=dtype)}
    snap = {"seg_001": freeze_snapshot("seg_001", d, equilibrium_analysis_eligible=True, phase_kind="production")}
    env = (lambda boost: None)
    return build_export_arrays(s, snap, BETA, sample_view=VIEW, envelope_factory=env, reconstruct=lambda cv1, cv2, w, b, **k: np.zeros((len(cv1), len(w))))


@pytest.mark.parametrize("lam", [0.0, 1.0, 0.1])
def test_float32_stored_lambda_round_trips(lam):
    assert _lam_export(lam, lam, np.float32)["N_k"].tolist() == [4]


def test_wrong_and_adjacent_float32_lambda_refused_float64_exact():
    nxt = float(np.nextafter(np.float32(0.1), np.float32(1.0)))
    with pytest.raises(IntegrityError, match="gamd_lambda"):
        _lam_export(nxt, 0.1, np.float32)
    with pytest.raises(IntegrityError, match="gamd_lambda"):
        _lam_export(0.2, 0.1, np.float32)
    assert _lam_export(0.1, 0.1, np.float64)["N_k"].tolist() == [4]
    with pytest.raises(IntegrityError, match="gamd_lambda"):
        _lam_export(float(np.float32(0.1)), 0.1, np.float64)


# Final fix wave M7: flag-off pins -----------------------------------------------------------------------------

def _non_aux_lam_export(stored, frozen=(0.0, 0.1)):
    """A non-aux (no aux model, no aux row) 2-rung ladder export with a float64 gamd_lambda column."""
    from aux_c_fixture import BOX, CV1
    r = [{"window_id": i, "center1": 0.2, "k1": 10.0, "center2": 0.0, "k2": 0.0, "gamd_lambda": lam}
         for i, lam in enumerate(frozen)]
    d = make_state_definition(r, physical_system_sha256="a" * 64, ensemble="NVT", temperature_k=300.0,
                              fixed_box_vectors_nm=BOX, cv1=CV1, cv2=None,
                              boost={"kind": "pep-gamd", "envelope": {"vmax": 1.0}})
    n = 4
    s = {"cv1": np.linspace(0.1, 0.3, n), "window_id": np.array([0, 1, 0, 1]),
         "segment_id": np.array(["seg_001"] * n), "gamd_lambda": np.asarray(stored, dtype=np.float64)}
    snap = {"seg_001": freeze_snapshot("seg_001", d, equilibrium_analysis_eligible=True, phase_kind="production")}
    return build_export_arrays(s, snap, BETA, sample_view=VIEW, envelope_factory=lambda boost: None,
                               reconstruct=lambda cv1, cv2, w, b, **k: np.zeros((len(cv1), len(w))))


def test_non_aux_float64_lambda_export_acceptance_unchanged():
    from gareus.correctness.observation_keys import lambda_equals_frozen
    assert _non_aux_lam_export([0.0, 0.1, 0.0, 0.1])["N_k"].tolist() == [2, 2]
    for bad in ([0.0, float(np.float32(0.1)), 0.0, 0.1], [0.0, 0.2, 0.0, 0.1], [0.1, 0.0, 0.0, 0.1]):
        with pytest.raises(IntegrityError, match="gamd_lambda"):
            _non_aux_lam_export(bad)
    # float64 columns: the new comparison is exactly the old np.array_equal rule
    rng = np.random.default_rng(3)
    frozen = rng.choice([0.0, 0.1, 0.47, 1.0], size=64)
    for stored in (frozen.copy(), np.where(rng.random(64) < 0.1, np.nextafter(frozen, 2.0), frozen),
                   frozen.astype(np.float32).astype(np.float64)):
        assert bool(lambda_equals_frozen(stored, frozen).all()) == bool(np.array_equal(stored, frozen))


def test_gamd_prep_steps_cli_defaults_are_5000(tmp_path):
    from gareus.cli import build_gareus_parser, parse_args
    a = parse_args(["--seq", "GA", "--cv1", "contacts", "--out", str(tmp_path / "o")])
    assert a.gamd_cmd_prep_steps == 5000 and a.gamd_equil_prep_steps == 5000
    p = build_gareus_parser()
    assert p.get_default("gamd_cmd_prep_steps") == 5000 and p.get_default("gamd_equil_prep_steps") == 5000
