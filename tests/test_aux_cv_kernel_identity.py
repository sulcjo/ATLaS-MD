# tests/test_aux_cv_kernel_identity.py
import csv
import json
import types

import numpy as np
import pytest

from aux_cv_fixture import model_payload
from gareus.auxiliary_cv.model import AuxModel
from gareus.kernel_identity import (ELIGIBLE_AUX_UNPERSISTED, ELIGIBLE_NOT_APPLICABLE,
                                    ELIGIBLE_UNKNOWN, ELIGIBLE_VERIFIED, EXCHANGE_ENERGY_VERSION,
                                    EXCHANGE_ENERGY_VERSION_AUX, RESIDUAL_EVALUATOR_VERSION,
                                    classify_segment_kernel, exchange_energy_version_for_args,
                                    kernel_identity_for_run)

LEGACY_ARGS = types.SimpleNamespace(secondary_cv="none", production_ensemble="npt",
                                    gamd_boost_type="pep-gamd-lower-dual")
# Digest of the legacy identity for LEGACY_ARGS, computed on main dc30285 (kernel_identity.py unchanged);
# re-confirmed by the verifier on unmodified code (2026-10-08).
LEGACY_DIGEST = "543f92e628d1777bdf6f9f228744eeb64159023e8ae16a36b2d775115d640bf6"


def _aux_args(tmp_path):
    m = AuxModel.from_mapping(model_payload([(0, 1, 2, 3)], [1.0, 0.0]))
    path = tmp_path / "aux.json"
    m.write(path)
    return types.SimpleNamespace(**vars(LEGACY_ARGS), aux_cv_model=str(path)), m


def test_legacy_identity_unchanged():
    ident = kernel_identity_for_run(LEGACY_ARGS, {})
    assert ident["exchange_energy_version"] == EXCHANGE_ENERGY_VERSION == "state_bias_matrix_v2"
    assert "aux_model_sha256" not in ident
    assert ident["digest"] == LEGACY_DIGEST


def test_legacy_identity_unchanged_when_the_aux_flag_is_present_but_unset():
    # parse_args always sets aux_cv_model (default None): a live non-aux campaign must keep its digest.
    args = types.SimpleNamespace(**vars(LEGACY_ARGS), aux_cv_model=None, _aux_runtime=None)
    ident = kernel_identity_for_run(args, {})
    assert ident["digest"] == LEGACY_DIGEST
    assert not any(k.startswith("aux_") for k in ident)
    assert exchange_energy_version_for_args(args) == EXCHANGE_ENERGY_VERSION


def test_aux_identity_and_version(tmp_path):
    args, m = _aux_args(tmp_path)
    assert exchange_energy_version_for_args(args) == EXCHANGE_ENERGY_VERSION_AUX == "state_bias_matrix_v3_aux"
    ident = kernel_identity_for_run(args, {})
    assert ident["exchange_energy_version"] == EXCHANGE_ENERGY_VERSION_AUX
    assert ident["aux_model_sha256"] == m.model_sha256
    assert ident["digest"] != LEGACY_DIGEST


def test_identity_uses_the_runtime_model_and_topology_without_rereading_the_file(tmp_path):
    args, m = _aux_args(tmp_path)
    args.aux_cv_model = str(tmp_path / "gone.json")            # the file is not read when a runtime exists
    args._aux_runtime = types.SimpleNamespace(info=types.SimpleNamespace(model_sha256=m.model_sha256),
                                              topology_sha256="c" * 64)
    ident = kernel_identity_for_run(args, {})
    assert ident["aux_model_sha256"] == m.model_sha256 and ident["aux_topology_sha256"] == "c" * 64


@pytest.mark.parametrize("cv2", ["residual-torsion-pc", "none", "torsion-pca"])
def test_unpersisted_aux_segments_are_ineligible_for_every_cv2(cv2):
    snap = {"cv2_type": cv2,
            "kernel_identity": {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                "exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX}}
    assert classify_segment_kernel(snap)[0] == ELIGIBLE_AUX_UNPERSISTED


def test_aux_model_sha_alone_marks_a_segment_aux():
    snap = {"cv2_type": "none", "kernel_identity": {"exchange_energy_version": EXCHANGE_ENERGY_VERSION,
                                                    "aux_model_sha256": "d" * 64}}
    assert classify_segment_kernel(snap)[0] == ELIGIBLE_AUX_UNPERSISTED


def test_v3_aux_without_payload_stays_unpersisted():
    """Stage C: persistence comes from the samples manifest payload, never from a snapshot key."""
    snap = {"cv2_type": "residual-torsion-pc", "aux_sample_schema": {"schema": "x"},
            "kernel_identity": {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                                "exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX}}
    assert classify_segment_kernel(snap)[0] == ELIGIBLE_AUX_UNPERSISTED


def test_legacy_classification_unchanged():
    res = {"cv2_type": "residual-torsion-pc",
           "kernel_identity": {"cv_evaluator_version": RESIDUAL_EVALUATOR_VERSION,
                               "exchange_energy_version": EXCHANGE_ENERGY_VERSION}}
    assert classify_segment_kernel(res)[0] == ELIGIBLE_VERIFIED
    assert classify_segment_kernel({"cv2_type": "none", "kernel_identity": {}})[0] == ELIGIBLE_NOT_APPLICABLE
    assert classify_segment_kernel({"cv2_type": "residual-torsion-pc"})[0] == ELIGIBLE_UNKNOWN
    affected = {"cv2_type": "residual-torsion-pc",
                "kernel_identity": {"cv_evaluator_version": None, "exchange_energy_version": "old"}}
    assert classify_segment_kernel(affected)[0] == "affected"


def _run_dir_with_snapshot(tmp_path, identity):
    run = tmp_path / "run"
    (run / "windows").mkdir(parents=True)
    (run / "segments.json").write_text(json.dumps([{"segment_id": "seg_001"}]))
    (run / "windows" / "seg_001.json").write_text(json.dumps({"segment_id": "seg_001", "cv1_type": "contacts",
                                                              "cv2_type": None, "windows": [],
                                                              "kernel_identity": identity}))
    return run


def test_segment_eligibility_excludes_unpersisted_aux_segments(tmp_path):
    from gareus.query import segment_eligibility
    run = _run_dir_with_snapshot(tmp_path, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX,
                                            "aux_model_sha256": "d" * 64})
    assert segment_eligibility(run)["seg_001"]["eligibility"] == ELIGIBLE_AUX_UNPERSISTED


def test_resume_of_an_aux_campaign_without_its_model_is_refused(tmp_path):
    from gareus.production import refuse_resume_of_aux_campaign
    run = _run_dir_with_snapshot(tmp_path, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX,
                                            "aux_model_sha256": "d" * 64})
    with pytest.raises(RuntimeError, match="auxiliary"):
        refuse_resume_of_aux_campaign(run, types.SimpleNamespace(aux_cv_model=None))


def test_resume_of_a_legacy_campaign_is_not_refused(tmp_path, capsys):
    from gareus.production import refuse_resume_of_aux_campaign
    refuse_resume_of_aux_campaign(_run_dir_with_snapshot(tmp_path, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION}),
                                  types.SimpleNamespace(aux_cv_model=None))
    refuse_resume_of_aux_campaign(tmp_path / "empty", types.SimpleNamespace(aux_cv_model=None))  # no windows/
    old = tmp_path / "old"
    (old / "windows").mkdir(parents=True)
    (old / "windows" / "seg_001.json").write_text(json.dumps({"segment_id": "seg_001", "windows": []}))
    refuse_resume_of_aux_campaign(old, types.SimpleNamespace(aux_cv_model=None))  # snapshot predating kernel identity
    assert capsys.readouterr().out == ""


def test_method_settings_record_aux_only_when_configured(tmp_path):
    from gareus.provenance import _method_settings
    from gareus.cli import parse_args
    a = parse_args(["--seq", "GA", "--out", str(tmp_path / "o"), "--cv1", "contacts"])
    s = _method_settings(a)
    assert s["exchange_energy_version"] == "state_bias_matrix_v2"
    assert "aux_cv_model_sha256" not in s and "aux_envelope_calibration" not in s
    args, m = _aux_args(tmp_path)
    a.aux_cv_model = args.aux_cv_model
    s2 = _method_settings(a)
    assert s2["exchange_energy_version"] == "state_bias_matrix_v3_aux"
    assert s2["aux_cv_model_sha256"] == m.model_sha256 and s2["aux_envelope_calibration"] == "aux_inactive"


# --- Controller ruling C3: the NPZ / CSV MBAR loaders refuse auxiliary-CV runs ----------------------------


def _write_windows_csv(prod):
    with (prod / "umbrella_windows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["center_A", "k_kcal_mol_A2"])
        w.writeheader()
        w.writerow({"center_A": "1.0", "k_kcal_mol_A2": "10.0"})


def _npz_prod(tmp_path):
    prod = tmp_path / "npz"
    prod.mkdir()
    n = 3
    np.savez(prod / "analysis_arrays.npz", cv_A=np.arange(n, dtype=float), window=np.zeros(n, int),
             replica=np.zeros(n, int), step=np.arange(n), umbrella_reduced_bias_nk=np.zeros((n, 1)))
    _write_windows_csv(prod)
    return prod


def _csv_prod(tmp_path):
    prod = tmp_path / "csv"
    prod.mkdir()
    _write_windows_csv(prod)
    with (prod / "samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cv_A", "window", "replica", "step"])
        w.writeheader()
        w.writerow({"cv_A": "1.0", "window": "0", "replica": "0", "step": "0"})
        w.writerow({"cv_A": "1.5", "window": "0", "replica": "0", "step": "1"})
    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    return prod


def _mark_manifest(prod, method):
    (prod / "run_manifest.json").write_text(json.dumps({"method_settings": method}))


def _mark_snapshot(prod, identity):
    (prod / "windows").mkdir(exist_ok=True)
    (prod / "windows" / "seg_001.json").write_text(json.dumps({"segment_id": "seg_001",
                                                              "kernel_identity": identity}))


def test_legacy_npz_and_csv_still_load_with_legacy_kernel_records(tmp_path):
    from gareus.mbar_analysis.loaders import load_csv, load_npz
    for prod, loader in ((_npz_prod(tmp_path), load_npz), (_csv_prod(tmp_path), load_csv)):
        _mark_manifest(prod, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION})
        _mark_snapshot(prod, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION})
        assert loader(prod).cv.size >= 2


@pytest.mark.parametrize("loader_name", ["load_npz", "load_csv"])
@pytest.mark.parametrize("marker", ["manifest_version", "manifest_sha", "snapshot_version", "snapshot_sha"])
def test_npz_and_csv_loaders_refuse_aux_runs(tmp_path, loader_name, marker):
    import gareus.mbar_analysis.loaders as loaders
    prod = _npz_prod(tmp_path) if loader_name == "load_npz" else _csv_prod(tmp_path)
    if marker == "manifest_version":
        _mark_manifest(prod, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX})
    elif marker == "manifest_sha":
        _mark_manifest(prod, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION, "aux_cv_model_sha256": "d" * 64})
    elif marker == "snapshot_version":
        _mark_snapshot(prod, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION_AUX})
    else:
        _mark_snapshot(prod, {"exchange_energy_version": EXCHANGE_ENERGY_VERSION, "aux_model_sha256": "d" * 64})
    with pytest.raises(RuntimeError, match="auxiliary"):
        getattr(loaders, loader_name)(prod)
