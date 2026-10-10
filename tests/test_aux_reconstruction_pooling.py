"""Task 10 (F02): z reconstruction governs pooling of an admitted aux campaign, in BOTH union paths.

A failed or missing-when-due spot check, a post-admission phase without recorded z, and a backfill whose
bytes are not the ones the admission record hashed all refuse pooling (never a warning)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aux_union_fixture import MODEL, build_campaign  # noqa: E402

from gareus.adaptive.aux_backfill import BACKFILL_FILENAME  # noqa: E402
from gareus.kernel_identity import AuxPoolingRefused, aux_admission_allows_pooling  # noqa: E402


def _driver(camp):
    from gareus.adaptive_production import build_union_state_mbar_inputs
    return build_union_state_mbar_inputs(camp.ad, camp.registry, output_prefix="t10")


def _analyzer(camp):
    from gareus.mbar_analysis.loaders_union_parquet import load_parquet_adaptive_union
    return load_parquet_adaptive_union(camp.ad, n_workers=1)


BOTH = pytest.mark.parametrize("build", [_driver, _analyzer], ids=["driver", "analyzer"])


@BOTH
def test_default_fixture_pools(tmp_path, build):
    build(build_campaign(tmp_path))


@BOTH
def test_failed_spot_check_refuses(tmp_path, build):
    camp = build_campaign(tmp_path, spot_check={"epoch": 1, "ok": False, "max_energy_err_kt": 100.0, "tol_kt": 1.0,
                                                "phase": "epoch_001", "n_compared": 10})
    with pytest.raises(AuxPoolingRefused, match=r"spot check.*100"):
        build(camp)


@BOTH
def test_spot_check_without_ok_true_refuses(tmp_path, build):
    camp = build_campaign(tmp_path, spot_check={"epoch": 1, "status": "not_checked", "errors": ["x"]})
    with pytest.raises(AuxPoolingRefused, match="spot check"):
        build(camp)


@BOTH
def test_due_but_missing_spot_check_refuses(tmp_path, build):
    camp = build_campaign(tmp_path, spot_check=None)      # epoch_001 and final ran with the model
    with pytest.raises(AuxPoolingRefused, match=r"spot check.*due"):
        build(camp)


def test_spot_check_not_due_without_a_post_admission_phase_run_with_the_model(tmp_path):
    import shutil
    camp = build_campaign(tmp_path, spot_check=None)
    shutil.rmtree(camp.ad / "epoch_001")
    shutil.rmtree(camp.ad / "final")
    assert aux_admission_allows_pooling(camp.ad)["model_sha256"] == MODEL.model_sha256


@BOTH
def test_modified_backfill_bytes_refuse(tmp_path, build):
    camp = build_campaign(tmp_path)
    p = camp.ad / "epoch_000" / BACKFILL_FILENAME
    t = pq.read_table(p)
    z = t.column("aux_z").to_numpy().copy()
    z[0] += 1e-3                                           # same rows, same model metadata, different bytes
    import pyarrow as pa
    pq.write_table(t.set_column(t.schema.get_field_index("aux_z"), "aux_z", pa.array(z)), p)
    with pytest.raises(AuxPoolingRefused, match="sha256"):
        build(camp)


@BOTH
def test_backfill_without_record_entry_refuses(tmp_path, build):
    camp = build_campaign(tmp_path, backfill_record=[])
    with pytest.raises(AuxPoolingRefused, match="no backfill entry"):
        build(camp)


@BOTH
def test_relocated_campaign_matches_backfill_by_label(tmp_path, build):
    camp = build_campaign(tmp_path)
    moved = tmp_path / "moved"
    camp.ad.rename(moved)
    camp.ad = moved
    build(camp)


def test_post_admission_phase_without_recorded_z_refuses(tmp_path):
    from gareus.adaptive.aux_pooling import phase_z
    camp = build_campaign(tmp_path)
    adm = json.loads((camp.ad / "aux_admission.json").read_text())
    with pytest.raises(AuxPoolingRefused, match="no recorded aux_z_00"):
        phase_z("final", camp.ad / "final", [0], [200_010], MODEL.model_sha256, recorded=None,
                admission=adm, adaptive_dir=camp.ad)


@BOTH
def test_post_admission_segment_without_aux_payload_refuses(tmp_path, build):
    # final ran with the admitted model (aux snapshot) but its samples carry no atlas-aux-samples-v1 payload:
    # kernel eligibility calls it aux_unpersisted; in an admitted campaign that refuses, it is never dropped
    camp = build_campaign(tmp_path, final_records_z=False)
    with pytest.raises(AuxPoolingRefused, match=r"final.*seg_001.*aux_unpersisted"):
        build(camp)


@BOTH
def test_pre_admission_aux_unpersisted_rule_untouched_for_phases_without_the_model(tmp_path, build):
    # epoch_000 (no aux kernel) is not aux_unpersisted: the default fixture pools through its backfill
    build(build_campaign(tmp_path))


def test_pre_admission_phase_still_uses_backfill(tmp_path):
    from gareus.adaptive.aux_pooling import phase_z
    camp = build_campaign(tmp_path)
    adm = json.loads((camp.ad / "aux_admission.json").read_text())
    (r, s), z = next(((k[1], k[2]), v) for k, v in camp.truth.items() if k[0] == "epoch_000")
    got = phase_z("epoch_000", camp.ad / "epoch_000", [r], [s], MODEL.model_sha256, admission=adm,
                  adaptive_dir=camp.ad)
    assert got.tolist() == [z]


def test_no_admission_record_returns_none_without_importing_pooling(tmp_path):
    camp = build_campaign(tmp_path, admission=False)
    code = ("import sys; from gareus.kernel_identity import aux_admission_allows_pooling as f; "
            f"assert f({str(camp.ad)!r}) is None; assert 'gareus.adaptive.aux_pooling' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(Path(__file__).resolve().parent.parent))


# ---- the driver's spot check records every outcome it reaches -------------------------------------------------

def _spot(camp, monkeypatch, check, phases):
    from gareus.adaptive import aux_admission_io as H
    import gareus.adaptive.aux_backfill as B
    p = camp.ad / "aux_admission.json"
    rec = json.loads(p.read_text())
    rec.pop("spot_check", None)
    p.write_text(json.dumps(rec))
    monkeypatch.setattr(B, "check_backfill_against_recorded", check)
    H._post_admission_spot_check(camp.ad, camp.ad / "epoch_001", 1, phases)
    return json.loads(p.read_text()).get("spot_check")


def test_spot_check_exception_is_recorded_as_failure(tmp_path, monkeypatch):
    camp = build_campaign(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("frames unreadable")
    sc = _spot(camp, monkeypatch, boom, [camp.ad / "epoch_001"])
    assert sc is not None and sc["ok"] is False and "frames unreadable" in json.dumps(sc)
    with pytest.raises(AuxPoolingRefused, match="spot check"):
        aux_admission_allows_pooling(camp.ad)


def test_spot_check_setup_failure_is_recorded_as_failure(tmp_path, monkeypatch):
    camp = build_campaign(tmp_path)
    (camp.ad / "aux_model.json").write_text("{not json")
    sc = _spot(camp, monkeypatch, lambda *a, **k: {"ok": True}, [camp.ad / "epoch_001"])
    assert sc is not None and sc["ok"] is False and sc.get("error")


def test_spot_check_skips_phases_run_without_the_model(tmp_path, monkeypatch):
    # re-admission: the boundary's phases ran without the model, nothing is due yet, nothing is recorded
    camp = build_campaign(tmp_path)
    called = []
    sc = _spot(camp, monkeypatch, lambda *a, **k: called.append(a) or {"ok": True}, [camp.ad / "epoch_000"])
    assert sc is None and called == []


def test_spot_check_failure_verdict_recorded(tmp_path, monkeypatch):
    camp = build_campaign(tmp_path)
    sc = _spot(camp, monkeypatch, lambda ph, *a, **k: {"phase": str(ph), "ok": False, "max_energy_err_kt": 100.0,
                                                        "tol_kt": 1.0, "n_compared": 3, "max_abs_dev": 1.0},
               [camp.ad / "epoch_001"])
    assert sc["ok"] is False and sc["max_energy_err_kt"] == 100.0
    with pytest.raises(AuxPoolingRefused, match="100"):
        aux_admission_allows_pooling(camp.ad)


def test_spot_check_error_record_is_rerun_at_the_next_boundary(tmp_path, monkeypatch):
    from gareus.adaptive import aux_admission_io as H
    import gareus.adaptive.aux_backfill as B
    camp = build_campaign(tmp_path, spot_check={"epoch": 1, "ok": False, "status": "error", "error": "RuntimeError: x"})
    with pytest.raises(AuxPoolingRefused, match="spot check"):
        aux_admission_allows_pooling(camp.ad)
    monkeypatch.setattr(B, "check_backfill_against_recorded", lambda ph, *a, **k: {
        "phase": str(ph), "ok": True, "max_energy_err_kt": 0.01, "tol_kt": 1.0, "n_compared": 3, "max_abs_dev": 0.01})
    H._post_admission_spot_check(camp.ad, camp.ad / "final", 2, [camp.ad / "final"])
    sc = json.loads((camp.ad / "aux_admission.json").read_text())["spot_check"]
    assert sc["ok"] is True and sc["epoch"] == 2 and sc["previous_errors"][0]["error"] == "RuntimeError: x"
    assert aux_admission_allows_pooling(camp.ad) is not None


def test_measured_spot_check_failure_is_final(tmp_path, monkeypatch):
    from gareus.adaptive import aux_admission_io as H
    import gareus.adaptive.aux_backfill as B
    measured = {"epoch": 1, "ok": False, "max_energy_err_kt": 100.0, "tol_kt": 1.0, "phase": "epoch_001",
                "n_compared": 10, "max_abs_dev": 1.0}
    camp = build_campaign(tmp_path, spot_check=measured)
    called = []
    monkeypatch.setattr(B, "check_backfill_against_recorded", lambda *a, **k: called.append(a) or {"ok": True})
    H._post_admission_spot_check(camp.ad, camp.ad / "final", 2, [camp.ad / "final"])
    assert called == [] and json.loads((camp.ad / "aux_admission.json").read_text())["spot_check"] == measured
    with pytest.raises(AuxPoolingRefused, match="100"):
        aux_admission_allows_pooling(camp.ad)


def test_due_but_missing_names_aux_discovery_off(tmp_path):
    camp = build_campaign(tmp_path, spot_check=None)
    (camp.ad / "decision_settings.json").write_text(json.dumps({"settings": {"aux_discovery": False}}))
    with pytest.raises(AuxPoolingRefused, match=r"spot check never ran: --ap-aux-discovery is off"):
        aux_admission_allows_pooling(camp.ad)
