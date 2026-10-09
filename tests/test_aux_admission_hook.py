import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gareus.adaptive import aux_admission_io as H
from gareus.adaptive.aux_discovery.validation import ValidationStatus
from gareus.adaptive_production import AdaptiveDecisionPolicy, AdaptiveProductionController, WindowStateRegistry


def _args(tmp):
    return SimpleNamespace(out=str(tmp), traj_interval=300, distance_output_interval=300, exchange_interval=3000,
                           timestep_fs=3.5, adaptive_production_aux_settings_override=False, temperature_k=300.0,
                           gamd_production_steps=60000)


def _registry():
    r = WindowStateRegistry()
    for c in (0.0, 0.5):
        r.add_state(c, 10.0, 0.0, 2.0, gamd_lambda=0.0)
    return r


def _run(ad, tmp, registry=None, epoch=1, args=None):
    return H.run_epoch_aux_discovery(
        adaptive_dir=ad, epoch_dir=ad / f"epoch_{epoch:03d}", epoch=epoch, registry=registry or _registry(),
        diagnostics={}, actions=[], policy=AdaptiveDecisionPolicy(aux_discovery=True, aux_reserve_slots=4),
        args=args or _args(tmp), out_dir=tmp, phase_dirs=[])


def _ok_validation(monkeypatch):
    monkeypatch.setattr(H, "check_validation_record", lambda *a, **k: ValidationStatus(True, "ok", 5.0))


def test_skips_epoch_zero_and_after_admission(tmp_path):
    ad = tmp_path / "adaptive_production"; ad.mkdir()
    pol = AdaptiveDecisionPolicy(aux_discovery=True)
    out = H.run_epoch_aux_discovery(adaptive_dir=ad, epoch_dir=ad / "epoch_000", epoch=0, registry=_registry(),
                                    diagnostics={}, actions=[("extend", 0, "x")], policy=pol, args=_args(tmp_path),
                                    out_dir=tmp_path, phase_dirs=[])
    assert out == [("extend", 0, "x")] and not (ad / "epoch_000" / "aux_discovery_report.json").exists()
    (ad / "aux_admission.json").write_text("{}")
    out = H.run_epoch_aux_discovery(adaptive_dir=ad, epoch_dir=ad / "epoch_001", epoch=1, registry=_registry(),
                                    diagnostics={}, actions=[], policy=pol, args=_args(tmp_path),
                                    out_dir=tmp_path, phase_dirs=[])
    assert out == []


def test_never_raises_and_reports_error(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    monkeypatch.setattr(H, "_build_frames", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    out = _run(ad, tmp_path)
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert out == [] and rep["status"] == "error" and "boom" in rep["error"]


def test_validation_missing_blocks_admission(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    fake = H.DiscoveryResultStub.ok_with_workers([(0, 1.2, 2.0)])
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    monkeypatch.setattr(H, "_discover", lambda **kw: fake)
    out = _run(ad, tmp_path)
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert out == [] and rep["status"] == "validation_missing" and not (ad / "aux_admission.json").exists()


def test_alignment_check_refuses_doomed_admission(tmp_path, monkeypatch):
    a = _args(tmp_path); a.traj_interval = 300; a.distance_output_interval = 300
    assert H.phase_alignment_ok(a, calib_steps=1_010_000)[0] is False
    assert H.phase_alignment_ok(a, calib_steps=0)[0] is True


def test_alignment_uses_production_distance_interval_default(tmp_path):
    a = _args(tmp_path); a.distance_output_interval = 0; a.report_interval = 2500
    ok, why = H.phase_alignment_ok(a, calib_steps=0)       # resolved: min(2500, 3000) = 2500 does not divide 3000
    assert ok is False and "distance_interval" in why


def test_calib_steps_follow_the_global_shared_setup_file(tmp_path):
    ad = tmp_path / "adaptive_production"
    (ad / "global_shared_gamd_setup").mkdir(parents=True)
    a = _args(tmp_path)
    assert H._calib_steps_for_phase(a, ad) == 0                    # nothing recorded, no GaMD step args
    a.gamd_cmd_prep_steps, a.gamd_cmd_steps, a.gamd_equil_prep_steps, a.gamd_equil_steps = 10, 20, 30, 40
    assert H._calib_steps_for_phase(a, ad) == 100                  # a phase that calibrates itself
    (ad / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json").write_text(
        json.dumps({"calibration_steps": 1_010_000}))
    assert H._calib_steps_for_phase(a, ad) == 1_010_000            # the imported record wins (production.py:4488)
    other = tmp_path / "elsewhere"; other.mkdir()
    (other / "shared_gamd_setup_globals.json").write_text(json.dumps({"calibration_steps": 7}))
    a.shared_gamd_setup_dir = str(other)
    assert H._calib_steps_for_phase(a, ad) == 7


def test_doomed_calibration_blocks_admission(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    (ad / "global_shared_gamd_setup").mkdir()
    (ad / "global_shared_gamd_setup" / "shared_gamd_setup_globals.json").write_text(
        json.dumps({"calibration_steps": 1_010_000}))
    _ok_validation(monkeypatch)
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    monkeypatch.setattr(H, "_discover", lambda **kw: H.DiscoveryResultStub.ok_with_workers([(0, 1.2, 2.0)]))
    out = _run(ad, tmp_path)
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert out == [] and rep["status"] == "alignment" and not (ad / "aux_admission.json").exists()


def test_admission_freezes_files_and_emits_actions(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    _ok_validation(monkeypatch)
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    monkeypatch.setattr(H, "_discover", lambda **kw: H.DiscoveryResultStub.ok_with_workers([(1, 1.2, 2.0)]))
    out = _run(ad, tmp_path)
    assert len(out) == 1 and out[0][0] == "admit_aux" and out[0][1] == 1
    assert out[0][2]["burnin_steps"] == 0 and out[0][4]["aux"]["burnin_phase_epoch"] == 2 and out[0][2]["aux_model_sha256"] == "e" * 64
    adm = json.loads((ad / "aux_admission.json").read_text())
    assert adm["schema"] == "atlas-aux-admission-v1" and adm["workers"][0]["parent_state_id"] == 1
    assert (ad / "aux_model.json").exists() and (ad / "aux_eval_partition.pkl").exists()


class _FailingPartition:
    def to_file(self, path):
        Path(path).write_bytes(b"half")
        Path(str(path) + ".sha256").write_text("x")
        raise OSError("disk full")


def test_failure_after_freeze_started_removes_partial_files(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    _ok_validation(monkeypatch)
    res = H.DiscoveryResultStub.ok_with_workers([(1, 1.2, 2.0)])
    res.eval_partition = _FailingPartition()
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    monkeypatch.setattr(H, "_discover", lambda **kw: res)
    out = _run(ad, tmp_path)
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert out == [] and rep["status"] == "error" and "disk full" in rep["error"]
    leftovers = [p.name for p in ad.iterdir() if p.name.startswith(("aux_model", "aux_eval_partition", "aux_admission"))]
    assert leftovers == []


def test_inject_phase_args(tmp_path):
    ad = tmp_path / "adaptive_production"; ad.mkdir()
    pa = SimpleNamespace(aux_cv_model=None, aux_phase_kind="pilot", aux_equilibrium_eligible=False,
                         windows_2d_csv=str(tmp_path / "w.csv"))
    (tmp_path / "w.csv").write_text("state_id,state_role\n0,ordinary\n")
    H.inject_aux_phase_args(pa, ad)
    assert pa.aux_cv_model is None
    (ad / "aux_admission.json").write_text(json.dumps({"schema": "atlas-aux-admission-v1"}))
    (ad / "aux_model.json").write_text("{}")
    (tmp_path / "w.csv").write_text("state_id,state_role\n0,ordinary\n1,auxiliary\n")
    H.inject_aux_phase_args(pa, ad)
    assert pa.aux_cv_model == str(ad / "aux_model.json") and pa.aux_phase_kind == "production"
    assert pa.aux_equilibrium_eligible is True


def test_refused_admission_is_unfrozen_so_next_boundary_retries(tmp_path):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    for name in ("aux_model.json", "aux_admission.json", "aux_eval_partition.pkl", "aux_eval_partition.pkl.sha256"):
        (ad / name).write_text("x")
    (ad / "epoch_001" / "aux_discovery_report.json").write_text(json.dumps({"status": "ok"}))
    actions = [("admit_aux", 1, {}, "t")]
    H.annotate_report_with_refusals(ad, ad / "epoch_001", actions, [{"action": "admit_aux", "reason": "aux_budget", "index": 0}])
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert rep["apply"]["refused"][0]["reason"] == "aux_budget" and rep["status"] == "refused_by_applier"
    assert not (ad / "aux_admission.json").exists() and not (ad / "aux_model.json").exists()



def test_calib_steps_are_zero_without_gamd(tmp_path):
    # Task 17 (CPU e2e, --run-mode cmd): production calibrates nothing (calib_steps = 0) but the hook summed the
    # gamd_* step defaults (310000), so the alignment check could refuse a cmd campaign for no reason.
    ad = tmp_path / "adaptive_production"; ad.mkdir()
    a = _args(tmp_path)
    a.gamd_cmd_prep_steps, a.gamd_cmd_steps, a.gamd_equil_prep_steps, a.gamd_equil_steps = 10, 20, 30, 40
    for mode in ("cmd", "hmr-cmd"):
        a.run_mode = mode
        assert H._calib_steps_for_phase(a, ad) == 0
    a.run_mode = "gamd"
    assert H._calib_steps_for_phase(a, ad) == 100


def test_inject_phase_args_for_a_worker_less_post_admission_segment(tmp_path):
    # Task 17 (CPU e2e): once the registry holds a worker every subset CSV carries the aux columns, also a
    # scheduled segment holding only ordinary states. It must get the model too (the loader refuses aux
    # columns without one, and its samples need recorded z for pooling: backfill covers pre-admission only).
    ad = tmp_path / "adaptive_production"; ad.mkdir()
    pa = SimpleNamespace(aux_cv_model=None, aux_phase_kind="pilot", aux_equilibrium_eligible=False,
                         windows_2d_csv=str(tmp_path / "w.csv"))
    (tmp_path / "w.csv").write_text("state_id,aux_center,aux_k_kcal_mol,state_role\n0,,0.0,ordinary\n")
    H.inject_aux_phase_args(pa, ad)                       # no admission: unchanged
    assert pa.aux_cv_model is None and pa.aux_phase_kind == "pilot" and pa.aux_equilibrium_eligible is False
    (ad / "aux_admission.json").write_text(json.dumps({"schema": "atlas-aux-admission-v1"}))
    (ad / "aux_model.json").write_text("{}")
    H.inject_aux_phase_args(pa, ad)
    assert pa.aux_cv_model == str(ad / "aux_model.json") and pa.aux_phase_kind == "production"
    assert pa.aux_equilibrium_eligible is True
    pb = SimpleNamespace(aux_cv_model=None, aux_phase_kind="pilot", aux_equilibrium_eligible=False,
                         windows_2d_csv=str(tmp_path / "plain.csv"))
    (tmp_path / "plain.csv").write_text("state_id,state_role\n0,ordinary\n")
    H.inject_aux_phase_args(pb, ad)                       # no aux columns, no worker: a plain phase
    assert pb.aux_cv_model is None


def test_admission_without_its_worker_is_re_emitted_never_rediscovered(tmp_path, monkeypatch):
    # Task 17: a job killed after the freeze but before registry.save leaves aux_admission.json with no worker
    # in the saved registry. The resumed boundary must re-emit the recorded worker (same parent, centre, k,
    # model) instead of returning early forever (pooling would then refuse the campaign).
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    _ok_validation(monkeypatch)
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    monkeypatch.setattr(H, "_discover", lambda **kw: H.DiscoveryResultStub.ok_with_workers([(1, 1.2, 2.0)]))
    first = _run(ad, tmp_path)
    assert len(first) == 1

    def _no_rediscovery(**kw):
        raise AssertionError("re-discovered")

    monkeypatch.setattr(H, "_build_frames", _no_rediscovery)
    monkeypatch.setattr(H, "_discover", _no_rediscovery)
    reg = _registry()                                         # the saved registry: worker never landed
    again = _run(ad, tmp_path, registry=reg)
    assert len(again) == 1 and again[0][0] == "admit_aux" and again[0][1] == 1
    assert again[0][2] == first[0][2]                        # centre, k, model sha, burnin_steps
    assert again[0][4]["aux"]["burnin_phase_epoch"] == 2
    rep = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert rep["status"] == "readmitted_from_record"
    ctl = AdaptiveProductionController(reg, policy=AdaptiveDecisionPolicy(aux_discovery=True, aux_reserve_slots=4))
    ctl.apply_actions(1, again)
    assert not ctl.refused_actions
    assert _run(ad, tmp_path, registry=reg) == []           # worker present: nothing to re-emit
