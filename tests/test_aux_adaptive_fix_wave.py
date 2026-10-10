"""Final fix wave of the adaptive aux-discovery branch: C1 (partial applier refusal stays poolable), I2 (no
admission at the last numbered epoch; admissions block convergence), I3 (spot check judged in kT), I4 (frozen
policy re-validated at driver start) and the placement / hook minors."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gareus.adaptive import aux_admission_io as H
from gareus.adaptive import aux_pooling as P
from gareus.adaptive.aux_discovery.validation import ValidationStatus
from gareus.adaptive_production import (AdaptiveDecisionPolicy, AdaptiveProductionController, WindowStateRegistry,
                                        _adaptive_production_converged, _apply_registry_actions)
from gareus.kernel_identity import AuxPoolingRefused


def _args(tmp):
    return SimpleNamespace(out=str(tmp), traj_interval=300, distance_output_interval=300, exchange_interval=3000,
                           timestep_fs=3.5, adaptive_production_aux_settings_override=False, temperature_k=300.0,
                           gamd_production_steps=60000)


def _registry(n=3):
    r = WindowStateRegistry()
    for c in np.linspace(0.0, 1.0, n):
        r.add_state(float(c), 10.0, 0.0, 2.0, gamd_lambda=0.0)
    return r


def _policy(**kw):
    return AdaptiveDecisionPolicy(aux_discovery=True, **{"aux_reserve_slots": 4, **kw})


def _stub(monkeypatch, workers):
    monkeypatch.setattr(H, "check_validation_record", lambda *a, **k: ValidationStatus(True, "ok", 5.0))
    monkeypatch.setattr(H, "_build_frames", lambda **kw: object())
    seen = {}

    def _discover(**kw):
        seen.update(kw)
        return H.DiscoveryResultStub.ok_with_workers(workers)
    monkeypatch.setattr(H, "_discover", _discover)
    return seen


def _hook(ad, tmp, registry, policy, *, actions=(), epoch=1, max_epochs=None):
    return H.run_epoch_aux_discovery(
        adaptive_dir=ad, epoch_dir=ad / f"epoch_{epoch:03d}", epoch=epoch, registry=registry, diagnostics={},
        actions=list(actions), policy=policy, args=_args(tmp), out_dir=tmp, phase_dirs=[], max_epochs=max_epochs)


def _report(ad, epoch=1):
    return json.loads((ad / f"epoch_{epoch:03d}" / "aux_discovery_report.json").read_text())


def _admission(ad):
    return json.loads((ad / "aux_admission.json").read_text())


def _workers(reg):
    return P.worker_table((s.state_id, s.metadata) for s in reg.all_states())


# --- C1 ----------------------------------------------------------------------------------------------------

def test_partial_applier_refusal_rewrites_admission_and_stays_poolable(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    _stub(monkeypatch, [(0, 1.2, 2.0), (1, -0.4, 3.0)])
    reg = _registry()
    actions = _hook(ad, tmp_path, reg, _policy())
    assert [a[1] for a in actions] == [0, 1] and len(_admission(ad)["workers"]) == 2
    # the applier then refuses the second (here: a replica cap that only fits one more state)
    refused = _apply_registry_actions(reg, actions, 1, policy=_policy(max_replicas_budget=4))
    assert [r["reason"] for r in refused] == ["max_replicas_budget"]
    H.annotate_report_with_refusals(ad, ad / "epoch_001", actions, refused, registry=reg)
    adm = _admission(ad)
    assert [(w["parent_state_id"], w["aux_center"]) for w in adm["workers"]] == [(0, 1.2)]
    assert [w["parent_state_id"] for w in adm["refused_workers"]] == [1]
    assert _report(ad)["apply"]["reconcile"]["status"] == "rewritten"
    P.require_admitted_workers(adm, _workers(reg), "test")        # poolable: record == registry
    assert (ad / "aux_model.json").exists()                         # the admission stays frozen


def test_every_worker_refused_unfreezes(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    _stub(monkeypatch, [(0, 1.2, 2.0)])
    reg = _registry()
    actions = _hook(ad, tmp_path, reg, _policy())
    refused = _apply_registry_actions(reg, actions, 1, policy=_policy(max_replicas_budget=3))
    H.annotate_report_with_refusals(ad, ad / "epoch_001", actions, refused, registry=reg)
    assert not (ad / "aux_admission.json").exists() and not (ad / "aux_model.json").exists()
    assert _report(ad)["status"] == "refused_by_applier"


def test_hook_truncates_to_free_aux_slots_and_budget(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    seen = _stub(monkeypatch, [(0, 1.2, 2.0), (1, -0.4, 3.0), (2, 0.3, 1.0)])
    out = _hook(ad, tmp_path, _registry(), _policy(aux_reserve_slots=1))
    assert [a[1] for a in out] == [0] and len(_admission(ad)["workers"]) == 1
    assert seen["settings"].max_workers == 1                        # placement's effective maximum
    assert _report(ad)["limits"]["n_slots"] == 1
    ad2 = tmp_path / "b" / "adaptive_production"; (ad2 / "epoch_001").mkdir(parents=True)
    out = _hook(ad2, tmp_path, _registry(), _policy(max_replicas_budget=5))      # 3 active + 2 free
    assert [a[1] for a in out] == [0, 1]


def test_hook_never_picks_a_parent_the_epoch_retires(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    seen = _stub(monkeypatch, [(0, 1.2, 2.0), (1, -0.4, 3.0)])
    reg = _registry()
    out = _hook(ad, tmp_path, reg, _policy(), actions=[("retire", 0, "redundant")])
    assert out[0] == ("retire", 0, "redundant") and [a[1] for a in out[1:]] == [1]
    assert 0 not in seen["eligible_parents"] and _report(ad)["placement_filtered"] == [0]
    assert len(reg.active_states()) == 3                            # the live registry was never touched


def test_no_slots_skips_discovery(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    seen = _stub(monkeypatch, [(0, 1.2, 2.0)])
    out = _hook(ad, tmp_path, _registry(), _policy(max_replicas_budget=3))
    assert out == [] and _report(ad)["status"] == "no_slots" and not seen


def test_pooling_requires_set_equality_both_directions():
    reg = _registry()
    ctl = AdaptiveProductionController(reg, policy=_policy())
    ctl.apply_actions(1, [("admit_aux", 0, {"aux_center": 1.2, "aux_k_kcal_mol": 2.0, "aux_model_sha256": "e" * 64},
                           "t", {"aux": {}}),
                          ("admit_aux", 1, {"aux_center": 0.1, "aux_k_kcal_mol": 2.0, "aux_model_sha256": "e" * 64},
                           "t", {"aux": {}})])
    w = _workers(reg)
    one = {"workers": [{"parent_state_id": 0, "aux_center": 1.2, "aux_k_kcal_mol": 2.0}]}
    with pytest.raises(AuxPoolingRefused, match="not in aux_admission.json"):
        P.require_admitted_workers(one, w, "t")
    three = {"workers": one["workers"] + [{"parent_state_id": 1, "aux_center": 0.1, "aux_k_kcal_mol": 2.0},
                                          {"parent_state_id": 2, "aux_center": 0.0, "aux_k_kcal_mol": 2.0}]}
    with pytest.raises(AuxPoolingRefused, match="no matching worker state"):
        P.require_admitted_workers(three, w, "t")
    two = {"workers": three["workers"][:2]}
    P.require_admitted_workers(two, w, "t")
    first = min(w)
    with pytest.raises(AuxPoolingRefused, match="not usable"):
        P.require_admitted_workers(two, w, "t", pooled={first: w[first]})


# --- I2 ----------------------------------------------------------------------------------------------------

def test_no_admission_at_the_last_numbered_epoch(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_002").mkdir(parents=True)
    seen = _stub(monkeypatch, [(0, 1.2, 2.0)])
    out = _hook(ad, tmp_path, _registry(), _policy(), epoch=2, max_epochs=3)
    assert out == [] and _report(ad, 2)["status"] == "last_epoch" and not seen
    assert not (ad / "aux_admission.json").exists()
    (ad / "epoch_001").mkdir()
    assert len(_hook(ad, tmp_path, _registry(), _policy(), epoch=1, max_epochs=3)) == 1


def test_missing_workers_are_not_readmitted_at_the_last_epoch(tmp_path, monkeypatch):
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True); (ad / "epoch_002").mkdir()
    _stub(monkeypatch, [(0, 1.2, 2.0)])
    assert len(_hook(ad, tmp_path, _registry(), _policy(), epoch=1, max_epochs=3)) == 1
    # killed before registry.save: the saved registry has no worker; the resumed job is at the last epoch
    out = _hook(ad, tmp_path, _registry(), _policy(), epoch=2, max_epochs=3)
    assert out == [] and not (ad / "aux_admission.json").exists()
    assert _report(ad, 2)["readmitted"][-1]["status"] == "last_epoch_not_readmitted"


def test_admission_blocks_convergence():
    a = ("admit_aux", 0, {"aux_center": 1.0, "aux_k_kcal_mol": 1.0, "aux_model_sha256": "e" * 64}, "t", {})
    pol = AdaptiveDecisionPolicy()
    assert _adaptive_production_converged([], {"edges": []}, pol) is True
    assert _adaptive_production_converged([a], {"edges": []}, pol) is False


def test_admission_is_a_convergence_gate_continue_reason(tmp_path):
    from gareus.adaptive_production import evaluate_adaptive_convergence_gate
    a = ("admit_aux", 0, {"aux_center": 1.0, "aux_k_kcal_mol": 1.0, "aux_model_sha256": "e" * 64}, "t", {})
    (tmp_path / "e").mkdir()
    gate = evaluate_adaptive_convergence_gate(tmp_path / "e", 1, _registry(1), {"edges": [], "states": []}, [a],
                                              AdaptiveDecisionPolicy(convergence_min_samples_per_state=0))
    assert gate["stop_adaptive"] is False
    assert any("aux worker admission" in r for r in gate["continue_reasons"])


# --- I3 ----------------------------------------------------------------------------------------------------

def test_worker_energy_error_in_kt():
    from gareus.adaptive.aux_backfill import worker_energy_error_kt
    RT = 0.0019872041 * 300.0
    z1 = np.array([1.0, 1.1, 5.0]); z2 = z1 + np.array([0.01, -0.02, 0.5])     # the 5.0 sample is far outside
    (w,) = worker_energy_error_kt(z1, z2, [(1.0, 2.0)], temperature_k=300.0)
    expect = 0.5 * 2.0 * abs((1.1 - 1.0) ** 2 - (1.08 - 1.0) ** 2) / RT
    assert w["n_within_2sigma"] == 2 and w["max_energy_err_kt"] == pytest.approx(expect)
    assert w["max_abs_dz_within"] == pytest.approx(0.02)


# --- I4 ----------------------------------------------------------------------------------------------------

def test_incompatibilities_shared_by_parse_and_driver():
    from gareus.adaptive.aux_discovery.settings import aux_discovery_incompatibilities
    ok = SimpleNamespace(window_mode="adaptive-production", traj_interval=300, distance_output_interval=300,
                         exchange_interval=3000, run_mode="cmd", ap_aux_reserve_slots=4, ap_continue_states=True)
    assert aux_discovery_incompatibilities(ok) == []
    assert "--ap-topups" in aux_discovery_incompatibilities(ok, topups=True)[0]
    bad = SimpleNamespace(**{**vars(ok), "exchange_mode": "neighbor", "us_auto_drop_bad_windows": True})
    msgs = aux_discovery_incompatibilities(bad)
    assert any("unrestricted exchange" in m for m in msgs) and any("auto-drop" in m for m in msgs)


def test_driver_refuses_frozen_aux_policy_with_incompatible_job_options(tmp_path):
    from gareus.adaptive_production import _resolve_decision_settings, run_adaptive_production_auto_loop
    ad = tmp_path / "adaptive_production"; ad.mkdir()
    _resolve_decision_settings(ad, AdaptiveDecisionPolicy(aux_discovery=True), override=False)  # frozen: aux on
    args = SimpleNamespace(window_mode="adaptive-production", traj_interval=300, distance_output_interval=300,
                           exchange_interval=3000, run_mode="cmd", adaptive_production_topups=True,
                           ap_continue_states=True)
    with pytest.raises(RuntimeError, match="frozen decision settings enable aux-CV discovery.*--ap-topups"):
        run_adaptive_production_auto_loop(args, tmp_path, None, None, None, None, None, None)


# --- minors ------------------------------------------------------------------------------------------------

def test_placement_skips_zero_variance_state():
    from gareus.adaptive.aux_discovery.placement import place_workers
    from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings
    rng = np.random.default_rng(0)
    sid = np.repeat([0, 1], 800)
    z = np.r_[np.full(800, 0.7), rng.normal(size=800)]
    lab = (rng.normal(size=1600) > 0).astype(int)
    lineage = np.array([f"e0:{i % 30}" for i in range(1600)])
    step = np.tile(np.arange(800) * 3000, 2)
    tr = np.tile(np.arange(800) < 600, 2)
    out = place_workers(z, lab, sid, lineage, step, tr, ~tr, AuxDiscoverySettings(), k_labels=2)
    assert {"state_id": 0, "n_train_frames": 600, "reason": "zero_variance"} in out["skipped_states"]
    assert all(c["state_id"] == 1 for c in out["all_candidates"])


def test_placement_respects_eligible_parents_and_zero_max():
    import dataclasses
    from gareus.adaptive.aux_discovery.placement import place_workers
    from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings
    rng = np.random.default_rng(0)
    n_states, per = 6, 600
    sid = np.repeat(np.arange(n_states), per)
    lineage = np.array([f"e0:{i % 30}" for i in range(n_states * per)])
    step = np.tile(np.arange(per) * 3000, n_states)
    z = rng.normal(size=sid.size)
    lab = (z + 0.5 * rng.normal(size=sid.size) > 0).astype(int)
    tr = np.tile(np.arange(per) < 450, n_states)
    s = AuxDiscoverySettings()
    full = place_workers(z, lab, sid, lineage, step, tr, ~tr, s, k_labels=2)
    if not full["chosen"]:
        pytest.skip("synthetic set chose no worker")
    banned = int(full["chosen"][0]["state_id"])
    out = place_workers(z, lab, sid, lineage, step, tr, ~tr, s, k_labels=2,
                        eligible_parents=[x for x in range(n_states) if x != banned])
    assert banned not in {c["state_id"] for c in out["chosen"]}
    assert any(e.get("reason") == "parent_not_eligible" for e in out["selection_log"])
    none = place_workers(z, lab, sid, lineage, step, tr, ~tr, dataclasses.replace(s, max_workers=0), k_labels=2)
    assert none["chosen"] == []


def test_physical_system_check_not_checked_without_a_system(tmp_path):
    rec = H.physical_system_check(_args(tmp_path), tmp_path)
    assert rec["status"] == "not_checked"


def test_cmap_system_blocks_admission_before_any_frame_is_read(tmp_path, monkeypatch):
    import shutil
    import openmm
    import gareus.system_setup as SS
    from aux_discovery_fixture import PDB
    shutil.copy(PDB, tmp_path / "01_solvated_start.pdb")

    def _cmap_system(app, unit, ff, topology, args, include_barostat, barostat_frequency=None):
        system = openmm.System()
        for _ in range(topology.getNumAtoms()):
            system.addParticle(12.0)
        system.addForce(openmm.CMAPTorsionForce())
        return system
    monkeypatch.setattr(SS, "create_system", _cmap_system)
    monkeypatch.setattr(SS, "make_forcefield_from_args", lambda app, args: None)
    ad = tmp_path / "adaptive_production"; (ad / "epoch_001").mkdir(parents=True)
    seen = _stub(monkeypatch, [(0, 1.2, 2.0)])
    monkeypatch.setattr(H, "_build_frames", lambda **kw: (_ for _ in ()).throw(AssertionError("frames read")))
    out = _hook(ad, tmp_path, _registry(), _policy())
    rep = _report(ad)
    assert out == [] and rep["status"] == "physical_system_unsupported" and not seen
    assert rep["physical_system"]["status"] == "refused" and "CMAP" in rep["physical_system"]["detail"]


def test_cv2_resolution_budget_holds_back_the_aux_slice_only_with_the_flag():
    # Task 4 minor: R1/R3 never spend the aux slice of the P1 reserve; flag off = byte-identical allowance
    from gareus.adaptive import cv2_resolution as cr
    from gareus.adaptive.cv2_resolution_io import budget_for_epoch
    reserve = {"fraction": 0.2, "max_replicas": 13}
    reg = _registry()
    off, _, _, _ = budget_for_epoch(reg, AdaptiveDecisionPolicy(max_replicas_budget=13, aux_reserve_slots=4), [],
                                    reserve, cr.ResolutionSettings())
    on, _, _, _ = budget_for_epoch(reg, _policy(max_replicas_budget=13), [], reserve, cr.ResolutionSettings())
    assert (off.free_slots, off.aux_slots) == (10, 0)
    assert (on.free_slots, on.aux_slots) == (6, 4) and on.resolution_slots < off.resolution_slots
    AdaptiveProductionController(reg, policy=_policy()).apply_actions(
        1, [("admit_aux", 0, {"aux_center": 1.0, "aux_k_kcal_mol": 2.0, "aux_model_sha256": "e" * 64}, "t", {})])
    held, _, _, _ = budget_for_epoch(reg, _policy(max_replicas_budget=13), [], reserve, cr.ResolutionSettings())
    assert held.aux_slots == 3 and held.free_slots == 6                  # 9 free after the worker, 3 still held
