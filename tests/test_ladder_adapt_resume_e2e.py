"""End-to-end: a crash after registry.save but before the summary write, then resume.

Drives the real run_adaptive_production_auto_loop with a fake run_gareus. The first drive
applies one action and is killed right after the registry save + applied-actions ledger;
the resumed drive must re-enter epoch 0 on the pre-action registry, must not propose or apply
again, and must reach the final phase with the action applied exactly once.
"""
import json

import pytest

import gareus.adaptive_production as ap
import gareus.production as prod
from gareus.adaptive_production import WindowStateRegistry
from gareus.cli import parse_args


class _Crash(Exception):
    pass


class _ReachedFinal(Exception):
    pass


def _campaign(tmp_path):
    out = tmp_path / "run"
    adaptive = out / "adaptive_production"
    adaptive.mkdir(parents=True)
    reg = WindowStateRegistry()
    for c in (0.1, 0.2, 0.3):
        reg.add_state(c, 800.0, epoch=0, source="seed")
    reg.save(adaptive)
    reg.write_active_window_csv(adaptive / "windows_epoch_000.csv")
    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(out)])
    args.adaptive_production_resume = True
    args.adaptive_production_allocation_scheduler = False
    args.adaptive_production_epochs = 1
    args.adaptive_production_epoch_steps = 1000
    args.adaptive_production_global_shared_gamd = False
    return args, out, adaptive


def _stub(monkeypatch, calls, crash_after_save):
    def fake_run_gareus(_args, run_dir, *rest, **kw):
        if "final" in str(run_dir):
            raise _ReachedFinal()

    def fake_propose(registry, diagnostics, **kw):
        calls["propose"] += 1
        return [("add", None, (0.15, 800.0), "test add")]

    def fake_diag(epoch_dir, registry, policy):
        calls["diag_states"].append(len(registry.active_states()))
        return {"states": [], "edges": []}

    real_pool = ap._write_runtime_pool_reports

    def pool_reports(*a, **k):
        if crash_after_save["armed"] and calls["propose"] >= 1:
            crash_after_save["armed"] = False
            raise _Crash()
        return real_pool(*a, **k)

    monkeypatch.setattr(prod, "run_gareus", fake_run_gareus)
    monkeypatch.setattr(ap, "propose_actions_from_diagnostics", fake_propose)
    monkeypatch.setattr(ap, "collect_epoch_diagnostics", fake_diag, raising=False)
    monkeypatch.setattr(ap, "collect_segmented_epoch_diagnostics", fake_diag)
    monkeypatch.setattr(ap, "_assert_epoch_has_samples", lambda *a, **k: None)
    monkeypatch.setattr(ap, "evaluate_adaptive_convergence_gate",
                        lambda *a, **k: {"status": "continue", "stop_adaptive": False, "continue_reasons": []})
    monkeypatch.setattr(ap, "_write_runtime_pool_reports", pool_reports)


def test_crash_after_save_then_resume_applies_the_action_exactly_once(tmp_path, monkeypatch):
    args, out, adaptive = _campaign(tmp_path)
    calls = {"propose": 0, "diag_states": []}
    crash = {"armed": True}
    _stub(monkeypatch, calls, crash)

    with pytest.raises(_Crash):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    ep = adaptive / "epoch_000"
    assert (ep / "actions_applied.json").exists() and (ep / "state_registry_pre_actions.json").exists()
    assert len(WindowStateRegistry.load(adaptive).active_states()) == 4
    summary = json.loads((adaptive / "adaptive_production_driver_summary.json").read_text()) \
        if (adaptive / "adaptive_production_driver_summary.json").exists() else {}
    assert int(summary.get("epochs_completed", 0) or 0) == 0          # the kill happened before this

    # A killed job leaves its lock behind with a dead PID, which the next job treats as stale;
    # in one test process the PID is still alive, so remove it as the stale-lock path would.
    (adaptive / ".gareus_run.lock").unlink(missing_ok=True)
    with pytest.raises(_ReachedFinal):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert calls["propose"] == 1                                       # never proposed again
    assert calls["diag_states"] == [3, 3]                             # both passes saw the states that ran
    reg = WindowStateRegistry.load(adaptive)
    assert len(reg.active_states()) == 4                              # applied exactly once
    assert sum(1 for s in reg.active_states() if abs(s.primary_center - 0.15) < 1e-9) == 1
    rows = (ep / "epoch_window_map.csv").read_text().strip().splitlines()
    assert len(rows) - 1 == 3                                         # epoch 0's map describes what ran


def test_a_normal_run_records_the_ledger_without_changing_behaviour(tmp_path, monkeypatch):
    args, out, adaptive = _campaign(tmp_path)
    calls = {"propose": 0, "diag_states": []}
    _stub(monkeypatch, calls, {"armed": False})
    with pytest.raises(_ReachedFinal):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert calls["propose"] == 1 and len(WindowStateRegistry.load(adaptive).active_states()) == 4
    led = json.loads((adaptive / "epoch_000" / "actions_applied.json").read_text())
    assert led["actions"][0][0] == "add"
