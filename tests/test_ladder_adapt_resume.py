"""Applied-actions ledger: an epoch's registry changes are resume-idempotent.

A job killed after ``registry.save`` but before the driver summary advances
``epochs_completed`` used to re-enter the epoch with states its window map never ran (and,
for a retiring action such as ``respace_ladder``, would propose again on the new ladder).
"""
import json
import os

from gareus.adaptive_production import (AdaptiveProductionController, WindowStateRegistry,
                                        _load_applied_actions, _pre_action_registry_path,
                                        _record_applied_actions, _snapshot_pre_action_registry)


def _registry():
    reg = WindowStateRegistry()
    for c in (0.1, 0.2):
        for lam in (0.0, 0.5, 1.0):
            reg.add_state(primary_center=c, primary_k=800.0, gamd_lambda=lam, epoch=0, source="initial", reason="s")
    return reg


def _apply_and_record(tmp_path, reg, epoch_dir, actions):
    _snapshot_pre_action_registry(epoch_dir, reg)
    AdaptiveProductionController(reg).apply_actions(0, actions)
    reg.save(tmp_path)
    _record_applied_actions(epoch_dir, 0, actions, tmp_path / "state_registry.json")


def test_recorded_ledger_is_found_while_the_registry_is_unchanged(tmp_path):
    reg, ep = _registry(), tmp_path / "epoch_000"
    ep.mkdir()
    actions = [("respace_ladder", (0.5,), (0.3,), "t")]
    _apply_and_record(tmp_path, reg, ep, actions)
    led = _load_applied_actions(ep, tmp_path / "state_registry.json")
    assert led is not None and led["epoch"] == 0
    assert led["actions"][0][0] == "respace_ladder"


def test_ledger_is_ignored_when_the_registry_changed_after_it(tmp_path, capsys):
    reg, ep = _registry(), tmp_path / "epoch_000"
    ep.mkdir()
    _apply_and_record(tmp_path, reg, ep, [("respace_ladder", (0.5,), (0.3,), "t")])
    reg.add_state(primary_center=0.3, primary_k=800.0, gamd_lambda=0.0, epoch=1, source="x", reason="x")
    reg.save(tmp_path)
    assert _load_applied_actions(ep, tmp_path / "state_registry.json") is None
    assert "digest" in capsys.readouterr().out


def test_no_ledger_means_no_recovery(tmp_path):
    ep = tmp_path / "epoch_000"
    ep.mkdir()
    _registry().save(tmp_path)
    assert _load_applied_actions(ep, tmp_path / "state_registry.json") is None


def test_pre_action_snapshot_holds_the_registry_before_the_actions(tmp_path):
    reg, ep = _registry(), tmp_path / "epoch_000"
    ep.mkdir()
    _apply_and_record(tmp_path, reg, ep, [("respace_ladder", (0.5,), (0.3,), "t")])
    pre = WindowStateRegistry.load_json(_pre_action_registry_path(ep))
    assert pre.rung_lambdas() == [0.0, 0.5, 1.0]
    assert reg.rung_lambdas() == [0.0, 0.3, 1.0]


def test_a_crash_mid_write_leaves_the_previous_ledger_intact(tmp_path, monkeypatch):
    reg, ep = _registry(), tmp_path / "epoch_000"
    ep.mkdir()
    _apply_and_record(tmp_path, reg, ep, [("respace_ladder", (0.5,), (0.3,), "first")])
    before = (ep / "actions_applied.json").read_text()

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(os, "replace", boom)
    try:
        _record_applied_actions(ep, 0, [("respace_ladder", (), (0.7,), "second")], tmp_path / "state_registry.json")
    except OSError:
        pass
    assert (ep / "actions_applied.json").read_text() == before
    json.loads(before)
