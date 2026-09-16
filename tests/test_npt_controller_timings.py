"""Per-phase timers on the volume move, and their checkpoint compatibility.

Timers are per-process diagnostics: written into the checkpoint for the record
but never restored from it, so each job reports its own rates.
"""
import json

import pytest

openmm = pytest.importorskip("openmm")

from gareus.npt import _CONTROLLER_SCHEMA_VERSION  # noqa: E402
from test_npt_controller_mechanics import (  # noqa: E402
    VolumeStubAdapter,
    _make_controller,
    _rigid_molecule_system,
)


def test_schema_version_is_unchanged_by_this_work():
    """Bumping it would break resume for the live chain at npt.py restore()."""
    assert _CONTROLLER_SCHEMA_VERSION == 1


def test_timings_start_at_zero_with_the_expected_keys():
    _ctx, ctrl = _make_controller(
        _rigid_molecule_system(), VolumeStubAdapter(), seed=11)
    timings = ctrl.state_dict()["timings"]
    assert set(timings) == {
        "read_s", "scale_s", "restore_s", "evaluate_s", "verify_s", "attempts",
    }
    assert all(v == 0 for v in timings.values())


def test_state_dict_carries_timings_and_stays_json_serialisable():
    _ctx, ctrl = _make_controller(
        _rigid_molecule_system(), VolumeStubAdapter(), seed=11)
    ctrl.attempt_due(ctrl.next_due_step)

    sd = json.loads(json.dumps(ctrl.state_dict()))
    assert sd["schema_version"] == 1
    assert sd["timings"]["attempts"] == 1
    assert sd["timings"]["scale_s"] >= 0.0
    assert sd["timings"]["read_s"] > 0.0


def test_every_phase_accumulates_over_several_moves():
    """A timer that never fires is worse than none: it reads as a free phase."""
    _ctx, ctrl = _make_controller(
        _rigid_molecule_system(n_mol=12),
        VolumeStubAdapter(physical=lambda v: 5.0 * (v - 8.5) ** 2), seed=4242)
    step = ctrl.next_due_step
    for _ in range(30):
        ctrl.attempt_due(step)
        step = ctrl.next_due_step

    t = ctrl.state_dict()["timings"]
    assert t["attempts"] == 30
    for key in ("read_s", "scale_s", "restore_s", "evaluate_s"):
        assert t[key] > 0.0, f"{key} never accumulated"
    # verify_s is strided (_RESTORE_VERIFY_STRIDE), so it may legitimately be
    # zero over 30 moves; assert only that it is present and non-negative.
    assert t["verify_s"] >= 0.0


def test_restore_accepts_a_checkpoint_written_before_timings_existed():
    """The live chain's in-flight checkpoints have no timings key."""
    from gareus.npt import BiasedMCBarostatController

    ctx, ctrl = _make_controller(
        _rigid_molecule_system(), VolumeStubAdapter(), seed=11)
    ctrl.attempt_due(ctrl.next_due_step)

    sd = json.loads(json.dumps(ctrl.state_dict()))
    sd.pop("timings")  # exactly what an a03f3f7-era checkpoint looks like

    restored = BiasedMCBarostatController.restore(
        ctx, VolumeStubAdapter(), state=sd,
        expected_pressure_bar=0.0, expected_temperature_k=300.0)
    assert restored.next_due_step == ctrl.next_due_step
    assert restored.state_dict()["timings"]["attempts"] == 0


def test_schema_v1_readers_ignore_unknown_optional_keys():
    """Adding a key without a version bump is only legitimate if v1 says this.

    Otherwise the timings addition is an undocumented schema fork. This test is
    the documentation: v1 readers must ignore what they do not recognise, so
    future optional additions have a stated rule to follow.
    """
    from gareus.npt import BiasedMCBarostatController

    ctx, ctrl = _make_controller(
        _rigid_molecule_system(), VolumeStubAdapter(), seed=11)
    sd = json.loads(json.dumps(ctrl.state_dict()))
    sd["a_key_from_a_future_version"] = {"nested": [1, 2, 3]}

    restored = BiasedMCBarostatController.restore(
        ctx, VolumeStubAdapter(), state=sd,
        expected_pressure_bar=0.0, expected_temperature_k=300.0)
    assert restored.next_due_step == ctrl.next_due_step
