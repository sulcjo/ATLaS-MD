from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_feedback_completed_resume_preserves_adaptive_production_resume(tmp_path: Path, monkeypatch) -> None:
    from gareus import cli

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "adaptive_feedback_driver_summary.json").write_text(
        json.dumps({"handoff_to_adaptive_production": True}),
        encoding="utf-8",
    )

    handoff_csv = out_dir / "handoff.csv"
    captured = []

    def fake_handoff(args, run_dir, feedback_summary):
        handoff_csv.write_text("window,primary_cv_center,primary_cv_k_kcal\n", encoding="utf-8")
        return handoff_csv

    def fake_adaptive_production(prod_args, *args, **kwargs):
        captured.append(prod_args)
        return {"status": "mocked"}

    monkeypatch.setattr(cli, "_write_double_adaptive_factorized_handoff_csv", fake_handoff)
    monkeypatch.setattr(cli, "run_adaptive_production_auto_loop", fake_adaptive_production)

    args = SimpleNamespace(
        adaptive_production_resume=True,
        window_mode="double-adaptive",
        resume=False,
    )

    payload = cli.run_double_adaptive_auto_loop(
        args,
        out_dir,
        openmm=None,
        app=None,
        unit=None,
        forcefield=None,
        topology=None,
        equil_state=None,
        progress=None,
    )

    assert captured
    assert captured[0].window_mode == "adaptive-production"
    assert captured[0].resume is False
    assert captured[0].adaptive_production_resume is True
    assert payload["resume_mode"] == "feedback_completed_production_resume"


def _write_pilot_shared_gamd_policy(out_dir: Path, shared_dir: Path, with_globals: bool = True) -> None:
    shared_dir.mkdir(parents=True, exist_ok=True)
    if with_globals:
        (shared_dir / "shared_gamd_setup_globals.json").write_text("{}", encoding="utf-8")
    (out_dir / "adaptive_feedback_shared_gamd_policy.json").write_text(
        json.dumps({"shared_gamd_dir": str(shared_dir)}), encoding="utf-8"
    )


def test_propagate_pilot_shared_gamd_dir_sets_dir_when_calibration_exists(tmp_path: Path) -> None:
    from gareus.cli import _propagate_pilot_shared_gamd_dir

    out_dir = tmp_path / "run"
    shared_dir = out_dir / "global_shared_gamd_setup"
    _write_pilot_shared_gamd_policy(out_dir, shared_dir)

    prod_args = SimpleNamespace(shared_gamd_setup_dir="")
    _propagate_pilot_shared_gamd_dir(prod_args, out_dir)

    assert prod_args.shared_gamd_setup_dir == str(shared_dir)


def test_propagate_pilot_shared_gamd_dir_noop_without_policy_file(tmp_path: Path) -> None:
    from gareus.cli import _propagate_pilot_shared_gamd_dir

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    prod_args = SimpleNamespace(shared_gamd_setup_dir="")
    _propagate_pilot_shared_gamd_dir(prod_args, out_dir)

    assert prod_args.shared_gamd_setup_dir == ""


def test_propagate_pilot_shared_gamd_dir_noop_when_calibration_incomplete(tmp_path: Path) -> None:
    from gareus.cli import _propagate_pilot_shared_gamd_dir

    out_dir = tmp_path / "run"
    shared_dir = out_dir / "global_shared_gamd_setup"
    _write_pilot_shared_gamd_policy(out_dir, shared_dir, with_globals=False)

    prod_args = SimpleNamespace(shared_gamd_setup_dir="")
    _propagate_pilot_shared_gamd_dir(prod_args, out_dir)

    assert prod_args.shared_gamd_setup_dir == ""


def test_propagate_pilot_shared_gamd_dir_respects_explicit_override(tmp_path: Path) -> None:
    from gareus.cli import _propagate_pilot_shared_gamd_dir

    out_dir = tmp_path / "run"
    shared_dir = out_dir / "global_shared_gamd_setup"
    _write_pilot_shared_gamd_policy(out_dir, shared_dir)

    prod_args = SimpleNamespace(shared_gamd_setup_dir="/user/chosen/dir")
    _propagate_pilot_shared_gamd_dir(prod_args, out_dir)

    assert prod_args.shared_gamd_setup_dir == "/user/chosen/dir"


def test_feedback_completed_resume_propagates_pilot_shared_gamd_dir(tmp_path: Path, monkeypatch) -> None:
    from gareus import cli

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "adaptive_feedback_driver_summary.json").write_text(
        json.dumps({"handoff_to_adaptive_production": True}),
        encoding="utf-8",
    )
    shared_dir = out_dir / "global_shared_gamd_setup"
    _write_pilot_shared_gamd_policy(out_dir, shared_dir)

    handoff_csv = out_dir / "handoff.csv"
    captured = []

    def fake_handoff(args, run_dir, feedback_summary):
        handoff_csv.write_text("window,primary_cv_center,primary_cv_k_kcal\n", encoding="utf-8")
        return handoff_csv

    def fake_adaptive_production(prod_args, *args, **kwargs):
        captured.append(prod_args)
        return {"status": "mocked"}

    monkeypatch.setattr(cli, "_write_double_adaptive_factorized_handoff_csv", fake_handoff)
    monkeypatch.setattr(cli, "run_adaptive_production_auto_loop", fake_adaptive_production)

    args = SimpleNamespace(
        adaptive_production_resume=True,
        window_mode="double-adaptive",
        resume=False,
        shared_gamd_setup_dir="",
    )

    cli.run_double_adaptive_auto_loop(
        args,
        out_dir,
        openmm=None,
        app=None,
        unit=None,
        forcefield=None,
        topology=None,
        equil_state=None,
        progress=None,
    )

    assert captured
    assert captured[0].shared_gamd_setup_dir == str(shared_dir)


def test_committed_shared_gamd_dir_recovers_from_driver_summary(tmp_path: Path) -> None:
    from gareus.cli import _committed_shared_gamd_dir

    out_dir = tmp_path / "run"
    adaptive_dir = out_dir / "adaptive_production"
    adaptive_dir.mkdir(parents=True)
    committed_dir = out_dir / "global_shared_gamd_setup"
    (adaptive_dir / "adaptive_production_driver_summary.json").write_text(
        json.dumps({"global_shared_gamd_setup_dir": str(committed_dir)}), encoding="utf-8"
    )

    assert _committed_shared_gamd_dir(out_dir) == str(committed_dir)


def test_committed_shared_gamd_dir_empty_without_driver_summary(tmp_path: Path) -> None:
    from gareus.cli import _committed_shared_gamd_dir

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    assert _committed_shared_gamd_dir(out_dir) == ""


def test_production_resume_from_registry_recovers_committed_shared_gamd_dir(tmp_path: Path, monkeypatch) -> None:
    """Once adaptive-production has its own committed epoch state (registry
    exists), resuming must reuse whatever shared_gamd_setup_dir that epoch state
    was actually calibrated against -- recovered from its own driver summary --
    rather than silently defaulting to a fresh, never-calibrated dir (which
    would force a pointless recalibration on every resume).
    """
    from gareus import cli

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    adaptive_dir = out_dir / "adaptive_production"
    adaptive_dir.mkdir()
    (adaptive_dir / "state_registry.json").write_text("{}", encoding="utf-8")
    # The dir epoch 0 actually calibrated against (may or may not be the pilot's
    # dir, depending on whether _propagate_pilot_shared_gamd_dir ran for this
    # campaign) -- recorded in the driver summary that gets written every epoch.
    committed_dir = out_dir / "global_shared_gamd_setup"
    (adaptive_dir / "adaptive_production_driver_summary.json").write_text(
        json.dumps({"global_shared_gamd_setup_dir": str(committed_dir)}), encoding="utf-8"
    )

    captured = []

    def fake_adaptive_production(prod_args, *args, **kwargs):
        captured.append(prod_args)
        return {"status": "mocked"}

    monkeypatch.setattr(cli, "run_adaptive_production_auto_loop", fake_adaptive_production)

    args = SimpleNamespace(
        adaptive_production_resume=True,
        window_mode="double-adaptive",
        resume=False,
        shared_gamd_setup_dir="",
    )

    cli.run_double_adaptive_auto_loop(
        args,
        out_dir,
        openmm=None,
        app=None,
        unit=None,
        forcefield=None,
        topology=None,
        equil_state=None,
        progress=None,
    )

    assert captured
    assert captured[0].shared_gamd_setup_dir == str(committed_dir)


def test_production_resume_from_registry_respects_explicit_shared_gamd_dir(tmp_path: Path, monkeypatch) -> None:
    from gareus import cli

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    adaptive_dir = out_dir / "adaptive_production"
    adaptive_dir.mkdir()
    (adaptive_dir / "state_registry.json").write_text("{}", encoding="utf-8")
    (adaptive_dir / "adaptive_production_driver_summary.json").write_text(
        json.dumps({"global_shared_gamd_setup_dir": str(out_dir / "global_shared_gamd_setup")}), encoding="utf-8"
    )

    captured = []

    def fake_adaptive_production(prod_args, *args, **kwargs):
        captured.append(prod_args)
        return {"status": "mocked"}

    monkeypatch.setattr(cli, "run_adaptive_production_auto_loop", fake_adaptive_production)

    args = SimpleNamespace(
        adaptive_production_resume=True,
        window_mode="double-adaptive",
        resume=False,
        shared_gamd_setup_dir="/user/chosen/dir",
    )

    cli.run_double_adaptive_auto_loop(
        args,
        out_dir,
        openmm=None,
        app=None,
        unit=None,
        forcefield=None,
        topology=None,
        equil_state=None,
        progress=None,
    )

    assert captured
    assert captured[0].shared_gamd_setup_dir == "/user/chosen/dir"
