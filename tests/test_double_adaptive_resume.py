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
