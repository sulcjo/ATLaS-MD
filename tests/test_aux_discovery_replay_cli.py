import json

import pytest


def test_replay_writes_only_into_out(tmp_path, monkeypatch):
    from gareus.adaptive.aux_discovery import __main__ as cli
    called = {}
    monkeypatch.setattr(cli, "_replay", lambda **kw: called.update(kw) or {"status": "insufficient_evidence"})
    run = tmp_path / "run"; (run / "adaptive_production").mkdir(parents=True)
    out = tmp_path / "out"
    assert cli.main(["replay", str(run), "--epoch", "1", "--out", str(out)]) == 0
    assert called["epoch"] == 1
    assert json.loads((out / "aux_discovery_report.json").read_text())["status"] == "insufficient_evidence"
    assert "wall_s" in json.loads((out / "timing.json").read_text())
    assert sorted(p.name for p in (run / "adaptive_production").iterdir()) == []


def test_out_inside_run_dir_is_refused(tmp_path, monkeypatch):
    from gareus.adaptive.aux_discovery import __main__ as cli
    monkeypatch.setattr(cli, "_replay", lambda **kw: {"status": "x"})
    run = tmp_path / "run"; run.mkdir()
    with pytest.raises(SystemExit):
        cli.main(["replay", str(run), "--epoch", "1", "--out", str(run / "sub")])
    assert not (run / "sub").exists()


def test_exchange_interval_read_from_manifest(tmp_path):
    from gareus.adaptive.aux_discovery.__main__ import _recorded_exchange_interval
    (tmp_path / "run_manifest.json").write_text(json.dumps({"resolved_args": {"exchange_interval": 2500}}))
    assert _recorded_exchange_interval(tmp_path) == 2500
    assert _recorded_exchange_interval(tmp_path / "none") is None
