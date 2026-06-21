from pathlib import Path
from gareus.adaptive_production import _epoch_sample_sources


def _mk_run(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "samples.csv").write_text("window,primary_cv\n0,1.0\n", encoding="utf-8")


def test_pilot_dirs_are_included(tmp_path):
    adaptive = tmp_path / "adaptive_production"
    _mk_run(adaptive / "epoch_000")
    _mk_run(adaptive / "final")
    pilot = tmp_path / "feedback" / "round_000"
    _mk_run(pilot)
    labels = {lbl for lbl, _ in _epoch_sample_sources(adaptive, True, pilot_dirs=[pilot])}
    assert any("epoch_000" in l for l in labels)
    assert any("final" == l for l in labels)
    assert any("round_000" in l or "pilot" in l for l in labels)


def test_no_pilots_when_none(tmp_path):
    adaptive = tmp_path / "adaptive_production"
    _mk_run(adaptive / "final")
    labels = {lbl for lbl, _ in _epoch_sample_sources(adaptive, False)}
    assert labels == {"final"}
