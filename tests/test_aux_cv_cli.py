import pytest

from gareus.cli import build_gareus_parser, parse_args

BASE = ["--seq", "GA", "--cv1", "contacts"]
OK = ["--aux-cv-model", "m.json", "--windows-2d-csv", "w.csv", "--run-mode", "cmd",
      "--exchange-mode", "gibbs-walk", "--aux-cv-allow-unpersisted"]


def _parse(extra, tmp_path):
    return parse_args(BASE + ["--out", str(tmp_path / "o")] + extra)


def test_off_by_default(tmp_path):
    a = _parse([], tmp_path)
    assert a.aux_cv_model is None and a.aux_cv_allow_unpersisted is False


def test_accepted_configuration(tmp_path):
    a = _parse(OK, tmp_path)
    assert a.aux_cv_model == "m.json"


def test_manual_window_mode_is_accepted(tmp_path):
    assert _parse(OK + ["--window-mode", "manual"], tmp_path).aux_cv_model == "m.json"


def test_pep_gamd_boost_is_accepted(tmp_path):
    extra = [x for x in OK if x not in ("--run-mode", "cmd")] + ["--run-mode", "gamd",
                                                                 "--gamd-boost-type", "pep-gamd-lower-dual"]
    assert _parse(extra, tmp_path).aux_cv_model == "m.json"


def test_flags_are_known_config_keys():
    dests = {a.dest for a in build_gareus_parser()._actions}
    assert {"aux_cv_model", "aux_cv_allow_unpersisted"} <= dests


@pytest.mark.parametrize("change, message", [
    (lambda o: [x for x in o if x not in ("--windows-2d-csv", "w.csv")], "windows-2d-csv"),
    (lambda o: [x for x in o if x not in ("--run-mode", "cmd")] + ["--run-mode", "gamd",
                                                                   "--gamd-boost-type", "lower-dihedral"], "group 0"),
    (lambda o: [x for x in o if x not in ("--run-mode", "cmd")] + ["--run-mode", "gamd",
                                                                   "--gamd-boost-type", "lower-dual"], "group 0"),
    (lambda o: [x for x in o if x not in ("--exchange-mode", "gibbs-walk")] + ["--exchange-mode", "neighbor"], "neighbor"),
    (lambda o: o + ["--resume"], "Stage C"),
    (lambda o: o + ["--extend"], "Stage C"),
    (lambda o: o + ["--us-auto-drop-bad-windows"], "auto-drop"),
    (lambda o: o + ["--window-mode", "adaptive-production"], "plain-run"),
    (lambda o: o + ["--window-mode", "adaptive-feedback"], "plain-run"),
    (lambda o: o + ["--window-mode", "double-adaptive"], "plain-run"),
    (lambda o: o + ["--window-mode", "delaunay-feedback"], "plain-run"),
    (lambda o: o + ["--swarm-stage", "run"], "swarm"),
    (lambda o: [x for x in o if x != "--aux-cv-allow-unpersisted"], "aux-cv-allow-unpersisted"),
])
def test_refusals(change, message, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse(change(list(OK)), tmp_path)
    assert message in capsys.readouterr().err


def test_ack_without_model_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse(["--aux-cv-allow-unpersisted"], tmp_path)
    assert "needs --aux-cv-model" in capsys.readouterr().err
