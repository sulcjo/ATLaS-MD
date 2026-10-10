import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gareus.adaptive.aux_discovery.settings import (
    AUX_CAMPAIGN_OPTIONS_FILENAME,
    AUX_SETTINGS_FILENAME,
    AuxDiscoverySettings,
    resolve_aux_campaign_options,
    resolve_aux_settings,
)
from test_aux_discovery_settings import _base, _parse


@pytest.mark.parametrize("mode", ["backbone", "sidechain", "mixed", "auto"])
def test_feature_space_cli_and_settings_roundtrip(tmp_path, mode):
    args = _parse(_base(tmp_path) + ["--ap-aux-discovery", "--ap-continue-states",
                                   "--ap-aux-feature-space", mode])
    assert args.ap_aux_feature_space == args.adaptive_production_aux_feature_space == mode
    settings = AuxDiscoverySettings(feature_space=mode)
    assert AuxDiscoverySettings.from_mapping(settings.to_mapping()) == settings


def test_feature_space_default_preserves_legacy_settings_identity(tmp_path):
    args = _parse(_base(tmp_path))
    assert args.ap_aux_feature_space == args.adaptive_production_aux_feature_space == "backbone"
    settings = AuxDiscoverySettings()
    assert settings.feature_space == "backbone"
    assert "feature_space" not in settings.to_mapping()
    digest = hashlib.sha256(json.dumps(settings.to_mapping(), sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    assert digest == "2b5f3b236836a1a58d5d06b7daa7d571ddc09c9ade189de97b5c1c41d0949006"
    assert AuxDiscoverySettings.from_mapping({}).feature_space == "backbone"
    assert resolve_aux_campaign_options(tmp_path, SimpleNamespace(ap_continue_states=True))["options"] == {
        "continue_states": True}


@pytest.mark.parametrize("mode", ["sidechain", "mixed", "auto"])
def test_nondefault_feature_space_requires_aux_discovery(tmp_path, capsys, mode):
    with pytest.raises(SystemExit):
        _parse(_base(tmp_path) + ["--ap-aux-feature-space", mode])
    assert "--ap-aux-feature-space needs --ap-aux-discovery" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["unknown", "", None, True, 3])
def test_feature_space_rejects_invalid_settings(bad):
    with pytest.raises(ValueError, match="feature_space"):
        AuxDiscoverySettings.from_mapping({"feature_space": bad})
    with pytest.raises(ValueError, match="feature_space"):
        AuxDiscoverySettings(feature_space=bad)


def test_invalid_feature_space_cli_and_yaml(tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse(_base(tmp_path) + ["--ap-aux-feature-space", "unknown"])
    assert "invalid choice" in capsys.readouterr().err
    config = tmp_path / "config.yaml"
    config.write_text("ap_aux_feature_space: unknown\n")
    with pytest.raises(SystemExit):
        _parse(_base(tmp_path) + ["--config", str(config)])
    assert "feature_space" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["sidechain", "mixed", "auto"])
def test_feature_space_yaml_and_campaign_roundtrip(tmp_path, mode):
    config = tmp_path / "config.yaml"
    config.write_text(f"ap_aux_feature_space: {mode}\n")
    args = _parse(_base(tmp_path) + ["--config", str(config), "--ap-aux-discovery", "--ap-continue-states"])
    rec = resolve_aux_campaign_options(tmp_path, args)
    assert rec["options"]["feature_space"] == mode
    before = (tmp_path / AUX_CAMPAIGN_OPTIONS_FILENAME).read_bytes()
    assert resolve_aux_campaign_options(tmp_path, args) == rec
    assert (tmp_path / AUX_CAMPAIGN_OPTIONS_FILENAME).read_bytes() == before
    with pytest.raises(RuntimeError, match="feature_space=.*frozen per campaign"):
        resolve_aux_campaign_options(tmp_path, SimpleNamespace(ap_continue_states=True))


@pytest.mark.parametrize("override", [False, True])
def test_feature_space_settings_override_cannot_change_campaign(tmp_path, override):
    resolve_aux_settings(tmp_path, AuxDiscoverySettings(feature_space="mixed"), override=False)
    before = (tmp_path / AUX_SETTINGS_FILENAME).read_bytes()
    with pytest.raises(RuntimeError, match="feature_space=.*frozen per campaign"):
        resolve_aux_settings(tmp_path, AuxDiscoverySettings(), override=override)
    assert (tmp_path / AUX_SETTINGS_FILENAME).read_bytes() == before


@pytest.mark.parametrize("record", ["options", "settings"])
def test_legacy_feature_space_is_backbone_and_resume_fails_before_md(tmp_path, record):
    from gareus.adaptive_production import (
        AdaptiveDecisionPolicy, _resolve_decision_settings, run_adaptive_production_auto_loop,
    )
    from test_aux_continue_states_required import OK

    ad = tmp_path / "adaptive_production"
    _resolve_decision_settings(ad, AdaptiveDecisionPolicy(aux_discovery=True), override=False)
    if record == "options":
        resolve_aux_campaign_options(ad, SimpleNamespace(ap_continue_states=True))
    else:
        resolve_aux_settings(ad, AuxDiscoverySettings(), override=False)
    args = SimpleNamespace(**OK, ap_aux_feature_space="sidechain")
    with pytest.raises(RuntimeError, match="feature_space=backbone.*frozen per campaign"):
        run_adaptive_production_auto_loop(args, tmp_path, None, None, None, None, None, None)
    assert not (ad / "epoch_000").exists()


@pytest.mark.parametrize("mode", ["sidechain", "mixed", "auto"])
def test_opt_in_feature_space_reaches_discovery_hook(tmp_path, monkeypatch, mode):
    from gareus.adaptive import aux_admission_io as hook
    from test_aux_admission_hook import _args, _run
    from gareus.adaptive.aux_discovery.pipeline import DiscoveryResult

    ad = tmp_path / "adaptive_production"
    (ad / "epoch_001").mkdir(parents=True)
    args = _args(tmp_path)
    args.ap_aux_feature_space = mode
    calls = []
    monkeypatch.setattr(hook, "_build_frames", lambda **kw: calls.append("frames") or object())
    monkeypatch.setattr(hook, "_discover", lambda **kw: calls.append("discovery") or DiscoveryResult("no_worker"))
    assert _run(ad, tmp_path, args=args) == []
    assert calls == ["frames", "discovery"]
    rec = json.loads((ad / AUX_SETTINGS_FILENAME).read_text())
    assert rec["settings"]["feature_space"] == mode
    report = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert report["status"] == "no_worker"


def test_mismatched_feature_space_refused_before_existing_admission(tmp_path, monkeypatch):
    from gareus.adaptive import aux_admission_io as hook
    from test_aux_admission_hook import _args, _run

    ad = tmp_path / "adaptive_production"
    (ad / "epoch_001").mkdir(parents=True)
    resolve_aux_settings(ad, AuxDiscoverySettings(), override=False)
    (ad / "aux_admission.json").write_text("{}")
    args = _args(tmp_path)
    args.ap_aux_feature_space = "auto"
    monkeypatch.setattr(hook, "_post_admission_spot_check", lambda *a: pytest.fail("admission must be refused"))
    assert _run(ad, tmp_path, args=args) == []
    report = json.loads((ad / "epoch_001" / "aux_discovery_report.json").read_text())
    assert report["status"] == "error" and "feature_space=backbone" in report["error"]


def test_default_path_does_not_import_new_kernels():
    script = """
import sys
from gareus.cli import parse_args
from gareus.adaptive.aux_discovery.settings import AuxDiscoverySettings
from gareus.adaptive import aux_admission_io
from gareus.adaptive.aux_discovery import pipeline
parse_args(['--seq', 'GYDPETGTWG', '--out', '/tmp/opencode/feature-space-parse-only'])
assert AuxDiscoverySettings().feature_space == 'backbone'
assert 'gareus.auxiliary_cv.sidechain_core' not in sys.modules
assert 'gareus.adaptive.aux_discovery.local_search_core' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_feature_space_refusal_remains_safe_when_report_write_fails(tmp_path, monkeypatch):
    from gareus.adaptive import aux_admission_io as hook
    from test_aux_admission_hook import _args, _run

    ad = tmp_path / "adaptive_production"
    resolve_aux_settings(ad, AuxDiscoverySettings(), override=False)
    args = _args(tmp_path)
    args.ap_aux_feature_space = "auto"

    def fail_write(*args):
        raise OSError("read-only report")

    monkeypatch.setattr(hook, "_write_json", fail_write)
    assert _run(ad, tmp_path, args=args) == []
