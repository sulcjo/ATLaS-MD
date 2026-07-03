"""config_profiles is an optional top-level module.

Simulates the deploy where config_profiles.py is absent (e.g. a stale remote
checkout) and asserts the workflow still resolves configs, only failing when a
`profile:` bundle is actually requested.
"""
import importlib
import sys

import pytest


def _reload_config_without_profiles(monkeypatch):
    # A `None` entry in sys.modules makes `import config_profiles` raise ImportError,
    # exactly as when the file is missing from the deploy.
    monkeypatch.setitem(sys.modules, "config_profiles", None)
    import gareus.config as config
    importlib.reload(config)
    assert config.config_profiles is None
    return config


@pytest.fixture
def config_no_profiles(monkeypatch):
    config = _reload_config_without_profiles(monkeypatch)
    yield config
    # Restore the real module for any later tests in the session.
    monkeypatch.undo()
    importlib.reload(config)


def _profile_less_yaml(tmp_path):
    p = tmp_path / "run.yaml"
    p.write_text("cvs:\n  cv1: contacts\n  cv2: none\noutput:\n  seq: AAA\n")
    return p


def test_loads_yaml_without_profiles_module(config_no_profiles, tmp_path):
    # Arrange
    cfg = _profile_less_yaml(tmp_path)
    # Act — dup-key lint is guarded, so this must not raise
    data = config_no_profiles._load_config_file(cfg)
    # Assert
    assert data["cvs"]["cv1"] == "contacts"


def test_resolve_without_profiles_module(config_no_profiles, tmp_path):
    from gareus.cli import build_gareus_parser
    parser = build_gareus_parser()
    res = config_no_profiles._apply_config_defaults_to_parser(parser, str(_profile_less_yaml(tmp_path)), None)
    assert res["unknown_keys"] == []


def test_profile_request_without_module_raises_clearly(config_no_profiles):
    from gareus.cli import build_gareus_parser
    parser = build_gareus_parser()
    with pytest.raises(ValueError, match="config_profiles"):
        config_no_profiles._apply_config_defaults_to_parser(parser, None, cli_profile="chignolin_fast")
