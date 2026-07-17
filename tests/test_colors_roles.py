import gareus.colors as colors_mod
from gareus.colors import (
    role_text,
    severity_role,
    ROLE_TITLE,
    ROLE_SECTION,
    ROLE_GOOD,
    ROLE_WARN,
    ROLE_BAD,
    ROLE_MUTED,
    ROLE_VALUE,
)


def test_role_text_disabled_returns_plain(monkeypatch):
    monkeypatch.setattr(colors_mod, "ASCII_COLOR_ENABLED", False)
    assert role_text("hello", ROLE_TITLE) == "hello"


def test_role_text_enabled_applies_style(monkeypatch):
    monkeypatch.setattr(colors_mod, "ASCII_COLOR_ENABLED", True)
    out = role_text("hello", ROLE_TITLE)
    assert out != "hello"
    assert "hello" in out


def test_role_text_unknown_role_falls_back_to_muted(monkeypatch):
    monkeypatch.setattr(colors_mod, "ASCII_COLOR_ENABLED", True)
    out = role_text("x", "not-a-real-role")
    fallback = role_text("x", ROLE_MUTED)
    assert out == fallback


def test_severity_role_thresholds():
    assert severity_role(0.0, warn_at=0.5, bad_at=1.5) == ROLE_GOOD
    assert severity_role(0.5, warn_at=0.5, bad_at=1.5) == ROLE_GOOD
    assert severity_role(0.51, warn_at=0.5, bad_at=1.5) == ROLE_WARN
    assert severity_role(1.5, warn_at=0.5, bad_at=1.5) == ROLE_WARN
    assert severity_role(1.51, warn_at=0.5, bad_at=1.5) == ROLE_BAD
    assert severity_role(-0.51, warn_at=0.5, bad_at=1.5) == ROLE_WARN
    assert severity_role(-1.51, warn_at=0.5, bad_at=1.5) == ROLE_BAD
