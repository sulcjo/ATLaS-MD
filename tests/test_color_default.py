"""`--color auto` turns colour on for the dashboard even when stdout is a log file."""

import io

import pytest

from gareus import colors


@pytest.fixture(autouse=True)
def _restore_colour_flag(monkeypatch):
    monkeypatch.setattr(colors, "ASCII_COLOR_ENABLED", False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("sys.stdout", io.StringIO())    # not a tty, like a SLURM log


def test_auto_enables_colour_for_the_dashboard_on_a_non_tty():
    colors.configure_color("auto", tui_mode="dashboard")
    assert colors.ASCII_COLOR_ENABLED is True


@pytest.mark.parametrize("tui_mode", ["line", "none", "interactive", None])
def test_auto_keeps_colour_off_on_a_non_tty_without_the_dashboard(tui_mode):
    colors.configure_color("auto", tui_mode=tui_mode)
    assert colors.ASCII_COLOR_ENABLED is False


def test_never_wins_over_the_dashboard_default():
    colors.configure_color("never", tui_mode="dashboard")
    assert colors.ASCII_COLOR_ENABLED is False


def test_no_color_env_disables_the_auto_default(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    colors.configure_color("auto", tui_mode="dashboard")
    assert colors.ASCII_COLOR_ENABLED is False


def test_always_ignores_no_color_because_it_was_asked_for_explicitly(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    colors.configure_color("always")
    assert colors.ASCII_COLOR_ENABLED is True
