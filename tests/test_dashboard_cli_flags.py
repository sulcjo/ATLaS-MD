import pytest

from gareus import cli

# Verified against this checkout: `cli.parse_args(["--seq", "AAAA", "--out", "out"])`
# succeeds and yields tui_mode='dashboard', color='auto', dashboard_density='auto',
# distance_ascii_max_replicas=32 -- i.e. the shim's values, pre-change.
BASE = ["--seq", "AAAA", "--out", "out"]      # minimal accepted invocation


def _parse(extra):
    return cli.parse_args(BASE + extra)


@pytest.mark.parametrize("attr,default", [
    ("tui_view", "auto"),
    ("tui_glyphs", "auto"),
    ("color", "auto"),
    ("tui_clear_mode", "always"),
    ("dashboard_density", "auto"),
    ("distance_ascii_max_replicas", 32),
    ("dashboard_render_interval_sec", 0.0),
])
def test_promoted_flag_defaults_match_the_previously_hardcoded_values(attr, default):
    assert getattr(_parse([]), attr) == default


@pytest.mark.parametrize("flag,attr,value,expected", [
    ("--tui-view", "tui_view", "physics", "physics"),
    ("--tui-glyphs", "tui_glyphs", "ascii", "ascii"),
    ("--color", "color", "never", "never"),
    ("--tui-clear-mode", "tui_clear_mode", "never", "never"),
    ("--dashboard-density", "dashboard_density", "compact", "compact"),
    ("--distance-ascii-max-replicas", "distance_ascii_max_replicas", "64", 64),
    ("--dashboard-render-interval-sec", "dashboard_render_interval_sec", "2.5", 2.5),
])
def test_promoted_flag_survives_the_compat_shims(flag, attr, value, expected):
    """A flag the shim used to overwrite must reach the parsed namespace intact."""
    assert getattr(_parse([flag, value]), attr) == expected


def test_shim_no_longer_assigns_the_promoted_names():
    import inspect
    source = inspect.getsource(cli._shim_output)
    for name in ("args.color", "args.tui_clear_mode", "args.dashboard_density",
                 "args.distance_ascii_max_replicas", "args.dashboard_render_interval_sec"):
        assert name not in source, f"{name} is still clobbered by _shim_output"


def test_shim_still_assigns_the_knobs_that_stayed_internal():
    import inspect
    source = inspect.getsource(cli._shim_output)
    assert "args.distance_history_limit" in source
    assert "args.dashboard_min_panel_width" in source


def test_repeated_parse_args_does_not_drift_a_promoted_default():
    """Guards the sticky-default pattern recorded for --sambar-* in CLAUDE.md."""
    assert _parse(["--distance-ascii-max-replicas", "64"]).distance_ascii_max_replicas == 64
    assert _parse([]).distance_ascii_max_replicas == 32
