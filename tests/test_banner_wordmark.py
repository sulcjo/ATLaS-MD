"""The startup wordmark reads as "ATLaS-MD": solid 5x7 pixel letters, one baseline."""

from gareus import colors
from gareus.banner import WIDTH, _wordmark, banner_lines
from gareus.tui import strip_ansi

EXPECTED = [
    " ███  █████ █            ████       █   █ ████",
    "█   █   █   █           █           ██ ██ █   █",
    "█   █   █   █      ███  █           █ █ █ █   █",
    "█████   █   █         █  ███  ████  █ █ █ █   █",
    "█   █   █   █      ████     █       █   █ █   █",
    "█   █   █   █     █   █     █       █   █ █   █",
    "█   █   █   █████  ████ ████        █   █ ████",
]


def test_wordmark_is_the_pixel_font_rendering_of_atlas_md():
    assert _wordmark("ATLaS-MD") == EXPECTED


def test_wordmark_letters_share_one_baseline_and_fit_the_banner():
    rows = _wordmark("ATLaS-MD")
    assert len(rows) == 7                     # no descender row below the baseline
    assert all(len(r) <= WIDTH for r in rows)
    assert set("".join(rows)) <= {"█", " "}


def test_banner_starts_with_the_centred_wordmark(tmp_path):
    import argparse

    lines = banner_lines(argparse.Namespace(), tmp_path, mbar_version="4.6.1")
    offset = (WIDTH - max(len(r) for r in EXPECTED)) // 2
    # Centred as one block: every row keeps the same left offset, or the
    # letters shear apart (rows differ in length once trailing blanks go).
    assert lines[:7] == [" " * offset + r for r in EXPECTED]


def test_coloured_wordmark_keeps_the_same_visible_text(tmp_path, monkeypatch):
    import argparse

    plain = banner_lines(argparse.Namespace(), tmp_path, mbar_version="4.6.1")[:7]
    monkeypatch.setattr(colors, "ASCII_COLOR_ENABLED", True)
    coloured = banner_lines(argparse.Namespace(), tmp_path, mbar_version="4.6.1")[:7]
    assert coloured != plain
    assert [strip_ansi(l) for l in coloured] == plain
