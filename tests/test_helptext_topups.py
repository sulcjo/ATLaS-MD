from gareus.cli import build_gareus_parser
from gareus.helptext import render_encyclopedia_help, _METHOD_ENCYCLOPEDIA


def test_the_encyclopedia_has_a_topups_topic():
    text = render_encyclopedia_help(build_gareus_parser(), _METHOD_ENCYCLOPEDIA, topic="top-ups", color=False)
    assert "off by default" in text and "--ap-topup-max-fraction" in text
