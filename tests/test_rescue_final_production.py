"""Tests for the rescue_in_final_production config key and argparse default.

These tests verify config parsing behavior only — no OpenMM, PeptideBuilder,
or gamd-openmm imports are used.
"""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# 1. Argparse default: rescue_in_final_production must be False by default
# ---------------------------------------------------------------------------

def test_rescue_in_final_production_default_is_false():
    """parse_args produces rescue_in_final_production=False when not set."""
    from gareus.cli import parse_args

    args = parse_args(["--seq", "A"])
    assert args.rescue_in_final_production is False


# ---------------------------------------------------------------------------
# 2. YAML key rescue_in_final_production: true is accepted
# ---------------------------------------------------------------------------

def test_rescue_in_final_production_true_yaml_accepted(tmp_path: Path):
    """rescue_in_final_production: true in YAML is parsed without unknown-key errors."""
    cfg = tmp_path / "rescue_true.yaml"
    cfg.write_text(
        textwrap.dedent("""\
            seq: A
            rescue_in_final_production: true
        """),
        encoding="utf-8",
    )
    from gareus.cli import parse_args

    args = parse_args(["--config", str(cfg)])
    assert args.rescue_in_final_production is True


# ---------------------------------------------------------------------------
# 3. YAML key rescue_in_final_production: false is accepted
# ---------------------------------------------------------------------------

def test_rescue_in_final_production_false_yaml_accepted(tmp_path: Path):
    """rescue_in_final_production: false in YAML is parsed without unknown-key errors."""
    cfg = tmp_path / "rescue_false.yaml"
    cfg.write_text(
        textwrap.dedent("""\
            seq: A
            rescue_in_final_production: false
        """),
        encoding="utf-8",
    )
    from gareus.cli import parse_args

    args = parse_args(["--config", str(cfg)])
    assert args.rescue_in_final_production is False
