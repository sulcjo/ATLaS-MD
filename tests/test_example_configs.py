"""Tests that all shipped example YAML configs parse without unknown-key errors."""

from __future__ import annotations

import pytest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"

# All YAML files in the examples/ directory must parse cleanly.
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob("*.yaml"))


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_parses_without_unknown_keys(config_path: Path) -> None:
    """Each example YAML must parse with no unknown-key warnings or errors.

    This test does NOT require OpenMM — it only imports gareus.cli which
    uses argparse and PyYAML.
    """
    from gareus.cli import parse_args

    # parse_args raises ValueError listing all unknown keys if any are found.
    # If the config provides --seq, the required arg is satisfied; if not,
    # the sequence block sets it via config defaults.
    parse_args(["--config", str(config_path)])
