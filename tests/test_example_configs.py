"""Tests that all shipped example YAML configs parse without unknown-key errors."""

from __future__ import annotations

import pytest
import yaml
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"

# All YAML files in the examples/ directory must parse cleanly.
EXAMPLE_CONFIGS = sorted(EXAMPLES_DIR.glob("*.yaml"))

# Top-level keys that are file metadata, not a workflow block.
_METADATA_KEYS = {"schema_version", "description"}


def _genpept_blocks():
    import GENPEPT
    return set(GENPEPT.GENPEPT_CONFIG_BLOCKS)


def _blocks(config_path: Path) -> set:
    raw = yaml.safe_load(config_path.read_text()) or {}
    return {str(k) for k in raw} - _METADATA_KEYS


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.name)
def test_example_config_parses_without_unknown_keys(config_path: Path) -> None:
    """Each example YAML must parse with no unknown-key warnings or errors.

    A config is checked by the parser of each workflow it configures: the ATLaS-MD
    blocks by ``gareus.cli.parse_args`` (unknown keys raise), and a GENPEPT block
    (``genpept``/``conformer_generation``/``seed_generation``) by ``GENPEPT.parse_args``
    with ``--strict-config`` (unknown keys inside the block are fatal). A GENPEPT-only
    file is not an ATLaS-MD config and is not given to the ATLaS-MD parser, which would
    stop at its required ``--seq`` (the sequence lives in the GENPEPT block there).

    This test does NOT require OpenMM.
    """
    blocks = _blocks(config_path)
    genpept = blocks & _genpept_blocks()
    atlas = blocks - _genpept_blocks()
    assert blocks, f"{config_path.name} configures no workflow"
    if atlas:
        from gareus.cli import parse_args
        parse_args(["--config", str(config_path)])
    if genpept:
        import GENPEPT
        GENPEPT.parse_args(["--config", str(config_path), "--strict-config"])


def test_a_genpept_block_with_an_unknown_key_is_rejected(tmp_path: Path) -> None:
    import GENPEPT
    bad = tmp_path / "bad.yaml"
    bad.write_text("genpept:\n  seq: GYDPETGTWG\n  out: x\n  bogus_key_xyz: 1\n")
    with pytest.raises(SystemExit):
        GENPEPT.parse_args(["--config", str(bad), "--strict-config"])
