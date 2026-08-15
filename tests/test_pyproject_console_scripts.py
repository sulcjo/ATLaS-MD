import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # dev dependency already covers this via pytest's own toolchain; if unavailable, this test documents the requirement

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_gareus_analyze_console_script_is_registered():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["gareus-analyze"] == "gareus.mbar_analysis.cli:main"


def test_existing_console_scripts_are_unchanged():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    scripts = data["project"]["scripts"]
    assert scripts["gareus"] == "gareus.core:main"
    assert scripts["gareus-peptide"] == "gareus.core:main"
    assert scripts["gareus-energy-decompose"] == "gareus.energy_decomposition:main"
    assert scripts["gareus-test-run"] == "gareus.integration_test:main"
    assert scripts["gareus-suggest-cvs"] == "gareus.cv_discovery:main"
    assert scripts["gareus-consolidate-traj"] == "gareus.traj_consolidate:main"
