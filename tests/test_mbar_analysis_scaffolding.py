import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mbar_analysis_package_is_importable():
    import gareus.mbar_analysis  # noqa: F401


def test_cli_main_is_callable_with_no_args_signature():
    from gareus.mbar_analysis.cli import main
    import inspect
    sig = inspect.signature(main)
    assert list(sig.parameters) == ["argv"]
    assert sig.parameters["argv"].default is None


def test_cli_help_output_matches_analyze_gareus_mbar_help_output():
    # Run both as real subprocesses (not in-process calls) since --help
    # triggers SystemExit, which is awkward to capture reliably in-process
    # across two different entry modules in the same test process.
    new_entry = subprocess.run(
        [sys.executable, "-c", "from gareus.mbar_analysis.cli import main; main(['--help'])"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    old_entry = subprocess.run(
        [sys.executable, "analyze_gareus_mbar.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert new_entry.stdout == old_entry.stdout
    assert new_entry.returncode == old_entry.returncode
