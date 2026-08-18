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


def test_cli_main_forwards_argv_none_to_real_sys_argv(monkeypatch, tmp_path, capsys):
    # main(argv=None) must let argparse fall through to the real process
    # sys.argv[1:], not silently swallow it. Use an invalid flag value so
    # argparse's own error path (SystemExit(2)) proves argv was actually read.
    monkeypatch.setattr(sys, "argv", ["gareus-analyze", "--bins", "NOT_AN_INT"])
    from gareus.mbar_analysis.cli import main
    import pytest
    with pytest.raises(SystemExit) as exc_info:
        main(None)
    assert exc_info.value.code == 2
    assert "invalid int value" in capsys.readouterr().err


def test_cli_main_restores_sys_argv0_after_call(monkeypatch):
    # The sys.argv[0] mutation used to make argparse's --help output match
    # the old script must not leak past the call, success or failure.
    sentinel = "sentinel-argv0-for-test"
    monkeypatch.setattr(sys, "argv", [sentinel, "--help"])
    from gareus.mbar_analysis.cli import main
    import pytest
    with pytest.raises(SystemExit):
        main(None)
    assert sys.argv[0] == sentinel


def test_cli_main_return_type_is_int():
    from gareus.mbar_analysis.cli import main
    import inspect
    sig = inspect.signature(main)
    assert sig.return_annotation in (int, "int")
