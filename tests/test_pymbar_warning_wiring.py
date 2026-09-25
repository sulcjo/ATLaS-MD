"""Test pymbar warning wiring in cli.py — verify once-per-process guard."""
import gareus.cli as cli
import gareus.pymbar_check as pc


def test_warn_pymbar_once_prints_only_once_when_invoked_multiple_times(monkeypatch, capsys):
    """Verify that _warn_pymbar_once() prints WARNING exactly once when called twice.

    Tests the module-level cli._warned_pymbar_in_cli guard flag ensures the
    warning prints at most once per process, even with multiple invocations.
    """
    # Clear the pymbar cache and monkeypatch it to return failure
    pc.pymbar_status.cache_clear()
    monkeypatch.setattr(pc, "pymbar_status", lambda: (False, "AttributeError: scipy.linalg.tril"))

    # Reset the cli-level guard to ensure a clean test
    monkeypatch.setattr(cli, "_warned_pymbar_in_cli", False)

    # First call to _warn_pymbar_once() should print
    cli._warn_pymbar_once()
    captured1 = capsys.readouterr()
    assert "WARNING [pymbar]" in captured1.out
    assert "AttributeError" in captured1.out
    assert "equilibration subsampling" in captured1.out
    warning_count_after_first = captured1.out.count("WARNING [pymbar]")
    assert warning_count_after_first == 1

    # Second call to _warn_pymbar_once() should NOT print (guard flag prevents it)
    cli._warn_pymbar_once()
    captured2 = capsys.readouterr()
    warning_count_after_second = captured2.out.count("WARNING [pymbar]")
    assert warning_count_after_second == 0, "Second call should not print due to guard flag"
