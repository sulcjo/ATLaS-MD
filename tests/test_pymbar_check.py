import builtins

import gareus.pymbar_check as pc


def test_reports_available_when_import_works():
    pc.pymbar_status.cache_clear()
    ok, msg = pc.pymbar_status()
    assert ok and msg.startswith("pymbar")


def test_reports_any_import_time_exception_not_just_importerror(monkeypatch, capsys):
    real_import = builtins.__import__

    def broken(name, *a, **k):
        if name == "pymbar":
            raise AttributeError("module 'scipy.linalg' has no attribute 'tril'")
        return real_import(name, *a, **k)

    pc.pymbar_status.cache_clear()
    monkeypatch.setattr(builtins, "__import__", broken)
    ok, msg = pc.pymbar_status()
    assert not ok and "AttributeError" in msg and "tril" in msg
    assert pc.warn_if_pymbar_unusable("top-up diagnostics") is False
    assert "WARNING [pymbar]" in capsys.readouterr().out
    pc.pymbar_status.cache_clear()


def test_warn_if_pymbar_unusable_prints_when_pymbar_fails(monkeypatch, capsys):
    """Verify that warn_if_pymbar_unusable prints when pymbar is unavailable."""
    pc.pymbar_status.cache_clear()
    # Monkeypatch to return failure
    monkeypatch.setattr(pc, "pymbar_status", lambda: (False, "AttributeError: scipy.linalg.tril"))

    # Call should print and return False
    result = pc.warn_if_pymbar_unusable("equilibration subsampling")
    captured = capsys.readouterr()
    assert result is False
    assert "WARNING [pymbar]" in captured.out
    assert "AttributeError" in captured.out
    assert "equilibration subsampling" in captured.out
