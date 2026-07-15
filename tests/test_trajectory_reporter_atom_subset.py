import sys
import types
from types import SimpleNamespace

import pytest

from gareus.system_setup import make_trajectory_reporter


class _FakeOpenmmDCDReporter:
    """Mimics openmm.app.DCDReporter: no atomSubset kwarg."""

    def __init__(self, file, reportInterval, append=False, enforcePeriodicBox=None):
        self.file = file
        self.reportInterval = reportInterval


class _FakeOpenmmXTCReporter:
    """Mimics openmm.app.XTCReporter: no atomSubset kwarg."""

    def __init__(self, file, reportInterval, append=False, enforcePeriodicBox=None):
        self.file = file
        self.reportInterval = reportInterval


class _FakeMdtrajDCDReporter:
    def __init__(self, file, reportInterval, atomSubset=None):
        self.file = file
        self.reportInterval = reportInterval
        self.atomSubset = atomSubset


class _FakeMdtrajXTCReporter:
    def __init__(self, file, reportInterval, atomSubset=None):
        self.file = file
        self.reportInterval = reportInterval
        self.atomSubset = atomSubset


@pytest.fixture
def fake_app():
    return SimpleNamespace(DCDReporter=_FakeOpenmmDCDReporter, XTCReporter=_FakeOpenmmXTCReporter)


@pytest.fixture(autouse=True)
def fake_mdtraj_reporters(monkeypatch):
    fake_module = types.ModuleType("mdtraj.reporters")
    fake_module.DCDReporter = _FakeMdtrajDCDReporter
    fake_module.XTCReporter = _FakeMdtrajXTCReporter
    fake_mdtraj = types.ModuleType("mdtraj")
    fake_mdtraj.reporters = fake_module
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)
    monkeypatch.setitem(sys.modules, "mdtraj.reporters", fake_module)
    yield


def test_no_subset_uses_openmm_reporter_without_atom_subset_kwarg(tmp_path, fake_app):
    args = SimpleNamespace(traj_format="dcd")
    reporter = make_trajectory_reporter(fake_app, tmp_path / "replica_000", 10, args, atom_subset=None)
    assert isinstance(reporter, _FakeOpenmmDCDReporter)


def test_subset_dispatches_to_mdtraj_dcd_reporter(tmp_path, fake_app):
    args = SimpleNamespace(traj_format="dcd")
    reporter = make_trajectory_reporter(fake_app, tmp_path / "replica_000", 10, args, atom_subset=[0, 1, 2])
    assert isinstance(reporter, _FakeMdtrajDCDReporter)
    assert reporter.atomSubset == [0, 1, 2]


def test_subset_dispatches_to_mdtraj_xtc_reporter(tmp_path, fake_app):
    args = SimpleNamespace(traj_format="xtc")
    reporter = make_trajectory_reporter(fake_app, tmp_path / "replica_000", 10, args, atom_subset=[3, 4])
    assert isinstance(reporter, _FakeMdtrajXTCReporter)
    assert reporter.atomSubset == [3, 4]


def test_subset_with_unsupported_format_raises(tmp_path, fake_app):
    args = SimpleNamespace(traj_format="bogus")
    with pytest.raises(ValueError, match="Unsupported --traj-format"):
        make_trajectory_reporter(fake_app, tmp_path / "replica_000", 10, args, atom_subset=[0])


def test_subset_missing_mdtraj_raises_runtime_error(tmp_path, fake_app, monkeypatch):
    monkeypatch.delitem(sys.modules, "mdtraj", raising=False)
    monkeypatch.delitem(sys.modules, "mdtraj.reporters", raising=False)
    import builtins

    real_import = builtins.__import__

    def _blocked_import(name, *a, **kw):
        if name.startswith("mdtraj"):
            raise ImportError("no mdtraj")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    args = SimpleNamespace(traj_format="dcd")
    with pytest.raises(RuntimeError, match="mdtraj"):
        make_trajectory_reporter(fake_app, tmp_path / "replica_000", 10, args, atom_subset=[0])
