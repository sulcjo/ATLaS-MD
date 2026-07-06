import importlib
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def analyzer(monkeypatch):
    fake_numba = SimpleNamespace(
        njit=lambda *args, **kwargs: (lambda func: func),
        prange=range,
        set_num_threads=lambda *_args, **_kwargs: None,
        get_num_threads=lambda: 1,
    )
    monkeypatch.setitem(sys.modules, "numba", fake_numba)
    sys.modules.pop("analyze_gareus_mbar", None)
    module = importlib.import_module("analyze_gareus_mbar")
    yield module
    sys.modules.pop("analyze_gareus_mbar", None)


def test_topology_search_prefers_solute_only_companion(tmp_path, analyzer):
    prod = tmp_path / "prod"
    prod.mkdir()
    solute = prod / "solute_only.pdb"
    full = prod / "03_npt_equilibrated.pdb"
    solute.write_text("solute\n")
    full.write_text("full\n")

    assert analyzer._find_rg_topology_path(prod, SimpleNamespace(rg_topology=None)) == solute


def test_adaptive_source_topology_prefers_solute_only_companion(tmp_path, analyzer):
    run = tmp_path / "run"
    ap = run / "adaptive_production"
    source = ap / "epoch_000"
    source.mkdir(parents=True)
    (run / "03_npt_equilibrated.pdb").write_text("full\n")
    solute = source / "solute_only.pdb"
    solute.write_text("solute\n")

    assert analyzer._find_trajectory_companion_topology_path([source]) == solute
