import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_pmf_module_is_importable_standalone():
    """Must import with ZERO dependency on analyze_gareus_mbar having been
    loaded first -- proves _bridge()'s deferred-import design (no eager
    cross-import at gareus/mbar_analysis/pmf.py's own module level)."""
    import gareus.mbar_analysis.pmf  # noqa: F401


def test_bridge_prefers_already_loaded_real_module(monkeypatch):
    import gareus.mbar_analysis.pmf as pmfmod
    sentinel = object()

    class _FakeModule:
        marker = sentinel

    monkeypatch.setitem(sys.modules, 'analyze_gareus_mbar', _FakeModule())
    assert pmfmod._bridge().marker is sentinel


def test_bridge_prefers_main_module_when_it_is_the_script(monkeypatch):
    import gareus.mbar_analysis.pmf as pmfmod
    monkeypatch.delitem(sys.modules, 'analyze_gareus_mbar', raising=False)

    class _FakeMain:
        __file__ = '/some/path/analyze_gareus_mbar.py'
        marker = 'main-copy'

    monkeypatch.setitem(sys.modules, '__main__', _FakeMain())
    assert pmfmod._bridge().marker == 'main-copy'


def test_bridge_falls_back_to_fresh_import_when_neither_present(monkeypatch):
    import gareus.mbar_analysis.pmf as pmfmod
    monkeypatch.delitem(sys.modules, 'analyze_gareus_mbar', raising=False)

    class _FakeMainUnrelated:
        __file__ = '/some/other/script.py'

    monkeypatch.setitem(sys.modules, '__main__', _FakeMainUnrelated())
    result = pmfmod._bridge()
    assert result.__name__ == 'analyze_gareus_mbar'


_TASK1_NAMES = [
    'make_bins', '_bin_indices', 'pmf_from_weights',
    '_cumulant_shared_stats', '_cumulant_from_shared', '_cumulant_expansion', '_cumulant_expansion_both',
    'cumulant2', 'cumulant3',
    'pmf2d_from_weights',
    '_cumulant_shared_stats_2d', '_cumulant_from_shared_2d', '_cumulant_expansion_2d', '_cumulant_expansion_2d_both',
    '_bootstrap_pmf_uncertainty_1d', '_bootstrap_pmf_uncertainty_2d',
    'cumulant2_2d', 'cumulant3_2d',
]


def test_task1_names_are_reexported_identically_by_analyze_gareus_mbar():
    import gareus.mbar_analysis.pmf as pmfmod
    import analyze_gareus_mbar as agm
    for name in _TASK1_NAMES:
        assert hasattr(pmfmod, name), f'{name} missing from gareus.mbar_analysis.pmf'
        assert getattr(agm, name) is getattr(pmfmod, name), (
            f'analyze_gareus_mbar.{name} is not the SAME object as '
            f'gareus.mbar_analysis.pmf.{name} -- re-export import did not replace the local def'
        )
