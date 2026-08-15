import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_gareus_mbar as agm

_NAMES = ["_render_health_section_md", "_key_diagnostics_md", "summary_md"]


def test_summary_module_is_importable():
    import gareus.mbar_analysis.summary  # noqa: F401


def test_every_relocated_name_is_the_same_object_via_reexport():
    import gareus.mbar_analysis.summary as summary

    for name in _NAMES:
        assert hasattr(summary, name), f"summary.py missing {name}"
        assert getattr(agm, name) is getattr(summary, name), f"{name}: re-export is broken"


def test_summary_module_has_no_cross_imports_from_sibling_a6a_modules():
    import ast

    src = Path("gareus/mbar_analysis/summary.py").read_text()
    tree = ast.parse(src)
    imported_modules = {
        n.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for n in [node]
    }
    assert "gareus.mbar_analysis.plotting" not in imported_modules
    assert "gareus.mbar_analysis.writers" not in imported_modules
    assert "gareus.mbar_analysis.convergence_reporting" not in imported_modules
