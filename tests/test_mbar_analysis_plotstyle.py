import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_plotstyle_module_is_importable_from_package():
    import gareus.mbar_analysis.plotstyle as ps  # noqa: F401


def test_plotstyle_exports_match_original_top_level_module():
    import gareus.mbar_analysis.plotstyle as ps

    assert ps.OKABE_ITO == [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    ]
    assert ps.METHOD_STYLE["gamd_cumulant2"] == ("#0072B2", "-")
    assert ps.pretty_method("gamd_cumulant2") == "GaMD cumulant-2"
    assert ps.pretty_method("unknown_method") == "unknown_method"
    assert ps.method_color("umbrella_only") == "#009E73"


def test_old_top_level_module_is_gone():
    assert not Path(__file__).resolve().parent.parent.joinpath("gareus_plotstyle.py").exists()


def test_analyze_secondary_cv_pmf_no_longer_references_bare_gareus_plotstyle():
    repo_root = Path(__file__).resolve().parent.parent
    text = (repo_root / "analyze_gareus_mbar.py").read_text()
    assert "import gareus_plotstyle" not in text
