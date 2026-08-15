import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import analyze_gareus_mbar as agm

_NAMES = [
    "write_2d_fes_csv", "write_2d_fes_npz", "write_cv1_cv2_2d_fes_csv",
    "write_cv1_cv2_2d_fes_npz", "write_pca_2d_fes_csv", "write_pca_2d_fes_npz",
    "_write_scalar_pmfs", "_write_generic_2d_fes", "_write_rama_2d",
    "write_rg_pmf", "write_rg_all", "write_pmf", "write_all", "write_cv2_pmf",
    "_write_csv_rows",
]


def test_writers_module_is_importable():
    import gareus.mbar_analysis.writers  # noqa: F401


def test_every_relocated_name_is_the_same_object_via_reexport():
    import gareus.mbar_analysis.writers as writers

    for name in _NAMES:
        assert hasattr(writers, name), f"writers.py missing {name}"
        assert getattr(agm, name) is getattr(writers, name), (
            f"{name}: re-export is broken"
        )


def test_writers_imports_from_plotting_not_from_agm():
    import gareus.mbar_analysis.writers as writers
    import gareus.mbar_analysis.plotting as plotting

    assert writers._smooth_pmf_1d is plotting._smooth_pmf_1d
    assert writers._plot_2d_fes_multirange is plotting._plot_2d_fes_multirange
    assert writers._want_gamd_method is plotting._want_gamd_method


def test_write_pmf_produces_expected_csv_shape():
    import gareus.mbar_analysis.writers as writers
    import numpy as np

    pmf = {
        "cv_A": np.array([1.0, 2.0]),
        "prob": np.array([0.5, 0.5]),
        "pmf": np.array([0.0, 0.1]),
        "counts": np.array([10, 20]),
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out.csv"
        writers.write_pmf(p, pmf, "umbrella_only")
        rows = list(csv.DictReader(p.open()))
    assert len(rows) == 2
    assert rows[0]["method"] == "umbrella_only"
    assert rows[1]["cv_A"] == "2.0"


def test_write_csv_rows_handles_heterogeneous_and_empty_input():
    import gareus.mbar_analysis.writers as writers

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rows.csv"
        writers._write_csv_rows(p, [{"a": 1, "b": 2}, {"a": 3, "c": 4}])
        rows = list(csv.DictReader(p.open()))
        assert [r["a"] for r in rows] == ["1", "3"]
        assert rows[0]["b"] == "2"

        p2 = Path(td) / "empty.csv"
        writers._write_csv_rows(p2, [])
        assert p2.read_text() == ""


def test_write_scalar_pmfs_resolves_plotting_dependencies():
    """Not in the brief's original test list -- added because
    ``_write_scalar_pmfs`` is the one relocated function with a real
    cross-module dependency (``_visible_pmfs``, needed in addition to the
    three names the brief specified) and its only call site sits inside a
    bare ``try/except Exception`` block. A missing/broken import there would
    silently degrade to a warning string instead of raising, so none of the
    brief's original 5 tests (which never call this function) would have
    caught it. This test calls it directly and asserts no such warning was
    swallowed.
    """
    import gareus.mbar_analysis.writers as writers
    import numpy as np

    pmfs = {
        "umbrella_only": {
            "x": np.array([1.0, 2.0]),
            "prob": np.array([0.5, 0.5]),
            "pmf": np.array([0.0, 0.1]),
            "counts": np.array([10, 20]),
        }
    }
    warnings: list = []
    with tempfile.TemporaryDirectory() as td:
        result = writers._write_scalar_pmfs(
            Path(td), "t", "label", "x", pmfs, "umbrella_only", warnings,
        )
    assert result["selected_method"] == "umbrella_only"
    assert not [w for w in warnings if "NameError" in w or "not defined" in w], warnings
