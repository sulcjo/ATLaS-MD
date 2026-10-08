# tests/test_aux_parity_report.py
from test_aux_loaders import _run          # synthetic aux run with Reference/double runtime
from gareus.auxiliary_cv.parity_report import parity_report


def test_parity_report_on_exact_data(tmp_path):
    _run(tmp_path, historical=False)
    rep = parity_report(tmp_path)
    (seg, row), = rep["segments"].items()
    assert row["precision"] == "double" and row["max_abs_dz"] == 0.0 and row["ok"] is True
    assert rep["ok"] is True
