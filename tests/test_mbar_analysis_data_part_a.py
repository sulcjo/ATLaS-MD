import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gareus.mbar_analysis.data import rjson, wjson, _json_default, read_windows, jvec, Data


def test_rjson_missing_file_returns_default():
    assert rjson(Path("/nonexistent/path/does/not/exist.json")) == {}
    assert rjson(Path("/nonexistent/path/does/not/exist.json"), default=[]) == []


def test_rjson_reads_real_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a": 1}')
    assert rjson(p) == {"a": 1}


def test_wjson_writes_and_rjson_reads_back(tmp_path):
    p = tmp_path / "sub" / "x.json"
    wjson(p, {"a": np.float64(1.5), "b": np.int64(3)})
    assert rjson(p) == {"a": 1.5, "b": 3}


def test_json_default_handles_numpy_types():
    assert _json_default(np.array([1, 2])) == [1, 2]
    assert _json_default(np.int64(5)) == 5
    assert _json_default(np.float64(float("nan"))) is None
    assert _json_default(np.bool_(True)) is True
    assert _json_default(Path("/a/b")) == "/a/b"


def test_read_windows_parses_center_and_k(tmp_path):
    p = tmp_path / "umbrella_windows.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["center_A", "k_kcal_mol_A2"])
        w.writeheader()
        w.writerow({"center_A": "1.0", "k_kcal_mol_A2": "10.0"})
        w.writerow({"center_A": "2.0", "k_kcal_mol_A2": "10.0"})
    centers, ks, rows = read_windows(p)
    assert list(centers) == [1.0, 2.0]
    assert list(ks) == [10.0, 10.0]
    assert len(rows) == 2


def test_read_windows_missing_file_returns_empty():
    centers, ks, rows = read_windows(Path("/nonexistent/umbrella_windows.csv"))
    assert centers.size == 0 and ks.size == 0 and rows == []


def test_jvec_parses_json_list_and_dict():
    assert jvec("[1.0, 2.0]") == [1.0, 2.0]
    assert jvec(json.dumps({"1": 2.0, "0": 1.0})) == [1.0, 2.0]
    assert jvec(None) == []
    assert jvec("") == []


def test_data_dataclass_holds_all_fields():
    d = Data(
        prod_dir=Path("/p"), out_dir=Path("/o"),
        cv=np.array([1.0]), cv2=np.array([np.nan]), rg_A=np.array([np.nan]),
        window=np.array([0]), replica=np.array([0]), step=np.array([0]),
        u_nk=np.array([[0.0]]), centers=np.array([1.0]), k_kcal=np.array([10.0]),
        beta=1.0, temp=300.0, boost_kj=np.array([np.nan]), potential_kj=None,
        source="test", meta={},
    )
    assert d.boost_dih_kj is None
    assert d.beta == 1.0
