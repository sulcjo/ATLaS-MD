import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from gareus.mbar_analysis.loaders import (
    _npz_sample_count_open, _npz_window_count_open, _Arrays, _ANALYSIS_VECTOR_KEYS,
    _discover_analysis_chunk_paths, _append_npz_arrays, _load_merged_arrays, load_npz,
    _load_secondary_cv_from_csv, _csv_row_count_fast, _npz_sample_count,
    _analysis_binary_sample_count, _window_float_array, load_csv,
    _parquet_sample_count, prod_dir_of, load_data,
)


def _write_umbrella_windows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["center_A", "k_kcal_mol_A2"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_npz_sample_count_open_prefers_step_key(tmp_path):
    npz_path = tmp_path / "a.npz"
    np.savez(npz_path, step=np.array([0, 1, 2]))
    with np.load(npz_path) as f:
        assert _npz_sample_count_open(f) == 3


def test_arrays_wrapper_exposes_files_property():
    a = _Arrays({"cv_A": np.array([1.0])})
    assert a.files == ["cv_A"]


def test_load_merged_arrays_and_load_npz_roundtrip(tmp_path):
    prod = tmp_path
    n = 3
    np.savez(
        prod / "analysis_arrays.npz",
        cv_A=np.arange(n, dtype=float), window=np.zeros(n, int),
        replica=np.zeros(n, int), step=np.arange(n),
        umbrella_reduced_bias_nk=np.zeros((n, 1)),
    )
    _write_umbrella_windows(prod / "umbrella_windows.csv", [{"center_A": "1.0", "k_kcal_mol_A2": "10.0"}])
    arr, notes = _load_merged_arrays(prod)
    assert arr["cv_A"].size == n
    d = load_npz(prod)
    assert d.cv.size == n
    assert d.centers.size == 1


def test_load_secondary_cv_from_csv_returns_nan_on_size_mismatch(tmp_path):
    p = tmp_path / "samples.csv"
    p.write_text("secondary_cv\n1.0\n2.0\n")
    out = _load_secondary_cv_from_csv(p, expected_size=5)
    assert out.size == 5
    assert np.all(np.isnan(out))


def test_csv_row_count_fast_counts_data_rows_not_header(tmp_path):
    p = tmp_path / "samples.csv"
    p.write_text("cv_A\n1.0\n2.0\n3.0\n")
    assert _csv_row_count_fast(p) == 3


def test_window_float_array_falls_back_to_default_on_bad_value():
    rows = [{"k": "1.5"}, {"k": "not-a-number"}]
    out = _window_float_array(rows, ("k",), default=-1.0)
    assert list(out) == [1.5, -1.0]


def test_load_csv_reconstructs_bias_matrix_from_windows(tmp_path):
    prod = tmp_path
    _write_umbrella_windows(prod / "umbrella_windows.csv", [
        {"center_A": "1.0", "k_kcal_mol_A2": "10.0"},
        {"center_A": "2.0", "k_kcal_mol_A2": "10.0"},
    ])
    with (prod / "samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cv_A", "window", "replica", "step"])
        w.writeheader()
        w.writerow({"cv_A": "1.0", "window": "0", "replica": "0", "step": "0"})
        w.writerow({"cv_A": "1.5", "window": "0", "replica": "0", "step": "1"})
    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    d = load_csv(prod)
    assert d.cv.size == 2
    assert d.u_nk.shape == (2, 2)


def test_parquet_sample_count_returns_zero_without_duckdb_data(tmp_path):
    assert _parquet_sample_count(tmp_path) == 0


def test_prod_dir_of_resolves_flat_npz_layout(tmp_path):
    (tmp_path / "analysis_arrays.npz").write_bytes(b"")
    assert prod_dir_of(tmp_path) == tmp_path.resolve()


def test_prod_dir_of_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        prod_dir_of(tmp_path)


def test_load_data_auto_selects_csv_when_only_csv_present(tmp_path):
    prod = tmp_path
    _write_umbrella_windows(prod / "umbrella_windows.csv", [{"center_A": "1.0", "k_kcal_mol_A2": "10.0"}])
    with (prod / "samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cv_A", "window", "replica", "step"])
        w.writeheader()
        w.writerow({"cv_A": "1.0", "window": "0", "replica": "0", "step": "0"})
    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    d = load_data(prod, out=None)
    assert d.cv.size == 1
    assert d.source.endswith("samples.csv")


def test_npz_sample_count_reads_real_npz_file(tmp_path):
    p = tmp_path / "analysis_arrays.npz"
    np.savez(p, cv_A=np.arange(5, dtype=float))
    assert _npz_sample_count(p) == 5


def test_npz_sample_count_missing_file_returns_zero(tmp_path):
    assert _npz_sample_count(tmp_path / "does_not_exist.npz") == 0


def test_analysis_binary_sample_count_sums_main_and_chunk_files(tmp_path):
    prod = tmp_path / "prod"
    prod.mkdir()
    np.savez(prod / "analysis_arrays.npz", cv_A=np.arange(5, dtype=float))
    chunks = prod / "analysis_chunks"
    chunks.mkdir()
    np.savez(chunks / "chunk_001.npz", cv_A=np.arange(3, dtype=float))
    assert _analysis_binary_sample_count(prod) == 8
