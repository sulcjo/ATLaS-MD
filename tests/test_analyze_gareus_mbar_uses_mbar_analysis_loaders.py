import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.mbar_analysis.loaders_adaptive as mladapt
import gareus.mbar_analysis.loaders_union_parquet as mlunion
import gareus.mbar_analysis.loaders as mloaders
import analyze_gareus_mbar as agm

_ADAPTIVE_NAMES = [
    "_find_adaptive_epoch_dirs", "_find_selfcontained_epoch_dirs",
    "_find_adaptive_epoch_csv_sources", "_has_epoch_csv_layout",
    "_find_adaptive_final_run_dirs", "_find_gareus_round_dirs",
    "_vectorized_map_lookup", "_vectorized_map_lookup_or_self", "_vectorized_map_index",
    "load_epoch_csv_adaptive", "load_union_npz", "_load_round_raw",
    "_build_union_window_table", "_round_window_to_union_map",
    "_augment_with_adaptive_rounds",
]
_UNION_NAMES = ["_is_usable_for_mbar", "_merge_missing_usable_states", "load_parquet_adaptive_union"]
_LOADERS_NAMES = [
    "_npz_sample_count_open", "_npz_window_count_open", "_Arrays", "_ANALYSIS_VECTOR_KEYS",
    "_discover_analysis_chunk_paths", "_append_npz_arrays", "_load_merged_arrays", "load_npz",
    "_load_secondary_cv_from_csv", "_csv_row_count_fast", "_npz_sample_count",
    "_analysis_binary_sample_count", "_window_float_array", "load_csv",
    "_parquet_sample_count", "load_parquet", "prod_dir_of", "load_data",
]


def test_adaptive_names_are_shared_objects():
    for name in _ADAPTIVE_NAMES:
        assert getattr(agm, name) is getattr(mladapt, name), name


def test_union_names_are_shared_objects():
    for name in _UNION_NAMES:
        assert getattr(agm, name) is getattr(mlunion, name), name


def test_loaders_names_are_shared_objects():
    for name in _LOADERS_NAMES:
        assert getattr(agm, name) is getattr(mloaders, name), name


def test_no_circular_import_via_fresh_subprocess():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", "import gareus.mbar_analysis.loaders"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
