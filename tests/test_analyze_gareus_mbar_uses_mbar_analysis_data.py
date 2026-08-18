import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.mbar_analysis.data as mdata
import analyze_gareus_mbar as agm

_NAMES = [
    "Data", "rjson", "wjson", "read_windows", "jvec", "infer_temp_beta",
    "clean", "_masked_data", "_apply_analysis_stride", "_filter_epoch_source",
    "_sample_block_ids", "_skip_first_n_frames", "_epoch_dir_index",
    "_epoch_number_for_run_dir", "_epoch_run_manifest_secondary_cv_type",
    "_epoch_zero_split_masks", "_secondary_cv_epoch_regime_masks",
]


def test_every_relocated_data_name_is_the_shared_object():
    for name in _NAMES:
        assert getattr(agm, name) is getattr(mdata, name), (
            f"{name}: analyze_gareus_mbar still defines its own copy "
            f"instead of importing gareus.mbar_analysis.data's"
        )
