"""Confirms gareus.analysis.validate_analysis_metadata_readiness tolerates a
non-finite umbrella_reduced_bias_nk as a warning, not a validation failure.

This matters because of a fix landing in the same modularization plan:
gareus.query.reconstruct_bias_matrix used to fabricate a zero deviation for
any sample with an unmeasured (NaN) secondary CV under a restrained 2D
window, instead of correctly excluding it via NaN. Once fixed,
export_analysis_arrays_npz (which does NOT row-mask, unlike
analyze_gareus_mbar.py's clean()) can, for the first time, persist NaN
entries in analysis_arrays.npz's umbrella_reduced_bias_nk for a real run with
genuinely-unmeasured secondary-CV samples. This test constructs that npz
directly (bypassing the Parquet/export pipeline, to isolate this consumer's
own tolerance from the reconstruct_bias_matrix fix itself) and confirms
validate_analysis_metadata_readiness surfaces it as an informational warning
rather than an error or a crash.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gareus.analysis")

from gareus.analysis import validate_analysis_metadata_readiness


def _write_minimal_1d_window_csv(out_dir, n_windows: int) -> None:
    with (out_dir / "umbrella_windows.csv").open("w") as f:
        f.write("window,distance_center_A,distance_k_kcal_mol_A2\n")
        for i in range(n_windows):
            f.write(f"{i},{i * 0.5},200.0\n")


def test_nonfinite_bias_matrix_produces_warning_not_error(tmp_path):
    n_samples, n_windows = 6, 2
    cv_a = np.linspace(-1.0, 1.0, n_samples)
    window = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    bias = np.full((n_samples, n_windows), 1.0)
    bias[2, 0] = np.nan  # one sample's reduced bias entry is unmeasured/excluded

    np.savez_compressed(
        tmp_path / "analysis_arrays.npz",
        cv_A=cv_a, window=window, umbrella_reduced_bias_nk=bias,
    )
    _write_minimal_1d_window_csv(tmp_path, n_windows)

    result = validate_analysis_metadata_readiness(tmp_path)

    assert any("umbrella_reduced_bias_nk contains non-finite values" in w
               for w in result["warnings"]), result["warnings"]
    assert not any("umbrella_reduced_bias_nk" in e for e in result["errors"]), result["errors"]


def test_fully_finite_bias_matrix_produces_no_such_warning(tmp_path):
    """Negative control: the warning must not fire when there is nothing
    non-finite -- proves the check above is specific, not a false positive
    that always fires."""
    n_samples, n_windows = 6, 2
    cv_a = np.linspace(-1.0, 1.0, n_samples)
    window = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    bias = np.full((n_samples, n_windows), 1.0)

    np.savez_compressed(
        tmp_path / "analysis_arrays.npz",
        cv_A=cv_a, window=window, umbrella_reduced_bias_nk=bias,
    )
    _write_minimal_1d_window_csv(tmp_path, n_windows)

    result = validate_analysis_metadata_readiness(tmp_path)

    assert not any("umbrella_reduced_bias_nk contains non-finite values" in w
                   for w in result["warnings"]), result["warnings"]
