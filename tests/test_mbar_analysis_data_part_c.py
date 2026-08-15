import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gareus.mbar_analysis.data import (
    Data, _epoch_dir_index, _epoch_number_for_run_dir,
    _epoch_run_manifest_secondary_cv_type, _epoch_zero_split_masks,
    _secondary_cv_epoch_regime_masks,
)


def _mk_data(n, epoch_src, run_dirs, cv=None):
    return Data(
        prod_dir=Path("/p"), out_dir=Path("/o"),
        cv=cv if cv is not None else np.arange(n, dtype=float),
        cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=np.zeros(n, int), replica=np.zeros(n, int), step=np.arange(n),
        u_nk=np.zeros((n, 1)), centers=np.zeros(1), k_kcal=np.zeros(1),
        beta=1.0, temp=300.0, boost_kj=np.full(n, np.nan), potential_kj=None,
        source="test",
        meta={"_epoch_source": epoch_src, "adaptive_epoch_run_dirs": [str(r) for r in run_dirs]},
    )


def test_epoch_dir_index_matches_epoch_nnn():
    assert _epoch_dir_index(Path("/a/epoch_003")) == 3
    assert _epoch_dir_index(Path("/a/final")) is None


def test_epoch_number_for_run_dir_checks_flat_and_subrun_layout():
    assert _epoch_number_for_run_dir(Path("/a/epoch_002")) == 2
    assert _epoch_number_for_run_dir(Path("/a/epoch_002/baseline")) == 2
    assert _epoch_number_for_run_dir(Path("/a/final/baseline")) is None


def test_epoch_run_manifest_secondary_cv_type_reads_resolved_args(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}), encoding="utf-8")
    assert _epoch_run_manifest_secondary_cv_type(tmp_path) == "torsion-pca"


def test_epoch_run_manifest_secondary_cv_type_missing_file_returns_empty(tmp_path):
    assert _epoch_run_manifest_secondary_cv_type(tmp_path / "nope") == ""


def test_epoch_zero_split_masks_splits_epoch0_from_rest(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    d = _mk_data(4, [0, 0, 1, 1], [e0, e1])
    result = _epoch_zero_split_masks(d)
    assert result is not None
    mask0, mask_rest = result
    assert list(mask0) == [True, True, False, False]
    assert list(mask_rest) == [False, False, True, True]


def test_epoch_zero_split_masks_returns_none_without_epoch_source():
    d = _mk_data(4, None, [])
    d.meta = {}
    assert _epoch_zero_split_masks(d) is None


def test_secondary_cv_epoch_regime_masks_detects_two_regimes(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}), encoding="utf-8")
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    (e1 / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "tica-linear"}}), encoding="utf-8")
    d = _mk_data(4, [0, 0, 1, 1], [e0, e1])
    regimes = _secondary_cv_epoch_regime_masks(d)
    assert regimes is not None
    assert set(regimes.keys()) == {"torsion-pca", "tica-linear"}
    # tica-linear is chronologically last -> dominant.
    assert regimes["tica-linear"][1] is True
    assert regimes["torsion-pca"][1] is False


def test_secondary_cv_epoch_regime_masks_returns_none_for_single_regime(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}), encoding="utf-8")
    d = _mk_data(2, [0, 0], [e0])
    assert _secondary_cv_epoch_regime_masks(d) is None
