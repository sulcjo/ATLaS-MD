from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# _load_epoch_data
# ---------------------------------------------------------------------------

def test_load_epoch_data_basic(tmp_path: Path) -> None:
    from gareus.epoch_plots import _load_epoch_data

    for i in range(3):
        epoch_dir = tmp_path / f"epoch_{i:03d}"
        epoch_dir.mkdir()
        np.savez(
            epoch_dir / "analysis_arrays.npz",
            cv_A=np.linspace(0.0, 1.0, 100),
            secondary_cv=np.linspace(-1.0, 1.0, 100),
            window=np.zeros(100, dtype=np.int32),
            step=np.arange(100, dtype=np.int64),
        )

    epochs = _load_epoch_data(tmp_path)
    assert len(epochs) == 3
    assert epochs[0]["epoch_idx"] == 0
    assert epochs[0]["cv_A"].shape == (100,)
    assert epochs[1]["epoch_idx"] == 1
    assert epochs[2]["epoch_idx"] == 2
    assert epochs[0]["n_samples"] == 100


def test_load_epoch_data_no_dirs(tmp_path: Path) -> None:
    from gareus.epoch_plots import _load_epoch_data

    with pytest.raises(ValueError):
        _load_epoch_data(tmp_path)


def test_load_epoch_data_malformed_dir(tmp_path: Path) -> None:
    from gareus.epoch_plots import _load_epoch_data

    # Create two valid epoch dirs
    for i in range(2):
        epoch_dir = tmp_path / f"epoch_{i:03d}"
        epoch_dir.mkdir()
        np.savez(
            epoch_dir / "analysis_arrays.npz",
            cv_A=np.linspace(0.0, 1.0, 50),
            secondary_cv=np.linspace(-1.0, 1.0, 50),
            window=np.zeros(50, dtype=np.int32),
            step=np.arange(50, dtype=np.int64),
        )

    # Create a malformed dir (bad name, no npz)
    bad_dir = tmp_path / "epoch_badname"
    bad_dir.mkdir()

    epochs = _load_epoch_data(tmp_path)
    # Malformed dir should be skipped; valid epochs still loaded
    assert len(epochs) == 2
    epoch_indices = [e["epoch_idx"] for e in epochs]
    assert 0 in epoch_indices
    assert 1 in epoch_indices


# ---------------------------------------------------------------------------
# _load_window_data
# ---------------------------------------------------------------------------

def test_load_window_data(tmp_path: Path) -> None:
    from gareus.epoch_plots import _load_window_data

    for epoch_idx in [1, 2]:
        df = pd.DataFrame({
            "primary_cv_center": [0.1, 0.2],
            "secondary_cv_center": [-0.5, 0.5],
            "state_id": [0, 1],
        })
        df.to_csv(tmp_path / f"windows_epoch_{epoch_idx:03d}.csv", index=False)

    result = _load_window_data(tmp_path)
    assert 1 in result
    assert 2 in result
    assert result[1].shape == (2, 3)
    assert list(result[1].columns) == ["primary_cv_center", "secondary_cv_center", "state_id"]


# ---------------------------------------------------------------------------
# _read_cv_labels
# ---------------------------------------------------------------------------

def test_read_cv_labels_contacts(tmp_path: Path) -> None:
    from gareus.epoch_plots import _read_cv_labels

    manifest = {
        "resolved_args": {
            "cv1": "contacts",
            "cv2": "rama-map",
        }
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    cv1_label, cv2_label = _read_cv_labels(tmp_path)
    assert cv1_label == "Contact fraction CV1"
    assert cv2_label == "Ramachandran CV2"


def test_read_cv_labels_missing(tmp_path: Path) -> None:
    from gareus.epoch_plots import _read_cv_labels

    # No run_manifest.json in tmp_path
    cv1_label, cv2_label = _read_cv_labels(tmp_path)
    assert cv1_label == "CV1"
    assert cv2_label == "CV2"


# ---------------------------------------------------------------------------
# plot_epoch_cv_exploration — 2D integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_plot_epoch_cv_exploration_2d(tmp_path: Path) -> None:
    from gareus.epoch_plots import plot_epoch_cv_exploration

    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir()

    rng = np.random.default_rng(42)
    for i in range(2):
        epoch_dir = adaptive_dir / f"epoch_{i:03d}"
        epoch_dir.mkdir()
        np.savez(
            epoch_dir / "analysis_arrays.npz",
            cv_A=rng.uniform(0.0, 1.0, 200),
            secondary_cv=rng.uniform(-1.0, 1.0, 200),
            window=np.zeros(200, dtype=np.int32),
            step=np.arange(200, dtype=np.int64),
        )

    df = pd.DataFrame({
        "primary_cv_center": rng.uniform(0.0, 1.0, 5),
        "secondary_cv_center": rng.uniform(-1.0, 1.0, 5),
        "state_id": np.arange(5),
    })
    df.to_csv(adaptive_dir / "windows_epoch_001.csv", index=False)

    manifest = {"resolved_args": {"cv1": "contacts", "cv2": "rama-map"}}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    generated = plot_epoch_cv_exploration(tmp_path)

    assert len(generated) == 3
    for p in generated:
        assert Path(p).exists(), f"Expected output file missing: {p}"
        assert Path(p).suffix == ".png", f"Expected .png, got: {p}"


# ---------------------------------------------------------------------------
# plot_epoch_cv_exploration — 1D fallback (secondary_cv all NaN)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
def test_plot_epoch_cv_exploration_1d_fallback(tmp_path: Path) -> None:
    from gareus.epoch_plots import plot_epoch_cv_exploration

    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir()

    rng = np.random.default_rng(7)
    for i in range(2):
        epoch_dir = adaptive_dir / f"epoch_{i:03d}"
        epoch_dir.mkdir()
        np.savez(
            epoch_dir / "analysis_arrays.npz",
            cv_A=rng.uniform(0.0, 1.0, 200),
            secondary_cv=np.full(200, np.nan),
            window=np.zeros(200, dtype=np.int32),
            step=np.arange(200, dtype=np.int64),
        )

    # Provide window CSV so _make_window_evolution_plot actually writes its file
    df = pd.DataFrame({
        "primary_cv_center": rng.uniform(0.0, 1.0, 5),
        "secondary_cv_center": rng.uniform(-1.0, 1.0, 5),
        "state_id": np.arange(5),
    })
    df.to_csv(adaptive_dir / "windows_epoch_001.csv", index=False)

    manifest = {"resolved_args": {"cv1": "contacts", "cv2": "none"}}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))

    generated = plot_epoch_cv_exploration(tmp_path)

    assert len(generated) == 3
    for p in generated:
        assert Path(p).exists(), f"Expected output file missing: {p}"
        assert Path(p).suffix == ".png", f"Expected .png, got: {p}"
