"""Regression tests for selecting adaptive MBAR epochs."""

from pathlib import Path

from analyze_gareus_mbar import (
    _find_adaptive_epoch_csv_sources,
    _find_adaptive_epoch_dirs,
    load_epoch_csv_adaptive,
    parse_args,
)


def _csv_source(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "samples.csv").write_text("cv_A\n1.0\n", encoding="utf-8")


def _parquet_source(path: Path) -> None:
    (path / "samples").mkdir(parents=True)
    (path / "segments.json").write_text("{}", encoding="utf-8")


def test_epoch_filter_selects_only_requested_baseline_and_topups(tmp_path):
    """Prevent CV2-incompatible epochs entering a selected epoch MBAR union."""
    ap = tmp_path / "adaptive_production"
    _csv_source(ap / "epoch_000")
    _csv_source(ap / "epoch_001" / "baseline")
    _csv_source(ap / "epoch_001" / "topup_001_10")
    _csv_source(ap / "epoch_002" / "baseline")

    _parquet_source(ap / "epoch_000")
    (ap / "epoch_000" / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n", encoding="utf-8")
    _parquet_source(ap / "epoch_001" / "baseline")
    _parquet_source(ap / "epoch_001" / "topup_001_10")
    (ap / "epoch_001" / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n", encoding="utf-8")

    expected = [
        ap / "epoch_001" / "baseline",
        ap / "epoch_001" / "topup_001_10",
    ]
    assert _find_adaptive_epoch_csv_sources(ap, epoch_ids={1}) == expected
    assert [path for path, _ in _find_adaptive_epoch_dirs(ap, epoch_ids={1})] == expected
    assert parse_args(["run", "--epoch", "1", "--epoch", "3"]).epochs == [1, 3]


def test_epoch_filter_always_includes_final_phase(tmp_path):
    """`--epoch` restricted to numbered epochs must not drop final/."""
    ap = tmp_path / "adaptive_production"

    _parquet_source(ap / "epoch_000")
    (ap / "epoch_000" / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n", encoding="utf-8")

    _parquet_source(ap / "epoch_001" / "baseline")
    (ap / "epoch_001" / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n", encoding="utf-8")

    _parquet_source(ap / "epoch_002" / "baseline")
    (ap / "epoch_002" / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n", encoding="utf-8")

    _parquet_source(ap / "final" / "baseline")
    _parquet_source(ap / "final" / "topup_001_10")
    (ap / "final" / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n", encoding="utf-8")

    expected = [
        ap / "epoch_001" / "baseline",
        ap / "final" / "baseline",
        ap / "final" / "topup_001_10",
    ]
    assert [path for path, _ in _find_adaptive_epoch_dirs(ap, epoch_ids={1})] == expected


def test_csv_epoch_filter_pools_requested_epoch_sources_only(tmp_path):
    """Prevent `--epoch 1` from retaining epoch-000 CV2 samples in MBAR."""
    ap = tmp_path / "adaptive_production"
    header = "cv_A,primary_cv_center,primary_cv_k,secondary_cv,secondary_cv_center,secondary_cv_k_kcal_mol\n"
    for path, value in (
        (ap / "epoch_000", "0.0"),
        (ap / "epoch_001" / "baseline", "1.0"),
        (ap / "epoch_001" / "topup_001_10", "2.0"),
    ):
        path.mkdir(parents=True, exist_ok=True)
        (path / "samples.csv").write_text(
            header + f"{value},1.0,1.0,0.0,0.0,0.0\n", encoding="utf-8")

    data = load_epoch_csv_adaptive(ap, epoch_ids={1})

    assert data.cv.tolist() == [1.0, 2.0]
    assert [Path(path).name for path in data.meta["adaptive_epoch_run_dirs"]] == [
        "baseline", "topup_001_10",
    ]
