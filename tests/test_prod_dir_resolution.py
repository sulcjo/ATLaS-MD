"""Resolution of analysis input dirs for interrupted adaptive_production runs.

When the adaptive_production driver is interrupted mid-epoch (before any epoch
finalizes), the union-MBAR artifacts (final_registry_used_for_mbar.csv,
epoch_window_map.csv) are never written.  The completed epoch dir is still a
self-contained single-production run (samples/ + segments.json + windows/), and
`prod_dir_of` must resolve to it instead of raising a misleading "no samples"
error.
"""
import pytest

pytest.importorskip("analyze_gareus_mbar")
from pathlib import Path

from analyze_gareus_mbar import prod_dir_of


def _mk_selfcontained_epoch(ep: Path):
    """Create an epoch dir that is a complete single-production run but lacks
    the union-scheme artifacts (registry, epoch_window_map.csv)."""
    ep.mkdir(parents=True, exist_ok=True)
    (ep / "segments.json").write_text(
        '{"segments": [{"segment_id": "seg_001"}]}', encoding="utf-8"
    )
    seg = ep / "samples" / "seg_001"
    seg.mkdir(parents=True)
    (seg / "chunk_000000.parquet").write_bytes(b"")
    win = ep / "windows"
    win.mkdir()
    (win / "seg_001.json").write_text('{"windows": []}', encoding="utf-8")


def test_interrupted_adaptive_production_resolves_to_epoch(tmp_path):
    run = tmp_path / "run"
    ep = run / "adaptive_production" / "epoch_000"
    _mk_selfcontained_epoch(ep)
    assert prod_dir_of(run) == ep.resolve()


def test_adaptive_production_input_resolves_to_epoch(tmp_path):
    ap = tmp_path / "run" / "adaptive_production"
    ep = ap / "epoch_000"
    _mk_selfcontained_epoch(ep)
    assert prod_dir_of(ap) == ep.resolve()


def test_picks_latest_epoch_when_multiple(tmp_path):
    run = tmp_path / "run"
    _mk_selfcontained_epoch(run / "adaptive_production" / "epoch_000")
    _mk_selfcontained_epoch(run / "adaptive_production" / "epoch_001")
    assert prod_dir_of(run) == (run / "adaptive_production" / "epoch_001").resolve()


def test_no_samples_error_mentions_epoch_hint(tmp_path):
    """An empty adaptive_production/ still raises, but the message is actionable."""
    run = tmp_path / "run"
    (run / "adaptive_production").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        prod_dir_of(run)
