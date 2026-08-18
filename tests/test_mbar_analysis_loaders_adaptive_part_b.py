import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from gareus.mbar_analysis.loaders_adaptive import (
    load_epoch_csv_adaptive, load_union_npz, _load_round_raw,
    _build_union_window_table, _round_window_to_union_map,
    _augment_with_adaptive_rounds,
)


def _write_samples_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_load_epoch_csv_adaptive_unions_windows_across_epoch_sources(tmp_path):
    ap = tmp_path / "adaptive_production"
    # secondary_cv_center/secondary_cv_k_kcal_mol included as finite 0.0, matching
    # the precedent in tests/test_mbar_epoch_filter.py and how a real CV1-only
    # samples.csv row actually looks on disk: gareus/production.py always writes
    # the secondary_cv_center *column*, but as "" (blank) for CV1-only runs, not
    # a numeric 0.0 -- and `_fkey`'s `float('') ` -> except -> a *fresh*
    # `float('nan')` default means two rows' "no secondary CV" keys are never
    # `==` to each other as dict keys (NaN != NaN, and they're different
    # objects). That's a real, reachable latent bug in the verbatim-relocated
    # `load_epoch_csv_adaptive` (see task-6 report) -- out of scope for this
    # pure-relocation task, so this fixture sidesteps it with an explicit
    # finite secondary axis instead of exercising the bug.
    fields = ["cv_A", "primary_cv_center", "primary_cv_k",
              "secondary_cv_center", "secondary_cv_k_kcal_mol", "step", "replica"]
    _write_samples_csv(
        ap / "epoch_000" / "samples.csv",
        [{"cv_A": "1.0", "primary_cv_center": "1.0", "primary_cv_k": "10.0",
          "secondary_cv_center": "0.0", "secondary_cv_k_kcal_mol": "0.0", "step": "0", "replica": "0"}],
        fields,
    )
    _write_samples_csv(
        ap / "epoch_001" / "samples.csv",
        [{"cv_A": "2.0", "primary_cv_center": "2.0", "primary_cv_k": "10.0",
          "secondary_cv_center": "0.0", "secondary_cv_k_kcal_mol": "0.0", "step": "0", "replica": "0"}],
        fields,
    )
    (ap.parent / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    d = load_epoch_csv_adaptive(ap)
    assert d.cv.size == 2
    assert d.u_nk.shape == (2, 2)
    assert d.meta["source"] == "epoch_csv_adaptive"


def test_load_union_npz_reads_arrays_and_infers_beta(tmp_path):
    ap = tmp_path / "adaptive_production"
    ap.mkdir(parents=True)
    np.savez(
        ap / "adaptive_union_mbar.npz",
        cv_A=np.array([1.0, 2.0]),
        secondary_cv=np.array([np.nan, np.nan]),
        sampled_state_ids=np.array([0, 1]),
        umbrella_reduced_bias_nk=np.zeros((2, 2)),
        primary_centers=np.array([1.0, 2.0]),
        primary_k=np.array([10.0, 10.0]),
    )
    (ap.parent / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    d = load_union_npz(ap)
    assert d.cv.size == 2
    assert d.meta["source"] == "adaptive_union_mbar"


def test_load_round_raw_concatenates_chunks(tmp_path):
    rd = tmp_path / "adaptive_feedback_round_01"
    chunks = rd / "analysis_chunks"
    chunks.mkdir(parents=True)
    np.savez(chunks / "chunk_000.npz", cv_A=np.array([1.0, 2.0]))
    np.savez(chunks / "chunk_001.npz", cv_A=np.array([3.0]))
    raw = _load_round_raw(rd)
    assert raw is not None
    assert list(raw["cv_A"]) == [1.0, 2.0, 3.0]
    assert raw["window"].size == 3


def test_build_union_window_table_deduplicates_by_tolerance(tmp_path):
    d1 = tmp_path / "d1"; d1.mkdir()
    (d1 / "umbrella_windows.csv").write_text(
        "window,center_A,k_kcal_mol_A2,primary_center,primary_k\n0,1.0,10.0,1.0,10.0\n")
    d2 = tmp_path / "d2"; d2.mkdir()
    (d2 / "umbrella_windows.csv").write_text(
        "window,center_A,k_kcal_mol_A2,primary_center,primary_k\n0,1.00001,10.0,1.00001,10.0\n"
        "1,2.0,10.0,2.0,10.0\n")
    table = _build_union_window_table([d1, d2])
    assert len(table) == 2  # d1's row and d2's first row merge (within tol)


def test_round_window_to_union_map_maps_local_to_union_index(tmp_path):
    rd = tmp_path / "round"; rd.mkdir()
    (rd / "umbrella_windows.csv").write_text(
        "window,center_A,k_kcal_mol_A2,primary_center,primary_k\n0,2.0,10.0,2.0,10.0\n")
    # secondary_cv_center=0.0 (not NaN): `_build_union_window_table` -- the
    # sibling function that actually produces `union_windows` on the real
    # `_augment_with_adaptive_rounds` call path -- reads a CSV lacking a
    # secondary_cv_center column via `float(row.get('secondary_cv_center', 0))`,
    # which always defaults to finite 0.0, never NaN. A NaN secondary center
    # here would compare `abs(nan - sc) < tol` -> always False against this
    # round's own `sc = float(row.get('secondary_cv_center', 0))` (also 0.0),
    # so a NaN fixture can never map -- a shape the real pipeline can't
    # produce, not a bug this function needs to handle.
    union_windows = [
        {"primary_center": 1.0, "secondary_cv_center": 0.0, "primary_k_kcal": 10.0, "secondary_k_kcal": 0.0},
        {"primary_center": 2.0, "secondary_cv_center": 0.0, "primary_k_kcal": 10.0, "secondary_k_kcal": 0.0},
    ]
    mapping = _round_window_to_union_map(rd, union_windows)
    assert mapping == {0: 1}


def test_augment_with_adaptive_rounds_returns_original_when_no_round_dirs(tmp_path, monkeypatch):
    import gareus.mbar_analysis.data as mdata
    d = mdata.Data(
        prod_dir=tmp_path, out_dir=tmp_path, cv=np.array([1.0]), cv2=np.array([np.nan]),
        rg_A=np.array([np.nan]), window=np.array([0]), replica=np.array([0]),
        step=np.array([0]), u_nk=np.zeros((1, 1)), centers=np.array([1.0]),
        k_kcal=np.array([10.0]), beta=1.0, temp=300.0, boost_kj=np.array([np.nan]),
        potential_kj=None, source="test", meta={},
    )
    out = _augment_with_adaptive_rounds(d, tmp_path)
    assert out is d


def test_augment_with_adaptive_rounds_merges_round_samples_with_final(tmp_path):
    """Real invocation of the full merge path (not just the no-round-dirs
    early return above) -- exercises the lazy
    `from analyze_gareus_mbar import _compute_u_nk_analytical` import inside
    the function body. _compute_u_nk_analytical now lives in
    gareus.mbar_analysis.bias (Plan A3); analyze_gareus_mbar.py re-exports it
    unchanged, so this lazy import (kept pointed at analyze_gareus_mbar
    deliberately, for the circular-import reason documented at the import
    site itself) still resolves correctly."""
    import gareus.mbar_analysis.data as mdata

    run_dir = tmp_path
    windows_csv = "window,center_A,k_kcal_mol_A2,primary_center,primary_k\n0,1.0,10.0,1.0,10.0\n"
    (run_dir / "umbrella_windows.csv").write_text(windows_csv)
    round_dir = run_dir / "adaptive_feedback_round_01"
    round_dir.mkdir()
    (round_dir / "umbrella_windows.csv").write_text(windows_csv)
    chunks = round_dir / "analysis_chunks"
    chunks.mkdir()
    np.savez(
        chunks / "chunk_000.npz",
        cv_A=np.array([1.1, 1.2]), step=np.array([0, 1]), replica=np.array([0, 0]),
        window=np.array([0, 0]), gamd_boost_total_kj_mol=np.array([0.5, 0.6]),
        potential_kj_mol=np.array([-10.0, -11.0]),
    )
    d = mdata.Data(
        prod_dir=run_dir, out_dir=run_dir, cv=np.array([1.0]), cv2=np.array([np.nan]),
        rg_A=np.array([np.nan]), window=np.array([0]), replica=np.array([0]),
        step=np.array([0]), u_nk=np.zeros((1, 1)), centers=np.array([1.0]),
        k_kcal=np.array([10.0]), beta=1.0, temp=300.0, boost_kj=np.array([np.nan]),
        potential_kj=None, source="test", meta={},
    )

    out = _augment_with_adaptive_rounds(d, run_dir)

    assert out.cv.size == 3  # 1 final-production sample + 2 round samples
    assert out.u_nk.shape == (3, 1)
    assert np.all(np.isfinite(out.u_nk))
