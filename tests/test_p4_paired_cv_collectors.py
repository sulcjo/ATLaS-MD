"""Spec P4: row-paired (CV1, CV2) data in every adaptive-production collector.

OpenMM-free.  Synthetic samples.csv fixtures, the three real collectors.
"""

from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gareus.adaptive_production as ap
from gareus.adaptive import paired_cv
from gareus.adaptive_production import (
    AdaptiveDecisionPolicy,
    collect_epoch_diagnostics,
    collect_final_combined_diagnostics,
    collect_segmented_epoch_diagnostics,
    registry_from_window_csv,
)

# (primary centre, secondary centre); states 0 and 1 share CV1 and differ only in CV2.
CENTRES = [(0.30, -0.6), (0.30, 0.6), (0.50, 0.0)]
SIGMA1, SIGMA2 = 0.02, 0.08


def _write_windows(path: Path, centres=CENTRES, secondary_k: Optional[Sequence[float]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["primary_cv_center", "primary_cv_k_kcal",
                                           "secondary_cv_center", "secondary_cv_k_kcal_mol"])
        w.writeheader()
        for i, (c1, c2) in enumerate(centres):
            w.writerow({"primary_cv_center": c1, "primary_cv_k_kcal": 80.0, "secondary_cv_center": c2,
                        "secondary_cv_k_kcal_mol": 100.0 if secondary_k is None else secondary_k[i]})
    return path


def _write_samples(path: Path, n_per_window: int, seed: int, *, nan_cv2_every: int = 0,
                   centres=CENTRES) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Write samples.csv; return the per-window (cv1, cv2) as written (NaN = blank)."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    written: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for wi, (c1, c2) in enumerate(centres):
        a = np.round(rng.normal(c1, SIGMA1, n_per_window), 6)
        b = np.round(rng.normal(c2, SIGMA2, n_per_window), 6)
        if nan_cv2_every:
            b[::nan_cv2_every] = np.nan
        written[wi] = (a, b)
        for x, y in zip(a, b):
            rows.append({"window": wi, "cv_A": float(x),
                         "secondary_cv": "" if not np.isfinite(y) else float(y),
                         "gamd_boost_total_kcal_mol": 1.0})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["window", "cv_A", "secondary_cv", "gamd_boost_total_kcal_mol"])
        w.writeheader()
        w.writerows(rows)
    return written


def _registry(tmp_path: Path, **kw):
    return registry_from_window_csv(_write_windows(tmp_path / "windows.csv", **kw), epoch=0, source="test")


def _strip_new(payload):
    """The payload with every P4 key removed (for the unchanged-keys comparison)."""
    out = copy.deepcopy(payload)
    out.pop("paired_cv", None)
    for s in out.get("states", []) or []:
        s.pop("paired_cv", None)
    for e in out.get("edges", []) or []:
        e.pop("overlap_joint_2d", None)
        e.pop("overlap_joint_2d_reason", None)
    return out


def _no_attach(monkeypatch):
    monkeypatch.setattr(ap, "attach_paired_cv", lambda payload, collector, json_path: payload)


def _state(payload, sid):
    return next(s for s in payload["states"] if int(s["state_id"]) == sid)


def _edge(payload, a, b):
    return next(e for e in payload["edges"]
                if {int(e["state_i"]), int(e["state_j"])} == {a, b})


# ---------------------------------------------------------------------------
# module units
# ---------------------------------------------------------------------------


def test_axis_moments_match_numpy_population_definitions():
    x = np.random.default_rng(1).gamma(2.0, 1.0, 5000)
    m = paired_cv.axis_moments(np.append(x, np.nan))
    d = x - x.mean()
    assert m["n"] == 5000
    assert m["mean"] == pytest.approx(x.mean())
    assert m["var"] == pytest.approx(x.var(ddof=0))
    assert m["skewness"] == pytest.approx(np.mean(d ** 3) / x.var() ** 1.5)
    assert m["kurtosis_excess"] == pytest.approx(np.mean(d ** 4) / x.var() ** 2 - 3.0)
    assert paired_cv.axis_moments(np.array([np.nan])) is None
    assert paired_cv.axis_moments(np.array([2.0]))["skewness"] is None


def test_stride_keeps_time_order_and_bound():
    idx, stride = paired_cv.stride_indices(5001, 2000)
    assert idx.size <= 2000 and stride == 3
    assert np.all(np.diff(idx) == stride) and idx[0] == 0
    idx, stride = paired_cv.stride_indices(10, 2000)
    assert idx.tolist() == list(range(10)) and stride == 1


def test_joint_overlap_sees_a_cv2_gap_the_marginal_cannot():
    rng = np.random.default_rng(3)
    a1, b1 = rng.normal(0, 1, 4000), rng.normal(0, 1, 4000)
    a2, b2 = rng.normal(-3, 0.5, 4000), rng.normal(3, 0.5, 4000)
    assert paired_cv.joint_overlap_2d(a1, a2, b1, b2) < 0.01
    assert paired_cv.joint_overlap_2d(a1, a2, b1, a2.copy()) > 0.8
    assert paired_cv.joint_overlap_2d(a1[:3], a2[:3], b1, b2) is None


# ---------------------------------------------------------------------------
# flat collector
# ---------------------------------------------------------------------------


def test_flat_collector_attaches_paired_data_moments_and_restraint(tmp_path):
    reg = _registry(tmp_path)
    epoch = tmp_path / "epoch_000"
    written = _write_samples(epoch / "samples.csv", 300, seed=7)
    diag = collect_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())

    assert diag["paired_cv"]["status"] == "ok"
    for sid, (c1, c2) in enumerate(CENTRES):
        pc = _state(diag, sid)["paired_cv"]
        assert pc["restraint"] == {
            "primary_center": pytest.approx(c1), "primary_k": pytest.approx(80.0),
            "secondary_center": pytest.approx(c2), "secondary_k": pytest.approx(100.0),
            "gamd_lambda": 0.0, "cv1_restrained": True, "cv2_restrained": True}
        a, b = written[sid]
        assert pc["n_rows"] == pc["n_pairs"] == 300
        assert pc["cv1"]["mean"] == pytest.approx(a.mean())
        assert pc["cv2"]["var"] == pytest.approx(b.var())
        assert pc["cov_cv1_cv2"] == pytest.approx(np.mean((a - a.mean()) * (b - b.mean())))
        # the existing independent-axis keys agree with the paired moments here
        assert _state(diag, sid)["cv_mean"] == pytest.approx(pc["cv1"]["mean"])

    loaded = paired_cv.load_paired_subsamples(diag["paired_cv"]["npz"])
    assert sorted(loaded) == [0, 1, 2]
    np.testing.assert_allclose(loaded[1]["cv1"], written[1][0])
    np.testing.assert_allclose(loaded[1]["cv2"], written[1][1])
    assert Path(diag["paired_cv"]["npz"]).parent == epoch


def test_flat_collector_pairs_at_row_level_and_drops_missing_cv2(tmp_path):
    reg = _registry(tmp_path)
    epoch = tmp_path / "epoch_000"
    written = _write_samples(epoch / "samples.csv", 300, seed=8, nan_cv2_every=10)
    diag = collect_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    pc = _state(diag, 0)["paired_cv"]
    assert pc["n_rows"] == 300 and pc["n_cv2_nonfinite_dropped"] == 30 and pc["n_pairs"] == 270
    assert pc["retained_fraction"] == pytest.approx(0.9)
    a, b = written[0]
    keep = np.isfinite(b)
    # CV1 moments are over the PAIRED rows, not over every CV1 value
    assert pc["cv1"]["mean"] == pytest.approx(a[keep].mean())
    loaded = paired_cv.load_paired_subsamples(diag["paired_cv"]["npz"])[0]
    assert np.all(np.isfinite(loaded["cv2"]))


def test_cv1_only_state_gets_no_cv2_and_no_joint_overlap(tmp_path):
    reg = _registry(tmp_path, centres=[(0.30, 0.0), (0.34, 0.0), (0.38, 0.0)], secondary_k=[0.0, 0.0, 0.0])
    epoch = tmp_path / "epoch_000"
    _write_samples(epoch / "samples.csv", 200, seed=9, nan_cv2_every=1,
                   centres=[(0.30, 0.0), (0.34, 0.0), (0.38, 0.0)])
    diag = collect_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    pc = _state(diag, 0)["paired_cv"]
    assert pc["restraint"]["cv2_restrained"] is False
    assert pc["cv2"] is None and pc["n_pairs"] == 200 and pc["n_cv2_nonfinite_dropped"] == 0
    for e in diag["edges"]:
        assert e["overlap_joint_2d"] is None and e["overlap_joint_2d_reason"] == "cv2_unrestrained"


def test_subsample_is_bounded_and_strided(tmp_path, monkeypatch):
    monkeypatch.setattr(paired_cv, "MAX_PAIRS_PER_STATE", 50)
    reg = _registry(tmp_path)
    epoch = tmp_path / "epoch_000"
    written = _write_samples(epoch / "samples.csv", 301, seed=10)
    diag = collect_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    sub = _state(diag, 2)["paired_cv"]["subsample"]
    assert sub == {"n_total": 301, "n_kept": 43, "stride": 7, "method": paired_cv.SUBSAMPLE_METHOD}
    loaded = paired_cv.load_paired_subsamples(diag["paired_cv"]["npz"])[2]
    assert loaded["row_index"].tolist() == list(range(0, 301, 7))
    np.testing.assert_allclose(loaded["cv1"], written[2][0][loaded["row_index"]])


# ---------------------------------------------------------------------------
# segmented and final collectors
# ---------------------------------------------------------------------------


def _segmented(tmp_path):
    reg = _registry(tmp_path)
    epoch = tmp_path / "epoch_001"
    w_base = _write_samples(epoch / "baseline" / "samples.csv", 200, seed=11)
    w_top = _write_samples(epoch / "topup_001" / "samples.csv", 100, seed=12)
    return reg, epoch, w_base, w_top


def test_segmented_collector_pools_pairs_across_segments(tmp_path):
    reg, epoch, w_base, w_top = _segmented(tmp_path)
    diag = collect_segmented_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    assert diag["paired_cv"]["status"] == "ok"
    for sid in range(3):
        pc = _state(diag, sid)["paired_cv"]
        a = np.concatenate([w_base[sid][0], w_top[sid][0]])
        b = np.concatenate([w_base[sid][1], w_top[sid][1]])
        assert pc["n_pairs"] == 300
        assert pc["cv1"]["var"] == pytest.approx(a.var())
        assert pc["cv2"]["kurtosis_excess"] == pytest.approx(np.mean((b - b.mean()) ** 4) / b.var() ** 2 - 3)
    loaded = paired_cv.load_paired_subsamples(diag["paired_cv"]["npz"])[0]
    # segment provenance survives the pooling, in segment order
    assert loaded["source_index"].tolist() == [0] * 200 + [1] * 100


def test_segmented_joint_overlap_is_pooled_and_separate_from_cv1_overlap(tmp_path):
    reg, epoch, w_base, w_top = _segmented(tmp_path)
    diag = collect_segmented_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    e01 = _edge(diag, 0, 1)
    # same CV1 centre: the marginal says "overlapping", the joint sees the CV2 gap
    assert e01["overlap"] > 0.6
    assert e01["overlap_joint_2d"] < 0.05 and e01["overlap_joint_2d_reason"] is None
    pooled = [np.concatenate([w_base[s][k], w_top[s][k]]) for s in (0, 1) for k in (0, 1)]
    assert e01["overlap_joint_2d"] == pytest.approx(paired_cv.joint_overlap_2d(*pooled))


def test_final_combined_collector_attaches_paired_data(tmp_path):
    reg = _registry(tmp_path)
    ad = tmp_path / "adaptive_production"
    w_base = _write_samples(ad / "final" / "baseline" / "samples.csv", 150, seed=13)
    w_ext = _write_samples(ad / "final_extension_001" / "samples.csv", 50, seed=14)
    diag = collect_final_combined_diagnostics(ad, reg, AdaptiveDecisionPolicy())
    assert diag["paired_cv"]["status"] == "ok"
    assert Path(diag["paired_cv"]["npz"]).name == "adaptive_final_combined_diagnostics_paired_cv.npz"
    pc = _state(diag, 2)["paired_cv"]
    assert pc["n_pairs"] == 200
    assert pc["cv2"]["mean"] == pytest.approx(np.concatenate([w_base[2][1], w_ext[2][1]]).mean())
    assert _edge(diag, 0, 1)["overlap_joint_2d"] < 0.05


# ---------------------------------------------------------------------------
# existing keys unchanged; failures never raise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["flat", "segmented", "final"])
def test_existing_keys_are_unchanged(tmp_path, monkeypatch, which):
    def run(root):
        reg = _registry(root)
        if which == "flat":
            _write_samples(root / "e" / "samples.csv", 120, seed=20)
            return collect_epoch_diagnostics(root / "e", reg, AdaptiveDecisionPolicy())
        if which == "segmented":
            _write_samples(root / "e" / "baseline" / "samples.csv", 120, seed=21)
            _write_samples(root / "e" / "topup_001" / "samples.csv", 60, seed=22)
            return collect_segmented_epoch_diagnostics(root / "e", reg, AdaptiveDecisionPolicy())
        _write_samples(root / "a" / "final" / "baseline" / "samples.csv", 120, seed=23)
        return collect_final_combined_diagnostics(root / "a", reg, AdaptiveDecisionPolicy())

    with_p4 = run(tmp_path / "on")
    with monkeypatch.context() as m:
        _no_attach(m)
        without = run(tmp_path / "off")

    def norm(p):
        s = _strip_new(p)
        text = repr(s).replace(str(tmp_path / "on"), "ROOT").replace(str(tmp_path / "off"), "ROOT")
        return text

    assert "paired_cv" in with_p4 and "paired_cv" not in without
    assert norm(with_p4) == norm(without)


def test_a_paired_cv_failure_is_reported_not_raised(tmp_path, monkeypatch):
    reg, epoch, _b, _t = _segmented(tmp_path)

    def boom(self, *a, **k):
        raise ValueError("synthetic")

    monkeypatch.setattr(paired_cv.PairedCVCollector, "state_summary", boom)
    diag = collect_segmented_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    assert diag["paired_cv"]["status"] == "error" and "synthetic" in diag["paired_cv"]["error"]
    assert all("paired_cv" not in s for s in diag["states"])
    assert sum(int(s["sample_count"]) for s in diag["states"]) == 900


def test_a_failure_while_adding_rows_is_reported_not_raised(tmp_path, monkeypatch):
    reg = _registry(tmp_path)
    epoch = tmp_path / "epoch_000"
    _write_samples(epoch / "samples.csv", 50, seed=30)
    monkeypatch.setattr(paired_cv, "_to_float", lambda v: (_ for _ in ()).throw(RuntimeError("row")))
    diag = collect_epoch_diagnostics(epoch, reg, AdaptiveDecisionPolicy())
    assert diag["paired_cv"]["status"] == "error" and "row" in diag["paired_cv"]["error"]
    assert _state(diag, 0)["sample_count"] == 50


def test_each_source_is_time_ordered_and_steps_are_recorded(tmp_path):
    reg = _registry(tmp_path)
    coll = paired_cv.PairedCVCollector(reg)
    coll.add_rows(0, [{"step": s, "cv_A": 0.3 + 1e-3 * s, "secondary_cv": -0.6} for s in (30, 10, 20)], "seg_a")
    coll.add_rows(0, [{"step": s, "cv_A": 0.3 + 1e-3 * s, "secondary_cv": -0.6} for s in (5, 6)], "seg_b")
    coll.write_npz(tmp_path / "p.npz", [0])
    got = paired_cv.load_paired_subsamples(tmp_path / "p.npz")[0]
    # sorted within a source, sources kept in the order they were added
    assert got["step"].tolist() == [10, 20, 30, 5, 6]
    assert got["source_index"].tolist() == [0, 0, 0, 1, 1]
    np.testing.assert_allclose(got["cv1"], 0.3 + 1e-3 * got["step"])
