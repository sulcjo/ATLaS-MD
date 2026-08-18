"""Regression tests for splitting secondary-CV PMF / CV1xCV2 2D FES output by
secondary-CV *regime* when a run's CV2 definition changed mid-campaign (e.g.
the torsion-pca -> tica-linear tICA auto-switch).

Pooling raw cv2 values sampled under two different CV2 projections into one
"secondary_cv" axis conflates two different order parameters -- they are not
a recentering of the same coordinate. `run_secondary_cv_analyses` keeps each
regime's own plot honest: the regime containing the run's last epoch/phase
("dominant") gets the existing unmodified output paths; any earlier,
different-regime epoch (e.g. epoch_000 under torsion-pca before a run's tICA
switch) gets its own separate output directory instead of being pooled in.
Runs with a single regime (the overwhelming majority) are unaffected --
`_secondary_cv_epoch_regime_masks` returns None and the wrapper degrades to
the original, unmodified calls.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("analyze_gareus_mbar")

import analyze_gareus_mbar
from analyze_gareus_mbar import (
    Data,
    _epoch_run_manifest_secondary_cv_type,
    _masked_data,
    _regime_slug,
    _secondary_cv_epoch_regime_masks,
    _secondary_cv_regions,
    _slug,
    parse_args,
    run_secondary_cv_analyses,
)


# --- _epoch_run_manifest_secondary_cv_type ----------------------------------

def test_reads_resolved_args_secondary_cv(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}), encoding="utf-8")
    assert _epoch_run_manifest_secondary_cv_type(tmp_path) == "torsion-pca"


def test_missing_manifest_returns_empty_string(tmp_path):
    assert _epoch_run_manifest_secondary_cv_type(tmp_path / "nope") == ""


# --- _regime_slug ------------------------------------------------------------

def test_regime_slug_is_filesystem_safe():
    assert _regime_slug("tica-linear") == "tica-linear"
    # _regime_slug now delegates to the shared _slug helper (Fix 4), which
    # strips leading/trailing underscores after escaping -- "rama_map_"
    # (the old hand-rolled escaping's output) is no longer produced.
    assert _regime_slug("rama map!") == "rama_map"
    assert _regime_slug("") == "unknown"


def test_regime_slug_matches_slug_dedup_and_strip_behavior():
    """Fix 4: _regime_slug delegates to _slug, so it must exhibit _slug's
    escaping/dedup/length-cap behavior exactly, not just its bare
    character-escaping -- verified directly against _slug's own output
    for a string with repeated/leading/trailing special characters, while
    still special-casing an empty/falsy regime to 'unknown' (bare _slug('')
    would return 'item', a worse label for a missing regime type here).
    """
    tricky = "  tica @@ linear!!  "
    assert _regime_slug(tricky) == _slug(tricky)
    # Sanity: this actually exercises dedup/strip, i.e. isn't a vacuous check.
    assert "__" not in _regime_slug(tricky)
    assert not _regime_slug(tricky).startswith("_")
    assert not _regime_slug(tricky).endswith("_")

    long_regime = "x" * 200
    assert _regime_slug(long_regime) == _slug(long_regime)
    assert len(_regime_slug(long_regime)) <= 80

    assert _regime_slug("") == "unknown"
    assert _regime_slug(None) == "unknown"


# --- _secondary_cv_epoch_regime_masks ---------------------------------------

def _data_with_epoch_source(epoch_src, run_dirs, n=None):
    n = n or len(epoch_src)
    return Data(
        prod_dir=Path("."), out_dir=Path("."),
        cv=np.zeros(n), cv2=np.zeros(n), rg_A=np.zeros(n),
        window=np.zeros(n, dtype=np.int32), replica=np.zeros(n, dtype=np.int32),
        step=np.arange(n, dtype=np.int64), u_nk=np.zeros((n, 1)),
        centers=np.zeros(1), k_kcal=np.zeros(1), beta=1.0, temp=300.0,
        boost_kj=np.zeros(n), potential_kj=None, source="test",
        meta={"_epoch_source": list(epoch_src), "adaptive_epoch_run_dirs": run_dirs},
    )


def test_returns_none_for_single_regime(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "contacts"}}))
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    (e1 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "contacts"}}))

    d = _data_with_epoch_source([0, 0, 1, 1], [str(e0), str(e1)])
    assert _secondary_cv_epoch_regime_masks(d) is None


def test_returns_none_without_epoch_source_metadata():
    d = _data_with_epoch_source([], [])
    d.meta = {}
    assert _secondary_cv_epoch_regime_masks(d) is None


def test_splits_two_regimes_last_epoch_is_dominant(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}))
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    (e1 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "tica-linear"}}))
    final = tmp_path / "final"; final.mkdir()
    (final / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "tica-linear"}}))

    epoch_src = [0, 0, 0, 1, 1, 2, 2]  # epoch_000 x3, epoch_001 x2, final x2
    d = _data_with_epoch_source(epoch_src, [str(e0), str(e1), str(final)])

    regimes = _secondary_cv_epoch_regime_masks(d)
    assert set(regimes) == {"torsion-pca", "tica-linear"}

    torsion_mask, torsion_dominant = regimes["torsion-pca"]
    tica_mask, tica_dominant = regimes["tica-linear"]
    assert not torsion_dominant
    assert tica_dominant
    np.testing.assert_array_equal(torsion_mask, [True, True, True, False, False, False, False])
    np.testing.assert_array_equal(tica_mask, [False, False, False, True, True, True, True])


def test_dominant_regime_resolves_by_mtime_not_list_position(tmp_path):
    """Fix 1: `run_dirs` load order (list position) comes from a lexicographic
    directory-name sort elsewhere in the pipeline, not chronological order --
    a quality-gate extension loop can produce a later-created directory that
    sorts *before* an earlier-created one by name (same bug class already
    fixed in plot_adaptive_diagnostics.py's discover_phases via mtime, per
    test_discover_phases_orders_topups_by_creation_time_not_step_suffix).

    Construct run_dirs where list index 0 is chronologically LAST (highest
    mtime) and list index 1 is chronologically EARLIER but occupies the last
    *list position*. The old `reversed(regime_by_epoch)` logic would pick
    index 1's regime ("torsion-pca") as dominant; the mtime-aware fix must
    pick index 0's regime ("tica-linear") instead.
    """
    now = time.time()

    run_dir_a = tmp_path / "aaa_last_created_first_in_list"
    run_dir_a.mkdir()
    (run_dir_a / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "tica-linear"}}))
    os.utime(run_dir_a, (now, now))  # chronologically LAST

    run_dir_b = tmp_path / "zzz_first_created_last_in_list"
    run_dir_b.mkdir()
    (run_dir_b / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}))
    os.utime(run_dir_b, (now - 1000.0, now - 1000.0))  # chronologically EARLIER

    # run_dirs list order: index 0 = run_dir_a (mtime-last), index 1 = run_dir_b
    # (mtime-earlier, but last list position) -- mirrors the real bug shape.
    epoch_src = [0, 0, 1, 1]
    d = _data_with_epoch_source(epoch_src, [str(run_dir_a), str(run_dir_b)])

    regimes = _secondary_cv_epoch_regime_masks(d)
    assert set(regimes) == {"torsion-pca", "tica-linear"}

    _, tica_dominant = regimes["tica-linear"]
    _, torsion_dominant = regimes["torsion-pca"]
    assert tica_dominant, "mtime-latest run_dir's regime must be dominant"
    assert not torsion_dominant


def test_unresolvable_epoch_folds_into_dominant_with_warning(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()  # no run_manifest.json -> unresolvable
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    (e1 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "tica-linear"}}))
    e2 = tmp_path / "epoch_002"; e2.mkdir()
    (e2 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}))

    epoch_src = [0, 1, 2]
    d = _data_with_epoch_source(epoch_src, [str(e0), str(e1), str(e2)])
    warnings: list = []

    regimes = _secondary_cv_epoch_regime_masks(d, warnings=warnings)

    # epoch_000 (unresolvable) folded into the dominant regime (last epoch = torsion-pca).
    torsion_mask, _ = regimes["torsion-pca"]
    np.testing.assert_array_equal(torsion_mask, [True, False, True])
    assert len(warnings) == 1
    assert "torsion-pca" in warnings[0]


# --- _masked_data ------------------------------------------------------------

def test_masked_data_slices_per_sample_fields_only():
    d = _data_with_epoch_source([0, 1, 0, 1], ["a", "b"])
    d.cv[:] = [1.0, 2.0, 3.0, 4.0]
    mask = np.array([True, False, True, False])

    sub = _masked_data(d, mask)

    np.testing.assert_array_equal(sub.cv, [1.0, 3.0])
    assert sub.centers is d.centers  # K-length field untouched, shared
    assert sub.beta == d.beta


def test_masked_data_applies_meta_override():
    d = _data_with_epoch_source([0], ["a"])
    sub = _masked_data(d, np.array([True]), meta_override={"secondary_cv": "torsion-pca"})
    assert sub.meta == {"secondary_cv": "torsion-pca"}
    assert d.meta.get("secondary_cv") != "torsion-pca"  # original untouched


def test_masked_data_slices_epoch_source_without_meta_override():
    """Fix 3: meta['_epoch_source'] is a per-sample-length list parallel to
    cv/window/etc, but lived inside the meta dict rather than as a Data field
    -- _masked_data must slice it consistently with every other per-sample
    array, not hand back the full-length original list.
    """
    # Distinguishable per-index values (not repeated like [0,1,0,1]) so a
    # broken implementation returning the wrong slice can't accidentally match.
    epoch_src = [10, 11, 12, 13]
    d = _data_with_epoch_source(epoch_src, ["a", "b"])
    d.cv[:] = [100.0, 101.0, 102.0, 103.0]
    mask = np.array([True, False, True, False])

    sub = _masked_data(d, mask)

    np.testing.assert_array_equal(sub.cv, [100.0, 102.0])
    assert sub.meta["_epoch_source"] == [10, 12]
    assert len(sub.meta["_epoch_source"]) == len(sub.cv)
    # adaptive_epoch_run_dirs is indexed by epoch position, not by sample --
    # must NOT be touched/sliced.
    assert sub.meta["adaptive_epoch_run_dirs"] == ["a", "b"]
    # Original untouched.
    assert d.meta["_epoch_source"] == epoch_src


def test_masked_data_slices_epoch_source_with_meta_override():
    """Same as above, but through the meta_override path (used by
    run_secondary_cv_analyses's per-regime call) -- the override dict's own
    _epoch_source must be sliced too, not passed through at full length.
    """
    epoch_src = [10, 11, 12, 13]
    d = _data_with_epoch_source(epoch_src, ["a", "b"])
    d.cv[:] = [100.0, 101.0, 102.0, 103.0]
    mask = np.array([True, False, True, False])
    override = {"secondary_cv": "torsion-pca", "_epoch_source": list(epoch_src)}

    sub = _masked_data(d, mask, meta_override=override)

    np.testing.assert_array_equal(sub.cv, [100.0, 102.0])
    assert sub.meta["_epoch_source"] == [10, 12]
    assert len(sub.meta["_epoch_source"]) == len(sub.cv)
    assert sub.meta["secondary_cv"] == "torsion-pca"
    # The override dict passed in by the caller must not be mutated in place.
    assert override["_epoch_source"] == epoch_src


# --- run_secondary_cv_analyses (integration) --------------------------------

def _full_data(cv, cv2, epoch_src=None, run_dirs=None, k=1):
    n = len(cv)
    return Data(
        prod_dir=Path("."), out_dir=Path("."),
        cv=np.asarray(cv, dtype=np.float64), cv2=np.asarray(cv2, dtype=np.float64),
        rg_A=np.full(n, np.nan),
        window=np.zeros(n, dtype=np.int32), replica=np.zeros(n, dtype=np.int32),
        step=np.arange(n, dtype=np.int64), u_nk=np.zeros((n, k)),
        centers=np.zeros(k), k_kcal=np.zeros(k), beta=0.4, temp=300.0,
        boost_kj=np.zeros(n), potential_kj=None, source="test",
        meta=({} if epoch_src is None else
              {"_epoch_source": epoch_src, "adaptive_epoch_run_dirs": run_dirs}),
    )


def _test_args():
    args = parse_args(["dummy_input", "--bins", "10", "--no-convergence"])
    args.no_poincare_map = True
    return args


def test_single_regime_writes_to_out_directly_no_breakdown(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    cv = rng.normal(0, 1, n)
    cv2 = rng.normal(0, 1, n)
    d = _full_data(cv, cv2)
    logw = np.zeros(n)

    pmf_info, fes_info = run_secondary_cv_analyses(
        d, _test_args(), logw, "umbrella_only", False, 0.6, tmp_path, [], None)

    assert pmf_info["available"]
    assert "regime_breakdown" not in pmf_info
    assert (tmp_path / "cv2_pmf_unbiased.csv").exists()
    assert not list(tmp_path.glob("secondary_cv_regime_*"))


def test_two_regimes_split_into_separate_directories_with_different_data(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}))
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    (e1 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "tica-linear"}}))

    rng = np.random.default_rng(1)
    n0, n1 = 150, 150
    # Deliberately different cv2 distributions per regime so a bug that pools
    # them would be visible as a shifted/blended PMF in the "dominant" output.
    cv2_epoch0 = rng.normal(-5.0, 0.5, n0)
    cv2_epoch1 = rng.normal(5.0, 0.5, n1)
    cv = rng.normal(0, 1, n0 + n1)
    cv2 = np.concatenate([cv2_epoch0, cv2_epoch1])
    epoch_src = [0] * n0 + [1] * n1
    out = tmp_path / "pmf_analysis"

    d = _full_data(cv, cv2, epoch_src=epoch_src, run_dirs=[str(e0), str(e1)])
    logw = np.zeros(n0 + n1)

    pmf_info, fes_info = run_secondary_cv_analyses(
        d, _test_args(), logw, "umbrella_only", False, 0.6, out, [], None)

    assert pmf_info["available"]
    breakdown = pmf_info["regime_breakdown"]
    assert set(breakdown) == {"torsion-pca", "tica-linear"}
    assert breakdown["tica-linear"]["is_dominant"]
    assert not breakdown["torsion-pca"]["is_dominant"]
    assert breakdown["torsion-pca"]["n_samples"] == n0
    assert breakdown["tica-linear"]["n_samples"] == n1

    # Dominant (tica-linear, last epoch) writes to the existing unmodified paths.
    assert (out / "cv2_pmf_unbiased.csv").exists()
    dominant_pmf = np.genfromtxt(out / "cv2_pmf_unbiased.csv", delimiter=",", names=True)
    assert np.nanmean(dominant_pmf["cv2_A"]) > 0  # centered near +5, not blended with -5

    # Minority regime (torsion-pca, epoch_000) gets its own separate directory.
    minority_dir = out / "secondary_cv_regime_torsion-pca"
    assert (minority_dir / "cv2_pmf_unbiased.csv").exists()
    minority_pmf = np.genfromtxt(minority_dir / "cv2_pmf_unbiased.csv", delimiter=",", names=True)
    assert np.nanmean(minority_pmf["cv2_A"]) < 0  # centered near -5, not blended with +5


def test_regime_meta_preserves_dict_shape_with_regions(tmp_path, monkeypatch):
    """Fix 2: when d.meta['secondary_cv'] is a dict (carrying region/basin
    reference-line annotations for the PMF/2D-FES plots), the per-regime
    meta built inside run_secondary_cv_analyses must preserve that dict
    shape -- only overriding its 'mode' field -- rather than collapsing it
    to a bare regime-name string, which would silently drop every region
    annotation from the per-regime plots.

    Spies on analyze_secondary_cv_pmf (called once per regime) to capture
    the actual d_regime.meta['secondary_cv'] each regime's analysis receives,
    since run_secondary_cv_analyses doesn't return the per-regime Data itself.
    """
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}))
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    (e1 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "tica-linear"}}))

    rng = np.random.default_rng(4)
    n0, n1 = 60, 60
    cv2 = np.concatenate([rng.normal(-5.0, 0.5, n0), rng.normal(5.0, 0.5, n1)])
    cv = rng.normal(0, 1, n0 + n1)
    epoch_src = [0] * n0 + [1] * n1
    out = tmp_path / "pmf_analysis"

    d = _full_data(cv, cv2, epoch_src=epoch_src, run_dirs=[str(e0), str(e1)])
    original_regions = [{"value": -5.0, "label": "unfolded"}, {"value": 5.0, "label": "folded"}]
    d.meta["secondary_cv"] = {"mode": "seed-mode-should-be-overridden", "label": "", "regions": original_regions}
    logw = np.zeros(n0 + n1)

    import gareus.mbar_analysis.pmf as pmfmod

    captured_secondary_cv = []
    real_analyze = pmfmod.analyze_secondary_cv_pmf

    def _spy(d_regime, *a, **kw):
        captured_secondary_cv.append(d_regime.meta.get("secondary_cv"))
        return real_analyze(d_regime, *a, **kw)

    monkeypatch.setattr(pmfmod, "analyze_secondary_cv_pmf", _spy)

    run_secondary_cv_analyses(d, _test_args(), logw, "umbrella_only", False, 0.6, out, [], None)

    assert len(captured_secondary_cv) == 2
    for sec_cv in captured_secondary_cv:
        assert isinstance(sec_cv, dict), "secondary_cv must stay a dict, not collapse to a bare string"
        assert sec_cv["mode"] in {"torsion-pca", "tica-linear"}
        # Region annotations survive unchanged into the per-regime metadata.
        assert _secondary_cv_regions({"secondary_cv": sec_cv}) == original_regions
    # The two regimes got distinct 'mode' overrides, not a shared/stale value.
    assert {sec_cv["mode"] for sec_cv in captured_secondary_cv} == {"torsion-pca", "tica-linear"}


def test_regime_breakdown_is_json_serializable_like_the_real_pmf_summary(tmp_path):
    """Regression: the dominant regime's own info dict must not end up nested
    inside its own regime_breakdown (a real circular reference caught by an
    actual end-to-end run against chignolin_5 -- json.dumps raised
    ValueError: Circular reference detected). analyze() serializes these
    exact dicts via wjson(out/'pmf_summary.json', s), so exercise that here
    instead of only checking file/content assertions that don't catch it.
    """
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}))
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    (e1 / "run_manifest.json").write_text(json.dumps({"resolved_args": {"secondary_cv": "tica-linear"}}))

    rng = np.random.default_rng(2)
    n0, n1 = 100, 100
    cv2 = np.concatenate([rng.normal(-5.0, 0.5, n0), rng.normal(5.0, 0.5, n1)])
    cv = rng.normal(0, 1, n0 + n1)
    epoch_src = [0] * n0 + [1] * n1
    out = tmp_path / "pmf_analysis"

    d = _full_data(cv, cv2, epoch_src=epoch_src, run_dirs=[str(e0), str(e1)])
    logw = np.zeros(n0 + n1)

    pmf_info, fes_info = run_secondary_cv_analyses(
        d, _test_args(), logw, "umbrella_only", False, 0.6, out, [], None)

    s = {"secondary_cv_pmf": pmf_info, "cv1_cv2_2d_fes": fes_info}
    json.dumps(s, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
