"""Regression tests: a post-pull window drop (--us-auto-drop-bad-windows) left the
phase's epoch_window_map.csv stale, so MBAR attributed samples to the wrong state.

Root cause (chignolin_6, docs/chignolin_6_low_ess_root_cause.md): the phase's
epoch_window_map.csv is written by the registry BEFORE the sub-run starts, as an
identity map (epoch_window i -> the i-th active state).  When the post-pull quality
gate then dropped windows, gareus/production.py's drop_bad_us_windows_and_rebuild
renumbered every window-indexed array around the survivors (0..N-1) but never
rewrote the map -- so local window 20 still claimed to be state 20 when it was
physically running state 21's Hamiltonian.  On the real run that mis-attributed
~1.82M of 12.1M samples (15%), fabricated up to 2,012 kT of self-bias in state 21,
made state 20 a phantom duplicate of 21, and pooled three different Hamiltonians
into state 22's slot -- base MBAR ESS 0.74%, health verdict FAIL.

The two fixture specs below are the real chignolin_6 phases, verbatim:
  final/baseline    -- 27 identity rows, dropped [20, 22, 23] -> 24 rows with
                       epoch_window 20->state 21, 21->24, 22->25, 23->26.
  epoch_001/baseline -- 24 identity rows, dropped [20] -> 23 rows with
                       epoch_window 20->state 21, 21->22, 22->23.
"""
from __future__ import annotations

import csv
import functools
from pathlib import Path

import numpy as np
import pytest

from gareus.production import (
    drop_bad_us_windows_and_rebuild,
    rewrite_epoch_window_map_after_drop,
)


# The real chignolin_6 maps carry exactly these four columns (no primary_k /
# secondary_k -- write_epoch_window_map has emitted those only since a later
# revision), so the 4-column shape is the one that must keep working.
REAL_RUN_FIELDS = ["epoch_window", "state_id", "primary_center", "secondary_center"]
FULL_FIELDS = ["epoch_window", "state_id", "primary_center", "primary_k", "secondary_center", "secondary_k"]


def _write_map(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _identity_map(path: Path, n: int, fields: list[str] = REAL_RUN_FIELDS) -> None:
    """Mirror registry.write_epoch_window_map's pre-drop identity output."""
    rows = []
    for i in range(n):
        row = {
            "epoch_window": i,
            "state_id": i,
            "primary_center": 0.05 * i,
            "secondary_center": -1.5 + 0.1 * i,
        }
        if "primary_k" in fields:
            row["primary_k"] = 200.0
            row["secondary_k"] = 148.5
        rows.append(row)
    _write_map(path, rows, fields)


def _read_map(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _pairs(path: Path) -> list[tuple[int, int]]:
    return [(int(r["epoch_window"]), int(r["state_id"])) for r in _read_map(path)]


def test_final_baseline_fixture_27_rows_drop_20_22_23(tmp_path):
    """Real chignolin_6 final/baseline: 27 identity rows, dropped [20, 22, 23]."""
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 27)

    summary = rewrite_epoch_window_map_after_drop(tmp_path, [20, 22, 23], 27)

    assert summary is not None
    assert summary["status"] == "rewritten"
    assert summary["n_windows_before"] == 27
    assert summary["n_windows_after"] == 24
    assert summary["dropped_state_ids"] == [20, 22, 23]

    pairs = _pairs(path)
    assert len(pairs) == 24
    # Local indices are dense 0..23 again ...
    assert [w for w, _ in pairs] == list(range(24))
    # ... and the four tail windows now point at the states physically running there.
    assert pairs[20:] == [(20, 21), (21, 24), (22, 25), (23, 26)]
    # Everything below the first dropped index is untouched.
    assert pairs[:20] == [(i, i) for i in range(20)]


def test_epoch_001_baseline_fixture_24_rows_drop_20(tmp_path):
    """Real chignolin_6 epoch_001/baseline: 24 identity rows, dropped [20]."""
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 24)

    summary = rewrite_epoch_window_map_after_drop(tmp_path, [20], 24)

    assert summary["status"] == "rewritten"
    pairs = _pairs(path)
    assert len(pairs) == 23
    assert pairs[20:] == [(20, 21), (21, 22), (22, 23)]


def test_rewrite_preserves_each_surviving_rows_own_centers_and_k(tmp_path):
    """State id AND restraint params must travel with the row, not with the index.

    This is the whole point: MBAR reconstructs each sample's bias from the params
    on the map row its local window index lands on.  Renumbering must move the
    index, never the physics.
    """
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 6, FULL_FIELDS)
    before = {int(r["state_id"]): dict(r) for r in _read_map(path)}

    rewrite_epoch_window_map_after_drop(tmp_path, [2, 3], 6)

    rows = _read_map(path)
    assert list(rows[0].keys()) == FULL_FIELDS  # column set preserved verbatim
    for row in rows:
        original = before[int(row["state_id"])]
        for field in ("primary_center", "primary_k", "secondary_center", "secondary_k"):
            assert row[field] == original[field]
    assert _pairs(path) == [(0, 0), (1, 1), (2, 4), (3, 5)]


def test_rewrite_handles_non_identity_topup_style_map(tmp_path):
    """A topup phase's map is a state subset (epoch_window dense, state_id not)."""
    path = tmp_path / "epoch_window_map.csv"
    _write_map(
        path,
        [
            {"epoch_window": 0, "state_id": 20, "primary_center": 0.114, "secondary_center": -1.628},
            {"epoch_window": 1, "state_id": 21, "primary_center": 0.114, "secondary_center": -1.438},
            {"epoch_window": 2, "state_id": 22, "primary_center": 0.114, "secondary_center": 1.407},
        ],
        REAL_RUN_FIELDS,
    )

    summary = rewrite_epoch_window_map_after_drop(tmp_path, [0], 3)

    assert summary["dropped_state_ids"] == [20]
    assert _pairs(path) == [(0, 21), (1, 22)]


def test_missing_map_is_a_silent_no_op(tmp_path):
    """Non-adaptive-production runs have no epoch_window_map.csv; that is not an error."""
    summary = rewrite_epoch_window_map_after_drop(tmp_path, [1], 4)
    assert summary is None
    assert not (tmp_path / "epoch_window_map.csv").exists()


def test_row_count_mismatch_leaves_the_file_untouched(tmp_path):
    """If the map does not cover the pre-drop window set, positional identity between
    local window index and map row is not established -- refuse rather than guess."""
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 5)
    original = path.read_bytes()

    summary = rewrite_epoch_window_map_after_drop(tmp_path, [1], 27)

    assert summary is not None
    assert summary["status"] == "skipped_row_count_mismatch"
    assert path.read_bytes() == original


def test_non_dense_epoch_window_column_leaves_the_file_untouched(tmp_path):
    """epoch_window must already be 0..N-1 for positional renumbering to be valid."""
    path = tmp_path / "epoch_window_map.csv"
    _write_map(
        path,
        [
            {"epoch_window": 0, "state_id": 0, "primary_center": 0.0, "secondary_center": 0.0},
            {"epoch_window": 2, "state_id": 1, "primary_center": 0.0, "secondary_center": 0.1},
        ],
        REAL_RUN_FIELDS,
    )
    original = path.read_bytes()

    summary = rewrite_epoch_window_map_after_drop(tmp_path, [0], 2)

    assert summary["status"] == "skipped_non_dense_epoch_window"
    assert path.read_bytes() == original


def test_out_of_range_dropped_index_leaves_the_file_untouched(tmp_path):
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 4)
    original = path.read_bytes()

    summary = rewrite_epoch_window_map_after_drop(tmp_path, [7], 4)

    assert summary["status"] == "skipped_dropped_index_out_of_range"
    assert path.read_bytes() == original


def test_dropping_every_window_writes_a_header_only_map(tmp_path):
    """us_auto_drop_max_fraction=1.0 makes an empty survivor set reachable; the
    writer must not IndexError on rows[0] when nothing survives."""
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 2)

    summary = rewrite_epoch_window_map_after_drop(tmp_path, [0, 1], 2)

    assert summary["n_windows_after"] == 0
    assert _read_map(path) == []
    assert path.read_text().splitlines()[0].split(",") == REAL_RUN_FIELDS


def test_warning_names_the_dropped_state_ids(tmp_path, capsys):
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 27)

    rewrite_epoch_window_map_after_drop(tmp_path, [20, 22, 23], 27)

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "epoch_window_map.csv" in out
    # State ids, not just local indices - the local indices happen to coincide here,
    # so also assert the states are labelled as such.
    assert "state_id" in out or "state ids" in out.lower()
    assert "20" in out and "22" in out and "23" in out


# ---------------------------------------------------------------------------
# Integration: the rewrite must actually be wired into the drop path, and the
# post-drop metadata must stop contradicting itself (A11).
# ---------------------------------------------------------------------------


def _drop_kwargs(out_dir: Path, nwin: int, dropped: list[int], args) -> dict:
    centers_a = np.linspace(0.0, 0.2, nwin)
    secondary = np.linspace(-1.5, 1.5, nwin)
    normalized_rows = [
        {"window": i, "primary_center": float(centers_a[i]), "secondary_center": float(secondary[i])}
        for i in range(nwin)
    ]
    return dict(
        out_dir=out_dir,
        dropped_window_indices=dropped,
        centers_a=centers_a,
        k_list=[200.0] * nwin,
        centers_nm=centers_a.copy(),
        ks_kj_nm2=np.full(nwin, 836.8),
        secondary_cv_centers=secondary,
        secondary_cv_k_kcal_list=[14.8] * nwin,
        secondary_cv_ks_kj=np.full(nwin, 62.1),
        window_start_positions=[None] * nwin,
        window_start_velocities=[None] * nwin,
        secondary_cv_metadata={"enabled": True},
        window_metadata={"n_windows": nwin, "normalized_rows": normalized_rows},
        args=args,
    )


class _Args:
    temperature_k = 300.0
    seq = "AAA"


def test_drop_bad_us_windows_and_rebuild_rewrites_the_map(tmp_path):
    _identity_map(tmp_path / "epoch_window_map.csv", 27)

    result = drop_bad_us_windows_and_rebuild(**_drop_kwargs(tmp_path, 27, [20, 22, 23], _Args()))

    assert len(result["centers_a"]) == 24
    assert _pairs(tmp_path / "epoch_window_map.csv")[20:] == [(20, 21), (21, 24), (22, 25), (23, 26)]
    rewrite = result["window_metadata"]["epoch_window_map_rewrite"]
    assert rewrite["status"] == "rewritten"
    assert rewrite["dropped_state_ids"] == [20, 22, 23]


def test_drop_bad_us_windows_and_rebuild_updates_n_windows_and_rows(tmp_path):
    """A11: window_metadata must not report the PRE-drop window count alongside
    POST-drop arrays, and normalized_rows must be renumbered the same way the
    pre-pull reachability filter already does it (seeding.py's
    filter_explicit_2d_windows_by_seed_reachability)."""
    _identity_map(tmp_path / "epoch_window_map.csv", 27)

    result = drop_bad_us_windows_and_rebuild(**_drop_kwargs(tmp_path, 27, [20, 22, 23], _Args()))

    wm = result["window_metadata"]
    assert wm["n_windows"] == 24
    rows = wm["normalized_rows"]
    assert len(rows) == 24
    assert [r["window"] for r in rows] == list(range(24))
    # Row identity travels with the row: local 20 now carries old window 21's center.
    assert rows[20]["secondary_center"] == pytest.approx(-1.5 + 21 * (3.0 / 26))
    # The window table summary the phase reports must agree with the arrays.
    assert int(wm["explicit_window_table"]["n_windows"]) == 24


def test_drop_with_no_map_present_still_succeeds(tmp_path):
    result = drop_bad_us_windows_and_rebuild(**_drop_kwargs(tmp_path, 6, [2], _Args()))
    assert len(result["centers_a"]) == 5
    assert result["window_metadata"].get("epoch_window_map_rewrite") is None


def test_drop_rewrites_per_window_provenance_from_the_survivors(tmp_path):
    """The canonical window table's provenance must follow the surviving windows.

    explicit_window_analysis_rows fills each post-drop window i's
    window_type/lifecycle/source_row from normalized_rows[i], so leaving those rows
    at their pre-drop indices hands every window at or past the first dropped one a
    different window's provenance.  Real chignolin_6 final/baseline: post-drop
    window 20 reported source_row 22 (the dropped window's), not 23.
    """
    _identity_map(tmp_path / "epoch_window_map.csv", 27)
    kwargs = _drop_kwargs(tmp_path, 27, [20, 22, 23], _Args())
    # The real run carries these under secondary_cv metadata, which is the copy
    # explicit_window_analysis_rows prefers.
    rows = [
        {"window": i, "source_row": i + 2, "window_type": f"state_{i}", "source_csv": "baseline_windows.csv"}
        for i in range(27)
    ]
    kwargs["secondary_cv_metadata"] = {"enabled": True, "explicit_2d_windows": True, "normalized_rows": rows}
    kwargs["window_metadata"] = {"n_windows": 27}

    result = drop_bad_us_windows_and_rebuild(**kwargs)

    kept = result["secondary_cv_metadata"]["normalized_rows"]
    assert len(kept) == 24
    assert [r["window"] for r in kept] == list(range(24))
    # local 20 -> original window 21 -> source_row 23
    assert kept[20]["source_row"] == 23
    assert [r["source_row"] for r in kept[20:]] == [23, 26, 27, 28]

    table = _read_map(tmp_path / "umbrella_explicit_windows.csv")
    assert len(table) == 24
    assert [int(r["source_row"]) for r in table[20:]] == [23, 26, 27, 28]
    assert table[20]["window_type"] == "state_21"
    # ... and the caller's original dicts were not mutated in place.
    assert len(rows) == 27 and rows[20]["window"] == 20


# ---------------------------------------------------------------------------
# The SECOND drop path: the PRE-pull seed-reachability filter.
#
# chignolin_6's epoch_001/topup_004_75786000 (map 3 rows -> 2 real windows) and
# final/topup_003_478000 (2 -> 1) were shifted by
# filter_explicit_2d_windows_by_seed_reachability, not by the post-pull auto-drop:
# neither phase has a dropped_post_pull_bad_windows record at all, so the
# post-pull rewrite above could never have covered them.  Ground truth from
# docs/chignolin_6_low_ess_root_cause.md's shift table:
#   epoch_001/topup_004: local 0 -> state 21, local 1 -> state 22 (map said 20, 21)
#   final/topup_003:     local 0 -> state 21                      (map said 20)
#
# These tests drive rewrite_epoch_window_map_after_unreachable_filter with the
# metadata dicts EXACTLY as the filter emits them (a list of per-drop dicts under
# "dropped_unreachable_windows", keyed "window"), so the extraction the call site
# depends on is what is under test, not a key invented by the test.  The call site
# wiring itself (run_gareus's explicit-2D window-loading block) needs a live
# OpenMM stack plus a GENPEPT seed library to reach and is not unit-testable, the
# same accepted seam as relax_to_window's crash recovery.
# ---------------------------------------------------------------------------

from gareus.production import (  # noqa: E402  (grouped with the fix under test)
    repair_epoch_window_map_from_surviving_windows,
    rewrite_epoch_window_map_after_unreachable_filter,
)

LEDGER = "epoch_window_map_rewrites.json"


def _filter_metadata(dropped_windows: list[int]) -> dict:
    """The literal shape filter_explicit_2d_windows_by_seed_reachability emits."""
    return {
        "enabled": True,
        "explicit_2d_windows": True,
        "dropped_unreachable_windows": [
            {
                "window": int(i),
                "primary_center": 0.114,
                "secondary_center": -1.6282,
                "nearest_seed_primary_cv": 0.05,
                "nearest_seed_secondary_cv": 0.4,
                "primary_score": 2.4,
                "secondary_score": 0.3,
            }
            for i in dropped_windows
        ],
    }


def _subset_map(path: Path, state_ids: list[int]) -> None:
    """Mirror write_state_subset_window_csv's map output for a topup segment."""
    _write_map(
        path,
        [
            {
                "epoch_window": local,
                "state_id": sid,
                "primary_center": 0.114,
                "secondary_center": -1.6282 + 0.19 * (sid - state_ids[0]),
            }
            for local, sid in enumerate(state_ids)
        ],
        REAL_RUN_FIELDS,
    )


def _ledger(tmp_path: Path) -> list[dict]:
    import json

    path = tmp_path / LEDGER
    if not path.exists():
        return []
    return json.loads(path.read_text())["applied"]


def _read_epoch_window_map_rows_via_production(phase: Path):
    """`(fieldnames, rows)` through production's own map reader."""
    from gareus.production import _read_epoch_window_map_rows

    return _read_epoch_window_map_rows(phase / "epoch_window_map.csv")


def _reorder_ledger(tmp_path: Path) -> list[dict]:
    """The ledger file's REORDER list, through the reader production itself uses.

    Read with the real function rather than by indexing the JSON, so a test can
    never agree with a payload the shipped reader would reject (the reader
    tolerates a missing/damaged file by design, and that tolerance is exactly
    what a hand-rolled `json.loads(...)["reorders"]` here would hide).
    """
    from gareus.production import _read_epoch_window_map_reorder_ledger

    return _read_epoch_window_map_reorder_ledger(tmp_path)


def test_unreachable_filter_fixes_epoch_001_topup_004(tmp_path):
    """Real chignolin_6 epoch_001/topup_004: 3 map rows (states 20, 21, 22), the
    window targeting state 20 was pre-pull unreachable -> local 0 is state 21."""
    path = tmp_path / "epoch_window_map.csv"
    _subset_map(path, [20, 21, 22])

    summary = rewrite_epoch_window_map_after_unreachable_filter(
        tmp_path,
        _filter_metadata([0]),
        {"n_windows": 2, "dropped_unreachable_windows": _filter_metadata([0])["dropped_unreachable_windows"]},
        3,
    )

    assert summary["status"] == "rewritten"
    assert summary["source"] == "pre_pull_seed_reachability_filter"
    assert summary["dropped_state_ids"] == [20]
    assert _pairs(path) == [(0, 21), (1, 22)]


def test_unreachable_filter_fixes_final_topup_003(tmp_path):
    """Real chignolin_6 final/topup_003: 2 map rows (states 20, 21) -> local 0 is 21."""
    path = tmp_path / "epoch_window_map.csv"
    _subset_map(path, [20, 21])

    summary = rewrite_epoch_window_map_after_unreachable_filter(
        tmp_path, _filter_metadata([0]), {"dropped_unreachable_windows": _filter_metadata([0])["dropped_unreachable_windows"]}, 2
    )

    assert summary["status"] == "rewritten"
    assert _pairs(path) == [(0, 21)]


def test_unreachable_filter_falls_back_to_secondary_cv_metadata(tmp_path):
    """The filter writes the drop record onto both metadata dicts; either alone
    must be enough (window_metadata is rebuilt/overwritten in several places)."""
    path = tmp_path / "epoch_window_map.csv"
    _subset_map(path, [20, 21, 22])

    summary = rewrite_epoch_window_map_after_unreachable_filter(
        tmp_path, _filter_metadata([0, 2]), {"n_windows": 1}, 3
    )

    assert summary["status"] == "rewritten"
    assert summary["dropped_state_ids"] == [20, 22]
    assert _pairs(path) == [(0, 21)]


def test_unreachable_filter_does_nothing_when_nothing_was_dropped(tmp_path):
    """The overwhelmingly common case: the filter keeps every window and does not
    put a drop record on the metadata at all, so the map must not be touched."""
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 5)
    original = path.read_bytes()

    summary = rewrite_epoch_window_map_after_unreachable_filter(
        tmp_path, {"enabled": True}, {"n_windows": 5}, 5
    )

    assert summary is None
    assert path.read_bytes() == original
    assert not (tmp_path / LEDGER).exists()


# ---------------------------------------------------------------------------
# Idempotence: two drop paths, one map.
#
# Both paths can fire in the SAME phase (pre-pull reachability filter, then
# post-pull auto-drop on its survivors), in which case the second one legitimately
# receives indices in the ALREADY-COMPACTED space and applying it is correct.
# What must be impossible is applying the same drop set twice.
# ---------------------------------------------------------------------------


def test_both_drop_paths_in_one_phase_compose(tmp_path):
    """Pre-pull filter then post-pull auto-drop, on the same phase's map.

    Map starts as states (20, 21, 22).  The filter drops local 0 (state 20),
    leaving (21, 22) renumbered 0, 1.  The auto-drop then flags local 0 -- which
    is now state 21 -- in the compacted index space, leaving state 22 alone at
    local 0.  Both rewrites must apply, and the second must NOT be mistaken for a
    replay of the first even though both were handed the drop set [0].
    """
    path = tmp_path / "epoch_window_map.csv"
    _subset_map(path, [20, 21, 22])

    first = rewrite_epoch_window_map_after_unreachable_filter(
        tmp_path, _filter_metadata([0]), {}, 3
    )
    assert first["status"] == "rewritten"
    assert _pairs(path) == [(0, 21), (1, 22)]

    second = rewrite_epoch_window_map_after_drop(tmp_path, [0], 2, source="post_pull_auto_drop")

    assert second["status"] == "rewritten"
    assert second["dropped_state_ids"] == [21]
    assert _pairs(path) == [(0, 22)]

    ledger = _ledger(tmp_path)
    assert [e["source"] for e in ledger] == [
        "pre_pull_seed_reachability_filter",
        "post_pull_auto_drop",
    ]
    # The two entries are distinguishable by the index space they were expressed
    # in, which is what makes "same set twice" detectable at all.
    assert [e["n_windows_before"] for e in ledger] == [3, 2]


def test_the_same_drop_set_is_never_applied_twice(tmp_path):
    """The double-rewrite hazard, head on: the identical (n_windows_before,
    dropped) key presented twice must leave the map exactly as the first pass
    wrote it.  Without the ledger the second pass would compact an already
    compacted map -- dropping the wrong three states and shifting the rest again."""
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 27)

    first = rewrite_epoch_window_map_after_drop(tmp_path, [20, 22, 23], 27)
    assert first["status"] == "rewritten"
    after_first = path.read_bytes()

    second = rewrite_epoch_window_map_after_drop(tmp_path, [20, 22, 23], 27)

    assert second["status"] == "skipped_already_applied"
    assert second["applied_by"] == "post_pull_auto_drop"
    assert path.read_bytes() == after_first
    assert _pairs(path)[20:] == [(20, 21), (21, 24), (22, 25), (23, 26)]
    assert len(_ledger(tmp_path)) == 1


def test_a_different_owner_replaying_the_same_set_is_also_refused(tmp_path):
    """Same key, different `source`: the refusal keys off the drop itself, not off
    who is asking.  (Exactly one site owns each drop path today; this makes a
    future second owner degrade to a no-op instead of double-compacting.)"""
    path = tmp_path / "epoch_window_map.csv"
    _subset_map(path, [20, 21, 22])

    rewrite_epoch_window_map_after_unreachable_filter(tmp_path, _filter_metadata([0]), {}, 3)
    after_first = path.read_bytes()

    replay = rewrite_epoch_window_map_after_drop(
        tmp_path, [0], 3, source="some_other_future_caller"
    )

    assert replay["status"] == "skipped_already_applied"
    assert replay["applied_by"] == "pre_pull_seed_reachability_filter"
    assert path.read_bytes() == after_first


def test_the_same_key_IS_applied_again_after_the_map_is_replaced(tmp_path):
    """The load-bearing half of the idempotence check.

    An interrupted phase re-run WITHOUT a usable production checkpoint gets a
    fresh identity map written by the adaptive driver from the (still un-pruned)
    registry, and then legitimately re-runs the same filter with the same drop
    set.  Keying the refusal on (n_windows_before, dropped) alone would call that
    a replay and leave the fresh map stale -- which is the very bug the rewrite
    exists to prevent.  The recorded surviving state_ids are what distinguish
    "the map still holds that rewrite" from "the map was replaced since".
    """
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 24)

    assert rewrite_epoch_window_map_after_drop(tmp_path, [20], 24)["status"] == "rewritten"
    assert _pairs(path)[20:] == [(20, 21), (21, 22), (22, 23)]

    # The driver rewrites the map from the registry on restart: 24 identity rows
    # again, ledger from the previous attempt still on disk.
    _identity_map(path, 24)

    again = rewrite_epoch_window_map_after_drop(tmp_path, [20], 24)

    assert again["status"] == "rewritten"
    assert _pairs(path)[20:] == [(20, 21), (21, 22), (22, 23)]
    assert len(_ledger(tmp_path)) == 2


def test_an_unlogged_prior_rewrite_is_caught_by_the_row_count_guard(tmp_path):
    """The other half of the invariant, kept honest.

    A successful rewrite always removes at least one row, so an already-compacted
    map can never have `n_windows_before` rows.  The ledger catches a LOGGED
    replay; this row-count guard catches an UNLOGGED one (something rewrote the
    map without recording it).  Together they cover both shapes.  This particular
    guard predates the ledger -- it is asserted here so the complementarity
    argument in the docstring stays true if either guard is ever touched.
    """
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 24)
    rewrite_epoch_window_map_after_drop(tmp_path, [20], 24)
    (tmp_path / LEDGER).unlink()
    after_first = path.read_bytes()

    replay = rewrite_epoch_window_map_after_drop(tmp_path, [20], 24)

    assert replay["status"] == "skipped_row_count_mismatch"
    assert path.read_bytes() == after_first


def test_an_empty_drop_set_is_an_explicit_no_op(tmp_path):
    """drop_bad_us_windows_and_rebuild's connectivity restore can hand back every
    flagged window, leaving nothing to drop.  Recording that as an applied rewrite
    would break the strictly-decreasing row count the idempotence invariant rests
    on, so it must be a no-op with no ledger entry."""
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 4)
    original = path.read_bytes()

    summary = rewrite_epoch_window_map_after_drop(tmp_path, [], 4)

    assert summary["status"] == "no_drop"
    assert path.read_bytes() == original
    assert not (tmp_path / LEDGER).exists()


def test_a_skipped_rewrite_records_nothing_in_the_ledger(tmp_path):
    """A refusal must not poison the ledger: the drop is still un-applied, and a
    later correct attempt must not be turned away by a record of the failure."""
    path = tmp_path / "epoch_window_map.csv"
    _identity_map(path, 5)

    assert rewrite_epoch_window_map_after_drop(tmp_path, [1], 27)["status"] == "skipped_row_count_mismatch"
    assert not (tmp_path / LEDGER).exists()

    assert rewrite_epoch_window_map_after_drop(tmp_path, [1], 5)["status"] == "rewritten"
    assert _pairs(path) == [(0, 0), (1, 2), (2, 3), (3, 4)]


# ---------------------------------------------------------------------------
# The durable half: re-deriving the map from the phase's own surviving window
# table, for a phase whose corrected map was overwritten after the drop.
#
# The adaptive driver rewrites epoch_window_map.csv from the registry every time
# it (re)starts a phase, and the registry still holds every dropped state as
# active.  A resumed phase then takes the production-checkpoint fast path and
# never re-runs the pull, hence never re-runs the drop that corrected the map --
# so for any interrupted phase (the normal case on a multi-day run) the drop-time
# rewrite above contributes nothing.  umbrella_explicit_windows.csv is rebuilt on
# every start from the window arrays actually in use (on resume, from the
# checkpoint manifest's own POST-drop windows_A), so it is the surviving window
# set and the map can be re-derived from it with nothing but on-disk state.
# ---------------------------------------------------------------------------


def _survivor_table(path: Path, primary_secondary_pairs: list[tuple[float, float]]) -> None:
    """Mirror write_explicit_window_analysis_files' surviving-window table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["window", "distance_center_A", "primary_center", "secondary_cv_center"],
        )
        writer.writeheader()
        for i, (c1, c2) in enumerate(primary_secondary_pairs):
            writer.writerow({
                "window": i,
                "distance_center_A": c1,
                "primary_center": c1,
                "secondary_cv_center": c2,
            })


def _identity_map_centers(i: int) -> tuple[float, float]:
    """The centers _identity_map writes for row i, so fixtures agree exactly."""
    return 0.05 * i, -1.5 + 0.1 * i


def test_repair_fixes_a_driver_overwritten_map_on_resume(tmp_path):
    """Real chignolin_6 final/baseline, resumed: post-pull drop of states 20, 22,
    23 was applied once, the phase was SIGTERM'd, and the driver rewrote a fresh
    27-row identity map from the un-pruned registry before the resumed segment
    started.  The 24 surviving windows are still on record in the phase's own
    window table, so the map is re-derivable with no drop record at all."""
    _identity_map(tmp_path / "epoch_window_map.csv", 27)
    survivors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 24, 25, 26]
    _survivor_table(
        tmp_path / "umbrella_explicit_windows.csv",
        [_identity_map_centers(i) for i in survivors],
    )

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "rewritten"
    assert summary["n_windows_before"] == 27
    assert summary["n_windows_after"] == 24
    assert summary["dropped_state_ids"] == [20, 22, 23]
    pairs = _pairs(tmp_path / "epoch_window_map.csv")
    assert [w for w, _ in pairs] == list(range(24))
    assert [s for _, s in pairs] == survivors
    assert pairs[20:] == [(20, 21), (21, 24), (22, 25), (23, 26)]
    assert _ledger(tmp_path)[0]["source"] == "surviving_window_table"


def test_repair_is_idempotent(tmp_path):
    """Second call sees a map that already agrees with the window set."""
    _identity_map(tmp_path / "epoch_window_map.csv", 6)
    _survivor_table(
        tmp_path / "umbrella_explicit_windows.csv",
        [_identity_map_centers(i) for i in (0, 1, 3, 4, 5)],
    )

    assert repair_epoch_window_map_from_surviving_windows(tmp_path)["status"] == "rewritten"
    after = (tmp_path / "epoch_window_map.csv").read_bytes()

    second = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert second["status"] == "consistent"
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == after
    assert len(_ledger(tmp_path)) == 1


def test_repair_is_a_no_op_for_a_phase_that_dropped_nothing(tmp_path):
    """The overwhelming majority of phases: map and window table already agree, so
    the repair must not touch the file even though it is called unconditionally."""
    _identity_map(tmp_path / "epoch_window_map.csv", 6)
    _survivor_table(
        tmp_path / "umbrella_explicit_windows.csv",
        [_identity_map_centers(i) for i in range(6)],
    )
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "consistent"
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original


def test_repair_refuses_when_the_centers_do_not_match(tmp_path):
    """A phase whose states were recentered between the interrupted attempt and the
    resume (e.g. the tICA CV2 recentering) has a window table that no longer
    matches its map.  Fail closed -- a wrongly-rewritten map is unrecoverable."""
    _identity_map(tmp_path / "epoch_window_map.csv", 6)
    _survivor_table(
        tmp_path / "umbrella_explicit_windows.csv",
        [(0.05 * i, 9.0 + i) for i in (0, 1, 3)],
    )
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "skipped_centers_do_not_match"
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original


def test_repair_refuses_on_duplicate_window_centers(tmp_path):
    """load_explicit_2d_window_csv deliberately keeps a user's duplicated
    (primary, secondary) rows as separate thermodynamic states.  Matching by
    center then cannot tell which of two identical rows survived, and guessing
    would attach the wrong state_id."""
    _write_map(
        tmp_path / "epoch_window_map.csv",
        [
            {"epoch_window": 0, "state_id": 7, "primary_center": 0.1, "secondary_center": -1.0},
            {"epoch_window": 1, "state_id": 8, "primary_center": 0.1, "secondary_center": -1.0},
            {"epoch_window": 2, "state_id": 9, "primary_center": 0.2, "secondary_center": 1.0},
        ],
        REAL_RUN_FIELDS,
    )
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", [(0.1, -1.0), (0.2, 1.0)])
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "skipped_duplicate_centers"
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original


def test_repair_refuses_when_the_map_covers_fewer_windows_than_exist(tmp_path):
    """No window drop can produce this; samples from the uncovered windows could
    not be attributed at all, so it is a different bug and must not be papered
    over by silently rewriting."""
    _identity_map(tmp_path / "epoch_window_map.csv", 2)
    _survivor_table(
        tmp_path / "umbrella_explicit_windows.csv",
        [_identity_map_centers(i) for i in range(4)],
    )
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "skipped_map_shorter_than_window_set"
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original


def test_repair_returns_none_without_the_artifacts_it_needs(tmp_path):
    """No map (plain non-adaptive run) or no window table: nothing to do, no error."""
    assert repair_epoch_window_map_from_surviving_windows(tmp_path) is None

    _identity_map(tmp_path / "epoch_window_map.csv", 3)
    assert repair_epoch_window_map_from_surviving_windows(tmp_path) is None
    assert not (tmp_path / LEDGER).exists()


def test_repair_and_a_later_drop_time_rewrite_do_not_collide(tmp_path):
    """A repaired map must still accept a genuinely new drop afterwards: the
    repair's ledger entry is keyed by its own index space (the pre-repair row
    count), so it cannot mask a subsequent drop on the compacted map."""
    _identity_map(tmp_path / "epoch_window_map.csv", 6)
    _survivor_table(
        tmp_path / "umbrella_explicit_windows.csv",
        [_identity_map_centers(i) for i in (0, 1, 3, 4, 5)],
    )
    assert repair_epoch_window_map_from_surviving_windows(tmp_path)["status"] == "rewritten"

    later = rewrite_epoch_window_map_after_drop(tmp_path, [0], 5, source="post_pull_auto_drop")

    assert later["status"] == "rewritten"
    assert _pairs(tmp_path / "epoch_window_map.csv") == [(0, 1), (1, 3), (2, 4), (3, 5)]


def test_a_second_owner_recomputing_the_count_from_the_compacted_map_is_refused(tmp_path):
    """The double-rewrite hazard neither count-based guard can see.

    If a second owner of the SAME drop path derives n_windows_before from the map
    as it stands now instead of from the pre-drop window count, both numbers have
    already shifted together: the map has 2 rows, it asks about 2 rows, and its
    drop set [0] is a perfectly plausible key.  The row-count guard passes it, and
    a ledger keyed only on (n_windows_before, dropped) passes it too -- it would
    compact [21, 22] down to [22], silently dropping the state that survived.

    It is refused because the map still holds this same drop path's own previous
    rewrite, and each drop path fires at most once over a given map.  Note this
    request is otherwise INDISTINGUISHABLE from the legitimate second drop in
    test_both_drop_paths_in_one_phase_compose, which differs only in `source` --
    that is why `source` has to be part of the rule.
    """
    path = tmp_path / "epoch_window_map.csv"
    _subset_map(path, [20, 21, 22])

    rewrite_epoch_window_map_after_unreachable_filter(tmp_path, _filter_metadata([0]), {}, 3)
    assert _pairs(path) == [(0, 21), (1, 22)]
    after_first = path.read_bytes()

    replay = rewrite_epoch_window_map_after_drop(
        tmp_path, [0], 2, source="pre_pull_seed_reachability_filter"
    )

    assert replay["status"] == "skipped_already_applied"
    assert path.read_bytes() == after_first
    assert _pairs(path) == [(0, 21), (1, 22)]


# ---------------------------------------------------------------------------
# Composition: the driver's flat-epoch map refresh and this module's rewrite
# ledger, driven together through the real adaptive-production loop.
#
# The ledger's own idempotence rule (_epoch_window_map_rewrite_already_applied)
# is written against a stated precondition: "a phase interrupted and re-run
# WITHOUT a usable production checkpoint gets a fresh identity map written by the
# adaptive driver and then legitimately re-runs the same filter with the same
# drop set".  For a flat (non-scheduled) numbered epoch >= 1 that precondition
# did not hold -- its map is written a whole epoch earlier by the previous loop
# iteration and nothing refreshed it at loop entry -- so a re-pulling attempt 2
# met a map still compacted by attempt 1's drop:
#
#   * the same drop set again  -> skipped_row_count_mismatch (3 rows against the
#     5 windows the drop indices are positions in), NOT the "already applied"
#     branch, and the map is never checked against the window set it now runs;
#   * a different drop set     -> the same refusal, leaving a map that describes
#     neither attempt's window set, and repair_epoch_window_map_from_surviving_windows
#     refuses too (skipped_map_shorter_than_window_set).
#
# These drive run_adaptive_production_auto_loop with a fake run_gareus that
# performs attempt 2's drop exactly as production does, so the ledger's real
# behaviour is what is asserted, not a re-implementation of it.
# ---------------------------------------------------------------------------

_LOOP_MAP_5_STATE_COMPACTED = (
    "epoch_window,state_id,primary_center,primary_k,secondary_center,secondary_k\n"
    "0,0,0.0,200.0,-1.4379,148.46\n"
    "1,2,0.0654,200.0,0.1264,113.09\n"
    "2,4,0.1141,200.0,-1.6282,89.88\n"
)


class _StopAfterDrop(Exception):
    """Ends the driver drive once attempt 2's drop has been applied."""


def _five_state_resumed_campaign(tmp_path):
    """The on-disk state of a campaign whose flat epoch 1 died after its drop.

    Registry holds 5 active states; ``windows_epoch_001.csv`` is the un-pruned
    table attempt 2 re-pulls from; ``epoch_001/`` holds attempt 1's compacted
    3-row map and the ledger entry that compaction wrote.  No production
    checkpoint, so attempt 2 starts fresh -- the case where the map on disk is a
    stale plan rather than a record.
    """
    import json

    from gareus.adaptive_production import WindowStateRegistry
    from gareus.cli import parse_args

    out = tmp_path / "run"
    adaptive = out / "adaptive_production"
    adaptive.mkdir(parents=True)

    registry = WindowStateRegistry()
    registry.add_state(0.0, 200.0, -1.4379, 148.46, epoch=0, source="seed")
    registry.add_state(0.02, 200.0, -1.0, 148.46, epoch=0, source="seed")
    registry.add_state(0.0654, 200.0, 0.1264, 113.09, epoch=0, source="seed")
    registry.add_state(0.09, 200.0, -0.5, 120.0, epoch=0, source="seed")
    registry.add_state(0.1141, 200.0, -1.6282, 89.88, epoch=0, source="seed")
    registry.save(adaptive)
    registry.write_active_window_csv(adaptive / "windows_epoch_001.csv")
    (adaptive / "adaptive_production_driver_summary.json").write_text(
        json.dumps({
            "schema_version": "adaptive_production_driver_summary_v1",
            "epochs_completed": 1,
            "epoch_summaries": [{"epoch": 0}],
        }),
        encoding="utf-8",
    )

    epoch_dir = adaptive / "epoch_001"
    epoch_dir.mkdir()
    (epoch_dir / "epoch_window_map.csv").write_text(_LOOP_MAP_5_STATE_COMPACTED, encoding="utf-8")
    (epoch_dir / LEDGER).write_text(
        json.dumps({
            "schema": "gareus_epoch_window_map_rewrite_ledger_v1",
            "applied": [{
                "status": "rewritten",
                "source": "post_pull_auto_drop",
                "n_windows_before": 5,
                "n_windows_after": 3,
                "dropped_window_indices": [1, 3],
                "dropped_state_ids": [1, 3],
                "surviving_state_ids": [0, 2, 4],
            }],
        }),
        encoding="utf-8",
    )

    args = parse_args(["--seq", "GYDPETGTWG", "--out", str(out)])
    args.adaptive_production_resume = True
    args.adaptive_production_allocation_scheduler = False
    args.adaptive_production_epochs = 2
    args.adaptive_production_epoch_steps = 1000
    args.adaptive_production_global_shared_gamd = False
    return args, out, epoch_dir


def _drive_with_attempt_2_dropping(monkeypatch, args, out, dropped):
    """Drive the real loop; attempt 2's ``run_gareus`` drops `dropped` and stops.

    The fake stands in for the post-pull auto-drop only: it calls the same
    ``rewrite_epoch_window_map_after_drop(out_dir, dropped, 5)`` that
    ``drop_bad_us_windows_and_rebuild`` calls, with the pre-drop window count the
    driver's own window table carries, and returns its verdict.
    """
    import json

    import gareus.adaptive_production as ap
    import gareus.production as prod

    seen = {}

    def _fake_run_gareus(_args, run_dir, *rest, **kw):
        seen["result"] = rewrite_epoch_window_map_after_drop(Path(run_dir), dropped, 5)
        seen["pairs"] = _pairs(Path(run_dir) / "epoch_window_map.csv")
        seen["ledger"] = json.loads((Path(run_dir) / LEDGER).read_text())["applied"]
        raise _StopAfterDrop()

    monkeypatch.setattr(prod, "run_gareus", _fake_run_gareus)
    with pytest.raises(_StopAfterDrop):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    return seen


def test_a_re_pulled_flat_epoch_can_apply_a_different_drop_set(tmp_path, monkeypatch):
    """The damaging case: attempt 2 pulls window 3 successfully and drops only 1.

    The phase then runs 4 windows (states 0, 2, 3, 4).  With attempt 1's 3-row
    map still on disk the rewrite refuses, the map covers 3 of the 4 windows the
    phase actually runs, and nothing downstream can repair it -- exactly the
    unrepairable combination.
    """
    args, out, epoch_dir = _five_state_resumed_campaign(tmp_path)

    seen = _drive_with_attempt_2_dropping(monkeypatch, args, out, [1])

    assert seen["result"]["status"] == "rewritten"
    assert seen["pairs"] == [(0, 0), (1, 2), (2, 3), (3, 4)]
    assert seen["result"]["dropped_state_ids"] == [1]


def test_the_same_drop_set_is_applied_again_after_the_driver_refreshed_the_map(tmp_path, monkeypatch):
    """The precondition the ledger's idempotence rule is written against.

    Attempt 2 re-pulls and flags the same two windows.  That is a legitimate
    re-application, not a replay: the map it is applied to is the fresh pre-drop
    one, so the ledger's "does the map still hold that rewrite" conjunct is
    false and the drop goes through.  The ledger stays append-only -- attempt 1's
    entry is still there afterwards, with attempt 2's appended after it.
    """
    args, out, epoch_dir = _five_state_resumed_campaign(tmp_path)

    seen = _drive_with_attempt_2_dropping(monkeypatch, args, out, [1, 3])

    assert seen["result"]["status"] == "rewritten"
    assert seen["pairs"] == [(0, 0), (1, 2), (2, 4)]
    assert len(seen["ledger"]) == 2, "the ledger must be appended to, never rewritten"
    assert [e["dropped_window_indices"] for e in seen["ledger"]] == [[1, 3], [1, 3]]
    assert seen["ledger"][0]["n_windows_before"] == 5


def test_a_true_replay_within_attempt_2_is_still_refused(tmp_path, monkeypatch):
    """The refresh must not weaken the idempotence rule it restores the input to.

    Two calls from the same drop path inside one attempt: the second is a replay
    against a map that still holds the first one's rewrite, and must be refused
    with the map untouched.
    """
    args, out, epoch_dir = _five_state_resumed_campaign(tmp_path)

    seen = _drive_with_attempt_2_dropping(monkeypatch, args, out, [1, 3])
    assert seen["result"]["status"] == "rewritten"
    before = (epoch_dir / "epoch_window_map.csv").read_bytes()

    replay = rewrite_epoch_window_map_after_drop(epoch_dir, [1, 3], 5)

    assert replay["status"] == "skipped_already_applied"
    assert (epoch_dir / "epoch_window_map.csv").read_bytes() == before


# ---------------------------------------------------------------------------
# Review round 4 -- a torn checkpoint does not un-write the Parquet rows
#
# Round 3's flat-epoch loop-entry refresh (above) closed the "attempt 1 compacted
# this map and nothing refreshes it" hole, but it keyed the no-clobber decision on
# `production_checkpoint_available()`, i.e. on whether a checkpoint survived.  The
# rule that actually protects a record is about samples, not checkpoints:
#
#   attempt 1 pulls -> --us-auto-drop-bad-windows drops D1 ->
#   rewrite_epoch_window_map_after_drop compacts the map ->
#   production writes samples into samples/seg_001 ->
#   the process dies leaving a checkpoint production_checkpoint_available()
#   rejects (torn write / SIGKILL / disk-full -- the exact case its
#   median-outlier branch exists for).
#
# `finalize_segment` runs from run_gareus's `finally`, so those rows are sealed
# "interrupted" with a real end_step and stay readable by
# gareus.query.load_samples whether or not a resumable checkpoint survived.  The
# round-2 guard is provably silent there (no usable checkpoint), so the round-3
# refresh replaced the very map those rows were labelled against.
#
# The second veto, `_phase_holds_samples_logged_against_its_window_map`, mirrors
# the loader's own per-segment filter rather than testing for file presence, so it
# narrows the round-3 refresh only where a *visible* record exists -- a "running"
# segment (SIGKILL, and the shape the refresh was written for) is still refreshed,
# because the next run_gareus's open_segment demotes it to non-last and the
# loader then skips it.
# ---------------------------------------------------------------------------


def _seal_samples_segment(phase_dir: Path, *, status: str, rows: int = 12, end_step=None):
    """Write real Parquet samples into `phase_dir` and seal the segment.

    Uses the production writers themselves (`ParquetSampleWriter`,
    `SegmentRegistry`) rather than hand-rolled files, so what the veto is asked
    about is the on-disk shape production actually leaves behind.  Returns the
    segment id.
    """
    from gareus.store import ParquetSampleWriter, SegmentRegistry

    phase_dir.mkdir(parents=True, exist_ok=True)
    reg = SegmentRegistry(phase_dir)
    seg_id = reg.open_segment("run_001", None, 1)
    writer = ParquetSampleWriter(phase_dir / "samples" / seg_id, flush_rows=1000)
    last_step = 0
    for i in range(rows):
        last_step = 50 * (i + 1)
        # window_id cycles over the three windows attempt 1's compacted map covers.
        writer.write_sample(last_step, 0, i % 3, 0.05 * (i % 3), -1.0 + 0.1 * i,
                            -100.0, 5.0, 2.0, 0.4)
    writer.close()
    if status == "complete":
        reg.close_segment(seg_id, end_step=last_step)
    elif status == "running":
        pass  # left exactly as open_segment wrote it
    else:
        reg.seal_segment(
            seg_id,
            absolute_end_step=last_step if end_step is None else int(end_step),
            status=status,
        )
    return seg_id


def _fake_resumable_checkpoint(phase_dir: Path, n_replicas: int = 4):
    """A checkpoint manifest `production_checkpoint_available` accepts.

    All replica files the same plausible size, so neither the absolute floor nor
    the median-outlier branch rejects it -- the shape that makes the round-2
    resume guard the deciding one.
    """
    from gareus.correctness.checkpoint_store import publish_generation

    publish_generation(
        phase_dir,
        [b"\0" * 200_000 for _ in range(n_replicas)],
        {
            "assignments": list(range(n_replicas)), "prod_done": 30_000,
            "absolute_step": 30_000, "parity": 0, "attempt": 0,
            "next_exchange": 30_000, "next_log": 30_000,
            "exchange_stats": {}, "rng_bit_generator": "PCG64", "rng_state": {},
        },
    )


def _absent_production_checkpoint(phase_dir: Path, n_replicas: int = 4):
    """No committed checkpoint: fresh-start path remains available.

    Deliberately not "no checkpoint at all": that is a state where the round-2
    guard was never going to preserve anything, so a test built on it proves less
    than it looks.  This is the median-outlier shape that predicate's own comment
    describes -- three healthy replica checkpoints and one cut short by a
    mid-write kill -- so the samples below are durable while round-2 is provably
    silent.
    """


def _drive_capturing_the_map_at_entry(monkeypatch, args, out):
    """Drive the real loop to the flat epoch's run_gareus and read its map there.

    Returns the (epoch_window, state_id) pairs the phase is about to sample
    against -- i.e. after every driver-side write decision for that phase has
    been made.
    """
    import gareus.adaptive_production as ap
    import gareus.production as prod

    captured = {}

    def _fake_run_gareus(_args, run_dir, *rest, **kw):
        captured["dir"] = Path(run_dir)
        captured["pairs"] = _pairs(Path(run_dir) / "epoch_window_map.csv")
        raise _StopAfterDrop()

    monkeypatch.setattr(prod, "run_gareus", _fake_run_gareus)
    with pytest.raises(_StopAfterDrop):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert captured["dir"].name == "epoch_001", "the drive must reach the flat epoch"
    return captured["pairs"]


def test_the_round_2_guard_is_silent_while_the_samples_veto_fires(tmp_path):
    """The two vetoes on one on-disk state: exactly the second one must decide.

    Pins the premise of every loop-drive test below -- that this is a state where
    preserving cannot be credited to the round-2 no-clobber guard -- instead of
    asserting it in a comment.
    """
    from gareus.checkpoints import production_checkpoint_available
    import gareus.adaptive_production as ap

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="interrupted")
    _absent_production_checkpoint(epoch_dir)
    map_path = epoch_dir / "epoch_window_map.csv"

    assert production_checkpoint_available(epoch_dir) is False, (
        "the fixture must be a torn checkpoint, not a resumable one")
    assert ap._phase_window_map_is_owned_by_a_resuming_phase(
        epoch_dir, map_path, resume_requested=True, expected_rows=5) is False
    assert ap._phase_holds_samples_logged_against_its_window_map(epoch_dir, map_path) is True


@pytest.mark.parametrize("status,end_step", [
    ("complete", None),
    ("interrupted", None),      # sealed at a real step boundary
    ("interrupted", -1),        # no valid boundary -> loader skips it
    ("abandoned", -1),
])
def test_the_veto_agrees_with_the_loader_about_what_is_readable(tmp_path, status, end_step):
    """Mirror-equivalence against the real loader, per segment status.

    `_phase_holds_samples_logged_against_its_window_map` re-implements
    gareus.query._load_segmented_parquet's per-segment filter by hand -- it has
    to, because the driver cannot open DuckDB and read millions of rows just to
    decide whether to write a CSV.  A hand mirror is only as good as the check
    that it still matches, so this asserts the two agree on the same on-disk
    state, using the real loader rather than a restatement of its rules.  Drift on
    either side (a new status, a changed boundary rule) fails here.
    """
    import gareus.adaptive_production as ap
    from gareus.query import load_samples

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status=status, rows=12, end_step=end_step)

    loader_sees_rows = len(load_samples(epoch_dir).get("window_id", [])) > 0
    veto = ap._phase_holds_samples_logged_against_its_window_map(
        epoch_dir, epoch_dir / "epoch_window_map.csv")

    assert veto is loader_sees_rows, (
        f"{status}/end_step={end_step}: the veto and the loader disagree about "
        "whether this phase's rows are readable")


def test_the_veto_diverges_from_the_loader_only_for_a_running_segment(tmp_path):
    """The one deliberate divergence from the mirror above, pinned as deliberate.

    The loader reads a `running` segment while it is the *last* one, so right now
    these rows ARE readable -- and the veto still says "not a record", because the
    next run_gareus opens its own segment before writing anything, which demotes
    this one to non-last (and the fresh-start branch then seals it `abandoned`).
    Written as its own test rather than an exception in the parametrisation so the
    divergence cannot be mistaken for the mirror being wrong: this is the
    self-closing residual the veto's docstring describes.
    """
    import gareus.adaptive_production as ap
    from gareus.query import load_samples

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="running", rows=12)

    samples = load_samples(epoch_dir)
    assert len(samples["window_id"]) == 12, (
        "a running-and-last segment really is readable until the next open_segment")
    assert ap._phase_holds_samples_logged_against_its_window_map(
        epoch_dir, epoch_dir / "epoch_window_map.csv") is False


def test_a_flat_epoch_with_samples_behind_a_torn_checkpoint_keeps_its_map(tmp_path, monkeypatch):
    """The finding, end to end through the real driver loop.

    Attempt 1 dropped windows 1 and 3, compacted this map to 3 rows, sampled
    against it, and died leaving a checkpoint that cannot be resumed from.
    Attempt 2 must not replace that map: those rows are labelled with local
    window indices only these three rows resolve, and re-attributing 12 real
    samples here is the same mechanism that mis-attributed 1.82M on chignolin_6.
    """
    args, out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="interrupted")
    _absent_production_checkpoint(epoch_dir)
    before = (epoch_dir / "epoch_window_map.csv").read_bytes()

    pairs = _drive_capturing_the_map_at_entry(monkeypatch, args, out)

    assert pairs == [(0, 0), (1, 2), (2, 4)], "attempt 1's compacted map must survive"
    assert (epoch_dir / "epoch_window_map.csv").read_bytes() == before


def test_a_completed_segment_behind_a_torn_checkpoint_also_keeps_its_map(tmp_path, monkeypatch):
    """The other analysis-visible status, same site.

    A segment sealed "complete" is read in full by the loader, so its rows are at
    least as protected as an interrupted one's.  Separate from the test above
    because "complete" and "interrupted" take different branches of the veto's
    status filter, and only one of them needs an end_step to be valid.
    """
    args, out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="complete")
    _absent_production_checkpoint(epoch_dir)

    pairs = _drive_capturing_the_map_at_entry(monkeypatch, args, out)

    assert pairs == [(0, 0), (1, 2), (2, 4)]


def test_a_sigkilled_attempts_running_segment_still_gets_the_refresh(tmp_path, monkeypatch):
    """The round-3 refresh must survive round 4 for the case it was written for.

    A SIGKILL leaves the segment "running", and the loader reads a running
    segment only while it is the *last* one -- which the next run_gareus stops
    being true before it writes anything (open_segment precedes the first
    ParquetSampleWriter).  Treating that as a record would disable the refresh
    permanently for any phase that ever flushed one 5,000-row chunk, so this is
    the regression guard on the narrowing, not on the fix.
    """
    args, out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="running")

    pairs = _drive_capturing_the_map_at_entry(monkeypatch, args, out)

    assert pairs == [(i, i) for i in range(5)], (
        "the pre-drop 5-window map attempt 2 re-pulls from must be restored")


def test_an_abandoned_segment_is_not_a_record(tmp_path):
    """`abandoned` means the query layer skips the segment entirely."""
    import gareus.adaptive_production as ap

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="abandoned", end_step=-1)

    assert ap._phase_holds_samples_logged_against_its_window_map(
        epoch_dir, epoch_dir / "epoch_window_map.csv") is False


def test_an_interrupted_segment_with_no_valid_boundary_is_not_a_record(tmp_path):
    """`interrupted` with end_step < 0 has no valid data boundary, so the loader
    skips it -- the seal shape production uses when it knows of no checkpoint
    step to trust."""
    import gareus.adaptive_production as ap

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="interrupted", end_step=-1)

    assert ap._phase_holds_samples_logged_against_its_window_map(
        epoch_dir, epoch_dir / "epoch_window_map.csv") is False


def test_samples_with_no_segments_json_are_a_record(tmp_path):
    """Legacy layout: with no registry the loader reads every Parquet file under
    samples/, so every one of them is a record."""
    import gareus.adaptive_production as ap

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="complete")
    (epoch_dir / "segments.json").unlink()

    assert ap._phase_holds_samples_logged_against_its_window_map(
        epoch_dir, epoch_dir / "epoch_window_map.csv") is True


def test_an_unreadable_segments_json_is_treated_as_a_record(tmp_path):
    """Fail safe: an unparseable registry makes the phase unloadable, not empty,
    so the map behind it must not be replaced on a guess."""
    import gareus.adaptive_production as ap

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="complete")
    (epoch_dir / "segments.json").write_text("{not json", encoding="utf-8")

    assert ap._phase_holds_samples_logged_against_its_window_map(
        epoch_dir, epoch_dir / "epoch_window_map.csv") is True


def test_a_phase_with_samples_but_no_map_yet_still_gets_one(tmp_path):
    """The veto needs an existing map to protect.

    With no map on disk there is nothing those samples were logged against, and
    vetoing would leave the phase with no map at all -- the MBAR loader would
    then fall back to the registry, which is the mis-attribution by another
    route.
    """
    import gareus.adaptive_production as ap

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="complete")
    (epoch_dir / "epoch_window_map.csv").unlink()

    assert ap._phase_holds_samples_logged_against_its_window_map(
        epoch_dir, epoch_dir / "epoch_window_map.csv") is False


def test_write_phase_window_map_honours_the_samples_veto(tmp_path):
    """The funnel four of the five driver sites go through.

    Same state, exercised at the helper rather than through the loop, so the
    frozen-final / extension-round / next-epoch sites are covered too.
    """
    import gareus.adaptive_production as ap

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="interrupted")
    map_path = epoch_dir / "epoch_window_map.csv"
    before = map_path.read_bytes()

    registry = ap.WindowStateRegistry.load(_out / "adaptive_production")
    # resume_requested=False: the round-2 guard cannot fire at all here, so any
    # preservation is attributable to the samples veto alone.
    written = ap._write_phase_window_map(registry, map_path, resume_requested=False)

    assert written is None
    assert map_path.read_bytes() == before


def test_the_subset_window_csv_site_honours_the_samples_veto_too(tmp_path):
    """run_segment's shape, at the level where the two claims differ.

    ``_write_phase_window_map`` (the funnel the other four sites use) has the
    behavioural test above.  The two sites that pass ``map_path=`` to a window-table
    writer themselves -- run_segment's scheduled sub-run and the frozen final phase
    -- consult the vetoes into a local and then hand that local to the writer, so
    "the veto is called" and "its answer is used" are separate claims and only the
    first is covered by the source-level test below.  This exercises the second:
    same predicate pair, same `map_path=None if preserved else map_path` writer
    call, asserting the map bytes are untouched while the windows table is still
    written (it is inert on a fast resume but stays a faithful record of what the
    driver asked for).

    `resume_requested=False` throughout, so nothing here can be credited to the
    round-2 resume guard.
    """
    import gareus.adaptive_production as ap

    _args, out, seg_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(seg_dir, status="interrupted")
    map_path = seg_dir / "epoch_window_map.csv"
    before = map_path.read_bytes()

    registry = ap.WindowStateRegistry.load(out / "adaptive_production")
    preserved = bool(
        ap._phase_window_map_is_owned_by_a_resuming_phase(
            seg_dir, map_path, resume_requested=False, expected_rows=5)
        or ap._phase_holds_samples_logged_against_its_window_map(seg_dir, map_path)
    )
    assert preserved is True

    windows_csv = seg_dir.parent / "topup_001_12000_windows.csv"
    ap.write_state_subset_window_csv(
        registry, windows_csv, [0, 1, 2, 3, 4],
        map_path=None if preserved else map_path)

    assert windows_csv.exists(), "the windows table must still be written"
    assert map_path.read_bytes() == before


def test_the_preserved_map_makes_a_replayed_drop_set_idempotent(tmp_path, monkeypatch):
    """Composition with the on-disk rewrite ledger, same drop set.

    Attempt 2 re-pulls and flags the same two windows.  Because the map it meets
    is still attempt 1's compaction, the ledger's "does the map still hold that
    rewrite" conjunct is TRUE and the replay is refused -- which is the correct
    outcome now: one window set, one map, and it already says the right thing.
    The ledger stays append-only and records nothing new for a skip.
    """
    args, out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="interrupted")
    _absent_production_checkpoint(epoch_dir)

    seen = _drive_with_attempt_2_dropping(monkeypatch, args, out, [1, 3])

    assert seen["result"]["status"] == "skipped_already_applied"
    assert seen["pairs"] == [(0, 0), (1, 2), (2, 4)]
    assert len(seen["ledger"]) == 1, "a skipped rewrite records nothing"


def test_a_different_drop_set_leaves_the_map_untouched_for_the_loader_to_catch(tmp_path, monkeypatch):
    """Composition with the ledger, differing drop sets -- the honest residual.

    There is no single map file that is correct for both attempts once their drop
    sets differ, so the choice is which failure mode to leave behind.  Preserving
    keeps the already-written rows correct and makes attempt 2's window set
    *detectably* inconsistent with the map (3 rows against the 4 windows it runs),
    which `_validate_and_repair_epoch_window_map` refuses to load rather than
    mis-attribute.  Replacing the map would instead have produced a 4-row map
    whose row count matches attempt 2's window set exactly -- nothing downstream
    would notice, and attempt 1's rows would be silently wrong.  Loud beats
    silent; that is the whole trade this test pins.

    The refusal comes from the ledger's *same drop path* clause rather than its
    (n_windows_before, dropped) key: the map still holds attempt 1's rewrite and
    the request comes from the same `post_pull_auto_drop` source, which the
    ledger treats as that path being replayed over its own rewrite.  Worth
    asserting on the exact status, because it is the clause that keeps the map
    intact here -- the key-equality clause would not have matched [1] against
    [1, 3].
    """
    args, out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _seal_samples_segment(epoch_dir, status="interrupted")
    _absent_production_checkpoint(epoch_dir)

    seen = _drive_with_attempt_2_dropping(monkeypatch, args, out, [1])

    assert seen["result"]["status"] == "skipped_already_applied"
    assert seen["result"]["applied_by"] == "post_pull_auto_drop"
    assert seen["result"]["dropped_window_indices"] == [1], (
        "the refusal must be about this request's own [1], not a re-reading of [1, 3]")
    assert seen["pairs"] == [(0, 0), (1, 2), (2, 4)]
    assert len(seen["ledger"]) == 1


def test_the_pull_and_drop_block_precedes_the_first_sample_segment():
    """Source-level pin for the fact the veto's `running` exclusion rests on.

    `run_gareus` must reach its pull/drop block -- the only thing that compacts a
    phase's epoch_window_map.csv mid-run -- strictly before it opens a sample
    segment.  That ordering is what makes "a compacted map with no samples behind
    it" a real state (so the round-3 refresh still has a job) while "samples
    written before the drop that compacted their map" is impossible within one
    attempt.  Reorder those and the veto's status filter stops being sound, in a
    way no unit test of the veto itself would notice.
    """
    import ast
    import inspect

    import gareus.production as prod

    tree = ast.parse(inspect.getsource(prod))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_gareus")

    def _first_stmt_index(needle):
        return next(i for i, stmt in enumerate(fn.body) if needle in ast.unparse(stmt))

    drop_at = _first_stmt_index("drop_bad_us_windows_and_rebuild(")
    repair_at = _first_stmt_index("repair_epoch_window_map_from_surviving_windows(")
    open_at = _first_stmt_index("open_segment(")
    writer_at = _first_stmt_index("ParquetSampleWriter(")

    assert repair_at < open_at < writer_at
    assert repair_at < drop_at, (
        "the map repair must precede the post-pull drop: the drop's indices are "
        "positions in `centers_a`, i.e. in the window table's order, so on a map "
        "whose rows are a permutation of that order the repair is what makes the "
        "drop land on the rows it names -- see "
        "test_a_reorder_makes_a_following_drop_time_rewrite_land_on_the_right_rows")
    assert drop_at < open_at, (
        "the post-pull window drop must precede SegmentRegistry.open_segment; "
        "_phase_holds_samples_logged_against_its_window_map's 'running' exclusion "
        "assumes a drop can never fire after samples have started flowing")


def test_both_no_clobber_vetoes_are_consulted_at_every_write_site():
    """Structural guard: the second veto must not be forgotten at a site.

    The resume guard is consulted directly at three places (the shared
    `_write_phase_window_map` funnel plus the two window-table writers that pass
    `map_path=` themselves).  Every one of them must consult the samples veto
    too, or that site silently keeps the checkpoint-only rule.
    """
    import ast
    import inspect

    import gareus.adaptive_production as ap

    tree = ast.parse(inspect.getsource(ap))
    resume_guard = "_phase_window_map_is_owned_by_a_resuming_phase"
    samples_veto = "_phase_holds_samples_logged_against_its_window_map"

    missing = []
    resume_sites = 0
    for scope in ast.walk(tree):
        if not isinstance(scope, ast.FunctionDef):
            continue
        calls = [n.func.id for n in ast.walk(scope)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        n_resume = calls.count(resume_guard)
        if not n_resume:
            continue
        resume_sites += n_resume
        if calls.count(samples_veto) < n_resume:
            missing.append(scope.name)

    assert resume_sites >= 3, f"expected at least three resume-guard call sites, found {resume_sites}"
    assert missing == [], f"write sites still on the checkpoint-only rule: {missing}"


def test_the_resume_guards_message_reports_the_row_count_without_diagnosing_it(tmp_path, capsys):
    """Finding 2's message fix.

    The guard used to append "looks like a post-pull window drop was already
    applied here" whenever the map's row count differed from the registry's
    active-state count.  That is one of several ways the counts can differ, and it
    is the *wrong* one at the epoch-0 post-run bootstrap call site, where the
    registry has just been rebuilt from this attempt's own post-drop window table
    -- there a difference would mean the file describes a different window set
    entirely, not that this map is the drop's own correction.  The message now
    states what was observed and stops there; the decision never depended on the
    cause.
    """
    import gareus.adaptive_production as ap

    _args, _out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    _fake_resumable_checkpoint(epoch_dir)
    capsys.readouterr()

    assert ap._phase_window_map_is_owned_by_a_resuming_phase(
        epoch_dir, epoch_dir / "epoch_window_map.csv",
        resume_requested=True, expected_rows=5) is True

    out = capsys.readouterr().out
    assert "map covers 3 window(s) against the 5 the caller was about to write" in out
    assert "post-pull window drop was already applied" not in out, (
        "the message must not assert a cause it cannot know at every call site")
    # `expected_rows` is `len(state_ids)` at run_segment's site and
    # `len(registry.active_states())` everywhere else, so naming either one is the
    # same defect in a new place.
    assert "active registry state" not in out, (
        "the message must not name expected_rows as the registry's active-state "
        "count -- run_segment hands it a scheduled subset instead")


# ---------------------------------------------------------------------------
# Review round 5 -- the two write sites that do NOT go through the funnel had
# no behavioural coverage at all, and the structural test could not tell a
# correct conditional from an inverted one.
#
# Four of the six driver map writes go through `_write_phase_window_map`, which
# owns both vetoes and has behavioural tests above.  The other two hand the
# vetoes' answer to a *window-table* writer as a keyword instead:
#
#   run_segment, nested in run_scheduled_adaptive_epoch:
#       write_state_subset_window_csv(..., map_path=None if _seg_map_preserved
#                                                    else _seg_map_path)
#   run_adaptive_production_auto_loop, its `if not _final_already_done` block:
#       registry.write_active_window_csv(..., map_path=None if _final_map_preserved
#                                                      else _final_map_path)
#
# Both are named here (and in the tests below) by enclosing function plus the
# distinguishing local, never by line number: this section originally pointed at
# them as ":5315" and ":6444" and gareus/adaptive_production.py grew ~240 lines
# within that same round, so those pointers landed in unrelated code almost
# immediately.  A rename or a move now breaks
# test_the_two_keyword_write_sites_are_named_by_something_that_cannot_rot rather
# than silently rotting the only map from a test to the site it covers.
#
# For those two, "the veto is called" and "its answer is used the right way
# round" are separate claims.  Round 5 inverted both to `path if preserved else
# None` and the entire map-rewrite suite still passed -- neither site was ever
# executed by a test (every loop-driving test set allocation_scheduler False, and
# `_StopAfterDrop` fires in the epoch loop long before the final phase), and the
# structural test only checked that *a* conditional gated on `_map_preserved` was
# present, which an inversion satisfies.
#
# The inverted behaviour is not a loud failure.  With `map_path=None` on a fresh
# phase, `write_state_subset_window_csv` writes the window table and no map;
# `_find_adaptive_epoch_dirs` then falls back to the PARENT's map -- a map over a
# different, larger state set -- or drops the segment from the analysis outright.
# That is the chignolin_6 mis-attribution class re-created at the load-bearing
# sites, which is why these drive the real loop rather than re-deriving the
# conditional in the test.
#
# Each site is covered by a matched pair, so an inversion at either one fails on
# its own rather than only as part of the pair:
#
#   preserved (phase already holds samples) -> the map on disk must survive byte
#                                              for byte
#   fresh     (nothing on disk)             -> the phase must get its own map
#
# In both preserved halves the round-2 resume guard is provably silent (the
# checkpoint is torn, so `production_checkpoint_available` is False), so what is
# demonstrated is the samples veto's answer being used, not the resume guard's.
# ---------------------------------------------------------------------------


def _drive_capturing_the_phase_map_at_run_gareus(monkeypatch, args, out):
    """Drive the real loop; read the map of whichever phase dir reaches run_gareus.

    Everything the driver decides about that phase's map has already happened by
    the time the fake fires -- both write sites under test run strictly before
    their phase's `run_gareus` call -- so this reads the decision, not a
    reconstruction of it.  Returns the phase directory, whether it has an
    `epoch_window_map.csv` by then, and that file's raw bytes and decoded
    (epoch_window, state_id) pairs.
    """
    import gareus.adaptive_production as ap
    import gareus.production as prod

    captured = {}

    def _fake_run_gareus(_args, run_dir, *rest, **kw):
        run_dir = Path(run_dir)
        map_path = run_dir / "epoch_window_map.csv"
        captured["dir"] = run_dir
        captured["exists"] = map_path.exists()
        captured["bytes"] = map_path.read_bytes() if map_path.exists() else None
        captured["pairs"] = _pairs(map_path) if map_path.exists() else None
        raise _StopAfterDrop()

    monkeypatch.setattr(prod, "run_gareus", _fake_run_gareus)
    with pytest.raises(_StopAfterDrop):
        ap.run_adaptive_production_auto_loop(args, out, None, None, None, None, None, None)
    assert "dir" in captured, "the drive never reached run_gareus at all"
    return captured


def _scheduled_epoch_campaign(tmp_path, *, segment_already_sampled: bool):
    """The round-3 campaign, re-pointed at the SCHEDULED (baseline+topup) layout.

    `allocation_scheduler=True` sends epoch 1 through `run_scheduled_adaptive_epoch`,
    whose `run_segment` writes each sub-run's own map -- the `_seg_map_preserved`
    site.  The phase under test is therefore `epoch_001/baseline`, not `epoch_001`
    itself.
    """
    args, out, epoch_dir = _five_state_resumed_campaign(tmp_path)
    args.adaptive_production_allocation_scheduler = True
    seg_dir = epoch_dir / "baseline"
    if segment_already_sampled:
        # Attempt 1 of THIS segment: it compacted its own map, flushed samples
        # against it, then died leaving a checkpoint too torn to resume from.
        seg_dir.mkdir(parents=True, exist_ok=True)
        (seg_dir / "epoch_window_map.csv").write_text(
            _LOOP_MAP_5_STATE_COMPACTED, encoding="utf-8")
        _seal_samples_segment(seg_dir, status="interrupted")
        _absent_production_checkpoint(seg_dir)
    return args, out, seg_dir


def _frozen_final_campaign(tmp_path, *, phase_already_sampled: bool):
    """The same campaign, arranged so the drive lands on the frozen final phase.

    `_StopAfterDrop` fires inside the epoch loop on every other loop-driving test,
    so the final phase's write site is never reached.  The campaign's driver
    summary records one completed epoch, so `start_epoch` is 1; setting
    `adaptive_production_epochs = 1` makes `range(1, 1)` empty and the loop is
    skipped outright, putting the final phase first in line.

    The final phase is forced flat (unsegmented): with the scheduled final path
    active the map under test would be a `run_segment` sub-run's, i.e. the
    `_seg_map_preserved` site again rather than `_final_map_preserved`.
    """
    args, out, _epoch_dir = _five_state_resumed_campaign(tmp_path)
    args.adaptive_production_epochs = 1
    args.adaptive_production_final_allocation_scheduler = False
    args.adaptive_production_scheduled_final_segments = False
    final_dir = out / "adaptive_production" / "final"
    if phase_already_sampled:
        final_dir.mkdir(parents=True, exist_ok=True)
        (final_dir / "epoch_window_map.csv").write_text(
            _LOOP_MAP_5_STATE_COMPACTED, encoding="utf-8")
        _seal_samples_segment(final_dir, status="interrupted")
        _absent_production_checkpoint(final_dir)
    return args, out, final_dir


def test_a_scheduled_segment_holding_samples_keeps_its_map(tmp_path, monkeypatch):
    """`run_segment`/`_seg_map_preserved`, preserved half -- driven through the
    real scheduled epoch.

    `epoch_001/baseline` already holds attempt 1's samples logged against attempt
    1's compacted 3-row map.  The identity map the registry would write covers a
    different, larger state set, so writing it re-attributes every one of those
    rows.  The bytes on disk must be exactly what they were.
    """
    import gareus.adaptive_production as ap

    args, out, seg_dir = _scheduled_epoch_campaign(tmp_path, segment_already_sampled=True)
    before = (seg_dir / "epoch_window_map.csv").read_bytes()
    # Pin the premise: the round-2 resume guard cannot be what preserves this, so
    # the assertions below are about the samples veto's answer being used.
    assert ap.production_checkpoint_available(seg_dir) is False

    captured = _drive_capturing_the_phase_map_at_run_gareus(monkeypatch, args, out)

    assert captured["dir"] == seg_dir, (
        "the drive must reach the scheduled sub-run, not some other phase")
    assert captured["pairs"] == [(0, 0), (1, 2), (2, 4)], (
        "the samples' own map was replaced -- every row is now attributed to a "
        f"different state (map is {captured['pairs']})")
    assert captured["bytes"] == before


def test_a_fresh_scheduled_segment_is_given_its_own_map(tmp_path, monkeypatch):
    """`run_segment`/`_seg_map_preserved`, fresh half -- the claim an inverted
    conditional breaks silently.

    Nothing on disk for this sub-run, so there is no record to protect and the
    registry's identity map must be written.  Without it `_find_adaptive_epoch_dirs`
    falls back to the parent `epoch_001/epoch_window_map.csv`, which here is
    attempt 1's 3-row compaction over a different state set -- asserted below to
    be genuinely different, so the fallback really would mis-attribute rather than
    coincidentally agree.
    """
    args, out, seg_dir = _scheduled_epoch_campaign(tmp_path, segment_already_sampled=False)
    parent_pairs = _pairs(seg_dir.parent / "epoch_window_map.csv")

    captured = _drive_capturing_the_phase_map_at_run_gareus(monkeypatch, args, out)

    assert captured["dir"] == seg_dir
    assert captured["exists"] is True, (
        "a fresh scheduled sub-run got no map of its own; the loader would fall "
        f"back to the parent's {parent_pairs}, which covers a different state set")
    assert captured["pairs"] == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    assert captured["pairs"] != parent_pairs


def test_a_frozen_final_phase_holding_samples_keeps_its_map(tmp_path, monkeypatch):
    """`_final_map_preserved` (the frozen final phase), preserved half -- the site
    no loop-driving test had ever reached.

    Same on-disk state as the scheduled-segment case, one phase kind over.  Note
    this site hands the resume guard `resume_requested` un-gated by
    `production_checkpoint_available` (unlike `run_segment`'s `_seg_will_resume`),
    so the torn checkpoint is what keeps the guard silent here -- pinned below,
    together with the frozen-final skip guard, whose "already done" branch would
    let the drive pass this test without the write site ever executing.
    """
    import gareus.adaptive_production as ap

    args, out, final_dir = _frozen_final_campaign(tmp_path, phase_already_sampled=True)
    before = (final_dir / "epoch_window_map.csv").read_bytes()
    assert ap.production_checkpoint_available(final_dir) is False
    assert ap._is_adaptive_production_completed(out / "adaptive_production") is False, (
        "the final phase must actually be entered, not skipped as already done")

    captured = _drive_capturing_the_phase_map_at_run_gareus(monkeypatch, args, out)

    assert captured["dir"] == final_dir, (
        "the drive must reach the frozen final phase, not stop in the epoch loop")
    assert captured["pairs"] == [(0, 0), (1, 2), (2, 4)], (
        "the final phase's samples were re-attributed by a fresh identity map "
        f"(map is {captured['pairs']})")
    assert captured["bytes"] == before


def test_a_fresh_frozen_final_phase_is_given_its_own_map(tmp_path, monkeypatch):
    """`_final_map_preserved` (the frozen final phase), fresh half.

    Nothing on disk under `final/`, so the registry's identity map must be
    written.  `final/` has no parent phase map to fall back to at all, so an
    inverted conditional here does not mis-attribute the phase -- it drops it
    from the analysis entirely.
    """
    args, out, final_dir = _frozen_final_campaign(tmp_path, phase_already_sampled=False)

    captured = _drive_capturing_the_phase_map_at_run_gareus(monkeypatch, args, out)

    assert captured["dir"] == final_dir
    assert captured["exists"] is True, (
        "the frozen final phase got no window map; with no parent map to fall "
        "back to, the whole phase drops out of the analysis")
    assert captured["pairs"] == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]


def test_the_two_keyword_write_sites_are_named_by_something_that_cannot_rot():
    """Anti-rot pin for the four docstrings above.

    Those four tests are the only map from a test to the write site it drives, and
    the sites are genuinely hard to find by grep: neither calls
    `_write_phase_window_map`, both hand the vetoes' answer to a *window-table*
    writer as a `map_path=` keyword instead.  The names the docstrings use must
    therefore keep resolving -- so this asserts each enclosing function still holds
    exactly one `map_path=` keyword gated on its own `_*_map_preserved` local.

    Fails on a rename or on the conditional being restructured, which is the point:
    a stale pointer here costs a future reader the whole trail, and a line number
    (what these docstrings used to carry) cannot be checked at all.
    """
    import ast
    import inspect

    import gareus.adaptive_production as ap

    tree = ast.parse(inspect.getsource(ap))

    def _scope(name, within=None):
        root = within if within is not None else tree
        matches = [n for n in ast.walk(root)
                   if isinstance(n, ast.FunctionDef) and n.name == name]
        assert len(matches) == 1, f"expected exactly one {name}, found {len(matches)}"
        return matches[0]

    scheduled_epoch = _scope("run_scheduled_adaptive_epoch")
    sites = {
        "_seg_map_preserved": _scope("run_segment", within=scheduled_epoch),
        "_final_map_preserved": _scope("run_adaptive_production_auto_loop"),
    }
    for local, scope in sites.items():
        gated = [kw for kw in ast.walk(scope)
                 if isinstance(kw, ast.keyword) and kw.arg == "map_path"
                 and local in ast.unparse(kw.value)]
        assert len(gated) == 1, (
            f"{scope.name} no longer has exactly one `map_path=` keyword gated on "
            f"`{local}` (found {len(gated)}); the docstrings above point at it by "
            "that name and must be updated with it")


# ---------------------------------------------------------------------------
# Review round 5, second finding -- the repair result that never reached disk.
#
# run_gareus persists repair_epoch_window_map_from_surviving_windows' verdict into
# the phase's gareus_metadata.json, which is the only place an operator can later
# ask "was this phase's window map ever actually checked against the windows it
# really ran?".  The guard used to be `status != "consistent"`, which is not the
# same question: the equal-count branch abstains with status "consistent" plus
# `centers_verified` False when duplicate restraint centres make the membership
# check ambiguous, so the ONE outcome that means "unverifiable" was the one dropped
# on the floor -- while both of its siblings (the equal-count mismatch and every
# skipped_*) were durable.  Read afterwards, an abstaining phase was
# indistinguishable from a verified one.
#
# The parametrisation below runs the REAL repair function over a fixture for every
# outcome it can reach from on-disk state, and evaluates run_gareus' OWN `if` test
# (lifted from its source and compiled, with `_map_repair` bound) against what the
# real function returned.  So neither half is re-derived in the test: an inverted
# or stale condition fails, and so does a repair branch that stops reporting
# `centers_verified`.  Expected values are written per case as literals.
#
# Deliberately not a warning: duplicate (primary, secondary) centres are legal --
# the explicit-2D loader keeps a user's duplicated rows as separate thermodynamic
# states -- so a run built that way records this on every phase.  Nothing consumes
# the key programmatically (grep: gareus/production.py writes it, nothing reads
# it), so making it durable cannot escalate a healthy run's health verdict; it only
# stops the metadata from claiming more than the check established.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _run_gareus_map_repair_guard():
    """run_gareus' own `if` test for persisting the repair result, compiled.

    Selected by the assignment in the guard's DIRECT body rather than by a substring
    over the whole subtree: `ast.walk` yields enclosing nodes first, so a future
    outer `if` wrapped around this block would otherwise be picked up instead and
    the test would silently start evaluating a condition that does not depend on
    `_map_repair` at all.
    """
    import ast
    import inspect

    import gareus.production as prod

    key = "epoch_window_map_repair_from_window_table"
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(prod)))
              if isinstance(n, ast.FunctionDef) and n.name == "run_gareus")
    guards = [n for n in ast.walk(fn)
              if isinstance(n, ast.If) and any(key in ast.unparse(stmt) for stmt in n.body)]
    assert len(guards) == 1, (
        f"expected exactly one guard writing {key} into window_metadata, found {len(guards)}")
    return compile(ast.Expression(guards[0].test), "<run_gareus>", "eval")


def _run_gareus_would_record(summary):
    """True when run_gareus would put `summary` into the phase's gareus_metadata."""
    import gareus.production as prod

    return bool(eval(_run_gareus_map_repair_guard(), vars(prod), {"_map_repair": summary}))


def _ladder_centers(n: int) -> list[tuple[float, float]]:
    return [_identity_map_centers(i) for i in range(n)]


def _map_from_centers(path: Path, centers: list[tuple[float, float]]) -> None:
    _write_map(
        path,
        [{"epoch_window": i, "state_id": i, "primary_center": c1, "secondary_center": c2}
         for i, (c1, c2) in enumerate(centers)],
        REAL_RUN_FIELDS,
    )


def _phase_rewritten(tmp_path: Path) -> None:
    """More map rows than real windows, matching centres: the ordinary repair."""
    _identity_map(tmp_path / "epoch_window_map.csv", 6)
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv",
                    [_identity_map_centers(i) for i in (0, 1, 3, 4, 5)])


def _phase_verified(tmp_path: Path) -> None:
    """The overwhelming majority: map and window table agree, centres check out."""
    _identity_map(tmp_path / "epoch_window_map.csv", 6)
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", _ladder_centers(6))


def _phase_equal_count_duplicate_centers(tmp_path: Path) -> None:
    """Counts agree, but two windows share a restraint centre, so which of the two
    identical rows a survivor is cannot be established.  The abstain case."""
    centers = [(0.1, -1.0), (0.1, -1.0), (0.2, 1.0)]
    _map_from_centers(tmp_path / "epoch_window_map.csv", centers)
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", centers)


def _phase_equal_count_different_window_set(tmp_path: Path) -> None:
    """Two attempts dropped equally many but different windows (residual #1)."""
    ladder = _ladder_centers(7)
    _map_from_centers(tmp_path / "epoch_window_map.csv", ladder[:3] + ladder[4:])
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", ladder[:5] + ladder[6:])


def _phase_equal_count_permutation(tmp_path: Path) -> None:
    """Same window set, wrong order (residual #1's benign twin): derivable, so it
    is repaired in place rather than refused."""
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])


def _phase_equal_count_reorder_not_positionally_verifiable(tmp_path: Path) -> None:
    """A permutation whose window table is not in local-window order, so which of
    the two orders is the local-window order cannot be read off either record."""
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3], table_file_order=[0, 2, 1, 3, 4])


def _phase_map_shorter_than_window_set(tmp_path: Path) -> None:
    _identity_map(tmp_path / "epoch_window_map.csv", 2)
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", _ladder_centers(4))


def _phase_longer_map_duplicate_centers(tmp_path: Path) -> None:
    _map_from_centers(tmp_path / "epoch_window_map.csv",
                      [(0.1, -1.0), (0.1, -1.0), (0.2, 1.0)])
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", [(0.1, -1.0), (0.2, 1.0)])


def _phase_centers_do_not_match(tmp_path: Path) -> None:
    """Recentered between the interrupted attempt and the resume (tICA CV2)."""
    _identity_map(tmp_path / "epoch_window_map.csv", 6)
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv",
                    [(0.05 * i, 9.0 + i) for i in (0, 1, 3)])


def _phase_unreadable_map(tmp_path: Path) -> None:
    """A map path that exists but cannot be parsed (here: a directory)."""
    (tmp_path / "epoch_window_map.csv").mkdir()
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", _ladder_centers(3))


def _phase_without_a_window_table(tmp_path: Path) -> None:
    """Nothing to check against: the repair returns None, not a status."""
    _identity_map(tmp_path / "epoch_window_map.csv", 3)


@pytest.mark.parametrize("build, expected_status, expected_recorded", [
    (_phase_rewritten, "rewritten", True),
    (_phase_verified, "consistent", False),
    (_phase_equal_count_duplicate_centers, "consistent", True),
    (_phase_equal_count_different_window_set, "inconsistent_equal_count_centers_do_not_match", True),
    (_phase_equal_count_permutation, "rewritten_reordered", True),
    (_phase_equal_count_reorder_not_positionally_verifiable,
     "inconsistent_equal_count_reorder_not_positionally_verifiable", True),
    (_phase_map_shorter_than_window_set, "skipped_map_shorter_than_window_set", True),
    (_phase_longer_map_duplicate_centers, "skipped_duplicate_centers", True),
    (_phase_centers_do_not_match, "skipped_centers_do_not_match", True),
    (_phase_unreadable_map, "skipped_unreadable", True),
    (_phase_without_a_window_table, None, False),
], ids=lambda v: getattr(v, "__name__", v))
def test_every_repair_outcome_except_a_verified_pass_is_recorded(
        tmp_path, build, expected_status, expected_recorded):
    """Exhaustive over the outcomes the real repair can reach from on-disk state.

    The bar is "was the map VERIFIED against this phase's window set", not "was the
    map left alone" -- the abstain case is left alone precisely because it could not
    be verified, and that is what has to survive into the metadata.  The single
    False here is the one outcome that really did establish the map is right; a
    condition that also drops the abstain (the round-5 defect) fails on the
    duplicate-centres case, and one that records everything fails on the verified
    case, which is the cry-wolf direction.
    """
    build(tmp_path)

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert (summary or {}).get("status") == expected_status
    assert _run_gareus_would_record(summary) is expected_recorded


def test_the_abstain_says_on_the_record_that_it_could_not_verify(tmp_path):
    """Durability is only half of it: what lands in the metadata has to answer the
    question.  `status: consistent` alone reads as a pass, so the record must carry
    the two fields that distinguish an abstain from a verification -- otherwise an
    operator reading the phase directory learns nothing from its being there."""
    _phase_equal_count_duplicate_centers(tmp_path)

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["centers_verified"] is False
    assert summary["centers_check"] == "skipped_duplicate_centers"
    assert _run_gareus_would_record(summary) is True


# ---------------------------------------------------------------------------
# The equal-count PERMUTATION: same window set, wrong order.
#
# Write-side/load-side divergence closed here.  Both sides compare the phase's
# epoch_window_map.csv against the phase's own post-drop
# umbrella_explicit_windows.csv, and both used to settle the equal-count case
# with the ORDER-PRESERVING subsequence matcher alone.  A permutation fails that
# matcher (it cannot reach back past a row it already consumed), so the write
# side reported `inconsistent_equal_count_centers_do_not_match` and printed "the
# map describes a DIFFERENT window set of the same size" -- of a map whose window
# set is in fact identical, and whose correct local->state mapping the load side
# (gareus/mbar_analysis/loaders_adaptive.py's `_consistent_map_membership_notes`)
# now derives from exactly these two artifacts and repairs in memory.
#
# Why a permutation is derivable and a different window set is not: local window
# *i*'s samples were generated at the centres the table records for row *i*, so
# the state_id that belongs to them is the one on the map row carrying those
# centres, whatever position that row sits at.  When every table row has such a
# map row that is a complete mapping, read off rather than inferred.  When a
# window that really ran has NO row, its state_id is not recorded anywhere in
# these two artifacts and no amount of matching invents it -- that case stays
# refused, which is the direction that matters.
#
# These drive the real `repair_epoch_window_map_from_surviving_windows` and
# assert the FILE it leaves behind, not just its status: a status-only assertion
# would pass for an implementation that reordered the rows wrongly.
# ---------------------------------------------------------------------------

# state_ids deliberately not 0..N-1 so a reorder that "worked" by rebuilding an
# identity map instead of moving the real rows cannot pass.
_PERMUTED_STATE_IDS = [20, 21, 22, 23, 24]


def _permuted_phase(tmp_path: Path, order: list[int], *,
                    table_file_order: list[int] | None = None) -> Path:
    """A phase whose map lists exactly the windows its table lists, in `order`.

    `order[j]` is the local window whose row sits at map position *j*, so
    `order == [0, 1, 2, 3, 4]` is a healthy phase and any other permutation is
    the defect.  Every row keeps its own state_id and centres; only where the row
    sits changes -- which is precisely what a map written from a registry whose
    active-state order is not the window-array order looks like.

    `table_file_order` writes the surviving-window table's rows in that file
    order while each keeps its own ``window`` value, i.e. what a writer that ever
    sorted that table by something other than window index would produce.
    """
    centers = _ladder_centers(len(_PERMUTED_STATE_IDS))
    _write_map(
        tmp_path / "epoch_window_map.csv",
        [{"epoch_window": j, "state_id": _PERMUTED_STATE_IDS[w],
          "primary_center": centers[w][0], "secondary_center": centers[w][1]}
         for j, w in enumerate(order)],
        REAL_RUN_FIELDS,
    )
    path = tmp_path / "umbrella_explicit_windows.csv"
    if table_file_order is None:
        _survivor_table(path, centers)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["window", "distance_center_A", "primary_center",
                                    "secondary_cv_center"])
            writer.writeheader()
            for w in table_file_order:
                writer.writerow({"window": w, "distance_center_A": centers[w][0],
                                 "primary_center": centers[w][0],
                                 "secondary_cv_center": centers[w][1]})
    return tmp_path


def _map_centers(path: Path) -> list[tuple[float, float]]:
    return [(float(r["primary_center"]), float(r["secondary_center"])) for r in _read_map(path)]


@pytest.mark.parametrize("order", [
    [4, 0, 1, 2, 3],           # rotation by one
    [0, 2, 1, 3, 4],           # adjacent swap
    [4, 3, 2, 1, 0],           # reversal
    [2, 0, 1, 4, 3],           # two disjoint cycles
], ids=["rotation", "adjacent_swap", "reversal", "two_cycles"])
def test_repair_reorders_a_permuted_map_onto_the_window_table(tmp_path, order):
    """The map lists exactly the windows this phase ran, at the wrong positions.

    The correct mapping is read off the two artifacts: table row *i*'s centres
    name the restraint local window *i* really ran under, and the map row
    carrying those centres names the state that belongs to it.  What is asserted
    is the resulting FILE -- row *i*'s centres equal table row *i*'s centres, the
    state_ids are the same multiset (no row invented, none lost), and
    epoch_window is 0..N-1 -- because a status alone cannot tell a correct
    reordering from a wrong one.
    """
    _permuted_phase(tmp_path, order)
    table = _ladder_centers(len(order))

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "rewritten_reordered"
    path = tmp_path / "epoch_window_map.csv"
    assert _map_centers(path) == table
    assert _pairs(path) == [(i, s) for i, s in enumerate(_PERMUTED_STATE_IDS)]
    assert sorted(s for _, s in _pairs(path)) == sorted(_PERMUTED_STATE_IDS)


def test_the_reorder_no_longer_claims_a_different_window_set(tmp_path, capsys):
    """The misdiagnosis this closes, head on.

    A permutation used to print "the map describes a DIFFERENT window set of the
    same size (two attempts at this phase dropped equally many but different
    windows)" -- a false claim about a map whose window set is identical, and a
    false claim about why.  That sentence must not appear for a permutation; it
    is still the right sentence for a genuinely different set, which
    tests/test_equal_count_map_fingerprint.py pins on its own fixture.
    """
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])

    repair_epoch_window_map_from_surviving_windows(tmp_path)

    out = capsys.readouterr().out
    assert "DIFFERENT window set" not in out
    assert "PERMUTATION" in out
    assert "REORDERED" in out


def test_a_reorder_is_invisible_to_a_row_count_comparison(tmp_path):
    """The proxy that made this a divergence in the first place.

    A reorder returns the same NUMBER of rows in a different order, so every
    "did anything change?" test written as a row-count (or before/after length)
    comparison reports "nothing happened" while the map was in fact rewritten.
    The summary therefore carries no `dropped_state_ids`/`dropped_window_indices`
    at all -- fields whose emptiness would invite exactly that inference -- and
    answers the question by status plus the moved windows themselves.
    """
    _permuted_phase(tmp_path, [2, 0, 1, 4, 3])
    before = len(_read_map(tmp_path / "epoch_window_map.csv"))

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert len(_read_map(tmp_path / "epoch_window_map.csv")) == before  # the proxy sees nothing
    assert "dropped_state_ids" not in summary
    assert "dropped_window_indices" not in summary
    assert summary["status"] == "rewritten_reordered"
    assert summary["n_rows_moved"] == 5
    assert summary["moved_windows"] == [
        {"epoch_window": 0, "state_id_before": 22, "state_id_after": 20},
        {"epoch_window": 1, "state_id_before": 20, "state_id_after": 21},
        {"epoch_window": 2, "state_id_before": 21, "state_id_after": 22},
        {"epoch_window": 3, "state_id_before": 24, "state_id_after": 23},
        {"epoch_window": 4, "state_id_before": 23, "state_id_after": 24},
    ]


def test_a_cv1_only_phase_reorders_on_its_primary_axis_alone(tmp_path):
    """A plain 1D ladder has no secondary centre in either record, and the two
    sides must agree that "absent in both" is agreement rather than a mismatch.

    This is the one place the shared matcher is reached through an adapter: the
    write side reads centres with its own column aliases and hands them over as
    text, so a CV1-only phase's secondary axis makes the whole trip as NaN.
    NaN != NaN under ordinary float comparison, so a matcher (or an adapter) that
    got this wrong would refuse every 1D phase outright -- a healthy-run refusal,
    the failure mode that is its own kind of bug.
    """
    centers = [(0.05 * i, float("nan")) for i in range(5)]
    _write_map(
        tmp_path / "epoch_window_map.csv",
        [{"epoch_window": j, "state_id": 10 + w, "primary_center": centers[w][0],
          "secondary_center": ""}
         for j, w in enumerate([4, 0, 1, 2, 3])],
        REAL_RUN_FIELDS,
    )
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", [(c1, "") for c1, _ in centers])

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "rewritten_reordered"
    assert _pairs(tmp_path / "epoch_window_map.csv") == [(i, 10 + i) for i in range(5)]
    rows = _read_map(tmp_path / "epoch_window_map.csv")
    assert [float(r["primary_center"]) for r in rows] == [c1 for c1, _ in centers]
    assert all(r["secondary_center"] == "" for r in rows)  # the axis stays absent


def test_the_rows_moved_count_is_not_a_restatement_of_the_state_id_changes(tmp_path):
    """`n_rows_moved` counts rows that physically changed position; `moved_windows`
    reports which local window changed STATE.  They are not the same number, and
    conflating them would make a real rewrite announce itself as "0 windows
    moved".

    The fixture is the case that separates them: a map carrying one state_id on
    two rows with different centres (corrupt, but the map is data on disk and the
    warning has to stay true whatever it holds).  Reordering by centre still moves
    those rows and still writes the file, while no local window's state_id
    changes -- so the state-level record is legitimately empty and the row-level
    count is not.
    """
    centers = _ladder_centers(4)
    _write_map(
        tmp_path / "epoch_window_map.csv",
        [{"epoch_window": j, "state_id": 30 + (w // 2), "primary_center": centers[w][0],
          "secondary_center": centers[w][1]}
         for j, w in enumerate([1, 0, 3, 2])],
        REAL_RUN_FIELDS,
    )
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", centers)

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "rewritten_reordered"
    assert summary["moved_windows"] == []          # no local window changed state
    assert summary["n_rows_moved"] == 4            # every row still changed place
    assert _map_centers(tmp_path / "epoch_window_map.csv") == centers


def test_reordering_is_idempotent_by_projection_and_stays_out_of_the_drop_ledger(tmp_path):
    """How a reorder interacts with `epoch_window_map_rewrites.json`'s DROP list:
    it does not touch it, and that separation is load-bearing.

    The `applied` list's entries are drop sets, keyed by `(n_windows_before,
    dropped_window_indices)`, and its idempotence rule rests on "a successful
    rewrite always removes at least one row, so that pair can never legitimately
    recur for one map file".  A reorder removes no rows, so it has no such key,
    and two reorders of one phase (the driver overwrites the map between attempts
    and it is permuted again) would produce byte-identical entries -- an entry
    shape that falsifies the invariant the whole branch rests on.

    It needs no replay guard either, and that is a stronger property than the
    ledger's rather than a weaker one: ordering rows into the table's order is a
    PROJECTION, so applying it twice is applying it once.  Asserted on the real
    function -- the second call finds the order-preserving matcher succeeding and
    returns `consistent`, having written nothing.

    What it DOES need is a record, which now lives in the file's separate
    `reorders` list -- see
    `test_a_reorder_is_recorded_on_disk_before_the_caller_can_store_anything`.
    This test's earlier revision asserted `not LEDGER.exists()`, i.e. that a
    reorder left no durable trace at all; that was the defect, not the contract.
    """
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])

    first = repair_epoch_window_map_from_surviving_windows(tmp_path)
    after_reorder = (tmp_path / "epoch_window_map.csv").read_bytes()
    second = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert first["status"] == "rewritten_reordered"
    assert second["status"] == "consistent"
    assert second["centers_verified"] is True
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == after_reorder
    # The drop list stays empty through both calls: no drop key was invented for
    # a rewrite that removed no rows.
    assert _ledger(tmp_path) == []
    # And the projection really is one: the second call added no second record.
    assert len(_reorder_ledger(tmp_path)) == 1


def test_a_reorder_is_recorded_on_disk_before_the_caller_can_store_anything(tmp_path):
    """The durable record of a reorder, in the only place that survives the
    process dying immediately after the map is rewritten.

    `run_gareus` stores the repair summary in the phase's gareus_metadata.json
    only after `repair_epoch_window_map_from_surviving_windows` returns.  A
    process killed in between leaves the map reordered on disk with the metadata
    channel empty -- and a resumed repair then finds the two artifacts agreeing
    and answers `consistent`, which `_epoch_window_map_repair_must_be_recorded`
    correctly declines to record.  Nothing anywhere would say a reorder happened,
    even though every sample already written for those windows was logged against
    the old order.

    Simulated exactly that way: the summary is deliberately NOT handed to the
    metadata channel, and the resume is the real function called a second time.
    """
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)
    # ... and here the process dies: nothing calls _run_gareus_would_record's
    # branch, so window_metadata never receives `summary`.
    resumed = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "rewritten_reordered"
    assert summary["ledger"]["appended"] is True
    assert summary["ledger"]["list"] == "reorders"
    # The metadata channel really is empty on the resume -- the gap this closes.
    assert resumed["status"] == "consistent"
    assert _run_gareus_would_record(resumed) is False
    # The ledger file is the record that remains, and it says what moved.
    records = _reorder_ledger(tmp_path)
    assert len(records) == 1
    assert records[0]["status"] == "rewritten_reordered"
    assert records[0]["state_ids_before"] == [24, 20, 21, 22, 23]
    assert records[0]["state_ids_after"] == [20, 21, 22, 23, 24]
    assert records[0]["moved_windows"][0] == {
        "epoch_window": 0, "state_id_before": 24, "state_id_after": 20}


def test_the_reorder_record_is_kept_off_the_drop_list_refusal_path(tmp_path):
    """Why the record goes in `reorders` and not in `applied`, measured on the
    refusal path itself rather than asserted about it.

    `_epoch_window_map_rewrite_already_applied` refuses a request whose `source`
    matches an entry's while the map still holds that entry's
    `surviving_state_ids`.  Put a reorder record in `applied` under source
    `surviving_window_table` and, the moment it carries that key, it refuses a
    later GENUINE drop rewrite from that same source: the ledger, whose entire
    purpose is to stop a map being wrongly compacted, becomes the reason a needed
    compaction never happens.

    Honest about how far away that is today: the reorder summary records
    `state_ids_before`/`state_ids_after`, not `surviving_state_ids`, so an entry
    misfiled into `applied` right now would be skipped by that matcher for a
    reason having nothing to do with the separation.  ONE key -- an obvious one to
    add when spelling out what a rewrite left behind -- closes that distance, and
    the second half below drives the real matcher with exactly that entry to show
    the refusal is real.  The first half is the outcome guard: whatever else
    changes, a later genuine drop from the same source must still be applied.
    """
    from gareus.production import (
        _append_epoch_window_map_ledger_entry,
        _epoch_window_map_rewrite_already_applied,
        _epoch_window_map_state_ids,
    )

    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])
    reorder = repair_epoch_window_map_from_surviving_windows(tmp_path)
    assert reorder["status"] == "rewritten_reordered"

    later_drop = rewrite_epoch_window_map_after_drop(
        tmp_path, [1], 5, source="surviving_window_table")

    assert later_drop["status"] == "rewritten"
    assert later_drop["dropped_state_ids"] == [21]
    assert _pairs(tmp_path / "epoch_window_map.csv") == [(0, 20), (1, 22), (2, 23), (3, 24)]

    # The hazard the separation avoids, measured: the same reorder record filed
    # into `applied` with the one key it is missing does match, and the matcher
    # hands back a refusal for a drop that has not been applied at all.
    hazard = tmp_path / "hazard"
    _permuted_phase(hazard, [4, 0, 1, 2, 3])
    hazard_reorder = repair_epoch_window_map_from_surviving_windows(hazard)
    misfiled = dict(hazard_reorder)
    misfiled["surviving_state_ids"] = misfiled["state_ids_after"]
    _append_epoch_window_map_ledger_entry(hazard, "applied", misfiled)
    _fields, hazard_rows = _read_epoch_window_map_rows_via_production(hazard)

    refusal = _epoch_window_map_rewrite_already_applied(
        hazard, [1], 5, _epoch_window_map_state_ids(hazard_rows), "surviving_window_table")

    assert refusal is not None
    assert refusal["status"] == "rewritten_reordered"

    # ...and with the record where it actually goes, that same lookup on the same
    # shape of phase finds nothing to refuse with.  A third directory, because the
    # first one has since taken a real drop from that source and its ledger
    # entry -- a genuine replay guard -- would answer this lookup correctly.
    clean = tmp_path / "clean"
    _permuted_phase(clean, [4, 0, 1, 2, 3])
    repair_epoch_window_map_from_surviving_windows(clean)
    _fields2, clean_rows = _read_epoch_window_map_rows_via_production(clean)
    assert _epoch_window_map_rewrite_already_applied(
        clean, [1], 5, _epoch_window_map_state_ids(clean_rows),
        "surviving_window_table") is None


def test_a_later_drop_rewrite_does_not_erase_the_reorder_record(tmp_path):
    """Both lists live in one file that is replaced wholesale on every append, so
    an appender that rebuilt the payload from its own list alone would delete the
    other one.  A phase that reorders and then drops must end up with both."""
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])
    repair_epoch_window_map_from_surviving_windows(tmp_path)

    rewrite_epoch_window_map_after_drop(tmp_path, [1], 5, source="post_pull_auto_drop")

    assert len(_reorder_ledger(tmp_path)) == 1
    assert len(_ledger(tmp_path)) == 1
    assert _ledger(tmp_path)[0]["source"] == "post_pull_auto_drop"


def test_a_reorder_is_recorded_in_the_phase_metadata(tmp_path):
    """The second, non-durable channel, unchanged: run_gareus stores any
    non-`consistent` repair verdict in the phase's gareus_metadata.json.  Kept
    because the ledger record above is a floor, not a replacement -- the metadata
    copy is what an operator reading the phase directory sees first."""
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "rewritten_reordered"
    assert _run_gareus_would_record(summary) is True


def test_repair_still_refuses_a_genuinely_different_window_set_of_equal_size(tmp_path, capsys):
    """The direction that must not move.  Attempt 1 dropped window 1, attempt 2
    dropped window 3, so the counts agree and the membership does not: window 3
    really ran and has no row in the map at all, its state_id is recorded nowhere
    in these two artifacts, and reordering cannot invent one.  Refuse, loudly,
    and leave the file alone."""
    centers = _ladder_centers(5)
    _map_from_centers(tmp_path / "epoch_window_map.csv", centers[:1] + centers[2:])
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", centers[:3] + centers[4:])
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "inconsistent_equal_count_centers_do_not_match"
    assert summary["centers_verified"] is False
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original
    assert "DIFFERENT window set of the same size" in capsys.readouterr().out


@pytest.mark.parametrize("wrecked", [
    ("one_centre_wrecked", 9.75),
    ("near_miss_at_1000x_tolerance", None),
], ids=lambda v: v[0] if isinstance(v, tuple) else v)
def test_a_permutation_with_a_centre_that_does_not_match_is_refused(tmp_path, wrecked):
    """Adversarial: a map that is *nearly* a permutation must not be reordered.

    Either one row's centre is grossly wrong (a recentered state, e.g. the tICA
    CV2 recentering between an interrupted attempt and its resume) or it is off
    by 1000x the 1e-6 centre-match tolerance -- still four orders below the
    closest two real windows of any phase on disk ever come, so accepting it
    would mean the tolerance, not the evidence, decided the mapping.
    """
    label, wrecked_center = wrecked
    centers = _ladder_centers(5)
    order = [4, 0, 1, 2, 3]
    rows = [{"epoch_window": j, "state_id": _PERMUTED_STATE_IDS[w],
             "primary_center": centers[w][0], "secondary_center": centers[w][1]}
            for j, w in enumerate(order)]
    if wrecked_center is None:
        rows[0]["secondary_center"] = centers[4][1] + 1e-3
    else:
        rows[0]["secondary_center"] = wrecked_center
    _write_map(tmp_path / "epoch_window_map.csv", rows, REAL_RUN_FIELDS)
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", centers)
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "inconsistent_equal_count_centers_do_not_match"
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original


def test_a_permutation_whose_window_table_is_not_in_local_window_order_is_refused(tmp_path, capsys):
    """The gate that keeps this from "repairing" a map into a wrong answer.

    The existing subsequence repair only ever REMOVES rows and keeps the map's
    own order, so a mis-ordered table makes it refuse.  A reorder takes its
    output order FROM the table -- so if the table's rows are not this phase's
    local windows 0..N-1 by position, the reordered map is wrong while looking
    right, because the sets still match.  The table's own `window` column is the
    only thing that contradicts file order, so a table that carries it and
    disagrees with it must stop the reorder.  Its own status and message: folding
    this into the different-window-set refusal would print a second false claim
    in place of the one being removed.
    """
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3], table_file_order=[0, 2, 1, 3, 4])
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "inconsistent_equal_count_reorder_not_positionally_verifiable"
    assert "window` column is not numbered" in summary["positional_problem"]
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original
    out = capsys.readouterr().out
    assert "DIFFERENT window set" not in out
    assert "same window set" in out


def test_a_permutation_whose_map_is_not_numbered_0_to_n_minus_1_is_refused(tmp_path):
    """The same gate on the other record.  A sample's local window index is
    resolved through the `epoch_window` VALUE downstream, so position and value
    have to agree before a positional reordering means anything -- and if they do
    not, which of the two orders is the local-window order is an inference, not a
    reading.  Abstain rather than guess."""
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])
    rows = _read_map(tmp_path / "epoch_window_map.csv")
    for i, row in enumerate(rows):
        row["epoch_window"] = 10 + i
    _write_map(tmp_path / "epoch_window_map.csv", rows, REAL_RUN_FIELDS)
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "inconsistent_equal_count_reorder_not_positionally_verifiable"
    assert "epoch_window column is not numbered" in summary["positional_problem"]
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original


def test_duplicate_centres_never_reach_a_written_reorder(tmp_path):
    """Duplicate restraint centres must never produce a reordered map.

    The shared permutation matcher is a greedy first-unused-match: two rows with
    identical (primary, secondary) centres are interchangeable to it and it pairs
    them arbitrarily, which would attach one of them the other's state_id (the
    bias is identical, so nothing downstream would look wrong; the MBAR state
    pooling is not).  The duplicate-centres abstain is what shields it, and this
    fixture is a genuine permutation that the matcher WOULD happily reorder --
    verified by deleting the abstain from an isolated copy of the source, which
    turns this phase into `rewritten_reordered`.

    What is pinned is the outcome, not the abstain's position in the branch:
    moving it after the reorder attempt (but still before the write) was measured
    to leave every observable identical, so a test claiming to pin the ordering
    would have been claiming teeth it does not have.
    """
    centers = [(0.1, -1.0), (0.1, -1.0), (0.2, 1.0), (0.3, 2.0)]
    _map_from_centers(tmp_path / "epoch_window_map.csv",
                      [centers[0], centers[1], centers[3], centers[2]])
    _survivor_table(tmp_path / "umbrella_explicit_windows.csv", centers)
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "consistent"
    assert summary["centers_verified"] is False
    assert summary["centers_check"] == "skipped_duplicate_centers"
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original


def test_a_failing_reorder_check_never_raises_out_of_the_repair(tmp_path, monkeypatch, capsys):
    """`repair_epoch_window_map_from_surviving_windows` documents "Never raises",
    and run_gareus calls it outside any try/except: an exception here kills a
    production phase whose MD is otherwise fine.  The reorder check is the one
    part of it that reaches into another module, so its failure has to degrade to
    a recorded refusal rather than propagate."""
    import gareus.production as prod

    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated import/read failure")

    monkeypatch.setattr(prod, "_reorder_map_rows_onto_surviving_windows", _boom)

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "skipped_reorder_check_failed"
    assert "simulated import/read failure" in summary["error"]
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original
    assert _run_gareus_would_record(summary) is True
    assert "simulated import/read failure" in capsys.readouterr().out


def test_the_write_side_and_the_load_side_agree_on_what_a_permutation_is(tmp_path):
    """The divergence itself, asserted rather than described.

    Two answers to "is this a permutation" is the drift that produced this
    finding, so the write side calls the load side's own matcher rather than
    carrying a second copy of the rule.  Both shapes are checked: a permutation
    that both must accept, and a genuinely different window set that both must
    refuse -- run through the real load-side function, not a re-derivation.
    """
    from gareus.mbar_analysis.loaders_adaptive import _permute_map_rows_onto_window_table

    centers = _ladder_centers(5)
    _permuted_phase(tmp_path, [4, 0, 1, 2, 3])
    rows = _read_map(tmp_path / "epoch_window_map.csv")

    assert _permute_map_rows_onto_window_table(rows, centers) is not None
    assert repair_epoch_window_map_from_surviving_windows(tmp_path)["status"] == "rewritten_reordered"

    other = tmp_path / "different_set"
    _map_from_centers(other / "epoch_window_map.csv", centers[:1] + centers[2:])
    _survivor_table(other / "umbrella_explicit_windows.csv", centers[:3] + centers[4:])
    other_rows = _read_map(other / "epoch_window_map.csv")

    assert _permute_map_rows_onto_window_table(other_rows, centers[:3] + centers[4:]) is None
    assert (repair_epoch_window_map_from_surviving_windows(other)["status"]
            == "inconsistent_equal_count_centers_do_not_match")


def test_a_reorder_makes_a_following_drop_time_rewrite_land_on_the_right_rows(tmp_path):
    """Composition, in run_gareus's real order: the map repair runs before the
    post-pull auto-drop (pinned by
    test_the_pull_and_drop_block_precedes_the_first_sample_segment, which reads
    that order out of the source), and that drop's indices are positions in
    `centers_a` -- i.e. in the window table's order, the very order the reorder
    installs.  So on a permuted map the reorder is what makes the drop land on
    the rows it names; without it the drop silently compacts the wrong states.

    Both halves run the real functions.  The same structural test also pins what
    makes changing a local->state mapping safe at all: it happens before any
    sample segment is opened, never after samples were logged against the old
    order.
    """
    _permuted_phase(tmp_path, [2, 0, 1, 4, 3])
    assert repair_epoch_window_map_from_surviving_windows(tmp_path)["status"] == "rewritten_reordered"

    after_repair = rewrite_epoch_window_map_after_drop(
        tmp_path, [1], 5, source="post_pull_auto_drop")

    assert after_repair["status"] == "rewritten"
    assert after_repair["dropped_state_ids"] == [21]
    assert _pairs(tmp_path / "epoch_window_map.csv") == [(0, 20), (1, 22), (2, 23), (3, 24)]

    # The same drop applied to the same phase's UNREPAIRED map removes a
    # different state and leaves the survivors in the wrong order: it is the
    # reorder above, not the drop, that makes local window 1 mean state 21.
    unrepaired = tmp_path / "unrepaired"
    _permuted_phase(unrepaired, [2, 0, 1, 4, 3])
    without_repair = rewrite_epoch_window_map_after_drop(
        unrepaired, [1], 5, source="post_pull_auto_drop")

    assert without_repair["dropped_state_ids"] == [20]
    assert _pairs(unrepaired / "epoch_window_map.csv") == [(0, 22), (1, 21), (2, 24), (3, 23)]
