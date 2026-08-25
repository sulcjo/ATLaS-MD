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
    import json

    chk = phase_dir / "checkpoints"
    chk.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n_replicas):
        name = f"replica_{i:03d}.chk"
        (chk / name).write_bytes(b"\0" * 200_000)
        names.append(name)
    (chk / "production_checkpoint_manifest.json").write_text(
        json.dumps({"replica_checkpoint_files": names, "prod_done": 30_000}),
        encoding="utf-8",
    )


def _torn_production_checkpoint(phase_dir: Path, n_replicas: int = 4):
    """A checkpoint manifest `production_checkpoint_available` must reject.

    Deliberately not "no checkpoint at all": that is a state where the round-2
    guard was never going to preserve anything, so a test built on it proves less
    than it looks.  This is the median-outlier shape that predicate's own comment
    describes -- three healthy replica checkpoints and one cut short by a
    mid-write kill -- so the samples below are durable while round-2 is provably
    silent.
    """
    import json

    chk = phase_dir / "checkpoints"
    chk.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(n_replicas):
        name = f"replica_{i:03d}.chk"
        size = 1024 if i == n_replicas - 1 else 200_000
        (chk / name).write_bytes(b"\0" * size)
        names.append(name)
    (chk / "production_checkpoint_manifest.json").write_text(
        json.dumps({"replica_checkpoint_files": names, "prod_done": 30_000}),
        encoding="utf-8",
    )


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
    _torn_production_checkpoint(epoch_dir)
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
    _torn_production_checkpoint(epoch_dir)
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
    _torn_production_checkpoint(epoch_dir)

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
    _torn_production_checkpoint(epoch_dir)

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
    _torn_production_checkpoint(epoch_dir)

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
