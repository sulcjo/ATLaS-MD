"""An epoch_window_map.csv with the RIGHT NUMBER of rows can still describe the
WRONG windows -- both sides now check the centres, not just the count.

Documented residual #1 of the auto-drop map-rewrite fix
(docs/chignolin_6_low_ess_root_cause.md): every guard shipped with that fix keys
off the map's ROW COUNT against the phase's real window count.  Two attempts at
the same phase that drop equally many but *different* windows therefore slip
through silently -- attempt 1 drops window a, the phase is interrupted, attempt 2
re-pulls and drops window b instead, and attempt 1's map has exactly the right
number of rows while naming the wrong states from index min(a, b) onwards.

Two independent checks close it, one on each side, and both were already paid
for before this change:

  * write time -- `repair_epoch_window_map_from_surviving_windows` returned
    ``{'status': 'consistent'}`` on an equal row count without ever consulting
    `_match_map_rows_to_surviving_windows`, which sits in the same file and is
    already called on the mismatch path.
  * load time -- `_validate_and_repair_epoch_window_map` early-returned on an
    equal row count without ever comparing the map against the phase's own
    post-drop surviving-window table, which the *mismatch* path already reads
    and already treats as the authoritative record of what ran.

The load-time check is the same comparison as the write-time one, on the same
two artifacts -- the map's per-row (primary, secondary) restraint centre against
``umbrella_explicit_windows.csv``'s -- because the write-time check only ever
runs inside a process that (re)runs the phase and so cannot protect an analysis
of a run already on disk, which is every run this branch exists to rescue.  It
is bookkeeping, not physics: one of the phase's own two records of what it ran
contradicting the other.

Deliberately NOT physics, and this file used to say the opposite.  An earlier
revision fired on a per-window bar on |<cv2> - assigned centre| in units of that
window's restraint width sqrt(kT/k2), calibrated on one run at 3.0.  Sweeping
that shipped implementation over every adaptive-production phase holding samples
under RUNS/ (125 phases, 1,409 correctly-mapped windows) put 160 of those
windows -- 11%, on 47 phases -- over the bar, all false.  Real umbrella windows
sit systematically off their CV2 centre (the pull of the underlying free energy,
-(dG/dcv2)/k2: median 0.78, max 9.04 restraint widths on those runs), while the
CV2 centres of a 2D grid can be a small fraction of that apart because its
windows differ mainly in CV1.  The two populations are not separable by any
statistic on the sampled cv2 mean, absolute or nearest-centre, so the sampled
cv2 is now quoted as corroboration inside an already-raised note and can never
raise one -- `test_a_phase_whose_windows_all_sit_far_off_centre_is_silent`
transcribes the real correct-population numbers that killed the bar.

A map that lists exactly the right windows in the WRONG ORDER is repaired, not
merely flagged.  Round 6 declined to, and said so in the note: "the map holds no
row at all for a window that really ran ... nothing to re-derive the mapping
from".  That is true of two attempts dropping *different* windows and false of a
permutation, which is the case it was built for -- every window that ran does
have a row, so local window *i*'s state_id is the one on the map row carrying
the centres the table records for row *i*, read off the same centre match that
detected the problem.  The two cases are now separated and the note says which
happened.  The repair is deliberately NOT gated on the sampled-CV2 fingerprint:
that would reinstate as a veto the firing rule this file's own history killed
(neighbouring windows are pulled off their centres by different amounts, so
`dev_after > dev_before` occurs for genuine permutations).

Every firing case below is paired with a non-firing twin, so an implementation
that always warns fails just as loudly as one that never does -- and every
repair asserts the resulting MAPPING, not only the note, because a single
adjacent swap is an involution: reordering the map onto the table and the table
onto the map give the same answer, so a swap alone cannot tell a correct
implementation from an inverted one.  `test_a_rotated_map_is_repaired_onto_the_
window_tables_order` is the fixture that can.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gareus.mbar_analysis.loaders_adaptive import (
    MAP_NOTE_REPAIRED,
    MAP_NOTE_STALE_LOADED,
    _map_cv2_corroboration,
    _sorted_map_rows,
    _validate_and_repair_epoch_window_map,
    window_map_note_kind,
)
from gareus.production import repair_epoch_window_map_from_surviving_windows

CHECK_TAG = "[window map check]"
REPAIR_TAG = "[stale window map]"
MAP_FIELDS = ["epoch_window", "state_id", "primary_center", "secondary_center"]


def _write_map(path: Path, rows: list[dict], fields: list[str] = MAP_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_window_table(path: Path, centers: list, with_k_column: bool = True) -> None:
    """Mirror write_explicit_window_analysis_files' surviving-window table.

    `centers` is either ``[secondary]`` (primary pinned at 0.0, the common 2D
    grid shape where every window shares one CV1 centre) or ``[(primary,
    secondary)]``.
    """
    fields = ["window", "primary_center", "secondary_cv_center"]
    if with_k_column:
        fields.append("secondary_cv_k_kcal_mol")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, c in enumerate(_pairs(centers)):
            row = {"window": i, "primary_center": c[0], "secondary_cv_center": c[1]}
            if with_k_column:
                row["secondary_cv_k_kcal_mol"] = 60.0
            writer.writerow(row)


def _pairs(centers: list) -> list[tuple]:
    """`[secondary]` or `[(primary, secondary)]` -> `[(primary, secondary)]`."""
    return [c if isinstance(c, tuple) else (0.0, c) for c in centers]


def _samples(window_means: list[float], spread: float = 0.05):
    """``(window_ids, cv2)`` whose per-window MEAN is exactly `window_means[i]`.

    Deterministic and symmetric about the target mean rather than random, so a
    failure is never a seed.  Only ever *corroborating* evidence now -- no
    assertion below turns on the value, except the ones that prove exactly that
    (the silent twins, where the means are deliberately absurd).
    """
    ids, vals = [], []
    for i, mean in enumerate(window_means):
        for delta in (-2.0 * spread, -spread, spread, 2.0 * spread):
            ids.append(i)
            vals.append(mean + delta)
    return np.asarray(ids, dtype=np.int64), np.asarray(vals, dtype=np.float64)


def _ladder(n: int, spacing: float = 0.5) -> list[float]:
    """A CV2 window ladder whose neighbours sit `spacing` apart."""
    return [-1.0 + spacing * i for i in range(n)]


def _phase(tmp_path: Path, map_centers: list, table_centers: list | None = None, *,
           with_table: bool = True) -> tuple[Path, list[dict]]:
    """A phase directory whose map lists `map_centers` and whose surviving-window
    table lists `table_centers` (the same, unless a defect is being staged).

    Deliberately writes NO run_manifest.json: the membership check reads neither
    a temperature nor a force constant, and
    `test_a_phase_with_no_resolvable_temperature_is_still_fully_checked` holds
    that property down.
    """
    phase = tmp_path / "final" / "baseline"
    rows = [{"epoch_window": i, "state_id": i, "primary_center": c[0], "secondary_center": c[1]}
            for i, c in enumerate(_pairs(map_centers))]
    _write_map(phase / "epoch_window_map.csv", rows)
    if with_table:
        _write_window_table(phase / "umbrella_explicit_windows.csv",
                            map_centers if table_centers is None else table_centers)
    with (phase / "epoch_window_map.csv").open(newline="") as handle:
        return phase, list(csv.DictReader(handle))


def _write_recorded_window_count(phase: Path, n_windows: int) -> None:
    """The post-drop window count a real phase records for itself.

    `_phase_recorded_window_count` prefers this file; every fixture that has
    samples gets the same number from the sample column instead, so only the
    no-samples test needs it.
    """
    phase.mkdir(parents=True, exist_ok=True)
    (phase / "analysis_metadata_validation.json").write_text(
        json.dumps({"n_windows": int(n_windows)}))


def _run(phase: Path, rows: list[dict], window_ids, cv2):
    return _validate_and_repair_epoch_window_map(phase, rows, window_ids=window_ids, cv2=cv2)


def _check_notes(notes: list[str]) -> list[str]:
    return [n for n in notes if n.startswith(CHECK_TAG)]


def _repair_notes(notes: list[str]) -> list[str]:
    """The notes that say the map was rewritten.

    A repair reuses the `[stale window map]` tag the row-count repair already
    uses, which is not decoration: `gareus_report.py` grades that tag as a loud
    warning and turns it into a CAUTION on the run's mapping check ("this
    analysis is correct, the run on disk is still stale"), while its
    `[window map check]` rules are for a fault that was NOT repaired.
    """
    return [n for n in notes if n.startswith(REPAIR_TAG)]


def _swap(centers: list, i: int) -> list:
    """`centers` with entries i and i+1 exchanged -- one adjacent-window swap."""
    out = list(centers)
    out[i], out[i + 1] = out[i + 1], out[i]
    return out


def _rotate(centers: list) -> list:
    """`centers` rotated one step left -- an n-cycle, not an involution.

    The shape the verifier ran against RUNS/chignolin_5/epoch_000's real 27-row
    map.  Unlike `_swap` it distinguishes the two directions a repair could
    match in, so it is what pins the repair to "map row onto TABLE row" rather
    than the reverse.
    """
    return list(centers[1:]) + list(centers[:1])


def _mapping(rows: list[dict]) -> dict:
    """Exactly what the union loader builds from the guard's returned rows
    (`gareus/mbar_analysis/loaders_union_parquet.py`): local window -> state_id.

    Every repair below is asserted here rather than on the note's prose: this is
    the value that decides which umbrella state a sample is attributed to, and
    it is the only assertion an inverted implementation cannot also satisfy.
    """
    return {int(r["epoch_window"]): int(r["state_id"]) for r in rows}


# ---------------------------------------------------------------------------
# The defect: the map describes a different window set of the same size
# ---------------------------------------------------------------------------


def test_a_map_that_matches_the_window_table_produces_no_note(tmp_path):
    """The non-firing twin of every test below, and the overwhelmingly common
    case: a map that lists exactly the windows the phase ran must stay silent.

    Measured on real data before this was written: of the 125 adaptive-production
    phases holding samples under RUNS/, the 117 whose row count agrees all agree
    row for row on the centres too, and none of them emits a note."""
    centers = _ladder(6)
    phase, rows = _phase(tmp_path, centers)
    ids, cv2 = _samples(centers)

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert notes == []
    assert [r["state_id"] for r in out_rows] == [str(i) for i in range(6)]


def test_an_equal_count_map_describing_a_shifted_window_set_is_flagged(tmp_path):
    """The residual itself.  Attempt 1 dropped window 2, attempt 2 dropped window
    5, so from local index 2 onwards attempt 1's map names the window one step up
    the ladder -- while the row COUNT still agrees with the phase's window count,
    which is what makes this invisible to every other guard."""
    ladder = _ladder(7)
    map_centers = ladder[:2] + ladder[3:]     # attempt 1 dropped window 2
    real_centers = ladder[:5] + ladder[6:]    # attempt 2 dropped window 5
    phase, rows = _phase(tmp_path, map_centers, real_centers)
    ids, cv2 = _samples(real_centers)

    out_rows, notes = _run(phase, rows, ids, cv2)

    hits = _check_notes(notes)
    assert len(hits) == 1
    note = hits[0]
    for local in (2, 3, 4):
        assert f"local window {local}" in note
    assert "final/baseline" in note
    assert "chignolin_6_low_ess_root_cause.md" in note
    # The diagnosis has to be the one that is true HERE: ladder entry 2 is in
    # the map and not in the table and entry 5 the other way round, so a window
    # that really ran (5) has no row at all and its state_id is recorded
    # nowhere.  That is what makes this one unrepairable, and it is a different
    # statement from the reordering below.
    assert "DIFFERENT window set of the same size" in note
    assert "has NO row in the map" in note
    assert "PERMUTATION" not in note
    # Warn, do not refuse: the rows come back untouched and loading continues.
    assert [r["secondary_center"] for r in out_rows] == [r["secondary_center"] for r in rows]
    assert _mapping(out_rows) == {i: i for i in range(6)}


def test_the_unrepairable_note_still_carries_the_phrase_the_report_fails_on(tmp_path):
    """Cross-module coupling, asserted where it can break.

    `gareus_report.py`'s `_MAP_MEMBERSHIP_RX` turns this note into a hard FAIL of
    the run's sample-to-state mapping check by matching the literal phrase
    "DIFFERENT window set of the same size" -- deliberately on the note's claim
    rather than on its `[window map check]` tag, so the weaker "could not
    cross-check" sibling cannot reach the same FAIL.  Rewording this note
    without knowing that would silently demote a real mapping failure to a
    warning, in a file this one cannot see.
    """
    import re
    membership_rx = re.compile(
        r"\[window map check\][^\n]*DIFFERENT window set of the same size", re.I)
    ladder = _ladder(7)
    phase, rows = _phase(tmp_path, ladder[:2] + ladder[3:], ladder[:5] + ladder[6:])
    ids, cv2 = _samples(ladder[:5] + ladder[6:])

    note = _check_notes(_run(phase, rows, ids, cv2)[1])[0]

    assert membership_rx.search(note)


def test_a_swap_between_two_adjacent_windows_is_repaired(tmp_path):
    """The hard case, and the one the self-bias diagnostic structurally cannot
    see: two NEIGHBOURING windows exchange rows, so each sample lands on a state
    whose restraint is nearly the one it was generated in and every per-state
    energy stays plausible.  Only the two windows involved are wrong, the counts
    agree, and both restraint centres are still present in the map -- just on
    each other's local index.

    Which is exactly why it is repairable, and round 6's note said otherwise:
    every window that ran has a row, so the state each window's samples belong
    to is the one on the row carrying that window's real centres."""
    centers = _ladder(6)
    phase, rows = _phase(tmp_path, _swap(centers, 3), centers)
    ids, cv2 = _samples(centers)

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert _mapping(out_rows) == {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 5}
    assert _mapping(out_rows) != {i: i for i in range(6)}      # the un-repaired answer
    assert len(_repair_notes(notes)) == 1 and _check_notes(notes) == []
    note = notes[0]
    assert "PERMUTATION" in note and "REPAIRED IN MEMORY" in note
    assert "3->4" in note and "4->3" in note
    assert "local window 3" in note and "local window 4" in note
    assert "local window 2" not in note and "local window 5" not in note
    # The claim round 6 made about this very fixture, and could not support.
    assert "no row at all" not in note
    assert "DIFFERENT window set of the same size" not in note


def test_a_rotated_map_is_repaired_onto_the_window_tables_order(tmp_path):
    """The fixture a swap cannot replace, and the shape the verifier ran against
    a real 27-row map (chignolin_5/epoch_000 rotated by one).

    A single swap is an involution: matching the map onto the table and the
    table onto the map produce the same rows, so every swap test above is also
    passed by an implementation that has the direction backwards.  A rotation is
    an n-cycle, so the two directions give different mappings and only one of
    them is right -- local window *i* was run at the table's row-*i* centres, so
    it must come out carrying the state_id of the map row holding those centres,
    which for this fixture is the row one step earlier."""
    n = 6
    centers = _ladder(n)
    phase, rows = _phase(tmp_path, _rotate(centers), centers)
    ids, cv2 = _samples(centers)

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert _mapping(out_rows) == {i: (i - 1) % n for i in range(n)}
    assert _mapping(out_rows) != {i: (i + 1) % n for i in range(n)}   # matched backwards
    assert _mapping(out_rows) != {i: i for i in range(n)}             # not repaired at all
    assert len(_repair_notes(notes)) == 1
    assert f"{n} of them sit at the wrong local window" in notes[0]


def test_the_repair_note_cannot_be_mistaken_for_the_stale_map_escape_hatch(tmp_path):
    """Cross-module coupling again, in the other direction.

    `gareus_report.py` matches `[stale window map]` notes twice: any of them is
    a loud warning and a CAUTION, but one that also says "Loaded with the STALE
    map anyway" / "GAREUS_ALLOW_STALE_WINDOW_MAP is set" is CRITICAL and a hard
    FAIL, because that is a run knowingly loaded with wrong attribution.  A
    repair is the opposite situation -- the attribution has just been made right
    -- so its wording must not drift into that rule."""
    import re
    override_rx = re.compile(
        r"\[stale window map\][^\n]*(Loaded with the STALE map anyway|"
        r"GAREUS_ALLOW_STALE_WINDOW_MAP is set)", re.I)
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, _swap(centers, 1), centers)
    ids, cv2 = _samples(centers)

    note = _repair_notes(_run(phase, rows, ids, cv2)[1])[0]

    assert not override_rx.search(note)


def test_a_swap_at_the_tightest_real_centre_gap_is_still_resolved(tmp_path):
    """Resolution, asserted rather than assumed -- on re-measured numbers.

    6.21e-5 CV2 units is the smallest gap between two distinct window centres of
    one phase over every table this check actually reads under RUNS/
    (chignolin_5/epoch_001/baseline, windows 13 and 20); the smallest max-axis
    separation between any two windows of a phase is 4.21e-4
    (chignolin_quicktest/final/baseline, windows 1-2).  An earlier revision of
    this docstring and of `_CENTER_MATCH_TOL`'s comment claimed 0.006 and ~0.02,
    320x and 14x too loose.  The tolerance (1e-6, sized for CSV text
    round-tripping) still sits far below the real number, so a swap that tight
    is detected AND correctly repaired.  The bar this check replaced could not
    see it at all: 6.21e-5 is ~3e-4 restraint widths and its threshold was
    3.0."""
    centers = [0.0, 6.21e-5, 0.5]
    phase, rows = _phase(tmp_path, _swap(centers, 0), centers)
    ids, cv2 = _samples(centers)

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert _mapping(out_rows) == {0: 1, 1: 0, 2: 2}
    assert len(_repair_notes(notes)) == 1
    assert "local window 0" in notes[0] and "local window 1" in notes[0]


def test_a_swap_between_two_windows_sharing_a_cv1_centre_is_repaired_on_cv2(tmp_path):
    """The real 2D-grid geometry: every window shares one CV1 centre and the
    windows are told apart by CV2 alone (chignolin_5's phases have 29-30 windows
    at primary_center 0.0).  Neither the check nor the repair may need the
    primary centre to discriminate."""
    centers = [(0.0, c) for c in _ladder(5)]
    phase, rows = _phase(tmp_path, _swap(centers, 1), centers)
    ids, cv2 = _samples([c[1] for c in centers])

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert _mapping(out_rows) == {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    assert len(_repair_notes(notes)) == 1
    assert "local window 1" in notes[0] and "local window 2" in notes[0]


def test_a_swap_between_two_windows_sharing_a_cv2_centre_is_repaired_on_cv1(tmp_path):
    """The mirror, and coverage the check this replaced could never have had: two
    windows at the SAME secondary centre that differ in CV1 are a legal, common
    shape (a 2D grid column), and a cv2-only statistic is blind to a swap between
    them by construction.  Comparing both centres sees it -- and matching on both
    is what lets the repair put them back."""
    centers = [(0.0, 0.3), (1.0, 0.3), (2.0, 0.3)]
    phase, rows = _phase(tmp_path, _swap(centers, 0), centers)
    ids, cv2 = _samples([0.3, 0.3, 0.3])

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert _mapping(out_rows) == {0: 1, 1: 0, 2: 2}
    assert len(_repair_notes(notes)) == 1
    assert "local window 0" in notes[0] and "local window 1" in notes[0]


def test_a_cv1_only_phase_with_a_shifted_ladder_is_repaired(tmp_path):
    """A phase with no secondary CV at all still gets checked and repaired, on
    its primary centres.  This too is new coverage: the previous check needed a
    sampled cv2 to compare against and returned unconditionally when there was
    none, so a CV1-only run's mapping was never cross-checked by anything -- and
    the repair must not quietly require the evidence the check does not."""
    ladder = [(0.5 * i, float("nan")) for i in range(6)]
    phase, rows = _phase(tmp_path, _swap(ladder, 2), ladder)
    ids, cv2 = _samples([0.0] * 6)
    cv2 = np.full_like(cv2, np.nan)

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert _mapping(out_rows) == {0: 0, 1: 1, 2: 3, 3: 2, 4: 4, 5: 5}
    assert len(_repair_notes(notes)) == 1
    assert "local window 2" in notes[0] and "local window 3" in notes[0]
    # Nothing to cross-check the reordering against, and the note says so rather
    # than quoting a fingerprint it does not have.
    assert "No sampled cv2 was available" in notes[0]


def test_a_cv1_only_phase_that_agrees_is_silent(tmp_path):
    """Non-firing twin of the test above: NaN secondary centres on both sides are
    "absent in both records", which is agreement, not disagreement."""
    ladder = [(0.5 * i, float("nan")) for i in range(6)]
    phase, rows = _phase(tmp_path, ladder)
    ids, cv2 = _samples([0.0] * 6)

    assert _run(phase, rows, ids, np.full_like(cv2, np.nan))[1] == []


# ---------------------------------------------------------------------------
# It must not cry wolf: the populations that made the previous check unusable
# ---------------------------------------------------------------------------


def test_a_phase_whose_windows_all_sit_far_off_centre_is_silent(tmp_path):
    """The regression guard against ever reintroducing a distance bar, written
    from the real correct population it would fire on.

    These are RUNS/chignolin_5/adaptive_production/final/baseline's own window
    centres and the mean cv2 its samples really produced in each one (300 K,
    secondary force constants 12-45 kcal/mol/CV^2, so one restraint width is
    ~0.16-0.28 CV2 units).  Every window is pulled the same way -- the free
    energy is downhill in +cv2 across this whole ladder -- and the last one sits
    1.49 units, 5.3 restraint widths, off its own centre.  The map is correct.
    A correct map on data like this must produce nothing at all.  RUNS/ is
    gitignored, so the numbers are transcribed rather than read."""
    real = [
        (1.1632, 1.2878), (0.9049, 1.1284), (-0.0166, 0.4371), (0.1745, 0.5060),
        (-0.0817, 0.1853), (0.3813, 0.5249), (0.4136, 0.9745), (0.6116, 1.0825),
        (0.8266, 1.1403), (-0.2889, 0.2003), (-0.4922, -0.1652), (-0.4264, -0.2088),
        (-0.8491, -0.1518), (-0.8253, -0.0591), (-2.4849, -0.9918), (1.8887, 1.3333),
    ]
    centers = [c for c, _mu in real]
    phase, rows = _phase(tmp_path, centers)
    ids, cv2 = _samples([mu for _c, mu in real])

    assert _run(phase, rows, ids, cv2)[1] == []


def test_duplicate_window_centres_are_not_a_disagreement(tmp_path):
    """Two windows may legitimately share a restraint centre -- the explicit-2D
    loader keeps a user's duplicated rows as separate thermodynamic states, and
    chignolin_5's final/baseline really does have two windows at secondary
    -0.4264.  Identical centres are not evidence of anything and must not fire."""
    centers = [-1.0, -0.4264, -0.4264, 0.5]
    phase, rows = _phase(tmp_path, centers)
    ids, cv2 = _samples(centers)

    assert _run(phase, rows, ids, cv2)[1] == []


def test_a_swap_between_two_identical_centre_rows_stays_undetected(tmp_path):
    """The honest bound, asserted rather than left to a comment.  Two windows
    whose (primary, secondary) centres are IDENTICAL can be exchanged without
    changing either artifact, so no comparison of centres can see it -- and
    neither can any statistic on the sampled CVs, since the two restraints are
    the same restraint.  Such a swap leaves the bias matrix correct but pools
    each window's samples under the other's state_id, so it is a real, unclosed
    residual, not a harmless one."""
    centers = [(0.0, -1.0), (0.0, 0.25), (0.0, 0.25), (0.0, 1.0)]
    phase, rows = _phase(tmp_path, _swap(centers, 1), centers)
    ids, cv2 = _samples([-1.0, 0.25, 0.25, 1.0])

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert notes == []
    # ...and the state_ids really are exchanged relative to the window table,
    # i.e. the test is asserting a blind spot, not a fixture that is secretly fine.
    assert [r["state_id"] for r in out_rows] == ["0", "1", "2", "3"]
    assert [float(r["secondary_center"]) for r in out_rows] == [-1.0, 0.25, 0.25, 1.0]


# ---------------------------------------------------------------------------
# It must SKIP where it cannot run, and never fake a pass
# ---------------------------------------------------------------------------


def test_a_phase_with_no_window_table_at_all_is_out_of_scope_and_silent(tmp_path):
    """A phase that never wrote a surviving-window table (a plain 1D ladder, a
    legacy layout) is not a phase whose record went missing -- it never had one.
    Announcing a non-check once per such phase would be noise; every pre-existing
    loader fixture in this repo has exactly this shape."""
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, centers, with_table=False)
    ids, cv2 = _samples(centers)

    assert _run(phase, rows, ids, cv2)[1] == []


def test_a_window_table_of_a_different_length_reports_that_it_could_not_check(tmp_path):
    """Both of the phase's own records exist and they disagree about how many
    windows it ran, while the row-count check -- which compares the map against
    the phase's *recorded* window count -- passed.  Nothing can be compared row
    for row, and saying nothing would let "could not check" read as "checked and
    clean".  Unreachable on all 125 real phases measured (every one has a table
    whose length equals its authoritative window count); kept because if it does
    happen it is a genuine contradiction."""
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, centers)
    _write_window_table(phase / "umbrella_explicit_windows.csv", centers[:4])
    ids, cv2 = _samples(centers)

    hits = _check_notes(_run(phase, rows, ids, cv2)[1])

    assert len(hits) == 1
    assert "Could not cross-check" in hits[0]
    assert "the membership check did not run" in hits[0]
    assert "4 window(s) while the map lists 5" in hits[0]


def test_a_map_not_numbered_from_zero_reports_that_it_could_not_check(tmp_path):
    """Position and value must agree before a positional comparison means
    anything: a sample's local window index is resolved downstream through the
    map's ``epoch_window`` VALUE, so a map numbered 1..N would have its rows
    lined up against the wrong table rows by any positional check -- and pass.
    No current writer produces one (the driver enumerates the registry, every
    rewrite path renumbers the survivors 0..N-1) and none of the 152 maps under
    RUNS/ is numbered otherwise, so this abstains rather than guessing."""
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, centers)
    for i, row in enumerate(rows):
        row["epoch_window"] = str(i + 1)
    ids, cv2 = _samples(centers)

    hits = _check_notes(_run(phase, rows, ids, cv2)[1])

    assert len(hits) == 1
    assert "Could not cross-check" in hits[0]
    assert "not numbered 0..4" in hits[0]


def _write_window_table_in_file_order(path: Path, centers: list, file_order: list) -> None:
    """The surviving-window table with its rows written in `file_order`, each
    keeping its own ``window`` value.

    What a writer that ever sorted this table by something other than window
    index would produce.  `centers[i]` is local window *i*'s centre, as always;
    only the order the rows appear in the file changes.
    """
    pairs = _pairs(centers)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["window", "primary_center",
                                                    "secondary_cv_center"])
        writer.writeheader()
        for w in file_order:
            writer.writerow({"window": w, "primary_center": pairs[w][0],
                             "secondary_cv_center": pairs[w][1]})


def test_a_window_table_not_in_window_order_reports_that_it_could_not_check(tmp_path):
    """The map's index column was guarded; the table's was not.

    Everything here compares the two artifacts POSITIONALLY -- table row *i* is
    local window *i* -- and the map's ``epoch_window`` column is checked to be
    0..N-1 in file order before that comparison is allowed (the test above).
    `umbrella_explicit_windows.csv` carries the same kind of column and it was
    discarded entirely, file order simply trusted.  Nothing is wrong today
    (audited: all 291 window tables under RUNS/ have the column and all 291 are
    0..N-1 in file order, and `explicit_window_analysis_rows` enumerates them in
    order), but a writer that ever sorted the table differently would have every
    phase compared pair-by-wrong-pair, silently.

    Worse than silence, in fact: this fixture's table is a healthy phase's,
    merely written in a different row order, and without the guard the two
    artifacts look permuted with respect to each other -- so the repair would
    rewrite a CORRECT map into a wrong one and announce it confidently.
    """
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, centers)                 # map and table agree
    _write_window_table_in_file_order(phase / "umbrella_explicit_windows.csv",
                                      centers, [1, 0, 2, 3, 4])
    ids, cv2 = _samples(centers)

    out_rows, notes = _run(phase, rows, ids, cv2)

    hits = _check_notes(notes)
    assert len(hits) == 1
    assert "Could not cross-check" in hits[0]
    assert "`window` column is not numbered 0..4 in file order" in hits[0]
    # Abstained, not guessed: the map is exactly as it was.
    assert _mapping(out_rows) == {i: i for i in range(5)}
    assert _repair_notes(notes) == []


def test_a_window_table_with_no_window_column_is_still_compared_by_file_order(tmp_path):
    """The non-firing twin, and the reason the guard checks the column rather
    than requiring it: a table that never wrote a ``window`` column has nothing
    contradicting file order, so it is trusted exactly as it always was."""
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, _swap(centers, 1), centers)
    path = phase / "umbrella_explicit_windows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["primary_center", "secondary_cv_center"])
        writer.writeheader()
        for c in _pairs(centers):
            writer.writerow({"primary_center": c[0], "secondary_cv_center": c[1]})
    ids, cv2 = _samples(centers)

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert _mapping(out_rows) == {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    assert len(_repair_notes(notes)) == 1


def test_a_window_table_not_in_window_order_is_not_matched_against_a_stale_map(tmp_path):
    """The same assumption on the *repair* path, where the cost is higher.

    A genuinely stale map (4 rows, 3 windows) is repaired by matching the map
    against this table.  Read in file order the table here says the phase ran
    windows A, B, C; its own ``window`` column says the local order was B, A, C.
    Matching against the former yields a complete, plausible-looking repair with
    two of the three windows on the wrong state, and the sampled-CV2 fingerprint
    cannot tell (the two windows swap deviations, so the aggregate is
    unchanged).  With the table refused there is no other record here, and this
    path's documented policy is to fail closed rather than guess.
    """
    map_centers = _ladder(4)
    phase, rows = _phase(tmp_path, map_centers, with_table=False)
    _write_window_table_in_file_order(phase / "umbrella_explicit_windows.csv",
                                      [map_centers[1], map_centers[0], map_centers[2]],
                                      [1, 0, 2])
    ids, cv2 = _samples([map_centers[1], map_centers[0], map_centers[2]])

    with pytest.raises(ValueError) as exc:
        _run(phase, rows, ids, cv2)

    msg = str(exc.value)
    assert "refusing to load" in msg
    assert "Neither a usable umbrella_explicit_windows.csv" in msg


def test_a_phase_with_no_resolvable_temperature_is_still_fully_checked(tmp_path):
    """The check reads neither a temperature nor a force constant, so the two
    inputs the previous one abstained without (`run_manifest.json` and the
    table's ``secondary_cv_k_kcal_mol`` column) are simply not needed.  No
    fixture here writes a manifest; this one also drops the k column, and the
    swap is still caught and repaired."""
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, _swap(centers, 1), centers)
    _write_window_table(phase / "umbrella_explicit_windows.csv", centers, with_k_column=False)
    ids, cv2 = _samples(centers)

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert _mapping(out_rows) == {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    assert len(_repair_notes(notes)) == 1
    assert "Could not cross-check" not in notes[0]


def test_a_phase_with_no_samples_at_all_is_still_checked(tmp_path):
    """`cv2=None`/`window_ids=None` is what the union loader passes for a phase
    that yielded no samples.  The membership check works off the two CSVs alone,
    so it still runs -- new coverage, since the previous check needed sampled cv2
    and returned unconditionally without it."""
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, _swap(centers, 2), centers)
    # With neither samples nor metadata there is no window count to check the
    # row count against at all, and the function returns before any of this --
    # a real phase records one, so the fixture does too.
    _write_recorded_window_count(phase, len(centers))

    out_rows, notes = _run(phase, rows, None, None)

    assert _mapping(out_rows) == {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}
    assert len(_repair_notes(notes)) == 1
    assert "local window 2" in notes[0] and "local window 3" in notes[0]
    # No sampled cv2 to quote, so the note simply does not quote any.
    assert "Sampled CV2 for the affected windows" not in notes[0]
    assert "No sampled cv2 was available" in notes[0]


def test_the_check_does_not_disturb_the_row_count_repair_path(tmp_path):
    """A genuinely stale map (more rows than windows) must still take the repair
    path and produce the repair's own note, not this one."""
    real = _ladder(4)
    stale = real[:2] + [1.7] + real[2:]  # a phantom row for a dropped window
    phase, rows = _phase(tmp_path, stale, real)
    ids, cv2 = _samples(real)

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert len(notes) == 1
    assert notes[0].startswith("[stale window map]")
    assert [int(r["epoch_window"]) for r in out_rows][:4] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# The sampled-CV2 evidence: quoted inside a note, never the reason for one
# ---------------------------------------------------------------------------


def test_the_unrepairable_note_quotes_the_sampled_cv2_for_both_candidate_centres(tmp_path):
    """What makes the un-repairable note actionable: the operator is shown the
    two candidate centres and the number the samples actually produced, which is
    what tells them WHICH of the two artifacts to believe -- the question a
    human has to answer here, because nothing in the pair of artifacts can.
    Quoted as corroboration, with the honest caveat attached: measured, not
    decisive."""
    ladder = _ladder(7)
    map_centers = ladder[:2] + ladder[3:]     # attempt 1 dropped window 2
    real_centers = ladder[:5] + ladder[6:]    # attempt 2 dropped window 5
    phase, rows = _phase(tmp_path, map_centers, real_centers)
    ids, cv2 = _samples(real_centers)

    note = _check_notes(_run(phase, rows, ids, cv2)[1])[0]

    assert "Sampled CV2 for the affected windows" in note
    assert "<cv2>=0.0000" in note
    assert "nearer the window table's centre in 3 of 3 comparable window(s)" in note
    assert "Corroboration only" in note


def test_the_repair_quotes_the_fingerprint_as_evidence_and_not_as_its_reason(tmp_path):
    """The rule this file's own history is the argument for, restated on the new
    path: the sampled-CV2 fingerprint may be REPORTED by a repair and must never
    GATE one.

    Round 6 demoted a per-window sigma bar to corroboration after measuring it
    firing on 160 of 1,409 correctly-mapped real windows, because neighbouring
    windows are pulled off their own centres by different amounts.  The same
    physics makes `dev_after > dev_before` reachable for a genuine permutation,
    so a veto on it would decline a correct repair while asserting a false claim
    about the data.  Here the numbers are quoted with that caveat attached."""
    centers = _ladder(5)
    phase, rows = _phase(tmp_path, _swap(centers, 1), centers)
    ids, cv2 = _samples(centers)

    note = _repair_notes(_run(phase, rows, ids, cv2)[1])[0]

    assert "Sampled-cv2 fingerprint across the reordering" in note
    assert "evidence, not the reason" in note


def test_a_permutation_is_repaired_even_when_the_fingerprint_gets_worse(tmp_path):
    """The executable half of the test above: an implementation that vetoes on
    the fingerprint passes everything else in this file, because every other
    fixture's samples sit exactly on their true centres.

    Here they do not.  The two swapped windows' samples are placed so that the
    aggregate |<cv2> - assigned centre| is LOWER under the map's (wrong) order
    than under the table's -- the heterogeneous-pull situation, staged
    deterministically: window 1's samples sit near centre 2 and window 2's sit
    far past centre 1.  The table is still the record of what ran, so the repair
    must go through and simply report the number."""
    centers = _ladder(5)                                   # -1.0 -0.5 0.0 0.5 1.0
    phase, rows = _phase(tmp_path, _swap(centers, 1), centers)
    # Local windows 1 and 2 are the swapped pair; their true centres are -0.5
    # and 0.0, the map assigns them 0.0 and -0.5.
    ids, cv2 = _samples([-1.0, 0.0, -3.0, 0.5, 1.0])

    out_rows, notes = _run(phase, rows, ids, cv2)

    assert _mapping(out_rows) == {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    assert len(_repair_notes(notes)) == 1
    # ...and the fixture really is the adverse one: the map's order scores better.
    from gareus.mbar_analysis.loaders_adaptive import (
        _map_cv2_fingerprint_deviation, _per_window_cv2_means)
    means = _per_window_cv2_means(ids, cv2)
    assert (_map_cv2_fingerprint_deviation(out_rows, means)
            > _map_cv2_fingerprint_deviation(rows, means))


def test_the_corroboration_cannot_raise_a_note_on_its_own():
    """Executable statement of the design rule this file's docstring explains:
    the sampled-CV2 helper renders text, it does not decide anything.  Handed
    the very shape that used to fire -- a window 20 restraint widths off its own
    centre -- with nothing flagged, it produces nothing."""
    assert _map_cv2_corroboration([], {0: 99.0, 1: -99.0}) == ""


def test_the_corroboration_is_omitted_rather_than_faked_when_cv2_is_unusable():
    """A flagged window with no finite sampled cv2 (or a non-finite centre on
    either side) contributes no evidence and must not be counted as agreeing
    with anything.  With no comparable window at all the whole clause is
    dropped, so the note never claims a corroboration it does not have."""
    bad = [(0, (0.0, float("nan")), (0.0, 1.0)), (1, (0.0, 2.0), (0.0, 3.0))]
    assert _map_cv2_corroboration(bad, {}) == ""
    assert _map_cv2_corroboration(bad, {0: 1.0}) == ""
    got = _map_cv2_corroboration(bad, {1: 2.9})
    assert "1 of 1 comparable window(s)" in got
    assert "local window 0" not in got


# ---------------------------------------------------------------------------
# Write side: repair_epoch_window_map_from_surviving_windows' equal-count branch
# ---------------------------------------------------------------------------


def _writer_phase(tmp_path: Path, map_centers: list[float], table_centers: list[float]) -> Path:
    rows = [{"epoch_window": i, "state_id": i, "primary_center": 0.0, "secondary_center": c2}
            for i, c2 in enumerate(map_centers)]
    _write_map(tmp_path / "epoch_window_map.csv", rows)
    _write_window_table(tmp_path / "umbrella_explicit_windows.csv", table_centers)
    return tmp_path


def test_writer_confirms_an_equal_count_map_that_really_matches(tmp_path):
    """Non-firing twin: the ordinary phase.  Still 'consistent', still untouched,
    and now it says the centres were actually looked at."""
    centers = _ladder(6)
    _writer_phase(tmp_path, centers, centers)
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "consistent"
    assert summary["centers_verified"] is True
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original


def test_writer_flags_an_equal_count_map_describing_a_different_window_set(tmp_path, capsys):
    """The residual, at write time.  Both attempts dropped one window, but not
    the same one, so the counts agree and the membership does not."""
    ladder = _ladder(7)
    map_centers = ladder[:3] + ladder[4:]     # attempt 1 dropped window 3
    table_centers = ladder[:5] + ladder[6:]   # attempt 2 dropped window 5
    _writer_phase(tmp_path, map_centers, table_centers)
    original = (tmp_path / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "inconsistent_equal_count_centers_do_not_match"
    assert summary["centers_verified"] is False
    # Never rewritten: with a row missing for a window that really ran there is
    # nothing to re-derive the correct mapping from.
    assert (tmp_path / "epoch_window_map.csv").read_bytes() == original
    out = capsys.readouterr().out
    assert "DIFFERENT window set of the same size" in out


def test_writer_abstains_rather_than_warns_when_centres_are_duplicated(tmp_path):
    """Duplicate (primary, secondary) centres are legal and make centre matching
    ambiguous about which of two identical rows a survivor is.  The equal-count
    branch must report the count agreement without claiming a verification it
    could not perform -- and must not emit the mismatch warning."""
    centers = [-1.0, -0.5, -0.5, 0.0]
    _writer_phase(tmp_path, centers, centers)

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "consistent"
    assert summary["centers_verified"] is False
    assert summary["centers_check"] == "skipped_duplicate_centers"


def test_writer_still_rewrites_a_genuinely_longer_map(tmp_path):
    """Guard against the new equal-count branch swallowing the existing repair:
    a map with MORE rows than windows must still be rewritten as before."""
    ladder = _ladder(6)
    _writer_phase(tmp_path, ladder, ladder[:3] + ladder[4:])

    summary = repair_epoch_window_map_from_surviving_windows(tmp_path)

    assert summary["status"] == "rewritten"
    assert summary["dropped_state_ids"] == [3]


# ---------------------------------------------------------------------------
# The COMPOUND stale map: more rows than windows AND those windows out of order
#
# Lives in this file rather than with the drop-rewrite tests because it is the
# same question this file is about -- what the phase's own two records say about
# each other's membership and order -- asked of the one shape neither matcher
# could see.  `_match_map_rows_to_real_windows` is order-preserving, so a
# permutation defeats it; `_permute_map_rows_onto_window_table` requires every
# map row to be consumed, so the extra rows defeat it.  Before this, the load
# side fell through both onto the drop-record fallback, which removes rows by
# state_id and never looks at a centre: it kept the map's own contradicted row
# order and returned the result under a REPAIRED note.  The write side, asked
# the same question about the same two files, refused.  Two sides disagreeing
# about one input is the drift these paired checks exist to remove; a wrong
# mapping announced as REPAIRED is the worst outcome available here.
# ---------------------------------------------------------------------------

# state_id 20+i sits at ladder centre i, the real chignolin_6 numbering.
_COMPOUND_LADDER = _ladder(5)
# The phase really ran windows 0, 1, 3, 4 (window 2 was dropped post-pull) and
# its map lists all five, in the order a registry whose active-state order is
# not the window-array order produces.
_COMPOUND_MAP_ORDER = [4, 0, 2, 1, 3]
_COMPOUND_REAL_WINDOWS = [0, 1, 3, 4]
_COMPOUND_CORRECT_MAPPING = {0: 20, 1: 21, 2: 23, 3: 24}
# What the drop-record fallback produces from the same fixture: the right rows
# kept, in the map's own (wrong) order.  Not a hypothetical -- this is what the
# load side returned, labelled REPAIRED, before the compound branch existed.
_COMPOUND_WRONG_MAPPING = {0: 24, 1: 20, 2: 21, 3: 23}


def _compound_phase(root: Path, map_windows: list[int], table_windows: list[int], *,
                    dropped_positions: list[int] | None = None,
                    table_centers: list | None = None) -> tuple[Path, list[dict]]:
    """A phase whose map lists `map_windows` (in that file order) and whose
    surviving-window table lists `table_windows`, both drawn from
    `_COMPOUND_LADDER` with state_id 20+window.

    `dropped_positions` writes the post-pull drop record -- pre-drop local
    positions, exactly as `gareus/production.py` records them.  It is what makes
    the fixture dangerous rather than merely broken: without it the loader has no
    fallback to reach and fails closed for a reason that has nothing to do with
    the defect under test.
    """
    phase = root / "final" / "baseline"
    rows = [{"epoch_window": i, "state_id": 20 + w, "primary_center": 0.0,
             "secondary_center": _COMPOUND_LADDER[w]}
            for i, w in enumerate(map_windows)]
    _write_map(phase / "epoch_window_map.csv", rows)
    centers = ([_COMPOUND_LADDER[w] for w in table_windows]
               if table_centers is None else table_centers)
    _write_window_table(phase / "umbrella_explicit_windows.csv", centers)
    _write_recorded_window_count(phase, len(centers))
    if dropped_positions is not None:
        (phase / "gareus_metadata.json").write_text(json.dumps(
            {"window_metadata": {"dropped_post_pull_bad_windows": list(dropped_positions)}}))
    with (phase / "epoch_window_map.csv").open(newline="") as handle:
        return phase, list(csv.DictReader(handle))


def _map_file_mapping(phase: Path) -> dict:
    """local window -> state_id, read back off the map file on disk.

    The write side's answer to the same question `_mapping` asks of the load
    side's returned rows, so the two can be compared directly.
    """
    with (phase / "epoch_window_map.csv").open(newline="") as handle:
        return {int(r["epoch_window"]): int(r["state_id"]) for r in csv.DictReader(handle)}


def test_the_loader_repairs_a_compound_map_onto_the_window_table(tmp_path):
    """The defect, at load time, with the fallback that used to swallow it fully
    armed: a drop record that reconciles by count and no sampled cv2 to
    cross-check against.

    Asserted on the MAPPING, not the note.  The drop-record fallback also returns
    four rows carrying the right four state_ids -- it removes exactly the row the
    record names -- so any assertion on row count, state_id membership or note
    text passes just as well for the wrong answer.  The only thing that separates
    them is which local window each state_id ends up at.
    """
    phase, rows = _compound_phase(tmp_path, _COMPOUND_MAP_ORDER, _COMPOUND_REAL_WINDOWS,
                                  dropped_positions=[2])

    repaired, notes = _run(phase, rows, None, None)

    real = [r for r in repaired if int(r["epoch_window"]) < 1_000_000]
    assert _mapping(real) == _COMPOUND_CORRECT_MAPPING
    assert _mapping(real) != _COMPOUND_WRONG_MAPPING
    assert len(_repair_notes(notes)) == 1
    # The dropped state keeps a row, parked out of band, so its epoch-native
    # window params still back its MBAR bias column.
    assert [int(r["state_id"]) for r in repaired if int(r["epoch_window"]) >= 1_000_000] == [22]


def test_the_loader_fails_closed_on_a_compound_map_it_cannot_derive(tmp_path):
    """The direction that must not move.  Window 4's centre appears nowhere in the
    map, so its state_id is recorded in neither artifact and nothing can invent
    one -- while the drop record still reconciles by count and would happily hand
    back four rows in the map's own order under a REPAIRED note.

    Fails closed through `_fail`, so the documented
    `GAREUS_ALLOW_STALE_WINDOW_MAP` escape hatch still applies; a bare raise here
    would take that away.
    """
    phase, rows = _compound_phase(tmp_path, [3, 0, 2, 1], [0, 1, 4],
                                  dropped_positions=[2])

    with pytest.raises(ValueError) as excinfo:
        _run(phase, rows, None, None)

    message = str(excinfo.value)
    assert "no row of the map carries those centres" in message
    assert "drop record is deliberately not used as a fallback" in message


def test_the_loader_refuses_a_compound_map_whose_row_is_ambiguous(tmp_path):
    """Two map rows carrying one real window's centres.  With more rows than
    windows, one of them may be a state this phase dropped and never sampled, so
    resolving the tie arbitrarily can hand a real window's samples to a state that
    never ran -- the chignolin_6 failure itself.  Refuse instead.

    The equal-count reordering matcher deliberately does NOT refuse this (there,
    both candidates are real windows of the phase, carrying identical restraint
    columns, so only the pooling differs).  That asymmetry is the reason the two
    matchers are separate functions.
    """
    # Local windows 0, 1, 2 ran at ladder centres 3, 1, 0 -- an order the map does
    # not share, so the order-preserving matcher fails and the compound matcher is
    # the one that answers.  Drop record [2, 4] removes two rows and reconciles
    # the count, so the old fallback was fully reachable on this fixture too.
    phase, rows = _compound_phase(tmp_path, [0, 1, 2, 3], [3, 1, 0],
                                  dropped_positions=[2, 4])
    # A second row at window 3's centre, carrying a different state_id: the
    # ambiguity, added after the fixture so the map is 5 rows for 3 windows.
    rows.append({"epoch_window": "4", "state_id": "99", "primary_center": "0.0",
                 "secondary_center": str(_COMPOUND_LADDER[3])})
    _write_map(phase / "epoch_window_map.csv", rows)

    with pytest.raises(ValueError) as excinfo:
        _run(phase, rows, None, None)

    assert "match 2 map rows" in str(excinfo.value)
    assert "not derivable" in str(excinfo.value)


def test_an_ordinary_longer_map_still_takes_the_order_preserving_path(tmp_path):
    """The non-firing twin, and the shape every real affected run on disk has:
    more rows than windows, but in order.  It must still be repaired, in the
    map's own row order, and must not be diverted into the compound branch --
    measured on disk, all 8 phases under RUNS/ that need a row-count repair are
    this shape and none is compound.

    Which matcher supplies the rows changed and this must not notice: the forced
    matcher now decides every longer map and the order-preserving one only says
    whether the answer is in the map's own order (see
    `test_the_loader_refuses_an_in_order_longer_map_whose_first_row_is_an_
    ambiguous_phantom` for what that closed).  Both pick the identical rows here,
    which is the whole point of the twin."""
    phase, rows = _compound_phase(tmp_path, [0, 1, 2, 3, 4], _COMPOUND_REAL_WINDOWS)

    repaired, notes = _run(phase, rows, None, None)

    real = [r for r in repaired if int(r["epoch_window"]) < 1_000_000]
    assert _mapping(real) == _COMPOUND_CORRECT_MAPPING
    assert len(_repair_notes(notes)) == 1


# ---------------------------------------------------------------------------
# The IN-ORDER half of the same defect: a LONGER map whose surviving rows are in
# the map's own order, with a phantom row duplicating a real window's centres
#
# `_match_map_rows_to_real_windows` is greedy -- it takes the first row whose
# centres match -- and it used to be what decided whether a longer map was
# derivable, with the forced matcher above reached only as its fallback.  So a
# phantom row (one for a state the phase dropped and never sampled) carrying real
# window 0's centres and sorting FIRST was consumed for window 0, every later
# window still matched, and the loader handed back that mapping labelled
# REPAIRED.  No reordering is involved anywhere, which is exactly why the
# compound branch never saw it: the map is in order, it is merely ambiguous about
# which of two identical-centre rows the extra one is.
#
# Why this is the original bug rather than the equal-count path's survivable
# degradation -- the asymmetry that decides why one path was tightened and the
# other deliberately was not: with as many rows as windows both candidates are
# real windows OF THE PHASE carrying identical restraint columns, so an arbitrary
# pick mis-pools two SAMPLED states and changes nothing else.  With more rows than
# windows one candidate can be a state that never ran, so the pick hands a real
# window's samples to a state with none of its own.  That is chignolin_6's state
# 20 exactly: a phantom duplicate of 21, given 861,547 samples and its own f_k.
#
# Both orderings of a longer map therefore go through the one forced matcher now.
# Measured cost on real data: none -- over all 152 maps under RUNS/, each of the 8
# phases needing this repair has its selection forced and gets the identical rows.
# ---------------------------------------------------------------------------

# The real windows: ladder centres 0, 1, 2, which the map also lists in that
# order, so the greedy matcher SUCCEEDS on this fixture and the forced matcher is
# the only thing that can refuse.
_PHANTOM_FIRST_TABLE = [_COMPOUND_LADDER[0], _COMPOUND_LADDER[1], _COMPOUND_LADDER[2]]
_PHANTOM_FIRST_CORRECT_MAPPING = {0: 20, 1: 21, 2: 22}
# What greedy returns instead, and returned before this was closed: real window
# 0's samples attributed to state 99, which this phase dropped and never sampled.
_PHANTOM_FIRST_WRONG_MAPPING = {0: 99, 1: 21, 2: 22}


def _phantom_first_phase(root: Path, first_row_center: float, *,
                         dropped_positions: list[int] | None = None) -> tuple[Path, list[dict]]:
    """3 real windows, 4 map rows, the extra row FIRST and carrying state 99.

    `first_row_center` is the ONLY difference between the firing fixture (real
    window 0's own centre -- ambiguous) and its non-firing twin (a centre no real
    window ran at -- an ordinary longer map).  Everything else is shared, so the
    twin cannot pass for some unrelated reason.

    First by ``epoch_window`` VALUE, not by line order, because
    `_sorted_map_rows` re-sorts before any matching happens: a phantom sorting
    LAST is never reached by the greedy matcher for window 0, and a fixture built
    that way would pass on the unfixed code -- a "would have caught it" test that
    would not have.  The tests assert that ordering rather than trusting it.
    """
    phase = root / "final" / "baseline"
    rows = [
        {"epoch_window": 0, "state_id": 99, "primary_center": 0.0,
         "secondary_center": first_row_center},
        {"epoch_window": 1, "state_id": 20, "primary_center": 0.0,
         "secondary_center": _COMPOUND_LADDER[0]},
        {"epoch_window": 2, "state_id": 21, "primary_center": 0.0,
         "secondary_center": _COMPOUND_LADDER[1]},
        {"epoch_window": 3, "state_id": 22, "primary_center": 0.0,
         "secondary_center": _COMPOUND_LADDER[2]},
    ]
    _write_map(phase / "epoch_window_map.csv", rows)
    _write_window_table(phase / "umbrella_explicit_windows.csv", _PHANTOM_FIRST_TABLE)
    _write_recorded_window_count(phase, len(_PHANTOM_FIRST_TABLE))
    if dropped_positions is not None:
        (phase / "gareus_metadata.json").write_text(json.dumps(
            {"window_metadata": {"dropped_post_pull_bad_windows": list(dropped_positions)}}))
    with (phase / "epoch_window_map.csv").open(newline="") as handle:
        return phase, list(csv.DictReader(handle))


def test_the_loader_refuses_an_in_order_longer_map_whose_first_row_is_an_ambiguous_phantom(tmp_path):
    """The defect: a longer map that is IN ORDER and ambiguous.

    Greedy consumes the phantom for window 0 and succeeds, so before this the
    loader returned `_PHANTOM_FIRST_WRONG_MAPPING` under a REPAIRED note -- a real
    window's 3 windows' worth of samples attributed to a state that never ran,
    announced as a successful repair.  Nothing in the map's row order is wrong, so
    the compound branch's refusal never applied.

    Fails closed through `_fail`, so the documented
    `GAREUS_ALLOW_STALE_WINDOW_MAP` escape hatch still applies; a bare raise here
    would take that away.
    """
    phase, rows = _phantom_first_phase(tmp_path, _COMPOUND_LADDER[0])

    # The fixture's teeth, asserted rather than assumed: the phantom is what the
    # greedy matcher meets FIRST after `_sorted_map_rows`.
    assert int(_sorted_map_rows(rows)[0]["state_id"]) == 99

    with pytest.raises(ValueError) as excinfo:
        _run(phase, rows, None, None)

    message = str(excinfo.value)
    # The refusal must not misdescribe its own evidence.  The map IS in order
    # here -- claiming its rows "do not carry those windows' restraint centres in
    # order", the compound branch's wording, would be the same failure shape one
    # step removed.
    assert "do carry those windows' restraint centres in order" in message
    assert "match 2 map rows (state_id [99, 20])" in message
    assert "a state this phase dropped and never sampled" in message
    assert "GAREUS_ALLOW_STALE_WINDOW_MAP" in message


def test_the_in_order_ambiguity_is_not_resolved_by_the_drop_record(monkeypatch, tmp_path):
    """The fallback stays shut, and the escape hatch stays open.

    Deliberate strictness, stated plainly: the drop record here names pre-drop
    position 0, which resolves to state 99, and removing that row WOULD reconcile
    the count and WOULD give the right answer on this fixture.  It is still not
    consulted, because it identifies rows by position in the very map whose
    reading is in question, can be inherited from a sibling sub-run
    (`_dropped_state_ids_for_phase`), and is only ever count-checked -- so leaning
    on it to break a centre tie is deciding by fiat.  The write side abstains on
    the same input for the same reason.

    The cost of that choice is one inspection run under
    `GAREUS_ALLOW_STALE_WINDOW_MAP`, which is asserted here rather than claimed:
    the map loads unrepaired, with the loud stale-loaded note.
    """
    phase, rows = _phantom_first_phase(tmp_path, _COMPOUND_LADDER[0], dropped_positions=[0])

    with pytest.raises(ValueError) as excinfo:
        _run(phase, rows, None, None)
    assert "drop record is deliberately not used as a fallback" in str(excinfo.value)

    monkeypatch.setenv("GAREUS_ALLOW_STALE_WINDOW_MAP", "1")
    loaded, notes = _run(phase, rows, None, None)

    assert _mapping(loaded) == _mapping(rows)          # loaded stale, as asked
    assert [window_map_note_kind(n) for n in notes] == [MAP_NOTE_STALE_LOADED]


def test_both_sides_refuse_an_in_order_longer_map_with_an_ambiguous_phantom(tmp_path):
    """The third input class in its in-order shape, so the both-sides agreement
    claim covers it too.

    Different mechanisms, same outcome, which is why the outcome is what is
    asserted: the write side's duplicate-centre abstain fires before its own
    in-order matcher runs, while the load side's forced matcher is what refuses.
    What must not differ is that neither answers -- and until this round the load
    side did answer, wrongly, on exactly this input.
    """
    write_side, _ = _phantom_first_phase(tmp_path / "write", _COMPOUND_LADDER[0],
                                         dropped_positions=[0])
    load_side, rows = _phantom_first_phase(tmp_path / "load", _COMPOUND_LADDER[0],
                                           dropped_positions=[0])
    original = (write_side / "epoch_window_map.csv").read_bytes()
    assert original == (load_side / "epoch_window_map.csv").read_bytes()

    write_summary = repair_epoch_window_map_from_surviving_windows(write_side)
    with pytest.raises(ValueError):
        _run(load_side, rows, None, None)

    assert write_summary["status"] == "skipped_duplicate_centers"
    assert (write_side / "epoch_window_map.csv").read_bytes() == original


def test_an_in_order_longer_map_with_a_distinct_extra_row_still_repairs(tmp_path):
    """The non-firing twin: the same fixture with the extra row moved to a centre
    no real window ran at, which is an ordinary longer map and the shape all 8
    affected phases under RUNS/ have.

    It must repair exactly as before -- same mapping, same REPAIRED kind, and the
    in-order wording, not the compound branch's "matched onto the map OUT OF
    ORDER".  A fix that closes the ambiguous case by refusing longer maps in
    general would pass the test above and fail this one.

    The partition is asserted too: survivors plus phantoms must still be the map's
    own rows with nothing dropped and no state_id counted twice, because the
    dropped state's row has to survive (parked out of band) to keep backing its
    own MBAR bias column.
    """
    phase, rows = _phantom_first_phase(tmp_path, _COMPOUND_LADDER[3])

    repaired, notes = _run(phase, rows, None, None)

    real = [r for r in repaired if int(r["epoch_window"]) < 1_000_000]
    assert _mapping(real) == _PHANTOM_FIRST_CORRECT_MAPPING
    assert _mapping(real) != _PHANTOM_FIRST_WRONG_MAPPING
    assert [window_map_note_kind(n) for n in notes] == [MAP_NOTE_REPAIRED]
    assert "OUT OF ORDER" not in notes[0]
    assert [int(r["state_id"]) for r in repaired if int(r["epoch_window"]) >= 1_000_000] == [99]
    assert (sorted(int(r["state_id"]) for r in repaired)
            == sorted(int(r["state_id"]) for r in rows))


def test_the_writer_repairs_the_same_compound_map_in_place(tmp_path):
    """The write side of the same defect, on the same fixture shape: it used to
    refuse with `skipped_centers_do_not_match` while the load side went on to
    'repair' it wrongly."""
    phase, _rows = _compound_phase(tmp_path, _COMPOUND_MAP_ORDER, _COMPOUND_REAL_WINDOWS)

    summary = repair_epoch_window_map_from_surviving_windows(phase)

    assert summary["status"] == "rewritten_reordered_and_pruned"
    assert summary["reordered_onto_window_table"] is True
    assert summary["dropped_state_ids"] == [22]
    assert _map_file_mapping(phase) == _COMPOUND_CORRECT_MAPPING
    # Rows were removed, so unlike a pure reorder this one has a real drop key
    # and belongs in the ledger's `applied` list.
    from gareus.production import _read_epoch_window_map_rewrite_ledger
    ledger = _read_epoch_window_map_rewrite_ledger(phase)
    assert len(ledger) == 1
    assert ledger[0]["dropped_window_indices"] == [2]


def test_the_writers_compound_repair_is_idempotent(tmp_path):
    """A second call finds the map already in the table's order with the right
    row count, takes the equal-count branch and writes nothing -- so the repair
    cannot compact a map twice."""
    phase, _rows = _compound_phase(tmp_path, _COMPOUND_MAP_ORDER, _COMPOUND_REAL_WINDOWS)
    first = repair_epoch_window_map_from_surviving_windows(phase)
    after = (phase / "epoch_window_map.csv").read_bytes()

    second = repair_epoch_window_map_from_surviving_windows(phase)

    assert first["status"] == "rewritten_reordered_and_pruned"
    assert second["status"] == "consistent"
    assert second["centers_verified"] is True
    assert (phase / "epoch_window_map.csv").read_bytes() == after


def test_the_writer_still_refuses_a_longer_map_missing_a_window_that_ran(tmp_path):
    """The write side's non-firing twin of the fail-closed case: the status and
    the untouched file are unchanged from before the compound branch existed."""
    phase, _rows = _compound_phase(tmp_path, [3, 0, 2, 1], [0, 1, 4])
    original = (phase / "epoch_window_map.csv").read_bytes()

    summary = repair_epoch_window_map_from_surviving_windows(phase)

    assert summary["status"] == "skipped_centers_do_not_match"
    assert "no row of the map carries those centres" in summary["select_problem"]
    assert (phase / "epoch_window_map.csv").read_bytes() == original


def test_both_sides_derive_the_same_mapping_for_a_compound_map(tmp_path):
    """The test the round-8 divergence would have failed.

    Not "both sides succeed" and not "both sides report the same status" -- the
    two functions produce different artifacts (one rewrites a file, one returns
    rows) and the only thing that can be compared is the answer they encode:
    which umbrella state each local window's samples are attributed to.  Both are
    run, for real, on byte-identical copies of one phase directory.

    A previous round asserted this agreement by reading the two implementations;
    measured, they did not agree -- the write side refused and the load side
    returned a wrong mapping labelled REPAIRED.
    """
    write_side, _ = _compound_phase(tmp_path / "write", _COMPOUND_MAP_ORDER,
                                    _COMPOUND_REAL_WINDOWS, dropped_positions=[2])
    load_side, rows = _compound_phase(tmp_path / "load", _COMPOUND_MAP_ORDER,
                                      _COMPOUND_REAL_WINDOWS, dropped_positions=[2])
    assert ((write_side / "epoch_window_map.csv").read_bytes()
            == (load_side / "epoch_window_map.csv").read_bytes())

    write_summary = repair_epoch_window_map_from_surviving_windows(write_side)
    loaded, _notes = _run(load_side, rows, None, None)

    from_writer = _map_file_mapping(write_side)
    from_loader = _mapping([r for r in loaded if int(r["epoch_window"]) < 1_000_000])
    assert write_summary["status"] == "rewritten_reordered_and_pruned"
    assert from_writer == from_loader == _COMPOUND_CORRECT_MAPPING


def test_both_sides_refuse_the_same_underivable_compound_map(tmp_path):
    """The other half of the agreement, which matters as much: neither side may
    answer a question the artifacts cannot settle, and neither may answer it
    while the other refuses."""
    write_side, _ = _compound_phase(tmp_path / "write", [3, 0, 2, 1], [0, 1, 4],
                                    dropped_positions=[2])
    load_side, rows = _compound_phase(tmp_path / "load", [3, 0, 2, 1], [0, 1, 4],
                                      dropped_positions=[2])
    original = (write_side / "epoch_window_map.csv").read_bytes()

    write_summary = repair_epoch_window_map_from_surviving_windows(write_side)
    with pytest.raises(ValueError):
        _run(load_side, rows, None, None)

    assert write_summary["status"] == "skipped_centers_do_not_match"
    assert (write_side / "epoch_window_map.csv").read_bytes() == original
    assert ((load_side / "epoch_window_map.csv").read_bytes()
            == (write_side / "epoch_window_map.csv").read_bytes())


def test_both_sides_refuse_a_compound_map_with_an_ambiguous_row(tmp_path):
    """The third input class, so the agreement claim covers all of them: a
    compound map one of whose real windows has two candidate rows.

    The two sides refuse it by different mechanisms, which is fine and is the
    point of asserting the outcome rather than the status -- the write side's
    duplicate-centre abstain fires first (it runs before any matching, and
    duplicate centres are legal), while the load side's longer path has no such
    pre-check and the forced matcher is what refuses.  What must not differ is
    that neither answers.
    """
    write_side, _ = _compound_phase(tmp_path / "write", [0, 1, 2, 3], [3, 1, 0],
                                    dropped_positions=[2, 4])
    load_side, rows = _compound_phase(tmp_path / "load", [0, 1, 2, 3], [3, 1, 0],
                                      dropped_positions=[2, 4])
    for phase in (write_side, load_side):
        with (phase / "epoch_window_map.csv").open(newline="") as handle:
            extended = list(csv.DictReader(handle))
        extended.append({"epoch_window": "4", "state_id": "99", "primary_center": "0.0",
                         "secondary_center": str(_COMPOUND_LADDER[3])})
        _write_map(phase / "epoch_window_map.csv", extended)
    original = (write_side / "epoch_window_map.csv").read_bytes()
    with (load_side / "epoch_window_map.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    write_summary = repair_epoch_window_map_from_surviving_windows(write_side)
    with pytest.raises(ValueError):
        _run(load_side, rows, None, None)

    assert write_summary["status"] == "skipped_duplicate_centers"
    assert (write_side / "epoch_window_map.csv").read_bytes() == original

def test_the_compound_repair_survives_the_sampled_cv2_veto_it_now_faces(tmp_path):
    """A compound repair goes through the mismatch path's fingerprint veto, unlike
    the equal-count reordering repair which quotes the same number as evidence
    only.  With samples sitting on their own restraint centres -- the correct
    mapping's signature -- the veto must not fire.

    Documented consequence, not a claim that the veto is safe in general: a
    correct compound repair whose fingerprint happens to worsen is REFUSED rather
    than published, which costs an inspection run under the escape hatch.  The
    opposite error costs a wrong free energy.

    And measured on the reverted code, because it sizes the finding: with the
    compound branch removed this test fails on the MAPPING, not on the veto --
    the old drop-record fallback returned `_COMPOUND_WRONG_MAPPING` under a
    REPAIRED note here too, with the sampled cv2 fully available.  The
    fingerprint's mean-deviation comparison happens not to separate the wrong
    mapping from the right one on this fixture, so the dangerous outcome did NOT
    require the fingerprint to be missing.
    """
    phase, rows = _compound_phase(tmp_path, _COMPOUND_MAP_ORDER, _COMPOUND_REAL_WINDOWS,
                                  dropped_positions=[2])
    window_ids, cv2 = _samples([_COMPOUND_LADDER[w] for w in _COMPOUND_REAL_WINDOWS])

    repaired, notes = _run(phase, rows, window_ids, cv2)

    real = [r for r in repaired if int(r["epoch_window"]) < 1_000_000]
    assert _mapping(real) == _COMPOUND_CORRECT_MAPPING
    assert "Sampled-cv2 fingerprint confirms it" in _repair_notes(notes)[0]


def test_the_survivor_renumbering_is_unchanged_for_an_unparseable_epoch_window(tmp_path):
    """The compound branch also changed how survivors are renumbered -- from
    `_renumber_epoch_window_rows` (which re-sorts by the stale `epoch_window`,
    and would put back exactly the order the compound matcher just corrected) to
    `_renumber_rows_in_order` (which keeps the matcher's chosen order).

    That is a no-op for the two order-preserving sources only because they select
    a SUBSEQUENCE of an already-sorted list, so re-sorting it is the identity.
    The awkward input for that argument is a row whose `epoch_window` does not
    parse: `_sorted_map_rows` keys those `(1, 0, i)` and sorts them after every
    parseable row, so a survivor list containing one is the case where "already
    sorted" is least obvious.  Measured here on the real functions rather than
    reasoned about, because reasoning instead of measuring is the failure this
    whole round exists to correct.
    """
    from gareus.mbar_analysis.loaders_adaptive import (
        _match_map_rows_to_real_windows, _phase_real_window_centers,
        _renumber_epoch_window_rows, _renumber_rows_in_order, _sorted_map_rows)

    # Sorted map order is [ew 0 -> state 20 @ c0, ew 1 -> state 22 @ c1,
    # ew 'x' -> state 21 @ c2]; windows 0 and 1 really ran, at c0 and c2.
    phase = tmp_path / "final" / "baseline"
    _write_map(phase / "epoch_window_map.csv", [
        {"epoch_window": 0, "state_id": 20, "primary_center": 0.0,
         "secondary_center": _COMPOUND_LADDER[0]},
        {"epoch_window": "x", "state_id": 21, "primary_center": 0.0,
         "secondary_center": _COMPOUND_LADDER[2]},
        {"epoch_window": 1, "state_id": 22, "primary_center": 0.0,
         "secondary_center": _COMPOUND_LADDER[1]},
    ])
    _write_window_table(phase / "umbrella_explicit_windows.csv",
                        [_COMPOUND_LADDER[0], _COMPOUND_LADDER[2]])
    _write_recorded_window_count(phase, 2)
    with (phase / "epoch_window_map.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    repaired, notes = _run(phase, rows, None, None)

    real = [r for r in repaired if int(r["epoch_window"]) < 1_000_000]
    assert _mapping(real) == {0: 20, 1: 21}
    assert len(_repair_notes(notes)) == 1
    # The identity itself, on the selection the real matcher produced: the two
    # renumberers agree, so the switch cannot have moved this path's answer.
    picked = _match_map_rows_to_real_windows(_sorted_map_rows(rows),
                                             _phase_real_window_centers(phase))
    assert picked is not None and len(picked) == 2
    assert _renumber_rows_in_order(picked) == _renumber_epoch_window_rows(picked)
