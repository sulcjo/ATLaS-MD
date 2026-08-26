"""The stale-`epoch_window_map.csv` guard, at the three consumers that are not
the union-Parquet loader.

Background (real run `RUNS/chignolin_6`, writeup in
`docs/chignolin_6_low_ess_root_cause.md`): `--us-auto-drop-bad-windows` prunes
umbrella windows post-pull and `gareus/production.py` renumbers every
window-indexed array around the survivors (0..N-1), but the phase's own
`epoch_window_map.csv` -- an identity map the registry wrote earlier over all
*active* states -- was never rewritten. Every local window index at or after
the first dropped one then names the wrong state (1.82M of 12.1M samples on
that run). `gareus/mbar_analysis/loaders_union_parquet.py` was taught to
validate/repair the map; these three were not:

* `compare_gareus_runs.load_npz_adaptive_union` -- a second, parallel union
  implementation for legacy NPZ-only adaptive runs, reached from `load_run`
  when `prod_dir_of` raises. It publishes comparison PMF numbers, so it fails
  closed exactly like the Parquet loader.
* `gareus.mbar_analysis.loaders.load_data`'s `adaptive_union_mbar.npz` branch
  -- the npz stores `sampled_state_ids`, i.e. the mapping the *driver* already
  applied at run time, so nothing about it can be recomputed (let alone
  repaired) from the npz. What can be checked is the maps it was built from.
* `plot_adaptive_diagnostics.py` -- figures, so it repairs where it can and
  stamps a loud warning where it cannot, rather than refusing to draw.

Fixture shape throughout, modelled on chignolin_6 at 1/9 scale: a 5-state
registry, a phase whose map still lists all 5 states, 3 windows actually run,
pre-drop local windows [1, 3] dropped. The correct local->state mapping is
therefore ``{0: 0, 1: 2, 2: 4}`` and the stale one is ``{0: 0, 1: 1, 2: 2}``.
Both are spelled out literally below rather than recomputed from the code
under test.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')

from gareus.mbar_analysis.loaders import (  # noqa: E402
    check_union_npz_window_map_provenance,
    figure_epoch_window_map_rows,
    load_data,
)

_STALE_ENV = 'GAREUS_ALLOW_STALE_WINDOW_MAP'

# Registry: 5 states, distinct primary AND secondary centres so a shifted
# mapping is visible in both channels.
_PRIMARY = [1.0, 1.1, 1.2, 1.3, 1.4]
_SECONDARY = [-1.0, -0.5, 0.0, 0.5, 1.0]

# The three windows that really ran are states 0, 2 and 4 (1 and 3 dropped).
_SURVIVOR_STATES = [0, 2, 4]
_REPAIRED_MAPPING = {0: 0, 1: 2, 2: 4}
_STALE_MAPPING = {0: 0, 1: 1, 2: 2}


# --- fixture builders -------------------------------------------------------

def _write_registry(ap: Path) -> None:
    ap.mkdir(parents=True, exist_ok=True)
    with (ap / 'final_registry_used_for_mbar.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['state_id', 'usable_for_mbar', 'primary_center',
                                          'primary_k', 'secondary_center', 'secondary_k'])
        w.writeheader()
        for sid in range(5):
            w.writerow({'state_id': sid, 'usable_for_mbar': 'True',
                        'primary_center': repr(_PRIMARY[sid]), 'primary_k': '10.0',
                        'secondary_center': repr(_SECONDARY[sid]), 'secondary_k': '5.0'})


def _write_map(phase: Path, state_ids: list) -> None:
    """epoch_window_map.csv listing `state_ids` at local windows 0..N-1."""
    phase.mkdir(parents=True, exist_ok=True)
    with (phase / 'epoch_window_map.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch_window', 'state_id', 'primary_center',
                                          'primary_k', 'secondary_center', 'secondary_k'])
        w.writeheader()
        for ew, sid in enumerate(state_ids):
            w.writerow({'epoch_window': ew, 'state_id': sid,
                        'primary_center': repr(_PRIMARY[sid]), 'primary_k': '10.0',
                        'secondary_center': repr(_SECONDARY[sid]), 'secondary_k': '5.0'})


def _write_window_table(phase: Path, state_ids: list) -> None:
    """The phase's post-drop surviving-window table, listing `state_ids`' centres.

    `umbrella_explicit_windows.csv` is the phase's own record of what it really
    ran, and it is the *other* half of the equal-row-count membership check: the
    map is compared against it row for row.
    """
    phase.mkdir(parents=True, exist_ok=True)
    with (phase / 'umbrella_explicit_windows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['window', 'primary_center', 'primary_k',
                                          'secondary_cv_center', 'secondary_cv_k_kcal_mol'])
        w.writeheader()
        for ew, sid in enumerate(state_ids):
            w.writerow({'window': ew, 'primary_center': repr(_PRIMARY[sid]), 'primary_k': '10.0',
                        'secondary_cv_center': repr(_SECONDARY[sid]),
                        'secondary_cv_k_kcal_mol': '5.0'})


def _write_drop_record(phase: Path, dropped: list, n_windows: int) -> None:
    (phase / 'gareus_metadata.json').write_text(
        json.dumps({'window_metadata': {'dropped_post_pull_bad_windows': list(dropped)}}),
        encoding='utf-8')
    (phase / 'analysis_metadata_validation.json').write_text(
        json.dumps({'n_windows': int(n_windows), 'status': 'ok'}), encoding='utf-8')


def _write_epoch_npz(phase: Path) -> None:
    """Six samples: two in each of the three windows that really ran.

    `secondary_cv` is each sample's true restraint centre, so the guard's
    independent sampled-CV2 fingerprint improves under the correct repair --
    exactly as it does on the real run.
    """
    phase.mkdir(parents=True, exist_ok=True)
    window = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
    cv2 = np.array([_SECONDARY[_SURVIVOR_STATES[w]] for w in window], dtype=np.float64)
    cv = np.array([_PRIMARY[_SURVIVOR_STATES[w]] for w in window], dtype=np.float64)
    np.savez(phase / 'analysis_arrays.npz', cv_A=cv, window=window, secondary_cv=cv2,
             step=np.arange(6, dtype=np.int64), replica=np.zeros(6, dtype=np.int32))
    # Every real phase dir carries one; the guard resolves the temperature its
    # CV2 restraint widths are expressed in from it.
    (phase / 'run_manifest.json').write_text(
        json.dumps({'resolved_args': {'temperature_k': 300.0}}), encoding='utf-8')


def _npz_run(tmp_path: Path, map_states: list, *, drop_record: bool) -> Path:
    """A legacy NPZ-only adaptive run for compare_gareus_runs.load_npz_adaptive_union."""
    ap = tmp_path / 'run' / 'adaptive_production'
    _write_registry(ap)
    phase = ap / 'epoch_000'
    _write_epoch_npz(phase)
    _write_map(phase, map_states)
    if drop_record:
        _write_drop_record(phase, [1, 3], n_windows=3)
    (ap.parent / 'run_args.json').write_text(json.dumps({'temperature_k': 300.0}))
    return ap


def _union_npz_run(tmp_path: Path, *, include_epochs: bool = True) -> Path:
    """A run whose only MBAR input is the driver-built adaptive_union_mbar.npz."""
    ap = tmp_path / 'run' / 'adaptive_production'
    ap.mkdir(parents=True)
    np.savez(ap / 'adaptive_union_mbar.npz',
             cv_A=np.array([1.0, 2.0]), secondary_cv=np.array([np.nan, np.nan]),
             sampled_state_ids=np.array([0, 1]), umbrella_reduced_bias_nk=np.zeros((2, 2)),
             primary_centers=np.array([1.0, 2.0]), primary_k=np.array([10.0, 10.0]))
    (ap / 'adaptive_union_mbar.json').write_text(
        json.dumps({'include_epochs': bool(include_epochs), 'n_samples': 2}), encoding='utf-8')
    (ap.parent / 'run_args.json').write_text(json.dumps({'temperature_k': 300.0}))
    return ap


def _write_parquet_samples(phase: Path) -> None:
    import pandas as pd
    sp = phase / 'samples'
    sp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'cv1': [1.0, 1.2, 1.4, 1.4],
                  'cv2': [-1.0, 0.0, 1.0, 1.0],
                  'window_id': [0, 1, 2, 2]}).to_parquet(sp / 'part-0.parquet')


# ---------------------------------------------------------------------------
# The note objects the consumers below branch on
# ---------------------------------------------------------------------------

def test_a_window_map_note_is_a_string_everywhere_and_a_kind_where_it_matters():
    """The contract that lets two consumers tell "stale" from "could not check"
    without any of the other six knowing there is a distinction.

    A note has to keep behaving as its own text: it is printed, extended into
    `meta['load_notes']`, JSON-serialised into pmf_summary.json's warnings and
    then regex-triaged there by gareus_report.py, which sees only the JSON.
    """
    import json
    from gareus.mbar_analysis.loaders_adaptive import (
        MAP_NOTE_UNCHECKED, WindowMapNote, window_map_note_kind,
        window_map_note_reports_a_fault)
    note = WindowMapNote('[window map check] Could not cross-check ...', MAP_NOTE_UNCHECKED)

    assert isinstance(note, str)
    assert json.loads(json.dumps({'warnings': [note]}))['warnings'][0] == str(note)
    assert note.startswith('[window map check]')
    assert window_map_note_kind(note) == MAP_NOTE_UNCHECKED
    assert not window_map_note_reports_a_fault(note)
    # A note that reaches a consumer without a kind (a plain str, a copy made by
    # something that drops it) must fall back to the STRICTER reading.
    assert window_map_note_reports_a_fault(str(note))
    assert window_map_note_reports_a_fault(f'{note}')


def test_a_window_map_note_survives_being_copied_and_pickled():
    """A warning that cannot be copied is worse than the drift it reports.

    `str` subclasses are reconstructed by `copy`/`copy.deepcopy`/`pickle` via
    `cls.__new__(cls, text)`, so a required second constructor argument makes
    every one of those raise `TypeError` -- turning a note that merely says "I
    could not check this phase" into a crash in whatever deep-copies the
    metadata dict it travels in.  Both the round trip and the kind it carries
    are pinned here.
    """
    import copy
    import pickle
    from gareus.mbar_analysis.loaders_adaptive import (
        MAP_NOTE_REPAIRED, WindowMapNote, window_map_note_kind, window_map_note_rewrote_rows)
    note = WindowMapNote('[stale window map] ... REPAIRED IN MEMORY ...', MAP_NOTE_REPAIRED)

    for clone in (copy.copy(note), copy.deepcopy(note), pickle.loads(pickle.dumps(note))):
        assert str(clone) == str(note)
        assert window_map_note_kind(clone) == MAP_NOTE_REPAIRED
        assert window_map_note_rewrote_rows(clone)
    # A whole notes list, which is the shape that actually travels in meta.
    assert [window_map_note_kind(n) for n in copy.deepcopy([note, note])] == [MAP_NOTE_REPAIRED] * 2


# ---------------------------------------------------------------------------
# I3a -- compare_gareus_runs.load_npz_adaptive_union
# ---------------------------------------------------------------------------

def _compare_module():
    import compare_gareus_runs
    return compare_gareus_runs


def test_compare_union_healthy_map_attributes_samples_exactly(tmp_path):
    """A consistent (non-identity) map is honoured verbatim and nothing is touched.

    The map itself already says 0->0, 1->2, 2->4, so the six samples must come
    out on registry rows 0, 0, 2, 2, 4, 4. A loader that ignored the map and
    used the raw local window_id would give 0, 0, 1, 1, 2, 2 instead.
    """
    ap = _npz_run(tmp_path, _SURVIVOR_STATES, drop_record=False)
    d = _compare_module().load_npz_adaptive_union(ap)
    assert list(d.window) == [0, 0, 2, 2, 4, 4]
    assert d.u_nk.shape == (6, 5)
    assert 'load_notes' not in d.meta


def test_compare_union_repairs_stale_map_before_attributing_samples(tmp_path):
    """The bug itself: a 5-row map for a 3-window phase.

    Un-guarded, the six samples land on states 0, 0, 1, 1, 2, 2 -- two of them
    on state 1, a window that was dropped and never sampled. Repaired, they
    land on 0, 0, 2, 2, 4, 4.
    """
    ap = _npz_run(tmp_path, [0, 1, 2, 3, 4], drop_record=True)
    d = _compare_module().load_npz_adaptive_union(ap)
    assert list(d.window) == [0, 0, 2, 2, 4, 4]
    assert list(d.window) != [0, 0, 1, 1, 2, 2]      # the un-guarded answer
    notes = d.meta.get('load_notes') or []
    assert any('stale window map' in n for n in notes)


def test_compare_union_bias_column_follows_the_repaired_state(tmp_path):
    """Not just the label: the reduced bias each sample is scored against.

    Sample 4 sits at state 4's restraint centre, so after the repair its own
    state's bias must be ~0 and state 2's must be large. Before the repair it
    was attributed to state 2, i.e. scored as though it sat 1.0 CV2 units off
    that restraint.
    """
    ap = _npz_run(tmp_path, [0, 1, 2, 3, 4], drop_record=True)
    d = _compare_module().load_npz_adaptive_union(ap)
    own = d.u_nk[4, d.window[4]]
    misattributed = d.u_nk[4, 2]
    assert own == pytest.approx(0.0, abs=1e-9)
    assert misattributed > 1.0


def test_compare_union_fails_closed_when_the_map_cannot_be_repaired(tmp_path):
    """No drop record and no surviving-window table: refuse rather than guess."""
    ap = _npz_run(tmp_path, [0, 1, 2, 3, 4], drop_record=False)
    with pytest.raises(ValueError) as exc:
        _compare_module().load_npz_adaptive_union(ap)
    assert 'refusing to load' in str(exc.value)


def test_compare_union_override_loads_the_stale_mapping_deliberately(tmp_path, monkeypatch):
    """The documented inspection escape hatch still reaches this loader."""
    monkeypatch.setenv(_STALE_ENV, '1')
    ap = _npz_run(tmp_path, [0, 1, 2, 3, 4], drop_record=False)
    d = _compare_module().load_npz_adaptive_union(ap)
    assert list(d.window) == [0, 0, 1, 1, 2, 2]      # stale, as asked for
    assert any('STALE map' in n for n in (d.meta.get('load_notes') or []))


# ---------------------------------------------------------------------------
# I3b -- load_data's adaptive_union_mbar.npz branch
# ---------------------------------------------------------------------------

def test_union_npz_healthy_phase_maps_load_silently(tmp_path):
    ap = _union_npz_run(tmp_path)
    phase = ap / 'final' / 'baseline'
    _write_map(phase, _SURVIVOR_STATES)
    _write_drop_record(phase, [], n_windows=3)
    assert check_union_npz_window_map_provenance(ap) == []
    d = load_data(ap.parent, None)
    assert d.cv.size == 2
    assert 'load_notes' not in d.meta


def test_union_npz_refuses_when_a_phase_map_was_stale(tmp_path):
    """The npz's sampled_state_ids were resolved through this very map while
    the run was live, so a stale map means the npz is wrong on disk -- and it
    stores no local window indices to repair it from."""
    ap = _union_npz_run(tmp_path)
    phase = ap / 'final' / 'baseline'
    _write_map(phase, [0, 1, 2, 3, 4])
    _write_drop_record(phase, [1, 3], n_windows=3)
    with pytest.raises(ValueError) as exc:
        load_data(ap.parent, None)
    msg = str(exc.value)
    assert 'adaptive_union_mbar.npz' in msg
    assert 'refusing to load' in msg
    assert 'CANNOT be repaired in memory' in msg


def test_union_npz_override_downgrades_the_refusal_to_a_note(tmp_path, monkeypatch):
    """The override must survive the wrapper: the guard stops raising when it
    is set and reports through its notes instead, so a wrapper treating any
    note as fatal would invert it."""
    monkeypatch.setenv(_STALE_ENV, '1')
    ap = _union_npz_run(tmp_path)
    phase = ap / 'final' / 'baseline'
    _write_map(phase, [0, 1, 2, 3, 4])
    _write_drop_record(phase, [1, 3], n_windows=3)
    d = load_data(ap.parent, None)
    notes = d.meta.get('load_notes') or []
    assert any('stale window map' in n for n in notes)
    assert d.cv.size == 2


def test_union_npz_still_refuses_when_the_stale_rows_move_no_sample(tmp_path):
    """RUNS/chignolin_5's real final/baseline shape: 4 map rows, 3 windows, and
    the one dropped row LAST -- so local windows 0..2 name the same states
    before and after the repair, and this phase's per-sample attribution is
    genuinely untouched.

    It is still refused, and the message must say which of the two it is. The
    dropped window was never retired from the state registry either, so it can
    have entered the npz's solve as a state of its own with an N_k the maps
    cannot account for -- passing the artifact on the strength of a check that
    does not cover that would be the silent-wrong-number trade this guard
    exists to refuse.
    """
    ap = _union_npz_run(tmp_path)
    phase = ap / 'final' / 'baseline'
    _write_map(phase, [0, 1, 2, 3])
    _write_drop_record(phase, [3], n_windows=3)
    with pytest.raises(ValueError) as exc:
        check_union_npz_window_map_provenance(ap)
    msg = str(exc.value)
    assert 'per-sample attribution is unaffected' in msg
    assert 'never retired from the state registry' in msg
    # ... and a shifting phase is NOT described that way.
    _write_drop_record(phase, [1], n_windows=3)
    with pytest.raises(ValueError) as exc2:
        check_union_npz_window_map_provenance(ap)
    assert 'per-sample attribution is unaffected' not in str(exc2.value)


def test_union_npz_says_so_when_provenance_cannot_be_checked(tmp_path):
    """A phase that recorded no window count is reported as unchecked, not as
    healthy: 'not checkable' and 'checked, fine' are different claims."""
    ap = _union_npz_run(tmp_path)
    _write_map(ap / 'final' / 'baseline', [0, 1, 2, 3, 4])
    d = load_data(ap.parent, None)
    notes = d.meta.get('load_notes') or []
    assert notes and notes[0].startswith('[unverified window map]')
    assert 'final/baseline' in notes[0]


def test_union_npz_loads_a_phase_whose_membership_check_could_not_run(tmp_path):
    """A note is no longer a boolean for "this map is stale", and this consumer
    is where treating it as one refused a healthy run.

    The equal-row-count branch of the guard can now report that a *check could
    not run* -- here the phase's map (4 rows) matches its own recorded window
    count (4), and only the cross-check against the surviving-window table (3
    rows) is impossible.  The guard grades that below a detected fault on
    purpose and `gareus_report.py` triages its wording a band lower still, but
    this function put any noted phase straight into its `stale` bucket and
    raised -- refusing to load a run on the strength of an absent check, which
    is the cry-wolf half of the same drift the guard exists to prevent.

    The note must still be reported: "could not check" is not "checked, fine".
    """
    ap = _union_npz_run(tmp_path)
    phase = ap / 'final' / 'baseline'
    _write_map(phase, [0, 1, 2, 3])
    _write_drop_record(phase, [], n_windows=4)
    _write_window_table(phase, [0, 1, 2])          # one row short: nothing to line up

    notes = check_union_npz_window_map_provenance(ap)      # must not raise

    assert len(notes) == 1
    assert notes[0].startswith('[window map check] Could not cross-check')
    assert 'final/baseline' in notes[0]
    assert 'the table lists 3 window(s) while the map lists 4' in notes[0]
    # And it reaches the caller the same way every other load note does.
    assert any('Could not cross-check' in n for n in (load_data(ap.parent, None)
                                                      .meta.get('load_notes') or []))


def test_union_npz_still_refuses_a_map_listing_a_different_window_set(tmp_path):
    """The non-firing twin of the test above, and the reason it cannot simply
    stop reading notes: an equal-count map that lists a genuinely DIFFERENT
    window set is a detected fault, is not repairable, and the npz's baked-in
    attribution was resolved through it.  Still fatal."""
    ap = _union_npz_run(tmp_path)
    phase = ap / 'final' / 'baseline'
    _write_map(phase, [0, 1, 2])
    _write_drop_record(phase, [], n_windows=3)
    _write_window_table(phase, [0, 1, 3])          # window 2 never ran; window 3 did

    with pytest.raises(ValueError) as exc:
        check_union_npz_window_map_provenance(ap)
    assert 'DIFFERENT window set of the same size' in str(exc.value)


def test_union_npz_still_refuses_a_map_the_guard_had_to_reorder(tmp_path):
    """The other non-firing twin: a repair is not a clean bill of health here.

    A permuted map is now repairable in memory, which fixes the per-epoch
    Parquet path -- but this artifact's per-sample states were resolved by the
    driver, at run time, through the map as it was.  A repair therefore means
    the npz on disk is wrong, and the npz stores no local window indices to
    redo it from.
    """
    ap = _union_npz_run(tmp_path)
    phase = ap / 'final' / 'baseline'
    _write_map(phase, [0, 2, 4])
    _write_drop_record(phase, [], n_windows=3)
    _write_window_table(phase, [0, 4, 2])          # same windows, wrong order

    with pytest.raises(ValueError) as exc:
        check_union_npz_window_map_provenance(ap)
    msg = str(exc.value)
    assert 'PERMUTATION' in msg
    assert 'CANNOT be repaired in memory' in msg


def test_union_npz_does_not_claim_an_unrepaired_fault_left_attribution_intact(tmp_path):
    """The qualifier on this message is a claim about a REPAIR, so it may only
    be made when there was one.

    It used to be decided by comparing the guard's returned rows against the
    rows handed in: equal, therefore "the repair moved nothing". For a fault
    the guard detects but cannot repair -- an equal-count map listing a
    genuinely different window set -- the guard returns the caller's own rows
    untouched, so that comparison is true VACUOUSLY, and the operator was told
    "its per-sample attribution is unaffected" about a phase whose mapping is
    not known at all. That is the reassuring half of a message that is
    otherwise a refusal, which is the worst place for it.

    Both halves are asserted together on purpose: an absent substring alone
    would also be satisfied by a typo in the source string, so the positive
    twin -- a phase the guard really did repair, and whose repair really does
    move no local window -- has to keep receiving it.
    """
    ap = _union_npz_run(tmp_path)
    phase = ap / 'final' / 'baseline'

    # (a) detected, not repairable: window 2 has a row and never ran, window 3
    #     ran and has no row anywhere, so no corrected mapping exists.
    _write_map(phase, [0, 1, 2])
    _write_drop_record(phase, [], n_windows=3)
    _write_window_table(phase, [0, 1, 3])
    with pytest.raises(ValueError) as exc:
        check_union_npz_window_map_provenance(ap)
    msg = str(exc.value)
    assert 'DIFFERENT window set of the same size' in msg
    assert 'per-sample attribution is unaffected' not in msg
    assert 'cannot be sized from these artifacts' in msg

    # (b) the positive twin, RUNS/chignolin_5's real final/baseline shape: 4 map
    #     rows, 3 windows, the dropped row LAST, so the repair renumbers nothing
    #     that a sample can reach and the claim is true.
    (phase / 'umbrella_explicit_windows.csv').unlink()
    _write_map(phase, [0, 1, 2, 3])
    _write_drop_record(phase, [3], n_windows=3)
    with pytest.raises(ValueError) as exc2:
        check_union_npz_window_map_provenance(ap)
    assert 'per-sample attribution is unaffected' in str(exc2.value)
    assert 'cannot be sized from these artifacts' not in str(exc2.value)


def test_union_npz_sizes_the_damage_of_a_repairable_phase_under_the_override(tmp_path, monkeypatch):
    """The second proxy in the same expression, and the same correction.

    The qualifier was also gated on the override being unset, on the reasoning
    that the override means the guard "hands back the stale rows unrepaired".
    That is true only of the refusal path: both repair paths run regardless of
    the override (it never suppresses a repair that IS possible), so a phase
    the guard repaired -- and whose repair moves none of its own samples -- had
    the true, useful half of the message withheld for no reason.

    Same fixture as the positive twin above, read under the override, where the
    check reports through a note instead of raising.
    """
    ap = _union_npz_run(tmp_path)
    phase = ap / 'final' / 'baseline'
    _write_map(phase, [0, 1, 2, 3])
    _write_drop_record(phase, [3], n_windows=3)

    monkeypatch.setenv(_STALE_ENV, '1')

    notes = check_union_npz_window_map_provenance(ap)           # must not raise

    assert len(notes) == 1
    assert 'per-sample attribution is unaffected' in notes[0]


def test_union_npz_ignores_epochs_it_was_not_built_from(tmp_path):
    """`include_epochs: false` means the union builder read only final/*, so a
    stale numbered epoch must not condemn this npz."""
    ap = _union_npz_run(tmp_path, include_epochs=False)
    healthy = ap / 'final' / 'baseline'
    _write_map(healthy, _SURVIVOR_STATES)
    _write_drop_record(healthy, [], n_windows=3)
    stale_epoch = ap / 'epoch_000'
    _write_map(stale_epoch, [0, 1, 2, 3, 4])
    _write_drop_record(stale_epoch, [1, 3], n_windows=3)
    assert check_union_npz_window_map_provenance(ap) == []
    # ... and the same epoch does condemn an npz that WAS built from it.
    (ap / 'adaptive_union_mbar.json').write_text(json.dumps({'include_epochs': True}))
    with pytest.raises(ValueError):
        check_union_npz_window_map_provenance(ap)


# ---------------------------------------------------------------------------
# M3 -- plot_adaptive_diagnostics.py
# ---------------------------------------------------------------------------

def _plot_module():
    import plot_adaptive_diagnostics
    return plot_adaptive_diagnostics


def test_figure_rows_drop_the_phantom_rows_the_mbar_path_keeps(tmp_path):
    """The two consumers want opposite halves of the guard's return value.

    MBAR keeps the dropped states' rows (parked past
    `_PHANTOM_EPOCH_WINDOW_BASE`) so their epoch-native window params still
    back their bias columns. A figure would draw one window ellipse per row,
    so it must see survivors only.
    """
    from gareus.mbar_analysis.loaders_adaptive import (
        _PHANTOM_EPOCH_WINDOW_BASE, _validate_and_repair_epoch_window_map)
    phase = tmp_path / 'epoch_000'
    _write_map(phase, [0, 1, 2, 3, 4])
    _write_drop_record(phase, [1, 3], n_windows=3)
    rows = list(csv.DictReader((phase / 'epoch_window_map.csv').open(newline='')))

    mbar_rows, _ = _validate_and_repair_epoch_window_map(phase, rows)
    assert len(mbar_rows) == 5
    assert max(int(r['epoch_window']) for r in mbar_rows) >= _PHANTOM_EPOCH_WINDOW_BASE

    fig_rows, note = figure_epoch_window_map_rows(phase, rows)
    assert [int(r['state_id']) for r in fig_rows] == _SURVIVOR_STATES
    assert [int(r['epoch_window']) for r in fig_rows] == [0, 1, 2]
    assert note is not None


def test_figure_rows_are_kept_whole_when_a_check_merely_could_not_run(tmp_path):
    """The figure consumer's half of the same contract drift.

    This path used to read "notes is non-empty" as "the guard rewrote the rows",
    and act on it twice: it dropped every row it could not place at a
    contiguous 0..N-1 index (right after a repair, wrong otherwise), and it
    handed the note to the plotter, which stamps a fixed "STALE
    epoch_window_map.csv" heading on the figure.  Applied to a note that says
    only that a cross-check could not run, that silently removes a real window
    from every per-state figure AND labels the run stale.

    The fixture is a map whose row count is right and one of whose
    ``epoch_window`` values is unparseable, which is exactly what makes the
    positional membership check abstain.  Nothing here is known to be wrong, so
    nothing may be removed and nothing may be announced.
    """
    phase = tmp_path / 'epoch_000'
    _write_map(phase, _SURVIVOR_STATES)
    _write_drop_record(phase, [], n_windows=3)
    _write_window_table(phase, _SURVIVOR_STATES)
    rows = list(csv.DictReader((phase / 'epoch_window_map.csv').open(newline='')))
    rows[2]['epoch_window'] = 'x'

    fig_rows, note = figure_epoch_window_map_rows(phase, rows)

    assert note is None
    assert [r['state_id'] for r in fig_rows] == [str(s) for s in _SURVIVOR_STATES]
    # ...while the MBAR/summary consumer, whose surface can carry the wording,
    # still gets the note.
    from gareus.mbar_analysis.loaders_adaptive import _validate_and_repair_epoch_window_map
    assert 'Could not cross-check' in _validate_and_repair_epoch_window_map(phase, rows)[1][0]


def test_figure_rows_still_carry_a_note_for_a_fault_that_was_not_repaired(tmp_path):
    """The non-firing twin: an over-broad fix that stopped reporting anything it
    had not rewritten would drop the warning that matters.  A map listing a
    different window set of the same size is a detected fault the guard cannot
    repair -- the rows are drawn as they are (a figure tool that refuses to draw
    is useless exactly when it is reached for) and the note travels with them.
    """
    phase = tmp_path / 'epoch_000'
    _write_map(phase, [0, 1, 2])
    _write_drop_record(phase, [], n_windows=3)
    _write_window_table(phase, [0, 1, 3])
    rows = list(csv.DictReader((phase / 'epoch_window_map.csv').open(newline='')))

    fig_rows, note = figure_epoch_window_map_rows(phase, rows)

    assert note is not None and 'DIFFERENT window set of the same size' in note
    assert [int(r['epoch_window']) for r in fig_rows] == [0, 1, 2]


def test_figure_rows_follow_a_reordering_the_guard_repaired(tmp_path):
    """And a repair the guard CAN make reaches the figures, so per-state sample
    counts and window labels are drawn against the states the samples really
    came from."""
    phase = tmp_path / 'epoch_000'
    _write_map(phase, [0, 2, 4])
    _write_drop_record(phase, [], n_windows=3)
    _write_window_table(phase, [0, 4, 2])
    rows = list(csv.DictReader((phase / 'epoch_window_map.csv').open(newline='')))

    fig_rows, note = figure_epoch_window_map_rows(phase, rows)

    assert [int(r['state_id']) for r in fig_rows] == [0, 4, 2]
    assert [int(r['epoch_window']) for r in fig_rows] == [0, 1, 2]
    assert note is not None and 'PERMUTATION' in note


def test_the_figure_note_describes_what_happened_to_the_figure_rows(tmp_path, monkeypatch):
    """The single returned note is the only thing the caller has left to ask.

    `plot_adaptive_diagnostics._phase_window_map` decides whether to follow the
    repair by testing `window_map_note_rewrote_rows` on the note it is handed,
    so if the rows were rewritten and the note handed back says otherwise, the
    figures silently fall back to the map they were repaired away from -- the
    same class of defect as the row-count proxy that branch replaced, one layer
    further in.

    The guard emits one note per phase today, so this is only reachable by
    stubbing it, which is what happens here: the guard's own rows and a note
    list whose repair note is NOT first. Everything downstream of the stub is
    the real `figure_epoch_window_map_rows`.
    """
    from gareus.mbar_analysis import loaders as _loaders
    from gareus.mbar_analysis.loaders_adaptive import (
        MAP_NOTE_FAULT_UNREPAIRED, MAP_NOTE_REPAIRED, WindowMapNote,
        window_map_note_rewrote_rows)
    phase = tmp_path / 'epoch_000'
    _write_map(phase, _SURVIVOR_STATES)
    rows = list(csv.DictReader((phase / 'epoch_window_map.csv').open(newline='')))
    other = WindowMapNote('[window map check] something else', MAP_NOTE_FAULT_UNREPAIRED)
    repair = WindowMapNote('[stale window map] REPAIRED IN MEMORY', MAP_NOTE_REPAIRED)
    monkeypatch.setattr(_loaders, '_validate_and_repair_epoch_window_map',
                        lambda *a, **k: (rows, [other, repair]))

    fig_rows, note = figure_epoch_window_map_rows(phase, rows)

    assert window_map_note_rewrote_rows(note)
    assert [int(r['state_id']) for r in fig_rows] == _SURVIVOR_STATES


def test_discover_phases_healthy_map_is_passed_through_untouched(tmp_path):
    plot = _plot_module()
    ap = tmp_path / 'run' / 'adaptive_production'
    phase = ap / 'epoch_000'
    (phase / 'samples').mkdir(parents=True)
    _write_map(phase, _SURVIVOR_STATES)
    _write_drop_record(phase, [], n_windows=3)
    phases = plot.discover_phases(ap)
    assert len(phases) == 1
    assert plot._wmap_to_sid(phases[0]['wmap']) == _REPAIRED_MAPPING
    assert phases[0]['wmap_note'] is None
    assert plot._phase_map_notes(phases) == []


def test_discover_phases_repairs_a_stale_map_for_the_figures(tmp_path):
    """The M3 finding: per-state figures for a legacy auto-drop run were
    labelled from the stale map."""
    plot = _plot_module()
    ap = tmp_path / 'run' / 'adaptive_production'
    phase = ap / 'epoch_000'
    (phase / 'samples').mkdir(parents=True)
    _write_map(phase, [0, 1, 2, 3, 4])
    _write_drop_record(phase, [1, 3], n_windows=3)
    phases = plot.discover_phases(ap)
    assert plot._wmap_to_sid(phases[0]['wmap']) == _REPAIRED_MAPPING
    assert plot._wmap_to_sid(phases[0]['wmap']) != _STALE_MAPPING
    # One ellipse per row: the never-run windows must be gone from the frame.
    assert len(phases[0]['wmap']) == 3
    assert [int(v) for v in phases[0]['wmap']['state_id']] == _SURVIVOR_STATES
    assert 'stale window map' in phases[0]['wmap_note']


def test_discover_phases_follows_a_reordering_the_guard_repaired(tmp_path):
    """The figures must be drawn against the mapping the guard reports, not the
    one it replaced.

    `_phase_window_map` decided whether the guard had rewritten its rows by
    comparing row COUNTS -- which answered the question correctly only as long
    as the only repair was removing never-run windows. A permutation repair
    (same window set, wrong order) removes nothing, so the counts agree, the
    short-circuit fired, and the plotter returned the ORIGINAL frame -- while
    stamping the guard's note, which says REPAIRED IN MEMORY, across the
    figure. Every per-state sample count and window label in that figure then
    belongs to a different state than its caption claims, which is worse than
    the unstamped stale figure this guard was added to prevent.

    Asserted on the mapping, not on the note: `wmap_note is not None` is
    satisfied with the defect fully intact, and the local-window -> state_id
    dict is the value that decides which state a figure attributes a sample to.
    `discover_phases` rather than `figure_epoch_window_map_rows`, because the
    inner function was already correct here (it branches on the note's kind) --
    the frame-level consumer one layer out is where the proxy was.
    """
    plot = _plot_module()
    ap = tmp_path / 'run' / 'adaptive_production'
    phase = ap / 'epoch_000'
    (phase / 'samples').mkdir(parents=True)
    _write_map(phase, [0, 2, 4])
    _write_drop_record(phase, [], n_windows=3)
    _write_window_table(phase, [0, 4, 2])          # same windows, wrong order

    phases = plot.discover_phases(ap)

    assert len(phases) == 1
    assert plot._wmap_to_sid(phases[0]['wmap']) == {0: 0, 1: 4, 2: 2}
    assert plot._wmap_to_sid(phases[0]['wmap']) != {0: 0, 1: 2, 2: 4}    # the map on disk
    # Nothing was dropped, so the frame keeps all three window ellipses -- and
    # its own dtypes, which is why the repair re-selects the typed frame's rows
    # instead of rebuilding one from the guard's string rows.
    assert len(phases[0]['wmap']) == 3
    assert [float(v) for v in phases[0]['wmap']['secondary_center']] == [
        _SECONDARY[0], _SECONDARY[4], _SECONDARY[2]]
    assert 'PERMUTATION' in phases[0]['wmap_note']


def _write_map_with_state_ids(phase: Path, centre_states: list, state_ids: list) -> None:
    """A map whose row *i* carries `centre_states[i]`'s restraint centres but
    `state_ids[i]`'s state_id.

    The two are separable on purpose: every other builder here derives the
    state_id from the centres, which makes a duplicated state_id imply duplicated
    centres and so unreachable by the repair at all.  A row that keeps its centres
    and carries the wrong state_id is a documented blind spot of BOTH centre
    checks (see `_MEMBERSHIP_CHECK_SCOPE`), which is exactly why the frame join
    downstream must not assume it away.
    """
    phase.mkdir(parents=True, exist_ok=True)
    with (phase / 'epoch_window_map.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch_window', 'state_id', 'primary_center',
                                          'primary_k', 'secondary_center', 'secondary_k'])
        w.writeheader()
        for ew, (centre_sid, sid) in enumerate(zip(centre_states, state_ids)):
            w.writerow({'epoch_window': ew, 'state_id': sid,
                        'primary_center': repr(_PRIMARY[centre_sid]), 'primary_k': '10.0',
                        'secondary_center': repr(_SECONDARY[centre_sid]), 'secondary_k': '5.0'})


def test_discover_phases_refuses_to_join_a_repair_onto_a_duplicated_state_id(tmp_path):
    """The frame join after a repair must check the property its own comment rests
    on, not a proxy a violation walks through.

    `_phase_window_map` re-selects the *typed* frame's rows for the repaired rows
    by state_id, via `pos = {int(state_id): row_index}`.  That dict is last-wins.
    If two rows of one phase's map carry the same state_id with different centres
    -- the one defect a centre comparison is blind to, on both sides, by
    construction -- `pos` collapses onto the later row, every lookup below still
    succeeds, the `len(keep) != len(survivors)` guard still passes, and
    `df.iloc[keep]` hands the figures a frame with one row duplicated and another
    silently gone.  A wrong frame that passes the guard is precisely what this
    file's guard exists to prevent.

    The map here is a genuine permutation (distinct centres, so the guard repairs
    it) whose third row carries state 4's id instead of state 2's.  Asserted on
    the FRAME, not the note: the note is present either way.
    """
    plot = _plot_module()
    ap = tmp_path / 'run' / 'adaptive_production'
    phase = ap / 'epoch_000'
    (phase / 'samples').mkdir(parents=True)
    _write_map_with_state_ids(phase, [0, 2, 4], [0, 4, 4])
    _write_drop_record(phase, [], n_windows=3)
    _write_window_table(phase, [0, 4, 2])          # same windows, wrong order

    phases = plot.discover_phases(ap)

    frame = phases[0]['wmap']
    centres = [float(v) for v in frame['secondary_center']]
    # The stale map, whole and in its own order -- the conservative fallback this
    # tool already takes for anything it cannot fix.
    assert centres == [_SECONDARY[0], _SECONDARY[2], _SECONDARY[4]]
    # ...and specifically NOT the collapsed join, which drops window 1's real
    # centre and draws window 2's twice.
    assert centres != [_SECONDARY[0], _SECONDARY[4], _SECONDARY[4]]
    assert len(frame) == 3
    note = phases[0]['wmap_note']
    assert 'state_id [4] on more than one row' in note
    assert 'fall back to the STALE map' in note


def test_discover_phases_keeps_the_frame_whole_for_a_fault_it_could_not_repair(tmp_path):
    """The non-firing twin: converting the row-count proxy to the note's own
    predicate must not start reordering frames the guard never rewrote.

    An equal-count map listing a genuinely DIFFERENT window set is detected and
    NOT repairable, so the rows come back exactly as they were read and the
    figures draw them -- with the warning, which is this tool's whole policy on
    a map it cannot fix.
    """
    plot = _plot_module()
    ap = tmp_path / 'run' / 'adaptive_production'
    phase = ap / 'epoch_000'
    (phase / 'samples').mkdir(parents=True)
    _write_map(phase, [0, 1, 2])
    _write_drop_record(phase, [], n_windows=3)
    _write_window_table(phase, [0, 1, 3])          # window 2 never ran; window 3 did

    phases = plot.discover_phases(ap)

    assert plot._wmap_to_sid(phases[0]['wmap']) == {0: 0, 1: 1, 2: 2}
    assert 'DIFFERENT window set of the same size' in phases[0]['wmap_note']


def test_discover_phases_scheduled_layout_is_guarded_too(tmp_path):
    """baseline/topup_* sub-runs go through the same helper as flat epochs."""
    plot = _plot_module()
    ap = tmp_path / 'run' / 'adaptive_production'
    phase = ap / 'final' / 'baseline'
    (phase / 'samples').mkdir(parents=True)
    _write_map(phase, [0, 1, 2, 3, 4])
    _write_drop_record(phase, [1, 3], n_windows=3)
    phases = plot.discover_phases(ap)
    assert len(phases) == 1
    assert plot._wmap_to_sid(phases[0]['wmap']) == _REPAIRED_MAPPING


def test_plot_tool_draws_an_unrepairable_map_but_says_so(tmp_path):
    """A figure tool that refuses to draw is useless exactly when it is
    reached for -- so the stale rows are still drawn, carrying a warning."""
    plot = _plot_module()
    ap = tmp_path / 'run' / 'adaptive_production'
    phase = ap / 'epoch_000'
    (phase / 'samples').mkdir(parents=True)
    _write_map(phase, [0, 1, 2, 3, 4])
    (phase / 'analysis_metadata_validation.json').write_text(
        json.dumps({'n_windows': 3, 'status': 'ok'}), encoding='utf-8')
    phases = plot.discover_phases(ap)                      # must not raise
    assert plot._wmap_to_sid(phases[0]['wmap']) == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    assert 'DRAWN FROM THE STALE MAP' in phases[0]['wmap_note']


def test_annotate_map_warnings_stamps_the_note_on_the_figure(tmp_path):
    """The warning has to travel with the picture, not just the terminal."""
    import matplotlib.pyplot as plt
    plot = _plot_module()
    fig = plt.figure()
    try:
        plot._annotate_map_warnings(fig, [])
        assert fig.texts == []
        plot._annotate_map_warnings(fig, ['[stale window map] epoch_000 is stale'])
        stamped = [t.get_text() for t in fig.texts]
        assert len(stamped) == 1
        assert 'STALE epoch_window_map.csv' in stamped[0]
        assert 'epoch_000 is stale' in stamped[0]
    finally:
        plt.close(fig)


def test_fig_topup_targeting_guards_its_own_map_read_and_annotates(tmp_path, monkeypatch):
    """fig_topup_targeting re-reads the maps itself rather than using the
    discovered phases, so it is wired separately -- end to end here."""
    import pandas as pd
    plot = _plot_module()
    ap = tmp_path / 'run' / 'adaptive_production'
    phase = ap / 'final' / 'baseline'
    _write_parquet_samples(phase)
    _write_map(phase, [0, 1, 2, 3, 4])
    _write_drop_record(phase, [1, 3], n_windows=3)
    state_reg = pd.DataFrame({'state_id': list(range(5)),
                              'primary_center': _PRIMARY,
                              'secondary_center': _SECONDARY})
    seen: list = []
    real = plot._annotate_map_warnings
    monkeypatch.setattr(plot, '_annotate_map_warnings',
                        lambda fig, notes: (seen.append(list(notes)), real(fig, notes))[1])
    out = tmp_path / 'fig4.png'
    plot.fig_topup_targeting(ap, state_reg, out)
    assert out.exists()
    assert seen and seen[0] and 'stale window map' in seen[0][0]
